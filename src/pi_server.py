import cv2
import argparse
import threading
import time
from flask import Flask, Response, render_template_string, jsonify
from flasgger import Swagger

from processor import CatFlapProcessor
from motion import MotionDetector, MotionGate
from catflap import SimulationCatflap, ServoCatflap
from state_machine import State, AccessState

app = Flask(__name__)
swagger = Swagger(app)

# Global variables for the background pipeline
pipeline_thread = None
current_frame_mjpeg = None
current_frame_seq = 0  # increases with every new JPEG, so the stream only sends new frames
current_mode = "idle"
frame_lock = threading.Lock()

# Global hardware
flap = None
auto_lock_timer = None

def schedule_auto_lock():
    """Schedules the flap to automatically lock after 30 seconds."""
    global auto_lock_timer
    
    # Cancel existing timer if there is one
    if auto_lock_timer is not None:
        auto_lock_timer.cancel()
        
    # Start a new 30 second timer
    auto_lock_timer = threading.Timer(30.0, flap.lock_in)
    auto_lock_timer.start()

def on_access_state_changed(access_state):
    """Event listener triggered by the StateMachine when the global access decision changes."""
    global auto_lock_timer
    
    if access_state == AccessState.GRANTED:
        print("[Event] Clean cat(s) detected. Unlocking flap for 30s.")
        flap.unlock_in()
        schedule_auto_lock()
        
    elif access_state == AccessState.DENIED:
        print("[Event] Cat WITH PREY detected! Locking flap immediately.")
        flap.lock_in()
        # Cancel any pending auto-lock since we are locking immediately
        if auto_lock_timer is not None:
            auto_lock_timer.cancel()
            auto_lock_timer = None

IDLE_INTERVAL = 0.2  # seconds between motion checks while idle (~5 Hz)

def camera_loop(save_uncertain_dir, use_tracking=True, history_length=10, prey_threshold=0.3,
                motion_gate=True, motion_threshold=0.005, idle_after=5.0, idle_probe=2.0):
    """Background thread that continuously processes frames from the Catflap.

    With motion_gate, the detection pipeline only runs while something moves (or a cat is still tracked);
    otherwise only the cheap MotionDetector looks at ~5 frames per second. Every idle_probe seconds one idle
    frame still goes through the full pipeline, so a cat that sits still (no motion) is found anyway.
    """
    global current_frame_mjpeg, current_frame_seq, current_mode

    processor = CatFlapProcessor(save_uncertain_dir=save_uncertain_dir, use_tracking=use_tracking,
                                 history_length=history_length, threshold=prey_threshold)
    # Subscribe to state changes to trigger the flap hardware
    processor.state_machine.subscribe(on_access_state_changed)

    detector = MotionDetector(min_area=motion_threshold)
    gate = MotionGate(idle_after=idle_after)
    current_mode = MotionGate.ACTIVE if not motion_gate else gate.mode
    print(f"[Config] history={history_length} prey_threshold={prey_threshold} "
          f"motion_gate={'on' if motion_gate else 'off'} motion_threshold={motion_threshold} "
          f"idle_after={idle_after}s idle_probe={idle_probe}s")

    frame_idx = 0  # counts processed frames only, so max_missed_frames keeps its meaning
    last_probe = float("-inf")
    try:
        # Loop continuously over frames from the smart flap hardware
        for frame in flap.capture_continuous():
            loop_start = time.time()
            processed = False

            if motion_gate:
                motion = detector.has_motion(frame)
                reason = f"changed pixels {detector.last_ratio:.1%}"
                if not motion and gate.mode == MotionGate.IDLE and idle_probe > 0 and loop_start - last_probe >= idle_probe:
                    # Idle probe: one full detection, wakes the gate if a cat face is visible
                    last_probe = loop_start
                    frame_idx += 1
                    processor.process_frame(frame, frame_idx)
                    processed = True
                    if processor.last_num_detections > 0:
                        motion = True
                        reason = "cat seen by idle probe"
                mode, changed = gate.step(motion, bool(processor.state_machine.tracks), loop_start)
                if changed:
                    print(f"[Motion] {mode.upper()} ({reason})")
                current_mode = mode

            if current_mode == MotionGate.ACTIVE and not processed:
                frame_idx += 1
                processor.process_frame(frame, frame_idx)
            elif current_mode == MotionGate.IDLE:
                cv2.putText(frame, "IDLE (motion watch)",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                with frame_lock:
                    current_frame_mjpeg = buffer.tobytes()
                    current_frame_seq += 1

            if current_mode == MotionGate.IDLE:
                # Keep the CPU (and the Pi) cool while nothing happens
                time.sleep(max(0.0, IDLE_INTERVAL - (time.time() - loop_start)))

    finally:
        flap.close()

# --- API Endpoints ---

@app.route('/api/status', methods=['GET'])
def get_status():
    """
    Get the current lock status of the cat flap
    ---
    responses:
      200:
        description: Returns the lock status
        schema:
          type: object
          properties:
            is_locked_in:
              type: boolean
            is_locked_out:
              type: boolean
            mode:
              type: string
              description: "idle (only motion watch) or active (detection running)"
    """
    return jsonify({"is_locked_in": flap.is_locked_in, "is_locked_out": flap.is_locked_out, "mode": current_mode})

@app.route('/api/lock', methods=['POST'])
def lock_flap():
    """
    Manually lock the cat flap
    ---
    responses:
      200:
        description: Flap locked successfully
    """
    flap.lock_in()
    return jsonify({"status": "success", "is_locked_in": flap.is_locked_in})

@app.route('/api/unlock', methods=['POST'])
def unlock_flap():
    """
    Manually unlock the cat flap (will auto-close after 30s)
    ---
    responses:
      200:
        description: Flap unlocked successfully
    """
    flap.unlock_in()
    schedule_auto_lock()
    return jsonify({"status": "success", "is_locked_in": flap.is_locked_in})

# --- Web UI ---

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Cat Flap Stream</title>
    <style>
        body { background-color: #333; color: white; font-family: Arial, sans-serif; text-align: center; }
        img { max-width: 100%; height: auto; border: 2px solid #555; }
        .nav { margin-bottom: 20px; }
        a { color: #4CAF50; text-decoration: none; font-size: 18px; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="nav">
        <a href="/apidocs" target="_blank">📄 Open Swagger API Documentation</a>
    </div>
    <h1>Cat Flap Monitor</h1>
    <img src="{{ url_for('video_feed') }}">
</body>
</html>
"""

def generate_mjpeg_stream():
    """Yields frames to the HTTP client from the global buffer."""
    last_seq = -1
    while True:
        with frame_lock:
            frame_data = current_frame_mjpeg
            seq = current_frame_seq

        if frame_data is not None and seq != last_seq:
            last_seq = seq
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        
        # Sleep slightly to prevent maxing out CPU in the stream generator
        time.sleep(0.05)

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raspberry Pi Flask Stream Server")
    parser.add_argument("--save_uncertain", type=str, default="", help="Directory to save uncertain frames")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--gpio", type=int, default=17, help="GPIO pin for the servo")
    parser.add_argument("--video", type=str, default=None, help="Optional: Path to an input video file to use instead of the camera")
    parser.add_argument("--simulate-servo", action="store_true", help="Run without initializing the physical servo motor")
    parser.add_argument("--flip", action="store_true", help="Flip the camera feed 180 degrees")
    parser.add_argument("--no-tracking", dest="no_tracking", action="store_true", help="Disable ByteTrack; all cats share one global history")
    parser.add_argument("--history", type=int, default=10, help="Frames per cat before a decision (state machine history length)")
    parser.add_argument("--prey-threshold", type=float, default=0.3, help="Mean prey confidence over the history from which a cat counts as carrying prey")
    parser.add_argument("--no-motion-gate", dest="motion_gate", action="store_false", help="Run detection on every frame instead of only after motion")
    parser.add_argument("--motion-threshold", type=float, default=0.005, help="Share of changed pixels (0..1) that counts as motion")
    parser.add_argument("--idle-after", type=float, default=5.0, help="Seconds without motion (and without tracked cats) before detection pauses")
    parser.add_argument("--idle-probe", type=float, default=2.0, help="While idle, run one full detection every N seconds to find cats that sit still (0 = off)")
    args = parser.parse_args()
    
    # Initialize the central Catflap hardware
    try:
        if args.simulate_servo:
            flap = SimulationCatflap(video_source=args.video, flip=args.flip)
        else:
            flap = ServoCatflap(gpio_pin=args.gpio, video_source=args.video, flip=args.flip)
    except RuntimeError as e:
        print(e)
        exit(1)
    
    # Start the camera processing loop in a background thread
    pipeline_thread = threading.Thread(
        target=camera_loop, 
        args=(args.save_uncertain if args.save_uncertain else None, not args.no_tracking,
              args.history, args.prey_threshold, args.motion_gate, args.motion_threshold, args.idle_after, args.idle_probe),
        daemon=True # Daemon thread dies when the main Flask app dies
    )
    pipeline_thread.start()
    
    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"Swagger documentation available at http://{args.host}:{args.port}/apidocs")
    
    app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)
