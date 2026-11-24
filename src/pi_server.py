import cv2
import os
import json
import argparse
import threading
import time
from collections import deque
from flask import Flask, Response, render_template, jsonify, request, send_from_directory, abort
from flasgger import Swagger

from processor import CatFlapProcessor
from motion import MotionDetector, MotionGate
from catflap import SimulationCatflap, ServoCatflap
from recorder import ClipRecorder, DropboxUploader, human_size
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
processor = None  # set by camera_loop, read by the status API
flap_action_lock = threading.Lock()  # camera thread, timer and web requests all move the servo

# --- Operation mode override ---
# auto:   detection decides (default)
# open:   flap stays unlocked, detection only logs (e.g. so the cats get used to the new flap)
# locked: flap stays locked, detection only logs
OVERRIDE_MODES = ("auto", "open", "locked")
OVERRIDE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pi_override.json")
override_mode = "auto"
events = deque(maxlen=15)  # recent events for the web page

recorder = None  # ClipRecorder, see recorder.py

def log_event(text):
    print(f"[Event] {text}")
    events.appendleft({"time": time.strftime("%H:%M:%S"), "text": text})
    if recorder is not None:
        recorder.note(text)

def load_override():
    try:
        with open(OVERRIDE_FILE) as f:
            mode = json.load(f).get("mode", "auto")
        return mode if mode in OVERRIDE_MODES else "auto"
    except (OSError, ValueError):
        return "auto"

def save_override(mode):
    try:
        with open(OVERRIDE_FILE, "w") as f:
            json.dump({"mode": mode}, f)
    except OSError as e:
        print(f"[Override] Could not save mode: {e}")

def cancel_auto_lock():
    global auto_lock_timer
    if auto_lock_timer is not None:
        auto_lock_timer.cancel()
        auto_lock_timer = None

def auto_lock():
    """Timer target: lock again after an unlock, unless the flap is meant to stay open."""
    if override_mode == "open":
        return
    with flap_action_lock:
        flap.lock_in()
    log_event("Auto-lock after 30 s")

def schedule_auto_lock():
    """Schedules the flap to automatically lock after 30 seconds."""
    global auto_lock_timer
    cancel_auto_lock()
    auto_lock_timer = threading.Timer(30.0, auto_lock)
    auto_lock_timer.daemon = True
    auto_lock_timer.start()

def apply_override(mode, persist=True):
    """Switches the operation mode and moves the flap accordingly."""
    global override_mode
    override_mode = mode
    if persist:
        save_override(mode)
    cancel_auto_lock()
    with flap_action_lock:
        if mode == "open":
            flap.unlock_in()
        else:
            # auto starts from the safe locked state; the next clean cat unlocks again
            flap.lock_in()
    if mode == "auto" and processor is not None:
        processor.state_machine.last_state = None  # a cat already in front of the flap triggers a fresh decision
    log_event({"auto": "Mode: automatic (detection decides)",
               "open": "Mode: always open (detection only logs)",
               "locked": "Mode: always locked (detection only logs)"}[mode])

def on_access_state_changed(access_state):
    """Event listener triggered by the StateMachine when the global access decision changes."""
    decision = "clean cat" if access_state == AccessState.GRANTED else "cat WITH PREY"
    if override_mode != "auto":
        log_event(f"Detected {decision} (flap not moved, mode {override_mode})")
        return

    if access_state == AccessState.GRANTED:
        log_event("Clean cat detected, unlocking for 30 s")
        with flap_action_lock:
            flap.unlock_in()
        schedule_auto_lock()

    elif access_state == AccessState.DENIED:
        log_event("Cat WITH PREY detected, locking")
        # Cancel any pending auto-lock since we are locking immediately
        cancel_auto_lock()
        with flap_action_lock:
            flap.lock_in()

IDLE_INTERVAL = 0.2  # seconds between motion checks while idle (~5 Hz)

def camera_loop(save_uncertain_dir, use_tracking=True, history_length=10, prey_threshold=0.3,
                motion_gate=True, motion_threshold=0.005, idle_after=5.0, idle_probe=2.0):
    """Background thread that continuously processes frames from the Catflap.

    With motion_gate, the detection pipeline only runs while something moves (or a cat is still tracked);
    otherwise only the cheap MotionDetector looks at ~5 frames per second. Every idle_probe seconds one idle
    frame still goes through the full pipeline, so a cat that sits still (no motion) is found anyway.
    """
    global current_frame_mjpeg, current_frame_seq, current_mode, processor

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
            raw = frame.copy() if recorder is not None else None  # clips keep the frames without overlays

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
                    if recorder is not None:
                        if mode == MotionGate.ACTIVE:
                            recorder.start(reason)
                        else:
                            clip = recorder.stop()
                            if clip:
                                print(f"[Recorder] Saved {os.path.basename(clip)}")
                current_mode = mode
            elif recorder is not None and not recorder.recording:
                recorder.start("motion gate off")  # without the gate: one clip until --record-max-minutes

            if current_mode == MotionGate.ACTIVE and not processed:
                frame_idx += 1
                processor.process_frame(frame, frame_idx)
            elif current_mode == MotionGate.IDLE:
                cv2.putText(frame, "IDLE (motion watch)",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
            if current_mode == MotionGate.ACTIVE and recorder is not None:
                recorder.add(raw)
            if override_mode != "auto":
                label ="OVERRIDE: ALWAYS OPEN" if override_mode == "open" else "OVERRIDE: ALWAYS LOCKED"
                cv2.putText(frame, label, (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

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
        if recorder is not None:
            recorder.stop()
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
            override:
              type: string
              description: "auto, open or locked"
            decision:
              type: string
              description: "last access decision of the state machine (GRANTED, DENIED or null)"
            cats:
              type: array
              items:
                type: object
            events:
              type: array
              items:
                type: object
    """
    cats = []
    decision = None
    params = {}
    if processor is not None:
        sm = processor.state_machine
        cats = [{"id": tid, "state": t.current_state.value, "history": len(t.confidence_history)}
                for tid, t in list(sm.tracks.items())]
        decision = sm.last_state.value if sm.last_state else None
        params = {"history_length": sm.history_length, "prey_threshold": sm.threshold}
    return jsonify({"is_locked_in": flap.is_locked_in, "is_locked_out": flap.is_locked_out, "mode": current_mode,
                    "override": override_mode, "decision": decision, "cats": cats, "params": params,
                    "events": list(events)})

@app.route('/api/override', methods=['GET', 'POST'])
def override():
    """
    Get or set the operation mode
    ---
    parameters:
      - in: body
        name: body
        required: false
        schema:
          type: object
          properties:
            mode:
              type: string
              enum: [auto, open, locked]
              description: "auto = detection decides, open = always unlocked (cats get used to the flap), locked = always locked"
    responses:
      200:
        description: Current operation mode
      400:
        description: Unknown mode
    """
    if request.method == 'POST':
        mode = (request.get_json(silent=True) or {}).get("mode")
        if mode not in OVERRIDE_MODES:
            return jsonify({"error": f"mode must be one of {list(OVERRIDE_MODES)}"}), 400
        apply_override(mode)
    return jsonify({"override": override_mode, "is_locked_in": flap.is_locked_in})

@app.route('/api/recordings', methods=['GET'])
def list_recordings():
    """
    List the recorded detection clips (newest first)
    ---
    responses:
      200:
        description: Clips with size and decision (prey, clean, nocat)
    """
    if recorder is None:
        return jsonify({"enabled": False, "clips": [], "count": 0})
    clips = recorder.clips()
    return jsonify({
        "enabled": True,
        "count": len(clips),
        "total": human_size(recorder.total_bytes()),
        "limit": human_size(recorder.max_bytes),
        "dropbox": recorder.uploader.status if recorder.uploader is not None else None,
        "clips": [{"name": name, "size": human_size(size), "decision": decision,
                   "label": time.strftime("%d.%m. %H:%M:%S", time.localtime(mtime))}
                  for name, size, decision, mtime in clips[:30]],
    })

@app.route('/recordings/<path:name>')
def download_recording(name):
    """
    Download a recorded clip (.mp4) or its metadata (.json)
    ---
    parameters:
      - in: path
        name: name
        type: string
        required: true
    responses:
      200:
        description: The file
    """
    if recorder is None or not (name.endswith(".mp4") or name.endswith(".json")):
        abort(404)
    return send_from_directory(recorder.directory, name, as_attachment=True)

@app.route('/api/lock', methods=['POST'])
def lock_flap():
    """
    Manually lock the cat flap
    ---
    responses:
      200:
        description: Flap locked successfully
    """
    cancel_auto_lock()
    with flap_action_lock:
        flap.lock_in()
    log_event("Locked manually")
    return jsonify({"status": "success", "is_locked_in": flap.is_locked_in})

@app.route('/api/unlock', methods=['POST'])
def unlock_flap():
    """
    Manually unlock the cat flap (will auto-close after 30s unless the mode is 'open')
    ---
    responses:
      200:
        description: Flap unlocked successfully
    """
    with flap_action_lock:
        flap.unlock_in()
    schedule_auto_lock()
    log_event("Unlocked manually for 30 s")
    return jsonify({"status": "success", "is_locked_in": flap.is_locked_in})

# --- Web UI ---

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
    return render_template("index.html")

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
    code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    parser.add_argument("--no-record", dest="record", action="store_false", help="Do not save a video clip per detection episode")
    parser.add_argument("--record-dir", type=str, default=os.path.join(code_dir, "recordings"), help="Folder for the recorded clips")
    parser.add_argument("--record-max-gb", type=float, default=5.0, help="Oldest clips are deleted above this folder size")
    parser.add_argument("--record-max-minutes", type=float, default=5.0, help="Maximum length of one clip")
    parser.add_argument("--dropbox-env", type=str, default=os.path.join(code_dir, "dropbox.env"),
                        help="File with DROPBOX_APP_KEY / DROPBOX_APP_SECRET / DROPBOX_REFRESH_TOKEN; upload is off without it")
    args = parser.parse_args()

    if args.record:
        uploader = DropboxUploader(args.dropbox_env)
        recorder = ClipRecorder(os.path.abspath(args.record_dir), max_bytes=int(args.record_max_gb * 1024 ** 3),
                                max_seconds=args.record_max_minutes * 60, uploader=uploader)
        print(f"[Recorder] Clips in {recorder.directory} (max {args.record_max_gb} GB), Dropbox: {uploader.status}")
    
    # Initialize the central Catflap hardware
    try:
        if args.simulate_servo:
            flap = SimulationCatflap(video_source=args.video, flip=args.flip)
        else:
            flap = ServoCatflap(gpio_pin=args.gpio, video_source=args.video, flip=args.flip)
    except RuntimeError as e:
        print(e)
        exit(1)

    # Restore the operation mode from the last run (e.g. 'open' while the cats get used to the flap)
    saved_mode = load_override()
    if saved_mode != "auto":
        apply_override(saved_mode, persist=False)
    else:
        log_event("Mode: automatic (detection decides)")

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
