import cv2
import argparse
import threading
import time
from flask import Flask, Response, render_template_string, jsonify
from flasgger import Swagger

from processor import CatFlapProcessor
from catflap import SimulationCatflap, ServoCatflap
from state_machine import State, AccessState

app = Flask(__name__)
swagger = Swagger(app)

# Global variables for the background pipeline
pipeline_thread = None
current_frame_mjpeg = None
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

def camera_loop(save_uncertain_dir):
    """Background thread that continuously processes frames from the Catflap."""
    global current_frame_mjpeg
    
    processor = CatFlapProcessor(save_uncertain_dir=save_uncertain_dir)
    # Subscribe to state changes to trigger the flap hardware
    processor.state_machine.subscribe(on_access_state_changed)
    
    frame_idx = 0
    try:
        # Loop continuously over frames from the smart flap hardware
        for frame in flap.capture_continuous():
            frame_idx += 1
            
            # Process the frame
            processor.process_frame(frame, frame_idx)
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                with frame_lock:
                    current_frame_mjpeg = buffer.tobytes()
                    
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
            is_locked:
              type: boolean
    """
    return jsonify({"is_locked_in": flap.is_locked_in, "is_locked_out": flap.is_locked_out})

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
    while True:
        with frame_lock:
            frame_data = current_frame_mjpeg
        
        if frame_data is not None:
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
        args=(args.save_uncertain if args.save_uncertain else None,),
        daemon=True # Daemon thread dies when the main Flask app dies
    )
    pipeline_thread.start()
    
    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"Swagger documentation available at http://{args.host}:{args.port}/apidocs")
    
    app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)
