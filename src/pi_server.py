import cv2
import argparse
import os
import numpy as np

try:
    from flask import Flask, Response, render_template_string
except ImportError:
    print("Error: flask library is not installed. Run 'pip install flask'")
    exit(1)

try:
    from picamera2 import Picamera2
except ImportError:
    print("Error: picamera2 library is not installed. Are you running this on a Raspberry Pi?")
    exit(1)

from models import CatFlapPipeline
from state_machine import StateMachine, State

app = Flask(__name__)

# Basic HTML template for the index page
INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Cat Flap Stream</title>
    <style>
        body { background-color: #333; color: white; font-family: Arial, sans-serif; text-align: center; }
        img { max-width: 100%; height: auto; border: 2px solid #555; }
    </style>
</head>
<body>
    <h1>Cat Flap Monitor</h1>
    <img src="{{ url_for('video_feed') }}">
</body>
</html>
"""

def draw_info(frame, box, track_id, state, prey_conf):
    x1, y1, x2, y2 = map(int, box)
    
    if state == State.CAT_WITH_PREY:
        color = (0, 0, 255)
    elif state == State.CAT_NO_PREY:
        color = (0, 255, 0)
    else:
        color = (255, 255, 255)
        
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID: {track_id} | {state.value} | Prey: {prey_conf:.2f}"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

def generate_frames(save_uncertain_dir=None):
    pipeline = CatFlapPipeline(
        detector_path="models/face_gray_float16.tflite",
        classifier_path="models/best_prey_23_07_V5_openvino_model",
        apply_clahe_detector=True,
        apply_clahe_classifier=True
    )
    
    state_machine = StateMachine(history_length=15, threshold=0.8, max_missed_frames=30)
    last_saved_frame = {}
    
    print("Initializing Picamera2...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    
    frame_idx = 0
    try:
        for request in picam2.capture_continuous(picam2.make_array("main")):
            frame = request.array
            frame_idx += 1
            pristine_frame = frame.copy()
            
            results = pipeline.run_detector(frame)
            
            if results and results.boxes:
                boxes = results.boxes.xyxy.cpu().numpy()
                if results.boxes.id is not None:
                    track_ids = results.boxes.id.int().cpu().tolist()
                else:
                    track_ids = [0] * len(boxes)
                
                for box, track_id in zip(boxes, track_ids):
                    prey_confidence = pipeline.run_classifier(frame, box)
                    current_state = state_machine.update(track_id, prey_confidence, frame_idx)
                    
                    if save_uncertain_dir and 0.15 <= prey_confidence <= 0.6:
                        if track_id not in last_saved_frame or (frame_idx - last_saved_frame[track_id]) > 30:
                            filename = os.path.join(save_uncertain_dir, f"uncertain_id{track_id}_f{frame_idx}_conf{prey_confidence:.2f}.jpg")
                            cv2.imwrite(filename, pristine_frame)
                            last_saved_frame[track_id] = frame_idx
                    
                    draw_info(frame, box, track_id, current_state, prey_confidence)
                    
            state_machine.cleanup_stale_tracks(frame_idx)
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            
            # Yield MJPEG frame format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
    finally:
        picam2.stop()

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(app.config.get('SAVE_UNCERTAIN_DIR')),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raspberry Pi Flask Stream Server")
    parser.add_argument("--save_uncertain", type=str, default="", help="Directory to save uncertain frames")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    args = parser.parse_args()
    
    app.config['SAVE_UNCERTAIN_DIR'] = args.save_uncertain if args.save_uncertain else None
    
    print(f"Starting server at http://{args.host}:{args.port}")
    # Disable reloader so it doesn't initialize the camera twice
    app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)
