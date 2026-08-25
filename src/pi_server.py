import cv2
import argparse
from flask import Flask, Response, render_template_string

try:
    from picamera2 import Picamera2
except ImportError:
    print("Error: picamera2 library is not installed. Are you running this on a Raspberry Pi?")
    exit(1)

from processor import CatFlapProcessor

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

def generate_frames(save_uncertain_dir=None):
    processor = CatFlapProcessor(save_uncertain_dir=save_uncertain_dir)
    
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
            
            # Process the frame
            processor.process_frame(frame, frame_idx)
            
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
