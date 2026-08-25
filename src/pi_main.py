import cv2
import argparse
import os
import numpy as np

# Try importing picamera2. If not on a Pi, fail gracefully.
try:
    from picamera2 import Picamera2
except ImportError:
    print("Error: picamera2 library is not installed. Are you running this on a Raspberry Pi?")
    exit(1)

from models import CatFlapPipeline
from state_machine import StateMachine, State

def draw_info(frame, box, track_id, state, prey_conf):
    """Draws bounding box, ID, State, and Confidence on the frame."""
    x1, y1, x2, y2 = map(int, box)
    
    # Choose color based on state
    if state == State.CAT_WITH_PREY:
        color = (0, 0, 255) # Red for prey
    elif state == State.CAT_NO_PREY:
        color = (0, 255, 0) # Green for no prey
    else:
        color = (255, 255, 255)
        
    # Draw Bounding Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw Label
    label = f"ID: {track_id} | {state.value} | Prey: {prey_conf:.2f}"
    
    # Background for text
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
    
    # Text
    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

def main(save_uncertain_dir=None, headless=False):
    # Initialize pipeline
    pipeline = CatFlapPipeline(
        detector_path="models/face_gray_float16.tflite",
        classifier_path="models/best_prey_23_07_V5_openvino_model",
        apply_clahe_detector=True,
        apply_clahe_classifier=True
    )
    
    # Initialize state machine
    state_machine = StateMachine(history_length=15, threshold=0.8, max_missed_frames=30)
    
    if save_uncertain_dir:
        os.makedirs(save_uncertain_dir, exist_ok=True)
        print(f"Hard Negative Mining enabled. Saving uncertain frames to {save_uncertain_dir}")
    
    last_saved_frame = {}
    
    # Initialize Picamera2
    print("Initializing Picamera2...")
    picam2 = Picamera2()
    
    # Configure the camera (adjust resolution and framerate as needed for the Pi)
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    
    print("Camera started. Press Ctrl+C to exit if running headless, or 'q' if viewing window.")
    
    frame_idx = 0
    try:
        # Loop continuously capturing frames from the camera array
        for request in picam2.capture_continuous(picam2.make_array("main")):
            # Extract the numpy array (the BGR frame)
            frame = request.array
            frame_idx += 1
            
            # Save a clean copy of the frame before bounding boxes
            pristine_frame = frame.copy()
            
            # 1 & 2. Run Object Detection and Tracking (ByteTrack)
            results = pipeline.run_detector(frame)
            
            if results and results.boxes:
                boxes = results.boxes.xyxy.cpu().numpy()
                if results.boxes.id is not None:
                    track_ids = results.boxes.id.int().cpu().tolist()
                else:
                    track_ids = [0] * len(boxes)
                
                for box, track_id in zip(boxes, track_ids):
                    # 3. Run Classification on the crop
                    prey_confidence = pipeline.run_classifier(frame, box)
                    
                    # 4. Update State Machine
                    current_state = state_machine.update(track_id, prey_confidence, frame_idx)
                    
                    # Hard Negative Mining: Save uncertain frames
                    if save_uncertain_dir and 0.15 <= prey_confidence <= 0.6:
                        if track_id not in last_saved_frame or (frame_idx - last_saved_frame[track_id]) > 30:
                            filename = os.path.join(save_uncertain_dir, f"uncertain_id{track_id}_f{frame_idx}_conf{prey_confidence:.2f}.jpg")
                            cv2.imwrite(filename, pristine_frame)
                            last_saved_frame[track_id] = frame_idx
                    
                    # Draw results on frame
                    draw_info(frame, box, track_id, current_state, prey_confidence)
                    
                    # Example: Trigger hardware flap lock if state == CAT_WITH_PREY
                    # if current_state == State.CAT_WITH_PREY:
                    #     lock_flap()
                    
            # Clean up stale tracks
            state_machine.cleanup_stale_tracks(frame_idx)
            
            if not headless:
                # Display the frame
                cv2.imshow("Cat Flap Prey Detection (PiCamera2)", frame)
                
                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        picam2.stop()
        if not headless:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raspberry Pi Picamera2 Cat Flap System")
    parser.add_argument("--save_uncertain", type=str, default="", help="Directory to save uncertain frames")
    parser.add_argument("--headless", action="store_true", help="Run without a display")
    args = parser.parse_args()
    
    main(args.save_uncertain if args.save_uncertain else None, args.headless)
