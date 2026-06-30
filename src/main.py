import cv2
import argparse
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

def main(video_path):
    # Initialize pipeline with placeholder model paths
    pipeline = CatFlapPipeline(
        detector_path="models/face_gray_float16.tflite",
        classifier_path="models/cat_prey_classify_full_int_v1.tflite"
    )
    
    # Initialize state machine
    state_machine = StateMachine(history_length=15, threshold=0.2, max_missed_frames=30)
    
    # Open video capture (0 for webcam, or path to video file)
    cap = cv2.VideoCapture(video_path if video_path else 0)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_path}")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        
        # 1 & 2. Run Object Detection and Tracking (ByteTrack)
        results = pipeline.run_detector(frame)
        
        if results and results.boxes and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                # 3. Run Classification on the crop
                prey_confidence = pipeline.run_classifier(frame, box)
                
                # 4. Update State Machine
                current_state = state_machine.update(track_id, prey_confidence, frame_idx)
                
                # Draw results on frame
                draw_info(frame, box, track_id, current_state, prey_confidence)
                
        # Clean up stale tracks
        state_machine.cleanup_stale_tracks(frame_idx)
        
        # Display the frame
        cv2.imshow("Cat Flap Prey Detection", frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cat Flap Prey Detection System")
    parser.add_argument("--source", type=str, default="", help="Path to video file or camera index")
    args = parser.parse_args()
    
    main(args.source)
