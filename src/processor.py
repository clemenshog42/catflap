import cv2
import os
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

class CatFlapProcessor:
    def __init__(self, save_uncertain_dir=None):
        """Initializes the Cat Flap models, state machine, and configuration."""
        self.pipeline = CatFlapPipeline(
            detector_path="models/face_gray_float16.tflite",
            classifier_path="models/prey_V7.pt",
            apply_clahe_detector=False,
            apply_clahe_classifier=True
        )
        self.state_machine = StateMachine(history_length=30, threshold=0.8, max_missed_frames=30)
        
        self.save_uncertain_dir = save_uncertain_dir
        self.last_saved_frame = {}
        self.latest_crop = None
        
        # FPS Tracking
        import time
        self.fps = 0.0
        self.last_time = time.time()
        self.frame_count = 0
        
        if save_uncertain_dir:
            os.makedirs(save_uncertain_dir, exist_ok=True)
            print(f"Hard Negative Mining enabled. Saving uncertain frames to {save_uncertain_dir}")

    def process_frame(self, frame, frame_idx):
        """
        Runs the full detection, classification, tracking, and drawing pipeline on a single frame.
        Modifies the frame in-place.
        """
        import time
        pristine_frame = frame.copy()
        
        # 1 & 2. Run Object Detection and Tracking
        results = self.pipeline.run_detector(frame)
        
        if results and results.boxes:
            boxes = results.boxes.xyxy.cpu().numpy()
            if results.boxes.id is not None:
                track_ids = results.boxes.id.int().cpu().tolist()
            else:
                track_ids = [0] * len(boxes)
            
            for box, track_id in zip(boxes, track_ids):
                # 3. Run Classification on the crop
                prey_confidence, crop_img = self.pipeline.run_classifier(frame, box)
                if crop_img is not None:
                    self.latest_crop = crop_img
                
                # 4. Update State Machine
                current_state = self.state_machine.update(track_id, prey_confidence, frame_idx)
                
                # Hard Negative Mining
                if self.save_uncertain_dir and 0.15 <= prey_confidence <= 0.6:
                    if track_id not in self.last_saved_frame or (frame_idx - self.last_saved_frame[track_id]) > 30:
                        filename = os.path.join(self.save_uncertain_dir, f"uncertain_id{track_id}_f{frame_idx}_conf{prey_confidence:.2f}.jpg")
                        cv2.imwrite(filename, pristine_frame)
                        self.last_saved_frame[track_id] = frame_idx
                
                # Draw results on frame
                draw_info(frame, box, track_id, current_state, prey_confidence)
                
        # Clean up stale tracks and evaluate global flap state
        self.state_machine.process_global_state(frame_idx)
        
        # Calculate and Draw FPS
        current_time = time.time()
        self.frame_count += 1
        elapsed = current_time - self.last_time
        
        # Update FPS string every 1 second
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = current_time
            
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
