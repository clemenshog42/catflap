import cv2
import os
from models import CatFlapPipeline
from state_machine import StateMachine, State

def draw_info(frame, box, track_id, state, prey_conf, cat_conf):
    """Draws bounding box, ID, State, and Confidence on the frame."""
    x1, y1, x2, y2 = map(int, box)
    
    # Choose color based on state
    if state == State.CAT_WITH_PREY:
        color = (0, 0, 255) # Red for prey
    elif state == State.CAT_NO_PREY:
        color = (0, 255, 0) # Green for no prey
    elif state == State.GATHERING_HISTORY:
        color = (0, 165, 255) # Orange for gathering history
    else:
        color = (255, 255, 255)
        
    # Draw Bounding Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw Label
    label = f"ID: {track_id} | {state.value} | Cat: {cat_conf:.2f} | Prey: {prey_conf:.2f}"
    
    # Background for text
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
    
    # Text
    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

class CatFlapProcessor:
    def __init__(self, save_uncertain_dir=None, use_tracking=True):
        """Initializes the Cat Flap models, state machine, and configuration.

        use_tracking: True  -> ByteTrack assigns an ID per cat, each cat gets its own confidence history.
                      False -> all cats share one global history (Global Track ID 1), e.g. for very low FPS.
        """
        self.use_tracking = use_tracking
        self.pipeline = CatFlapPipeline(
            detector_path="models/cat_face.tflite",
            classifier_path="models/cat_prey.tflite",
            apply_clahe_detector=False,
            apply_clahe_classifier=False,
            color_mode_detector="grayscale",
            color_mode_classifier="grayscale"
        )
        self.state_machine = StateMachine(history_length=5, threshold=0.5, max_missed_frames=30)
        
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
        results = self.pipeline.run_detector(frame, track=self.use_tracking)
        
        if results and results.boxes:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            if self.use_tracking:
                # ByteTrack only reports IDs for confirmed tracks; unconfirmed boxes are skipped this frame
                track_ids = results.boxes.id.int().cpu().tolist() if results.boxes.id is not None else [None] * len(boxes)
            else:
                track_ids = [1] * len(boxes)  # Global Track ID 1: all cats share one history
            
            prey_results = []
            
            for box, cat_conf, track_id in zip(boxes, confidences, track_ids):
                if track_id is None:
                    continue
                # 3. Run prey segmentation on the (bottom-extended) face crop
                prey_confidence, crop_img = self.pipeline.run_prey_detector(frame, box)
                if crop_img is not None:
                    self.latest_crop = crop_img
                prey_results.append((box, cat_conf, prey_confidence, track_id))
                
            # 4. Update State Machine
            if self.use_tracking:
                # One update per tracked cat -> separate confidence history per cat
                for box, cat_conf, prey_confidence, track_id in prey_results:
                    self.state_machine.update(track_id, prey_confidence, frame_idx)
            elif prey_results:
                # One update for the whole frame (highest prey confidence of all cats)
                frame_max_prey_conf = max(p[2] for p in prey_results)
                self.state_machine.update(1, frame_max_prey_conf, frame_idx)
            
            for box, cat_conf, prey_confidence, track_id in prey_results:
                current_state = self.state_machine.get_state(track_id)
                history_len = len(self.state_machine.tracks[track_id].confidence_history) if track_id in self.state_machine.tracks else 0
                print(f"[Frame {frame_idx}] Cat {track_id} | Prey Conf: {prey_confidence:.2f} | State: {current_state.value} | History: {history_len}/{self.state_machine.history_length}")
                
                # Hard Negative Mining (per track, at most every 30 frames)
                if self.save_uncertain_dir and 0.3 <= prey_confidence <= 0.6:
                    if track_id not in self.last_saved_frame or (frame_idx - self.last_saved_frame[track_id]) > 30:
                        filename = os.path.join(self.save_uncertain_dir, f"uncertain_f{frame_idx}_id{track_id}_conf{prey_confidence:.2f}.jpg")
                        cv2.imwrite(filename, pristine_frame)
                        self.last_saved_frame[track_id] = frame_idx
                
                # Draw results on frame
                draw_info(frame, box, track_id, current_state, prey_confidence, cat_conf)
        else:
            # If we want to debug silent frames (Optional, can be spammy)
            if frame_idx % 10 == 0:
                print(f"[Frame {frame_idx}] No cat detected in this frame.")
                
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
