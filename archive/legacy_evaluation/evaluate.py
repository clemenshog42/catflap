import os
import cv2
import time
from src.models import CatFlapPipeline
from src.state_machine import StateMachine, State

def evaluate_video(video_path, pipeline):
    state_machine = StateMachine(history_length=15, threshold=0.8, max_missed_frames=30)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, 0
        
    detected_prey = False
    frame_idx = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        results = pipeline.run_detector(frame)
        
        if results and results.boxes and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                prey_confidence = pipeline.run_classifier(frame, box)
                current_state = state_machine.update(track_id, prey_confidence, frame_idx)
                
                if current_state == State.CAT_WITH_PREY:
                    detected_prey = True
                    break # Break inner loop
                    
        state_machine.cleanup_stale_tracks(frame_idx)
        if detected_prey:
            break # Break outer loop since we already reached CAT_WITH_PREY
            
    cap.release()
    # Calculate FPS only if we processed frames
    process_time = time.time() - start_time
    fps = frame_idx / process_time if process_time > 0 and frame_idx > 0 else 0
    return detected_prey, fps

def run_evaluation():
    print("Loading models...")
    pipeline = CatFlapPipeline(
        detector_path="models/face_gray_float16.tflite",
        classifier_path="models/best_prey_30_06_V2_openvino_model"
    )
    
    results = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    total_fps = []
    
    # Eval with_prey
    with_prey_dir = "videos/with_prey"
    if os.path.exists(with_prey_dir):
        print("\n--- EVALUATING 'WITH PREY' VIDEOS ---")
        for f in os.listdir(with_prey_dir):
            if f.endswith(".mp4"):
                path = os.path.join(with_prey_dir, f)
                print(f"Evaluating {f}...")
                detected, fps = evaluate_video(path, pipeline)
                if fps > 0: total_fps.append(fps)
                if detected:
                    print(f"  -> TRUE POSITIVE (Correct!) [FPS: {fps:.1f}]")
                    results["TP"] += 1
                else:
                    print(f"  -> FALSE NEGATIVE (Missed!) [FPS: {fps:.1f}]")
                    results["FN"] += 1
                    
    # Eval no_prey
    no_prey_dir = "videos/no_prey"
    if os.path.exists(no_prey_dir):
        print("\n--- EVALUATING 'NO PREY' VIDEOS ---")
        for f in os.listdir(no_prey_dir):
            if f.endswith(".mp4"):
                path = os.path.join(no_prey_dir, f)
                print(f"Evaluating {f}...")
                detected, fps = evaluate_video(path, pipeline)
                if fps > 0: total_fps.append(fps)
                if detected:
                    print(f"  -> FALSE POSITIVE (False Alarm!) [FPS: {fps:.1f}]")
                    results["FP"] += 1
                else:
                    print(f"  -> TRUE NEGATIVE (Correct!) [FPS: {fps:.1f}]")
                    results["TN"] += 1
                    
    print("\n=== EVALUATION RESULTS ===")
    print(f"TP (Correctly blocked prey): {results['TP']}")
    print(f"FN (MISSED prey)           : {results['FN']}")
    print(f"FP (False alarms)          : {results['FP']}")
    print(f"TN (Correctly let in)      : {results['TN']}")
    
    total = results['TP'] + results['FN'] + results['FP'] + results['TN']
    if total > 0:
        recall = results['TP'] / (results['TP'] + results['FN']) if (results['TP'] + results['FN']) > 0 else 0
        precision = results['TP'] / (results['TP'] + results['FP']) if (results['TP'] + results['FP']) > 0 else 0
        print(f"\nRecall (Sensitivity): {recall*100:.1f}%")
        print(f"Precision           : {precision*100:.1f}%")
        
    if len(total_fps) > 0:
        print(f"\nAverage Processing FPS: {sum(total_fps)/len(total_fps):.2f}")

if __name__ == '__main__':
    run_evaluation()
