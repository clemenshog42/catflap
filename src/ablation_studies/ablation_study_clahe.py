import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import glob
from processor import CatFlapProcessor
from state_machine import StateMachine, AccessState
from ultralytics import YOLO

def run_evaluation(video_path, use_clahe):
    processor = CatFlapProcessor()
    
    if use_clahe:
        processor.pipeline.detector = YOLO(r"C:\Public\Studium\Bachelorarbeit\models\cat_face_clahe.pt")
        processor.pipeline.classifier = YOLO(r"C:\Public\Studium\Bachelorarbeit\models\prey_clahe.pt")
        processor.pipeline.apply_clahe_detector = True
        processor.pipeline.apply_clahe_classifier = True
    else:
        processor.pipeline.detector = YOLO(r"C:\Public\Studium\Bachelorarbeit\models\cat_face.pt")
        processor.pipeline.classifier = YOLO(r"C:\Public\Studium\Bachelorarbeit\models\prey.pt")
        processor.pipeline.apply_clahe_detector = False
        processor.pipeline.apply_clahe_classifier = False
        
    # FORCE history_length=1 to isolate the effect of the classifier
    processor.state_machine = StateMachine(history_length=1, threshold=0.8, max_missed_frames=30)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Fehler: Konnte Video nicht öffnen: {video_path}")
        return None
        
    state_changes = 0
    denied_triggered = False
    
    def on_state_change(state):
        nonlocal state_changes, denied_triggered
        state_changes += 1
        if state == AccessState.DENIED:
            denied_triggered = True
            
    processor.state_machine.subscribe(on_state_change)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        processor.process_frame(frame, frame_idx)
        frame_idx += 1
        
    cap.release()
    
    return {
        "state_changes": state_changes,
        "denied_triggered": denied_triggered,
        "total_frames": frame_idx
    }

def main():
    base_dir = r"C:\Public\Studium\Bachelorarbeit\videos"
    categories = ["no_prey", "with_prey"]
    results_file = r"C:\Public\Studium\Bachelorarbeit\ablation_clahe_results.txt"
    
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("Ablationsstudie: Bedeutung des CLAHE-Filters\n")
        f.write("="*60 + "\n\n")
        
        for category in categories:
            f.write(f"KATEGORIE: {category.upper()}\n")
            f.write("-" * 60 + "\n")
            
            video_files = glob.glob(os.path.join(base_dir, category, "*_gray.mp4"))
            for video in video_files:
                video_name = os.path.basename(video)
                print(f"Verarbeite {video_name} in Kategorie {category}...")
                
                # Run with CLAHE (Baseline)
                res_true = run_evaluation(video, True)
                
                # Run without CLAHE (Colour Ablation)
                res_false = run_evaluation(video, False)
                
                if res_true is None or res_false is None:
                    continue
                
                f.write(f"Video: {video_name} ({res_true['total_frames']} Frames)\n")
                f.write(f"  Baseline (Mit CLAHE):\n")
                f.write(f"    - Anzahl Zustandswechsel: {res_true['state_changes']}\n")
                f.write(f"    - Klappe verriegelt: {'Ja' if res_true['denied_triggered'] else 'Nein'}\n")
                f.write(f"  Ablation (Ohne CLAHE):\n")
                f.write(f"    - Anzahl Zustandswechsel: {res_false['state_changes']}\n")
                f.write(f"    - Klappe verriegelt: {'Ja' if res_false['denied_triggered'] else 'Nein'}\n")
                f.write("\n")
                
    print(f"\nFertig! Auswertung gespeichert in: {results_file}")

if __name__ == "__main__":
    main()





