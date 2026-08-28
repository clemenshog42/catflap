import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
﻿import os
import cv2
import glob
from processor import CatFlapProcessor
from state_machine import StateMachine

def run_evaluation(video_path, use_asymmetric_crop, category):
    processor = CatFlapProcessor()
    
    # Toggle asymmetric crop
    processor.pipeline.use_asymmetric_crop = use_asymmetric_crop
    processor.state_machine = StateMachine(history_length=1, threshold=0.8, max_missed_frames=30)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Fehler: Konnte Video nicht öffnen: {video_path}")
        return None
        
    errors = 0
    total_frames_with_cat = 0
    
    # We will monkeypatch the run_classifier to intercept the confidence
    original_run_classifier = processor.pipeline.run_classifier
    def hooked_run_classifier(frame, box):
        conf, crop_img = original_run_classifier(frame, box)
        nonlocal errors, total_frames_with_cat
        total_frames_with_cat += 1
        
        if category == "no_prey":
            if conf >= 0.8:
                errors += 1
        else: # with_prey
            if conf < 0.8:
                errors += 1
                
        return conf, crop_img
        
    processor.pipeline.run_classifier = hooked_run_classifier

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        processor.process_frame(frame, frame_idx)
        
    cap.release()
    
    return {
        "errors": errors,
        "total_frames_with_cat": total_frames_with_cat,
        "total_frames": frame_idx
    }

def main():
    base_dir = r"C:\Public\Studium\Bachelorarbeit\videos"
    categories = ["no_prey", "with_prey"]
    results_file = r"C:\Public\Studium\Bachelorarbeit\ablation_crop_results.txt"
    
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("Ablationsstudie: Asymmetrisches Zuschneiden (True vs False)\n")
        f.write("="*60 + "\n\n")
        
        for category in categories:
            f.write(f"KATEGORIE: {category.upper()}\n")
            f.write("-" * 60 + "\n")
            
            video_files = glob.glob(os.path.join(base_dir, category, "*.mp4"))
            for video in video_files:
                video_name = os.path.basename(video)
                print(f"Verarbeite {video_name} in Kategorie {category}...")
                
                res_true = run_evaluation(video, use_asymmetric_crop=True, category=category)
                res_false = run_evaluation(video, use_asymmetric_crop=False, category=category)
                
                if res_true is None or res_false is None:
                    continue
                
                f.write(f"Video: {video_name} ({res_true['total_frames']} Frames)\n")
                f.write(f"  Mit asymmetrischem Cropping (Baseline):\n")
                f.write(f"    - Fehler: {res_true['errors']} / {res_true['total_frames_with_cat']} Katzen-Frames\n")
                
                f.write(f"  Symmetrisches Cropping (Ablation):\n")
                f.write(f"    - Fehler: {res_false['errors']} / {res_false['total_frames_with_cat']} Katzen-Frames\n")
                f.write("\n")
                
    print(f"\nFertig! Auswertung gespeichert in: {results_file}")

if __name__ == "__main__":
    main()
