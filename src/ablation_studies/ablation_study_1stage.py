import sys
import os
import cv2
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processor import CatFlapProcessor
from ultralytics import YOLO

def run_evaluation(video_path, category, processor, detector_1stage):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    baseline_prey_err = 0
    baseline_cat_miss = 0
    ablation_prey_err = 0
    ablation_cat_miss = 0
    total_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        total_frames += 1
        
        # 1. Baseline
        face_results = processor.pipeline.run_detector(frame)
        face_found = False
        baseline_prey_found = False
        
        if face_results is not None:
            for r in face_results:
                if len(r.boxes) > 0:
                    face_found = True
                    for box in r.boxes:
                        conf, _ = processor.pipeline.run_classifier(frame, box.xyxy[0])
                        if conf >= 0.5:
                            baseline_prey_found = True
                            
        if not face_found:
            baseline_cat_miss += 1
            
        if category == "no_prey" and baseline_prey_found:
            baseline_prey_err += 1
        elif category == "with_prey" and not baseline_prey_found:
            baseline_prey_err += 1
            
        # 2. Ablation
        res_1stage = detector_1stage(frame, conf=0.5, verbose=False)
        cat_found_1stage = False
        prey_found_1stage = False
        
        for r in res_1stage:
            if len(r.boxes) > 0:
                cat_found_1stage = True
                for box in r.boxes:
                    cls = int(box.cls[0].item())
                    if cls == 1:
                        prey_found_1stage = True
                        
        if not cat_found_1stage:
            ablation_cat_miss += 1
            
        if category == "no_prey" and prey_found_1stage:
            ablation_prey_err += 1
        elif category == "with_prey" and not prey_found_1stage:
            ablation_prey_err += 1
            
    cap.release()
    return {
        "total_frames": total_frames,
        "baseline_prey_err": baseline_prey_err,
        "baseline_cat_miss": baseline_cat_miss,
        "ablation_prey_err": ablation_prey_err,
        "ablation_cat_miss": ablation_cat_miss
    }

def main():
    base_dir = r"C:\Public\Studium\Bachelorarbeit\videos"
    categories = ["no_prey", "with_prey"]
    results_file = r"C:\Public\Studium\Bachelorarbeit\ablation_1stage_results.txt"
    
    print("Lade Modelle...")
    processor = CatFlapProcessor()
    processor.pipeline.detector = YOLO(r"C:\Public\Studium\Bachelorarbeit\models\cat_face_3108_colour.pt")
    processor.pipeline.classifier = YOLO(r"C:\Public\Studium\Bachelorarbeit\models\prey_3108_colour.pt")
    
    detector_1stage = YOLO(r"C:\Public\Studium\Bachelorarbeit\models\1stage_ablation.pt")
    
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("Ablationsstudie: 2-Stage (Baseline) vs 1-Stage (Detector)\n")
        f.write("="*60 + "\n\n")
        
        for category in categories:
            f.write(f"KATEGORIE: {category.upper()}\n")
            f.write("-" * 60 + "\n")
            
            video_files = glob.glob(os.path.join(base_dir, category, "*.mp4"))
            for video in video_files:
                video_name = os.path.basename(video)
                print(f"Verarbeite {video_name} in Kategorie {category}...")
                
                res = run_evaluation(video, category, processor, detector_1stage)
                
                if res is None:
                    continue
                
                f.write(f"Video: {video_name} ({res['total_frames']} Frames)\n")
                f.write(f"  Baseline (2-Stage):\n")
                f.write(f"    - Beute-Fehler: {res['baseline_prey_err']} / {res['total_frames']}\n")
                f.write(f"    - Katze übersehen: {res['baseline_cat_miss']} / {res['total_frames']}\n")
                f.write(f"  Ablation (1-Stage):\n")
                f.write(f"    - Beute-Fehler: {res['ablation_prey_err']} / {res['total_frames']}\n")
                f.write(f"    - Katze übersehen: {res['ablation_cat_miss']} / {res['total_frames']}\n")
                f.write("\n")
                
    print(f"\nFertig! Auswertung gespeichert in: {results_file}")

if __name__ == "__main__":
    main()


