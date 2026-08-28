import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import cv2
import glob
from processor import CatFlapProcessor
from state_machine import StateMachine, AccessState

def run_evaluation(video_path, history_length):
    # Initialisiere den Processor
    processor = CatFlapProcessor()
    
    # Überschreibe den Zustandsautomaten mit der gewünschten history_length
    processor.state_machine = StateMachine(history_length=history_length, threshold=0.8, max_missed_frames=30)
    
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

    # Auf Statuswechsel-Event registrieren
    processor.state_machine.subscribe(on_state_change)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        # Frame verarbeiten (ohne cv2.imshow, damit es so schnell wie möglich durchläuft)
        processor.process_frame(frame, frame_idx)
        
    cap.release()
    
    return {
        "state_changes": state_changes,
        "denied_triggered": denied_triggered,
        "total_frames": frame_idx
    }

def main():
    base_dir = r"C:\Public\Studium\Bachelorarbeit\videos"
    categories = ["no_prey", "with_prey"]
    
    results_file = r"C:\Public\Studium\Bachelorarbeit\ablation_results.txt"
    
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("Ablationsstudie: Zustandsautomat (history_length 1 vs 15)\n")
        f.write("="*60 + "\n\n")
        
        for category in categories:
            f.write(f"KATEGORIE: {category.upper()}\n")
            f.write("-" * 60 + "\n")
            
            video_files = glob.glob(os.path.join(base_dir, category, "*.mp4"))
            
            if not video_files:
                f.write("Keine Videos in diesem Ordner gefunden.\n\n")
                continue
                
            for video in video_files:
                video_name = os.path.basename(video)
                print(f"Verarbeite {video_name} in Kategorie {category}...")
                
                # 1. Baseline (mit State Machine)
                res_15 = run_evaluation(video, history_length=15)
                
                # 2. Ablation (ohne zeitliche Glättung)
                res_1 = run_evaluation(video, history_length=1)
                
                if res_15 is None or res_1 is None:
                    continue
                    
                # Auswertung, ob die Entscheidung korrekt war:
                # no_prey -> Darf niemals DENIED auslösen
                # with_prey -> Muss zwingend DENIED auslösen
                if category == "no_prey":
                    correct_15 = not res_15["denied_triggered"]
                    correct_1 = not res_1["denied_triggered"]
                else:
                    correct_15 = res_15["denied_triggered"]
                    correct_1 = res_1["denied_triggered"]
                
                f.write(f"Video: {video_name} ({res_15['total_frames']} Frames)\n")
                f.write(f"  history_length=15 (Baseline):\n")
                f.write(f"    - Anzahl Zustandswechsel: {res_15['state_changes']}\n")
                f.write(f"    - Klappe gesperrt?:       {res_15['denied_triggered']} (Korrekt? {correct_15})\n")
                
                f.write(f"  history_length=1  (Ohne Glättung):\n")
                f.write(f"    - Anzahl Zustandswechsel: {res_1['state_changes']}\n")
                f.write(f"    - Klappe gesperrt?:       {res_1['denied_triggered']} (Korrekt? {correct_1})\n")
                f.write("\n")
                
    print(f"\nFertig! Die Auswertung wurde gespeichert in: {results_file}")

if __name__ == "__main__":
    main()
