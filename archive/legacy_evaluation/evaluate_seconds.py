import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import cv2
import os
import glob
from processor import CatFlapProcessor

def evaluate_videos():
    base_dir = r"C:\Public\Studium\Bachelorarbeit\code\videos"
    categories = ["no_prey", "with_prey"]
    
    total_evaluations = 0
    total_errors = 0
    
    print("Starte manuelle Evaluation. Das Video wird schnell abgespielt.")
    print("Das Bild pausiert jede Sekunde (alle 30 Frames) UND bei jedem Zustandswechsel!")
    print("Drücke 'j' (Ja), wenn der Zustand der Klappe KORREKT ist.")
    print("Drücke 'n' (Nein), wenn der Zustand FALSCH ist.")
    print("Drücke 'q' zum Abbrechen.")
    print("-" * 50)
    
    for category in categories:
        video_files = glob.glob(os.path.join(base_dir, category, "*.mp4"))
        video_files = [v for v in video_files if not v.endswith('_gray.mp4')]
        
        for video_path in video_files:
            print(f"\nEvaluiere Video: {os.path.basename(video_path)} ({category})")
            
            processor = CatFlapProcessor()
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"Fehler beim Öffnen: {video_path}")
                continue
                
            frame_idx = 0
            video_errors = 0
            video_evals = 0
            
            previous_state = False # assume unlocked at start
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                processor.process_frame(frame, frame_idx)
                from state_machine import AccessState
                current_state = (processor.state_machine.last_state == AccessState.DENIED)
                
                state_changed = (current_state != previous_state) and frame_idx > 0
                previous_state = current_state
                
                display_frame = frame.copy()
                state_text = "LOCKED (Beute)" if current_state else "UNLOCKED (Sicher)"
                color = (0, 0, 255) if current_state else (0, 255, 0)
                
                cv2.putText(display_frame, f"Frame: {frame_idx} | State: {state_text}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Ask user every 30 frames OR if state changed
                if frame_idx % 30 == 0 or state_changed:
                    reason = "ZUSTANDSWECHSEL!" if state_changed else "Regelmäßige Prüfung (1 Sek)"
                    cv2.putText(display_frame, f"Grund: {reason}", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(display_frame, "Ist das KORREKT? (j = Ja / n = Nein)", (10, 110), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    cv2.imshow("Evaluation", display_frame)
                    
                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord('j'):
                            video_evals += 1
                            break
                        elif key == ord('n'):
                            video_evals += 1
                            video_errors += 1
                            break
                        elif key == ord('q'):
                            cap.release()
                            cv2.destroyAllWindows()
                            return
                else:
                    cv2.imshow("Evaluation", display_frame)
                    cv2.waitKey(1)
                    
                frame_idx += 1
                
            cap.release()
            
            if video_evals > 0:
                error_rate = (video_errors / video_evals) * 100
                print(f"-> Ergebnis für {os.path.basename(video_path)}: {video_errors} Fehler bei {video_evals} Bewertungen.")
                print(f"-> Entspricht ca. {error_rate:.1f} Fehlern pro 100 Bewertungen (extrapoliert).")
                
                total_errors += video_errors
                total_evaluations += video_evals

    cv2.destroyAllWindows()
    
    if total_evaluations > 0:
        overall_error_rate = (total_errors / total_evaluations) * 100
        print("\n" + "="*50)
        print("GESAMTERGEBNIS DER EVALUATION:")
        print(f"Total Bewertungen (Sekunden + Wechsel): {total_evaluations}")
        print(f"Total Fehler gemacht: {total_errors}")
        print(f"-> Das System macht im Schnitt {overall_error_rate:.1f} Fehler pro 100 Situationen.")
        print("="*50)

if __name__ == '__main__':
    evaluate_videos()



