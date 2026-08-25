import cv2
import argparse
from processor import CatFlapProcessor

def main(video_path, save_uncertain_dir=None):
    processor = CatFlapProcessor(save_uncertain_dir=save_uncertain_dir)
    
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
        
        # Process the frame (handles detection, classification, state machine, and drawing)
        processor.process_frame(frame, frame_idx)
        
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
    parser.add_argument("--save_uncertain", type=str, default="", help="Directory to save uncertain frames")
    args = parser.parse_args()
    
    main(args.source, args.save_uncertain if args.save_uncertain else None)
