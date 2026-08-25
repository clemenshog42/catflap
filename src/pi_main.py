import cv2
import argparse
import numpy as np

# Try importing picamera2. If not on a Pi, fail gracefully.
try:
    from picamera2 import Picamera2
except ImportError:
    print("Warning: picamera2 library is not installed. Live camera mode will not work.")
    Picamera2 = None

from processor import CatFlapProcessor

def main(save_uncertain_dir=None, headless=False, video_path=None, output_path=None):
    processor = CatFlapProcessor(save_uncertain_dir=save_uncertain_dir)
    
    # SETUP VIDEO SOURCE
    if video_path:
        print(f"Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    else:
        if Picamera2 is None:
            print("Error: picamera2 is not installed. Please provide a --video file to run in video mode.")
            return
            
        print("Initializing Picamera2...")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (640, 480)})
        picam2.configure(config)
        picam2.start()
        
        width, height = 640, 480
        fps = 30.0
    
    # SETUP VIDEO WRITER
    writer = None
    if output_path:
        # Use mp4v codec for standard mp4 output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving processed video to: {output_path}")

    if not headless:
        print("Press 'q' in the video window to exit.")

    frame_idx = 0
    try:
        if video_path:
            # --- VIDEO FILE LOOP ---
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Finished processing video.")
                    break
                    
                frame_idx += 1
                processor.process_frame(frame, frame_idx)
                
                if writer:
                    writer.write(frame)
                    
                if not headless:
                    cv2.imshow("Cat Flap Prey Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        else:
            # --- PICAMERA2 LOOP ---
            for request in picam2.capture_continuous(picam2.make_array("main")):
                frame = request.array
                frame_idx += 1
                
                processor.process_frame(frame, frame_idx)
                
                if writer:
                    writer.write(frame)
                    
                if not headless:
                    cv2.imshow("Cat Flap Prey Detection (PiCamera2)", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if video_path:
            cap.release()
        else:
            picam2.stop()
            
        if writer:
            writer.release()
            print(f"Successfully saved {output_path}")
            
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raspberry Pi Picamera2 Cat Flap System (with Video Support)")
    parser.add_argument("--save_uncertain", type=str, default="", help="Directory to save uncertain frames")
    parser.add_argument("--headless", action="store_true", help="Run without a display")
    parser.add_argument("--video", type=str, default=None, help="Path to an input video file (overrides live camera)")
    parser.add_argument("--output", type=str, default=None, help="Path to save the output video (e.g., output.mp4)")
    args = parser.parse_args()
    
    main(
        save_uncertain_dir=args.save_uncertain if args.save_uncertain else None, 
        headless=args.headless,
        video_path=args.video,
        output_path=args.output
    )
