import cv2
import argparse
import numpy as np

# Try importing picamera2. If not on a Pi, fail gracefully.
try:
    from picamera2 import Picamera2
except ImportError:
    print("Error: picamera2 library is not installed. Are you running this on a Raspberry Pi?")
    exit(1)

from processor import CatFlapProcessor

def main(save_uncertain_dir=None, headless=False):
    processor = CatFlapProcessor(save_uncertain_dir=save_uncertain_dir)
    
    # Initialize Picamera2
    print("Initializing Picamera2...")
    picam2 = Picamera2()
    
    # Configure the camera (adjust resolution and framerate as needed for the Pi)
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    
    print("Camera started. Press Ctrl+C to exit if running headless, or 'q' if viewing window.")
    
    frame_idx = 0
    try:
        # Loop continuously capturing frames from the camera array
        for request in picam2.capture_continuous(picam2.make_array("main")):
            frame = request.array
            frame_idx += 1
            
            # Process the frame (handles detection, classification, state machine, and drawing)
            processor.process_frame(frame, frame_idx)
            
            if not headless:
                # Display the frame
                cv2.imshow("Cat Flap Prey Detection (PiCamera2)", frame)
                
                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        picam2.stop()
        if not headless:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raspberry Pi Picamera2 Cat Flap System")
    parser.add_argument("--save_uncertain", type=str, default="", help="Directory to save uncertain frames")
    parser.add_argument("--headless", action="store_true", help="Run without a display")
    args = parser.parse_args()
    
    main(args.save_uncertain if args.save_uncertain else None, args.headless)
