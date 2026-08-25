import cv2
import argparse
from processor import CatFlapProcessor
from catflap import Catflap

def main(save_uncertain_dir=None, headless=False, video_path=None, output_path=None, simulate_servo=False):
    processor = CatFlapProcessor(save_uncertain_dir=save_uncertain_dir)
    
    # Initialize the central Catflap hardware (handles both the lock and the camera/video source)
    try:
        flap = Catflap(video_source=video_path, simulate_servo=simulate_servo)
    except RuntimeError as e:
        print(e)
        return

    # Optional: Setup Video Writer to save output
    writer = None
    if output_path:
        width, height, fps = flap.get_video_properties()
        # Use mp4v codec for standard mp4 output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving processed video to: {output_path}")

    if not headless:
        print("Press 'q' in the video window to exit.")

    frame_idx = 0
    try:
        # Loop continuously over frames from the smart flap hardware
        for frame in flap.capture_continuous():
            frame_idx += 1
            
            processor.process_frame(frame, frame_idx)
            
            if writer:
                writer.write(frame)
                
            if not headless:
                cv2.imshow("Cat Flap Prey Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        flap.close()
            
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
    parser.add_argument("--simulate-servo", action="store_true", help="Run without initializing the physical servo motor")
    args = parser.parse_args()
    
    main(
        save_uncertain_dir=args.save_uncertain if args.save_uncertain else None, 
        headless=args.headless,
        video_path=args.video,
        output_path=args.output,
        simulate_servo=args.simulate_servo
    )
