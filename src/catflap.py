import time
import cv2

try:
    from gpiozero import Servo
except ImportError:
    print("Warning: gpiozero library not found. Catflap servo control will run in simulation mode.")
    Servo = None

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

class Catflap:
    """
    Central object representing the physical Cat Flap hardware.
    Encapsulates both the locking mechanism (Servo) and the visual sensor (Camera/Video).
    """
    
    def __init__(self, gpio_pin=17, lock_value=-1.0, unlock_value=1.0, video_source=None, simulate_servo=False):
        """
        Initialize the Catflap.
        If video_source is None, it attempts to use the physical Raspberry Pi Camera (Picamera2).
        If video_source is a path (str) or integer, it uses cv2.VideoCapture (for mp4 files or webcams).
        If simulate_servo is True, it skips hardware PWM initialization and only prints lock states.
        """
        # --- Servo Lock Initialization ---
        self.gpio_pin = gpio_pin
        self.lock_value = lock_value
        self.unlock_value = unlock_value
        self.is_locked = False
        
        if Servo is not None and not simulate_servo:
            self.servo = Servo(gpio_pin)
            self.unlock() 
        else:
            self.servo = None
            mode = "Forced Simulation" if simulate_servo else "Missing Library Simulation"
            print(f"[{mode}] Lock initialized on GPIO {gpio_pin}")
            self.unlock()
            
        # --- Camera Sensor Initialization ---
        self.video_source = video_source
        self.picam2 = None
        self.cap = None
        
        if self.video_source is None:
            if Picamera2 is None:
                raise RuntimeError("Error: Picamera2 is not installed and no alternative video_source was provided.")
            print("[Catflap Sensor] Initializing physical Picamera2...")
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(main={"size": (640, 480)})
            self.picam2.configure(config)
            self.picam2.start()
        else:
            print(f"[Catflap Sensor] Opening video source: {self.video_source}")
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Error: Could not open video source {self.video_source}")

    def capture_continuous(self):
        """
        Generator that yields frames continuously from the active sensor (Camera or Video file).
        Automatically handles differences between Picamera2 streams and OpenCV video reading.
        """
        if self.picam2 is not None:
            # Native Picamera2 hardware loop
            for request in self.picam2.capture_continuous(self.picam2.make_array("main")):
                yield request.array
        elif self.cap is not None:
            # Standard OpenCV video/webcam loop
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("[Catflap Sensor] End of video stream reached.")
                    break
                yield frame

    def get_video_properties(self):
        """Returns the (width, height, fps) of the active sensor."""
        if self.cap is not None:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            return width, height, fps
        return 640, 480, 30.0

    def close(self):
        """Safely shuts down the camera hardware and releases resources."""
        if self.picam2 is not None:
            self.picam2.stop()
        if self.cap is not None:
            self.cap.release()

    # --- Servo Methods ---
            
    def lock(self):
        """Locks the cat flap by rotating the servo to the lock_value."""
        if not self.is_locked:
            if self.servo is not None:
                self.servo.value = self.lock_value
                time.sleep(0.5) # Give the servo 500ms to physically move
                self.servo.value = None # Stop sending PWM signal to prevent jitter/buzzing
            else:
                print("[Simulated Catflap] 🔒 Flap LOCKED")
            self.is_locked = True
            
    def unlock(self):
        """Unlocks the cat flap by rotating the servo to the unlock_value."""
        if self.is_locked or self.is_locked is False: # Trigger on init too
            if self.servo is not None:
                self.servo.value = self.unlock_value
                time.sleep(0.5) # Give the servo 500ms to physically move
                self.servo.value = None # Stop sending PWM signal to prevent jitter/buzzing
            else:
                print("[Simulated Catflap] 🔓 Flap UNLOCKED")
            self.is_locked = False
