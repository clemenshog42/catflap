import time
import cv2

try:
    from gpiozero import Servo
except ImportError:
    print("Warning: gpiozero library not found. Servo control will not be available.")
    Servo = None

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

class Catflap:
    """
    Base interface representing the physical Cat Flap hardware.
    Handles the camera initialization and defines the locking interface.
    """
    
    def __init__(self, video_source=None, flip=False):
        self.flip = flip
        self.is_locked_in = False
        self.is_locked_out = False
        
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
        if self.picam2 is not None:
            while True:
                try:
                    image = self.picam2.capture_array("main")
                    if self.flip:
                        image = cv2.flip(image, -1)
                    yield image
                except Exception as e:
                    print(f"[Catflap Sensor] Camera stopped: {e}")
                    break
        elif self.cap is not None:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("[Catflap Sensor] End of video stream reached.")
                    break
                if self.flip:
                    frame = cv2.flip(frame, -1)
                yield frame

    def get_video_properties(self):
        if self.cap is not None:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            return width, height, fps
        return 640, 480, 30.0

    def close(self):
        if self.picam2 is not None:
            self.picam2.stop()
        if self.cap is not None:
            self.cap.release()

    # --- Hardware Interface Methods ---
    def lock_in(self):
        raise NotImplementedError
        
    def unlock_in(self):
        raise NotImplementedError
        
    def lock_out(self):
        raise NotImplementedError
        
    def unlock_out(self):
        raise NotImplementedError


class SimulationCatflap(Catflap):
    """A simulated cat flap that just prints state changes."""
    def __init__(self, video_source=None, flip=False):
        super().__init__(video_source, flip)
        self.unlock_in()
        self.unlock_out()

    def lock_in(self):
        print("[Simulated Flap] 🔒 Entry LOCKED (Prey detected)")
        self.is_locked_in = True
        
    def unlock_in(self):
        print("[Simulated Flap] 🔓 Entry UNLOCKED (Clean cat)")
        self.is_locked_in = False
        
    def lock_out(self):
        print("[Simulated Flap] 🔒 Exit LOCKED")
        self.is_locked_out = True
        
    def unlock_out(self):
        print("[Simulated Flap] 🔓 Exit UNLOCKED")
        self.is_locked_out = False


class ServoCatflap(Catflap):
    """A simple DIY cat flap with a single servo blocking the door."""
    def __init__(self, gpio_pin=17, lock_value=-1.0, unlock_value=1.0, video_source=None, flip=False):
        super().__init__(video_source, flip)
        self.lock_value = lock_value
        self.unlock_value = unlock_value
        
        if Servo is None:
            print("WARNING: gpiozero not installed, servo won't move.")
            self.servo = None
        else:
            self.servo = Servo(gpio_pin)
            
        self.unlock_in()

    def _move_servo(self, value):
        if self.servo is not None:
            self.servo.value = value
            time.sleep(0.5)
            self.servo.value = None

    def lock_in(self):
        if not self.is_locked_in:
            self._move_servo(self.lock_value)
            self.is_locked_in = True
            
    def unlock_in(self):
        if self.is_locked_in or self.is_locked_in is False:
            self._move_servo(self.unlock_value)
            self.is_locked_in = False
            
    def lock_out(self):
        # A simple single-servo flap blocks both ways usually, or is mechanically entry-only.
        print("Warning: lock_out not mechanically supported on basic ServoCatflap.")
        self.is_locked_out = True
        
    def unlock_out(self):
        self.is_locked_out = False


class SureCatflap(Catflap):
    """Advanced integration with a Sure Petcare API flap (Dual Lock)."""
    def __init__(self, video_source=None, flip=False):
        super().__init__(video_source, flip)
        # TODO: Initialize SureFlap API authentication here
        self.unlock_in()
        self.unlock_out()

    def lock_in(self):
        print("[SureFlap API] Sending lock_in command...")
        self.is_locked_in = True
        
    def unlock_in(self):
        print("[SureFlap API] Sending unlock_in command...")
        self.is_locked_in = False
        
    def lock_out(self):
        print("[SureFlap API] Sending lock_out command...")
        self.is_locked_out = True
        
    def unlock_out(self):
        print("[SureFlap API] Sending unlock_out command...")
        self.is_locked_out = False
