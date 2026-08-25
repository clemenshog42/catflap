import time

try:
    from gpiozero import Servo
except ImportError:
    print("Warning: gpiozero library not found. Catflap servo control will run in simulation mode.")
    Servo = None

class Catflap:
    """Controls the physical locking mechanism of the cat flap using a servo motor."""
    
    def __init__(self, gpio_pin, lock_value=-1.0, unlock_value=1.0):
        """
        Initialize the Catflap with the GPIO pin connected to the servo.
        lock_value and unlock_value correspond to servo positions (-1.0 to 1.0).
        You may need to adjust these values based on how your physical servo is mounted.
        """
        self.gpio_pin = gpio_pin
        self.lock_value = lock_value
        self.unlock_value = unlock_value
        self.is_locked = False
        
        if Servo is not None:
            # We initialize the servo. For common SG90 servos, you might need to adjust 
            # min_pulse_width and max_pulse_width if it doesn't rotate a full 180 degrees.
            self.servo = Servo(gpio_pin)
            self.unlock() # Ensure it starts unlocked
        else:
            self.servo = None
            print(f"[Simulated Catflap] Initialized on GPIO {gpio_pin}")
            self.unlock()
            
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
