"""Cheap motion gate for the Raspberry Pi server.

While nothing moves in front of the flap, the expensive cat/prey pipeline is not run; only this frame
difference check (well under 1 ms per frame) watches the scene. Not used by the offline evaluation.
"""
import cv2


class MotionDetector:
    """Frame difference against a slowly adapting background on a small, blurred greyscale image.

    min_area: share of changed pixels (0..1) that counts as motion.
    alpha:    adaptation rate of the background; slow light changes (IR switch, auto exposure) are absorbed.
    """

    def __init__(self, min_area=0.005, pixel_threshold=25, alpha=0.05, size=(160, 120)):
        self.min_area = min_area
        self.pixel_threshold = pixel_threshold
        self.alpha = alpha
        self.size = size
        self.background = None
        self.last_ratio = 0.0

    def _prepare(self, frame):
        small = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        if small.ndim == 3:
            gray = cv2.cvtColor(small, cv2.COLOR_BGRA2GRAY if small.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
        else:
            gray = small
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def has_motion(self, frame):
        gray = self._prepare(frame)
        if self.background is None:
            self.background = gray.astype("float32")
            self.last_ratio = 0.0
            return False
        diff = cv2.absdiff(gray, cv2.convertScaleAbs(self.background))
        _, mask = cv2.threshold(diff, self.pixel_threshold, 255, cv2.THRESH_BINARY)
        self.last_ratio = cv2.countNonZero(mask) / mask.size
        cv2.accumulateWeighted(gray, self.background, self.alpha)
        return self.last_ratio >= self.min_area


class MotionGate:
    """Decides whether the detection pipeline runs.

    idle -> active: as soon as motion is seen.
    active -> idle: only when the state machine tracks no cat any more AND there was no motion for idle_after
                    seconds, so a running decision (or a prey lock) is never cut short.
    """

    IDLE = "idle"
    ACTIVE = "active"

    def __init__(self, idle_after=5.0):
        self.idle_after = idle_after
        self.mode = self.IDLE
        self.last_motion = float("-inf")

    def step(self, motion, has_tracks, now):
        """Returns (mode, changed)."""
        previous = self.mode
        if motion:
            self.last_motion = now
            self.mode = self.ACTIVE
        elif self.mode == self.ACTIVE and not has_tracks and now - self.last_motion >= self.idle_after:
            self.mode = self.IDLE
        return self.mode, self.mode != previous
