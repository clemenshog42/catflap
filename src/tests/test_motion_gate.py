import unittest
import numpy as np
from motion import MotionDetector, MotionGate


def scene(brightness=80, rect_x=None):
    """640x480 BGR frame with some texture and an optional bright 'cat' rectangle."""
    frame = np.full((480, 640, 3), brightness, dtype=np.uint8)
    frame[100:140, 50:600] = brightness + 40  # static structure (flap frame)
    if rect_x is not None:
        frame[200:360, rect_x:rect_x + 160] = 230
    return frame


class TestMotionDetector(unittest.TestCase):
    def test_static_scene_has_no_motion(self):
        md = MotionDetector()
        results = [md.has_motion(scene()) for _ in range(20)]
        self.assertFalse(any(results))

    def test_moving_object_is_motion(self):
        md = MotionDetector()
        for _ in range(5):
            md.has_motion(scene())
        self.assertTrue(md.has_motion(scene(rect_x=100)))

    def test_slow_brightness_ramp_is_no_motion(self):
        md = MotionDetector()
        results = [md.has_motion(scene(brightness=60 + i)) for i in range(60)]  # +1 grey level per frame
        self.assertFalse(any(results))

    def test_four_channel_camera_frame(self):
        md = MotionDetector()
        md.has_motion(np.zeros((480, 640, 4), dtype=np.uint8))
        frame = np.zeros((480, 640, 4), dtype=np.uint8)
        frame[100:300, 100:300] = 255
        self.assertTrue(md.has_motion(frame))


class TestMotionGate(unittest.TestCase):
    def test_starts_idle_and_wakes_on_motion(self):
        gate = MotionGate(idle_after=5.0)
        self.assertEqual(gate.mode, MotionGate.IDLE)
        mode, changed = gate.step(True, False, now=0.0)
        self.assertEqual(mode, MotionGate.ACTIVE)
        self.assertTrue(changed)

    def test_stays_active_while_cats_are_tracked(self):
        gate = MotionGate(idle_after=5.0)
        gate.step(True, False, now=0.0)
        # Cat sits still in front of the flap: no motion, but still tracked
        mode, _ = gate.step(False, True, now=60.0)
        self.assertEqual(mode, MotionGate.ACTIVE)

    def test_goes_idle_only_after_tracks_empty_and_quiet(self):
        gate = MotionGate(idle_after=5.0)
        gate.step(True, False, now=0.0)
        self.assertEqual(gate.step(False, False, now=4.9)[0], MotionGate.ACTIVE)
        mode, changed = gate.step(False, False, now=5.0)
        self.assertEqual(mode, MotionGate.IDLE)
        self.assertTrue(changed)

    def test_motion_resets_idle_timer(self):
        gate = MotionGate(idle_after=5.0)
        gate.step(True, False, now=0.0)
        gate.step(True, False, now=4.0)
        self.assertEqual(gate.step(False, False, now=8.0)[0], MotionGate.ACTIVE)
        self.assertEqual(gate.step(False, False, now=9.0)[0], MotionGate.IDLE)


if __name__ == '__main__':
    unittest.main()
