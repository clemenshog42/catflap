import json
import os
import tempfile
import unittest

import numpy as np
from recorder import ClipRecorder, DropboxUploader


def frame(v=0, channels=3):
    return np.full((480, 640, channels), v, dtype=np.uint8)


class TestClipRecorder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_clip_with_prey_decision(self):
        rec = ClipRecorder(self.dir)
        rec.start("changed pixels 3.0%")
        for i in range(10):
            rec.add(frame(i * 20, channels=4))  # camera frames have 4 channels
        rec.note("Clean cat detected, unlocking for 30 s")
        rec.note("Cat WITH PREY detected, locking")
        path = rec.stop()
        self.assertTrue(path.endswith("_prey.mp4"))
        self.assertGreater(os.path.getsize(path), 0)
        with open(path[:-4] + ".json") as f:
            meta = json.load(f)
        self.assertEqual(meta["frames"], 10)
        self.assertEqual(meta["decision"], "prey")
        self.assertEqual(len(meta["events"]), 2)
        self.assertEqual(rec.clips()[0][2], "prey")

    def test_episode_without_cat(self):
        rec = ClipRecorder(self.dir)
        rec.start()
        rec.add(frame())
        self.assertTrue(rec.stop().endswith("_nocat.mp4"))

    def test_stop_without_frames_writes_nothing(self):
        rec = ClipRecorder(self.dir)
        rec.start()
        self.assertIsNone(rec.stop())
        self.assertEqual(os.listdir(self.dir), [])

    def test_size_limit_deletes_oldest(self):
        rec = ClipRecorder(self.dir)
        for name in ("20260101_000000_nocat", "20260101_000100_nocat", "20260101_000200_clean"):
            with open(os.path.join(self.dir, name + ".mp4"), "wb") as f:
                f.write(b"x" * 1000)
            with open(os.path.join(self.dir, name + ".json"), "w") as f:
                f.write("{}")
        for i, (fn, *_rest) in enumerate(sorted(rec.clips(), key=lambda c: c[0])):
            os.utime(os.path.join(self.dir, fn), (i, i))  # file names are in time order
        rec.max_bytes = 2100
        rec.enforce_limit()
        names = sorted(os.listdir(self.dir))
        self.assertNotIn("20260101_000000_nocat.mp4", names)
        self.assertNotIn("20260101_000000_nocat.json", names)
        self.assertIn("20260101_000200_clean.mp4", names)


class TestDropboxUploader(unittest.TestCase):
    def test_disabled_without_config(self):
        up = DropboxUploader(os.path.join(tempfile.gettempdir(), "does_not_exist.env"))
        self.assertFalse(up.enabled)
        self.assertEqual(up.status, "not configured")
        up.enqueue("anything.mp4")  # ignored, no error


if __name__ == '__main__':
    unittest.main()
