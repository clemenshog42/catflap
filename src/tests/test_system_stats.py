import unittest
from system_stats import decode_throttled, health, SystemMonitor


class TestSystemStats(unittest.TestCase):
    def test_decode_throttled(self):
        # 0xe0000: clock capped, throttled and soft temperature limit since boot, nothing right now
        now, since_boot = decode_throttled(0xe0000)
        self.assertEqual(now, [])
        self.assertEqual(since_boot, ["clock capped", "throttled", "soft temperature limit"])
        now, _ = decode_throttled(0x50005)
        self.assertEqual(now, ["under-voltage", "throttled"])
        self.assertEqual(decode_throttled(None), (None, None))

    def test_health_levels(self):
        self.assertEqual(health(52.0, [])[0], "ok")
        self.assertEqual(health(72.0, [])[0], "warm")
        self.assertEqual(health(81.0, [])[0], "hot")
        self.assertEqual(health(60.0, ["under-voltage"])[0], "hot")  # active throttling beats temperature
        self.assertEqual(health(None, None)[0], "ok")

    def test_snapshot_without_pi_hardware(self):
        # On a PC the Pi files and vcgencmd are missing: values become None, nothing raises
        s = SystemMonitor(interval=3600).snapshot()
        for key in ("temperature_c", "health", "cpu_total_pct", "history", "temp_hot"):
            self.assertIn(key, s)
        self.assertEqual(len(s["history"]), 1)


if __name__ == '__main__':
    unittest.main()
