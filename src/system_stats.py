"""Health of the Raspberry Pi for the web page: temperature, throttling, CPU, fan, memory, disk.

A background thread samples every few seconds and keeps a short history (temperature/CPU chart). Every value
is read defensively: on another machine (e.g. a Windows test run) missing values are simply None.
"""
import glob
import shutil
import subprocess
import threading
import time
from collections import deque

try:
    import psutil
except ImportError:
    psutil = None

TEMP_WARN = 70.0   # °C: getting warm, the fan should be running
TEMP_HOT = 80.0    # °C: the Pi 5 starts to reduce its clock (soft limit), 85 °C = hard limit

# Bits of `vcgencmd get_throttled`
THROTTLE_BITS = {
    0: "under-voltage", 1: "clock capped", 2: "throttled", 3: "soft temperature limit",
}


def _read(path, cast=float):
    try:
        with open(path) as f:
            return cast(f.read().strip())
    except (OSError, ValueError):
        return None


def _first(pattern):
    paths = glob.glob(pattern)
    return paths[0] if paths else None


def decode_throttled(value):
    """Returns (active now, occurred since boot) as lists of reasons."""
    if value is None:
        return None, None
    now = [name for bit, name in THROTTLE_BITS.items() if value & (1 << bit)]
    since_boot = [name for bit, name in THROTTLE_BITS.items() if value & (1 << (bit + 16))]
    return now, since_boot


def health(temp, throttled_now):
    """ok / warm / hot plus a short explanation for the page."""
    if throttled_now:
        return "hot", "Pi is limiting itself now: " + ", ".join(throttled_now)
    if temp is not None and temp >= TEMP_HOT:
        return "hot", f"Temperature at or above {TEMP_HOT:.0f} °C"
    if temp is not None and temp >= TEMP_WARN:
        return "warm", f"Temperature above {TEMP_WARN:.0f} °C"
    return "ok", "Within safe limits"


class SystemMonitor:
    def __init__(self, interval=5.0, history_minutes=30, disk_path="/"):
        self.interval = interval
        self.disk_path = disk_path
        self.history = deque(maxlen=max(1, int(history_minutes * 60 / interval)))
        self.latest = {}
        self.proc = psutil.Process() if psutil else None
        self.fan_dir = _first("/sys/devices/platform/cooling_fan/hwmon/*")
        if psutil:
            psutil.cpu_percent(None)  # first call only primes the counters
            self.proc.cpu_percent(None)
        self.sample()
        threading.Thread(target=self._run, daemon=True).start()

    def _throttled(self):
        try:
            out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2).stdout
            return int(out.strip().split("=")[1], 16)
        except Exception:
            return None

    def sample(self):
        temp = _read("/sys/class/thermal/thermal_zone0/temp")
        temp = temp / 1000 if temp is not None else None
        throttled = self._throttled()
        now, since_boot = decode_throttled(throttled)
        cores = psutil.cpu_count() if psutil else None
        freq = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
        freq_max = _read("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
        pwm = _read(f"{self.fan_dir}/pwm1") if self.fan_dir else None
        rpm = _read(f"{self.fan_dir}/fan1_input") if self.fan_dir else None
        level, reason = health(temp, now)
        s = {
            "time": time.time(),
            "temperature_c": round(temp, 1) if temp is not None else None,
            "health": level, "health_reason": reason,
            "throttled_now": now, "throttled_since_boot": since_boot,
            "cpu_total_pct": psutil.cpu_percent(None) if psutil else None,
            # catflap process, as share of the whole CPU (all cores)
            "cpu_process_pct": round(self.proc.cpu_percent(None) / cores, 1) if psutil and cores else None,
            "cpu_cores": cores,
            "load_1m": round(psutil.getloadavg()[0], 2) if psutil and hasattr(psutil, "getloadavg") else None,
            "cpu_mhz": round(freq / 1000) if freq else None,
            "cpu_max_mhz": round(freq_max / 1000) if freq_max else None,
            "fan_pct": round(pwm / 255 * 100) if pwm is not None else None,
            "fan_rpm": int(rpm) if rpm is not None else None,
            "memory_pct": psutil.virtual_memory().percent if psutil else None,
            "process_mb": round(self.proc.memory_info().rss / 1024 ** 2) if psutil else None,
            "uptime_s": int(time.time() - psutil.boot_time()) if psutil else None,
        }
        try:
            du = shutil.disk_usage(self.disk_path)
            s["disk_free_gb"] = round(du.free / 1024 ** 3, 1)
            s["disk_pct"] = round(du.used / du.total * 100)
        except OSError:
            s["disk_free_gb"] = s["disk_pct"] = None
        self.latest = s
        self.history.append((round(s["time"]), s["temperature_c"], s["cpu_total_pct"]))
        return s

    def _run(self):
        while True:
            time.sleep(self.interval)
            try:
                self.sample()
            except Exception as e:  # never let monitoring take the server down
                print(f"[System] Sampling failed: {e}")

    def snapshot(self):
        return dict(self.latest, history=list(self.history), temp_warn=TEMP_WARN, temp_hot=TEMP_HOT)
