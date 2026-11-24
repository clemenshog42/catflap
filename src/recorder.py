"""Records one video clip per detection episode on the Pi (for later analysis) and optionally uploads it to Dropbox.

A clip starts when the motion gate switches to active and ends when it goes idle again. It contains the raw
camera frames (no overlays), so the clips can be run through the pipeline/evaluation again later. Next to each
clip, a .json file holds the decision and the events of the episode. The oldest clips are deleted when the
folder grows above max_bytes.
"""
import json
import os
import queue
import threading
import time

import cv2

CLIP_FPS = 5.0  # roughly the detection rate on the Pi; only affects playback speed


def human_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024


class ClipRecorder:
    def __init__(self, directory, max_bytes=5 * 1024 ** 3, max_seconds=300, uploader=None):
        self.directory = directory
        self.max_bytes = max_bytes
        self.max_frames = int(max_seconds * CLIP_FPS)  # safety cap for very long episodes
        self.uploader = uploader
        self.writer = None
        self.base = None
        self.frames = 0
        self.started = None
        self.meta = {}
        os.makedirs(directory, exist_ok=True)

    @property
    def recording(self):
        return self.writer is not None

    def start(self, reason=""):
        if self.recording:
            return
        self.started = time.time()
        self.base = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.started))
        self.frames = 0
        self.meta = {"start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started)),
                     "reason": reason, "events": [], "decisions": []}
        # The writer is opened with the first frame (frame size known then)

    def add(self, frame):
        if self.base is None or self.frames >= self.max_frames:
            return
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if self.writer is None:
            h, w = frame.shape[:2]
            path = os.path.join(self.directory, self.base + ".partial.mp4")
            self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), CLIP_FPS, (w, h))
        self.writer.write(frame)
        self.frames += 1

    def note(self, text):
        """Event during the episode (decision, flap movement, ...)."""
        if self.base is not None:
            self.meta["events"].append({"t": round(time.time() - self.started, 1), "text": text})
            if "prey" in text.lower():
                self.meta["decisions"].append("prey")
            elif "clean" in text.lower():
                self.meta["decisions"].append("clean")

    def stop(self):
        """Closes the clip; returns the final video path or None."""
        if self.base is None:
            return None
        base, writer = self.base, self.writer
        self.base, self.writer = None, None
        if writer is None:
            return None
        writer.release()
        decisions = self.meta["decisions"]
        decision = "prey" if "prey" in decisions else "clean" if "clean" in decisions else "nocat"
        self.meta.update({"decision": decision, "frames": self.frames,
                          "duration_s": round(time.time() - self.started, 1), "fps": CLIP_FPS})
        name = f"{base}_{decision}"
        final = os.path.join(self.directory, name + ".mp4")
        os.replace(os.path.join(self.directory, base + ".partial.mp4"), final)
        with open(os.path.join(self.directory, name + ".json"), "w") as f:
            json.dump(self.meta, f, indent=1)
        self.enforce_limit()
        if self.uploader is not None:
            self.uploader.enqueue(final)
            self.uploader.enqueue(os.path.join(self.directory, name + ".json"))
        return final

    def clips(self):
        """Finished clips, newest first: list of (name, size, decision, mtime)."""
        out = []
        for fn in os.listdir(self.directory):
            if fn.endswith(".mp4") and ".partial" not in fn:
                p = os.path.join(self.directory, fn)
                decision = fn[:-4].rsplit("_", 1)[-1]
                out.append((fn, os.path.getsize(p), decision, os.path.getmtime(p)))
        return sorted(out, key=lambda c: c[3], reverse=True)

    def total_bytes(self):
        return sum(os.path.getsize(os.path.join(self.directory, f)) for f in os.listdir(self.directory))

    def enforce_limit(self):
        """Deletes the oldest clips (and their .json) until the folder fits into max_bytes."""
        clips = sorted(self.clips(), key=lambda c: c[3])
        total = self.total_bytes()
        while total > self.max_bytes and len(clips) > 1:
            fn = clips.pop(0)[0]
            for f in (fn, fn[:-4] + ".json"):
                p = os.path.join(self.directory, f)
                if os.path.exists(p):
                    total -= os.path.getsize(p)
                    os.remove(p)


class DropboxUploader:
    """Uploads finished clips in a background thread (Dropbox HTTP API, no extra package).

    Configured by a file with three lines KEY=value (default ~/catflap/dropbox.env, not in git):
        DROPBOX_APP_KEY=...
        DROPBOX_APP_SECRET=...
        DROPBOX_REFRESH_TOKEN=...
    Files go to /<folder>/<file name> inside the app folder.
    """

    def __init__(self, env_file, folder="/catflap"):
        import requests  # installed with ultralytics
        self.requests = requests
        self.folder = folder.rstrip("/")
        self.cfg = self._read_env(env_file)
        self.enabled = all(k in self.cfg for k in ("DROPBOX_APP_KEY", "DROPBOX_APP_SECRET", "DROPBOX_REFRESH_TOKEN"))
        self.status = "not configured" if not self.enabled else "ready"
        self.token, self.token_expires = None, 0
        self.q = queue.Queue()
        if self.enabled:
            threading.Thread(target=self._run, daemon=True).start()

    @staticmethod
    def _read_env(path):
        cfg = {}
        try:
            with open(path) as f:
                for line in f:
                    if "=" in line and not line.strip().startswith("#"):
                        k, v = line.strip().split("=", 1)
                        cfg[k.strip()] = v.strip()
        except OSError:
            pass
        return cfg

    def enqueue(self, path):
        if self.enabled:
            self.q.put(path)

    def _access_token(self):
        if self.token and time.time() < self.token_expires - 60:
            return self.token
        r = self.requests.post("https://api.dropboxapi.com/oauth2/token", timeout=30, data={
            "grant_type": "refresh_token", "refresh_token": self.cfg["DROPBOX_REFRESH_TOKEN"],
            "client_id": self.cfg["DROPBOX_APP_KEY"], "client_secret": self.cfg["DROPBOX_APP_SECRET"]})
        r.raise_for_status()
        data = r.json()
        self.token, self.token_expires = data["access_token"], time.time() + data.get("expires_in", 14400)
        return self.token

    def _upload(self, path):
        with open(path, "rb") as f:
            body = f.read()
        r = self.requests.post("https://content.dropboxapi.com/2/files/upload", timeout=300, data=body, headers={
            "Authorization": f"Bearer {self._access_token()}",
            "Content-Type": "application/octet-stream",
            "Dropbox-API-Arg": json.dumps({"path": f"{self.folder}/{os.path.basename(path)}", "mode": "overwrite"})})
        r.raise_for_status()

    def _run(self):
        while True:
            path = self.q.get()
            for attempt in range(5):
                if not os.path.exists(path):
                    break  # deleted by the size limit in the meantime
                try:
                    self._upload(path)
                    self.status = f"last upload {time.strftime('%H:%M')}"
                    print(f"[Dropbox] Uploaded {os.path.basename(path)}")
                    break
                except Exception as e:
                    self.status = f"error ({type(e).__name__}), retrying"
                    print(f"[Dropbox] Upload of {os.path.basename(path)} failed: {e}")
                    time.sleep(30 * (attempt + 1))
