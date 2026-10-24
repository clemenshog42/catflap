"""Renders evaluation videos with the logged detections and the replayed state machine drawn in.

For every video of a run (results/raw/<run>/*.csv) an overlay video is written to results/videos/<run>/.
Drawn in: face boxes coloured by track state, current and averaged prey confidence per track,
the flap state of two variants (as described in the thesis, and with 'prey lock until the track leaves'),
the annotated prey state and a marker for every frame that counts as a safety or comfort error.

Usage: python3 src/tools/render_overlay.py [--run two_stage_track_grayscale] [--videos 01 06]
"""
import argparse
import glob
import os
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd

CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(CODE_DIR, "src"))
sys.path.insert(0, os.path.join(CODE_DIR, "src", "evaluation"))
from state_machine import StateMachine, TrackState, State, AccessState  # noqa: E402
import evaluate as ev  # noqa: E402


class StickyTrack(TrackState):
    """Variant: once a track was seen with prey it stays locked until it is removed."""
    def update(self, prey_confidence, frame_idx):
        if self.current_state == State.CAT_WITH_PREY:
            self.confidence_history.append(prey_confidence)
            self.last_seen = frame_idx
            return self.current_state
        return super().update(prey_confidence, frame_idx)


class StickyStateMachine(StateMachine):
    def update(self, track_id, prey_confidence, frame_idx):
        if track_id not in self.tracks:
            self.tracks[track_id] = StickyTrack(track_id, self.history_length, self.threshold)
        return self.tracks[track_id].update(prey_confidence, frame_idx)


def replay(df, sm):
    door = {"open": False}
    sm.subscribe(lambda s: door.__setitem__("open", s == AccessState.GRANTED))
    out = {}
    for i, ((fr, t), rows) in enumerate(df.groupby(["frame", "time_s"], sort=True)):
        dets = []
        for _, r in rows.iterrows():
            if pd.notna(r.track_id):
                tid = int(r.track_id)
                st = sm.update(tid, float(r.prey_conf), i)
                avg = float(np.mean(sm.tracks[tid].confidence_history))
                dets.append((tid, r.x1, r.y1, r.x2, r.y2, float(r.prey_conf), avg, st))
        sm.process_global_state(i, current_time=float(t))
        if len(sm.tracks) == 0:
            door["open"] = False
        out[int(fr)] = (float(t), dets, door["open"])
    return out


COL = {State.GATHERING_HISTORY: (0, 170, 255), State.CAT_NO_PREY: (60, 200, 60), State.CAT_WITH_PREY: (50, 50, 230)}
NAME = {State.GATHERING_HISTORY: "prueft", State.CAT_NO_PREY: "sauber", State.CAT_WITH_PREY: "Beute"}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def text(img, s, org, scale=0.6, col=(255, 255, 255), bg=None, th=2):
    (tw, tht), _ = cv2.getTextSize(s, FONT, scale, th)
    x, y = org
    if bg is not None:
        cv2.rectangle(img, (x - 6, y - tht - 8), (x + tw + 6, y + 8), bg, -1)
    cv2.putText(img, s, (x, y), FONT, scale, col, th, cv2.LINE_AA)
    return tw


def render(video_path, csv_path, out_path, h_base, t_base, h_var, t_var):
    df = pd.read_csv(csv_path)
    video = df["video"].iloc[0]
    gt = ev.load_gt(video)
    cats = gt["cats"]
    first_seen = min((iv[0] for c in cats for iv in c.get("face_visible", c["visible"])), default=0.0)
    base = replay(df, StateMachine(h_base, t_base, 30))
    var = replay(df, StickyStateMachine(h_var, t_var, 30))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = out_path + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    cur_b, cur_v = (0.0, [], False), (0.0, [], False)
    idx = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        cur_b = base.get(idx, cur_b)
        cur_v = var.get(idx, cur_v)
        t = idx / fps
        _, dets, open_b = cur_b
        open_v = cur_v[2]
        for tid, x1, y1, x2, y2, pc, avg, st in dets:
            c = COL.get(st, (200, 200, 200))
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(f, (x1, y1), (x2, y2), c, 3)
            text(f, f"Katze {tid}: {NAME.get(st, '')}", (x1 + 4, y1 - 34), 0.6, bg=c)
            text(f, f"Beute {pc:.2f}  Schnitt {avg:.2f}", (x1 + 4, y1 - 8), 0.55, bg=c)

        # ground truth and error marking (same rules as evaluate.score)
        visible = [c for c in cats if ev.in_any(t, c["visible"])]
        need_lock = any(ev.in_any(t, c.get("prey", [])) for c in visible)
        def err(is_open):
            if not visible:
                return ""
            if need_lock and is_open:
                return "  SICHERHEITSFEHLER"
            if not need_lock and not is_open and t - first_seen > 3.0:
                return "  KOMFORTFEHLER"
            return ""
        gt_txt = "Annotation: Beute" if need_lock else ("Annotation: keine Beute" if visible else "Annotation: keine Katze")

        y = 50
        for label, op in ((f"Wie in der Arbeit ({h_base}/{t_base:g})", open_b), (f"Mit Sperre ({h_var}/{t_var:g})", open_v)):
            s = f"{label}: {'OFFEN' if op else 'GESPERRT'}{err(op)}"
            text(f, s, (30, y), 0.75, bg=(60, 160, 60) if op else (50, 50, 200))
            y += 48
        text(f, gt_txt, (30, y), 0.7, bg=(40, 40, 40))
        text(f, f"t = {t:4.2f} s", (W - 190, 50), 0.8, bg=(40, 40, 40))
        vw.write(f)
        idx += 1
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "26", "-preset", "veryfast", out_path], check=True)
    os.remove(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="two_stage_track_grayscale")
    ap.add_argument("--videos", nargs="*", help="filter, e.g. with_prey__01 06_black")
    ap.add_argument("--base", nargs=2, type=float, default=[20, 0.3], metavar=("H", "T"))
    ap.add_argument("--variant", nargs=2, type=float, default=[20, 0.25], metavar=("H", "T"))
    a = ap.parse_args()
    out_dir = os.path.join(CODE_DIR, "results", "videos", a.run)
    os.makedirs(out_dir, exist_ok=True)
    for csv_path in sorted(glob.glob(os.path.join(CODE_DIR, "results", "raw", a.run, "*.csv"))):
        name = os.path.basename(csv_path)[:-4]
        if a.videos and not any(v in name for v in a.videos):
            continue
        out_path = os.path.join(out_dir, name + ".mp4")
        if os.path.exists(out_path):
            continue
        video = pd.read_csv(csv_path, nrows=1)["video"].iloc[0]
        render(os.path.join(CODE_DIR, "videos", "veo", video), csv_path, out_path,
               int(a.base[0]), a.base[1], int(a.variant[0]), a.variant[1])
        print("geschrieben:", out_path, flush=True)


if __name__ == "__main__":
    main()
