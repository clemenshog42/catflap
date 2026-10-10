"""
Step 2 of the evaluation: replay the state machine offline on the raw CSV logs
from run_inference.py and compute error metrics against the per-frame ground truth.

Ground truth: videos/veo/annotations/<category>__<video>.json
  {
    "cats": [
      {"id": "A", "visible": [[t_start, t_end]], "face_visible": [[t_start, t_end]], "prey": [[t_start, t_end]]}
    ]
  }
Times are in seconds. "prey" marks the time the cat carries prey (not only when the prey is visible).
A frame "needs a lock" if any visible cat carries prey.

Door model: the flap is locked by default; it is only open after the state machine
emitted GRANTED and before the next DENIED (or before all cats left the image).

Metrics per video and configuration
  safety_errors     : frames in which the door is open although a cat with prey is visible
  comfort_errors    : frames in which the door is locked although only clean cats are visible,
                      counted after the allowed decision time (--grace seconds after the cat appeared)
  prey_passed       : 1 if a cat with prey had the door open at least once (event level)
  clean_locked_out  : 1 if a clean cat never got the door opened while visible (event level)
  latency_s         : time from first visibility to the first correct decision
                      (DENIED for prey videos, GRANTED for clean videos); empty if never reached
  frame-level precision / recall / F1 of the decision "lock" over all frames with a visible cat

Usage (from code/):
  python src/evaluation/evaluate.py --run results/raw/two_stage_track_grayscale --history 5 --threshold 0.5
  python src/evaluation/evaluate.py --run results/raw/two_stage_track_grayscale --sweep
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from state_machine import StateMachine, AccessState  # noqa: E402

CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANN_DIR = os.path.join(CODE_DIR, "videos", "veo", "annotations")


def in_any(t, intervals):
    return any(s <= t <= e for s, e in intervals)


def load_gt(video):
    with open(os.path.join(ANN_DIR, video.replace("/", "__").replace(".mp4", ".json"))) as f:
        return json.load(f)


def replay(df, history, threshold, max_missed_s=4.0, fps=8.0):
    """Replays the state machine; returns a list of (time_s, door_open) per processed frame."""
    sm = StateMachine(history_length=history, threshold=threshold, max_missed_frames=int(round(max_missed_s * fps)))
    door = {"open": False}

    def cb(state):
        door["open"] = (state == AccessState.GRANTED)
    sm.subscribe(cb)

    out = []
    for (frame, t), rows in df.groupby(["frame", "time_s"], sort=True):
        for _, r in rows.iterrows():
            if pd.notna(r["track_id"]) and r["track_id"] != "":
                sm.update(int(r["track_id"]), float(r["prey_conf"]), int(frame))
        sm.process_global_state(int(frame), current_time=float(t))
        if len(sm.tracks) == 0:
            door["open"] = False  # flap auto-locks when no cat is present
        out.append((float(t), door["open"]))
    return out


def score(video, timeline, gt, grace):
    cats = gt["cats"]
    # decision time is measured from the moment a cat face is recognisable (face_visible), else from first visibility
    first_seen = min((iv[0] for c in cats for iv in c.get("face_visible", c["visible"])), default=None)
    has_prey_video = any(c.get("prey") for c in cats)
    m = dict(video=video, frames=0, cat_frames=0, safety_errors=0, comfort_errors=0,
             tp=0, fp=0, fn=0, tn=0, prey_passed=0, clean_locked_out=0, latency_s=None)
    ever_open_while_clean = False
    for t, is_open in timeline:
        m["frames"] += 1
        visible = [c for c in cats if in_any(t, c["visible"])]
        if not visible:
            continue
        m["cat_frames"] += 1
        need_lock = any(in_any(t, c.get("prey", [])) for c in visible)
        locked = not is_open
        if need_lock and not locked:
            m["safety_errors"] += 1
            m["prey_passed"] = 1
        if not need_lock and locked and first_seen is not None and t - first_seen > grace:
            m["comfort_errors"] += 1
        if not need_lock and is_open:
            ever_open_while_clean = True
        m["tp"] += need_lock and locked
        m["fn"] += need_lock and not locked
        m["fp"] += (not need_lock) and locked
        m["tn"] += (not need_lock) and not locked
        if m["latency_s"] is None and first_seen is not None:
            if (has_prey_video and need_lock and locked) or (not has_prey_video and is_open):
                m["latency_s"] = round(t - first_seen, 3)
    if not has_prey_video and m["cat_frames"] > 0 and not ever_open_while_clean:
        m["clean_locked_out"] = 1
    m["category"] = "with_prey" if has_prey_video else "no_prey"
    return m


def evaluate_run(run_dir, history, threshold, grace):
    rows = []
    for csv_path in sorted(glob.glob(os.path.join(run_dir, "*.csv"))):
        df = pd.read_csv(csv_path, dtype={"track_id": "string"})
        video = df["video"].iloc[0]
        timeline = replay(df, history, threshold)
        m = score(video, timeline, load_gt(video), grace)
        m.update(history=history, threshold=threshold)
        rows.append(m)
    return pd.DataFrame(rows)


def summarize(res):
    tp, fp, fn = res.tp.sum(), res.fp.sum(), res.fn.sum()
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    prey = res[res.category == "with_prey"]
    clean = res[res.category == "no_prey"]
    return dict(precision=round(prec, 3), recall=round(rec, 3), f1=round(f1, 3),
                safety_error_rate=round(res.safety_errors.sum() / max(1, (res.tp + res.fn).sum()), 3),
                comfort_error_rate=round(res.comfort_errors.sum() / max(1, (res.fp + res.tn).sum()), 3),
                prey_passed=f"{int(prey.prey_passed.sum())}/{len(prey)}",
                clean_locked_out=f"{int(clean.clean_locked_out.sum())}/{len(clean)}",
                latency_prey_s=round(prey.latency_s.dropna().mean(), 2) if len(prey) else None,
                latency_clean_s=round(clean.latency_s.dropna().mean(), 2) if len(clean) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="folder with raw CSVs from run_inference.py")
    ap.add_argument("--history", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--grace", type=float, default=3.0, help="seconds allowed for the initial decision")
    ap.add_argument("--sweep", action="store_true", help="grid over history length and threshold")
    args = ap.parse_args()

    out_dir = os.path.join(CODE_DIR, "results", "metrics")
    os.makedirs(out_dir, exist_ok=True)
    run = os.path.basename(os.path.normpath(args.run))

    if args.sweep:
        summary = []
        for h in [1, 3, 5, 8, 10, 15, 20]:
            for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                res = evaluate_run(args.run, h, th, args.grace)
                summary.append(dict(history=h, threshold=th, **summarize(res)))
        out = pd.DataFrame(summary)
        out.to_csv(os.path.join(out_dir, f"{run}_sweep.csv"), index=False)
        print(out.to_string(index=False))
    else:
        res = evaluate_run(args.run, args.history, args.threshold, args.grace)
        res.to_csv(os.path.join(out_dir, f"{run}_h{args.history}_t{args.threshold}.csv"), index=False)
        print(res.to_string(index=False))
        print(summarize(res))


if __name__ == "__main__":
    main()
