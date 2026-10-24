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

Metrics (per video and summed over all videos)
  safety errors   : frames in which the door is open although a cat with prey is visible
  comfort errors  : frames in which the door is locked although only clean cats are visible,
                    counted only after the decision time (--grace seconds after the cat face appeared)
  precision / recall / F1 of the decision "lock" (clean frames inside the decision time are excluded)
  prey passed     : the door was open at least once while a cat with prey was visible (event level)
  locked out      : a clean cat never got the door opened (event level)
  decision time   : seconds from the cat face appearing to the first DENIED (prey videos) or GRANTED (clean videos)
  missed cats     : frames with a recognisable cat face (face_visible) in which no cat was detected

Crop rule (two-stage runs): frames in which the extended face box for the segmentation reaches beyond the
image border are not added to the history (see crop.py and StateMachine.update). --no-crop-rule switches it off
(one-stage runs have no segmentation crop).

Usage (from code/):
  python src/evaluation/evaluate.py --run results/raw/two_stage_track_grayscale --history 30 --threshold 0.3
  python src/evaluation/evaluate.py --run results/raw/two_stage_track_grayscale --sweep
"""
import argparse
import glob
import json
import os
import sys

import cv2
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from state_machine import StateMachine, AccessState  # noqa: E402
from crop import crop_region  # noqa: E402

CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIDEO_DIR = os.path.join(CODE_DIR, "videos", "veo")
ANN_DIR = os.path.join(VIDEO_DIR, "annotations")
_SIZES = {}


def video_size(video):
    """(width, height) of a test video, needed to check whether the segmentation crop stays inside the image."""
    if video not in _SIZES:
        cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, video))
        _SIZES[video] = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    return _SIZES[video]


def in_any(t, intervals):
    return any(s <= t <= e for s, e in intervals)


def load_gt(video):
    with open(os.path.join(ANN_DIR, video.replace("/", "__").replace(".mp4", ".json"))) as f:
        return json.load(f)


def replay(df, history, threshold, max_missed_frames=30, crop_rule=True):
    """Replays the state machine.
    Frames are counted as processed frames (as on the Pi), not as indices of the original 24 FPS video,
    so max_missed_frames has the same meaning as in processor.py (30 frames = 3.75 s at 8 FPS).
    crop_rule: frames whose segmentation crop leaves the image are not added to the history.
    Returns the timeline [(time_s, door_open, cat_detected)] and the emitted decisions [(time_s, AccessState)]."""
    sm = StateMachine(history_length=history, threshold=threshold, max_missed_frames=max_missed_frames)
    door = {"open": False, "t": 0.0}
    decisions = []

    def cb(state):
        door["open"] = (state == AccessState.GRANTED)
        decisions.append((door["t"], state))
    sm.subscribe(cb)
    img_w, img_h = video_size(df["video"].iloc[0]) if crop_rule else (0, 0)

    timeline = []
    for frame, ((_, t), rows) in enumerate(df.groupby(["frame", "time_s"], sort=True)):
        door["t"] = float(t)
        detected = False
        for _, r in rows.iterrows():
            if pd.notna(r["track_id"]) and r["track_id"] != "":
                detected = True
                valid = crop_region((r["x1"], r["y1"], r["x2"], r["y2"]), img_w, img_h)[1] if crop_rule else True
                sm.update(int(r["track_id"]), float(r["prey_conf"]), int(frame), valid)
        sm.process_global_state(int(frame), current_time=float(t))
        if len(sm.tracks) == 0:
            door["open"] = False  # flap auto-locks when no cat is present
        timeline.append((float(t), door["open"], detected))
    return timeline, decisions


def score(video, timeline, decisions, gt, grace):
    cats = gt["cats"]
    # decision time is measured from the moment a cat face is recognisable (face_visible), else from first visibility
    first_seen = min((iv[0] for c in cats for iv in c.get("face_visible", c["visible"])), default=0.0)
    prey_video = any(c.get("prey") for c in cats)
    m = dict(video=video, category="with_prey" if prey_video else "no_prey", frames=0,
             tp=0, fp=0, fn=0, tn=0, safety_errors=0, comfort_errors=0, prey_frames=0, clean_frames=0,
             face_frames=0, missed_cat_frames=0, prey_passed=0, locked_out=0, decision_time_s=None)
    ever_open_while_clean = False
    for t, is_open, detected in timeline:
        m["frames"] += 1
        if any(in_any(t, c.get("face_visible", c["visible"])) for c in cats):
            m["face_frames"] += 1
            m["missed_cat_frames"] += not detected
        visible = [c for c in cats if in_any(t, c["visible"])]
        if not visible:
            continue
        need_lock = any(in_any(t, c.get("prey", [])) for c in visible)
        locked = not is_open
        if need_lock:
            m["prey_frames"] += 1
            m["tp"] += locked
            m["fn"] += not locked
            m["safety_errors"] += not locked
            m["prey_passed"] = max(m["prey_passed"], int(not locked))
        else:
            ever_open_while_clean |= is_open
            if t - first_seen <= grace:
                continue  # the state machine is still allowed to collect its history
            m["clean_frames"] += 1
            m["fp"] += locked
            m["tn"] += not locked
            m["comfort_errors"] += locked
    if not prey_video and m["clean_frames"] > 0 and not ever_open_while_clean:
        m["locked_out"] = 1
    target = AccessState.DENIED if prey_video else AccessState.GRANTED
    hit = next((t for t, s in decisions if s == target), None)
    if hit is not None:
        m["decision_time_s"] = round(max(0.0, hit - first_seen), 2)
    return m


def evaluate_run(run_dir, history, threshold, grace, crop_rule=True):
    rows = []
    for csv_path in sorted(glob.glob(os.path.join(run_dir, "*.csv"))):
        df = pd.read_csv(csv_path, dtype={"track_id": "string"})
        video = df["video"].iloc[0]
        timeline, decisions = replay(df, history, threshold, crop_rule=crop_rule)
        m = score(video, timeline, decisions, load_gt(video), grace)
        m.update(history=history, threshold=threshold)
        rows.append(m)
    return pd.DataFrame(rows)


def ratio(a, b):
    return a / b if b else float("nan")


def summarize(res):
    tp, fp, fn = res.tp.sum(), res.fp.sum(), res.fn.sum()
    prec, rec = ratio(tp, tp + fp), ratio(tp, tp + fn)
    prey = res[res.category == "with_prey"]
    clean = res[res.category == "no_prey"]
    return {
        "precision": prec,
        "recall": rec,
        "f1": ratio(2 * prec * rec, prec + rec),
        "safety_error_rate": ratio(res.safety_errors.sum(), res.prey_frames.sum()),
        "comfort_error_rate": ratio(res.comfort_errors.sum(), res.clean_frames.sum()),
        "missed_cat_rate": ratio(res.missed_cat_frames.sum(), res.face_frames.sum()),
        "prey_passed": int(prey.prey_passed.sum()), "prey_videos": len(prey),
        "locked_out": int(clean.locked_out.sum()), "clean_videos": len(clean),
        "decision_time_prey_s": prey.decision_time_s.dropna().mean() if len(prey) else float("nan"),
        "decision_time_clean_s": clean.decision_time_s.dropna().mean() if len(clean) else float("nan"),
    }


# ---------- human readable output ----------

def pct(x):
    return "   -   " if pd.isna(x) else f"{100 * x:6.1f} %"


def sec(x):
    return "  -  " if x is None or pd.isna(x) else f"{x:4.1f} s"


def print_table(headers, rows):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def print_report(run, res, history, threshold, grace):
    print(f"\nRun: {run}   |   Zeitfenster: {history} Bilder   |   Schwellenwert: {threshold}   |   Entscheidungszeit erlaubt: {grace} s\n")
    rows = []
    for _, r in res.iterrows():
        if r.category == "with_prey":
            errors = f"{r.safety_errors}/{r.prey_frames} Bilder offen"
            outcome = "BEUTE DURCHGELASSEN" if r.prey_passed else "ok"
        else:
            errors = f"{r.comfort_errors}/{r.clean_frames} Bilder gesperrt"
            outcome = "AUSGESPERRT" if r.locked_out else "ok"
        rows.append([r.video, "mit Beute" if r.category == "with_prey" else "ohne Beute", errors,
                     f"{r.missed_cat_frames}/{r.face_frames}", sec(r.decision_time_s), outcome])
    print_table(["Video", "Kategorie", "Fehler", "Katze übersehen", "Entscheidung nach", "Ergebnis"], rows)

    s = summarize(res)
    print("\nZusammenfassung")
    print(f"  Sicherheitsfehler (offen trotz Beute):        {pct(s['safety_error_rate'])}")
    print(f"  Komfortfehler (gesperrt obwohl sauber):       {pct(s['comfort_error_rate'])}")
    print(f"  Präzision / Trefferquote / F1 (sperren):      {pct(s['precision'])} / {pct(s['recall'])} / {pct(s['f1'])}")
    print(f"  Übersehene Katzen:                            {pct(s['missed_cat_rate'])}")
    print(f"  Beute durchgelassen:                          {s['prey_passed']} von {s['prey_videos']} Videos")
    print(f"  Saubere Katze ausgesperrt:                    {s['locked_out']} von {s['clean_videos']} Videos")
    print(f"  Mittlere Entscheidungszeit mit / ohne Beute:  {sec(s['decision_time_prey_s'])} / {sec(s['decision_time_clean_s'])}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="folder with raw CSVs from run_inference.py")
    ap.add_argument("--history", type=int, default=30)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--grace", type=float, default=3.0, help="seconds allowed for the initial decision")
    ap.add_argument("--sweep", action="store_true", help="grid over history length and threshold")
    ap.add_argument("--no-crop-rule", dest="crop_rule", action="store_false",
                    help="add every frame to the history, also if the segmentation crop leaves the image (one-stage runs)")
    args = ap.parse_args()

    out_dir = os.path.join(CODE_DIR, "results", "metrics")
    os.makedirs(out_dir, exist_ok=True)
    run = os.path.basename(os.path.normpath(args.run))

    if args.sweep:
        summary = []
        for h in [1, 3, 5, 8, 10, 15, 20, 25, 30]:
            for th in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                # a window of one frame is the system without state machine: no frame is held back there
                s = summarize(evaluate_run(args.run, h, th, args.grace, args.crop_rule and h > 1))
                summary.append(dict(history=h, threshold=th, **s))
        out = pd.DataFrame(summary)
        path = os.path.join(out_dir, f"{run}_sweep.csv")
        out.to_csv(path, index=False)
        print(f"\nParameterreihe für {run}  (alle Werte über alle Videos)\n")
        print_table(["Fenster", "Schwelle", "Sicherheitsf.", "Komfortf.", "F1", "Beute durch", "Ausgesperrt", "Entsch. Beute", "Entsch. sauber"],
                    [[r.history, r.threshold, pct(r.safety_error_rate), pct(r.comfort_error_rate), pct(r.f1),
                      f"{r.prey_passed}/{r.prey_videos}", f"{r.locked_out}/{r.clean_videos}",
                      sec(r.decision_time_prey_s), sec(r.decision_time_clean_s)] for r in out.itertuples()])
        print(f"\nGespeichert: {path}\n")
    else:
        res = evaluate_run(args.run, args.history, args.threshold, args.grace, args.crop_rule)
        path = os.path.join(out_dir, f"{run}_h{args.history}_t{args.threshold}.csv")
        res.to_csv(path, index=False)
        print_report(run, res, args.history, args.threshold, args.grace)
        print(f"Details gespeichert: {path}\n")


if __name__ == "__main__":
    main()
