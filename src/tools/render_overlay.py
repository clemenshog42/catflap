"""Renders evaluation videos with the logged detections and the replayed state machine drawn in.

For every video of a run (results/raw/<run>/*.csv) an overlay video is written to results/videos/<run>/.
Drawn in:
  - face box per cat, coloured by its state (orange: checking, green: clean, red: prey),
    with the current and the averaged prey confidence
  - the segmentation crop (extended face box): cyan if complete, magenta if it leaves the image
    (then the frame is not added to the history)
  - inset at the bottom right: the image the segmentation model gets (square, black borders, greyscale)
  - the flap state of the final system and, for comparison, of the configuration without the crop rule
  - the annotated prey state and a marker for every frame that counts as a safety or comfort error

--clean (e.g. for the presentation): only the final system, no comparison, no annotation (no ground truth needed).

Usage: python3 src/tools/render_overlay.py [--run two_stage_track_grayscale] [--videos 01 06] [--overwrite]
       python3 src/tools/render_overlay.py --run two_stage_track_grayscale_demo --video-dir videos/demo --clean
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
from state_machine import StateMachine, State, AccessState  # noqa: E402
from crop import crop_region, CROP_PAD  # noqa: E402
import evaluate as ev  # noqa: E402


def replay(df, history, threshold, crop_rule, img_w, img_h):
    """Same logic as evaluate.replay, but keeps per detection what is needed for drawing."""
    sm = StateMachine(history, threshold, 30)
    door = {"open": False}
    sm.subscribe(lambda s: door.__setitem__("open", s == AccessState.GRANTED))
    out = {}
    for i, ((fr, t), rows) in enumerate(df.groupby(["frame", "time_s"], sort=True)):
        dets = []
        for _, r in rows.iterrows():
            if pd.notna(r.track_id):
                tid = int(r.track_id)
                box = (r.x1, r.y1, r.x2, r.y2)
                region, inside = crop_region(box, img_w, img_h)
                st = sm.update(tid, float(r.prey_conf), i, inside if crop_rule else True)
                hist = sm.tracks[tid].confidence_history
                avg = float(np.mean(hist)) if len(hist) else 0.0
                dets.append(dict(tid=tid, box=box, region=region, inside=inside, prey=float(r.prey_conf),
                                 avg=avg, n=len(hist), state=st))
        sm.process_global_state(i, current_time=float(t))
        if len(sm.tracks) == 0:
            door["open"] = False
        out[int(fr)] = (float(t), dets, door["open"])
    return out


COL = {State.GATHERING_HISTORY: (0, 170, 255), State.CAT_NO_PREY: (60, 200, 60), State.CAT_WITH_PREY: (50, 50, 230)}
NAME = {State.GATHERING_HISTORY: "prueft", State.CAT_NO_PREY: "sauber", State.CAT_WITH_PREY: "Beute"}
FONT = cv2.FONT_HERSHEY_SIMPLEX
CYAN, MAGENTA = (230, 210, 0), (200, 0, 220)


def text(img, s, org, scale=0.6, col=(255, 255, 255), bg=None, th=2):
    (tw, tht), _ = cv2.getTextSize(s, FONT, scale, th)
    x, y = org
    if bg is not None:
        cv2.rectangle(img, (x - 6, y - tht - 8), (x + tw + 6, y + 8), bg, -1)
    cv2.putText(img, s, (x, y), FONT, scale, col, th, cv2.LINE_AA)
    return tw


def segmentation_input(frame, region, size):
    """The crop as the segmentation model sees it: padded to a square with black borders, greyscale."""
    x1, y1, x2, y2 = region
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((size, size, 3), np.uint8)
    ch, cw = crop.shape[:2]
    m = max(ch, cw)
    t, l = (m - ch) // 2, (m - cw) // 2
    sq = cv2.copyMakeBorder(crop, t, m - ch - t, l, m - cw - l, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    g = cv2.cvtColor(cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return cv2.resize(g, (size, size))


def render(video_path, csv_path, out_path, final, compare, clean_mode=False):
    df = pd.read_csv(csv_path)
    video = df["video"].iloc[0]
    cats = [] if clean_mode else ev.load_gt(video)["cats"]
    first_seen = min((iv[0] for c in cats for iv in c.get("face_visible", c["visible"])), default=0.0)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    main = replay(df, final[0], final[1], True, W, H)
    comp = replay(df, compare[0], compare[1], False, W, H)

    tmp = out_path + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    cur_m, cur_c = (0.0, [], False), (0.0, [], False)
    inset = int(H * 0.36)
    idx = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        clean = f.copy()
        cur_m = main.get(idx, cur_m)
        cur_c = comp.get(idx, cur_c)
        t = idx / fps
        _, dets, open_m = cur_m
        open_c = cur_c[2]

        for k, d in enumerate(dets):
            # segmentation crop
            rx1, ry1, rx2, ry2 = d["region"]
            rc = CYAN if d["inside"] else MAGENTA
            cv2.rectangle(f, (rx1, ry1), (rx2 - 1, ry2 - 1), rc, 2)
            # face box and state
            c = COL.get(d["state"], (200, 200, 200))
            x1, y1, x2, y2 = map(int, d["box"])
            cv2.rectangle(f, (x1, y1), (x2, y2), c, 3)
            text(f, f"Katze {d['tid']}: {NAME.get(d['state'], '')}", (x1 + 4, y1 - 34), 0.6, bg=c)
            text(f, f"Beute {d['prey']:.2f}  Schnitt {d['avg']:.2f} ({d['n']}/{final[0]})", (x1 + 4, y1 - 8), 0.55, bg=c)
            # inset: what the segmentation model gets
            img = segmentation_input(clean, d["region"], inset)
            cv2.rectangle(img, (0, 0), (inset - 1, inset - 1), rc, 4)
            ox, oy = W - (k + 1) * (inset + 12), H - inset - 12
            f[oy:oy + inset, ox:ox + inset] = img
            text(f, f"Segmentierung, Katze {d['tid']}", (ox + 8, oy + 26), 0.55, bg=(40, 40, 40))
            if not d["inside"]:
                text(f, "ueber dem Bildrand: zaehlt nicht", (ox + 8, oy + inset - 14), 0.5, bg=MAGENTA)

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

        if clean_mode:
            text(f, f"Klappe: {'OFFEN' if open_m else 'GESPERRT'}", (30, 56), 1.0,
                 bg=(60, 160, 60) if open_m else (50, 50, 200))
            text(f, f"t = {t:4.1f} s", (W - 170, 52), 0.8, bg=(40, 40, 40))
            vw.write(f)
            idx += 1
            continue
        gt_txt = "Annotation: Beute" if need_lock else ("Annotation: keine Beute" if visible else "Annotation: keine Katze")
        y = 50
        for label, op in ((f"System ({final[0]}/{final[1]:g}, Randregel)", open_m),
                          (f"Vergleich ({compare[0]}/{compare[1]:g}, ohne Randregel)", open_c)):
            text(f, f"{label}: {'OFFEN' if op else 'GESPERRT'}{err(op)}", (30, y), 0.75 if y == 50 else 0.6,
                 bg=(60, 160, 60) if op else (50, 50, 200))
            y += 46
        text(f, gt_txt, (30, y), 0.65, bg=(40, 40, 40))
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
    ap.add_argument("--final", nargs=2, type=float, default=[30, 0.3], metavar=("H", "T"))
    ap.add_argument("--compare", nargs=2, type=float, default=[20, 0.3], metavar=("H", "T"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--video-dir", default=os.path.join(CODE_DIR, "videos", "veo"))
    ap.add_argument("--clean", action="store_true", help="only the final system, without comparison and annotation")
    a = ap.parse_args()
    out_dir = os.path.join(CODE_DIR, "results", "videos", a.run + ("_clean" if a.clean else ""))
    os.makedirs(out_dir, exist_ok=True)
    for csv_path in sorted(glob.glob(os.path.join(CODE_DIR, "results", "raw", a.run, "*.csv"))):
        name = os.path.basename(csv_path)[:-4]
        if a.videos and not any(v in name for v in a.videos):
            continue
        out_path = os.path.join(out_dir, name + ".mp4")
        if os.path.exists(out_path) and not a.overwrite:
            continue
        video = pd.read_csv(csv_path, nrows=1)["video"].iloc[0]
        render(os.path.join(a.video_dir, video), csv_path, out_path,
               (int(a.final[0]), a.final[1]), (int(a.compare[0]), a.compare[1]), a.clean)
        print("geschrieben:", out_path, flush=True)


if __name__ == "__main__":
    main()
