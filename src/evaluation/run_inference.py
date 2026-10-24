"""
Step 1 of the evaluation: run the vision models once per video and configuration
and log the raw per-frame results to CSV. The state machine is NOT applied here;
it is replayed offline by evaluate.py, so parameter sweeps are cheap.

Videos are subsampled to the Raspberry Pi frame rate (default 8 FPS).

Configurations
  two_stage : cat face detector (+ optional ByteTrack) -> prey segmentation on the extended face crop
  one_stage : single detector with classes "cat with prey" / "cat without prey" on the full frame

Usage (from code/):
  python src/evaluation/run_inference.py --config two_stage
  python src/evaluation/run_inference.py --config two_stage --no-tracking
  python src/evaluation/run_inference.py --config one_stage --one-stage-model models/1stage_ablation.pt
  python src/evaluation/run_inference.py --config two_stage --segmenter models/prey_rgb.pt --seg-color rgb --tag rgbseg   (grayscale ablation)
"""
import argparse
import csv
import glob
import os
import sys

import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import CatFlapPipeline  # noqa: E402

CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIELDS = ["video", "frame", "time_s", "track_id", "x1", "y1", "x2", "y2", "det_conf", "prey_conf"]


def to_input(frame, color):
    if color == "grayscale":
        return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return frame


def iter_frames(path, target_fps):
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    step = src_fps / target_fps
    next_pick, idx = 0.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= next_pick - 1e-6:
            yield idx, idx / src_fps, frame
            next_pick += step
        idx += 1
    cap.release()


def run_two_stage(pipeline, path, args, writer, name):
    if args.tracking:
        pipeline.detector.predictor = None  # reset ByteTrack state between videos
    for idx, t, frame in iter_frames(path, args.fps):
        res = pipeline.run_detector(frame, track=args.tracking)
        if res is None or res.boxes is None or len(res.boxes) == 0:
            writer.writerow({"video": name, "frame": idx, "time_s": f"{t:.3f}", "track_id": ""})
            continue
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        ids = res.boxes.id.int().cpu().tolist() if (args.tracking and res.boxes.id is not None) else [None] * len(boxes)
        written = 0
        for box, conf, tid in zip(boxes, confs, ids):
            if args.tracking and tid is None:
                continue  # ByteTrack has not confirmed this detection yet; the pipeline ignores it as well
            prey_conf, _ = pipeline.run_prey_detector(frame, box)  # colour mode handled inside the pipeline
            writer.writerow({"video": name, "frame": idx, "time_s": f"{t:.3f}",
                             "track_id": tid if args.tracking else 1,
                             "x1": int(box[0]), "y1": int(box[1]), "x2": int(box[2]), "y2": int(box[3]),
                             "det_conf": f"{conf:.4f}", "prey_conf": f"{prey_conf:.4f}"})
            written += 1
        if written == 0:  # keep every processed frame in the log, otherwise the frame is lost for the evaluation
            writer.writerow({"video": name, "frame": idx, "time_s": f"{t:.3f}", "track_id": ""})


def run_one_stage(model, path, args, writer, name):
    if args.tracking:
        model.predictor = None
    for idx, t, frame in iter_frames(path, args.fps):
        inp = to_input(frame, args.color)
        if args.tracking:
            res = model.track(inp, conf=0.1, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        else:
            res = model.predict(inp, conf=0.1, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            writer.writerow({"video": name, "frame": idx, "time_s": f"{t:.3f}", "track_id": ""})
            continue
        ids = res.boxes.id.int().cpu().tolist() if (args.tracking and res.boxes.id is not None) else [None] * len(res.boxes)
        written = 0
        for b, tid in zip(res.boxes, ids):
            if args.tracking and tid is None:
                continue
            written += 1
            cls_name = res.names[int(b.cls[0])].lower()
            conf = float(b.conf[0])
            is_prey = ("prey" in cls_name and "no" not in cls_name and "without" not in cls_name)
            prey_conf = conf if is_prey else 1.0 - conf  # probability-like prey score
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            writer.writerow({"video": name, "frame": idx, "time_s": f"{t:.3f}",
                             "track_id": tid if args.tracking else 1,
                             "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                             "det_conf": f"{conf:.4f}", "prey_conf": f"{prey_conf:.4f}"})
        if written == 0:
            writer.writerow({"video": name, "frame": idx, "time_s": f"{t:.3f}", "track_id": ""})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["two_stage", "one_stage"], required=True)
    ap.add_argument("--videos", default=os.path.join(CODE_DIR, "videos", "veo"))
    ap.add_argument("--out", default=os.path.join(CODE_DIR, "results", "raw"))
    ap.add_argument("--fps", type=float, default=8.0, help="simulated Raspberry Pi frame rate")
    ap.add_argument("--no-tracking", dest="tracking", action="store_false")
    ap.add_argument("--color", choices=["grayscale", "rgb"], default="grayscale")
    ap.add_argument("--seg-color", dest="seg_color", choices=["grayscale", "rgb"], default=None,
                    help="colour mode of the segmentation model only (default: same as --color); the face detector keeps --color")
    ap.add_argument("--detector", default=os.path.join(CODE_DIR, "models", "cat_face.pt"))  # greyscale model (1 channel)
    ap.add_argument("--segmenter", default=os.path.join(CODE_DIR, "models", "prey_gray.pt"))
    ap.add_argument("--one-stage-model", default=os.path.join(CODE_DIR, "models", "1stage_ablation.pt"))
    ap.add_argument("--crop-pad", dest="crop_pad", nargs=3, type=float, default=None, metavar=("SIDE", "TOP", "BOTTOM"),
                    help="extension of the face box for the segmentation crop, relative to the box size (default 0.8 0 2.0)")
    ap.add_argument("--tag", default="", help="extra name for the output folder")
    ap.add_argument("--overwrite", action="store_true", help="recompute videos that already have a CSV")
    args = ap.parse_args()

    run_name = f"{args.config}_{'track' if args.tracking else 'notrack'}_{args.color}" + (f"_{args.tag}" if args.tag else "")
    out_dir = os.path.join(args.out, run_name)
    os.makedirs(out_dir, exist_ok=True)

    if args.config == "two_stage":
        pipeline = CatFlapPipeline(args.detector, args.segmenter, apply_clahe_detector=False, apply_clahe_classifier=False,
                                   color_mode_detector=args.color, color_mode_classifier=args.seg_color or args.color)
        if args.crop_pad:
            pipeline.crop_pad = tuple(args.crop_pad)
    else:
        from ultralytics import YOLO
        model = YOLO(args.one_stage_model)

    videos = sorted(glob.glob(os.path.join(args.videos, "*", "*.mp4")))
    print(f"{len(videos)} videos -> {out_dir}")
    for path in videos:
        name = f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"
        csv_path = os.path.join(out_dir, name.replace("/", "__").replace(".mp4", ".csv"))
        if os.path.exists(csv_path) and not args.overwrite:
            print("  skip (already done)", name)
            continue
        part_path = csv_path + ".part"  # only renamed to .csv once the video is complete
        with open(part_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            if args.config == "two_stage":
                run_two_stage(pipeline, path, args, writer, name)
            else:
                run_one_stage(model, path, args, writer, name)
        os.replace(part_path, csv_path)
        print("  done", name)


if __name__ == "__main__":
    main()
