"""Compare face detector variants on the Veo test videos (8 FPS, like the evaluation).

For every variant it counts, per video, the processed frames in which at least one cat face
is detected with a confidence of at least --conf, and prints the share of frames with a detection.

  venv\\Scripts\\python src\\tools\\compare_face_detectors.py
"""
import argparse
import glob
import os
import cv2
from ultralytics import YOLO

CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VARIANTS = [
    # name, model file, input colour mode
    ("tflite (2609) + Farbe", "models/cat_face.tflite", "rgb"),
    ("tflite (2609) + Graustufen", "models/cat_face.tflite", "gray3"),
    ("cat_face.pt (1 Kanal) + Graustufen", "models/cat_face.pt", "gray1"),
]


def prepare(frame, mode):
    if mode == "rgb":
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray[..., None] if mode == "gray1" else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.25, help="ByteTrack starts new tracks from about 0.25")
    ap.add_argument("--step", type=int, default=3, help="24 FPS / 3 = 8 FPS")
    args = ap.parse_args()
    videos = sorted(glob.glob(os.path.join(CODE_DIR, "videos", "veo", "*", "*.mp4")))
    models = {}
    print(f"Anteil Einzelbilder mit erkanntem Gesicht (Konfidenz >= {args.conf})\n")
    header = f"{'Video':45s}" + "".join(f"{v[0][:22]:>24s}" for v in VARIANTS)
    print(header)
    totals = [[0, 0] for _ in VARIANTS]
    for path in videos:
        name = os.path.relpath(path, os.path.join(CODE_DIR, "videos", "veo"))
        row = f"{name[:45]:45s}"
        for k, (label, model_path, mode) in enumerate(VARIANTS):
            if model_path not in models:
                task = "detect"
                models[model_path] = YOLO(os.path.join(CODE_DIR, model_path), task=task)
            model = models[model_path]
            cap = cv2.VideoCapture(path)
            i = hits = n = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % args.step == 0:
                    r = model.predict(prepare(frame, mode), conf=args.conf, verbose=False)[0]
                    hits += int(len(r.boxes) > 0)
                    n += 1
                i += 1
            totals[k][0] += hits
            totals[k][1] += n
            row += f"{hits / max(n, 1):>23.0%} "
        print(row)
    print("-" * len(header))
    print(f"{'Gesamt':45s}" + "".join(f"{h / max(n, 1):>23.0%} " for h, n in totals))


if __name__ == "__main__":
    main()
