"""Tracking stability per run: number of track IDs per video compared with the number of annotated cats.
Every ID beyond the number of cats is an ID switch or a short-lived extra track.

  python src/tools/track_stats.py two_stage_track_grayscale two_stage_track_grayscale_bt_relaxed
"""
import glob
import os
import sys

import pandas as pd

CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(CODE_DIR, "src", "evaluation"))
sys.path.insert(0, os.path.join(CODE_DIR, "src"))
import evaluate as ev  # noqa: E402


def stats(run):
    rows = []
    for f in sorted(glob.glob(os.path.join(CODE_DIR, "results", "raw", run, "*.csv"))):
        df = pd.read_csv(f)
        cats = len(ev.load_gt(df["video"].iloc[0])["cats"])
        d = df.dropna(subset=["track_id"])
        lens = d.groupby("track_id").size().sort_values(ascending=False).tolist()
        rows.append(dict(video=os.path.basename(f)[:-4], cats=cats, tracks=len(lens),
                         extra=max(0, len(lens) - cats), short=sum(n < 8 for n in lens), lengths=lens))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    runs = sys.argv[1:] or ["two_stage_track_grayscale"]
    res = {r: stats(r) for r in runs}
    for r, s in res.items():
        print(f"\n{r}")
        print(s.to_string(index=False))
    print("\nZusammenfassung (zusätzliche Tracks über alle Videos, davon kürzer als 1 s)")
    for r, s in res.items():
        print(f"  {r:45s} {int(s.extra.sum()):3d}   kurz: {int(s.short.sum())}")
