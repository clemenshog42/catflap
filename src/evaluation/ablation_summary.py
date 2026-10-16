"""Summary of all ablation runs: single-frame decision (window 1, threshold 0.5) and the final state machine
configuration (window 20, threshold 0.3), split into day and infrared videos.

  python src/evaluation/ablation_summary.py
"""
import os
import re
import pandas as pd
from evaluate import evaluate_run, summarize, CODE_DIR

RUNS = ["two_stage_track_grayscale", "two_stage_notrack_grayscale", "one_stage_track_grayscale",
        "one_stage_notrack_grayscale", "two_stage_track_grayscale_rgbseg"]
CONFIGS = [(1, 0.5), (20, 0.3)]
IR_VIDEOS = {"06", "07", "08", "10"}  # infrared night vision, all others are day light


def group_of(video):
    return "IR" if re.search(r"/(\d\d)_", video).group(1) in IR_VIDEOS else "Tag"


def main():
    rows = []
    for run in RUNS:
        run_dir = os.path.join(CODE_DIR, "results", "raw", run)
        if not os.path.isdir(run_dir):
            continue
        for h, t in CONFIGS:
            res = evaluate_run(run_dir, h, t, 3.0)
            res["group"] = res.video.map(group_of)
            rows.append(dict(summarize(res), run=run, h=h, t=t, group="alle"))
            for g in ["Tag", "IR"]:
                rows.append(dict(summarize(res[res.group == g]), run=run, h=h, t=t, group=g))
    out = pd.DataFrame(rows)
    path = os.path.join(CODE_DIR, "results", "metrics", "ablation_summary.csv")
    out.to_csv(path, index=False)
    print(out[["run", "h", "t", "group", "safety_error_rate", "comfort_error_rate", "precision", "recall", "f1",
               "missed_cat_rate", "prey_passed", "prey_videos", "locked_out", "decision_time_prey_s",
               "decision_time_clean_s"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
