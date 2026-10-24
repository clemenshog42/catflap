@echo off
REM Runs the complete evaluation on all Veo test videos and writes everything to results\run_log.txt
cd /d "%~dp0"
call venv\Scripts\activate.bat
if not exist results mkdir results
set LOG=results\run_log.txt
echo ===== Evaluation started %date% %time% ===== > %LOG%

REM Ablations: (1) state machine = parameter sweep of evaluate.py, (2) detection+segmentation vs. detection = two_stage vs. one_stage,
REM Two-stage runs use the greyscale face detector models\cat_face.pt and the crop extension from src\crop.py and are recomputed (--overwrite);
REM one-stage runs do not use the face detector and are kept.
REM (3) grayscale vs. RGB = segmentation model prey_gray.pt vs. prey_rgb.pt (only if models\prey_rgb.pt exists)
echo [1/5] two_stage with tracking
python src\evaluation\run_inference.py --config two_stage --overwrite >> %LOG% 2>&1
echo [2/5] two_stage without tracking
python src\evaluation\run_inference.py --config two_stage --no-tracking --overwrite >> %LOG% 2>&1
echo [3/5] one_stage with tracking
python src\evaluation\run_inference.py --config one_stage >> %LOG% 2>&1
echo [4/5] one_stage without tracking
python src\evaluation\run_inference.py --config one_stage --no-tracking >> %LOG% 2>&1
set RUNS=two_stage_track_grayscale two_stage_notrack_grayscale one_stage_track_grayscale one_stage_notrack_grayscale
if exist models\prey_rgb.pt (
  echo [5/5] two_stage with RGB segmentation model
  python src\evaluation\run_inference.py --config two_stage --segmenter models\prey_rgb.pt --seg-color rgb --tag rgbseg --overwrite >> %LOG% 2>&1
  set RUNS=%RUNS% two_stage_track_grayscale_rgbseg
) else (
  echo [5/5] skipped: models\prey_rgb.pt not found
)

echo Evaluating...
setlocal enabledelayedexpansion
REM One-stage runs have no segmentation crop, so the crop rule of the state machine does not apply (--no-crop-rule)
for %%R in (!RUNS!) do (
  set NAME=%%R
  set EXTRA=
  if "!NAME:~0,9!"=="one_stage" set EXTRA=--no-crop-rule
  python src\evaluation\evaluate.py --run results\raw\%%R !EXTRA! >> %LOG% 2>&1
  python src\evaluation\evaluate.py --run results\raw\%%R --sweep !EXTRA! >> %LOG% 2>&1
)
python src\evaluation\ablation_summary.py >> %LOG% 2>&1
echo ===== Evaluation finished %date% %time% ===== >> %LOG%
echo Done. Results in %LOG%
pause
