@echo off
REM Runs the complete evaluation on all Veo test videos and writes everything to results\run_log.txt
cd /d "%~dp0"
call venv\Scripts\activate.bat
if not exist results mkdir results
set LOG=results\run_log.txt
echo ===== Evaluation started %date% %time% ===== > %LOG%

REM Ablations: (1) state machine = parameter sweep of evaluate.py, (2) detection+segmentation vs. detection = two_stage vs. one_stage,
REM (3) grayscale vs. RGB = segmentation model prey_gray.pt vs. prey_rgb.pt (only if models\prey_rgb.pt exists)
echo [1/5] two_stage with tracking
python src\evaluation\run_inference.py --config two_stage >> %LOG% 2>&1
echo [2/5] two_stage without tracking
python src\evaluation\run_inference.py --config two_stage --no-tracking >> %LOG% 2>&1
echo [3/5] one_stage with tracking
python src\evaluation\run_inference.py --config one_stage >> %LOG% 2>&1
echo [4/5] one_stage without tracking
python src\evaluation\run_inference.py --config one_stage --no-tracking >> %LOG% 2>&1
set RUNS=two_stage_track_grayscale two_stage_notrack_grayscale one_stage_track_grayscale one_stage_notrack_grayscale
if exist models\prey_rgb.pt (
  echo [5/5] two_stage with RGB segmentation model
  python src\evaluation\run_inference.py --config two_stage --segmenter models\prey_rgb.pt --seg-color rgb --tag rgbseg >> %LOG% 2>&1
  set RUNS=%RUNS% two_stage_track_grayscale_rgbseg
) else (
  echo [5/5] skipped: models\prey_rgb.pt not found
)

echo Evaluating...
setlocal enabledelayedexpansion
for %%R in (!RUNS!) do (
  python src\evaluation\evaluate.py --run results\raw\%%R >> %LOG% 2>&1
  python src\evaluation\evaluate.py --run results\raw\%%R --sweep >> %LOG% 2>&1
)
echo ===== Evaluation finished %date% %time% ===== >> %LOG%
echo Done. Results in %LOG%
pause
