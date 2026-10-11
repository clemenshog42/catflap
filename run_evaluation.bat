@echo off
REM Runs the complete evaluation on all Veo test videos and writes everything to results\run_log.txt
cd /d "%~dp0"
call venv\Scripts\activate.bat
if not exist results mkdir results
set LOG=results\run_log.txt
echo ===== Evaluation started %date% %time% ===== > %LOG%

echo [1/4] two_stage with tracking
python src\evaluation\run_inference.py --config two_stage >> %LOG% 2>&1
echo [2/4] two_stage without tracking
python src\evaluation\run_inference.py --config two_stage --no-tracking >> %LOG% 2>&1
echo [3/4] one_stage with tracking
python src\evaluation\run_inference.py --config one_stage >> %LOG% 2>&1
echo [4/4] one_stage without tracking
python src\evaluation\run_inference.py --config one_stage --no-tracking >> %LOG% 2>&1

echo Evaluating...
for %%R in (two_stage_track_grayscale two_stage_notrack_grayscale one_stage_track_grayscale one_stage_notrack_grayscale) do (
  python src\evaluation\evaluate.py --run results\raw\%%R >> %LOG% 2>&1
  python src\evaluation\evaluate.py --run results\raw\%%R --sweep >> %LOG% 2>&1
)
echo ===== Evaluation finished %date% %time% ===== >> %LOG%
echo Done. Results in %LOG%
pause
