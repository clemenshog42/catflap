@echo off
REM Exports the greyscale models for the Raspberry Pi (TFLite) and checks them against the .pt models.
REM Log: results\export_pi_log.txt
cd /d "%~dp0"
call venv\Scripts\activate.bat
set LOG=results\export_pi_log.txt
echo ===== Export started %date% %time% ===== > %LOG%

echo [1/4] Export face detector models\cat_face.pt (1 channel)
python src\export_model.py --model_path models\cat_face.pt --format tflite >> %LOG% 2>&1
copy /y models\cat_face_saved_model\cat_face_float32.tflite models\cat_face_gray.tflite >> %LOG% 2>&1

echo [2/4] Export segmentation model models\prey_gray.pt (1 channel)
python src\export_model.py --model_path models\prey_gray.pt --format tflite >> %LOG% 2>&1
copy /y models\prey_gray_saved_model\prey_gray_float32.tflite models\prey_gray.tflite >> %LOG% 2>&1

echo [3/4] Inference on the test videos with the TFLite models
python src\evaluation\run_inference.py --config two_stage --detector models\cat_face_gray.tflite --segmenter models\prey_gray.tflite --tag tflite --overwrite >> %LOG% 2>&1

echo [4/4] Evaluation (compare with two_stage_track_grayscale)
python src\evaluation\evaluate.py --run results\raw\two_stage_track_grayscale_tflite >> %LOG% 2>&1

echo ===== Export finished %date% %time% ===== >> %LOG%
echo Done. Results in %LOG%
pause
