# catflap

Prey detection for a cat flap: YOLO26 cat face detection, ByteTrack tracking, prey segmentation on the
(bottom-extended) face crop and a state machine that aggregates the prey confidence over time.

## Structure

| Path | Content |
|---|---|
| `src/` | Pipeline: `models.py` (detector + segmentation), `processor.py`, `state_machine.py`, hardware (`catflap.py`) |
| `src/main.py`, `src/pi_main.py`, `src/pi_server.py` | Local test, Raspberry Pi test, Raspberry Pi server (`--no-tracking` disables ByteTrack) |
| `src/evaluation/` | Evaluation: `run_inference.py` (raw per-frame logs at 8 FPS) and `evaluate.py` (offline state machine replay + metrics) |
| `src/face_detection/`, `src/prey_detection/` | Dataset building and training |
| `src/tools/` | Utilities (Gemini image generation, grayscale conversion, model checks) |
| `src/tests/` | Unit tests |
| `models/` | Final models (`cat_face.tflite`, `cat_prey.tflite`, `.pt` originals, `1stage_ablation.pt`) |
| `videos/veo/` | Test videos (not versioned), `annotations/` ground truth, `PROMPTS.md` |
| `notebooks/` | Kaggle training notebooks |
| `archive/` | Earlier ablation studies (CLAHE, crop, state changes per 100 frames) and one-off thesis scripts |

## Evaluation

```
python src/evaluation/run_inference.py --config two_stage
python src/evaluation/run_inference.py --config one_stage
python src/evaluation/evaluate.py --run results/raw/two_stage_track_grayscale --sweep
```
