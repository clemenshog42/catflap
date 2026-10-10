from ultralytics import YOLO
model = YOLO('models/1stage_ablation.pt')
print("Model classes:", model.names)
