import argparse
from ultralytics import YOLO

def export_model(model_path, format_type, half=False, imgsz=None):
    """
    Exports a YOLO model to a specific format.
    Supported formats include: 'openvino', 'tflite', 'onnx', 'engine', 'coreml', etc.
    """
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    export_kwargs = {"format": format_type, "half": half}
    if imgsz:
        export_kwargs["imgsz"] = imgsz
        
    print(f"Exporting model to {format_type} (half precision: {half})...")
    
    try:
        exported_path = model.export(**export_kwargs)
        print(f"✅ Export complete! Model saved at: {exported_path}")
    except Exception as e:
        print(f"❌ Export failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO models to different formats.")
    parser.add_argument("--model_path", required=True, help="Path to the trained .pt model (e.g., best.pt)")
    parser.add_argument("--format", required=True, choices=["openvino", "tflite", "onnx", "engine", "coreml", "pb"], help="Target format for export")
    parser.add_argument("--half", action="store_true", help="Use FP16 half-precision for faster inference (recommended if hardware supports it)")
    parser.add_argument("--imgsz", type=int, help="Optional specific image size for export (e.g., 224 or 640)")
    
    args = parser.parse_args()
    export_model(args.model_path, args.format, args.half, args.imgsz)
