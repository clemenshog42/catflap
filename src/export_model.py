import argparse
from ultralytics import YOLO

def export_model(model_path, format_type, half=False, int8=False, imgsz=None):
    """
    Exports a YOLO model to a specific format.
    Supported formats include: 'openvino', 'tflite', 'litert', 'onnx', 'engine', 'coreml', etc.
    """
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    export_kwargs = {"format": format_type}
    
    if format_type == "litert":
        if int8:
            print("Using 'w8a32' (Dynamic INT8) quantization for LiteRT...")
            export_kwargs["quantize"] = "w8a32"
        elif half:
            print("Warning: '--half' (FP16) is not supported by Ultralytics for LiteRT. Defaulting to FP32.")
    else:
        if half:
            export_kwargs["half"] = True
        if int8:
            export_kwargs["int8"] = True
        
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
    parser.add_argument("--format", required=True, choices=["openvino", "tflite", "litert", "onnx", "engine", "coreml", "pb"], help="Target format for export")
    parser.add_argument("--half", action="store_true", help="Use FP16 half-precision for faster inference")
    parser.add_argument("--int8", action="store_true", help="Use INT8 quantization (recommended for Raspberry Pi)")
    parser.add_argument("--imgsz", type=int, help="Optional specific image size for export (e.g., 224 or 640)")
    
    args = parser.parse_args()
    export_model(args.model_path, args.format, args.half, args.int8, args.imgsz)
