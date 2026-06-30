import os
import cv2
import argparse
import random
from pathlib import Path
from ultralytics import YOLO

def test_padding(input_dir, output_dir, crop_model_path, pad_w=10, pad_h=10, num_samples=10, seed=42):
    random.seed(seed)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not crop_model_path:
        raise ValueError("crop_model_path must be provided to test padding")
    crop_model = YOLO(crop_model_path)
    
    # Get all valid files
    files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    if not files:
        print(f"No images found in {input_dir}")
        return
        
    sampled_files = random.sample(files, min(len(files), num_samples))
    print(f"Testing padding on {len(sampled_files)} images...")
    
    for i, f in enumerate(sampled_files):
        img = cv2.imread(str(f))
        if img is None:
            continue
            
        # We will save two versions: one with a drawn bounding box, one actual crop
        results = crop_model(str(f), verbose=False)[0]
        if len(results.boxes) > 0:
            box = results.boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            
            h, w = img.shape[:2]
            
            # Original box in RED
            img_drawn = img.copy()
            cv2.rectangle(img_drawn, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Add padding
            x1_pad = max(0, x1 - pad_w)
            y1_pad = max(0, y1 - pad_h)
            x2_pad = min(w, x2 + pad_w)
            y2_pad = min(h, y2 + pad_h)
            
            # Padded box in GREEN
            cv2.rectangle(img_drawn, (x1_pad, y1_pad), (x2_pad, y2_pad), (0, 255, 0), 2)
            
            # Actual crop
            img_cropped = img[y1_pad:y2_pad, x1_pad:x2_pad]
            
            # Save visual test
            cv2.imwrite(str(output_path / f"test_{i}_boxes.jpg"), img_drawn)
            cv2.imwrite(str(output_path / f"test_{i}_crop.jpg"), img_cropped)
        else:
            print(f"No face detected in {f.name}")

    print(f"✅ Saved padding tests to {output_dir}")
    print("Green box = Padded Crop, Red Box = Original Detection")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Input directory of sample images")
    parser.add_argument("--output_dir", required=True, help="Output directory for test visualizations")
    parser.add_argument("--crop_model_path", required=True, help="Path to YOLO cat face model")
    parser.add_argument("--pad_w", type=int, default=10, help="Horizontal padding (pixels) to test")
    parser.add_argument("--pad_h", type=int, default=10, help="Vertical padding (pixels) to test")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of random samples to test")
    
    args = parser.parse_args()
    test_padding(args.input_dir, args.output_dir, args.crop_model_path, args.pad_w, args.pad_h, args.num_samples)
