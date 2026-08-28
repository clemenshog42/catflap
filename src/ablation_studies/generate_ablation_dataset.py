import os
import cv2
import glob
import argparse
import random
import shutil
from ultralytics import YOLO

def process_directory(img_paths, face_detector, class_id, apply_padding):
    results_list = []
    
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        
        # Graustufen und CLAHE für den Face Detector (falls er darauf trainiert ist)`n        gray_img_for_det = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`n        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`n        gray_img_for_det = clahe.apply(gray_img_for_det)`n        # Manche Modelle erwarten 3 Kanäle (RGB), andere 1 Kanal. Wir lassen YOLO das regeln.`n        # Wir konvertieren sicherheitshalber zurück zu 3 Kanälen falls YOLO das intern braucht (YOLO packt das meist in 3 Kanäle).`n        gray_3ch = cv2.cvtColor(gray_img_for_det, cv2.COLOR_GRAY2BGR)`n        results = face_detector(gray_3ch, verbose=False)
        boxes_to_save = []
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                face_w, face_h = x2 - x1, y2 - y1
                
                if apply_padding:
                    pad_top = int(face_h * -0.35)
                    pad_bottom = int(face_h * 0.35)
                else:
                    pad_top = 0
                    pad_bottom = 0
                    
                new_y1 = max(0, y1 - pad_top)
                new_y2 = min(h, y2 + pad_bottom)
                new_x1 = max(0, x1)
                new_x2 = min(w, x2)
                
                box_w, box_h = new_x2 - new_x1, new_y2 - new_y1
                cx, cy = new_x1 + (box_w / 2), new_y1 + (box_h / 2)
                
                yolo_x, yolo_y = cx / w, cy / h
                yolo_w, yolo_h = box_w / w, box_h / h
                
                boxes_to_save.append(f"{class_id} {yolo_x:.6f} {yolo_y:.6f} {yolo_w:.6f} {yolo_h:.6f}")
                
        if boxes_to_save:
            results_list.append({
                'img_path': img_path,
                'boxes': boxes_to_save
            })
            
    return results_list

def main():
    parser = argparse.ArgumentParser(description='Auto-Annotate 1-Stage Detector Dataset')
    
    # Standard-Werte für lokale Ausführung hinzugefügt
    default_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    parser.add_argument('--dir_no_prey', type=str, 
                        default=os.path.join(default_base, 'videos', 'no_prey'),
                        help='Directory containing images without prey')
    parser.add_argument('--dir_with_prey', type=str, 
                        default=os.path.join(default_base, 'videos', 'with_prey'),
                        help='Directory containing images with prey')
    parser.add_argument('--output_dir', type=str, 
                        default=os.path.join(default_base, 'ablation_1stage_dataset'),
                        help='Output directory for YOLO dataset')
    parser.add_argument('--face_model', type=str, 
                        default=os.path.join(default_base, 'models', 'cat_face_2708.pt'),
                        help='Path to face detector model')
    
    args = parser.parse_args()
    
    print(f"Verwende dir_no_prey: {args.dir_no_prey}")
    print(f"Verwende dir_with_prey: {args.dir_with_prey}")
    print(f"Verwende face_model: {args.face_model}")
    print(f"Output-Verzeichnis: {args.output_dir}")
    
    images_train = os.path.join(args.output_dir, 'images', 'train')
    labels_train = os.path.join(args.output_dir, 'labels', 'train')
    os.makedirs(images_train, exist_ok=True)
    os.makedirs(labels_train, exist_ok=True)
    
    print("Loading face detector model...")
    face_detector = YOLO(args.face_model)
    
    paths_no_prey = []
    paths_with_prey = []
    for ext in ['*.jpg', '*.png', '*.jpeg', '*.mp4']: # MP4 reading fallback support below
        paths_no_prey.extend(glob.glob(os.path.join(args.dir_no_prey, '**', ext), recursive=True))
        paths_with_prey.extend(glob.glob(os.path.join(args.dir_with_prey, '**', ext), recursive=True))
        
    print(f"Found {len(paths_no_prey)} files in NO PREY folder.")
    print(f"Found {len(paths_with_prey)} files in WITH PREY folder.")
    
    # If the user provides MP4s instead of JPGs, extract frames temporarily
    def extract_if_video(paths):
        img_paths = []
        for p in paths:
            if p.lower().endswith('.mp4'):
                cap = cv2.VideoCapture(p)
                v_name = os.path.splitext(os.path.basename(p))[0]
                f_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    f_name = os.path.join(os.path.dirname(p), f"{v_name}_frame{f_idx}.jpg")
                    cv2.imwrite(f_name, frame)
                    img_paths.append(f_name)
                    f_idx += 1
                cap.release()
            else:
                img_paths.append(p)
        return img_paths
        
    print("Prüfe auf MP4 Videos und extrahiere Frames falls nötig...")
    paths_no_prey = extract_if_video(paths_no_prey)
    paths_with_prey = extract_if_video(paths_with_prey)
    
    print("Processing NO PREY images...")
    results_no_prey = process_directory(paths_no_prey, face_detector, class_id=0, apply_padding=False)
    
    print("Processing WITH PREY images...")
    results_with_prey = process_directory(paths_with_prey, face_detector, class_id=1, apply_padding=True)
    
    print(f"Successfully detected cats in {len(results_no_prey)} NO PREY images and {len(results_with_prey)} WITH PREY images.")
    
    min_count = min(len(results_no_prey), len(results_with_prey))
    print(f"Balancing dataset to {min_count} images per class...")
    
    random.shuffle(results_no_prey)
    random.shuffle(results_with_prey)
    
    final_dataset = results_no_prey[:min_count] + results_with_prey[:min_count]
    
    print("Saving dataset...")
    for idx, item in enumerate(final_dataset):
        orig_img_path = item['img_path']
        boxes = item['boxes']
        
        save_name = f"auto_ann_{idx:06d}.jpg"
        dest_img_path = os.path.join(images_train, save_name)
        img = cv2.imread(orig_img_path)`n        if img is not None:`n            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`n            # Optional: CLAHE anwenden (wie in der Baseline)`n            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`n            gray_img = clahe.apply(gray_img)`n            cv2.imwrite(dest_img_path, gray_img)`n
        
        label_path = os.path.join(labels_train, save_name.replace('.jpg', '.txt'))
        with open(label_path, 'w') as f:
            f.write('\n'.join(boxes))
            
    print(f"Generated {len(final_dataset)} annotated images in total.")
    
    yaml_path = os.path.join(args.output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(args.output_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/train\n\n")
        f.write("names:\n")
        f.write("  0: cat_without_prey\n")
        f.write("  1: cat_with_prey\n")
        
    print(f"data.yaml saved to {yaml_path}")

if __name__ == '__main__':
    main()



