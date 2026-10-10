import cv2
import os

def convert_to_grayscale(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening {input_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    
    print(f"Converting {input_path} to {output_path}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(gray)
        
    cap.release()
    out.release()
    print("Done.")

base_dir = r"C:\Public\Studium\Bachelorarbeit\code\videos"
for cat in ["no_prey", "with_prey"]:
    for vid in ["black.mp4", "orange.mp4"]:
        in_p = os.path.join(base_dir, cat, vid)
        out_p = os.path.join(base_dir, cat, vid.replace(".mp4", "_gray.mp4"))
        if os.path.exists(in_p):
            convert_to_grayscale(in_p, out_p)
