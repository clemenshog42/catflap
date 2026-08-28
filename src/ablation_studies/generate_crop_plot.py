import matplotlib.pyplot as plt
import numpy as np
import re

file_path = r'C:\Public\Studium\Bachelorarbeit\ablation_crop_results.txt'

data = []
current_category = ""

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i].strip()
    if 'KATEGORIE: NO_PREY' in line:
        current_category = "Ohne Beute"
    elif 'KATEGORIE: WITH_PREY' in line:
        current_category = "Mit Beute"
        
    if line.startswith('Video: '):
        match = re.match(r'Video: (.*) \((\d+) Frames\)', line)
        if match:
            v_name = match.group(1)
            t_frames = int(match.group(2))
            
            # Baseline
            while not 'Mit asymmetrischem Cropping' in lines[i]: i += 1
            i += 1
            b_match = re.search(r'Fehler: (\d+) /', lines[i])
            b_err = int(b_match.group(1))
            
            # Ablation
            while not 'Symmetrisches Cropping' in lines[i]: i += 1
            i += 1
            a_match = re.search(r'Fehler: (\d+) /', lines[i])
            a_err = int(a_match.group(1))
            
            data.append({
                'name': f"{v_name}\n({current_category})",
                'total_frames': t_frames,
                'baseline': b_err,
                'ablation': a_err
            })
    i += 1

labels = [d['name'] for d in data]
b_raw = [d['baseline'] for d in data]
a_raw = [d['ablation'] for d in data]
t_frames = [d['total_frames'] for d in data]

# Normalizing to errors per 100 frames
b_norm = [b / f * 100 if f > 0 else 0 for b, f in zip(b_raw, t_frames)]
a_norm = [a / f * 100 if f > 0 else 0 for a, f in zip(a_raw, t_frames)]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, a_norm, width, label='Symmetrisches Padding', color='#ff7f0e')
rects2 = ax.bar(x + width/2, b_norm, width, label='Asymmetrisches Padding (Baseline)', color='#1f77b4')

ax.set_ylabel('Fehlerhafte Vorhersagen (pro 100 Frames)')
ax.set_title('Einfluss des Paddings auf die Modellgenauigkeit')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.legend()

ax.bar_label(rects1, padding=3, fmt='%.1f')
ax.bar_label(rects2, padding=3, fmt='%.1f')

max_val = max(max(b_norm) if b_norm else 0, max(a_norm) if a_norm else 0)
ax.set_ylim(0, max_val * 1.1 + 1.0)

fig.tight_layout()
plt.savefig(r'C:\Public\Studium\Bachelorarbeit\images\ablation_crop_state_changes.png', dpi=300)
plt.close(fig)

print("Plot generated successfully!")
