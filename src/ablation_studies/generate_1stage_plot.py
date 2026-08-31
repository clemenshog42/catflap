import matplotlib.pyplot as plt
import numpy as np
import re
import os

file_path = r'C:\Public\Studium\Bachelorarbeit\ablation_1stage_results.txt'

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
            while not 'Baseline' in lines[i]: i += 1
            i += 1
            b_prey_match = re.search(r'Beute-Fehler: (\d+) /', lines[i])
            b_prey_err = int(b_prey_match.group(1))
            i += 1
            b_cat_match = re.search(r'Katze übersehen: (\d+) /', lines[i])
            b_cat_miss = int(b_cat_match.group(1))
            
            # 1-stufig
            while not 'Ablation (1-Stage)' in lines[i]: i += 1
            i += 1
            a_prey_match = re.search(r'Beute-Fehler: (\d+) /', lines[i])
            a_prey_err = int(a_prey_match.group(1))
            i += 1
            a_cat_match = re.search(r'Katze übersehen: (\d+) /', lines[i])
            a_cat_miss = int(a_cat_match.group(1))
            
            data.append({
                'name': f"{v_name}\n({current_category})",
                'total_frames': t_frames,
                'b_prey': b_prey_err,
                'b_cat': b_cat_miss,
                'a_prey': a_prey_err,
                'a_cat': a_cat_miss
            })
    i += 1

labels = [d['name'] for d in data]
t_frames = [d['total_frames'] for d in data]

# Normalizing to errors per 100 frames
b_prey_norm = [d['b_prey'] / f * 100 for d, f in zip(data, t_frames)]
a_prey_norm = [d['a_prey'] / f * 100 for d, f in zip(data, t_frames)]

b_cat_norm = [d['b_cat'] / f * 100 for d, f in zip(data, t_frames)]
a_cat_norm = [d['a_cat'] / f * 100 for d, f in zip(data, t_frames)]

x = np.arange(len(labels))
width = 0.35

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# Subplot 1: Beute-Fehler
rects1_b = ax1.bar(x - width/2, b_prey_norm, width, label='2-stufig (Baseline)', color='#1f77b4')
rects1_a = ax1.bar(x + width/2, a_prey_norm, width, label='1-stufig (Ablation)', color='#ff7f0e')

ax1.set_ylabel('Fehlentscheidungen (pro 100 Bilder)')
ax1.set_title('Beute-Erkennung: Fehlerhafte Klassifikationen')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=45, ha="right")
ax1.legend()
ax1.bar_label(rects1_b, padding=3, fmt='%.1f')
ax1.bar_label(rects1_a, padding=3, fmt='%.1f')
max_val1 = max(max(b_prey_norm) if b_prey_norm else 0, max(a_prey_norm) if a_prey_norm else 0)
ax1.set_ylim(0, max_val1 * 1.2 + 1.0)

# Subplot 2: Verpasste Katzen
rects2_b = ax2.bar(x - width/2, b_cat_norm, width, label='2-stufig (Baseline)', color='#1f77b4')
rects2_a = ax2.bar(x + width/2, a_cat_norm, width, label='1-stufig (Ablation)', color='#ff7f0e')

ax2.set_ylabel('Übersehene Katzen (pro 100 Bilder)')
ax2.set_title('Katzen-Detektion: Komplett verpasste Katzen (False Negatives)')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=45, ha="right")
ax2.legend()
ax2.bar_label(rects2_b, padding=3, fmt='%.1f')
ax2.bar_label(rects2_a, padding=3, fmt='%.1f')
max_val2 = max(max(b_cat_norm) if b_cat_norm else 0, max(a_cat_norm) if a_cat_norm else 0)
ax2.set_ylim(0, max_val2 * 1.2 + 1.0)

fig.tight_layout()
plt.savefig(r'C:\Public\Studium\Bachelorarbeit\images\ablation_1stage_evaluation.png', dpi=300)
plt.close(fig)
print("Plot generated successfully!")

