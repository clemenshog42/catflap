import matplotlib.pyplot as plt
import numpy as np
import re
import os

file_path = r'C:\Public\Studium\Bachelorarbeit\ablation_clahe_results.txt'

data = []

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

i = 0
current_category = ""
while i < len(lines):
    line = lines[i].strip()
    
    if line.startswith('KATEGORIE:'):
        current_category = "Mit Beute" if "WITH_PREY" in line else "Ohne Beute"
        
    if line.startswith('Video: '):
        match = re.match(r'Video: (.*) \((\d+) Frames\)', line)
        if match:
            v_name = match.group(1).replace('_gray', '')
            f_count = int(match.group(2))
            
            # Baseline is next
            while not 'Baseline' in lines[i]: i += 1
            i += 1
            b_changes = int(re.search(r'Anzahl Zustandswechsel: (\d+)', lines[i]).group(1))
            
            # Ablation is next
            while not 'Ablation' in lines[i]: i += 1
            i += 1
            a_changes = int(re.search(r'Anzahl Zustandswechsel: (\d+)', lines[i]).group(1))
            
            data.append({
                'name': f"{v_name}\n({current_category})",
                'frames': f_count,
                'baseline': b_changes,
                'ablation': a_changes
            })
    i += 1

labels = [d['name'] for d in data]
frames = [d['frames'] for d in data]

b_raw = [d['baseline'] for d in data]
a_raw = [d['ablation'] for d in data]

b_norm = [b / f * 100 for b, f in zip(b_raw, frames)]
a_norm = [a / f * 100 for a, f in zip(a_raw, frames)]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - width/2, a_norm, width, label='Ohne CLAHE (Ablation)', color='#9467bd')
rects2 = ax.bar(x + width/2, b_norm, width, label='Mit CLAHE (Baseline)', color='#17becf')

ax.set_ylabel('Zustandswechsel (pro 100 Bilder)', fontsize=12)
ax.set_title('Auswirkung des CLAHE Filters auf das Flackern', fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0, fontsize=11)

# Put legend OUTSIDE or higher up to avoid overlap
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=12)

ax.bar_label(rects1, padding=3, fmt='%.1f')
ax.bar_label(rects2, padding=3, fmt='%.1f')

max_val = max(max(b_norm) if b_norm else 0, max(a_norm) if a_norm else 0)
ax.set_ylim(0, max_val + 2.0)

fig.tight_layout()
plt.savefig(r'C:\Public\Studium\Bachelorarbeit\images\ablation_clahe_state_changes.png', dpi=300, bbox_inches='tight')
print("CLAHE plot successfully generated from file!")

