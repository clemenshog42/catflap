import matplotlib.pyplot as plt
import numpy as np
import re

file_path = r'C:\Public\Studium\Bachelorarbeit\ablation_results.txt'

no_prey_data = []
with_prey_data = []

current_list = None

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i].strip()
    if 'KATEGORIE: NO_PREY' in line:
        current_list = no_prey_data
    elif 'KATEGORIE: WITH_PREY' in line:
        current_list = with_prey_data
        
    if line.startswith('Video: '):
        match = re.match(r'Video: (.*) \((\d+) Frames\)', line)
        if match:
            v_name = match.group(1)
            f_count = int(match.group(2))
            
            # Baseline is next
            while not 'history_length=15' in lines[i]: i += 1
            i += 1
            b_changes = int(re.search(r'Anzahl Zustandswechsel: (\d+)', lines[i]).group(1))
            
            # Ablation is next
            while not 'history_length=1' in lines[i]: i += 1
            i += 1
            a_changes = int(re.search(r'Anzahl Zustandswechsel: (\d+)', lines[i]).group(1))
            
            current_list.append({
                'name': v_name,
                'frames': f_count,
                'baseline': b_changes,
                'ablation': a_changes
            })
    i += 1

def make_plot(data, title, out_path, color_ab, color_base):
    labels = [d['name'] for d in data]
    frames = [d['frames'] for d in data]
    
    b_raw = [d['baseline'] for d in data]
    a_raw = [d['ablation'] for d in data]
    
    b_norm = [b / f * 100 for b, f in zip(b_raw, frames)]
    a_norm = [a / f * 100 for a, f in zip(a_raw, frames)]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, a_norm, width, label='Ohne Glättung (history=1)', color=color_ab)
    rects2 = ax.bar(x + width/2, b_norm, width, label='Mit Zustandsautomat (history=15)', color=color_base)
    
    ax.set_ylabel('Zustandswechsel (pro 100 Frames)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')
    
    max_val = max(max(b_norm) if b_norm else 0, max(a_norm) if a_norm else 0)
    ax.set_ylim(0, max_val + 1.0)
    
    fig.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

make_plot(
    no_prey_data, 
    'Auswirkung der zeitlichen Glättung auf False Positives (Normalisiert)', 
    r'C:\Public\Studium\Bachelorarbeit\images\ablation_state_changes.png',
    '#ff7f0e', '#1f77b4'
)

make_plot(
    with_prey_data, 
    'Auswirkung der zeitlichen Glättung auf True Positives (Normalisiert)', 
    r'C:\Public\Studium\Bachelorarbeit\images\ablation_state_changes_with_prey.png',
    '#d62728', '#2ca02c'
)

print("Plots generated from file successfully!")
