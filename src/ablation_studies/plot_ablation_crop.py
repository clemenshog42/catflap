import matplotlib.pyplot as plt
import numpy as np

# Data from ablation_crop_results.txt (history_length=1)
labels = ['no_prey (two_cats)', 'with_prey (black)', 'with_prey (two_cats)']
frames = [1395, 240, 240]
baseline_raw = [30, 11, 1]       # Asymmetric (pad_bottom=30)
ablation_raw = [12, 17, 15]      # Symmetric (pad_bottom=0)

# Normalize to state changes per 100 frames
baseline = [b / f * 100 for b, f in zip(baseline_raw, frames)]
ablation = [a / f * 100 for a, f in zip(ablation_raw, frames)]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, ablation, width, label='Ohne Randerweiterung (Ablation)', color='#d62728')
rects2 = ax.bar(x + width/2, baseline, width, label='Nur nach unten (Baseline)', color='#2ca02c')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Zustandswechsel (pro 100 Bilder)')
ax.set_title('Auswirkung der Randerweiterung nach unten (ohne Automaten)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)

# Format labels to 2 decimal places
ax.bar_label(rects1, padding=3, fmt='%.2f')
ax.bar_label(rects2, padding=3, fmt='%.2f')
ax.set_ylim(0, max(max(baseline), max(ablation)) + 0.5)

fig.tight_layout()
plt.savefig(r'C:\Public\Studium\Bachelorarbeit\images\ablation_crop_state_changes.png', dpi=300, bbox_inches='tight')
print("Plot gespeichert!")
