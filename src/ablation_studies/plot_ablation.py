import matplotlib.pyplot as plt
import numpy as np

# Data from ablation_results.txt
labels = ['black.mp4', 'orange.mp4', 'two_cats.mp4']
frames = [240, 240, 1395]
baseline_raw = [2, 1, 1]       # history_length = 15
ablation_raw = [6, 7, 42]      # history_length = 1

# Normalize to state changes per 100 frames to make the y-axis readable
baseline = [b / f * 100 for b, f in zip(baseline_raw, frames)]
ablation = [a / f * 100 for a, f in zip(ablation_raw, frames)]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, ablation, width, label='Ohne Glättung (history=1)', color='#ff7f0e')
rects2 = ax.bar(x + width/2, baseline, width, label='Mit Zustandsautomat (history=15)', color='#1f77b4')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Zustandswechsel (pro 100 Bilder)')
ax.set_title('Auswirkung der zeitlichen Glättung auf False Positives (Normalisiert)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Format labels to 2 decimal places
ax.bar_label(rects1, padding=3, fmt='%.2f')
ax.bar_label(rects2, padding=3, fmt='%.2f')

fig.tight_layout()
plt.savefig(r'C:\Public\Studium\Bachelorarbeit\images\ablation_state_changes.png', dpi=300)
print("Plot 1 gespeichert!")
