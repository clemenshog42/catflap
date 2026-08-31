import matplotlib.pyplot as plt
import numpy as np

# These will be updated based on the ablation_results.txt once finished
labels = ['black.mp4', 'orange.mp4', 'orange2.mp4', 'two_cats.mp4']
frames = [240, 164, 222, 240]

baseline_raw = [1, 1, 1, 1]       # history_length = 15
ablation_raw = [11, 1, 3, 15]     # history_length = 1 (we expect these values based on crop ablation baseline)

# Normalize to state changes per 100 frames
baseline = [b / f * 100 for b, f in zip(baseline_raw, frames)]
ablation = [a / f * 100 for a, f in zip(ablation_raw, frames)]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, ablation, width, label='Ohne Glättung (history=1)', color='#ff7f0e')
rects2 = ax.bar(x + width/2, baseline, width, label='Mit Zustandsautomat (history=15)', color='#1f77b4')

ax.set_ylabel('Zustandswechsel (pro 100 Bilder)')
ax.set_title('Auswirkung der zeitlichen Glättung auf True Positives (Normalisiert)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3, fmt='%.2f')
ax.bar_label(rects2, padding=3, fmt='%.2f')
ax.set_ylim(0, max(max(baseline), max(ablation)) + 1.0)

fig.tight_layout()
plt.savefig(r'C:\Public\Studium\Bachelorarbeit\images\ablation_state_changes_with_prey.png', dpi=300)
print("Plot saved as ablation_state_changes_with_prey.png")
