import matplotlib.pyplot as plt
import numpy as np

# Data from ablation_clahe_results.txt (history_length=1)
labels = ['no_prey (two_cats)', 'with_prey (black)']
frames = [1395, 240]
baseline_raw = [42, 1]       # Mit CLAHE
ablation_raw = [14, 21]      # Ohne CLAHE

# Normalize to state changes per 100 frames
baseline = [b / f * 100 for b, f in zip(baseline_raw, frames)]
ablation = [a / f * 100 for a, f in zip(ablation_raw, frames)]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, ablation, width, label='Ohne CLAHE (Ablation)', color='#9467bd')
rects2 = ax.bar(x + width/2, baseline, width, label='Mit CLAHE (Baseline)', color='#17becf')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Zustandswechsel (pro 100 Bilder)')
ax.set_title('Auswirkung des CLAHE Filters (ohne Automaten)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)

# Format labels to 2 decimal places
ax.bar_label(rects1, padding=3, fmt='%.2f')
ax.bar_label(rects2, padding=3, fmt='%.2f')
ax.set_ylim(0, max(max(baseline), max(ablation)) + 0.5)

fig.tight_layout()
plt.savefig(r'C:\Public\Studium\Bachelorarbeit\images\ablation_clahe_state_changes.png', dpi=300, bbox_inches='tight')
print("Plot 3 gespeichert!")
