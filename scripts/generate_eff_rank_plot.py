import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
})

stages = ['res2', 'res3', 'res4', 'res5']
x = np.arange(len(stages))

data_no_ds = {
    0.25: [0.81, 0.82, 0.76, 0.27],
    0.35: [0.80, 0.84, 0.71, 0.48],
    0.40: [0.81, 0.84, 0.66, 0.50],
    0.50: [0.82, 0.83, 0.69, 0.55]
}

data_with_ds = {
    0.25: [0.81, 0.82, 0.80, 0.73],
    0.35: [0.81, 0.84, 0.74, 0.72],
    0.40: [0.82, 0.84, 0.70, 0.72],
    0.50: [0.81, 0.85, 0.72, 0.70]
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

colors = {0.5: '#95a5a6', 0.4: '#7f8c8d', 0.35: '#34495e', 0.25: '#e74c3c'} # Grays + Red
styles = {0.5: '--', 0.4: '--', 0.35: '--', 0.25: '-'} # Dash others, Solid Red
markers = {0.5: 'o', 0.4: 'o', 0.35: 'o', 0.25: 'o'} # Distinct markers

for ds, vals in data_no_ds.items():
    ax1.plot(x, vals, marker=markers[ds], label=f'ds_rate={ds}', 
             color=colors[ds], linestyle=styles[ds], linewidth=5 if ds==0.25 else 4, markersize=8)

ax1.set_title("Without Deep Supervision", fontsize=22)
ax1.set_xticks(x)
ax1.set_xticklabels(stages, fontsize=18)
ax1.set_ylabel("Normalized Effective Rank", fontsize=18)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.set_ylim(0.25, 0.9) 

for ds, vals in data_with_ds.items():
    ax2.plot(x, vals, marker=markers[ds], label=f'$d_s={ds}$', 
             color=colors[ds], linestyle=styles[ds], linewidth=5 if ds==0.25 else 4, markersize=8)

ax2.set_title("Deep Supervision", fontsize=22)
ax2.set_xticks(x)
ax2.set_xticklabels(stages, fontsize=18)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='lower left', fontsize=16, frameon=True)

plt.tight_layout()
plt.savefig('rank_collapse_zoomed.png', dpi=300)
plt.show()