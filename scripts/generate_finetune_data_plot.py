import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# 1. Matching the PSD Professional Style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in"
})

percentages = [5, 25, 50, 75, 100]
vit_scores = [0.2529, 0.4805, 0.5231, 0.5788, 0.6054]
aff_scores = [0.3910, 0.4968, 0.5364, 0.5747, 0.6030]

# Professional Palette from PSD (Black and Deep Blue/Orange)
c_vit = '#0072B2' # Blue
c_aff = '#D55E00' # Vermillion/Orange

fig, ax = plt.subplots(figsize=(8, 6))

# 2. Plotting with higher z-order for markers
ax.plot(percentages, vit_scores, marker='o', markersize=9, linestyle='-', 
         linewidth=3.5, color=c_vit, label='ViT', zorder=3)

ax.plot(percentages, aff_scores, marker='s', markersize=9, linestyle='-', 
         linewidth=3.5, color=c_aff, label='AFF', zorder=3)

# Shaded area for the gap
ax.fill_between(percentages, vit_scores, aff_scores, 
                 color='gray', alpha=0.5, label='Performance Gap')

# 3. Refined Labeling
ax.set_xlabel("Training Data Used (%)", fontsize=20, labelpad=10)
ax.set_ylabel("mIoU Score", fontsize=20, labelpad=10)

# 4. Ticks and Grid
ax.set_xticks(percentages)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax.set_ylim(0.24, 0.61) 

# Scientific grid style
ax.grid(True, which="major", ls="-", lw=1, alpha=0.4)
ax.minorticks_on()
ax.grid(True, which="minor", ls=":", lw=0.5, alpha=0.2)

# 5. Legend (No frame, matching PSD style)
ax.legend(fontsize=14, frameon=False, loc='lower right')

# Remove the top and right spines for that clean "paper" look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('icml_convergence_plot.png', bbox_inches='tight', dpi=300)
plt.show()