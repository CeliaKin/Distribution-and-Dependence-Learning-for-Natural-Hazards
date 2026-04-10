# EXPLORATPRY SCATTERPLOT


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr  


df = pd.read_csv('wind_flood_loss_pairs_events2.0.csv')
x = df['wind_loss'].dropna()
y = df['flood_loss'].dropna()


s_corr, _ = spearmanr(x, y)
p_corr, _ = pearsonr(x, y)  

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Times New Roman"],
    "axes.linewidth": 0.8,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5)) 

def draw_dependence_plot(ax, title):

    ax.scatter(x, y, color='#0077b6', alpha=0.5, s=15, 
                edgecolors='white', linewidth=0.3)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(False)

    ax.set_xlabel('Wind Loss ($ USD, Log)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Flood Loss ($ USD, Log)', fontsize=9, fontweight='bold')
    ax.set_title(title, fontsize=10, fontweight='bold', pad=12)

stats_text = f'Spearman: {s_corr:.2f} | Pearson: {p_corr:.2f}'

draw_dependence_plot(ax1, f'Dataset Analysis A\n{stats_text}')
draw_dependence_plot(ax2, f'Dataset Analysis B\n{stats_text}')

plt.tight_layout()
plt.savefig('side_by_side_comparison.png', dpi=600, bbox_inches='tight')
plt.show()

