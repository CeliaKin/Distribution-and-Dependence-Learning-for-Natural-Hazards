#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr  # Added pearsonr

# 1. Load your data
df = pd.read_csv('wind_flood_loss_pairs_events2.0.csv')
x = df['wind_loss'].dropna()
y = df['flood_loss'].dropna()

# Calculate both correlations
s_corr, _ = spearmanr(x, y)
p_corr, _ = pearsonr(x, y)  # Added Pearson calculation

# 2. Styling Configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Times New Roman"],
    "axes.linewidth": 0.8,
})

# 3. Create Side-by-Side Subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5)) # Widened slightly for text

def draw_dependence_plot(ax, title):
    # Scatter plot with refined markers
    ax.scatter(x, y, color='#0077b6', alpha=0.5, s=15, 
                edgecolors='white', linewidth=0.3)
    
    # Log-Log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Remove "Box" and Grid
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(False)
    
    # Labeling
    ax.set_xlabel('Wind Loss ($ USD, Log)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Flood Loss ($ USD, Log)', fontsize=9, fontweight='bold')
    ax.set_title(title, fontsize=10, fontweight='bold', pad=12)

# 4. Generate the plots with both metrics in the title
# Using a pipe '|' or newline to separate the two metrics
stats_text = f'Spearman: {s_corr:.2f} | Pearson: {p_corr:.2f}'

draw_dependence_plot(ax1, f'Dataset Analysis A\n{stats_text}')
draw_dependence_plot(ax2, f'Dataset Analysis B\n{stats_text}')

# 5. Final Polishing
plt.tight_layout()
plt.savefig('side_by_side_comparison.png', dpi=600, bbox_inches='tight')
plt.show()

