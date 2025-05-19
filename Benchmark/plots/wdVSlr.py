import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import LogFormatterSciNotation

# Dati da plottare
df = pd.read_csv("mse_results.csv")

# Ordina valori
learning_rates = sorted(df["learning_rate"].unique())
weight_decays = sorted(df["weight_decay"].unique())

# Matrice MSE [lr x wd]
mse_results = np.zeros((len(learning_rates), len(weight_decays)))
for i, lr in enumerate(learning_rates):
    for j, wd in enumerate(weight_decays):
        mse = df[(df["learning_rate"] == lr) & (df["weight_decay"] == wd)]["mse"].values[0]
        mse_results[i, j] = mse

# Labels per assi x ed y
lr_labels = ['1/512', '1/256', '1/128', '1/64', '1/32', '1/16', '1/8', '1/4', '1/2', '1']
wd_labels = ['1/128', '1/64', '1/32', '1/16', '1/8', '1/4', '1/2', '1', '2', '4']
X, Y = np.meshgrid(range(len(weight_decays)), range(len(learning_rates)))
min_mse = np.min(mse_results)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))

# Colormap/bar logaritmica con valori sopra 1 mappati a rosso
base_cmap = plt.cm.get_cmap('turbo', 256)
new_colors = base_cmap(np.linspace(0, 1, 256))
new_colors[-1] = plt.cm.turbo(1.0)
custom_cmap = colors.ListedColormap(new_colors)

levels = np.logspace(-3, 0, 256)
mse_clipped = np.clip(mse_results, a_min=1e-3, a_max=1.0)
contour = ax.contourf(X, Y, mse_clipped, levels=levels,
                      norm=colors.LogNorm(vmin=1e-3, vmax=1.0),
                      cmap=custom_cmap)

cbar = plt.colorbar(contour, ticks=[1e-3, 1e-2, 1e-1, 1e0])
cbar.set_label("MSE")
cbar.ax.yaxis.set_major_formatter(LogFormatterSciNotation())

# Evidenzia il minimo MSE
min_mse = np.min(mse_results)
min_indices = np.argwhere(mse_results == min_mse)
for idx in min_indices:
    x, y = idx[1], idx[0]
    ax.plot(x, y, 'wo', markersize=6, markerfacecolor=None)
    ax.text(x + 0.3, y, f"{min_mse:.6f}", color='white', fontsize=9,
            va='center', ha='left', bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.6))

# Etichette assi
ax.set_xticks(range(len(weight_decays)))
ax.set_xticklabels(wd_labels, rotation=45)
ax.set_yticks(range(len(learning_rates)))
ax.set_yticklabels(lr_labels)

ax.set_xlabel("Weight decay to be multiplied by 0.001")
ax.set_ylabel("Initial learning rate to be multiplied by 0.1")
ax.set_title("Test MSE - LR vs WD")

# Layout e salvataggio
plt.tight_layout()
plt.savefig("wdVSlr_.png", dpi=300)
plt.show()

