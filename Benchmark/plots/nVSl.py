import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import LogFormatterSciNotation

# Dati da plottare
df = pd.read_csv("mse_results.csv")

# Ordina valori
neurons = sorted(df["Neurons"].unique())
layers = sorted(df["Layers"].unique())

# Matrice MSE [lr x wd]
mse_results = np.zeros((len(neurons), len(layers)))
for i, ne in enumerate(neurons):
    for j, la in enumerate(layers):
        mse = df[(df["Neurons"] == ne) & (df["Layers"] == la)]["mse"].values[0]
        mse_results[i, j] = mse

# Labels per assi x e y
ne_labels = [str(int(n)) for n in neurons]
la_labels = [str(int(l)) for l in layers]
X, Y = np.meshgrid(range(len(layers)), range(len(neurons)))
min_mse = np.min(mse_results)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))

# Colormap/bar lineare
levels = np.linspace(np.min(mse_results), np.max(mse_results), 256)
contour = ax.contourf(X, Y, mse_results, levels=levels,
                      cmap='turbo')
cbar = plt.colorbar(contour)
cbar.set_label("MSE")

# Evidenzia il minimo MSE
min_indices = np.argwhere(mse_results == min_mse)
for idx in min_indices:
    x, y = idx[1], idx[0]
    ax.plot(x, y, 'wo', markersize=6, markerfacecolor=None)
    # Aggiunta del valore del minimo MSE
    ax.text(x + 0.3, y, f"{min_mse:.6f}", color='white', fontsize=9,
            va='center', ha='left', bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.6))

# Etichette assi
ax.set_xticks(range(len(layers)))
ax.set_xticklabels(la_labels, rotation=45)
ax.set_yticks(range(len(neurons)))
ax.set_yticklabels(ne_labels)

ax.set_xlabel("Number of Layers")
ax.set_ylabel("Number of Neurons")
ax.set_title("Test MSE - Layers vs neurons")

# Layout e salvataggio
plt.tight_layout()
plt.savefig("nVSl_.png", dpi=300)
plt.show()

