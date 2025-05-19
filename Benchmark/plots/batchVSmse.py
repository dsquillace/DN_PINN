import pandas as pd
import matplotlib.pyplot as plt

# Dati da plottare
df = pd.read_csv("mse_results.csv")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df["Batch_size"], df["mse"], marker='o', linestyle='-', color='green')
plt.xlabel("Batch Size")
plt.ylabel("MSE")
plt.title("MSE vs Batch Size")
plt.grid(True)
plt.tight_layout()
plt.savefig("mse_vs_batch_size.png", dpi=300)
plt.show()
