import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers.schedules import CosineDecay, CosineDecayRestarts
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import LogFormatterSciNotation
import time
import os

# Configurazione CPU
os.environ["OMP_NUM_THREADS"] = "32"  
tf.config.threading.set_intra_op_parallelism_threads(30)
tf.config.threading.set_inter_op_parallelism_threads(4)

################################################
################# FUNCTIONS ####################
################################################

def build_model(num_layers, neurons, activation, input_dim=11, output_dim=40):
    model = Sequential()
    model.add(Dense(neurons, activation=activation, input_shape=(input_dim,)))
    for _ in range(num_layers - 1):
        model.add(Dense(neurons, activation=activation))
    model.add(Dense(output_dim))
    return model

################################################
##################### MAIN #####################
################################################

# Caricamento dati
data = np.loadtxt("filtered_dataset.txt")
np.random.seed(42)
np.random.shuffle(data)

X = data[:, :11]
y = data[:, 11:]

test_data = np.loadtxt("test_data.txt")
X_test = test_data[:, :11]
y_test = test_data[:, 11:]

# Parametri principali
epochs = 1		# Numero epoche
validation_split = 0.2	# Divisione train-validation
activation = 'swish'	# Funzione di attivazione
neurons = 64		# Numero neuroni
batch_size = 8		# Batch size
ntrain = int(len(X) - validation_split * len(X))	# Numero effettivo di campioni per training
steps_per_epoch = ntrain // batch_size			# Steps per ogni epoca

learning_rates = [1/512, 1/256, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1]	# 10 Learning Rates (base)
weight_decays =  [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4]		# 10 Weight decays
mse_results = np.zeros((len(learning_rates), len(weight_decays)))

T_max = 150*steps_per_epoch	# Numero di step per il primo ciclo
T_mul = 2.0  			# Moltiplica il numero di epoche per ogni ciclo successivo
m_mul = 1.0  			# Moltiplica l'ampiezza del learning
results = []

checkloss = 0	# Plot grafico train-validation loss [0-1]
saveloss = 0	# Salva grafico[0-1]

start_time = time.perf_counter()

# Ciclo su tutte le combinazioni
for i, lr in enumerate(learning_rates):
	for j, wd in enumerate(weight_decays):
		print(f"\nTraining with INITIAL LR={0.1*lr}, WD={0.001*wd}")
		
		### Vari modelli LR ###
	
		base_lr = 0.1 * lr	# learning rate iniziale
		end_lr = 0.001 * base_lr	# learning rate finale

		# Step Decay
		boundaries = [(T_max), (T_max)]  
		values = [base_lr, base_lr * 0.1, end_lr]
		SD = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
			boundaries = boundaries,
			values = values
		)

		#Cosine Decay
		CD = CosineDecay(
			initial_learning_rate = base_lr,
			decay_steps = steps_per_epoch * epochs,
			alpha = (end_lr / base_lr)  
		)

		#Cosine Decay Restarts
		CDR = CosineDecayRestarts(
			initial_learning_rate = base_lr,
			first_decay_steps = T_max,
			t_mul = T_mul, 
			m_mul = m_mul,
			alpha = (end_lr / base_lr)
		)

		# Nuovo modello ogni ciclo
		model = Sequential([
		Dense(neurons, activation=activation, input_shape=(11,)),
		Dense(neurons, activation=activation),
		Dense(neurons, activation=activation),
		Dense(40)
		])

		optimizer = tfa.optimizers.AdamW(weight_decay=0.001*wd, learning_rate=base_lr)

		model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

		# Nuovo file checkpoint per ogni run
		checkpoint_path = f"best_model_lr{lr}_wd{wd}.h5"
		checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', verbose=0)

		history = model.fit(
		X, y,
		validation_split=validation_split,
		epochs=epochs,
		batch_size=batch_size,
		verbose=0,
		callbacks=[checkpoint]
		)

		best_model = tf.keras.models.load_model(checkpoint_path)
		mse, mae = best_model.evaluate(X_test, y_test, verbose=0)
		mse_results[i, j] = mse
		
		results.append({
		'learning_rate': lr,
		'weight_decay': wd,
		'mse': mse_results[i, j]
		})
		
		print(f"Test MSE: {mse:.5f}")

		
		if checkloss == 1:
			# Plot delle curve di training e validation con scala logaritmica sull'asse Y
			plt.figure(figsize=(10, 6))
			plt.plot(history.history['loss'], label='Training Loss')
			plt.plot(history.history['val_loss'], label='Validation Loss')
			plt.xlabel('Epochs')
			plt.ylabel('Loss (log scale)')
			plt.title(f'Training and Validation Loss (Log Scale) - Batch Size {batch_size}')
			plt.yscale('log')  # Set the Y-axis to log scale
			plt.legend()
			plt.grid(True, which="both", linestyle="--", linewidth=0.5)
			plt.show()

			if saveloss == 1:
				plot_filename = f"loss_bs_{batch_size}_nl.png"
				plt.savefig(plot_filename, dpi=300)

				print(f"Plot saved as {plot_filename}")

s# Salvataggio risultati
df_results = pd.DataFrame(results)
df_results.to_csv("mse_results.csv", index=False)
print("Risultati salvati in 'mse_results.csv'")

elapsed = time.perf_counter() - start_time
print(f"Elapsed time: {elapsed:.3f} secondi")

################################################
##################### PLOTS ####################
################################################

# Labels per assi x ed y
lr_labels = ['1/512', '1/256', '1/128', '1/64', '1/32', '1/16', '1/8', '1/4', '1/2', '1']
wd_labels = ['1/128', '1/64', '1/32', '1/16', '1/8', '1/4', '1/2', '1', '2', '4']
X, Y = np.meshgrid(range(len(weight_decays)), range(len(learning_rates)))

# Clipping per i valori MSE per rientrare nel range [1e-3, 1]
mse_clipped = np.clip(mse_results, a_min=1e-3, a_max=1.0)

# Colormap/bar logaritmica con valori sopra 1 mappati a rosso
base_cmap = plt.cm.get_cmap('turbo', 256)
new_colors = base_cmap(np.linspace(0, 1, 256))
new_colors[-1] = plt.cm.turbo(1.0)  # ultimo colore rosso pieno
custom_cmap = colors.ListedColormap(new_colors)

levels = np.logspace(-3, 0, 256)
cbar = plt.colorbar(contour, ticks=[1e-3, 1e-2, 1e-1, 1e0])
cbar.set_label("MSE")
cbar.ax.yaxis.set_major_formatter(LogFormatterSciNotation())

fig, ax = plt.subplots(figsize=(8, 6))

contour = ax.contourf(
    X, Y, mse_clipped, levels=levels,
    norm=colors.LogNorm(vmin=1e-3, vmax=1.0),
    cmap=custom_cmap
)

# Evidenzia minimo MSE con cerchio e testo
min_mse = np.min(mse_results)
min_indices = np.argwhere(mse_results == min_mse)
for idx in min_indices:
    x, y = idx[1], idx[0]
    ax.plot(x, y, 'wo', markersize=6, markerfacecolor=None)
    ax.text(
        x + 0.3, y, f"{min_mse:.6f}",
        color='white', fontsize=9,
        va='center', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.6)
    )

ax.set_xticks(range(len(weight_decays)))
ax.set_xticklabels(wd_labels, rotation=45)
ax.set_yticks(range(len(learning_rates)))
ax.set_yticklabels(lr_labels)

ax.set_xlabel("Weight decay to be multiplied by 0.001")
ax.set_ylabel("Initial learning rate to be multiplied by 0.1")
ax.set_title("Test MSE - LR vs WD")

plt.tight_layout()
plt.savefig("wdVSlr_.png", dpi=300)
plt.show()




