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
epochs = 1000		# Numero epoche
validation_split = 0.2	# Divisione train-validation
activation = 'swish'	# Funzione di attivazione
neurons = 64		# Numero neuroni
batch_size = 8		# Batch size
ntrain = int(len(X) - validation_split * len(X))	# Numero effettivo di campioni per training
steps_per_epoch = ntrain // batch_size			# Steps per ogni epoca

base_lr = 0.1 * 1/64
end_lr = 0.001 * base_lr
wd = 0.001 * 1/16

T_max = 150*steps_per_epoch	# Numero di step per il primo ciclo
T_mul = 2.0  			# Moltiplica il numero di epoche per ogni ciclo successivo
m_mul = 1.0  			# Moltiplica l'ampiezza del learning

neurons = [40, 80, 120, 160, 200, 240, 280, 320, 360, 400]
#neurons = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]	# Numero neuroni per layer 
layers =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]		# Numero di layers

mse_results = np.zeros((len(neurons), len(layers)))

### Vari modelli LR ###

# Step Decay
boundaries = [(T_max), (3*T_max)]  # step per i cambiamenti
values = [base_lr, base_lr * 0.1, end_lr]  # valori scalati

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

results = []
checkloss = 0	# Plot grafico train-validation loss [0-1]
saveloss = 0	# Salva grafico[0-1]

start_time = time.perf_counter()
# Ciclo su tutte le combinazioni
for i, ne in enumerate(neurons):
	for j, la in enumerate(layers):
		print(f"\nTraining with Neu={ne}, Lay={la}")
		
		# Nuovo modello ogni ciclo

		model = build_model(la,ne, activation, 11, 40)



		optimizer = tfa.optimizers.AdamW(weight_decay=wd, learning_rate=CDR)
		model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

		# Nuovo file checkpoint per ogni run
		checkpoint_path = f"best_model_N{ne}_L{la}.h5"
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
		'Neurons': ne,
		'Layers': la,
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

# Converti in DataFrame
df_results = pd.DataFrame(results)

# Salva in CSV
df_results.to_csv("mse_results.csv", index=False)
print("Risultati salvati in 'mse_results.csv'")

elapsed = time.perf_counter() - start_time
print(f"Elapsed time: {elapsed:.3f} secondi")
################################################
##################### PLOTS ####################
################################################

# Labels per assi x ed y
ne_labels = ['10', '20', '30', '40', '50', '60', '70', '90', '90', '100']
la_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
X, Y = np.meshgrid(range(len(layers)), range(len(neurons)))
min_mse = np.min(mse_results)

# Creazione del plot
fig, ax = plt.subplots(figsize=(8, 6))
levels = np.linspace(np.min(mse_results), np.max(mse_results), 256)
contour = ax.contourf(X, Y, mse_results, levels=levels,
                      cmap='turbo')

# Evidenzia minimo MSE con cerchio
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

# Colorbar lineare
cbar = plt.colorbar(contour)
cbar.set_label("MSE")

# Etichette assi con le frazioni
ax.set_xticks(range(len(layers)))
ax.set_xticklabels(la_labels, rotation=45)
ax.set_yticks(range(len(neurons)))
ax.set_yticklabels(ne_labels)

# Etichette e titolo
ax.set_xlabel("Number of Layers")
ax.set_ylabel("Number of Neurons")
ax.set_title("Test MSE - Layers vs neurons")

plt.tight_layout()
plt.savefig("mse_contour_map.png", dpi=300)
plt.show()
