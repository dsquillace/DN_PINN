import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers.schedules import CosineDecay, CosineDecayRestarts
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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
ntrain = int(len(X) - validation_split * len(X))

batch_size = list(range(2, ntrain + 1, 2))
mse_results = np.zeros((len(batch_size)))

lr = 0.001
wd = 0.0001
T_mul = 2.0	# Moltiplica il numero di epoche per ogni ciclo successivo (CDR)
m_mul = 1.0	# Moltiplica ampiezza del learning (CDR)
results = []

checkloss = 0	# Plot grafico train-validation loss [0-1]
saveloss = 0	# Salva grafico[0-1]

start_time = time.perf_counter()

# Ciclo su tutte le combinazioni

for i, bs in enumerate(batch_size):

	print(f"\nTraining with Batch Size={bs}")

	steps_per_epoch = ntrain // bs	#Numero step per epoca
	T_max = 150*steps_per_epoch  # Numero di step per il primo ciclo
	
	### Vari modelli LR ###
	
	base_lr = lr	# learning rate iniziale
	end_lr = 0.001 * lr	# learning rate finale 

	# Step Decay
	boundaries = [(T_max), (T_max*3)]  # step per i cambiamenti
	values = [base_lr, base_lr * 0.1, end_lr]  # valori scalati
	SD = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
		boundaries = boundaries,
		values = values
		)

	#Cosine Decay
	CD = CosineDecay(
		initial_learning_rate = base_lr,
		decay_steps = 1500 * steps_per_epoch,
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

	optimizer = tfa.optimizers.AdamW(weight_decay=wd, learning_rate=base_lr)
	model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

	# Nuovo file checkpoint per ogni run
	checkpoint_path = f"best_model_bs{bs}.h5"
	checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', verbose=0)

	history = model.fit(
		X, y,
		validation_split=validation_split,
		epochs=epochs,
		batch_size=bs,
		verbose=0,
		callbacks=[checkpoint]
		)

	best_model = tf.keras.models.load_model(checkpoint_path)
	mse, mae = best_model.evaluate(X_test, y_test, verbose=0)
	mse_results[i] = mse
		
	results.append({
	'Batch_size': bs,
	'mse': mse_results[i]
	})
		
	print(f"Test MSE: {mse:.5f}")

	if checkloss == 1:
			# Plot delle curve di training e validation con scala logaritmica sull'asse Y
		plt.figure(figsize=(10, 6))
		plt.plot(history.history['loss'], label='Training Loss')
		plt.plot(history.history['val_loss'], label='Validation Loss')
		plt.xlabel('Epochs')
		plt.ylabel('Loss (log scale)')
		plt.title(f'Training and Validation Loss (Log Scale) - Batch Size {bs}')
		plt.yscale('log')  # Set the Y-axis to log scale
		plt.legend()
		plt.grid(True, which="both", linestyle="--", linewidth=0.5)
		plt.show()
		
		if saveloss == 1:
				# Salvataggio del plot
			plot_filename = f"loss_bs{bs}.png"
			plt.savefig(plot_filename, dpi=300)
			print(f"Plot saved as {plot_filename}")



# Salvataggio risultati
df_results = pd.DataFrame(results)
df_results.to_csv("mse_results.csv", index=False)
print("Risultati salvati in 'mse_results.csv'")

elapsed = time.perf_counter() - start_time
print(f"Elapsed time: {elapsed:.3f} secondi")

################################################
##################### PLOTS ####################
################################################

plt.figure(figsize=(10, 6))
plt.plot(batch_size, mse_results, marker='o', linestyle='-', color='blue')
plt.xlabel('Numero di Batch')
plt.ylabel('MSE')
plt.title('MSE vs Numero di Batch')
plt.grid(True)
plt.tight_layout()
plt.savefig("mse_vs_num_batches.png", dpi=300)
plt.show()
