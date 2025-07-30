import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.data import AUTOTUNE
import subprocess

# Enable mixed precision
set_global_policy('mixed_float16')

#Load x coords 

BASE_PATH = '/home/damiano/NeuralNetwork/resources/airf'
AIRFOIL_FILE = 'NACA0012_120.dat'
AIRFOIL_PATH = os.path.join(BASE_PATH, AIRFOIL_FILE)
with open(AIRFOIL_PATH, 'r') as infile:
    xo, yo = np.loadtxt(infile, dtype=float, unpack=True, skiprows=1)
    
xc = (xo[:-1] + xo[1:]) / 2
n = len(xc)//2
dx = xc[1:n] - xc[:(n-1)]
dx = np.float32(dx)


################################################
################# FUNCTIONS ####################
################################################

def build_model(input_dim, output_dim, num_layers, neurons, activation):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    for _ in range(num_layers):
        model.add(Dense(neurons, activation=activation))
    model.add(Dense(output_dim, dtype='float32'))  # Ensure output is float32
    return model

def compute_cl(cp, dx):
    n = cp.shape[1] // 2
    cp_l = cp[:, :n]
    cp_u = cp[:, n:]
    cp_l_avg = 0.5 * (cp_l[:, 1:] + cp_l[:, :-1])
    cp_u_avg = 0.5 * (cp_u[:, 1:] + cp_u[:, :-1])
    cl = -tf.reduce_sum((cp_l_avg - cp_u_avg) * tf.expand_dims(dx, axis=0), axis=1)
    return cl


def custom_loss(y_true, y_pred):
    overall_mse = tf.reduce_mean(tf.square(y_pred - y_true))
    cl_true = compute_cl(y_true, dx)
    cl_pred = compute_cl(y_pred,dx)
    mse_cl = tf.reduce_mean(tf.square(cl_pred-cl_true))
    return overall_mse + 1 * mse_cl

################################################
##################### MAIN #####################
################################################

# Load and prepare data
data = np.loadtxt("filtered_dataset.txt")
np.random.seed(42)
np.random.shuffle(data)

n_par = 122
input_dim, output_dim = n_par, 120
X = data[:, :n_par]
y = data[:, n_par:]

# Hyperparameters
epochs = 2000
validation_split = 0.2
activation = 'swish'
neurons = 144 * 2
layers = 4 * 2
steps_per_epoch = 15
fixed_batch_size = 128

base_lr = 0.1 * 1/32
end_lr = 0.0001 * base_lr
wd = 0.001 * 1/64
T_max = 150 * steps_per_epoch

CDR = CosineDecayRestarts(
    initial_learning_rate=base_lr,
    first_decay_steps=T_max,
    t_mul=2.0,
    m_mul=1.0,
    alpha=(end_lr / base_lr)
)

checkpoint_path = "best_model.keras"
max_iterations = 50
error_threshold = 0.0001
num_to_add = 500

all_train_loss, all_val_loss, mse_history = [], [], []
start_time = time.perf_counter()

for loop_idx in range(max_iterations):
    print(f"\n=== Iteration {loop_idx + 1} ===")

    # Rebuild model from scratch
    if fixed_batch_size == 0:
        batch_size = max(4, len(X) // steps_per_epoch)
    else:
        batch_size = fixed_batch_size

    model = build_model(input_dim, output_dim, layers, neurons, activation)
    optimizer = AdamW(learning_rate=CDR, weight_decay=wd)
    model.compile(optimizer=optimizer, loss=custom_loss, metrics=['mae'])

    # Load previous best weights if available
    if loop_idx > 0 and os.path.exists(checkpoint_path):
        print("🔄 Loading existing weights...")
        model.load_weights(checkpoint_path)

    # Dataset preparation with tf.data
    perm = np.random.permutation(len(X)) #shuffle the dataset everytime
    X = X[perm]
    y = y[perm]

    val_size = int(validation_split * len(X))
    X_val, y_val = X[:val_size], y[:val_size]
    X_train, y_train = X[val_size:], y[val_size:]

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=len(X_train)).batch(batch_size).prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(AUTOTUNE)

    # Checkpointing
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True,
                                 monitor='val_loss', mode='min', verbose=1)

    # Training
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1,
        callbacks=[checkpoint]
    )

    all_train_loss.extend(history.history['loss'])
    all_val_loss.extend(history.history['val_loss'])

    # Evaluate on external test set
    model.load_weights(checkpoint_path)
    print("🔍 Evaluating model...")

    subprocess.run(["python", "/home/damiano/NeuralNetwork/compute_dn_par.py"], check=True)
    test_data = np.loadtxt("test_data.txt")
    X_test = test_data[:, :n_par]
    y_test = test_data[:, n_par:]

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(AUTOTUNE)
    y_pred = model.predict(test_ds, verbose=0)
    mse = np.mean((y_test - y_pred)**2)
    mse_history.append(mse)

    print(f"Test MSE: {mse:.5f}")
    if mse <= error_threshold:
        print(f"✅ Convergence reached: Test MSE ({mse:.5f}) <= Threshold ({error_threshold})")
        break

    # Add worst samples to training set
    rmse_errors = np.sqrt(np.mean((y_test - y_pred)**2, axis=1))
    worst_indices = np.argsort(rmse_errors)[-num_to_add:]
    new_X = X_test[worst_indices]
    new_y = y_test[worst_indices]

    X = np.vstack((X, new_X))
    y = np.vstack((y, new_y))

    # Save updated dataset
    np.savetxt("new_filtered_dataset.txt", np.hstack((X, y)))

# Final summary
elapsed = time.perf_counter() - start_time
print(f"\n🕒 Total elapsed time: {elapsed:.2f} seconds")
print(f"Final training set size: {len(X)} samples")

with open("epoch_loss_log.txt", "w") as f:
    f.write("Epoch\tTrain_Loss\tVal_Loss\n")
    for i, (train, val) in enumerate(zip(all_train_loss, all_val_loss)):
        f.write(f"{i+1}\t{train:.8f}\t{val:.8f}\n")

# Plot loss
plt.figure(figsize=(10, 6))
plt.plot(all_train_loss, label='Training Loss')
plt.plot(all_val_loss, label='Validation Loss')
plt.xlabel('Epochs (Cumulative)')
plt.ylabel('Loss (log scale)')
plt.title('Training and Validation Loss Across All Iterations')
plt.yscale('log')
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve_all_iterations.png", dpi=300)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(mse_history) + 1), mse_history, marker='o', color='tab:blue')
plt.xlabel('Iteration')
plt.ylabel('Test MSE')
plt.title('Test MSE Over Iterations')
plt.yscale('log')
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("mse_over_iterations.png", dpi=300)
