import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib 
import time

################################################
################# FUNCTIONS ####################
################################################

def lr_scheduler(epoch, lr):
    if (epoch + 1) % 1000 == 0:  
        return lr * 1.0
    return lr

def finite_difference_check(points):

    # Estrazione dei valori
    xc, yc, uc, vc = points['center']
    xl, yl, ul, vl = points['left']
    xr, yr, ur, vr = points['right']
    xt, yt, ut, vt = points['top']
    xb, yb, ub, vb = points['bottom']
    
    # Calcolo dello spazio in x e y (passi uniformi)
    dx = xr - xc  # oppure xc - xl
    dy = yt - yc  # oppure yc - yb
    
    # Differenze finite centrate
    dudx = (ur - ul) / (2 * dx)
    dvdy = (vt - vb) / (2 * dy)
    
    # Somma delle derivate
    divergence = dudx + dvdy
    
    return divergence

def OG_custom_loss(y_true, y_pred):

    # Valori originali
    vx_true, vz_true = y_true[:, 0], y_true[:, 1]

    # Ricerca vettorializzata degli indici
    condition = tf.reduce_all(tf.equal(data_ref[:, None, :], y_true[None, :, :]), axis=2)
    batch_indices = tf.argmax(tf.cast(condition, tf.int32), axis=0)
    batch_indices = tf.cast(batch_indices, tf.int32)

    # Estrazione batch senza loop
    X_batch = tf.gather(tfdata, batch_indices, axis=0)

    # Compute velocity losses
    vx_loss = tf.reduce_mean(tf.square(y_pred[:, 0] - vx_true))
    vz_loss = tf.reduce_mean(tf.square(y_pred[:, 1] - vz_true))

    MSE = (vx_loss + vz_loss)
    
    dx = 0.001
    dy = 0.001

    # Creiamo i nuovi set di input modificando le colonne specifiche

    X_left   = tf.concat([X_batch[:, :79], X_batch[:, 79:80] - dx, X_batch[:, 80:81]], axis=1)
    X_right  = tf.concat([X_batch[:, :79], X_batch[:, 79:80] + dx, X_batch[:, 80:81]], axis=1)
    X_top    = tf.concat([X_batch[:, :80], X_batch[:, 80:81] + dy], axis=1)
    X_bottom = tf.concat([X_batch[:, :80], X_batch[:, 80:81] - dy], axis=1)

    ul, vl = model(X_left,   training=True)[:, 0], model(X_left,   training=True)[:, 1]  # Previsione per il punto sinistra
    ur, vr = model(X_right,  training=True)[:, 0], model(X_right,  training=True)[:, 1]  # Previsione per il punto destra
    ut, vt = model(X_top,    training=True)[:, 0], model(X_top,    training=True)[:, 1]  # Previsione per il punto sopra
    ub, vb = model(X_bottom, training=True)[:, 0], model(X_bottom, training=True)[:, 1]  # Previsione per il punto sotto


    # Continuita
    points = {
        'center': ( X_batch[:, 79:80],     X_batch[:, 80:81],     y_pred[:, 0], y_pred[:, 1]),
        'left':   ((X_batch[:, 79:80]-dx), X_batch[:, 80:81],     ul, vl),
        'right':  ((X_batch[:, 79:80]+dx), X_batch[:, 80:81],     ur, vr),
        'top':    ( X_batch[:, 79:80],    (X_batch[:, 80:81]+dy), ut, vt),
        'bottom': ( X_batch[:, 79:80],    (X_batch[:, 80:81]-dy), ub, vb)
    }

    continuity_loss = tf.reduce_mean(tf.square(finite_difference_check(points)))

    tf.print('Continuity=', continuity_loss)
    tf.print('MSE=',MSE)
    
    # Total loss
    total_loss = 1 * MSE  + 0.01 * continuity_loss

    return total_loss




def custom_loss(y_true, y_pred):
    # True values
    vx_true, vz_true = y_true[:, 0], y_true[:, 1]

    # Ricerca vettorializzata degli indici
    condition = tf.reduce_all(tf.equal(data_ref[:, None, :], y_true[None, :, :]), axis=2)
    batch_indices = tf.argmax(tf.cast(condition, tf.int32), axis=0)
    batch_indices = tf.cast(batch_indices, tf.int32)

    # Input batch associato
    X_batch = tf.gather(tfdata, batch_indices, axis=0)

    # Compute MSE loss
    vx_loss = tf.reduce_mean(tf.square(y_pred[:, 0] - vx_true))
    vz_loss = tf.reduce_mean(tf.square(y_pred[:, 1] - vz_true))
    MSE = vx_loss + vz_loss

    # Coordinate input
    x = tf.reshape(X_batch[:, 79], (-1, 1))  # x-coordinate
    y = tf.reshape(X_batch[:, 80], (-1, 1))  # y-coordinate

    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, y])

        # Costruisci di nuovo l'input per il modello
        input_vars = tf.concat([X_batch[:, :79], x, y], axis=1)
        output = model(input_vars, training=True)

        u = output[:, 0:1]  # vx
        v = output[:, 1:2]  # vz

    # Derivate automatiche
    du_dx = tape.gradient(u, x)
    dv_dy = tape.gradient(v, y)
    del tape  # libera memoria

    # Calcolo della divergenza
    divergence = du_dx + dv_dy

    # Loss di continuita
    continuity_loss = tf.reduce_mean(tf.square(divergence))

    #tf.print('Continuity=', continuity_loss)

    # Loss totale
    total_loss = 1 * MSE + 0.1 * continuity_loss
    return total_loss


################################################
##################### MAIN #####################
################################################

start_time = time.perf_counter()	# Start Time
file_path = "filtered_dataset.txt"  
data = np.loadtxt(file_path) 

tfdata = tf.convert_to_tensor(data, dtype=tf.float32)
data_ref = tfdata[:, 81:83]  # Estrai i valori dal dataset originale che corrispondono a vx_true

# Split into inputs (X) and outputs (y)
X = data[:, :81]  # Prime 81 colonne come input (39x 39y alpha 1xp 1yp)
y = data[:, 81:83]  # Ultime 2 colonne come output  (vxp, vyp)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Normalizzazione degli input
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

# Salvataggio del modello per la normalizzazione
#scaler_filename = "scaler_model.pkl"
#joblib.dump(scaler, scaler_filename)
#print(f"Scaler saved to {scaler_filename}")


# Costruzione rete neurale
activation='swish'

model = Sequential([
   Dense(256, activation=activation, input_shape=(81,)),
  
   Dense(256, activation=activation), 
   Dense(128, activation=activation), 
   Dense(128, activation=activation), 
   Dense(64,  activation=activation),  

   Dense(2) 
])

# Compilazione modello
initial_lr = 0.001
optimizer = Adam(learning_rate=initial_lr)
model.compile(optimizer=optimizer, loss=custom_loss, metrics=['mae'])

lr_schedule_callback = LearningRateScheduler(lr_scheduler)

# Callback per salvare il miglior modello basandosi sulla validation loss
checkpoint_filepath = "best_model.h5"
checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, monitor='val_loss', mode='min', verbose=1)


batch_size = 32	# Dimensione batch-size

# Allenamento del modello
history = model.fit(
	X_train,
	y_train,
	validation_split=0.2, 
	epochs=10000, 
	batch_size=batch_size,
	verbose=0, 
	callbacks=[checkpoint, lr_schedule_callback])



# Caricare il miglior modello per fare previsione
best_model = tf.keras.models.load_model(checkpoint_filepath,custom_objects={'custom_loss': custom_loss}) # Da usare nel caso in cui sfrutto una custom loss
#best_model = tf.keras.models.load_model(checkpoint_filepath)

sample_input = X_test[0:5]  # Esempio di previsione
predicted_output = best_model.predict(sample_input)
print(f"Predicted Output: {predicted_output[0]}, Actual Output: {y_test[0]}")

################################################
##################### PLOTS ####################
################################################

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

# Salvataggio del plot
plot_filename = f"loss_bs_{batch_size}_nl.png"
plt.savefig(plot_filename, dpi=300)
#plt.show()

print(f"Plot saved as {plot_filename}")

elapsed = time.perf_counter() - start_time
print(f"Elapsed time: {elapsed:.3f} secondi")
