import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from joblib import load
import matplotlib.pyplot as plt
################################################
############## LOSS FUNCTION ###################
################################################

file_path = "filtered_dataset.txt"  
data = np.loadtxt(file_path) 
tfdata = tf.convert_to_tensor(data, dtype=tf.float32)
data_ref = tfdata[:, 81:83]
def custom_loss(y_true, y_pred):

    vx_true, vz_true = y_true[:, 0], y_true[:, 1]

    condition = tf.reduce_all(tf.equal(data_ref[:, None, :], y_true[None, :, :]), axis=2)
    batch_indices = tf.argmax(tf.cast(condition, tf.int32), axis=0)
    batch_indices = tf.cast(batch_indices, tf.int32)

    X_batch = tf.gather(tfdata, batch_indices, axis=0)

    vx_loss = tf.reduce_mean(tf.square(y_pred[:, 0] - vx_true))
    vz_loss = tf.reduce_mean(tf.square(y_pred[:, 1] - vz_true))
    MSE = vx_loss + vz_loss

    x = tf.reshape(X_batch[:, 79], (-1, 1)) 
    y = tf.reshape(X_batch[:, 80], (-1, 1))  

    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, y])

        input_vars = tf.concat([X_batch[:, :79], x, y], axis=1)
        output = model(input_vars, training=True)

        u = output[:, 0:1] 
        v = output[:, 1:2] 

    du_dx = tape.gradient(u, x)
    dv_dy = tape.gradient(v, y)
    del tape 

    divergence = du_dx + dv_dy

    continuity_loss = tf.reduce_mean(tf.square(divergence))

    total_loss = 1 * MSE + 0.1 * continuity_loss
    return total_loss

################################################
################# PREDICTION ###################
################################################

# Percorso per il modello
model_path = "best_model.h5"

# Caricamento rete neurale con funzione loss
model = load_model(model_path, custom_objects={'custom_loss': custom_loss})
#model = load_model(model_path) #caso in cui si è utilizzata una funzione loss default

def predict(file_path, output_file):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Coverti righe in un DataFrame & estrazione input ed output
        data = pd.DataFrame([line.strip().split() for line in lines]).astype(float)
        input_data = data.iloc[:, :81]
        true_values = data.iloc[:, 81:83]
        
        # Se presente una normalizzazione viene effettuata 
        #data_scaled = scaler.transform(input_data)
        data_scaled = input_data

	# Effettua la previsione
        predictions = model.predict(data_scaled)

        # Calcolo degli errori nelle previsioni
        vx_error = predictions[:, 0] - true_values.iloc[:, 0]
        vz_error = predictions[:, 1] - true_values.iloc[:, 1]

        # Creazione dei grafici
        plt.figure(figsize=(12, 6))

        # Grafico errore vx come scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(range(len(vx_error)), vx_error, color='red', label='Errore in previsione vx')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.title('Errore nella previsione di vx')
        plt.xlabel('Campioni')
        plt.ylabel('Errore (vx)')

        # Grafico errore vz come scatter plot
        plt.subplot(1, 2, 2)
        plt.scatter(range(len(vz_error)), vz_error, color='blue', label='Errore in previsione vz')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.title('Errore nella previsione di vz')
        plt.xlabel('Campioni')
        plt.ylabel('Errore (vz)')

        plt.tight_layout()
        plt.savefig('Prediction_Error', dpi=300)
        plt.show()
        
        # Creazione DataFrame per le predizioni e per i valori veri
        predictions_df = pd.DataFrame(predictions, columns=[f"PREDICTION_{i+1}" for i in range(2)])
        true_values.columns = [f"TRUE_{i+1}" for i in range(2)]
        results = pd.concat([true_values, predictions_df], axis=1)
        
        # Salvataggio risultati su output
        results.to_csv(output_file, index=False, header=True, sep=' ')
        print(f"Predictions saved to {output_file}")
    
    except Exception as e:
        print(f"Error: {e}")

################################################
#################### MAIN ######################
################################################

if __name__ == "__main__":
    input_file = "dataTEST.txt"  # File di dati da utilizzare
    output_file = "V_prediction"  # Nome file output
    predict(input_file, output_file)

