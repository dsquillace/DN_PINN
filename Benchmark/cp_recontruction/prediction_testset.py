import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error

################################################
############### NN-PREDICTION ##################
################################################

def predict(file_path, output_file):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        data = pd.DataFrame([line.strip().split() for line in lines]).astype(float)
        input_data = data.iloc[:, :11]
        true_values = data.iloc[:, 11:]

        predictions = model.predict(input_data)

        general_mse = mean_squared_error(true_values.values, predictions)
        print(f"General MSE: {general_mse:.6f}")

        # Preparazione output
        predictions_df = pd.DataFrame(predictions, columns=[f"PREDICTION_{i+1}" for i in range(predictions.shape[1])])
        true_values.columns = [f"TRUE_{i+1}" for i in range(true_values.shape[1])]
        results = pd.concat([true_values, predictions_df], axis=1)
        results.to_csv(output_file, index=False, header=True, sep=' ')
        print(f"Predictions saved to {output_file}")

    except Exception as e:
        print(f"Error: {e}")

################################################
#################### MAIN ######################
################################################

if __name__ == "__main__":
    input_file = "testset.txt"
    output_file = "prediction"
    model = load_model("best_model.h5", compile=False)
    predict(input_file, output_file)

