import numpy as np
import pandas as pd
import subprocess,shutil
from tensorflow.keras.models import load_model

fn = 'NACA0012_120.dat'
sp = '/home/damiano/NeuralNetwork/resources/bin/run_wg2aer.sh -save'
obj_data = np.loadtxt("objective.txt", comments="#")
cp_obj = obj_data[:, 2]

n_par = 20
input_file = "params.txt"

model = load_model("best_model.keras", compile=False)

################################################
############### NN-PREDICTION ##################
################################################

def predict(input_vector, output_file=None):
	try:
		input_df = pd.DataFrame([input_vector]).astype(float)
		prediction = model.predict(input_df)
		return prediction
	except Exception as e:
		print(f"Error: {e}")

################################################
#################### MAIN ######################
################################################

if __name__ == "__main__":

    with open(input_file, "r") as f:
        numbers = [float(x) for x in f.read().split()]

    param = numbers[:n_par]
    alpha = numbers[-1]

    lines = []
    lines.append('BASELINE_AIRFOIL=' + fn + '\n')
    for i, value in enumerate(param):
        lines.append(f'X({i}) = {value:.6f}\n')

    with open('eval_obj.in', 'w') as f:
        f.writelines(lines)

    subprocess.run(sp, shell=True, cwd='./')

    with open('modified_airfoil.dat', 'r') as infile:
        xo, yo = np.loadtxt(infile, dtype=float, unpack=True, skiprows=1)

    nn_input = yo
    nn_input = np.append(nn_input, alpha)

    ### Previsione
    cp_pred = predict(nn_input)
    cp_pred = np.array(cp_pred).flatten()

    rmse = np.sqrt(np.mean((cp_pred - cp_obj) ** 2))

    print(f"RMSE with CP_obj: {rmse:.6f}")
    with open("cp_distribution.txt", "w") as f:
        header_line = "# x    z     Cp"
        f.write(f"# RMSE = {rmse:.6f}\n")
        f.write(header_line + "\n")
        for xi, zi, cpi in zip(xo, yo, cp_pred):
            f.write(f"{xi:.6f}  {zi:.6f}  {cpi:.6f}\n")
