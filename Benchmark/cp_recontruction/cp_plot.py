import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import subprocess,shutil,random
from tensorflow.keras.models import load_model

################################################
############## Douglass-Neumann ################
################################################

def dn(airfoil, alpha):
    alphar = (np.pi / 180) * alpha

    n = airfoil.shape[0]

    x0 = airfoil[:-1, 0]
    z0 = airfoil[:-1, 1]
    x1 = airfoil[1:, 0]
    z1 = airfoil[1:, 1]

    xc = (x0 + x1) / 2
    zc = (z0 + z1) / 2

    elle = np.sqrt((x1 - x0)**2 + (z1 - z0)**2)
    sinb = (z1 - z0) / elle
    cosb = (x1 - x0) / elle

    ra = np.array([(xc[i] - x0)**2 + (zc[i] - z0)**2 for i in range(n-1)])
    rb = np.array([(xc[i] - x1)**2 + (zc[i] - z1)**2 for i in range(n-1)])

    u = np.log(ra / rb) / (4 * np.pi)

    cosbibj = np.outer(cosb, cosb) + np.outer(sinb, sinb)
    sinbibj = np.outer(sinb, cosb) - np.outer(cosb, sinb)

    w = np.zeros((n-1, n-1))
    for i in range(n-1):
        w[i, :] = np.arctan2(
            (zc[i] - z1) * (xc[i] - x0) - (zc[i] - z0) * (xc[i] - x1),
            (xc[i] - x1) * (xc[i] - x0) + (zc[i] - z0) * (zc[i] - z1)
        ) / (2 * np.pi)
        w[i, i] = 0.5

    vt = cosbibj * u + sinbibj * w
    vn = -sinbibj * u + cosbibj * w

    rhs = np.cos(alphar) * sinb - cosb * np.sin(alphar)

    vtv = np.sum(vn, axis=1)
    vnv = -np.sum(vt, axis=1)

    vn = np.hstack((vn, vnv.reshape(-1, 1)))
    vt = np.hstack((vt, vtv.reshape(-1, 1)))

    vn = np.vstack((vn, vt[0, :] + vt[-1, :]))
    vn[-1, -1] = np.sum(vn[0, :] + vn[-2, :])

    rhs = np.append(rhs, -(
        np.cos(alphar) * cosb[-1] + np.sin(alphar) * sinb[-1] +
        np.cos(alphar) * cosb[0] + np.sin(alphar) * sinb[0]
    ))

    siga = np.linalg.solve(vn, rhs)

    veta = vt @ siga + np.cos(alphar) * cosb + np.sin(alphar) * sinb

    vx = veta * cosb
    vz = veta * sinb 

    phi = np.cumsum(veta * elle)

    cp = 1 - veta**2

    cl = 2 * np.dot(veta, elle)

    # Cl calcolato con cp
    n_pan = len(cp) // 2

    cp_lower = cp[:n_pan]
    cp_upper = cp[n_pan:]

    x_lower = xc[:n_pan]
    x_upper = xc[n_pan:]

    # Differenze in x per integrazione
    dx = x_lower[1:] - x_lower[:-1]

    # Media dei cp su ciascun intervallo
    cp_l_avg = 0.5 * (cp_lower[1:] + cp_lower[:-1])
    cp_u_avg = 0.5 * (cp_upper[1:] + cp_upper[:-1])

    # Calcolo Cl da Cp
    cl_cp_int = -np.sum((cp_l_avg - cp_u_avg) * dx)

    print('Angolo di incidenza =', f"{alpha:.3f}")
    print('Coefficiente di portanza tramite veta =', f"{cl:.3f}")
    print('Coefficiente di portanza tramite CP =', f"{cl_cp_int:.3f}")

    return cl, siga, veta, phi, cp, xc, zc, elle, vx, vz, cl_cp_int


################################################
############### NN-PREDICTION ##################
################################################

# Percorso per il modello
model_path = "best_model.h5"

# Caricamento NN
model = load_model(model_path, compile=False)

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

wg2aer_range = 1.0
alpha = random.uniform(-10,10)

#### Soluzione reale problema

#Preparazione directory
shutil.copy('/home/squillace/DEEPONET/HOMEMADE/resources/NACA0012_40.dat', './')	# Copiare la geometria di base voluta

#Creazione input wg2aer
fileinput = 'eval_obj.in'
lines = []
lines.append('BASELINE_AIRFOIL=NACA0012_40.dat\n')	# Sostituire con il nome del file dati corretto
param = []

for i in range(10):
    value = random.uniform(-wg2aer_range, wg2aer_range) 	# Per generare geometrie modificate sfruttando WG2AER
    lines.append(f'X({i}) = {value:.6f}\n')
    param.append(value)

param.append(alpha)

with open(fileinput, 'w') as f:
    f.writelines(lines)

subprocess.run('/home/squillace/DEEPONET/HOMEMADE/bin/run_wg2aer.sh -save', shell=True, cwd='./')

with open('modified_airfoil.dat', 'r') as infile:
    xo, yo = np.loadtxt(infile, dtype=float, unpack=True, skiprows=1)
    n = (len(xo)) 	# Numero di punti del profilo

# Controllo intreccio profilo
intersection_found = False
for i in range(1, (n//2+1)):
	if yo[i - 1] > yo[n - i]:
		print('***ERRORE-GEOMETRIA INTRECCIATA***')
		# Scrivi una riga di 1 nel file di output e interrompi il ciclo di controllo
		output_file.write(' '.join('1' for count in range(20)) + ' 1\n')
		intersection_found = True
		break


with open('modified_airfoil.dat', 'r') as file:
    header = file.readline().strip()
    airfoil = np.loadtxt(file)

print(f"Geometria creata con successo!")

# Soluazione problema
cl, siga, veta, phi, cp, xc, zc, elle, vx, vz, cl_cp_int = dn(airfoil, alpha)


### Previsione
cp_pred = predict(param)
cp_pred = cp_pred.reshape(-1)

# Calcolo cl con cp come in D-N
n_pan = len(cp) // 2  # 20
cp_lower = cp_pred[:n_pan]
cp_upper = cp_pred[n_pan:]
x_lower = xc[:n_pan]
x_upper = xc[n_pan:]
dx = x_lower[1:] - x_lower[:-1]
cp_l_avg = 0.5 * (cp_lower[1:] + cp_lower[:-1])
cp_u_avg = 0.5 * (cp_upper[1:] + cp_upper[:-1])

cl_pred = -np.sum((cp_l_avg - cp_u_avg) * dx)

################################################
#################### PLOT ######################
################################################

#Rendere il profilo alare più visibile
cp_range = (-cp).max() - (-cp).min()
scale_factor = cp_range / 4.0 / (yo.max() - yo.min())
yo_scaled = yo * scale_factor


# Original Airfoil-CP
#plt.plot(xo, yo, label="Airfoil")
#plt.plot(xc, -cp, label="-Cp")
#plt.legend()
#plt.show()

# Predicted Airfoil-CP
#plt.plot(xo, yo, label="Airfoil")
#plt.plot(xc, -cp_pred, label="-Cp")
#plt.legend()
#plt.show()


# Confronto Cp_pred, Cp e CL

fig, ax = plt.subplots()

ax.plot(xo[:21], yo_scaled[:21], 'b', linewidth = 0.8)
ax.plot(xo[20:], yo_scaled[20:], 'r', linewidth = 0.8)

line1, = ax.plot(xc[:21], -cp[:21], 'b', label=r"Real -$C_p$ (Upper)", linewidth = 2)
line2, = ax.plot(xc[20:], -cp[20:], 'r', label=r"Real -$C_p$ (Lower)", linewidth = 2)
line3, = ax.plot(xc, -cp_pred, 'k', linestyle=(0,(4,4)), label=r"Predicted $C_p$", linewidth = 2)

legend1 = ax.legend(handles=[line1, line2, line3], loc='upper right')
ax.add_artist(legend1)

ax.set_title(r'$\alpha$ = %1.3f' % alpha)

cl_text = [
    Line2D([0], [0], color='none', label=f"Real $C_L$ = {cl_cp_int:.3f}"),
    Line2D([0], [0], color='none', label=f"Predicted $C_L$ = {cl_pred:.3f}")
]

legend2 = ax.legend(handles=cl_text, loc='lower right')

ax.grid()
plt.show()

