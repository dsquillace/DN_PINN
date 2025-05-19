import numpy as np
import matplotlib.pyplot as plt
import subprocess,shutil,random
import math
import matplotlib.path as mplPath

################################################
################# FUNCTIONS ####################
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

    #print('Numero punti profilo =', n)
    print('Angolo di incidenza =', f"{alpha:.3f}")
    print('Coefficiente di portanza =', f"{cl:.3f}")

    return cl, siga, veta, phi, cp, xc, zc, elle, vx, vz



def velocity_on_grid(x_grid, z_grid, airfoil, siga, alpha):
    alphar = (np.pi / 180) * alpha
    n = airfoil.shape[0]

    x0 = airfoil[:-1, 0]
    z0 = airfoil[:-1, 1]
    x1 = airfoil[1:, 0]
    z1 = airfoil[1:, 1]

    # Inizializza le velocità su tutta la griglia
    u_x_grid = np.cos(alphar)*np.ones_like(x_grid)
    u_z_grid = np.sin(alphar)*np.ones_like(z_grid)

    n_panels = len(x0)

    for j in range(n_panels):
        # Lunghezza e inclinazione del pannello
        dx = x1[j] - x0[j]
        dz = z1[j] - z0[j]
        elle = np.sqrt(dx**2 + dz**2)
        xc = (x0 + x1) / 2
        zc = (z0 + z1) / 2

        sinb = dz / elle
        cosb = dx / elle

        # Distanze dai punti della griglia agli estremi del pannello
        ra = (x_grid - x0[j])**2 + (z_grid - z0[j])**2
        rb = (x_grid - x1[j])**2 + (z_grid - z1[j])**2


        # Contributo del pannello j
        u = (np.log(ra / rb) / (4 * np.pi)) 

        w = np.arctan2(
            (z_grid - z1[j]) * (x_grid - x0[j]) - (z_grid - z0[j]) * (x_grid - x1[j]),
            (x_grid - x1[j]) * (x_grid - x0[j]) + (z_grid - z0[j]) * (z_grid - z1[j])
            ) / (2 * np.pi)
 
        u_x_grid += siga[j] * (cosb * u -  sinb * w)
        u_z_grid += siga[j] * (sinb * u + cosb * w)

    return u_x_grid, u_z_grid


def verify_continuity(x_grid, z_grid, u_x_grid, u_z_grid):

    # Griglia (valida in caso di griglia strutturata)
    dx = np.gradient(x_grid, axis=1)
    dz = np.gradient(z_grid, axis=0) 

    # Derivate parziali di u e v
    du_dx = np.gradient(u_x_grid, axis=1) / dx
    dv_dz = np.gradient(u_z_grid, axis=0) / dz

    # Calcolo continuità
    continuity_error = du_dx + dv_dz

    # Metriche di valutazione
    max_error = np.max(np.abs(continuity_error))
    avg_error = np.mean(np.abs(continuity_error))

    return continuity_error, max_error, avg_error


def get_neighbors(index, shape):
    i, j = np.unravel_index(index, shape)  # Ottieni coordinate 2D
    neighbors = [
        (i - 1, j),   # sopra
        (i, j - 1),   # sinistra
        (i, j),       # centro
        (i, j + 1),   # destra
        (i + 1, j)    # sotto
    ]
    return [(r, c) for r, c in neighbors if 0 <= r < shape[0] and 0 <= c < shape[1]]

################################################
##################### MAIN #####################
################################################

num_profiles = 1			# Numero di profili da generare
n_points_dataset = 500			# Numero punti da prendere a random nel dominio
dataset_file = 'dataset.txt'  		# Nome del file di output per il dataset
filtered_file = 'filtered_dataset.txt' 	# Dataset ordinato e pulito
wg2aer_range = 0.0
n_par = 10

# Parametri della griglia

n_points = 1000	# Numero totale di punti nel dominio
xpar = 0.7	# Distanza X da TE e LE del dominio
zpar = 0.7	# Distanza Y dall'orgine del dominio

x_domain_low, x_domain_up = -xpar, 1+xpar	# Dominio asse X
z_domain_low, z_domain_up = -zpar, zpar		# Dominio asse y
x = np.linspace(x_domain_low, x_domain_up, n_points)	# Creazione vettore punti X equidistanti  
z = np.linspace(z_domain_low, z_domain_up, n_points)  	# Creazione vettore punti Y equidistanti 
x_grid, z_grid = np.meshgrid(x, z)	# Creazione griglia strutturata 2D
rows, cols = x_grid.shape			# Dimensioni della griglia
grid_points = np.column_stack((x_grid.flatten(), z_grid.flatten()))  # Converte in array Nx2


count = 0
with open(dataset_file, 'w') as output_file:
	while count < num_profiles:

		#Pulizia directory per sicurezza
		subprocess.run('rm modified_airfoil.dat eval_obj.in', shell=True, cwd='./')

		#Preparazione directory
		shutil.copy('/home/squillace/DEEPONET/HOMEMADE/resources/naca0012_39correct.dat', './')	# Copiare la geometria di base voluta

		fileinput = 'eval_obj.in'
		lines = []
		lines.append('BASELINE_AIRFOIL=naca0012_39correct.dat\n')	# Sostituire con il nome del file dati corretto
		param = []

		for i in range(n_par):
		    value = random.uniform(-wg2aer_range, wg2aer_range) 	# Per generare geometrie modificate sfruttando WG2AER
		    lines.append(f'X({i}) = {value:.6f}\n')
		    param.append(value)

		with open(fileinput, 'w') as f:
		    f.writelines(lines)

		subprocess.run('/home/squillace/DEEPONET/HOMEMADE/bin/run_wg2aer.sh -save', shell=True, cwd='./')

		with open('modified_airfoil.dat', 'r') as infile:
		    xo, yo = np.loadtxt(infile, dtype=float, unpack=True, skiprows=1)
		    n = (len(xo)) 	# Numero di punti del profilo

		intersection_found = False

		# Controllo intreccio profilo

		for i in range(1, (n//2+1)):
			if yo[i - 1] > yo[n - i]:
				print('***ERRORE-GEOMETRIA INTRECCIATA***')
				# Scrivi una riga di 1 nel file di output e interrompi il ciclo di controllo
				output_file.write(' '.join('1' for count in range(20)) + ' 1\n')
				intersection_found = True
				break

		if intersection_found:
			continue

		print(f"Geometria creata con successo!")

		with open('modified_airfoil.dat', 'r') as file:
		    header = file.readline().strip()
		    airfoil = np.loadtxt(file)

		# Soluzione del problema sul corpo

		#alpha = 5
		alpha = random.uniform(-10,10)
		cl, siga, veta, phi, cp, xc, zc, elle, vx, vz = dn(airfoil, alpha)

		# Calcolo delle velocità nel dominio

#		print(f"Calcolo campo di velocità...")
#		ux_grid, uz_grid = velocity_on_grid(x_grid, z_grid, airfoil, siga, alpha)
#		continuity_error, max_error, avg_error = verify_continuity(x_grid, z_grid, ux_grid, uz_grid)
#		cp_grid = 1 - (ux_grid**2 + uz_grid**2)

		# Definizione del dominio dove prendere i punti (ESCLUSIONE DENTRO IL CORPO E BORDI)

#		airfoil_path = mplPath.Path(airfoil)  # Definisce il contorno del profilo
#		inside_mask = airfoil_path.contains_points(grid_points)  # Array 1D di booleani
#		airfoil_mask = inside_mask.reshape(rows, cols)
#		valid_indices = [
#				i * cols + j
#				for i in range(1, rows - 1)   # Evita il bordo superiore e inferiore
#				for j in range(1, cols - 1)   # Evita il bordo sinistro e destro
#				if not airfoil_mask.flatten()[i * cols + j]  # Escludi i punti interni al profilo
#
#				]
#		random_indices = random.sample(valid_indices, n_points_dataset)
		################################################
		##################### PLOTS ####################
		################################################

		# Airfoil-CP
#		plt.plot(xo, yo, label="Airfoil")
#		plt.plot(xc, -cp, label="-Cp")
#		plt.legend()
#		plt.show()

		# Errore continuità
#		contour = plt.contourf(x_grid, z_grid, continuity_error, levels=100, cmap="coolwarm")#
#		plt.colorbar(label="Errore di Continuità (du/dx + dv/dz)")
#		contour_lines = plt.contour(x_grid, z_grid, continuity_error, levels=100, colors='black', linewidths=0.5)
#		plt.clabel(contour_lines, inline=True, fontsize=10, fmt="%.2e")  # Etichette dei livelli
#		plt.xlabel("x")
#		plt.ylabel("z")
#		plt.title("Errore dell'Equazione di Continuità")
#		plt.show()


		# Streamlines
#		plt.figure(figsize=(10, 5))
#		plt.streamplot(x, z, ux_grid, uz_grid, density=2, linewidth=1, arrowsize=1, arrowstyle='->')
#		plt.fill(airfoil[:, 0], airfoil[:, 1], 'k')
#		plt.plot(airfoil[:, 0], airfoil[:, 1], 'k-', linestyle='solid', lw=2, zorder=2)  # Plot airfoil
#		plt.xlabel('x')
#		plt.ylabel('z')
#		plt.title('Velocity Field with Freestream')
#		plt.axis('equal')
#		plt.show()


		# Plot del campo di pressione Cp
#		plt.figure(figsize=(10, 5))
#		contour = plt.contourf(x_grid, z_grid, cp_grid, levels=100, cmap='plasma')
#		plt.colorbar(contour, label='Pressure Coefficient $C_p$')
#		plt.fill(airfoil[:, 0], airfoil[:, 1], 'k')  # Profilo nero
#		plt.plot(airfoil[:, 0], airfoil[:, 1], 'k-', linestyle='solid', lw=2, zorder=2)
#		plt.xlabel('x')
#		plt.ylabel('z')
#		plt.title('Pressure Coefficient $C_p$ Field')
#		plt.axis('equal')
#		plt.show()

		################################################
		#################### DATASET ###################
		################################################

		####### Data set benchmark [10_param_wg2aer, 1 alpha, cp]

		wg2aer = ' '.join(f'{wg2aer_val:.6f}' for wg2aer_val in param)
		cp_output = ' '.join(f'{cp_val:.6f}' for cp_val in cp)
		output_file.write(wg2aer + f' {alpha:.6f} ' + cp_output + '\n')


		####### Data set semplice [x_airf,y_airf, alpha, siga, cl]

#		data_x = ' '.join(f'{x_val:.6f}' for x_val in xo)
#		data_y = ' '.join(f'{y_val:.6f}' for y_val in yo)
#		siga = ' '.join(f'{siga_val:.6f}' for siga_val in siga)
#		output_file.write(data_x + '   ' + data_y + f' {alpha:.6f} ' + siga + f' {cl:.6f}\n')


		####### Data set continuity 1pt [x_airf,y_airf, alpha, xp, yp, up, vp]

#		print('Estrazione punti nel dominio...')
#		for i in random_indices:
#			x_pt = x_grid.flatten()[i]
#			z_pt = z_grid.flatten()[i]
#			ux_pt = ux_grid.flatten()[i]
#			uz_pt = uz_grid.flatten()[i]
#
#			output_file.write(
#			f"{' '.join(f'{val:.6f}' for val in airfoil[:, 0])}   "  # X airfoil
#			f"{' '.join(f'{val:.6f}' for val in airfoil[:, 1])}   "  # Z airfoil
#			f"{alpha:.6f}   {x_pt:.6f}   {z_pt:.6f}   {ux_pt:.6f}   {uz_pt:.6f}\n"
#			)

		####### Data set continuity 5pt a croce (differenze finite centrate) [x_airf,y_airf, alpha, 5xp, 5yp, 5up, 5vp]

#		rows, cols = x_grid.shape
#		for idx in random_indices:
#		    neighbors = get_neighbors(idx, (rows, cols))
#		    
#		    x_pts = [x_grid[r, c] for r, c in neighbors]
#		    z_pts = [z_grid[r, c] for r, c in neighbors]
#		    ux_pts = [ux_grid[r, c] for r, c in neighbors]
#		    uz_pts = [uz_grid[r, c] for r, c in neighbors]
#		    
#		    output_file.write(
#			f"{' '.join(f'{val:.6f}' for val in airfoil[:, 0])}   "  # X airfoil
#			f"{' '.join(f'{val:.6f}' for val in airfoil[:, 1])}   "  # Z airfoil
#			f"{alpha:.6f}   "
#			f"{' '.join(f'{val:.6f}' for val in x_pts)}   "
#			f"{' '.join(f'{val:.6f}' for val in z_pts)}   "
#			f"{' '.join(f'{val:.6f}' for val in ux_pts)}   "
#			f"{' '.join(f'{val:.6f}' for val in uz_pts)}\n"
#		    )

		count = count + 1	
		print('######## Profili completati:', (count), '/', num_profiles)

print(f"COMPLIMENTI! Dataset creato con successo e salvato in '{dataset_file}'")


with open('dataset.txt', "r") as input_file, open(filtered_file, "w") as output_file:
    # Lista per memorizzare tutte le righe valide (per l'allineamento delle colonne)
    valid_lines = []
    
    for line in input_file:
        # Rimuove gli spazi e ottiene una lista di valori
        values = line.strip().split()
        
        # Controlla se la riga contiene solo 0 o solo 1 (ERRORI DA RIMUOVERE)
        if not all(value == '0.0' for value in values) and not all(value == '1' for value in values):
            valid_lines.append(values)
    
    # Trova il numero massimo di caratteri in ciascuna colonna per il formato
    max_lengths = [max(len(str(value)) for value in column) for column in zip(*valid_lines)]
    
    # Scrive le righe formattate nel file di output
    for values in valid_lines:
        # Format della riga, allineando le colonne in base alla lunghezza massima
        formatted_line = "   ".join(f"{value:>{max_lengths[i]}}" for i, value in enumerate(values))
        output_file.write(formatted_line + "\n")

print(f"Dataset ordinato e salvato in '{filtered_file}'")


