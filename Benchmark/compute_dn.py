import numpy as np
import matplotlib.pyplot as plt
import subprocess, shutil, random
import math, os
import matplotlib.path as mplPath
################################################
################# CONSTANTS ####################
################################################

bp = '/home/damiano/NeuralNetwork/resources/airf'  # Base path for airfoil files
fn = 'NACA0012_120.dat'  # Filename
fp = os.path.join(bp, fn)
sp = '/home/damiano/NeuralNetwork/resources/bin/run_wg2aer.sh' #wg2aer path

num_profiles = 1000                    # Number of airfoils to generate
n_points_dataset = 500             # Number of domain points to sample
dataset_file = 'dataset.txt'       # Output file for dataset
filtered_file = 'filtered_dataset.txt'  # Optional cleaned/ordered version
wg2aer_range = 2.                 # Range for WG2AER parameters (-x, x)
n_par = 20                         # Number of WG2AER parameters (1 to 32)

# Grid parameters
n_points = 1000                    # Total number of domain points
xpar = 0.7                         # X-axis extension beyond trailing and leading edge
zpar = 0.7                         # Y-axis extension from center (0, 0)

x_domain_low, x_domain_up = -xpar, 1 + xpar
z_domain_low, z_domain_up = -zpar, zpar
x = np.linspace(x_domain_low, x_domain_up, n_points)
z = np.linspace(z_domain_low, z_domain_up, n_points)
x_grid, z_grid = np.meshgrid(x, z)  # 2D structured grid
rows, cols = x_grid.shape
grid_points = np.column_stack((x_grid.flatten(), z_grid.flatten()))

################################################
################# FUNCTIONS ####################
################################################

def dn(airfoil, alpha):
    """Panel method solution for lift and flow properties around the airfoil"""
    alphar = (np.pi / 180) * alpha
    n = airfoil.shape[0]

    # Panel geometry
    x0 = airfoil[:-1, 0]
    z0 = airfoil[:-1, 1]
    x1 = airfoil[1:, 0]
    z1 = airfoil[1:, 1]

    xc = (x0 + x1) / 2
    zc = (z0 + z1) / 2

    elle = np.sqrt((x1 - x0)**2 + (z1 - z0)**2)
    sinb = (z1 - z0) / elle
    cosb = (x1 - x0) / elle

    ra = np.array([(xc[i] - x0)**2 + (zc[i] - z0)**2 for i in range(n - 1)])
    rb = np.array([(xc[i] - x1)**2 + (zc[i] - z1)**2 for i in range(n - 1)])

    u = np.log(ra / rb) / (4 * np.pi)

    cosbibj = np.outer(cosb, cosb) + np.outer(sinb, sinb)
    sinbibj = np.outer(sinb, cosb) - np.outer(cosb, sinb)

    w = np.zeros((n - 1, n - 1))
    for i in range(n - 1):
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

    #print('Angle of attack =', f"{alpha:.3f}")
    #print('Lift coefficient =', f"{cl:.3f}")

    return cl, siga, veta, phi, cp, xc, zc, elle, vx, vz


def velocity_on_grid(x_grid, z_grid, airfoil, siga, alpha):
    """Compute velocity field over the domain using panel influence"""
    alphar = (np.pi / 180) * alpha
    n = airfoil.shape[0]
    x0, z0 = airfoil[:-1, 0], airfoil[:-1, 1]
    x1, z1 = airfoil[1:, 0], airfoil[1:, 1]

    u_x_grid = np.cos(alphar) * np.ones_like(x_grid)
    u_z_grid = np.sin(alphar) * np.ones_like(z_grid)

    for j in range(len(x0)):
        dx, dz = x1[j] - x0[j], z1[j] - z0[j]
        elle = np.sqrt(dx**2 + dz**2)
        sinb, cosb = dz / elle, dx / elle

        ra = (x_grid - x0[j])**2 + (z_grid - z0[j])**2
        rb = (x_grid - x1[j])**2 + (z_grid - z1[j])**2
        u = np.log(ra / rb) / (4 * np.pi)
        w = np.arctan2(
            (z_grid - z1[j]) * (x_grid - x0[j]) - (z_grid - z0[j]) * (x_grid - x1[j]),
            (x_grid - x1[j]) * (x_grid - x0[j]) + (z_grid - z0[j]) * (z_grid - z1[j])
        ) / (2 * np.pi)

        u_x_grid += siga[j] * (cosb * u - sinb * w)
        u_z_grid += siga[j] * (sinb * u + cosb * w)

    return u_x_grid, u_z_grid


def verify_continuity(x_grid, z_grid, u_x_grid, u_z_grid):
    """Compute continuity error from ∂u/∂x + ∂v/∂z"""
    dx = np.gradient(x_grid, axis=1)
    dz = np.gradient(z_grid, axis=0)

    du_dx = np.gradient(u_x_grid, axis=1) / dx
    dv_dz = np.gradient(u_z_grid, axis=0) / dz

    continuity_error = du_dx + dv_dz
    max_error = np.max(np.abs(continuity_error))
    avg_error = np.mean(np.abs(continuity_error))

    return continuity_error, max_error, avg_error


def get_neighbors(index, shape):
    """Get 5-point cross neighbors (center + up/down/left/right) for finite difference stencil"""
    i, j = np.unravel_index(index, shape)
    neighbors = [
        (i - 1, j),  # top
        (i, j - 1),  # left
        (i, j),      # center
        (i, j + 1),  # right
        (i + 1, j)   # bottom
    ]
    return [(r, c) for r, c in neighbors if 0 <= r < shape[0] and 0 <= c < shape[1]]

################################################
##################### MAIN #####################
################################################

if __name__ == "__main__":
    count = 0
    with open(dataset_file, 'w') as output_file:
        while count < num_profiles:

            # Clean working directory (just in case)
            subprocess.run('rm modified_airfoil.dat eval_obj.in', shell=True, cwd='./')

            # Prepare working directory
            shutil.copy(fp, './')  # Copy baseline geometry

            fileinput = 'eval_obj.in'
            lines = ['BASELINE_AIRFOIL='+fn+'\n']  # Replace with correct data filename
            param = []

            for i in range(n_par):
                value = random.uniform(-wg2aer_range, wg2aer_range)  # Generate parameters for WG2AER
                lines.append(f'X({i}) = {value:.6f}\n')
                param.append(value)

            with open(fileinput, 'w') as f:
                f.writelines(lines)

            subprocess.run(f'{sp} -save', shell=True, cwd='./')

            angle_found = False
            # === Check for leading edge radius ===
            eval_out_path = 'eval_obj.out'
            if os.path.exists(eval_out_path):
                with open(eval_out_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if "Airfoil_leading_edge_radius" in line:
                            value_str = line.strip().split('=')[1]
                            radius = float(value_str)
                            if (radius < 0.0025):
                                print('***ERROR - SHARP LEADING EDGE***')
                                output_file.write(' '.join('1' for _ in range(n_par)) + ' 1\n')
                                angle_found = True
                                break
            if angle_found:
                continue


            with open('modified_airfoil.dat', 'r') as infile:
                xo, yo = np.loadtxt(infile, dtype=float, unpack=True, skiprows=1)
                n = len(xo)  # Number of points in the airfoil

            intersection_found = False

            # Check for overlapping geometry
            for i in range(1, (n // 2 + 1)):
                if yo[i - 1] > yo[n - i]:
                    print('***ERROR - INTERSECTING GEOMETRY***')
                    output_file.write(' '.join('1' for count in range(20)) + ' 1\n')
                    intersection_found = True
                    break

            if intersection_found:
                continue

            print(f"Geometry created successfully!")

            with open('modified_airfoil.dat', 'r') as file:
                header = file.readline().strip()
                airfoil = np.loadtxt(file)

            # Solve airfoil flow
            alpha = random.uniform(-10, 10)
            cl, siga, veta, phi, cp, xc, zc, elle, vx, vz = dn(airfoil, alpha)

            # Optional: Compute velocity field, pressure coefficient, or continuity error
            # Uncomment following lines to enable them:

            # print(f"Computing velocity field...")
            # ux_grid, uz_grid = velocity_on_grid(x_grid, z_grid, airfoil, siga, alpha)
            # continuity_error, max_error, avg_error = verify_continuity(x_grid, z_grid, ux_grid, uz_grid)
            # cp_grid = 1 - (ux_grid**2 + uz_grid**2)

            # Optional: Domain point selection excluding inside airfoil and borders
            # airfoil_path = mplPath.Path(airfoil)
            # inside_mask = airfoil_path.contains_points(grid_points)
            # airfoil_mask = inside_mask.reshape(rows, cols)
            # valid_indices = [
            #     i * cols + j
            #     for i in range(1, rows - 1)
            #     for j in range(1, cols - 1)
            #     if not airfoil_mask.flatten()[i * cols + j]
            # ]
            # random_indices = random.sample(valid_indices, n_points_dataset)

            #######################################
            ############### PLOTS #################
            #######################################

            ###### Errore continuità
            #contour = plt.contourf(x_grid, z_grid, continuity_error, levels=100, cmap="coolwarm")
            #plt.colorbar(label="Errore di Continuità (du/dx + dv/dz)")
            #contour_lines = plt.contour(x_grid, z_grid, continuity_error, levels=100, colors='black', linewidths=0.5)
            #plt.clabel(contour_lines, inline=True, fontsize=10, fmt="%.2e")  # Etichette dei livelli
            #plt.xlabel("x")
            #plt.ylabel("z")
            #plt.title("Errore dell'Equazione di Continuità")
            #plt.show()

            ###### Streamlines
#            plt.figure(figsize=(10, 5))
#            plt.streamplot(x, z, ux_grid, uz_grid, density=2, linewidth=1, arrowsize=1, arrowstyle='->')
#            plt.fill(airfoil[:, 0], airfoil[:, 1], 'k')
#            plt.plot(airfoil[:, 0], airfoil[:, 1], 'k-', linestyle='solid', lw=2, zorder=2)  # Plot airfoil
#            plt.xlabel('x')
#            plt.ylabel('z')
#            plt.title('Velocity Field with Freestream')
#            plt.axis('equal')
#            plt.show()

            ###### Plot Cp field
#            plt.figure(figsize=(10, 5))
#            contour = plt.contourf(x_grid, z_grid, cp_grid, levels=100, cmap='plasma')
#            plt.colorbar(contour, label='Pressure Coefficient $C_p$')
#            plt.fill(airfoil[:, 0], airfoil[:, 1], 'k')  # Profilo nero
#            plt.plot(airfoil[:, 0], airfoil[:, 1], 'k-', linestyle='solid', lw=2, zorder=2)
#            plt.xlabel('x')
#            plt.ylabel('z')
#            plt.title('Pressure Coefficient $C_p$ Field')
#            plt.axis('equal')
#            plt.show()

             #print(f"Lift coefficient: {cl}")
             #print(f"Number of columns in data_line: {len(yo)}")
             #print(f"Number of columns in alpha: 1")
             #print(f"Number of columns in velocities: {len(vx) * 2}")
             #print(f"Number of columns in cl: 1")

             #print('n_points', n_points)
             #print('Max_error', max_error)
             #print('Avg_error', avg_error)

            ##### Plotting airfoil with cp
            #plt.plot(xc, zc, label="Airfoil")
            #plt.plot(xc, -cp, label="-Cp")
            #plt.legend()
            #plt.savefig('prova.png')


            #######################################
            ########### DATASET OUTPUT ############
            #######################################

            # Dataset with WG2AER parameters, 1 alpha, and cp values
            wg2aer = ' '.join(f'{wg2aer_val:.6f}' for wg2aer_val in param)
            cp_output = ' '.join(f'{cp_val:.6f}' for cp_val in cp)
            output_file.write(wg2aer + f' {alpha:.6f} ' + cp_output + '\n')

            # Optional dataset formats:
            # Dataset with full airfoil, alpha, singularities, cl
            # data_x = ' '.join(f'{x_val:.6f}' for x_val in xo)
            # data_y = ' '.join(f'{y_val:.6f}' for y_val in yo)
            # siga_str = ' '.join(f'{siga_val:.6f}' for siga_val in siga)
            # output_file.write(data_x + '   ' + data_y + f' {alpha:.6f} ' + siga_str + f' {cl:.6f}\n')

            # Dataset for a single point: airfoil + alpha + point coords + velocity
            # for i in random_indices:
            #     x_pt = x_grid.flatten()[i]
            #     z_pt = z_grid.flatten()[i]
            #     ux_pt = ux_grid.flatten()[i]
            #     uz_pt = uz_grid.flatten()[i]
            #     output_file.write(
            #         f"{' '.join(f'{val:.6f}' for val in airfoil[:, 0])}   "
            #         f"{' '.join(f'{val:.6f}' for val in airfoil[:, 1])}   "
            #         f"{alpha:.6f}   {x_pt:.6f}   {z_pt:.6f}   {ux_pt:.6f}   {uz_pt:.6f}\n"
            #     )

            # Dataset for 5-point cross (centered finite difference)
            # for idx in random_indices:
            #     neighbors = get_neighbors(idx, (rows, cols))
            #     x_pts = [x_grid[r, c] for r, c in neighbors]
            #     z_pts = [z_grid[r, c] for r, c in neighbors]
            #     ux_pts = [ux_grid[r, c] for r, c in neighbors]
            #     uz_pts = [uz_grid[r, c] for r, c in neighbors]
            #     output_file.write(
            #         f"{' '.join(f'{val:.6f}' for val in airfoil[:, 0])}   "
            #         f"{' '.join(f'{val:.6f}' for val in airfoil[:, 1])}   "
            #         f"{alpha:.6f}   "
            #         f"{' '.join(f'{val:.6f}' for val in x_pts)}   "
            #         f"{' '.join(f'{val:.6f}' for val in z_pts)}   "
            #         f"{' '.join(f'{val:.6f}' for val in ux_pts)}   "
            #         f"{' '.join(f'{val:.6f}' for val in uz_pts)}\n"
            #     )

            count += 1

    print(f"CONGRATS!! Dataset created with success and saved as '{dataset_file}'.")

    with open('dataset.txt', "r") as input_file, open(filtered_file, "w") as output_file:
        valid_lines = []
        for line in input_file:
            values = line.strip().split()
            if not all(value == '0.0' for value in values) and not all(value == '1' for value in values):
                valid_lines.append(values)

        max_lengths = [max(len(str(value)) for value in column) for column in zip(*valid_lines)]

        # Formatted output
        for values in valid_lines:
            formatted_line = "   ".join(f"{value:>{max_lengths[i]}}" for i, value in enumerate(values))
            output_file.write(formatted_line + "\n")

    print(f"Dataset ordered and filtered saved as '{filtered_file}'.")  
