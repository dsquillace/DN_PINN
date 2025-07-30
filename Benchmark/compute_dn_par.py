import os, subprocess, shutil, random, concurrent.futures
import numpy as np
from tempfile import TemporaryDirectory
from concurrent.futures import ProcessPoolExecutor


################################################
################# CONSTANTS ####################
################################################

bp = '/home/damiano/NeuralNetwork/resources/airf'   #Airfoil folder path
fn = 'NACA0012_120.dat' #Selected airfoil
fp = os.path.join(bp, fn)
sp = '/home/damiano/NeuralNetwork/resources/bin/run_wg2aer.sh' #wg2aer path
num_profiles = 10000  #Total number of samples
n_par = 20  #Number of wg2aer parameters (1 to 32)
wg2aer_range = 2.0  #Variability range of the parameters
alpha_range = 10    #Variability range of AoA (-x, x)
output_file = 'test_data.txt'   #Output file name

up_cp, low_cp = 1.2, -7     #CP limits constrain
LE_radius = 0.0015      #LE radius constrain
min_thickness  = 0.0025     #Constrain on the minimum thickness

################################################
################# FUNCTIONS ####################
################################################

def dn(airfoil, alpha):
    alphar = np.deg2rad(alpha)
    n = airfoil.shape[0]

    x0, z0 = airfoil[:-1, 0], airfoil[:-1, 1]
    x1, z1 = airfoil[1:, 0], airfoil[1:, 1]

    xc = (x0 + x1) / 2
    zc = (z0 + z1) / 2

    dx = x1 - x0
    dz = z1 - z0
    elle = np.sqrt(dx**2 + dz**2)
    sinb = dz / elle
    cosb = dx / elle

    dx0 = xc[:, None] - x0[None, :]
    dz0 = zc[:, None] - z0[None, :]
    dx1 = xc[:, None] - x1[None, :]
    dz1 = zc[:, None] - z1[None, :]
    ra = dx0**2 + dz0**2
    rb = dx1**2 + dz1**2
    u = np.log(ra / rb) / (4 * np.pi)

    cosbibj = np.outer(cosb, cosb) + np.outer(sinb, sinb)
    sinbibj = np.outer(sinb, cosb) - np.outer(cosb, sinb)

    x0_m, x1_m = x0[None, :], x1[None, :]
    z0_m, z1_m = z0[None, :], z1[None, :]
    xc_m, zc_m = xc[:, None], zc[:, None]
    num = (zc_m - z1_m) * (xc_m - x0_m) - (zc_m - z0_m) * (xc_m - x1_m)
    den = (xc_m - x1_m) * (xc_m - x0_m) + (zc_m - z0_m) * (zc_m - z1_m)
    w = np.arctan2(num, den) / (2 * np.pi)
    np.fill_diagonal(w, 0.5)

    vt = cosbibj * u + sinbibj * w
    vn = -sinbibj * u + cosbibj * w

    rhs = np.cos(alphar) * sinb - np.sin(alphar) * cosb
    vtv = np.sum(vn, axis=1)
    vnv = -np.sum(vt, axis=1)

    vn = np.hstack((vn, vnv[:, None]))
    vt = np.hstack((vt, vtv[:, None]))
    vn = np.vstack((vn, [vt[0, :] + vt[-1, :]]))
    vn[-1, -1] = np.sum(vn[0, :] + vn[-2, :])

    rhs = np.append(rhs, -(
        np.cos(alphar) * (cosb[-1] + cosb[0]) +
        np.sin(alphar) * (sinb[-1] + sinb[0])
    ))

    siga = np.linalg.solve(vn, rhs)
    veta = vt @ siga + np.cos(alphar) * cosb + np.sin(alphar) * sinb

    vx = veta * cosb
    vz = veta * sinb

    phi = np.cumsum(veta * elle)
    cp = 1 - veta**2
    cl = 2 * np.dot(veta, elle)

    #print('Angolo di incidenza =', f"{alpha:.3f}")
    #print('Coefficiente di portanza =', f"{cl:.3f}")

    return cl, siga, veta, phi, cp, xc, zc, elle, vx, vz

def check_leading_edge_radius(tmpdir, min_radius):
    eval_out_path = os.path.join(tmpdir, 'eval_obj.out')
    if os.path.exists(eval_out_path):
        with open(eval_out_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "Airfoil_leading_edge_radius" in line:
                    value_str = line.strip().split('=')[1]
                    radius = float(value_str)
                    if radius < min_radius:
                        raise ValueError(f'***ERROR - SHARP LEADING EDGE ***')
    return None

def check_intersecting_geometry(yo, n, min_thick):
    for i in range(1, n // 2 + 1):
        lower = yo[i - 1]
        upper = yo[n - i]
        if lower > upper or upper - lower < min_thick:
            raise ValueError(f'***ERROR - INTERSECTING GEOMETRY at index {i} | thickness = {upper - lower:.6f} ***')
    return None

def check_pressure_coefficient(cp, min_cp, max_cp):
    for i, value in enumerate(cp):
        if value > max_cp or value < min_cp:
            print(f"cp[{i}] = {value} is out of bounds")
            raise ValueError(f'***ERROR - cp[{i}] = {value} is out of bounds ***')
    return None

def generate_profile(index):
    with TemporaryDirectory() as tmpdir:
        try:
            shutil.copy(fp, os.path.join(tmpdir, fn))

            param = [random.uniform(-wg2aer_range, wg2aer_range) for _ in range(n_par)]
            lines = ['BASELINE_AIRFOIL='+ fn + '\n']
            lines += [f'X({i}) = {p:.6f}\n' for i, p in enumerate(param)]

            with open(os.path.join(tmpdir, 'eval_obj.in'), 'w') as f:
                f.writelines(lines)

            subprocess.run(f'{sp} -save', shell=True, cwd=tmpdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            eval_out_path = os.path.join(tmpdir, 'eval_obj.out')
            check_leading_edge_radius(tmpdir, LE_radius)

            airfoil_file = os.path.join(tmpdir, 'modified_airfoil.dat')
            with open(airfoil_file, 'r') as infile:
                xo, yo = np.loadtxt(infile, dtype=float, unpack=True, skiprows=1)

            n = len(xo)
            check_intersecting_geometry(yo, n, min_thickness)

            with open(airfoil_file, 'r') as file:
                file.readline()
                airfoil = np.loadtxt(file)

            alpha = random.uniform(-alpha_range, alpha_range)
            cl, siga, veta, phi, cp, xc, zc, elle, vx, vz = dn(airfoil, alpha)

            check_pressure_coefficient(cp, low_cp, up_cp)

            data_y = ' '.join(f'{y_val:.6f}' for y_val in yo)
            cp_output = ' '.join(f'{cp_val:.6f}' for cp_val in cp)
            return f"{data_y} {alpha:.6f} {cp_output}\n"

        except Exception as e:
            print(f"[Worker {index}] Error: {e}")
            return None

def is_broken_sample(line, n_par):
    parts = line.strip().split()
    return all(p == '1' for p in parts[:n_par])

################################################
##################### MAIN #####################
################################################


if __name__ == "__main__":
    dataset_file = output_file
    results = []

    print(f"Generating {num_profiles} valid profiles (excluding BROKEN or failed ones)...")

    index = 0  # Counter for the job ID
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {}

        while len(results) < num_profiles:
            # Launch more tasks to reach the needed total
            while len(futures) < (num_profiles - len(results)):
                futures[executor.submit(generate_profile, index)] = index
                index += 1

            # As tasks complete, process them
            done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

            for future in done:
                line = future.result()
                if line is not None and not is_broken_sample(line, n_par):
                    results.append(line)

                # Remove completed future
                del futures[future]

                if len(results) % 100 == 0 or len(results) == num_profiles:
                    print("####################################################")
                    print(f"########### {len(results)} valid profiles collected ##########")
                    print("####################################################")

    # Write final dataset
    with open(dataset_file, 'w') as f:
        f.writelines(results)

    print(f"Dataset with {num_profiles} valid profiles saved to '{dataset_file}'.")
