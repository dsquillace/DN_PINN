import numpy as np
import matplotlib.pyplot as plt
import subprocess, os
import math

################################################
################# CONSTANTS ####################
################################################

fn = 'NACA0012_120.dat'                            # Baseline airfoil filename
sp = '/home/damiano/NeuralNetwork/resources/bin/run_wg2aer.sh' # wg2aer path

fileinput = 'eval_obj.in'       # File for WG2AER input
n_par = 20                      # Number of WG2AER parameters (first 20 numbers in file)
input_file = 'params.txt'       # File containing 21 numbers (20 params + 1 alpha)

obj_data = np.loadtxt("objective.txt", comments="#")
cp_obj = obj_data[:, 2]
################################################
################# FUNCTIONS ####################
################################################

def dn(airfoil, alpha):
    """Panel method solution for lift and Cp distribution"""
    alphar = np.deg2rad(alpha)  # convert to radians
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

    phi = np.cumsum(veta * elle)
    cp = 1 - veta**2
    cl = 2 * np.dot(veta, elle)

    return cl, cp, xc, zc

################################################
##################### MAIN #####################
################################################

if __name__ == "__main__":

    with open(input_file, 'r') as f:
        numbers = [float(x) for x in f.read().split()]

    param = numbers[:n_par]   # shape parameters
    alpha = numbers[-1]       # angle of attack

    lines = [f'BASELINE_AIRFOIL={fn}\n']
    for i, value in enumerate(param):
        lines.append(f'X({i}) = {value:.6f}\n')

    with open(fileinput, 'w') as f:
        f.writelines(lines)

    subprocess.run(f'{sp} -save', shell=True, cwd='./')

    with open('modified_airfoil.dat', 'r') as infile:
        xo, yo = np.loadtxt(infile, dtype=float, unpack=True, skiprows=1)
    with open('modified_airfoil.dat', 'r') as file:
        file.readline()
        airfoil = np.loadtxt(file)

    cl, cp, xc, zc = dn(airfoil, alpha)

    rmse = np.sqrt(np.mean((cp - cp_obj) ** 2))
    print(f"RMSE with CP_obj: {rmse:.6f}")
    with open("cp_distribution.txt", "w") as f:
        header_line = "# x    z    Cp"
        f.write(f"# RMSE = {rmse:.6f}\n")
        f.write(header_line + "\n")
        for xi, zi, cpi in zip(xc, zc, cp):
            f.write(f"{xi:.6f}  {zi:.6f}  {cpi:.6f}\n")
