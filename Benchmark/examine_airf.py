import os

bp = '/home/damiano/NeuralNetwork/resources/airf'  # Airfoil folder path
fn = 'NACA0012_120.dat'  # Selected airfoil file
airfoil_path = os.path.join(bp, fn)

with open(airfoil_path, 'r') as f:
    lines = f.readlines()

# Extract header (first line)
header = lines[0].strip()

# Extract first column (X coordinates)
x_coords = [float(line.split()[0]) for line in lines[1:] if line.strip()]

test_file = 'test_data.txt'
with open(test_file, 'r') as f:
    test_lines = f.readlines()

# Input: row number (from user or hardcoded here)
row_index = int(input(f"Enter row index (0 to {len(test_lines)-1}): "))

# Safety check
if row_index < 0 or row_index >= len(test_lines):
    raise IndexError("Invalid row index")

# Extract the first 121 values from the selected row
selected_row = test_lines[row_index].strip().split()
y_values = [float(val) for val in selected_row[:121]]

output_file = 'requested_airf.dat'
with open(output_file, 'w') as f:
    f.write(f"{header}\n")
    for x, y in zip(x_coords, y_values):
        f.write(f"{x:.6f} {y:.6f}\n")

print(f"File '{output_file}' written successfully.")
