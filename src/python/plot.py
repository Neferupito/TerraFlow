import numpy as np
import matplotlib.pyplot as plt

fname = "/mnt/c/Users/fdall/OneDrive/docs/BLENDER/gg.xyz"


# Function to read the file and parse data
def read_fsw2d_data(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Skip commented and empty lines
    data = [
        list(map(float, line.split()))
        for line in lines
        if not line.startswith("#") and line.strip()
    ]

    # Convert to NumPy array
    data = np.array(data)
    return data


data = read_fsw2d_data(fname)

# Extract columns (adjust column indices as per your structure)
gg = (data[:, 1] - min(data[:, 1])) * 1000
dx = gg[1] - gg[0]
x = (data[:, 0] - min(data[:, 0])) * 1000 + dx / 2  # x-coordinates
y = (data[:, 1] - min(data[:, 1])) * 1000 + dx / 2  # y-coordinates
z = data[:, 2] * 200  # z-coordinates

file_name = "topography.txt"
with open(file_name, "w") as file:
    # Write header
    file.write("# x y z\n")
    # Write data
    for xi, yi, zi in zip(x, y, z):
        file.write(f"{xi:.6f} {yi:.6f} {zi:.6f}\n")
print(y[1] - y[0], max(y) - min(y))


# File path
file_path = "/mnt/c/Users/fdall/Downloads/FullSWOF_2D/FullSWOF_2D/Examples/Simple/Outputs/huz_final.dat"

# Read the data
data = read_fsw2d_data(file_path)

# Extract columns (adjust column indices as per your structure)
x = data[:, 0]  # x-coordinates
y = data[:, 1]  # y-coordinates
h = data[:, 2]  # h values
u = data[:, 3]  # u values
v = data[:, 4]  # v values

# Compute sqrt(u^2 + v^2)
velocity_norm = np.sqrt(u**2 + v**2)

# Reshape data for colormesh plotting
# Assuming x and y are evenly spaced in a grid
x_unique = np.unique(x)
y_unique = np.unique(y)
x_grid, y_grid = np.meshgrid(x_unique, y_unique)
h_grid = h.reshape(len(y_unique), len(x_unique), order="F")
velocity_norm_grid = velocity_norm.reshape(len(y_unique), len(x_unique), order="F")
er_lim = (h_grid > 0.5) & (velocity_norm_grid > 2)
dep_lim = (h_grid > 0.5) & (velocity_norm_grid < 1)
limit = 1
mask = (limit - velocity_norm_grid) * 0.01 * h_grid

# Plot h as a colormesh
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.pcolormesh(x_grid, y_grid, h_grid, shading="auto", cmap="viridis")
plt.colorbar(label="h")
plt.title("h values")
plt.xlabel("x")
plt.ylabel("y")

# Plot velocity norm as a colormesh
plt.subplot(1, 3, 2)
plt.pcolormesh(x_grid, y_grid, velocity_norm_grid, shading="auto", cmap="plasma")
plt.colorbar(label="sqrt(u^2 + v^2)")
plt.title("Velocity Norm")
plt.xlabel("x")
plt.ylabel("y")

# Plot velocity norm as a colormesh
plt.subplot(1, 3, 3)
plt.pcolormesh(x_grid, y_grid, mask, shading="auto", cmap="plasma")
plt.colorbar(label="dep erosion")
plt.title("Velocity Norm")
plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
plt.show()
