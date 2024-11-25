import numpy as np
from SWE_finite_difference_2D import navier_stokes_2D
import matplotlib.pyplot as plt


def read_xyz_and_infer_grid_size(filepath):
    """
    Reads an XYZ file, infers the grid size, and reshapes the coordinates into X, Y, and Z grids.

    Args:
        filepath: Path to the XYZ file.

    Returns:
        x_grid: 2D numpy array of X coordinates.
        y_grid: 2D numpy array of Y coordinates.
        z_grid: 2D numpy array of Z (elevation) coordinates.
        grid_size: Tuple (rows, cols) representing the inferred grid size.
    """
    # Read the XYZ data
    coordinates = []

    with open(filepath, "r") as file:
        for line in file:
            if line.strip() and not line.startswith("#"):  # Skip empty or comment lines
                try:
                    x, y, z = map(float, line.split())
                    coordinates.append((x, y, z))
                except ValueError:
                    print(f"Warning: Could not parse line: {line.strip()}")

    # Convert list of coordinates to numpy array
    coordinates = np.array(coordinates)

    # Extract unique X and Y values to infer the grid size
    unique_x = np.unique(coordinates[:, 0])
    unique_y = np.unique(coordinates[:, 1])

    # Infer the grid size from the number of unique X and Y values
    n_cols = len(unique_x)  # Number of unique X values
    n_rows = len(unique_y)  # Number of unique Y values

    print(f"Inferred grid size: {n_rows} rows and {n_cols} columns")

    # Reshape the XYZ coordinates into grids based on inferred grid size
    x_grid = (
        coordinates[:, 0].reshape((n_rows, n_cols)).T
    )  # Reshape X coordinates into 2D grid
    y_grid = (
        coordinates[:, 1].reshape((n_rows, n_cols)).T
    )  # Reshape Y coordinates into 2D grid
    z_grid = (
        coordinates[:, 2].reshape((n_rows, n_cols)).T
    )  # Reshape Z coordinates (elevation) into 2D grid

    return x_grid, y_grid, z_grid, (n_rows, n_cols)


# Example usage
xyz_file = "../../../BLENDER/gg.xyz"  # Path to your XYZ file
x_grid, y_grid, z_grid, grid_size = read_xyz_and_infer_grid_size(xyz_file)

# Print the inferred grid size and first few rows of the grids
print(f"Inferred Grid Size: {grid_size}")
print("\nX Grid (First 5 rows):")
print(x_grid[:5, :])

print("\nY Grid (First 5 rows):")
print(y_grid[:5, :])

print("\nZ Grid (First 5 rows):")
print(z_grid[:5, :])

dt = 0.0001
navier_stokes_2D(x_grid, y_grid, z_grid, dt, 1000)

# You can also access individual coordinates, e.g., coordinates[0] for the first vertex
