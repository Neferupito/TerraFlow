import numpy as np
from SWE_finite_difference_2D import navier_stokes_2D
import matplotlib.pyplot as plt
from SWE_finite_volume_2D import finite_volume_2D


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


import numpy as np
from scipy.interpolate import interp2d


def increase_sampling_by_factor(grid, factor):
    """
    Increase the resolution of a 2D grid by a scaling factor using interpolation.

    Parameters:
        grid (2D array): Original grid data.
        factor (float): Scaling factor (e.g., 2.0 to double the resolution, 0.5 to halve it).

    Returns:
        2D array: Interpolated grid with increased resolution.
    """
    # Get the original shape of the grid
    original_rows, original_cols = grid.shape

    # Compute the new resolution
    new_rows = int(original_rows * factor)
    new_cols = int(original_cols * factor)

    # Create the original and new coordinate grids
    x_old = np.linspace(0, original_cols - 1, original_cols)
    y_old = np.linspace(0, original_rows - 1, original_rows)
    x_new = np.linspace(0, original_cols - 1, new_cols)
    y_new = np.linspace(0, original_rows - 1, new_rows)

    # Perform the interpolation
    interpolator = interp2d(x_old, y_old, grid, kind="linear")
    new_grid = interpolator(x_new, y_new)

    return new_grid


# Example usage
xyz_file = "../../../BLENDER/gg.xyz"  # Path to your XYZ file
x_grid, y_grid, z_grid, grid_size = read_xyz_and_infer_grid_size(xyz_file)
x_grid = increase_sampling_by_factor(x_grid, 1)
y_grid = increase_sampling_by_factor(y_grid, 1)
z_grid = increase_sampling_by_factor(z_grid, 1)
dt = 0.000001
rain_mm_h = 0
dx = x_grid[0, 1] - x_grid[0, 0]
finite_volume_2D(dt, dx, z_grid, 1000, rain_mm_h)

# You can also access individual coordinates, e.g., coordinates[0] for the first vertex
