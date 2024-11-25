import numpy as np
import constants
import matplotlib.pyplot as plt


def check_vector_sizes(vec1, vec2, vec3):
    """
    Check if three vectors have the same size.

    Parameters:
        vec1, vec2, vec3: Arrays or lists representing the vectors.

    Raises:
        ValueError: If the sizes of the vectors are not the same.
    """
    if len(vec1) != len(vec2) or len(vec2) != len(vec3):
        raise ValueError(
            f"Vectors have different sizes: "
            f"len(vec1)={len(vec1)}, len(vec2)={len(vec2)}, len(vec3)={len(vec3)}"
        )
    print("All vectors have the same size.")


import numpy as np


def get_dx_dy(x_grid, y_grid):
    """
    Calculates the grid spacing (dx, dy) from X and Y grids.

    Args:
        x_grid: 2D numpy array of X coordinates.
        y_grid: 2D numpy array of Y coordinates.

    Returns:
        dx: Grid spacing in the X direction.
        dy: Grid spacing in the Y direction.
    """
    # Calculate the differences between consecutive values in X and Y directions
    dx = np.diff(x_grid, axis=1)  # Difference along columns (X direction)
    dy = np.diff(y_grid, axis=0)  # Difference along rows (Y direction)

    # Assuming the grid is regularly spaced, check if dx and dy are consistent
    if not np.allclose(dx, dx[0, 0]):  # Check if all dx values are the same
        raise ValueError("Inconsistent dx values in X grid.")

    if not np.allclose(dy, dy[0, 0]):  # Check if all dy values are the same
        raise ValueError("Inconsistent dy values in Y grid.")

    # Return the grid spacing (assuming regular grid spacing)
    return dx[0, 0], dy[0, 0]


def compute_slope(z, dx, dy):
    """
    Compute the gradient of the topography z (slope in x and y directions).
    Args:
        z: 2D numpy array of topography.
        dx: Grid spacing in the x-direction.
        dy: Grid spacing in the y-direction.
    Returns:
        grad_z_x: Gradient of z in the x-direction.
        grad_z_y: Gradient of z in the y-direction.
    """
    grad_z_x = (np.roll(z, -1, axis=1) - np.roll(z, 1, axis=1)) / (2 * dx)
    grad_z_y = (np.roll(z, -1, axis=0) - np.roll(z, 1, axis=0)) / (2 * dy)

    return grad_z_x, grad_z_y


def shallow_water_step(h, u, v, z, rain, dx, dy, dt):
    """
    Perform one time step of the shallow water equations with slope effect and rain input.

    Args:
        h: Water height (2D numpy array).
        u: Velocity in the x-direction (2D numpy array).
        v: Velocity in the y-direction (2D numpy array).
        z: Topography (2D numpy array).
        rain: Rainfall rate (2D numpy array) in m/s.
        dx: Grid spacing in the x-direction.
        dy: Grid spacing in the y-direction.
        dt: Time step.

    Returns:
        h_new, u_new, v_new: Updated water height and velocities after one time step.
    """
    # Compute gradients of the topography (slope)
    grad_z_x, grad_z_y = compute_slope(z, dx, dy)

    g = constants.EARTH_GRAVITY
    # Update water height using the continuity equation with rain effect
    h_new = (
        h
        - dt * (np.roll(u * h, -1, axis=1) - np.roll(u * h, 1, axis=1)) / dx
        - dt * (np.roll(v * h, -1, axis=0) - np.roll(v * h, 1, axis=0)) / dy
        + rain * dt
    )  # Adding rain contribution to water height

    # Update velocity in the x-direction
    u_new = (
        u
        - dt * (np.roll(u**2 * h, -1, axis=1) - np.roll(u**2 * h, 1, axis=1)) / dx
        - dt * (np.roll(u * v * h, -1, axis=0) - np.roll(u * v * h, 1, axis=0)) / dy
        - g * dt * grad_z_x
    )  # Slope effect in x-direction

    # Update velocity in the y-direction
    v_new = (
        v
        - dt * (np.roll(u * v * h, -1, axis=1) - np.roll(u * v * h, 1, axis=1)) / dx
        - dt * (np.roll(v**2 * h, -1, axis=0) - np.roll(v**2 * h, 1, axis=0)) / dy
        - g * dt * grad_z_y
    )  # Slope effect in y-direction

    return h_new, u_new, v_new


def navier_stokes_2D(x, y, z, dt, max_iter):

    check_vector_sizes(x, y, z)

    dx, dy = get_dx_dy(x, y)
    print(dx, dy)

    rain_mm_h = 5
    rain_m_s = rain_mm_h * 0.001 / 3600

    grid_size = x.shape
    rain = np.ones(grid_size) * rain_m_s
    h = np.zeros(grid_size)
    u = np.zeros(grid_size)
    v = np.zeros(grid_size)
    print(dx, dy)
    for i in range(max_iter):
        h, u, v = shallow_water_step(h, u, v, z, rain, dx, dy, dt)
    print(np.min(h), np.max(h))
    print(np.min(u), np.max(u))
    print(np.min(v), np.max(v))
    plt.subplot(1, 3, 1)
    plt.pcolormesh(h)
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.pcolormesh(u)
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.pcolormesh(v)
    plt.colorbar()
    plt.show()
