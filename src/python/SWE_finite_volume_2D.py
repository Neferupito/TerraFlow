import numpy as np
from constants import EARTH_GRAVITY as g
import matplotlib.pyplot as plt


def build_U(h, u, v):
    return np.array([h, h * u, h * v])


def U2huv(U):
    h = np.maximum(U[0], 1e-8)  # Ensure no zero values in h
    u = U[1] / h
    v = U[2] / h
    return h, u, v


def build_Fx(h, u, v):
    return np.array([h * u, h * u**2 + 1 / 2 * g * h**2, h * u * v])


def build_Fy(h, u, v):
    return np.array([h * v, h * u * v, h * v**2 + 1 / 2 * g * h**2])


def build_S(dx, z, h, rain_mm_h):
    rain_m_s = rain_mm_h * 0.001 / 3600
    rain = np.ones(z[1:-1, 1:-1].shape) * rain_m_s
    slope_x, slope_y = compute_centered_scheme(dx, z)
    hx, hy = compute_centered_scheme(dx, h)
    return np.array([rain, -g * hx * slope_x, -g * hy * slope_y])


def compute_centered_scheme(dx, z):
    slope_x = (z[1:-1, 2:] - z[1:-1, :-2]) / (2 * dx)
    slope_y = (z[2:, 1:-1] - z[:-2, 1:-1]) / (2 * dx)
    return slope_x, slope_y


def compute_forward_scheme(dx, X):
    slope_x = (X[1:, 1:] - X[1:, :-1]) / dx
    slope_y = (X[1:, 1:] - X[:-1, 1:]) / dx
    return slope_x, slope_y


def build_S_interface(dx, z, h):
    slope_x, slope_y = compute_forward_scheme(dx, z)
    hx, hy = compute_forward_scheme(dx, h)
    return np.array([np.zeros(slope_x.shape), -g * hx * slope_x, -g * hy * slope_y])


def get_interface_LR_from_state(X):
    if X.ndim == 2:
        return X[:-1, :-1], X[:-1, 1:], X[:-1, :-1], X[1:, :-1]
    elif X.ndim == 3:
        return X[:, :-1, :-1], X[:, :-1, 1:], X[:, :-1, :-1], X[:, 1:, :-1]
    else:
        raise ValueError("Input array must be 2D or 3D.")


def flux_at_interface(U, dx, z):
    h, u, v = U2huv(U)
    S = build_S_interface(dx, z, h)
    UxL, UxR, UyL, UyR = get_interface_LR_from_state(U)

    # Compute left/right states
    hxL, uxL, vxL = U2huv(UxL)
    hxR, uxR, vxR = U2huv(UxR)
    hyL, uyL, vyL = U2huv(UyL)
    hyR, uyR, vyR = U2huv(UyR)

    # Compute fluxes
    FxL = build_Fx(hxL, uxL, vxL)
    FxR = build_Fx(hxR, uxR, vxR)
    FyL = build_Fy(hyL, uyL, vyL)
    FyR = build_Fy(hyR, uyR, vyR)

    # Topography effects
    zxL, zxR, zyL, zyR = get_interface_LR_from_state(z)
    hxL_tot = np.maximum(0, hxL + zxL)
    hxR_tot = np.maximum(0, hxR + zxR)
    hyL_tot = np.maximum(0, hyL + zyL)
    hyR_tot = np.maximum(0, hyR + zyR)

    # Compute wave speeds
    SxL = np.minimum(uxL - np.sqrt(g * hxL_tot), uxR - np.sqrt(g * hxR_tot))
    SxR = np.maximum(uxL + np.sqrt(g * hxL_tot), uxR + np.sqrt(g * hxR_tot))
    SyL = np.minimum(vyL - np.sqrt(g * hyL_tot), vyR - np.sqrt(g * hyR_tot))
    SyR = np.maximum(vyL + np.sqrt(g * hyL_tot), vyR + np.sqrt(g * hyR_tot))

    # Add small tolerance for division stability
    epsilon = 1e-8

    Fx_interface = (SxR * FxL - SxL * FxR + SxR * SxL * (UxR - UxL)) / (
        SxR - SxL + epsilon
    )
    Fy_interface = (SyR * FyL - SyL * FyR + SyR * SyL * (UyR - UyL)) / (
        SyR - SyL + epsilon
    )

    return Fx_interface, Fy_interface


def finite_volume_2D(dt, dx, z, iter, rain_mm_h):

    h = np.zeros(z.shape)
    u = np.zeros(z.shape)
    v = np.zeros(z.shape)

    U = build_U(h, u, v)
    for i in range(iter):
        Fx_interface, Fy_interface = flux_at_interface(U, dx, z)

        h, u, v = U2huv(U)
        U[:, 1:-1, 1:-1] += (
            -dt / dx * (Fx_interface[:, :-1, 1:] - Fx_interface[:, :-1, :-1])
            - dt / dx * (Fy_interface[:, 1:, :-1] - Fy_interface[:, :-1, :-1])
            + dt * build_S(dx, z, h, rain_mm_h)
        )
        total_mass = np.sum(h)
        print(f"Iteration {i}: Total Mass = {total_mass}")

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
