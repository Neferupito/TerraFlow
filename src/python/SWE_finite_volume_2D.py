import numpy as np
import constants


def build_U(h, u, v):
    return np.array([h, h * u, h * v])


def U2huv(U):
    h = U[0]
    u = U[1] / h
    v = U[2] / h
    return h, u, v


def build_F(h, u, v):
    return np.array(
        [h * u, h * u**2 + 1 / 2 * constants.EARTH_GRAVITY * h**2, h * u * v]
    )


def build_G(h, u, v):
    return np.array(
        [h * v, h * u * v, h * v**2 + 1 / 2 * constants.EARTH_GRAVITY * h**2]
    )


def build_S(dx, z, rain_mm_h):
    rain_m_s = rain_mm_h * 0.001 / 3600
    rain = np.ones(z.shape) * rain_m_s
    slope_x, slope_y = compute_slope(dx, z)
    return np.array([rain, slope_x, slope_y])


def compute_slope(dx, z):
    slope_x = (z[1:-1, 2:] - z[1:-1, :-2]) / (2 * dx)
    slope_y = (z[2:, 1:-1] - z[:-2, 1:-1]) / (2 * dx)
    return slope_x, slope_y


def flux_at_interface(U, dx, z):
    slope_x, slope_y = compute_slope(dx, z)
    UL = F[1:-1, :-1]
    UR = F[1:-1, 1:]
    h, u, v = U2huv(U)
    h_total = np.maximum(0, h + z)
    F = build_F(h, u, v)
    FL = F[1:-1, :-1]
    FR = F[1:-1, 1:]
    G = build_G(h, u, v)
    GL = G[:-1, 1:-1]
    GR = G[1:, 1:-1]
    F_interface = np.zeros(z[1:-1, 1:-1].shape)
    G_interface = np.zeros(z[1:-1, 1:-1].shape)
    lambda_min_u = np.minimum(
        u[1:-1, :-1] - np.sqrt(constants.EARTH_GRAVITY * h_total[1:-1, :-1]),
        u[1:-1, 1:] - np.sqrt(constants.EARTH_GRAVITY * h_total[1:-1, 1:]),
    )
    lambda_min_v = np.minimum(
        v[:-1, 1:-1] - np.sqrt(constants.EARTH_GRAVITY * h_total[:-1, 1:-1]),
        v[1:, 1:-1] - np.sqrt(constants.EARTH_GRAVITY * h_total[1:, 1:-1]),
    )
    lambda_max_u = np.maximum(
        u[1:-1, :-1] + np.sqrt(constants.EARTH_GRAVITY * h_total[1:-1, :-1]),
        u[1:-1, 1:] + np.sqrt(constants.EARTH_GRAVITY * h_total[1:-1, 1:]),
    )
    lambda_max_v = np.maximum(
        v[:-1, 1:-1] + np.sqrt(constants.EARTH_GRAVITY * h_total[:-1, 1:-1]),
        v[1:, 1:-1] + np.sqrt(constants.EARTH_GRAVITY * h_total[1:, 1:-1]),
    )

    lambda_min_u_mask = lambda_min_u > 0
    lambda_min_v_mask = lambda_min_v > 0
    lambda_max_u_mask = lambda_max_u < 0
    lambda_max_v_mask = lambda_max_v < 0

    not_in_both_mask_u = ~(lambda_min_u_mask | lambda_max_u_mask)
    not_in_both_mask_v = ~(lambda_min_v_mask | lambda_max_v_mask)

    F_interface[lambda_max_u_mask] = FL[lambda_max_u_mask]
    F_interface[lambda_min_u_mask] = FR[lambda_min_u_mask]
    F_interface[not_in_both_mask_u] = (
        lambda_max_u[not_in_both_mask_u] * FL[not_in_both_mask_u]
        - lambda_min_u[not_in_both_mask_u] * FR[not_in_both_mask_u]
        + lambda_max_u[not_in_both_mask_u]
        * lambda_min_u[not_in_both_mask_u]
        * (UR[not_in_both_mask_u] - UL[not_in_both_mask_u])
    )

    G_interface[lambda_max_v_mask] = GL[lambda_max_v_mask]
    G_interface[lambda_min_v_mask] = GR[lambda_min_v_mask]
    G_interface[not_in_both_mask_v] = (
        lambda_max_u[not_in_both_mask_v] * GL[not_in_both_mask_v]
        - lambda_min_u[not_in_both_mask_v] * GR[not_in_both_mask_v]
        + lambda_max_u[not_in_both_mask_v]
        * lambda_min_u[not_in_both_mask_v]
        * (UR[not_in_both_mask_v] - UL[not_in_both_mask_v])
    )

    F_interface -= constants.EARTH_GRAVITY * slope_x
    G_interface -= constants.EARTH_GRAVITY * slope_y
    return (F_interface, G_interface)


def finite_volume_2D(dt, dx, z, iter, rain_mm_h):

    h = np.zeros(z.shape)
    u = np.zeros(z.shape)
    v = np.zeros(z.shape)

    U = build_U(h, u, v)
    for i in range(iter):
        F, G = flux_at_interface(U, dx, z)
        U += (
            -dt / dx * (F[:, 1:] - F[:, :-1])
            - dt / dx * (G[1:, :] - G[:-1, :])
            + dt * build_S(dx, z, rain_mm_h)
        )
