"""Optional accelerated numerical kernels.

The public functions in this module must remain usable without Numba.  When
``numba`` is installed, selected pure-array kernels are JIT-compiled; otherwise
the NumPy fallback is used.  This keeps the research scripts and desktop app
portable while still allowing faster runs on machines with Numba available.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

try:  # pragma: no cover - availability depends on the local environment.
    from numba import njit as _numba_njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised when numba is absent.
    _numba_njit = None
    NUMBA_AVAILABLE = False


def _optional_njit(**kwargs):
    def decorator(func):
        if _numba_njit is None:
            return func
        return _numba_njit(**kwargs)(func)

    return decorator


@_optional_njit(cache=True, fastmath=True)
def _visibility_mask_ecef_numba(
    r_ecef: np.ndarray,
    r_moon_ecef: np.ndarray,
    station_ecef: np.ndarray,
    station_lat_rad: np.ndarray,
    station_lon_rad: np.ndarray,
    min_elev_rad: float,
    r_moon_m: float,
) -> np.ndarray:
    n_steps = r_ecef.shape[0]
    n_stations = station_ecef.shape[0]
    out = np.zeros((n_steps, n_stations), dtype=np.bool_)
    r2_safe = r_moon_m * r_moon_m * 0.999999999

    for si in range(n_stations):
        phi = station_lat_rad[si]
        lam = station_lon_rad[si]
        zx = np.cos(phi) * np.cos(lam)
        zy = np.cos(phi) * np.sin(lam)
        zz = np.sin(phi)

        sx = station_ecef[si, 0]
        sy = station_ecef[si, 1]
        sz = station_ecef[si, 2]

        for ti in range(n_steps):
            rx = r_ecef[ti, 0] - sx
            ry = r_ecef[ti, 1] - sy
            rz = r_ecef[ti, 2] - sz
            rng = np.sqrt(rx * rx + ry * ry + rz * rz)
            if rng <= 0.0:
                continue

            sin_el = (rx * zx + ry * zy + rz * zz) / rng
            if sin_el > 1.0:
                sin_el = 1.0
            elif sin_el < -1.0:
                sin_el = -1.0

            if np.arcsin(sin_el) <= min_elev_rad:
                continue

            # Segment occultation from station to spacecraft, body center at
            # r_moon_ecef[ti].  Equivalent to visibility.is_occulted_by_moon.
            ax = sx - r_moon_ecef[ti, 0]
            ay = sy - r_moon_ecef[ti, 1]
            az = sz - r_moon_ecef[ti, 2]

            bx = r_ecef[ti, 0] - r_moon_ecef[ti, 0]
            by = r_ecef[ti, 1] - r_moon_ecef[ti, 1]
            bz = r_ecef[ti, 2] - r_moon_ecef[ti, 2]

            mag2_a = ax * ax + ay * ay + az * az
            mag2_b = bx * bx + by * by + bz * bz
            dot_ab = ax * bx + ay * by + az * bz
            len2_segment = mag2_a + mag2_b - 2.0 * dot_ab
            if len2_segment <= 1e-15:
                out[ti, si] = True
                continue

            tau = (mag2_a - dot_ab) / len2_segment
            if tau < 0.0 or tau > 1.0:
                out[ti, si] = True
                continue

            cx = ax + (bx - ax) * tau
            cy = ay + (by - ay) * tau
            cz = az + (bz - az) * tau
            dist2 = cx * cx + cy * cy + cz * cz
            out[ti, si] = dist2 >= r2_safe

    return out


def _visibility_mask_ecef_numpy(
    r_ecef: np.ndarray,
    r_moon_ecef: np.ndarray,
    station_ecef: np.ndarray,
    station_lat_rad: np.ndarray,
    station_lon_rad: np.ndarray,
    min_elev_rad: float,
    r_moon_m: float,
) -> np.ndarray:
    n_steps = r_ecef.shape[0]
    out = np.zeros((n_steps, station_ecef.shape[0]), dtype=bool)
    r2_safe = float(r_moon_m) ** 2 * 0.999999999

    for si in range(station_ecef.shape[0]):
        rho_vec = r_ecef - station_ecef[si, :]
        range_m = np.linalg.norm(rho_vec, axis=1)
        phi = float(station_lat_rad[si])
        lam = float(station_lon_rad[si])
        zenith_unit = np.array(
            [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)],
            dtype=float,
        )
        sin_el = np.divide(
            rho_vec @ zenith_unit,
            range_m,
            out=np.zeros(n_steps, dtype=float),
            where=range_m > 0.0,
        )
        in_view = np.arcsin(np.clip(sin_el, -1.0, 1.0)) > min_elev_rad
        if not np.any(in_view):
            continue

        idx = np.where(in_view)[0]
        vec_a = station_ecef[si, :].reshape(1, 3) - r_moon_ecef[idx, :]
        vec_b = r_ecef[idx, :] - r_moon_ecef[idx, :]
        mag2_a = np.sum(vec_a * vec_a, axis=1)
        mag2_b = np.sum(vec_b * vec_b, axis=1)
        dot_ab = np.sum(vec_a * vec_b, axis=1)
        len2_segment = mag2_a + mag2_b - 2.0 * dot_ab
        tau = np.divide(
            mag2_a - dot_ab,
            len2_segment,
            out=np.zeros_like(len2_segment),
            where=len2_segment > 1e-15,
        )
        valid_tau = (tau >= 0.0) & (tau <= 1.0)
        vec_c = vec_a + (vec_b - vec_a) * tau[:, None]
        dist2 = np.sum(vec_c * vec_c, axis=1)
        occulted = (len2_segment > 1e-15) & valid_tau & (dist2 < r2_safe)
        out[idx[~occulted], si] = True

    return out


def visibility_mask_ecef(
    r_ecef: ArrayLike,
    r_moon_ecef: ArrayLike,
    station_ecef: ArrayLike,
    station_lat_rad: ArrayLike,
    station_lon_rad: ArrayLike,
    min_elev_rad: float,
    r_moon_m: float,
) -> np.ndarray:
    """Return station visibility mask for ECEF spacecraft/Moon positions.

    Parameters use plain arrays only, making this suitable for Numba JIT.  The
    output has shape ``(num_epochs, num_stations)``.
    """
    r_ecef_arr = np.asarray(r_ecef, dtype=float)
    r_moon_arr = np.asarray(r_moon_ecef, dtype=float)
    station_arr = np.asarray(station_ecef, dtype=float)
    lat_arr = np.asarray(station_lat_rad, dtype=float).reshape(-1)
    lon_arr = np.asarray(station_lon_rad, dtype=float).reshape(-1)

    if r_ecef_arr.ndim != 2 or r_ecef_arr.shape[1] != 3:
        raise ValueError("r_ecef must have shape (N, 3).")
    if r_moon_arr.shape != r_ecef_arr.shape:
        raise ValueError("r_moon_ecef must have shape (N, 3).")
    if station_arr.ndim != 2 or station_arr.shape[1] != 3:
        raise ValueError("station_ecef must have shape (M, 3).")
    if lat_arr.size != station_arr.shape[0] or lon_arr.size != station_arr.shape[0]:
        raise ValueError("station latitude/longitude arrays must match station_ecef rows.")

    if NUMBA_AVAILABLE:
        return _visibility_mask_ecef_numba(
            r_ecef_arr,
            r_moon_arr,
            station_arr,
            lat_arr,
            lon_arr,
            float(min_elev_rad),
            float(r_moon_m),
        )
    return _visibility_mask_ecef_numpy(
        r_ecef_arr,
        r_moon_arr,
        station_arr,
        lat_arr,
        lon_arr,
        float(min_elev_rad),
        float(r_moon_m),
    )


ACCELERATION_BACKEND = "numba" if NUMBA_AVAILABLE else "numpy"


@_optional_njit(cache=True)
def _position_observables_numba(
    state_history_mci: np.ndarray,
    obs_data: np.ndarray,
    earth_pos_mci: np.ndarray,
    xforms: np.ndarray,
    station_ecef: np.ndarray,
    station_lat_rad: np.ndarray,
    station_lon_rad: np.ndarray,
) -> np.ndarray:
    n_obs = obs_data.shape[0]
    out = np.zeros((n_obs, 3), dtype=np.float64)
    two_pi = 2.0 * np.pi

    for oi in range(n_obs):
        station_id = int(obs_data[oi, 4]) - 1
        time_idx = int(obs_data[oi, 5]) - 1

        rx_eci = state_history_mci[time_idx, 0] - earth_pos_mci[time_idx, 0]
        ry_eci = state_history_mci[time_idx, 1] - earth_pos_mci[time_idx, 1]
        rz_eci = state_history_mci[time_idx, 2] - earth_pos_mci[time_idx, 2]

        x00 = xforms[time_idx, 0, 0]
        x01 = xforms[time_idx, 0, 1]
        x02 = xforms[time_idx, 0, 2]
        x10 = xforms[time_idx, 1, 0]
        x11 = xforms[time_idx, 1, 1]
        x12 = xforms[time_idx, 1, 2]
        x20 = xforms[time_idx, 2, 0]
        x21 = xforms[time_idx, 2, 1]
        x22 = xforms[time_idx, 2, 2]

        rho_x = x00 * rx_eci + x01 * ry_eci + x02 * rz_eci - station_ecef[station_id, 0]
        rho_y = x10 * rx_eci + x11 * ry_eci + x12 * rz_eci - station_ecef[station_id, 1]
        rho_z = x20 * rx_eci + x21 * ry_eci + x22 * rz_eci - station_ecef[station_id, 2]

        az, el, rng = _razel_scalar(rho_x, rho_y, rho_z, station_lat_rad[station_id], station_lon_rad[station_id], two_pi)
        out[oi, 0] = rng
        out[oi, 1] = az
        out[oi, 2] = el

    return out


@_optional_njit(cache=True)
def _range_rate_observables_numba(
    state_history_mci: np.ndarray,
    obs_data: np.ndarray,
    earth_pos_mci: np.ndarray,
    earth_vel_mci: np.ndarray,
    xforms: np.ndarray,
    station_ecef: np.ndarray,
    station_lat_rad: np.ndarray,
    station_lon_rad: np.ndarray,
) -> np.ndarray:
    n_obs = obs_data.shape[0]
    out = np.zeros((n_obs, 4), dtype=np.float64)
    two_pi = 2.0 * np.pi

    for oi in range(n_obs):
        station_id = int(obs_data[oi, 5]) - 1
        time_idx = int(obs_data[oi, 6]) - 1

        rx_eci = state_history_mci[time_idx, 0] - earth_pos_mci[time_idx, 0]
        ry_eci = state_history_mci[time_idx, 1] - earth_pos_mci[time_idx, 1]
        rz_eci = state_history_mci[time_idx, 2] - earth_pos_mci[time_idx, 2]
        vx_eci = state_history_mci[time_idx, 3] - earth_vel_mci[time_idx, 0]
        vy_eci = state_history_mci[time_idx, 4] - earth_vel_mci[time_idx, 1]
        vz_eci = state_history_mci[time_idx, 5] - earth_vel_mci[time_idx, 2]

        r00 = xforms[time_idx, 0, 0]
        r01 = xforms[time_idx, 0, 1]
        r02 = xforms[time_idx, 0, 2]
        r10 = xforms[time_idx, 1, 0]
        r11 = xforms[time_idx, 1, 1]
        r12 = xforms[time_idx, 1, 2]
        r20 = xforms[time_idx, 2, 0]
        r21 = xforms[time_idx, 2, 1]
        r22 = xforms[time_idx, 2, 2]

        d00 = xforms[time_idx, 3, 0]
        d01 = xforms[time_idx, 3, 1]
        d02 = xforms[time_idx, 3, 2]
        d10 = xforms[time_idx, 4, 0]
        d11 = xforms[time_idx, 4, 1]
        d12 = xforms[time_idx, 4, 2]
        d20 = xforms[time_idx, 5, 0]
        d21 = xforms[time_idx, 5, 1]
        d22 = xforms[time_idx, 5, 2]

        r_x = r00 * rx_eci + r01 * ry_eci + r02 * rz_eci
        r_y = r10 * rx_eci + r11 * ry_eci + r12 * rz_eci
        r_z = r20 * rx_eci + r21 * ry_eci + r22 * rz_eci

        v_x = d00 * rx_eci + d01 * ry_eci + d02 * rz_eci + r00 * vx_eci + r01 * vy_eci + r02 * vz_eci
        v_y = d10 * rx_eci + d11 * ry_eci + d12 * rz_eci + r10 * vx_eci + r11 * vy_eci + r12 * vz_eci
        v_z = d20 * rx_eci + d21 * ry_eci + d22 * rz_eci + r20 * vx_eci + r21 * vy_eci + r22 * vz_eci

        rho_x = r_x - station_ecef[station_id, 0]
        rho_y = r_y - station_ecef[station_id, 1]
        rho_z = r_z - station_ecef[station_id, 2]

        az, el, rng = _razel_scalar(rho_x, rho_y, rho_z, station_lat_rad[station_id], station_lon_rad[station_id], two_pi)
        rr = 0.0 if rng <= 0.0 else (rho_x * v_x + rho_y * v_y + rho_z * v_z) / rng

        out[oi, 0] = rng
        out[oi, 1] = rr
        out[oi, 2] = az
        out[oi, 3] = el

    return out


@_optional_njit(cache=True)
def _razel_scalar(rho_x: float, rho_y: float, rho_z: float, lat_rad: float, lon_rad: float, two_pi: float):
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    south = sin_lat * cos_lon * rho_x + sin_lat * sin_lon * rho_y - cos_lat * rho_z
    east = -sin_lon * rho_x + cos_lon * rho_y
    zenith = cos_lat * cos_lon * rho_x + cos_lat * sin_lon * rho_y + sin_lat * rho_z
    rng = np.sqrt(south * south + east * east + zenith * zenith)

    if rng < 1e-9:
        return 0.0, 0.0, rng

    sin_el = zenith / rng
    if sin_el > 1.0:
        sin_el = 1.0
    elif sin_el < -1.0:
        sin_el = -1.0
    el = np.arcsin(sin_el)

    if np.sqrt(south * south + east * east) < 1e-12:
        return 0.0, el, rng

    az = np.arctan2(east, -south)
    if az < 0.0:
        az += two_pi
    return az, el, rng


def station_arrays(stations) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert station objects into array form for accelerated kernels."""
    if len(stations) == 0:
        return (
            np.zeros((0, 3), dtype=float),
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
        )
    station_ecef = np.vstack([np.asarray(station.r_ecef_m, dtype=float).reshape(3) for station in stations])
    station_lat_rad = np.asarray([float(station.lat_rad) for station in stations], dtype=float)
    station_lon_rad = np.asarray([float(station.lon_rad) for station in stations], dtype=float)
    return station_ecef, station_lat_rad, station_lon_rad


def position_observables(
    state_history_mci: ArrayLike,
    obs_data: ArrayLike,
    earth_pos_mci: ArrayLike,
    xforms: ArrayLike,
    stations,
) -> np.ndarray:
    station_ecef, station_lat_rad, station_lon_rad = station_arrays(stations)
    return _position_observables_numba(
        np.asarray(state_history_mci, dtype=float),
        np.asarray(obs_data, dtype=float),
        np.asarray(earth_pos_mci, dtype=float),
        np.asarray(xforms, dtype=float),
        station_ecef,
        station_lat_rad,
        station_lon_rad,
    )


def geometric_range_rate_observables(
    state_history_mci: ArrayLike,
    obs_data: ArrayLike,
    earth_pos_mci: ArrayLike,
    earth_vel_mci: ArrayLike,
    xforms: ArrayLike,
    stations,
) -> np.ndarray:
    station_ecef, station_lat_rad, station_lon_rad = station_arrays(stations)
    return _range_rate_observables_numba(
        np.asarray(state_history_mci, dtype=float),
        np.asarray(obs_data, dtype=float),
        np.asarray(earth_pos_mci, dtype=float),
        np.asarray(earth_vel_mci, dtype=float),
        np.asarray(xforms, dtype=float),
        station_ecef,
        station_lat_rad,
        station_lon_rad,
    )


@_optional_njit(cache=True)
def _apply_stm_to_jacobian_numba(
    obs_data: np.ndarray,
    x_aug_hist: np.ndarray,
    h_tilde: np.ndarray,
    block_size: int,
    time_col: int,
) -> np.ndarray:
    """Apply per-observation STM (stored Fortran-order in x_aug_hist[:,6:]) to h_tilde blocks."""
    n_obs = obs_data.shape[0]
    h_out = np.zeros((n_obs * block_size, 6), dtype=np.float64)
    for obs_idx in range(n_obs):
        time_idx = int(obs_data[obs_idx, time_col]) - 1
        row0 = obs_idx * block_size
        # phi_k[k, c] = x_aug_hist[time_idx, 6 + k + c*6]  (Fortran/column-major layout)
        for r in range(block_size):
            for c in range(6):
                acc = 0.0
                for k in range(6):
                    acc += h_tilde[row0 + r, k] * x_aug_hist[time_idx, 6 + k + c * 6]
                h_out[row0 + r, c] = acc
    return h_out


def apply_stm_to_jacobian(
    obs_data: np.ndarray,
    x_aug_hist: np.ndarray,
    h_tilde: np.ndarray,
    block_size: int,
    time_col: int,
) -> np.ndarray:
    """Apply STM blocks to local Jacobian rows; vectorized replacement for::

        for obs_idx in range(n_obs):
            row0 = obs_idx * block_size
            time_idx = int(obs_data[obs_idx, time_col]) - 1
            phi_k = x_aug_hist[time_idx, 6:].reshape((6, 6), order='F')
            h_out[row0:row0+block_size] = h_tilde[row0:row0+block_size] @ phi_k
    """
    if NUMBA_AVAILABLE:
        return _apply_stm_to_jacobian_numba(
            np.asarray(obs_data, dtype=float),
            np.asarray(x_aug_hist, dtype=float),
            np.asarray(h_tilde, dtype=float),
            int(block_size),
            int(time_col),
        )
    n_obs = obs_data.shape[0]
    h_out = np.zeros((n_obs * block_size, 6), dtype=float)
    for obs_idx in range(n_obs):
        row0 = obs_idx * block_size
        time_idx = int(obs_data[obs_idx, time_col]) - 1
        phi_k = x_aug_hist[time_idx, 6:].reshape((6, 6), order="F")
        h_out[row0 : row0 + block_size, :] = h_tilde[row0 : row0 + block_size, :] @ phi_k
    return h_out


# ---------------------------------------------------------------------------
# Inlined dynamics ODE kernels (zero Python overhead in hot loop)
# ---------------------------------------------------------------------------

@_optional_njit(cache=True, fastmath=True)
def _f3body_rhs_numba(
    x6: np.ndarray,
    mu_moon: float,
    mu_earth: float,
    mu_sun: float,
    r_earth: np.ndarray,
    r_sun: np.ndarray,
    j2_moon: float,
    moon_r: float,
    c_bf: np.ndarray,
) -> np.ndarray:
    """6-state f3body_moon derivative — all arithmetic inlined, no Python calls."""
    rx = x6[0]; ry = x6[1]; rz = x6[2]
    vx = x6[3]; vy = x6[4]; vz = x6[5]

    # Moon point-mass
    r2 = rx*rx + ry*ry + rz*rz
    r  = np.sqrt(r2);  r3 = r * r2
    ax = -mu_moon * rx / r3
    ay = -mu_moon * ry / r3
    az = -mu_moon * rz / r3

    # Earth third-body
    ex = r_earth[0]; ey = r_earth[1]; ez = r_earth[2]
    dex = ex - rx;   dey = ey - ry;   dez = ez - rz
    de2 = dex*dex + dey*dey + dez*dez
    de  = np.sqrt(de2);  de3 = de * de2
    re2 = ex*ex + ey*ey + ez*ez
    re  = np.sqrt(re2);  re3 = re * re2
    ax += mu_earth * (dex/de3 - ex/re3)
    ay += mu_earth * (dey/de3 - ey/re3)
    az += mu_earth * (dez/de3 - ez/re3)

    # Sun third-body
    sx = r_sun[0];  sy = r_sun[1];  sz = r_sun[2]
    dsx = sx - rx;  dsy = sy - ry;  dsz = sz - rz
    ds2 = dsx*dsx + dsy*dsy + dsz*dsz
    ds  = np.sqrt(ds2);  ds3 = ds * ds2
    rs2 = sx*sx + sy*sy + sz*sz
    rs  = np.sqrt(rs2);  rs3 = rs * rs2
    ax += mu_sun * (dsx/ds3 - sx/rs3)
    ay += mu_sun * (dsy/ds3 - sy/rs3)
    az += mu_sun * (dsz/ds3 - sz/rs3)

    # J2 lunar oblateness (no-op when j2_moon == 0)
    if j2_moon != 0.0:
        rbfx = c_bf[0,0]*rx + c_bf[0,1]*ry + c_bf[0,2]*rz
        rbfy = c_bf[1,0]*rx + c_bf[1,1]*ry + c_bf[1,2]*rz
        rbfz = c_bf[2,0]*rx + c_bf[2,1]*ry + c_bf[2,2]*rz
        r2bf = rbfx*rbfx + rbfy*rbfy + rbfz*rbfz
        rbf5 = r2bf * r2bf * np.sqrt(r2bf)
        z2r2 = rbfz * rbfz / r2bf
        scl  = -1.5 * j2_moon * mu_moon * moon_r * moon_r / rbf5
        abfx = scl * rbfx * (1.0 - 5.0 * z2r2)
        abfy = scl * rbfy * (1.0 - 5.0 * z2r2)
        abfz = scl * rbfz * (3.0 - 5.0 * z2r2)
        ax += c_bf[0,0]*abfx + c_bf[1,0]*abfy + c_bf[2,0]*abfz
        ay += c_bf[0,1]*abfx + c_bf[1,1]*abfy + c_bf[2,1]*abfz
        az += c_bf[0,2]*abfx + c_bf[1,2]*abfy + c_bf[2,2]*abfz

    out = np.empty(6)
    out[0] = vx;  out[1] = vy;  out[2] = vz
    out[3] = ax;  out[4] = ay;  out[5] = az
    return out


@_optional_njit(cache=True, fastmath=True)
def _ode42_rhs_numba(
    x42: np.ndarray,
    mu_moon: float,
    mu_earth: float,
    mu_sun: float,
    r_earth: np.ndarray,
    r_sun: np.ndarray,
    j2_moon: float,
    moon_r: float,
    c_bf: np.ndarray,
) -> np.ndarray:
    """42-state augmented ODE derivative [xdot(6) | phi_dot(36)] — fully inlined.

    phi is stored Fortran-order: phi[i,j] = x42[6 + i + j*6].
    A matrix: upper-right = I, lower-left = gravity gradient G, rest = 0.
    phi_dot = A @ phi  →  rows 0-2: phi_dot[i,j] = phi[i+3,j];
                          rows 3-5: phi_dot[i+3,j] = G[i,:] @ phi[:,j].
    """
    rx = x42[0];  ry = x42[1];  rz = x42[2]
    vx = x42[3];  vy = x42[4];  vz = x42[5]

    # Moon point-mass acceleration + gravity gradient
    r2 = rx*rx + ry*ry + rz*rz
    r  = np.sqrt(r2);  r3 = r * r2
    ax = -mu_moon * rx / r3
    ay = -mu_moon * ry / r3
    az = -mu_moon * rz / r3
    f1 = -mu_moon / r3                       # diag contribution
    f3 =  3.0 * mu_moon / (r3 * r2)         # outer-product coefficient

    # Earth third-body acceleration + gradient
    ex = r_earth[0]; ey = r_earth[1]; ez = r_earth[2]
    dex = ex - rx;   dey = ey - ry;   dez = ez - rz
    de2 = dex*dex + dey*dey + dez*dez
    de  = np.sqrt(de2);  de3 = de * de2
    re2 = ex*ex + ey*ey + ez*ez
    re  = np.sqrt(re2);  re3 = re * re2
    ax += mu_earth * (dex/de3 - ex/re3)
    ay += mu_earth * (dey/de3 - ey/re3)
    az += mu_earth * (dez/de3 - ez/re3)
    fe1 = -mu_earth / de3
    fe3 =  3.0 * mu_earth / (de3 * de2)

    # Sun third-body acceleration + gradient
    sx = r_sun[0];  sy = r_sun[1];  sz = r_sun[2]
    dsx = sx - rx;  dsy = sy - ry;  dsz = sz - rz
    ds2 = dsx*dsx + dsy*dsy + dsz*dsz
    ds  = np.sqrt(ds2);  ds3 = ds * ds2
    rs2 = sx*sx + sy*sy + sz*sz
    rs  = np.sqrt(rs2);  rs3 = rs * rs2
    ax += mu_sun * (dsx/ds3 - sx/rs3)
    ay += mu_sun * (dsy/ds3 - sy/rs3)
    az += mu_sun * (dsz/ds3 - sz/rs3)
    fs1 = -mu_sun / ds3
    fs3 =  3.0 * mu_sun / (ds3 * ds2)

    # Gravity gradient G (3×3 lower-left block of A matrix)
    # G[i,j] = (f1+fe1+fs1)*I[i,j] + f3*r[i]*r[j] + fe3*de[i]*de[j] + fs3*ds[i]*ds[j]
    diag = f1 + fe1 + fs1
    g00 = diag + f3*rx*rx + fe3*dex*dex + fs3*dsx*dsx
    g01 =        f3*rx*ry + fe3*dex*dey + fs3*dsx*dsy
    g02 =        f3*rx*rz + fe3*dex*dez + fs3*dsx*dsz
    g10 =        f3*ry*rx + fe3*dey*dex + fs3*dsy*dsx
    g11 = diag + f3*ry*ry + fe3*dey*dey + fs3*dsy*dsy
    g12 =        f3*ry*rz + fe3*dey*dez + fs3*dsy*dsz
    g20 =        f3*rz*rx + fe3*dez*dex + fs3*dsz*dsx
    g21 =        f3*rz*ry + fe3*dez*dey + fs3*dsz*dsy
    g22 = diag + f3*rz*rz + fe3*dez*dez + fs3*dsz*dsz

    # J2 lunar oblateness — acceleration + analytic gravity gradient (no-op when j2_moon == 0)
    if j2_moon != 0.0:
        rbfx = c_bf[0,0]*rx + c_bf[0,1]*ry + c_bf[0,2]*rz
        rbfy = c_bf[1,0]*rx + c_bf[1,1]*ry + c_bf[1,2]*rz
        rbfz = c_bf[2,0]*rx + c_bf[2,1]*ry + c_bf[2,2]*rz
        r2bf = rbfx*rbfx + rbfy*rbfy + rbfz*rbfz
        rbf  = np.sqrt(r2bf)
        r4bf = r2bf * r2bf
        r5bf = r4bf * rbf
        z2r2 = rbfz * rbfz / r2bf
        z2r4 = rbfz * rbfz / r4bf
        scl  = -1.5 * j2_moon * mu_moon * moon_r * moon_r / r5bf
        # J2 acceleration in body-fixed, rotated back to MCI
        abfx = scl * rbfx * (1.0 - 5.0 * z2r2)
        abfy = scl * rbfy * (1.0 - 5.0 * z2r2)
        abfz = scl * rbfz * (3.0 - 5.0 * z2r2)
        ax += c_bf[0,0]*abfx + c_bf[1,0]*abfy + c_bf[2,0]*abfz
        ay += c_bf[0,1]*abfx + c_bf[1,1]*abfy + c_bf[2,1]*abfz
        az += c_bf[0,2]*abfx + c_bf[1,2]*abfy + c_bf[2,2]*abfz
        # Analytic J2 gravity gradient in body-fixed (G_bf), then G_mci = C_bf^T @ G_bf @ C_bf
        base_bf = 1.0 - 5.0 * z2r2
        gj00 = scl * (base_bf - 5.0*rbfx*rbfx/r2bf + 35.0*rbfx*rbfx*z2r4)
        gj11 = scl * (base_bf - 5.0*rbfy*rbfy/r2bf + 35.0*rbfy*rbfy*z2r4)
        gj22 = scl * (3.0 - 30.0*z2r2 + 35.0*rbfz*rbfz*z2r4)
        gj01 = scl * rbfx*rbfy * (-5.0/r2bf + 35.0*z2r4)
        gj02 = scl * rbfx*rbfz * (-15.0/r2bf + 35.0*z2r4)
        gj12 = scl * rbfy*rbfz * (-15.0/r2bf + 35.0*z2r4)
        # tmp = G_bf @ C_bf  (G_bf symmetric: gj10=gj01, gj20=gj02, gj21=gj12)
        t00 = gj00*c_bf[0,0] + gj01*c_bf[1,0] + gj02*c_bf[2,0]
        t01 = gj00*c_bf[0,1] + gj01*c_bf[1,1] + gj02*c_bf[2,1]
        t02 = gj00*c_bf[0,2] + gj01*c_bf[1,2] + gj02*c_bf[2,2]
        t10 = gj01*c_bf[0,0] + gj11*c_bf[1,0] + gj12*c_bf[2,0]
        t11 = gj01*c_bf[0,1] + gj11*c_bf[1,1] + gj12*c_bf[2,1]
        t12 = gj01*c_bf[0,2] + gj11*c_bf[1,2] + gj12*c_bf[2,2]
        t20 = gj02*c_bf[0,0] + gj12*c_bf[1,0] + gj22*c_bf[2,0]
        t21 = gj02*c_bf[0,1] + gj12*c_bf[1,1] + gj22*c_bf[2,1]
        t22 = gj02*c_bf[0,2] + gj12*c_bf[1,2] + gj22*c_bf[2,2]
        # G_mci += C_bf^T @ tmp  (C_bf^T[i,k] = C_bf[k,i])
        g00 += c_bf[0,0]*t00 + c_bf[1,0]*t10 + c_bf[2,0]*t20
        g01 += c_bf[0,0]*t01 + c_bf[1,0]*t11 + c_bf[2,0]*t21
        g02 += c_bf[0,0]*t02 + c_bf[1,0]*t12 + c_bf[2,0]*t22
        g10 += c_bf[0,1]*t00 + c_bf[1,1]*t10 + c_bf[2,1]*t20
        g11 += c_bf[0,1]*t01 + c_bf[1,1]*t11 + c_bf[2,1]*t21
        g12 += c_bf[0,1]*t02 + c_bf[1,1]*t12 + c_bf[2,1]*t22
        g20 += c_bf[0,2]*t00 + c_bf[1,2]*t10 + c_bf[2,2]*t20
        g21 += c_bf[0,2]*t01 + c_bf[1,2]*t11 + c_bf[2,2]*t21
        g22 += c_bf[0,2]*t02 + c_bf[1,2]*t12 + c_bf[2,2]*t22

    out = np.empty(42)
    out[0] = vx;  out[1] = vy;  out[2] = vz
    out[3] = ax;  out[4] = ay;  out[5] = az

    # phi_dot = A @ phi  (column-by-column, Fortran layout)
    for j in range(6):
        base = 6 + j * 6
        p0 = x42[base];     p1 = x42[base + 1];  p2 = x42[base + 2]
        p3 = x42[base + 3]; p4 = x42[base + 4];  p5 = x42[base + 5]
        out[base]     = p3                           # phi_dot[0,j] = phi[3,j]
        out[base + 1] = p4                           # phi_dot[1,j] = phi[4,j]
        out[base + 2] = p5                           # phi_dot[2,j] = phi[5,j]
        out[base + 3] = g00*p0 + g01*p1 + g02*p2   # phi_dot[3,j]
        out[base + 4] = g10*p0 + g11*p1 + g12*p2   # phi_dot[4,j]
        out[base + 5] = g20*p0 + g21*p1 + g22*p2   # phi_dot[5,j]

    return out


def f3body_rhs(
    x6: np.ndarray,
    mu_moon: float,
    mu_earth: float,
    mu_sun: float,
    r_earth: np.ndarray,
    r_sun: np.ndarray,
    j2_moon: float = 0.0,
    moon_r: float = 0.0,
    c_bf: np.ndarray | None = None,
) -> np.ndarray:
    """6-state ODE RHS — numba-accelerated when available, numpy fallback otherwise."""
    c_bf_arr = np.eye(3, dtype=np.float64) if c_bf is None else np.asarray(c_bf, dtype=np.float64)
    return _f3body_rhs_numba(
        np.asarray(x6, dtype=np.float64),
        float(mu_moon), float(mu_earth), float(mu_sun),
        np.asarray(r_earth, dtype=np.float64).reshape(3),
        np.asarray(r_sun, dtype=np.float64).reshape(3),
        float(j2_moon), float(moon_r), c_bf_arr,
    )


def ode42_rhs(
    x42: np.ndarray,
    mu_moon: float,
    mu_earth: float,
    mu_sun: float,
    r_earth: np.ndarray,
    r_sun: np.ndarray,
    j2_moon: float = 0.0,
    moon_r: float = 0.0,
    c_bf: np.ndarray | None = None,
) -> np.ndarray:
    """42-state augmented ODE RHS — numba-accelerated when available, numpy fallback otherwise."""
    c_bf_arr = np.eye(3, dtype=np.float64) if c_bf is None else np.asarray(c_bf, dtype=np.float64)
    return _ode42_rhs_numba(
        np.asarray(x42, dtype=np.float64),
        float(mu_moon), float(mu_earth), float(mu_sun),
        np.asarray(r_earth, dtype=np.float64).reshape(3),
        np.asarray(r_sun, dtype=np.float64).reshape(3),
        float(j2_moon), float(moon_r), c_bf_arr,
    )


# ---------------------------------------------------------------------------
# Fast ephemeris interpolation helper
# ---------------------------------------------------------------------------

@_optional_njit(cache=True, fastmath=True)
def _lerp_vec3_numba(t: float, t_grid: np.ndarray, data_grid: np.ndarray) -> np.ndarray:
    """Binary-search linear interpolation of an (N, 3) grid at scalar t."""
    n = len(t_grid)
    lo = 0; hi = n - 1
    while lo < hi - 1:
        mid = (lo + hi) >> 1
        if t_grid[mid] <= t:
            lo = mid
        else:
            hi = mid
    i = lo if lo < n - 1 else n - 2
    alpha = (t - t_grid[i]) / (t_grid[i + 1] - t_grid[i])
    out = np.empty(3)
    out[0] = data_grid[i, 0] + alpha * (data_grid[i + 1, 0] - data_grid[i, 0])
    out[1] = data_grid[i, 1] + alpha * (data_grid[i + 1, 1] - data_grid[i, 1])
    out[2] = data_grid[i, 2] + alpha * (data_grid[i + 1, 2] - data_grid[i, 2])
    return out


@_optional_njit(cache=True, fastmath=True)
def _rk4_6state_numba(
    y0: np.ndarray,
    t0: float,
    tf: float,
    dt: float,
    mu_moon: float,
    mu_earth: float,
    mu_sun: float,
    t_grid: np.ndarray,
    earth_grid: np.ndarray,
    sun_grid: np.ndarray,
    j2_moon: float,
    moon_r: float,
    c_bf: np.ndarray,
) -> np.ndarray:
    """Fixed-step RK4 for 6-state f3body_moon dynamics — pure numba, no Python overhead.

    Uses binary-search linear ephemeris interpolation.  Suitable for short
    propagation intervals (UKF sigma steps) where 3 mm accuracy is sufficient.
    """
    n = max(1, int(round((tf - t0) / dt)))
    h = (tf - t0) / n
    y = y0.copy()
    t = t0
    for _ in range(n):
        re = _lerp_vec3_numba(t,           t_grid, earth_grid)
        rs = _lerp_vec3_numba(t,           t_grid, sun_grid)
        k1 = _f3body_rhs_numba(y,               mu_moon, mu_earth, mu_sun, re, rs, j2_moon, moon_r, c_bf)
        re = _lerp_vec3_numba(t + 0.5 * h, t_grid, earth_grid)
        rs = _lerp_vec3_numba(t + 0.5 * h, t_grid, sun_grid)
        k2 = _f3body_rhs_numba(y + 0.5*h*k1, mu_moon, mu_earth, mu_sun, re, rs, j2_moon, moon_r, c_bf)
        k3 = _f3body_rhs_numba(y + 0.5*h*k2, mu_moon, mu_earth, mu_sun, re, rs, j2_moon, moon_r, c_bf)
        re = _lerp_vec3_numba(t + h,        t_grid, earth_grid)
        rs = _lerp_vec3_numba(t + h,        t_grid, sun_grid)
        k4 = _f3body_rhs_numba(y + h * k3,  mu_moon, mu_earth, mu_sun, re, rs, j2_moon, moon_r, c_bf)
        y = y + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        t = t + h
    return y


def rk4_6state(
    y0: ArrayLike,
    t0: float,
    tf: float,
    mu_moon: float,
    mu_earth: float,
    mu_sun: float,
    t_grid: ArrayLike,
    earth_grid: ArrayLike,
    sun_grid: ArrayLike,
    *,
    dt: float = 10.0,
    j2_moon: float = 0.0,
    moon_r: float = 0.0,
    c_bf: np.ndarray | None = None,
) -> np.ndarray:
    """Fixed-step RK4 for 6-state dynamics — numba-accelerated.

    ``dt`` is the nominal step size in seconds.  The actual step is adjusted so
    the integration lands exactly on ``tf``.

    Accuracy: ~3 mm for 60 s arcs (dominated by linear ephemeris interpolation
    vs PCHIP).  Use for UKF sigma propagation, not for BLS arc integration.
    """
    c_bf_arr = np.eye(3, dtype=np.float64) if c_bf is None else np.asarray(c_bf, dtype=np.float64)
    return _rk4_6state_numba(
        np.asarray(y0, dtype=np.float64),
        float(t0), float(tf), float(dt),
        float(mu_moon), float(mu_earth), float(mu_sun),
        np.asarray(t_grid,    dtype=np.float64),
        np.asarray(earth_grid, dtype=np.float64),
        np.asarray(sun_grid,   dtype=np.float64),
        float(j2_moon), float(moon_r), c_bf_arr,
    )


def make_lerp_vec3(t_grid: ArrayLike, data_grid: ArrayLike):
    """Return a fast scalar-time → (3,) interpolant for an ephemeris grid.

    Uses numba binary-search linear interpolation when available; falls back to
    scipy interp1d.  Suitable when the grid is dense enough that linear accuracy
    suffices (roughly ≤ 1-minute spacing for Earth/Sun ephemerides).
    """
    t_g = np.asarray(t_grid, dtype=np.float64).reshape(-1)
    d_g = np.asarray(data_grid, dtype=np.float64)
    if d_g.ndim != 2 or d_g.shape[1] != 3:
        raise ValueError("data_grid must have shape (N, 3).")
    if t_g.size != d_g.shape[0]:
        raise ValueError("t_grid and data_grid must have the same length.")
    if NUMBA_AVAILABLE:
        def interp(t: float) -> np.ndarray:
            return _lerp_vec3_numba(float(t), t_g, d_g)
    else:
        from scipy.interpolate import interp1d
        _f = interp1d(t_g, d_g, axis=0, kind="linear", fill_value="extrapolate")
        def interp(t: float) -> np.ndarray:
            return _f(float(t))
    return interp
