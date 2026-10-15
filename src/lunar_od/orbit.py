"""Orbit-state helpers for the Python Lunar OD port."""

from __future__ import annotations

import numpy as np


def rot_x(angle_rad: float) -> np.ndarray:
    """Passive DCM about the X axis, matching MATLAB `rot_x.m`."""
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, s],
            [0.0, -s, c],
        ],
        dtype=float,
    )


def rot_z(angle_rad: float) -> np.ndarray:
    """Passive DCM about the Z axis, matching MATLAB `rot_z.m`."""
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array(
        [
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def coe2rv(
    semi_major_axis_m: float,
    eccentricity: float,
    inclination_rad: float,
    raan_rad: float,
    arg_periapsis_rad: float,
    true_anomaly_rad: float,
    mu_m3_s2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert classical orbital elements to Cartesian state.

    The implementation mirrors MATLAB `coe2rv_with_utils.m`, including the
    passive 3-1-3 DCM convention.
    """
    if eccentricity < 0.0 or eccentricity >= 1.0:
        raise ValueError("Only closed orbits with 0 <= e < 1 are supported.")

    p_m = semi_major_axis_m * (1.0 - eccentricity**2)
    r_mag_m = p_m / (1.0 + eccentricity * np.cos(true_anomaly_rad))

    r_pqw_m = r_mag_m * np.array(
        [np.cos(true_anomaly_rad), np.sin(true_anomaly_rad), 0.0],
        dtype=float,
    )
    v_pqw_mps = np.sqrt(mu_m3_s2 / p_m) * np.array(
        [-np.sin(true_anomaly_rad), eccentricity + np.cos(true_anomaly_rad), 0.0],
        dtype=float,
    )

    dcm_inertial_to_pqw = rot_z(arg_periapsis_rad) @ rot_x(inclination_rad) @ rot_z(raan_rad)
    r_pqw_to_inertial = dcm_inertial_to_pqw.T

    return r_pqw_to_inertial @ r_pqw_m, r_pqw_to_inertial @ v_pqw_mps


def lunar_initial_state_mci(
    moon_radius_m: float,
    altitude_m: float,
    eccentricity: float,
    inclination_rad: float,
    raan_rad: float,
    arg_periapsis_rad: float,
    true_anomaly_rad: float,
    mu_moon_m3_s2: float,
    moon_pa_to_j2000_sxform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the initial lunar state in MOON_PA and transform it to MCI/J2000."""
    semi_major_axis_m = moon_radius_m + altitude_m
    r_mpa_m, v_mpa_mps = coe2rv(
        semi_major_axis_m,
        eccentricity,
        inclination_rad,
        raan_rad,
        arg_periapsis_rad,
        true_anomaly_rad,
        mu_moon_m3_s2,
    )
    state_mpa = np.concatenate([r_mpa_m, v_mpa_mps])
    state_mci = np.asarray(moon_pa_to_j2000_sxform, dtype=float) @ state_mpa
    return r_mpa_m, v_mpa_mps, state_mci

