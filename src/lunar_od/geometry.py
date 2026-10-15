"""Geometry helpers ported from the MATLAB Lunar OD utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

WGS84_A_M = 6378137.0
WGS84_INV_F = 298.257223563


def geodetic_to_ecef_wgs84(
    lat_deg: ArrayLike,
    lon_deg: ArrayLike,
    alt_m: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert WGS84 geodetic coordinates to ECEF meters.

    Inputs follow the MATLAB function `geodetic_to_ecef_wgs84.m`:
    latitude and longitude are degrees, altitude is meters.
    """
    lat_rad = np.asarray(lat_deg, dtype=float) * (np.pi / 180.0)
    lon_rad = np.asarray(lon_deg, dtype=float) * (np.pi / 180.0)
    alt_m = np.asarray(alt_m, dtype=float)

    f = 1.0 / WGS84_INV_F
    e2 = 2.0 * f - f**2

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    n = WGS84_A_M / np.sqrt(1.0 - e2 * sin_lat**2)

    x_ecef_m = (n + alt_m) * cos_lat * cos_lon
    y_ecef_m = (n + alt_m) * cos_lat * sin_lon
    z_ecef_m = (n * (1.0 - e2) + alt_m) * sin_lat
    return x_ecef_m, y_ecef_m, z_ecef_m


def ecef2sez_dcm(lat_rad: float, lon_rad: float) -> np.ndarray:
    """Return the ECEF-to-SEZ direction cosine matrix.

    SEZ axes are South, East, Zenith. The returned matrix satisfies
    `rho_sez = C_SEZ_ECEF @ rho_ecef`.
    """
    if np.ndim(lat_rad) != 0 or np.ndim(lon_rad) != 0:
        raise ValueError("Latitude and longitude must be scalar radians.")

    sin_lat = np.sin(float(lat_rad))
    cos_lat = np.cos(float(lat_rad))
    sin_lon = np.sin(float(lon_rad))
    cos_lon = np.cos(float(lon_rad))

    return np.array(
        [
            [sin_lat * cos_lon, sin_lat * sin_lon, -cos_lat],
            [-sin_lon, cos_lon, 0.0],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=float,
    )


def ecef2razel_sez(
    r_rel_ecef_m: ArrayLike,
    lat_rad: float,
    lon_rad: float,
) -> tuple[float, float, float]:
    """Compute azimuth, elevation, and range from a relative ECEF vector.

    Azimuth is clockwise from North and returned in `[0, 2*pi)`.
    """
    r_rel_ecef_m = np.asarray(r_rel_ecef_m, dtype=float).reshape(-1)
    if r_rel_ecef_m.size != 3:
        raise ValueError("Relative vector must have exactly 3 elements.")

    rho_sez_m = ecef2sez_dcm(lat_rad, lon_rad) @ r_rel_ecef_m
    south_m, east_m, zenith_m = rho_sez_m
    range_m = float(np.linalg.norm(rho_sez_m))

    if range_m < 1e-9:
        return 0.0, 0.0, range_m

    sin_el = float(np.clip(zenith_m / range_m, -1.0, 1.0))
    elevation_rad = float(np.arcsin(sin_el))

    if float(np.hypot(south_m, east_m)) < 1e-12:
        return 0.0, elevation_rad, range_m

    azimuth_rad = float(np.arctan2(east_m, -south_m))
    if azimuth_rad < 0.0:
        azimuth_rad += 2.0 * np.pi

    return azimuth_rad, elevation_rad, range_m


def wrap_to_pi(angle_rad: ArrayLike) -> np.ndarray:
    """Wrap angles to the principal interval using atan2(sin, cos)."""
    angle_rad = np.asarray(angle_rad, dtype=float)
    return np.arctan2(np.sin(angle_rad), np.cos(angle_rad))
