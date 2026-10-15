"""Visibility, lunar occultation, and arc segmentation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import ArrayLike

from .accelerated import visibility_mask_ecef


@dataclass(frozen=True)
class VisibilityConfig:
    r_moon_mean_m: float
    earth_rotation_rad_s: float
    epoch_utc: str
    min_elevation_deg: float = 10.0


def calc_gst_curtis(
    year: ArrayLike,
    month: ArrayLike,
    day: ArrayLike,
    hour: ArrayLike,
    minute: ArrayLike,
    second: ArrayLike,
) -> np.ndarray:
    """Compute Greenwich sidereal time using MATLAB `calc_GST_Curtis.m`."""
    year = np.asarray(year, dtype=float)
    month = np.asarray(month, dtype=float)
    day = np.asarray(day, dtype=float)
    hour = np.asarray(hour, dtype=float)
    minute = np.asarray(minute, dtype=float)
    second = np.asarray(second, dtype=float)

    is_jan_feb = month <= 2
    y = year - is_jan_feb.astype(float)
    m = month + 12.0 * is_jan_feb.astype(float)

    a = np.floor(y / 100.0)
    b = 2.0 - a + np.floor(a / 4.0)

    j0 = (
        np.floor(365.25 * (y + 4716.0))
        + np.floor(30.6001 * (m + 1.0))
        + day
        + b
        - 1524.5
    )

    t0 = (j0 - 2451545.0) / 36525.0
    theta_g0_deg = 100.4606184 + 36000.77004 * t0 + 0.000387933 * t0**2 - 2.583e-8 * t0**3
    theta_g0_deg = np.mod(theta_g0_deg, 360.0)

    ut_hours = hour + minute / 60.0 + second / 3600.0
    theta_deg = theta_g0_deg + (360.98564724 / 24.0) * ut_hours
    return np.deg2rad(np.mod(theta_deg, 360.0))


def is_occulted_by_moon(
    r_observer_m: ArrayLike,
    r_target_m: ArrayLike,
    r_body_m: ArrayLike,
    radius_body_m: float,
) -> np.ndarray:
    """Line-segment spherical occultation test matching `isOccultedByMoon.m`."""
    r_observer_m = np.asarray(r_observer_m, dtype=float).reshape(1, 3)
    r_target_m = _as_n_by_3(r_target_m, "r_target_m")
    r_body_m = _as_n_by_3(r_body_m, "r_body_m")
    if r_body_m.shape[0] == 1 and r_target_m.shape[0] > 1:
        r_body_m = np.repeat(r_body_m, r_target_m.shape[0], axis=0)
    if r_body_m.shape[0] != r_target_m.shape[0]:
        raise ValueError("r_body_m must have one row or the same number of rows as r_target_m.")

    vec_a = r_observer_m - r_body_m
    vec_b = r_target_m - r_body_m

    mag2_a = np.sum(vec_a * vec_a, axis=1)
    mag2_b = np.sum(vec_b * vec_b, axis=1)
    dot_ab = np.sum(vec_a * vec_b, axis=1)
    len2_segment = mag2_a + mag2_b - 2.0 * dot_ab

    tau_min = (mag2_a - dot_ab) / len2_segment
    valid_tau = (tau_min >= 0.0) & (tau_min <= 1.0)

    vec_ab = vec_b - vec_a
    vec_c = vec_a + vec_ab * tau_min[:, None]
    dist2_closest = np.sum(vec_c * vec_c, axis=1)
    r2_safe = radius_body_m**2 * 0.999999999

    return (len2_segment > 1e-15) & valid_tau & (dist2_closest < r2_safe)


def analyze_visibility_gap(
    t_sim_s: ArrayLike,
    state_history_mci: ArrayLike,
    stations,
    get_earth_pos,
    max_gap_s: float,
    config: VisibilityConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute raw station visibility and network gap-bridged arcs."""
    t_sim_s = np.asarray(t_sim_s, dtype=float).reshape(-1)
    state_history_mci = np.asarray(state_history_mci, dtype=float)
    if state_history_mci.ndim != 2 or state_history_mci.shape[1] < 3:
        raise ValueError("state_history_mci must have shape (N, >=3).")
    if state_history_mci.shape[0] != t_sim_s.size:
        raise ValueError("state_history_mci row count must match t_sim_s.")
    if t_sim_s.size < 2:
        raise ValueError("At least two simulation epochs are required.")

    num_steps = t_sim_s.size
    dt_s = float(t_sim_s[1] - t_sim_s[0])
    min_elev_rad = np.deg2rad(config.min_elevation_deg)

    dt = _parse_epoch(config.epoch_utc)
    gst0_rad = float(
        calc_gst_curtis(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)
    )
    gst_vec = gst0_rad + config.earth_rotation_rad_s * t_sim_s
    c_g = np.cos(gst_vec)
    s_g = np.sin(gst_vec)

    r_earth_vec = np.asarray(get_earth_pos(t_sim_s), dtype=float)
    if r_earth_vec.shape == (3, num_steps):
        r_earth_vec = r_earth_vec.T
    if r_earth_vec.shape != (num_steps, 3):
        raise ValueError("get_earth_pos(t_sim_s) must return shape (N, 3) or (3, N).")

    r_mci = state_history_mci[:, :3]
    r_eci = r_mci - r_earth_vec
    r_moon_eci = -r_earth_vec

    r_ecef = np.column_stack(
        [
            r_eci[:, 0] * c_g + r_eci[:, 1] * s_g,
            -r_eci[:, 0] * s_g + r_eci[:, 1] * c_g,
            r_eci[:, 2],
        ]
    )
    r_moon_ecef = np.column_stack(
        [
            r_moon_eci[:, 0] * c_g + r_moon_eci[:, 1] * s_g,
            -r_moon_eci[:, 0] * s_g + r_moon_eci[:, 1] * c_g,
            r_moon_eci[:, 2],
        ]
    )

    station_ecef, station_lat_rad, station_lon_rad = _station_arrays(stations)
    vis_mask_raw = visibility_mask_ecef(
        r_ecef,
        r_moon_ecef,
        station_ecef,
        station_lat_rad,
        station_lon_rad,
        min_elev_rad,
        config.r_moon_mean_m,
    )

    is_net_vis_raw = np.any(vis_mask_raw, axis=1)
    gap_steps = max(int(np.ceil(max_gap_s / dt_s)), 1)
    is_net_vis_filled = is_net_vis_raw.copy()

    padded_raw = np.concatenate([[False], is_net_vis_raw, [False]]).astype(int)
    d_net = np.diff(padded_raw)
    starts = np.where(d_net == 1)[0]
    ends = np.where(d_net == -1)[0] - 1

    for m in range(len(starts) - 1):
        gap_len = starts[m + 1] - ends[m] - 1
        if gap_len > 0 and gap_len <= gap_steps:
            is_net_vis_filled[ends[m] + 1 : starts[m + 1]] = True

    padded_filled = np.concatenate([[False], is_net_vis_filled, [False]]).astype(int)
    d_filled = np.diff(padded_filled)
    seg_starts = np.where(d_filled == 1)[0]
    seg_ends = np.where(d_filled == -1)[0] - 1

    return seg_starts, seg_ends, vis_mask_raw, is_net_vis_filled


def analyze_visibility_gap_with_transforms(
    t_sim_s: ArrayLike,
    state_history_mci: ArrayLike,
    stations,
    get_earth_pos,
    x_j2000_to_itrf93: ArrayLike,
    max_gap_s: float,
    config: VisibilityConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute visibility using supplied J2000->ITRF93 position transforms.

    This variant mirrors the measurement model's frame path more closely than
    the fast GST rotation used by `analyze_visibility_gap`.
    """
    t_sim_s = np.asarray(t_sim_s, dtype=float).reshape(-1)
    state_history_mci = np.asarray(state_history_mci, dtype=float)
    if state_history_mci.ndim != 2 or state_history_mci.shape[1] < 3:
        raise ValueError("state_history_mci must have shape (N, >=3).")
    if state_history_mci.shape[0] != t_sim_s.size:
        raise ValueError("state_history_mci row count must match t_sim_s.")
    if t_sim_s.size < 2:
        raise ValueError("At least two simulation epochs are required.")

    xforms = np.asarray(x_j2000_to_itrf93, dtype=float)
    if xforms.shape == (t_sim_s.size, 6, 6):
        rot_j2k_itrf = xforms[:, :3, :3]
    elif xforms.shape == (t_sim_s.size, 3, 3):
        rot_j2k_itrf = xforms
    else:
        raise ValueError("x_j2000_to_itrf93 must have shape (N, 6, 6) or (N, 3, 3).")

    num_steps = t_sim_s.size
    dt_s = float(t_sim_s[1] - t_sim_s[0])
    min_elev_rad = np.deg2rad(config.min_elevation_deg)

    r_earth_vec = np.asarray(get_earth_pos(t_sim_s), dtype=float)
    if r_earth_vec.shape == (3, num_steps):
        r_earth_vec = r_earth_vec.T
    if r_earth_vec.shape != (num_steps, 3):
        raise ValueError("get_earth_pos(t_sim_s) must return shape (N, 3) or (3, N).")

    r_mci = state_history_mci[:, :3]
    r_eci = r_mci - r_earth_vec
    r_moon_eci = -r_earth_vec

    r_ecef = np.einsum("nij,nj->ni", rot_j2k_itrf, r_eci)
    r_moon_ecef = np.einsum("nij,nj->ni", rot_j2k_itrf, r_moon_eci)

    station_ecef, station_lat_rad, station_lon_rad = _station_arrays(stations)
    vis_mask_raw = visibility_mask_ecef(
        r_ecef,
        r_moon_ecef,
        station_ecef,
        station_lat_rad,
        station_lon_rad,
        min_elev_rad,
        config.r_moon_mean_m,
    )

    is_net_vis_filled = _gap_fill_network_visibility(vis_mask_raw, max_gap_s, dt_s)
    seg_starts, seg_ends = _segments_from_mask(is_net_vis_filled)
    return seg_starts, seg_ends, vis_mask_raw, is_net_vis_filled


def sample_j2000_to_itrf93_transforms(et0: float, t_s: ArrayLike) -> np.ndarray:
    """Sample SPICE J2000->ITRF93 6x6 state transforms on a time grid."""
    import spiceypy as spice

    t_s = np.asarray(t_s, dtype=float).reshape(-1)
    xforms = np.zeros((t_s.size, 6, 6), dtype=float)
    for idx, rel_t_s in enumerate(t_s):
        xforms[idx, :, :] = np.asarray(spice.sxform("J2000", "ITRF93", float(et0 + rel_t_s)), dtype=float)
    return xforms


def _as_n_by_3(value: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 1 and array.size == 3:
        return array.reshape(1, 3)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3) or 3 elements.")
    return array


def _parse_epoch(epoch_utc: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(epoch_utc, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unsupported epoch format: {epoch_utc}")


def _station_arrays(stations) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _gap_fill_network_visibility(vis_mask_raw: np.ndarray, max_gap_s: float, dt_s: float) -> np.ndarray:
    is_net_vis_raw = np.any(vis_mask_raw, axis=1)
    gap_steps = max(int(np.ceil(max_gap_s / dt_s)), 1)
    is_net_vis_filled = is_net_vis_raw.copy()

    starts, ends = _segments_from_mask(is_net_vis_raw)
    for m in range(len(starts) - 1):
        gap_len = starts[m + 1] - ends[m] - 1
        if gap_len > 0 and gap_len <= gap_steps:
            is_net_vis_filled[ends[m] + 1 : starts[m + 1]] = True
    return is_net_vis_filled


def _segments_from_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]]).astype(int)
    d_mask = np.diff(padded)
    starts = np.where(d_mask == 1)[0]
    ends = np.where(d_mask == -1)[0] - 1
    return starts, ends
