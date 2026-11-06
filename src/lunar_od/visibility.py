"""Visibility, lunar occultation, and arc segmentation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import ArrayLike

from .accelerated import visibility_mask_ecef
from .geometry import ecef2sez_dcm
from .history_domain import require_supported_epoch

#: F2 WP2.7 (MEAS-14): mask geometries of :func:`analyze_visibility_gap_with_transforms`.
VISIBILITY_GEOMETRIES = ("geometric", "light_time_downlink", "light_time_two_way")
#: F2 WP2.7: a grid whose spacings all lie within this relative tolerance of the first is uniform.
UNIFORM_GRID_RTOL = 1e-9
_SPEED_OF_LIGHT_MPS = 299792458.0
_LIGHT_TIME_TOL_S = 1e-12
_LIGHT_TIME_MAX_ITERATIONS = 10


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
    _validate_time_grid(t_sim_s)

    num_steps = t_sim_s.size
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

    is_net_vis_filled = _bridge_network_gaps(t_sim_s, vis_mask_raw, max_gap_s)
    seg_starts, seg_ends = _segments_from_mask(is_net_vis_filled)
    return seg_starts, seg_ends, vis_mask_raw, is_net_vis_filled


def analyze_visibility_gap_with_transforms(
    t_sim_s: ArrayLike,
    state_history_mci: ArrayLike,
    stations,
    get_earth_pos,
    x_j2000_to_itrf93: ArrayLike,
    max_gap_s: float,
    config: VisibilityConfig,
    *,
    geometry: str = "geometric",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute visibility using supplied J2000->ITRF93 position transforms.

    This variant mirrors the measurement model's frame path more closely than
    the fast GST rotation used by `analyze_visibility_gap`.

    ``geometry`` (F2 WP2.7, MEAS-14) selects the mask geometry:

    - ``"geometric"`` (default): spacecraft, station and Moon at the same epoch.
    - ``"light_time_downlink"``: ``t_sim_s`` is the reception epoch at the
      station; the received signal left the spacecraft at ``t - tau``
      (geocentric Newtonian light time). The apparent direction (light time
      and diurnal aberration) must clear the elevation mask at ``t`` and the
      ray must clear the lunar sphere, the Moon taken at the epoch the ray
      passes its closest point.
    - ``"light_time_two_way"``: the downlink condition and the uplink from the
      same station, transmitted at ``t1 = t - tau - tau_up``: its point-ahead
      direction above the mask at ``t1`` and its ray clear of the Moon.

    The light-time modes need ``(N, 6, 6)`` transforms and a state history
    with positions and velocities (``(N, >=6)``). Network gaps are bridged on
    the sample count for a uniform grid and on time otherwise
    (:func:`_bridge_network_gaps`).
    """
    t_sim_s = np.asarray(t_sim_s, dtype=float).reshape(-1)
    state_history_mci = np.asarray(state_history_mci, dtype=float)
    if state_history_mci.ndim != 2 or state_history_mci.shape[1] < 3:
        raise ValueError("state_history_mci must have shape (N, >=3).")
    if state_history_mci.shape[0] != t_sim_s.size:
        raise ValueError("state_history_mci row count must match t_sim_s.")
    if t_sim_s.size < 2:
        raise ValueError("At least two simulation epochs are required.")
    _validate_time_grid(t_sim_s)
    if geometry not in VISIBILITY_GEOMETRIES:
        raise ValueError(f"geometry must be one of {VISIBILITY_GEOMETRIES}; got {geometry!r}.")

    xforms = np.asarray(x_j2000_to_itrf93, dtype=float)
    if xforms.shape == (t_sim_s.size, 6, 6):
        rot_j2k_itrf = xforms[:, :3, :3]
    elif xforms.shape == (t_sim_s.size, 3, 3):
        rot_j2k_itrf = xforms
    else:
        raise ValueError("x_j2000_to_itrf93 must have shape (N, 6, 6) or (N, 3, 3).")

    min_elev_rad = np.deg2rad(config.min_elevation_deg)
    if geometry != "geometric":
        if xforms.shape != (t_sim_s.size, 6, 6):
            raise ValueError(f"geometry={geometry!r} needs (N, 6, 6) J2000->ITRF93 state transforms.")
        if state_history_mci.shape[1] < 6:
            raise ValueError(f"geometry={geometry!r} needs a state history with velocities, shape (N, >=6).")
        vis_mask_raw = _light_time_visibility_mask(
            t_sim_s,
            state_history_mci,
            stations,
            get_earth_pos,
            xforms,
            float(min_elev_rad),
            float(config.r_moon_mean_m),
            two_way=geometry == "light_time_two_way",
        )
        is_net_vis_filled = _bridge_network_gaps(t_sim_s, vis_mask_raw, max_gap_s)
        seg_starts, seg_ends = _segments_from_mask(is_net_vis_filled)
        return seg_starts, seg_ends, vis_mask_raw, is_net_vis_filled

    num_steps = t_sim_s.size

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

    is_net_vis_filled = _bridge_network_gaps(t_sim_s, vis_mask_raw, max_gap_s)
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


def _validate_time_grid(t_s: np.ndarray) -> None:
    """F2 WP2.7: the visibility grid must be finite and strictly increasing."""
    if not np.all(np.isfinite(t_s)):
        raise ValueError("t_sim_s must be finite.")
    if np.any(np.diff(t_s) <= 0.0):
        raise ValueError("t_sim_s must be strictly increasing.")


def _bridge_network_gaps(t_s: ArrayLike, vis_mask_raw: np.ndarray, max_gap_s: float) -> np.ndarray:
    """Network visibility with short gaps bridged (F2 WP2.7, MEAS-14).

    On a uniform grid (every spacing within ``UNIFORM_GRID_RTOL`` of the first)
    this is :func:`_gap_fill_network_visibility`, unchanged: a run of ``n``
    invisible samples is bridged when ``n <= max(ceil(max_gap_s / dt), 1)``.
    On a non-uniform grid the run between visible samples ``e`` and ``s`` is
    bridged when its midpoint duration
    ``0.5 * ((t[s-1] + t[s]) - (t[e] + t[e+1]))`` is at most ``max_gap_s``.
    The midpoint duration of a uniform run is ``n * dt``, so the two rules agree
    whenever ``max_gap_s`` is a multiple of ``dt``.
    """
    t_s = np.asarray(t_s, dtype=float).reshape(-1)
    dt_s = np.diff(t_s)
    if np.all(np.abs(dt_s - dt_s[0]) <= UNIFORM_GRID_RTOL * abs(dt_s[0])):
        return _gap_fill_network_visibility(vis_mask_raw, max_gap_s, float(dt_s[0]))

    is_net_vis_filled = np.any(vis_mask_raw, axis=1)
    starts, ends = _segments_from_mask(is_net_vis_filled)
    for m in range(len(starts) - 1):
        e, s = ends[m], starts[m + 1]
        if 0.5 * ((t_s[s - 1] + t_s[s]) - (t_s[e] + t_s[e + 1])) <= max_gap_s:
            is_net_vis_filled[e + 1 : s] = True
    return is_net_vis_filled


def _earth_rows(get_earth_pos, t_s: np.ndarray) -> np.ndarray:
    r_earth = np.asarray(get_earth_pos(t_s), dtype=float)
    if r_earth.shape != (t_s.size, 3) and r_earth.shape == (3, t_s.size):
        r_earth = r_earth.T
    if r_earth.shape != (t_s.size, 3):
        raise ValueError("get_earth_pos(t) must return shape (N, 3) or (3, N).")
    return r_earth


def _hermite_positions(t_grid_s: np.ndarray, states: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    """Cubic Hermite positions from positions and velocities: the polynomial of
    ``radiometrics._interp_state``, vectorised, the edge intervals continued at
    most ``GENERIC_HISTORY_MAX_EXTRAPOLATION_S`` (HARDEN-D19)."""
    from .radiometrics import GENERIC_HISTORY_MAX_EXTRAPOLATION_S

    for epoch in (float(np.min(t_s)), float(np.max(t_s))):
        require_supported_epoch(
            epoch,
            float(t_grid_s[0]),
            float(t_grid_s[-1]),
            history_name="state_history_mci",
            model_context="light-time visibility (spacecraft at the light-time epoch)",
            consumer="visibility._hermite_positions",
            max_extrapolation_s=GENERIC_HISTORY_MAX_EXTRAPOLATION_S,
        )
    i1 = np.clip(np.searchsorted(t_grid_s, t_s, side="right"), 1, t_grid_s.size - 1)
    i0 = i1 - 1
    h = t_grid_s[i1] - t_grid_s[i0]
    s = ((t_s - t_grid_s[i0]) / h)[:, None]
    h = h[:, None]
    return (
        (2.0 * s**3 - 3.0 * s**2 + 1.0) * states[i0, :3]
        + (s**3 - 2.0 * s**2 + s) * h * states[i0, 3:6]
        + (-2.0 * s**3 + 3.0 * s**2) * states[i1, :3]
        + (s**3 - s**2) * h * states[i1, 3:6]
    )


def _segment_clear(r_a: np.ndarray, r_b: np.ndarray, r_body: np.ndarray, r2_safe: float):
    """Is the segment ``r_a -> r_b`` clear of the sphere about ``r_body``? The
    test of ``visibility_mask_ecef``, row-wise; also returns the closest-point
    fraction from ``r_a`` (clipped to [0, 1])."""
    vec_a = r_a - r_body
    vec_b = r_b - r_body
    mag2_a = np.sum(vec_a * vec_a, axis=1)
    dot_ab = np.sum(vec_a * vec_b, axis=1)
    len2_segment = mag2_a + np.sum(vec_b * vec_b, axis=1) - 2.0 * dot_ab
    frac = np.divide(mag2_a - dot_ab, len2_segment, out=np.zeros_like(len2_segment), where=len2_segment > 1e-15)
    closest = vec_a + (vec_b - vec_a) * frac[:, None]
    occulted = (len2_segment > 1e-15) & (frac >= 0.0) & (frac <= 1.0) & (np.sum(closest * closest, axis=1) < r2_safe)
    return ~occulted, np.clip(frac, 0.0, 1.0)


def _rotation_after(xforms6: np.ndarray, dt_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """J2000->ITRF93 rotation and its rate at ``t + dt`` from the 6x6 transform at
    ``t``: ``R(t + dt) = exp(Omega dt) R(t)`` with ``Omega = dR R^T`` (Rodrigues;
    the Earth's rotation is constant over a light time)."""
    rot = xforms6[:, :3, :3]
    omega = np.einsum("nij,nkj->nik", xforms6[:, 3:6, :3], rot)
    w = 0.5 * np.stack(
        [omega[:, 2, 1] - omega[:, 1, 2], omega[:, 0, 2] - omega[:, 2, 0], omega[:, 1, 0] - omega[:, 0, 1]], axis=1
    )
    phi = w * dt_s[:, None]
    theta = np.linalg.norm(phi, axis=1)
    k = np.zeros((phi.shape[0], 3, 3))
    k[:, 0, 1], k[:, 0, 2], k[:, 1, 2] = -phi[:, 2], phi[:, 1], -phi[:, 0]
    k[:, 1, 0], k[:, 2, 0], k[:, 2, 1] = phi[:, 2], -phi[:, 1], phi[:, 0]
    safe = np.where(theta > 0.0, theta, 1.0)
    a = np.where(theta > 0.0, np.sin(safe) / safe, 1.0)
    b = np.where(theta > 0.0, (1.0 - np.cos(safe)) / safe**2, 0.5)
    expm = np.eye(3) + a[:, None, None] * k + b[:, None, None] * np.einsum("nij,njk->nik", k, k)
    rot_dt = np.einsum("nij,njk->nik", expm, rot)
    return rot_dt, np.einsum("nij,njk->nik", omega, rot_dt)


def _elevation(rot_j2k_itrf: np.ndarray, los_j2k: np.ndarray, velocity_j2k: np.ndarray, sign: float, c_sez_ecef):
    """Elevation of ``los`` after first-order aberration ``u + b - (u.b) u`` with
    ``b = sign * v / c`` (``+1`` for reception, ``-1`` for the point-ahead of a
    transmission)."""
    u = los_j2k / np.linalg.norm(los_j2k, axis=1)[:, None]
    beta = sign * velocity_j2k / _SPEED_OF_LIGHT_MPS
    u = u + beta - np.sum(u * beta, axis=1)[:, None] * u
    sez = np.einsum("ij,nj->ni", c_sez_ecef, np.einsum("nij,nj->ni", rot_j2k_itrf, u))
    return np.arctan2(sez[:, 2], np.hypot(sez[:, 0], sez[:, 1]))


def _light_time_visibility_mask(
    t_s: np.ndarray,
    state_history_mci: np.ndarray,
    stations,
    get_earth_pos,
    xforms6: np.ndarray,
    min_elev_rad: float,
    r_moon_m: float,
    *,
    two_way: bool,
) -> np.ndarray:
    """F2 WP2.7 (MEAS-14): visibility of the signal received at ``t_s``.

    Geocentric J2000 Newtonian light time by fixed point; the spacecraft by
    :func:`_hermite_positions`, the Earth (Moon-centred) from ``get_earth_pos``.
    Shape ``(N, num_stations)``, the station mask of ``visibility_mask_ecef``.
    """
    c = _SPEED_OF_LIGHT_MPS
    r2_safe = float(r_moon_m) ** 2 * 0.999999999
    states = state_history_mci[:, :6]

    def spacecraft_geocentric(t_tx):
        return _hermite_positions(t_s, states, t_tx) - _earth_rows(get_earth_pos, t_tx)

    def fixed_point(update, tau):
        for _ in range(_LIGHT_TIME_MAX_ITERATIONS):
            tau_next = update(tau)
            if np.max(np.abs(tau_next - tau)) < _LIGHT_TIME_TOL_S:
                return tau_next
            tau = tau_next
        raise ValueError("light-time visibility: the light-time fixed point did not converge.")

    rot = xforms6[:, :3, :3]
    out = np.zeros((t_s.size, len(stations)), dtype=bool)
    for si, station in enumerate(stations):
        r_ecef = np.asarray(station.r_ecef_m, dtype=float).reshape(3)
        c_sez = ecef2sez_dcm(station.lat_rad, station.lon_rad)
        r_rx = np.einsum("nji,j->ni", rot, r_ecef)
        v_rx = np.einsum("nji,j->ni", xforms6[:, 3:6, :3], r_ecef)

        tau = fixed_point(lambda tau: np.linalg.norm(spacecraft_geocentric(t_s - tau) - r_rx, axis=1) / c,
                          np.zeros(t_s.size))
        t_sc = t_s - tau
        r_sc = spacecraft_geocentric(t_sc)
        visible = _elevation(rot, r_sc - r_rx, v_rx, 1.0, c_sez) > min_elev_rad
        _, frac = _segment_clear(r_rx, r_sc, -_earth_rows(get_earth_pos, t_sc), r2_safe)
        clear, _ = _segment_clear(r_rx, r_sc, -_earth_rows(get_earth_pos, t_s - frac * tau), r2_safe)
        visible &= clear

        if two_way:
            def uplink_station(tau_up):
                rot_up, _ = _rotation_after(xforms6, (t_sc - tau_up) - t_s)
                return np.einsum("nji,j->ni", rot_up, r_ecef)

            tau_up = fixed_point(lambda tau_up: np.linalg.norm(r_sc - uplink_station(tau_up), axis=1) / c, tau)
            t_tx = t_sc - tau_up
            rot_up, rot_rate_up = _rotation_after(xforms6, t_tx - t_s)
            r_tx = np.einsum("nji,j->ni", rot_up, r_ecef)
            v_tx = np.einsum("nji,j->ni", rot_rate_up, r_ecef)
            visible &= _elevation(rot_up, r_sc - r_tx, v_tx, -1.0, c_sez) > min_elev_rad
            _, frac = _segment_clear(r_tx, r_sc, -_earth_rows(get_earth_pos, t_sc), r2_safe)
            clear, _ = _segment_clear(r_tx, r_sc, -_earth_rows(get_earth_pos, t_tx + frac * tau_up), r2_safe)
            visible &= clear
        out[:, si] = visible
    return out
