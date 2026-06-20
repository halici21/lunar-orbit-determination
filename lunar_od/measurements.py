"""Synthetic measurement generation and residual helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .geometry import ecef2razel_sez, ecef2sez_dcm, wrap_to_pi
from .accelerated import geometric_range_rate_observables, position_observables
from .radiometrics import (
    RangeRatePhysicsConfig,
    instantaneous_geometric_range_rate,
    range_rate_physics_config,
    two_way_counted_doppler_observable,
    interp_state_history,
)

C_LIGHT_MPS = 299792458.0


@dataclass(frozen=True)
class PassGeometry:
    t_s: np.ndarray
    earth_pos_mci_m: np.ndarray
    earth_vel_mci_mps: np.ndarray
    x_j2000_to_itrf93: np.ndarray
    stations: tuple
    measurement_type: str
    range_rate_physics: RangeRatePhysicsConfig | None = None
    apply_light_time: bool = False
    apply_stellar_aberration: bool = False


@dataclass(frozen=True)
class LightTimeSolution:
    range_m: float
    light_time_s: float
    transmit_time_s: float
    iterations: int
    converged: bool
    target_position_m: np.ndarray


def solve_one_way_light_time(
    receive_time_s: float,
    observer_position_m: ArrayLike,
    get_target_position_m,
    *,
    light_speed_mps: float = C_LIGHT_MPS,
    tolerance_s: float = 1e-12,
    max_iter: int = 10,
) -> LightTimeSolution:
    """Iterate one-way geometric light-time from target transmit to receive time."""
    observer_position_m = np.asarray(observer_position_m, dtype=float).reshape(3)
    if light_speed_mps <= 0.0:
        raise ValueError("light_speed_mps must be positive.")
    if tolerance_s <= 0.0:
        raise ValueError("tolerance_s must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    receive_time_s = float(receive_time_s)
    target_position = np.asarray(get_target_position_m(receive_time_s), dtype=float).reshape(3)
    light_time_s = float(np.linalg.norm(target_position - observer_position_m) / light_speed_mps)
    converged = False

    for iteration in range(1, max_iter + 1):
        transmit_time_s = receive_time_s - light_time_s
        target_position = np.asarray(get_target_position_m(transmit_time_s), dtype=float).reshape(3)
        new_light_time_s = float(np.linalg.norm(target_position - observer_position_m) / light_speed_mps)
        if abs(new_light_time_s - light_time_s) <= tolerance_s:
            light_time_s = new_light_time_s
            converged = True
            break
        light_time_s = new_light_time_s
    else:
        iteration = max_iter

    transmit_time_s = receive_time_s - light_time_s
    range_m = light_time_s * light_speed_mps
    return LightTimeSolution(
        range_m=range_m,
        light_time_s=light_time_s,
        transmit_time_s=transmit_time_s,
        iterations=iteration,
        converged=converged,
        target_position_m=target_position,
    )


def apply_stellar_aberration(
    rho_vec_inertial: ArrayLike,
    observer_velocity_inertial: ArrayLike,
    *,
    light_speed_mps: float = C_LIGHT_MPS,
) -> np.ndarray:
    """Reception-case Newtonian stellar aberration (SPICE ``+S`` model).

    Rotate the light-time corrected inertial line-of-sight vector
    ``rho_vec_inertial`` *toward* the observer's inertial velocity by the
    aberration angle ``phi`` where ``sin(phi) = (v / c) * sin(w)`` and ``w`` is
    the angle between the line of sight and the observer velocity. The rotation
    axis is ``h = rho x v_obs`` and the rotation is evaluated with the Rodrigues
    formula. The returned vector has the same magnitude as the input (a pure
    rotation), so range derived from it is unchanged.

    ``observer_velocity_inertial`` must be expressed in the same inertial frame
    as ``rho_vec_inertial`` (here Moon-centred J2000); do not pass an ECEF
    velocity. Note that the dominant SPICE CN+S correction uses the observer
    velocity relative to the solar-system barycentre; this frame-relative
    formulation omits the Moon's barycentric velocity and is intended as a
    self-consistent synthetic-measurement model (the identical model is used in
    generation and prediction, so no estimator bias is introduced).
    """
    r = np.asarray(rho_vec_inertial, dtype=float).reshape(3)
    v = np.asarray(observer_velocity_inertial, dtype=float).reshape(3)
    r_norm = float(np.linalg.norm(r))
    v_norm = float(np.linalg.norm(v))
    if r_norm == 0.0 or v_norm == 0.0:
        return r.copy()
    h = np.cross(r, v)
    h_norm = float(np.linalg.norm(h))
    if h_norm == 0.0:
        return r.copy()  # line of sight parallel to velocity: sin(w) = 0
    k_hat = h / h_norm
    sin_w = h_norm / (r_norm * v_norm)
    sin_phi = float(np.clip((v_norm / light_speed_mps) * sin_w, -1.0, 1.0))
    phi = float(np.arcsin(sin_phi))
    cos_phi = float(np.cos(phi))
    # Rodrigues rotation of r about unit axis k_hat by +phi (toward v_obs).
    return (
        r * cos_phi
        + np.cross(k_hat, r) * sin_phi
        + k_hat * float(np.dot(k_hat, r)) * (1.0 - cos_phi)
    )


def _apparent_position_observable(
    receive_time_s: float,
    station,
    t_grid_s: ArrayLike,
    state_history_mci: ArrayLike,
    earth_pos_mci_rx: ArrayLike,
    x_j2k_itrf_rx: ArrayLike,
    *,
    earth_vel_mci_rx: ArrayLike | None = None,
    apply_stellar: bool = False,
    light_speed_mps: float = C_LIGHT_MPS,
    tolerance_s: float = 1e-12,
    max_iter: int = 10,
) -> tuple[np.ndarray, float, float, int]:
    """Apparent (one-way light-time corrected) [range, az, el] for one receive epoch.

    The spacecraft position is evaluated at the transmit time ``t_t = t_r - tau``;
    the Earth centre, station, and topocentric SEZ frame are evaluated at the
    receive time ``t_r``. ``earth_pos_mci_rx`` and ``x_j2k_itrf_rx`` are the
    receive-epoch values (already indexed, not interpolated). The transmit-time
    spacecraft state is cubic-Hermite interpolated from ``state_history_mci``;
    linear extrapolation is used when ``t_t`` falls just before the grid start
    (e.g. at the first receive epoch).

    When ``apply_stellar`` is set, the converged-light-time inertial line of
    sight is additionally rotated by the reception-case stellar aberration
    correction (:func:`apply_stellar_aberration`) before az/el/range are formed.
    This requires ``earth_vel_mci_rx`` (Earth-centre velocity in the MCI frame at
    the receive time) so the observer's inertial velocity can be assembled. The
    light-time solution itself is unchanged.
    """
    earth_pos_mci_rx = np.asarray(earth_pos_mci_rx, dtype=float).reshape(3)
    x_rx = np.asarray(x_j2k_itrf_rx, dtype=float)
    r_rot_rx = x_rx[:3, :3]
    station_ecef = np.asarray(station.r_ecef_m, dtype=float).reshape(3)
    # Station inertial (MCI) state at the receive time: Earth centre plus the
    # ITRF93->J2000 transform of the (fixed) station ECEF state (6x6 solve). The
    # velocity part carries the inertial velocity due to Earth rotation.
    station_ecef_state = np.concatenate([station_ecef, np.zeros(3)])
    station_rel_state_j2000 = np.linalg.solve(x_rx, station_ecef_state)
    station_rel_j2000 = station_rel_state_j2000[:3]
    station_mci_rx = earth_pos_mci_rx + station_rel_j2000

    solution = solve_one_way_light_time(
        receive_time_s,
        station_mci_rx,
        lambda t: interp_state_history(t_grid_s, state_history_mci, t)[:3],
        light_speed_mps=light_speed_mps,
        tolerance_s=tolerance_s,
        max_iter=max_iter,
    )
    transmit_time_s = solution.transmit_time_s
    r_sc_tt = interp_state_history(t_grid_s, state_history_mci, transmit_time_s)[:3]

    if apply_stellar:
        if earth_vel_mci_rx is None:
            raise ValueError("apply_stellar=True requires earth_vel_mci_rx.")
        earth_vel_mci_rx = np.asarray(earth_vel_mci_rx, dtype=float).reshape(3)
        # Observer (station) inertial velocity = Earth-centre MCI velocity plus
        # the inertial velocity from Earth rotation (velocity part of the solve).
        observer_vel_inertial = earth_vel_mci_rx + station_rel_state_j2000[3:]
        # CN-corrected inertial line of sight, then stellar aberration rotation.
        rho_inertial = apply_stellar_aberration(
            r_sc_tt - station_mci_rx,
            observer_vel_inertial,
            light_speed_mps=light_speed_mps,
        )
        rho_ecef = r_rot_rx @ rho_inertial
    else:
        # Apparent line of sight: spacecraft at transmit time, Earth/frame/station
        # at receive time. ||rho_ecef|| equals the converged light-time range.
        rho_ecef = r_rot_rx @ (r_sc_tt - earth_pos_mci_rx) - station_ecef

    az_rad, el_rad, range_m = ecef2razel_sez(rho_ecef, station.lat_rad, station.lon_rad)
    z = np.array([range_m, az_rad, el_rad], dtype=float)
    return z, transmit_time_s, solution.light_time_s, int(solution.iterations)


def _apparent_position_rowwise(
    state_history_mci: np.ndarray,
    obs_data: np.ndarray,
    pass_geo: "PassGeometry",
    *,
    apply_stellar_aberration: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Row-by-row apparent observables for an obs_data block.

    Returns ``(h_meas, r_transmit_mci)`` where ``h_meas`` is (n_obs, 3) and
    ``r_transmit_mci`` is the (n_obs, 3) spacecraft position evaluated at each
    transmit time (used by the approximate light-time analytic Jacobian).

    When stellar aberration is active (``pass_geo.apply_stellar_aberration`` or
    the ``apply_stellar_aberration`` override), the apparent line of sight is
    rotated by the reception-case stellar aberration correction; ``r_transmit``
    is unaffected (it is the transmit-time spacecraft position).
    """
    use_stellar = bool(
        getattr(pass_geo, "apply_stellar_aberration", False)
        if apply_stellar_aberration is None
        else apply_stellar_aberration
    )
    earth_vel = getattr(pass_geo, "earth_vel_mci_mps", None)
    n_obs = obs_data.shape[0]
    h_meas = np.zeros((n_obs, 3), dtype=float)
    r_tx = np.zeros((n_obs, 3), dtype=float)
    for i in range(n_obs):
        station_id = int(obs_data[i, 4]) - 1
        time_idx = int(obs_data[i, 5]) - 1
        receive_time_s = float(obs_data[i, 0])
        z, transmit_time_s, _lt, _it = _apparent_position_observable(
            receive_time_s,
            pass_geo.stations[station_id],
            pass_geo.t_s,
            state_history_mci,
            pass_geo.earth_pos_mci_m[time_idx],
            pass_geo.x_j2000_to_itrf93[time_idx],
            earth_vel_mci_rx=(
                earth_vel[time_idx] if (use_stellar and earth_vel is not None) else None
            ),
            apply_stellar=use_stellar,
        )
        h_meas[i] = z
        r_tx[i] = interp_state_history(pass_geo.t_s, state_history_mci, transmit_time_s)[:3]
    return h_meas, r_tx


def generate_position_measurements(
    t_pass_s: ArrayLike,
    state_history_mci: ArrayLike,
    stations,
    vis_mask_raw: ArrayLike,
    get_earth_pos,
    get_earth_vel,
    et0: float,
    *,
    noise: bool = True,
    rng: np.random.Generator | None = None,
    arc_id: int | None = None,
    apply_light_time: bool = False,
    apply_stellar_aberration: bool = False,
) -> tuple[np.ndarray, PassGeometry, np.ndarray]:
    """Generate range/azimuth/elevation measurements.

    ObsData follows MATLAB's convention:
    `[t, range, az, el, station_id_1based, time_index_1based, optional_arc_id]`.
    """
    import spiceypy as spice

    t_pass_s = np.asarray(t_pass_s, dtype=float).reshape(-1)
    state_history_mci = np.asarray(state_history_mci, dtype=float)
    vis_mask_raw = np.asarray(vis_mask_raw, dtype=bool)
    if vis_mask_raw.ndim == 1:
        vis_mask_raw = vis_mask_raw.reshape(-1, 1)

    n_steps = t_pass_s.size
    if state_history_mci.shape[0] != n_steps or state_history_mci.shape[1] < 3:
        raise ValueError("state_history_mci must have shape (N, >=3).")
    if apply_light_time and state_history_mci.shape[1] < 6:
        raise ValueError(
            "apply_light_time=True requires a 6-state (position+velocity) history."
        )
    if apply_stellar_aberration and not apply_light_time:
        raise ValueError(
            "apply_stellar_aberration=True requires apply_light_time=True "
            "(stellar aberration is applied on top of the light-time solution)."
        )
    if vis_mask_raw.shape != (n_steps, len(stations)):
        raise ValueError("vis_mask_raw must have shape (N, num_stations).")

    r_earth_mci = _ensure_n_by_3(get_earth_pos(t_pass_s), n_steps, "r_earth_mci")
    v_earth_mci = _ensure_n_by_3(get_earth_vel(t_pass_s), n_steps, "v_earth_mci")
    xforms = np.zeros((n_steps, 6, 6), dtype=float)

    include_arc_id = arc_id is not None
    num_cols = 7 if include_arc_id else 6
    total_measurements = int(np.sum(vis_mask_raw))
    obs_data = np.zeros((total_measurements, num_cols), dtype=float)
    clean_obs_data = np.zeros((total_measurements, num_cols), dtype=float)
    rng = rng or np.random.default_rng()

    obs_counter = 0
    for k, t_s in enumerate(t_pass_s):
        x_j2k_itrf = np.asarray(spice.sxform("J2000", "ITRF93", float(et0 + t_s)), dtype=float)
        xforms[k, :, :] = x_j2k_itrf

        active_station_cols = np.where(vis_mask_raw[k, :])[0]
        if active_station_cols.size == 0:
            continue

        r_sat_mci = state_history_mci[k, :3]
        dr_eci = r_sat_mci - r_earth_mci[k, :]
        r_sat_ecef = x_j2k_itrf[:3, :3] @ dr_eci

        for station_col in active_station_cols:
            station = stations[station_col]
            if apply_light_time:
                z_clean, _t_t, _lt, _it = _apparent_position_observable(
                    float(t_s), station, t_pass_s, state_history_mci,
                    r_earth_mci[k], x_j2k_itrf,
                    earth_vel_mci_rx=v_earth_mci[k] if apply_stellar_aberration else None,
                    apply_stellar=apply_stellar_aberration,
                )
            else:
                rho_vec_ecef = r_sat_ecef - station.r_ecef_m
                az_rad, el_rad, range_m = ecef2razel_sez(rho_vec_ecef, station.lat_rad, station.lon_rad)
                z_clean = np.array([range_m, az_rad, el_rad], dtype=float)

            bias_vec = np.asarray(getattr(station, "bias", np.zeros(3)), dtype=float).reshape(-1)
            if bias_vec.size != 3:
                bias_vec = np.zeros(3)
            if noise:
                noise_vec = np.array(
                    [
                        station.sigma_range_m * rng.standard_normal(),
                        station.sigma_angle_rad * rng.standard_normal(),
                        station.sigma_angle_rad * rng.standard_normal(),
                    ],
                    dtype=float,
                )
            else:
                noise_vec = np.zeros(3)
            z_noisy = z_clean + noise_vec + bias_vec

            station_id_1based = station_col + 1
            time_index_1based = k + 1
            row_noisy = [t_s, z_noisy[0], z_noisy[1], z_noisy[2], station_id_1based, time_index_1based]
            row_clean = [t_s, z_clean[0], z_clean[1], z_clean[2], station_id_1based, time_index_1based]
            if include_arc_id:
                row_noisy.append(float(arc_id))
                row_clean.append(float(arc_id))
            obs_data[obs_counter, :] = row_noisy
            clean_obs_data[obs_counter, :] = row_clean
            obs_counter += 1

    pass_geo = PassGeometry(
        t_s=t_pass_s,
        earth_pos_mci_m=r_earth_mci,
        earth_vel_mci_mps=v_earth_mci,
        x_j2000_to_itrf93=xforms,
        stations=tuple(stations),
        measurement_type="position",
        apply_light_time=apply_light_time,
        apply_stellar_aberration=apply_stellar_aberration,
    )
    return obs_data[:obs_counter, :], pass_geo, clean_obs_data[:obs_counter, :]


def compute_position_residuals(
    state_history_mci: ArrayLike,
    obs_data: ArrayLike,
    pass_geo: PassGeometry,
    *,
    apply_light_time: bool = False,
    apply_stellar_aberration: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute position-only observed-minus-computed residuals.

    When ``pass_geo.apply_light_time`` (or the ``apply_light_time`` override) is
    set, the computed measurements use the one-way light-time corrected apparent
    geometry, row by row, matching :func:`generate_position_measurements`.
    Otherwise the Numba instantaneous fast path is used unchanged. The residual
    sign convention (observed - computed) and angle wrapping are unchanged."""
    state_history_mci = np.asarray(state_history_mci, dtype=float)
    obs_data = np.asarray(obs_data, dtype=float)
    use_stellar = bool(
        getattr(pass_geo, "apply_stellar_aberration", False)
        if apply_stellar_aberration is None
        else apply_stellar_aberration
    )
    use_light_time = bool(
        getattr(pass_geo, "apply_light_time", False) or apply_light_time or use_stellar
    )
    if not use_light_time:
        h_meas = position_observables(
            state_history_mci,
            obs_data,
            pass_geo.earth_pos_mci_m,
            pass_geo.x_j2000_to_itrf93,
            pass_geo.stations,
        )
    else:
        h_meas, _ = _apparent_position_rowwise(
            state_history_mci, obs_data, pass_geo, apply_stellar_aberration=use_stellar
        )

    diff_raw = obs_data[:, 1:4] - h_meas
    diff_raw[:, 1] = wrap_to_pi(diff_raw[:, 1])
    diff_raw[:, 2] = wrap_to_pi(diff_raw[:, 2])
    _zero_tiny_position_residuals(diff_raw)
    residuals = diff_raw.T.reshape(-1, order="F")
    return residuals, h_meas


def generate_range_rate_measurements(
    t_pass_s: ArrayLike,
    state_history_mci: ArrayLike,
    stations,
    vis_mask_raw: ArrayLike,
    get_earth_pos,
    get_earth_vel,
    et0: float,
    *,
    bias_range_m: float = 0.0,
    bias_rr_mps: float = 0.0,
    bias_az_rad: float = 0.0,
    bias_el_rad: float = 0.0,
    noise: bool = True,
    rng: np.random.Generator | None = None,
    arc_id: int | None = None,
    range_rate_physics: RangeRatePhysicsConfig | str | None = None,
) -> tuple[np.ndarray, PassGeometry]:
    """Generate range/range-rate/azimuth/elevation measurements."""
    import spiceypy as spice

    t_pass_s = np.asarray(t_pass_s, dtype=float).reshape(-1)
    state_history_mci = np.asarray(state_history_mci, dtype=float)
    vis_mask_raw = np.asarray(vis_mask_raw, dtype=bool)
    if vis_mask_raw.ndim == 1:
        vis_mask_raw = vis_mask_raw.reshape(-1, 1)

    n_steps = t_pass_s.size
    if state_history_mci.shape[0] != n_steps or state_history_mci.shape[1] != 6:
        raise ValueError("state_history_mci must have shape (N, 6).")
    if vis_mask_raw.shape != (n_steps, len(stations)):
        raise ValueError("vis_mask_raw must have shape (N, num_stations).")
    rr_physics = range_rate_physics_config(range_rate_physics)

    r_earth_mci = _ensure_n_by_3(get_earth_pos(t_pass_s), n_steps, "r_earth_mci")
    v_earth_mci = _ensure_n_by_3(get_earth_vel(t_pass_s), n_steps, "v_earth_mci")
    xforms = np.zeros((n_steps, 6, 6), dtype=float)
    for k, t_s in enumerate(t_pass_s):
        xforms[k, :, :] = np.asarray(spice.sxform("J2000", "ITRF93", float(et0 + t_s)), dtype=float)

    include_arc_id = arc_id is not None
    num_cols = 8 if include_arc_id else 7
    total_measurements = int(np.sum(vis_mask_raw))
    obs_data = np.zeros((total_measurements, num_cols), dtype=float)
    rng = rng or np.random.default_rng()

    obs_counter = 0
    for k, t_s in enumerate(t_pass_s):
        active_station_cols = np.where(vis_mask_raw[k, :])[0]
        if active_station_cols.size == 0:
            continue

        state_sat_mci = state_history_mci[k, :]
        state_earth_mci = np.concatenate([r_earth_mci[k, :], v_earth_mci[k, :]])
        state_sat_eci = state_sat_mci - state_earth_mci
        state_sat_ecef = xforms[k, :, :] @ state_sat_eci
        r_sat_ecef = state_sat_ecef[:3]
        v_sat_ecef = state_sat_ecef[3:]

        for station_col in active_station_cols:
            station = stations[station_col]
            rho_vec_ecef = r_sat_ecef - station.r_ecef_m
            rho_dot_ecef = v_sat_ecef

            range_ideal = float(np.linalg.norm(rho_vec_ecef))
            if rr_physics.mode == "geometric_instantaneous":
                rr_ideal = instantaneous_geometric_range_rate(rho_vec_ecef, rho_dot_ecef)
            else:
                rr_ideal = two_way_counted_doppler_observable(
                    float(t_s),
                    station,
                    t_pass_s,
                    state_history_mci,
                    r_earth_mci,
                    v_earth_mci,
                    xforms,
                    rr_physics,
                )
            az_ideal, el_ideal, _ = ecef2razel_sez(rho_vec_ecef, station.lat_rad, station.lon_rad)

            station_bias = np.asarray(getattr(station, "bias", []), dtype=float).reshape(-1)
            if station_bias.size == 4:
                bias_vec = station_bias
            else:
                bias_vec = np.array([bias_range_m, bias_rr_mps, bias_az_rad, bias_el_rad], dtype=float)

            if noise:
                noise_vec = np.array(
                    [
                        station.sigma_range_m * rng.standard_normal(),
                        station.sigma_range_rate_mps * rng.standard_normal(),
                        station.sigma_angle_rad * rng.standard_normal(),
                        station.sigma_angle_rad * rng.standard_normal(),
                    ],
                    dtype=float,
                )
            else:
                noise_vec = np.zeros(4)

            z = np.array([range_ideal, rr_ideal, az_ideal, el_ideal], dtype=float) + bias_vec + noise_vec
            station_id_1based = station_col + 1
            time_index_1based = k + 1
            row = [t_s, z[0], z[1], z[2], z[3], station_id_1based, time_index_1based]
            if include_arc_id:
                row.append(float(arc_id))
            obs_data[obs_counter, :] = row
            obs_counter += 1

    pass_geo = PassGeometry(
        t_s=t_pass_s,
        earth_pos_mci_m=r_earth_mci,
        earth_vel_mci_mps=v_earth_mci,
        x_j2000_to_itrf93=xforms,
        stations=tuple(stations),
        measurement_type="range_rate",
        range_rate_physics=rr_physics,
    )
    return obs_data[:obs_counter, :], pass_geo


def compute_range_rate_residuals(
    state_history_mci: ArrayLike,
    obs_data: ArrayLike,
    pass_geo: PassGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute range-rate observed-minus-computed residuals."""
    state_history_mci = np.asarray(state_history_mci, dtype=float)
    obs_data = np.asarray(obs_data, dtype=float)
    n_obs = obs_data.shape[0]
    h_meas = np.zeros((n_obs, 4), dtype=float)
    rr_physics = range_rate_physics_config(pass_geo.range_rate_physics)

    if rr_physics.mode == "geometric_instantaneous":
        h_meas = geometric_range_rate_observables(
            state_history_mci,
            obs_data,
            pass_geo.earth_pos_mci_m,
            pass_geo.earth_vel_mci_mps,
            pass_geo.x_j2000_to_itrf93,
            pass_geo.stations,
        )
        diff_raw = obs_data[:, 1:5] - h_meas
        diff_raw[:, 2] = wrap_to_pi(diff_raw[:, 2])
        diff_raw[:, 3] = wrap_to_pi(diff_raw[:, 3])
        _zero_tiny_range_rate_residuals(diff_raw)
        residuals = diff_raw.T.reshape(-1, order="F")
        return residuals, h_meas

    for obs_idx in range(n_obs):
        station_id = int(obs_data[obs_idx, 5]) - 1
        time_idx = int(obs_data[obs_idx, 6]) - 1
        station = pass_geo.stations[station_id]

        state_sat_mci = state_history_mci[time_idx, :]
        state_earth_mci = np.concatenate(
            [pass_geo.earth_pos_mci_m[time_idx, :], pass_geo.earth_vel_mci_mps[time_idx, :]]
        )
        state_sat_eci = state_sat_mci - state_earth_mci
        state_sat_ecef = pass_geo.x_j2000_to_itrf93[time_idx, :, :] @ state_sat_eci
        r_sat_ecef = state_sat_ecef[:3]
        v_sat_ecef = state_sat_ecef[3:]

        r_rel_ecef = r_sat_ecef - station.r_ecef_m
        v_rel_ecef = v_sat_ecef
        range_val = float(np.linalg.norm(r_rel_ecef))
        if rr_physics.mode == "geometric_instantaneous":
            rr_val = instantaneous_geometric_range_rate(r_rel_ecef, v_rel_ecef)
        else:
            rr_val = two_way_counted_doppler_observable(
                float(pass_geo.t_s[time_idx]),
                station,
                pass_geo.t_s,
                state_history_mci,
                pass_geo.earth_pos_mci_m,
                pass_geo.earth_vel_mci_mps,
                pass_geo.x_j2000_to_itrf93,
                rr_physics,
            )
        az_rad, el_rad, _ = ecef2razel_sez(r_rel_ecef, station.lat_rad, station.lon_rad)
        h_meas[obs_idx, :] = [range_val, rr_val, az_rad, el_rad]

    diff_raw = obs_data[:, 1:5] - h_meas
    diff_raw[:, 2] = wrap_to_pi(diff_raw[:, 2])
    diff_raw[:, 3] = wrap_to_pi(diff_raw[:, 3])
    _zero_tiny_range_rate_residuals(diff_raw)
    residuals = diff_raw.T.reshape(-1, order="F")
    return residuals, h_meas


def _zero_tiny_position_residuals(diff_raw: np.ndarray) -> None:
    """Suppress insignificant arithmetic roundoff in synthetic closure tests."""
    if diff_raw.size == 0:
        return
    diff_raw[np.abs(diff_raw[:, 0]) < 1e-7, 0] = 0.0
    diff_raw[np.abs(diff_raw[:, 1]) < 1e-14, 1] = 0.0
    diff_raw[np.abs(diff_raw[:, 2]) < 1e-14, 2] = 0.0


def _zero_tiny_range_rate_residuals(diff_raw: np.ndarray) -> None:
    """Suppress insignificant arithmetic roundoff in synthetic closure tests."""
    if diff_raw.size == 0:
        return
    diff_raw[np.abs(diff_raw[:, 0]) < 1e-7, 0] = 0.0
    diff_raw[np.abs(diff_raw[:, 1]) < 1e-10, 1] = 0.0
    diff_raw[np.abs(diff_raw[:, 2]) < 1e-14, 2] = 0.0
    diff_raw[np.abs(diff_raw[:, 3]) < 1e-14, 3] = 0.0


def compute_position_residuals_analytic(
    state_history_mci: ArrayLike,
    obs_data: ArrayLike,
    pass_geo: PassGeometry,
    *,
    apply_light_time: bool = False,
    apply_stellar_aberration: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Position-only residuals and local analytic 3x6 H_tilde blocks.

    Light-time corrected residuals are exact for the implemented model, but the
    analytic Jacobian uses a first-order receive-epoch STM approximation and
    neglects implicit light-time derivatives (d(tau)/dx and the difference
    between STM(t_t) and STM(t_r)). This is intended as a first-stage
    implementation, not a finite-difference-exact light-time Jacobian."""
    residuals, h_meas = compute_position_residuals(
        state_history_mci, obs_data, pass_geo,
        apply_light_time=apply_light_time,
        apply_stellar_aberration=apply_stellar_aberration,
    )
    state_history_mci = np.asarray(state_history_mci, dtype=float)
    obs_data = np.asarray(obs_data, dtype=float)
    n_obs = obs_data.shape[0]

    station_ids = obs_data[:, 4].astype(int) - 1    # (n_obs,)
    time_idxs   = obs_data[:, 5].astype(int) - 1    # (n_obs,)

    n_stations = len(pass_geo.stations)
    c_sez_all = np.stack(
        [ecef2sez_dcm(pass_geo.stations[i].lat_rad, pass_geo.stations[i].lon_rad)
         for i in range(n_stations)]
    )  # (n_stations, 3, 3)
    r_ecef_all = np.stack(
        [pass_geo.stations[i].r_ecef_m for i in range(n_stations)]
    )  # (n_stations, 3)

    use_stellar = bool(
        getattr(pass_geo, "apply_stellar_aberration", False)
        if apply_stellar_aberration is None
        else apply_stellar_aberration
    )
    use_light_time = bool(
        getattr(pass_geo, "apply_light_time", False) or apply_light_time or use_stellar
    )
    if use_light_time:
        # Apparent geometry: spacecraft evaluated at each transmit time. The H
        # block below reuses the instantaneous formula at this apparent LOS
        # (first-order receive-epoch STM approximation; the stellar aberration
        # rotation derivative is neglected per the first-stage design).
        _, r_sat = _apparent_position_rowwise(
            state_history_mci, obs_data, pass_geo, apply_stellar_aberration=use_stellar
        )
    else:
        r_sat = state_history_mci[time_idxs, :3]                    # (n_obs, 3)
    dr_eci = r_sat - pass_geo.earth_pos_mci_m[time_idxs, :]         # (n_obs, 3)
    R_j2k  = pass_geo.x_j2000_to_itrf93[time_idxs, :3, :3]          # (n_obs, 3, 3)

    dr_ecef  = np.einsum('nij,nj->ni', R_j2k, dr_eci)               # (n_obs, 3)
    rho_ecef = dr_ecef - r_ecef_all[station_ids]                     # (n_obs, 3)

    c_sez = c_sez_all[station_ids]                                    # (n_obs, 3, 3)
    rho_sez = np.einsum('nij,nj->ni', c_sez, rho_ecef)              # (n_obs, 3)

    south, east, zenith = rho_sez[:, 0], rho_sez[:, 1], rho_sez[:, 2]
    range_m = np.linalg.norm(rho_sez, axis=1)                        # (n_obs,)
    rho_h2  = south**2 + east**2                                      # (n_obs,)
    rho2    = range_m**2                                              # (n_obs,)
    rho_h   = np.sqrt(np.maximum(rho_h2, 1e-12))

    valid    = range_m > 1e-9
    rng_safe = np.where(valid, range_m, 1.0)
    rho2_safe = np.where(valid, rho2, 1.0)
    az_valid  = rho_h2 >= 1e-12
    rho_h2_safe = np.where(az_valid, rho_h2, 1.0)

    d_range = np.where(valid[:, None], rho_sez / rng_safe[:, None], 0.0)

    d_az = np.zeros((n_obs, 3))
    d_az[:, 0] = np.where(az_valid,  east  / rho_h2_safe, 0.0)
    d_az[:, 1] = np.where(az_valid, -south / rho_h2_safe, 0.0)

    d_el = np.column_stack([
        np.where(valid, -(zenith * south) / (rho2_safe * rho_h), 0.0),
        np.where(valid, -(zenith * east)  / (rho2_safe * rho_h), 0.0),
        np.where(valid,  rho_h / rho2_safe,                      0.0),
    ])

    d_meas_dsez = np.stack([d_range, d_az, d_el], axis=1)           # (n_obs, 3, 3)
    dsez_drmci  = np.einsum('nij,njk->nik', c_sez, R_j2k)           # (n_obs, 3, 3)
    dmeas_drmci = np.einsum('nij,njk->nik', d_meas_dsez, dsez_drmci) # (n_obs, 3, 3)

    h_tilde = np.zeros((3 * n_obs, 6), dtype=float)
    h_tilde[:, :3] = dmeas_drmci.reshape(3 * n_obs, 3)

    return residuals, h_meas, h_tilde


def compute_range_rate_residuals_analytic(
    state_history_mci: ArrayLike,
    obs_data: ArrayLike,
    pass_geo: PassGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RR residuals and local analytic 4x6 H_tilde blocks."""
    rr_physics = range_rate_physics_config(pass_geo.range_rate_physics)
    if rr_physics.mode != "geometric_instantaneous":
        raise NotImplementedError(
            "Analytic H_tilde is currently implemented for geometric_instantaneous RR only; "
            "use compute_range_rate_residuals or UKF-style numerical propagation for two_way_counted_doppler."
        )
    residuals, h_meas = compute_range_rate_residuals(state_history_mci, obs_data, pass_geo)
    state_history_mci = np.asarray(state_history_mci, dtype=float)
    obs_data = np.asarray(obs_data, dtype=float)
    n_obs = obs_data.shape[0]

    station_ids = obs_data[:, 5].astype(int) - 1    # (n_obs,)
    time_idxs   = obs_data[:, 6].astype(int) - 1    # (n_obs,)

    n_stations = len(pass_geo.stations)
    c_sez_all = np.stack(
        [ecef2sez_dcm(pass_geo.stations[i].lat_rad, pass_geo.stations[i].lon_rad)
         for i in range(n_stations)]
    )  # (n_stations, 3, 3)
    r_ecef_all = np.stack(
        [pass_geo.stations[i].r_ecef_m for i in range(n_stations)]
    )  # (n_stations, 3)

    state_sat  = state_history_mci[time_idxs, :]                     # (n_obs, 6)
    earth_pos  = pass_geo.earth_pos_mci_m[time_idxs, :]              # (n_obs, 3)
    earth_vel  = pass_geo.earth_vel_mci_mps[time_idxs, :]            # (n_obs, 3)
    r_eci      = state_sat[:, :3] - earth_pos                        # (n_obs, 3)
    v_eci      = state_sat[:, 3:] - earth_vel                        # (n_obs, 3)

    R_j2k  = pass_geo.x_j2000_to_itrf93[time_idxs, :3, :3]          # (n_obs, 3, 3)
    dR_j2k = pass_geo.x_j2000_to_itrf93[time_idxs, 3:6, :3]         # (n_obs, 3, 3)

    r_ecef   = np.einsum('nij,nj->ni', R_j2k, r_eci)                # (n_obs, 3)
    v_ecef   = (np.einsum('nij,nj->ni', dR_j2k, r_eci)
                + np.einsum('nij,nj->ni', R_j2k,  v_eci))           # (n_obs, 3)
    rho_ecef = r_ecef - r_ecef_all[station_ids]                      # (n_obs, 3)
    v_rel    = v_ecef                                                 # (n_obs, 3)

    range_m  = np.linalg.norm(rho_ecef, axis=1)                     # (n_obs,)
    valid    = range_m >= 1e-3
    rng_safe = np.where(valid, range_m, 1.0)

    u_hat = rho_ecef / rng_safe[:, None]                             # (n_obs, 3)
    rr    = np.einsum('ni,ni->n', rho_ecef, v_rel) / rng_safe       # (n_obs,)

    c_sez   = c_sez_all[station_ids]                                  # (n_obs, 3, 3)
    rho_sez = np.einsum('nij,nj->ni', c_sez, rho_ecef)              # (n_obs, 3)

    south, east, zenith = rho_sez[:, 0], rho_sez[:, 1], rho_sez[:, 2]
    rho_h2    = south**2 + east**2
    rho2      = rng_safe**2
    rho_h     = np.sqrt(np.maximum(rho_h2, 1e-12))
    az_valid  = rho_h2 >= 1e-12
    rho_h2_safe = np.where(az_valid, rho_h2, 1.0)

    d_range_sez = np.where(valid[:, None], rho_sez / rng_safe[:, None], 0.0)

    d_az_sez = np.zeros((n_obs, 3))
    d_az_sez[:, 0] = np.where(az_valid & valid,  east  / rho_h2_safe, 0.0)
    d_az_sez[:, 1] = np.where(az_valid & valid, -south / rho_h2_safe, 0.0)

    d_el_sez = np.column_stack([
        np.where(valid, -(zenith * south) / (rho2 * rho_h), 0.0),
        np.where(valid, -(zenith * east)  / (rho2 * rho_h), 0.0),
        np.where(valid,  rho_h / rho2,                      0.0),
    ])

    dsez_drmci    = np.einsum('nij,njk->nik', c_sez, R_j2k)         # (n_obs, 3, 3)
    d_range_drmci = np.einsum('ni,nij->nj', d_range_sez, dsez_drmci) # (n_obs, 3)
    d_az_drmci    = np.einsum('ni,nij->nj', d_az_sez,    dsez_drmci) # (n_obs, 3)
    d_el_drmci    = np.einsum('ni,nij->nj', d_el_sez,    dsez_drmci) # (n_obs, 3)

    v_orth       = v_rel - rr[:, None] * u_hat                       # (n_obs, 3)
    drr_drmci    = (np.einsum('ni,nij->nj', v_orth / rng_safe[:, None], R_j2k)
                    + np.einsum('ni,nij->nj', u_hat, dR_j2k))        # (n_obs, 3)
    drr_drmci    = np.where(valid[:, None], drr_drmci, 0.0)
    drr_dvmci    = np.einsum('ni,nij->nj', u_hat, R_j2k)             # (n_obs, 3)
    drr_dvmci    = np.where(valid[:, None], drr_dvmci, 0.0)

    h_tilde = np.zeros((4 * n_obs, 6), dtype=float)
    rows = np.arange(n_obs)
    h_tilde[rows * 4,     :3] = d_range_drmci
    h_tilde[rows * 4 + 1, :3] = drr_drmci
    h_tilde[rows * 4 + 1, 3:] = drr_dvmci
    h_tilde[rows * 4 + 2, :3] = d_az_drmci
    h_tilde[rows * 4 + 3, :3] = d_el_drmci

    return residuals, h_meas, h_tilde


def measurement_sigma_vector(
    obs_data: ArrayLike,
    pass_geo: PassGeometry,
    measurement_type: str | None = None,
) -> np.ndarray:
    """Return per-component measurement standard deviations in residual order."""
    measurement_type = (measurement_type or pass_geo.measurement_type).lower()
    obs_data = np.asarray(obs_data, dtype=float)
    if measurement_type == "position":
        sigma = np.zeros(obs_data.shape[0] * 3, dtype=float)
        for obs_idx in range(obs_data.shape[0]):
            station = pass_geo.stations[int(obs_data[obs_idx, 4]) - 1]
            row0 = obs_idx * 3
            sigma[row0 : row0 + 3] = [
                station.sigma_range_m,
                station.sigma_angle_rad,
                station.sigma_angle_rad,
            ]
    elif measurement_type == "range_rate":
        sigma = np.zeros(obs_data.shape[0] * 4, dtype=float)
        for obs_idx in range(obs_data.shape[0]):
            station = pass_geo.stations[int(obs_data[obs_idx, 5]) - 1]
            if station.sigma_range_rate_mps is None:
                raise ValueError("range-rate measurements require station.sigma_range_rate_mps.")
            row0 = obs_idx * 4
            sigma[row0 : row0 + 4] = [
                station.sigma_range_m,
                station.sigma_range_rate_mps,
                station.sigma_angle_rad,
                station.sigma_angle_rad,
            ]
    else:
        raise ValueError("measurement_type must be 'position' or 'range_rate'.")

    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("Measurement sigmas must be finite and positive.")
    return sigma


def measurement_covariance_matrix(
    obs_data: ArrayLike,
    pass_geo: PassGeometry,
    measurement_type: str | None = None,
) -> np.ndarray:
    """Return diagonal measurement covariance R in residual-vector order."""
    sigma = measurement_sigma_vector(obs_data, pass_geo, measurement_type)
    return np.diag(sigma**2)


def _ensure_n_by_3(value: ArrayLike, n_rows: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape == (3, n_rows):
        array = array.T
    if array.shape != (n_rows, 3):
        raise ValueError(f"{name} must have shape (N, 3) or (3, N).")
    return array


def _range_az_el_partials_sez(rho_sez: ArrayLike) -> np.ndarray:
    rho_sez = np.asarray(rho_sez, dtype=float).reshape(3)
    south, east, zenith = rho_sez
    range_m = float(np.linalg.norm(rho_sez))
    if range_m < 1e-9:
        return np.zeros((3, 3), dtype=float)

    rho_h2 = south**2 + east**2
    rho2 = range_m**2
    d_range_dsez = (rho_sez / range_m).reshape(1, 3)

    if rho_h2 < 1e-12:
        d_az_dsez = np.zeros((1, 3), dtype=float)
    else:
        d_az_dsez = np.array([[east / rho_h2, -south / rho_h2, 0.0]], dtype=float)

    rho_h = float(np.sqrt(max(rho_h2, 1e-12)))
    d_el_dsez = np.array(
        [[-(zenith * south) / (rho2 * rho_h), -(zenith * east) / (rho2 * rho_h), rho_h / rho2]],
        dtype=float,
    )

    return np.vstack([d_range_dsez, d_az_dsez, d_el_dsez])
