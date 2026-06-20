"""Radiometric observable models for range-rate-like measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

C_LIGHT_MPS = 299792458.0
DEFAULT_X_BAND_UPLINK_HZ = 7.2e9
DEFAULT_X_BAND_TURNAROUND_RATIO = 880.0 / 749.0
TAYLOR3_MAX_COUNT_INTERVAL_S = 60.0

RangeRatePhysicsMode = Literal["geometric_instantaneous", "two_way_counted_doppler"]


@dataclass(frozen=True)
class RangeRatePhysicsConfig:
    """Select the physics used for the second RR observable component.

    The default mode reproduces the existing instantaneous geometric line-of-sight
    range-rate. The two-way mode returns a range-rate equivalent derived from a
    constant-uplink counted Doppler observable over ``count_interval_s``.
    """

    mode: RangeRatePhysicsMode = "geometric_instantaneous"
    count_interval_s: float = 60.0
    uplink_frequency_hz: float = DEFAULT_X_BAND_UPLINK_HZ
    turnaround_ratio: float = DEFAULT_X_BAND_TURNAROUND_RATIO
    output_unit: Literal["mps_equivalent", "hz"] = "mps_equivalent"
    light_speed_mps: float = C_LIGHT_MPS
    light_time_tolerance_s: float = 1e-10
    light_time_max_iter: int = 20
    local_state_model: Literal["ode", "taylor3"] = "ode"
    station_clock_offset_s: float = 0.0
    station_clock_drift: float = 0.0
    clock_reference_time_s: float = 0.0
    transponder_delay_s: float = 0.0

    def __post_init__(self) -> None:
        normalized = _normalize_range_rate_mode(self.mode)
        object.__setattr__(self, "mode", normalized)
        if self.count_interval_s <= 0.0:
            raise ValueError("count_interval_s must be positive.")
        if self.uplink_frequency_hz <= 0.0:
            raise ValueError("uplink_frequency_hz must be positive.")
        if self.turnaround_ratio <= 0.0:
            raise ValueError("turnaround_ratio must be positive.")
        if self.output_unit not in {"mps_equivalent", "hz"}:
            raise ValueError("output_unit must be 'mps_equivalent' or 'hz'.")
        if self.light_speed_mps <= 0.0:
            raise ValueError("light_speed_mps must be positive.")
        if self.light_time_tolerance_s <= 0.0:
            raise ValueError("light_time_tolerance_s must be positive.")
        if self.light_time_max_iter <= 0:
            raise ValueError("light_time_max_iter must be positive.")
        if self.local_state_model not in {"ode", "taylor3"}:
            raise ValueError("local_state_model must be 'ode' or 'taylor3'.")
        if self.local_state_model == "taylor3" and self.count_interval_s > TAYLOR3_MAX_COUNT_INTERVAL_S:
            raise ValueError(
                "local_state_model='taylor3' is limited to count_interval_s <= "
                f"{TAYLOR3_MAX_COUNT_INTERVAL_S:g}."
            )
        if not np.isfinite(self.station_clock_offset_s):
            raise ValueError("station_clock_offset_s must be finite.")
        if not np.isfinite(self.station_clock_drift):
            raise ValueError("station_clock_drift must be finite.")
        if not np.isfinite(self.clock_reference_time_s):
            raise ValueError("clock_reference_time_s must be finite.")
        if self.transponder_delay_s < 0.0 or not np.isfinite(self.transponder_delay_s):
            raise ValueError("transponder_delay_s must be finite and non-negative.")


@dataclass(frozen=True)
class RoundTripLightTimeSolution:
    """Round-trip light-time solution for a two-way coherent observable."""

    receive_time_s: float
    transmit_time_s: float
    transponder_time_s: float
    round_trip_light_time_s: float
    uplink_light_time_s: float
    downlink_light_time_s: float
    iterations: int
    converged: bool


def range_rate_physics_config(config: RangeRatePhysicsConfig | str | None) -> RangeRatePhysicsConfig:
    """Normalize a user-provided range-rate physics selector."""
    if config is None:
        return RangeRatePhysicsConfig()
    if isinstance(config, RangeRatePhysicsConfig):
        return config
    if isinstance(config, str):
        return RangeRatePhysicsConfig(mode=_normalize_range_rate_mode(config))
    raise TypeError("range_rate_physics must be None, a string, or RangeRatePhysicsConfig.")


def instantaneous_geometric_range_rate(r_rel_m: ArrayLike, v_rel_mps: ArrayLike) -> float:
    """Return line-of-sight range-rate from relative position and velocity."""
    r_rel_m = np.asarray(r_rel_m, dtype=float).reshape(3)
    v_rel_mps = np.asarray(v_rel_mps, dtype=float).reshape(3)
    range_m = float(np.linalg.norm(r_rel_m))
    if range_m < 1e-9:
        return 0.0
    return float(np.dot(r_rel_m, v_rel_mps) / range_m)


def two_way_counted_doppler_observable(
    receive_mid_time_s: float,
    station,
    t_grid_s: ArrayLike,
    state_history_mci: ArrayLike,
    earth_pos_mci_m: ArrayLike,
    earth_vel_mci_mps: ArrayLike,
    x_j2000_to_itrf93: ArrayLike,
    config: RangeRatePhysicsConfig | str | None = None,
) -> float:
    """Compute simplified two-way counted Doppler or its m/s equivalent.

    This controlled DSN-like model assumes a constant uplink frequency and fixed
    coherent turnaround ratio. Optional station clock and transponder-delay
    errors can be enabled for mismatch campaigns. Media corrections are not yet
    modeled.
    """
    cfg = range_rate_physics_config(config)
    if cfg.mode != "two_way_counted_doppler":
        raise ValueError("two_way_counted_doppler_observable requires two_way_counted_doppler mode.")

    half_tc = 0.5 * cfg.count_interval_s
    t_start = float(receive_mid_time_s) - half_tc
    t_end = float(receive_mid_time_s) + half_tc
    receive_start = _clock_corrected_receive_time(t_start, cfg)
    receive_end = _clock_corrected_receive_time(t_end, cfg)
    rho_start = solve_two_way_light_time(
        receive_start,
        station,
        t_grid_s,
        state_history_mci,
        earth_pos_mci_m,
        earth_vel_mci_mps,
        x_j2000_to_itrf93,
        cfg,
    ).round_trip_light_time_s
    rho_end = solve_two_way_light_time(
        receive_end,
        station,
        t_grid_s,
        state_history_mci,
        earth_pos_mci_m,
        earth_vel_mci_mps,
        x_j2000_to_itrf93,
        cfg,
    ).round_trip_light_time_s
    rho_rate = (rho_end - rho_start) / cfg.count_interval_s
    doppler_hz = cfg.turnaround_ratio * cfg.uplink_frequency_hz * rho_rate
    if cfg.output_unit == "hz":
        return float(doppler_hz)
    return float(cfg.light_speed_mps * rho_rate / 2.0)


def two_way_counted_doppler_initial_state_jacobian(
    receive_mid_time_s: float,
    station,
    t_grid_s: ArrayLike,
    augmented_state_history_mci: ArrayLike,
    earth_pos_mci_m: ArrayLike,
    earth_vel_mci_mps: ArrayLike,
    x_j2000_to_itrf93: ArrayLike,
    config: RangeRatePhysicsConfig | str | None = None,
) -> np.ndarray:
    """Return analytic counted-Doppler partials with respect to arc initial state."""
    cfg = range_rate_physics_config(config)
    if cfg.mode != "two_way_counted_doppler":
        raise ValueError("two_way_counted_doppler_initial_state_jacobian requires two_way_counted_doppler mode.")

    half_tc = 0.5 * cfg.count_interval_s
    receive_start = _clock_corrected_receive_time(float(receive_mid_time_s) - half_tc, cfg)
    receive_end = _clock_corrected_receive_time(float(receive_mid_time_s) + half_tc, cfg)
    d_tau_start = round_trip_light_time_initial_state_jacobian(
        receive_start,
        station,
        t_grid_s,
        augmented_state_history_mci,
        earth_pos_mci_m,
        earth_vel_mci_mps,
        x_j2000_to_itrf93,
        cfg,
    )
    d_tau_end = round_trip_light_time_initial_state_jacobian(
        receive_end,
        station,
        t_grid_s,
        augmented_state_history_mci,
        earth_pos_mci_m,
        earth_vel_mci_mps,
        x_j2000_to_itrf93,
        cfg,
    )
    if cfg.output_unit == "hz":
        scale = cfg.turnaround_ratio * cfg.uplink_frequency_hz / cfg.count_interval_s
    else:
        scale = cfg.light_speed_mps / (2.0 * cfg.count_interval_s)
    return scale * (d_tau_end - d_tau_start)


def round_trip_light_time_initial_state_jacobian(
    receive_time_s: float,
    station,
    t_grid_s: ArrayLike,
    augmented_state_history_mci: ArrayLike,
    earth_pos_mci_m: ArrayLike,
    earth_vel_mci_mps: ArrayLike,
    x_j2000_to_itrf93: ArrayLike,
    config: RangeRatePhysicsConfig | str | None = None,
) -> np.ndarray:
    """Return d(round-trip light-time)/d(initial spacecraft state)."""
    cfg = range_rate_physics_config(config)
    x_aug = np.asarray(augmented_state_history_mci, dtype=float)
    if x_aug.ndim != 2 or x_aug.shape[1] < 42:
        raise ValueError("augmented_state_history_mci must contain state plus 6x6 STM columns.")

    solution = solve_two_way_light_time(
        receive_time_s,
        station,
        t_grid_s,
        x_aug[:, :6],
        earth_pos_mci_m,
        earth_vel_mci_mps,
        x_j2000_to_itrf93,
        cfg,
    )
    t1 = solution.transmit_time_s
    t2 = solution.transponder_time_s
    t3 = solution.receive_time_s

    station_rx_state = _station_state_mci(
        t3,
        station,
        t_grid_s,
        earth_pos_mci_m,
        earth_vel_mci_mps,
        x_j2000_to_itrf93,
    )
    station_tx_state, station_tx_slope = _station_state_mci_with_time_slope(
        t1,
        station,
        t_grid_s,
        earth_pos_mci_m,
        earth_vel_mci_mps,
        x_j2000_to_itrf93,
    )
    sc_t2_state = _interp_state(t_grid_s, x_aug[:, :6], t2)
    phi_history = np.array([row.reshape((6, 6), order="F") for row in x_aug[:, 6:]], dtype=float)
    phi_position_t2 = _interp_state_transition_position(t_grid_s, phi_history, t2)

    r2 = sc_t2_state[:3]
    v2 = sc_t2_state[3:6]
    g3 = station_rx_state[:3]
    g1 = station_tx_state[:3]
    vg1 = station_tx_slope[:3]
    a2 = phi_position_t2

    rho_down = r2 - g3
    rho_up = r2 - g1
    norm_down = float(np.linalg.norm(rho_down))
    norm_up = float(np.linalg.norm(rho_up))
    if norm_down < 1e-9 or norm_up < 1e-9:
        return np.zeros(6, dtype=float)

    u_down = rho_down / norm_down
    u_up = rho_up / norm_up
    c = cfg.light_speed_mps

    dt2_dx = -((u_down.reshape(1, 3) @ a2).reshape(6)) / (c + float(np.dot(u_down, v2)))
    dt1_denom = -1.0 + float(np.dot(u_up, vg1)) / c
    dt1_dx = (
        ((u_up.reshape(1, 3) @ a2).reshape(6) / c)
        - (1.0 - float(np.dot(u_up, v2)) / c) * dt2_dx
    ) / dt1_denom

    return -dt1_dx


def solve_two_way_light_time(
    receive_time_s: float,
    station,
    t_grid_s: ArrayLike,
    state_history_mci: ArrayLike,
    earth_pos_mci_m: ArrayLike,
    earth_vel_mci_mps: ArrayLike,
    x_j2000_to_itrf93: ArrayLike,
    config: RangeRatePhysicsConfig | str | None = None,
) -> RoundTripLightTimeSolution:
    """Solve station-spacecraft-station geometric round-trip light-time."""
    cfg = range_rate_physics_config(config)
    receive_time_s = float(receive_time_s)
    station_rx_state = _station_state_mci(
        receive_time_s,
        station,
        t_grid_s,
        earth_pos_mci_m,
        earth_vel_mci_mps,
        x_j2000_to_itrf93,
    )

    sc_rx_state = _interp_state(t_grid_s, state_history_mci, receive_time_s)
    downlink_lt = float(np.linalg.norm(sc_rx_state[:3] - station_rx_state[:3]) / cfg.light_speed_mps)
    converged = False
    iteration_count = 0
    t2 = receive_time_s - downlink_lt

    for iteration_count in range(1, cfg.light_time_max_iter + 1):
        sc_t2_state = _interp_state(t_grid_s, state_history_mci, t2)
        new_downlink_lt = float(np.linalg.norm(sc_t2_state[:3] - station_rx_state[:3]) / cfg.light_speed_mps)
        new_t2 = receive_time_s - new_downlink_lt
        if abs(new_t2 - t2) <= cfg.light_time_tolerance_s:
            t2 = new_t2
            downlink_lt = new_downlink_lt
            converged = True
            break
        t2 = new_t2
        downlink_lt = new_downlink_lt

    sc_t2_state = _interp_state(t_grid_s, state_history_mci, t2)
    t1 = t2 - cfg.transponder_delay_s - downlink_lt
    uplink_converged = False
    for uplink_iter in range(1, cfg.light_time_max_iter + 1):
        station_tx_state = _station_state_mci(
            t1,
            station,
            t_grid_s,
            earth_pos_mci_m,
            earth_vel_mci_mps,
            x_j2000_to_itrf93,
        )
        uplink_lt = float(np.linalg.norm(sc_t2_state[:3] - station_tx_state[:3]) / cfg.light_speed_mps)
        new_t1 = t2 - cfg.transponder_delay_s - uplink_lt
        if abs(new_t1 - t1) <= cfg.light_time_tolerance_s:
            t1 = new_t1
            uplink_converged = True
            break
        t1 = new_t1
    else:
        uplink_lt = float(np.linalg.norm(sc_t2_state[:3] - station_tx_state[:3]) / cfg.light_speed_mps)

    return RoundTripLightTimeSolution(
        receive_time_s=receive_time_s,
        transmit_time_s=float(t1),
        transponder_time_s=float(t2),
        round_trip_light_time_s=float(receive_time_s - t1),
        uplink_light_time_s=float(uplink_lt),
        downlink_light_time_s=float(downlink_lt),
        iterations=int(iteration_count + uplink_iter),
        converged=bool(converged and uplink_converged),
    )


def _station_state_mci(
    t_s: float,
    station,
    t_grid_s: ArrayLike,
    earth_pos_mci_m: ArrayLike,
    earth_vel_mci_mps: ArrayLike,
    x_j2000_to_itrf93: ArrayLike,
) -> np.ndarray:
    earth_state = np.concatenate(
        [
            _interp_vector(t_grid_s, earth_pos_mci_m, t_s),
            _interp_vector(t_grid_s, earth_vel_mci_mps, t_s),
        ]
    )
    xform = _interp_matrix(t_grid_s, x_j2000_to_itrf93, t_s)
    station_ecef_state = np.concatenate([np.asarray(station.r_ecef_m, dtype=float).reshape(3), np.zeros(3)])
    station_rel_j2000 = np.linalg.solve(xform, station_ecef_state)
    return earth_state + station_rel_j2000


def _clock_corrected_receive_time(t_s: float, config: RangeRatePhysicsConfig) -> float:
    return float(
        t_s
        + config.station_clock_offset_s
        + config.station_clock_drift * (t_s - config.clock_reference_time_s)
    )


def _station_state_mci_with_time_slope(
    t_s: float,
    station,
    t_grid_s: ArrayLike,
    earth_pos_mci_m: ArrayLike,
    earth_vel_mci_mps: ArrayLike,
    x_j2000_to_itrf93: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    earth_pos, earth_pos_slope = _interp_array_and_slope(t_grid_s, earth_pos_mci_m, t_s)
    earth_vel, earth_vel_slope = _interp_array_and_slope(t_grid_s, earth_vel_mci_mps, t_s)
    xform, xform_slope = _interp_array_and_slope(t_grid_s, x_j2000_to_itrf93, t_s)
    station_ecef_state = np.concatenate([np.asarray(station.r_ecef_m, dtype=float).reshape(3), np.zeros(3)])
    station_rel_j2000 = np.linalg.solve(xform, station_ecef_state)
    station_rel_slope = -np.linalg.solve(xform, xform_slope @ station_rel_j2000)
    return (
        np.concatenate([earth_pos, earth_vel]) + station_rel_j2000,
        np.concatenate([earth_pos_slope, earth_vel_slope]) + station_rel_slope,
    )


def _interp_state(t_grid_s: ArrayLike, state_history: ArrayLike, t_s: float) -> np.ndarray:
    t_grid = np.asarray(t_grid_s, dtype=float).reshape(-1)
    states = np.asarray(state_history, dtype=float)
    if states.ndim != 2 or states.shape[0] != t_grid.size or states.shape[1] < 6:
        raise ValueError("state_history must have shape (N, 6+) matching t_grid_s.")
    t = float(t_s)
    if t <= t_grid[0] or t >= t_grid[-1]:
        return _interp_vector(t_grid, states, t)

    i1 = int(np.searchsorted(t_grid, t, side="right"))
    i0 = i1 - 1
    h = float(t_grid[i1] - t_grid[i0])
    if h <= 0.0:
        raise ValueError("t_grid_s must be strictly increasing.")
    s = (t - float(t_grid[i0])) / h
    r0 = states[i0, :3]
    v0 = states[i0, 3:6]
    r1 = states[i1, :3]
    v1 = states[i1, 3:6]

    position = (
        (2.0 * s**3 - 3.0 * s**2 + 1.0) * r0
        + (s**3 - 2.0 * s**2 + s) * h * v0
        + (-2.0 * s**3 + 3.0 * s**2) * r1
        + (s**3 - s**2) * h * v1
    )
    velocity = (
        (6.0 * s**2 - 6.0 * s) * r0 / h
        + (3.0 * s**2 - 4.0 * s + 1.0) * v0
        + (-6.0 * s**2 + 6.0 * s) * r1 / h
        + (3.0 * s**2 - 2.0 * s) * v1
    )
    result = states[i0].copy()
    result[:3] = position
    result[3:6] = velocity
    return result


def interp_state_history(t_grid_s: ArrayLike, state_history: ArrayLike, t_s: float) -> np.ndarray:
    """Cubic-Hermite (position+velocity) interpolation of a 6-state history with
    linear extrapolation outside the grid. Shared helper for light-time models."""
    return _interp_state(t_grid_s, state_history, t_s)


def _interp_state_transition_position(
    t_grid_s: ArrayLike,
    phi_history: ArrayLike,
    t_s: float,
) -> np.ndarray:
    t_grid = np.asarray(t_grid_s, dtype=float).reshape(-1)
    phi = np.asarray(phi_history, dtype=float)
    if phi.shape != (t_grid.size, 6, 6):
        raise ValueError("phi_history must have shape (N, 6, 6) matching t_grid_s.")
    t = float(t_s)
    if t <= t_grid[0] or t >= t_grid[-1]:
        return _interp_matrix(t_grid, phi, t)[:3, :]

    i1 = int(np.searchsorted(t_grid, t, side="right"))
    i0 = i1 - 1
    h = float(t_grid[i1] - t_grid[i0])
    if h <= 0.0:
        raise ValueError("t_grid_s must be strictly increasing.")
    s = (t - float(t_grid[i0])) / h
    return (
        (2.0 * s**3 - 3.0 * s**2 + 1.0) * phi[i0, :3, :]
        + (s**3 - 2.0 * s**2 + s) * h * phi[i0, 3:6, :]
        + (-2.0 * s**3 + 3.0 * s**2) * phi[i1, :3, :]
        + (s**3 - s**2) * h * phi[i1, 3:6, :]
    )


def _interp_array_and_slope(t_grid_s: ArrayLike, values: ArrayLike, t_s: float) -> tuple[np.ndarray, np.ndarray]:
    t_grid = np.asarray(t_grid_s, dtype=float).reshape(-1)
    array = np.asarray(values, dtype=float)
    if array.shape[0] != t_grid.size:
        raise ValueError("values must have the same number of rows as t_grid_s.")
    if t_grid.size < 2:
        raise ValueError("At least two time samples are required for interpolation.")

    t = float(t_s)
    if t <= t_grid[0]:
        i0, i1 = 0, 1
    elif t >= t_grid[-1]:
        i0, i1 = t_grid.size - 2, t_grid.size - 1
    else:
        i1 = int(np.searchsorted(t_grid, t, side="right"))
        i0 = i1 - 1

    dt = t_grid[i1] - t_grid[i0]
    if dt == 0.0:
        raise ValueError("t_grid_s must be strictly increasing.")
    frac = (t - t_grid[i0]) / dt
    slope = (array[i1] - array[i0]) / dt
    value = array[i0] + frac * (array[i1] - array[i0])
    return value, slope


def _interp_vector(t_grid_s: ArrayLike, values: ArrayLike, t_s: float) -> np.ndarray:
    t_grid_s = np.asarray(t_grid_s, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape[0] != t_grid_s.size:
        raise ValueError("values must have the same number of rows as t_grid_s.")
    return np.array([_interp_1d_linear_extrap(t_grid_s, values[:, col], float(t_s)) for col in range(values.shape[1])])


def _interp_matrix(t_grid_s: ArrayLike, values: ArrayLike, t_s: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    original_shape = values.shape[1:]
    flat = values.reshape(values.shape[0], -1)
    return _interp_vector(t_grid_s, flat, t_s).reshape(original_shape)


def _interp_1d_linear_extrap(t_grid_s: np.ndarray, y: np.ndarray, t_s: float) -> float:
    if t_grid_s.size < 2:
        raise ValueError("At least two time samples are required for interpolation.")
    if t_s <= t_grid_s[0]:
        i0, i1 = 0, 1
    elif t_s >= t_grid_s[-1]:
        i0, i1 = t_grid_s.size - 2, t_grid_s.size - 1
    else:
        return float(np.interp(t_s, t_grid_s, y))
    dt = t_grid_s[i1] - t_grid_s[i0]
    if dt == 0.0:
        raise ValueError("t_grid_s must be strictly increasing.")
    slope = (y[i1] - y[i0]) / dt
    return float(y[i0] + slope * (t_s - t_grid_s[i0]))


def _normalize_range_rate_mode(mode: str) -> RangeRatePhysicsMode:
    normalized = mode.strip().lower().replace("-", "_")
    aliases = {
        "geometric": "geometric_instantaneous",
        "instantaneous": "geometric_instantaneous",
        "geometric_rr": "geometric_instantaneous",
        "geometric_range_rate": "geometric_instantaneous",
        "geometric_instantaneous": "geometric_instantaneous",
        "two_way": "two_way_counted_doppler",
        "twoway": "two_way_counted_doppler",
        "two_way_doppler": "two_way_counted_doppler",
        "counted_doppler": "two_way_counted_doppler",
        "two_way_counted": "two_way_counted_doppler",
        "two_way_counted_doppler": "two_way_counted_doppler",
    }
    if normalized not in aliases:
        raise ValueError("range-rate physics mode must be geometric_instantaneous or two_way_counted_doppler.")
    return aliases[normalized]  # type: ignore[return-value]
