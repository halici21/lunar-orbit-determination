"""Opt-in counted-Doppler fidelity references for the R2 campaign.

This module is deliberately outside the production measurement dispatch.  It
compares three explicitly labelled models while leaving
``lunar_od.radiometrics`` byte-for-byte unchanged:

``L``
    The legacy production single-bounce model with a linearly interpolated
    station transform grid.
``S``
    The same single-bounce event equations and spacecraft history as ``L``,
    with the station site state evaluated by exact event-epoch
    ``spice.sxform``.
``F``
    The M3 four-event chain ``t1 < t2u <= t2d < t3`` with exact event-epoch
    station states and a fixed coordinate-time transponder delay.

"Exact" in this module is intentionally limited to the station transform and
event epochs.  Spacecraft states and STMs retain the accepted cubic-Hermite
history interpolation, and the Earth-centre translation retains the legacy
linear grid interpolation so ``S - L`` isolates the station-transform effect.
The four-event solve and its implicit 3x3 sensitivity are reused directly from
``lunar_od.two_way_range`` rather than re-derived here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable, Literal

import numpy as np
from numpy.typing import ArrayLike

from .radiometrics import (
    C_LIGHT_MPS,
    DEFAULT_X_BAND_TURNAROUND_RATIO,
    DEFAULT_X_BAND_UPLINK_HZ,
    RangeRatePhysicsConfig,
    _clock_corrected_receive_time,
    _interp_state,
    _interp_state_transition_position,
    _interp_vector,
    _station_state_mci,
    range_rate_physics_config,
    two_way_counted_doppler_initial_state_jacobian,
    two_way_counted_doppler_observable,
)
from .two_way_range import (
    DEFAULT_EVENT_EQUATION_TOLERANCE_S,
    DEFAULT_EVENT_MAX_ITER,
    DEFAULT_EVENT_UPDATE_TOLERANCE_S,
    TwoWayEventConvergenceError,
    TwoWayEventHistoryError,
    TwoWayRangeConfig,
    TwoWayRangeEventSensitivity,
    TwoWayRangeEventSolution,
    solve_two_way_range_events,
    two_way_range_event_sensitivity,
    two_way_range_from_solution,
)

ReferenceOutputUnit = Literal["mps_equivalent", "hz"]

LEGACY_MODEL_ID = "L_legacy_production_single_bounce_interpolated_station"
EXACT_STATION_MODEL_ID = "S_exact_event_station_single_bounce"
FOUR_EVENT_MODEL_ID = "F_exact_event_station_four_event"
REFERENCE_STATION_STATE_METHOD = "exact_event_epoch_sxform"
REFERENCE_EARTH_EPHEMERIS_METHOD = "legacy_linear_grid_interpolation"
REFERENCE_SPACECRAFT_INTERPOLATION_METHOD = "cubic_hermite"
REFERENCE_EVENT_SOLVER_ID = "m3_nested_scalar_four_event"
REFERENCE_EVENT_SENSITIVITY_ID = "m3_scaled_implicit_3x3"


@dataclass(frozen=True)
class CountedDopplerReferenceConfig:
    """Configuration shared by the opt-in S/F counted-Doppler references."""

    count_interval_s: float = 60.0
    uplink_frequency_hz: float = DEFAULT_X_BAND_UPLINK_HZ
    turnaround_ratio: float = DEFAULT_X_BAND_TURNAROUND_RATIO
    output_unit: ReferenceOutputUnit = "mps_equivalent"
    light_speed_mps: float = C_LIGHT_MPS
    tolerance_s: float = DEFAULT_EVENT_UPDATE_TOLERANCE_S
    equation_tolerance_s: float = DEFAULT_EVENT_EQUATION_TOLERANCE_S
    max_iter: int = DEFAULT_EVENT_MAX_ITER
    station_clock_offset_s: float = 0.0
    station_clock_drift: float = 0.0
    clock_reference_time_s: float = 0.0
    transponder_delay_s: float = 0.0
    range_convention: str = "delay_calibrated_half_round_trip"

    def __post_init__(self) -> None:
        positive = {
            "count_interval_s": self.count_interval_s,
            "uplink_frequency_hz": self.uplink_frequency_hz,
            "turnaround_ratio": self.turnaround_ratio,
            "light_speed_mps": self.light_speed_mps,
            "tolerance_s": self.tolerance_s,
            "equation_tolerance_s": self.equation_tolerance_s,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.output_unit not in {"mps_equivalent", "hz"}:
            raise ValueError("output_unit must be 'mps_equivalent' or 'hz'.")
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        for name, value in {
            "station_clock_offset_s": self.station_clock_offset_s,
            "station_clock_drift": self.station_clock_drift,
            "clock_reference_time_s": self.clock_reference_time_s,
        }.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if not np.isfinite(self.transponder_delay_s) or self.transponder_delay_s < 0.0:
            raise ValueError("transponder_delay_s must be finite and non-negative.")
        # Reuse M3 validation for the range convention and all event-solver
        # scalars.  This keeps the reference contract aligned with the accepted
        # four-event implementation.
        TwoWayRangeConfig(
            transponder_delay_s=float(self.transponder_delay_s),
            convention=self.range_convention,
            light_speed_mps=float(self.light_speed_mps),
            tolerance_s=float(self.tolerance_s),
            equation_tolerance_s=float(self.equation_tolerance_s),
            max_iter=int(self.max_iter),
        )

    @classmethod
    def from_legacy(
        cls,
        config: RangeRatePhysicsConfig | str | None,
        *,
        transponder_delay_s: float = 0.0,
    ) -> "CountedDopplerReferenceConfig":
        """Map the production counted-Doppler convention into a reference config."""

        legacy = range_rate_physics_config(config)
        if legacy.mode != "two_way_counted_doppler":
            raise ValueError("Reference counted Doppler requires two_way_counted_doppler mode.")
        return cls(
            count_interval_s=float(legacy.count_interval_s),
            uplink_frequency_hz=float(legacy.uplink_frequency_hz),
            turnaround_ratio=float(legacy.turnaround_ratio),
            output_unit=legacy.output_unit,
            light_speed_mps=float(legacy.light_speed_mps),
            tolerance_s=float(legacy.light_time_tolerance_s),
            equation_tolerance_s=float(legacy.light_time_equation_tolerance_s),
            max_iter=int(legacy.light_time_max_iter),
            station_clock_offset_s=float(legacy.station_clock_offset_s),
            station_clock_drift=float(legacy.station_clock_drift),
            clock_reference_time_s=float(legacy.clock_reference_time_s),
            transponder_delay_s=float(transponder_delay_s),
        )

    def as_two_way_range_config(self) -> TwoWayRangeConfig:
        """Return the accepted M3 event-solver configuration for model F."""

        return TwoWayRangeConfig(
            transponder_delay_s=float(self.transponder_delay_s),
            convention=self.range_convention,
            light_speed_mps=float(self.light_speed_mps),
            tolerance_s=float(self.tolerance_s),
            equation_tolerance_s=float(self.equation_tolerance_s),
            max_iter=int(self.max_iter),
        )


@dataclass
class ExactEventStationStateProvider:
    """Duck-compatible M3 station provider with an exact sxform call counter."""

    state_fn: Callable[[float], np.ndarray]
    station_state_method: str = REFERENCE_STATION_STATE_METHOD
    earth_ephemeris_method: str = REFERENCE_EARTH_EPHEMERIS_METHOD
    _sxform_counter: list[int] | None = None

    def state(self, t_s: float) -> np.ndarray:
        state = np.asarray(self.state_fn(float(t_s)), dtype=float).reshape(6)
        if not np.all(np.isfinite(state)):
            raise TwoWayEventHistoryError(
                f"Reference station state is non-finite at t={float(t_s):.16g} s."
            )
        return state

    @property
    def exact_sxform_call_count(self) -> int:
        return 0 if self._sxform_counter is None else int(self._sxform_counter[0])


def make_exact_event_station_state_provider(
    station,
    et0_s: float,
    t_grid_s: ArrayLike,
    earth_pos_mci_m: ArrayLike,
    earth_vel_mci_mps: ArrayLike,
    *,
    sxform_fn: Callable[[str, str, float], ArrayLike] | None = None,
) -> ExactEventStationStateProvider:
    """Build the shared exact-event station helper used by S and F.

    The default path calls ``spice.sxform`` at every requested event epoch.
    ``sxform_fn`` exists only for deterministic SPICE-free unit fixtures; it
    must obey the same J2000-to-ITRF93 state-transform convention.

    Earth position and velocity use the same linear grid interpolation as the
    legacy counted-Doppler path.  Thus the S-L comparison changes only the
    site-state transform evaluation, not the Earth ephemeris policy.
    """

    t_grid = np.asarray(t_grid_s, dtype=float).reshape(-1)
    earth_pos = np.asarray(earth_pos_mci_m, dtype=float)
    earth_vel = np.asarray(earth_vel_mci_mps, dtype=float)
    if t_grid.size < 2 or np.any(np.diff(t_grid) <= 0.0):
        raise ValueError("t_grid_s must contain at least two strictly increasing epochs.")
    if earth_pos.shape != (t_grid.size, 3) or earth_vel.shape != (t_grid.size, 3):
        raise ValueError("earth ephemeris histories must have shape (N, 3) matching t_grid_s.")
    if not np.isfinite(et0_s):
        raise ValueError("et0_s must be finite.")

    if sxform_fn is None:
        import spiceypy as spice

        sxform_fn = spice.sxform

    station_ecef_state = np.concatenate(
        [np.asarray(station.r_ecef_m, dtype=float).reshape(3), np.zeros(3)]
    )
    counter = [0]
    et0 = float(et0_s)

    def state_fn(t_s: float) -> np.ndarray:
        t = float(t_s)
        earth_state = np.concatenate(
            [
                _interp_vector(t_grid, earth_pos, t),
                _interp_vector(t_grid, earth_vel, t),
            ]
        )
        xform = np.asarray(sxform_fn("J2000", "ITRF93", et0 + t), dtype=float)
        if xform.shape != (6, 6) or not np.all(np.isfinite(xform)):
            raise TwoWayEventHistoryError(
                f"Exact event-epoch sxform is invalid at t={t:.16g} s: shape {xform.shape}."
            )
        counter[0] += 1
        station_rel_j2000 = np.linalg.solve(xform, station_ecef_state)
        return earth_state + station_rel_j2000

    return ExactEventStationStateProvider(state_fn=state_fn, _sxform_counter=counter)


@dataclass(frozen=True)
class SingleBounceEventSolution:
    """Converged exact-station single-bounce event chain for model S."""

    t1_s: float
    t2_s: float
    t3_s: float
    uplink_light_time_s: float
    downlink_light_time_s: float
    round_trip_light_time_s: float
    uplink_iterations: int
    downlink_iterations: int
    uplink_update_residual_s: float
    downlink_update_residual_s: float
    uplink_equation_residual_s: float
    downlink_equation_residual_s: float
    converged: bool
    station_state_method: str
    earth_ephemeris_method: str
    spacecraft_state_interpolation_method: str = REFERENCE_SPACECRAFT_INTERPOLATION_METHOD

    @property
    def range_m(self) -> float:
        return 0.5 * C_LIGHT_MPS * self.round_trip_light_time_s


@dataclass(frozen=True)
class SingleBounceEventSensitivity:
    """Implicit two-event initial-state sensitivity for model S."""

    dt1_dx0: np.ndarray
    dt2_dx0: np.ndarray
    event_time_sensitivity: np.ndarray
    range_jacobian_dx0: np.ndarray
    event_matrix: np.ndarray
    event_matrix_condition_number: float


@dataclass(frozen=True)
class CountedDopplerReferenceResult:
    """One two-endpoint counted-Doppler reference result."""

    model_id: str
    observable: float
    output_unit: str
    receive_mid_time_s: float
    nominal_count_interval_s: float
    receive_start_s: float
    receive_end_s: float
    start_range_m: float
    end_range_m: float
    start_solution: SingleBounceEventSolution | TwoWayRangeEventSolution
    end_solution: SingleBounceEventSolution | TwoWayRangeEventSolution
    station_state_method: str
    earth_ephemeris_method: str
    exact_sxform_call_count: int
    endpoint_solution_count: int = 2
    physical_event_count: int = 0

    @property
    def counted_range_change_m(self) -> float:
        return float(self.end_range_m - self.start_range_m)


@dataclass(frozen=True)
class CountedDopplerReferenceJacobian:
    """Counted-Doppler initial-state row and its endpoint sensitivities."""

    model_id: str
    jacobian_dx0: np.ndarray
    start_range_jacobian_dx0: np.ndarray
    end_range_jacobian_dx0: np.ndarray
    start_sensitivity: SingleBounceEventSensitivity | TwoWayRangeEventSensitivity
    end_sensitivity: SingleBounceEventSensitivity | TwoWayRangeEventSensitivity
    exact_sxform_call_count: int
    endpoint_solution_count: int = 2
    physical_event_count: int = 0


@dataclass(frozen=True)
class CountedDopplerModelDecomposition:
    """L/S/F observable and Jacobian decomposition for one campaign row."""

    legacy_observable: float
    exact_station_observable: float
    four_event_observable: float
    frame_interpolation_error: float
    four_event_model_error: float
    total_reference_error: float
    observable_triangle_residual: float
    legacy_jacobian: np.ndarray
    exact_station_jacobian: np.ndarray
    four_event_jacobian: np.ndarray
    frame_jacobian_error: np.ndarray
    four_event_jacobian_error: np.ndarray
    total_jacobian_error: np.ndarray
    jacobian_triangle_residual: np.ndarray
    exact_station_sxform_calls: int
    four_event_sxform_calls: int


def _require_spacecraft_epoch(
    label: str,
    epoch_s: float,
    t_grid_s: np.ndarray,
    events: dict[str, float],
) -> None:
    if not np.isfinite(epoch_s):
        raise TwoWayEventHistoryError(f"Reference event {label} is not finite: {epoch_s!r}.")
    start = float(t_grid_s[0])
    end = float(t_grid_s[-1])
    if epoch_s < start or epoch_s > end:
        raise TwoWayEventHistoryError(
            f"Reference event {label}={epoch_s:.16g} s is outside spacecraft history "
            f"[{start:.16g}, {end:.16g}] s; events={events!r}."
        )


def solve_exact_station_single_bounce_events(
    receive_time_s: float,
    station_state_provider: ExactEventStationStateProvider,
    t_grid_s: ArrayLike,
    state_history_mci: ArrayLike,
    config: CountedDopplerReferenceConfig | None = None,
) -> SingleBounceEventSolution:
    """Solve the legacy two-leg/single-bounce equations with exact station states."""

    cfg = config or CountedDopplerReferenceConfig()
    if cfg.transponder_delay_s != 0.0:
        raise ValueError(
            "Model S is the zero-delay legacy single-bounce reference; nonzero "
            "transponder delay is represented only by model F."
        )
    t_grid = np.asarray(t_grid_s, dtype=float).reshape(-1)
    states = np.asarray(state_history_mci, dtype=float)
    if t_grid.size < 2 or states.ndim != 2 or states.shape[0] != t_grid.size or states.shape[1] < 6:
        raise ValueError("state_history_mci must have shape (N, 6+) matching t_grid_s.")
    t3 = float(receive_time_s)
    c = float(cfg.light_speed_mps)
    events = {"t3": t3}

    station_rx = station_state_provider.state(t3)
    _require_spacecraft_epoch("t3 initial guess", t3, t_grid, events)
    sc_rx = _interp_state(t_grid, states, t3)
    downlink_lt = float(np.linalg.norm(sc_rx[:3] - station_rx[:3]) / c)
    t2 = t3 - downlink_lt
    downlink_converged = False
    downlink_update = float("inf")
    downlink_iterations = 0

    for downlink_iterations in range(1, cfg.max_iter + 1):
        _require_spacecraft_epoch("t2", t2, t_grid, {**events, "t2": t2})
        sc_t2 = _interp_state(t_grid, states, t2)
        new_downlink_lt = float(np.linalg.norm(sc_t2[:3] - station_rx[:3]) / c)
        new_t2 = t3 - new_downlink_lt
        downlink_update = abs(new_t2 - t2)
        t2 = new_t2
        downlink_lt = new_downlink_lt
        if downlink_update <= cfg.tolerance_s:
            downlink_converged = True
            break

    _require_spacecraft_epoch("t2", t2, t_grid, {**events, "t2": t2})
    sc_t2 = _interp_state(t_grid, states, t2)
    downlink_range = float(np.linalg.norm(sc_t2[:3] - station_rx[:3]))
    downlink_equation_residual = abs(t3 - t2 - downlink_range / c)
    events["t2"] = t2

    # This initial guess and fixed-point iteration intentionally mirror the
    # legacy single-bounce path, with only the station provider changed.
    t1 = t2 - downlink_lt
    uplink_converged = False
    uplink_update = float("inf")
    uplink_iterations = 0
    uplink_lt = downlink_lt
    for uplink_iterations in range(1, cfg.max_iter + 1):
        station_tx = station_state_provider.state(t1)
        uplink_lt = float(np.linalg.norm(sc_t2[:3] - station_tx[:3]) / c)
        new_t1 = t2 - uplink_lt
        uplink_update = abs(new_t1 - t1)
        t1 = new_t1
        if uplink_update <= cfg.tolerance_s:
            uplink_converged = True
            break

    station_tx = station_state_provider.state(t1)
    uplink_range = float(np.linalg.norm(sc_t2[:3] - station_tx[:3]))
    uplink_equation_residual = abs(t2 - t1 - uplink_range / c)
    events["t1"] = t1

    converged = bool(
        downlink_converged
        and uplink_converged
        and downlink_equation_residual <= cfg.equation_tolerance_s
        and uplink_equation_residual <= cfg.equation_tolerance_s
    )
    if not converged:
        raise TwoWayEventConvergenceError(
            "Exact-station single-bounce solve failed the dual convergence "
            f"criterion at t3={t3:.16g} s: updates [downlink {downlink_update:.3e}, "
            f"uplink {uplink_update:.3e}] s; equations [downlink "
            f"{downlink_equation_residual:.3e}, uplink {uplink_equation_residual:.3e}] s."
        )
    if not (t1 < t2 < t3):
        raise TwoWayEventHistoryError(
            f"Single-bounce event ordering violated: t1={t1!r}, t2={t2!r}, t3={t3!r}."
        )

    return SingleBounceEventSolution(
        t1_s=float(t1),
        t2_s=float(t2),
        t3_s=t3,
        uplink_light_time_s=float(uplink_lt),
        downlink_light_time_s=float(downlink_lt),
        round_trip_light_time_s=float(t3 - t1),
        uplink_iterations=int(uplink_iterations),
        downlink_iterations=int(downlink_iterations),
        uplink_update_residual_s=float(uplink_update),
        downlink_update_residual_s=float(downlink_update),
        uplink_equation_residual_s=float(uplink_equation_residual),
        downlink_equation_residual_s=float(downlink_equation_residual),
        converged=True,
        station_state_method=station_state_provider.station_state_method,
        earth_ephemeris_method=station_state_provider.earth_ephemeris_method,
    )


def exact_station_single_bounce_event_sensitivity(
    solution: SingleBounceEventSolution,
    station_state_provider: ExactEventStationStateProvider,
    t_grid_s: ArrayLike,
    augmented_state_history_mci: ArrayLike,
    config: CountedDopplerReferenceConfig | None = None,
) -> SingleBounceEventSensitivity:
    """Solve the single-bounce implicit two-event sensitivity system."""

    cfg = config or CountedDopplerReferenceConfig()
    if cfg.transponder_delay_s != 0.0:
        raise ValueError("Model S sensitivity requires zero transponder delay.")
    t_grid = np.asarray(t_grid_s, dtype=float).reshape(-1)
    x_aug = np.asarray(augmented_state_history_mci, dtype=float)
    if x_aug.ndim != 2 or x_aug.shape != (t_grid.size, 42):
        raise ValueError("augmented_state_history_mci must have shape (N, 42).")
    events = {"t1": solution.t1_s, "t2": solution.t2_s, "t3": solution.t3_s}
    _require_spacecraft_epoch("t2", solution.t2_s, t_grid, events)

    states = x_aug[:, :6]
    phi_history = np.array(
        [row.reshape((6, 6), order="F") for row in x_aug[:, 6:42]], dtype=float
    )
    sc_t2 = _interp_state(t_grid, states, solution.t2_s)
    phi_r_t2 = _interp_state_transition_position(t_grid, phi_history, solution.t2_s)
    station_tx = station_state_provider.state(solution.t1_s)
    station_rx = station_state_provider.state(solution.t3_s)

    rho_u_vec = sc_t2[:3] - station_tx[:3]
    rho_d_vec = sc_t2[:3] - station_rx[:3]
    rho_u = float(np.linalg.norm(rho_u_vec))
    rho_d = float(np.linalg.norm(rho_d_vec))
    if rho_u <= 0.0 or rho_d <= 0.0:
        raise TwoWayEventConvergenceError("Single-bounce sensitivity requires positive ranges.")
    u_u = rho_u_vec / rho_u
    u_d = rho_d_vec / rho_d
    c = float(cfg.light_speed_mps)

    event_matrix = np.array(
        [
            [
                -1.0 + float(np.dot(u_u, station_tx[3:6])) / c,
                1.0 - float(np.dot(u_u, sc_t2[3:6])) / c,
            ],
            [0.0, -1.0 - float(np.dot(u_d, sc_t2[3:6])) / c],
        ],
        dtype=float,
    )
    g_x = np.vstack([-(u_u @ phi_r_t2) / c, -(u_d @ phi_r_t2) / c])
    if not np.all(np.isfinite(event_matrix)) or not np.all(np.isfinite(g_x)):
        raise TwoWayEventConvergenceError("Single-bounce event sensitivity is non-finite.")
    condition_number = float(np.linalg.cond(event_matrix))
    if not np.isfinite(condition_number):
        raise TwoWayEventConvergenceError("Single-bounce event matrix condition is non-finite.")
    try:
        event_sensitivity = np.linalg.solve(event_matrix, -g_x)
    except np.linalg.LinAlgError as exc:
        raise TwoWayEventConvergenceError("Single-bounce event matrix is singular.") from exc
    range_jacobian = -0.5 * c * event_sensitivity[0, :]
    return SingleBounceEventSensitivity(
        dt1_dx0=np.asarray(event_sensitivity[0, :], dtype=float),
        dt2_dx0=np.asarray(event_sensitivity[1, :], dtype=float),
        event_time_sensitivity=np.asarray(event_sensitivity, dtype=float),
        range_jacobian_dx0=np.asarray(range_jacobian, dtype=float),
        event_matrix=event_matrix,
        event_matrix_condition_number=condition_number,
    )


def _clock_corrected_endpoint(t_s: float, config: CountedDopplerReferenceConfig) -> float:
    legacy = RangeRatePhysicsConfig(
        mode="geometric_instantaneous",
        count_interval_s=float(config.count_interval_s),
        uplink_frequency_hz=float(config.uplink_frequency_hz),
        turnaround_ratio=float(config.turnaround_ratio),
        output_unit=config.output_unit,
        light_speed_mps=float(config.light_speed_mps),
        light_time_tolerance_s=float(config.tolerance_s),
        light_time_max_iter=int(config.max_iter),
        station_clock_offset_s=float(config.station_clock_offset_s),
        station_clock_drift=float(config.station_clock_drift),
        clock_reference_time_s=float(config.clock_reference_time_s),
        transponder_delay_s=0.0,
        light_time_equation_tolerance_s=float(config.equation_tolerance_s),
    )
    return _clock_corrected_receive_time(float(t_s), legacy)


def _count_endpoints(
    receive_mid_time_s: float,
    config: CountedDopplerReferenceConfig,
) -> tuple[float, float]:
    half = 0.5 * float(config.count_interval_s)
    start = _clock_corrected_endpoint(float(receive_mid_time_s) - half, config)
    end = _clock_corrected_endpoint(float(receive_mid_time_s) + half, config)
    if not np.isfinite(start) or not np.isfinite(end) or not start < end:
        raise ValueError(
            f"Clock-corrected count endpoints must be finite and ordered; got {start!r}, {end!r}."
        )
    return start, end


def _observable_from_range_difference(
    range_difference_m: float,
    config: CountedDopplerReferenceConfig,
) -> float:
    rate_mps = float(range_difference_m) / float(config.count_interval_s)
    if config.output_unit == "mps_equivalent":
        return rate_mps
    return float(2.0 * config.turnaround_ratio * config.uplink_frequency_hz * rate_mps / config.light_speed_mps)


def _jacobian_from_range_difference(
    range_jacobian_difference: np.ndarray,
    config: CountedDopplerReferenceConfig,
) -> np.ndarray:
    row = np.asarray(range_jacobian_difference, dtype=float) / float(config.count_interval_s)
    if config.output_unit == "mps_equivalent":
        return row
    return 2.0 * config.turnaround_ratio * config.uplink_frequency_hz * row / config.light_speed_mps


def _solve_single_bounce_counted_endpoints(
    receive_mid_time_s: float,
    station_state_provider: ExactEventStationStateProvider,
    t_grid_s: ArrayLike,
    state_history_mci: ArrayLike,
    config: CountedDopplerReferenceConfig,
) -> tuple[float, float, SingleBounceEventSolution, SingleBounceEventSolution]:
    if config.transponder_delay_s != 0.0:
        raise ValueError("Model S endpoint solver requires zero transponder delay.")
    start, end = _count_endpoints(receive_mid_time_s, config)
    start_solution = solve_exact_station_single_bounce_events(
        start, station_state_provider, t_grid_s, state_history_mci, config
    )
    end_solution = solve_exact_station_single_bounce_events(
        end, station_state_provider, t_grid_s, state_history_mci, config
    )
    return start, end, start_solution, end_solution


def _solve_four_event_counted_endpoints(
    receive_mid_time_s: float,
    station_state_provider: ExactEventStationStateProvider,
    t_grid_s: ArrayLike,
    state_history_mci: ArrayLike,
    config: CountedDopplerReferenceConfig,
) -> tuple[float, float, TwoWayRangeEventSolution, TwoWayRangeEventSolution]:
    """Shared two-endpoint/eight-event helper for F generation and evaluation."""

    start, end = _count_endpoints(receive_mid_time_s, config)
    range_config = config.as_two_way_range_config()
    start_solution = solve_two_way_range_events(
        start, station_state_provider, t_grid_s, state_history_mci, range_config
    )
    end_solution = solve_two_way_range_events(
        end, station_state_provider, t_grid_s, state_history_mci, range_config
    )
    return start, end, start_solution, end_solution


def exact_station_single_bounce_counted_doppler_reference(
    receive_mid_time_s: float,
    station_state_provider: ExactEventStationStateProvider,
    t_grid_s: ArrayLike,
    state_history_mci: ArrayLike,
    config: CountedDopplerReferenceConfig | None = None,
) -> CountedDopplerReferenceResult:
    """Evaluate model S at both count endpoints."""

    cfg = config or CountedDopplerReferenceConfig()
    count_before = station_state_provider.exact_sxform_call_count
    start, end, start_solution, end_solution = _solve_single_bounce_counted_endpoints(
        receive_mid_time_s, station_state_provider, t_grid_s, state_history_mci, cfg
    )
    start_range = 0.5 * cfg.light_speed_mps * start_solution.round_trip_light_time_s
    end_range = 0.5 * cfg.light_speed_mps * end_solution.round_trip_light_time_s
    observable = _observable_from_range_difference(end_range - start_range, cfg)
    return CountedDopplerReferenceResult(
        model_id=EXACT_STATION_MODEL_ID,
        observable=float(observable),
        output_unit=cfg.output_unit,
        receive_mid_time_s=float(receive_mid_time_s),
        nominal_count_interval_s=float(cfg.count_interval_s),
        receive_start_s=float(start),
        receive_end_s=float(end),
        start_range_m=float(start_range),
        end_range_m=float(end_range),
        start_solution=start_solution,
        end_solution=end_solution,
        station_state_method=station_state_provider.station_state_method,
        earth_ephemeris_method=station_state_provider.earth_ephemeris_method,
        exact_sxform_call_count=station_state_provider.exact_sxform_call_count - count_before,
        physical_event_count=6,
    )


def exact_station_single_bounce_counted_doppler_jacobian(
    receive_mid_time_s: float,
    station_state_provider: ExactEventStationStateProvider,
    t_grid_s: ArrayLike,
    augmented_state_history_mci: ArrayLike,
    config: CountedDopplerReferenceConfig | None = None,
) -> CountedDopplerReferenceJacobian:
    """Evaluate model S's analytic arc-initial-state Jacobian."""

    cfg = config or CountedDopplerReferenceConfig()
    x_aug = np.asarray(augmented_state_history_mci, dtype=float)
    count_before = station_state_provider.exact_sxform_call_count
    _, _, start_solution, end_solution = _solve_single_bounce_counted_endpoints(
        receive_mid_time_s, station_state_provider, t_grid_s, x_aug[:, :6], cfg
    )
    start_sensitivity = exact_station_single_bounce_event_sensitivity(
        start_solution, station_state_provider, t_grid_s, x_aug, cfg
    )
    end_sensitivity = exact_station_single_bounce_event_sensitivity(
        end_solution, station_state_provider, t_grid_s, x_aug, cfg
    )
    row = _jacobian_from_range_difference(
        end_sensitivity.range_jacobian_dx0 - start_sensitivity.range_jacobian_dx0,
        cfg,
    )
    return CountedDopplerReferenceJacobian(
        model_id=EXACT_STATION_MODEL_ID,
        jacobian_dx0=np.asarray(row, dtype=float),
        start_range_jacobian_dx0=start_sensitivity.range_jacobian_dx0,
        end_range_jacobian_dx0=end_sensitivity.range_jacobian_dx0,
        start_sensitivity=start_sensitivity,
        end_sensitivity=end_sensitivity,
        exact_sxform_call_count=station_state_provider.exact_sxform_call_count - count_before,
        physical_event_count=6,
    )


def exact_station_four_event_counted_doppler_reference(
    receive_mid_time_s: float,
    station_state_provider: ExactEventStationStateProvider,
    t_grid_s: ArrayLike,
    state_history_mci: ArrayLike,
    config: CountedDopplerReferenceConfig | None = None,
) -> CountedDopplerReferenceResult:
    """Evaluate model F as two independent four-event endpoint solves."""

    cfg = config or CountedDopplerReferenceConfig()
    count_before = station_state_provider.exact_sxform_call_count
    start, end, start_solution, end_solution = _solve_four_event_counted_endpoints(
        receive_mid_time_s, station_state_provider, t_grid_s, state_history_mci, cfg
    )
    range_config = cfg.as_two_way_range_config()
    start_range = two_way_range_from_solution(start_solution, range_config)
    end_range = two_way_range_from_solution(end_solution, range_config)
    observable = _observable_from_range_difference(end_range - start_range, cfg)
    return CountedDopplerReferenceResult(
        model_id=FOUR_EVENT_MODEL_ID,
        observable=float(observable),
        output_unit=cfg.output_unit,
        receive_mid_time_s=float(receive_mid_time_s),
        nominal_count_interval_s=float(cfg.count_interval_s),
        receive_start_s=float(start),
        receive_end_s=float(end),
        start_range_m=float(start_range),
        end_range_m=float(end_range),
        start_solution=start_solution,
        end_solution=end_solution,
        station_state_method=station_state_provider.station_state_method,
        earth_ephemeris_method=station_state_provider.earth_ephemeris_method,
        exact_sxform_call_count=station_state_provider.exact_sxform_call_count - count_before,
        physical_event_count=8,
    )


def exact_station_four_event_counted_doppler_jacobian(
    receive_mid_time_s: float,
    station_state_provider: ExactEventStationStateProvider,
    t_grid_s: ArrayLike,
    augmented_state_history_mci: ArrayLike,
    config: CountedDopplerReferenceConfig | None = None,
) -> CountedDopplerReferenceJacobian:
    """Evaluate model F using M3's implicit 3x3 endpoint sensitivities."""

    cfg = config or CountedDopplerReferenceConfig()
    x_aug = np.asarray(augmented_state_history_mci, dtype=float)
    if x_aug.ndim != 2 or x_aug.shape[1] < 42:
        raise ValueError("augmented_state_history_mci must contain state plus 6x6 STM columns.")
    count_before = station_state_provider.exact_sxform_call_count
    _, _, start_solution, end_solution = _solve_four_event_counted_endpoints(
        receive_mid_time_s, station_state_provider, t_grid_s, x_aug[:, :6], cfg
    )
    range_config = cfg.as_two_way_range_config()
    start_sensitivity = two_way_range_event_sensitivity(
        start_solution, station_state_provider, t_grid_s, x_aug, range_config
    )
    end_sensitivity = two_way_range_event_sensitivity(
        end_solution, station_state_provider, t_grid_s, x_aug, range_config
    )
    row = _jacobian_from_range_difference(
        end_sensitivity.two_way_range_jacobian_dx0
        - start_sensitivity.two_way_range_jacobian_dx0,
        cfg,
    )
    return CountedDopplerReferenceJacobian(
        model_id=FOUR_EVENT_MODEL_ID,
        jacobian_dx0=np.asarray(row, dtype=float),
        start_range_jacobian_dx0=start_sensitivity.two_way_range_jacobian_dx0,
        end_range_jacobian_dx0=end_sensitivity.two_way_range_jacobian_dx0,
        start_sensitivity=start_sensitivity,
        end_sensitivity=end_sensitivity,
        exact_sxform_call_count=station_state_provider.exact_sxform_call_count - count_before,
        physical_event_count=8,
    )


def generate_four_event_counted_doppler_reference(
    receive_mid_times_s: Iterable[float],
    station_state_provider: ExactEventStationStateProvider,
    t_grid_s: ArrayLike,
    state_history_mci: ArrayLike,
    config: CountedDopplerReferenceConfig | None = None,
    *,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, tuple[CountedDopplerReferenceResult, ...]]:
    """Generate opt-in F reference values through the same endpoint helper.

    This is not connected to production measurement dispatch.  It exists so
    generation and evaluation in the R2 campaign provably share the identical
    two-endpoint/eight-event implementation.
    """

    cfg = config or CountedDopplerReferenceConfig()
    if not np.isfinite(noise_sigma) or noise_sigma < 0.0:
        raise ValueError("noise_sigma must be finite and non-negative.")
    generator = rng or np.random.default_rng(0)
    results: list[CountedDopplerReferenceResult] = []
    rows: list[list[float]] = []
    for receive_mid in receive_mid_times_s:
        result = exact_station_four_event_counted_doppler_reference(
            float(receive_mid), station_state_provider, t_grid_s, state_history_mci, cfg
        )
        noise = float(noise_sigma * generator.standard_normal()) if noise_sigma else 0.0
        rows.append([float(receive_mid), float(result.observable + noise)])
        results.append(result)
    return np.asarray(rows, dtype=float).reshape(-1, 2), tuple(results)


def legacy_interpolated_station_state(
    t_s: float,
    station,
    t_grid_s: ArrayLike,
    earth_pos_mci_m: ArrayLike,
    earth_vel_mci_mps: ArrayLike,
    x_j2000_to_itrf93: ArrayLike,
) -> np.ndarray:
    """Reference-campaign access to model L's unchanged station-state helper."""

    return _station_state_mci(
        float(t_s),
        station,
        t_grid_s,
        earth_pos_mci_m,
        earth_vel_mci_mps,
        x_j2000_to_itrf93,
        endpoint_label="R2 station comparison",
        event_label="station-state",
        consumer="two_way_counted_doppler_reference",
    )


def evaluate_lsf_decomposition(
    receive_mid_time_s: float,
    station,
    et0_s: float,
    t_grid_s: ArrayLike,
    augmented_state_history_mci: ArrayLike,
    earth_pos_mci_m: ArrayLike,
    earth_vel_mci_mps: ArrayLike,
    x_j2000_to_itrf93: ArrayLike,
    legacy_config: RangeRatePhysicsConfig | str | None,
    *,
    sxform_fn: Callable[[str, str, float], ArrayLike] | None = None,
) -> CountedDopplerModelDecomposition:
    """Evaluate the complete L/S/F zero-delay observable/Jacobian decomposition."""

    legacy = range_rate_physics_config(legacy_config)
    if legacy.mode != "two_way_counted_doppler":
        raise ValueError("L/S/F decomposition requires two_way_counted_doppler mode.")
    if legacy.transponder_delay_s != 0.0:
        raise ValueError("Legacy model L must retain its zero-delay fail-closed contract.")
    x_aug = np.asarray(augmented_state_history_mci, dtype=float)
    if x_aug.ndim != 2 or x_aug.shape[1] < 42:
        raise ValueError("augmented_state_history_mci must contain state plus 6x6 STM columns.")
    reference = CountedDopplerReferenceConfig.from_legacy(legacy)

    legacy_observable = two_way_counted_doppler_observable(
        receive_mid_time_s,
        station,
        t_grid_s,
        x_aug[:, :6],
        earth_pos_mci_m,
        earth_vel_mci_mps,
        x_j2000_to_itrf93,
        legacy,
    )
    legacy_jacobian = two_way_counted_doppler_initial_state_jacobian(
        receive_mid_time_s,
        station,
        t_grid_s,
        x_aug,
        earth_pos_mci_m,
        earth_vel_mci_mps,
        x_j2000_to_itrf93,
        legacy,
    )

    station_s = make_exact_event_station_state_provider(
        station,
        et0_s,
        t_grid_s,
        earth_pos_mci_m,
        earth_vel_mci_mps,
        sxform_fn=sxform_fn,
    )
    s_result = exact_station_single_bounce_counted_doppler_reference(
        receive_mid_time_s, station_s, t_grid_s, x_aug[:, :6], reference
    )
    s_jacobian = exact_station_single_bounce_counted_doppler_jacobian(
        receive_mid_time_s, station_s, t_grid_s, x_aug, reference
    )

    station_f = make_exact_event_station_state_provider(
        station,
        et0_s,
        t_grid_s,
        earth_pos_mci_m,
        earth_vel_mci_mps,
        sxform_fn=sxform_fn,
    )
    f_result = exact_station_four_event_counted_doppler_reference(
        receive_mid_time_s, station_f, t_grid_s, x_aug[:, :6], reference
    )
    f_jacobian = exact_station_four_event_counted_doppler_jacobian(
        receive_mid_time_s, station_f, t_grid_s, x_aug, reference
    )

    frame_error = float(s_result.observable - legacy_observable)
    four_event_error = float(f_result.observable - s_result.observable)
    total_error = float(f_result.observable - legacy_observable)
    frame_jacobian = s_jacobian.jacobian_dx0 - legacy_jacobian
    four_event_jacobian = f_jacobian.jacobian_dx0 - s_jacobian.jacobian_dx0
    total_jacobian = f_jacobian.jacobian_dx0 - legacy_jacobian
    return CountedDopplerModelDecomposition(
        legacy_observable=float(legacy_observable),
        exact_station_observable=float(s_result.observable),
        four_event_observable=float(f_result.observable),
        frame_interpolation_error=frame_error,
        four_event_model_error=four_event_error,
        total_reference_error=total_error,
        observable_triangle_residual=float(total_error - (frame_error + four_event_error)),
        legacy_jacobian=np.asarray(legacy_jacobian, dtype=float),
        exact_station_jacobian=np.asarray(s_jacobian.jacobian_dx0, dtype=float),
        four_event_jacobian=np.asarray(f_jacobian.jacobian_dx0, dtype=float),
        frame_jacobian_error=np.asarray(frame_jacobian, dtype=float),
        four_event_jacobian_error=np.asarray(four_event_jacobian, dtype=float),
        total_jacobian_error=np.asarray(total_jacobian, dtype=float),
        jacobian_triangle_residual=np.asarray(
            total_jacobian - (frame_jacobian + four_event_jacobian), dtype=float
        ),
        exact_station_sxform_calls=int(station_s.exact_sxform_call_count),
        four_event_sxform_calls=int(station_f.exact_sxform_call_count),
    )


def reference_config_with_delay(
    config: CountedDopplerReferenceConfig,
    transponder_delay_s: float,
) -> CountedDopplerReferenceConfig:
    """Return a validated F configuration for one delay-sweep sample."""

    return replace(config, transponder_delay_s=float(transponder_delay_s))


__all__ = [
    "CountedDopplerModelDecomposition",
    "CountedDopplerReferenceConfig",
    "CountedDopplerReferenceJacobian",
    "CountedDopplerReferenceResult",
    "EXACT_STATION_MODEL_ID",
    "ExactEventStationStateProvider",
    "FOUR_EVENT_MODEL_ID",
    "LEGACY_MODEL_ID",
    "REFERENCE_EARTH_EPHEMERIS_METHOD",
    "REFERENCE_EVENT_SENSITIVITY_ID",
    "REFERENCE_EVENT_SOLVER_ID",
    "REFERENCE_SPACECRAFT_INTERPOLATION_METHOD",
    "REFERENCE_STATION_STATE_METHOD",
    "SingleBounceEventSensitivity",
    "SingleBounceEventSolution",
    "evaluate_lsf_decomposition",
    "exact_station_four_event_counted_doppler_jacobian",
    "exact_station_four_event_counted_doppler_reference",
    "exact_station_single_bounce_counted_doppler_jacobian",
    "exact_station_single_bounce_counted_doppler_reference",
    "exact_station_single_bounce_event_sensitivity",
    "generate_four_event_counted_doppler_reference",
    "legacy_interpolated_station_state",
    "make_exact_event_station_state_provider",
    "reference_config_with_delay",
    "solve_exact_station_single_bounce_events",
]
