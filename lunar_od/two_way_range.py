"""Converged two-way range observable with explicit uplink/downlink events.

This module implements the M3 two-way range measurement model.  It is
deliberately separate from :mod:`lunar_od.radiometrics` so that the existing
two-way counted-Doppler production path (``solve_two_way_light_time``) is not
modified.  The event contract is:

.. code-block:: text

    t1   station uplink transmit epoch
    t2u  spacecraft uplink receive epoch
    t2d  spacecraft downlink transmit epoch  (t2d = t2u + transponder delay)
    t3   station downlink receive epoch and measurement time tag (fixed)

with ordering ``t1 < t2u <= t2d < t3`` and the event equations, all in
seconds:

.. code-block:: text

    G_u  = t2u - t1  - |r_sc(t2u) - r_st(t1)| / c = 0
    G_d  = t3  - t2d - |r_sc(t2d) - r_st(t3)| / c = 0
    G_tr = t2d - t2u - transponder_delay          = 0

Station states are evaluated with exact event-epoch ``spice.sxform``
(production policy Option A); the Earth-center MCI ephemeris translation is a
separate cubic-Hermite interpolation of the supplied ephemeris grid and is
reported independently in metadata.  Spacecraft state and STM position blocks
are cubic-Hermite interpolated at ``t2u`` and ``t2d`` separately; event
epochs outside the propagated history raise :class:`TwoWayEventHistoryError`
instead of extrapolating.  The station uplink epoch ``t1`` lies up to one
round-trip light-time before the first receive tag; the Earth-center
ephemeris is allowed to extrapolate linearly over that pre-grid interval
because the Earth-Moon relative acceleration (~2.7e-3 m/s^2) bounds the
extrapolation error by ~0.5*a*dt^2 ~ 1 cm for dt ~ 2.7 s, far below the
configured range noise.

Initial-state sensitivities solve the uniform-unit implicit 3x3 event system

.. code-block:: text

    G_y @ dy/dx0 = -G_x,   y = [t1, t2u, t2d]

with ``np.linalg.solve`` (never an explicit inverse), and the two-way range
row is ``H = -(c/2) * dt1/dx0`` for both the raw and the delay-calibrated
half-round-trip conventions (the fixed transponder delay has zero
initial-state derivative).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from .radiometrics import (
    C_LIGHT_MPS,
    _interp_state,
    _interp_state_transition_position,
)

TWO_WAY_RANGE_CONVENTIONS = (
    "raw_half_round_trip",
    "delay_calibrated_half_round_trip",
)

DEFAULT_EVENT_UPDATE_TOLERANCE_S = 1e-12
DEFAULT_EVENT_EQUATION_TOLERANCE_S = 1e-11
DEFAULT_EVENT_MAX_ITER = 25


class TwoWayEventHistoryError(ValueError):
    """Raised when a two-way event epoch falls outside the propagated history."""


class TwoWayEventConvergenceError(RuntimeError):
    """Raised when the two-way event solve fails its convergence criteria."""


@dataclass(frozen=True)
class TwoWayRangeConfig:
    """Configuration for the converged two-way range observable."""

    transponder_delay_s: float = 0.0
    convention: str = "delay_calibrated_half_round_trip"
    light_speed_mps: float = C_LIGHT_MPS
    tolerance_s: float = DEFAULT_EVENT_UPDATE_TOLERANCE_S
    equation_tolerance_s: float = DEFAULT_EVENT_EQUATION_TOLERANCE_S
    max_iter: int = DEFAULT_EVENT_MAX_ITER

    def __post_init__(self) -> None:
        if not np.isfinite(self.transponder_delay_s) or self.transponder_delay_s < 0.0:
            raise ValueError("transponder_delay_s must be finite and non-negative.")
        if self.convention not in TWO_WAY_RANGE_CONVENTIONS:
            raise ValueError(
                f"convention must be one of {TWO_WAY_RANGE_CONVENTIONS}; got {self.convention!r}."
            )
        if self.light_speed_mps <= 0.0:
            raise ValueError("light_speed_mps must be positive.")
        if self.tolerance_s <= 0.0:
            raise ValueError("tolerance_s must be positive.")
        if self.equation_tolerance_s <= 0.0:
            raise ValueError("equation_tolerance_s must be positive.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")


def two_way_range_config(config: TwoWayRangeConfig | None) -> TwoWayRangeConfig:
    """Normalize a user-provided two-way range configuration."""
    if config is None:
        return TwoWayRangeConfig()
    if isinstance(config, TwoWayRangeConfig):
        return config
    raise TypeError("two-way range config must be None or TwoWayRangeConfig.")


@dataclass(frozen=True)
class TwoWayStationStateProvider:
    """Station MCI state source with explicit method labels for metadata.

    ``state_fn(t_s)`` must return the station state ``[r, v]`` (6,) in
    Moon-centered J2000-aligned coordinates (m, m/s) at scenario time ``t_s``.
    """

    state_fn: Callable[[float], np.ndarray]
    station_state_method: str
    earth_ephemeris_method: str

    def state(self, t_s: float) -> np.ndarray:
        return np.asarray(self.state_fn(float(t_s)), dtype=float).reshape(6)


def make_exact_sxform_station_state_provider(
    station,
    et0_s: float,
    t_grid_s: ArrayLike,
    earth_pos_mci_m: ArrayLike,
    earth_vel_mci_mps: ArrayLike,
) -> TwoWayStationStateProvider:
    """Build the production Option-A station-state provider.

    The ITRF93->J2000 site-state transformation uses exact event-epoch
    ``spice.sxform`` (no transform-grid interpolation).  The Earth-center
    Moon-relative translation is cubic-Hermite interpolated from the supplied
    position/velocity ephemeris grid; that interpolation is a separate,
    independently reported approximation.
    """
    import spiceypy as spice

    t_grid = np.asarray(t_grid_s, dtype=float).reshape(-1)
    earth_pos = np.asarray(earth_pos_mci_m, dtype=float)
    earth_vel = np.asarray(earth_vel_mci_mps, dtype=float)
    if earth_pos.shape != (t_grid.size, 3) or earth_vel.shape != (t_grid.size, 3):
        raise ValueError("earth ephemeris histories must have shape (N, 3) matching t_grid_s.")
    earth_state_history = np.hstack([earth_pos, earth_vel])
    station_ecef_state = np.concatenate(
        [np.asarray(station.r_ecef_m, dtype=float).reshape(3), np.zeros(3)]
    )
    et0 = float(et0_s)

    def state_fn(t_s: float) -> np.ndarray:
        earth_state = _interp_state(t_grid, earth_state_history, t_s)
        xform = np.asarray(spice.sxform("J2000", "ITRF93", et0 + float(t_s)), dtype=float)
        station_rel_j2000 = np.linalg.solve(xform, station_ecef_state)
        return earth_state + station_rel_j2000

    return TwoWayStationStateProvider(
        state_fn=state_fn,
        station_state_method="exact_event_epoch_sxform",
        earth_ephemeris_method="cubic_hermite_grid_interpolation",
    )


@dataclass(frozen=True)
class TwoWayRangeEventSolution:
    """Converged two-way event chain and derived range quantities."""

    t1_s: float
    t2u_s: float
    t2d_s: float
    t3_s: float

    uplink_light_time_s: float
    downlink_light_time_s: float
    round_trip_light_time_s: float
    transponder_delay_s: float

    uplink_range_m: float
    downlink_range_m: float
    raw_half_round_trip_range_m: float
    delay_calibrated_half_round_trip_range_m: float

    uplink_iterations: int
    downlink_iterations: int
    converged: bool

    uplink_equation_residual_s: float
    downlink_equation_residual_s: float
    transponder_equation_residual_s: float

    station_state_method: str
    earth_ephemeris_method: str
    spacecraft_state_interpolation_method: str = "cubic_hermite"

    @property
    def total_iterations(self) -> int:
        return self.uplink_iterations + self.downlink_iterations

    @property
    def uplink_equation_residual_m(self) -> float:
        return C_LIGHT_MPS * self.uplink_equation_residual_s

    @property
    def downlink_equation_residual_m(self) -> float:
        return C_LIGHT_MPS * self.downlink_equation_residual_s


@dataclass(frozen=True)
class TwoWayRangeEventSensitivity:
    """Initial-state sensitivities of the converged two-way event chain."""

    dt1_dx0: np.ndarray
    dt2u_dx0: np.ndarray
    dt2d_dx0: np.ndarray
    event_time_sensitivity: np.ndarray  # (3, 6): rows [t1, t2u, t2d]

    two_way_range_jacobian_dx0: np.ndarray  # (6,) m per state-unit
    scaled_event_matrix: np.ndarray  # (3, 3) uniform seconds
    event_matrix_condition_number: float

    uplink_unit_los: np.ndarray
    downlink_unit_los: np.ndarray

    transponder_delay_is_solve_for: bool = False


def _spacecraft_history_bounds(t_grid: np.ndarray) -> tuple[float, float]:
    return float(t_grid[0]), float(t_grid[-1])


def _require_spacecraft_epoch_support(
    label: str,
    epoch_s: float,
    t_grid: np.ndarray,
    events: dict,
) -> None:
    start, end = _spacecraft_history_bounds(t_grid)
    if not np.isfinite(epoch_s):
        raise TwoWayEventHistoryError(
            f"Two-way event epoch {label} is not finite: {epoch_s!r}. Events so far: {events}."
        )
    if epoch_s < start or epoch_s > end:
        pre_roll = max(0.0, start - epoch_s)
        raise TwoWayEventHistoryError(
            f"Two-way event epoch {label}={epoch_s:.9f} s lies outside the propagated "
            f"spacecraft history [{start:.9f}, {end:.9f}] s. Events: "
            f"t1={events.get('t1')!r}, t2u={events.get('t2u')!r}, "
            f"t2d={events.get('t2d')!r}, t3={events.get('t3')!r}. "
            f"Required additional pre-roll before history start: {pre_roll:.9f} s. "
            "Extend the propagation (pre-roll) or delay the first receive tag."
        )


def solve_two_way_range_events(
    receive_time_s: float,
    station_state_provider: TwoWayStationStateProvider,
    t_grid_s: ArrayLike,
    state_history_mci: ArrayLike,
    config: TwoWayRangeConfig | None = None,
) -> TwoWayRangeEventSolution:
    """Solve the receive-tagged two-way event chain t3 -> t2d -> t2u -> t1.

    The nested causal structure (fixed ``t3``, downlink fixed-point solve for
    ``t2d``, transponder relation ``t2u = t2d - delay``, uplink fixed-point
    solve for ``t1``) is used for the solve; the sensitivity is computed
    separately on the converged solution through the uniform-unit 3x3
    implicit system (see :func:`two_way_range_event_sensitivity`).
    Convergence requires both the event-time update tolerance and the
    light-time equation residual tolerance; on failure a controlled
    :class:`TwoWayEventConvergenceError` is raised instead of returning the
    last iterate.
    """
    cfg = two_way_range_config(config)
    t_grid = np.asarray(t_grid_s, dtype=float).reshape(-1)
    states = np.asarray(state_history_mci, dtype=float)
    t3 = float(receive_time_s)
    c = cfg.light_speed_mps
    events: dict = {"t3": t3}

    if not np.isfinite(t3):
        raise TwoWayEventHistoryError(f"receive_time_s must be finite; got {receive_time_s!r}.")

    station_rx_state = station_state_provider.state(t3)
    if not np.all(np.isfinite(station_rx_state)):
        raise TwoWayEventHistoryError(
            f"Station receive state at t3={t3:.9f} s is not finite: {station_rx_state!r}."
        )

    # Downlink: G_d = t3 - t2d - |r_sc(t2d) - r_st(t3)| / c = 0, solved by
    # fixed-point iteration t2d <- t3 - rho_d/c (contraction ratio ~ v/c).
    _require_spacecraft_epoch_support("t3 (downlink initial guess)", t3, t_grid, events)
    sc_state = _interp_state(t_grid, states, t3)
    downlink_lt = float(np.linalg.norm(sc_state[:3] - station_rx_state[:3]) / c)
    t2d = t3 - downlink_lt
    downlink_converged = False
    downlink_iterations = 0
    for downlink_iterations in range(1, cfg.max_iter + 1):
        _require_spacecraft_epoch_support("t2d", t2d, t_grid, {**events, "t2d": t2d})
        sc_t2d_state = _interp_state(t_grid, states, t2d)
        new_downlink_lt = float(np.linalg.norm(sc_t2d_state[:3] - station_rx_state[:3]) / c)
        new_t2d = t3 - new_downlink_lt
        update = abs(new_t2d - t2d)
        t2d = new_t2d
        downlink_lt = new_downlink_lt
        if update <= cfg.tolerance_s:
            downlink_converged = True
            break
    _require_spacecraft_epoch_support("t2d", t2d, t_grid, {**events, "t2d": t2d})
    sc_t2d_state = _interp_state(t_grid, states, t2d)
    downlink_range_m = float(np.linalg.norm(sc_t2d_state[:3] - station_rx_state[:3]))
    # Q1-F01: the residual is evaluated against the AUTHORITATIVE LOCAL light time
    # ``downlink_lt``, not against ``t3 - t2d``. Re-deriving the interval from two
    # rounded large epochs gives it the granularity of ulp(t3) rather than
    # ulp(tau) -- ~1.5e-11 s versus ~4.4e-16 s at a one-day pass-relative epoch --
    # which makes the accepted 1e-11 s tolerance unreachable beyond ~12.5 h. The
    # physical equation and its tolerance are unchanged; only the conditioning of
    # its evaluation is repaired.
    downlink_equation_residual_s = abs(downlink_lt - downlink_range_m / c)
    events["t2d"] = t2d

    # Transponder relation: t2u = t2d - delay (fixed coordinate-time delay).
    t2u = t2d - cfg.transponder_delay_s
    events["t2u"] = t2u
    _require_spacecraft_epoch_support("t2u", t2u, t_grid, events)
    sc_t2u_state = _interp_state(t_grid, states, t2u)
    # The transponder relation is a representation check: t2u is CONSTRUCTED as
    # t2d - delay, so this measures how well the epoch pair carries the delay. It
    # is already well conditioned for every frozen delay (1e-6 s >> ulp(t2d)) and
    # is deliberately left on the epoch representation, because detecting a delay
    # too small for the epochs to carry is exactly what it should do.
    transponder_equation_residual_s = abs((t2d - t2u) - cfg.transponder_delay_s)

    # Uplink: G_u = t2u - t1 - |r_sc(t2u) - r_st(t1)| / c = 0, fixed-point
    # t1 <- t2u - rho_u/c with the station state re-evaluated exactly at each
    # trial t1 (station inertial speed / c ~ 1.5e-6 keeps this contractive).
    uplink_lt = float(np.linalg.norm(sc_t2u_state[:3] - station_rx_state[:3]) / c)
    t1 = t2u - uplink_lt
    uplink_converged = False
    uplink_iterations = 0
    station_tx_state = station_rx_state
    for uplink_iterations in range(1, cfg.max_iter + 1):
        station_tx_state = station_state_provider.state(t1)
        if not np.all(np.isfinite(station_tx_state)):
            raise TwoWayEventHistoryError(
                f"Station uplink state at t1={t1:.9f} s is not finite. Events: {events}."
            )
        new_uplink_lt = float(np.linalg.norm(sc_t2u_state[:3] - station_tx_state[:3]) / c)
        new_t1 = t2u - new_uplink_lt
        update = abs(new_t1 - t1)
        t1 = new_t1
        uplink_lt = new_uplink_lt
        if update <= cfg.tolerance_s:
            uplink_converged = True
            break
    station_tx_state = station_state_provider.state(t1)
    uplink_range_m = float(np.linalg.norm(sc_t2u_state[:3] - station_tx_state[:3]))
    # Q1-F01: as for the downlink, compared against the authoritative local
    # ``uplink_lt`` rather than the re-derived ``t2u - t1``.
    uplink_equation_residual_s = abs(uplink_lt - uplink_range_m / c)
    events["t1"] = t1

    converged = (
        downlink_converged
        and uplink_converged
        and downlink_equation_residual_s <= cfg.equation_tolerance_s
        and uplink_equation_residual_s <= cfg.equation_tolerance_s
        and transponder_equation_residual_s <= cfg.equation_tolerance_s
    )
    if not converged:
        raise TwoWayEventConvergenceError(
            "Two-way event solve failed convergence at receive time "
            f"t3={t3:.9f} s: downlink update converged={downlink_converged} "
            f"({downlink_iterations} it), uplink update converged={uplink_converged} "
            f"({uplink_iterations} it), equation residuals "
            f"[uplink {uplink_equation_residual_s:.3e} s / "
            f"{c * uplink_equation_residual_s:.3e} m, "
            f"downlink {downlink_equation_residual_s:.3e} s / "
            f"{c * downlink_equation_residual_s:.3e} m, "
            f"transponder {transponder_equation_residual_s:.3e} s] versus "
            f"equation tolerance {cfg.equation_tolerance_s:.3e} s. "
            f"Last events: t1={t1!r}, t2u={t2u!r}, t2d={t2d!r}, t3={t3!r}."
        )

    # Event ordering and light-time domain validation.
    if not (t1 < t2u <= t2d < t3):
        raise TwoWayEventHistoryError(
            f"Two-way event ordering violated: t1={t1!r}, t2u={t2u!r}, "
            f"t2d={t2d!r}, t3={t3!r} (expected t1 < t2u <= t2d < t3)."
        )
    if uplink_lt <= 0.0 or downlink_lt <= 0.0:
        raise TwoWayEventHistoryError(
            f"Two-way light times must be positive: uplink {uplink_lt!r} s, "
            f"downlink {downlink_lt!r} s."
        )

    # Q1-F01: the physical round-trip light time is assembled from the three
    # well-conditioned LOCAL intervals rather than from ``t3 - t1``. The two
    # forms are mathematically identical, but ``t3 - t1`` inherits ulp(t3), and
    # the counted-Doppler observable amplifies that by c/(2*Tc) into a floor of
    # c*ulp(t)/(2*Tc) -- measured at 2.18e-4 m/s for Tc = 10 s at a one-day
    # epoch, which swamps the delay physics it is meant to resolve.
    round_trip_light_time_s = downlink_lt + cfg.transponder_delay_s + uplink_lt
    raw_range_m = 0.5 * c * round_trip_light_time_s
    calibrated_range_m = 0.5 * c * (round_trip_light_time_s - cfg.transponder_delay_s)

    return TwoWayRangeEventSolution(
        t1_s=float(t1),
        t2u_s=float(t2u),
        t2d_s=float(t2d),
        t3_s=t3,
        uplink_light_time_s=float(uplink_lt),
        downlink_light_time_s=float(downlink_lt),
        round_trip_light_time_s=float(round_trip_light_time_s),
        transponder_delay_s=float(cfg.transponder_delay_s),
        uplink_range_m=uplink_range_m,
        downlink_range_m=downlink_range_m,
        raw_half_round_trip_range_m=float(raw_range_m),
        delay_calibrated_half_round_trip_range_m=float(calibrated_range_m),
        uplink_iterations=int(uplink_iterations),
        downlink_iterations=int(downlink_iterations),
        converged=True,
        uplink_equation_residual_s=float(uplink_equation_residual_s),
        downlink_equation_residual_s=float(downlink_equation_residual_s),
        transponder_equation_residual_s=float(transponder_equation_residual_s),
        station_state_method=station_state_provider.station_state_method,
        earth_ephemeris_method=station_state_provider.earth_ephemeris_method,
    )


def two_way_range_from_solution(
    solution: TwoWayRangeEventSolution,
    config: TwoWayRangeConfig | None = None,
) -> float:
    """Return the configured production two-way range convention in metres."""
    cfg = two_way_range_config(config)
    if cfg.convention == "raw_half_round_trip":
        return solution.raw_half_round_trip_range_m
    return solution.delay_calibrated_half_round_trip_range_m


def two_way_range_event_sensitivity(
    solution: TwoWayRangeEventSolution,
    station_state_provider: TwoWayStationStateProvider,
    t_grid_s: ArrayLike,
    augmented_state_history_mci: ArrayLike,
    config: TwoWayRangeConfig | None = None,
) -> TwoWayRangeEventSensitivity:
    """Solve the uniform-unit implicit 3x3 event system on a converged solution.

    With ``y = [t1, t2u, t2d]`` and the event equations G_u, G_d, G_tr in
    seconds, the implicit-function theorem gives ``dy/dx0 =
    solve(G_y, -G_x)``.  The station and spacecraft velocities enter through
    the light-time equation time-derivatives; the fixed transponder delay has
    zero initial-state partial.  The returned two-way range row
    ``-(c/2) dt1/dx0`` is already an arc-initial-state Jacobian: the STM is
    embedded through the interpolated position-sensitivity blocks and must
    not be applied again downstream.
    """
    cfg = two_way_range_config(config)
    t_grid = np.asarray(t_grid_s, dtype=float).reshape(-1)
    x_aug = np.asarray(augmented_state_history_mci, dtype=float)
    if x_aug.ndim != 2 or x_aug.shape[1] < 42:
        raise ValueError("augmented_state_history_mci must contain state plus 6x6 STM columns.")
    c = cfg.light_speed_mps

    events = {
        "t1": solution.t1_s,
        "t2u": solution.t2u_s,
        "t2d": solution.t2d_s,
        "t3": solution.t3_s,
    }
    _require_spacecraft_epoch_support("t2u", solution.t2u_s, t_grid, events)
    _require_spacecraft_epoch_support("t2d", solution.t2d_s, t_grid, events)

    states = x_aug[:, :6]
    phi_history = np.array([row.reshape((6, 6), order="F") for row in x_aug[:, 6:42]], dtype=float)

    sc_t2u = _interp_state(t_grid, states, solution.t2u_s)
    sc_t2d = _interp_state(t_grid, states, solution.t2d_s)
    phi_r_t2u = _interp_state_transition_position(t_grid, phi_history, solution.t2u_s)
    phi_r_t2d = _interp_state_transition_position(t_grid, phi_history, solution.t2d_s)

    station_tx = station_state_provider.state(solution.t1_s)
    station_rx = station_state_provider.state(solution.t3_s)

    rho_u_vec = sc_t2u[:3] - station_tx[:3]
    rho_d_vec = sc_t2d[:3] - station_rx[:3]
    rho_u = float(np.linalg.norm(rho_u_vec))
    rho_d = float(np.linalg.norm(rho_d_vec))
    if rho_u <= 0.0 or rho_d <= 0.0:
        raise TwoWayEventConvergenceError(
            "Two-way event sensitivity requires positive uplink/downlink ranges; "
            f"got uplink {rho_u!r} m, downlink {rho_d!r} m."
        )
    u_u = rho_u_vec / rho_u
    u_d = rho_d_vec / rho_d

    v_st_t1 = station_tx[3:6]
    v_sc_t2u = sc_t2u[3:6]
    v_sc_t2d = sc_t2d[3:6]

    # Uniform-unit event matrix G_y (rows: G_u, G_d, G_tr; cols: t1, t2u, t2d).
    g_y = np.array(
        [
            [-1.0 + float(np.dot(u_u, v_st_t1)) / c, 1.0 - float(np.dot(u_u, v_sc_t2u)) / c, 0.0],
            [0.0, 0.0, -1.0 - float(np.dot(u_d, v_sc_t2d)) / c],
            [0.0, -1.0, 1.0],
        ],
        dtype=float,
    )
    g_x = np.zeros((3, 6), dtype=float)
    g_x[0, :] = -(u_u @ phi_r_t2u) / c
    g_x[1, :] = -(u_d @ phi_r_t2d) / c
    # G_tr has zero initial-state partial for a fixed transponder delay.

    if not np.all(np.isfinite(g_y)):
        raise TwoWayEventConvergenceError(
            f"Two-way event matrix contains non-finite entries: {g_y!r}."
        )
    condition_number = float(np.linalg.cond(g_y))
    try:
        dy_dx0 = np.linalg.solve(g_y, -g_x)
    except np.linalg.LinAlgError as exc:
        raise TwoWayEventConvergenceError(
            f"Two-way event matrix is singular (condition number {condition_number:.3e})."
        ) from exc

    dt1_dx0 = dy_dx0[0, :]
    range_jacobian = -0.5 * c * dt1_dx0

    return TwoWayRangeEventSensitivity(
        dt1_dx0=np.asarray(dt1_dx0, dtype=float),
        dt2u_dx0=np.asarray(dy_dx0[1, :], dtype=float),
        dt2d_dx0=np.asarray(dy_dx0[2, :], dtype=float),
        event_time_sensitivity=np.asarray(dy_dx0, dtype=float),
        two_way_range_jacobian_dx0=np.asarray(range_jacobian, dtype=float),
        scaled_event_matrix=g_y,
        event_matrix_condition_number=condition_number,
        uplink_unit_los=np.asarray(u_u, dtype=float),
        downlink_unit_los=np.asarray(u_d, dtype=float),
    )


def two_way_range_measurement_metadata(
    config: TwoWayRangeConfig,
    *,
    noise_enabled: bool | None = None,
    noise_seed: int | None = None,
    dropped_measurements: int = 0,
    stations=(),
) -> dict:
    """Build traceable metadata recording the actual implemented M3 physics."""
    station_sigmas = [
        {
            "name": getattr(station, "name", None),
            "sigma_range_m": float(getattr(station, "sigma_range_m", np.nan)),
        }
        for station in stations
    ]
    return {
        "measurement_type": "two_way_range",
        "measurement_time_tag": "receive",
        "measurement_model_profile": "two_way_light_time",
        "two_way_event_model": "converged_uplink_downlink",
        "two_way_event_solver": "nested_scalar",
        "two_way_event_sensitivity": "scaled_implicit_3x3",
        "two_way_range_convention": config.convention,
        "transponder_delay_model": "fixed_coordinate_time",
        "transponder_delay_s": float(config.transponder_delay_s),
        "station_uplink_epoch": "t1",
        "spacecraft_uplink_receive_epoch": "t2u",
        "spacecraft_downlink_transmit_epoch": "t2d",
        "station_downlink_receive_epoch": "t3",
        "station_fixed_frame": "ITRF93",
        "station_inertial_axes": "J2000",
        "station_reference_center": "MOON",
        "station_state_evaluation": "exact_event_epoch_sxform",
        "earth_center_ephemeris_interpolation": "cubic_hermite_grid_interpolation",
        "spacecraft_state_interpolation": "cubic_hermite",
        "stm_interpolation": "cubic_hermite",
        "two_way_range_jacobian_model": "scaled_implicit_event_system",
        "jacobian_state_reference": "arc_initial_state",
        "stm_application_count": "one",
        "two_way_range_noise_source": "station_sigma_range_m",
        "ukf_supported": False,
        "event_update_tolerance_s": float(config.tolerance_s),
        "event_equation_tolerance_s": float(config.equation_tolerance_s),
        "event_max_iterations": int(config.max_iter),
        "dropped_measurements": int(dropped_measurements),
        "dropped_measurement_reason": (
            "two-way event chain outside propagated history (insufficient pre-roll)"
            if dropped_measurements
            else None
        ),
        "noise_enabled": noise_enabled,
        "noise_seed": noise_seed,
        "station_sigmas": station_sigmas,
    }


def _station_state_providers_for_pass(pass_geo) -> list[TwoWayStationStateProvider]:
    """Build one exact-sxform provider per station from a two-way pass geometry."""
    if pass_geo.et0_s is None:
        raise ValueError(
            "two_way_range pass geometry must carry et0_s (SPICE ET of scenario "
            "t=0) so station states can use exact event-epoch sxform."
        )
    return [
        make_exact_sxform_station_state_provider(
            station,
            float(pass_geo.et0_s),
            pass_geo.t_s,
            pass_geo.earth_pos_mci_m,
            pass_geo.earth_vel_mci_mps,
        )
        for station in pass_geo.stations
    ]


def generate_two_way_range_measurements(
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
    config: TwoWayRangeConfig | None = None,
    noise_seed: int | None = None,
):
    """Generate receive-tagged two-way range measurements.

    Observation rows are ``[t3, range_m, station_id_1based, time_index_1based
    (, arc_id)]``.  Receive tags whose event chain would require spacecraft
    history before the propagated grid are dropped (not extrapolated); the
    dropped count and reason are recorded in the pass metadata.
    """
    from .measurements import PassGeometry

    import spiceypy as spice

    cfg = two_way_range_config(config)
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

    r_earth_mci = np.asarray(get_earth_pos(t_pass_s), dtype=float).reshape(n_steps, 3)
    v_earth_mci = np.asarray(get_earth_vel(t_pass_s), dtype=float).reshape(n_steps, 3)
    xforms = np.zeros((n_steps, 6, 6), dtype=float)
    for k, t_s in enumerate(t_pass_s):
        xforms[k, :, :] = np.asarray(
            spice.sxform("J2000", "ITRF93", float(et0 + t_s)), dtype=float
        )

    providers = [
        make_exact_sxform_station_state_provider(
            station, float(et0), t_pass_s, r_earth_mci, v_earth_mci
        )
        for station in stations
    ]

    include_arc_id = arc_id is not None
    num_cols = 5 if include_arc_id else 4
    total = int(np.sum(vis_mask_raw))
    obs_data = np.zeros((total, num_cols), dtype=float)
    rng = rng or np.random.default_rng()

    obs_counter = 0
    dropped = 0
    for k, t_s in enumerate(t_pass_s):
        for station_col in np.where(vis_mask_raw[k, :])[0]:
            station = stations[station_col]
            try:
                solution = solve_two_way_range_events(
                    float(t_s), providers[station_col], t_pass_s, state_history_mci, cfg
                )
            except TwoWayEventHistoryError:
                dropped += 1
                continue
            range_ideal = two_way_range_from_solution(solution, cfg)
            noise_m = station.sigma_range_m * rng.standard_normal() if noise else 0.0
            row = [float(t_s), range_ideal + noise_m, station_col + 1, k + 1]
            if include_arc_id:
                row.append(float(arc_id))
            obs_data[obs_counter, :] = row
            obs_counter += 1

    metadata = two_way_range_measurement_metadata(
        cfg,
        noise_enabled=noise,
        noise_seed=noise_seed,
        dropped_measurements=dropped,
        stations=stations,
    )
    pass_geo = PassGeometry(
        t_s=t_pass_s,
        earth_pos_mci_m=r_earth_mci,
        earth_vel_mci_mps=v_earth_mci,
        x_j2000_to_itrf93=xforms,
        stations=tuple(stations),
        measurement_type="two_way_range",
        measurement_model_profile="two_way_light_time",
        jacobian_model="implicit_light_time",
        measurement_metadata=metadata,
        et0_s=float(et0),
        two_way_range=cfg,
    )
    return obs_data[:obs_counter, :], pass_geo


def compute_two_way_range_residuals(
    state_history_mci: ArrayLike,
    obs_data: ArrayLike,
    pass_geo,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute two-way range observed-minus-computed residuals in metres.

    Nonconvergent or out-of-history event chains raise their controlled
    errors; measurements are never silently skipped during estimation.
    """
    state_history_mci = np.asarray(state_history_mci, dtype=float)
    obs_data = np.asarray(obs_data, dtype=float)
    cfg = two_way_range_config(pass_geo.two_way_range)
    providers = _station_state_providers_for_pass(pass_geo)

    n_obs = obs_data.shape[0]
    h_meas = np.zeros(n_obs, dtype=float)
    for obs_idx in range(n_obs):
        station_id = int(obs_data[obs_idx, 2]) - 1
        solution = solve_two_way_range_events(
            float(obs_data[obs_idx, 0]),
            providers[station_id],
            pass_geo.t_s,
            state_history_mci,
            cfg,
        )
        h_meas[obs_idx] = two_way_range_from_solution(solution, cfg)

    residuals = obs_data[:, 1] - h_meas
    residuals[np.abs(residuals) < 1e-7] = 0.0  # suppress synthetic-closure roundoff
    return residuals, h_meas


def two_way_range_nominal_and_initial_jacobian(
    obs_data: ArrayLike,
    pass_geo,
    augmented_state_history_mci: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nominal two-way ranges (N,) and the initial-state Jacobian (N, 6).

    Each observation's event chain is solved once on the current nominal
    trajectory and the same converged solution feeds the implicit 3x3
    sensitivity, so the residual and the Jacobian share one linearization
    point.  The returned rows are with respect to the arc initial state; the
    STM must not be applied again.
    """
    obs_data = np.asarray(obs_data, dtype=float)
    x_aug = np.asarray(augmented_state_history_mci, dtype=float)
    cfg = two_way_range_config(pass_geo.two_way_range)
    providers = _station_state_providers_for_pass(pass_geo)

    n_obs = obs_data.shape[0]
    h_nominal = np.zeros(n_obs, dtype=float)
    h_initial = np.zeros((n_obs, 6), dtype=float)
    for obs_idx in range(n_obs):
        station_id = int(obs_data[obs_idx, 2]) - 1
        provider = providers[station_id]
        solution = solve_two_way_range_events(
            float(obs_data[obs_idx, 0]), provider, pass_geo.t_s, x_aug[:, :6], cfg
        )
        h_nominal[obs_idx] = two_way_range_from_solution(solution, cfg)
        sensitivity = two_way_range_event_sensitivity(
            solution, provider, pass_geo.t_s, x_aug, cfg
        )
        h_initial[obs_idx, :] = sensitivity.two_way_range_jacobian_dx0
    return h_nominal, h_initial
