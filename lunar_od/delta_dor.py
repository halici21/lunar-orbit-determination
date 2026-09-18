"""Production OD-level ΔDOR (Delta Differential One-way Ranging) observable.

Phase 17-R1O-D.  This module implements the calibrated, OD-level ΔDOR
observable consumed by orbit determination: a spacecraft interferometric
differential delay minus a quasar (far-field calibrator) differential delay,
both referred to a common baseline pair.  It does NOT implement raw RF
sampling, VLBI correlation, PN-code generation, or antenna receiver DSP —
those belong to the mission's signal chain, not to an OD framework (R1O-D
§14).

## Governing physics (Moyer 2000, DSN 810-005 formulation; Curkendall & Border
2013, IPN PR 42-193)

Two ground stations A, B receive the SAME spacecraft-transmitted wavefront.
The spacecraft interferometric differential delay is

    D_S = tau_{S,B} - tau_{S,A}

where tau_{S,X} is the one-way light time from the common transmit event to
station X.  A far-field quasar at inertial direction s_hat provides the same
differential quantity for a plane wave,

    D_Q = tau_{Q,B} - tau_{Q,A} = -(r_B - r_A) . s_hat / c

and the calibrated ΔDOR observable is the difference

    DDOR = D_S - D_Q        (seconds)

This module works entirely in LOCAL delay variables (never differencing two
large absolute epochs), per the Phase 17C numerical lesson: R3/R4's original
`t3 - t1` formulation lost 4574x of precision to subtractive cancellation
before that repair, and every event-time quantity here is carried as a light
time or a small event-time offset rather than reconstructed from raw epoch
subtraction.

## Same-transmit-event solver

The observation is time-tagged by RECEPTION AT STATION A: a fixed constant
``T_obs``.  ``t_tx`` (the common transmit event) is solved BACKWARD from
``T_obs`` at station A -- this reuses the existing, qualified
``solve_one_way_light_time`` unchanged.  ``t_B`` (station B's reception of
THE SAME wavefront) is then solved FORWARD from the now-fixed ``t_tx`` --a
new fixed-point iteration, since the existing one-way solver is anchored at
the receive end and this leg is anchored at the transmit end.  Both stations
are therefore provably tied to one shared spacecraft transmit event
(R1O-D §17); ``D_S = t_B - T_obs`` follows by construction (see the module
docstring's algebra: ``tau_{S,A} = T_obs - t_tx``, ``tau_{S,B} = t_B - t_tx``,
so ``D_S = tau_{S,B} - tau_{S,A} = t_B - T_obs``).

## State and K Jacobians

Both solved event times depend on the arc-initial state and (through the
already-qualified trajectory sensitivity) on K_SRP.  The Jacobian is built
with the SAME implicit-event-matrix construction `two_way_range_event_
sensitivity` already uses for the two-way range chain: a small event matrix
``G_y`` (partial derivatives of the light-time equations w.r.t. the SOLVED
event times) and ``G_x`` (partials w.r.t. the state, event times held fixed
at their converged values), giving ``dy/dx0 = solve(G_y, -G_x)`` by the
implicit function theorem. The K column is obtained by the identical linear
solve with ``Phi_r`` replaced by the trajectory's K-sensitivity column
``S_K[:3]`` -- exactly the same substitution `_two_way_range_k_srp_column`
uses for two-way range, confirming ``DIRECT_MEASUREMENT_K_DEPENDENCE = NO``
by construction: nothing in ``G_x`` or the quasar term depends on K except
through the spacecraft position at the transmit event.

## What this module does NOT model (R1O-D §22/§27, explicitly deferred)

Quasar catalog frame is a fixed J2000 unit vector (a synthetic approximation
to the ICRF radio-source catalog frame, not a rigorous frame transformation).
No media (troposphere/ionosphere/solar-plasma), clock, or instrumental delay
calibration model is implemented; `MEDIA_CALIBRATION_PHYSICS =
CHARACTERIZED_NOT_FULLY_IMPLEMENTED` and residual error is injected only as
an explicit, labeled synthetic term in the analysis harness, never fabricated
inside this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .radiometrics import _interp_state, _interp_state_transition_position
from .measurements import (
    C_LIGHT_MPS,
    ONE_WAY_LIGHT_TIME_EQUATION_TOLERANCE_S,
    ONE_WAY_LIGHT_TIME_MAX_ITERATIONS,
    ONE_WAY_LIGHT_TIME_TOLERANCE_S,
    LightTimeSolution,
    solve_one_way_light_time,
)


class DeltaDorEventError(RuntimeError):
    """The common-transmit-event / forward light-time solve did not converge."""


@dataclass(frozen=True)
class QuasarDirection:
    """A far-field calibrator direction, as a fixed inertial (J2000-like) unit vector.

    R1O-D §21/§22: the quasar is represented as a plane-wave source at a
    catalog direction, NOT as a finite Cartesian object. The frame is a
    SYNTHETIC APPROXIMATION to the ICRF radio-source catalog frame -- no
    frame-bias rotation between J2000 mean-equator-and-equinox and ICRF is
    applied (their frame difference is sub-milliarcsecond and would need its
    own qualification if ever load-bearing; see the R1O-D report §22).
    """

    ra_rad: float
    dec_rad: float

    @property
    def unit_vector_j2000(self) -> np.ndarray:
        cos_dec = float(np.cos(self.dec_rad))
        return np.array([
            cos_dec * np.cos(self.ra_rad),
            cos_dec * np.sin(self.ra_rad),
            np.sin(self.dec_rad),
        ], dtype=float)

    @classmethod
    def from_unit_vector(cls, unit_vector: ArrayLike) -> "QuasarDirection":
        v = np.asarray(unit_vector, dtype=float).reshape(3)
        norm = float(np.linalg.norm(v))
        if norm <= 0.0:
            raise ValueError("Quasar direction unit vector must be nonzero.")
        v = v / norm
        dec = float(np.arcsin(np.clip(v[2], -1.0, 1.0)))
        ra = float(np.arctan2(v[1], v[0]))
        return cls(ra_rad=ra, dec_rad=dec)


@dataclass(frozen=True)
class CommonTransmitEventSolution:
    """The solved (t_tx, t_B) pair for one DDOR observation."""

    t_obs_s: float                 # fixed: station A receive time (the tag)
    transmit_time_s: float         # solved backward from t_obs at station A
    station_b_receive_time_s: float  # solved forward from transmit_time_s
    spacecraft_differential_delay_s: float  # D_S = t_B - T_obs
    backward_converged: bool
    forward_converged: bool
    forward_iterations: int
    backward_equation_residual_s: float
    forward_equation_residual_s: float
    station_a_range_m: float
    station_b_range_m: float


def solve_forward_one_way_light_time(
    transmit_time_s: float,
    spacecraft_position_tx_m: ArrayLike,
    get_station_position_m,
    *,
    light_speed_mps: float = C_LIGHT_MPS,
    tolerance_s: float = ONE_WAY_LIGHT_TIME_TOLERANCE_S,
    max_iter: int = ONE_WAY_LIGHT_TIME_MAX_ITERATIONS,
    equation_tolerance_s: float = ONE_WAY_LIGHT_TIME_EQUATION_TOLERANCE_S,
) -> LightTimeSolution:
    """Iterate one-way light time FORWARD from a fixed transmit event.

    Mirrors `solve_one_way_light_time` (the existing, qualified backward
    solver) exactly in numerical structure -- same dual convergence
    criterion (update tolerance + independently re-evaluated equation
    residual, FA-03A), same fixed-point contraction -- but with the roles of
    transmit and receive reversed: here the SPACECRAFT position is frozen
    (it is a `float` position, not a callback) and the STATION position is
    looked up at the moving iterate, because the observable's precision-
    sensitive quantity is `t_B - T_obs`, a LOCAL delay, and every iterate
    below stays in that local-delay domain rather than differencing two
    absolute epochs (the Phase 17C lesson, R1O-D §19).
    """
    spacecraft_position_tx_m = np.asarray(spacecraft_position_tx_m, dtype=float).reshape(3)
    if light_speed_mps <= 0.0:
        raise ValueError("light_speed_mps must be positive.")
    if tolerance_s <= 0.0:
        raise ValueError("tolerance_s must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if not np.isfinite(equation_tolerance_s) or equation_tolerance_s <= 0.0:
        raise ValueError("equation_tolerance_s must be finite and positive.")

    transmit_time_s = float(transmit_time_s)
    station_position = np.asarray(
        get_station_position_m(transmit_time_s), dtype=float
    ).reshape(3)
    light_time_s = float(
        np.linalg.norm(station_position - spacecraft_position_tx_m) / light_speed_mps
    )
    update_converged = False
    update_residual_s = float("inf")

    for iteration in range(1, max_iter + 1):
        receive_time_s = transmit_time_s + light_time_s
        station_position = np.asarray(
            get_station_position_m(receive_time_s), dtype=float
        ).reshape(3)
        new_light_time_s = float(
            np.linalg.norm(station_position - spacecraft_position_tx_m) / light_speed_mps
        )
        update_residual_s = abs(new_light_time_s - light_time_s)
        if update_residual_s <= tolerance_s:
            light_time_s = new_light_time_s
            update_converged = True
            break
        light_time_s = new_light_time_s
    else:
        iteration = max_iter

    receive_time_s = transmit_time_s + light_time_s
    range_m = light_time_s * light_speed_mps
    final_station_position = np.asarray(
        get_station_position_m(receive_time_s), dtype=float
    ).reshape(3)
    equation_residual_s = abs(
        light_time_s
        - float(
            np.linalg.norm(final_station_position - spacecraft_position_tx_m)
            / light_speed_mps
        )
    )
    converged = update_converged and equation_residual_s <= equation_tolerance_s
    return LightTimeSolution(
        range_m=range_m,
        light_time_s=light_time_s,
        transmit_time_s=transmit_time_s,
        iterations=iteration,
        converged=converged,
        target_position_m=final_station_position,
        equation_residual_s=equation_residual_s,
        update_converged=update_converged,
        update_residual_s=float(update_residual_s),
    )


def solve_common_transmit_event(
    t_obs_s: float,
    station_a_position_at_t_obs_m: ArrayLike,
    get_station_b_position_m,
    get_spacecraft_position_m,
    *,
    light_speed_mps: float = C_LIGHT_MPS,
    tolerance_s: float = ONE_WAY_LIGHT_TIME_TOLERANCE_S,
    max_iter: int = ONE_WAY_LIGHT_TIME_MAX_ITERATIONS,
    equation_tolerance_s: float = ONE_WAY_LIGHT_TIME_EQUATION_TOLERANCE_S,
) -> CommonTransmitEventSolution:
    """Solve (t_tx, t_B) for one DDOR observation tagged by t_obs at station A.

    R1O-D §17 hard requirement: both stations are tied to the identical
    spacecraft transmit event.  Step 1 solves t_tx backward from the FIXED
    t_obs at station A (station A's position frozen at t_obs -- exactly the
    existing qualified `solve_one_way_light_time`).  Step 2 solves t_B
    forward from that converged, now-fixed t_tx (spacecraft position frozen
    at t_tx) using `solve_forward_one_way_light_time` above.  Station A's
    receive event is never re-solved once t_obs is fixed; station B's receive
    event is never anchored to anything but the single t_tx from step 1.
    """
    t_obs_s = float(t_obs_s)
    station_a_pos = np.asarray(station_a_position_at_t_obs_m, dtype=float).reshape(3)

    backward = solve_one_way_light_time(
        t_obs_s,
        station_a_pos,
        get_spacecraft_position_m,
        light_speed_mps=light_speed_mps,
        tolerance_s=tolerance_s,
        max_iter=max_iter,
        equation_tolerance_s=equation_tolerance_s,
    )
    if not backward.converged:
        raise DeltaDorEventError(
            "Backward (station A) light-time solve did not converge: "
            f"update_residual_s={backward.update_residual_s:.3e}, "
            f"equation_residual_s={backward.equation_residual_s:.3e}."
        )
    t_tx = backward.transmit_time_s
    spacecraft_position_tx = backward.target_position_m

    forward = solve_forward_one_way_light_time(
        t_tx,
        spacecraft_position_tx,
        get_station_b_position_m,
        light_speed_mps=light_speed_mps,
        tolerance_s=tolerance_s,
        max_iter=max_iter,
        equation_tolerance_s=equation_tolerance_s,
    )
    if not forward.converged:
        raise DeltaDorEventError(
            "Forward (station B) light-time solve did not converge: "
            f"update_residual_s={forward.update_residual_s:.3e}, "
            f"equation_residual_s={forward.equation_residual_s:.3e}."
        )
    # forward.transmit_time_s IS t_tx (the anchor passed in); t_B itself is
    # only ever formed as t_tx + light_time_B inside the forward solve.
    t_b = forward.transmit_time_s + forward.light_time_s

    # D_S = tau_{S,B} - tau_{S,A} = light_time_B - light_time_A.
    #
    # An earlier version of this function computed D_S as `t_b - t_obs_s`,
    # reasoning that both operands sit within about one baseline light-time
    # (~tens of ms) of each other and so the subtraction "costs no
    # precision, the way t3-t1 did in Phase 17C". That reasoning was wrong:
    # t_b is ITSELF constructed as `t_tx + light_time_B`, and t_tx already
    # carries the full magnitude of T_obs. Once T_obs is large, forming that
    # sum already rounds light_time_B to ulp(t_tx) -- the precision is lost
    # at CONSTRUCTION time, not at the final subtraction, so subtracting
    # t_obs_s afterwards cannot recover it. This was confirmed empirically
    # (examples/phase17_r1od_oracles.py s19): D_S drifted by up to 1.3e-4
    # relative across binades out to T_obs ~ 2^36 s under the old formula.
    #
    # The genuinely local form -- exactly the two_way_range.py Q1-F01/F08
    # repair pattern (work with the AUTHORITATIVE LOCAL light time, never an
    # epoch difference reconstructed from two large absolute numbers) -- is
    # to subtract the two SMALL light times directly. Neither operand here
    # is larger than the light time across the baseline (~tens of ms), so
    # the result carries full double-precision significance regardless of
    # how large T_obs is.
    d_s = forward.light_time_s - backward.light_time_s

    return CommonTransmitEventSolution(
        t_obs_s=t_obs_s,
        transmit_time_s=t_tx,
        station_b_receive_time_s=t_b,
        spacecraft_differential_delay_s=d_s,
        backward_converged=backward.converged,
        backward_equation_residual_s=backward.equation_residual_s,
        forward_converged=forward.converged,
        forward_iterations=forward.iterations,
        forward_equation_residual_s=forward.equation_residual_s,
        station_a_range_m=backward.range_m,
        station_b_range_m=forward.range_m,
    )


def quasar_differential_delay_s(
    station_a_position_m: ArrayLike,
    station_b_position_m: ArrayLike,
    quasar: QuasarDirection,
    *,
    light_speed_mps: float = C_LIGHT_MPS,
) -> float:
    """D_Q = -(r_B - r_A) . s_hat / c, the analytic plane-wave differential delay.

    Both station positions must be evaluated at the SAME reference epoch by
    the caller (R1O-D §27 characterizes the sensitivity to that choice: the
    baseline rotates with Earth, so evaluating at t_A vs t_B vs their mean
    is not interchangeable at nanoradian precision).
    """
    r_a = np.asarray(station_a_position_m, dtype=float).reshape(3)
    r_b = np.asarray(station_b_position_m, dtype=float).reshape(3)
    s_hat = quasar.unit_vector_j2000
    return float(-np.dot(r_b - r_a, s_hat) / light_speed_mps)


def delta_dor_observable_s(
    common_event: CommonTransmitEventSolution,
    quasar_differential_delay_value_s: float,
) -> float:
    """DDOR = D_S - D_Q, in seconds.  Sign convention QUALIFIED by oracle (R1O-D §35)."""
    return common_event.spacecraft_differential_delay_s - quasar_differential_delay_value_s


@dataclass(frozen=True)
class DeltaDorSensitivity:
    """Implicit-event-matrix Jacobian of D_S w.r.t. arc-initial state and K.

    Built with the SAME 2x2-implicit-system pattern
    `two_way_range_event_sensitivity` uses for the two-way range chain: an
    event matrix `g_y` (partials of the two light-time equations w.r.t. the
    solved event times [t_tx, t_B]) and `g_x`/`g_k` (partials w.r.t. state/K,
    event times held fixed).  `dy/dx0 = solve(g_y, -g_x)`;
    `d(D_S)/dx0 = dt_B/dx0` (row 1 of the solve), since T_obs is a constant.
    """

    event_time_sensitivity_dx0: np.ndarray   # (2, 6): [dt_tx/dx0; dt_B/dx0]
    d_spacecraft_delay_dx0: np.ndarray        # (6,)  == event_time_sensitivity_dx0[1]
    d_spacecraft_delay_dk: float               # scalar, chained through S_K
    event_matrix: np.ndarray                   # (2, 2) g_y
    event_matrix_condition_number: float
    station_a_unit_los: np.ndarray
    station_b_unit_los: np.ndarray


def delta_dor_spacecraft_sensitivity_full(
    common_event: CommonTransmitEventSolution,
    t_grid_s: ArrayLike,
    augmented_state_history_mci: ArrayLike,
    station_a_position_m: ArrayLike,
    station_b_position_at_tb_m: ArrayLike,
    station_b_velocity_at_tb_mps: ArrayLike,
    *,
    light_speed_mps: float = C_LIGHT_MPS,
) -> DeltaDorSensitivity:
    """Full 2x2 implicit-event Jacobian; the K column is chained if x_aug has 48 cols."""
    t_grid = np.asarray(t_grid_s, dtype=float).reshape(-1)
    x_aug = np.asarray(augmented_state_history_mci, dtype=float)
    if x_aug.ndim != 2 or x_aug.shape[1] < 42:
        raise ValueError("augmented_state_history_mci must contain state plus 6x6 STM columns.")
    has_k = x_aug.shape[1] >= 48
    c = float(light_speed_mps)

    t_tx = common_event.transmit_time_s
    states = x_aug[:, :6]
    # Phi is stored column-major flattened (order="F"), matching the exact
    # reshape two_way_range_event_sensitivity uses for the same 42-column
    # augmented history -- reusing a different order here would silently
    # transpose every Phi lookup.
    phi_history = np.array(
        [row.reshape((6, 6), order="F") for row in x_aug[:, 6:42]], dtype=float
    )
    sc_tx = _interp_state(t_grid, states, t_tx)
    phi_r_tx = _interp_state_transition_position(t_grid, phi_history, t_tx)
    v_sc_tx = sc_tx[3:6]

    station_a_pos = np.asarray(station_a_position_m, dtype=float).reshape(3)
    station_b_pos = np.asarray(station_b_position_at_tb_m, dtype=float).reshape(3)
    v_b_tb = np.asarray(station_b_velocity_at_tb_mps, dtype=float).reshape(3)

    rho_a_vec = sc_tx[:3] - station_a_pos
    rho_a = float(np.linalg.norm(rho_a_vec))
    rho_b_vec = sc_tx[:3] - station_b_pos
    rho_b = float(np.linalg.norm(rho_b_vec))
    if rho_a <= 0.0 or rho_b <= 0.0:
        raise DeltaDorEventError("Station ranges must be positive for sensitivity evaluation.")
    u_a = rho_a_vec / rho_a
    u_b = rho_b_vec / rho_b

    g_y = np.array([
        [1.0 + float(np.dot(u_a, v_sc_tx)) / c, 0.0],
        [-1.0 - float(np.dot(u_b, v_sc_tx)) / c, 1.0 + float(np.dot(u_b, v_b_tb)) / c],
    ], dtype=float)

    g_x = np.zeros((2, 6), dtype=float)
    g_x[0, :] = (u_a @ phi_r_tx) / c
    g_x[1, :] = -(u_b @ phi_r_tx) / c

    if not np.all(np.isfinite(g_y)):
        raise DeltaDorEventError(f"DDOR event matrix contains non-finite entries: {g_y!r}.")
    cond = float(np.linalg.cond(g_y))
    try:
        dy_dx0 = np.linalg.solve(g_y, -g_x)
    except np.linalg.LinAlgError as exc:
        raise DeltaDorEventError(
            f"DDOR event matrix is singular (condition number {cond:.3e})."
        ) from exc
    d_spacecraft_delay_dx0 = dy_dx0[1, :]

    d_spacecraft_delay_dk = float("nan")
    if has_k:
        # S_K is a (N,6) dx/dK history, not a matrix like Phi -- interpolated
        # with the same Hermite state interpolator used for the trajectory
        # itself (`_interp_state` only requires shape (N, 6+)), then only the
        # position rows (columns 0:3) are used, exactly as `phi_r_tx` above
        # uses only Phi's position rows.
        s_k_flat = x_aug[:, 42:48]
        s_k_tx = _interp_state(t_grid, s_k_flat, t_tx)[:3]
        g_k = np.zeros((2, 1), dtype=float)
        g_k[0, 0] = (u_a @ s_k_tx) / c
        g_k[1, 0] = -(u_b @ s_k_tx) / c
        dy_dk = np.linalg.solve(g_y, -g_k)
        d_spacecraft_delay_dk = float(dy_dk[1, 0])

    return DeltaDorSensitivity(
        event_time_sensitivity_dx0=np.asarray(dy_dx0, dtype=float),
        d_spacecraft_delay_dx0=np.asarray(d_spacecraft_delay_dx0, dtype=float),
        d_spacecraft_delay_dk=d_spacecraft_delay_dk,
        event_matrix=g_y,
        event_matrix_condition_number=cond,
        station_a_unit_los=u_a,
        station_b_unit_los=u_b,
    )
