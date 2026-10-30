"""Extended Kalman filter and Rauch-Tung-Striebel smoother for Lunar OD arcs.

The EKF shares every model with the SR-UKF in ``filters.py`` so the two are
comparable on the same arc:

- the measurement function, noise and residual (angle wrapping) come from
  ``filters._measurement_context`` / ``filters._measurement_residual``;
- the process noise is coerced and discretised by the same helpers and added
  after propagation, as in the UKF;
- the dynamics STM is the variational STM of ``propagate_augmented_state``
  (the UKF's ``use_stm_linearization`` path); bias states are constants.

The measurement Jacobian is a central difference of that shared measurement
function, so the EKF linearises exactly the physics the UKF evaluates. The
covariance update uses the Joseph form.

``rts_smooth`` is the fixed-interval smoother over the stored filter history.
For a linear model it equals the batch least-squares solution with the same
prior (tests/test_ekf_smoother.py checks this exactly).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike

from .dynamics import propagate_augmented_state
from .filters import (
    MeasurementFunction,
    ProcessNoiseModel,
    ResidualFunction,
    _coerce_process_noise,
    _discretize_process_noise,
    _measurement_context,
    _measurement_residual,
    _resolve_ukf_bias_config,
    _symmetrize,
    normalized_innovation_squared,
    validate_ukf_measurement_support,
)
from .propagation_contract import numerical_metadata, resolve_profile
from .radiometrics import range_rate_physics_config

_IDENTITY_6_COL: np.ndarray = np.eye(6).reshape(-1, order="F")

#: Central-difference steps of the measurement Jacobian: position 1 m, velocity 1 mm/s.
#: Bias states enter the measurement additively, so any step is exact for them.
DEFAULT_JACOBIAN_STEPS = (1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3)
DEFAULT_BIAS_JACOBIAN_STEP = 1.0


@dataclass(frozen=True)
class EKFUpdateDiagnostics:
    innovation: np.ndarray
    innovation_covariance: np.ndarray
    kalman_gain: np.ndarray
    predicted_measurement: np.ndarray
    normalized_innovation_squared: float
    accepted: bool


@dataclass(frozen=True)
class RTSSmootherResult:
    """Smoothed states and covariances at the filter's update epochs."""

    state_estimates: np.ndarray
    covariances: np.ndarray
    gains: np.ndarray


@dataclass(frozen=True)
class LunarEKFResult:
    """Sequential EKF estimates for one measurement arc, plus what the RTS smoother needs."""

    t_update_s: np.ndarray
    obs_indices: np.ndarray
    state_estimates: np.ndarray
    covariances: np.ndarray
    predicted_states: np.ndarray
    predicted_covariances: np.ndarray
    transition_matrices: np.ndarray
    innovations: np.ndarray
    innovation_covariances: np.ndarray
    predicted_measurements: np.ndarray
    normalized_innovation_squared: np.ndarray
    accepted_updates: np.ndarray
    final_state: np.ndarray
    final_covariance: np.ndarray
    elapsed_s: float
    numerical_profile: dict | None = None
    smoothed: RTSSmootherResult | None = None


def ekf_predict(
    p: ArrayLike,
    x_pred: ArrayLike,
    phi: ArrayLike,
    process_noise: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Covariance time update P- = Phi P Phi^T + Q around an already propagated mean."""
    phi_m = np.asarray(phi, dtype=float)
    p_pred = phi_m @ np.asarray(p, dtype=float) @ phi_m.T
    if process_noise is not None:
        p_pred = p_pred + np.asarray(process_noise, dtype=float)
    return np.asarray(x_pred, dtype=float).copy(), _symmetrize(p_pred)


def ekf_update(
    x_pred: ArrayLike,
    p_pred: ArrayLike,
    measurement: ArrayLike,
    predicted_measurement: ArrayLike,
    h: ArrayLike,
    r: ArrayLike,
    *,
    residual_fn: ResidualFunction | None = None,
    nis_gate: float | None = None,
) -> tuple[np.ndarray, np.ndarray, EKFUpdateDiagnostics]:
    """Joseph-form measurement update; a NIS above ``nis_gate`` rejects the update."""
    x_m = np.asarray(x_pred, dtype=float).reshape(-1)
    p_m = _symmetrize(np.asarray(p_pred, dtype=float))
    z = np.asarray(measurement, dtype=float).reshape(-1)
    z_pred = np.asarray(predicted_measurement, dtype=float).reshape(-1)
    h_m = np.asarray(h, dtype=float)
    r_m = np.asarray(r, dtype=float)
    innovation = z - z_pred if residual_fn is None else np.asarray(residual_fn(z, z_pred), dtype=float)
    s = _symmetrize(h_m @ p_m @ h_m.T + r_m)
    nis = normalized_innovation_squared(innovation, s)
    if nis_gate is not None and nis > nis_gate:
        diag = EKFUpdateDiagnostics(innovation, s, np.zeros((x_m.size, z.size)), z_pred, float(nis), False)
        return x_m.copy(), p_m.copy(), diag
    gain = np.linalg.solve(s, h_m @ p_m).T
    x_upd = x_m + gain @ innovation
    i_kh = np.eye(x_m.size) - gain @ h_m
    p_upd = _symmetrize(i_kh @ p_m @ i_kh.T + gain @ r_m @ gain.T)
    return x_upd, p_upd, EKFUpdateDiagnostics(innovation, s, gain, z_pred, float(nis), True)


def finite_difference_measurement_jacobian(
    measurement_fn: MeasurementFunction,
    x: ArrayLike,
    steps: ArrayLike,
    residual_fn: ResidualFunction | None = None,
) -> np.ndarray:
    """Central-difference dh/dx of the shared measurement function (angle differences wrapped)."""
    x0 = np.asarray(x, dtype=float).reshape(-1)
    hs = np.asarray(steps, dtype=float).reshape(-1)
    if hs.size != x0.size or np.any(hs <= 0.0):
        raise ValueError("steps must be positive, one per state element.")
    columns = []
    for j, step in enumerate(hs):
        plus, minus = x0.copy(), x0.copy()
        plus[j] += step
        minus[j] -= step
        z_plus = np.asarray(measurement_fn(plus), dtype=float)
        z_minus = np.asarray(measurement_fn(minus), dtype=float)
        diff = z_plus - z_minus if residual_fn is None else np.asarray(residual_fn(z_plus, z_minus), dtype=float)
        columns.append(diff / (2.0 * step))
    return np.column_stack(columns)


def rts_smooth(
    filtered_states: ArrayLike,
    filtered_covariances: ArrayLike,
    predicted_states: ArrayLike,
    predicted_covariances: ArrayLike,
    transition_matrices: ArrayLike,
) -> RTSSmootherResult:
    """Fixed-interval Rauch-Tung-Striebel smoother.

    Row k of ``predicted_*`` and ``transition_matrices`` describe the step from
    update k-1 to update k (row 0 is the step from the prior and is unused).
    Same-epoch updates have Phi = I and Q = 0.
    """
    xf = np.asarray(filtered_states, dtype=float)
    pf = np.asarray(filtered_covariances, dtype=float)
    xp = np.asarray(predicted_states, dtype=float)
    pp = np.asarray(predicted_covariances, dtype=float)
    phis = np.asarray(transition_matrices, dtype=float)
    n = xf.shape[0]
    xs, ps = xf.copy(), pf.copy()
    gains = np.zeros_like(pf)
    for k in range(n - 2, -1, -1):
        # C_k = P_k|k Phi_k+1^T (P_k+1|k)^-1, solved without forming the inverse
        c = np.linalg.solve(pp[k + 1], phis[k + 1] @ pf[k]).T
        gains[k] = c
        xs[k] = xf[k] + c @ (xs[k + 1] - xp[k + 1])
        ps[k] = _symmetrize(pf[k] + c @ (ps[k + 1] - pp[k + 1]) @ c.T)
    return RTSSmootherResult(state_estimates=xs, covariances=ps, gains=gains)


def run_lunar_ekf(
    t_pass_s: ArrayLike,
    obs_data: ArrayLike,
    x0_mci: ArrayLike,
    p0: ArrayLike,
    pass_geo,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    *,
    measurement_type: str | None = None,
    bias_mode: str | None = None,
    process_noise: ArrayLike | None = None,
    process_noise_model: ProcessNoiseModel = "discrete",
    nis_gate: float | None = None,
    jacobian_steps: ArrayLike | None = None,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    j2_moon: float = 0.0,
    srp=None,
    propagation_profile=None,
    smooth: bool = False,
) -> LunarEKFResult:
    """Run an EKF (optionally followed by the RTS smoother) over one prepared lunar OD arc.

    Arguments mirror ``run_lunar_ukf``. The prior ``(x0_mci, p0)`` holds at the
    first observation epoch, as in the UKF. Not supported here: K_SRP
    solve-for, state freezing/regularisation, adaptive Q/R and robust updates.
    """
    start_time = perf_counter()
    propagation_profile = resolve_profile(srp, propagation_profile)
    propagation_kwargs = {} if propagation_profile is None else propagation_profile.propagator_kwargs()
    measurement_type = (measurement_type or pass_geo.measurement_type).lower()
    if measurement_type not in {"position", "range_rate"}:
        raise ValueError("measurement_type must be 'position' or 'range_rate'.")
    # same support gate as the UKF: the EKF evaluates the UKF's measurement function
    validate_ukf_measurement_support(
        "ukf",
        measurement_type,
        pass_geo.measurement_model_profile,
        apply_light_time=pass_geo.apply_light_time,
        apply_stellar_aberration=pass_geo.apply_stellar_aberration,
    )
    t_pass = np.asarray(t_pass_s, dtype=float).reshape(-1)
    obs = np.asarray(obs_data, dtype=float)
    if obs.ndim != 2:
        raise ValueError("obs_data must be a 2-D array.")
    x = np.asarray(x0_mci, dtype=float).reshape(-1).copy()
    nx = x.size
    if nx < 6:
        raise ValueError("x0_mci must contain at least the 6 dynamic state elements.")
    p = _symmetrize(np.asarray(p0, dtype=float))
    if p.shape != (nx, nx):
        raise ValueError("p0 must be an NxN covariance matrix matching x0_mci.")
    bias_cfg = _resolve_ukf_bias_config(measurement_type, nx, len(pass_geo.stations), bias_mode)
    if process_noise_model not in {"discrete", "continuous_white_acceleration"}:
        raise ValueError("process_noise_model must be 'discrete' or 'continuous_white_acceleration'.")
    q = _coerce_process_noise(process_noise, nx, process_noise_model)
    steps = (np.asarray(list(DEFAULT_JACOBIAN_STEPS) + [DEFAULT_BIAS_JACOBIAN_STEP] * (nx - 6), dtype=float)
             if jacobian_steps is None else np.asarray(jacobian_steps, dtype=float).reshape(-1))
    residual_fn = _measurement_residual(measurement_type)

    order = np.argsort(obs[:, 0], kind="stable")
    current_t = float(t_pass[0])
    rows: dict[str, list] = {key: [] for key in (
        "t", "idx", "x", "p", "xp", "pp", "phi", "innov", "s", "zpred", "nis", "accepted")}
    for obs_idx in order:
        row = obs[obs_idx, :]
        target_t = float(row[0])
        phi = np.eye(nx)
        x_prop = x.copy()
        if target_t != current_t:
            aug = propagate_augmented_state(
                [current_t, target_t], np.concatenate([x[:6], _IDENTITY_6_COL]),
                mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2, get_earth_pos, get_sun_pos,
                rtol=rtol, atol=atol, j2_moon=j2_moon, srp=srp, **propagation_kwargs,
            )[-1, :]
            x_prop[:6] = aug[:6]
            phi[:6, :6] = aug[6:42].reshape(6, 6, order="F")
            q_step = _discretize_process_noise(q, abs(target_t - current_t), nx, process_noise_model)
        else:
            q_step = None
        x_pred, p_pred = ekf_predict(p, x_prop, phi, q_step)
        z, r, measurement_fn, _, _ = _measurement_context(
            measurement_type, row, pass_geo, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
            get_earth_pos, get_sun_pos, rtol, atol, bias_cfg,
            j2_moon=j2_moon, srp=srp, solve_for_k_srp=False, propagation_profile=propagation_profile,
        )
        z_pred = np.asarray(measurement_fn(x_pred), dtype=float)
        h = finite_difference_measurement_jacobian(measurement_fn, x_pred, steps, residual_fn)
        x, p, diag = ekf_update(x_pred, p_pred, z, z_pred, h, r, residual_fn=residual_fn, nis_gate=nis_gate)
        current_t = target_t
        for key, value in (("t", target_t), ("idx", int(obs_idx)), ("x", x), ("p", p), ("xp", x_pred),
                           ("pp", p_pred), ("phi", phi), ("innov", diag.innovation),
                           ("s", diag.innovation_covariance), ("zpred", diag.predicted_measurement),
                           ("nis", diag.normalized_innovation_squared), ("accepted", diag.accepted)):
            rows[key].append(value)

    xs, ps = np.asarray(rows["x"]), np.asarray(rows["p"])
    xps, pps, phis = np.asarray(rows["xp"]), np.asarray(rows["pp"]), np.asarray(rows["phi"])
    smoothed = rts_smooth(xs, ps, xps, pps, phis) if smooth and xs.shape[0] > 0 else None
    return LunarEKFResult(
        t_update_s=np.asarray(rows["t"], dtype=float),
        obs_indices=np.asarray(rows["idx"], dtype=int),
        state_estimates=xs,
        covariances=ps,
        predicted_states=xps,
        predicted_covariances=pps,
        transition_matrices=phis,
        innovations=np.asarray(rows["innov"]),
        innovation_covariances=np.asarray(rows["s"]),
        predicted_measurements=np.asarray(rows["zpred"]),
        normalized_innovation_squared=np.asarray(rows["nis"], dtype=float),
        accepted_updates=np.asarray(rows["accepted"], dtype=bool),
        final_state=x.copy(),
        final_covariance=p.copy(),
        elapsed_s=perf_counter() - start_time,
        numerical_profile=numerical_metadata(
            srp, propagation_profile, rtol, atol, fast_sigma=False, adaptive_tolerances=False,
            local_state_model=range_rate_physics_config(pass_geo.range_rate_physics).local_state_model
            if measurement_type == "range_rate" else "not_applicable"),
        smoothed=smoothed,
    )
