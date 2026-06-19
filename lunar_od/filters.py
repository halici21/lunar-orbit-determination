"""Sequential filtering helpers for Lunar OD experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import erf, sqrt
from time import perf_counter
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .dynamics import dynamics_jacobian_a_matrix, f3body_moon, propagate_augmented_state, propagate_state

_IDENTITY_6_COL: np.ndarray = np.eye(6).reshape(-1, order="F")
from .geometry import ecef2razel_sez, wrap_to_pi
from .measurements import PassGeometry
from .radiometrics import (
    instantaneous_geometric_range_rate,
    range_rate_physics_config,
    two_way_counted_doppler_observable,
)


@dataclass(frozen=True)
class UnscentedTransformConfig:
    """Tuning parameters for sigma-point generation."""

    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0
    jitter: float = 1e-12

    def __post_init__(self) -> None:
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive.")
        if self.beta < 0.0:
            raise ValueError("beta must be non-negative.")
        if self.jitter <= 0.0:
            raise ValueError("jitter must be positive.")


@dataclass(frozen=True)
class UKFState:
    """Filtered state and covariance at one epoch."""

    x: np.ndarray
    p: np.ndarray


@dataclass(frozen=True)
class SquareRootUKFState:
    """Filtered state and lower-triangular covariance square root."""

    x: np.ndarray
    sqrt_p: np.ndarray

    @property
    def p(self) -> np.ndarray:
        return _symmetrize(self.sqrt_p @ self.sqrt_p.T)


@dataclass(frozen=True)
class UKFUpdateDiagnostics:
    """Innovation diagnostics for one UKF measurement update."""

    innovation: np.ndarray
    innovation_covariance: np.ndarray
    kalman_gain: np.ndarray
    predicted_measurement: np.ndarray
    normalized_innovation_squared: float
    measurement_noise_scale: float
    robust_component_weights: np.ndarray
    accepted: bool
    accepted_components: np.ndarray


@dataclass(frozen=True)
class UKFPerformanceDiagnostics:
    elapsed_s: float
    process_function_evaluations: int
    unique_dynamic_propagations: int
    dynamic_propagation_cache_hits: int
    measurement_function_evaluations: int
    unique_measurement_model_evaluations: int
    measurement_model_cache_hits: int


@dataclass(frozen=True)
class UKFStabilityDiagnostics:
    finite_state_history: bool
    finite_covariance_history: bool
    positive_semidefinite_covariances: bool
    min_covariance_eigenvalue: float
    max_covariance_condition_number: float
    max_normalized_innovation_squared: float
    accepted_update_fraction: float
    robust_reweighted_fraction: float
    stable: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class UKFAdaptiveConfig:
    """Optional consistency and covariance tuning controls for UKF updates."""

    covariance_inflation: float = 1.0
    adaptive_process_noise: bool = False
    initial_process_noise_scale: float = 1.0
    min_process_noise_scale: float = 0.1
    max_process_noise_scale: float = 100.0
    process_noise_adaptation_gain: float = 0.2
    adaptive_measurement_noise: bool = False
    max_measurement_noise_scale: float = 100.0
    nis_gate: float | None = None
    component_nis_gate: float | None = None
    component_gate_mode: Literal["marginal", "conditional"] = "marginal"
    robust_measurement_update: bool = False
    robust_loss: Literal["student_t", "huber"] = "student_t"
    robust_student_t_dof: float = 5.0
    robust_huber_threshold: float = 3.0
    robust_min_component_weight: float = 0.05

    def __post_init__(self) -> None:
        if self.covariance_inflation < 1.0:
            raise ValueError("covariance_inflation must be at least 1.0.")
        if self.initial_process_noise_scale <= 0.0:
            raise ValueError("initial_process_noise_scale must be positive.")
        if self.min_process_noise_scale <= 0.0:
            raise ValueError("min_process_noise_scale must be positive.")
        if self.max_process_noise_scale < self.min_process_noise_scale:
            raise ValueError("max_process_noise_scale must be at least min_process_noise_scale.")
        if not (self.min_process_noise_scale <= self.initial_process_noise_scale <= self.max_process_noise_scale):
            raise ValueError("initial_process_noise_scale must be within the min/max process-noise scale bounds.")
        if not (0.0 <= self.process_noise_adaptation_gain <= 1.0):
            raise ValueError("process_noise_adaptation_gain must be between 0 and 1.")
        if self.max_measurement_noise_scale < 1.0:
            raise ValueError("max_measurement_noise_scale must be at least 1.0.")
        if self.nis_gate is not None and self.nis_gate <= 0.0:
            raise ValueError("nis_gate must be positive when provided.")
        if self.component_nis_gate is not None and self.component_nis_gate <= 0.0:
            raise ValueError("component_nis_gate must be positive when provided.")
        if self.component_gate_mode not in {"marginal", "conditional"}:
            raise ValueError("component_gate_mode must be 'marginal' or 'conditional'.")
        if self.robust_loss not in {"student_t", "huber"}:
            raise ValueError("robust_loss must be 'student_t' or 'huber'.")
        if self.robust_student_t_dof <= 2.0:
            raise ValueError("robust_student_t_dof must be greater than 2.")
        if self.robust_huber_threshold <= 0.0:
            raise ValueError("robust_huber_threshold must be positive.")
        if not (0.0 < self.robust_min_component_weight <= 1.0):
            raise ValueError("robust_min_component_weight must be in (0, 1].")


@dataclass(frozen=True)
class LunarUKFResult:
    """Sequential UKF estimates for one measurement arc."""

    t_update_s: np.ndarray
    obs_indices: np.ndarray
    state_estimates: np.ndarray
    covariances: np.ndarray
    innovations: np.ndarray
    innovation_covariances: np.ndarray
    predicted_measurements: np.ndarray
    normalized_innovation_squared: np.ndarray
    measurement_noise_scales: np.ndarray
    robust_component_weights: np.ndarray
    process_noise_scales: np.ndarray
    accepted_updates: np.ndarray
    accepted_components: np.ndarray
    final_state: np.ndarray
    final_covariance: np.ndarray
    performance: UKFPerformanceDiagnostics


def assess_ukf_operational_stability(
    result: LunarUKFResult,
    *,
    covariance_eigenvalue_floor: float = -1e-10,
    max_covariance_condition_number: float = 1e18,
    max_normalized_innovation_squared: float = 1e9,
    min_accepted_update_fraction: float = 0.8,
) -> UKFStabilityDiagnostics:
    """Summarize numerical health of a completed UKF arc."""
    states = np.asarray(result.state_estimates, dtype=float)
    covariances = np.asarray(result.covariances, dtype=float)
    nis_values = np.asarray(result.normalized_innovation_squared, dtype=float)
    accepted = np.asarray(result.accepted_updates, dtype=bool)
    robust_weights = np.asarray(result.robust_component_weights, dtype=float)

    failures: list[str] = []
    finite_states = bool(np.all(np.isfinite(states)))
    finite_covariances = bool(np.all(np.isfinite(covariances)))
    finite_nis = bool(np.all(np.isfinite(nis_values)))
    if not finite_states:
        failures.append("nonfinite_state_history")
    if not finite_covariances:
        failures.append("nonfinite_covariance_history")
    if not finite_nis:
        failures.append("nonfinite_nis_history")

    min_eigenvalue = float("nan")
    max_condition = float("nan")
    psd_covariances = False
    if covariances.size and finite_covariances:
        eigenvalue_rows = np.asarray([np.linalg.eigvalsh(_symmetrize(covariance)) for covariance in covariances])
        min_eigenvalue = float(np.min(eigenvalue_rows))
        psd_covariances = bool(min_eigenvalue >= covariance_eigenvalue_floor)
        conditions = [float(np.linalg.cond(_symmetrize(covariance))) for covariance in covariances]
        max_condition = float(np.max(conditions)) if conditions else float("nan")
    if not psd_covariances:
        failures.append("covariance_not_psd")
    if np.isfinite(max_condition) and max_condition > max_covariance_condition_number:
        failures.append("covariance_condition_too_large")

    max_nis = float(np.max(nis_values)) if nis_values.size and finite_nis else float("nan")
    if np.isfinite(max_nis) and max_nis > max_normalized_innovation_squared:
        failures.append("nis_too_large")

    accepted_fraction = float(np.mean(accepted)) if accepted.size else 0.0
    if accepted_fraction < min_accepted_update_fraction:
        failures.append("accepted_update_fraction_too_low")

    robust_reweighted_fraction = (
        float(np.mean(robust_weights < (1.0 - 1e-12))) if robust_weights.size else 0.0
    )
    return UKFStabilityDiagnostics(
        finite_state_history=finite_states,
        finite_covariance_history=finite_covariances,
        positive_semidefinite_covariances=psd_covariances,
        min_covariance_eigenvalue=min_eigenvalue,
        max_covariance_condition_number=max_condition,
        max_normalized_innovation_squared=max_nis,
        accepted_update_fraction=accepted_fraction,
        robust_reweighted_fraction=robust_reweighted_fraction,
        stable=not failures,
        failures=tuple(failures),
    )


ProcessFunction = Callable[[np.ndarray], np.ndarray]
MeasurementFunction = Callable[[np.ndarray], np.ndarray]
ResidualFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]
ProcessNoiseModel = Literal["discrete", "continuous_white_acceleration"]
CovarianceForm = Literal["standard", "square_root"]


def chi_square_nis_gate(dimension: int, *, sigma: float = 3.0, probability: float | None = None) -> float:
    """Return a chi-square NIS gate for a vector measurement.

    With ``probability=None``, the two-sided 1-D Gaussian probability contained
    inside ``+/- sigma`` is mapped to a chi-square quantile with ``dimension``
    degrees of freedom. For example, ``sigma=3`` corresponds to about
    99.73 percent probability.
    """
    if dimension <= 0:
        raise ValueError("dimension must be positive.")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    if probability is None:
        probability = erf(float(sigma) / sqrt(2.0))
    if not (0.0 < float(probability) < 1.0):
        raise ValueError("probability must be in (0, 1).")
    from scipy.stats import chi2

    return float(chi2.ppf(float(probability), int(dimension)))


def sigma_points(
    mean: ArrayLike,
    covariance: ArrayLike,
    config: UnscentedTransformConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sigma points and mean/covariance weights."""
    cfg = config or UnscentedTransformConfig()
    x = np.asarray(mean, dtype=float).reshape(-1)
    p = _symmetrize(np.asarray(covariance, dtype=float))
    if p.shape != (x.size, x.size):
        raise ValueError("covariance shape must match mean size.")

    n = x.size
    lam = cfg.alpha**2 * (n + cfg.kappa) - n
    scale = n + lam
    if scale <= 0.0:
        raise ValueError("unscented transform scale must be positive.")

    sqrt_p = _cholesky_psd(scale * p, cfg.jitter)
    points = np.zeros((2 * n + 1, n), dtype=float)
    points[0, :] = x
    for idx in range(n):
        col = sqrt_p[:, idx]
        points[1 + idx, :] = x + col
        points[1 + n + idx, :] = x - col

    wm = np.full(2 * n + 1, 0.5 / scale, dtype=float)
    wc = wm.copy()
    wm[0] = lam / scale
    wc[0] = wm[0] + (1.0 - cfg.alpha**2 + cfg.beta)
    return points, wm, wc


def _sigma_points_from_sqrt(
    mean: ArrayLike,
    sqrt_covariance: ArrayLike,
    config: UnscentedTransformConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(mean, dtype=float).reshape(-1)
    sqrt_p = np.asarray(sqrt_covariance, dtype=float)
    if sqrt_p.shape != (x.size, x.size):
        raise ValueError("sqrt_covariance shape must match mean size.")
    n = x.size
    lam = config.alpha**2 * (n + config.kappa) - n
    scale = n + lam
    if scale <= 0.0:
        raise ValueError("unscented transform scale must be positive.")
    spread = np.sqrt(scale) * sqrt_p
    points = np.zeros((2 * n + 1, n), dtype=float)
    points[0] = x
    for idx in range(n):
        points[1 + idx] = x + spread[:, idx]
        points[1 + n + idx] = x - spread[:, idx]
    wm = np.full(2 * n + 1, 0.5 / scale, dtype=float)
    wc = wm.copy()
    wm[0] = lam / scale
    wc[0] = wm[0] + (1.0 - config.alpha**2 + config.beta)
    return points, wm, wc


def unscented_mean_and_covariance(
    transformed_points: ArrayLike,
    wm: ArrayLike,
    wc: ArrayLike,
    *,
    additive_covariance: ArrayLike | None = None,
    residual_fn: ResidualFunction | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover a mean and covariance from transformed sigma points."""
    points = np.asarray(transformed_points, dtype=float)
    mean_weights = np.asarray(wm, dtype=float).reshape(-1)
    cov_weights = np.asarray(wc, dtype=float).reshape(-1)
    if points.ndim != 2:
        raise ValueError("transformed_points must be a 2-D array.")
    if points.shape[0] != mean_weights.size or mean_weights.size != cov_weights.size:
        raise ValueError("weight lengths must match the number of sigma points.")

    mean = _unscented_mean(points, mean_weights, residual_fn)
    cov = np.zeros((points.shape[1], points.shape[1]), dtype=float)
    for idx, point in enumerate(points):
        delta = _residual(point, mean, residual_fn)
        cov += cov_weights[idx] * np.outer(delta, delta)
    if additive_covariance is not None:
        cov += np.asarray(additive_covariance, dtype=float)
    return mean, _symmetrize(cov)


def ukf_predict(
    state: UKFState,
    process_fn: ProcessFunction,
    *,
    process_noise: ArrayLike | None = None,
    covariance_inflation: float = 1.0,
    config: UnscentedTransformConfig | None = None,
    residual_fn: ResidualFunction | None = None,
) -> tuple[UKFState, np.ndarray, np.ndarray, np.ndarray]:
    """Run one UKF prediction step through a nonlinear process function."""
    if covariance_inflation < 1.0:
        raise ValueError("covariance_inflation must be at least 1.0.")
    points, wm, wc = sigma_points(state.x, state.p, config)
    propagated = np.asarray([process_fn(point) for point in points], dtype=float)
    q = None if process_noise is None else np.asarray(process_noise, dtype=float)
    x_pred, p_pred = unscented_mean_and_covariance(
        propagated,
        wm,
        wc,
        additive_covariance=q,
        residual_fn=residual_fn,
    )
    p_pred *= covariance_inflation
    p_pred = _symmetrize(p_pred)
    if q is not None or covariance_inflation != 1.0:
        propagated, wm, wc = sigma_points(x_pred, p_pred, config)
    return UKFState(x_pred, p_pred), propagated, wm, wc


def square_root_ukf_predict(
    state: SquareRootUKFState,
    process_fn: ProcessFunction,
    *,
    process_noise: ArrayLike | None = None,
    covariance_inflation: float = 1.0,
    config: UnscentedTransformConfig | None = None,
    residual_fn: ResidualFunction | None = None,
    stm_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[SquareRootUKFState, np.ndarray, np.ndarray, np.ndarray]:
    """Run a square-root UKF prediction using QR and Cholesky updates.

    When stm_fn is provided, it is called once on the mean state to obtain
    the 42-element augmented result [x_new(6), phi_flat(36)] and all sigma
    points are linearized via the STM, reducing 2n+1 ODE calls to one.
    """
    if covariance_inflation < 1.0:
        raise ValueError("covariance_inflation must be at least 1.0.")
    cfg = config or UnscentedTransformConfig()
    points, wm, wc = _sigma_points_from_sqrt(state.x, state.sqrt_p, cfg)
    if stm_fn is not None:
        x_mean = state.x[:6]
        aug_end = stm_fn(x_mean)              # (42,): [x_new(6), phi_flat(36)]
        x_mean_new = aug_end[:6]
        phi = aug_end[6:].reshape(6, 6, order="F")
        offsets_dyn = points[:, :6] - x_mean[None, :]
        propagated_dyn = x_mean_new[None, :] + (phi @ offsets_dyn.T).T
        if points.shape[1] > 6:
            propagated = np.concatenate([propagated_dyn, points[:, 6:]], axis=1)
        else:
            propagated = propagated_dyn
    else:
        propagated = np.asarray([process_fn(point) for point in points], dtype=float)
    x_pred = _unscented_mean(propagated, wm, residual_fn)
    sqrt_pred = _unscented_sqrt_covariance(
        propagated,
        x_pred,
        wc,
        additive_covariance=process_noise,
        residual_fn=residual_fn,
        jitter=cfg.jitter,
    )
    sqrt_pred *= np.sqrt(covariance_inflation)
    predicted = SquareRootUKFState(x_pred, sqrt_pred)
    propagated, wm, wc = _sigma_points_from_sqrt(predicted.x, predicted.sqrt_p, cfg)
    return predicted, propagated, wm, wc


def ukf_update(
    predicted_state: UKFState,
    predicted_sigma_points: ArrayLike,
    wm: ArrayLike,
    wc: ArrayLike,
    measurement: ArrayLike,
    measurement_fn: MeasurementFunction,
    measurement_noise: ArrayLike,
    *,
    state_residual_fn: ResidualFunction | None = None,
    measurement_residual_fn: ResidualFunction | None = None,
    adaptive_config: UKFAdaptiveConfig | None = None,
    config: UnscentedTransformConfig | None = None,
) -> tuple[UKFState, UKFUpdateDiagnostics]:
    """Run one UKF measurement update."""
    cfg = config or UnscentedTransformConfig()
    adaptive = adaptive_config or UKFAdaptiveConfig()
    x_pred = np.asarray(predicted_state.x, dtype=float).reshape(-1)
    p_pred = _symmetrize(np.asarray(predicted_state.p, dtype=float))
    sigma_x = np.asarray(predicted_sigma_points, dtype=float)
    z_obs = np.asarray(measurement, dtype=float).reshape(-1)
    r_base = np.asarray(measurement_noise, dtype=float)

    sigma_z = np.asarray([measurement_fn(point) for point in sigma_x], dtype=float)
    z_pred, s_cov = unscented_mean_and_covariance(
        sigma_z,
        wm,
        wc,
        additive_covariance=r_base,
        residual_fn=measurement_residual_fn,
    )
    innovation = _residual(z_obs, z_pred, measurement_residual_fn)
    measurement_noise_scale = 1.0
    robust_component_weights = np.ones(z_pred.size, dtype=float)
    r_effective = r_base.copy()
    nis = normalized_innovation_squared(innovation, s_cov)
    if adaptive.adaptive_measurement_noise and z_pred.size > 0 and nis > float(z_pred.size):
        measurement_noise_scale = min(adaptive.max_measurement_noise_scale, nis / float(z_pred.size))
        r_effective = measurement_noise_scale * r_base
        s_cov = _symmetrize(s_cov + (r_effective - r_base))
        nis = normalized_innovation_squared(innovation, s_cov)
    if adaptive.robust_measurement_update and z_pred.size > 0:
        r_robust, robust_component_weights = _robust_measurement_covariance(
            innovation,
            s_cov,
            r_effective,
            adaptive,
            cfg.jitter,
        )
        s_cov = _symmetrize(s_cov + (r_robust - r_effective))
        r_effective = r_robust
        measurement_noise_scale = max(
            measurement_noise_scale,
            _relative_covariance_scale(r_base, r_effective, cfg.jitter),
        )
        nis = normalized_innovation_squared(innovation, s_cov)

    cov_weights = np.asarray(wc, dtype=float).reshape(-1)
    if state_residual_fn is None and measurement_residual_fn is None:
        dx_all = sigma_x - x_pred[None, :]
        dz_all = sigma_z - z_pred[None, :]
    else:
        n = sigma_x.shape[0]
        dx_all = np.array([_residual(sigma_x[i], x_pred, state_residual_fn) for i in range(n)])
        dz_all = np.array([_residual(sigma_z[i], z_pred, measurement_residual_fn) for i in range(n)])
    pxz = np.einsum("n,ni,nj->ij", cov_weights, dx_all, dz_all)

    s_cov = _symmetrize(s_cov)
    accepted_components = _component_gate_mask(
        innovation,
        s_cov,
        adaptive.component_nis_gate,
        adaptive.component_gate_mode,
        cfg.jitter,
    )

    accepted = bool(np.any(accepted_components))
    if adaptive.nis_gate is not None and nis > adaptive.nis_gate:
        accepted = False
        accepted_components[:] = False
    if not accepted:
        diagnostics = UKFUpdateDiagnostics(
            innovation=innovation,
            innovation_covariance=s_cov,
            kalman_gain=np.zeros((x_pred.size, z_pred.size), dtype=float),
            predicted_measurement=z_pred,
            normalized_innovation_squared=float(nis),
            measurement_noise_scale=float(measurement_noise_scale),
            robust_component_weights=robust_component_weights,
            accepted=False,
            accepted_components=accepted_components,
        )
        return UKFState(x_pred.copy(), p_pred.copy()), diagnostics

    active = np.flatnonzero(accepted_components)
    s_active = s_cov[np.ix_(active, active)]
    pxz_active = pxz[:, active]
    innovation_active = innovation[active]
    gain_active = np.linalg.solve(s_active.T, pxz_active.T).T
    gain = np.zeros_like(pxz)
    gain[:, active] = gain_active
    x_upd = x_pred + gain_active @ innovation_active
    p_upd = _symmetrize(p_pred - gain_active @ s_active @ gain_active.T)
    p_upd = _ensure_psd(p_upd, cfg.jitter)
    diagnostics = UKFUpdateDiagnostics(
        innovation=innovation,
        innovation_covariance=s_cov,
        kalman_gain=gain,
        predicted_measurement=z_pred,
        normalized_innovation_squared=float(nis),
        measurement_noise_scale=float(measurement_noise_scale),
        robust_component_weights=robust_component_weights,
        accepted=True,
        accepted_components=accepted_components,
    )
    return UKFState(x_upd, p_upd), diagnostics


def square_root_ukf_update(
    predicted_state: SquareRootUKFState,
    predicted_sigma_points: ArrayLike,
    wm: ArrayLike,
    wc: ArrayLike,
    measurement: ArrayLike,
    measurement_fn: MeasurementFunction,
    measurement_noise: ArrayLike,
    *,
    state_residual_fn: ResidualFunction | None = None,
    measurement_residual_fn: ResidualFunction | None = None,
    adaptive_config: UKFAdaptiveConfig | None = None,
    config: UnscentedTransformConfig | None = None,
) -> tuple[SquareRootUKFState, UKFUpdateDiagnostics]:
    """Run a square-root UKF measurement update."""
    cfg = config or UnscentedTransformConfig()
    adaptive = adaptive_config or UKFAdaptiveConfig()
    x_pred = np.asarray(predicted_state.x, dtype=float).reshape(-1)
    sigma_x = np.asarray(predicted_sigma_points, dtype=float)
    z_obs = np.asarray(measurement, dtype=float).reshape(-1)
    r_base = np.asarray(measurement_noise, dtype=float)
    sigma_z = np.asarray([measurement_fn(point) for point in sigma_x], dtype=float)
    z_pred = _unscented_mean(sigma_z, np.asarray(wm, dtype=float), measurement_residual_fn)
    sqrt_s = _unscented_sqrt_covariance(
        sigma_z,
        z_pred,
        np.asarray(wc, dtype=float),
        additive_covariance=r_base,
        residual_fn=measurement_residual_fn,
        jitter=cfg.jitter,
    )
    s_cov = _symmetrize(sqrt_s @ sqrt_s.T)
    innovation = _residual(z_obs, z_pred, measurement_residual_fn)
    measurement_noise_scale = 1.0
    robust_component_weights = np.ones(z_pred.size, dtype=float)
    r_effective = r_base.copy()
    nis = normalized_innovation_squared(innovation, s_cov)
    if adaptive.adaptive_measurement_noise and z_pred.size > 0 and nis > float(z_pred.size):
        measurement_noise_scale = min(adaptive.max_measurement_noise_scale, nis / float(z_pred.size))
        r_effective = measurement_noise_scale * r_base
        sqrt_s = _unscented_sqrt_covariance(
            sigma_z,
            z_pred,
            np.asarray(wc, dtype=float),
            additive_covariance=r_effective,
            residual_fn=measurement_residual_fn,
            jitter=cfg.jitter,
        )
        s_cov = _symmetrize(sqrt_s @ sqrt_s.T)
        nis = normalized_innovation_squared(innovation, s_cov)
    if adaptive.robust_measurement_update and z_pred.size > 0:
        r_effective, robust_component_weights = _robust_measurement_covariance(
            innovation,
            s_cov,
            r_effective,
            adaptive,
            cfg.jitter,
        )
        sqrt_s = _unscented_sqrt_covariance(
            sigma_z,
            z_pred,
            np.asarray(wc, dtype=float),
            additive_covariance=r_effective,
            residual_fn=measurement_residual_fn,
            jitter=cfg.jitter,
        )
        s_cov = _symmetrize(sqrt_s @ sqrt_s.T)
        measurement_noise_scale = max(
            measurement_noise_scale,
            _relative_covariance_scale(r_base, r_effective, cfg.jitter),
        )
        nis = normalized_innovation_squared(innovation, s_cov)

    cov_weights_sr = np.asarray(wc, dtype=float).reshape(-1)
    if state_residual_fn is None and measurement_residual_fn is None:
        dx_all = sigma_x - x_pred[None, :]
        dz_all = sigma_z - z_pred[None, :]
    else:
        n = sigma_x.shape[0]
        dx_all = np.array([_residual(sigma_x[i], x_pred, state_residual_fn) for i in range(n)])
        dz_all = np.array([_residual(sigma_z[i], z_pred, measurement_residual_fn) for i in range(n)])
    pxz = np.einsum("n,ni,nj->ij", cov_weights_sr, dx_all, dz_all)

    accepted_components = _component_gate_mask(
        innovation,
        s_cov,
        adaptive.component_nis_gate,
        adaptive.component_gate_mode,
        cfg.jitter,
    )
    accepted = bool(np.any(accepted_components))
    if adaptive.nis_gate is not None and nis > adaptive.nis_gate:
        accepted = False
        accepted_components[:] = False
    if not accepted:
        diagnostics = UKFUpdateDiagnostics(
            innovation=innovation,
            innovation_covariance=s_cov,
            kalman_gain=np.zeros((x_pred.size, z_pred.size), dtype=float),
            predicted_measurement=z_pred,
            normalized_innovation_squared=float(nis),
            measurement_noise_scale=float(measurement_noise_scale),
            robust_component_weights=robust_component_weights,
            accepted=False,
            accepted_components=accepted_components,
        )
        return predicted_state, diagnostics

    active = np.flatnonzero(accepted_components)
    s_active = s_cov[np.ix_(active, active)]
    pxz_active = pxz[:, active]
    gain_active = np.linalg.solve(s_active.T, pxz_active.T).T
    gain = np.zeros_like(pxz)
    gain[:, active] = gain_active
    x_upd = x_pred + gain_active @ innovation[active]
    update_root = gain_active @ _cholesky_psd(s_active, cfg.jitter)
    sqrt_upd = predicted_state.sqrt_p.copy()
    try:
        for col in range(update_root.shape[1]):
            sqrt_upd = _cholupdate_lower(sqrt_upd, update_root[:, col], sign=-1)
    except np.linalg.LinAlgError:
        p_upd = predicted_state.p - gain_active @ s_active @ gain_active.T
        sqrt_upd = _cholesky_psd(_ensure_psd(p_upd, cfg.jitter), cfg.jitter)
    diagnostics = UKFUpdateDiagnostics(
        innovation=innovation,
        innovation_covariance=s_cov,
        kalman_gain=gain,
        predicted_measurement=z_pred,
        normalized_innovation_squared=float(nis),
        measurement_noise_scale=float(measurement_noise_scale),
        robust_component_weights=robust_component_weights,
        accepted=True,
        accepted_components=accepted_components,
    )
    return SquareRootUKFState(x_upd, sqrt_upd), diagnostics


def ukf_predict_update(
    state: UKFState,
    process_fn: ProcessFunction,
    measurement: ArrayLike,
    measurement_fn: MeasurementFunction,
    measurement_noise: ArrayLike,
    *,
    process_noise: ArrayLike | None = None,
    covariance_inflation: float = 1.0,
    config: UnscentedTransformConfig | None = None,
    state_residual_fn: ResidualFunction | None = None,
    measurement_residual_fn: ResidualFunction | None = None,
    adaptive_config: UKFAdaptiveConfig | None = None,
) -> tuple[UKFState, UKFUpdateDiagnostics]:
    """Convenience wrapper for one predict/update cycle."""
    predicted, sigma_x, wm, wc = ukf_predict(
        state,
        process_fn,
        process_noise=process_noise,
        covariance_inflation=covariance_inflation,
        config=config,
        residual_fn=state_residual_fn,
    )
    return ukf_update(
        predicted,
        sigma_x,
        wm,
        wc,
        measurement,
        measurement_fn,
        measurement_noise,
        state_residual_fn=state_residual_fn,
        measurement_residual_fn=measurement_residual_fn,
        adaptive_config=adaptive_config,
        config=config,
    )


def run_lunar_ukf(
    t_pass_s: ArrayLike,
    obs_data: ArrayLike,
    x0_mci: ArrayLike,
    p0: ArrayLike,
    pass_geo: PassGeometry,
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
    covariance_form: CovarianceForm = "square_root",
    adaptive_config: UKFAdaptiveConfig | None = None,
    config: UnscentedTransformConfig | None = None,
    frozen_state_indices: Sequence[int] = (),
    regularization_std_by_state: Mapping[int, float] | None = None,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    fast_sigma_propagator: Callable | None = None,
    use_stm_linearization: bool = False,
) -> LunarUKFResult:
    """Run a sequential UKF over one prepared lunar OD arc.

    The first 6 elements are the dynamic lunar state. Optional trailing bias
    states follow the BLS/SRIF conventions: global_full, station_angles, or
    station_full. Bias states are modeled as constants unless process noise is
    supplied for them.
    """
    start_time = perf_counter()
    measurement_type = (measurement_type or pass_geo.measurement_type).lower()
    if measurement_type not in {"position", "range_rate"}:
        raise ValueError("measurement_type must be 'position' or 'range_rate'.")
    adaptive = adaptive_config or UKFAdaptiveConfig()

    t_pass = np.asarray(t_pass_s, dtype=float).reshape(-1)
    obs = np.asarray(obs_data, dtype=float)
    if obs.ndim != 2:
        raise ValueError("obs_data must be a 2-D array.")
    if measurement_type == "position" and obs.shape[1] < 6:
        raise ValueError("position obs_data must have at least 6 columns.")
    if measurement_type == "range_rate" and obs.shape[1] < 7:
        raise ValueError("range_rate obs_data must have at least 7 columns.")

    order = np.argsort(obs[:, 0], kind="stable")
    x0 = np.asarray(x0_mci, dtype=float).reshape(-1)
    if x0.size < 6:
        raise ValueError("x0_mci must contain at least the 6 dynamic state elements.")
    nx = x0.size
    frozen_indices = _validate_state_constraints(
        nx,
        frozen_state_indices,
        regularization_std_by_state,
    )
    regularization = dict(regularization_std_by_state or {})
    constraints_active = bool(frozen_indices or regularization)
    state = UKFState(x0, _symmetrize(np.asarray(p0, dtype=float)))
    if state.p.shape != (nx, nx):
        raise ValueError("p0 must be an NxN covariance matrix matching x0_mci.")
    bias_cfg = _resolve_ukf_bias_config(measurement_type, nx, len(pass_geo.stations), bias_mode)

    if process_noise_model not in {"discrete", "continuous_white_acceleration"}:
        raise ValueError("process_noise_model must be 'discrete' or 'continuous_white_acceleration'.")
    if covariance_form not in {"standard", "square_root"}:
        raise ValueError("covariance_form must be 'standard' or 'square_root'.")
    q = _coerce_process_noise(process_noise, nx, process_noise_model)
    sqrt_state = SquareRootUKFState(state.x.copy(), _cholesky_psd(state.p, (config or UnscentedTransformConfig()).jitter))
    current_t = float(t_pass[0])
    states: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    innovations: list[np.ndarray] = []
    innovation_covariances: list[np.ndarray] = []
    predicted_measurements: list[np.ndarray] = []
    nis_values: list[float] = []
    measurement_noise_scales: list[float] = []
    robust_component_weight_rows: list[np.ndarray] = []
    process_noise_scales: list[float] = []
    accepted_updates: list[bool] = []
    accepted_component_rows: list[np.ndarray] = []
    t_updates: list[float] = []
    obs_indices: list[int] = []
    measurement_dim = 3 if measurement_type == "position" else 4
    q_scale = adaptive.initial_process_noise_scale
    epoch_accepted_nis: list[float] = []
    process_function_evaluations = 0
    unique_dynamic_propagations = 0
    dynamic_propagation_cache_hits = 0
    measurement_function_evaluations = 0
    unique_measurement_model_evaluations = 0
    measurement_model_cache_hits = 0

    for obs_idx in order:
        row = obs[obs_idx, :]
        target_t = float(row[0])
        is_time_advance = not np.isclose(current_t, target_t)
        if is_time_advance and epoch_accepted_nis:
            q_scale = adapt_process_noise_scale(
                q_scale,
                float(np.mean(epoch_accepted_nis)),
                measurement_dim,
                adaptive,
                enabled=q is not None,
            )
            epoch_accepted_nis = []
        dt_s = abs(target_t - current_t)
        q_step = _discretize_process_noise(q, dt_s, nx, process_noise_model) if is_time_advance else None
        propagation_cache: dict[bytes, np.ndarray] = {}

        def process_fn(x: np.ndarray, t0: float = current_t, t1: float = target_t) -> np.ndarray:
            nonlocal process_function_evaluations
            nonlocal unique_dynamic_propagations
            nonlocal dynamic_propagation_cache_hits
            process_function_evaluations += 1
            if np.isclose(t0, t1):
                return x.copy()
            if fast_sigma_propagator is not None:
                unique_dynamic_propagations += 1
                return np.concatenate([fast_sigma_propagator(t0, t1, x[:6]), x[6:]])
            key = np.ascontiguousarray(x[:6]).tobytes()
            if key in propagation_cache:
                dynamic_propagation_cache_hits += 1
                return np.concatenate([propagation_cache[key], x[6:]])
            propagated_dyn = propagate_state(
                [t0, t1],
                x[:6],
                mu_moon_m3_s2,
                mu_earth_m3_s2,
                mu_sun_m3_s2,
                get_earth_pos,
                get_sun_pos,
                rtol=rtol,
                atol=atol,
            )[-1, :]
            unique_dynamic_propagations += 1
            propagation_cache[key] = propagated_dyn
            return np.concatenate([propagated_dyn, x[6:]])

        def _stm_fn(x6: np.ndarray, _t0: float = current_t, _t1: float = target_t) -> np.ndarray:
            if np.isclose(_t0, _t1):
                return np.concatenate([x6.copy(), _IDENTITY_6_COL])
            x_aug0 = np.concatenate([x6, _IDENTITY_6_COL])
            return propagate_augmented_state(
                [_t0, _t1], x_aug0,
                mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
                get_earth_pos, get_sun_pos,
                rtol=rtol, atol=atol,
            )[-1, :]

        if covariance_form == "square_root":
            predicted, sigma_x, wm, wc = square_root_ukf_predict(
                sqrt_state,
                process_fn,
                process_noise=_scaled_process_noise(q_step, q_scale),
                covariance_inflation=adaptive.covariance_inflation if is_time_advance else 1.0,
                config=config,
                stm_fn=_stm_fn if (use_stm_linearization and is_time_advance) else None,
            )
        else:
            predicted, sigma_x, wm, wc = ukf_predict(
                state,
                process_fn,
                process_noise=_scaled_process_noise(q_step, q_scale),
                covariance_inflation=adaptive.covariance_inflation if is_time_advance else 1.0,
                config=config,
            )
        z, r, measurement_fn, measurement_dim, measurement_stats = _measurement_context(
            measurement_type,
            row,
            pass_geo,
            mu_moon_m3_s2,
            mu_earth_m3_s2,
            mu_sun_m3_s2,
            get_earth_pos,
            get_sun_pos,
            rtol,
            atol,
            bias_cfg,
        )
        measurement_function_evaluations += int(sigma_x.shape[0])
        if covariance_form == "square_root":
            sqrt_state, diagnostics = square_root_ukf_update(
                predicted,
                sigma_x,
                wm,
                wc,
                z,
                measurement_fn,
                r,
                measurement_residual_fn=_measurement_residual(measurement_type),
                adaptive_config=adaptive,
                config=config,
            )
            state = UKFState(sqrt_state.x.copy(), sqrt_state.p)
        else:
            state, diagnostics = ukf_update(
                predicted,
                sigma_x,
                wm,
                wc,
                z,
                measurement_fn,
                r,
                measurement_residual_fn=_measurement_residual(measurement_type),
                adaptive_config=adaptive,
                config=config,
            )
        if constraints_active:
            state = _apply_state_constraints(
                state,
                UKFState(predicted.x.copy(), predicted.p),
                x0,
                frozen_indices,
                regularization,
                (config or UnscentedTransformConfig()).jitter,
            )
            if covariance_form == "square_root":
                sqrt_state = SquareRootUKFState(
                    state.x.copy(),
                    _cholesky_psd(state.p, (config or UnscentedTransformConfig()).jitter),
                )
        unique_measurement_model_evaluations += measurement_stats["evaluations"]
        measurement_model_cache_hits += measurement_stats["cache_hits"]
        current_t = target_t
        states.append(state.x.copy())
        covariances.append(state.p.copy())
        innovations.append(diagnostics.innovation.copy())
        innovation_covariances.append(diagnostics.innovation_covariance.copy())
        predicted_measurements.append(diagnostics.predicted_measurement.copy())
        nis_values.append(float(diagnostics.normalized_innovation_squared))
        measurement_noise_scales.append(float(diagnostics.measurement_noise_scale))
        robust_component_weight_rows.append(diagnostics.robust_component_weights.copy())
        process_noise_scales.append(float(q_scale))
        accepted_updates.append(bool(diagnostics.accepted))
        accepted_component_rows.append(diagnostics.accepted_components.copy())
        if diagnostics.accepted:
            epoch_accepted_nis.append(float(diagnostics.normalized_innovation_squared))
        t_updates.append(target_t)
        obs_indices.append(int(obs_idx))

    state_estimates = np.asarray(states, dtype=float).reshape(len(states), nx)
    covariance_history = np.asarray(covariances, dtype=float).reshape(len(covariances), nx, nx)
    innovation_history = np.asarray(innovations, dtype=float).reshape(len(innovations), measurement_dim)
    innovation_covariance_history = np.asarray(innovation_covariances, dtype=float).reshape(
        len(innovation_covariances), measurement_dim, measurement_dim
    )
    predicted_history = np.asarray(predicted_measurements, dtype=float).reshape(len(predicted_measurements), measurement_dim)
    return LunarUKFResult(
        t_update_s=np.asarray(t_updates, dtype=float),
        obs_indices=np.asarray(obs_indices, dtype=int),
        state_estimates=state_estimates,
        covariances=covariance_history,
        innovations=innovation_history,
        innovation_covariances=innovation_covariance_history,
        predicted_measurements=predicted_history,
        normalized_innovation_squared=np.asarray(nis_values, dtype=float),
        measurement_noise_scales=np.asarray(measurement_noise_scales, dtype=float),
        robust_component_weights=np.asarray(robust_component_weight_rows, dtype=float).reshape(
            len(robust_component_weight_rows), measurement_dim
        ),
        process_noise_scales=np.asarray(process_noise_scales, dtype=float),
        accepted_updates=np.asarray(accepted_updates, dtype=bool),
        accepted_components=np.asarray(accepted_component_rows, dtype=bool).reshape(
            len(accepted_component_rows), measurement_dim
        ),
        final_state=state.x.copy(),
        final_covariance=state.p.copy(),
        performance=UKFPerformanceDiagnostics(
            elapsed_s=perf_counter() - start_time,
            process_function_evaluations=process_function_evaluations,
            unique_dynamic_propagations=unique_dynamic_propagations,
            dynamic_propagation_cache_hits=dynamic_propagation_cache_hits,
            measurement_function_evaluations=measurement_function_evaluations,
            unique_measurement_model_evaluations=unique_measurement_model_evaluations,
            measurement_model_cache_hits=measurement_model_cache_hits,
        ),
    )


def _residual(value: np.ndarray, reference: np.ndarray, residual_fn: ResidualFunction | None) -> np.ndarray:
    if residual_fn is None:
        return np.asarray(value, dtype=float) - np.asarray(reference, dtype=float)
    return np.asarray(residual_fn(value, reference), dtype=float)


def _validate_state_constraints(
    state_size: int,
    frozen_state_indices: Sequence[int],
    regularization_std_by_state: Mapping[int, float] | None,
) -> tuple[int, ...]:
    frozen = tuple(sorted(set(int(index) for index in frozen_state_indices)))
    regularization = regularization_std_by_state or {}
    indices = set(frozen) | {int(index) for index in regularization}
    if any(index < 0 or index >= state_size for index in indices):
        raise ValueError("State constraint index is outside the state vector.")
    if any(float(std) <= 0.0 or not np.isfinite(float(std)) for std in regularization.values()):
        raise ValueError("State regularization standard deviations must be finite and positive.")
    return frozen


def _apply_state_constraints(
    updated: UKFState,
    predicted: UKFState,
    prior_mean: np.ndarray,
    frozen_indices: Sequence[int],
    regularization_std_by_state: Mapping[int, float],
    jitter: float,
) -> UKFState:
    x = updated.x.copy()
    p = updated.p.copy()
    frozen = set(frozen_indices)
    for index, std in regularization_std_by_state.items():
        idx = int(index)
        if idx in frozen:
            continue
        covariance_column = p[:, idx].copy()
        innovation_variance = float(p[idx, idx] + float(std) ** 2)
        gain = covariance_column / max(innovation_variance, jitter)
        x += gain * (prior_mean[idx] - x[idx])
        p -= np.outer(covariance_column, covariance_column) / max(innovation_variance, jitter)
        p = _symmetrize(p)
    for idx in frozen_indices:
        x[idx] = predicted.x[idx]
        p[idx, :] = predicted.p[idx, :]
        p[:, idx] = predicted.p[:, idx]
    return UKFState(x=x, p=_ensure_psd(p, jitter))


def _component_gate_mask(
    innovation: np.ndarray,
    covariance: np.ndarray,
    gate: float | None,
    mode: str,
    jitter: float,
) -> np.ndarray:
    if gate is None:
        return np.ones(innovation.size, dtype=bool)
    if mode == "marginal":
        statistics = innovation**2 / np.clip(np.diag(covariance), jitter, None)
        return statistics <= gate

    active = list(range(innovation.size))
    while len(active) > 1:
        active_covariance = covariance[np.ix_(active, active)]
        active_innovation = innovation[active]
        statistics = _conditional_component_nis(active_innovation, active_covariance, jitter)
        worst_local = int(np.argmax(statistics))
        if statistics[worst_local] <= gate:
            break
        del active[worst_local]
    if len(active) == 1:
        last = active[0]
        if innovation[last] ** 2 / max(covariance[last, last], jitter) > gate:
            active = []
    mask = np.zeros(innovation.size, dtype=bool)
    mask[active] = True
    return mask


def _conditional_component_nis(
    innovation: np.ndarray,
    covariance: np.ndarray,
    jitter: float,
) -> np.ndarray:
    statistics = np.zeros(innovation.size, dtype=float)
    for idx in range(innovation.size):
        others = np.array([j for j in range(innovation.size) if j != idx], dtype=int)
        if others.size == 0:
            statistics[idx] = innovation[idx] ** 2 / max(covariance[idx, idx], jitter)
            continue
        covariance_others = covariance[np.ix_(others, others)]
        cross = covariance[idx, others]
        solved_innovation = np.linalg.solve(covariance_others, innovation[others])
        solved_cross = np.linalg.solve(covariance_others, covariance[others, idx])
        conditional_residual = innovation[idx] - cross @ solved_innovation
        conditional_variance = covariance[idx, idx] - cross @ solved_cross
        statistics[idx] = conditional_residual**2 / max(float(conditional_variance), jitter)
    return statistics


def _robust_measurement_covariance(
    innovation: np.ndarray,
    innovation_covariance: np.ndarray,
    measurement_covariance: np.ndarray,
    adaptive: UKFAdaptiveConfig,
    jitter: float,
) -> tuple[np.ndarray, np.ndarray]:
    residual = np.asarray(innovation, dtype=float).reshape(-1)
    s_cov = _symmetrize(np.asarray(innovation_covariance, dtype=float))
    r_cov = _symmetrize(np.asarray(measurement_covariance, dtype=float))
    if r_cov.shape != (residual.size, residual.size):
        raise ValueError("measurement_covariance shape must match innovation size.")
    if residual.size == 0:
        return r_cov.copy(), np.ones(0, dtype=float)

    variance = np.clip(np.diag(s_cov), jitter, None)
    statistic = residual**2 / variance
    if adaptive.robust_loss == "student_t":
        dof = float(adaptive.robust_student_t_dof)
        weights = (dof + 1.0) / (dof + statistic)
    else:
        normalized_abs = np.sqrt(statistic)
        threshold = float(adaptive.robust_huber_threshold)
        weights = np.ones_like(normalized_abs)
        outlier = normalized_abs > threshold
        weights[outlier] = threshold / np.clip(normalized_abs[outlier], jitter, None)

    weights = np.clip(weights, adaptive.robust_min_component_weight, 1.0)
    scale = 1.0 / np.sqrt(weights)
    scaled_covariance = scale[:, None] * r_cov * scale[None, :]
    return _symmetrize(scaled_covariance), weights


def _relative_covariance_scale(
    base_covariance: np.ndarray,
    effective_covariance: np.ndarray,
    jitter: float,
) -> float:
    base_diag = np.diag(_symmetrize(np.asarray(base_covariance, dtype=float)))
    effective_diag = np.diag(_symmetrize(np.asarray(effective_covariance, dtype=float)))
    valid = base_diag > jitter
    if not np.any(valid):
        return 1.0
    ratio = np.max(effective_diag[valid] / np.clip(base_diag[valid], jitter, None))
    if not np.isfinite(ratio):
        return 1.0
    return float(max(1.0, ratio))


def _unscented_mean(
    points: np.ndarray,
    weights: np.ndarray,
    residual_fn: ResidualFunction | None,
) -> np.ndarray:
    if residual_fn is None:
        return weights @ points

    mean = np.asarray(points[0], dtype=float).copy()
    for _ in range(20):
        correction = np.zeros(points.shape[1], dtype=float)
        for idx, point in enumerate(points):
            correction += weights[idx] * _residual(point, mean, residual_fn)
        mean = mean + correction
        if np.linalg.norm(correction, ord=np.inf) <= 1e-12:
            break
    return mean


def _unscented_sqrt_covariance(
    points: np.ndarray,
    mean: np.ndarray,
    weights: np.ndarray,
    *,
    additive_covariance: ArrayLike | None,
    residual_fn: ResidualFunction | None,
    jitter: float,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    signed_columns: list[tuple[np.ndarray, int]] = []
    for idx, point in enumerate(np.asarray(points, dtype=float)):
        weight = float(weights[idx])
        if weight == 0.0:
            continue
        column = np.sqrt(abs(weight)) * _residual(point, mean, residual_fn)
        if weight > 0.0:
            columns.append(column)
        else:
            signed_columns.append((column, -1))
    if additive_covariance is not None:
        q = _symmetrize(np.asarray(additive_covariance, dtype=float))
        if q.shape != (mean.size, mean.size):
            raise ValueError("additive_covariance shape must match transformed point dimension.")
        q_root = _cholesky_psd(q, jitter)
        columns.extend(q_root[:, idx] for idx in range(q_root.shape[1]))

    if columns:
        matrix = np.column_stack(columns)
        _, upper = np.linalg.qr(matrix.T, mode="reduced")
        sqrt_covariance = upper.T
        signs = np.where(np.diag(sqrt_covariance) < 0.0, -1.0, 1.0)
        sqrt_covariance = sqrt_covariance @ np.diag(signs)
    else:
        sqrt_covariance = np.sqrt(jitter) * np.eye(mean.size)
    for column, sign in signed_columns:
        sqrt_covariance = _cholupdate_lower(sqrt_covariance, column, sign=sign)
    return sqrt_covariance


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def _cholupdate_lower(lower: np.ndarray, vector: ArrayLike, *, sign: int) -> np.ndarray:
    """Rank-one update or downdate of a lower Cholesky factor."""
    if sign not in {-1, 1}:
        raise ValueError("sign must be +1 or -1.")
    result = np.asarray(lower, dtype=float).copy()
    x = np.asarray(vector, dtype=float).reshape(-1).copy()
    if result.shape != (x.size, x.size):
        raise ValueError("lower shape must match vector size.")
    for idx in range(x.size):
        diagonal_sq = result[idx, idx] ** 2 + sign * x[idx] ** 2
        if diagonal_sq <= 0.0:
            raise np.linalg.LinAlgError("Cholesky downdate is not positive definite.")
        r = np.sqrt(diagonal_sq)
        c = r / result[idx, idx]
        s = x[idx] / result[idx, idx]
        result[idx, idx] = r
        if idx + 1 < x.size:
            column = result[idx + 1 :, idx].copy()
            result[idx + 1 :, idx] = (column + sign * s * x[idx + 1 :]) / c
            x[idx + 1 :] = c * x[idx + 1 :] - s * result[idx + 1 :, idx]
    return result


def _cholesky_psd(matrix: np.ndarray, jitter: float) -> np.ndarray:
    matrix = _symmetrize(matrix)
    eye = np.eye(matrix.shape[0], dtype=float)
    attempt = jitter
    for _ in range(8):
        try:
            return np.linalg.cholesky(matrix + attempt * eye)
        except np.linalg.LinAlgError:
            attempt *= 10.0
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.clip(vals, attempt, None)
    return vecs @ np.diag(np.sqrt(vals))


def _ensure_psd(matrix: np.ndarray, jitter: float) -> np.ndarray:
    matrix = _symmetrize(matrix)
    vals, vecs = np.linalg.eigh(matrix)
    if np.min(vals) >= jitter:
        return matrix
    vals = np.clip(vals, jitter, None)
    return _symmetrize((vecs * vals) @ vecs.T)


def normalized_innovation_squared(innovation: ArrayLike, innovation_covariance: ArrayLike) -> float:
    """Return innovation.T @ inv(S) @ innovation for consistency monitoring."""
    residual = np.asarray(innovation, dtype=float).reshape(-1)
    covariance = _symmetrize(np.asarray(innovation_covariance, dtype=float))
    if covariance.shape != (residual.size, residual.size):
        raise ValueError("innovation_covariance shape must match innovation size.")
    return float(residual @ np.linalg.solve(covariance, residual))


def normalized_estimation_error_squared(
    estimate: ArrayLike,
    truth: ArrayLike,
    covariance: ArrayLike,
    *,
    residual_fn: ResidualFunction | None = None,
) -> float:
    """Return estimation-error NEES when a truth state is available."""
    error = _residual(np.asarray(estimate, dtype=float), np.asarray(truth, dtype=float), residual_fn).reshape(-1)
    covariance_arr = _symmetrize(np.asarray(covariance, dtype=float))
    if covariance_arr.shape != (error.size, error.size):
        raise ValueError("covariance shape must match estimate size.")
    return float(error @ np.linalg.solve(covariance_arr, error))


def _scaled_process_noise(process_noise: np.ndarray | None, scale: float) -> np.ndarray | None:
    if process_noise is None:
        return None
    return float(scale) * np.asarray(process_noise, dtype=float)


def discretize_white_acceleration_process_noise(
    acceleration_psd: ArrayLike,
    dt_s: float,
    *,
    state_size: int = 6,
) -> np.ndarray:
    """Discretize continuous white acceleration PSD for [position, velocity] states."""
    if dt_s < 0.0:
        raise ValueError("dt_s must be non-negative.")
    if state_size < 6:
        raise ValueError("state_size must be at least 6.")
    spectral_density = np.asarray(acceleration_psd, dtype=float)
    if spectral_density.ndim == 0:
        spectral_density = float(spectral_density) * np.eye(3)
    elif spectral_density.shape == (3,):
        spectral_density = np.diag(spectral_density)
    if spectral_density.shape != (3, 3):
        raise ValueError("acceleration_psd must be scalar, length 3, or 3x3.")
    spectral_density = _symmetrize(spectral_density)
    if np.any(np.linalg.eigvalsh(spectral_density) < -1e-15):
        raise ValueError("acceleration_psd must be positive semidefinite.")

    dt = float(dt_s)
    q = np.zeros((state_size, state_size), dtype=float)
    q[:3, :3] = (dt**3 / 3.0) * spectral_density
    q[:3, 3:6] = (dt**2 / 2.0) * spectral_density
    q[3:6, :3] = q[:3, 3:6]
    q[3:6, 3:6] = dt * spectral_density
    return q


def _coerce_process_noise(
    process_noise: ArrayLike | None,
    state_size: int,
    model: ProcessNoiseModel,
) -> np.ndarray | None:
    if process_noise is None:
        return None
    q = np.asarray(process_noise, dtype=float)
    if model == "continuous_white_acceleration":
        if q.ndim == 0 or q.shape in {(3,), (3, 3)}:
            return q
        raise ValueError(
            "continuous_white_acceleration process_noise must be scalar, length 3, or 3x3 acceleration PSD."
        )
    if q.shape == (state_size, state_size):
        return q
    if state_size > 6 and q.shape == (6, 6):
        q_aug = np.zeros((state_size, state_size), dtype=float)
        q_aug[:6, :6] = q
        return q_aug
    raise ValueError("process_noise must be either 6x6 or NxN matching x0_mci.")


def _discretize_process_noise(
    process_noise: np.ndarray | None,
    dt_s: float,
    state_size: int,
    model: ProcessNoiseModel,
) -> np.ndarray | None:
    if process_noise is None:
        return None
    if model == "continuous_white_acceleration":
        return discretize_white_acceleration_process_noise(process_noise, dt_s, state_size=state_size)
    return process_noise


def adapt_process_noise_scale(
    current_scale: float,
    nis: float,
    measurement_dim: int,
    adaptive: UKFAdaptiveConfig,
    *,
    enabled: bool,
) -> float:
    if not enabled or not adaptive.adaptive_process_noise:
        return float(current_scale)
    if measurement_dim <= 0:
        return float(current_scale)
    normalized_nis = float(nis) / float(measurement_dim)
    if not np.isfinite(normalized_nis) or normalized_nis <= 0.0:
        return float(current_scale)
    new_scale = float(current_scale) * normalized_nis ** adaptive.process_noise_adaptation_gain
    return float(np.clip(new_scale, adaptive.min_process_noise_scale, adaptive.max_process_noise_scale))


def _resolve_ukf_bias_config(
    measurement_type: str,
    state_size: int,
    num_stations: int,
    bias_mode: str | None,
) -> dict:
    nx_dyn = 6
    if state_size < nx_dyn:
        raise ValueError("state vector must contain at least 6 dynamic elements.")
    measurement_dim = 3 if measurement_type == "position" else 4

    if bias_mode is None:
        if state_size == nx_dyn:
            bias_mode = "none"
        elif state_size == nx_dyn + measurement_dim:
            bias_mode = "global_full"
        else:
            raise ValueError(
                "Provide bias_mode for station-specific UKF bias states. "
                "Supported modes: global_full, station_angles, station_full."
            )

    mode = bias_mode.lower()
    aliases = {
        "global": "global_full",
        "global_rr_full": "global_full",
        "station": "station_full",
        "station_rr_full": "station_full",
    }
    mode = aliases.get(mode, mode)

    if mode == "none":
        expected = nx_dyn
        block_size = 0
    elif mode == "global_full":
        expected = nx_dyn + measurement_dim
        block_size = measurement_dim
    elif mode == "station_angles":
        expected = nx_dyn + 2 * num_stations
        block_size = 2
    elif mode == "station_full":
        expected = nx_dyn + measurement_dim * num_stations
        block_size = measurement_dim
    else:
        raise ValueError(f"Unsupported UKF bias mode: {bias_mode}")

    if state_size != expected:
        raise ValueError(f"bias_mode={mode} expects state size {expected}, got {state_size}.")
    return {
        "mode": mode,
        "size": state_size - nx_dyn,
        "block_size": block_size,
        "measurement_type": measurement_type,
        "measurement_dim": measurement_dim,
        "num_stations": num_stations,
    }


def _apply_ukf_measurement_bias(
    measurement: np.ndarray,
    obs_row: np.ndarray,
    state: ArrayLike,
    bias_cfg: dict,
) -> np.ndarray:
    bias = np.asarray(state, dtype=float).reshape(-1)[6:]
    if bias.size == 0 or bias_cfg["mode"] == "none":
        return np.asarray(measurement, dtype=float)

    value = np.asarray(measurement, dtype=float).copy()
    mode = bias_cfg["mode"]
    measurement_type = bias_cfg["measurement_type"]
    if mode == "global_full":
        return value + bias.reshape(value.shape)

    station_col = 4 if measurement_type == "position" else 5
    station_id = int(obs_row[station_col]) - 1
    if station_id < 0 or station_id >= bias_cfg["num_stations"]:
        raise ValueError("Observation station id is out of range for UKF bias state.")

    if mode == "station_angles":
        col0 = station_id * 2
        value[-2:] += bias[col0 : col0 + 2]
    elif mode == "station_full":
        block_size = int(bias_cfg["block_size"])
        col0 = station_id * block_size
        value += bias[col0 : col0 + block_size]
    return value


def _position_measurement_from_state(state_mci: ArrayLike, obs_row: np.ndarray, pass_geo: PassGeometry) -> np.ndarray:
    station_id = int(obs_row[4]) - 1
    time_idx = int(obs_row[5]) - 1
    station = pass_geo.stations[station_id]
    r_mci = np.asarray(state_mci, dtype=float).reshape(-1)[:3]
    dr_eci = r_mci - pass_geo.earth_pos_mci_m[time_idx, :]
    r_ecef = pass_geo.x_j2000_to_itrf93[time_idx, :3, :3] @ dr_eci
    rho_ecef = r_ecef - station.r_ecef_m
    az_rad, el_rad, range_m = ecef2razel_sez(rho_ecef, station.lat_rad, station.lon_rad)
    return np.array([range_m, az_rad, el_rad], dtype=float)


def _position_measurement_covariance(obs_row: np.ndarray, pass_geo: PassGeometry) -> np.ndarray:
    station_id = int(obs_row[4]) - 1
    station = pass_geo.stations[station_id]
    sigma = np.array([station.sigma_range_m, station.sigma_angle_rad, station.sigma_angle_rad], dtype=float)
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("Position measurement sigmas must be finite and positive.")
    return np.diag(sigma**2)


def _position_measurement_residual(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    diff = np.asarray(value, dtype=float).reshape(3) - np.asarray(reference, dtype=float).reshape(3)
    diff[1] = wrap_to_pi(diff[1])
    diff[2] = wrap_to_pi(diff[2])
    return diff


def _measurement_context(
    measurement_type: str,
    obs_row: np.ndarray,
    pass_geo: PassGeometry,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    rtol: float,
    atol: float,
    bias_cfg: dict,
) -> tuple[np.ndarray, np.ndarray, MeasurementFunction, int, dict[str, int]]:
    physical_cache: dict[bytes, np.ndarray] = {}
    stats = {"evaluations": 0, "cache_hits": 0}

    def cached_physical_measurement(x: np.ndarray) -> np.ndarray:
        key = np.ascontiguousarray(np.asarray(x, dtype=float).reshape(-1)[:6]).tobytes()
        if key in physical_cache:
            stats["cache_hits"] += 1
            return physical_cache[key]
        stats["evaluations"] += 1
        if measurement_type == "position":
            value = _position_measurement_from_state(x, obs_row, pass_geo)
        else:
            value = _range_rate_measurement_from_state(
                np.asarray(x, dtype=float).reshape(-1)[:6],
                obs_row,
                pass_geo,
                mu_moon_m3_s2,
                mu_earth_m3_s2,
                mu_sun_m3_s2,
                get_earth_pos,
                get_sun_pos,
                rtol,
                atol,
            )
        physical_cache[key] = value
        return value

    if measurement_type == "position":
        z = obs_row[1:4]
        r = _position_measurement_covariance(obs_row, pass_geo)
        measurement_fn = lambda x, row=obs_row: _apply_ukf_measurement_bias(
            cached_physical_measurement(x),
            row,
            x,
            bias_cfg,
        )
        return z, r, measurement_fn, 3, stats
    z = obs_row[1:5]
    r = _range_rate_measurement_covariance(obs_row, pass_geo)
    measurement_fn = lambda x, row=obs_row: _apply_ukf_measurement_bias(
        cached_physical_measurement(x),
        row,
        x,
        bias_cfg,
    )
    return z, r, measurement_fn, 4, stats


def _measurement_residual(measurement_type: str) -> ResidualFunction:
    if measurement_type == "position":
        return _position_measurement_residual
    return _range_rate_measurement_residual


def _range_rate_measurement_from_state(
    state_mci: ArrayLike,
    obs_row: np.ndarray,
    pass_geo: PassGeometry,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    rtol: float,
    atol: float,
    bias_cfg: dict | None = None,
) -> np.ndarray:
    station_id = int(obs_row[5]) - 1
    time_idx = int(obs_row[6]) - 1
    station = pass_geo.stations[station_id]
    rr_physics = range_rate_physics_config(pass_geo.range_rate_physics)
    state_full = np.asarray(state_mci, dtype=float).reshape(-1)
    state = state_full[:6]
    earth_state = np.concatenate([pass_geo.earth_pos_mci_m[time_idx, :], pass_geo.earth_vel_mci_mps[time_idx, :]])
    state_sat_eci = state - earth_state
    state_sat_ecef = pass_geo.x_j2000_to_itrf93[time_idx, :, :] @ state_sat_eci
    rho_ecef = state_sat_ecef[:3] - station.r_ecef_m
    v_rel_ecef = state_sat_ecef[3:]
    range_m = float(np.linalg.norm(rho_ecef))
    if rr_physics.mode == "geometric_instantaneous":
        rr_mps = instantaneous_geometric_range_rate(rho_ecef, v_rel_ecef)
    else:
        local_t, local_state, local_earth_pos, local_earth_vel, local_xforms = _two_way_local_histories(
            float(obs_row[0]),
            state,
            range_m,
            pass_geo,
            rr_physics.count_interval_s,
            rr_physics.light_speed_mps,
            mu_moon_m3_s2,
            mu_earth_m3_s2,
            mu_sun_m3_s2,
            get_earth_pos,
            get_sun_pos,
            rtol,
            atol,
            rr_physics.local_state_model,
        )
        rr_mps = two_way_counted_doppler_observable(
            float(obs_row[0]),
            station,
            local_t,
            local_state,
            local_earth_pos,
            local_earth_vel,
            local_xforms,
            rr_physics,
        )
    az_rad, el_rad, _ = ecef2razel_sez(rho_ecef, station.lat_rad, station.lon_rad)
    measurement = np.array([range_m, rr_mps, az_rad, el_rad], dtype=float)
    if bias_cfg is None:
        bias_cfg = _resolve_ukf_bias_config("range_rate", state_full.size, len(pass_geo.stations), None)
    return _apply_ukf_measurement_bias(measurement, obs_row, state_full, bias_cfg)


def _range_rate_measurement_covariance(obs_row: np.ndarray, pass_geo: PassGeometry) -> np.ndarray:
    station_id = int(obs_row[5]) - 1
    station = pass_geo.stations[station_id]
    if station.sigma_range_rate_mps is None:
        raise ValueError("range-rate UKF measurements require station.sigma_range_rate_mps.")
    sigma = np.array(
        [station.sigma_range_m, station.sigma_range_rate_mps, station.sigma_angle_rad, station.sigma_angle_rad],
        dtype=float,
    )
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("Range-rate measurement sigmas must be finite and positive.")
    return np.diag(sigma**2)


def _range_rate_measurement_residual(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    diff = np.asarray(value, dtype=float).reshape(4) - np.asarray(reference, dtype=float).reshape(4)
    diff[2] = wrap_to_pi(diff[2])
    diff[3] = wrap_to_pi(diff[3])
    return diff


def _two_way_local_histories(
    receive_mid_time_s: float,
    state_mid_mci: np.ndarray,
    range_estimate_m: float,
    pass_geo: PassGeometry,
    count_interval_s: float,
    light_speed_mps: float,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    rtol: float,
    atol: float,
    local_state_model: str = "ode",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    half_count = 0.5 * count_interval_s
    one_way_light_time_s = max(float(range_estimate_m) / light_speed_mps, 0.0)
    light_time_margin_s = max(2.0, 2.5 * one_way_light_time_s + 1.0)
    start_t = receive_mid_time_s - half_count - light_time_margin_s
    end_t = receive_mid_time_s + half_count
    step_s = min(5.0, max(1.0, count_interval_s / 6.0))
    n_grid = max(2, int(np.ceil((end_t - start_t) / step_s)) + 1)
    local_t = np.linspace(start_t, end_t, n_grid)
    local_t = np.unique(
        np.concatenate(
            [
                local_t,
                np.array(
                    [
                        receive_mid_time_s - half_count,
                        receive_mid_time_s,
                        receive_mid_time_s + half_count,
                    ],
                    dtype=float,
                ),
            ]
        )
    )
    local_t.sort()

    if local_state_model == "taylor3":
        local_state = _taylor3_local_state_history(
            local_t,
            receive_mid_time_s,
            state_mid_mci,
            mu_moon_m3_s2,
            mu_earth_m3_s2,
            mu_sun_m3_s2,
            get_earth_pos,
            get_sun_pos,
        )
    else:
        local_state = _propagate_local_state_history(
            local_t,
            receive_mid_time_s,
            state_mid_mci,
            mu_moon_m3_s2,
            mu_earth_m3_s2,
            mu_sun_m3_s2,
            get_earth_pos,
            get_sun_pos,
            rtol,
            atol,
        )

    local_earth_pos = _interp_pass_values(pass_geo.t_s, pass_geo.earth_pos_mci_m, local_t)
    local_earth_vel = _interp_pass_values(pass_geo.t_s, pass_geo.earth_vel_mci_mps, local_t)
    local_xforms = _interp_pass_values(pass_geo.t_s, pass_geo.x_j2000_to_itrf93, local_t)
    return local_t, local_state, local_earth_pos, local_earth_vel, local_xforms


def _taylor3_local_state_history(
    local_t_s: np.ndarray,
    mid_time_s: float,
    state_mid_mci: np.ndarray,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
) -> np.ndarray:
    local_t = np.asarray(local_t_s, dtype=float).reshape(-1)
    state_mid = np.asarray(state_mid_mci, dtype=float).reshape(6)
    earth_pos = np.asarray(get_earth_pos(float(mid_time_s)), dtype=float).reshape(-1, 3)[0]
    sun_pos = np.asarray(get_sun_pos(float(mid_time_s)), dtype=float).reshape(-1, 3)[0]
    acceleration = f3body_moon(
        state_mid,
        mu_moon_m3_s2,
        mu_earth_m3_s2,
        mu_sun_m3_s2,
        earth_pos,
        sun_pos,
    )[3:6]
    jacobian = dynamics_jacobian_a_matrix(
        state_mid,
        mu_moon_m3_s2,
        mu_earth_m3_s2,
        mu_sun_m3_s2,
        earth_pos,
        sun_pos,
    )
    jerk = jacobian[3:6, 0:3] @ state_mid[3:6]
    dt = (local_t - float(mid_time_s)).reshape(-1, 1)
    state_history = np.zeros((local_t.size, 6), dtype=float)
    state_history[:, :3] = (
        state_mid[:3]
        + dt * state_mid[3:6]
        + 0.5 * dt**2 * acceleration
        + (dt**3 / 6.0) * jerk
    )
    state_history[:, 3:6] = state_mid[3:6] + dt * acceleration + 0.5 * dt**2 * jerk
    return state_history


def _propagate_local_state_history(
    local_t_s: np.ndarray,
    mid_time_s: float,
    state_mid_mci: np.ndarray,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    rtol: float,
    atol: float,
) -> np.ndarray:
    local_t = np.asarray(local_t_s, dtype=float).reshape(-1)
    state_mid = np.asarray(state_mid_mci, dtype=float).reshape(6)
    state_history = np.zeros((local_t.size, 6), dtype=float)
    before_idx = np.where(local_t < mid_time_s)[0]
    mid_idx = np.where(np.isclose(local_t, mid_time_s))[0]
    after_idx = np.where(local_t > mid_time_s)[0]

    if before_idx.size:
        backward_times = np.concatenate([[mid_time_s], local_t[before_idx][::-1]])
        backward_history = propagate_state(
            backward_times,
            state_mid,
            mu_moon_m3_s2,
            mu_earth_m3_s2,
            mu_sun_m3_s2,
            get_earth_pos,
            get_sun_pos,
            rtol=rtol,
            atol=atol,
        )
        state_history[before_idx, :] = backward_history[1:, :][::-1, :]
    if mid_idx.size:
        state_history[mid_idx, :] = state_mid
    if after_idx.size:
        forward_times = np.concatenate([[mid_time_s], local_t[after_idx]])
        forward_history = propagate_state(
            forward_times,
            state_mid,
            mu_moon_m3_s2,
            mu_earth_m3_s2,
            mu_sun_m3_s2,
            get_earth_pos,
            get_sun_pos,
            rtol=rtol,
            atol=atol,
        )
        state_history[after_idx, :] = forward_history[1:, :]
    return state_history


def _interp_pass_values(source_t_s: ArrayLike, values: ArrayLike, target_t_s: ArrayLike) -> np.ndarray:
    source_t = np.asarray(source_t_s, dtype=float).reshape(-1)
    target_t = np.asarray(target_t_s, dtype=float).reshape(-1)
    array = np.asarray(values, dtype=float)
    original_shape = array.shape[1:]
    flat = array.reshape(source_t.size, -1)
    out = np.empty((target_t.size, flat.shape[1]), dtype=float)
    left_mask = target_t < source_t[0]
    right_mask = target_t > source_t[-1]
    for col in range(flat.shape[1]):
        col_vals = flat[:, col]
        out[:, col] = np.interp(target_t, source_t, col_vals)
        if left_mask.any():
            sl = (col_vals[1] - col_vals[0]) / (source_t[1] - source_t[0])
            out[left_mask, col] = col_vals[0] + sl * (target_t[left_mask] - source_t[0])
        if right_mask.any():
            sl = (col_vals[-1] - col_vals[-2]) / (source_t[-1] - source_t[-2])
            out[right_mask, col] = col_vals[-1] + sl * (target_t[right_mask] - source_t[-1])
    return out.reshape((target_t.size, *original_shape))


def _interp_1d_linear_extrap(source_t: np.ndarray, values: np.ndarray, target_t: float) -> float:
    if source_t.size < 2:
        raise ValueError("At least two time samples are required for interpolation.")
    if target_t <= source_t[0]:
        i0, i1 = 0, 1
    elif target_t >= source_t[-1]:
        i0, i1 = source_t.size - 2, source_t.size - 1
    else:
        return float(np.interp(target_t, source_t, values))
    dt = source_t[i1] - source_t[i0]
    if dt == 0.0:
        raise ValueError("source_t_s must be strictly increasing.")
    slope = (values[i1] - values[i0]) / dt
    return float(values[i0] + slope * (target_t - source_t[i0]))
