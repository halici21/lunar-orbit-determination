"""Estimator helpers for the Python Lunar OD port."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike

from .accelerated import apply_stm_to_jacobian
from .dynamics import propagate_augmented_state, propagate_state
from .geometry import wrap_to_pi
from .measurements import PassGeometry, compute_position_residuals_analytic
from .measurements import compute_range_rate_residuals
from .measurements import compute_range_rate_residuals_analytic, measurement_sigma_vector
from .radiometrics import RangeRatePhysicsConfig, range_rate_physics_config
from .radiometrics import two_way_counted_doppler_initial_state_jacobian


_IDENTITY_6_COL: np.ndarray = np.eye(6).reshape(-1, order="F")


def _adaptive_tol(iteration: int, max_iter: int, rtol: float, atol: float) -> tuple[float, float]:
    """Loosen ODE tolerances for early BLS iterations; tighten as convergence nears."""
    if max_iter <= 1:
        return rtol, atol
    frac = (iteration - 1) / (max_iter - 1)
    if frac < 0.40:
        return max(rtol, 1e-8), max(atol, 1e-9)
    if frac < 0.75:
        return max(rtol, 1e-10), max(atol, 1e-11)
    return rtol, atol


@dataclass(frozen=True)
class EstimatorStats:
    iterations: int
    final_cost: float
    position_step_norm_m: float
    velocity_step_norm_mps: float
    condition_number: float = float("nan")
    rank: int = 0
    rejected_components: int = 0
    active_weight_fraction: float = 1.0
    posterior_information: np.ndarray | None = None
    posterior_covariance: np.ndarray | None = None
    posterior_sqrt_information: np.ndarray | None = None


def estimate_position_srif(
    t_pass_s: ArrayLike,
    obs_data: ArrayLike,
    x_nominal0: ArrayLike,
    pass_geo: PassGeometry,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    *,
    max_iter: int = 40,
    tol_step_norm: float = 1e-8,
    tol_cost_stability: float = 1e-8,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    j2_moon: float = 0.0,
    bias_mode: str | None = None,
    prior_covariance: ArrayLike | None = None,
    prior_sqrt_information: ArrayLike | None = None,
    return_posterior: bool = False,
) -> tuple[np.ndarray, str, EstimatorStats]:
    """Position-only SRIF/QR estimator for the initial state.

    Supported solve-for layouts:
    - 6 elements: dynamic state only
    - 9 elements: dynamic state plus global `[range, az, el]` bias
    - 6 + 2*num_stations elements with `bias_mode="station_angles"`
    - 6 + 3*num_stations elements with `bias_mode="station_full"`
    """
    t_pass_s = np.asarray(t_pass_s, dtype=float).reshape(-1)
    obs_data = np.asarray(obs_data, dtype=float)
    x_nominal = np.asarray(x_nominal0, dtype=float).reshape(-1).copy()

    nx = 6
    bias_cfg = _resolve_position_bias_config(x_nominal.size, nx, len(pass_geo.stations), bias_mode)
    nb = bias_cfg["size"]
    na = x_nominal.size

    has_explicit_prior = prior_covariance is not None or prior_sqrt_information is not None
    x_prior = x_nominal.copy()
    prior_inv, prior_sqrt_info, scale, _, prior_sqrt_scaled = _prior_information_and_scale(
        nx,
        bias_cfg,
        _position_bias_prior_and_scale,
        prior_covariance,
        prior_sqrt_information,
    )
    r_bar = prior_sqrt_scaled

    w_diag = _position_weight_diagonal(obs_data, pass_geo)
    w_sqrt = np.sqrt(w_diag)

    x_best = x_nominal.copy()
    best_cost = np.inf
    stop_reason = "MaxIter"
    last_step = np.zeros(6)
    last_condition_number = float("nan")
    last_rank = 0

    for iteration in range(1, max_iter + 1):
        x_dyn_nominal = x_nominal[:nx]
        b_nominal = x_nominal[nx:] if nb else np.zeros(0)
        _atol_r, _atol_a = _adaptive_tol(iteration, max_iter, rtol, atol)

        x_aug0 = np.concatenate([x_dyn_nominal, _IDENTITY_6_COL])
        x_aug_hist = propagate_augmented_state(
            t_pass_s,
            x_aug0,
            mu_moon_m3_s2,
            mu_earth_m3_s2,
            mu_sun_m3_s2,
            get_earth_pos,
            get_sun_pos,
            rtol=_atol_r,
            atol=_atol_a,
            j2_moon=j2_moon,
        )
        x_hist = x_aug_hist[:, :6]
        _, h_nom, h_tilde = compute_position_residuals_analytic(x_hist, obs_data, pass_geo)
        h_nom_aug = _apply_position_bias(h_nom, obs_data, b_nominal, bias_cfg)
        residual = _position_residual_from_h(obs_data, h_nom_aug)
        current_cost = float(np.dot(w_diag * residual, residual))

        if iteration == 1:
            best_cost = current_cost
            x_best = x_nominal.copy()

        h_initial_state = apply_stm_to_jacobian(obs_data, x_aug_hist, h_tilde, 3, 5)

        if nb:
            h_initial = np.hstack([h_initial_state, _position_bias_jacobian(obs_data, bias_cfg)])
        else:
            h_initial = h_initial_state

        h_scaled = h_initial @ scale
        weighted_h = h_scaled * w_sqrt[:, None]
        weighted_r = residual * w_sqrt
        z_bar = prior_sqrt_info @ (x_prior - x_nominal) if has_explicit_prior else np.zeros(na)
        combined = np.vstack(
            [
                np.column_stack([r_bar, z_bar]),
                np.column_stack([weighted_h, weighted_r]),
            ]
        )
        _, r_qr = np.linalg.qr(combined, mode="reduced")
        r_hat = r_qr[:na, :na]
        z_hat = r_qr[:na, na]
        last_condition_number = float(np.linalg.cond(r_hat))
        last_rank = int(np.linalg.matrix_rank(r_hat))

        if not np.all(np.isfinite(r_hat)) or last_condition_number > 1e14:
            stop_reason = "Singular"
            break

        step_bar = np.linalg.solve(r_hat, z_hat)
        step = scale @ step_bar
        pos_step_norm = float(np.linalg.norm(step[:3]))
        if pos_step_norm > 20000.0:
            step *= 20000.0 / pos_step_norm
        last_step = step

        x_candidate = x_nominal + step
        x_dyn_candidate = x_candidate[:nx]
        b_candidate = x_candidate[nx:] if nb else np.zeros(0)
        x_hist_candidate = propagate_state(
            t_pass_s, x_dyn_candidate, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
            get_earth_pos, get_sun_pos, rtol=_atol_r, atol=_atol_a, j2_moon=j2_moon,
        )
        _, h_candidate, _ = compute_position_residuals_analytic(x_hist_candidate, obs_data, pass_geo)
        h_candidate_aug = _apply_position_bias(h_candidate, obs_data, b_candidate, bias_cfg)
        residual_candidate = _position_residual_from_h(obs_data, h_candidate_aug)
        candidate_cost = float(np.dot(w_diag * residual_candidate, residual_candidate))

        if candidate_cost < current_cost:
            relative_improvement = abs(current_cost - candidate_cost) / max(current_cost, np.finfo(float).eps)
            x_nominal = x_candidate
            x_best = x_nominal.copy()
            best_cost = candidate_cost

            if relative_improvement < tol_cost_stability:
                stop_reason = "J-Stab"
                break
            if np.linalg.norm(step) < tol_step_norm:
                stop_reason = "Converged"
                break
        else:
            x_nominal = x_best.copy()

    posterior_information, posterior_covariance, posterior_sqrt_information = (None, None, None)
    if return_posterior:
        posterior_information = _position_posterior_information(
            t_pass_s,
            obs_data,
            x_best[:nx],
            pass_geo,
            mu_moon_m3_s2,
            mu_earth_m3_s2,
            mu_sun_m3_s2,
            get_earth_pos,
            get_sun_pos,
            bias_cfg,
            prior_inv,
            w_diag,
            rtol,
            atol,
        )
        posterior_covariance = _safe_covariance_from_information(posterior_information)
        posterior_sqrt_information = _position_posterior_sqrt_information(
            t_pass_s,
            obs_data,
            x_best[:nx],
            pass_geo,
            mu_moon_m3_s2,
            mu_earth_m3_s2,
            mu_sun_m3_s2,
            get_earth_pos,
            get_sun_pos,
            bias_cfg,
            prior_sqrt_info,
            w_diag,
            rtol,
            atol,
        )

    stats = EstimatorStats(
        iterations=iteration,
        final_cost=best_cost,
        position_step_norm_m=float(np.linalg.norm(last_step[:3])),
        velocity_step_norm_mps=float(np.linalg.norm(last_step[3:])),
        condition_number=last_condition_number,
        rank=last_rank,
        posterior_information=posterior_information,
        posterior_covariance=posterior_covariance,
        posterior_sqrt_information=posterior_sqrt_information,
    )
    return x_best, stop_reason, stats


def estimate_range_rate_srif(
    t_pass_s: ArrayLike,
    obs_data: ArrayLike,
    x_nominal0: ArrayLike,
    pass_geo: PassGeometry,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    *,
    max_iter: int = 40,
    tol_step_norm: float = 1e-8,
    tol_cost_stability: float = 1e-8,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    j2_moon: float = 0.0,
    bias_mode: str | None = None,
    robust_outlier_rejection: bool = False,
    outlier_sigma: float = 3.0,
    max_outlier_fraction: float = 0.30,
    prior_covariance: ArrayLike | None = None,
    prior_sqrt_information: ArrayLike | None = None,
    return_posterior: bool = False,
) -> tuple[np.ndarray, str, EstimatorStats]:
    """Range/range-rate/azimuth/elevation SRIF estimator.

    Supported solve-for layouts:
    - 6 elements: dynamic state only
    - 10 elements: dynamic state plus global `[range, rr, az, el]` bias
    - 6 + 2*num_stations elements with `bias_mode="station_angles"`
    - 6 + 4*num_stations elements with `bias_mode="station_full"`
    """
    t_pass_s = np.asarray(t_pass_s, dtype=float).reshape(-1)
    obs_data = np.asarray(obs_data, dtype=float)
    x_nominal = np.asarray(x_nominal0, dtype=float).reshape(-1).copy()
    nx = 6
    bias_cfg = _resolve_range_rate_bias_config(x_nominal.size, nx, len(pass_geo.stations), bias_mode)
    nb = bias_cfg["size"]
    na = x_nominal.size

    has_explicit_prior = prior_covariance is not None or prior_sqrt_information is not None
    x_prior = x_nominal.copy()
    prior_inv, prior_sqrt_info, scale, _, prior_sqrt_scaled = _prior_information_and_scale(
        nx,
        bias_cfg,
        _range_rate_bias_prior_and_scale,
        prior_covariance,
        prior_sqrt_information,
    )
    r_bar = prior_sqrt_scaled

    w_diag = _range_rate_weight_diagonal(obs_data, pass_geo)
    w_curr_diag = w_diag.copy()

    x_best = x_nominal.copy()
    best_cost = np.inf
    stop_reason = "MaxIter"
    last_step = np.zeros(6)
    last_condition_number = float("nan")
    last_rank = 0
    last_rejected_components = 0
    last_active_weight_fraction = 1.0
    max_rejected_components = 0
    min_active_weight_fraction = 1.0

    for iteration in range(1, max_iter + 1):
        x_dyn_nominal = x_nominal[:nx]
        b_nominal = x_nominal[nx:] if nb else np.zeros(0)
        _atol_r, _atol_a = _adaptive_tol(iteration, max_iter, rtol, atol)
        x_aug0 = np.concatenate([x_dyn_nominal, _IDENTITY_6_COL])
        x_aug_hist = propagate_augmented_state(
            t_pass_s,
            x_aug0,
            mu_moon_m3_s2,
            mu_earth_m3_s2,
            mu_sun_m3_s2,
            get_earth_pos,
            get_sun_pos,
            rtol=_atol_r,
            atol=_atol_a,
            j2_moon=j2_moon,
        )
        h_nom, h_initial_state = _range_rate_nominal_and_initial_jacobian(
            t_pass_s,
            obs_data,
            x_dyn_nominal,
            pass_geo,
            mu_moon_m3_s2,
            mu_earth_m3_s2,
            mu_sun_m3_s2,
            get_earth_pos,
            get_sun_pos,
            x_aug_hist,
            rtol,
            atol,
        )
        h_nom_aug = _apply_range_rate_bias(h_nom, obs_data, b_nominal, bias_cfg)
        residual = _range_rate_residual_from_h(obs_data, h_nom_aug)
        w_curr_diag, last_rejected_components, last_active_weight_fraction = _robust_weight_diagonal(
            w_diag,
            residual,
            iteration,
            enabled=robust_outlier_rejection,
            outlier_sigma=outlier_sigma,
            max_outlier_fraction=max_outlier_fraction,
        )
        w_sqrt = np.sqrt(w_curr_diag)
        max_rejected_components = max(max_rejected_components, last_rejected_components)
        min_active_weight_fraction = min(min_active_weight_fraction, last_active_weight_fraction)
        current_cost = float(np.dot(w_curr_diag * residual, residual))

        if iteration == 1:
            best_cost = current_cost
            x_best = x_nominal.copy()

        if nb:
            h_initial = np.hstack([h_initial_state, _range_rate_bias_jacobian(obs_data, bias_cfg)])
        else:
            h_initial = h_initial_state

        h_scaled = h_initial @ scale
        weighted_h = h_scaled * w_sqrt[:, None]
        weighted_r = residual * w_sqrt
        z_bar = prior_sqrt_info @ (x_prior - x_nominal) if has_explicit_prior else np.zeros(na)
        combined = np.vstack(
            [
                np.column_stack([r_bar, z_bar]),
                np.column_stack([weighted_h, weighted_r]),
            ]
        )
        _, r_qr = np.linalg.qr(combined, mode="reduced")
        r_hat = r_qr[:na, :na]
        z_hat = r_qr[:na, na]
        last_condition_number = float(np.linalg.cond(r_hat))
        last_rank = int(np.linalg.matrix_rank(r_hat))

        if not np.all(np.isfinite(r_hat)) or last_condition_number > 1e14:
            stop_reason = "Singular"
            break

        step_bar = np.linalg.solve(r_hat, z_hat)
        step = scale @ step_bar
        pos_step_norm = float(np.linalg.norm(step[:3]))
        if pos_step_norm > 20000.0:
            step *= 20000.0 / pos_step_norm
        last_step = step

        x_candidate = x_nominal + step
        x_dyn_candidate = x_candidate[:nx]
        b_candidate = x_candidate[nx:] if nb else np.zeros(0)
        x_hist_candidate = propagate_state(
            t_pass_s, x_dyn_candidate, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
            get_earth_pos, get_sun_pos, rtol=_atol_r, atol=_atol_a, j2_moon=j2_moon,
        )
        _, h_candidate = compute_range_rate_residuals(x_hist_candidate, obs_data, pass_geo)
        h_candidate_aug = _apply_range_rate_bias(h_candidate, obs_data, b_candidate, bias_cfg)
        residual_candidate = _range_rate_residual_from_h(obs_data, h_candidate_aug)
        candidate_cost = float(np.dot(w_curr_diag * residual_candidate, residual_candidate))

        if candidate_cost < current_cost:
            relative_improvement = abs(current_cost - candidate_cost) / max(current_cost, np.finfo(float).eps)
            x_nominal = x_candidate
            x_best = x_nominal.copy()
            best_cost = candidate_cost

            if relative_improvement < tol_cost_stability:
                stop_reason = "J-Stab"
                break
            if np.linalg.norm(step) < tol_step_norm:
                stop_reason = "Converged"
                break
        else:
            x_nominal = x_best.copy()

    posterior_information, posterior_covariance, posterior_sqrt_information = (None, None, None)
    if return_posterior:
        posterior_information = _range_rate_posterior_information(
            t_pass_s,
            obs_data,
            x_best[:nx],
            pass_geo,
            mu_moon_m3_s2,
            mu_earth_m3_s2,
            mu_sun_m3_s2,
            get_earth_pos,
            get_sun_pos,
            bias_cfg,
            prior_inv,
            w_curr_diag,
            rtol,
            atol,
        )
        posterior_covariance = _safe_covariance_from_information(posterior_information)
        posterior_sqrt_information = _range_rate_posterior_sqrt_information(
            t_pass_s,
            obs_data,
            x_best[:nx],
            pass_geo,
            mu_moon_m3_s2,
            mu_earth_m3_s2,
            mu_sun_m3_s2,
            get_earth_pos,
            get_sun_pos,
            bias_cfg,
            prior_sqrt_info,
            w_curr_diag,
            rtol,
            atol,
        )

    stats = EstimatorStats(
        iterations=iteration,
        final_cost=best_cost,
        position_step_norm_m=float(np.linalg.norm(last_step[:3])),
        velocity_step_norm_mps=float(np.linalg.norm(last_step[3:])),
        condition_number=last_condition_number,
        rank=last_rank,
        rejected_components=max_rejected_components,
        active_weight_fraction=min_active_weight_fraction,
        posterior_information=posterior_information,
        posterior_covariance=posterior_covariance,
        posterior_sqrt_information=posterior_sqrt_information,
    )
    return x_best, stop_reason, stats


def estimate_position_bls_lm(
    t_pass_s: ArrayLike,
    obs_data: ArrayLike,
    x_nominal0: ArrayLike,
    pass_geo: PassGeometry,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    *,
    max_iter: int = 60,
    tol_step_norm: float = 1e-8,
    tol_cost_stability: float = 1e-8,
    lambda0: float = 1e-2,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    j2_moon: float = 0.0,
    bias_mode: str | None = None,
    prior_covariance: ArrayLike | None = None,
    prior_sqrt_information: ArrayLike | None = None,
    return_posterior: bool = False,
) -> tuple[np.ndarray, str, EstimatorStats]:
    """Position-only batch least-squares with Levenberg-Marquardt damping."""
    t_pass_s = np.asarray(t_pass_s, dtype=float).reshape(-1)
    obs_data = np.asarray(obs_data, dtype=float)
    x_nominal = np.asarray(x_nominal0, dtype=float).reshape(-1).copy()

    nx = 6
    bias_cfg = _resolve_position_bias_config(x_nominal.size, nx, len(pass_geo.stations), bias_mode)
    nb = bias_cfg["size"]
    na = x_nominal.size

    has_explicit_prior = prior_covariance is not None or prior_sqrt_information is not None
    x_prior = x_nominal.copy()
    prior_inv, _, scale, prior_inv_scaled, _ = _prior_information_and_scale(
        nx,
        bias_cfg,
        _position_bias_prior_and_scale,
        prior_covariance,
        prior_sqrt_information,
    )
    w_diag = _position_weight_diagonal(obs_data, pass_geo)

    x_best = x_nominal.copy()
    best_cost = np.inf
    stop_reason = "MaxIter"
    lambda_damping = float(lambda0)
    last_step = np.zeros(na)
    last_condition_number = float("nan")
    last_rank = 0

    for iteration in range(1, max_iter + 1):
        x_dyn_nominal = x_nominal[:nx]
        b_nominal = x_nominal[nx:] if nb else np.zeros(0)
        _atol_r, _atol_a = _adaptive_tol(iteration, max_iter, rtol, atol)

        x_aug0 = np.concatenate([x_dyn_nominal, _IDENTITY_6_COL])
        x_aug_hist = propagate_augmented_state(
            t_pass_s,
            x_aug0,
            mu_moon_m3_s2,
            mu_earth_m3_s2,
            mu_sun_m3_s2,
            get_earth_pos,
            get_sun_pos,
            rtol=_atol_r,
            atol=_atol_a,
            j2_moon=j2_moon,
        )
        x_hist = x_aug_hist[:, :6]
        _, h_nom, h_tilde = compute_position_residuals_analytic(x_hist, obs_data, pass_geo)
        h_nom_aug = _apply_position_bias(h_nom, obs_data, b_nominal, bias_cfg)
        residual = _position_residual_from_h(obs_data, h_nom_aug)
        current_cost = float(np.dot(w_diag * residual, residual))

        if iteration == 1:
            best_cost = current_cost
            x_best = x_nominal.copy()

        h_initial_state = apply_stm_to_jacobian(obs_data, x_aug_hist, h_tilde, 3, 5)

        if nb:
            h_initial = np.hstack([h_initial_state, _position_bias_jacobian(obs_data, bias_cfg)])
        else:
            h_initial = h_initial_state

        step, last_condition_number, last_rank, singular = _lm_step(
            h_initial,
            residual,
            w_diag,
            scale,
            prior_inv_scaled,
            scale.T @ prior_inv @ (x_prior - x_nominal) if has_explicit_prior else np.zeros(na),
            lambda_damping,
        )
        if singular:
            stop_reason = "Singular"
            break

        step = _limit_step(step, pos_limit_m=20000.0)
        last_step = step

        x_candidate = x_nominal + step
        x_dyn_candidate = x_candidate[:nx]
        b_candidate = x_candidate[nx:] if nb else np.zeros(0)
        x_aug0_cand = np.concatenate([x_dyn_candidate, _IDENTITY_6_COL])
        x_aug_hist_cand = propagate_augmented_state(
            t_pass_s, x_aug0_cand, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
            get_earth_pos, get_sun_pos, rtol=_atol_r, atol=_atol_a, j2_moon=j2_moon,
        )
        _, h_candidate, _ = compute_position_residuals_analytic(x_aug_hist_cand[:, :6], obs_data, pass_geo)
        h_candidate_aug = _apply_position_bias(h_candidate, obs_data, b_candidate, bias_cfg)
        residual_candidate = _position_residual_from_h(obs_data, h_candidate_aug)
        candidate_cost = float(np.dot(w_diag * residual_candidate, residual_candidate))

        if candidate_cost < current_cost:
            relative_improvement = abs(current_cost - candidate_cost) / max(current_cost, np.finfo(float).eps)
            x_nominal = x_candidate
            x_best = x_nominal.copy()
            best_cost = candidate_cost
            lambda_damping = max(lambda_damping / 5.0, 1e-12)

            if relative_improvement < tol_cost_stability:
                stop_reason = "J-Stab"
                break
            if np.linalg.norm(step) < tol_step_norm:
                stop_reason = "Converged"
                break
        else:
            x_nominal = x_best.copy()
            lambda_damping *= 10.0
            if lambda_damping > 1e12:
                stop_reason = "Singular"
                break

    posterior_information, posterior_covariance = (None, None)
    if return_posterior:
        posterior_information = _position_posterior_information(
            t_pass_s,
            obs_data,
            x_best[:nx],
            pass_geo,
            mu_moon_m3_s2,
            mu_earth_m3_s2,
            mu_sun_m3_s2,
            get_earth_pos,
            get_sun_pos,
            bias_cfg,
            prior_inv,
            w_diag,
            rtol,
            atol,
        )
        posterior_covariance = _safe_covariance_from_information(posterior_information)

    stats = EstimatorStats(
        iterations=iteration,
        final_cost=best_cost,
        position_step_norm_m=float(np.linalg.norm(last_step[:3])),
        velocity_step_norm_mps=float(np.linalg.norm(last_step[3:6])),
        condition_number=last_condition_number,
        rank=last_rank,
        posterior_information=posterior_information,
        posterior_covariance=posterior_covariance,
    )
    return x_best, stop_reason, stats


def estimate_range_rate_bls_lm(
    t_pass_s: ArrayLike,
    obs_data: ArrayLike,
    x_nominal0: ArrayLike,
    pass_geo: PassGeometry,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    *,
    max_iter: int = 80,
    tol_step_norm: float = 1e-8,
    tol_cost_stability: float = 1e-8,
    lambda0: float = 1e-2,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    j2_moon: float = 0.0,
    bias_mode: str | None = None,
    robust_outlier_rejection: bool = False,
    outlier_sigma: float = 3.0,
    max_outlier_fraction: float = 0.30,
    prior_covariance: ArrayLike | None = None,
    prior_sqrt_information: ArrayLike | None = None,
    return_posterior: bool = False,
) -> tuple[np.ndarray, str, EstimatorStats]:
    """Range-rate batch least-squares with LM damping and optional outlier rejection."""
    t_pass_s = np.asarray(t_pass_s, dtype=float).reshape(-1)
    obs_data = np.asarray(obs_data, dtype=float)
    x_nominal = np.asarray(x_nominal0, dtype=float).reshape(-1).copy()

    nx = 6
    bias_cfg = _resolve_range_rate_bias_config(x_nominal.size, nx, len(pass_geo.stations), bias_mode)
    nb = bias_cfg["size"]
    na = x_nominal.size

    has_explicit_prior = prior_covariance is not None or prior_sqrt_information is not None
    x_prior = x_nominal.copy()
    prior_inv, _, scale, prior_inv_scaled, _ = _prior_information_and_scale(
        nx,
        bias_cfg,
        _range_rate_bias_prior_and_scale,
        prior_covariance,
        prior_sqrt_information,
    )
    w_diag = _range_rate_weight_diagonal(obs_data, pass_geo)
    w_curr_diag = w_diag.copy()

    x_best = x_nominal.copy()
    best_cost = np.inf
    stop_reason = "MaxIter"
    lambda_damping = float(lambda0)
    last_step = np.zeros(na)
    last_condition_number = float("nan")
    last_rank = 0
    last_rejected_components = 0
    last_active_weight_fraction = 1.0
    max_rejected_components = 0
    min_active_weight_fraction = 1.0

    for iteration in range(1, max_iter + 1):
        x_dyn_nominal = x_nominal[:nx]
        b_nominal = x_nominal[nx:] if nb else np.zeros(0)
        _atol_r, _atol_a = _adaptive_tol(iteration, max_iter, rtol, atol)

        x_aug0 = np.concatenate([x_dyn_nominal, _IDENTITY_6_COL])
        x_aug_hist = propagate_augmented_state(
            t_pass_s,
            x_aug0,
            mu_moon_m3_s2,
            mu_earth_m3_s2,
            mu_sun_m3_s2,
            get_earth_pos,
            get_sun_pos,
            rtol=_atol_r,
            atol=_atol_a,
            j2_moon=j2_moon,
        )

        h_nom, h_initial_state = _range_rate_nominal_and_initial_jacobian(
            t_pass_s,
            obs_data,
            x_dyn_nominal,
            pass_geo,
            mu_moon_m3_s2,
            mu_earth_m3_s2,
            mu_sun_m3_s2,
            get_earth_pos,
            get_sun_pos,
            x_aug_hist,
            rtol,
            atol,
        )
        h_nom_aug = _apply_range_rate_bias(h_nom, obs_data, b_nominal, bias_cfg)
        residual = _range_rate_residual_from_h(obs_data, h_nom_aug)
        w_curr_diag, last_rejected_components, last_active_weight_fraction = _robust_weight_diagonal(
            w_diag,
            residual,
            iteration,
            enabled=robust_outlier_rejection,
            outlier_sigma=outlier_sigma,
            max_outlier_fraction=max_outlier_fraction,
        )
        max_rejected_components = max(max_rejected_components, last_rejected_components)
        min_active_weight_fraction = min(min_active_weight_fraction, last_active_weight_fraction)
        current_cost = float(np.dot(w_curr_diag * residual, residual))

        if iteration == 1:
            best_cost = current_cost
            x_best = x_nominal.copy()

        if nb:
            h_initial = np.hstack([h_initial_state, _range_rate_bias_jacobian(obs_data, bias_cfg)])
        else:
            h_initial = h_initial_state

        step, last_condition_number, last_rank, singular = _lm_step(
            h_initial,
            residual,
            w_curr_diag,
            scale,
            prior_inv_scaled,
            scale.T @ prior_inv @ (x_prior - x_nominal) if has_explicit_prior else np.zeros(na),
            lambda_damping,
        )
        if singular:
            stop_reason = "Singular"
            break

        step = _limit_step(step, pos_limit_m=20000.0)
        last_step = step

        x_candidate = x_nominal + step
        x_dyn_candidate = x_candidate[:nx]
        b_candidate = x_candidate[nx:] if nb else np.zeros(0)
        x_aug0_cand = np.concatenate([x_dyn_candidate, _IDENTITY_6_COL])
        x_aug_hist_cand = propagate_augmented_state(
            t_pass_s, x_aug0_cand, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
            get_earth_pos, get_sun_pos, rtol=_atol_r, atol=_atol_a, j2_moon=j2_moon,
        )
        _, h_candidate = compute_range_rate_residuals(x_aug_hist_cand[:, :6], obs_data, pass_geo)
        h_candidate_aug = _apply_range_rate_bias(h_candidate, obs_data, b_candidate, bias_cfg)
        residual_candidate = _range_rate_residual_from_h(obs_data, h_candidate_aug)
        candidate_cost = float(np.dot(w_curr_diag * residual_candidate, residual_candidate))

        if candidate_cost < current_cost:
            relative_improvement = abs(current_cost - candidate_cost) / max(current_cost, np.finfo(float).eps)
            x_nominal = x_candidate
            x_best = x_nominal.copy()
            best_cost = candidate_cost
            lambda_damping = max(lambda_damping / 5.0, 1e-12)

            min_cost_stability_iteration = 3 if robust_outlier_rejection else 1
            if relative_improvement < tol_cost_stability and iteration >= min_cost_stability_iteration:
                stop_reason = "J-Stab"
                break
            if np.linalg.norm(step) < tol_step_norm:
                stop_reason = "Converged"
                break
        else:
            x_nominal = x_best.copy()
            lambda_damping *= 10.0
            if lambda_damping > 1e12:
                stop_reason = "Singular"
                break

    posterior_information, posterior_covariance = (None, None)
    if return_posterior:
        posterior_information = _range_rate_posterior_information(
            t_pass_s,
            obs_data,
            x_best[:nx],
            pass_geo,
            mu_moon_m3_s2,
            mu_earth_m3_s2,
            mu_sun_m3_s2,
            get_earth_pos,
            get_sun_pos,
            bias_cfg,
            prior_inv,
            w_curr_diag,
            rtol,
            atol,
        )
        posterior_covariance = _safe_covariance_from_information(posterior_information)

    stats = EstimatorStats(
        iterations=iteration,
        final_cost=best_cost,
        position_step_norm_m=float(np.linalg.norm(last_step[:3])),
        velocity_step_norm_mps=float(np.linalg.norm(last_step[3:6])),
        condition_number=last_condition_number,
        rank=last_rank,
        rejected_components=max_rejected_components,
        active_weight_fraction=min_active_weight_fraction,
        posterior_information=posterior_information,
        posterior_covariance=posterior_covariance,
    )
    return x_best, stop_reason, stats


def _prior_information_and_scale(
    nx: int,
    bias_cfg: dict,
    bias_prior_and_scale,
    prior_covariance: ArrayLike | None,
    prior_sqrt_information: ArrayLike | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nb = int(bias_cfg["size"])
    na = nx + nb
    dyn_prior_diag = [1.0 / 15000.0**2] * 3 + [1.0 / 5.0**2] * 3
    scale_diag = [1e6] * 3 + [1e3] * 3

    if nb:
        bias_prior_diag, bias_scale_diag = bias_prior_and_scale(bias_cfg)
        scale_diag.extend(bias_scale_diag)
    else:
        bias_prior_diag = []

    if prior_covariance is not None and prior_sqrt_information is not None:
        raise ValueError("Use either prior_covariance or prior_sqrt_information, not both.")

    if prior_sqrt_information is not None:
        sqrt_info = np.asarray(prior_sqrt_information, dtype=float)
        if sqrt_info.shape == (nx, nx):
            prior_sqrt_info = np.zeros((na, na), dtype=float)
            prior_sqrt_info[:nx, :nx] = sqrt_info
            if nb:
                prior_sqrt_info[nx:, nx:] = np.diag(np.sqrt(bias_prior_diag))
        elif sqrt_info.shape == (na, na):
            prior_sqrt_info = sqrt_info.copy()
        else:
            raise ValueError(
                f"prior_sqrt_information must have shape ({nx},{nx}) or ({na},{na})."
            )
        prior_inv = _symmetrize(prior_sqrt_info.T @ prior_sqrt_info)
    elif prior_covariance is None:
        prior_inv = np.diag(dyn_prior_diag + bias_prior_diag)
        prior_sqrt_info = np.diag(np.sqrt(dyn_prior_diag + bias_prior_diag))
    else:
        cov = np.asarray(prior_covariance, dtype=float)
        if cov.shape == (nx, nx):
            prior_inv = np.zeros((na, na), dtype=float)
            prior_inv[:nx, :nx] = _safe_information_from_covariance(cov)
            if nb:
                prior_inv[nx:, nx:] = np.diag(bias_prior_diag)
        elif cov.shape == (na, na):
            prior_inv = _safe_information_from_covariance(cov)
        else:
            raise ValueError(f"prior_covariance must have shape ({nx},{nx}) or ({na},{na}).")
        prior_sqrt_info = _sqrt_information_from_information(prior_inv)

    prior_inv = _symmetrize(prior_inv)
    scale = np.diag(scale_diag)
    prior_inv_scaled = scale.T @ prior_inv @ scale
    prior_sqrt_scaled = prior_sqrt_info @ scale
    return prior_inv, prior_sqrt_info, scale, prior_inv_scaled, prior_sqrt_scaled


def _position_posterior_information(
    t_pass_s: np.ndarray,
    obs_data: np.ndarray,
    x_dyn: np.ndarray,
    pass_geo: PassGeometry,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    bias_cfg: dict,
    prior_inv: np.ndarray,
    w_diag: np.ndarray,
    rtol: float,
    atol: float,
) -> np.ndarray:
    x_aug0 = np.concatenate([x_dyn, np.eye(6).reshape(-1, order="F")])
    x_aug_hist = propagate_augmented_state(
        t_pass_s,
        x_aug0,
        mu_moon_m3_s2,
        mu_earth_m3_s2,
        mu_sun_m3_s2,
        get_earth_pos,
        get_sun_pos,
        rtol=rtol,
        atol=atol,
    )
    _, _, h_tilde = compute_position_residuals_analytic(x_aug_hist[:, :6], obs_data, pass_geo)
    h_initial = np.zeros((obs_data.shape[0] * 3, 6), dtype=float)
    for obs_idx in range(obs_data.shape[0]):
        row0 = obs_idx * 3
        time_idx = int(obs_data[obs_idx, 5]) - 1
        phi_k = x_aug_hist[time_idx, 6:].reshape((6, 6), order="F")
        h_initial[row0 : row0 + 3, :] = h_tilde[row0 : row0 + 3, :] @ phi_k
    if bias_cfg["size"]:
        h_initial = np.hstack([h_initial, _position_bias_jacobian(obs_data, bias_cfg)])
    return _symmetrize(h_initial.T @ (w_diag[:, None] * h_initial) + prior_inv)


def _range_rate_nominal_and_initial_jacobian(
    t_pass_s: np.ndarray,
    obs_data: np.ndarray,
    x_dyn: np.ndarray,
    pass_geo: PassGeometry,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    x_aug_hist: np.ndarray | None,
    rtol: float,
    atol: float,
) -> tuple[np.ndarray, np.ndarray]:
    if x_aug_hist is None:
        x_aug0 = np.concatenate([x_dyn, np.eye(6).reshape(-1, order="F")])
        x_aug_hist = propagate_augmented_state(
            t_pass_s,
            x_aug0,
            mu_moon_m3_s2,
            mu_earth_m3_s2,
            mu_sun_m3_s2,
            get_earth_pos,
            get_sun_pos,
            rtol=rtol,
            atol=atol,
        )

    rr_physics = range_rate_physics_config(pass_geo.range_rate_physics)
    if rr_physics.mode == "geometric_instantaneous":
        _, h_nom, h_tilde = compute_range_rate_residuals_analytic(x_aug_hist[:, :6], obs_data, pass_geo)
        return h_nom, _range_rate_initial_jacobian_from_local(obs_data, x_aug_hist, h_tilde)

    _, h_nom = compute_range_rate_residuals(x_aug_hist[:, :6], obs_data, pass_geo)
    h_initial = _range_rate_two_way_analytic_initial_jacobian(obs_data, pass_geo, x_aug_hist, rr_physics)
    return h_nom, h_initial


def _range_rate_two_way_analytic_initial_jacobian(
    obs_data: np.ndarray,
    pass_geo: PassGeometry,
    x_aug_hist: np.ndarray,
    rr_physics: RangeRatePhysicsConfig,
) -> np.ndarray:
    geometric_pass_geo = replace(pass_geo, range_rate_physics=RangeRatePhysicsConfig())
    _, _, h_tilde = compute_range_rate_residuals_analytic(x_aug_hist[:, :6], obs_data, geometric_pass_geo)
    h_initial = _range_rate_initial_jacobian_from_local(obs_data, x_aug_hist, h_tilde)
    for obs_idx in range(obs_data.shape[0]):
        row0 = obs_idx * 4
        station_id = int(obs_data[obs_idx, 5]) - 1
        time_idx = int(obs_data[obs_idx, 6]) - 1
        station = pass_geo.stations[station_id]
        h_initial[row0 + 1, :] = two_way_counted_doppler_initial_state_jacobian(
            float(pass_geo.t_s[time_idx]),
            station,
            pass_geo.t_s,
            x_aug_hist,
            pass_geo.earth_pos_mci_m,
            pass_geo.earth_vel_mci_mps,
            pass_geo.x_j2000_to_itrf93,
            rr_physics,
        )
    return h_initial


def _range_rate_initial_jacobian_from_local(
    obs_data: np.ndarray,
    x_aug_hist: np.ndarray,
    h_tilde: np.ndarray,
) -> np.ndarray:
    return apply_stm_to_jacobian(obs_data, x_aug_hist, h_tilde, 4, 6)


def _range_rate_numerical_initial_jacobian(
    t_pass_s: np.ndarray,
    obs_data: np.ndarray,
    x_dyn: np.ndarray,
    pass_geo: PassGeometry,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    rtol: float,
    atol: float,
) -> np.ndarray:
    warnings.warn(
        "_range_rate_numerical_initial_jacobian runs 12 ODE integrations per call "
        "and is ~10x slower than the analytic path; use only in tests.",
        RuntimeWarning,
        stacklevel=2,
    )
    h_initial = np.zeros((obs_data.shape[0] * 4, 6), dtype=float)
    steps = _range_rate_initial_fd_steps(x_dyn)
    for col_idx, step in enumerate(steps):
        perturb = np.zeros(6, dtype=float)
        perturb[col_idx] = step
        h_plus = _range_rate_h_vector_from_initial(
            t_pass_s,
            obs_data,
            x_dyn + perturb,
            pass_geo,
            mu_moon_m3_s2,
            mu_earth_m3_s2,
            mu_sun_m3_s2,
            get_earth_pos,
            get_sun_pos,
            rtol,
            atol,
        )
        h_minus = _range_rate_h_vector_from_initial(
            t_pass_s,
            obs_data,
            x_dyn - perturb,
            pass_geo,
            mu_moon_m3_s2,
            mu_earth_m3_s2,
            mu_sun_m3_s2,
            get_earth_pos,
            get_sun_pos,
            rtol,
            atol,
        )
        delta = (h_plus - h_minus).reshape(-1, 4)
        delta[:, 2] = wrap_to_pi(delta[:, 2])
        delta[:, 3] = wrap_to_pi(delta[:, 3])
        h_initial[:, col_idx] = delta.reshape(-1) / (2.0 * step)
    return h_initial


def _range_rate_h_vector_from_initial(
    t_pass_s: np.ndarray,
    obs_data: np.ndarray,
    x_dyn: np.ndarray,
    pass_geo: PassGeometry,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    rtol: float,
    atol: float,
) -> np.ndarray:
    x_aug0 = np.concatenate([x_dyn, np.eye(6).reshape(-1, order="F")])
    x_aug_hist = propagate_augmented_state(
        t_pass_s,
        x_aug0,
        mu_moon_m3_s2,
        mu_earth_m3_s2,
        mu_sun_m3_s2,
        get_earth_pos,
        get_sun_pos,
        rtol=rtol,
        atol=atol,
    )
    _, h_meas = compute_range_rate_residuals(x_aug_hist[:, :6], obs_data, pass_geo)
    return h_meas.reshape(-1)


def _range_rate_initial_fd_steps(x_dyn: np.ndarray) -> np.ndarray:
    x_dyn = np.asarray(x_dyn, dtype=float).reshape(6)
    steps = np.empty(6, dtype=float)
    steps[:3] = np.maximum(np.abs(x_dyn[:3]) * 1e-7, 10.0)
    steps[3:] = np.maximum(np.abs(x_dyn[3:]) * 1e-7, 1e-3)
    return steps


def _range_rate_posterior_information(
    t_pass_s: np.ndarray,
    obs_data: np.ndarray,
    x_dyn: np.ndarray,
    pass_geo: PassGeometry,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    bias_cfg: dict,
    prior_inv: np.ndarray,
    w_diag: np.ndarray,
    rtol: float,
    atol: float,
) -> np.ndarray:
    x_aug0 = np.concatenate([x_dyn, np.eye(6).reshape(-1, order="F")])
    x_aug_hist = propagate_augmented_state(
        t_pass_s,
        x_aug0,
        mu_moon_m3_s2,
        mu_earth_m3_s2,
        mu_sun_m3_s2,
        get_earth_pos,
        get_sun_pos,
        rtol=rtol,
        atol=atol,
    )
    _, h_initial = _range_rate_nominal_and_initial_jacobian(
        t_pass_s,
        obs_data,
        x_dyn,
        pass_geo,
        mu_moon_m3_s2,
        mu_earth_m3_s2,
        mu_sun_m3_s2,
        get_earth_pos,
        get_sun_pos,
        x_aug_hist,
        rtol,
        atol,
    )
    if bias_cfg["size"]:
        h_initial = np.hstack([h_initial, _range_rate_bias_jacobian(obs_data, bias_cfg)])
    return _symmetrize(h_initial.T @ (w_diag[:, None] * h_initial) + prior_inv)


def _position_posterior_sqrt_information(
    t_pass_s: np.ndarray,
    obs_data: np.ndarray,
    x_dyn: np.ndarray,
    pass_geo: PassGeometry,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    bias_cfg: dict,
    prior_sqrt_info: np.ndarray,
    w_diag: np.ndarray,
    rtol: float,
    atol: float,
) -> np.ndarray:
    h_initial = _position_initial_jacobian(
        t_pass_s,
        obs_data,
        x_dyn,
        pass_geo,
        mu_moon_m3_s2,
        mu_earth_m3_s2,
        mu_sun_m3_s2,
        get_earth_pos,
        get_sun_pos,
        bias_cfg,
        rtol,
        atol,
    )
    rows = np.vstack([prior_sqrt_info, h_initial * np.sqrt(w_diag)[:, None]])
    return _upper_triangular_qr_factor(rows)


def _range_rate_posterior_sqrt_information(
    t_pass_s: np.ndarray,
    obs_data: np.ndarray,
    x_dyn: np.ndarray,
    pass_geo: PassGeometry,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    bias_cfg: dict,
    prior_sqrt_info: np.ndarray,
    w_diag: np.ndarray,
    rtol: float,
    atol: float,
) -> np.ndarray:
    h_initial = _range_rate_initial_jacobian(
        t_pass_s,
        obs_data,
        x_dyn,
        pass_geo,
        mu_moon_m3_s2,
        mu_earth_m3_s2,
        mu_sun_m3_s2,
        get_earth_pos,
        get_sun_pos,
        bias_cfg,
        rtol,
        atol,
    )
    rows = np.vstack([prior_sqrt_info, h_initial * np.sqrt(w_diag)[:, None]])
    return _upper_triangular_qr_factor(rows)


def _position_initial_jacobian(
    t_pass_s: np.ndarray,
    obs_data: np.ndarray,
    x_dyn: np.ndarray,
    pass_geo: PassGeometry,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    bias_cfg: dict,
    rtol: float,
    atol: float,
) -> np.ndarray:
    x_aug0 = np.concatenate([x_dyn, np.eye(6).reshape(-1, order="F")])
    x_aug_hist = propagate_augmented_state(
        t_pass_s,
        x_aug0,
        mu_moon_m3_s2,
        mu_earth_m3_s2,
        mu_sun_m3_s2,
        get_earth_pos,
        get_sun_pos,
        rtol=rtol,
        atol=atol,
    )
    _, _, h_tilde = compute_position_residuals_analytic(x_aug_hist[:, :6], obs_data, pass_geo)
    h_initial = apply_stm_to_jacobian(obs_data, x_aug_hist, h_tilde, 3, 5)
    if bias_cfg["size"]:
        h_initial = np.hstack([h_initial, _position_bias_jacobian(obs_data, bias_cfg)])
    return h_initial


def _range_rate_initial_jacobian(
    t_pass_s: np.ndarray,
    obs_data: np.ndarray,
    x_dyn: np.ndarray,
    pass_geo: PassGeometry,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    bias_cfg: dict,
    rtol: float,
    atol: float,
) -> np.ndarray:
    x_aug0 = np.concatenate([x_dyn, np.eye(6).reshape(-1, order="F")])
    x_aug_hist = propagate_augmented_state(
        t_pass_s,
        x_aug0,
        mu_moon_m3_s2,
        mu_earth_m3_s2,
        mu_sun_m3_s2,
        get_earth_pos,
        get_sun_pos,
        rtol=rtol,
        atol=atol,
    )
    _, h_initial = _range_rate_nominal_and_initial_jacobian(
        t_pass_s,
        obs_data,
        x_dyn,
        pass_geo,
        mu_moon_m3_s2,
        mu_earth_m3_s2,
        mu_sun_m3_s2,
        get_earth_pos,
        get_sun_pos,
        x_aug_hist,
        rtol,
        atol,
    )
    if bias_cfg["size"]:
        h_initial = np.hstack([h_initial, _range_rate_bias_jacobian(obs_data, bias_cfg)])
    return h_initial


def _safe_information_from_covariance(covariance: np.ndarray) -> np.ndarray:
    cov = _symmetrize(np.asarray(covariance, dtype=float))
    vals, vecs = np.linalg.eigh(cov)
    max_val = float(np.max(np.abs(vals))) if vals.size else 1.0
    floor = max(max_val * 1e-14, np.finfo(float).eps)
    vals = np.clip(vals, floor, None)
    return _symmetrize((vecs / vals) @ vecs.T)


def _sqrt_information_from_information(information: np.ndarray) -> np.ndarray:
    info = _symmetrize(np.asarray(information, dtype=float))
    vals, vecs = np.linalg.eigh(info)
    max_val = float(np.max(np.abs(vals))) if vals.size else 1.0
    floor = max(max_val * 1e-14, np.finfo(float).eps)
    if np.min(vals) <= floor:
        vals = np.clip(vals, floor, None)
        info = _symmetrize((vecs * vals) @ vecs.T)
    try:
        return np.linalg.cholesky(info).T
    except np.linalg.LinAlgError:
        vals = np.clip(vals, floor, None)
        sqrt_info = (np.sqrt(vals)[:, None] * vecs.T)
        return _upper_triangular_qr_factor(sqrt_info)


def _upper_triangular_qr_factor(rows: np.ndarray) -> np.ndarray:
    _, r_factor = np.linalg.qr(np.asarray(rows, dtype=float), mode="reduced")
    n_cols = rows.shape[1]
    r_factor = r_factor[:n_cols, :n_cols]
    signs = np.where(np.diag(r_factor) < 0.0, -1.0, 1.0)
    return signs[:, None] * r_factor


def _safe_covariance_from_information(information: np.ndarray) -> np.ndarray:
    info = _symmetrize(np.asarray(information, dtype=float))
    vals, vecs = np.linalg.eigh(info)
    max_val = float(np.max(np.abs(vals))) if vals.size else 1.0
    floor = max(max_val * 1e-14, np.finfo(float).eps)
    vals = np.clip(vals, floor, None)
    return _symmetrize((vecs / vals) @ vecs.T)


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _lm_step(
    h_initial: np.ndarray,
    residual: np.ndarray,
    w_diag: np.ndarray,
    scale: np.ndarray,
    prior_inv_scaled: np.ndarray,
    prior_rhs_scaled: np.ndarray,
    lambda_damping: float,
) -> tuple[np.ndarray, float, int, bool]:
    h_scaled = h_initial @ scale
    weighted_h = h_scaled * w_diag[:, None]
    atwa = h_scaled.T @ weighted_h
    atwb = h_scaled.T @ (w_diag * residual) + prior_rhs_scaled
    damping_diag = np.diag(np.diag(atwa))
    if not np.any(np.diag(damping_diag)):
        damping_diag = np.eye(atwa.shape[0])
    normal_matrix = atwa + lambda_damping * damping_diag + prior_inv_scaled
    condition_number = float(np.linalg.cond(normal_matrix))
    rank = int(np.linalg.matrix_rank(normal_matrix))
    if not np.all(np.isfinite(normal_matrix)) or condition_number > 1e15:
        return np.zeros(h_initial.shape[1]), condition_number, rank, True
    try:
        step_bar = np.linalg.solve(normal_matrix, atwb)
    except np.linalg.LinAlgError:
        return np.zeros(h_initial.shape[1]), condition_number, rank, True
    return scale @ step_bar, condition_number, rank, False


def _limit_step(step: np.ndarray, *, pos_limit_m: float) -> np.ndarray:
    step = np.asarray(step, dtype=float).copy()
    pos_step_norm = float(np.linalg.norm(step[:3]))
    if pos_step_norm > pos_limit_m:
        step *= pos_limit_m / pos_step_norm
    return step


def _robust_weight_diagonal(
    base_w_diag: np.ndarray,
    residual: np.ndarray,
    iteration: int,
    *,
    enabled: bool,
    outlier_sigma: float,
    max_outlier_fraction: float,
) -> tuple[np.ndarray, int, float]:
    base_w_diag = np.asarray(base_w_diag, dtype=float)
    if not enabled or iteration <= 2:
        return base_w_diag.copy(), 0, 1.0

    sigma = np.full(base_w_diag.shape, np.inf, dtype=float)
    valid = base_w_diag > 0.0
    sigma[valid] = 1.0 / np.sqrt(base_w_diag[valid])
    normalized = np.abs(residual) / sigma
    bad = normalized > outlier_sigma
    rejected = int(np.sum(bad))
    max_rejected = int(np.floor(max_outlier_fraction * base_w_diag.size))
    if rejected == 0 or rejected > max_rejected:
        return base_w_diag.copy(), 0, 1.0

    w_diag = base_w_diag.copy()
    w_diag[bad] = 0.0
    active_fraction = float(np.count_nonzero(w_diag > 0.0) / w_diag.size)
    return w_diag, rejected, active_fraction


def _position_weight_diagonal(obs_data: np.ndarray, pass_geo: PassGeometry) -> np.ndarray:
    sigma = measurement_sigma_vector(obs_data, pass_geo, "position")
    return 1.0 / sigma**2


def _range_rate_weight_diagonal(obs_data: np.ndarray, pass_geo: PassGeometry) -> np.ndarray:
    sigma = measurement_sigma_vector(obs_data, pass_geo, "range_rate")
    return 1.0 / sigma**2


def _position_residual_from_h(obs_data: np.ndarray, h_meas: np.ndarray) -> np.ndarray:
    diff = obs_data[:, 1:4] - h_meas
    diff[:, 1] = np.arctan2(np.sin(diff[:, 1]), np.cos(diff[:, 1]))
    diff[:, 2] = np.arctan2(np.sin(diff[:, 2]), np.cos(diff[:, 2]))
    return diff.T.reshape(-1, order="F")


def _range_rate_residual_from_h(obs_data: np.ndarray, h_meas: np.ndarray) -> np.ndarray:
    diff = obs_data[:, 1:5] - h_meas
    diff[:, 2] = np.arctan2(np.sin(diff[:, 2]), np.cos(diff[:, 2]))
    diff[:, 3] = np.arctan2(np.sin(diff[:, 3]), np.cos(diff[:, 3]))
    return diff.T.reshape(-1, order="F")


def _resolve_range_rate_bias_config(
    solve_for_size: int,
    nx: int,
    num_stations: int,
    bias_mode: str | None,
) -> dict:
    if solve_for_size < nx:
        raise ValueError("x_nominal0 must contain at least the 6 dynamic state elements.")

    if bias_mode is None:
        if solve_for_size == nx:
            bias_mode = "none"
        elif solve_for_size == nx + 4:
            bias_mode = "global_full"
        else:
            raise ValueError(
                "Provide bias_mode for station-specific RR solve-for vectors. "
                "Supported modes: station_angles, station_full."
            )

    bias_mode = bias_mode.lower()
    if bias_mode == "global_rr_full":
        bias_mode = "global_full"
    elif bias_mode == "station_rr_full":
        bias_mode = "station_full"

    if bias_mode == "none":
        expected = nx
        block_size = 0
    elif bias_mode == "global_full":
        expected = nx + 4
        block_size = 4
    elif bias_mode == "station_angles":
        expected = nx + 2 * num_stations
        block_size = 2
    elif bias_mode == "station_full":
        expected = nx + 4 * num_stations
        block_size = 4
    else:
        raise ValueError(f"Unsupported RR bias mode: {bias_mode}")

    if solve_for_size != expected:
        raise ValueError(f"bias_mode={bias_mode} expects solve-for size {expected}, got {solve_for_size}.")

    return {
        "mode": bias_mode,
        "size": solve_for_size - nx,
        "block_size": block_size,
        "num_stations": num_stations,
    }


def _range_rate_bias_prior_and_scale(bias_cfg: dict) -> tuple[list[float], list[float]]:
    if bias_cfg["mode"] == "none":
        return [], []

    sig_range = 100.0
    sig_rr = 1e-2
    sig_angle = np.deg2rad(0.05)

    if bias_cfg["mode"] in {"global_full", "station_full"}:
        prior_block = [1.0 / sig_range**2, 1.0 / sig_rr**2, 1.0 / sig_angle**2, 1.0 / sig_angle**2]
        scale_block = [1e2, 1e-3, 1e-5, 1e-5]
    elif bias_cfg["mode"] == "station_angles":
        prior_block = [1.0 / sig_angle**2, 1.0 / sig_angle**2]
        scale_block = [1e-5, 1e-5]
    else:
        return [], []

    n_blocks = 1 if bias_cfg["mode"] == "global_full" else bias_cfg["num_stations"]
    return prior_block * n_blocks, scale_block * n_blocks


def _apply_range_rate_bias(
    h_meas: np.ndarray,
    obs_data: np.ndarray,
    bias: np.ndarray,
    bias_cfg: dict,
) -> np.ndarray:
    if bias.size == 0 or bias_cfg["mode"] == "none":
        return h_meas

    h_aug = h_meas.copy()
    if bias_cfg["mode"] == "global_full":
        return h_aug + bias.reshape(1, 4)

    for obs_idx in range(obs_data.shape[0]):
        station_id = int(obs_data[obs_idx, 5]) - 1
        if bias_cfg["mode"] == "station_angles":
            col0 = station_id * 2
            h_aug[obs_idx, 2:4] += bias[col0 : col0 + 2]
        elif bias_cfg["mode"] == "station_full":
            col0 = station_id * 4
            h_aug[obs_idx, :] += bias[col0 : col0 + 4]
    return h_aug


def _range_rate_bias_jacobian(obs_data: np.ndarray, bias_cfg: dict) -> np.ndarray:
    n_obs = obs_data.shape[0]
    if bias_cfg["mode"] == "global_full":
        return np.tile(np.eye(4), (n_obs, 1))
    if bias_cfg["mode"] == "station_angles":
        hb = np.zeros((n_obs * 4, bias_cfg["size"]), dtype=float)
        for obs_idx in range(n_obs):
            row0 = obs_idx * 4
            station_id = int(obs_data[obs_idx, 5]) - 1
            col0 = station_id * 2
            hb[row0 + 2, col0] = 1.0
            hb[row0 + 3, col0 + 1] = 1.0
        return hb
    if bias_cfg["mode"] == "station_full":
        hb = np.zeros((n_obs * 4, bias_cfg["size"]), dtype=float)
        for obs_idx in range(n_obs):
            row0 = obs_idx * 4
            station_id = int(obs_data[obs_idx, 5]) - 1
            col0 = station_id * 4
            hb[row0 : row0 + 4, col0 : col0 + 4] = np.eye(4)
        return hb
    return np.zeros((n_obs * 4, 0), dtype=float)


def _resolve_position_bias_config(
    solve_for_size: int,
    nx: int,
    num_stations: int,
    bias_mode: str | None,
) -> dict:
    if solve_for_size < nx:
        raise ValueError("x_nominal0 must contain at least the 6 dynamic state elements.")

    if bias_mode is None:
        if solve_for_size == nx:
            bias_mode = "none"
        elif solve_for_size == nx + 3:
            bias_mode = "global_full"
        else:
            raise ValueError(
                "Provide bias_mode for station-specific solve-for vectors. "
                "Supported modes: station_angles, station_full."
            )

    bias_mode = bias_mode.lower()
    if bias_mode == "none":
        expected = nx
        block_size = 0
    elif bias_mode == "global_full":
        expected = nx + 3
        block_size = 3
    elif bias_mode == "station_angles":
        expected = nx + 2 * num_stations
        block_size = 2
    elif bias_mode == "station_full":
        expected = nx + 3 * num_stations
        block_size = 3
    else:
        raise ValueError(f"Unsupported position bias mode: {bias_mode}")

    if solve_for_size != expected:
        raise ValueError(f"bias_mode={bias_mode} expects solve-for size {expected}, got {solve_for_size}.")

    return {
        "mode": bias_mode,
        "size": solve_for_size - nx,
        "block_size": block_size,
        "num_stations": num_stations,
    }


def _position_bias_prior_and_scale(bias_cfg: dict) -> tuple[list[float], list[float]]:
    sig_range = 100.0
    sig_angle = np.deg2rad(0.05)

    if bias_cfg["mode"] in {"global_full", "station_full"}:
        prior_block = [1.0 / sig_range**2, 1.0 / sig_angle**2, 1.0 / sig_angle**2]
        scale_block = [1e2, 1e-5, 1e-5]
    elif bias_cfg["mode"] == "station_angles":
        prior_block = [1.0 / sig_angle**2, 1.0 / sig_angle**2]
        scale_block = [1e-5, 1e-5]
    else:
        return [], []

    n_blocks = 1 if bias_cfg["mode"] == "global_full" else bias_cfg["num_stations"]
    return prior_block * n_blocks, scale_block * n_blocks


def _apply_position_bias(h_meas: np.ndarray, obs_data: np.ndarray, bias: np.ndarray, bias_cfg: dict) -> np.ndarray:
    if bias.size == 0 or bias_cfg["mode"] == "none":
        return h_meas

    h_aug = h_meas.copy()
    mode = bias_cfg["mode"]
    if mode == "global_full":
        return h_aug + bias.reshape(1, 3)

    for obs_idx in range(obs_data.shape[0]):
        station_id = int(obs_data[obs_idx, 4]) - 1
        if mode == "station_angles":
            col0 = station_id * 2
            h_aug[obs_idx, 1:3] += bias[col0 : col0 + 2]
        elif mode == "station_full":
            col0 = station_id * 3
            h_aug[obs_idx, :] += bias[col0 : col0 + 3]
    return h_aug


def _position_bias_jacobian(obs_data: np.ndarray, bias_cfg: dict) -> np.ndarray:
    n_obs = obs_data.shape[0]
    mode = bias_cfg["mode"]

    if mode == "global_full":
        return np.tile(np.eye(3), (n_obs, 1))
    if mode == "station_angles":
        hb = np.zeros((n_obs * 3, bias_cfg["size"]), dtype=float)
        for obs_idx in range(n_obs):
            row0 = obs_idx * 3
            station_id = int(obs_data[obs_idx, 4]) - 1
            col0 = station_id * 2
            hb[row0 + 1, col0] = 1.0
            hb[row0 + 2, col0 + 1] = 1.0
        return hb
    if mode == "station_full":
        hb = np.zeros((n_obs * 3, bias_cfg["size"]), dtype=float)
        for obs_idx in range(n_obs):
            row0 = obs_idx * 3
            station_id = int(obs_data[obs_idx, 4]) - 1
            col0 = station_id * 3
            hb[row0 : row0 + 3, col0 : col0 + 3] = np.eye(3)
        return hb
    return np.zeros((n_obs * 3, 0), dtype=float)
