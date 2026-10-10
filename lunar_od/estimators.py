"""Estimator helpers for the Python Lunar OD port."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike
# Phase 17-R1COV.  scipy is already a hard dependency of lunar_od (dynamics,
# filters, ephemeris), so this adds no new requirement; it is imported here for
# the triangular solves of the square-root covariance path, which must never
# form or invert the normal matrix.
from scipy.linalg import solve_triangular

from .accelerated import apply_stm_to_jacobian
from .dynamics import (
    propagate_augmented_state,
    propagate_state,
    propagate_state_with_k_sensitivity,
)
from .geometry import wrap_to_pi
from .measurements import (
    PassGeometry,
    compute_position_residuals_analytic,
    position_initial_state_jacobian_from_augmented_history,
)
from .measurements import compute_range_rate_residuals
from .measurements import compute_range_rate_residuals_analytic, measurement_sigma_vector
from .radiometrics import RangeRatePhysicsConfig, range_rate_physics_config
from .radiometrics import two_way_counted_doppler_initial_state_jacobian
from .two_way_range import (
    compute_two_way_range_residuals,
    two_way_range_nominal_and_initial_jacobian,
)


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
    termination_detail: str = ""
    #: Phase 17-R (opt-in K_SRP solve-for). None unless solve_for_k_srp=True
    #: was passed to the estimator; existing callers are unaffected.
    k_srp_estimate: float | None = None
    k_srp_prior_status: str | None = None
    k_srp_negative_step_rejections: int = 0


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

        h_initial_state = _position_initial_state_jacobian(
            obs_data, x_aug_hist, h_tilde, pass_geo
        )

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
            j2_moon=j2_moon,
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
            j2_moon=j2_moon,
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
            j2_moon=j2_moon,
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
            j2_moon=j2_moon,
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

        h_initial_state = _position_initial_state_jacobian(
            obs_data, x_aug_hist, h_tilde, pass_geo
        )

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
                stop_reason = "DampingLimit"
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
            j2_moon=j2_moon,
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
                stop_reason = "DampingLimit"
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
            j2_moon=j2_moon,
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
    *,
    j2_moon: float = 0.0,
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
        j2_moon=j2_moon,
    )
    _, _, h_tilde = compute_position_residuals_analytic(x_aug_hist[:, :6], obs_data, pass_geo)
    h_initial = _position_initial_state_jacobian(
        obs_data, x_aug_hist, h_tilde, pass_geo
    )
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
    *,
    j2_moon: float = 0.0,
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
            j2_moon=j2_moon,
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
            # R3: transport the scenario ET origin so this route resolves the
            # same station-state method the observable used (R3-P09).
            et0_s=pass_geo.et0_s,
        )
    return h_initial


def _range_rate_initial_jacobian_from_local(
    obs_data: np.ndarray,
    x_aug_hist: np.ndarray,
    h_tilde: np.ndarray,
) -> np.ndarray:
    return apply_stm_to_jacobian(obs_data, x_aug_hist, h_tilde, 4, 6)


def _position_initial_state_jacobian(
    obs_data: np.ndarray,
    x_aug_hist: np.ndarray,
    h_tilde: np.ndarray,
    pass_geo: PassGeometry,
) -> np.ndarray:
    """Apply the shared position initial-state mapping contract."""
    return position_initial_state_jacobian_from_augmented_history(
        obs_data, x_aug_hist, h_tilde, pass_geo
    )


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
    *,
    j2_moon: float = 0.0,
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
        j2_moon=j2_moon,
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
        j2_moon=j2_moon,
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
    *,
    j2_moon: float = 0.0,
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
        j2_moon=j2_moon,
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
    *,
    j2_moon: float = 0.0,
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
        j2_moon=j2_moon,
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
    *,
    j2_moon: float = 0.0,
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
        j2_moon=j2_moon,
    )
    _, _, h_tilde = compute_position_residuals_analytic(x_aug_hist[:, :6], obs_data, pass_geo)
    h_initial = _position_initial_state_jacobian(
        obs_data, x_aug_hist, h_tilde, pass_geo
    )
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
    *,
    j2_moon: float = 0.0,
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
        j2_moon=j2_moon,
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
        j2_moon=j2_moon,
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
    if not np.all(np.isfinite(info)):
        raise ValueError(
            "Posterior information matrix contains nonfinite values; "
            "covariance decomposition was not attempted."
        )
    vals, vecs = np.linalg.eigh(info)
    max_val = float(np.max(np.abs(vals))) if vals.size else 1.0
    floor = max(max_val * 1e-14, np.finfo(float).eps)
    vals = np.clip(vals, floor, None)
    return _symmetrize((vecs / vals) @ vecs.T)


class RankDeficientCovarianceError(np.linalg.LinAlgError):
    """No finite covariance exists for at least one direction of the problem.

    Phase 17-R1COV.  Raised in preference to returning a plausible-looking
    finite matrix: a caller told "no covariance exists" can act correctly,
    whereas a caller handed a floor-derived number cannot tell that anything
    went wrong.
    """

    def __init__(self, message: str, *, rank: int, n: int,
                 singular_ratio: float, threshold: float):
        super().__init__(message)
        self.rank = rank
        self.n = n
        self.singular_ratio = singular_ratio
        self.threshold = threshold


def _square_root_covariance_from_design(
    h_full: np.ndarray,
    w_diag: np.ndarray,
    prior_inv: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase 17-R1COV.  Posterior covariance without forming H^T W H.

    Returns ``(covariance_physical, sqrt_information_physical)``.

    WHY THIS EXISTS.  ``_safe_covariance_from_information`` recovers covariance
    by eigendecomposing the normal matrix and clipping eigenvalues at
    ``max_eig * 1e-14``.  For the K-augmented problem that clip is not a
    safety net, it is the answer: R1M measured that the reported sigma_K is
    reproduced to 1.5e-10 by ``sqrt(K_scale**2 / floor)`` alone, and that it
    scales linearly with the arbitrary K bookkeeping constant while every
    physical quantity stays invariant.

    The root cause is that forming the normal matrix squares the condition
    number exactly -- measured ratio cond(H^T W H) / cond(H_w)**2 = 1.0000 on
    every tested arc.  The whitened, scaled DESIGN matrix is conditioned at
    1e8-2.6e9 against a double-precision limit of 1/eps = 4.5e15, so the weak
    K direction is comfortably resolvable; squaring pushes it past the limit.

    This routine therefore never forms the normal matrix.  It builds

        A = [ W^(1/2) H S ; L^T ]      with  L L^T = S^T P0^-1 S

    factors A = Q R, and recovers P = R^-1 R^-T by two triangular solves.

    Qualified in Phase 17-R1COV against an EXACT rational-arithmetic oracle
    (agreement <= 1.6e-13 on sigma_K and <= 6.8e-12 across the full
    covariance), for scaling invariance (spread 0.0 across K scales spanning
    4x), for correct prior response, and against a sequentially accumulated
    SRIF R factor.

    Note that ``_sqrt_information_from_information`` cannot serve this purpose:
    it applies the SAME eigenvalue floor and then factors the already-floored
    matrix, so it is a square root OF the defect (measured 90-99.6% error).
    """
    h_full = np.asarray(h_full, dtype=float)
    w_diag = np.asarray(w_diag, dtype=float)
    n = h_full.shape[1]

    rows = [np.sqrt(w_diag)[:, None] * (h_full @ scale)]
    if prior_inv is not None:
        prior_scaled = _symmetrize(scale.T @ np.asarray(prior_inv, float) @ scale)
        # eigendecomposition rather than Cholesky: the prior information matrix
        # is routinely only SEMI-definite here (data-only is all zeros, and a
        # partial prior constrains some parameters and not others), and
        # Cholesky raises on both.
        p_vals, p_vecs = np.linalg.eigh(prior_scaled)
        keep = p_vals > 0.0
        if np.any(keep):
            rows.append(np.sqrt(p_vals[keep])[:, None] * p_vecs[:, keep].T)
    a_aug = np.vstack(rows)

    r_factor = np.asarray(np.linalg.qr(a_aug, mode="r"))[:n, :n]
    signs = np.where(np.diag(r_factor) < 0.0, -1.0, 1.0)
    r_factor = signs[:, None] * r_factor

    # Rank test on the SINGULAR VALUES of R (a triangular factor's diagonal can
    # be arbitrarily unrepresentative of them).  The criterion is a pure ratio,
    # hence invariant to any uniform rescaling -- precisely the scale-awareness
    # the absolute eigenvalue floor lacked.  On the weak-K lunar arcs the ratio
    # is ~3.8e-10 against a threshold of ~1.6e-15, five orders of margin, so
    # the weak direction is retained on its own merits.
    sv = np.linalg.svd(r_factor, compute_uv=False)
    threshold = float(n) * float(np.finfo(float).eps)
    ratio = float(sv[-1] / sv[0]) if sv[0] > 0.0 else 0.0
    rank = int(np.sum(sv > sv[0] * threshold))
    if rank < n or not np.all(np.isfinite(r_factor)):
        raise RankDeficientCovarianceError(
            "augmented problem is numerically rank %d of %d "
            "(sigma_min/sigma_max = %.3e, resolvability threshold %.3e); "
            "no finite covariance exists for the unresolved direction"
            % (rank, n, ratio, threshold),
            rank=rank, n=n, singular_ratio=ratio, threshold=threshold)

    eye = np.eye(n)
    y_factor = solve_triangular(r_factor, eye, trans="T", lower=False)
    cov_scaled = solve_triangular(r_factor, y_factor, lower=False)
    covariance = scale @ _symmetrize(cov_scaled) @ scale.T

    # R is expressed in scaled coordinates; map it back so the reported
    # sqrt-information satisfies R_phys^T R_phys = posterior_information.
    # scale is diagonal, so dividing columns preserves upper-triangularity.
    sqrt_information = r_factor / np.diag(scale)[None, :]
    return _symmetrize(covariance), sqrt_information


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


# ---------------------------------------------------------------------------
# Two-way range (M3): scalar converged-event observable.
#
# Both estimators consume the same (N, 6) arc-initial-state Jacobian from
# two_way_range_nominal_and_initial_jacobian; the rows already embed the
# event-epoch STMs, so no STM is applied here.  Bias solve-for states are not
# supported for this observable in M3 (deferred to the noise/bias phase).
# ---------------------------------------------------------------------------

_TWO_WAY_RANGE_NO_BIAS_CFG: dict = {"size": 0, "mode": None}

#: Phase 17-R.  K_SRP solve-for is a separate opt-in mechanism from the M3
#: bias states above: it is activated by srp=/solve_for_k_srp=, never by
#: bias_mode, and x_nominal0 always stays a 6-vector (K is tracked alongside,
#: not concatenated into it).  See estimate_two_way_range_bls_lm.


def _two_way_range_k_srp_column(
    obs_data: np.ndarray,
    x_aug_hist: np.ndarray,
    h_initial: np.ndarray,
) -> np.ndarray:
    """K_SRP design-matrix column for the two-way range Jacobian.

    Reuses the Phase-17A-R-qualified composition, rather than re-deriving the
    event-solver mathematics (out of Phase 17-R's scope):

        H_K(t3_i) = H_x0(t3_i) . Phi(t3_i)^-1 . S_K(t3_i)

    ``H_x0`` is the already-qualified range Jacobian returned by
    ``two_way_range_nominal_and_initial_jacobian``; ``Phi`` and ``S_K`` are the
    already-qualified STM and trajectory sensitivity from
    ``propagate_state_with_k_sensitivity``.  ``obs_data`` column 3 is the
    1-based grid index used to GENERATE each receive tag, so ``Phi``/``S_K``
    are read at the exact epoch, not interpolated.  Phase 17A-R independently
    validated this composition against the production observable evaluated at
    perturbed inputs (never against this same formula) to a best relative
    error of about 4.6e-09 (measurement composition) and 4.9e-08 (full
    end-to-end), with a resolved plateau-then-roundoff convergence structure
    across 14 finite-difference steps.
    """
    time_idx = np.asarray(obs_data[:, 3], dtype=int) - 1
    phi_all = x_aug_hist[:, 6:42]
    s_k_all = x_aug_hist[:, 42:48]
    h_k = np.zeros(obs_data.shape[0], dtype=float)
    for i, row in enumerate(time_idx):
        phi_i = phi_all[row].reshape((6, 6), order="F")
        v_i = np.linalg.solve(phi_i, s_k_all[row])
        h_k[i] = float(h_initial[i] @ v_i)
    return h_k


def _two_way_range_bias_prior_and_scale(bias_cfg: dict) -> tuple[list[float], list[float]]:
    raise ValueError("two_way_range estimation does not support bias solve-for states.")


def _two_way_range_weight_diagonal(obs_data: np.ndarray, pass_geo: PassGeometry) -> np.ndarray:
    sigma = measurement_sigma_vector(obs_data, pass_geo, "two_way_range")
    return 1.0 / sigma**2


def _two_way_range_residual_from_h(obs_data: np.ndarray, h_meas: np.ndarray) -> np.ndarray:
    return np.asarray(obs_data[:, 1], dtype=float) - np.asarray(h_meas, dtype=float).reshape(-1)


def _validate_two_way_range_solve_for(x_nominal: np.ndarray, bias_mode: str | None) -> None:
    if bias_mode is not None:
        raise ValueError(
            "two_way_range estimation does not support bias_mode in M3; got "
            f"{bias_mode!r}."
        )
    if x_nominal.size != 6:
        raise ValueError(
            "two_way_range estimation solves for the 6-element dynamic state only; "
            f"got a {x_nominal.size}-element solve-for vector."
        )


def _two_way_range_posterior_information(
    t_pass_s: np.ndarray,
    obs_data: np.ndarray,
    x_dyn: np.ndarray,
    pass_geo: PassGeometry,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    prior_inv: np.ndarray,
    w_diag: np.ndarray,
    rtol: float,
    atol: float,
    j2_moon: float,
) -> np.ndarray:
    x_aug0 = np.concatenate([x_dyn, _IDENTITY_6_COL])
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
        j2_moon=j2_moon,
    )
    _, h_initial = two_way_range_nominal_and_initial_jacobian(obs_data, pass_geo, x_aug_hist)
    return _symmetrize(h_initial.T @ (w_diag[:, None] * h_initial) + prior_inv)


def _two_way_range_posterior_information_with_k(
    t_pass_s: np.ndarray,
    obs_data: np.ndarray,
    x_dyn: np.ndarray,
    k_srp: float,
    srp,
    pass_geo: PassGeometry,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    prior_inv: np.ndarray,
    w_diag: np.ndarray,
    rtol: float,
    atol: float,
    j2_moon: float,
) -> np.ndarray:
    """Phase 17-R.  (7,7) posterior information at the converged (x0, K)."""
    h_full = _two_way_range_posterior_design_with_k(
        t_pass_s, obs_data, x_dyn, k_srp, srp, pass_geo,
        mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
        get_earth_pos, get_sun_pos, rtol, atol, j2_moon,
    )
    return _symmetrize(h_full.T @ (w_diag[:, None] * h_full) + prior_inv)


def _two_way_range_posterior_design_with_k(
    t_pass_s: np.ndarray,
    obs_data: np.ndarray,
    x_dyn: np.ndarray,
    k_srp: float,
    srp,
    pass_geo: PassGeometry,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    rtol: float,
    atol: float,
    j2_moon: float,
) -> np.ndarray:
    """Phase 17-R1COV.  The (M,7) augmented design matrix at the solution.

    Split out of ``_two_way_range_posterior_information_with_k`` so the
    square-root covariance path can reach the design rows WITHOUT a second
    trajectory propagation, and without the information matrix ever being
    formed on its behalf.  The arithmetic of the information matrix is
    unchanged by the split, so ``posterior_information`` stays bitwise
    identical to Phase 17-R.
    """
    x_aug_hist = propagate_state_with_k_sensitivity(
        t_pass_s, x_dyn, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
        get_earth_pos, get_sun_pos, srp=replace(srp, k_srp_m2_per_kg=k_srp),
        rtol=rtol, atol=atol, j2_moon=j2_moon,
    )
    _, h_x0 = two_way_range_nominal_and_initial_jacobian(obs_data, pass_geo, x_aug_hist)
    h_k = _two_way_range_k_srp_column(obs_data, x_aug_hist, h_x0)
    return np.hstack([h_x0, h_k[:, None]])


def estimate_two_way_range_bls_lm(
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
    harmonic_model=None,
    harmonic_epoch_et0: float | None = None,
    srp=None,
    solve_for_k_srp: bool = False,
    k_srp_initial: float | None = None,
    k_srp_prior_sigma: float | None = None,
    k_srp_scale: float = 1e-2,
) -> tuple[np.ndarray, str, EstimatorStats]:
    """Two-way range batch least-squares with LM damping.

    Lunar spherical harmonics (default-off, opt-in): pass ``harmonic_model``
    plus ``harmonic_epoch_et0`` (the SPICE ET at ``t = 0``) to run the estimator
    on a real GRAIL field evaluated in the exact-epoch ``MOON_PA_DE440`` frame.
    Both the nominal 42-state propagation AND the LM candidate propagation
    receive the SAME model, and the augmented pass sets
    ``harmonic_stm_opt_in=True`` so the variational equations use the MATCHED
    analytic Pines gradient rather than a lower-order one. Leaving the model
    ``None`` keeps the historical central/J2 behaviour bit-for-bit.

    Phase 17-R (opt-in K_SRP solve-for).  Three independent switches (s13):

    - ``srp=None`` (default): SRP off, identical to every prior phase.
    - ``srp=SRPOptions(...)``, ``solve_for_k_srp=False``: SRP on with a FIXED
      coefficient; the estimator still solves the 6-state orbit only.
    - ``srp=SRPOptions(...)``, ``solve_for_k_srp=True``: K_SRP becomes a 7th
      solve-for scalar.  ``x_nominal0`` stays a 6-vector always; K is tracked
      alongside it and returned in ``stats.k_srp_estimate`` /
      ``stats.posterior_covariance`` (7x7), never concatenated into the
      returned state.  This keeps ``x_best`` a 6-vector in every mode, so
      existing callers are structurally unaffected.

    ``k_srp_initial`` sets the solve's starting K (distinct from
    ``srp.k_srp_m2_per_kg``, which stays the FIXED value when
    ``solve_for_k_srp=False``); if omitted, the solve starts from
    ``srp.k_srp_m2_per_kg``.  ``k_srp_prior_sigma`` is an explicit
    QUALIFICATION_PRIOR sigma on K -- omit it for a data-only solve (s23).
    ``k_srp_scale`` is a NUMERICAL scale factor only (s19/s20); it carries no
    prior information and must not change the physical estimate.

    Production SRP rejects negative K (fail-closed).  A trial LM step whose
    candidate K would go negative is REJECTED without evaluating the
    observable -- not clipped to zero, which would bias the estimate low (s18)
    -- via the same damping-increase path already used for a rejected step.
    """
    _harm = (
        {} if harmonic_model is None
        else {"harmonic_model": harmonic_model,
              "harmonic_epoch_et0": harmonic_epoch_et0}
    )
    t_pass_s = np.asarray(t_pass_s, dtype=float).reshape(-1)
    obs_data = np.asarray(obs_data, dtype=float)
    x_nominal = np.asarray(x_nominal0, dtype=float).reshape(-1).copy()
    _validate_two_way_range_solve_for(x_nominal, bias_mode)

    if solve_for_k_srp:
        if srp is None or not srp.enabled:
            raise ValueError(
                "solve_for_k_srp=True requires an active srp=SRPOptions(...). "
                "SRP-enabled and K-solve-for-enabled are independent switches "
                "(Phase 17-R s13); solving for K without an active SRP force "
                "has no meaning."
            )
        k_current = (float(k_srp_initial) if k_srp_initial is not None
                     else float(srp.k_srp_m2_per_kg))
        if k_current < 0.0:
            raise ValueError(f"k_srp_initial/srp.k_srp_m2_per_kg must be non-negative; got {k_current}.")
        if k_srp_prior_sigma is not None and not (
            math.isfinite(k_srp_prior_sigma) and k_srp_prior_sigma > 0.0
        ):
            raise ValueError("k_srp_prior_sigma must be a finite positive number.")
        if not (math.isfinite(k_srp_scale) and k_srp_scale > 0.0):
            raise ValueError("k_srp_scale must be a finite positive number.")
    else:
        if k_srp_initial is not None or k_srp_prior_sigma is not None:
            raise ValueError(
                "k_srp_initial/k_srp_prior_sigma require solve_for_k_srp=True."
            )
        k_current = None

    nx = 6
    na = 7 if solve_for_k_srp else 6
    has_explicit_prior = prior_covariance is not None or prior_sqrt_information is not None
    k_prior_active = solve_for_k_srp and k_srp_prior_sigma is not None
    prior_active = has_explicit_prior or k_prior_active
    x_prior = x_nominal.copy()
    k_prior_mean = k_current  # the solve's own starting point IS the prior mean
    prior_inv6, _, scale6, _, _ = _prior_information_and_scale(
        nx,
        _TWO_WAY_RANGE_NO_BIAS_CFG,
        _two_way_range_bias_prior_and_scale,
        prior_covariance,
        prior_sqrt_information,
    )
    if solve_for_k_srp:
        prior_inv = np.zeros((7, 7), dtype=float)
        prior_inv[:6, :6] = prior_inv6
        if k_prior_active:
            prior_inv[6, 6] = 1.0 / float(k_srp_prior_sigma) ** 2
        prior_inv = _symmetrize(prior_inv)
        scale = np.diag(list(np.diag(scale6)) + [float(k_srp_scale)])
    else:
        prior_inv, scale = prior_inv6, scale6
    prior_inv_scaled = scale.T @ prior_inv @ scale
    w_diag = _two_way_range_weight_diagonal(obs_data, pass_geo)
    w_curr_diag = w_diag.copy()

    x_best = x_nominal.copy()
    k_best = k_current
    best_cost = np.inf
    stop_reason = "MaxIter"
    termination_detail = "MAX_ITER"
    lambda_damping = float(lambda0)
    last_step = np.zeros(na)
    last_condition_number = float("nan")
    last_rank = 0
    max_rejected_components = 0
    min_active_weight_fraction = 1.0
    negative_k_rejections = 0

    def _active_srp(k_state):
        if srp is None:
            return None
        if not srp.enabled:
            return srp
        return replace(srp, k_srp_m2_per_kg=k_state if k_state is not None else srp.k_srp_m2_per_kg)

    def _acceptance_residual(x_state, k_state=None):
        """Residual used ONLY for the LM accept/reject comparison.

        PHASE 6.2 repair.  Previously the current state was scored from the
        42-state augmented propagation while the candidate was scored from a
        plain 6-state propagation.  Both carried the same nominal rtol label,
        but the augmented integrator also error-controls the 36 STM components
        and so takes a far smaller step sequence: measured 0.049-0.449 m
        trajectory error against 1.06-6.75 m for the 6-state path at the same
        label.  Near the solution, where residuals are small, that made a
        candidate look up to 1900x worse than the incumbent regardless of the
        step, so every candidate was rejected, lambda escalated to the 1e12
        guard, and the solve stopped with a "Singular" label despite a
        well-conditioned system.

        Both states now go through this one path at the CALLER'S requested
        tolerance, so the comparison is like-for-like.  The augmented pass keeps
        its own adaptive tolerance for the Jacobian, which does not need this
        accuracy.  Consequence: identical states produce identical costs.
        """
        hist = propagate_state(
            t_pass_s, x_state, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
            get_earth_pos, get_sun_pos, rtol=rtol, atol=atol, j2_moon=j2_moon,
            srp=_active_srp(k_state), **_harm,
        )
        res, _ = compute_two_way_range_residuals(hist, obs_data, pass_geo)
        return np.asarray(res, dtype=float)

    def _acceptance_cost(x_state, k_state, residual_vec, weight_diag):
        """Accept/reject scalar: the objective _lm_step actually minimises.

        PHASE 6.2 repair.  The LM normal equations carry both the
        measurement information and the prior information, so they are
        Gauss-Newton for the MAP objective

            J_MAP = 0.5 r^T W r + 0.5 (x - x_prior)^T P0^-1 (x - x_prior).

        Acceptance used to score only the measurement half, so a step the
        normal equations had proposed as a MAP descent could be rejected,
        and a step that worsened J_MAP could be accepted.  Both were
        observed in a real solve, at 0.13-0.14 m steps.

        The returned scalar is 2 * J_MAP: the measurement term keeps the
        factor of two it already carried and the prior term is given the
        same one, so only the objective changes, not its scaling.

        Phase 17-R: extended to the 7-parameter [x0, K] state.  When K carries
        no explicit prior, prior_inv's K row/column is exactly zero, so this
        reduces algebraically to the original 6-state formula -- no
        special-casing, and default invariance follows from that identity
        rather than from a separate code path.
        """
        cost = float(np.dot(weight_diag * residual_vec, residual_vec))
        if prior_active:
            delta = np.asarray(x_state, dtype=float) - x_prior
            if solve_for_k_srp:
                delta = np.concatenate([delta, [k_state - k_prior_mean]])
            cost += float(delta @ prior_inv @ delta)
        return cost

    for iteration in range(1, max_iter + 1):
        _atol_r, _atol_a = _adaptive_tol(iteration, max_iter, rtol, atol)
        if solve_for_k_srp:
            x_aug_hist = propagate_state_with_k_sensitivity(
                t_pass_s, x_nominal, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
                get_earth_pos, get_sun_pos, srp=replace(srp, k_srp_m2_per_kg=k_current),
                rtol=_atol_r, atol=_atol_a, j2_moon=j2_moon,
                **({} if not _harm else {**_harm, "harmonic_stm_opt_in": True}),
            )
            h_nom, h_x0 = two_way_range_nominal_and_initial_jacobian(obs_data, pass_geo, x_aug_hist)
            h_k = _two_way_range_k_srp_column(obs_data, x_aug_hist, h_x0)
            h_initial = np.hstack([h_x0, h_k[:, None]])
        else:
            x_aug0 = np.concatenate([x_nominal, _IDENTITY_6_COL])
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
                srp=srp,
                **({} if not _harm else {**_harm, "harmonic_stm_opt_in": True}),
            )
            h_nom, h_initial = two_way_range_nominal_and_initial_jacobian(
                obs_data, pass_geo, x_aug_hist
            )
        residual = _two_way_range_residual_from_h(obs_data, h_nom)
        w_curr_diag, rejected, active_fraction = _robust_weight_diagonal(
            w_diag,
            residual,
            iteration,
            enabled=robust_outlier_rejection,
            outlier_sigma=outlier_sigma,
            max_outlier_fraction=max_outlier_fraction,
        )
        max_rejected_components = max(max_rejected_components, rejected)
        min_active_weight_fraction = min(min_active_weight_fraction, active_fraction)
        _acc_residual = _acceptance_residual(x_nominal, k_current)
        current_cost = _acceptance_cost(x_nominal, k_current, _acc_residual, w_curr_diag)

        if iteration == 1:
            best_cost = current_cost
            x_best = x_nominal.copy()
            k_best = k_current

        x_prior_full = (np.concatenate([x_prior, [k_prior_mean]]) if solve_for_k_srp
                        else x_prior)
        x_current_full = (np.concatenate([x_nominal, [k_current]]) if solve_for_k_srp
                          else x_nominal)
        prior_rhs_scaled = (scale.T @ prior_inv @ (x_prior_full - x_current_full)
                            if prior_active else np.zeros(na))
        step, last_condition_number, last_rank, singular = _lm_step(
            h_initial, residual, w_curr_diag, scale, prior_inv_scaled,
            prior_rhs_scaled, lambda_damping,
        )
        if singular and solve_for_k_srp:
            # K's information content is routinely ~1e13-1e16x smaller than
            # the orbital state's in atwa units (range is extremely sensitive
            # to position, only weakly and indirectly to K), so lambda0's
            # damping can leave the matrix singular on the very FIRST attempt,
            # before the escalation loop below ever gets a chance to run (a
            # "singular" result breaks immediately, unlike a rejected step).
            # LM damping resolves any conditioning for a large enough lambda
            # whenever the diagonal is nonzero (diagonally-dominant limit), so
            # retry with escalating lambda before concluding genuine failure.
            # The na=6 path above is untouched: this retry is unreachable
            # there, so default behaviour keeps its original single-attempt
            # semantics exactly.
            retry_lambda = lambda_damping
            while singular and retry_lambda <= 1e12:
                retry_lambda *= 10.0
                step, last_condition_number, last_rank, singular = _lm_step(
                    h_initial, residual, w_curr_diag, scale, prior_inv_scaled,
                    prior_rhs_scaled, retry_lambda,
                )
            if not singular:
                lambda_damping = retry_lambda
        if singular:
            stop_reason = "Singular"
            termination_detail = "MATRIX_SINGULAR"
            break

        step = _limit_step(step, pos_limit_m=20000.0)
        last_step = step

        x_candidate = x_nominal + step[:6]
        if solve_for_k_srp:
            k_candidate = k_current + step[6]
            if k_candidate < 0.0:
                # s18: domain-infeasible trial point. REJECT without ever
                # calling propagate() with a negative K (which SRPOptions
                # would raise on) -- never clip toward zero, which would bias
                # the estimate low. Same LM safeguard as a worse-cost step.
                negative_k_rejections += 1
                x_nominal = x_best.copy()
                k_current = k_best
                lambda_damping *= 10.0
                if lambda_damping > 1e12:
                    stop_reason = "DampingLimit"
                    termination_detail = "LAMBDA_HARD_LIMIT_NEGATIVE_K"
                    break
                continue
        else:
            k_candidate = None
        residual_candidate = _acceptance_residual(x_candidate, k_candidate)
        candidate_cost = _acceptance_cost(x_candidate, k_candidate, residual_candidate, w_curr_diag)

        if candidate_cost < current_cost:
            relative_improvement = abs(current_cost - candidate_cost) / max(current_cost, np.finfo(float).eps)
            x_nominal = x_candidate
            if solve_for_k_srp:
                k_current = k_candidate
            x_best = x_nominal.copy()
            k_best = k_current
            best_cost = candidate_cost
            lambda_damping = max(lambda_damping / 5.0, 1e-12)

            min_cost_stability_iteration = 3 if robust_outlier_rejection else 1
            if relative_improvement < tol_cost_stability and iteration >= min_cost_stability_iteration:
                stop_reason = "J-Stab"
                termination_detail = "COST_STABLE"
                break
            if np.linalg.norm(step) < tol_step_norm:
                stop_reason = "Converged"
                termination_detail = "STEP_NORM"
                break
        else:
            x_nominal = x_best.copy()
            k_current = k_best
            lambda_damping *= 10.0
            if lambda_damping > 1e12:
                # PHASE 6.2: damping exhaustion, NOT a singular linear solve.
                # _lm_step reported no singularity to reach here; the LM step
                # simply stopped being accepted.  The 1e12 limit is unchanged.
                stop_reason = "DampingLimit"
                termination_detail = "LAMBDA_HARD_LIMIT"
                break

    posterior_information, posterior_covariance = (None, None)
    if return_posterior:
        if solve_for_k_srp:
            # Phase 17-R1COV.  The design matrix is read once and used for BOTH
            # the reported information matrix and the covariance.
            #
            # Phase 17-R conditioned on `scale` before the floored inverse and
            # believed that sufficed.  Phase 17-R1M showed it does not: the
            # reported sigma_K was reproduced to 1.5e-10 by the eigenvalue
            # floor alone and moved linearly with the arbitrary K bookkeeping
            # scale, i.e. it was an artifact rather than an uncertainty.  The
            # cause is that forming H^T W H squares the condition number
            # exactly, pushing a resolvable K direction past double precision.
            #
            # The covariance is therefore taken from an orthogonal
            # factorization of the design matrix, which never forms the normal
            # matrix.  posterior_information is still reported, unchanged, for
            # callers that want it -- it is only no longer INVERTED.
            posterior_design = _two_way_range_posterior_design_with_k(
                t_pass_s, obs_data, x_best, k_best, srp, pass_geo,
                mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
                get_earth_pos, get_sun_pos, rtol, atol, j2_moon,
            )
            posterior_information = _symmetrize(
                posterior_design.T @ (w_curr_diag[:, None] * posterior_design)
                + prior_inv
            )
            posterior_covariance, _ = _square_root_covariance_from_design(
                posterior_design, w_curr_diag, prior_inv, scale
            )
        else:
            posterior_information = _two_way_range_posterior_information(
                t_pass_s,
                obs_data,
                x_best,
                pass_geo,
                mu_moon_m3_s2,
                mu_earth_m3_s2,
                mu_sun_m3_s2,
                get_earth_pos,
                get_sun_pos,
                prior_inv,
                w_curr_diag,
                rtol,
                atol,
                j2_moon,
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
        termination_detail=termination_detail,
        k_srp_estimate=k_best if solve_for_k_srp else None,
        k_srp_prior_status=(
            ("QUALIFICATION_PRIOR" if k_prior_active else "DATA_ONLY")
            if solve_for_k_srp else None
        ),
        k_srp_negative_step_rejections=negative_k_rejections,
    )
    return x_best, stop_reason, stats


def estimate_two_way_range_srif(
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
    srp=None,
    solve_for_k_srp: bool = False,
    k_srp_initial: float | None = None,
    k_srp_prior_sigma: float | None = None,
    k_srp_scale: float = 1e-2,
) -> tuple[np.ndarray, str, EstimatorStats]:
    """Two-way range square-root information filter estimator.

    Phase 17-R (opt-in K_SRP solve-for): same three independent switches as
    ``estimate_two_way_range_bls_lm`` (s13), and the same contract -- K is
    tracked alongside the 6-vector state, never concatenated into it.

    Unlike BLS-LM, this SRIF carries no Levenberg-Marquardt damping: a
    rejected step is simply retried at the same linearization next
    iteration. K's information content is routinely orders of magnitude
    smaller than the orbital state's (measured in the Phase 17-R campaign
    report), so an UNREGULARIZED K column here would frequently leave
    ``r_hat`` singular with no escape.  ``k_srp_prior_sigma`` is therefore
    REQUIRED (not optional) when solving for K -- exactly the s31 SRIF prior
    semantics: "sequential filtering requires an initial uncertainty."
    """
    t_pass_s = np.asarray(t_pass_s, dtype=float).reshape(-1)
    obs_data = np.asarray(obs_data, dtype=float)
    x_nominal = np.asarray(x_nominal0, dtype=float).reshape(-1).copy()
    _validate_two_way_range_solve_for(x_nominal, bias_mode)

    if solve_for_k_srp:
        if srp is None or not srp.enabled:
            raise ValueError(
                "solve_for_k_srp=True requires an active srp=SRPOptions(...) "
                "(Phase 17-R s13)."
            )
        if k_srp_prior_sigma is None or not (
            math.isfinite(k_srp_prior_sigma) and k_srp_prior_sigma > 0.0
        ):
            raise ValueError(
                "k_srp_prior_sigma is required for SRIF K_SRP solve-for (s31): "
                "sequential filtering requires an initial uncertainty; this "
                "estimator carries no LM damping to fall back on."
            )
        k_current = (float(k_srp_initial) if k_srp_initial is not None
                     else float(srp.k_srp_m2_per_kg))
        if k_current < 0.0:
            raise ValueError(f"k_srp_initial/srp.k_srp_m2_per_kg must be non-negative; got {k_current}.")
        if not (math.isfinite(k_srp_scale) and k_srp_scale > 0.0):
            raise ValueError("k_srp_scale must be a finite positive number.")
    else:
        if k_srp_initial is not None or k_srp_prior_sigma is not None:
            raise ValueError(
                "k_srp_initial/k_srp_prior_sigma require solve_for_k_srp=True."
            )
        k_current = None

    nx = 6
    na = 7 if solve_for_k_srp else 6
    has_explicit_prior = prior_covariance is not None or prior_sqrt_information is not None
    x_prior = x_nominal.copy()
    k_prior_mean = k_current
    prior_inv6, prior_sqrt_info6, scale6, _, prior_sqrt_scaled6 = _prior_information_and_scale(
        nx,
        _TWO_WAY_RANGE_NO_BIAS_CFG,
        _two_way_range_bias_prior_and_scale,
        prior_covariance,
        prior_sqrt_information,
    )
    if solve_for_k_srp:
        prior_inv = np.zeros((7, 7), dtype=float)
        prior_inv[:6, :6] = prior_inv6
        prior_inv[6, 6] = 1.0 / float(k_srp_prior_sigma) ** 2
        prior_inv = _symmetrize(prior_inv)
        prior_sqrt_info = np.zeros((7, 7), dtype=float)
        prior_sqrt_info[:6, :6] = prior_sqrt_info6
        prior_sqrt_info[6, 6] = 1.0 / float(k_srp_prior_sigma)
        scale = np.diag(list(np.diag(scale6)) + [float(k_srp_scale)])
        r_bar = prior_sqrt_info @ scale
    else:
        prior_inv, scale = prior_inv6, scale6
        prior_sqrt_info = prior_sqrt_info6
        r_bar = prior_sqrt_scaled6

    w_diag = _two_way_range_weight_diagonal(obs_data, pass_geo)
    w_curr_diag = w_diag.copy()

    x_best = x_nominal.copy()
    k_best = k_current
    best_cost = np.inf
    stop_reason = "MaxIter"
    last_step = np.zeros(na)
    last_condition_number = float("nan")
    last_rank = 0
    max_rejected_components = 0
    min_active_weight_fraction = 1.0
    negative_k_rejections = 0

    def _active_srp(k_state):
        if srp is None:
            return None
        if not srp.enabled:
            return srp
        return replace(srp, k_srp_m2_per_kg=k_state if k_state is not None else srp.k_srp_m2_per_kg)

    for iteration in range(1, max_iter + 1):
        _atol_r, _atol_a = _adaptive_tol(iteration, max_iter, rtol, atol)
        if solve_for_k_srp:
            x_aug_hist = propagate_state_with_k_sensitivity(
                t_pass_s, x_nominal, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
                get_earth_pos, get_sun_pos, srp=replace(srp, k_srp_m2_per_kg=k_current),
                rtol=_atol_r, atol=_atol_a, j2_moon=j2_moon,
            )
            h_nom, h_x0 = two_way_range_nominal_and_initial_jacobian(obs_data, pass_geo, x_aug_hist)
            h_k = _two_way_range_k_srp_column(obs_data, x_aug_hist, h_x0)
            h_initial = np.hstack([h_x0, h_k[:, None]])
        else:
            x_aug0 = np.concatenate([x_nominal, _IDENTITY_6_COL])
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
                srp=srp,
            )
            h_nom, h_initial = two_way_range_nominal_and_initial_jacobian(
                obs_data, pass_geo, x_aug_hist
            )
        residual = _two_way_range_residual_from_h(obs_data, h_nom)
        w_curr_diag, rejected, active_fraction = _robust_weight_diagonal(
            w_diag,
            residual,
            iteration,
            enabled=robust_outlier_rejection,
            outlier_sigma=outlier_sigma,
            max_outlier_fraction=max_outlier_fraction,
        )
        w_sqrt = np.sqrt(w_curr_diag)
        max_rejected_components = max(max_rejected_components, rejected)
        min_active_weight_fraction = min(min_active_weight_fraction, active_fraction)
        current_cost = float(np.dot(w_curr_diag * residual, residual))

        if iteration == 1:
            best_cost = current_cost
            x_best = x_nominal.copy()
            k_best = k_current

        h_scaled = h_initial @ scale
        weighted_h = h_scaled * w_sqrt[:, None]
        weighted_r = residual * w_sqrt
        x_prior_full = (np.concatenate([x_prior, [k_prior_mean]]) if solve_for_k_srp
                        else x_prior)
        x_current_full = (np.concatenate([x_nominal, [k_current]]) if solve_for_k_srp
                          else x_nominal)
        z_bar = (prior_sqrt_info @ (x_prior_full - x_current_full)
                 if (has_explicit_prior or solve_for_k_srp) else np.zeros(na))
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

        if solve_for_k_srp:
            # s18: no LM damping to lean on here, so an infeasible trial K is
            # handled by backtracking ALONG THE SAME QR-computed direction
            # (halving magnitude, never flipping sign or clipping toward
            # zero) until the domain is respected -- a standard trust-region
            # safeguard, not a change to what the step means statistically.
            attempts = 0
            while k_current + step[6] < 0.0 and attempts < 40:
                step = step * 0.5
                attempts += 1
            if k_current + step[6] < 0.0:
                stop_reason = "NegativeKDomainLimit"
                break
            if attempts > 0:
                negative_k_rejections += 1
        last_step = step

        x_candidate = x_nominal + step[:6]
        k_candidate = k_current + step[6] if solve_for_k_srp else None
        if solve_for_k_srp:
            hist_srp = replace(srp, k_srp_m2_per_kg=k_candidate)
        else:
            hist_srp = srp
        x_hist_candidate = propagate_state(
            t_pass_s, x_candidate, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
            get_earth_pos, get_sun_pos, rtol=_atol_r, atol=_atol_a, j2_moon=j2_moon,
            srp=hist_srp,
        )
        residual_candidate, _ = compute_two_way_range_residuals(
            x_hist_candidate, obs_data, pass_geo
        )
        candidate_cost = float(np.dot(w_curr_diag * residual_candidate, residual_candidate))

        if candidate_cost < current_cost:
            relative_improvement = abs(current_cost - candidate_cost) / max(current_cost, np.finfo(float).eps)
            x_nominal = x_candidate
            if solve_for_k_srp:
                k_current = k_candidate
            x_best = x_nominal.copy()
            k_best = k_current
            best_cost = candidate_cost

            if relative_improvement < tol_cost_stability:
                stop_reason = "J-Stab"
                break
            if np.linalg.norm(step) < tol_step_norm:
                stop_reason = "Converged"
                break
        else:
            x_nominal = x_best.copy()
            k_current = k_best

    posterior_information, posterior_covariance, posterior_sqrt_information = (None, None, None)
    if return_posterior:
        if solve_for_k_srp:
            # Phase 17-R1COV.  Same square-root covariance path as the BLS
            # branch (see its comment for the full rationale).  Unreachable for
            # solve_for_k_srp=False.
            posterior_design = _two_way_range_posterior_design_with_k(
                t_pass_s, obs_data, x_best, k_best, srp, pass_geo,
                mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
                get_earth_pos, get_sun_pos, rtol, atol, j2_moon,
            )
            posterior_information = _symmetrize(
                posterior_design.T @ (w_curr_diag[:, None] * posterior_design)
                + prior_inv
            )
            posterior_covariance, k_sqrt_information = (
                _square_root_covariance_from_design(
                    posterior_design, w_curr_diag, prior_inv, scale
                )
            )
        else:
            posterior_information = _two_way_range_posterior_information(
                t_pass_s,
                obs_data,
                x_best,
                pass_geo,
                mu_moon_m3_s2,
                mu_earth_m3_s2,
                mu_sun_m3_s2,
                get_earth_pos,
                get_sun_pos,
                prior_inv,
                w_curr_diag,
                rtol,
                atol,
                j2_moon,
            )
            posterior_covariance = _safe_covariance_from_information(posterior_information)
            k_sqrt_information = None
        # Phase 17-R1COV.  _sqrt_information_from_information applies the SAME
        # eigenvalue floor and then factors the already-floored matrix, so on
        # the K path it is a square root OF the defect (measured 90-99.6%
        # error against an exact oracle).  On that path the genuine R factor
        # from the design-matrix QR is used instead.  The six-state path is
        # untouched: shipping a qualified covariance beside a contaminated
        # sqrt-information in the same stats object would be a trap.
        posterior_sqrt_information = (
            k_sqrt_information if k_sqrt_information is not None
            else _sqrt_information_from_information(posterior_information)
        )

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
        posterior_sqrt_information=posterior_sqrt_information,
        k_srp_estimate=k_best if solve_for_k_srp else None,
        k_srp_prior_status="QUALIFICATION_PRIOR" if solve_for_k_srp else None,
        k_srp_negative_step_rejections=negative_k_rejections,
    )
    return x_best, stop_reason, stats
