"""Batch least squares with dynamic model compensation (WP4.8d, EST-21).

A batch fit has no process noise, so the Gauss-Markov empirical acceleration of
``dmc.py`` enters it the standard way (Tapley, Schutz & Born 2004, section 4.9): the
estimated parameters are the epoch state ``[r0, v0, w0]``, ``w`` decays
deterministically as ``w(t) = w0 exp(-t / tau)`` (in inertial or RTN axes), and the
9x9 STM of :func:`propagate_dmc_state` maps the local measurement partials to the
epoch.  ``w0`` carries the prior ``N(0, sigma^2)`` of the Gauss-Markov steady state
(per axis with ``sigma_axes_mps2``); ``r0, v0`` carry only the caller's prior, if any.

Two solvers share one objective, one damping schedule and the WP4.2 certificate:
``method="normal"`` solves the damped normal equations (BLS-LM, :func:`_lm_step`),
``method="square_root"`` the same system by QR without forming them
(:func:`_lm_step_square_root`, the SRIF-form solve).  Position observations
(:func:`estimate_position_dmc_batch`) and instantaneous range-rate observations
(:func:`estimate_range_rate_dmc_batch`, WP4.8e); the measurement partials are mapped
through the receive-epoch STM, so ``jacobian_model="implicit_light_time"`` and the
counted-Doppler range-rate physics (whose partials integrate over the count interval)
are refused.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from .dmc import DMC_SIZE, DMCOptions, propagate_dmc_state
from .estimators import (
    EstimatorStats,
    _LMDamping,
    _gauss_markov_whitening,
    _lm_step,
    _lm_step_square_root,
    _predicted_decrease,
    _limit_step,
    _position_residual_from_h,
    _position_weight_diagonal,
    _range_rate_residual_from_h,
    _range_rate_weight_diagonal,
    _square_root_covariance_from_design,
    _stationarity_certified,
    _weighted_map_cost,
)
from .measurements import (
    PassGeometry,
    compute_position_residuals_analytic,
    compute_range_rate_residuals_analytic,
)
from .radiometrics import range_rate_physics_config

N_STATE = 6 + DMC_SIZE
_SOLVERS = {"normal": _lm_step, "square_root": _lm_step_square_root}
# measurement type -> (weights, local residual and partials, residual vector, components, time column)
_MEASUREMENTS = {
    "position": (_position_weight_diagonal, compute_position_residuals_analytic, _position_residual_from_h, 3, 5),
    "range_rate": (_range_rate_weight_diagonal, compute_range_rate_residuals_analytic,
                   _range_rate_residual_from_h, 4, 6),
}


def estimate_position_dmc_batch(*args, **kwargs) -> tuple[np.ndarray, str, EstimatorStats]:
    """Position-only batch fit of ``[r0, v0, w0]``; arguments as :func:`_estimate_dmc_batch`."""
    return _estimate_dmc_batch("position", *args, **kwargs)


def estimate_range_rate_dmc_batch(*args, **kwargs) -> tuple[np.ndarray, str, EstimatorStats]:
    """WP4.8e: the same fit on instantaneous range-rate observations (``obs_data`` rows
    ``[t, range, range_rate, az, el, station, time index]``); arguments as
    :func:`_estimate_dmc_batch`.  Counted-Doppler physics is refused."""
    return _estimate_dmc_batch("range_rate", *args, **kwargs)


def _estimate_dmc_batch(
    measurement_type: str,
    t_pass_s: ArrayLike,
    obs_data: ArrayLike,
    x_nominal0: ArrayLike,
    pass_geo: PassGeometry,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    dmc: DMCOptions,
    *,
    method: str = "normal",
    state_prior_covariance: ArrayLike | None = None,
    max_iter: int = 60,
    tol_step_norm: float = 1e-8,
    tol_cost_stability: float = 1e-8,
    lambda0: float = 1e-2,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    j2_moon: float = 0.0,
    srp=None,
    return_posterior: bool = False,
    damping_schedule: str = "fixed",
    measurement_correlation_time_s: float | None = None,
) -> tuple[np.ndarray, str, EstimatorStats]:
    """Batch fit of ``[r0, v0, w0]`` with a Gauss-Markov empirical acceleration.

    ``x_nominal0`` is the 9-element initial guess (prior mean of ``w0`` is 0; the prior
    mean of ``r0, v0`` is the guess itself when ``state_prior_covariance`` (6x6) is given,
    otherwise ``r0, v0`` are data-only).  Returns ``(x_best, status, stats)`` like
    :func:`estimate_position_bls_lm`; ``stats.posterior_covariance`` is 9x9.
    ``damping_schedule`` and ``measurement_correlation_time_s`` as in
    :func:`estimate_position_bls_lm` (WP4.2b, WP4.5b).
    """
    if method not in _SOLVERS:
        raise ValueError(f"method must be one of {tuple(_SOLVERS)}, got {method!r}.")
    if getattr(pass_geo, "jacobian_model", "analytic_exact_geometric") == "implicit_light_time":
        raise ValueError("the DMC batch maps partials at the receive epoch; "
                         "jacobian_model='implicit_light_time' is not supported.")
    weights_of, local_model, residual_vector, components, time_col = _MEASUREMENTS[measurement_type]
    if pass_geo.measurement_type != measurement_type:
        raise ValueError(f"pass_geo.measurement_type is {pass_geo.measurement_type!r}, not {measurement_type!r}.")
    if (measurement_type == "range_rate"
            and range_rate_physics_config(pass_geo.range_rate_physics).mode != "geometric_instantaneous"):
        raise ValueError("the DMC batch supports instantaneous range-rate only; counted-Doppler partials "
                         "integrate over the count interval and are not mapped through the DMC STM.")
    t_pass_s = np.asarray(t_pass_s, dtype=float).reshape(-1)
    obs_data = np.asarray(obs_data, dtype=float)
    x_nominal = np.asarray(x_nominal0, dtype=float).reshape(-1).copy()
    if x_nominal.size != N_STATE:
        raise ValueError("x_nominal0 must be [r0(3), v0(3), w0(3)].")

    sigma_w = (np.full(3, dmc.sigma_mps2) if dmc.sigma_axes_mps2 is None
               else np.asarray(dmc.sigma_axes_mps2, dtype=float))
    if np.any(sigma_w <= 0.0):
        raise ValueError("every DMC axis needs a positive sigma: it is the prior of w0.")
    prior_inv = np.zeros((N_STATE, N_STATE))
    prior_inv[6:, 6:] = np.diag(1.0 / sigma_w**2)
    x_prior = np.concatenate([x_nominal[:6], np.zeros(DMC_SIZE)])
    if state_prior_covariance is not None:
        p6 = np.asarray(state_prior_covariance, dtype=float)
        if p6.shape != (6, 6):
            raise ValueError("state_prior_covariance must be 6x6 (r0, v0).")
        prior_inv[:6, :6] = np.linalg.inv(p6)
    ev, vec = np.linalg.eigh(0.5 * (prior_inv + prior_inv.T))
    prior_sqrt = np.sqrt(np.clip(ev, 0.0, None))[:, None] * vec.T
    # bookkeeping units: m, m/s, and the DMC sigma for w (the solves equilibrate anyway)
    scale = np.diag(np.concatenate([np.full(3, 1e3), np.ones(3), sigma_w]))
    prior_inv_scaled = scale.T @ prior_inv @ scale
    w_diag = weights_of(obs_data, pass_geo)
    whitening = _gauss_markov_whitening(obs_data, w_diag, measurement_correlation_time_s, components)
    if whitening is not None:
        w_diag = np.ones_like(w_diag)          # the whitened rows carry unit weight
    white = (lambda v: v) if whitening is None else whitening
    solve = _SOLVERS[method]

    def propagate(x, stm):
        return propagate_dmc_state(t_pass_s, x, dmc, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
                                   get_earth_pos, get_sun_pos, rtol=rtol, atol=atol, j2_moon=j2_moon,
                                   srp=srp, stm=stm)

    def residual_of(hist):
        _, h_nom, h_tilde = local_model(hist[:, :6], obs_data, pass_geo)
        return white(residual_vector(obs_data, h_nom)), h_tilde

    def linearize(x):
        hist = propagate(x, True)
        residual, h_tilde = residual_of(hist)
        h = np.zeros((h_tilde.shape[0], N_STATE))
        for k in range(obs_data.shape[0]):
            phi = hist[int(obs_data[k, time_col]) - 1, N_STATE:].reshape(N_STATE, N_STATE, order="F")
            rows = slice(components * k, components * (k + 1))
            h[rows] = h_tilde[rows] @ phi[:6, :]
        return white(h), residual

    def cost_of(x, residual):
        return _weighted_map_cost(residual, w_diag, x, x_prior, prior_sqrt)

    x_best = x_nominal.copy()
    status, detail = "MaxIter", ""
    gn_decrement = float("nan")
    damping = _LMDamping(lambda0, damping_schedule)
    pending = None
    last_step = np.zeros(N_STATE)
    condition_number, rank = float("nan"), 0
    best_cost = np.inf
    for iteration in range(1, max_iter + 1):
        h, residual = linearize(x_nominal)
        current_cost = cost_of(x_nominal, residual)
        if iteration == 1:
            best_cost = current_cost
        prior_rhs_scaled = scale.T @ prior_inv @ (x_prior - x_nominal)
        if pending is not None:
            certified, gn_decrement = _stationarity_certified(
                h, residual, w_diag, scale, prior_inv_scaled, prior_rhs_scaled)
            if certified:
                status, detail = pending
                break
            pending = None
        step, condition_number, rank, singular = solve(
            h, residual, w_diag, scale, prior_inv_scaled, prior_rhs_scaled, damping.value,
            damping.damping_diag(h, w_diag, scale, prior_inv_scaled))
        if singular:
            status = "Singular"
            break
        step = _limit_step(step, pos_limit_m=20000.0)
        last_step = step
        x_candidate = x_nominal + step
        candidate_residual, _ = residual_of(propagate(x_candidate, False))
        candidate_cost = cost_of(x_candidate, candidate_residual)
        if candidate_cost < current_cost:
            improvement = abs(current_cost - candidate_cost) / max(current_cost, np.finfo(float).eps)
            damping.accepted(current_cost - candidate_cost, _predicted_decrease(
                h, residual, w_diag, step, x_nominal - x_prior, prior_inv))
            x_nominal = x_candidate
            x_best = x_nominal.copy()
            best_cost = candidate_cost
            if improvement < tol_cost_stability or np.linalg.norm(step) < tol_step_norm:
                pending = ("J-Stab", "COST_STABLE") if improvement < tol_cost_stability else ("Converged", "STEP_NORM")
        else:
            x_nominal = x_best.copy()
            damping.rejected()
            if damping.exhausted:
                certified, gn_decrement = _stationarity_certified(
                    h, residual, w_diag, scale, prior_inv_scaled, prior_rhs_scaled)
                status, detail = (("Converged", "STATIONARY_AT_DAMPING_LIMIT") if certified
                                  else ("DampingLimit", "LAMBDA_HARD_LIMIT"))
                break

    posterior_covariance = None
    if return_posterior:
        h_best, _ = linearize(x_best)
        posterior_covariance, _ = _square_root_covariance_from_design(h_best, w_diag, prior_inv, scale)

    stats = EstimatorStats(
        iterations=iteration,
        final_cost=best_cost,
        position_step_norm_m=float(np.linalg.norm(last_step[:3])),
        velocity_step_norm_mps=float(np.linalg.norm(last_step[3:6])),
        condition_number=condition_number,
        rank=rank,
        posterior_covariance=posterior_covariance,
        termination_detail=detail,
        gauss_newton_decrement=gn_decrement,
    )
    return x_best, status, stats
