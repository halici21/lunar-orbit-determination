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

With ``dmc=`` (Phase D5) the state is ``[r, v, w, biases]``: ``w`` is the
Gauss-Markov empirical acceleration of ``dmc.py``, propagated with its 9x9 STM
and van Loan process noise. The measurement function never sees ``w``.

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

from .dmc import DMC_SIZE, DMCOptions, dmc_process_noise, propagate_dmc_state
from .dynamics import propagate_augmented_state, propagate_state
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
    linearization: ArrayLike | None = None,
    dmc: DMCOptions | None = None,
) -> LunarEKFResult:
    """Run an EKF (optionally followed by the RTS smoother) over one prepared lunar OD arc.

    Arguments mirror ``run_lunar_ukf``. The prior ``(x0_mci, p0)`` holds at the
    first observation epoch, as in the UKF. Not supported here: K_SRP
    solve-for, state freezing/regularisation, adaptive Q/R and robust updates.

    ``linearization`` (one state per observation, in time order) makes this a
    linearised filter about that trajectory instead of about its own
    prediction: dynamics and measurement are expanded around it. The iterated
    smoother (``run_lunar_iterated_smoother``) passes its previous solution here.

    ``dmc`` adds the empirical acceleration ``w`` as state elements 6:9 (``x0_mci``,
    ``p0`` and ``linearization`` include it). It replaces ``process_noise``, which
    must then be None, and propagates with DOP853 (``propagation_profile`` is not
    applied on that path).
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
    n_dyn = 6 if dmc is None else 6 + DMC_SIZE
    if nx < n_dyn:
        raise ValueError(f"x0_mci must contain at least the {n_dyn} dynamic state elements.")
    if dmc is not None and process_noise is not None:
        raise ValueError("dmc= supplies the process noise; process_noise must be None.")
    p = _symmetrize(np.asarray(p0, dtype=float))
    if p.shape != (nx, nx):
        raise ValueError("p0 must be an NxN covariance matrix matching x0_mci.")
    bias_cfg = _resolve_ukf_bias_config(measurement_type, nx - (n_dyn - 6), len(pass_geo.stations), bias_mode)
    if process_noise_model not in {"discrete", "continuous_white_acceleration"}:
        raise ValueError("process_noise_model must be 'discrete' or 'continuous_white_acceleration'.")
    q = _coerce_process_noise(process_noise, nx, process_noise_model)
    # the measurement does not depend on w, so its step only has to be nonzero
    steps = (np.asarray(list(DEFAULT_JACOBIAN_STEPS) + [DEFAULT_BIAS_JACOBIAN_STEP] * (nx - 6), dtype=float)
             if jacobian_steps is None else np.asarray(jacobian_steps, dtype=float).reshape(-1))
    residual_fn = _measurement_residual(measurement_type)

    order = np.argsort(obs[:, 0], kind="stable")
    current_t = float(t_pass[0])
    nominal = None
    if linearization is not None:
        nominal = np.asarray(linearization, dtype=float)
        if nominal.shape != (order.size, nx):
            raise ValueError("linearization must hold one full state per observation.")
        if float(obs[order[0], 0]) != current_t:
            raise ValueError("a linearization trajectory requires the first observation at t_pass_s[0].")
    x_bar_prev = None
    rows: dict[str, list] = {key: [] for key in (
        "t", "idx", "x", "p", "xp", "pp", "phi", "innov", "s", "zpred", "nis", "accepted")}
    for step, obs_idx in enumerate(order):
        row = obs[obs_idx, :]
        target_t = float(row[0])
        x_bar = None if nominal is None else nominal[step]
        phi = np.eye(nx)
        x_prop = x.copy()
        if target_t != current_t:
            centre = x if x_bar_prev is None else x_bar_prev
            if dmc is None:
                aug = propagate_augmented_state(
                    [current_t, target_t], np.concatenate([centre[:6], _IDENTITY_6_COL]),
                    mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2, get_earth_pos, get_sun_pos,
                    rtol=rtol, atol=atol, j2_moon=j2_moon, srp=srp, **propagation_kwargs,
                )[-1, :]
                q_step = _discretize_process_noise(q, abs(target_t - current_t), nx, process_noise_model)
            else:
                aug = propagate_dmc_state(
                    [current_t, target_t], centre[:n_dyn], dmc, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
                    get_earth_pos, get_sun_pos, rtol=rtol, atol=atol, j2_moon=j2_moon, srp=srp,
                )[-1, :]
                q_step = np.zeros((nx, nx))
                q_step[:n_dyn, :n_dyn] = dmc_process_noise(dmc, target_t - current_t)
            phi[:n_dyn, :n_dyn] = aug[n_dyn:n_dyn + n_dyn * n_dyn].reshape(n_dyn, n_dyn, order="F")
            # f(x) about the linearization centre: exact when the centre is the filter state itself
            x_prop[:n_dyn] = aug[:n_dyn] + phi[:n_dyn, :n_dyn] @ (x[:n_dyn] - centre[:n_dyn])
        else:
            q_step = None
        x_pred, p_pred = ekf_predict(p, x_prop, phi, q_step)
        z, r, measurement_fn, _, _ = _measurement_context(
            measurement_type, row, pass_geo, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
            get_earth_pos, get_sun_pos, rtol, atol, bias_cfg,
            j2_moon=j2_moon, srp=srp, solve_for_k_srp=False, propagation_profile=propagation_profile,
        )
        if dmc is not None:
            measurement_fn = _without_dmc(measurement_fn)
        centre = x_pred if x_bar is None else x_bar
        h = finite_difference_measurement_jacobian(measurement_fn, centre, steps, residual_fn)
        z_pred = np.asarray(measurement_fn(centre), dtype=float) + h @ (x_pred - centre)
        x, p, diag = ekf_update(x_pred, p_pred, z, z_pred, h, r, residual_fn=residual_fn, nis_gate=nis_gate)
        current_t = target_t
        x_bar_prev = x_bar
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


def _without_dmc(measurement_fn: MeasurementFunction) -> MeasurementFunction:
    """The shared measurement function evaluated on [r, v, biases] of a [r, v, w, biases] state."""
    def wrapped(x):
        x = np.asarray(x, dtype=float)
        return measurement_fn(np.concatenate([x[:6], x[6 + DMC_SIZE:]]))
    return wrapped


@dataclass(frozen=True)
class IteratedSmootherResult:
    """The last linearised EKF+RTS pass of the iterated smoother and how it got there."""

    result: LunarEKFResult
    status: str
    iterations: int
    history: tuple[dict, ...]
    final_gn_decrement: float | None = None


def run_lunar_iterated_smoother(
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
    max_iter: int = 20,
    tol_sigma: float = 1e-3,
    max_halvings: int = 10,
    stationarity_chi2: float = 0.01,
    **ekf_options,
) -> IteratedSmootherResult:
    """Iterated EKF + RTS smoother: Gauss-Newton on the arc's MAP problem, one EKF+RTS pass per iteration.

    Pass 0 is the ordinary EKF+RTS. Each later pass relinearises about the previous solution:

    - without process noise the solution is one trajectory, so the next linearization is that trajectory
      propagated from the smoothed first-epoch state; the step is accepted only if it lowers the MAP cost
      (prior + whitened measurement residuals), halving it up to ``max_halvings`` times (damped Gauss-Newton);
    - with process noise the next linearization is the smoothed state sequence itself (the iterated
      extended Kalman smoother); no line search. ``dmc=`` (Phase D5) is process noise too.

    Stops when the largest change of the linearization point, in smoothed sigmas, is below ``tol_sigma``,
    or (without process noise) when the Gauss-Newton decrement of the next step, the chi2 decrease the
    linearised problem predicts, is at most ``stationarity_chi2`` (D4-R; the F2 TRF certified-stop bound).
    Status: ``Converged``, ``MaxIter`` or ``NoDescent`` (the line search found no lower cost).
    """
    if ekf_options.pop("smooth", True) is not True or "linearization" in ekf_options:
        raise ValueError("the iterated smoother controls smooth= and linearization= itself.")
    common = (t_pass_s, obs_data, x0_mci, p0, pass_geo, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
              get_earth_pos, get_sun_pos)
    with_q = ekf_options.get("process_noise") is not None or ekf_options.get("dmc") is not None
    x_prior = np.asarray(x0_mci, dtype=float).reshape(-1)
    p_prior = np.asarray(p0, dtype=float)
    obs = np.asarray(obs_data, dtype=float)
    order = np.argsort(obs[:, 0], kind="stable")
    t_obs = obs[order, 0]
    cost_of = _map_cost_function(obs, order, x_prior, p_prior, pass_geo, mu_moon_m3_s2, mu_earth_m3_s2,
                                 mu_sun_m3_s2, get_earth_pos, get_sun_pos, ekf_options) if not with_q else None

    result = run_lunar_ekf(*common, smooth=True, **ekf_options)
    if not with_q and float(t_obs[0]) != float(np.asarray(t_pass_s, dtype=float).reshape(-1)[0]):
        raise ValueError("the iterated smoother requires the first observation at t_pass_s[0].")
    x_bar0 = x_prior.copy()
    nominal = None
    cost = None if with_q else cost_of(x_bar0)[0]
    history = [dict(iteration=0, cost=cost, step_sigma=None, halvings=0, gn_decrement=None)]
    status = "MaxIter"
    final_gn_decrement = None
    for iteration in range(1, max_iter + 1):
        smoothed = result.smoothed
        sigma = np.sqrt(np.diag(smoothed.covariances[0]))
        if with_q:
            previous = result.predicted_states if nominal is None else nominal
            change = float(np.max(np.abs(smoothed.state_estimates - previous) / np.sqrt(
                np.diagonal(smoothed.covariances, axis1=1, axis2=2))))
            if change < tol_sigma:
                status = "Converged"
                break
            nominal = smoothed.state_estimates.copy()
            history.append(dict(iteration=iteration, cost=None, step_sigma=change, halvings=0, gn_decrement=None))
        else:
            step = smoothed.state_estimates[0] - x_bar0
            change = float(np.max(np.abs(step) / sigma))
            gn_decrement = float(step @ np.linalg.solve(smoothed.covariances[0], step))
            if change < tol_sigma or gn_decrement <= stationarity_chi2:
                status = "Converged"
                final_gn_decrement = gn_decrement
                break
            for halvings in range(max_halvings + 1):
                candidate = x_bar0 + step * 0.5 ** halvings
                cand_cost, trajectory = cost_of(candidate)
                if np.isfinite(cand_cost) and cand_cost < cost:
                    break
            else:
                status = "NoDescent"
                break
            x_bar0, cost, nominal = candidate, cand_cost, trajectory
            history.append(dict(iteration=iteration, cost=cost, step_sigma=change, halvings=halvings,
                                gn_decrement=gn_decrement))
        result = run_lunar_ekf(*common, smooth=True, linearization=nominal, **ekf_options)
    return IteratedSmootherResult(result=result, status=status, iterations=len(history) - 1,
                                  history=tuple(history), final_gn_decrement=final_gn_decrement)


def _map_cost_function(obs, order, x_prior, p_prior, pass_geo, mu_moon, mu_earth, mu_sun, get_earth_pos,
                       get_sun_pos, options):
    """MAP cost of a process-noise-free arc as a function of the first-epoch state, and its trajectory."""
    rtol, atol = options.get("rtol", 1e-11), options.get("atol", 1e-12)
    srp, j2_moon = options.get("srp"), options.get("j2_moon", 0.0)
    profile = resolve_profile(srp, options.get("propagation_profile"))
    kwargs = {} if profile is None else profile.propagator_kwargs()
    measurement_type = (options.get("measurement_type") or pass_geo.measurement_type).lower()
    nx = x_prior.size
    bias_cfg = _resolve_ukf_bias_config(measurement_type, nx, len(pass_geo.stations), options.get("bias_mode"))
    residual_fn = _measurement_residual(measurement_type)
    rows = [obs[i, :] for i in order]
    t_obs = np.array([row[0] for row in rows])
    epochs, inverse = np.unique(t_obs, return_inverse=True)
    prior_info = np.linalg.inv(p_prior)

    def cost_of(x0):
        x0 = np.asarray(x0, dtype=float)
        try:
            states = propagate_state(epochs, x0[:6], mu_moon, mu_earth, mu_sun, get_earth_pos, get_sun_pos,
                                     rtol=rtol, atol=atol, j2_moon=j2_moon, srp=srp, **kwargs)
        except Exception:
            return np.inf, None
        trajectory = np.column_stack([states[inverse], np.tile(x0[6:], (len(rows), 1))])
        d = x0 - x_prior
        total = float(d @ prior_info @ d)
        for row, x in zip(rows, trajectory):
            z, r, measurement_fn, _, _ = _measurement_context(
                measurement_type, row, pass_geo, mu_moon, mu_earth, mu_sun, get_earth_pos, get_sun_pos,
                rtol, atol, bias_cfg, j2_moon=j2_moon, srp=srp, solve_for_k_srp=False,
                propagation_profile=profile)
            res = np.asarray(residual_fn(np.asarray(z, float), np.asarray(measurement_fn(x), float)))
            total += float(res @ np.linalg.solve(r, res))
        return total, trajectory

    return cost_of
