"""Analytic local measurement partials for the EKF (Phase D6).

The EKF needs dh/dx at the update epoch. These are the BLS-LM partials, taken
locally instead of mapped to the arc's initial state:

- position and instantaneous range rate: the local H_tilde blocks of
  ``compute_position_residuals_analytic`` / ``compute_range_rate_residuals_analytic``;
- two-way counted Doppler: range/az/el rows from the geometric H_tilde (as BLS),
  the Doppler row from ``two_way_counted_doppler_initial_state_jacobian`` on the
  observable's own local window, with a 6+36 augmented history propagated from
  the update-epoch state (STM = I there), so "initial state" is that state.

Only combinations whose analytic partials are the exact derivative of the
observable the UKF/EKF evaluates are accepted; anything else raises.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
from numpy.typing import ArrayLike

from .dynamics import PropagationProfile, propagate_augmented_state
from .filters import _two_way_local_histories
from .measurements import (
    PassGeometry,
    compute_position_residuals_analytic,
    compute_range_rate_residuals_analytic,
    normalize_companion_geometry,
)
from .radiometrics import RangeRatePhysicsConfig, range_rate_physics_config, two_way_counted_doppler_initial_state_jacobian

_IDENTITY_6_COL: np.ndarray = np.eye(6).reshape(-1, order="F")


def validate_analytic_support(measurement_type: str, pass_geo: PassGeometry) -> None:
    """Raise unless the analytic partials are exact for this observable (D6 scope)."""
    if measurement_type == "position":
        if getattr(pass_geo, "measurement_model_profile", "geometric_instantaneous") != "geometric_instantaneous":
            raise ValueError("analytic EKF partials support only the geometric_instantaneous position profile.")
        return
    rr = range_rate_physics_config(pass_geo.range_rate_physics)
    if normalize_companion_geometry(getattr(pass_geo, "companion_geometry", "instantaneous")) != "instantaneous":
        raise ValueError("analytic EKF partials require companion_geometry='instantaneous'.")
    if rr.mode == "two_way_counted_doppler" and rr.local_state_model != "ode":
        raise ValueError("analytic counted-Doppler EKF partials require local_state_model='ode'.")
    if rr.mode not in {"geometric_instantaneous", "two_way_counted_doppler"}:
        raise ValueError(f"analytic EKF partials do not support range-rate mode {rr.mode!r}.")


def _local_history(x6: np.ndarray, time_idx: int, n_t: int) -> np.ndarray:
    # the analytic blocks read the state history only at the observation's own index
    history = np.zeros((n_t, 6))
    history[time_idx] = x6
    return history


def analytic_measurement_jacobian(
    measurement_type: str,
    obs_row: np.ndarray,
    pass_geo: PassGeometry,
    state_mci: ArrayLike,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    rtol: float,
    atol: float,
    *,
    j2_moon: float = 0.0,
    srp=None,
    propagation_profile=None,
) -> np.ndarray:
    """dh/d[r, v] at the observation epoch: 3x6 (position) or 4x6 (range, range rate, az, el)."""
    x6 = np.asarray(state_mci, dtype=float).reshape(-1)[:6]
    row = np.asarray(obs_row, dtype=float).reshape(1, -1)
    n_t = len(pass_geo.t_s)
    if measurement_type == "position":
        time_idx = int(row[0, 5]) - 1
        h_tilde = compute_position_residuals_analytic(_local_history(x6, time_idx, n_t), row, pass_geo)[2]
        return np.asarray(h_tilde, dtype=float).reshape(3, 6)

    rr = range_rate_physics_config(pass_geo.range_rate_physics)
    time_idx = int(row[0, 6]) - 1
    geometric_geo = pass_geo if rr.mode == "geometric_instantaneous" else replace(
        pass_geo, range_rate_physics=RangeRatePhysicsConfig())
    h = np.asarray(compute_range_rate_residuals_analytic(_local_history(x6, time_idx, n_t), row, geometric_geo)[2],
                   dtype=float).reshape(4, 6).copy()
    if rr.mode == "geometric_instantaneous":
        return h

    # counted Doppler: the observable's own local window, re-propagated with the STM from this state
    t_mid = float(row[0, 0])
    station = pass_geo.stations[int(row[0, 5]) - 1]
    earth_state = np.concatenate([pass_geo.earth_pos_mci_m[time_idx], pass_geo.earth_vel_mci_mps[time_idx]])
    rho_ecef = (pass_geo.x_j2000_to_itrf93[time_idx] @ (x6 - earth_state))[:3] - station.r_ecef_m
    local_t, _, local_earth_pos, local_earth_vel, local_xforms = _two_way_local_histories(
        t_mid, x6, float(np.linalg.norm(rho_ecef)), pass_geo, rr.count_interval_s, rr.light_speed_mps,
        mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2, get_earth_pos, get_sun_pos, rtol, atol,
        rr.local_state_model, rr_physics=rr, j2_moon=j2_moon, srp=srp, propagation_profile=propagation_profile)
    aug = _augmented_local_history(local_t, t_mid, x6, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
                                   get_earth_pos, get_sun_pos, rtol, atol, j2_moon=j2_moon, srp=srp,
                                   propagation_profile=propagation_profile)
    h[1] = two_way_counted_doppler_initial_state_jacobian(
        t_mid, station, local_t, aug, local_earth_pos, local_earth_vel, local_xforms, rr, et0_s=pass_geo.et0_s)
    return h


def _augmented_local_history(local_t, t_mid, x6, mu_moon, mu_earth, mu_sun, get_earth_pos, get_sun_pos, rtol, atol,
                             *, j2_moon, srp, propagation_profile):
    """State + STM Phi(t, t_mid) on ``local_t``, propagated both ways from t_mid (as the observable's history)."""
    if propagation_profile is None and srp is not None and srp.enabled:
        propagation_profile = PropagationProfile()
    kwargs = {} if propagation_profile is None else propagation_profile.propagator_kwargs()
    y0 = np.concatenate([x6, _IDENTITY_6_COL])
    out = np.zeros((local_t.size, 42))
    out[local_t == t_mid] = y0
    for idx in (np.where(local_t < t_mid)[0][::-1], np.where(local_t > t_mid)[0]):
        if idx.size:
            hist = propagate_augmented_state(np.concatenate([[t_mid], local_t[idx]]), y0, mu_moon, mu_earth, mu_sun,
                                             get_earth_pos, get_sun_pos, rtol=rtol, atol=atol, j2_moon=j2_moon,
                                             srp=srp, **kwargs)
            out[idx] = hist[1:, :42]
    return out
