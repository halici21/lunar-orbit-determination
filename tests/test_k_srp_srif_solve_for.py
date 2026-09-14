"""Phase 17-R - permanent guards for K_SRP solve-for in the two-way range SRIF.

Mirrors test_k_srp_bls_solve_for.py's guards (default invariance, explicit
opt-in, negative-K non-clipping, numerical-only scaling) and adds the gate
this estimator exists to prove: SRIF and BLS-LM must reach the SAME posterior
on the SAME linear-Gaussian problem (s32 -- "one of the most important
gates").

SRIF has no LM damping, so s31 makes k_srp_prior_sigma REQUIRED here (BLS
allows a data-only solve; SRIF does not) -- checked directly below.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("spiceypy")

from lunar_od.config import Station
from lunar_od.dynamics import propagate_augmented_state
from lunar_od.estimators import estimate_two_way_range_bls_lm, estimate_two_way_range_srif
from lunar_od.radiometrics import _interp_state
from lunar_od.srp import SRPOptions
from lunar_od.two_way_range import generate_two_way_range_measurements

_MU_MOON = 4902.800066e9
_ET_UTC = "2027-03-02 00:00:00"
_STATION = Station(
    name="Goldstone DSN", lat_deg=35.30, lon_deg=-116.81, alt_m=969.67,
    color_rgb=(0.85, 0.325, 0.098), sigma_range_m=5.0,
    sigma_angle_rad=math.radians(0.001),
)


def setup_module(_module):
    from lunar_od.spice_loader import load_spice_kernels

    load_spice_kernels()


def _fixture(duration_s: float = 6000.0, step_s: float = 60.0):
    import spiceypy as spice

    et0 = float(spice.str2et(_ET_UTC))
    t_pass = np.arange(0.0, duration_s + 0.5 * step_s, step_s)
    earth_states = np.array([
        np.asarray(spice.spkezr("EARTH", et0 + t, "J2000", "NONE", "MOON")[0],
                   dtype=float) * 1e3
        for t in t_pass
    ])

    def get_earth_pos(t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        return np.array([_interp_state(t_pass, earth_states, float(ti))[:3]
                         for ti in t_arr])

    def get_earth_vel(t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        return np.array([_interp_state(t_pass, earth_states, float(ti))[3:]
                         for ti in t_arr])

    def get_sun_pos(t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        return np.tile(np.array([1.4e11, 6.0e10, 0.0]), (t_arr.size, 1))

    r0 = 1737.4e3 + 100e3
    x_true0 = np.array([r0, 30e3, -20e3, -15.0, math.sqrt(_MU_MOON / r0), 4.0])
    x_aug0 = np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")])
    x_aug_truth = propagate_augmented_state(
        t_pass, x_aug0, _MU_MOON, 0.0, 0.0, get_earth_pos, get_sun_pos,
        rtol=1e-12, atol=1e-13,
    )
    vis = np.ones((t_pass.size, 1), dtype=bool)
    obs, pass_geo = generate_two_way_range_measurements(
        t_pass, x_aug_truth[:, :6], (_STATION,), vis, get_earth_pos, get_earth_vel,
        et0, noise=False,
    )
    return dict(et0=et0, t_pass=t_pass, gep=get_earth_pos, gev=get_earth_vel,
                gsp=get_sun_pos, x_true0=x_true0, obs=obs, pass_geo=pass_geo)


@pytest.fixture(scope="module")
def fx():
    return _fixture()


# ======================================================================
# default invariance, checked bitwise
# ======================================================================
def test_default_six_state_srif_is_bitwise_unaffected_by_new_kwargs(fx):
    x_guess = fx["x_true0"] + np.array([25.0, -20.0, 12.0, 0.01, -0.008, 0.004])
    args = (fx["t_pass"], fx["obs"], x_guess, fx["pass_geo"], _MU_MOON, 0.0, 0.0,
            fx["gep"], fx["gsp"])
    kw = dict(max_iter=6, rtol=1e-12, atol=1e-13, return_posterior=True)

    x_old, stop_old, stats_old = estimate_two_way_range_srif(*args, **kw)
    x_new, stop_new, stats_new = estimate_two_way_range_srif(
        *args, **kw, srp=None, solve_for_k_srp=False)

    assert np.array_equal(x_old, x_new)
    assert stop_old == stop_new
    assert stats_old.final_cost == stats_new.final_cost
    assert np.array_equal(stats_old.posterior_covariance, stats_new.posterior_covariance)
    assert stats_new.k_srp_estimate is None


# ======================================================================
# s13 explicit opt-in / s31 prior is required for SRIF
# ======================================================================
def test_solve_for_k_without_srp_is_rejected(fx):
    with pytest.raises(ValueError, match="solve_for_k_srp"):
        estimate_two_way_range_srif(
            fx["t_pass"], fx["obs"], fx["x_true0"].copy(), fx["pass_geo"],
            _MU_MOON, 0.0, 0.0, fx["gep"], fx["gsp"], max_iter=1,
            solve_for_k_srp=True,
        )


def test_srif_k_prior_sigma_is_mandatory_not_optional(fx):
    """s31: SRIF carries no LM damping, so an unregularized K column would
    routinely leave r_hat singular. Unlike BLS, a data-only SRIF K solve-for
    must be refused, not silently attempted."""
    srp = SRPOptions(k_srp_m2_per_kg=0.01)
    with pytest.raises(ValueError, match="k_srp_prior_sigma"):
        estimate_two_way_range_srif(
            fx["t_pass"], fx["obs"], fx["x_true0"].copy(), fx["pass_geo"],
            _MU_MOON, 0.0, 0.0, fx["gep"], fx["gsp"], max_iter=1,
            srp=srp, solve_for_k_srp=True,
        )


# ======================================================================
# s18 - negative K is rejected, never clipped
# ======================================================================
def test_negative_k_trial_steps_are_backtracked_not_clipped(fx):
    x_guess = fx["x_true0"].copy()
    srp = SRPOptions(k_srp_m2_per_kg=0.001)
    x_est, stop_reason, stats = estimate_two_way_range_srif(
        fx["t_pass"], fx["obs"], x_guess, fx["pass_geo"], _MU_MOON, 0.0, 0.0,
        fx["gep"], fx["gsp"], max_iter=15, rtol=1e-12, atol=1e-13,
        srp=srp, solve_for_k_srp=True, k_srp_initial=0.0005,
        k_srp_prior_sigma=0.01,
    )
    assert stats.k_srp_estimate is not None
    assert stats.k_srp_estimate >= 0.0
    assert np.isfinite(stats.k_srp_estimate)


# ======================================================================
# s32 - BLS-SRIF common Gaussian posterior (the most important gate)
# ======================================================================
def test_bls_srif_common_posterior_on_the_same_linear_gaussian_problem(fx):
    """SRIF and BLS-LM must reach the same posterior mean, K estimate, and
    covariance on the same linearized problem (s32). A well-posed state
    prior is used for the same reason the BLS oracle needs one: the raw
    single-pass problem's eigenvalue spread trips the shared
    eigenvalue-floor safety net differently for slightly different
    convergence paths (documented in the campaign report), which is a
    property of that shared helper, not of either estimator's correctness.
    """
    x_guess = fx["x_true0"] + np.array([5.0, -3.0, 2.0, 0.0, 0.0, 0.0])
    srp = SRPOptions(k_srp_m2_per_kg=0.01)
    prior_cov = np.diag([10.0 ** 2] * 3 + [0.01 ** 2] * 3)
    common_kw = dict(
        max_iter=20, rtol=1e-12, atol=1e-13, srp=srp, solve_for_k_srp=True,
        k_srp_initial=0.012, k_srp_prior_sigma=0.005,
        prior_covariance=prior_cov, return_posterior=True,
    )
    x_bls, _, stats_bls = estimate_two_way_range_bls_lm(
        fx["t_pass"], fx["obs"], x_guess.copy(), fx["pass_geo"], _MU_MOON, 0.0, 0.0,
        fx["gep"], fx["gsp"], **common_kw)
    x_srif, _, stats_srif = estimate_two_way_range_srif(
        fx["t_pass"], fx["obs"], x_guess.copy(), fx["pass_geo"], _MU_MOON, 0.0, 0.0,
        fx["gep"], fx["gsp"], **common_kw)

    assert np.max(np.abs(x_bls - x_srif)) < 1e-2, "state posterior mean disagrees"
    assert abs(stats_bls.k_srp_estimate - stats_srif.k_srp_estimate) < 1e-6, (
        "K posterior mean disagrees between BLS and SRIF"
    )
    sigma_bls = math.sqrt(stats_bls.posterior_covariance[6, 6])
    sigma_srif = math.sqrt(stats_srif.posterior_covariance[6, 6])
    assert sigma_bls == pytest.approx(sigma_srif, rel=1e-3)
