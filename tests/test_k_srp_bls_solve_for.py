"""Phase 17-R - permanent guards for K_SRP solve-for in the two-way range BLS.

These pin the hard-scope requirements that must hold for every future change
to this code path, not just today's implementation:

  - default invariance (s12): solve_for_k_srp defaults to False and must not
    change six-state behaviour at all -- checked bitwise, not to tolerance.
  - explicit opt-in (s13): SRP-enabled and K-solve-for-enabled are independent
    switches; enabling one must never silently enable the other.
  - negative-K non-clipping (s18): a trial step that would drive K negative
    must be rejected outright, never clamped to zero (clamping would bias the
    estimate low).
  - numerical-only scaling (s19/s20): k_srp_scale must not change the physical
    estimate.
  - reference configuration untouched (s14): this module must never assign a
    value to LTB-IRIS-DSN34X-v1's K_SRP.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("spiceypy")

from lunar_od.config import Station
from lunar_od.dynamics import propagate_augmented_state
from lunar_od.estimators import estimate_two_way_range_bls_lm
from lunar_od.radiometrics import _interp_state
from lunar_od.srp import SRPOptions
from lunar_od.two_way_range import compute_two_way_range_residuals, generate_two_way_range_measurements

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
    """A short LLO pass with real Earth ephemeris and illuminated SRP geometry."""
    import spiceypy as spice

    et0 = float(spice.str2et(_ET_UTC))
    t_pass = np.arange(0.0, duration_s + 0.5 * step_s, step_s)
    earth_states = np.array([
        np.asarray(spice.spkezr("EARTH", et0 + t, "J2000", "NONE", "MOON")[0],
                   dtype=float) * 1e3
        for t in t_pass
    ])
    earth_pos, earth_vel = earth_states[:, :3], earth_states[:, 3:]

    def get_earth_pos(t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        return np.array([_interp_state(t_pass, earth_states, float(ti))[:3]
                         for ti in t_arr])

    def get_earth_vel(t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        return np.array([_interp_state(t_pass, earth_states, float(ti))[3:]
                         for ti in t_arr])

    def get_sun_pos(t):
        # A real Sun direction, fixed magnitude, at a geometry that keeps the
        # spacecraft mostly illuminated over this short pass -- SRP sensitivity
        # requires the geometry to actually see sunlight.
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
# s12 - default invariance, checked bitwise
# ======================================================================
def test_default_six_state_bls_is_bitwise_unaffected_by_new_kwargs(fx):
    """Passing the new kwargs at their defaults must be indistinguishable
    from a call site that has never heard of K_SRP."""
    x_guess = fx["x_true0"] + np.array([25.0, -20.0, 12.0, 0.01, -0.008, 0.004])
    args = (fx["t_pass"], fx["obs"], x_guess, fx["pass_geo"], _MU_MOON, 0.0, 0.0,
            fx["gep"], fx["gsp"])
    kw = dict(max_iter=6, rtol=1e-12, atol=1e-13, return_posterior=True)

    x_old, stop_old, stats_old = estimate_two_way_range_bls_lm(*args, **kw)
    x_new, stop_new, stats_new = estimate_two_way_range_bls_lm(
        *args, **kw, srp=None, solve_for_k_srp=False)

    assert np.array_equal(x_old, x_new)
    assert stop_old == stop_new
    assert stats_old.final_cost == stats_new.final_cost
    assert stats_old.iterations == stats_new.iterations
    assert np.array_equal(stats_old.posterior_covariance, stats_new.posterior_covariance)
    assert stats_new.k_srp_estimate is None
    assert stats_new.k_srp_prior_status is None


def test_default_x_best_stays_a_six_vector_in_every_mode(fx):
    """x_best must never grow to 7 elements; K always comes back via stats."""
    x_guess = fx["x_true0"] + np.array([25.0, -20.0, 12.0, 0.01, -0.008, 0.004])
    srp = SRPOptions(k_srp_m2_per_kg=0.01)
    x_est, _, stats = estimate_two_way_range_bls_lm(
        fx["t_pass"], fx["obs"], x_guess, fx["pass_geo"], _MU_MOON, 0.0, 0.0,
        fx["gep"], fx["gsp"], max_iter=4, rtol=1e-12, atol=1e-13,
        srp=srp, solve_for_k_srp=True, return_posterior=True,
    )
    assert x_est.shape == (6,)
    assert stats.k_srp_estimate is not None
    assert stats.posterior_covariance.shape == (7, 7)


# ======================================================================
# s13 - explicit opt-in; the three modes stay distinct
# ======================================================================
def test_solve_for_k_without_srp_is_rejected(fx):
    x_guess = fx["x_true0"].copy()
    with pytest.raises(ValueError, match="solve_for_k_srp"):
        estimate_two_way_range_bls_lm(
            fx["t_pass"], fx["obs"], x_guess, fx["pass_geo"], _MU_MOON, 0.0, 0.0,
            fx["gep"], fx["gsp"], max_iter=1, solve_for_k_srp=True,
        )


def test_srp_enabled_fixed_k_does_not_silently_solve_for_k(fx):
    """SRP ON does not imply K solve-for ON (s13)."""
    x_guess = fx["x_true0"] + np.array([10.0, -5.0, 3.0, 0.0, 0.0, 0.0])
    srp = SRPOptions(k_srp_m2_per_kg=0.01)
    _, _, stats = estimate_two_way_range_bls_lm(
        fx["t_pass"], fx["obs"], x_guess, fx["pass_geo"], _MU_MOON, 0.0, 0.0,
        fx["gep"], fx["gsp"], max_iter=3, rtol=1e-12, atol=1e-13,
        srp=srp, solve_for_k_srp=False,
    )
    assert stats.k_srp_estimate is None


def test_k_srp_initial_requires_solve_for_k_srp(fx):
    x_guess = fx["x_true0"].copy()
    with pytest.raises(ValueError):
        estimate_two_way_range_bls_lm(
            fx["t_pass"], fx["obs"], x_guess, fx["pass_geo"], _MU_MOON, 0.0, 0.0,
            fx["gep"], fx["gsp"], max_iter=1, k_srp_initial=0.02,
        )


# ======================================================================
# s18 - negative K is rejected, never clipped
# ======================================================================
def test_negative_k_trial_steps_are_rejected_not_clipped(fx):
    """Start K very close to zero so an early LM step is likely to try a
    negative trial value; the estimate must never be clamped to zero and the
    negative-K rejection counter must be consistent with the final estimate.
    """
    x_guess = fx["x_true0"].copy()
    srp = SRPOptions(k_srp_m2_per_kg=0.001)
    x_est, stop_reason, stats = estimate_two_way_range_bls_lm(
        fx["t_pass"], fx["obs"], x_guess, fx["pass_geo"], _MU_MOON, 0.0, 0.0,
        fx["gep"], fx["gsp"], max_iter=15, rtol=1e-12, atol=1e-13,
        srp=srp, solve_for_k_srp=True, k_srp_initial=0.0005,
    )
    assert stats.k_srp_estimate is not None
    assert stats.k_srp_estimate >= 0.0
    assert np.isfinite(stats.k_srp_estimate)
    # A K estimate of exactly 0.0 with rejections recorded would be the
    # clipping signature this guard exists to catch.
    if stats.k_srp_negative_step_rejections > 0:
        assert stats.k_srp_estimate != 0.0


# ======================================================================
# s19/s20 - k_srp_scale is numerical-only
# ======================================================================
def test_k_scale_choice_does_not_change_the_physical_estimate(fx):
    """Two reasonable scale choices (s20 requires >= 2) must agree.

    A moderately informative state prior (sigma_pos=10 m, sigma_vel=0.01 m/s
    -- a realistic "already have a decent orbit" starting point, not an
    unconstrained guess) is used to keep the augmented information matrix's
    eigenvalue spread inside double-precision range.  Without it, this
    fixture's single 5.5-hour single-station pass gives K's information
    content in atwa units roughly 1e13-1e16x smaller than the orbital
    state's, which pushed the shared _safe_covariance_from_information
    eigenvalue floor (max_eig*1e-14) to engage differently at different
    scales.

    Phase 17-R1COV has since REMOVED that limitation from the K solve-for
    path: covariance there now comes from an orthogonal factorization of the
    design matrix, which never forms the normal matrix, and scale invariance
    holds with no prior at all (see test_k_srp_square_root_covariance.py).
    The prior is retained here so this test keeps testing what it always
    tested -- the K ESTIMATE's scale invariance in a well-posed regime -- and
    so its assertions remain comparable with the Phase 17-R baseline.
    """
    x_guess = fx["x_true0"] + np.array([5.0, -3.0, 2.0, 0.0, 0.0, 0.0])
    srp = SRPOptions(k_srp_m2_per_kg=0.01)
    prior_cov = np.diag([10.0 ** 2] * 3 + [0.01 ** 2] * 3)
    results = {}
    for scale in (1e-2, 2e-2):
        _, _, stats = estimate_two_way_range_bls_lm(
            fx["t_pass"], fx["obs"], x_guess, fx["pass_geo"], _MU_MOON, 0.0, 0.0,
            fx["gep"], fx["gsp"], max_iter=15, rtol=1e-12, atol=1e-13,
            srp=srp, solve_for_k_srp=True, k_srp_initial=0.012,
            k_srp_prior_sigma=0.005, prior_covariance=prior_cov,
            k_srp_scale=scale, return_posterior=True,
        )
        results[scale] = stats
    a, b = results[1e-2], results[2e-2]
    assert a.k_srp_estimate == b.k_srp_estimate
    assert a.final_cost == b.final_cost
    assert a.posterior_covariance[6, 6] == pytest.approx(
        b.posterior_covariance[6, 6], rel=1e-9)


# ======================================================================
# s14 - reference configuration must never be mutated by this module
# ======================================================================
def test_estimators_module_never_imports_reference_config():
    """The reference spacecraft's K_SRP stays UNKNOWN by never being touched:
    ``estimators.py`` must not import ``lunar_od.reference_config`` at all,
    so a solve-for run has no path to write into it."""
    import inspect

    from lunar_od import estimators

    src = inspect.getsource(estimators)
    assert "reference_config" not in src


# ======================================================================
# Response responsiveness / zero-sensitivity controls (s55/s56)
# ======================================================================
def test_k_correction_is_nonzero_for_an_informative_arc_with_wrong_initial_k(fx):
    x_guess = fx["x_true0"].copy()
    srp = SRPOptions(k_srp_m2_per_kg=0.01)
    _, _, stats = estimate_two_way_range_bls_lm(
        fx["t_pass"], fx["obs"], x_guess, fx["pass_geo"], _MU_MOON, 0.0, 0.0,
        fx["gep"], fx["gsp"], max_iter=15, rtol=1e-12, atol=1e-13,
        srp=srp, solve_for_k_srp=True, k_srp_initial=0.02,
    )
    assert stats.k_srp_estimate != pytest.approx(0.02, rel=1e-9)


def test_k_information_is_negligible_for_the_earliest_available_observation(fx):
    """Zero-sensitivity control (s56): S_K(t0) = 0 exactly (pinned separately
    in test_srp_k_sensitivity.py), so the EARLIEST observation this fixture
    can generate -- one grid step later, since the receive tag at t = t_grid0
    itself is dropped for want of pre-roll -- must still carry only a tiny
    fraction of the K information a well-informed later observation carries.
    This guards against the estimator inventing K information from nothing.
    """
    from lunar_od.estimators import _two_way_range_posterior_information_with_k

    # pass_geo.t_s (fixed at fixture-build time) must match the grid passed
    # here; only the OBSERVATION subset varies between the two cases.
    obs_early = fx["obs"][fx["obs"][:, 3] == 2.0]
    obs_late = fx["obs"][fx["obs"][:, 3] == float(fx["obs"][:, 3].max())]
    assert obs_early.shape[0] > 0 and obs_late.shape[0] > 0
    srp = SRPOptions(k_srp_m2_per_kg=0.01)
    prior_inv = np.diag([0.0] * 6 + [1.0 / 0.005**2])

    info_early = _two_way_range_posterior_information_with_k(
        fx["t_pass"], obs_early, fx["x_true0"], 0.01, srp, fx["pass_geo"],
        _MU_MOON, 0.0, 0.0, fx["gep"], fx["gsp"],
        prior_inv, np.ones(obs_early.shape[0]), 1e-12, 1e-13, 0.0,
    )
    info_late = _two_way_range_posterior_information_with_k(
        fx["t_pass"], obs_late, fx["x_true0"], 0.01, srp, fx["pass_geo"],
        _MU_MOON, 0.0, 0.0, fx["gep"], fx["gsp"],
        prior_inv, np.ones(obs_late.shape[0]), 1e-12, 1e-13, 0.0,
    )
    data_info_early = info_early[6, 6] - prior_inv[6, 6]
    data_info_late = info_late[6, 6] - prior_inv[6, 6]
    assert data_info_early >= 0.0 and data_info_late >= 0.0
    assert data_info_early < 1e-3 * max(data_info_late, 1e-300)
