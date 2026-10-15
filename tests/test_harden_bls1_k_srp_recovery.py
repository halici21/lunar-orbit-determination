"""Phase 17-HARDEN-BLS1 - production BLS K_SRP solve-for recovers from a realistic a priori.

Before BLS1 the K_SRP path of ``estimate_two_way_range_bls_lm`` could not leave a
~100 m a-priori orbit: every case below ended MaxIter / falsely "Converged" with the
orbit still ~120 m off and K walked to a meaningless value (KP2 frozen BLS gate
FAIL).  Three defects, each shown necessary by ablation, were repaired:

  1. the damped LM system was solved through the normal matrix formed in the
     bookkeeping units (1e6 m, 1e3 m/s, k_srp_scale): cond ~1e16 on a design whose
     equilibrated cond is ~1e8, so every small-lambda step was declared singular
     -> ``_lm_step_square_root`` (same system, equilibrated QR);
  2. a trial K < 0 discarded the whole step and raised lambda, which on a coupled
     problem makes the K component MORE negative -> the K-fixed (bound-active) step;
  3. the Jacobian was integrated at the loosened early-iteration tolerance, whose
     error along the weak state/K direction moves the Gauss-Newton fixed point
     -> caller tolerance on the K path.

These tests pin the step's numerical contract and the end-to-end recovery against
an INDEPENDENT stationarity check (column-normalised lstsq at the caller tolerance)
and a solve started from the truth -- never against the answer used as a start.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("spiceypy")

import lunar_od.estimators as E  # noqa: E402
from lunar_od.dynamics import propagate_augmented_state  # noqa: E402
from lunar_od.srp import SRPOptions  # noqa: E402
from lunar_od.two_way_range import generate_two_way_range_measurements  # noqa: E402

from tests import test_k_srp_bls_solve_for as base  # noqa: E402

# the KP2 cross-check a-priori offset (rng([98, 0, 0]) draw, rounded)
DX_APRIORI = np.array([-108.8, -124.5, 56.1, 0.069, 0.043, 0.107])
TOL = dict(rtol=1e-12, atol=1e-13)


def setup_module(_module):
    from lunar_od.spice_loader import load_spice_kernels

    load_spice_kernels()


# ----------------------------------------------------------------------
# 1. the square-root step: same system, scale-free arithmetic
# ----------------------------------------------------------------------
def _random_problem(rng, col_spread):
    m, n = 200, 7
    h = rng.normal(size=(m, n)) * col_spread
    r = rng.normal(size=m)
    w = rng.uniform(0.5, 2.0, m)
    return h, r, w


@pytest.mark.parametrize("with_prior", [False, True])
def test_square_root_step_solves_the_same_damped_system(with_prior):
    rng = np.random.default_rng(11)
    h, r, w = _random_problem(rng, np.ones(7))
    scale = np.diag(rng.uniform(0.5, 2.0, 7))
    prior = np.zeros((7, 7))
    if with_prior:
        prior[:6, :6] = np.diag(rng.uniform(0.1, 1.0, 6))      # K row/column zero: the data-only K case
    prior_s = scale.T @ prior @ scale
    rhs = scale.T @ prior @ rng.normal(size=7)
    for lam in (1e-2, 1.0, 1e3):
        a, *_, sing_a = E._lm_step(h, r, w, scale, prior_s, rhs, lam)
        b, *_, sing_b = E._lm_step_square_root(h, r, w, scale, prior_s, rhs, lam)
        assert not sing_a and not sing_b
        assert np.allclose(a, b, rtol=1e-10, atol=1e-13 * np.max(np.abs(a)))


def test_square_root_step_is_free_of_the_bookkeeping_units():
    """The production structure: a K column nearly in the span of the state columns
    (the weak direction) and range-like unit spreads (pos ~1, vel ~1e4, K ~2e4).
    In the production bookkeeping units the old normal-matrix solve calls the
    small-lambda step singular; the square-root solve does not, two unit choices give
    the same physical step, and lambda -> 0 recovers Gauss-Newton."""
    rng = np.random.default_rng(5)
    m = 200
    hs = rng.normal(size=(m, 6))
    hk = hs @ rng.normal(size=6) + 1e-3 * rng.normal(size=m)
    h = np.column_stack([hs, hk]) * np.array([1.0, 1.0, 1.0, 1e4, 1e4, 1e4, 2e4])
    r, w = rng.normal(size=m), rng.uniform(0.5, 2.0, m)
    z, zr = np.zeros((7, 7)), np.zeros(7)
    s1 = np.diag([1e6, 1e6, 1e6, 1e3, 1e3, 1e3, 1e-2])       # the production bookkeeping units
    s2 = np.diag([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3, 1e-5])
    lam = 1e-14
    _, cond_old, _, sing_old = E._lm_step(h, r, w, s1, z, zr, lam)
    assert sing_old and cond_old > 1e15                      # the BLS1 defect, reproduced
    d1, cond1, _, sing1 = E._lm_step_square_root(h, r, w, s1, z, zr, lam)
    d2, _, _, sing2 = E._lm_step_square_root(h, r, w, s2, z, zr, lam)
    assert not sing1 and not sing2 and cond1 < 1e9           # the problem's own conditioning
    assert np.allclose(d1, d2, rtol=1e-10, atol=0.0)
    ref = np.linalg.lstsq(h * np.sqrt(w)[:, None], np.sqrt(w) * r, rcond=None)[0]
    assert np.allclose(d1, ref, rtol=1e-5)                   # lambda -> 0 limit is Gauss-Newton


# ----------------------------------------------------------------------
# 2. end-to-end recovery from a ~100 m a priori
# ----------------------------------------------------------------------
def _srp_fixture(k_truth, noise_seed):
    fx = base._fixture(duration_s=20000.0, step_s=120.0)
    srp = SRPOptions(k_srp_m2_per_kg=k_truth)
    aug0 = np.concatenate([fx["x_true0"], np.eye(6).reshape(-1, order="F")])
    hist = propagate_augmented_state(fx["t_pass"], aug0, base._MU_MOON, 0.0, 0.0, fx["gep"], fx["gsp"],
                                     srp=srp, **TOL)
    vis = np.ones((fx["t_pass"].size, 1), dtype=bool)
    obs, pg = generate_two_way_range_measurements(fx["t_pass"], hist[:, :6], (base._STATION,), vis,
                                                  fx["gep"], fx["gev"], fx["et0"], noise=False)
    obs = np.array(obs, float, copy=True)
    obs[:, 1] += np.random.default_rng(noise_seed).normal(0.0, 5.0, obs.shape[0])
    return fx, obs, pg


def _solve(fx, obs, pg, x_start, k_start):
    return E.estimate_two_way_range_bls_lm(
        fx["t_pass"], obs, x_start, pg, base._MU_MOON, 0.0, 0.0, fx["gep"], fx["gsp"], **TOL,
        srp=SRPOptions(k_srp_m2_per_kg=k_start), solve_for_k_srp=True, k_srp_initial=k_start)


def _independent_gauss_newton(fx, obs, pg, x, k):
    """Undamped GN step at (x, k) and the formal sigmas, by column-normalised lstsq/QR."""
    srp = SRPOptions(k_srp_m2_per_kg=k)
    aug = E.propagate_state_with_k_sensitivity(fx["t_pass"], x, base._MU_MOON, 0.0, 0.0, fx["gep"],
                                               fx["gsp"], srp=srp, **TOL)
    _, hx = E.two_way_range_nominal_and_initial_jacobian(obs, pg, aug)
    hk = E._two_way_range_k_srp_column(obs, aug, hx)
    hist = E.propagate_state(fx["t_pass"], x, base._MU_MOON, 0.0, 0.0, fx["gep"], fx["gsp"], srp=srp, **TOL)
    r, _ = E.compute_two_way_range_residuals(hist, obs, pg)
    sw = np.sqrt(E._two_way_range_weight_diagonal(obs, pg))
    a = np.hstack([hx, hk[:, None]]) * sw[:, None]
    c = np.linalg.norm(a, axis=0)
    step = np.linalg.lstsq(a / c, sw * np.asarray(r), rcond=None)[0] / c
    r_inv = np.linalg.inv(np.linalg.qr(a / c, mode="r"))
    sd = np.sqrt(np.sum(r_inv ** 2, axis=1)) / c
    return step, sd


def test_k_srp_bls_reaches_the_least_squares_solution_from_a_100m_apriori():
    fx, obs, pg = _srp_fixture(k_truth=0.01, noise_seed=7)
    x_hat, status, stats = _solve(fx, obs, pg, fx["x_true0"] + DX_APRIORI, 0.01)
    assert status in ("Converged", "J-Stab"), status
    step, sd = _independent_gauss_newton(fx, obs, pg, np.asarray(x_hat), stats.k_srp_estimate)
    assert np.max(np.abs(step / sd)) <= 1e-2                 # stationary (pre-BLS1: up to 58 sigma)
    # the same solution from the truth start (a different basin entry, not the answer)
    x_t, _, stats_t = _solve(fx, obs, pg, fx["x_true0"], 0.01)
    diff = np.concatenate([np.asarray(x_t) - x_hat, [stats_t.k_srp_estimate - stats.k_srp_estimate]])
    assert np.max(np.abs(diff / sd)) <= 2e-2


def test_bound_active_step_lets_the_orbit_converge_when_k_sits_at_zero():
    """Truth K = 0 (noise-free): the optimum is ON the bound.  A negative-K trial
    must neither be evaluated nor clipped; the K-fixed step carries the orbit in."""
    fx = base._fixture(duration_s=20000.0)
    x_hat, _, stats = E.estimate_two_way_range_bls_lm(
        fx["t_pass"], fx["obs"], fx["x_true0"] + DX_APRIORI, fx["pass_geo"], base._MU_MOON, 0.0, 0.0,
        fx["gep"], fx["gsp"], **TOL, srp=SRPOptions(k_srp_m2_per_kg=0.01), solve_for_k_srp=True,
        k_srp_initial=0.01)
    assert stats.k_srp_negative_step_rejections >= 1
    assert stats.k_srp_bound_active_steps >= 1
    assert 0.0 < stats.k_srp_estimate < 1e-6                 # approached, never clipped to exactly 0
    assert np.linalg.norm(np.asarray(x_hat)[:3] - fx["x_true0"][:3]) < 1e-2   # pre-BLS1: 122 m
