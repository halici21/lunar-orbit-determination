"""Phase 17-HARDEN-D29 - the SRIF K_SRP path solves data-only state + the required K prior.

Before D29, ``estimate_two_way_range_srif(solve_for_k_srp=True)`` built its prior right-hand side
from the FULL prior square root, so with no explicit state prior the 15 km / 5 m/s step
regularizer (MATLAB P_bar_inv heritage, D28) silently became a Bayesian state prior in the
ESTIMATE, pulling the state toward its starting value. The declared contract
(docs/phase17_harden_d29_srif_k_prior_contract.md) is the BLS one:

    J = r^T W r + (K - K0)^2 / sigma_K^2          (no explicit state prior)

with the default diagonal kept only as the step regularizer (r_bar rows, zero right-hand side).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("spiceypy")

import lunar_od.estimators as E  # noqa: E402
from lunar_od.srp import SRPOptions  # noqa: E402

from tests import test_harden_d25_posterior_dynamics_contract as D25  # noqa: E402

TOL = D25.TOL
K0, SIG_K = 0.01, 0.005
DX = np.array([30.0, -25.0, 20.0, 0.03, -0.02, 0.015])


def setup_module(_module):
    D25.setup_module(_module)


@pytest.fixture(scope="module")
def fx():
    return _build_fixture()


def _build_fixture():
    """3.3 h single-station two-way range, SRP-only dynamics (K = 0.01), seeded 5 m noise."""
    from lunar_od.dynamics import propagate_augmented_state
    from lunar_od.two_way_range import generate_two_way_range_measurements
    f = D25.base._fixture(duration_s=12000.0, step_s=120.0)
    srp = SRPOptions(k_srp_m2_per_kg=K0)
    aug0 = np.concatenate([f["x_true0"], np.eye(6).reshape(-1, order="F")])
    hist = propagate_augmented_state(f["t_pass"], aug0, *D25._dyn(f), srp=srp, **TOL)
    obs, pg = generate_two_way_range_measurements(f["t_pass"], hist[:, :6], (D25.base._STATION,),
                                                  np.ones((f["t_pass"].size, 1), bool), f["gep"], f["gev"], f["et0"],
                                                  noise=False)
    obs = np.array(obs, float, copy=True)
    obs[:, 1] += np.random.default_rng(29).normal(0.0, 5.0, obs.shape[0])
    return dict(f, obs=obs, pg=pg, srp=srp)


def _srif_k(fx, x0, **kw):
    return E.estimate_two_way_range_srif(fx["t_pass"], fx["obs"], x0, fx["pg"], *D25._dyn(fx), max_iter=40, **TOL,
                                         srp=fx["srp"], solve_for_k_srp=True, k_srp_initial=K0,
                                         k_srp_prior_sigma=SIG_K, return_posterior=True, **kw)


def _declared(fx, x, k, state_prior_inv=None):
    """Independent design of the declared problem at (x, k): whitened data rows + the K prior row
    (+ an explicit state prior); returns (covariance, information, data residual)."""
    srp = SRPOptions(k_srp_m2_per_kg=k)
    aug = E.propagate_state_with_k_sensitivity(fx["t_pass"], x, *D25._dyn(fx), srp=srp, **TOL)
    _, hx = E.two_way_range_nominal_and_initial_jacobian(fx["obs"], fx["pg"], aug)
    h = np.hstack([hx, E._two_way_range_k_srp_column(fx["obs"], aug, hx)[:, None]])
    w = E._two_way_range_weight_diagonal(fx["obs"], fx["pg"])
    info = h.T @ (w[:, None] * h)
    info[6, 6] += 1.0 / SIG_K ** 2
    if state_prior_inv is not None:
        info[:6, :6] += state_prior_inv
    hist = E.propagate_state(fx["t_pass"], x, *D25._dyn(fx), srp=srp, **TOL)
    r, _ = E.compute_two_way_range_residuals(hist, fx["obs"], fx["pg"])
    grad = h.T @ (w * np.asarray(r))
    grad[6] += (K0 - k) / SIG_K ** 2
    return np.linalg.inv(info), info, grad


def _sig(c):
    return np.sqrt(np.diag(np.asarray(c)))


def test_srif_k_without_a_state_prior_is_data_only_plus_the_k_prior(fx):
    x, _, st = _srif_k(fx, fx["x_true0"])
    cov, _, grad = _declared(fx, np.asarray(x), st.k_srp_estimate)
    assert np.max(np.abs(_sig(st.posterior_covariance) / _sig(cov) - 1)) <= 1e-8     # posterior = declared problem
    step = cov @ grad                                                                 # Gauss-Newton step of J
    assert np.max(np.abs(step) / _sig(cov)) <= 1e-2                                   # stationary for J
    # the hidden prior (default state rows) is visibly not in the posterior
    _, info, _ = _declared(fx, np.asarray(x), st.k_srp_estimate)
    hist = info.copy()
    hist[:6, :6] += E._prior_information_and_scale(6, E._TWO_WAY_RANGE_NO_BIAS_CFG,
                                                   E._two_way_range_bias_prior_and_scale, None, None)[0]
    assert np.max(np.abs(_sig(np.linalg.inv(hist)) / _sig(cov) - 1)) > 1e-5


def test_srif_k_solution_no_longer_depends_on_the_starting_state(fx):
    xa, _, sa = _srif_k(fx, fx["x_true0"])
    xb, _, sb = _srif_k(fx, fx["x_true0"] + DX)
    d = np.r_[np.asarray(xa) - np.asarray(xb), sa.k_srp_estimate - sb.k_srp_estimate]
    assert np.max(np.abs(d) / _sig(sa.posterior_covariance)) <= 1e-2


def test_srif_k_explicit_state_prior_is_carried_exactly(fx):
    p0 = np.diag([1e4] * 3 + [1e-2] * 3)
    x, _, st = _srif_k(fx, fx["x_true0"], prior_covariance=p0)
    cov, _, _ = _declared(fx, np.asarray(x), st.k_srp_estimate, state_prior_inv=np.linalg.inv(p0))
    assert np.max(np.abs(_sig(st.posterior_covariance) / _sig(cov) - 1)) <= 1e-8


def test_bls_and_srif_k_solve_the_same_problem(fx):
    """Identical declared problem (same data, start, K prior, no state prior): same estimate to the
    solvers' convergence floor, and each posterior equals the declared covariance at its own point."""
    xs, _, ss = _srif_k(fx, fx["x_true0"])
    xb, _, sb = E.estimate_two_way_range_bls_lm(fx["t_pass"], fx["obs"], fx["x_true0"], fx["pg"], *D25._dyn(fx),
                                                max_iter=40, **TOL, srp=fx["srp"], solve_for_k_srp=True,
                                                k_srp_initial=K0, k_srp_prior_sigma=SIG_K, return_posterior=True)
    d = np.r_[np.asarray(xs) - np.asarray(xb), ss.k_srp_estimate - sb.k_srp_estimate]
    assert np.max(np.abs(d) / _sig(sb.posterior_covariance)) <= 2e-2
    cov_b, _, _ = _declared(fx, np.asarray(xb), sb.k_srp_estimate)
    assert np.max(np.abs(_sig(sb.posterior_covariance) / _sig(cov_b) - 1)) <= 1e-8


def test_a_strengthened_step_regularizer_still_does_not_pull_the_estimate(fx, monkeypatch):
    """The historical defect, made visible. The real 15 km / 5 m/s diagonal moves the estimate by only
    ~1e-4 sigma on this arc, so it is strengthened 1e4x (150 m, 5 cm/s): as a step regularizer it must
    leave the solution independent of the start (measured 6.2e-4 sigma); the pre-D29 code, which used it
    as a state prior, depends on the start by 0.27 sigma."""
    real = E._prior_information_and_scale

    def strong(nx, cfg, fn, prior_cov, prior_sqrt):
        pi, ps, sc, pis, pss = real(nx, cfg, fn, prior_cov, prior_sqrt)
        if prior_cov is None and prior_sqrt is None:
            return pi * 1e4, ps * 1e2, sc, pis * 1e4, pss * 1e2
        return pi, ps, sc, pis, pss

    monkeypatch.setattr(E, "_prior_information_and_scale", strong)
    xa, _, sa = _srif_k(fx, fx["x_true0"])
    xb, _, sb = _srif_k(fx, fx["x_true0"] + DX)
    cov, _, _ = _declared(fx, np.asarray(xa), sa.k_srp_estimate)
    d = np.r_[np.asarray(xa) - np.asarray(xb), sa.k_srp_estimate - sb.k_srp_estimate]
    assert np.max(np.abs(d) / _sig(cov)) <= 1e-2
