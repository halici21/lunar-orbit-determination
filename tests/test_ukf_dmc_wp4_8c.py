"""WP4.8c (EST-21): dynamic model compensation in the SR-UKF.

Before WP4.8c only the EKF/IEKS carried the Gauss-Markov empirical acceleration, so a
UKF-vs-EKF comparison on an arc with unmodelled forces compared different models.
Oracles: the EKF with the same ``dmc=`` (an independent linearised implementation of
the same model; on a dense, near-linear position arc both are close to the optimal
filter), the plain UKF when DMC is negligible, and the injected along-track push.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from lunar_od.dmc import DMCOptions
from lunar_od.ekf import run_lunar_ekf
from lunar_od.filters import UnscentedTransformConfig, run_lunar_ukf
from tests.test_ekf_smoother import MU_MOON, _add_noise, _arc, _earth, _sun
from tests.test_filters import _build_clean_position_observations

TOL = dict(rtol=1e-12, atol=1e-13)
CFG = UnscentedTransformConfig(alpha=0.35)
PUSH = 2e-6


def _p9(p0, sigma):
    p9 = np.zeros((9, 9))
    p9[:6, :6] = p0
    p9[6:, 6:] = sigma**2 * np.eye(3)
    return p9


def _pushed_arc(seed):
    x_true0, t_pass, _, pass_geo, _ = _arc("position", duration_s=1801.0)

    def rhs(_t, y):
        r, v = y[:3], y[3:]
        return np.concatenate([v, -MU_MOON * r / np.linalg.norm(r) ** 3 + PUSH * v / np.linalg.norm(v)])

    truth = solve_ivp(rhs, (0.0, t_pass[-1]), x_true0, method="DOP853", t_eval=t_pass, **TOL).y.T
    rng = np.random.default_rng(seed)
    obs = _add_noise(_build_clean_position_observations(t_pass, truth, pass_geo), "position", rng)
    p0 = np.diag([200.0**2] * 3 + [0.2**2] * 3)
    x_guess = x_true0 + rng.multivariate_normal(np.zeros(6), p0)
    return t_pass, obs, pass_geo, truth, p0, x_guess


def test_negligible_dmc_reproduces_the_plain_ukf():
    """Near the linear regime: a 9-state UKF spreads its sigma points with a different
    lambda(n) than a 6-state one, so with a wide prior the two differ at second order
    (2e-5 relative in sigma at a 20 m prior); at 1 m / 1 mm/s that term vanishes."""
    x_true0, t_pass, _, pass_geo, clean = _arc("position")
    rng = np.random.default_rng(3)
    obs = _add_noise(clean, "position", rng)
    p0 = np.diag([1.0**2] * 3 + [1e-3**2] * 3)
    x_guess = x_true0 + rng.multivariate_normal(np.zeros(6), p0)
    common = (t_pass, obs)
    plain = run_lunar_ukf(*common, x_guess, p0, pass_geo, MU_MOON, 0.0, 0.0, _earth, _sun, config=CFG, **TOL)
    tiny = DMCOptions(sigma_mps2=1e-15, tau_s=1800.0)
    with_dmc = run_lunar_ukf(*common, np.concatenate([x_guess, np.zeros(3)]), _p9(p0, tiny.sigma_mps2), pass_geo,
                             MU_MOON, 0.0, 0.0, _earth, _sun, config=CFG, dmc=tiny, **TOL)
    # the runs also differ in the integrator (DOP853 vs the production propagator; measured
    # 2e-6 m, test_dmc bounds the same gap at 1e-3 m) and a 1e-15 m/s^2 floor
    np.testing.assert_allclose(with_dmc.state_estimates[:, :6], plain.state_estimates, rtol=0, atol=1e-5)
    sigma = np.sqrt(np.einsum("kii->ki", plain.covariances))
    np.testing.assert_allclose(np.sqrt(np.einsum("kii->ki", with_dmc.covariances[:, :6, :6])), sigma,
                               rtol=1e-7)                                   # measured 3.5e-9


@pytest.mark.parametrize("axes", ["inertial", "rtn"])
def test_ukf_dmc_agrees_with_ekf_dmc(axes):
    """In the linear regime (1 m / 1 mm/s prior on the truth) UKF and EKF are the same
    optimal filter, so they must agree to integration accuracy; with the 200 m prior
    they differ by ~0.13 sigma with DMC and 1.4 sigma without, which is nonlinearity."""
    t_pass, obs, pass_geo, truth, _, _ = _pushed_arc(seed=5)
    options = DMCOptions(sigma_mps2=5e-6, tau_s=1800.0, axes=axes)
    x9 = np.concatenate([truth[0], np.zeros(3)])
    p9 = _p9(np.diag([1.0] * 3 + [1e-3**2] * 3), options.sigma_mps2)
    args = (t_pass, obs, x9, p9, pass_geo, MU_MOON, 0.0, 0.0, _earth, _sun)
    ukf = run_lunar_ukf(*args, config=CFG, dmc=options, **TOL)
    ekf = run_lunar_ekf(*args, dmc=options, **TOL)
    sigma = np.sqrt(np.diag(ekf.covariances[-1]))
    assert np.max(np.abs(ukf.final_state - ekf.state_estimates[-1]) / sigma) < 1e-4    # measured 1.5e-6
    np.testing.assert_allclose(np.sqrt(np.diag(ukf.final_covariance)), sigma, rtol=1e-7)  # measured 5.5e-10


def test_ukf_dmc_absorbs_an_unmodelled_acceleration():
    t_pass, obs, pass_geo, truth, p0, x_guess = _pushed_arc(seed=7)
    common = (t_pass, obs)
    plain = run_lunar_ukf(*common, x_guess, p0, pass_geo, MU_MOON, 0.0, 0.0, _earth, _sun, config=CFG, **TOL)
    options = DMCOptions(sigma_mps2=0.0, tau_s=3600.0, axes="rtn", sigma_axes_mps2=(1e-7, 5e-6, 1e-7))
    dmc = run_lunar_ukf(*common, np.concatenate([x_guess, np.zeros(3)]), _p9(p0, 5e-6), pass_geo, MU_MOON,
                        0.0, 0.0, _earth, _sun, config=CFG, dmc=options, **TOL)
    err_plain = np.linalg.norm(plain.final_state[:3] - truth[-1, :3])
    err_dmc = np.linalg.norm(dmc.final_state[:3] - truth[-1, :3])
    assert err_dmc < err_plain
    w_t = dmc.final_state[7]                         # along-track empirical acceleration
    assert abs(w_t - PUSH) < 3.0 * np.sqrt(dmc.final_covariance[7, 7])


def test_ukf_dmc_rejects_unsupported_combinations():
    x_true0, t_pass, _, pass_geo, obs = _arc("position")
    x9 = np.concatenate([x_true0, np.zeros(3)])
    args = (t_pass, obs, x9, np.eye(9), pass_geo, MU_MOON, 0.0, 0.0, _earth, _sun)
    options = DMCOptions(5e-6, 1800.0)
    with pytest.raises(ValueError):
        run_lunar_ukf(*args, dmc=options, process_noise=np.eye(9))
    with pytest.raises(ValueError):
        run_lunar_ukf(*args, dmc=options, use_stm_linearization=True)
    with pytest.raises(ValueError):
        run_lunar_ukf(t_pass, obs, x_true0, np.eye(6), pass_geo, MU_MOON, 0.0, 0.0, _earth, _sun, dmc=options)
