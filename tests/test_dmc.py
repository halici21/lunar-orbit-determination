"""Phase D5: dynamic model compensation (Gauss-Markov empirical acceleration) and its EKF/IEKS hook."""

import math

import numpy as np
import pytest

from lunar_od.dmc import DMCOptions, dmc_process_noise, propagate_dmc_state
from lunar_od.dynamics import propagate_state
from lunar_od.ekf import run_lunar_ekf, run_lunar_iterated_smoother
from tests.test_ekf_smoother import MU_MOON, _add_noise, _arc, _earth, _sun

MU_EARTH = 3.986004418e14
J2_MOON = 2.03e-4
TOL = dict(rtol=1e-12, atol=1e-13)
OPTIONS = DMCOptions(sigma_mps2=5e-6, tau_s=1800.0)


def _state9():
    r0 = 1737.4e3 + 100e3
    return np.array([r0, 30e3, -20e3, -15.0, math.sqrt(MU_MOON / r0), 4.0, 3e-6, -2e-6, 1e-6])


def test_dmc_stm_matches_central_finite_differences():
    x0 = _state9()
    t = [0.0, 3600.0]
    force = dict(j2_moon=J2_MOON, **TOL)
    phi = propagate_dmc_state(t, x0, OPTIONS, MU_MOON, MU_EARTH, 0.0, _earth, _sun, **force)[-1, 9:].reshape(
        9, 9, order="F")
    steps = [1.0] * 3 + [1e-3] * 3 + [1e-6] * 3
    for j, h in enumerate(steps):
        plus, minus = x0.copy(), x0.copy()
        plus[j] += h
        minus[j] -= h
        f = [propagate_dmc_state(t, x, OPTIONS, MU_MOON, MU_EARTH, 0.0, _earth, _sun, stm=False, **force)[-1]
             for x in (plus, minus)]
        column = (f[0] - f[1]) / (2.0 * h)
        assert np.linalg.norm(column - phi[:, j]) <= 1e-6 * np.linalg.norm(phi[:, j]), j
    # the w block is the exact Gauss-Markov decay
    np.testing.assert_allclose(phi[6:9, 6:9], math.exp(-3600.0 / OPTIONS.tau_s) * np.eye(3), rtol=1e-12)


def test_dmc_with_zero_acceleration_follows_the_production_propagator():
    x0 = _state9()
    x0[6:] = 0.0
    t = np.arange(0.0, 3601.0, 300.0)
    force = dict(j2_moon=J2_MOON, **TOL)
    dmc = propagate_dmc_state(t, x0, OPTIONS, MU_MOON, MU_EARTH, 0.0, _earth, _sun, stm=False, **force)
    ref = propagate_state(t, x0[:6], MU_MOON, MU_EARTH, 0.0, _earth, _sun, **force)
    assert np.max(np.linalg.norm(dmc[:, :3] - ref[:, :3], axis=1)) <= 1e-3
    assert np.all(dmc[:, 6:] == 0.0)


def test_van_loan_noise_matches_the_gauss_markov_limits():
    for dt in (1.0, 60.0, 1800.0, 7200.0):
        q = dmc_process_noise(OPTIONS, dt)
        expected = OPTIONS.sigma_mps2**2 * (1.0 - math.exp(-2.0 * dt / OPTIONS.tau_s))
        np.testing.assert_allclose(np.diag(q)[6:], expected, rtol=1e-10)
        assert np.all(np.linalg.eigvalsh(q) > -1e-12 * np.max(np.abs(q)))
        np.testing.assert_array_equal(q, q.T)
    # small step: Q -> Qc dt, all of it on w
    dt = 1e-3
    q = dmc_process_noise(OPTIONS, dt)
    np.testing.assert_allclose(q[6:, 6:], 2.0 * OPTIONS.sigma_mps2**2 / OPTIONS.tau_s * dt * np.eye(3), rtol=1e-6)
    assert np.max(np.abs(q[:6, :6])) < 1e-6 * np.max(np.abs(q[6:, 6:]))
    assert not np.any(dmc_process_noise(DMCOptions(0.0, 100.0), 60.0))
    with pytest.raises(ValueError):
        DMCOptions(1e-6, 0.0)


def test_negligible_dmc_reproduces_the_plain_ekf():
    x_true0, t_pass, _, pass_geo, clean = _arc("position")
    rng = np.random.default_rng(3)
    obs = _add_noise(clean, "position", rng)
    p0 = np.diag([20.0**2] * 3 + [0.02**2] * 3)
    x_guess = x_true0 + rng.multivariate_normal(np.zeros(6), p0)
    args = (t_pass, obs, x_guess, p0, pass_geo, MU_MOON, 0.0, 0.0, _earth, _sun)
    plain = run_lunar_ekf(*args, smooth=True, **TOL)
    tiny = DMCOptions(sigma_mps2=1e-15, tau_s=1800.0)
    p9 = np.zeros((9, 9))
    p9[:6, :6] = p0
    p9[6:, 6:] = tiny.sigma_mps2**2 * np.eye(3)
    with_dmc = run_lunar_ekf(t_pass, obs, np.concatenate([x_guess, np.zeros(3)]), p9, pass_geo, MU_MOON, 0.0,
                             0.0, _earth, _sun, smooth=True, dmc=tiny, **TOL)
    # the two runs differ only in the integrator (DOP853 vs VODE Adams) and a 1e-15 m/s^2 noise floor
    np.testing.assert_allclose(with_dmc.state_estimates[:, :6], plain.state_estimates, rtol=0, atol=1e-5)
    np.testing.assert_allclose(with_dmc.smoothed.covariances[:, :6, :6], plain.smoothed.covariances,
                               rtol=1e-5, atol=1e-12)


def test_dmc_rejects_explicit_process_noise():
    x_true0, t_pass, _, pass_geo, obs = _arc("position")
    with pytest.raises(ValueError):
        run_lunar_ekf(t_pass, obs, np.concatenate([x_true0, np.zeros(3)]), np.eye(9), pass_geo, MU_MOON, 0.0,
                      0.0, _earth, _sun, dmc=OPTIONS, process_noise=np.eye(9))


def test_iterated_smoother_with_dmc_absorbs_an_unmodelled_acceleration():
    """A constant 2e-6 m/s^2 along-track push the filter does not model: DMC recovers the orbit."""
    x_true0, t_pass, _, pass_geo, _ = _arc("position", duration_s=1801.0)
    push = 2e-6

    def rhs(_t, y):
        r, v = y[:3], y[3:]
        return np.concatenate([v, -MU_MOON * r / np.linalg.norm(r) ** 3 + push * v / np.linalg.norm(v)])

    from scipy.integrate import solve_ivp
    from tests.test_filters import _build_clean_position_observations
    truth = solve_ivp(rhs, (0.0, t_pass[-1]), x_true0, method="DOP853", t_eval=t_pass, **TOL).y.T
    rng = np.random.default_rng(5)
    obs = _add_noise(_build_clean_position_observations(t_pass, truth, pass_geo), "position", rng)
    p0 = np.diag([200.0**2] * 3 + [0.2**2] * 3)
    x_guess = x_true0 + rng.multivariate_normal(np.zeros(6), p0)
    common = (t_pass, obs)
    plain = run_lunar_iterated_smoother(*common, x_guess, p0, pass_geo, MU_MOON, 0.0, 0.0, _earth, _sun, **TOL)
    p9 = np.zeros((9, 9))
    p9[:6, :6] = p0
    p9[6:, 6:] = OPTIONS.sigma_mps2**2 * np.eye(3)
    dmc = run_lunar_iterated_smoother(*common, np.concatenate([x_guess, np.zeros(3)]), p9, pass_geo, MU_MOON,
                                      0.0, 0.0, _earth, _sun, dmc=OPTIONS, **TOL)
    assert dmc.status == "Converged"

    def rms(run):
        ref = truth[np.searchsorted(t_pass, run.result.t_update_s), :3]
        return float(np.sqrt(np.mean(np.sum((run.result.smoothed.state_estimates[:, :3] - ref) ** 2, axis=1))))

    # a 30 min arc only partly separates the push from the orbit; the D5 study measures the gain
    assert rms(dmc) < rms(plain)
