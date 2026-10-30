import math

import numpy as np
import pytest

from lunar_od import (
    PassGeometry,
    UnscentedTransformConfig,
    normalized_estimation_error_squared,
    propagate_augmented_state,
    run_lunar_ukf,
)
from lunar_od.ekf import (
    ekf_predict,
    ekf_update,
    finite_difference_measurement_jacobian,
    rts_smooth,
    run_lunar_ekf,
)
from lunar_od.geometry import wrap_to_pi
from tests.test_filters import (
    _build_clean_position_observations,
    _build_clean_range_rate_observations,
    _synthetic_position_station,
    _synthetic_range_rate_station,
)

# ----------------------------------------------------------------------------------------------
# Linear-Gaussian oracle: on a linear model the filter is the Kalman filter and the RTS smoother
# is the batch maximum a posteriori solution over the whole state sequence.
# ----------------------------------------------------------------------------------------------


def _linear_case(seed, n_steps=12, q_scale=0.0):
    rng = np.random.default_rng(seed)
    dt = 10.0
    phi = np.eye(4)
    phi[:2, 2:] = dt * np.eye(2)
    h = np.array([[1.0, 0.0, 0.0, 0.0], [0.3, 1.0, 0.0, 0.0]])
    r = np.diag([4.0, 9.0])
    q = q_scale * np.diag([dt**3 / 3, dt**3 / 3, dt, dt])
    x0 = np.array([100.0, -50.0, 1.0, 2.0])
    p0 = np.diag([25.0, 25.0, 0.04, 0.04])
    z = rng.normal(0.0, 1.0, (n_steps, 2)) * 3.0 + np.arange(n_steps)[:, None] * 10.0
    return phi, h, r, q, x0, p0, z


def _run_linear_filter(phi, h, r, q, x0, p0, z):
    x, p = x0.copy(), p0.copy()
    xs, ps, xps, pps, phis = [], [], [], [], []
    for k in range(z.shape[0]):
        step_phi = np.eye(4) if k == 0 else phi
        step_q = None if k == 0 else q
        x_pred, p_pred = ekf_predict(p, step_phi @ x, step_phi, step_q)
        x, p, diag = ekf_update(x_pred, p_pred, z[k], h @ x_pred, h, r)
        assert diag.accepted
        xs.append(x)
        ps.append(p)
        xps.append(x_pred)
        pps.append(p_pred)
        phis.append(step_phi)
    return [np.asarray(v) for v in (xs, ps, xps, pps, phis)]


def _batch_map(phi, h, r, q, x0, p0, z):
    """Whitened least squares over the stacked states x_0..x_{N-1} (the MAP smoother)."""
    n, nx = z.shape[0], x0.size
    rows, rhs = [], []

    def add(block_cols, value, cov):
        w = np.linalg.cholesky(np.linalg.inv(cov)).T
        row = np.zeros((cov.shape[0], n * nx))
        for col, mat in block_cols:
            row[:, col * nx:(col + 1) * nx] = mat
        rows.append(w @ row)
        rhs.append(w @ value)

    add([(0, np.eye(nx))], x0, p0)
    for k in range(n):
        add([(k, h)], z[k], r)
    for k in range(1, n):
        add([(k, np.eye(nx)), (k - 1, -phi)], np.zeros(nx), q)
    a, b = np.vstack(rows), np.concatenate(rhs)
    info = a.T @ a
    sol = np.linalg.solve(info, a.T @ b)
    cov = np.linalg.inv(info)
    return sol.reshape(n, nx), np.array([cov[k * nx:(k + 1) * nx, k * nx:(k + 1) * nx] for k in range(n)])


def test_rts_equals_batch_map_with_process_noise():
    phi, h, r, q, x0, p0, z = _linear_case(1, q_scale=1e-4)
    xs, ps, xps, pps, phis = _run_linear_filter(phi, h, r, q, x0, p0, z)
    sm = rts_smooth(xs, ps, xps, pps, phis)
    x_map, p_map = _batch_map(phi, h, r, q, x0, p0, z)
    np.testing.assert_allclose(sm.state_estimates, x_map, rtol=0, atol=1e-8)
    np.testing.assert_allclose(sm.covariances, p_map, rtol=1e-8, atol=1e-12)
    # the last smoothed point is the last filtered point
    np.testing.assert_allclose(sm.state_estimates[-1], xs[-1], rtol=0, atol=0)


def test_rts_without_process_noise_equals_batch_least_squares_mapped_through_phi():
    phi, h, r, _, x0, p0, z = _linear_case(2)
    xs, ps, xps, pps, phis = _run_linear_filter(phi, h, r, np.zeros((4, 4)), x0, p0, z)
    sm = rts_smooth(xs, ps, xps, pps, phis)
    # batch least squares for the initial state, then mapped deterministically
    info = np.linalg.inv(p0)
    vec = info @ x0
    mapping = np.eye(4)
    for k in range(z.shape[0]):
        if k:
            mapping = phi @ mapping
        hk = h @ mapping
        info = info + hk.T @ np.linalg.solve(r, hk)
        vec = vec + hk.T @ np.linalg.solve(r, z[k])
    x0_hat = np.linalg.solve(info, vec)
    np.testing.assert_allclose(sm.state_estimates[0], x0_hat, rtol=0, atol=1e-8)
    np.testing.assert_allclose(sm.covariances[0], np.linalg.inv(info), rtol=1e-7, atol=1e-12)
    # with Q = 0 every smoothed point lies on one trajectory
    np.testing.assert_allclose(sm.state_estimates[5], np.linalg.matrix_power(phi, 5) @ x0_hat, rtol=0, atol=1e-7)


def test_filter_last_state_equals_batch_map_last_state():
    phi, h, r, q, x0, p0, z = _linear_case(3, q_scale=1e-3)
    xs, ps, *_ = _run_linear_filter(phi, h, r, q, x0, p0, z)
    x_map, p_map = _batch_map(phi, h, r, q, x0, p0, z)
    np.testing.assert_allclose(xs[-1], x_map[-1], rtol=0, atol=1e-8)
    np.testing.assert_allclose(ps[-1], p_map[-1], rtol=1e-8, atol=1e-12)


def test_joseph_update_matches_standard_form_and_nis_gate_rejects():
    p = np.array([[4.0, 1.0], [1.0, 3.0]])
    h = np.array([[1.0, 2.0]])
    r = np.array([[0.5]])
    x, p_upd, diag = ekf_update(np.zeros(2), p, [1.0], [0.0], h, r)
    k = p @ h.T / (h @ p @ h.T + r)
    np.testing.assert_allclose(p_upd, p - k @ h @ p, rtol=1e-12)
    np.testing.assert_allclose(x, (k * 1.0).ravel(), rtol=1e-12)
    _, p_gated, gated = ekf_update(np.zeros(2), p, [100.0], [0.0], h, r, nis_gate=9.0)
    assert not gated.accepted and gated.normalized_innovation_squared > 9.0
    np.testing.assert_array_equal(p_gated, p)


def test_finite_difference_jacobian_is_exact_for_linear_and_wraps_angles():
    a = np.array([[1.0, -2.0, 0.5], [0.0, 3.0, 1.0]])
    jac = finite_difference_measurement_jacobian(lambda x: a @ x, np.array([1.0, 2.0, 3.0]), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(jac, a, rtol=1e-12)
    # an angle measurement near +-pi: the wrapped difference keeps the derivative
    wrapped = finite_difference_measurement_jacobian(
        lambda x: np.array([wrap_to_pi(x[0])]), np.array([math.pi - 1e-4]), [1e-3],
        residual_fn=lambda u, v: wrap_to_pi(u - v))
    np.testing.assert_allclose(wrapped, [[1.0]], rtol=1e-9)


# ----------------------------------------------------------------------------------------------
# Lunar arcs: the EKF on the SR-UKF's synthetic fixtures, same truth/observations/prior/noise.
# ----------------------------------------------------------------------------------------------

MU_MOON = 4902.800066e9


def _earth(t):
    return np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))


def _sun(t):
    return np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))


def _arc(measurement_type, duration_s=361.0):
    r0 = 1737.4e3 + 100e3
    x_true0 = np.array([r0, 30e3, -20e3, -15.0, math.sqrt(MU_MOON / r0), 4.0])
    t_pass = np.arange(0.0, duration_s, 60.0)
    truth = propagate_augmented_state(t_pass, np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")]),
                                      MU_MOON, 0.0, 0.0, _earth, _sun, rtol=1e-12, atol=1e-13)[:, :6]
    station = _synthetic_position_station if measurement_type == "position" else _synthetic_range_rate_station
    pass_geo = PassGeometry(
        t_s=t_pass, earth_pos_mci_m=np.zeros((t_pass.size, 3)), earth_vel_mci_mps=np.zeros((t_pass.size, 3)),
        x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass.size, axis=0),
        stations=(station(0.0, 0.0, 0.0), station(0.0, 90.0, 0.0), station(45.0, -30.0, 500.0)),
        measurement_type=measurement_type)
    build = _build_clean_position_observations if measurement_type == "position" else _build_clean_range_rate_observations
    return x_true0, t_pass, truth, pass_geo, build(t_pass, truth, pass_geo)


def _add_noise(obs, measurement_type, rng):
    obs = obs.copy()
    if measurement_type == "position":
        obs[:, 1] += rng.normal(0.0, 1.0, obs.shape[0])
        obs[:, 2] = wrap_to_pi(obs[:, 2] + rng.normal(0.0, 1e-5, obs.shape[0]))
        obs[:, 3] = wrap_to_pi(obs[:, 3] + rng.normal(0.0, 1e-5, obs.shape[0]))
    else:
        obs[:, 1] += rng.normal(0.0, 1.0, obs.shape[0])
        obs[:, 2] += rng.normal(0.0, 1e-4, obs.shape[0])
        obs[:, 3] = wrap_to_pi(obs[:, 3] + rng.normal(0.0, 1e-5, obs.shape[0]))
        obs[:, 4] = wrap_to_pi(obs[:, 4] + rng.normal(0.0, 1e-5, obs.shape[0]))
    return obs


def _both(measurement_type, obs, x_guess, p0, t_pass, pass_geo, **kwargs):
    common = (t_pass, obs, x_guess, p0, pass_geo, MU_MOON, 0.0, 0.0, _earth, _sun)
    ekf = run_lunar_ekf(*common, rtol=1e-12, atol=1e-13, smooth=True, **kwargs)
    ukf = run_lunar_ukf(*common, rtol=1e-12, atol=1e-13, config=UnscentedTransformConfig(alpha=0.35), **kwargs)
    return ekf, ukf


@pytest.mark.parametrize("measurement_type", ["position", "range_rate"])
def test_lunar_ekf_agrees_with_srukf_on_the_same_arc(measurement_type):
    x_true0, t_pass, truth, pass_geo, clean = _arc(measurement_type)
    rng = np.random.default_rng(20261030)
    obs = _add_noise(clean, measurement_type, rng)
    p0 = np.diag([20.0**2] * 3 + [0.02**2] * 3)
    x_guess = x_true0 + rng.multivariate_normal(np.zeros(6), p0)
    ekf, ukf = _both(measurement_type, obs, x_guess, p0, t_pass, pass_geo)
    np.testing.assert_array_equal(ekf.obs_indices, ukf.obs_indices)
    sigma = np.sqrt(np.diag(ukf.final_covariance))
    # a 20 m prior is well inside the linear regime: the two filters agree to a small fraction of sigma
    assert np.all(np.abs(ekf.final_state - ukf.final_state) < 0.05 * sigma)
    np.testing.assert_allclose(np.sqrt(np.diag(ekf.final_covariance)), sigma, rtol=0.02)
    np.testing.assert_allclose(ekf.normalized_innovation_squared, ukf.normalized_innovation_squared,
                               rtol=0.05, atol=0.05)
    # the smoother ends on the filter and improves the first epoch
    np.testing.assert_array_equal(ekf.smoothed.state_estimates[-1], ekf.final_state)
    assert np.all(np.diag(ekf.smoothed.covariances[0]) < np.diag(ekf.covariances[0]))
    assert (np.linalg.norm(ekf.smoothed.state_estimates[0][:3] - truth[0, :3])
            < np.linalg.norm(ekf.state_estimates[0][:3] - truth[0, :3]))


def test_lunar_ekf_position_global_bias_recovery():
    x_true0, t_pass, truth, pass_geo, obs = _arc("position")
    true_bias = np.array([20.0, 1.2e-5, -8.0e-6])
    obs[:, 1:4] += true_bias
    x_guess = np.concatenate([x_true0, np.zeros(3)])
    p0 = np.diag([5.0**2] * 3 + [0.01**2] * 3 + [100.0**2, 1e-4**2, 1e-4**2])
    q = np.diag([0.01**2] * 3 + [1e-5**2] * 3)
    ekf, ukf = _both("position", obs, x_guess, p0, t_pass, pass_geo, process_noise=q)
    bias_error = np.abs(ekf.final_state[6:9] - true_bias)
    assert bias_error[0] < 3.0 and bias_error[1] < 4e-6 and bias_error[2] < 4e-6
    assert np.all(np.linalg.eigvalsh(ekf.final_covariance) > 0.0)
    # with Q > 0 the smoothed bias is constant only up to the smoother's spread; it stays on the truth
    assert np.all(np.abs(ekf.smoothed.state_estimates[0][6:9] - true_bias) < [3.0, 4e-6, 4e-6])


@pytest.mark.parametrize("measurement_type", ["position", "range_rate"])
def test_lunar_ekf_and_smoother_are_statistically_consistent(measurement_type):
    x_true0, t_pass, truth, pass_geo, clean = _arc(measurement_type)
    p0 = np.diag([20.0**2] * 3 + [0.02**2] * 3)
    rng = np.random.default_rng(20260608)
    nis, nees_filter, nees_smoothed = [], [], []
    runs = 20
    for _ in range(runs):
        obs = _add_noise(clean, measurement_type, rng)
        x_guess = x_true0 + rng.multivariate_normal(np.zeros(6), p0)
        ekf = run_lunar_ekf(t_pass, obs, x_guess, p0, pass_geo, MU_MOON, 0.0, 0.0, _earth, _sun,
                            rtol=1e-12, atol=1e-13, smooth=True)
        nis.extend(ekf.normalized_innovation_squared)
        nees_filter.append(normalized_estimation_error_squared(ekf.final_state, truth[-1], ekf.final_covariance))
        nees_smoothed.append(normalized_estimation_error_squared(
            ekf.smoothed.state_estimates[0], truth[0], ekf.smoothed.covariances[0]))
    dim = 3 if measurement_type == "position" else 4
    assert abs(np.mean(nis) - dim) < 0.2 * dim
    # mean NEES of 20 runs of a 6-dof chi-square: 95% interval about [4.3, 7.9]
    assert 4.0 < np.mean(nees_filter) < 8.5
    assert 4.0 < np.mean(nees_smoothed) < 8.5
