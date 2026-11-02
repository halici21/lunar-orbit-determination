"""WP4.5 (EST-15): station-wise, time-based AR(1) noise and its whitening.

Before WP4.5 the AR(1) generator chained consecutive ROWS: interleaved stations were
correlated with each other, and the last sample of one pass with the first of the
next however long the gap.  Nothing whitened correlated noise either.  Now the noise
is a first-order Gauss-Markov process per station, rho_k = exp(-dt_k / tau), and a
Prais-Winsten whitening matrix W with W Sigma W^T = I is available.  Oracles: the
exact Gauss-Markov covariance Sigma_ij = exp(-|t_i - t_j| / tau) within a station,
0 across stations, and the closed-form variance of a mean under that covariance.
"""
from __future__ import annotations

import numpy as np
import pytest

from lunar_od.noise_models import (
    MeasurementNoiseConfig,
    ar1_variance_inflation,
    gauss_markov_covariance,
    gauss_markov_whitening_matrix,
    generate_measurement_noise,
)

TAU = 300.0


def _two_stations_two_passes():
    pass_a = np.arange(0.0, 600.0, 60.0)
    times = np.concatenate([pass_a, pass_a + 30.0, pass_a + 20_000.0])
    groups = np.array(["A"] * 10 + ["B"] * 10 + ["A"] * 10)
    order = np.argsort(times, kind="stable")          # interleaved rows, as a schedule is
    return times[order], groups[order]


def _sample(times, groups, n_trials, seed):
    cfg = MeasurementNoiseConfig(model="ar1", correlation_time_s=TAU)
    rng = np.random.default_rng(seed)
    return np.stack([
        generate_measurement_noise(times.size, np.eye(1), rng=rng, config=cfg,
                                   times=times, groups=groups)[:, 0]
        for _ in range(n_trials)
    ])


def test_gauss_markov_covariance_is_station_and_time_based():
    times, groups = _two_stations_two_passes()
    sigma = gauss_markov_covariance(times, groups, TAU)
    for i in range(times.size):
        for j in range(times.size):
            expected = np.exp(-abs(times[i] - times[j]) / TAU) if groups[i] == groups[j] else 0.0
            assert sigma[i, j] == pytest.approx(expected, rel=1e-14, abs=1e-300)


def test_ar1_noise_matches_the_gauss_markov_covariance():
    times, groups = _two_stations_two_passes()
    samples = _sample(times, groups, 20_000, seed=45)
    empirical = samples.T @ samples / samples.shape[0]
    np.testing.assert_allclose(empirical, gauss_markov_covariance(times, groups, TAU), atol=0.04)


def test_ar1_inter_pass_and_inter_station_correlation_near_zero():
    times, groups = _two_stations_two_passes()
    samples = _sample(times, groups, 20_000, seed=46)
    a = np.flatnonzero(groups == "A")
    b = np.flatnonzero(groups == "B")
    last_of_pass1, first_of_pass2 = a[9], a[10]       # 19 460 s apart, 65 tau
    assert abs(np.corrcoef(samples[:, last_of_pass1], samples[:, first_of_pass2])[0, 1]) < 0.03
    neighbours = [(i, i + 1) for i in range(times.size - 1) if groups[i] != groups[i + 1]]
    assert neighbours                                   # stations do interleave
    for i, j in neighbours:                              # was rho for adjacent rows
        assert abs(np.corrcoef(samples[:, i], samples[:, j])[0, 1]) < 0.03


def test_whitening_matrix_is_exact():
    times, groups = _two_stations_two_passes()
    times = times + np.random.default_rng(7).uniform(0.0, 20.0, times.size)   # irregular
    order = np.argsort(times, kind="stable")
    times, groups = times[order], groups[order]
    sigma = gauss_markov_covariance(times, groups, TAU)
    w = gauss_markov_whitening_matrix(times, groups, TAU)
    np.testing.assert_allclose(w @ sigma @ w.T, np.eye(times.size), atol=1e-12)


def test_whitened_residuals_are_white():
    times = np.arange(0.0, 3000.0, 10.0)
    groups = np.zeros(times.size, dtype=int)
    samples = _sample(times, groups, 400, seed=47)
    w = gauss_markov_whitening_matrix(times, groups, TAU)
    raw_lag1 = np.mean([np.corrcoef(s[:-1], s[1:])[0, 1] for s in samples])
    white = samples @ w.T
    white_lag1 = np.mean([np.corrcoef(s[:-1], s[1:])[0, 1] for s in white])
    assert raw_lag1 == pytest.approx(np.exp(-10.0 / TAU), abs=0.02)
    assert abs(white_lag1) < 0.01
    assert np.var(white) == pytest.approx(1.0, abs=0.02)


def test_variance_inflation_matches_the_exact_mean_variance():
    """Var(mean) = 1^T Sigma 1 / n^2; white weighting reports 1/n, so the factor is
    n * Var(mean); the closed form for a uniform grid is (1+rho)/(1-rho) - 2 rho (1 -
    rho^n) / (n (1-rho)^2)."""
    for dt, n in [(10.0, 300), (60.0, 50), (600.0, 20)]:
        times = np.arange(n) * dt
        groups = np.zeros(n, dtype=int)
        exact = gauss_markov_covariance(times, groups, TAU).sum() / n
        rho = np.exp(-dt / TAU)
        closed = (1 + rho) / (1 - rho) - 2 * rho * (1 - rho**n) / (n * (1 - rho) ** 2)
        assert exact == pytest.approx(closed, rel=1e-12)
        assert ar1_variance_inflation(times, groups, TAU) == pytest.approx(exact, rel=1e-12)


def test_legacy_index_ar1_is_unchanged_without_times():
    cfg = MeasurementNoiseConfig(model="ar1", ar1_coefficient=0.8)
    a = generate_measurement_noise(50, np.eye(2), rng=np.random.default_rng(3), config=cfg)
    rng = np.random.default_rng(3)
    innovations = rng.standard_normal((50, 2))
    expected = np.zeros_like(innovations)
    expected[0] = innovations[0]
    for k in range(1, 50):
        expected[k] = 0.8 * expected[k - 1] + np.sqrt(1 - 0.64) * innovations[k]
    np.testing.assert_allclose(a, expected, rtol=1e-14)


def test_time_based_ar1_requires_times_and_a_positive_tau():
    with pytest.raises(ValueError):
        MeasurementNoiseConfig(model="ar1", correlation_time_s=0.0)
    cfg = MeasurementNoiseConfig(model="ar1", correlation_time_s=TAU)
    with pytest.raises(ValueError):
        generate_measurement_noise(5, np.eye(1), rng=np.random.default_rng(0), config=cfg)
    with pytest.raises(ValueError):
        gauss_markov_whitening_matrix(np.array([0.0, 10.0, 5.0]), np.zeros(3), TAU)
