"""WP4.5b (EST-15): Gauss-Markov measurement noise whitened inside the batch estimators.

WP4.5 made time-correlated noise generable, but every estimator still weighted it as
white, so the posterior covariance was optimistic by the AR(1) variance inflation
(measured: mean NEES 77 instead of 6 at tau = 600 s on 60 s sampling).  With
``measurement_correlation_time_s`` the residuals and the design are Prais-Winsten
whitened per station and component.  Oracles: the dense whitening matrix and the dense
Gauss-Markov covariance of ``noise_models`` (r^T Sigma^-1 r), a Monte Carlo NEES
against chi-square bounds with noise drawn from a dense Cholesky factor (independent of
the whitening code), and the white fit when tau -> 0.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2

from lunar_od.dmc import DMCOptions
from lunar_od.dmc_batch import estimate_position_dmc_batch
from lunar_od.estimators import (
    _gauss_markov_whitening,
    _position_weight_diagonal,
    estimate_position_bls_lm,
    estimate_range_rate_bls_lm,
)
from lunar_od.measurements import measurement_sigma_vector
from lunar_od.noise_models import gauss_markov_covariance, gauss_markov_whitening_matrix
from tests.test_ekf_smoother import MU_MOON, _add_noise, _arc, _earth, _sun

TOL = dict(rtol=1e-12, atol=1e-13)
TAU = 600.0
LAYOUT = {"position": (estimate_position_bls_lm, 3, 4), "range_rate": (estimate_range_rate_bls_lm, 4, 5)}


def _series(obs, components, station_col):
    times = np.repeat(obs[:, 0], components)
    groups = (np.repeat(obs[:, station_col].astype(int), components) * components
              + np.tile(np.arange(components), obs.shape[0]))
    return times, groups


def test_whitening_is_the_dense_prais_winsten_transform():
    _, _, _, geo, obs = _arc("position", duration_s=1801.0)
    w = _position_weight_diagonal(obs, geo)
    sigma = 1.0 / np.sqrt(w)
    times, groups = _series(obs, 3, 4)
    whiten = _gauss_markov_whitening(obs, w, TAU, 3)
    r = np.random.default_rng(1).standard_normal(times.size) * sigma
    dense = gauss_markov_whitening_matrix(times, groups, TAU)
    np.testing.assert_allclose(whiten(r), dense @ (r / sigma), rtol=1e-13, atol=1e-13)
    h = np.random.default_rng(2).standard_normal((times.size, 6))
    reference = dense @ (h / sigma[:, None])
    # rows are differences of ~1e5 (angle / 1e-5 rad) terms: roundoff 1e-15 of the row scale
    np.testing.assert_allclose(whiten(h), reference, rtol=0.0, atol=1e-14 * np.max(np.abs(reference)))
    covariance = sigma[:, None] * gauss_markov_covariance(times, groups, TAU) * sigma[None, :]
    assert whiten(r) @ whiten(r) == pytest.approx(r @ np.linalg.solve(covariance, r), rel=1e-10)


@pytest.mark.parametrize("measurement_type", ["position", "range_rate"])
def test_whitened_posterior_is_consistent_under_gauss_markov_noise(measurement_type):
    """50 trials: the mean NEES of the whitened fit lies inside the two-sided 99.9 %
    chi-square(300)/50 interval [4.40, 7.79]; white weighting of the same data does not
    (measured mean NEES ~76 vs ~6.8 with 20 trials)."""
    estimator, components, station_col = LAYOUT[measurement_type]
    x0, t, _, geo, clean = _arc(measurement_type, duration_s=3601.0)
    sigma = measurement_sigma_vector(clean, geo, measurement_type)
    times, groups = _series(clean, components, station_col)
    root = np.linalg.cholesky(gauss_markov_covariance(times, groups, TAU))
    trials = 50
    nees = {None: [], TAU: []}
    for k in range(trials):
        rng = np.random.default_rng(1000 + k)
        obs = clean.copy()
        obs[:, 1:1 + components] += (sigma * (root @ rng.standard_normal(times.size))).reshape(-1, components)
        for tau in nees:
            x, status, stats = estimator(t, obs, x0, geo, MU_MOON, 0.0, 0.0, _earth, _sun,
                                         measurement_correlation_time_s=tau, return_posterior=True, **TOL)
            assert status in {"Converged", "J-Stab"}
            err = x - x0
            nees[tau].append(float(err @ np.linalg.solve(stats.posterior_covariance, err)))
    lo, hi = chi2.ppf([0.0005, 0.9995], 6 * trials) / trials
    assert lo < np.mean(nees[TAU]) < hi
    assert np.mean(nees[None]) > hi


@pytest.mark.parametrize("measurement_type", ["position", "range_rate"])
def test_vanishing_correlation_time_reproduces_the_white_fit(measurement_type):
    estimator, _, _ = LAYOUT[measurement_type]
    x0, t, _, geo, clean = _arc(measurement_type, duration_s=1801.0)
    obs = _add_noise(clean, measurement_type, np.random.default_rng(3))
    white = estimator(t, obs, x0, geo, MU_MOON, 0.0, 0.0, _earth, _sun, return_posterior=True, **TOL)
    tiny = estimator(t, obs, x0, geo, MU_MOON, 0.0, 0.0, _earth, _sun, return_posterior=True,
                     measurement_correlation_time_s=1e-6, **TOL)
    sigma = np.sqrt(np.diag(white[2].posterior_covariance))
    assert np.max(np.abs(white[0] - tiny[0]) / sigma) < 1e-6
    np.testing.assert_allclose(tiny[2].posterior_covariance, white[2].posterior_covariance, rtol=1e-9)
    np.testing.assert_allclose(tiny[2].posterior_information, white[2].posterior_information, rtol=1e-9)


def test_dmc_batch_whitening():
    options = DMCOptions(sigma_mps2=1e-5, tau_s=1800.0)
    x0, t, _, geo, clean = _arc("position", duration_s=1801.0)
    obs = _add_noise(clean, "position", np.random.default_rng(4))
    guess = np.concatenate([x0, np.zeros(3)])
    fit = lambda tau: estimate_position_dmc_batch(t, obs, guess, geo, MU_MOON, 0.0, 0.0, _earth, _sun, options,
                                                  measurement_correlation_time_s=tau, return_posterior=True, **TOL)
    white, tiny, correlated = fit(None), fit(1e-6), fit(TAU)
    sigma = np.sqrt(np.diag(white[2].posterior_covariance))
    # both stop at the WP4.2 certificate (decrement <= 0.1 sigma), not at an exact
    # minimum; the weakly determined w0 leaves 1.3e-4 sigma between the two endpoints
    assert np.max(np.abs(white[0] - tiny[0]) / sigma) < 1e-3
    # correlated noise carries less information: every posterior sigma grows
    assert np.all(np.diag(correlated[2].posterior_covariance) > np.diag(white[2].posterior_covariance))


def test_invalid_whitening_inputs_rejected():
    x0, t, _, geo, obs = _arc("range_rate")
    args = (t, obs, x0, geo, MU_MOON, 0.0, 0.0, _earth, _sun)
    with pytest.raises(ValueError):
        estimate_range_rate_bls_lm(*args, measurement_correlation_time_s=TAU, robust_outlier_rejection=True)
    with pytest.raises(ValueError):
        estimate_range_rate_bls_lm(*args, measurement_correlation_time_s=0.0)
