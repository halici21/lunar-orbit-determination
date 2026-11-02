"""WP4.6 (EST-16): chi-square intervals for ANEES/ANIS, rejected-update exclusion,
Ljung-Box whiteness and per-channel residual autocorrelation.

Oracles: Bar-Shalom, Li & Kirubarajan (2001) Example 5.4 (ANEES, N = 50 runs,
n = 2: 95 % interval [1.5, 2.6]); the Wilson-Hilferty approximation for large
degrees of freedom; seeded Monte Carlo calibration of the nominal 5 % error rate.
"""
from __future__ import annotations

import numpy as np
import pytest

from lunar_od.diagnostics import (
    analyze_average_consistency,
    analyze_residuals,
    channel_lag1_autocorrelation,
    chi_square_average_interval,
    ljung_box_test,
)


def test_anees_interval_matches_bar_shalom_example():
    lower, upper = chi_square_average_interval(2, 50, confidence=0.95)
    assert lower == pytest.approx(1.5, abs=0.05)
    assert upper == pytest.approx(2.6, abs=0.05)


@pytest.mark.parametrize("dimension, count", [(6, 100), (3, 1000)])
def test_interval_matches_wilson_hilferty(dimension, count):
    """chi2_p(k) ~ k (1 - 2/(9k) + z_p sqrt(2/(9k)))^3; averaged over `count`."""
    k = dimension * count
    z = 1.959963984540054
    approx = [k * (1 - 2 / (9 * k) + s * z * np.sqrt(2 / (9 * k))) ** 3 / count for s in (-1, 1)]
    np.testing.assert_allclose(chi_square_average_interval(dimension, count), approx, rtol=2e-3)


def test_consistent_anees_covers_its_interval_at_the_nominal_rate():
    rng = np.random.default_rng(46)
    dimension, runs, repeats = 6, 100, 2000
    nees = np.sum(rng.standard_normal((repeats, runs, dimension)) ** 2, axis=2)
    inside = np.mean([analyze_average_consistency(row, dimension).within_interval for row in nees])
    assert 0.93 <= inside <= 0.97          # binomial 3-sigma band around 0.95 is +-0.015


def test_overconfident_covariance_fails_the_interval():
    rng = np.random.default_rng(461)
    nees = np.sum((1.3 * rng.standard_normal((100, 6))) ** 2, axis=1)   # P too small by 1.69x
    result = analyze_average_consistency(nees, 6)
    assert not result.within_interval and result.mean > result.upper


def test_rejected_updates_are_excluded_from_average_nis():
    rng = np.random.default_rng(462)
    nis = np.sum(rng.standard_normal((400, 3)) ** 2, axis=1)
    gated = nis.copy()
    gated[::20] = 400.0                     # outliers the filter rejected
    accepted = np.ones(nis.size, bool)
    accepted[::20] = False
    with_rejected = analyze_average_consistency(gated, 3)
    without = analyze_average_consistency(gated, 3, accepted=accepted)
    assert not with_rejected.within_interval
    assert without.within_interval and without.excluded == 20 and without.count == 380


def test_ljung_box_statistic_matches_the_definition():
    x = np.array([0.3, -1.2, 0.8, 0.1, -0.4, 1.5, -0.9, 0.2, 0.6, -0.7, 0.05, 1.1])
    n, h = x.size, 3
    c = x - x.mean()
    rho = [np.sum(c[:-k] * c[k:]) / np.sum(c * c) for k in range(1, h + 1)]
    q = n * (n + 2) * sum(r * r / (n - k) for k, r in zip(range(1, h + 1), rho))
    result = ljung_box_test(x, max_lag=h)
    assert result.statistic == pytest.approx(q, rel=1e-12)
    assert result.dof == h
    np.testing.assert_allclose(result.autocorrelations, rho, rtol=1e-12)


def test_ljung_box_size_and_power():
    rng = np.random.default_rng(463)
    white = rng.standard_normal((2000, 200))
    size = np.mean([ljung_box_test(row).p_value < 0.05 for row in white])
    assert 0.035 <= size <= 0.065
    ar = np.zeros((200, 200))
    e = rng.standard_normal((200, 200))
    for t in range(1, 200):
        ar[:, t] = 0.5 * ar[:, t - 1] + e[:, t]
    power = np.mean([ljung_box_test(row).p_value < 0.05 for row in ar])
    assert power > 0.99


def test_ljung_box_rejects_too_short_sequences():
    with pytest.raises(ValueError):
        ljung_box_test(np.arange(5.0), max_lag=5)


def test_per_channel_lag1_sees_what_the_interleaved_vector_hides():
    """Range channel AR(1) 0.9 in time, az/el white: the flattened lag-1 mixes channels."""
    rng = np.random.default_rng(464)
    epochs = 3000
    rng_channel = np.zeros(epochs)
    for t in range(1, epochs):
        rng_channel[t] = 0.9 * rng_channel[t - 1] + rng.standard_normal() * np.sqrt(1 - 0.81)
    table = np.column_stack([rng_channel, rng.standard_normal(epochs), rng.standard_normal(epochs)])
    flat = table.reshape(-1)
    per_channel = channel_lag1_autocorrelation(flat, 3)
    assert per_channel[0] == pytest.approx(0.9, abs=0.03)
    assert np.all(np.abs(per_channel[1:]) < 0.06)
    assert abs(analyze_residuals(flat).lag1_autocorrelation) < 0.06
