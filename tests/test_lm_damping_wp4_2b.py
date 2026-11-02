"""WP4.2b (EST-08): gain-ratio Levenberg-Marquardt damping.

The historical BLS-LM schedule divides lambda by 5 after every accepted step and
multiplies it by 10 after every rejected one, whatever the linear model predicted,
and damps with diag(A^T A), which leaves a column carried only by the prior undamped.
``damping_schedule="nielsen"`` updates lambda from the gain ratio
rho = actual / predicted decrease (Nielsen 1999; Madsen, Nielsen & Tingleff 2004,
eq. 3.16) and damps with More's (1978) running maximum of diag(A^T A + P).
Oracles: the published update rule at hand-computed rho values, and agreement with
the fixed schedule at the minimum (one objective, one certificate: only the path
to the minimum may differ).
"""
from __future__ import annotations

import numpy as np
import pytest

from lunar_od.dmc import DMCOptions
from lunar_od.dmc_batch import estimate_position_dmc_batch
from lunar_od.estimators import (
    LM_DAMPING_SCHEDULES,
    _LMDamping,
    estimate_position_bls_lm,
    estimate_range_rate_bls_lm,
)
from tests.test_ekf_smoother import MU_MOON, _add_noise, _arc, _earth, _sun

TOL = dict(rtol=1e-12, atol=1e-13)
CERTIFIED = {"Converged", "J-Stab"}


@pytest.mark.parametrize("rho, factor", [(1.0, 1.0 / 3.0), (0.75, 0.875), (0.5, 1.0), (0.25, 1.125),
                                         (0.0, 2.0), (3.0, 1.0 / 3.0)])
def test_nielsen_update_follows_the_published_rule(rho, factor):
    """lambda <- lambda max(1/3, 1 - (2 rho - 1)^3) after an accepted step."""
    damping = _LMDamping(1e-2, "nielsen")
    damping.accepted(rho * 4.0, 4.0)
    assert damping.value == pytest.approx(1e-2 * factor, rel=1e-15)


def test_nielsen_rejections_double_nu_and_an_acceptance_resets_it():
    damping = _LMDamping(1.0, "nielsen")
    for expected in (2.0, 8.0, 64.0):                   # nu = 2, 4, 8
        damping.rejected()
        assert damping.value == expected
    damping.accepted(1.0, 1.0)                          # rho = 1 -> / 3, nu back to 2
    damping.rejected()
    assert damping.value == pytest.approx(64.0 / 3.0 * 2.0, rel=1e-15)


def test_fixed_schedule_is_the_historical_one():
    damping = _LMDamping(1e-2, "fixed")
    damping.accepted(0.0, 1.0)                          # gain ratio ignored
    assert damping.value == 1e-2 / 5.0
    damping.rejected()
    assert damping.value == 1e-2 / 5.0 * 10.0
    assert damping.damping_diag(np.eye(2), np.ones(2), np.eye(2), np.eye(2)) is None


def test_more_damping_matrix_includes_the_prior_and_never_shrinks():
    damping = _LMDamping(1e-2, "nielsen")
    h = np.array([[2.0, 0.0], [0.0, 0.0]])              # column 2 has no data information
    prior = np.diag([0.0, 9.0])
    d1 = damping.damping_diag(h, np.ones(2), np.eye(2), prior).copy()
    np.testing.assert_array_equal(d1, [4.0, 9.0])
    d2 = damping.damping_diag(0.5 * h, np.ones(2), np.eye(2), prior)
    np.testing.assert_array_equal(d2, [4.0, 9.0])       # running maximum


def test_unknown_schedule_rejected():
    assert LM_DAMPING_SCHEDULES == ("fixed", "nielsen")
    with pytest.raises(ValueError):
        _LMDamping(1e-2, "marquardt")


def _guesses(x_true0, rng, n):
    p0 = np.diag([4000.0**2] * 3 + [4.0**2] * 3)        # 20x the 200 m / 0.2 m/s cold start
    return [x_true0 + rng.multivariate_normal(np.zeros(6), p0) for _ in range(n)]


@pytest.mark.parametrize("measurement_type", ["position", "range_rate"])
def test_nielsen_reaches_the_fixed_schedule_minimum(measurement_type):
    """Measured on 12 position cases (1x and 20x cold start): both certified, the two
    minima within 1.7e-4 posterior sigma; iterations mean 12.3 (fixed) vs 10.1, worst
    25 vs 18."""
    estimator = estimate_position_bls_lm if measurement_type == "position" else estimate_range_rate_bls_lm
    x_true0, t, _, geo, clean = _arc(measurement_type, duration_s=3601.0)
    rng = np.random.default_rng(21)
    obs = _add_noise(clean, measurement_type, rng)
    for guess in _guesses(x_true0, rng, 2):
        fits = {s: estimator(t, obs, guess, geo, MU_MOON, 0.0, 0.0, _earth, _sun, damping_schedule=s,
                             return_posterior=True, **TOL) for s in LM_DAMPING_SCHEDULES}
        assert {fits[s][1] for s in fits} <= CERTIFIED
        sigma = np.sqrt(np.diag(fits["fixed"][2].posterior_covariance))
        assert np.max(np.abs(fits["fixed"][0] - fits["nielsen"][0]) / sigma) < 1e-3


def test_nielsen_in_the_dmc_batch():
    options = DMCOptions(sigma_mps2=1e-5, tau_s=1800.0, axes="rtn")
    x_true0, t, _, geo, clean = _arc("position", duration_s=3601.0)
    obs = _add_noise(clean, "position", np.random.default_rng(22))
    guess = np.concatenate([x_true0 + np.array([150.0, -80.0, 60.0, 0.05, -0.03, 0.02]), np.zeros(3)])
    fits = {s: estimate_position_dmc_batch(t, obs, guess, geo, MU_MOON, 0.0, 0.0, _earth, _sun, options,
                                           damping_schedule=s, return_posterior=True, **TOL)
            for s in LM_DAMPING_SCHEDULES}
    assert {fits[s][1] for s in fits} <= CERTIFIED
    sigma = np.sqrt(np.diag(fits["fixed"][2].posterior_covariance))
    assert np.max(np.abs(fits["fixed"][0] - fits["nielsen"][0]) / sigma) < 1e-3
