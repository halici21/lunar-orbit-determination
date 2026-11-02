"""WP4.3 (EST-06/07): batch posterior covariance by design QR, one unit-free singularity test.

Before WP4.3 the six-state batch posteriors inverted the information matrix after
clipping its eigenvalues at max*1e-14 in physical units.  With station_full biases the
information spread reached 6e-14 although the equilibrated problem has cond ~1e4, and
the reported position sigma came out 0.25 % small (outputs/wp43_probe_*.txt in the
WP4.3 worktree).  Oracle: the inverse of the same information matrix after Jacobi
equilibration, which is well conditioned.  BLS-LM and SRIF also tested different
quantities against different limits (normal-matrix cond 1e15 vs cond(R) 1e14, both in
bookkeeping units).
"""
from __future__ import annotations

import numpy as np
import pytest

import lunar_od.estimators as E
from tests.test_srif_acceptance_wp4_1 import MU, OFFSET, X_TRUE, _arc, _earth, _sun


def _equilibrated_inverse(info):
    d = 1.0 / np.sqrt(np.diag(info))
    return d[:, None] * np.linalg.inv(d[:, None] * info * d[None, :]) * d[None, :]


@pytest.fixture(scope="module")
def position_arc():
    return _arc("position")


@pytest.mark.parametrize("estimator", [E.estimate_position_bls_lm, E.estimate_position_srif])
@pytest.mark.parametrize("bias_mode", ["station_angles", "station_full"])
def test_bias_augmented_covariance_matches_equilibrated_reference(position_arc, estimator, bias_mode):
    t, obs, geo = position_arc
    nb = {"station_angles": 2, "station_full": 3}[bias_mode] * len(geo.stations)
    x0 = np.concatenate([X_TRUE + OFFSET, np.zeros(nb)])
    _, reason, stats = estimator(t, obs, x0, geo, MU, 0.0, 0.0, _earth, _sun, max_iter=40,
                                 rtol=1e-12, atol=1e-13, bias_mode=bias_mode, return_posterior=True)
    assert reason == "Converged"
    reference = _equilibrated_inverse(stats.posterior_information)
    np.testing.assert_allclose(np.sqrt(np.diag(stats.posterior_covariance)),
                               np.sqrt(np.diag(reference)), rtol=1e-8)
    np.testing.assert_allclose(stats.posterior_covariance, reference, rtol=1e-6,
                               atol=1e-10 * np.max(np.abs(reference)))


def test_bls_and_srif_report_the_same_condition_number(position_arc):
    t, obs, geo = position_arc
    x0 = np.concatenate([X_TRUE + OFFSET, np.zeros(3 * len(geo.stations))])
    kwargs = dict(max_iter=40, rtol=1e-12, atol=1e-13, bias_mode="station_full")
    _, _, bls = E.estimate_position_bls_lm(t, obs, x0, geo, MU, 0.0, 0.0, _earth, _sun, **kwargs)
    _, _, srif = E.estimate_position_srif(t, obs, x0, geo, MU, 0.0, 0.0, _earth, _sun, **kwargs)
    assert bls.rank == srif.rank == x0.size
    assert bls.condition_number == pytest.approx(srif.condition_number, rel=1e-2)


def _random_design(seed, m=40, n=7, spread=1e5):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((m, n)) * np.logspace(0, np.log10(spread), n)


@pytest.mark.parametrize("seed", range(4))
def test_lm_step_condition_is_invariant_to_bookkeeping_units(seed):
    """The same problem in other units must get the same condition, rank and verdict."""
    h = _random_design(seed)
    residual = np.random.default_rng(100 + seed).standard_normal(h.shape[0])
    w = np.full(h.shape[0], 1e4)
    zero = np.zeros((h.shape[1], h.shape[1]))
    reference = E._lm_step(h, residual, w, np.eye(h.shape[1]), zero, np.zeros(h.shape[1]), 1e-6)
    units = np.diag(np.logspace(-6, 9, h.shape[1]))   # metres-to-micro, K scale, ...
    other = E._lm_step(h, residual, w, units, zero, np.zeros(h.shape[1]), 1e-6)
    assert other[1] == pytest.approx(reference[1], rel=1e-6)
    assert other[2] == reference[2] and other[3] == reference[3] is False
    np.testing.assert_allclose(other[0], reference[0], rtol=1e-8)


@pytest.mark.parametrize("seed", range(4))
def test_factor_and_normal_conditions_agree(seed):
    """cond(R_eq)**2 (SRIF) and cond(N_eq) (BLS) are one quantity, so one limit serves both."""
    h = _random_design(seed, spread=1e3)
    r = np.linalg.qr(h, mode="r")
    from_factor = E._equilibrated_condition(r, normal=False)
    from_normal = E._equilibrated_condition(h.T @ h, normal=True)
    assert from_factor[1] == from_normal[1] == h.shape[1]
    assert from_factor[0] == pytest.approx(from_normal[0], rel=1e-6)


def test_singular_limit_is_shared():
    assert E.SINGULAR_CONDITION_LIMIT == 1e15
    h = _random_design(9)
    h[:, -1] = h[:, 0] * 3.0 + 1e-12 * h[:, 1]     # numerically dependent column
    _, cond, _, singular = E._lm_step(h, np.ones(h.shape[0]), np.ones(h.shape[0]), np.eye(7),
                                      np.zeros((7, 7)), np.zeros(7), 0.0)
    assert singular and cond > E.SINGULAR_CONDITION_LIMIT
