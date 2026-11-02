"""WP4.4a (EST-11/12): SR-UKF mean in deviation form; jitter only when needed, unit-free.

Before WP4.4a, at the library default alpha = 1e-3 (Wm0 = Wc0 = -1e6) the unscented
mean was formed as sum(w_i X_i) on ~1.8e6 m states and missed an identity transform's
exact mean by 3e-5 m; and every Cholesky added an ABSOLUTE 1e-12 to the diagonal, so a
1 microradian angle variance (1e-12 rad^2) came out doubled and _ensure_psd lifted a
1e-14 variance to 1e-12 (outputs/wp44_probe_before.txt in the WP4.4a worktree).
EST-10 (Cholesky downdate failure) was not reproduced: none of 1200 random strongly
quadratic transforms nor range/az/el transforms up to sigma = 3000 km raised, and an
indefinite covariance raising LinAlgError is the honest outcome, so it is unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

import lunar_od.filters as F

X = np.array([1837.4e3, 30e3, -20e3, -15.0, 1633.0, 4.0])
P = np.diag([100.0**2] * 3 + [0.1**2] * 3)


@pytest.mark.parametrize("alpha", [1e-3, 0.35, 1.0])
def test_identity_transform_recovers_mean_and_covariance(alpha):
    cfg = F.UnscentedTransformConfig(alpha=alpha)
    points, wm, wc = F._sigma_points_from_sqrt(X, np.linalg.cholesky(P), cfg)
    mean, cov = F.unscented_mean_and_covariance(points, wm, wc)
    np.testing.assert_allclose(mean, X, rtol=0.0, atol=1e-9)          # was 3e-5 m at 1e-3
    np.testing.assert_allclose(cov, P, rtol=1e-9, atol=1e-12)
    sqrt_cov = F._unscented_sqrt_covariance(points, mean, wc, additive_covariance=None,
                                            residual_fn=None, jitter=cfg.jitter)
    np.testing.assert_allclose(sqrt_cov @ sqrt_cov.T, P, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("variance", [1e-12, 1e-16, 1e4])
def test_cholesky_of_a_positive_definite_matrix_adds_nothing(variance):
    r = np.diag([variance, 2 * variance])                 # e.g. 1 microrad angles
    lower = F._cholesky_psd(r, F.UnscentedTransformConfig().jitter)
    np.testing.assert_allclose(lower @ lower.T, r, rtol=1e-14, atol=0.0)


def test_jitter_is_relative_to_each_diagonal():
    """A singular block gets jitter in its own units, never in the largest one's."""
    m = np.zeros((3, 3))
    m[0, 0] = 1e4                                          # position, m^2
    m[1:, 1:] = 1e-12 * np.ones((2, 2))                    # two identical angle biases, rad^2
    jitter = F.UnscentedTransformConfig().jitter
    lower = F._cholesky_psd(m, jitter)
    rebuilt = lower @ lower.T
    np.testing.assert_allclose(np.diag(rebuilt), np.diag(m), rtol=1e-6)
    psd = F._ensure_psd(m, jitter)
    np.testing.assert_allclose(np.diag(psd), np.diag(m), rtol=1e-6)


def test_ensure_psd_leaves_small_positive_variances_alone():
    r = np.diag([1e-14, 1e-14])
    np.testing.assert_array_equal(F._ensure_psd(r, F.UnscentedTransformConfig().jitter), r)


def test_ensure_psd_still_repairs_an_indefinite_matrix():
    m = np.array([[1.0, 2.0], [2.0, 1.0]])                 # eigenvalues 3, -1
    out = F._ensure_psd(m, F.UnscentedTransformConfig().jitter)
    assert np.min(np.linalg.eigvalsh(out)) > 0.0
