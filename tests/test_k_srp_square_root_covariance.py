"""Phase 17-R1COV - qualification tests for the square-root covariance path.

These exercise `_square_root_covariance_from_design` directly, without SPICE or
a trajectory propagation, so they are fast and depend only on linear algebra.

The reference throughout is EXACT rational arithmetic via `fractions.Fraction`.
Every IEEE-754 double is a dyadic rational, so the normal matrix is
representable exactly and an exact Gauss-Jordan inverse has zero arithmetic
error. A float64 method can then be compared against the true answer rather
than against another float64 method.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from lunar_od.estimators import (
    RankDeficientCovarianceError,
    _safe_covariance_from_information,
    _square_root_covariance_from_design,
)

K = 6
N = 7
ZERO_PRIOR = np.zeros((N, N))


def _scale(k_scale: float = 1e-2, pos: float = 1e6, vel: float = 1e3) -> np.ndarray:
    return np.diag([pos, pos, pos, vel, vel, vel, k_scale])


def _exact_sigma_k(h: np.ndarray, w: np.ndarray, prior=None) -> float:
    """sigma_K from exact rational arithmetic; the reference for these tests."""
    hf = [[Fraction(float(v)) for v in row] for row in np.asarray(h, float)]
    wf = [Fraction(float(v)) for v in np.asarray(w, float)]
    n = len(hf[0])
    m = [[Fraction(0)] * n for _ in range(n)]
    for row, wi in zip(hf, wf):
        for i in range(n):
            ri = row[i] * wi
            if ri:
                for j in range(n):
                    m[i][j] += ri * row[j]
    if prior is not None:
        for i in range(n):
            for j in range(n):
                m[i][j] += Fraction(float(np.asarray(prior, float)[i, j]))
    aug = [m[i][:] + [Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if aug[piv][col] == 0:
            raise ZeroDivisionError("exactly singular")
        aug[col], aug[piv] = aug[piv], aug[col]
        p = aug[col][col]
        aug[col] = [v / p for v in aug[col]]
        for r in range(n):
            if r != col and aug[r][col]:
                f = aug[r][col]
                aug[r] = [vr - f * vc for vr, vc in zip(aug[r], aug[col])]
    return float(np.sqrt(float(aug[K][n + K])))


def _design(weak: float, rows: int = 60, seed: int = 11):
    """A 7-parameter design whose last (K) direction has singular value `weak`."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((rows, N)))
    v, _ = np.linalg.qr(rng.standard_normal((N, N)))
    sv = np.concatenate([np.geomspace(1.0, 0.5, N - 1), [weak]])
    return q @ (sv[:, None] * v.T), np.ones(rows)


# ======================================================================
# Agreement with the exact oracle
# ======================================================================
@pytest.mark.parametrize("weak", [1e-2, 1e-4, 1e-6, 1e-8])
def test_square_root_covariance_matches_exact_rational_oracle(weak):
    h, w = _design(weak)
    cov, _ = _square_root_covariance_from_design(h, w, ZERO_PRIOR, np.eye(N))
    sigma = float(np.sqrt(cov[K, K]))
    assert sigma == pytest.approx(_exact_sigma_k(h, w), rel=1e-8)


def test_square_root_covariance_is_symmetric_and_positive_definite():
    h, w = _design(1e-6)
    cov, _ = _square_root_covariance_from_design(h, w, ZERO_PRIOR, np.eye(N))
    assert np.allclose(cov, cov.T, rtol=0, atol=0)
    assert float(np.min(np.linalg.eigvalsh(cov))) > 0.0


def test_returned_sqrt_information_reproduces_the_information_matrix():
    """R_phys^T R_phys must equal H^T W H, the matrix that was never inverted."""
    h, w = _design(1e-5)
    scale = _scale()
    _, r = _square_root_covariance_from_design(h, w, ZERO_PRIOR, scale)
    assert np.allclose(r, np.triu(r), rtol=0, atol=1e-12 * np.abs(r).max())
    info = h.T @ (w[:, None] * h)
    assert r.T @ r == pytest.approx(info, rel=1e-9)


# ======================================================================
# The properties the floored path failed
# ======================================================================
@pytest.mark.parametrize("weak", [1e-4, 1e-7])
def test_sigma_k_is_invariant_to_the_k_bookkeeping_scale(weak):
    """The defect R1M identified: the old path's sigma_K scaled with K_scale.

    K_scale is a numerical bookkeeping choice, so a physical uncertainty may
    not depend on it. No prior is used, which is exactly the regime where the
    old floored path failed this.
    """
    h, w = _design(weak)
    sigmas = []
    for k_scale in (5e-3, 1e-2, 2e-2):
        cov, _ = _square_root_covariance_from_design(
            h, w, ZERO_PRIOR, _scale(k_scale=k_scale))
        sigmas.append(float(np.sqrt(cov[K, K])))
    assert max(sigmas) / min(sigmas) - 1.0 < 1e-9


def test_floored_path_is_the_one_that_moves_with_the_scale():
    """Characterization of the defect, so the repair cannot silently regress."""
    h, w = _design(1e-7)
    floored = []
    for k_scale in (5e-3, 1e-2, 2e-2):
        s = _scale(k_scale=k_scale)
        info = s.T @ (h.T @ (w[:, None] * h)) @ s
        cov = s @ _safe_covariance_from_information(info) @ s.T
        floored.append(float(np.sqrt(cov[K, K])))
    assert max(floored) / min(floored) > 1.5


def test_posterior_sigma_k_responds_to_the_prior():
    """Broad and moderate K priors must NOT give the same answer.

    R1G observed identical sigma_K for sigma=1.0 and sigma=0.01 priors; R1M
    showed the reason was that neither prior was reaching the answer.
    """
    h, w = _design(1e-7)
    out = {}
    for name, psig in (("broad", 1.0), ("moderate", 0.01)):
        prior = np.zeros((N, N))
        prior[K, K] = 1.0 / psig ** 2
        cov, _ = _square_root_covariance_from_design(h, w, prior, _scale())
        out[name] = float(np.sqrt(cov[K, K]))
        assert out[name] == pytest.approx(_exact_sigma_k(h, w, prior), rel=1e-8)
    assert out["broad"] > out["moderate"]
    assert abs(out["broad"] - out["moderate"]) / out["moderate"] > 1e-3


def test_semidefinite_prior_is_accepted():
    """A prior constraining only K is singular; Cholesky would raise on it."""
    h, w = _design(1e-5)
    prior = np.zeros((N, N))
    prior[K, K] = 100.0
    cov, _ = _square_root_covariance_from_design(h, w, prior, _scale())
    assert np.isfinite(cov).all()
    assert float(np.sqrt(cov[K, K])) == pytest.approx(
        _exact_sigma_k(h, w, prior), rel=1e-8)


# ======================================================================
# Rank semantics
# ======================================================================
@pytest.mark.parametrize("weak", [1e-6, 1e-8, 1e-10])
def test_weak_but_resolvable_direction_is_retained(weak):
    """Weak must mean a LARGE uncertainty, never a discarded direction."""
    h, w = _design(weak)
    cov, _ = _square_root_covariance_from_design(h, w, ZERO_PRIOR, np.eye(N))
    sigma = float(np.sqrt(cov[K, K]))
    assert sigma == pytest.approx(_exact_sigma_k(h, w), rel=1e-6)
    assert np.isfinite(sigma) and sigma > 0.0


def test_sigma_k_grows_inversely_with_the_weak_singular_value():
    """sigma_K ~ 1/weak, so weakening the direction must inflate the uncertainty.

    The proportionality constant is the overlap between the K coordinate and
    the weak singular direction, which this fixture's random rotation fixes at
    ~4.6e-3; asserting proportionality avoids baking that constant in.
    """
    products = []
    for weak in (1e-6, 1e-8, 1e-10):
        h, w = _design(weak)
        cov, _ = _square_root_covariance_from_design(h, w, ZERO_PRIOR, np.eye(N))
        products.append(float(np.sqrt(cov[K, K])) * weak)
    assert max(products) / min(products) == pytest.approx(1.0, rel=1e-6)


def test_exactly_rank_deficient_problem_raises_instead_of_fabricating():
    """A duplicated column carries no new information; refuse, do not invent."""
    h, w = _design(1.0)
    h[:, K] = h[:, 0]                     # bitwise duplicate: exactly dependent
    with pytest.raises(RankDeficientCovarianceError) as excinfo:
        _square_root_covariance_from_design(h, w, ZERO_PRIOR, np.eye(N))
    assert excinfo.value.rank < excinfo.value.n
    assert excinfo.value.singular_ratio < excinfo.value.threshold


def test_floored_path_fabricates_a_finite_covariance_where_we_now_refuse():
    """Characterization of the behaviour being replaced."""
    h, w = _design(1.0)
    h[:, K] = h[:, 0]
    info = h.T @ (w[:, None] * h)
    cov = _safe_covariance_from_information(info)
    assert np.isfinite(cov[K, K]) and cov[K, K] > 0.0


def test_rank_decision_is_invariant_to_uniform_rescaling():
    """The criterion is a singular-value RATIO, unlike the absolute floor."""
    h, w = _design(1e-8)
    base, _ = _square_root_covariance_from_design(h, w, ZERO_PRIOR, np.eye(N))
    for mult in (1e-6, 1e6):
        cov, _ = _square_root_covariance_from_design(
            h * mult, w, ZERO_PRIOR, np.eye(N))
        assert float(np.sqrt(cov[K, K])) == pytest.approx(
            float(np.sqrt(base[K, K])) / mult, rel=1e-8)


# ======================================================================
# Scope protection
# ======================================================================
def test_default_six_state_helper_is_unchanged_and_still_floors():
    """s39/s40: the shared helper keeps its exact historical behaviour."""
    rng = np.random.default_rng(3)
    a = rng.standard_normal((6, 6))
    info = a @ a.T
    vals, vecs = np.linalg.eigh(0.5 * (info + info.T))
    floor = max(float(np.max(np.abs(vals))) * 1e-14, np.finfo(float).eps)
    expected = (vecs / np.clip(vals, floor, None)) @ vecs.T
    got = _safe_covariance_from_information(info)
    assert got == pytest.approx(0.5 * (expected + expected.T), rel=0, abs=0)
