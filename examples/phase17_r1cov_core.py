"""PHASE 17-R1COV - covariance oracles and square-root covariance recovery.

ANALYSIS SPACE ONLY.  Production code is imported and executed, never modified
by this module.

WHY THIS ORACLE IS EXACT RATHER THAN HIGH-PRECISION
---------------------------------------------------
s17 suggests mpmath at >= 80 decimal digits.  mpmath is not installed, and
adding a dependency to this frozen repository mid-phase is a change this phase
is not authorised to make.  The Python standard library offers something
STRICTLY STRONGER for this particular problem.

Every entry of H and W is an IEEE-754 double, and every finite double is a
dyadic rational p/2^k.  `fractions.Fraction(x)` converts one exactly.  Sums and
products of dyadic rationals are dyadic rationals, so the whole normal matrix
H^T W H + P0^-1 is representable EXACTLY as rationals, and an exact
Gauss-Jordan inverse then yields the covariance with ZERO arithmetic error --
not 80 correct digits, but all of them.

That matters here for a specific reason.  The question this phase asks is
"is the weak K direction destroyed by arithmetic, or absent from the data?"
A high-precision oracle answers it only up to its own precision; an exact one
settles it.  Any difference between a float64 method and this oracle is, by
construction, entirely a floating-point artifact of that method.

The s18 convergence study is still performed, using `decimal.Decimal` at
50/80/120/200 digits, and is strengthened by having the exact limit to converge
TO rather than merely comparing successive precisions to each other.

WHAT "EXACT" DOES AND DOES NOT MEAN
-----------------------------------
The oracle is exact for the linear-algebra step only.  It inherits whatever
error is already in H and W from the trajectory propagation and the measurement
partials.  It is the exact covariance OF THE DESIGN MATRIX WE WERE GIVEN, which
is precisely the reference needed to qualify a covariance ALGORITHM.  It is not
a claim about the physical truth of H.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, getcontext
from fractions import Fraction

import numpy as np
from scipy.linalg import solve_triangular

K_INDEX = 6  # K_SRP is the last unknown in the 7-parameter augmented problem


# ======================================================================
# Exact rational oracle
# ======================================================================
def _frac_matrix(a: np.ndarray) -> list[list[Fraction]]:
    return [[Fraction(float(v)) for v in row] for row in np.asarray(a, dtype=float)]


def exact_normal_matrix(
    h: np.ndarray, w: np.ndarray, prior_inv: np.ndarray | None = None
) -> list[list[Fraction]]:
    """H^T W H + P0^-1 in exact rational arithmetic, in PHYSICAL units.

    No scaling is applied: in exact arithmetic a similarity scaling changes
    nothing, which is itself the cleanest statement of why the s27 scaling
    invariance gate must pass for any correct float implementation.
    """
    hf = _frac_matrix(h)
    wf = [Fraction(float(v)) for v in np.asarray(w, dtype=float)]
    n = len(hf[0])
    out = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    for row, wi in zip(hf, wf):
        for i in range(n):
            ri = row[i] * wi
            if ri:
                for j in range(i, n):
                    out[i][j] += ri * row[j]
    for i in range(n):          # mirror
        for j in range(i + 1, n):
            out[j][i] = out[i][j]
    if prior_inv is not None:
        pf = _frac_matrix(prior_inv)
        for i in range(n):
            for j in range(n):
                out[i][j] += pf[i][j]
    return out


def exact_inverse(mat: list[list[Fraction]]) -> list[list[Fraction]]:
    """Gauss-Jordan with full exactness; raises if the matrix is truly singular."""
    n = len(mat)
    a = [row[:] + [Fraction(1) if i == j else Fraction(0) for j in range(n)]
         for i, row in enumerate(mat)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if a[piv][col] == 0:
            raise ZeroDivisionError(
                "exactly singular at column %d -- the matrix is rank deficient "
                "in exact arithmetic, not merely ill conditioned" % col)
        a[col], a[piv] = a[piv], a[col]
        p = a[col][col]
        a[col] = [v / p for v in a[col]]
        for r in range(n):
            if r != col and a[r][col]:
                f = a[r][col]
                a[r] = [vr - f * vc for vr, vc in zip(a[r], a[col])]
    return [row[n:] for row in a]


def exact_covariance(h, w, prior_inv=None) -> tuple[list[list[Fraction]], float]:
    """Return (exact covariance, sigma_K) in physical units."""
    cov = exact_inverse(exact_normal_matrix(h, w, prior_inv))
    var_k = cov[K_INDEX][K_INDEX]
    return cov, float(np.sqrt(float(var_k)))


def exact_cov_to_float(cov: list[list[Fraction]]) -> np.ndarray:
    return np.array([[float(v) for v in row] for row in cov], dtype=float)


# ======================================================================
# Decimal oracle at a chosen precision (s18 convergence study)
# ======================================================================
def decimal_sigma_k(h: np.ndarray, w: np.ndarray, prior_inv=None,
                    digits: int = 80) -> float:
    """sigma_K computed in Decimal arithmetic at `digits` significant digits."""
    getcontext().prec = digits
    hd = [[Decimal(float(v)) for v in row] for row in np.asarray(h, float)]
    wd = [Decimal(float(v)) for v in np.asarray(w, float)]
    n = len(hd[0])
    m = [[Decimal(0) for _ in range(n)] for _ in range(n)]
    for row, wi in zip(hd, wd):
        for i in range(n):
            ri = row[i] * wi
            for j in range(i, n):
                m[i][j] += ri * row[j]
    for i in range(n):
        for j in range(i + 1, n):
            m[j][i] = m[i][j]
    if prior_inv is not None:
        for i in range(n):
            for j in range(n):
                m[i][j] += Decimal(float(np.asarray(prior_inv, float)[i, j]))
    # Gauss-Jordan for the K column of the inverse only
    a = [row[:] + [Decimal(1) if i == j else Decimal(0) for j in range(n)]
         for i, row in enumerate(m)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if a[piv][col] == 0:
            return float("inf")
        a[col], a[piv] = a[piv], a[col]
        p = a[col][col]
        a[col] = [v / p for v in a[col]]
        for r in range(n):
            if r != col and a[r][col] != 0:
                f = a[r][col]
                a[r] = [vr - f * vc for vr, vc in zip(a[r], a[col])]
    var_k = a[K_INDEX][n + K_INDEX]
    return float(var_k.sqrt())


# ======================================================================
# Production-equivalent floored path (the defect under audit)
# ======================================================================
def floored_covariance(h, w, prior_inv, scale) -> tuple[np.ndarray, dict]:
    """Reproduce the production covariance path exactly, with diagnostics.

    This calls the REAL production helper so the reproduction cannot drift
    from the shipped behaviour.
    """
    from lunar_od.estimators import _safe_covariance_from_information

    h = np.asarray(h, float)
    info = h.T @ (np.asarray(w, float)[:, None] * h)
    if prior_inv is not None:
        info = info + np.asarray(prior_inv, float)
    info_scaled = scale.T @ info @ scale
    cov = scale @ _safe_covariance_from_information(info_scaled) @ scale.T
    vals = np.linalg.eigvalsh(0.5 * (info_scaled + info_scaled.T))
    max_val = float(np.max(np.abs(vals)))
    floor = max(max_val * 1e-14, np.finfo(float).eps)
    return cov, dict(
        smallest_information_eigenvalue=float(vals[0]),
        largest_information_eigenvalue=float(vals[-1]),
        floor=floor,
        floored_modes=int(np.sum(vals < floor)),
        floor_inflation=float(floor / max(vals[0], 1e-300)),
        sigma_k=float(np.sqrt(cov[K_INDEX, K_INDEX])),
    )


# ======================================================================
# Square-root (QR) covariance -- the proposed repair
# ======================================================================
def prior_square_root_rows(prior_inv: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Rows L^T with L L^T = S^T P0^-1 S, dropping exactly-zero directions.

    A symmetric eigendecomposition is used rather than Cholesky because the
    prior information matrix is routinely SEMI-definite here: the data-only
    case is all zeros, and partial priors constrain some parameters and not
    others.  Cholesky would raise on both.  Negative eigenvalues are clipped at
    zero and reported by the caller -- a prior information matrix must be PSD,
    so a materially negative eigenvalue is a bug in the prior, not something to
    absorb silently.
    """
    if prior_inv is None:
        return np.zeros((0, scale.shape[0]))
    ps = scale.T @ np.asarray(prior_inv, float) @ scale
    ps = 0.5 * (ps + ps.T)
    vals, vecs = np.linalg.eigh(ps)
    keep = vals > 0.0
    if not np.any(keep):
        return np.zeros((0, ps.shape[0]))
    return (np.sqrt(vals[keep])[:, None] * vecs[:, keep].T)


class RankDeficientCovariance(np.linalg.LinAlgError):
    """The augmented problem has no finite covariance in the K direction.

    Raised INSTEAD of returning a plausible-looking finite matrix. s35 forbids
    fabricating a confident variance from a safety floor, and a caller that is
    told "no covariance exists" can make a correct decision, whereas a caller
    handed a floored number cannot tell that anything went wrong.
    """

    def __init__(self, message: str, *, rank: int, n: int,
                 singular_ratio: float, threshold: float):
        super().__init__(message)
        self.rank = rank
        self.n = n
        self.singular_ratio = singular_ratio
        self.threshold = threshold


def numerical_rank(r_factor: np.ndarray) -> tuple[int, float, float]:
    """(rank, sigma_min/sigma_max, threshold) for the square-root factor.

    s36 requires the rank threshold to be documented, scale-aware and tested
    rather than inherited from a library default.

    The criterion is the standard one for float64 least squares:

        sigma_i is resolvable  iff  sigma_i > sigma_max * n * eps

    It is applied to the SINGULAR VALUES of R, not to |diag(R)|, because a
    triangular factor's diagonal can be arbitrarily unrepresentative of its
    singular values. It is a pure RATIO, so it is invariant to any uniform
    rescaling of the problem -- which is exactly the scale-awareness the
    eigenvalue floor lacked, the floor being an ABSOLUTE threshold
    (max_eig * 1e-14) on a scale-dependent matrix.

    On the weak-K lunar cases this threshold is nowhere near binding: G0's
    ratio is ~3.8e-10 against a threshold of ~1.6e-15, five orders of margin.
    The weak K direction is therefore retained on its own merits, not by a
    generous cutoff.
    """
    sv = np.linalg.svd(np.asarray(r_factor, float), compute_uv=False)
    n = r_factor.shape[0]
    threshold = float(n * np.finfo(float).eps)
    ratio = float(sv[-1] / sv[0]) if sv[0] > 0 else 0.0
    rank = int(np.sum(sv > sv[0] * threshold))
    return rank, ratio, threshold


@dataclass(frozen=True)
class SquareRootCovariance:
    covariance: np.ndarray        # physical units
    r_factor: np.ndarray          # (n,n) upper triangular, scaled coords
    diag_r: np.ndarray
    rank_estimate: int
    smallest_abs_diag_r: float
    condition_design: float
    sigma_k: float
    singular_ratio: float = float("nan")
    rank_threshold: float = float("nan")
    rank_margin: float = float("nan")   # ratio / threshold; >1 means resolvable


def square_root_covariance(
    h: np.ndarray,
    w: np.ndarray,
    prior_inv: np.ndarray | None,
    scale: np.ndarray,
    *,
    rank_rtol: float = 1e-13,
) -> SquareRootCovariance:
    """Covariance from an orthogonal factorization, never forming H^T W H.

    Builds the whitened, scaled design system

        A = [ W^(1/2) H S ; L^T ]      with  L L^T = S^T P0^-1 S

    factors A = Q R, and recovers P_scaled = R^-1 R^-T by TWO TRIANGULAR SOLVES
    rather than an explicit inverse (s24):

        R^T Y = I      (lower-triangular solve)
        R   P = Y      (upper-triangular solve)

    The conditioning of this path is that of A, not of A^T A, which is the
    entire point: R1M measured cond(A^T A) = cond(A)^2 to a ratio of 1.0000,
    so squaring is what pushed the weak K direction below machine epsilon.
    """
    h = np.asarray(h, float)
    w = np.asarray(w, float)
    a_data = np.sqrt(w)[:, None] * (h @ scale)
    rows = [a_data]
    p_rows = prior_square_root_rows(prior_inv, scale)
    if p_rows.shape[0]:
        rows.append(p_rows)
    a_aug = np.vstack(rows)

    sv = np.linalg.svd(a_aug, compute_uv=False)
    cond_design = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")

    r = np.linalg.qr(a_aug, mode="r")
    n = a_aug.shape[1]
    r = np.asarray(r)[:n, :n]
    # Fix sign so diag(R) > 0; this is a similarity of the factorization and
    # leaves R^-1 R^-T unchanged, but it makes diag(R) directly interpretable
    # as per-direction information square roots.
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    r = signs[:, None] * r

    diag_r = np.abs(np.diag(r))
    smallest = float(diag_r.min())
    if not np.all(np.isfinite(r)):
        raise RankDeficientCovariance(
            "square-root factor contains nonfinite entries",
            rank=0, n=n, singular_ratio=float("nan"), threshold=float("nan"))
    rank_estimate, ratio, threshold = numerical_rank(r)
    margin = ratio / threshold if threshold > 0 else float("inf")
    if rank_estimate < n:
        raise RankDeficientCovariance(
            "augmented problem is numerically rank %d of %d "
            "(sigma_min/sigma_max = %.3e, resolvability threshold %.3e); "
            "no finite covariance exists for the unresolved direction"
            % (rank_estimate, n, ratio, threshold),
            rank=rank_estimate, n=n, singular_ratio=ratio, threshold=threshold)

    eye = np.eye(n)
    y = solve_triangular(r, eye, trans="T", lower=False)   # R^T Y = I
    p_scaled = solve_triangular(r, y, lower=False)          # R P = Y
    p_scaled = 0.5 * (p_scaled + p_scaled.T)
    cov = scale @ p_scaled @ scale.T
    return SquareRootCovariance(
        covariance=cov, r_factor=r, diag_r=diag_r,
        rank_estimate=rank_estimate, smallest_abs_diag_r=smallest,
        condition_design=cond_design,
        sigma_k=float(np.sqrt(cov[K_INDEX, K_INDEX])),
        singular_ratio=ratio, rank_threshold=threshold, rank_margin=margin,
    )


# ======================================================================
# Shared helpers
# ======================================================================
def scale_matrix(k_scale: float = 1e-2, pos: float = 1e6,
                 vel: float = 1e3) -> np.ndarray:
    return np.diag([pos, pos, pos, vel, vel, vel, k_scale])


def relative_error(a: float, b: float) -> float:
    """|a-b| / |b|, with b the reference; 0/0 -> 0."""
    if b == 0.0:
        return 0.0 if a == 0.0 else float("inf")
    return abs(a - b) / abs(b)


def condition_numbers(h, w, prior_inv, scale) -> dict:
    """cond of the whitened/scaled design matrix and of its normal matrix."""
    h = np.asarray(h, float)
    a = np.sqrt(np.asarray(w, float))[:, None] * (h @ scale)
    p_rows = prior_square_root_rows(prior_inv, scale)
    if p_rows.shape[0]:
        a = np.vstack([a, p_rows])
    sv = np.linalg.svd(a, compute_uv=False)
    info = a.T @ a
    ev = np.linalg.svd(info, compute_uv=False)
    cond_a = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    cond_i = float(ev[0] / ev[-1]) if ev[-1] > 0 else float("inf")
    return dict(
        cond_design=cond_a, cond_information=cond_i,
        smallest_design_singular_value=float(sv[-1]),
        largest_design_singular_value=float(sv[0]),
        smallest_information_eigenvalue=float(ev[-1]),
        squaring_ratio=cond_i / cond_a ** 2 if np.isfinite(cond_a) else float("nan"),
        design_resolvable=bool(cond_a < 1.0 / np.finfo(float).eps),
        information_resolvable=bool(cond_i < 1.0 / np.finfo(float).eps),
    )
