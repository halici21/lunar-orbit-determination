"""PHASE 17-KP1 - alternative K_SRP parameterizations: core (analysis only).

A candidate model writes the SRP coefficient as a deterministic basis expansion

    K(t) = sum_j p_j b_j(t),     p_j in m^2/kg,   truth p = K_truth * c (c: the constant's coordinates),

so a_SRP = K(t) g(r, t) with the production kernel g, and each parameter's
trajectory sensitivity solves the production variational equation with forcing
b_j(t) g:  dS_j/dt = A S_j + [0; b_j g].  For a basis that is piecewise constant
on grid intervals this is EXACT from the production STM and dx/dK alone:

    J(t) = Phi(t)^-1 dx/dK(t)          (dJ/dt = Phi^-1 [0; g])
    S_window[a,b](t) = Phi(t) (J(clip(t, a, b)) - J(a))

No new integration, no quadrature, no production change.  Every basis below is
piecewise constant on the 90 s grid intervals.  The design columns of a
parameter come from the production range / counted-Doppler Jacobians with S_j
in the STM slot (the HARDEN/F1 K construction); several parameters are packed
into the six slot columns of one call, which is linear in the slot.

Information geometry is computed on the whitened, scaled design
(state scale = phase17_r1od2_core.SCALE; parameter scale = K_truth, i.e.
fractional-K units), never from a normal matrix.
"""
from __future__ import annotations

import json
import math
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parent / "artifacts"
PRE_PATH = ARTIFACTS / "phase17_kp1_predeclaration.json"

K_TRUTH = 0.01                       # m^2/kg, SYNTHETIC_CAMPAIGN_TRUTH
STATE_SCALE = np.array([1e6, 1e6, 1e6, 1e3, 1e3, 1e3])
MODELS = ("M0_SCALAR", "M1_TWO_SEGMENT", "M2_LINEAR_DRIFT", "M3_THREE_SEGMENT")
#: columns of the shared per-cell parameter-sensitivity bank (all piecewise constant on grid intervals)
BANK = ("K", "SEG2_A", "SEG3_A", "SEG3_B", "DRIFT")


class PredeclarationMismatch(RuntimeError):
    pass


def canonical_sha256(d: dict) -> str:
    import hashlib
    body = {k: v for k, v in d.items() if k != "canonical_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_predeclaration(pinned: str, path: Path = PRE_PATH) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    got = canonical_sha256(d)
    if not (got == d.get("canonical_sha256") == pinned):
        raise PredeclarationMismatch("KP1 predeclaration hash mismatch: file %s stored %s pinned %s"
                                     % (got, d.get("canonical_sha256"), pinned))
    return d


# ======================================================================
# Bases
# ======================================================================
def boundaries(t_grid: np.ndarray, n_segments: int) -> list[int]:
    """Node indices of equal-duration segment boundaries (nearest grid node)."""
    t_end = float(t_grid[-1])
    return [0] + [int(np.argmin(np.abs(t_grid - t_end * j / n_segments))) for j in range(1, n_segments)] \
        + [t_grid.size - 1]


def drift_interval_values(t_grid: np.ndarray) -> np.ndarray:
    """Normalized drift s(t) = (t - T/2)/(T/2) at each grid-interval midpoint (staircase basis)."""
    t_end = float(t_grid[-1])
    mid = 0.5 * (t_grid[:-1] + t_grid[1:])
    return (mid - 0.5 * t_end) / (0.5 * t_end)


def model_matrix(model: str) -> np.ndarray:
    """Map bank columns (BANK order) to the model's parameter columns: H_p = H_bank @ M.

    Truth coordinates: every model contains the constant K = K_truth exactly
    (segments all = K_truth; drift = 0)."""
    m = {
        "M0_SCALAR": [[1], [0], [0], [0], [0]],
        # segments: [A, B] with B = K - A
        "M1_TWO_SEGMENT": [[0, 1], [1, -1], [0, 0], [0, 0], [0, 0]],
        # constant + drift
        "M2_LINEAR_DRIFT": [[1, 0], [0, 0], [0, 0], [0, 0], [0, 1]],
        # segments [A, B, C] with C = K - A - B
        "M3_THREE_SEGMENT": [[0, 0, 1], [0, 0, 0], [1, 0, -1], [0, 1, -1], [0, 0, 0]],
    }[model]
    return np.asarray(m, dtype=float)


# ======================================================================
# Exact composition of parameter sensitivities
# ======================================================================
def sensitivity_bank(nom48: np.ndarray, t_grid: np.ndarray) -> dict:
    """{name: (N, 6) sensitivity history} for BANK, from the production STM and dx/dK."""
    phi = np.stack([nom48[i, 6:42].reshape((6, 6), order="F") for i in range(nom48.shape[0])])
    s_k = nom48[:, 42:48]
    j_hist = np.linalg.solve(phi, s_k[:, :, None])[:, :, 0]          # J_n = Phi_n^-1 S_K(t_n)
    n = t_grid.size

    def window(a: int, b: int) -> np.ndarray:
        idx = np.clip(np.arange(n), a, b)
        coeff = j_hist[idx] - j_hist[a]
        return np.einsum("nij,nj->ni", phi, coeff)

    b2, b3 = boundaries(t_grid, 2), boundaries(t_grid, 3)
    drift = drift_interval_values(t_grid)
    d_j = np.diff(j_hist, axis=0) * drift[:, None]                     # sum_i s_i (J_{i+1} - J_i)
    coeff = np.vstack([np.zeros((1, 6)), np.cumsum(d_j, axis=0)])
    return dict(K=s_k.copy(), SEG2_A=window(b2[0], b2[1]), SEG3_A=window(b3[0], b3[1]),
                SEG3_B=window(b3[1], b3[2]), DRIFT=np.einsum("nij,nj->ni", phi, coeff),
                boundaries=dict(two=b2, three=b3), j_hist=j_hist)


def slot_history(nom48: np.ndarray, columns: list[np.ndarray]) -> np.ndarray:
    """42-column history whose STM slot columns hold the given sensitivities (<= 6)."""
    if len(columns) > 6:
        raise ValueError("at most 6 slot columns")
    fake = np.array(nom48[:, :42], dtype=float, copy=True)
    fake[:, 6:42] = 0.0
    for j, col in enumerate(columns):
        fake[:, 6 + 6 * j:12 + 6 * j] = col
    return fake


def range_parameter_columns(rb, bank: dict) -> np.ndarray:
    """(n_range, len(BANK)) production range parameter columns (_two_way_range_k_srp_column)."""
    from lunar_od.estimators import _two_way_range_k_srp_column
    out = []
    for name in BANK:
        fake = np.array(rb.nom48, dtype=float, copy=True)
        fake[:, 42:48] = bank[name]
        out.append(np.asarray(_two_way_range_k_srp_column(rb.obs, fake, rb.h_x0), float))
    return np.column_stack(out)


def doppler_parameter_columns(rb, rows, stations, tc_s: float, bank: dict) -> np.ndarray:
    """(n_doppler, len(BANK)) production counted-Doppler parameter columns on CD1 rows."""
    from lunar_od.radiometrics import (resolve_counted_doppler_station_state_provider,
                                       two_way_counted_doppler_initial_state_jacobian)
    import phase17_r1o_cd1_core as CD
    pg = rb.pass_geo
    cfg = CD.doppler_config(tc_s)
    hist = slot_history(rb.nom48, [bank[n] for n in BANK])
    provs = {s: resolve_counted_doppler_station_state_provider(cfg, stations[s], rb.t_grid, pg.earth_pos_mci_m,
                                                                pg.earth_vel_mci_mps, et0_s=pg.et0_s)
             for s in set(int(x) for x in rows.station)}
    out = np.zeros((rows.n, len(BANK)))
    for i in range(rows.n):
        s = int(rows.station[i])
        h = two_way_counted_doppler_initial_state_jacobian(
            float(rows.t_mid[i]), stations[s], rb.t_grid, hist, pg.earth_pos_mci_m, pg.earth_vel_mci_mps,
            pg.x_j2000_to_itrf93, cfg, et0_s=pg.et0_s, station_state_provider=provs[s])
        out[i] = np.asarray(h, float)[:len(BANK)]
    return out


# ======================================================================
# Information geometry of a parameter block
# ======================================================================
def _orth(a: np.ndarray, rel_tol: float = 1e-13) -> np.ndarray:
    """Orthonormal basis of span(a) by SVD (rank by relative singular-value tolerance)."""
    if a.shape[1] == 0:
        return np.zeros((a.shape[0], 0))
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    keep = s > s[0] * max(rel_tol, a.shape[0] * np.finfo(float).eps)
    return u[:, keep]


def assemble(blocks, nuisance=None, prior_sigma=None):
    """Whitened, scaled [H_x | N] (conditioning set) and H_bank, with optional nuisance prior rows.

    blocks: [(h_x0 (n,6), h_bank (n,nb), w (n,))]; nuisance (N_total, q) or None."""
    hx = np.vstack([b[0] for b in blocks])
    hb = np.vstack([b[1] for b in blocks])
    w = np.concatenate([b[2] for b in blocks])
    sw = np.sqrt(w)
    c = sw[:, None] * (hx * STATE_SCALE)
    p = sw[:, None] * (hb * K_TRUTH)
    if nuisance is not None:
        nu = sw[:, None] * nuisance
        c = np.hstack([c, nu])
        if prior_sigma is not None:
            q = nuisance.shape[1]
            c = np.vstack([c, np.hstack([np.zeros((q, 6)), np.eye(q) / prior_sigma])])
            p = np.vstack([p, np.zeros((q, p.shape[1]))])
    return c, p


def block_metrics(c: np.ndarray, p_bank: np.ndarray, model: str) -> dict:
    """Information geometry of model's parameter block against the conditioning set c."""
    hp = p_bank @ model_matrix(model)
    q = hp.shape[1]
    qc = _orth(c)
    b = hp - qc @ (qc.T @ hp)                                     # P_perp(state[, nuisance]) H_p
    # basis-invariant: principal sines between span(H_p) and span(c)
    qp = _orth(hp)
    s_all = np.linalg.svd(qp - qc @ (qc.T @ qp), compute_uv=False)
    out = dict(model=model, q=q, sines=[float(x) for x in s_all],
               s_min=float(s_all.min()), s_max=float(s_all.max()))
    # variation subspace: span(H_p) against span(c, h_K) -> drop the zero angle of h_K itself
    hk = p_bank[:, [0]]
    if q > 1:
        qck = _orth(np.hstack([c, hk]))
        sv = np.sort(np.linalg.svd(qp - qck @ (qck.T @ qp), compute_uv=False))[::-1][:q - 1]
        out.update(var_sines=[float(x) for x in sv], s_var_min=float(sv.min()), s_var_max=float(sv.max()))
    # conditioned information in the declared (fractional-K) basis, via the projected design
    _, sb, vt = np.linalg.svd(b, full_matrices=False)
    out.update(conditioned_singular_values=[float(x) for x in sb],
               condition=float(sb[0] / sb[-1]) if sb[-1] > 0 else float("inf"),
               weakest_combination=[float(x) for x in vt[-1]])
    if sb[-1] > 0:
        r = np.linalg.qr(b, mode="r")
        rinv = np.linalg.solve(r, np.eye(q))
        cov = rinv @ rinv.T                                            # fractional-K^2 units
        sd = np.sqrt(np.diag(cov))
        corr = cov / np.outer(sd, sd)
        out.update(sigma_frac=[float(x) for x in sd],
                   sigma_m2kg=[float(x * K_TRUTH) for x in sd],
                   max_abs_corr=float(np.max(np.abs(corr - np.eye(q)))) if q > 1 else 0.0,
                   corr=[[float(x) for x in row] for row in corr])
    return out


# ======================================================================
# Independent rebuild route (no _orth, no SVD of the stacked basis)
# ======================================================================
def independent_sines(c: np.ndarray, hp: np.ndarray) -> np.ndarray:
    """Principal sines via lstsq residuals and the generalized eigenproblem
    s^2 = eig(G_p^-1 S), S = R_p^T R_p (R_p = lstsq residual of H_p on c), G_p = H_p^T H_p."""
    coef, *_ = np.linalg.lstsq(c, hp, rcond=None)
    res = hp - c @ coef
    s_mat = res.T @ res
    g = hp.T @ hp
    lam = np.linalg.eigvals(np.linalg.solve(g, s_mat)).real
    return np.sqrt(np.clip(np.sort(lam)[::-1], 0.0, None))


def exact_sines(c: np.ndarray, hp: np.ndarray) -> list[Decimal]:
    """Principal sines from the exact rational Schur complement (float64 inputs are dyadic).

    S = H_p^T H_p - H_p^T C (C^T C)^-1 C^T H_p exactly (Fraction Gauss elimination),
    then s^2 = eig(G_p^-1 S) in 50-digit Decimal via the characteristic polynomial (q <= 3)."""
    a = np.hstack([c, hp])
    n, q = a.shape[1], hp.shape[1]
    af = [[Fraction(float(v)) for v in row] for row in a]
    g = [[Fraction(0)] * n for _ in range(n)]
    for row in af:
        nz = [(i, v) for i, v in enumerate(row) if v]
        for i, vi in nz:
            for j, vj in nz:
                if j >= i:
                    g[i][j] += vi * vj
    for i in range(n):
        for j in range(i):
            g[i][j] = g[j][i]
    gp = [[g[n - q + i][n - q + j] for j in range(q)] for i in range(q)]
    m = [row[:] for row in g]
    k = n - q
    for col in range(k):                          # eliminate the conditioning block
        piv = m[col][col]
        if piv == 0:
            raise ZeroDivisionError("exactly singular conditioning block")
        for r in range(col + 1, n):
            if m[r][col]:
                f = m[r][col] / piv
                m[r] = [vr - f * vc for vr, vc in zip(m[r], m[col])]
    s = [[m[k + i][k + j] for j in range(q)] for i in range(q)]
    # characteristic polynomial det(S - lam G_p), exact, by exact interpolation at q + 1 points
    def det(mat):
        mat = [row[:] for row in mat]
        d = Fraction(1)
        for col in range(len(mat)):
            piv = next((r for r in range(col, len(mat)) if mat[r][col] != 0), None)
            if piv is None:
                return Fraction(0)
            if piv != col:
                mat[col], mat[piv] = mat[piv], mat[col]
                d = -d
            d *= mat[col][col]
            for r in range(col + 1, len(mat)):
                f = mat[r][col] / mat[col][col]
                mat[r] = [vr - f * vc for vr, vc in zip(mat[r], mat[col])]
        return d

    xs = [Fraction(i) for i in range(q + 1)]
    ys = [det([[s[i][j] - x * gp[i][j] for j in range(q)] for i in range(q)]) for x in xs]
    coef = [Fraction(0)] * (q + 1)                                  # Lagrange -> monomial, exact
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        basis = [Fraction(1)]
        den = Fraction(1)
        for j, xj in enumerate(xs):
            if j != i:
                basis = [Fraction(0)] + basis
                for t in range(len(basis) - 1):
                    basis[t] -= xj * basis[t + 1]
                den *= (xi - xj)
        for t in range(q + 1):
            coef[t] += yi * basis[t] / den
    getcontext().prec = 60
    dec = lambda fr: Decimal(fr.numerator) / Decimal(fr.denominator)  # noqa: E731
    cd = [dec(c) for c in coef]
    dd = [cd[t] * t for t in range(1, q + 1)]
    pv = lambda x: sum(cd[t] * x ** t for t in range(q + 1))  # noqa: E731
    dv = lambda x: sum(dd[t - 1] * x ** (t - 1) for t in range(1, q + 1))  # noqa: E731
    g_f = np.array([[float(v) for v in row] for row in gp])
    s_f = np.array([[float(v) for v in row] for row in s])
    guesses = sorted(np.linalg.eigvals(np.linalg.solve(g_f, s_f)).real, reverse=True)
    roots = []
    for g0 in guesses:                                              # Newton refinement at 60 digits
        x = Decimal(repr(float(g0)))
        for _ in range(100):
            step = pv(x) / dv(x)
            x -= step
            if abs(step) < Decimal("1e-45"):
                break
        roots.append(x)
    return [max(r, Decimal(0)).sqrt() for r in sorted(roots, reverse=True)]
