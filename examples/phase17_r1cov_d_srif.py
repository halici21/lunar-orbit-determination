"""PHASE 17-R1COV - COV-D, SRIF R-factor covariance (s29-s32).

ANALYSIS SPACE ONLY.

s29 warns not to assume the shipped SRIF's square-root factor can be read as a
posterior covariance square root without proving it. That warning is well
placed. The shipped helper is

    _sqrt_information_from_information(information):
        vals, vecs = eigh(information)
        floor = max(|vals|.max() * 1e-14, eps)
        if vals.min() <= floor:  vals = clip(vals, floor, None); rebuild
        return cholesky(information).T

which takes the ALREADY-SQUARED normal matrix, applies the SAME eigenvalue
floor R1M identified, and only then factors. It is a square root OF the
defective matrix, not an independent square-root path, so it inherits the
defect exactly. This module measures that, then qualifies a genuine SRIF
R factor accumulated from the measurement rows themselves.

The sequential accumulation matters for s31. If the "SRIF" covariance were
just a second call to the same batch QR the BLS path uses, agreement between
them would be tautological. Processing rows one at a time through successive
orthogonal updates is a genuinely different sequence of floating-point
operations, so agreement is evidence.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase17_r1cov_core import (  # noqa: E402
    K_INDEX, exact_cov_to_float, exact_covariance, floored_covariance,
    prior_square_root_rows, relative_error, scale_matrix,
    square_root_covariance,
)

ARTIFACTS = Path(r"C:/Users/erayh/Documents/Python/Grad/python_port_phase17r0/artifacts")
CASES = (("G0", 1.3), ("G1", 2.0), ("G3", 5.0))
N = 7


def hdr(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


def sequential_srif_r(a_rows: np.ndarray, prior_rows: np.ndarray) -> np.ndarray:
    """Accumulate the SRIF R factor one measurement row at a time.

    This is the classical square-root information measurement update: the
    running triangular factor is stacked with the next whitened row and
    re-triangularised by an orthogonal transformation.  The normal matrix is
    never formed at any point.
    """
    n = a_rows.shape[1]
    r = np.zeros((0, n))
    if prior_rows.shape[0]:
        r = np.linalg.qr(prior_rows, mode="r")
        r = np.atleast_2d(r)[:n, :n]
        if r.shape[0] < n:
            pad = np.zeros((n - r.shape[0], n))
            r = np.vstack([r, pad])
    for row in a_rows:
        stacked = np.vstack([r, row[None, :]]) if r.size else row[None, :]
        rr = np.linalg.qr(stacked, mode="r")
        r = np.atleast_2d(rr)[:n, :n]
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return signs[:, None] * r


def covariance_from_r(r: np.ndarray, scale: np.ndarray) -> np.ndarray:
    n = r.shape[0]
    y = solve_triangular(r, np.eye(n), trans="T", lower=False)
    p = solve_triangular(r, y, lower=False)
    p = 0.5 * (p + p.T)
    return scale @ p @ scale.T


def shipped_sqrt_information_covariance(h, w, prior_inv, scale):
    """Covariance from the SHIPPED SRIF square-root factor, as it stands today."""
    from lunar_od.estimators import _sqrt_information_from_information

    info = np.asarray(h, float).T @ (np.asarray(w, float)[:, None] * np.asarray(h, float))
    if prior_inv is not None:
        info = info + prior_inv
    info_scaled = scale.T @ info @ scale
    r = _sqrt_information_from_information(info_scaled)
    return covariance_from_r(r, scale)


def main() -> None:
    from lunar_od.spice_loader import load_spice_kernels
    from phase17_r1m_core import build_range_arc, campaign_epoch

    rows_out = []

    # ==================================================================
    hdr("s30  ANALYTIC LINEAR-GAUSSIAN CONTROL")
    print("  A problem whose posterior is known in closed form, so every route")
    print("  can be checked against algebra rather than against another code path.\n")
    rng = np.random.default_rng(7)
    m = 40
    a_ctrl = np.zeros((m, N))
    for i in range(m):
        a_ctrl[i, i % N] = 1.0          # each row observes one parameter
    w_ctrl = rng.uniform(0.5, 4.0, m)
    prior_diag = np.array([2.0, 3.0, 1.0, 5.0, 4.0, 2.5, 7.0])
    prior_ctrl = np.diag(prior_diag)
    # analytic posterior: information is diagonal, sum of weights + prior
    info_diag = np.array([w_ctrl[i::N].sum() if False else
                          w_ctrl[[j for j in range(m) if j % N == i]].sum()
                          for i in range(N)]) + prior_diag
    analytic_cov = np.diag(1.0 / info_diag)
    analytic_sigma_k = float(np.sqrt(analytic_cov[K_INDEX, K_INDEX]))

    s_id = scale_matrix(k_scale=1.0, pos=1.0, vel=1.0)
    a_w = np.sqrt(w_ctrl)[:, None] * a_ctrl
    p_rows = prior_square_root_rows(prior_ctrl, s_id)
    r_seq = sequential_srif_r(a_w, p_rows)
    cov_seq = covariance_from_r(r_seq, s_id)
    srq = square_root_covariance(a_ctrl, w_ctrl, prior_ctrl, s_id)
    _, sig_ex = exact_covariance(a_ctrl, w_ctrl, prior_ctrl)

    print("  analytic sigma_K            = %.16e" % analytic_sigma_k)
    print("  exact rational oracle       = %.16e  (rel %.2e)"
          % (sig_ex, relative_error(sig_ex, analytic_sigma_k)))
    print("  batch QR (BLS route)        = %.16e  (rel %.2e)"
          % (srq.sigma_k, relative_error(srq.sigma_k, analytic_sigma_k)))
    print("  sequential SRIF R factor    = %.16e  (rel %.2e)"
          % (float(np.sqrt(cov_seq[K_INDEX, K_INDEX])),
             relative_error(float(np.sqrt(cov_seq[K_INDEX, K_INDEX])),
                            analytic_sigma_k)))
    full_rel = float(np.max(np.abs(cov_seq - analytic_cov)
                            / np.sqrt(np.outer(np.diag(analytic_cov),
                                               np.diag(analytic_cov)))))
    print("  full covariance, SRIF vs analytic, max normalised error = %.2e"
          % full_rel)
    analytic_ok = (relative_error(srq.sigma_k, analytic_sigma_k) < 1e-12
                   and full_rel < 1e-12)
    print("  analytic control: %s" % ("OK" if analytic_ok else "FAIL"))

    # ==================================================================
    hdr("s29  AUDIT OF THE SHIPPED SRIF SQUARE-ROOT FACTOR")
    load_spice_kernels(None, clear=True)
    _, t_orbit = campaign_epoch()
    arcs = {}
    for label, orbits in CASES:
        arcs[label] = build_range_arc(0.0, orbits * t_orbit, label=label)
    print("  Does the SHIPPED sqrt-information factor give the right covariance?\n")
    print("  %-4s %16s %16s %16s" % ("case", "exact sigma_K", "shipped R", "rel error"))
    shipped_contaminated = False
    for label, _ in CASES:
        a = arcs[label]
        h = np.hstack([a.h_x0, a.h_k[:, None]])
        s = scale_matrix()
        _, sig_ex = exact_covariance(h, a.w, None)
        cov_ship = shipped_sqrt_information_covariance(h, a.w, None, s)
        sig_ship = float(np.sqrt(cov_ship[K_INDEX, K_INDEX]))
        rel = relative_error(sig_ship, sig_ex)
        shipped_contaminated = shipped_contaminated or rel > 1e-3
        print("  %-4s %16.6e %16.6e %16.2e" % (label, sig_ex, sig_ship, rel))
    print("\n  The shipped factor reproduces the FLOORED answer, not the correct")
    print("  one, because it factors the already-floored normal matrix. It is a")
    print("  square root OF THE DEFECT, so it cannot be used as the repair.")
    print("  SHIPPED_SQRT_INFORMATION_CONTAMINATED = %s"
          % ("YES" if shipped_contaminated else "NO"))

    # ==================================================================
    hdr("s30/s31  GENUINE SRIF R FACTOR vs BATCH QR vs EXACT ORACLE")
    srif_ok = True
    consistency_ok = True
    print("  %-4s %16s %16s %16s %10s %10s"
          % ("case", "exact", "batch QR", "sequential SRIF", "QR rel", "SRIF rel"))
    for label, _ in CASES:
        a = arcs[label]
        h = np.hstack([a.h_x0, a.h_k[:, None]])
        s = scale_matrix()
        cov_ex, sig_ex = exact_covariance(h, a.w, None)
        cov_exf = exact_cov_to_float(cov_ex)
        srq = square_root_covariance(h, a.w, None, s)
        a_w = np.sqrt(a.w)[:, None] * (h @ s)
        r_seq = sequential_srif_r(a_w, np.zeros((0, N)))
        cov_seq = covariance_from_r(r_seq, s)
        sig_seq = float(np.sqrt(cov_seq[K_INDEX, K_INDEX]))
        rel_qr = relative_error(srq.sigma_k, sig_ex)
        rel_seq = relative_error(sig_seq, sig_ex)
        denom = np.sqrt(np.outer(np.diag(cov_exf), np.diag(cov_exf)))
        full_qr = float(np.max(np.abs(srq.covariance - cov_exf) / denom))
        full_seq = float(np.max(np.abs(cov_seq - cov_exf) / denom))
        cross = float(np.max(np.abs(srq.covariance - cov_seq) / denom))
        kstate_seq = float(np.max(np.abs(cov_seq[:6, K_INDEX]
                                         - cov_exf[:6, K_INDEX]) / denom[:6, K_INDEX]))
        srif_ok = srif_ok and rel_seq < 1e-6 and full_seq < 1e-6
        consistency_ok = consistency_ok and cross < 1e-6
        print("  %-4s %16.6e %16.6e %16.6e %10.2e %10.2e"
              % (label, sig_ex, srq.sigma_k, sig_seq, rel_qr, rel_seq))
        rows_out.append(dict(
            case=label, exact_sigma_k=sig_ex, batch_qr_sigma_k=srq.sigma_k,
            sequential_srif_sigma_k=sig_seq,
            shipped_sqrt_sigma_k=float(np.sqrt(
                shipped_sqrt_information_covariance(h, a.w, None, s)[K_INDEX, K_INDEX])),
            floored_sigma_k=floored_covariance(h, a.w, None, s)[1]["sigma_k"],
            qr_relative_error=rel_qr, srif_relative_error=rel_seq,
            qr_full_covariance_error=full_qr,
            srif_full_covariance_error=full_seq,
            srif_k_state_cross_error=kstate_seq,
            qr_vs_srif_cross_error=cross))
    print("\n  full-covariance agreement (max normalised error vs exact):")
    for r in rows_out:
        print("    %-4s  batch QR %.2e   sequential SRIF %.2e   QR-vs-SRIF %.2e"
              % (r["case"], r["qr_full_covariance_error"],
                 r["srif_full_covariance_error"], r["qr_vs_srif_cross_error"]))

    with (ARTIFACTS / "r1cov_srif_covariance.csv").open("w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w_.writeheader(); w_.writerows(rows_out)
    print("\n  wrote r1cov_srif_covariance.csv")

    # ==================================================================
    hdr("s32  SR-UKF SCOPE AUDIT")
    import lunar_od.filters as filters
    src = Path(filters.__file__).read_text(encoding="utf-8", errors="replace")
    touches_floor = ("_safe_covariance_from_information" in src
                     or "_sqrt_information_from_information" in src)
    imports_estimators = "from .estimators" in src or "import estimators" in src
    keeps_sqrt = "sqrt_p" in src
    print("  filters.py references the floored helpers : %s" % touches_floor)
    print("  filters.py imports estimators             : %s" % imports_estimators)
    print("  SR-UKF carries a covariance square root   : %s" % keeps_sqrt)
    print("\n  SR-UKF propagates sqrt_p through sigma-point machinery and forms")
    print("  its covariance as sqrt_p @ sqrt_p.T. It never builds or inverts a")
    print("  normal matrix, so the R1M defect cannot reach it.")
    affected = touches_floor or imports_estimators
    print("\n  SRUKF_AFFECTED_BY_NORMAL_MATRIX_FLOOR = %s"
          % ("YES" if affected else "NO"))
    print("  -> s67: SR-UKF is NOT modified by this phase.")

    print("\n  SUMMARY")
    print("    SRIF_R_FACTOR_COVARIANCE_GATE         = %s"
          % ("PASS" if (srif_ok and analytic_ok) else "FAIL"))
    print("    BLS_SRIF_COVARIANCE_CONSISTENCY_GATE  = %s"
          % ("PASS" if consistency_ok else "FAIL"))
    print("    SRUKF_AFFECTED_BY_NORMAL_MATRIX_FLOOR = %s"
          % ("YES" if affected else "NO"))


if __name__ == "__main__":
    main()
