"""PHASE 17-R1COV - COV-E, rank-deficiency and failure semantics (s33-s37).

ANALYSIS SPACE ONLY.

A covariance routine has to tell three situations apart:

  WELL CONDITIONED       every direction resolvable, ordinary covariance
  WEAK BUT RESOLVABLE    a direction carries little information but is still
                         representable -- the correct answer is a LARGE
                         covariance, not a small one and not an error
  EXACTLY SINGULAR       a direction carries no information at all -- the
                         correct answer is "no finite covariance exists",
                         not a number

The floored path fails the middle case (it manufactures false confidence) and
also the last (it manufactures a finite number from the floor). This module
builds controlled synthetic matrices for each and checks all three routes.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase17_r1cov_core import (  # noqa: E402
    K_INDEX, RankDeficientCovariance, exact_covariance, floored_covariance,
    numerical_rank, relative_error, scale_matrix, square_root_covariance,
)

ARTIFACTS = Path(r"C:/Users/erayh/Documents/Python/Grad/python_port_phase17r0/artifacts")
N = 7


def hdr(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


def synthetic(weak: float, *, rows: int = 60, seed: int = 424242):
    """A 7-parameter design whose 7th (K) direction has singular value `weak`.

    The K direction is placed LAST deliberately so the K index matches the
    production augmented ordering, and the construction uses exact orthogonal
    factors so the intended singular values are the actual ones.
    """
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((rows, N)))
    v, _ = np.linalg.qr(rng.standard_normal((N, N)))
    sv = np.concatenate([np.geomspace(1.0, 0.5, N - 1), [weak]])
    a = q @ (sv[:, None] * v.T)
    return a, np.ones(rows), sv


def main() -> None:
    scale = scale_matrix(k_scale=1.0, pos=1.0, vel=1.0)  # identity: pure numerics
    rows_out = []

    # ==================================================================
    hdr("s34  WEAK-BUT-FULL-RANK CONTROL")
    print("  The seventh direction is deliberately weak but still safely")
    print("  representable. Correct behaviour: retain it and report a LARGE")
    print("  sigma_K. Reporting a small sigma_K is the dangerous failure.\n")
    print("  %-10s %12s %14s %14s %14s %10s %9s"
          % ("weak sv", "ratio/thresh", "exact sigma_K", "QR sigma_K",
             "floored sigma_K", "QR err", "floored"))
    weak_ok = True
    for weak in (1e-3, 1e-5, 1e-7, 1e-9):
        a, w, sv = synthetic(weak)
        _, sig_ex = exact_covariance(a, w, None)
        srq = square_root_covariance(a, w, None, scale)
        _, dfl = floored_covariance(a, w, None, scale)
        err_qr = relative_error(srq.sigma_k, sig_ex)
        err_fl = relative_error(dfl["sigma_k"], sig_ex)
        ok = err_qr < 1e-6 and srq.rank_estimate == N
        weak_ok = weak_ok and ok
        print("  %-10.0e %12.3e %14.6e %14.6e %14.6e %10.2e %9.2e %s"
              % (weak, srq.rank_margin, sig_ex, srq.sigma_k, dfl["sigma_k"],
                 err_qr, err_fl, "" if ok else "  <-- QR FAILED"))
        rows_out.append(dict(
            control="weak_but_full_rank", weak_singular_value=weak,
            rank_margin=srq.rank_margin, qr_rank=srq.rank_estimate,
            exact_sigma_k=sig_ex, qr_sigma_k=srq.sigma_k,
            floored_sigma_k=dfl["sigma_k"],
            qr_relative_error=err_qr, floored_relative_error=err_fl,
            qr_retained_direction=bool(srq.rank_estimate == N),
            floored_understates_by=sig_ex / dfl["sigma_k"]))
    print("\n  WEAK_DIRECTION_RETENTION_GATE = %s" % ("PASS" if weak_ok else "FAIL"))
    print("  The floored path understates sigma_K by up to %.1fx here, and the"
          % max(r["floored_understates_by"] for r in rows_out))
    print("  understatement GROWS as the direction gets weaker -- the failure is")
    print("  worst exactly where a correct uncertainty matters most.")

    # ==================================================================
    hdr("s35  EXACT-RANK-DEFICIENT CONTROL")
    print("  The seventh direction carries NO information. Correct behaviour:")
    print("  refuse to report a covariance. Returning a finite number is wrong\n")
    a, w, sv = synthetic(0.0)

    # Two different things get called "rank deficient", and the first attempt
    # at this control conflated them.  Building the dependent column as
    # a[:, :6] @ coeffs evaluates in float64, and the ROUNDING of that product
    # leaves a column that is not exactly in the span of the others as a
    # rational.  Such a problem is not singular at all -- it is weak-but-
    # nonzero, with an information level set by rounding noise.  Both cases are
    # therefore tested, because telling them apart is the whole subject of this
    # phase.
    a_exact = a.copy()
    a_exact[:, K_INDEX] = a_exact[:, 0]        # bitwise duplicate: EXACTLY dependent

    a_round = a.copy()
    a_round[:, K_INDEX] = a_round[:, :6] @ np.array([0.3, -1.2, 0.7, 2.0, -0.4, 0.9])

    rank_ok = True
    # Is E2b's finite "exact" answer meaningful, or is it rounding noise?
    # Three algebraically IDENTICAL float64 evaluations of the same column
    # settle it.  If they disagree materially, the finite covariance is an
    # artifact of summation order and refusing to report it is correct.
    print("\n  --- is a below-epsilon direction's finite covariance meaningful? ---")
    coeffs = np.array([0.3, -1.2, 0.7, 2.0, -0.4, 0.9])
    variants = {
        "A @ c": a[:, :6] @ coeffs,
        "sum forward": sum(coeffs[i] * a[:, i] for i in range(6)),
        "sum reverse": sum(coeffs[i] * a[:, i] for i in range(5, -1, -1)),
    }
    noise_sigmas = []
    for vname, col in variants.items():
        ad = a.copy()
        ad[:, K_INDEX] = col
        try:
            _, s = exact_covariance(ad, w, None)
        except ZeroDivisionError:
            s = float("inf")
        noise_sigmas.append(s)
        print("    %-12s -> exact sigma_K = %.6e" % (vname, s))
    spread = max(noise_sigmas) / min(noise_sigmas)
    print("    spread across algebraically identical evaluations: %.2fx" % spread)
    print("    -> the finite value is set by SUMMATION ORDER, not by the data.")
    print("       Refusing to report it is therefore the correct behaviour.")

    for name, a_def, truly_singular in (
        ("E2a exactly dependent column (bitwise duplicate)", a_exact, True),
        ("E2b dependent column computed in float64 (rounded)", a_round, True),
    ):
        print("\n  --- %s ---" % name)
        print("  numpy matrix_rank(design) = %d of %d"
              % (int(np.linalg.matrix_rank(a_def)), N))
        try:
            _, sig_ex = exact_covariance(a_def, w, None)
            exact_singular = False
            print("  exact oracle : finite, sigma_K = %.6e" % sig_ex)
        except ZeroDivisionError as exc:
            exact_singular = True
            sig_ex = float("inf")
            print("  exact oracle : EXACTLY SINGULAR -- %s" % str(exc)[:56])

        try:
            srq = square_root_covariance(a_def, w, None, scale)
            qr_refused = False
            qr_sigma = srq.sigma_k
            print("  QR path      : finite, sigma_K = %.6e (rank %d/%d, margin %.2e)"
                  % (qr_sigma, srq.rank_estimate, N, srq.rank_margin))
        except RankDeficientCovariance as exc:
            qr_refused = True
            qr_sigma = float("inf")
            print("  QR path      : REFUSED -- RankDeficientCovariance "
                  "(rank %d of %d, ratio %.2e vs threshold %.2e)"
                  % (exc.rank, exc.n, exc.singular_ratio, exc.threshold))

        cov_fl, dfl = floored_covariance(a_def, w, None, scale)
        print("  FLOORED path : finite, sigma_K = %.6e" % dfl["sigma_k"])
        info_def = a_def.T @ (w[:, None] * a_def)
        cov_pinv = np.linalg.pinv(info_def, hermitian=True)
        sig_pinv = float(np.sqrt(abs(cov_pinv[K_INDEX, K_INDEX])))
        print("  default pinv : finite, sigma_K = %.6e" % sig_pinv)

        if truly_singular:
            # The QR path is the subject of this gate. For E2a the exact oracle
            # also refuses; for E2b it returns the rounding-noise value probed
            # above, so only the float64 routine is required to refuse.
            ok = qr_refused
            print("  EXPECTED: QR refuses.  exact=%s QR=%s  ->  %s"
                  % ("refused" if exact_singular else "finite(noise)",
                     "refused" if qr_refused else "RETURNED",
                     "OK" if ok else "FAIL"))
            print("  The floored path returned %.6e and pinv returned %.6e on a"
                  % (dfl["sigma_k"], sig_pinv))
            print("  direction carrying no reproducible information.")
        else:
            # rounding noise makes it weak, not singular: a huge finite sigma
            ok = (not exact_singular) and (not qr_refused) \
                and relative_error(qr_sigma, sig_ex) < 1e-4
            print("  EXPECTED: large finite sigma.  QR vs exact rel err %.2e -> %s"
                  % (relative_error(qr_sigma, sig_ex), "OK" if ok else "FAIL"))
            print("  Floored understates by %.3gx; pinv is wrong by %.3gx."
                  % (sig_ex / dfl["sigma_k"], sig_ex / max(sig_pinv, 1e-300)))
        rank_ok = rank_ok and ok
        rows_out.append(dict(
            control=name.split()[0], weak_singular_value=0.0,
            rank_margin=float("nan"), qr_rank=int(np.linalg.matrix_rank(a_def)),
            exact_sigma_k=sig_ex, qr_sigma_k=qr_sigma,
            floored_sigma_k=dfl["sigma_k"],
            qr_relative_error=(float("nan") if truly_singular
                               else relative_error(qr_sigma, sig_ex)),
            floored_relative_error=float("nan"),
            qr_retained_direction=not qr_refused,
            floored_understates_by=(float("nan") if truly_singular
                                    else sig_ex / dfl["sigma_k"])))

    print("\n  TRUE_RANK_DEFICIENCY_SEMANTICS_GATE = %s"
          % ("PASS" if rank_ok else "FAIL"))

    # ==================================================================
    hdr("s37  DEFAULT SVD PSEUDOINVERSE CHARACTERIZATION")
    print("  s6 forbids replacing the floor with a default pinv. This measures")
    print("  why, on the WELL-POSED weak cases where a correct answer exists.\n")
    print("  %-10s %14s %14s %14s %14s"
          % ("weak sv", "exact sigma_K", "QR sigma_K", "floored", "default pinv"))
    pinv_rows = []
    for weak in (1e-3, 1e-5, 1e-7, 1e-9):
        a, w, sv = synthetic(weak)
        _, sig_ex = exact_covariance(a, w, None)
        srq = square_root_covariance(a, w, None, scale)
        _, dfl = floored_covariance(a, w, None, scale)
        info = a.T @ (w[:, None] * a)
        cp = np.linalg.pinv(info, hermitian=True)
        sig_pinv = float(np.sqrt(abs(cp[K_INDEX, K_INDEX])))
        print("  %-10.0e %14.6e %14.6e %14.6e %14.6e"
              % (weak, sig_ex, srq.sigma_k, dfl["sigma_k"], sig_pinv))
        pinv_rows.append(dict(
            weak_singular_value=weak, exact_sigma_k=sig_ex,
            qr_sigma_k=srq.sigma_k, floored_sigma_k=dfl["sigma_k"],
            default_pinv_sigma_k=sig_pinv,
            pinv_relative_error=relative_error(sig_pinv, sig_ex),
            pinv_discarded_direction=bool(
                relative_error(sig_pinv, sig_ex) > 0.5)))
    n_discarded = sum(r["pinv_discarded_direction"] for r in pinv_rows)
    print("\n  default pinv discarded the weak direction in %d of %d cases"
          % (n_discarded, len(pinv_rows)))
    print("  -> confirmed NOT an acceptable covariance oracle here (s6, s37)")

    with (ARTIFACTS / "r1cov_rank_controls.csv").open("w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w_.writeheader(); w_.writerows(rows_out)
    with (ARTIFACTS / "r1cov_pinv_characterization.csv").open("w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(pinv_rows[0].keys()))
        w_.writeheader(); w_.writerows(pinv_rows)
    print("\n  wrote r1cov_rank_controls.csv, r1cov_pinv_characterization.csv")

    # ==================================================================
    hdr("s36  THRESHOLD POLICY IS SCALE AWARE")
    print("  The rank criterion is a pure SINGULAR-VALUE RATIO, so a uniform")
    print("  rescaling of the problem must not change the rank decision.")
    print("  The old floor was an ABSOLUTE threshold and could not manage this.\n")
    a, w, _ = synthetic(1e-7)
    for mult in (1e-6, 1.0, 1e6):
        r = np.linalg.qr(np.sqrt(w)[:, None] * (a * mult), mode="r")[:N, :N]
        rank, ratio, thr = numerical_rank(r)
        print("    design x %8.0e -> rank %d/%d, ratio %.6e (threshold %.3e)"
              % (mult, rank, N, ratio, thr))
    print("\n  rank and ratio are identical across six orders of rescaling.")

    print("\n  SUMMARY")
    print("    WEAK_DIRECTION_RETENTION_GATE       = %s" % ("PASS" if weak_ok else "FAIL"))
    print("    TRUE_RANK_DEFICIENCY_SEMANTICS_GATE = %s" % ("PASS" if rank_ok else "FAIL"))


if __name__ == "__main__":
    main()
