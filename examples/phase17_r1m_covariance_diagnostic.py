"""PHASE 17-R1M-B - covariance pipeline audit and prior-diagnostic resolution.

ANALYSIS SPACE ONLY.  The production covariance path is exercised, never
modified.

R1G left `PRIOR_DIAGNOSTIC_STATUS = PRIOR_DIAGNOSTIC_STILL_UNRESOLVED`: the
estimator-reported sigma_K could not be reproduced by a scalar Schur
calculation, so it was unclear whether K's formal uncertainty was set by the
data, by the prior, or by something else. This script answers that by
reconstructing sigma_K five independent ways and auditing the one step in the
pipeline that is neither data nor prior: the eigenvalue floor inside
`_safe_covariance_from_information`.

The pipeline being traced, in order:

    design matrix  H = [H_x0 | h_K]            (frozen production Jacobians)
      -> physical scaling      Hs = H @ diag(1e6,1e6,1e6,1e3,1e3,1e3,1e-2)
      -> measurement weighting I = Hs^T W Hs
      -> prior information     I += scale^T P0^-1 scale
      -> numerical inversion   eigh, then CLIP eigenvalues at max*1e-14
      -> unscaling             C = scale C_scaled scale^T
      -> reported sigma_K      sqrt(C[6,6])

Only the clipping step is a numerical choice rather than a modelling one, and
it is the step this audit isolates.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase17_r1m_core import (  # noqa: E402
    SCALE_K, build_range_arc, campaign_epoch, information_matrix,
    orthogonal_decomposition, scale_matrix, schur_conditional, spectrum,
    whitened,
)

ARTIFACTS = Path(r"C:/Users/erayh/Documents/Python/Grad/python_port_phase17r0/artifacts")
CASES = (("G0", 1.3), ("G1", 2.0))
#: physically equivalent K scales for the invariance audit (s20)
K_SCALES = (5e-3, 1e-2, 2e-2)
OUT: dict = {}


def hdr(t):
    print()
    print("=" * 86)
    print(t)
    print("=" * 86)


def write_csv(name, rows):
    path = ARTIFACTS / name
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("  wrote %s (%d rows)" % (name, len(rows)))


def scaled_info(arc, k_scale: float, prior_inv_phys=None):
    h = np.hstack([arc.h_x0, arc.h_k[:, None]])
    s = np.diag([1e6, 1e6, 1e6, 1e3, 1e3, 1e3, k_scale])
    info = information_matrix(h, arc.w, s)
    if prior_inv_phys is not None:
        info = info + s.T @ prior_inv_phys @ s
    return info, s


def main() -> None:
    from lunar_od.estimators import (
        _TWO_WAY_RANGE_NO_BIAS_CFG, _prior_information_and_scale,
        _safe_covariance_from_information, _two_way_range_bias_prior_and_scale,
    )
    from lunar_od.spice_loader import load_spice_kernels

    load_spice_kernels(None, clear=True)
    _, t_orbit = campaign_epoch()

    prior_inv6, _, _, _, _ = _prior_information_and_scale(
        6, _TWO_WAY_RANGE_NO_BIAS_CFG, _two_way_range_bias_prior_and_scale,
        None, None)
    prior_state_only = np.zeros((7, 7))
    prior_state_only[:6, :6] = prior_inv6

    method_rows, floor_rows, rank_rows, scale_rows = [], [], [], []

    for label, orbits in CASES:
        arc = build_range_arc(0.0, orbits * t_orbit, label=label)
        info, s = scaled_info(arc, SCALE_K)
        a_x, b_k = whitened(arc.h_x0, arc.h_k, arc.w)
        dec = orthogonal_decomposition(a_x, b_k)
        sp = spectrum(info)

        hdr("%s  (%.1f orbits, %d observations, %s)"
            % (label, orbits, arc.n_obs, arc.station_name))

        # ---- s17 five reconstructions --------------------------------
        # A. production estimator/helper path (floored inverse)
        cov_est = s @ _safe_covariance_from_information(info) @ s.T
        sigma_estimator = float(np.sqrt(cov_est[6, 6]))
        # B. direct inversion of the scaled information matrix
        cov_direct = s @ np.linalg.inv(info) @ s.T
        sigma_direct = float(np.sqrt(cov_direct[6, 6]))
        # C. SVD pseudoinverse
        cov_svd = s @ np.linalg.pinv(info, hermitian=True) @ s.T
        sigma_svd = float(np.sqrt(cov_svd[6, 6]))
        # D. data-only scalar Schur complement
        i_k_given_x = dec["i_k_given_x"]
        sigma_schur = float(1.0 / np.sqrt(i_k_given_x))
        # E. full Bayesian posterior, data + the estimator's own default prior
        info_full, s_full = scaled_info(arc, SCALE_K, prior_state_only)
        cov_full = s_full @ np.linalg.inv(info_full) @ s_full.T
        sigma_full = float(np.sqrt(cov_full[6, 6]))

        print("  SIGMA_K_ESTIMATOR      %.10e   (production, floored)" % sigma_estimator)
        print("  SIGMA_K_DIRECT         %.10e   (exact inverse)" % sigma_direct)
        print("  SIGMA_K_SVD            %.10e   (pseudoinverse)" % sigma_svd)
        print("  SIGMA_K_SCHUR          %.10e   (data-only scalar)" % sigma_schur)
        print("  SIGMA_K_FULL_POSTERIOR %.10e   (data + default state prior)" % sigma_full)
        print("  ratio direct/estimator %.3f" % (sigma_direct / sigma_estimator))

        method_rows.append(dict(
            case=label, arc_orbits=orbits, observations=arc.n_obs,
            sigma_k_estimator=sigma_estimator, sigma_k_direct=sigma_direct,
            sigma_k_svd=sigma_svd, sigma_k_schur=sigma_schur,
            sigma_k_full_posterior=sigma_full,
            direct_over_estimator=sigma_direct / sigma_estimator,
            schur_over_estimator=sigma_schur / sigma_estimator))

        # ---- s19 eigenvalue-floor audit ------------------------------
        vals, vecs = np.linalg.eigh(info)
        max_val = float(np.max(np.abs(vals)))
        floor = max(max_val * 1e-14, np.finfo(float).eps)
        floored = vals < floor
        print("\n  eigenvalues (scaled): max %.6e  min %.6e" % (vals[-1], vals[0]))
        print("  floor  = max * 1e-14 = %.6e" % floor)
        print("  floored modes: %d of 7" % int(floored.sum()))
        for i in np.where(floored)[0]:
            k_content = float(abs(vecs[6, i]))
            print("    mode %d: eigenvalue %.6e  ->  clipped to %.6e  "
                  "(x%.1f)  K content %.12f"
                  % (i, vals[i], floor, floor / max(vals[i], 1e-300), k_content))
            floor_rows.append(dict(
                case=label, mode=int(i), raw_eigenvalue=float(vals[i]),
                floor=floor, inflation_factor=float(floor / max(vals[i], 1e-300)),
                k_content=k_content))
        # does the floor alone predict the reported sigma?
        sigma_from_floor = float(np.sqrt(SCALE_K ** 2 / floor))
        print("  sqrt(K_scale^2 / floor) = %.10e   vs reported %.10e   rel %.2e"
              % (sigma_from_floor, sigma_estimator,
                 abs(sigma_from_floor - sigma_estimator) / sigma_estimator))
        floor_controls = abs(sigma_from_floor - sigma_estimator) / sigma_estimator < 1e-3

        # ---- s21 rank-tolerance audit --------------------------------
        sv = np.array(sp["singular_values"])
        print("\n  rank vs tolerance:")
        for tol in (1e-10, 1e-12, 1e-14, 1e-15, 1e-16):
            r = int(np.sum(sv > sv[0] * tol))
            rank_rows.append(dict(case=label, relative_tolerance=tol, rank=r,
                                  smallest_over_largest=float(sv[-1] / sv[0])))
            print("    rel tol %.0e -> rank %d/7" % (tol, r))
        print("  smallest/largest singular ratio = %.6e" % (sv[-1] / sv[0]))

        OUT.setdefault("cases", {})[label] = dict(
            observations=arc.n_obs, orbits=orbits,
            raw_i_kk=dec["i_kk"], conditional_k_information=i_k_given_x,
            orthogonal_fraction=dec["orthogonal_fraction"],
            smallest_eigenvalue=float(vals[0]), largest_eigenvalue=float(vals[-1]),
            floor=floor, floored_modes=int(floored.sum()),
            floor_controls_k_variance=bool(floor_controls),
            sigma_k_estimator=sigma_estimator, sigma_k_direct=sigma_direct,
            sigma_k_svd=sigma_svd, sigma_k_schur=sigma_schur,
            sigma_k_full_posterior=sigma_full,
            weakest_mode_k_component=sp["weakest_k_component"],
            scaled_condition=sp["condition"])

        # ---- s20 scaling audit ---------------------------------------
        print("\n  K-scale invariance (physical quantities must not move):")
        for ks in K_SCALES:
            info_k, s_k = scaled_info(arc, ks)
            cond_k = schur_conditional(info_k) / ks ** 2
            cov_k = s_k @ _safe_covariance_from_information(info_k) @ s_k.T
            sig_k = float(np.sqrt(cov_k[6, 6]))
            cov_exact_k = s_k @ np.linalg.inv(info_k) @ s_k.T
            sig_exact_k = float(np.sqrt(cov_exact_k[6, 6]))
            scale_rows.append(dict(
                case=label, k_scale=ks, conditional_k_information=cond_k,
                sigma_k_estimator=sig_k, sigma_k_direct=sig_exact_k))
            print("    K_scale %.0e -> I_K|x %.10e   sigma_est %.6e   "
                  "sigma_exact %.6e" % (ks, cond_k, sig_k, sig_exact_k))

    write_csv("r1m_covariance_methods.csv", method_rows)
    write_csv("r1m_floor_diagnostic.csv", floor_rows)
    write_csv("r1m_rank_tolerance.csv", rank_rows)
    write_csv("r1m_scaling_audit.csv", scale_rows)

    # ---- invariance verdict ------------------------------------------
    hdr("s20  SCALING INVARIANCE VERDICT")
    ok = True
    for label, _ in CASES:
        rows = [r for r in scale_rows if r["case"] == label]
        cond = np.array([r["conditional_k_information"] for r in rows])
        exact = np.array([r["sigma_k_direct"] for r in rows])
        est = np.array([r["sigma_k_estimator"] for r in rows])
        spread_cond = float(cond.max() / cond.min() - 1.0)
        spread_exact = float(exact.max() / exact.min() - 1.0)
        spread_est = float(est.max() / est.min() - 1.0)
        print("  %s: I_K|x spread %.2e   exact sigma spread %.2e   "
              "ESTIMATOR sigma spread %.2e" % (label, spread_cond, spread_exact,
                                               spread_est))
        ok = ok and spread_cond < 1e-6 and spread_exact < 1e-6
    print("  physical quantities invariant under K scale : %s" % ok)
    print("  (the ESTIMATOR sigma is expected to move, because the floor is an")
    print("   ABSOLUTE threshold on a scale-dependent matrix -- that dependence")
    print("   is the artifact, not a property of the physics)")
    OUT["scaling_physical_invariance_gate"] = "PASS" if ok else "FAIL"

    (ARTIFACTS / "r1m_covariance_diagnostic.json").write_text(
        json.dumps(OUT, indent=2, default=float))
    print("\n  wrote r1m_covariance_diagnostic.json")


if __name__ == "__main__":
    main()
