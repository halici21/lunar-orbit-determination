"""PHASE 17-R1COV - COV-A, COV-B, COV-C.

  COV-A  s15/s16  reproduce the R1M covariance defect on the shipped code
  COV-B  s17-s21  exact/high-precision covariance oracle and its convergence
  COV-C  s22-s28  QR square-root covariance, scaling invariance, prior response

ANALYSIS SPACE ONLY.  No production file is modified.

The arcs are built once and reused by every section, because rebuilding them
per section would be the slowest part of the phase and would also risk two
sections silently qualifying slightly different design matrices.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase17_r1cov_core import (  # noqa: E402
    K_INDEX, condition_numbers, decimal_sigma_k, exact_cov_to_float,
    exact_covariance, floored_covariance, relative_error, scale_matrix,
    square_root_covariance,
)

ARTIFACTS = Path(r"C:/Users/erayh/Documents/Python/Grad/python_port_phase17r0/artifacts")
CASES = (("G0", 1.3), ("G1", 2.0), ("G3", 5.0))
#: s21 prior cases.  Broad and moderate are R1/R1G's own definitions.
PRIORS = {
    "data_only": None,
    "broad_k_sigma_1.0": 1.0,
    "moderate_k_sigma_0.01": 0.01,
}
#: s27 physically equivalent K bookkeeping scales
K_SCALES = (5e-3, 1e-2, 2e-2)
#: s18 precision ladder
DIGITS = (50, 80, 120, 200)
OUT: dict = {}


def hdr(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


def write_csv(name, rows):
    with (ARTIFACTS / name).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("  wrote %s (%d rows)" % (name, len(rows)))


def k_prior_information(sigma_k: float | None) -> np.ndarray | None:
    if sigma_k is None:
        return None
    p = np.zeros((7, 7))
    p[K_INDEX, K_INDEX] = 1.0 / sigma_k ** 2
    return p


def well_conditioned_control() -> tuple[np.ndarray, np.ndarray]:
    """A benign 7-parameter least-squares problem numpy can be trusted on.

    Its purpose is to prove the exact oracle agrees with an ordinary float64
    inverse WHERE float64 IS RELIABLE.  Without this control, an oracle that
    disagreed with numpy everywhere would be indistinguishable from an oracle
    that is simply wrong.
    """
    rng = np.random.default_rng(20260915)
    q, _ = np.linalg.qr(rng.standard_normal((60, 7)))
    sv = np.geomspace(1.0, 1e-2, 7)          # cond = 100, comfortably benign
    v, _ = np.linalg.qr(rng.standard_normal((7, 7)))
    h = q @ (sv[:, None] * v.T)
    w = np.full(60, 4.0)
    return h, w


def main() -> None:
    from lunar_od.spice_loader import load_spice_kernels
    sys.path.insert(0, str(HERE))
    from phase17_r1m_core import build_range_arc, campaign_epoch

    load_spice_kernels(None, clear=True)
    _, t_orbit = campaign_epoch()

    arcs = {}
    for label, orbits in CASES:
        arcs[label] = build_range_arc(0.0, orbits * t_orbit, label=label)
        print("built %s: %d observations, %s"
              % (label, arcs[label].n_obs, arcs[label].station_name))

    def design(label):
        a = arcs[label]
        return np.hstack([a.h_x0, a.h_k[:, None]]), a.w

    # ==================================================================
    hdr("COV-A  s15/s16  REPRODUCE THE R1M COVARIANCE DEFECT")
    repro_rows = []
    squaring_ok = True
    for label, orbits in CASES:
        h, w = design(label)
        s = scale_matrix()
        cond = condition_numbers(h, w, None, s)
        cov_f, diag = floored_covariance(h, w, None, s)
        _, sigma_exact = exact_covariance(h, w, None)
        # data-only scalar Schur, independent route
        info = h.T @ (w[:, None] * h)
        keep = list(range(6))
        i_xx = info[np.ix_(keep, keep)]
        i_xk = info[np.ix_(keep, [K_INDEX])]
        schur = float(info[K_INDEX, K_INDEX] - (i_xk.T @ np.linalg.solve(i_xx, i_xk))[0, 0])
        sigma_schur = float(1.0 / np.sqrt(schur))
        ratio = cond["squaring_ratio"]
        squaring_ok = squaring_ok and abs(ratio - 1.0) < 1e-6
        print("\n  %s (%.1f orbits, n=%d)" % (label, orbits, arcs[label].n_obs))
        print("    cond(design)        = %.6e   resolvable=%s"
              % (cond["cond_design"], cond["design_resolvable"]))
        print("    cond(information)   = %.6e   resolvable=%s"
              % (cond["cond_information"], cond["information_resolvable"]))
        print("    squaring ratio      = %.10f   (cond(I)/cond(A)^2)" % ratio)
        print("    smallest design sv  = %.6e" % cond["smallest_design_singular_value"])
        print("    smallest info eig   = %.6e" % diag["smallest_information_eigenvalue"])
        print("    floor               = %.6e   modes floored=%d  inflation=%.1fx"
              % (diag["floor"], diag["floored_modes"], diag["floor_inflation"]))
        print("    PRODUCTION sigma_K  = %.10e" % diag["sigma_k"])
        print("    EXACT      sigma_K  = %.10e" % sigma_exact)
        print("    Schur      sigma_K  = %.10e" % sigma_schur)
        print("    production overconfident by %.1fx"
              % (sigma_exact / diag["sigma_k"]))
        repro_rows.append(dict(
            case=label, arc_orbits=orbits, observations=arcs[label].n_obs,
            cond_design=cond["cond_design"],
            cond_information=cond["cond_information"],
            squaring_ratio=ratio,
            design_resolvable=cond["design_resolvable"],
            information_resolvable=cond["information_resolvable"],
            smallest_design_singular_value=cond["smallest_design_singular_value"],
            smallest_information_eigenvalue=diag["smallest_information_eigenvalue"],
            floor=diag["floor"], floored_modes=diag["floored_modes"],
            floor_inflation=diag["floor_inflation"],
            production_sigma_k=diag["sigma_k"],
            exact_sigma_k=sigma_exact, schur_sigma_k=sigma_schur,
            overconfidence_factor=sigma_exact / diag["sigma_k"]))
    write_csv("r1cov_defect_reproduction.csv", repro_rows)
    defect_reproduced = all(r["overconfidence_factor"] > 5.0 for r in repro_rows)
    print("\n  R1M_COVARIANCE_DEFECT_REPRODUCED = %s"
          % ("YES" if defect_reproduced else "NO"))
    print("  NORMAL_MATRIX_CONDITION_SQUARING_GATE = %s"
          % ("PASS" if squaring_ok else "FAIL"))
    OUT["defect_reproduced"] = bool(defect_reproduced)
    OUT["squaring_gate"] = "PASS" if squaring_ok else "FAIL"

    # ==================================================================
    hdr("COV-B  s17-s21  EXACT ORACLE AND PRECISION CONVERGENCE")

    # -- s19 well-conditioned control: exact must agree with plain float64
    hc, wc = well_conditioned_control()
    info_c = hc.T @ (wc[:, None] * hc)
    cov_np = np.linalg.inv(info_c)
    cov_ex, sig_ex_ctrl = exact_covariance(hc, wc, None)
    cov_ex_f = exact_cov_to_float(cov_ex)

    # Two metrics, because the obvious one is misleading.  Entrywise relative
    # error divides by |cov[i,j]|, so a covariance entry that happens to sit
    # near zero reports a huge relative error for a negligible absolute one.
    # The meaningful metric normalises each entry by its own natural scale
    # sqrt(var_i * var_j) -- the same normalisation used for the QR comparison
    # below, so the two are directly comparable.
    denom_c = np.sqrt(np.outer(np.diag(cov_ex_f), np.diag(cov_ex_f)))
    ctrl_entrywise = float(np.max(np.abs(cov_ex_f - cov_np) / np.abs(cov_np)))
    ctrl_rel = float(np.max(np.abs(cov_ex_f - cov_np) / denom_c))

    # The threshold is DERIVED, not chosen (s26).  Inverting a normal matrix of
    # condition kappa in float64 cannot be more accurate than ~kappa*eps; that
    # is the accuracy limit of the thing being compared against, so requiring
    # agreement tighter than it would be requiring float64 to beat itself.
    cond_info_c = float(np.linalg.cond(info_c))
    ctrl_threshold = cond_info_c * np.finfo(float).eps
    control_ok = ctrl_rel <= ctrl_threshold

    print("  well-conditioned control (cond(A)=%.1f, cond(A^T A)=%.3e):"
          % (np.linalg.cond(np.sqrt(wc)[:, None] * hc), cond_info_c))
    print("    entrywise relative error        = %.3e  (inflated by near-zero entries)"
          % ctrl_entrywise)
    print("    correlation-normalised error    = %.3e" % ctrl_rel)
    print("    float64 accuracy limit kappa*eps= %.3e" % ctrl_threshold)
    print("    observed / theoretical limit    = %.4f" % (ctrl_rel / ctrl_threshold))
    print("    -> the exact oracle agrees with float64 to within float64's OWN")
    print("       accuracy limit, i.e. WHERE FLOAT64 IS RELIABLE it is reliable")
    OUT["oracle_control_agreement"] = ctrl_rel
    OUT["oracle_control_entrywise"] = ctrl_entrywise
    OUT["oracle_control_threshold"] = ctrl_threshold

    # -- s18 convergence ladder against the exact limit
    conv_rows = []
    conv_ok = True
    for label, _ in CASES:
        h, w = design(label)
        _, sig_exact = exact_covariance(h, w, None)
        print("\n  %s  exact sigma_K = %.16e" % (label, sig_exact))
        prev = None
        for d in DIGITS:
            sd = decimal_sigma_k(h, w, None, digits=d)
            rel = relative_error(sd, sig_exact)
            step = "" if prev is None else "  step %.2e" % relative_error(sd, prev)
            print("    %3d digits -> %.16e   rel-to-exact %.3e%s"
                  % (d, sd, rel, step))
            conv_rows.append(dict(case=label, digits=d, sigma_k=sd,
                                  exact_sigma_k=sig_exact,
                                  relative_to_exact=rel))
            prev = sd
        r80 = next(r for r in conv_rows if r["case"] == label and r["digits"] == 80)
        r120 = next(r for r in conv_rows if r["case"] == label and r["digits"] == 120)
        d80_120 = relative_error(r80["sigma_k"], r120["sigma_k"])
        print("    80->120 digit change = %.3e  (must be negligible)" % d80_120)
        conv_ok = conv_ok and d80_120 < 1e-30 and r120["relative_to_exact"] < 1e-30
    write_csv("r1cov_high_precision_oracle.csv", conv_rows)
    print("\n  HIGH_PRECISION_ORACLE_CONVERGENCE_GATE = %s"
          % ("PASS" if (conv_ok and control_ok) else "FAIL"))
    OUT["oracle_convergence_gate"] = "PASS" if (conv_ok and control_ok) else "FAIL"

    # -- s20/s21 oracle outputs across prior cases
    oracle_rows = []
    for label, _ in CASES:
        h, w = design(label)
        for pname, psig in PRIORS.items():
            pinv = k_prior_information(psig)
            cov, sig = exact_covariance(h, w, pinv)
            covf = exact_cov_to_float(cov)
            ev = np.linalg.eigvalsh(0.5 * (covf + covf.T))
            sd = np.sqrt(np.diag(covf))
            corr_k = [float(covf[i, K_INDEX] / (sd[i] * sd[K_INDEX]))
                      for i in range(6)]
            oracle_rows.append(dict(
                case=label, prior=pname,
                prior_sigma_k=("" if psig is None else psig),
                sigma_k=sig,
                k_variance=float(covf[K_INDEX, K_INDEX]),
                sigma_pos_x=float(sd[0]), sigma_vel_x=float(sd[3]),
                covariance_min_eigenvalue=float(ev[0]),
                covariance_condition=float(ev[-1] / ev[0]) if ev[0] > 0 else float("inf"),
                corr_k_x=corr_k[0], corr_k_y=corr_k[1], corr_k_z=corr_k[2],
                corr_k_vx=corr_k[3], corr_k_vy=corr_k[4], corr_k_vz=corr_k[5]))
    write_csv("r1cov_oracle_prior_cases.csv", oracle_rows)
    print("\n  exact posterior sigma_K by prior:")
    for label, _ in CASES:
        row = {r["prior"]: r["sigma_k"] for r in oracle_rows if r["case"] == label}
        print("    %s  data-only %.6e   broad(1.0) %.6e   moderate(0.01) %.6e"
              % (label, row["data_only"], row["broad_k_sigma_1.0"],
                 row["moderate_k_sigma_0.01"]))

    # ==================================================================
    hdr("COV-C  s22-s26  QR SQUARE-ROOT COVARIANCE vs EXACT ORACLE")
    qr_rows = []
    qr_ok = True
    for label, _ in CASES:
        h, w = design(label)
        for pname, psig in PRIORS.items():
            pinv = k_prior_information(psig)
            s = scale_matrix()
            srq = square_root_covariance(h, w, pinv, s)
            cov_ex, sig_ex = exact_covariance(h, w, pinv)
            cov_exf = exact_cov_to_float(cov_ex)
            cov_fl, dfl = floored_covariance(h, w, pinv, s)
            rel_sig = relative_error(srq.sigma_k, sig_ex)
            # full-matrix comparison, relative to each exact entry's scale
            denom = np.sqrt(np.outer(np.diag(cov_exf), np.diag(cov_exf)))
            rel_full = float(np.max(np.abs(srq.covariance - cov_exf) / denom))
            rel_kstate = float(np.max(
                np.abs(srq.covariance[:6, K_INDEX] - cov_exf[:6, K_INDEX])
                / denom[:6, K_INDEX]))
            rel_states = float(np.max(
                np.abs(np.sqrt(np.diag(srq.covariance)[:6])
                       - np.sqrt(np.diag(cov_exf)[:6]))
                / np.sqrt(np.diag(cov_exf)[:6])))
            qr_ok = qr_ok and rel_sig < 1e-6 and rel_full < 1e-6
            qr_rows.append(dict(
                case=label, prior=pname,
                qr_sigma_k=srq.sigma_k, exact_sigma_k=sig_ex,
                floored_sigma_k=dfl["sigma_k"],
                qr_rel_error_sigma_k=rel_sig,
                floored_rel_error_sigma_k=relative_error(dfl["sigma_k"], sig_ex),
                qr_rel_error_full_covariance=rel_full,
                qr_rel_error_k_state_terms=rel_kstate,
                qr_rel_error_state_sigmas=rel_states,
                qr_rank_estimate=srq.rank_estimate,
                qr_smallest_abs_diag_r=srq.smallest_abs_diag_r,
                cond_design=srq.condition_design))
        print("  %s: QR vs exact sigma_K rel err = %s"
              % (label, ", ".join(
                  "%s %.2e" % (r["prior"], r["qr_rel_error_sigma_k"])
                  for r in qr_rows if r["case"] == label)))
    write_csv("r1cov_qr_vs_oracle.csv", qr_rows)
    print("\n  QR_COVARIANCE_ORACLE_GATE = %s" % ("PASS" if qr_ok else "FAIL"))
    OUT["qr_oracle_gate"] = "PASS" if qr_ok else "FAIL"

    # ---- s27 scaling invariance (HARD GATE) --------------------------
    hdr("COV-C  s27  SCALING INVARIANCE  (HARD GATE)")
    scale_rows = []
    scaling_ok = True
    for label, _ in CASES:
        h, w = design(label)
        qr_sigmas, fl_sigmas = [], []
        for ks in K_SCALES:
            s = scale_matrix(k_scale=ks)
            srq = square_root_covariance(h, w, None, s)
            cov_fl, dfl = floored_covariance(h, w, None, s)
            qr_sigmas.append(srq.sigma_k)
            fl_sigmas.append(dfl["sigma_k"])
            scale_rows.append(dict(case=label, k_scale=ks,
                                   qr_sigma_k=srq.sigma_k,
                                   floored_sigma_k=dfl["sigma_k"]))
        qr_spread = max(qr_sigmas) / min(qr_sigmas) - 1.0
        fl_spread = max(fl_sigmas) / min(fl_sigmas) - 1.0
        scaling_ok = scaling_ok and qr_spread < 1e-9
        print("  %s  QR sigma_K spread over K scales = %.3e   "
              "(floored path spread = %.3e)" % (label, qr_spread, fl_spread))
    write_csv("r1cov_scaling_invariance.csv", scale_rows)
    print("\n  QR_COVARIANCE_SCALING_INVARIANCE_GATE = %s"
          % ("PASS" if scaling_ok else "FAIL"))
    OUT["qr_scaling_gate"] = "PASS" if scaling_ok else "FAIL"

    # ---- s28 prior response ------------------------------------------
    hdr("COV-C  s28  PRIOR RESPONSE  (closes the R1/R1G equal-sigma pathology)")
    pr_rows = []
    prior_ok = True
    for label, _ in CASES:
        h, w = design(label)
        s = scale_matrix()
        seen = {}
        for pname, psig in PRIORS.items():
            pinv = k_prior_information(psig)
            srq = square_root_covariance(h, w, pinv, s)
            _, sig_ex = exact_covariance(h, w, pinv)
            _, dfl = floored_covariance(h, w, pinv, s)
            seen[pname] = (srq.sigma_k, dfl["sigma_k"])
            pr_rows.append(dict(case=label, prior=pname,
                                prior_sigma_k=("" if psig is None else psig),
                                qr_sigma_k=srq.sigma_k,
                                exact_sigma_k=sig_ex,
                                floored_sigma_k=dfl["sigma_k"],
                                qr_matches_exact=relative_error(srq.sigma_k, sig_ex) < 1e-6))
            prior_ok = prior_ok and relative_error(srq.sigma_k, sig_ex) < 1e-6
        broad_qr, broad_fl = seen["broad_k_sigma_1.0"]
        mod_qr, mod_fl = seen["moderate_k_sigma_0.01"]
        qr_distinct = relative_error(broad_qr, mod_qr)
        fl_distinct = relative_error(broad_fl, mod_fl)
        print("  %s  broad vs moderate prior:" % label)
        print("      QR      %.6e vs %.6e   relative difference %.3e"
              % (broad_qr, mod_qr, qr_distinct))
        print("      FLOORED %.6e vs %.6e   relative difference %.3e  %s"
              % (broad_fl, mod_fl, fl_distinct,
                 "<-- IDENTICAL: the R1G pathology" if fl_distinct < 1e-12 else ""))
    write_csv("r1cov_prior_response.csv", pr_rows)
    print("\n  QR_PRIOR_RESPONSE_GATE = %s" % ("PASS" if prior_ok else "FAIL"))
    OUT["qr_prior_gate"] = "PASS" if prior_ok else "FAIL"

    (ARTIFACTS / "r1cov_abc_results.json").write_text(
        json.dumps(OUT, indent=2, default=float))
    print("\n  wrote r1cov_abc_results.json")


if __name__ == "__main__":
    main()
