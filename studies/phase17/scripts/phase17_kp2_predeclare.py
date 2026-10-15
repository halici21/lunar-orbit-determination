"""PHASE 17-KP2 - write the predeclaration BEFORE any high-fidelity recovery result.

Refuses to overwrite.  Canonical sha256 over the sorted-key compact JSON body.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase17_kp2_core as K2  # noqa: E402


def main() -> None:
    if K2.PRE_PATH.exists():
        raise SystemExit("predeclaration exists; refusing to overwrite %s" % K2.PRE_PATH)
    pre = {
        "phase": "17-KP2 nonlinear M3 K_SRP recovery confirmation",
        "written_before_any_high_fidelity_recovery_result": True,
        "entering_head": "720c19b",
        "question": "Are the M3 variation directions that KP1 found locally distinguishable recoverable by a "
                    "nonlinear batch estimator on production measurement VALUES under a free per-pass Doppler bias, "
                    "without claiming the common K level?",
        "model": {"K": "M3_THREE_SEGMENT: K(t) = K_j on equal thirds of the 15-orbit arc (KP1 boundaries)",
                  "propagation": "segmented production propagate_state_with_k_sensitivity, restart at boundaries, "
                                 "chained STM, exact per-segment dx/dK_j",
                  "physics_changed": False},
        "truth": {"level": "SYNTHETIC_CAMPAIGN_TRUTH, common level K0 = 0.01 m^2/kg in every truth",
                  "truths_fractional_segment_deviation": {k: list(v) for k, v in K2.TRUTHS.items()},
                  "rationale": "T0 = constant (false-detection control); T1, T2 = zero-mean 10 % variations spanning "
                               "the 2-D variation subspace (variable effective area, GEO contract S1)",
                  "doppler_bias_truth": "N(0, 1e-4 m/s) per pass, seeded",
                  "noise": "white, range 5 m, counted Doppler 1e-4 m/s (CD1 structural), seeded"},
        "population": {"cells": "5 orbits (CD1/KP1 set) x D3 epochs k in {0, 12}, HIGH_FIDELITY (primary)",
                       "trials": "one seed per (cell, truth): 30 primary M3 trials",
                       "linear_prediction": "KP1 committed s_var_min (RPD_D1FREE, M3, HIGH_FIDELITY) per cell: "
                                            "LIN_SEP if >= 0.35 else LIN_WEAK",
                       "seed": "numpy default_rng([cell_index, truth_index, 0])"},
        "estimator": {
            "kind": "analysis-layer batch Gauss-Newton on production range and counted-Doppler VALUES and production "
                    "Jacobians (D23-qualified path); production estimators have no counted-Doppler + range joint "
                    "solve, per-pass bias or M3 K, and are not changed",
            "parameters": "x0 (6), K_1..K_3, one free constant bias per Doppler pass (CD1 D1)",
            "a_priori": "x0_ap = x0_true + N(0, diag(100 m, 0.1 m/s)^2), the same as the prior covariance; "
                        "K start K0 for every segment; biases start 0; no prior on K or biases",
            "staging": "stage 1: x0 + biases with K held at K0 (<= 4 evaluations, 1e-2 sigma); stage 2: all "
                       "parameters (<= 10 evaluations, converged when every Gauss-Newton step < 1e-2 sigma; the model evaluation carries ~1e-3 sigma integrator noise, measured in a simplified-dynamics smoke); step "
                       "halving on cost increase; non-positive K candidates rejected without evaluation",
            "covariance": "QR of the column-normalised whitened design at the converged point; never a normal "
                          "matrix inverse"},
        "metrics": {
            "variation": "v = Q^T f, f = K_j / K0 - 1, Q orthonormal basis of the zero-mean subspace; "
                         "chi2_err = (v_hat - v)^T C_v^-1 (v_hat - v); chi2_detect = v_hat^T C_v^-1 v_hat",
            "level": "m = mean(f); level error / sigma reported; never claimed as identifiable",
            "nees9": "(x0, f) error against the 9x9 posterior block",
            "fit": "whitened residual rms per measurement type"},
        "decision": {
            "trial_recovered": "converged AND chi2_err <= 13.8155 (chi2_2 0.999) AND chi2_detect >= 13.8155",
            "R_sep": "recovered fraction over T1 and T2 trials in LIN_SEP cells (all cells if fewer than 3 LIN_SEP)",
            "FP": "fraction of T0 trials with chi2_detect >= 13.8155",
            "CONS": "fraction of converged M3 trials (all truths) with chi2_err <= 13.8155",
            "A": "M3_VARIATION_NONLINEARLY_RECOVERABLE: R_sep >= 0.80 and FP <= 0.20 and CONS >= 0.80",
            "B": "M3_VARIATION_PARTIALLY_RECOVERABLE: not A and R_sep >= 0.30",
            "C": "M3_LINEAR_SEPARATION_DOES_NOT_TRANSLATE_TO_NONLINEAR_RECOVERY: R_sep < 0.30",
            "consistency_status": "CONSISTENT if CONS >= 0.9 and fraction(nees9 <= 27.877) >= 0.9; MARGINAL if both "
                                  ">= 0.7; else INCONSISTENT",
            "common_level_status": "NOT_CLAIMED (KP1: common mode state-degenerate); the formal level error / "
                                   "sigma is reported as a consistency figure only",
            "translation_contrast": "recovered fraction in LIN_WEAK cells reported separately",
            "doppler_bias_impact": "T1 at k = 0 on all 5 orbits re-solved with zero truth bias and NO bias "
                                   "parameters (D0); MATERIAL if median sigma_v(D1)/sigma_v(D0) >= 1.5 or the "
                                   "recovered count drops, else MINOR"},
        "verification": {
            "fd": "segmented HF 2-orbit arcs (CANON k0, O2 k12) with the T1 nominal: perturb each K_j by "
                  "+/- {1e-3, 3e-3} m^2/kg, repropagate segmented, central differences vs the chained dx/dK_j at "
                  "every node; best-step relative error <= 1e-4",
            "kp1_anchor": "at each T0 truth (constant K) the segmented design reproduces KP1 committed M3 "
                          "sigma_frac and s_var_min for the cell (RPD_D1FREE, HF): relative difference <= 1e-5",
            "noise_free": "CANON k0 HF, T1, zero noise and zero bias (biases still estimated): |f_hat - f| <= "
                          "0.02 sigma and |x0_hat - x0| <= 0.02 sigma (twice the convergence criterion)",
            "production_bls": "CANON k0 HF, T0, range only, scalar K, no prior: KP2 solver vs production "
                              "estimate_two_way_range_bls_lm(solve_for_k_srp=True) from the same start; every "
                              "estimate difference <= 0.02 sigma (two convergence floors) and every sigma within 1 %",
            "m0_baseline": "scalar K on T0 at CANON k0 and O5 k0 (secondary): formal sigma vs CD1 committed "
                           "sigma_k_frac_D1FREE within 5 %",
            "d23": "focused D23 regression in the full suite"},
        "forbidden": ["changing truths, amplitudes, priors, nuisance treatment, thresholds or cells after results",
                      "stochastic K", "constraining the Doppler biases", "claiming the common K level"],
    }
    pre["canonical_sha256"] = K2.canonical_sha256(pre)
    K2.PRE_PATH.write_text(json.dumps(pre, indent=1, sort_keys=True), encoding="utf-8")
    print(pre["canonical_sha256"])


if __name__ == "__main__":
    main()
