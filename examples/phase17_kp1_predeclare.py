"""PHASE 17-KP1 - write the predeclaration BEFORE any KP1 parameter-separation metric.

Refuses to overwrite an existing file.  The canonical SHA-256 is taken over the
sorted-key compact JSON body (without the hash field) and embedded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase17_kp1_core as KP  # noqa: E402
import phase17_r1o_cd1_core as CD  # noqa: E402


def main() -> None:
    if KP.PRE_PATH.exists():
        raise SystemExit("predeclaration exists; refusing to overwrite %s" % KP.PRE_PATH)
    cd1 = json.loads(CD.PRE_PATH.read_text(encoding="utf-8"))
    pre = {
        "phase": "17-KP1 alternative K_SRP parameterization characterization (linear information geometry only)",
        "written_before_any_kp1_separation_metric": True,
        "entering_head": "12839fe884dedabdb568cc19eb4d07885c12970d",
        "truth_level": "SYNTHETIC_CAMPAIGN_TRUTH: K_truth = 0.01 m^2/kg constant; no time-varying truth is "
                       "fabricated; KP1 is local distinguishability at the constant truth",
        "question": "Is there a low-dimensional, physically interpretable alternative K_SRP parameterization whose "
                    "components are distinguishable from the six-state initial condition and from each other "
                    "under the qualified range + counted-Doppler set?",
        "literature": {
            "source_contract": "docs/phase17_geo_literature_contract.md (S1 Slojkowski NTRS 20140008968, "
                               "S2 Slojkowski-Lowe-Woodburn ISSFD 2015; full text read in GEO)",
            "LITERATURE_RESULT": [
                "S1 s6.3: SRP variations due to variable spacecraft area are time-dependent dynamical errors "
                "(arc-length choice 36 h vs 60 h)",
                "S2: operations model the SRP scale factor as a stochastic process (Vasicek better than "
                "Gauss-Markov); C_R is not treated as a well-observed static constant",
                "S1 s6.2: consistent cannonball C_R estimation fails in full-Sun periods"],
            "REPOSITORY_HYPOTHESIS": "a deterministic low-dimensional time variation of K (batch analogue of a "
                                     "stochastic scale factor) carries measurement directions that the six-state "
                                     "initial condition and the constant K cannot reproduce",
            "not_established_by_literature": "any KP1 identifiability result"},
        "k_semantics": "production a_SRP = K * P(d) * nu * r_hat / |r|, K = C_R A/m [m^2/kg] constant "
                       "(lunar_od.srp.srp_acceleration_kernel); dx/dK from propagate_state_with_k_sensitivity "
                       "(F1 A matrix)",
        "candidates": {
            "M0_SCALAR": {"definition": "K(t) = p0", "params": ["p0 [m^2/kg]"], "truth": [0.01],
                          "role": "baseline (CD1 metrics reproduced exactly)"},
            "M1_TWO_SEGMENT": {
                "definition": "K(t) = pA on [0, T/2), pB on [T/2, T] (boundary at the grid node nearest T/2)",
                "params": ["pA [m^2/kg]", "pB [m^2/kg]"], "truth": [0.01, 0.01],
                "physics": "arc-wise constant scale factor: the deterministic batch analogue of a stochastic "
                           "scale factor with ~half-arc (~14.7 h) correlation time (S2); variable area (S1)"},
            "M2_LINEAR_DRIFT": {
                "definition": "K(t) = p0 + p1 s(t), s(t) = (t - T/2)/(T/2) in [-1, 1], realised as a staircase "
                              "constant on each 90 s grid interval at the interval-midpoint value (differs from "
                              "the exact ramp by <= h/T ~ 8.5e-4 in s)",
                "params": ["p0 [m^2/kg]", "p1 [m^2/kg, K change from mid-arc to arc end]"], "truth": [0.01, 0.0],
                "physics": "first-order Taylor term of ANY slow K(t) variation over the arc (variable area, "
                           "beta-angle evolution); mechanism-agnostic"},
            "M3_THREE_SEGMENT": {
                "definition": "K(t) = pA, pB, pC on equal thirds (boundaries at nodes nearest T/3, 2T/3)",
                "params": ["pA", "pB", "pC [m^2/kg]"], "truth": [0.01, 0.01, 0.01],
                "role": "complexity escalation: does one more degree of freedom add information or split it"}},
        "rejected": {
            "once_per_rev_attitude_modulation": "needs an attitude / multi-plate area model; production SRP is "
                                                "cannonball, so this would add physics (stop condition 1)",
            "per_orbit_K": "15 parameters; large piecewise basis (brief s2)",
            "splines_high_order_polynomials": "no physical basis at this arc length (brief s2)",
            "stochastic_Gauss_Markov_Vasicek_K": "physically established by S2 but deferred by the brief until "
                                                 "deterministic block identifiability is understood",
            "shadow_regime_dependent_K": "unphysical: K is a spacecraft property; shadow is in nu"},
        "sensitivity_construction": {
            "method": "exact composition from production quantities: J(t) = Phi(t)^-1 dx/dK(t); "
                      "S_window[a,b](t) = Phi(t)(J(clip(t,a,b)) - J(a)); staircase drift = "
                      "Phi(t) sum_i s_i (J_{i+1} - J_i); all boundaries on grid nodes",
            "measurement_columns": "production _two_way_range_k_srp_column (range) and "
                                   "two_way_counted_doppler_initial_state_jacobian with the sensitivity in the STM "
                                   "slot (counted Doppler), the HARDEN/F1 K construction",
            "production_code_changed": False},
        "scaling": {"state": "phase17_r1od2_core.SCALE state part [1e6 m x3, 1e3 m/s x3]",
                    "parameters": "fractional K: every p_j column multiplied by K_truth = 0.01 m^2/kg "
                                  "(1 unit = 100 % of K_truth; drift unit = K_truth change mid-arc to end)",
                    "invariant_metrics": "principal sines are invariant to any invertible reparameterization "
                                         "of the parameter block and of the state/nuisance block"},
        "dynamics": {"PRIMARY": "HIGH_FIDELITY phase17_r1od2_fidelity.propagate_fidelity (as CD1)",
                     "CONTROL": "SIMPLIFIED phase17_r1od2_core.propagate_nom48, diagnostic only, never decides"},
        "population": {"orbits": cd1["orbits"], "epoch_grid": cd1["epoch_grid"],
                       "cells": "5 orbits x 24 epochs = 120 HIGH_FIDELITY (+120 SIMPLIFIED diagnostic)",
                       "feasible": "CD1 rule: >= 7 counted-Doppler rows"},
        "measurements": {
            "range": "CD1 range block (production two-way range, DSN3, sigma 5 m, no range nuisance)",
            "doppler": "CD1 rows: ALL_VISIBLE, Tc 60 s, 90 s cadence, sigma 1e-4 m/s structural",
            "cases": {"RANGE": "range only",
                      "RPD_D1FREE": "PRIMARY: range + counted Doppler, one free constant bias per pass (CD1 D1)",
                      "RPD_D0": "secondary: ideal Doppler",
                      "RPD_D1P1E-4": "secondary: per-pass bias with 1e-4 m/s prior"},
            "doppler_values_used": False,
            "d23": "design/partials only; CD1 closure showed D23 moves f_perp <= 4e-10"},
        "metrics": {
            "s_min / s_max": "principal sines between span(H_p) and span(conditioning set = state [+ nuisance]); "
                             "for M0 equals f_perp",
            "s_var_min": "smallest nonzero principal sine between span(H_p) and span(conditioning set + h_K): "
                         "separation of the K-VARIATION directions from state AND the constant K",
            "conditioned": "SVD of B = P_perp H_p (declared fractional-K basis): singular values, condition, "
                           "weakest combination; sigma and correlations from the QR of B (never a normal-matrix "
                           "inverse, never a rank gate)",
            "nuisance_impact": "D0 vs D1 FREE change of s_var_min and sigma",
            "individual_column_f_perp": "diagnostic only"},
        "thresholds": {"separation": "STRONG >= 0.50, MODERATE >= 0.35, else WEAK (phase17_r1od2_core."
                                     "absolute_class, applied to s_var_min and s_min)",
                       "internal": "max |conditioned correlation| <= 0.90 INTERNALLY_SEPARATED, <= 0.99 "
                                   "PARTIALLY, else COLLINEAR (repository convention, declared here)"},
        "decision": {
            "per_model_on_PRIMARY": {
                "VAR": "fraction of feasible HF cells with s_var_min >= 0.35",
                "INT": "fraction of feasible HF cells with max |corr| <= 0.90",
                "DISTINGUISHABLE": "VAR >= 0.50 and INT >= 0.50",
                "PARTIAL": "not DISTINGUISHABLE and VAR >= 0.10",
                "NO_NEW_INFORMATION": "VAR < 0.10"},
            "order": ["A", "B", "C"],
            "A": "ALTERNATIVE_K_PARAMETERIZATION_ADDS_DISTINGUISHABLE_DIRECTIONS: any model DISTINGUISHABLE; "
                 "recommend the lowest-dimension one (tie: M1 before M2)",
            "B": "ALTERNATIVE_K_PARAMETERIZATION_REVEALS_ONLY_PARTIAL_SEPARATION: any model PARTIAL",
            "C": "ADDITIONAL_K_PARAMETERS_INCREASE_DEGENERACY_WITHOUT_NEW_INFORMATION: otherwise; "
                 "recommend M0_SCALAR",
            "complexity_test": "SPLITS_WEAK_DIRECTION if median s_var_min(M3) < median s_var_min(M1) on PRIMARY",
            "nonlinear_confirmation": "may be recommended only under A; never started",
            "stochastic_followup": "justified only if B or A shows variation directions with VAR >= 0.10"},
        "verification": {
            "anchors": "M0 s_min == CD1 committed f_perp (RANGE, RPD_D0, RPD_D1FREE) |diff| <= 1e-10; "
                       "bank K columns == CD1 production h_k bitwise",
            "fd_oracle": "segmented HF repropagation with restarts on 2-orbit arcs (CANON k0, O2 k12), "
                         "K(t) = K_truth + eps b(t), central differences eps in {1e-3, 3e-3}, rtol 1e-13: "
                         "SEG2_A, SEG3_B, DRIFT vs exact composition; state-level per-column "
                         "min_eps max_t ||fd - an|| / max_t ||an|| <= 1e-4; a zero-perturbation segmented run "
                         "must reproduce the unsegmented propagation",
            "independent_rebuild": "every HF cell, every case x model: principal sines by lstsq residual + "
                                   "generalized eigenproblem, |diff| <= 1e-8",
            "exact_oracle": "CANON k = 0, 6, 12, 18 HF, RPD_D1FREE, M1-M3: exact rational Schur complement, "
                            "|wrapper - exact| <= 1e-12"},
        "pre_metric_feasibility": "one SIMPLIFIED cell run for anchors and timing only; no separation metric "
                                  "printed or inspected",
        "forbidden": ["nonlinear estimation", "Monte Carlo", "stochastic K", "Doppler values or residuals",
                      "changing candidates, scaling or thresholds after results", "DeltaDOR search",
                      "production code change"],
    }
    pre["canonical_sha256"] = KP.canonical_sha256(pre)
    KP.PRE_PATH.write_text(json.dumps(pre, indent=1, sort_keys=True), encoding="utf-8")
    print(pre["canonical_sha256"])


if __name__ == "__main__":
    main()
