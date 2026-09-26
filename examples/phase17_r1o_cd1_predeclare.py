"""PHASE 17-R1O-CD1 - write the predeclaration BEFORE any CD1 K-information metric.

Refuses to overwrite an existing file.  The canonical SHA-256 is taken over the
sorted-key compact JSON body (without the hash field) and embedded.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase17_r1o_cd1_core as CD  # noqa: E402

OUT = CD.PRE_PATH
ORBITS = ("CANON", "O1_LOW_BETA_ECLIPSE_RICH", "O2_FULL_SUN_HIGH_BETA", "O3_HIGH_ALTITUDE_500KM",
          "O5_FAVOURABLE_GEO_MAX_FPERP")


def _x0_hashes() -> dict:
    g = json.loads((CD.ARTIFACTS / "phase17_r1od2_predeclared_grid.json").read_text(encoding="utf-8"))
    return {x["case_id"]: dict(source=x["source"], x0_sha256=hashlib.sha256(json.dumps(x["x0"]).encode()).hexdigest())
            for x in g["geometries"] if x["case_id"] in ORBITS}


def main() -> None:
    if OUT.exists():
        raise SystemExit("predeclaration exists; refusing to overwrite %s" % OUT)
    d3grid = json.loads((CD.ARTIFACTS / "phase17_r1od3_predeclared_grid.json").read_text(encoding="utf-8"))
    pre = {
        "phase": "17-R1O-CD1 production counted-Doppler K_SRP complementarity characterization",
        "written_before_any_cd1_k_metric": True,
        "entering_head": "cfa1ce7293c60024c9ee04dfeca801ef7ba72c5d",
        "truth_level": "SYNTHETIC_CAMPAIGN_TRUTH (K_truth = 0.01 m^2/kg, constant scalar K; no reparameterization)",
        "literature": {"contract": "docs/phase17_r1o_cd1_literature_contract.md",
                       "dsn_module": "810-005 Module 202 Rev. E (2022-07-19), sha256 29a66259...4708",
                       "moyer": "DESCANSO Monograph 2, s10.2.3.1.2 / s13.3"},
        "observable": {
            "function": "lunar_od.radiometrics.two_way_counted_doppler_observable / _initial_state_jacobian "
                        "(production default R3 exact-station model == R4 four-event at zero delay)",
            "units": "mps_equivalent (rate of change of one-way range, m/s)",
            "uplink_hz": 7.2e9, "turnaround_ratio": "880/749",
            "k_column": "production Jacobian with dx/dK placed in the first STM column (HARDEN/F1 construction)",
            "earth": "canonical FXR fixture, linear interpolation inside the counted path (R3-P22 / HARDEN-D23)",
            "production_code_changed": False},
        "epoch_grid": {"source": "phase17_r1od3_predeclared_grid.json (D3), canonical_sha256 %s"
                                 % d3grid["canonical_sha256"],
                       "k": list(range(24)), "spacing_deg": 15.0,
                       "et0": "canonical_et0 + k * synodic_month_s / 24 (phase17_r1od3_core.epoch_et)",
                       "canonical_epoch_included": "k = 0 exactly"},
        "orbits": {"set": _x0_hashes(), "source": "artifacts/phase17_r1od2_predeclared_grid.json (D4 set, unchanged)"},
        "arc": {"duration": "15 canonical orbits", "grid_s": 90.0, "k_srp": 0.01, "shadow": "CONICAL_PENUMBRA"},
        "dynamics": {
            "PRIMARY": "HIGH_FIDELITY: phase17_r1od2_fidelity.propagate_fidelity (GL1800F 50x50 MOON_PA_DE440, "
                       "Earth + Sun third body, production SRP + F1 position gradient, harmonic STM opt-in, "
                       "rtol 1e-12) for EVERY orbit x epoch cell",
            "CONTROL": "SIMPLIFIED: phase17_r1od2_core.propagate_nom48, same cells, decomposition only; never decides",
            "matrix": "FULL 5 orbits x 24 epochs x {HF, SIMPLIFIED}; one propagation per cell reused by all schedules",
            "fallback": "none needed (feasibility measured: simplified cell ~13 s, HF ~330 s)"},
        "range_baseline": {"definition": "phase17_r1od2_core.build_range_block on the SAME propagation (orbit, "
                                         "epoch, dynamics, visibility), production two-way range, DSN3, "
                                         "sigma 5 m (station.sigma_range_m), no range bias",
                           "anchors": ["CANON SIMPLIFIED range f_perp, all 24 k, vs artifacts/phase17_r1od3_primary_results.csv (1e-12)",
                                       "all orbits HF and SIMPLIFIED range f_perp at k = 4..11 vs artifacts/phase17_r1od4_raw_results.json (1e-12)"]},
        "doppler_schedule": {
            "PRIMARY": "ALL_VISIBLE: one row per (90 s grid node, DSN3 station visible there) - the production "
                       "candidate rule - kept only if every grid node from floor(t_mid - Tc/2) to ceil(t_mid + Tc/2) "
                       "is in the same visibility window; HistoryDomainError candidates dropped (production)",
            "SECONDARY": "REGULAR_SESSION: per station pass lasting >= 3600 s, the rows within the centred 3600 s",
            "not_used": "no K-informed selection, no schedule search; DeltaDOR not included in any primary block"},
        "count_interval": {"PRIMARY_S": 60.0, "PRIMARY_CADENCE_S": 90.0,
                           "sensitivity": {"10": 90.0, "540": 540.0},
                           "rule": "cadence >= Tc so counts never overlap (independent white noise per count); "
                                   "540 s = 6 grid steps, midpoints on every 6th node",
                           "justification": "60 s = production default and HARDEN/F1-qualified value; Moyer M2 "
                                            "'e.g. 60 s or 600 s'; 202E figures use 5/60/1000 s"},
        "noise": {
            "PRIMARY": "STRUCTURAL_REFERENCE sigma_D = 1.0e-4 m/s at Tc = 60 s, white, per row "
                       "(repository DSN sigma_range_rate_mps; inside 202E X/X scintillation span at 60 s)",
            "bracket_60s_mps": list(CD.SIGMA_D60_BRACKET_MPS),
            "tc_scalings": {"SCINT_T^-0.175": "sigma(T) = sigma60 (60/T)^0.175 (202E Eq 30)",
                            "WHITE_PHASE_T^-1": "sigma(T) = sigma60 (60/T) (202E Eq 25/26 thermal)"},
            "OPERATIONAL_LOWER_BOUND": "per row: 202E Eq (30) X/X (C_band 2.7e-6) at the midpoint "
                                       "Sun-Earth-spacecraft angle, T = 60 s, RSS 2.12e-6 m/s H-maser floor; "
                                       "thermal PARAMETRIC and excluded -> optimistic lower bound",
            "range_sigma_m": 5.0,
            "labels": "SYNTHETIC_CAMPAIGN_TRUTH; not DSN flight performance"},
        "nuisance": {
            "models": {"D0_IDEAL": "none",
                       "D1_SESSION_CONSTANT_BIAS": "one constant range-rate bias per station pass (frequency / "
                                                   "slowly varying media offset over a pass)",
                       "D2_SESSION_BIAS_PLUS_LINEAR_DRIFT": "D1 plus a linear drift per pass, regressor "
                                                            "(t - t_mid)/half_span in [-1, 1]"},
            "priors_mps": ["FREE", 1.0e-5, 1.0e-4, 1.0e-3],
            "PRIMARY": "D1_SESSION_CONSTANT_BIAS, FREE (no operational prior is claimed)",
            "range": "no range nuisance (unchanged D2-D4 baseline)",
            "design_unit": "nuisance parameters in units of 1e-4 m/s for the scaled SVD; f_perp invariant"},
        "blocks": {"PRIMARY": ["RANGE_ONLY", "DOPPLER_ONLY", "RANGE_PLUS_DOPPLER"],
                   "secondary_RANGE_PLUS_DOPPLER_PLUS_DDOR": "NOT evaluated (no DeltaDOR search in CD1)"},
        "metrics": ["observation counts", "range f_perp", "Doppler-only f_perp", "combined f_perp", "Delta f_perp",
                    "theta_K", "I(K|x) (state-projected orthogonal K energy)", "I_kk", "sigma_K", "sigma_K/K",
                    "||b_K||", "scaled design singular values", "condition number",
                    "weakest right-singular vector and its K component", "K-bias correlation",
                    "K-drift correlation", "sigma_K inflation vs D0", "f_perp loss vs D0"],
        "implementation": "phase17_r1od2_core.info_metrics (QR projector f_perp, R1COV QR sigma_K, SVD); "
                          "rank(H^T W H) is NOT an observability gate",
        "thresholds": {"relative": {"STRONG_DIRECTION_GAIN": "delta > 0.05",
                                    "MODEST_DIRECTION_GAIN": "0.02 < delta <= 0.05",
                                    "MAGNITUDE_ONLY": "-0.02 <= delta <= 0.02",
                                    "DIRECTION_LOSS": "delta < -0.02",
                                    "implementation": "phase17_r1od2_core.relative_gain_class"},
                       "absolute": {"STRONG_K_SEPARATION": "f >= 0.50", "MODERATE_K_SEPARATION": "f >= 0.35",
                                    "WEAK_K_SEPARATION": "otherwise",
                                    "implementation": "phase17_r1od2_core.absolute_class"},
                       "delta": "f_perp(range + Doppler) - f_perp(range), same propagation"},
        "hypotheses": {"H1": "counted Doppler adds a more robust K-specific direction than DeltaDOR did",
                       "H2": "range + Doppler gain persists across epoch and orbit under HF dynamics",
                       "H3": "the direction survives Doppler nuisance freedom"},
        "population": {
            "cells": "HF, RANGE_PLUS_DOPPLER, ALL_VISIBLE, Tc 60 s, sigma 1e-4 m/s; 5 orbits x 24 epochs = 120; "
                     "feasible = >= 7 Doppler rows",
            "POP(X)": "fraction(delta_X > 0) >= 0.75 AND fraction(delta_X > 0.02) >= 0.50 over feasible cells",
            "ABS": "fraction(combined f_perp under D1 FREE >= 0.35) >= 0.50",
            "headline_nuisance": "D1_SESSION_CONSTANT_BIAS FREE (D0 reported alongside)"},
        "decision_tree": {
            "order": ["A", "B", "C", "D", "E"],
            "A": "COUNTED_DOPPLER_ROBUSTLY_ADDS_K_SRP_DIRECTION: POP(D0) and POP(D1 FREE) and ABS",
            "B": "COUNTED_DOPPLER_ROBUSTLY_IMPROVES_DIRECTION_BUT_K_REMAINS_WEAK: POP(D0) and POP(D1 FREE) and not ABS",
            "C": "COUNTED_DOPPLER_GAIN_IS_GEOMETRY_DEPENDENT: not POP(D0) and fraction_D0(delta > 0.02) >= 0.10",
            "D": "COUNTED_DOPPLER_K_INFORMATION_IS_SYSTEMATICALLY_FRAGILE: POP(D0) and not POP(D1 FREE)",
            "E": "COUNTED_DOPPLER_ADDS_INFORMATION_MAGNITUDE_WITHOUT_BREAKING_K_STATE_DEGENERACY: "
                 "not POP(D0) and fraction_D0(delta > 0.02) < 0.10",
            "nonlinear_confirmation": "may become justified only for A or B (never started automatically)",
            "alternative_parameterization": "justified as the next characterization for C, D or E"},
        "classes": {
            "NUISANCE_RETENTION": "cells with delta_D0 > 0.02: median delta_D1FREE / delta_D0 >= 0.75 RETAINED; "
                                  ">= 0.25 PARTIALLY_RETAINED; else LOST; no such cell -> NOT_APPLICABLE_NO_IDEAL_GAIN",
            "FIDELITY_EFFECT": "POP(D0) differs HF vs SIMPLIFIED -> FIDELITY_CHANGES_POPULATION_VERDICT; else RMS over "
                               "cells of (delta_HF - delta_SIMP) > 0.02 -> FIDELITY_MATERIAL_VERDICT_STABLE; else "
                               "FIDELITY_NEGLIGIBLE",
            "decomposition": "per cell: d(range f), d(Doppler-only f), d(combined f), d(delta), HF - SIMPLIFIED, D0"},
        "representative_cells": {"cells": [["CANON", 0], ["CANON", 6], ["CANON", 12], ["CANON", 18]],
                                 "dynamics": "HF", "use": "time localization + physical signature (interpretation only)",
                                 "time_localization": "cumulative range+Doppler and Doppler-only f_perp and I(K|x) "
                                                      "using rows with t <= tau, tau = 10..100% of the arc; LOCALIZED "
                                                      "if one 10% slice adds > 50% of the final combined I(K|x), else BROAD",
                                 "signature": "corr(Doppler h_K, LOS . dv/dK), corr(range h_K, LOS . dr/dK); share of "
                                              "Doppler-row orthogonal energy within +-1 h of eclipse transitions and "
                                              "within +-10 min of pass start/end, vs the row fraction"},
        "fd_spot_checks": {
            "cases": [["CANON", 0], ["O2_FULL_SUN_HIGH_BETA", 12]],
            "arc": "2 canonical orbits (FD tractability), same x0 / epoch / HF force model",
            "rows": "6 Doppler rows per case, evenly spaced over the ALL_VISIBLE Tc 60 rows",
            "analytic": "production Jacobian on the HF propagation at rtol 1e-12",
            "oracle": "central differences of the production observable on repropagated HF trajectories at "
                      "rtol 1e-13, atol 1e-14; steps x {10, 100} m, v {0.01, 0.1} m/s, K {1e-3, 3e-3} m^2/kg",
            "metric": "per column min over steps of ||fd - an|| / ||an|| over the 6 rows",
            "gates": {"COUNTED_DOPPLER_STATE_FD_GATE": "every state column <= 1e-5 in both cases",
                      "COUNTED_DOPPLER_K_FD_GATE": "K column <= 1e-3 in both cases"}},
        "independent_rebuild": {
            "f_perp": "every HF primary cell: lstsq route (phase17_r1o_cd1_core.independent_f_perp) for "
                      "DOPPLER_ONLY D0, RANGE_PLUS_DOPPLER D0, D1 FREE, D2 FREE and D1 prior 1e-4: |diff| <= 1e-10",
            "doppler_rows": "every HF cell, 12 rows: R4 four-event module (M3 3x3 implicit sensitivity) at zero delay "
                            "vs production R3 rows; state rel <= 1e-9, K abs / max|h_K| <= 1e-9",
            "range_k_column": "every HF cell, 12 range rows: dx/dK-in-STM-slot (event-epoch interpolation) "
                              "through two_way_range_nominal_and_initial_jacobian vs production "
                              "_two_way_range_k_srp_column (H_x0 Phi(t3)^-1 S_K(t3) at the receive-tag node, "
                              "Phase 17A-R qualified to 4.9e-8); abs / max|h_K| <= 1e-7 (pre-measured before "
                              "any metric: 2.7e-9, CANON k0 SIMPLIFIED)",
            "whitening": "Doppler w == 1/sigma^2 exactly; range w == 1/5^2 exactly",
            "nuisance_ownership": "each nuisance column non-zero only on its own pass's Doppler rows; range rows zero"},
        "forbidden": ["nonlinear K estimation", "BLS/SRIF/SRUKF K recovery", "Monte Carlo", "K reparameterization",
                      "stochastic K / empirical accelerations", "schedule optimisation or K-informed row selection",
                      "DeltaDOR epoch / baseline / session / phase < 60 deg search",
                      "threshold changes after results", "production physics or measurement changes",
                      "rank(H^T W H) as an observability gate"],
        "bug_policy": "IMPLEMENTATION_BUG_FIX logged, preregistration preserved, affected results re-run and "
                      "decision fields compared; a change of the experiment definition => PREDECLARATION_INVALIDATED",
        "production_defect_policy": "a real production correctness defect stops the science campaign; isolate and "
                                    "recommend a repair; no silent production repair",
    }
    pre["canonical_sha256"] = CD.canonical_sha256(pre)
    OUT.write_text(json.dumps(pre, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(pre["canonical_sha256"])


if __name__ == "__main__":
    main()
