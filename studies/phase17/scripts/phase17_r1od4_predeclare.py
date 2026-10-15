"""PHASE 17-R1O-D4 - write the confirmatory predeclaration (run ONCE, before any D4 K metric).

Canonical serialisation (sorted keys, no whitespace); SHA-256 stored inside the
file as "canonical_sha256" (computed with that key absent) and pinned in
phase17_r1od4_campaign.PREDECLARATION_SHA256.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parents[3] / "studies" / "phase17" / "results"
PATH = ARTIFACTS.parent / "configs" / "phase17_r1od4_predeclaration.json"


def canonical_sha256(d: dict) -> str:
    body = {k: v for k, v in d.items() if k != "canonical_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _x0_hashes() -> dict:
    g = json.loads((ARTIFACTS.parent / "configs" / "phase17_r1od2_predeclared_grid.json").read_text(encoding="utf-8"))
    ids = ("CANON", "O1_LOW_BETA_ECLIPSE_RICH", "O2_FULL_SUN_HIGH_BETA", "O3_HIGH_ALTITUDE_500KM",
           "O5_FAVOURABLE_GEO_MAX_FPERP")
    return {x["case_id"]: dict(source=x["source"], x0_sha256=hashlib.sha256(json.dumps(x["x0"]).encode()).hexdigest())
            for x in g["geometries"] if x["case_id"] in ids}


PHASES = [60.0 + 5.0 * i for i in range(22)]

PRE = {
    "phase": "17-R1O-D4 Goldstone-Madrid narrow-window confirmatory robustness qualification",
    "written_before_any_d4_k_metric": True,
    "entering_head": "89942a618731ab44151a4b9c3396540f1949105a",
    "hypothesis_source": "D3 secondary observation (studies/phase17/reports/phase17_r1od3_tracking_design_characterization.md s22): "
                         "GM FIRST_SESSION STRONG at D3 k = 5, 6, 7 (75/90/105 deg), C1 nominal, simplified dynamics; "
                         "exploratory, 1 of 13 patterns (multiple-comparison exposed)",
    "d3_primary_conclusion_not_reopened": True,
    "truth_level": "SYNTHETIC_CAMPAIGN_TRUTH (K_truth = 0.01 m^2/kg)",
    "literature": {"status": "D3_CONTRACT_RECONFIRMED_NO_REVISION",
                   "evidence": "DSN 810-005 index lists 210F (CL 26-2234, as read by D3), 107F (CL 25-3456); "
                               "107F PDF sha256 3a8da003e866ce1169164e5d1f733055923af208062acb19b696847fb2fc1d1d "
                               "identical to D3; CCSDS 500.1-G-2 retained",
                   "random_delay_sigma_ns": 0.048, "nominal_bias_prior_ns": 0.0317,
                   "prior_status": "REPOSITORY arithmetic from DSN-210F Table 8 systematic rows; not a DSN Gaussian claim"},
    "phase_grid": {
        "min_deg": 60.0, "max_deg": 165.0, "step_deg": 5.0, "count": 22, "phases_deg": PHASES,
        "d3_anchors_deg": [75.0, 90.0, 105.0, 135.0, 150.0],
        "epoch_convention": "et0 = 857806356.99998 + (phase_deg / 15) * synodic_month_s / 24, synodic_month_s = "
                            "2551442.8896 (identical to D3 k * S / 24 at phase = 15 k; D3 grid a651c4de)",
        "held_invariant": "same J2000/LME state vector per orbit, 15 canonical orbits, 90 s grid, K = 0.01",
        "recomputed_per_epoch": ["Sun", "Earth", "Earth orientation", "stations", "visibility", "range baseline",
                                 "DDOR rows", "sessions"],
    },
    "baseline": {"primary": "GM_ONLY (Goldstone-Madrid)", "not_searched": ["GC", "GC_PLUS_GM", "Madrid-Canberra"]},
    "schedules": {
        "primary": "FIRST_SESSION (earliest qualifying GM window, chronological only)",
        "reference": "ALL_FEASIBLE_SESSIONS (continuity with D3 primary)",
        "session_abstraction": "D3 exactly: contiguous dual-visibility window >= 20 min; session <= 60 min centred; "
                               "3 normal points at 1/6, 1/2, 5/6 of the session rows (phase17_r1od3_core.build_sessions)",
        "forbidden": ["MID", "LAST", "FIRST_PLUS_LAST", "custom best-session", "K-selected session"],
    },
    "calibration": {
        "weight": "random delay 0.048 ns per normal point in every case (no systematic term in the weight)",
        "cases": {"C0_IDEAL": "no bias nuisance", "C1_HALF": "prior 0.5 x 0.0317 ns", "C1_NOMINAL": "prior 0.0317 ns",
                  "C1_DOUBLE": "prior 2 x 0.0317 ns", "C2_FREE": "free bias per session"},
        "primary": "C1_NOMINAL",
        "d3_core_names": {"C0_IDEAL": "C0_IDEAL", "C1_HALF": "C1_x0.5", "C1_NOMINAL": "C1_x1.0",
                          "C1_DOUBLE": "C1_x2.0", "C2_FREE": "C2_FREE"},
    },
    "orbits": {
        "set": _x0_hashes(),
        "roles": {"CANON": "CANONICAL", "O1_LOW_BETA_ECLIPSE_RICH": "ECLIPSE_RICH / LOW_BETA",
                  "O2_FULL_SUN_HIGH_BETA": "FULL_SUN / HIGH_BETA", "O3_HIGH_ALTITUDE_500KM": "HIGH_ALTITUDE",
                  "O5_FAVOURABLE_GEO_MAX_FPERP": "PREVIOUSLY FAVOURABLE / HIGH f_perp"},
        "count": 5,
        "source": "studies/phase17/configs/phase17_r1od2_predeclared_grid.json (D2 / GEO predeclared configurations), unchanged",
    },
    "dynamics": {
        "HIGH_FIDELITY": "examples/phase17_r1od2_fidelity.propagate_fidelity: GL1800F 50x50 MOON_PA_DE440 + Earth and "
                         "Sun point-mass third body (SPICE) + production SRP (conical penumbra), harmonic STM opt-in, "
                         "F1-qualified variational equations, rtol 1e-12 (the D3 Stage C configuration)",
        "SIMPLIFIED": "phase17_r1od2_core.propagate_nom48 (Moon point mass + lunar J2 + SRP), controlled comparison",
        "matrix": "FULL: both dynamics for every orbit x phase cell (5 x 22 x 2 = 220 geometries); feasibility "
                  "measured before this file: one HF propagation ~257 s, 110 HF over 15 workers ~30 min",
        "fallback_only_if_full_matrix_fails_to_compute": "HF on all 22 CANON phases plus all 5 orbits at 75/90/105/"
                                                         "135/150 deg and the two nearest 5-deg shoulder points on each "
                                                         "side of each D3 cluster (65, 70, 110, 115, 125, 130, 155, 160)",
    },
    "thresholds": {
        "relative": {"STRONG_DIRECTION_GAIN": "delta > 0.05", "MODEST_DIRECTION_GAIN": "0.02 < delta <= 0.05",
                     "MAGNITUDE_ONLY": "-0.02 <= delta <= 0.02", "DIRECTION_LOSS": "delta < -0.02",
                     "implementation": "phase17_r1od2_core.relative_gain_class (unchanged)"},
        "absolute": {"STRONG_K_SEPARATION": "f >= 0.50", "MODERATE_K_SEPARATION": "0.35 <= f < 0.50",
                     "WEAK_K_SEPARATION": "f < 0.35", "implementation": "phase17_r1od2_core.absolute_class"},
        "delta_definition": "f_perp(range + DDOR) - f_perp(range), same orbit / epoch / dynamics range baseline",
    },
    "window": {
        "required_consecutive_points": 7,
        "required_phase_span_deg": 30.0,
        "equivalence": "D3: >= 3 consecutive points on a 15-deg grid = 30 deg endpoint span; D4 5-deg grid: >= 7 "
                       "consecutive STRONG points = 30 deg endpoint span (3 points on a 5-deg grid do NOT qualify)",
        "duration_days_per_30deg": 30.0 / 360.0 * 2551442.8896 / 86400.0,
        "overlap_region_deg": [75.0, 150.0],
        "overlap_rule": "a window overlaps the D3 candidate region if [start, end] intersects [75, 150]",
        "invalid_cells": "a phase with no qualifying GM session (FIRST infeasible) is NOT STRONG",
    },
    "tests": {
        "R0_canonical": "GM FIRST_SESSION, CANON, HIGH_FIDELITY, C1_NOMINAL: >= 7-point STRONG window overlapping [75,150]",
        "R1_orbit": "same criterion per orbit (HIGH_FIDELITY, C1_NOMINAL, FIRST_SESSION); report start, end, width, "
                    "days, max and median delta inside the window",
        "R2_calibration": "for each canonical HF window found under C1_NOMINAL (or, if none, report the C0/C1_HALF "
                          "windows): recompute the window under all five cases; persist / shrink / fragment / disappear",
        "R3_fidelity": "canonical SIMPLIFIED vs HIGH_FIDELITY (C1_NOMINAL, FIRST): existence, boundaries, class changes",
    },
    "classes": {
        "ORBIT_ROBUSTNESS": {"ORBIT_ROBUST": ">= 4/5 orbits satisfy R1", "ORBIT_DEPENDENT": "2-3/5",
                             "CANONICAL_ONLY": "1/5", "NO_CONFIRMED_WINDOW": "0/5"},
        "CALIBRATION": {"CALIBRATION_ROBUST": "canonical HF window (>= 7 pts, overlapping) survives C1_DOUBLE",
                        "CALIBRATION_SENSITIVE": "survives C1_NOMINAL but not C1_DOUBLE",
                        "CALIBRATION_FRAGILE": "exists only for C1_HALF and/or C0, or no window at C1_NOMINAL while "
                                               "C0/C1_HALF has one",
                        "NO_WINDOW_ANY_CALIBRATION": "no canonical HF window under any case"},
        "FIDELITY": {"FIDELITY_STABLE": "existence unchanged (both or neither) and, if both, each boundary moves <= 5 deg",
                     "FIDELITY_SHIFTS_WINDOW": "both have a window, a boundary moves > 5 deg",
                     "FIDELITY_CHANGES_EXISTENCE": "exactly one of SIMPLIFIED / HIGH_FIDELITY has a window"},
        "MECHANISM": {
            "quantities": "per canonical HF phase where ALL has > 1 session: d_diff = delta_ALL - delta_FIRST (C1 nominal); "
                          "ikk_ratio = I_kk(ALL) / I_kk(FIRST); added_orth_frac = (I_perp(ALL) - I_perp(FIRST)) / "
                          "(I_kk(ALL) - I_kk(FIRST)) where I_perp = conditional K information; first_orth_frac = "
                          "f_perp(FIRST)^2",
            "LATER_SESSIONS_DILUTE_DIRECTION": "median d_diff < -0.01 AND median ikk_ratio > 1.1 AND median "
                                               "added_orth_frac < median first_orth_frac",
            "LATER_SESSIONS_ADD_DIRECTION": "median d_diff > +0.01",
            "NO_SYSTEMATIC_DIFFERENCE": "otherwise",
        },
    },
    "confirmatory_success": [
        "1 canonical HF C1_NOMINAL FIRST has a >= 7-point (30 deg) STRONG window",
        "2 it overlaps [75, 150]",
        "3 >= 4 of the 5 orbits (CANON included) have an overlapping >= 7-point STRONG window (HF, C1_NOMINAL)",
        "4 every normal point of every counted window cell is contract-valid (visibility, occultation, coverage, events)",
        "5 the canonical window is not created solely by C0_IDEAL (C1_NOMINAL window exists by 1)",
        "6 independent design reconstruction reproduces the metrics (max |f_perp difference| <= 1e-10)",
        "7 no primary rule changed after inspection (predeclaration hash unchanged)",
    ],
    "decision_tree": {
        "order": ["A", "B", "C", "D", "E"],
        "A": "GM_NARROW_WINDOW_CONFIRMED_AND_ROBUST: all confirmatory_success items true",
        "B": "GM_WINDOW_EXISTS_BUT_IS_ORBIT_GEOMETRY_DEPENDENT: R0 true (canonical HF C1 window) but < 4/5 orbits",
        "C": "GM_WINDOW_IS_DYNAMICS_FIDELITY_FRAGILE: canonical SIMPLIFIED and HIGH_FIDELITY disagree on window "
             "existence (C1_NOMINAL, FIRST)",
        "D": "GM_WINDOW_IS_CALIBRATION_FRAGILE: canonical HF has no C1_NOMINAL window but has one under C0_IDEAL or C1_HALF",
        "E": "D3_SECONDARY_GM_WINDOW_DOES_NOT_CONFIRM: otherwise (no canonical HF C1_NOMINAL window)",
        "nonlinear_confirmation": "YES only for A (narrowly, GM design only); NO otherwise",
        "branch_status_if_E": "CLOSED_FOR_CURRENT_K_SRP_FORMULATION",
    },
    "mechanism_quantities": ["session midpoint geometry (geocentric spacecraft declination, Sun-Earth-spacecraft angle)",
                             "projected baseline", "session DDOR-only f_perp (dense rows inside the session interval)",
                             "pair DDOR-only f_perp (all dense rows)", "combined conditional K information",
                             "theta_K", "K-bias correlation", "added-session orthogonal fraction"],
    "literature_overlay": {"per_phase": ["geocentric lunar declination", "spacecraft geocentric declination at FIRST "
                                         "midpoint", "Sun-Earth-spacecraft angle at FIRST midpoint",
                                         "GM qualifying dual-visibility duration"],
                           "dsn_good_gm_geometry": "Sun-Earth-spacecraft > 10 deg AND declination > -15 deg "
                                                   "(LITERATURE_RESULT context, not a validity rule)"},
    "source_overlay": {"catalog": "DSN-107F Table 3 coordinates (D3 parser)", "criterion": "a source within 0.1 rad of "
                       "the geocentric spacecraft direction at every FIRST_SESSION midpoint",
                       "labels": ["COORDINATE_SOURCE_AVAILABLE", "NO_COORDINATE_SOURCE_WITHIN_LIMIT"]},
    "independent_verification": ["A predeclaration hash refusal", "B independent phase->ET reconstruction",
                                 "C per-normal-point visibility / occultation / coverage", "D session ownership",
                                 "E bias-column ownership", "F independent lstsq reconstruction of range + DDOR + bias + "
                                 "prior + whitening for every primary cell", "G D3 anchor reproduction at 75/90/105/"
                                 "135/150 (canonical SIMPLIFIED, FIRST and ALL, all five calibrations) vs "
                                 "studies/phase17/results/phase17_r1od3_primary_results.csv to 1e-12"],
    "metrics": ["range n_obs", "GM visibility-window count", "qualified session count", "normal-point count",
                "range f_perp", "DDOR-only f_perp", "combined f_perp", "delta f_perp", "theta_K", "I(K|x)", "sigma_K",
                "sigma_K/K", "||b_K||", "design singular values", "condition number", "weakest right-singular K component",
                "bias posterior sigma", "K-bias correlation"],
    "forbidden": ["rank(H^T W H) as an observability gate", "post-hoc schedules / phases / orbits", "threshold changes",
                  "nonlinear estimation", "production code changes"],
    "bug_policy": "IMPLEMENTATION_BUG_FIX logged, preregistration preserved, affected results re-run and decision fields "
                  "compared; a fix that changes the experiment definition => PREDECLARATION_INVALIDATED (stop)",
}


def main() -> None:
    if PATH.exists():
        raise SystemExit("predeclaration exists; refusing to overwrite")
    d = dict(PRE)
    d["canonical_sha256"] = canonical_sha256(d)
    PATH.write_text(json.dumps(d, indent=1, sort_keys=True), encoding="utf-8")
    print(d["canonical_sha256"])


if __name__ == "__main__":
    main()
