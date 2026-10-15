"""PHASE 17-R1O-D3 - write the predeclared tracking-design grid (run ONCE, before any metric).

The grid is serialised canonically (sorted keys, no whitespace); its SHA-256 is
stored inside the file under "canonical_sha256" (computed with that key absent)
and pinned in phase17_r1od3_core.GRID_SHA256.  The campaign refuses to run if
either disagrees with the file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parents[3] / "studies" / "phase17" / "results"
PATH = ARTIFACTS.parent / "configs" / "phase17_r1od3_predeclared_grid.json"

#: D2's synodic month (E1-E3 offsets = k/4 of it): 29.530589 d.
SYNODIC_S = 2551442.8896
CANONICAL_ET0 = 857806356.99998


def canonical_sha256(grid: dict) -> str:
    body = {k: v for k, v in grid.items() if k != "canonical_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


GRID = {
    "phase": "17-R1O-D3",
    "written_before_any_d3_k_metric": True,
    "entering_head": "d071d4c09d511de9eaa6e36b6ee32ba9b4056ba5",
    "truth_level": "SYNTHETIC_CAMPAIGN_TRUTH (K_truth = 0.01 m^2/kg)",
    "production_contracts": {
        "dynamics": "F1-qualified variational equations (SRP position gradient); canonical simplified force model "
                    "(Moon point mass + lunar J2 + cannonball SRP, conical penumbra; mu_earth = mu_sun = 0)",
        "earth": "EARTH_FIXTURE_CANONICAL, checked provider (measurement_contracts); legacy table guarded",
        "visibility": "PRODUCTION_VISIBILITY_ELEVATION_AND_OCCULTATION (10 deg mask + lunar occultation)",
        "forbidden": ["EARTH_FIXTURE_LEGACY_OFFSET_CLAMPED", "DDOR_ROW_RULE_LEGACY_ELEVATION_ONLY",
                      "frozen-Earth bridges", "pre-F1 variational model"],
    },
    "stage_A_epoch_grid": {
        "count": 24,
        "synodic_month_s": SYNODIC_S,
        "spacing_s": SYNODIC_S / 24,
        "spacing_deg_synodic_phase": 15.0,
        "canonical_et0": CANONICAL_ET0,
        "definition": "et0_k = canonical_et0 + k * synodic_month_s / 24, k = 0..23 (k = 0 is the canonical epoch exactly)",
        "held_invariant": "the SAME J2000/LME spacecraft state vector x0 (campaign initial state) at every epoch, "
                          "arc 15 canonical orbits, 90 s grid, K = 0.01 -- the D2 epoch method (E1-E3 at k/4) "
                          "densified; Sun, Earth, Earth orientation, stations and visibility are recomputed at each "
                          "absolute epoch; the trajectory is NOT carried continuously through the month",
        "wraparound": "NONE: k = 23 and k = 0 are not treated as adjacent (the geometry at et0 + S is not that at et0)",
    },
    "baseline_strategies": {
        "primary": ["GC_ONLY", "GM_ONLY", "GC_PLUS_GM"],
        "pairs": {"GC": ["Goldstone DSN", "Canberra DSN"], "GM": ["Goldstone DSN", "Madrid DSN"]},
        "madrid_canberra": "not a DSN DeltaDOR baseline (DSN-210F s2.1.4): excluded",
        "joint_rule": "GC_PLUS_GM stacks separately qualified GC and GM sessions in the same arc; no simultaneity "
                      "required; every row still needs its own pair's dual visibility",
    },
    "session_rules": {
        "window": "a contiguous run of converged production DeltaDOR rows of one pair on the 90 s grid "
                  "(D2 sessions_from_indices; each row requires both stations visible: elevation >= 10 deg AND "
                  "no lunar occultation)",
        "window_duration": "t_last - t_first + 90 s",
        "min_window_s": 1200.0,
        "session_max_s": 3600.0,
        "session_interval": "if window >= 3600 s: the 3600 s interval centred on the window midpoint; "
                            "else the whole window",
        "normal_points": "3 per session: the session's rows at fractions 1/6, 1/2, 5/6 of the row list "
                         "(index round(f * (n-1))), deduplicated -- the D2/CB13 operational abstraction",
        "source": "DSN-210F s2.1 / s2.1.4 (~1 h pass of 5-10 scans); D2 OPS convention",
    },
    "schedule_patterns": {
        "single_baseline": {
            "FIRST_SESSION": "earliest qualifying window (by start time)",
            "MID_SESSION": "qualifying window whose midpoint is closest to the arc midpoint (tie -> earlier)",
            "LAST_SESSION": "latest qualifying window",
            "FIRST_PLUS_LAST": "FIRST and LAST (requires >= 2 qualifying windows, else not feasible)",
            "ALL_FEASIBLE_SESSIONS": "every qualifying window",
        },
        "joint": {
            "FIRST_GC_PLUS_FIRST_GM": "FIRST_SESSION of GC + FIRST_SESSION of GM",
            "FIRST_LAST_GC_PLUS_FIRST_LAST_GM": "FIRST_PLUS_LAST of GC + FIRST_PLUS_LAST of GM",
            "ALL_GC_PLUS_ALL_GM": "ALL_FEASIBLE_SESSIONS of GC + ALL_FEASIBLE_SESSIONS of GM",
        },
        "primary_pattern_for_decision": {"GC_ONLY": "ALL_FEASIBLE_SESSIONS", "GM_ONLY": "ALL_FEASIBLE_SESSIONS",
                                         "GC_PLUS_GM": "ALL_GC_PLUS_ALL_GM"},
        "selection_uses_k_information": False,
    },
    "resource_conflict": {
        "rule": "Goldstone complex-level no-overlap: a GC session and a GM session conflict if their session "
                "intervals overlap (closed intervals). Antenna identity is not represented: conservative "
                "complex-level rule, labelled as a limitation.",
        "resolution": "GC priority: every GC session of the pattern is kept; a GM session overlapping any kept GC "
                      "session is dropped and logged (RESOURCE_CONFLICT = YES for the raw schedule)",
        "primary_joint_schedules_double_book": False,
    },
    "noise": {
        "primary": {"semantics": "CONSTANT_DELAY_SIGMA", "sigma_ns": 0.048,
                    "source": "DSN-210F Table 9 RSS random (identical to Rev. E)"},
        "historical_continuity_diagnostic": {"semantics": "ANGULAR_SIGMA_PROJECTED_BY_BPERP", "sigma_nrad": 5.0,
                                             "rows": "DENSE (every dual-visible grid row)",
                                             "use": "canonical reproduction and predictor only"},
        "dense_delay_diagnostic": {"semantics": "CONSTANT_DELAY_SIGMA", "sigma_ns": 0.048,
                                   "rows": "DENSE", "use": "DDOR-only predictor, geometric upper bound"},
    },
    "calibration": {
        "nominal_bias_prior_ns": 0.0317,
        "nominal_source": "RSS of DSN-210F Table 8 systematic rows = sqrt(0.007^2+0.007^2+0.012^2+0.019^2+0.020^2) "
                          "(corrects D2's 0.033 arithmetic; REPOSITORY modelling choice, not a DSN Gaussian claim)",
        "cases": {
            "C0_IDEAL": "no bias nuisance",
            "C1_NOMINAL_CALIBRATED_x0.5": "one delay bias per session, Gaussian prior 0.5 x nominal",
            "C1_NOMINAL_CALIBRATED_x1.0": "one delay bias per session, Gaussian prior 1.0 x nominal (THE nominal)",
            "C1_NOMINAL_CALIBRATED_x2.0": "one delay bias per session, Gaussian prior 2.0 x nominal",
            "C2_FREE_SESSION_BIAS": "one free delay bias per session",
        },
        "weight_policy": "random 0.048 ns in every case; systematics only via the nuisance (no double counting)",
        "bias_ownership": "one column per (baseline, session); column j is 1 exactly on the rows of session j",
    },
    "metrics": ["range n_obs", "DDOR normal points", "n_sessions", "session_hours", "station_hours",
                "range f_perp", "DDOR-only f_perp (dense delay, predictor)", "combined f_perp", "delta_f_perp",
                "theta_K", "I(K|x)", "sigma_K", "sigma_K/K", "||b_K||", "design singular values",
                "design condition", "weakest right-singular-vector K component",
                "bias posterior sigma", "max |K-bias correlation|", "sigma_K inflation vs C0",
                "delta loss vs C0"],
    "forbidden_metrics": ["rank(H^T W H) or any normal-matrix numerical rank as an observability gate"],
    "thresholds": {
        "relative": {"STRONG_DIRECTION_GAIN": "delta > 0.05", "MODEST_DIRECTION_GAIN": "0.02 < delta <= 0.05",
                     "MAGNITUDE_ONLY": "-0.02 <= delta <= 0.02", "DIRECTION_LOSS": "delta < -0.02",
                     "implementation": "phase17_r1od2_core.relative_gain_class (verified identical)"},
        "absolute": {"STRONG_K_SEPARATION": "f >= 0.50", "MODERATE_K_SEPARATION": "0.35 <= f < 0.50",
                     "WEAK_K_SEPARATION": "f < 0.35", "implementation": "phase17_r1od2_core.absolute_class"},
        "materiality_tolerance_delta": 0.005,
    },
    "tracking_window": {
        "definition": "for a primary strategy with its primary pattern under C1 nominal (x1.0) and operational "
                      "sampling: delta_f_perp > 0.05 at >= 3 CONSECUTIVE epoch-grid points (no wraparound), with no "
                      "visibility violation and no resource double-booking",
        "min_consecutive": 3,
        "absolute_label": "reported independently; not required",
    },
    "robustness": {
        "window_robust_if": "the window exists AND the Stage B orbit check at the window-centre epoch gives "
                            "ORBIT_ROBUSTNESS_CLASS = PRESERVES AND Stage C gives PRESERVES_PRIMARY_INTERPRETATION",
        "adjacent_epoch": "a window of >= 3 consecutive points contains an epoch with STRONG neighbours on both sides",
    },
    "stage_B_orbit_robustness": {
        "orbits": {"CANON": "campaign x0", "O1_LOW_BETA_ECLIPSE_RICH": "D2 grid / GEO G1_b00",
                   "O2_FULL_SUN_HIGH_BETA": "D2 grid / GEO G1_b80",
                   "O3_HIGH_ALTITUDE_500KM": "D2 grid / GEO G2_alt500_b20_fixD",
                   "O4_POLAR": "D2 grid / GEO G3_i90.0_bcan",
                   "O5_FAVOURABLE_GEO_MAX_FPERP": "D2 grid / GEO G3_i26.7_bcan"},
        "orbit_state": "each orbit's D2 J2000 state vector, held fixed while the epoch shifts (same method as Stage A)",
        "epochs": "k in {0, 6, 12, 18} (the D2 quarter-phase epochs) PLUS, by rule, the centre epoch of the longest "
                  "primary window (tie -> earliest; if no window: none added)",
        "strategies": "GC_ONLY, GM_ONLY, GC_PLUS_GM with their primary patterns, C0 and C1 nominal",
        "class_rule": "at the window-centre epoch for the window's strategy: PRESERVES if >= 4 of the 5 non-canonical "
                      "orbits keep STRONG under C1 nominal; PARTIAL if 2-3; FAILS if 0-1; NO_WINDOW_TO_TEST if no "
                      "primary window exists (results still reported)",
    },
    "stage_C_high_fidelity": {
        "force_model": "D2 fidelity path: GL1800F 50x50 MOON_PA_DE440 + Earth/Sun point-mass third body (SPICE) + "
                       "same SRP, harmonic STM opt-in (F1-qualified variational equations)",
        "cases_by_rule": [
            "HF1: k = 0, GC_ONLY, ALL_FEASIBLE_SESSIONS (canonical favourable historical cell)",
            "HF2: first epoch of the longest primary window, its strategy/pattern; fallback if no window: the "
            "GC_ONLY epoch with the largest C1-nominal delta",
            "HF3: the GC_ONLY epoch with the smallest C1-nominal delta (unfavourable)",
            "HF4: k = 0, GC_PLUS_GM, ALL_GC_PLUS_ALL_GM (joint)",
        ],
        "calibration": "C0 and C1 nominal",
        "class_rule": "PRESERVES_PRIMARY_INTERPRETATION if every case keeps its C1-nominal relative class, or moves "
                      "by < materiality tolerance 0.005 across a class boundary; else CHANGES_PRIMARY_INTERPRETATION",
        "sessions": "rebuilt from the high-fidelity trajectory with the same rules",
    },
    "predictors": {
        "candidates": ["DDOR-only f_perp (dense rows, delay 0.048 ns)", "range-only f_perp",
                       "projected (plane-of-sky) baseline magnitude, mean over the pattern's normal points",
                       "Sun-Moon-Earth angle (mid arc)", "range LOS vs SRP-direction angle (mid arc)",
                       "number of feasible sessions", "normal-point count",
                       "baseline vs plane-of-sky K-displacement |cos| (dense rows; D2 pair_geometry)"],
        "response": "delta_f_perp, C1 nominal, primary pattern, per (strategy, epoch)",
        "method": "Spearman rank correlation over the 24 epochs per strategy, and pooled GC+GM single-baseline cells",
        "strength": {"STRONG": "|rho| >= 0.7", "MODERATE": "0.5 <= |rho| < 0.7", "WEAK": "0.3 <= |rho| < 0.5",
                     "NONE": "|rho| < 0.3"},
        "best_supported": "the candidate with the largest pooled |rho|; 'NONE_ADEQUATE' if its strength is WEAK or NONE",
    },
    "joint_baseline_effect": {
        "compare": "GC_PLUS_GM (ALL_GC_PLUS_ALL_GM) vs GC_ONLY (ALL_FEASIBLE_SESSIONS), same range data, 24 epochs",
        "order": ["DEGRADES_DIRECTION", "BROADENS_WINDOW", "STRENGTHENS_EXISTING_WINDOW", "CALIBRATION_HELP_ONLY",
                  "NO_MATERIAL_BENEFIT"],
        "DEGRADES_DIRECTION": "joint C1-nominal STRONG epoch count < GC count AND median over epochs of "
                              "(delta_joint - delta_GC) < -0.01",
        "BROADENS_WINDOW": "longest joint window > longest GC window, OR joint STRONG count >= GC STRONG count + 2 (C1 nominal)",
        "STRENGTHENS_EXISTING_WINDOW": "median over GC-STRONG epochs of (delta_joint - delta_GC) >= +0.01 (C1 nominal)",
        "CALIBRATION_HELP_ONLY": "median over epochs of (C2 retained fraction joint - GC) >= +0.10 while the C1 "
                                 "median improvement < 0.01; retained fraction = delta_C2 / delta_C0 for cells with "
                                 "delta_C0 > 0.02",
        "NO_MATERIAL_BENEFIT": "otherwise",
    },
    "calibration_effect_labels": {
        "IDEAL_VS_NOMINAL": "median over GC_ONLY primary epochs with delta_C0 > 0.02 of delta_C1/delta_C0: "
                            "> 0.75 NOMINAL_RETAINS_MOST; 0.25-0.75 NOMINAL_RETAINS_PART; < 0.25 NOMINAL_REMOVES_MOST",
        "NOMINAL_VS_FREE": "same with delta_C2/delta_C1 (cells with delta_C1 > 0.02): > 0.75 FREE_RETAINS_MOST; "
                           "0.25-0.75 FREE_RETAINS_PART; < 0.25 FREE_REMOVES_MOST",
    },
    "hypotheses": {
        "H1": "The D2 canonical GC gain belongs to a finite, multi-day region of geometry, not one isolated point.",
        "H2": "A joint GC+GM schedule broadens or stabilises useful K-direction information relative to GC-only, "
              "especially with session biases estimated.",
        "H3": "Calibration is a first-order limiter: nominal prior preserves materially more than free biases, "
              "materially less than ideal.",
    },
    "decision_tree": {
        "order": ["A", "B", "C", "D"],
        "A": "DDOR_TRACKING_DESIGN_HAS_ROBUST_FAVORABLE_WINDOWS: some primary strategy has a tracking window "
             "(>= 3 consecutive STRONG, C1 nominal) that is ROBUST (orbit PRESERVES and high fidelity PRESERVES)",
        "B": "DDOR_TRACKING_GAIN_REMAINS_CONFINED_TO_NARROW_GEOMETRIC_WINDOWS: C1-nominal STRONG at >= 2 epochs "
             "(any primary strategy) but no robust window",
        "C": "DDOR_TRACKING_GEOMETRY_EXISTS_BUT_IS_CALIBRATION_LIMITED: some primary strategy has a C0 window "
             "(>= 3 consecutive STRONG) but the total C1-nominal STRONG count is <= 10% of the C0 STRONG count",
        "D": "DDOR_TRACKING_DESIGN_DOES_NOT_GENERALIZE_THE_D2_GAIN: otherwise",
        "absolute_note": "no outcome implies absolute K identifiability; the absolute label is reported separately",
        "nonlinear_confirmation": "YES only for outcome A",
    },
    "escalation": {
        "orbit": "Stage B adds the window-centre epoch only if a primary window exists",
        "high_fidelity": "Stage C cases fixed by rule above; no further escalation",
        "conflicts": "if resource conflicts drop > 50% of GM sessions in ALL_GC_PLUS_ALL_GM at any epoch, report it "
                     "(no schedule re-optimisation)",
    },
    "continuity_checks": {
        "canonical": "k = 0 CANON: range f_perp 0.091502 and DENSE_ANGULAR 5 nrad GC combined 0.172343 must reproduce "
                     "studies/phase17/reports/phase17_authoritative_status.md (post-F1) to 1e-6",
    },
    "real_source_overlay": {
        "catalog": "DSN 810-005 Module 107 Rev. F Table 3 (ICRF3-SX), coordinates only",
        "criterion": "at least one source within 0.1 rad (DSN-210F Table 7 nominal) of the geocentric spacecraft "
                     "direction at each primary session midpoint",
        "not_assessed": ["flux", "SNR", "structure", "Ka/X interchangeability"],
        "if_unavailable": "REAL_SOURCE_AVAILABILITY = UNRESOLVED",
    },
    "grid_changes": "none permitted; implementation fixes are logged as IMPLEMENTATION_BUG_FIX without editing this file",
}


def main() -> None:
    if PATH.exists():
        raise SystemExit("predeclared grid already exists; refusing to overwrite")
    grid = dict(GRID)
    grid["canonical_sha256"] = canonical_sha256(grid)
    PATH.write_text(json.dumps(grid, indent=1, sort_keys=True), encoding="utf-8")
    print(grid["canonical_sha256"])


if __name__ == "__main__":
    main()
