"""PHASE 17-R1O-D2 s11 - write the predeclared robustness grid.

Run BEFORE any new D2 headline metric.  The only numbers computed here are
SPICE geometry descriptors (Sun-Moon-Earth angle) that document WHY the epochs
were chosen; no range, ΔDOR or K quantity is evaluated.

The grid hash is the SHA-256 of the newline-normalized (LF) JSON text, the
convention GEO's guard and git blobs use.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ARTIFACTS = HERE.parent / "artifacts"
SYNODIC_MONTH_D = 29.530589
GEO_ORBIT_CASES = {
    "O1_LOW_BETA_ECLIPSE_RICH": "G1_b00",
    "O2_FULL_SUN_HIGH_BETA": "G1_b80",
    "O3_HIGH_ALTITUDE_500KM": "G2_alt500_b20_fixD",
    "O4_POLAR": "G3_i90.0_bcan",
    "O5_FAVOURABLE_GEO_MAX_FPERP": "G3_i26.7_bcan",
}


def main() -> None:
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels(None, clear=True)
    import phase17_geo_core as G
    import phase17_r1m_core as M

    et0, t_orbit = M.campaign_epoch()
    x0 = M.campaign_initial_state().tolist()
    dur = 15.0 * t_orbit

    def sme(et):
        s, e = G.sun_mci(et), G.earth_state_mci(et)[:3]
        return math.degrees(math.acos(float(np.dot(s, e) / (np.linalg.norm(s) * np.linalg.norm(e)))))

    geos = [dict(case_id="CANON", family="canonical", et0=et0, x0=x0, duration_s=dur,
                 shadow_model="CONICAL_PENUMBRA", k_srp=M.K_TRUTH,
                 source="campaign epoch + campaign x0 (FXR canonical, GEO G0_consistent)")]
    for k, name in ((1, "E1_FIRST_QUARTER_PLUS"), (2, "E2_HALF_SYNODIC"), (3, "E3_THREE_QUARTER")):
        geos.append(dict(case_id=name, family="epoch", et0=et0 + k * SYNODIC_MONTH_D / 4 * 86400.0,
                         x0=x0, duration_s=dur, shadow_model="CONICAL_PENUMBRA", k_srp=M.K_TRUTH,
                         source="same J2000/LME state vector as CANON, epoch + %d/4 synodic month" % k))
    grid_geo = json.loads((ARTIFACTS / "phase17_geo_predeclared_grid.json").read_text())
    by_id = {c["id"]: c for c in grid_geo["cases"]}
    for name, gid in GEO_ORBIT_CASES.items():
        c = by_id[gid]
        geos.append(dict(case_id=name, family="orbit", et0=c["epoch_et"], x0=c["x0"],
                         duration_s=c["duration_s"], shadow_model=c["shadow_model"],
                         k_srp=c["k_srp"], source="GEO predeclared case %s" % gid))
    for g in geos:
        mid = g["et0"] + 0.5 * g["duration_s"]
        g["sun_moon_earth_deg_mid_arc_descriptor"] = sme(mid)

    grid = dict(
        phase="17-R1O-D2",
        written_before_any_new_headline_metric=True,
        entering_head="a4304fc35a27b212fcab1eaf6f936e3d45fe56c2",
        earth_fixture="EARTH_FIXTURE_CANONICAL (legacy table guarded: LegacyEarthLeakError)",
        dynamics="inherited canonical simplified: Moon point mass + lunar J2 + cannonball SRP "
                 "(conical penumbra), mu_earth = mu_sun = 0 (D7 spot check excepted)",
        tracking=dict(range="production two-way range, DSN3, 10 deg mask + lunar occultation, 90 s, "
                            "station sigma", ddor_grid_cadence_s=90.0,
                      ddor_earth_anchor="MEASUREMENT_EPOCH (Earth position and velocity at each grid epoch)"),
        ddor_row_rules=dict(
            reference="PRODUCTION_VISIBILITY_ELEVATION_AND_OCCULTATION",
            historical_for_D0_only="ELEVATION_ONLY_HISTORICAL_R1OD",
            reason="the R1O-D/FXR harness never tested lunar occultation: 93 of 263 canonical "
                   "FXR rows had the spacecraft behind the Moon for >= 1 station (row count only; "
                   "no K metric computed before this grid)"),
        geometries=geos,
        pairs=dict(GC=["Goldstone DSN", "Canberra DSN"], GM=["Goldstone DSN", "Madrid DSN"],
                   MC=["Madrid DSN", "Canberra DSN"],
                   standard=["GC", "GM"],
                   note="MC is not a DSN ΔDOR baseline (DSN-210 s2.1.4): reported, excluded from the "
                        "primary robustness population"),
        realizations=dict(
            DENSE_ANGULAR=dict(rows="every dual-visible 90 s grid epoch", semantics="ANGULAR",
                               sigmas_nrad=[1, 2, 5, 10, 30, 100], reference_nrad=5,
                               note="R1O-D/FXR convention, kept for continuity"),
            DENSE_DELAY=dict(rows="every dual-visible 90 s grid epoch", semantics="DELAY",
                             sigmas_ns=[0.024, 0.048, 0.1, 0.3, 1.0], reference_ns=0.048),
            OPS_EVERY_WINDOW=dict(rows="3 normal points (1/6, 1/2, 5/6) per dual-visible window "
                                       ">= 20 min", semantics="DELAY", sigma_ns=0.048,
                                  source="DSN-210 s2.1.4 (~1 h pass), CB13 (3 QSQ normal points/session), "
                                         "DSN-210 Table 9 random RSS 0.048 ns"),
            OPS_TWO_SESSIONS=dict(rows="3 normal points in the longest qualifying window of each arc half",
                                  semantics="DELAY", sigma_ns=0.048),
        ),
        primary_robustness_population=dict(
            members="all 9 geometries x pairs {GC, GM} x DENSE_ANGULAR 5 nrad, PRODUCTION_VISIBILITY rows",
            count=18,
            secondary_populations=["same 18 x OPS_EVERY_WINDOW 0.048 ns",
                                   "same 18 x DENSE_DELAY 0.048 ns",
                                   "MC pair (all geometries, reported only)"]),
        quasar_grid=dict(
            kind="GEOMETRY_ONLY_SYNTHETIC_QUASAR, one per session, placed relative to the geocentric "
                 "spacecraft direction at the session's middle row",
            separations_rad=[0.02, 0.05, 0.1, 0.2, 0.3], separation_reference_rad=0.1,
            position_angles_deg=[0, 45, 90, 135, 180, 225, 270, 315],
            evaluated_on="CANON x {GC, GM} x {DENSE_ANGULAR 5 nrad, OPS_EVERY_WINDOW}",
            formal_invariance_check="ΔDOR observable FD with 3 quasar directions on a perturbed trajectory",
            systematic_models=dict(station_location_m_per_component=0.02,
                                   ut1_1day_prediction_m=0.02, polar_motion_m=0.01,
                                   quasar_coordinate_rad=0.75e-9,
                                   station_location_and_EOP="coherent over the arc",
                                   quasar_coordinate="independent per session (one quasar per session)",
                                   source="DSN-210 Tables 7/10; CCSDS 500.1-G-2 eq. 25/26/32"),
            real_catalog="not locally available -> OPERATIONAL_SOURCE_AVAILABILITY = UNRESOLVED"),
        bias_cases=dict(
            B0="ideal, no bias",
            B1="common bias on all ΔDOR rows, injected unestimated: nominal 0.033 ns "
               "(RSS of DSN-210 Table 8 systematic rows), stress 1 ns",
            B2="independent per-session bias, sigma 0.033 ns, injected unestimated (RSS K shift)",
            B3="one delay bias per session solved, no prior",
            B4="one delay bias per session solved, prior sigma 0.033 ns",
            weighting="random component only when biases are solved (DSN-210 p. 20): "
                      "OPS 0.048 ns; DENSE keeps the historical 5 nrad (labelled)",
            evaluated_on="primary population x {DENSE_ANGULAR 5 nrad, OPS_EVERY_WINDOW}"),
        thresholds=dict(
            relative_gain="delta = f_perp(range+ΔDOR) - f_perp(range) on the SAME geometry: "
                          "STRONG > 0.05; MODEST (0.02, 0.05]; MAGNITUDE_ONLY [-0.02, 0.02]; "
                          "DIRECTION_LOSS < -0.02 (R1O-D thresholds + symmetric loss class)",
            absolute_separation="GEO gates: f_perp >= 0.50 STRONG; 0.35-0.50 MODERATE; < 0.35 WEAK",
            rank_and_weakest_mode="SVD of the whitened scaled DESIGN matrix only; normal-matrix rank "
                                  "never used as an observability gate",
            calibration_survival="a case survives nuisance treatment if B3 keeps delta >= 0.02 AND "
                                 "retains >= 50% of its B0 delta"),
        outcome_rules=dict(
            order="evaluate D first, then C, then B, else A",
            D="fewer than 25% of the primary population STRONG",
            C="primary population meets A's geometry test but fewer than 50% of it survive B3 in "
              "DENSE or in OPS",
            B="25-75% STRONG, or a single geometry family/pair holds all the non-strong cases",
            A="at least 75% STRONG, at most 10% DIRECTION_LOSS, at least 50% survive B3 in DENSE "
              "and in OPS, and at least 50% of OPS_EVERY_WINDOW cases at least MODEST"),
        fd_cases=dict(
            FD1=dict(geometry="CANON", pair="GC"), FD2=dict(geometry="E2_HALF_SYNODIC", pair="GC"),
            FD3=dict(geometry="O4_POLAR", pair="GM"),
            rows="the nominal PRODUCTION_VISIBILITY row set, frozen (no visibility re-evaluation)",
            oracle="independent 6-state re-propagation (lunar_od.dynamics.propagate_state) + "
                   "production solve_common_transmit_event at the same receive tags",
            position_steps_m=[10, 30, 100, 300, 1000], velocity_steps_mps=[0.01, 0.03, 0.1, 0.3, 1.0],
            k_steps=[1e-4, 3e-4, 1e-3, 3e-3],
            state_gate="every state column: best-step relative error (Frobenius over rows) <= 1e-4",
            k_gate="K column: best-step relative error <= 1e-3",
            materiality="|f_perp(range + FD ΔDOR) - f_perp(range + analytic ΔDOR)| <= 1e-3; "
                        "if violated with delta comparable to the science effect -> STOP"),
        surrogate_bridge="every primary case + CANON elevation-only: R1O surrogate on the same epochs",
        fidelity_spot_check=dict(
            force_model="GL1800F truncated 50x50 in MOON_PA_DE440 + Earth and Sun point-mass third "
                        "body (SPICE, scenario epoch) + same SRP; harmonic STM opt-in",
            cases=["CANON GC", "O2_FULL_SUN_HIGH_BETA GC",
                   "the primary-population case with the smallest delta (rule, not choice)"],
            classes=["PRESERVES_CLASS", "CHANGES_CLASS", "INCONCLUSIVE", "NOT_RUN"]),
        jacobian_materiality=dict(range_state_jacobian_f_perp_bound=3.77e-6,
                                  note="GEO; carried, not investigated"),
    )
    text = json.dumps(grid, indent=1, default=float)
    path = ARTIFACTS / "phase17_r1od2_predeclared_grid.json"
    path.write_text(text, encoding="utf-8", newline="\n")
    sha = hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()
    print("wrote", path, "sha256(LF) =", sha)
    for g in geos:
        print("  %-30s et0=%.3f SME(mid)=%.1f deg  %s" % (g["case_id"], g["et0"],
              g["sun_moon_earth_deg_mid_arc_descriptor"], g["source"]))


if __name__ == "__main__":
    main()
