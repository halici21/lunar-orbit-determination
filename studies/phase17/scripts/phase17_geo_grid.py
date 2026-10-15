"""PHASE 17-GEO s19/s45 - freeze the geometry grid BEFORE any K metric exists.

Writes artifacts/phase17_geo_predeclared_grid.json.  This script computes only
geometry (states, beta, Earth-view angle); it imports no information-metric code
and propagates nothing.  The campaign script refuses to run a case that is not in
this file, and records this file's SHA-256 next to every result.
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

import phase17_geo_core as G  # noqa: E402
from phase17_r1m_core import K_TRUTH, campaign_epoch, campaign_initial_state  # noqa: E402

DSN3 = ("Goldstone DSN", "Madrid DSN", "Canberra DSN")
#: Nearest quarter phase to the canonical epoch: Sun-Earth separation seen from
#: the Moon = 90.13 deg at +6.85 d (0.05 d search grid over +/-10 d).
QUARTER_OFFSET_DAYS = 6.85
GRID_V1_SHA256 = "40965dcae03eb68a94508598090da6d381cbe952f107af7e8125bdf43562d8e7"
AMENDMENTS = [dict(
    id="A1", when="after v1 was written, before any K metric was computed",
    reason=("v1 geometry showed beta and Earth-view are collinear at the new-moon canonical epoch "
            "(Earth-view ~= 90 + beta for polar orbits; RAAN-root pairs differ by only ~10 deg). "
            "Added three polar cases at the nearest quarter-phase epoch (+6.85 d) where the "
            "beta -> Earth-view mapping flips. No existing case was moved or removed."),
    based_on="geometry only (beta, Earth-view, Sun-Earth separation); no f_perp or sigma_K existed",
    v1_sha256=GRID_V1_SHA256)]

# ---------------------------------------------------------------- thresholds
THRESHOLDS = {
    "reference": "G0 canonical-consistent range-only f_perp (f_ref)",
    "delta": "d = f_perp(case) - f_ref, absolute units of f_perp",
    "MAGNITUDE_ONLY": "|d| < 0.02",
    "MARGINAL_DIRECTION_CHANGE": "0.02 <= |d| < 0.10 (sign reported: gain or loss)",
    "MODERATE_DIRECTION_GAIN": "0.10 <= d < 0.25 and f_perp >= 0.35",
    "STRONG_DIRECTION_GAIN": "d >= 0.25 and f_perp >= 0.50",
    "DIRECTION_LOSS": "d <= -0.10",
    "robustness": ("A gain class is claimed only if it holds, to at most one class lower, under the "
                   "matched-observation-count variant AND at its predeclared validation neighbours."),
    "stage_effect_class": ("class of the stage's best matched-count case, with the within-stage "
                           "f_perp range (max - min) reported alongside"),
}

PREDICTIONS = {
    "G5_k_magnitude": (
        "The force is exactly linear in K (a = K g, lunar_od/srp.py), and the K column is dz/dK, "
        "which depends on K only through the negligible K-induced change of the linearisation "
        "trajectory.  Predicted BEFORE running: ||b_K||, f_perp, theta_K, I_K|x and absolute "
        "sigma_K are invariant to <1e-3 relative; sigma_K/K scales as 1/K.  Classification: "
        "INFORMATION_MAGNITUDE_ONLY (relative precision), not a direction change."),
    "H1": "f_perp(eclipse-rich) > f_perp(full-Sun) (literature contract C1)",
    "H2": "f_perp falls as |beta| crosses the full-Sun boundary (C2)",
    "H3": "matched-|beta| face-on vs edge-on pairs differ in f_perp (C3)",
}

RULES = {
    "epoch": "campaign manifest epoch for Sun, Earth position AND Earth rotation (consistent fixture)",
    "force_model": ("Moon point mass + lunar J2 (mean pole) + cannonball SRP with conical penumbra; "
                    "mu_earth = mu_sun = 0 exactly as the canonical fixture; frozen (s17)"),
    "tracking": "two-way range, 3 DSN stations unless stated, 10 deg mask, 90 s cadence, 5 m sigma",
    "orbit_construction": ("circular (e = 0) in the lunar mean-equator frame at the canonical semi-major "
                           "axis unless stated; argument of latitude 0 at epoch"),
    "raan_root_rule": ("each (inclination, beta) has two RAAN roots; the PRIMARY case takes the root "
                       "with the smaller RAAN in [0,360); G4 adds the other root as the matched partner"),
    "fixed_duration": "D = 15 canonical revolutions (the published W15 span)",
    "fixed_revolutions": "15 revolutions of the case's own Keplerian period",
    "matched_count": ("per stage: N_match = the minimum observation count over that stage's cases; "
                      "each case keeps rows at round(linspace(0, N-1, N_match)) of its time-ordered rows"),
    "G6_interaction_rule": ("run a 3x3 interaction surface of factor X with beta ONLY if stage X's "
                            "effect class is MODERATE_DIRECTION_GAIN or stronger; otherwise G6 is empty"),
    "validation_neighbours": ("for any case classed MODERATE or STRONG: beta +/-5 deg, or altitude "
                              "x0.8 and x1.25, points not in this grid"),
    "eclipse_counterfactual": ("ALWAYS run, analysis-only: identical orbits with the production "
                               "shadow_model='NO_SHADOW' for G0-consistent, polar beta 0 and polar "
                               "beta 20. Never a production default, never mission truth."),
    "jacobian_fd_cases": ["G2_alt30_b20_fixD", "G2_alt1000_b20_fixD", "G1_b00", "G1_b88",
                          "G3_i26.7_b00", "G0_consistent"],
    "jacobian_fd_arc": "4 revolutions of each geometry (same builder), step sweeps, floor-aware",
    "doppler_secondary_cases": ["G0_consistent", "G1_b00", "G1_b88", "G2_alt30_b20_fixD",
                                "G2_alt1000_b20_fixD"],
    "physical_validity": ("min altitude > 5 km over the arc, propagation success, finite states, "
                          "period within 2% of Keplerian, finite illumination"),
}


def case(cid, stage, x0, et0, duration_s, **kw):
    el = G.elements_from_state(x0)
    h = np.array(el["h_hat_mci"])
    return dict(
        id=cid, stage=stage, epoch_et=float(et0), x0=[float(v) for v in x0],
        duration_s=float(duration_s),
        k_srp=float(kw.pop("k_srp", K_TRUTH)), stations=list(kw.pop("stations", DSN3)),
        shadow_model=kw.pop("shadow_model", "CONICAL_PENUMBRA"),
        elements={k: el[k] for k in ("a_m", "e", "inc_deg", "raan_deg", "period_s")},
        beta_deg_epoch=G.beta_deg(h, et0), earth_view_deg_epoch=G.earth_view_deg(h, et0),
        **kw)


def main() -> None:
    from lunar_od.spice_loader import load_spice_kernels

    load_spice_kernels(None, clear=True)
    et0, t_canon = campaign_epoch()
    x_canon = campaign_initial_state()
    a_canon = G.elements_from_state(x_canon)["a_m"]
    D = 15.0 * t_canon
    rm = G.r_moon()
    mu = G.mu_moon()
    cases = []

    # G0 -- canonical orbit, as published fixture and consistent fixture
    cases.append(case("G0_published", "G0", x_canon, et0, D, fixture="PUBLISHED_R1M",
                      role="exact reproduction of the published W15 baseline"))
    cases.append(case("G0_consistent", "G0", x_canon, et0, D, fixture="CONSISTENT",
                      role="same orbit, epoch-consistent fixture = f_ref"))

    def circ(inc, beta, a=a_canon, root=0):
        roots = G.raan_for_beta(inc, beta, et0)
        if len(roots) <= root:
            return None, roots
        return G.state_from_elements(a, 0.0, inc, roots[root], 0.0, 0.0), roots

    # G1 -- polar beta sweep across eclipse-rich -> grazing -> full Sun
    for b in (0, 20, 40, 55, 65, 69, 71, 73, 80, 88):
        x0, roots = circ(90.0, float(b))
        cases.append(case(f"G1_b{b:02d}", "G1", x0, et0, D, fixture="CONSISTENT",
                          beta_target=b, raan_roots=roots))

    # G2 -- altitude, fixed duration and fixed revolutions, eclipse-rich and full-Sun
    for alt_km in (30, 50, 100, 200, 500, 1000):
        a = a_canon if alt_km == 100 else rm + alt_km * 1e3
        period = 2 * math.pi * math.sqrt(a ** 3 / mu)
        for beta, tag in ((20.0, "b20"), (85.0, "b85")):
            x0, roots = circ(90.0, beta, a=a)
            cases.append(case(f"G2_alt{alt_km}_{tag}_fixD", "G2", x0, et0, D, fixture="CONSISTENT",
                              altitude_nominal_km=(a - rm) / 1e3, beta_target=beta,
                              normalization="FIXED_DURATION", revolutions=D / period))
            if tag == "b20":
                cases.append(case(f"G2_alt{alt_km}_{tag}_fixN", "G2", x0, et0, 15.0 * period,
                                  fixture="CONSISTENT", altitude_nominal_km=(a - rm) / 1e3,
                                  beta_target=beta, normalization="FIXED_REVOLUTIONS",
                                  revolutions=15.0))

    # G3 -- inclination at matched beta (0 and the canonical beta)
    beta_canon = G.beta_deg(np.array(G.elements_from_state(x_canon)["h_hat_mci"]), et0)
    inc_canon = G.elements_from_state(x_canon)["inc_deg"]
    for inc in (round(inc_canon, 1), 45.0, 60.0, 90.0, 120.0, 150.0):
        for beta, tag in ((0.0, "b00"), (round(beta_canon, 1), "bcan")):
            x0, roots = circ(inc, beta)
            if x0 is None:
                cases.append(dict(id=f"G3_i{inc}_{tag}", stage="G3", invalid="beta unreachable",
                                  inc_deg=inc, beta_target=beta))
                continue
            cases.append(case(f"G3_i{inc}_{tag}", "G3", x0, et0, D, fixture="CONSISTENT",
                              inc_target=inc, beta_target=beta, raan_roots=roots))

    # G4 -- Earth-view at matched |beta| (second RAAN root), plus network subsets
    for b in (20, 80):
        x0, roots = circ(90.0, float(b), root=1)
        cases.append(case(f"G4_b{b:02d}_root2", "G4", x0, et0, D, fixture="CONSISTENT",
                          beta_target=b, partner=f"G1_b{b:02d}", raan_roots=roots))
    # AMENDMENT A1 (geometry-only, before any K metric): at the canonical epoch the
    # Moon is at new moon (Sun-Earth separation 174.4 deg seen from the Moon), so
    # for polar orbits Earth-view ~= 90 + beta and RAAN roots cannot decouple them.
    # At quarter phase the beta -> Earth-view mapping flips (beta 0 -> face-on,
    # beta 90 -> edge-on), so the same beta at the two epochs isolates Earth-view.
    et_q = et0 + QUARTER_OFFSET_DAYS * 86400.0
    for b in (0, 20, 80):
        roots = G.raan_for_beta(90.0, float(b), et_q)
        x0 = G.state_from_elements(a_canon, 0.0, 90.0, roots[0], 0.0, 0.0)
        cases.append(case(f"G4_quarter_b{b:02d}", "G4", x0, et_q, D, fixture="CONSISTENT",
                          beta_target=b, partner=f"G1_b{b:02d}", raan_roots=roots,
                          role="same beta as partner, quarter-phase epoch: Earth-view decoupled"))
    for st in ("Goldstone DSN", "Madrid DSN", "Canberra DSN"):
        cases.append(case(f"G4_net_{st.split()[0]}", "G4", x_canon, et0, D, fixture="CONSISTENT",
                          stations=(st,), role="identical dynamics, different measurement projection"))

    # G5 -- K magnitude control on the canonical-consistent geometry
    for k in (0.0025, 0.005, 0.02, 0.04):
        cases.append(case(f"G5_k{k}", "G5", x_canon, et0, D, fixture="CONSISTENT", k_srp=k))

    # eclipse counterfactual (analysis-only)
    for src in ("G0_consistent", "G1_b00", "G1_b20"):
        base = next(c for c in cases if c["id"] == src)
        cases.append(case(f"CF_noshadow_{src}", "CF", np.array(base["x0"]), et0, base["duration_s"],
                          fixture="CONSISTENT", shadow_model="NO_SHADOW", counterfactual_of=src))

    # Sun-Earth separation seen from the Moon (Earth-shadow applicability)
    s = G.sun_mci(et0)
    e = G.earth_state_mci(et0)[:3]
    sep = math.degrees(math.acos(np.dot(s, e) / np.linalg.norm(s) / np.linalg.norm(e)))

    grid = dict(
        phase="17-GEO", written_before_any_k_metric=True,
        epoch_et=et0, canonical_period_s=t_canon, fixed_duration_s=D,
        canonical_semi_major_axis_m=a_canon, canonical_inclination_lme_deg=inc_canon,
        canonical_beta_deg=beta_canon, full_sun_boundary_beta_deg=math.degrees(math.asin(rm / a_canon)),
        sun_earth_separation_from_moon_deg=sep, grid_version=2, amendments=AMENDMENTS,
        quarter_epoch_et=et0 + QUARTER_OFFSET_DAYS * 86400.0,
        thresholds=THRESHOLDS, predictions=PREDICTIONS, rules=RULES, cases=cases)
    ARTIFACTS.mkdir(exist_ok=True)
    out = ARTIFACTS / "phase17_geo_predeclared_grid.json"
    text = json.dumps(grid, indent=1)
    out.write_text(text)
    print("wrote", out, "sha256", hashlib.sha256(text.encode()).hexdigest())
    n_valid = sum(1 for c in cases if "invalid" not in c)
    print("cases:", len(cases), "valid:", n_valid)
    for c in cases:
        if "invalid" in c:
            print("  INVALID", c["id"], c["invalid"])
        else:
            print(f"  {c['id']:24s} i={c['elements']['inc_deg']:7.2f} raan={c['elements']['raan_deg']:7.2f} "
                  f"alt={(c['elements']['a_m'] - rm) / 1e3:7.1f}km beta={c['beta_deg_epoch']:7.2f} "
                  f"earthview={c['earth_view_deg_epoch']:6.1f} D={c['duration_s'] / 3600:5.1f}h")
    print(f"Sun-Earth separation seen from Moon at epoch: {sep:.1f} deg")


if __name__ == "__main__":
    main()
