"""PHASE 17-FXR s35 - how were GEO's G5 K-magnitude cases generated?

ANALYSIS SPACE ONLY.  Limited audit, not a new campaign.

Question: was the nominal trajectory re-propagated for each K truth value, or
was one design matrix reused with only the sigma_K/K denominator changed?

Evidence used:
  1. the code path (phase17_geo_grid -> phase17_geo_campaign.run_case ->
     phase17_geo_core.build_geo_range_arc(k_srp=...) -> propagate48(k_srp))
  2. the committed G5 artifact (phase17_geo_k_magnitude_control.csv)
  3. a re-run of the two extreme cases through that same path, checking that
     the nominal trajectories and K columns differ (they must, if propagated)
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ARTIFACTS = HERE.parent / "artifacts"


def main() -> None:
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels(None, clear=True)
    import phase17_geo_core as G
    from phase17_r1m_core import campaign_epoch, campaign_initial_state

    rows = list(csv.DictReader((ARTIFACTS / "phase17_geo_k_magnitude_control.csv").open()))
    g0 = [r for r in rows if r["id"] == "G0_consistent"][0]
    g5 = [r for r in rows if r["stage"] == "G5"]
    table = []
    for r in [g0] + g5:
        k = float(r["id"].split("_k")[1]) if r["stage"] == "G5" else 0.01
        table.append(dict(id=r["id"], k=k, f_perp=float(r["f_perp"]),
                          k_column_norm=float(r["k_column_norm"]),
                          sigma_k_abs=float(r["sigma_k_frac"]) * k,
                          alt_mean_km=float(r["alt_mean_km"])))
    f = np.array([t["f_perp"] for t in table])
    s = np.array([t["sigma_k_abs"] for t in table])
    b = np.array([t["k_column_norm"] for t in table])

    et0, t_orbit = campaign_epoch()
    x0 = campaign_initial_state()
    arcs = {k: G.build_geo_range_arc(x0, et0, 15.0 * t_orbit, label="g5_%g" % k, k_srp=k)
            for k in (0.0025, 0.04)}
    sha = {k: hashlib.sha256(np.ascontiguousarray(a.nom48)).hexdigest() for k, a in arcs.items()}
    m = {k: G.information_metrics(a.h_x0, a.h_k, a.w, k_truth=k) for k, a in arcs.items()}
    rerun_matches_artifact = all(
        m[k]["f_perp"] == [t for t in table if t["k"] == k][0]["f_perp"] for k in m)
    out = dict(
        code_path=["phase17_geo_grid: case(f'G5_k{k}', ..., k_srp=k)",
                   "phase17_geo_campaign.run_case: G.build_geo_range_arc(..., k_srp=case['k_srp'])",
                   "phase17_geo_core.build_geo_range_arc: nom48 = propagate48(t_grid, x0, et0, k_srp)",
                   "phase17_geo_campaign: G.information_metrics(..., k_truth=case['k_srp'])"],
        artifact_rows=table,
        f_perp_min=float(f.min()), f_perp_max=float(f.max()),
        f_perp_rel_spread=float((f.max() - f.min()) / f.mean()),
        sigma_k_abs_rel_spread=float((s.max() - s.min()) / s.mean()),
        k_column_norm_rel_spread=float((b.max() - b.min()) / b.mean()),
        rerun_nom48_differs_between_k=sha[0.0025] != sha[0.04],
        rerun_h_k_differs_between_k=not np.array_equal(arcs[0.0025].h_k, arcs[0.04].h_k),
        rerun_matches_artifact_bitwise=rerun_matches_artifact,
        implementation_class="PHYSICAL_K_TRUTH_TRAJECTORY_SWEEP",
        report_wording_note=("GEO s28 prints f_perp = 0.091508 and ||b_K|| = 4302 for every K and calls "
                             "the invariance exact; the committed values differ in the 5th-6th "
                             "significant digit because each K has its own propagated nominal "
                             "trajectory. The linearity interpretation holds to ~1e-4 relative, "
                             "not exactly."),
    )
    for k, v in out.items():
        if k != "artifact_rows":
            print("  %-34s %s" % (k, v))
    for t in table:
        print("  %-16s K=%-7g f_perp=%.9f ||b_K||=%.4f sigma_K=%.7e alt=%.5f km"
              % (t["id"], t["k"], t["f_perp"], t["k_column_norm"], t["sigma_k_abs"],
                 t["alt_mean_km"]))
    (ARTIFACTS / "phase17_fxr_g5_audit.json").write_text(json.dumps(out, indent=1, default=float))
    print("  wrote artifacts/phase17_fxr_g5_audit.json")


if __name__ == "__main__":
    main()
