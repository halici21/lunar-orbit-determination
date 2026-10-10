"""PHASE 17-HARDEN - R1O surrogate / Earth-LOS rows under the visibility contract.

ANALYSIS ONLY, NO NEW SCIENCE.  FXR requalified R1O's surrogate rows on the
canonical Earth fixture, but with the elevation-only rule: lunar occultation was
never tested (HARDEN-D04).  This script:

1. Rebuilds FXR's canonical R1O surrogate / Earth-LOS / bridge rows with the
   named legacy rule and checks that they reproduce the published FXR numbers
   BITWISE.  This proves that the new row-rule plumbing changed nothing in the
   legacy path.
2. Rebuilds the same surrogate rows under the production visibility contract
   (elevation AND occultation), and reports how many rows are removed and the
   corrected metrics.

The production ΔDOR numbers themselves are Phase 17-R1O-D2's (range 0.091508,
combined 0.172351, sigma_K/K 7.46%); this script restates them from D2's artifact
and does not recompute them.

Output: artifacts/phase17_harden_visibility_requalification.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
ARTIFACTS = ROOT / "artifacts"

import phase17_r1m_core as M  # noqa: E402

DSN3 = ("Goldstone DSN", "Madrid DSN", "Canberra DSN")
DDOR_PAIR = ("Goldstone DSN", "Canberra DSN")
KEYS = ("n_obs", "orthogonal_fraction", "conditional_k_information", "fractional_sigma_k")


def _m(d: dict) -> dict:
    return {k: d[k] for k in KEYS}


def main() -> None:
    import od_gravity_covariance_campaign as C  # noqa: N811
    from lunar_od.spice_loader import load_spice_kernels
    from phase17_r1o_core import build_ddor_arc, combined_metrics
    from phase17_r1o_r_requalification import bridge_reduced_rows, earth_los_rows

    warnings.simplefilter("ignore", M.SupersededVisibilityWarning)
    load_spice_kernels(None, clear=True)
    fx = M.EARTH_FIXTURE_CANONICAL
    et0, t_orbit = M.campaign_epoch()
    w15 = M.build_range_arc(0.0, 15.0 * t_orbit, label="W15_range", station_filter=DSN3,
                            cadence_s=90.0, earth_fixture=fx)
    nom, rb = w15.nom48, (w15.h_x0, w15.h_k, w15.w)
    fxr = json.loads((ARTIFACTS / "phase17_fxr_requalification.json").read_text(encoding="utf-8"))
    pub = fxr["tasks"]["r1o"]["CANONICAL"]

    out = {"phase": "17-HARDEN", "fixture": "CANONICAL", "legacy": {}, "visible": {},
           "legacy_reproduces_fxr_bitwise": {}}
    for rule, tag in ((M.DDOR_ROW_RULE_LEGACY_ELEVATION_ONLY, "legacy"),
                      (M.DDOR_ROW_RULE_VISIBLE, "visible")):
        for nrad in (1.0, 5.0):
            d = build_ddor_arc(nom, w15.t_grid, et0, DDOR_PAIR, sigma_angle_rad=nrad * 1e-9,
                               label="ddor", earth_fixture=fx, row_rule=rule)
            key = "ddor_surrogate_%gnrad" % nrad
            out[tag][key] = _m(combined_metrics([rb, (d.h_x0, d.h_k, d.w)]))
            out[tag][key]["n_ddor_rows"] = int(d.h_k.size)
        ex, ek, _ = earth_los_rows(nom, w15.t_grid, earth_fixture=fx, row_rule=rule)
        w_e = np.full(ek.size, 1.0 / (1.0 * 1e-6) ** 2)
        out[tag]["earth_los_1urad"] = _m(combined_metrics([rb, (ex, ek, w_e)]))
        out[tag]["earth_los_1urad"]["n_los_rows"] = int(ek.size)
    by = {s.name: s for s in C.STATIONS}
    bx, bk = bridge_reduced_rows(nom, w15.t_grid, et0, by[DDOR_PAIR[0]], by[DDOR_PAIR[1]],
                                 earth_fixture=fx)
    w_b = np.full(bk.size, 1.0 / (5.0 * 1e-9) ** 2 * (1e7 / 299792458.0) ** 2)
    out["legacy"]["bridge_surrogate_5nrad"] = _m(combined_metrics([rb, (bx, bk, w_b)]))
    out["bridge_note"] = ("bridge_reduced_rows is HISTORICAL ONLY (Earth frozen at t=0, "
                          "elevation-only); it has no visibility-correct variant by design")

    for key in ("ddor_surrogate_1nrad", "ddor_surrogate_5nrad", "earth_los_1urad",
                "bridge_surrogate_5nrad"):
        out["legacy_reproduces_fxr_bitwise"][key] = all(
            out["legacy"][key][k] == pub[key][k] for k in KEYS)
    out["range_only"] = _m(pub["range_only"])

    d2 = json.loads((ARTIFACTS / "phase17_r1od2_summary.json").read_text(encoding="utf-8"))
    canon = next(r for r in d2["primary_population"] if r["case"] == "CANON" and r["pair"] == "GC")
    out["production_ddor_authoritative"] = {
        "source": "artifacts/phase17_r1od2_summary.json primary_population CANON/GC",
        "range_only_f_perp": canon["range_f"], "combined_f_perp": canon["comb_f"],
        "delta_f_perp": canon["delta"], "sigma_k_over_k": canon["sigma_frac"],
        "historical_intermediate_f_perp": pub["ddor_surrogate_5nrad"]["orthogonal_fraction"],
        "historical_intermediate_production_f_perp":
            fxr["tasks"]["r1od_MEASUREMENT_EPOCH"]["CANONICAL"]["range_ddor_5nrad"]["orthogonal_fraction"],
        "historical_intermediate_label": "EPOCH_CORRECTED_BUT_ELEVATION_ONLY_HISTORICAL_INTERMEDIATE",
    }
    # surrogate (visible) vs production (visible): same rows, different observable model
    out["visible_surrogate_minus_production_f_perp"] = (
        out["visible"]["ddor_surrogate_5nrad"]["orthogonal_fraction"] - canon["comb_f"])
    (ARTIFACTS / "phase17_harden_visibility_requalification.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    for tag in ("legacy", "visible"):
        for key, v in out[tag].items():
            print("%-8s %-24s n=%5d f_perp=%.6f sigmaK/K=%.4f%%"
                  % (tag, key, v["n_obs"], v["orthogonal_fraction"], 100 * v["fractional_sigma_k"]))
    print("legacy bitwise == FXR:", out["legacy_reproduces_fxr_bitwise"])


if __name__ == "__main__":
    main()
