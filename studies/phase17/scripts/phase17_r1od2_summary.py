"""PHASE 17-R1O-D2 s36-s41 - robustness summary and predeclared outcome rules.

Reads only the D2 artifacts; applies the grid's predeclared classification and
outcome rules mechanically.  Writes studies/phase17/results/phase17_r1od2_summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ART = Path(__file__).resolve().parent.parent / "studies" / "phase17" / "results"
CLASSES = ("STRONG_DIRECTION_GAIN", "MODEST_DIRECTION_GAIN", "MAGNITUDE_ONLY", "DIRECTION_LOSS")


def _stats(v):
    v = np.asarray(v, float)
    return dict(n=int(v.size), min=float(v.min()), median=float(np.median(v)), max=float(v.max()))


def main() -> None:
    R = json.loads((ART / "phase17_r1od2_results.json").read_text())["geometries"]
    fd = json.loads((ART / "phase17_r1od2_fd.json").read_text())
    kref = json.loads((ART / "phase17_r1od2_fd_k_refinement.json").read_text())
    fid_path = ART / "phase17_r1od2_fidelity.json"
    fid = json.loads(fid_path.read_text()) if fid_path.exists() else None

    prim = []
    for cid, r in R.items():
        for p in ("GC", "GM"):
            pr = r["pairs"][p]
            if "dense_angular" in pr:
                d5 = pr["dense_angular"]["5"]
                ops = pr["ops"]["OPS_EVERY_WINDOW"]
                prim.append(dict(case=cid, family=r["family"], pair=p, has_rows=True,
                                 range_f=r["range_only"]["f_perp"], comb_f=d5["f_perp"],
                                 delta=d5["delta_f_perp"], cls=d5["relative_class"],
                                 abs_cls=d5["absolute_class"], sigma_frac=d5["sigma_k_frac"],
                                 ops_delta=ops.get("delta_f_perp"), ops_cls=ops.get("relative_class"),
                                 delay_delta=pr["dense_delay"]["0.048"]["delta_f_perp"],
                                 delay_cls=pr["dense_delay"]["0.048"]["relative_class"],
                                 surv_dense=pr["bias"]["DENSE_5NRAD"]["survives_calibration_freedom"],
                                 surv_ops=pr["bias"].get("OPS_EVERY_WINDOW", {}).get("survives_calibration_freedom", False),
                                 b3_dense=pr["bias"]["DENSE_5NRAD"]["B3"]["delta_f_perp"],
                                 b4_dense=pr["bias"]["DENSE_5NRAD"]["B4"]["delta_f_perp"],
                                 b4_ops=pr["bias"].get("OPS_EVERY_WINDOW", {}).get("B4", {}).get("delta_f_perp"),
                                 weakest_k=d5["weakest_mode_k_component"],
                                 design_cond=d5["design_condition"]))
            else:
                prim.append(dict(case=cid, family=r["family"], pair=p, has_rows=False,
                                 range_f=r["range_only"]["f_perp"], comb_f=r["range_only"]["f_perp"],
                                 delta=0.0, cls="MAGNITUDE_ONLY", abs_cls=None, sigma_frac=None,
                                 ops_delta=None, ops_cls="MAGNITUDE_ONLY", surv_dense=False,
                                 surv_ops=False, note="NO_DUAL_VISIBILITY"))
    n = len(prim)
    counts = {c: sum(1 for x in prim if x["cls"] == c) for c in CLASSES}
    data = [x for x in prim if x["has_rows"]]
    strong_frac = counts["STRONG_DIRECTION_GAIN"] / n
    loss_frac = counts["DIRECTION_LOSS"] / n
    surv_dense = sum(1 for x in prim if x["surv_dense"]) / n
    surv_ops = sum(1 for x in prim if x["surv_ops"]) / n
    ops_modest = sum(1 for x in prim if x["ops_cls"] in ("STRONG_DIRECTION_GAIN", "MODEST_DIRECTION_GAIN")) / n
    # predeclared outcome rules, in order D, C, B, else A
    a_geometry = strong_frac >= 0.75 and loss_frac <= 0.10
    if strong_frac < 0.25:
        outcome = "D"
    elif a_geometry and (surv_dense < 0.5 or surv_ops < 0.5):
        outcome = "C"
    elif strong_frac <= 0.75:
        outcome = "B"
    elif a_geometry and ops_modest >= 0.5:
        outcome = "A"
    else:
        outcome = "B"
    primary = {"A": "PRODUCTION_DDOR_ROBUSTLY_ADDS_COMPLEMENTARY_K_SRP_INFORMATION",
               "B": "DDOR_K_INFORMATION_GAIN_IS_STRONGLY_GEOMETRY_DEPENDENT",
               "C": "DDOR_DIRECTION_GAIN_IS_FORMALLY_STRONG_BUT_CALIBRATION_FRAGILE",
               "D": "CANONICAL_DDOR_DIRECTION_GAIN_DOES_NOT_GENERALIZE"}[outcome]

    def breakdown(key):
        out = {}
        for x in prim:
            b = out.setdefault(x[key], {c: 0 for c in CLASSES})
            b[x["cls"]] += 1
        return out

    strong_cases = [(x["case"], x["pair"]) for x in prim if x["cls"] == "STRONG_DIRECTION_GAIN"]
    canon_epoch = {"CANON", "O1_LOW_BETA_ECLIPSE_RICH", "O2_FULL_SUN_HIGH_BETA", "O3_HIGH_ALTITUDE_500KM",
                   "O4_POLAR", "O5_FAVOURABLE_GEO_MAX_FPERP"}
    # pair medians
    by_pair = {p: _stats([x["delta"] for x in data if x["pair"] == p]) for p in ("GC", "GM")}
    # quasar (CANON)
    q = R["CANON"]["pairs"]
    qsum = {}
    for p in ("GC", "GM"):
        for real in ("DENSE_5NRAD", "OPS_EVERY_WINDOW"):
            rows = [x for x in q[p]["quasar_systematics"] if x["realization"] == real]
            sl = {s: float(np.median([x["k_bias_rss_by_group_frac"]["station_location"]
                                      for x in rows if x["sep_rad"] == s])) for s in sorted({x["sep_rad"] for x in rows})}
            pa_spread = {s: float(np.ptp([x["k_bias_rss_total_frac"] for x in rows if x["sep_rad"] == s]))
                         for s in sorted({x["sep_rad"] for x in rows})}
            qsum["%s_%s" % (p, real)] = dict(
                formal_sigma_k_frac=rows[0]["formal_sigma_k_frac"],
                total_bias_frac_range=[min(x["k_bias_rss_total_frac"] for x in rows),
                                       max(x["k_bias_rss_total_frac"] for x in rows)],
                station_location_bias_frac_median_by_sep=sl,
                pa_spread_of_total_bias_frac_by_sep=pa_spread,
                quasar_coordinate_share_at_0p1=float(np.median([
                    x["k_bias_rss_by_group_frac"]["quasar_coordinate"] / x["k_bias_rss_total_frac"]
                    for x in rows if abs(x["sep_rad"] - 0.1) < 1e-12])))
    # surrogate agreement
    surr = [abs(R[c]["pairs"][p]["surrogate"]["f_perp_combined_surrogate_weights"]
                - R[c]["pairs"][p]["surrogate"]["f_perp_combined_production"])
            for c in R for p in ("GC", "GM") if "surrogate" in R[c]["pairs"][p]]
    k_ref_best = max(min(v["rtol_1e-14"]["rel_err"]) for v in kref.values())
    d0 = R["CANON"]["elevation_only_historical"]["GC"]
    rows_rule_effect = d0["dense_angular"]["5"]["f_perp"] - R["CANON"]["pairs"]["GC"]["dense_angular"]["5"]["f_perp"]
    summary = dict(
        primary_population=prim, population_size=n, class_counts=counts,
        strong_fraction=strong_frac, direction_loss_fraction=loss_frac,
        survives_b3_fraction_dense=surv_dense, survives_b3_fraction_ops=surv_ops,
        ops_at_least_modest_fraction=ops_modest,
        combined_f_perp_stats_cases_with_data=_stats([x["comb_f"] for x in data]),
        delta_f_perp_stats_cases_with_data=_stats([x["delta"] for x in data]),
        delta_f_perp_stats_all_primary=_stats([x["delta"] for x in prim]),
        breakdown_by_pair=breakdown("pair"), breakdown_by_family=breakdown("family"),
        delta_by_pair=by_pair,
        strong_cases=strong_cases,
        all_strong_cases_are_GC_at_canonical_epoch=all(c in canon_epoch and p == "GC" for c, p in strong_cases),
        gc_at_canonical_epoch_all_strong=all(x["cls"] == "STRONG_DIRECTION_GAIN" for x in prim
                                             if x["pair"] == "GC" and x["case"] in canon_epoch),
        absolute_class_counts={c: sum(1 for x in data if x["abs_cls"] == c)
                               for c in ("STRONG_K_SEPARATION", "MODERATE_K_SEPARATION", "WEAK_K_SEPARATION")},
        ops_class_counts={c: sum(1 for x in prim if x["ops_cls"] == c) for c in CLASSES},
        dense_delay_0048_class_counts={c: sum(1 for x in data if x["delay_cls"] == c) for c in CLASSES},
        b4_dense_delta_stats=_stats([x["b4_dense"] for x in data]),
        outcome_rule=outcome, PRIMARY_CLASS=primary,
        quasar=qsum,
        surrogate_max_abs_diff=float(max(surr)),
        d0=dict(range=R["CANON"]["range_only"]["f_perp"], combined_5nrad=d0["dense_angular"]["5"]["f_perp"],
                sigma_frac=d0["dense_angular"]["5"]["sigma_k_frac"],
                one_nrad=d0["dense_angular"]["1"]["f_perp"], hundred_nrad=d0["dense_angular"]["100"]["f_perp"],
                surrogate=d0["surrogate"]["f_perp_combined_surrogate_weights"], n_rows=d0["n_rows"],
                row_rule_effect_on_combined_f_perp=rows_rule_effect),
        fd=dict(state_gate=fd["gates"]["DDOR_STATE_JACOBIAN_FD_GATE"],
                k_gate_predeclared_oracle=fd["gates"]["DDOR_K_JACOBIAN_FD_GATE"],
                k_gate_refined_oracle="PASS" if k_ref_best <= 1e-3 else "FAIL",
                max_state_relerr=fd["gates"]["MAX_DDOR_STATE_FD_RELERR"],
                max_k_relerr_predeclared_oracle=fd["gates"]["MAX_DDOR_K_FD_RELERR"],
                max_k_relerr_refined_oracle=k_ref_best,
                max_f_perp_materiality=fd["gates"]["MAX_F_PERP_MATERIALITY"],
                quasar_invariance=fd["quasar_invariance"]),
        fidelity=None if fid is None else dict(effect=fid["PRODUCTION_FIDELITY_DYNAMICS_EFFECT"],
                                               weakest=fid["weakest_rule_selection"]),
    )
    (ART / "phase17_r1od2_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    for k in ("class_counts", "strong_fraction", "survives_b3_fraction_dense", "survives_b3_fraction_ops",
              "ops_at_least_modest_fraction", "combined_f_perp_stats_cases_with_data",
              "delta_f_perp_stats_cases_with_data", "breakdown_by_pair", "breakdown_by_family",
              "delta_by_pair", "strong_cases", "all_strong_cases_are_GC_at_canonical_epoch",
              "gc_at_canonical_epoch_all_strong", "absolute_class_counts", "ops_class_counts",
              "dense_delay_0048_class_counts", "b4_dense_delta_stats", "outcome_rule", "PRIMARY_CLASS",
              "surrogate_max_abs_diff", "d0", "fd", "fidelity"):
        print("%-44s %s" % (k, summary[k]))
    print("quasar:", json.dumps(qsum, indent=1)[:1500])


if __name__ == "__main__":
    main()
