"""PHASE 17-F1 - pre/post comparison of the materiality runs (no computation of physics).

Reads studies/phase17/results/phase17_f1_science_materiality.json ("pre", "post" from
phase17_f1_materiality.py), applies the predeclared rules of
studies/phase17/configs/phase17_f1_predeclaration.json and writes the "comparison" block.
"""
from __future__ import annotations

import json
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parents[3] / "studies" / "phase17" / "results"
KEYS = ("f_perp", "theta_k_deg", "conditional_k_information", "sigma_k", "sigma_k_frac", "k_column_norm",
        "weakest_mode_k_component")


def main() -> None:
    path = ARTIFACTS / "phase17_f1_science_materiality.json"
    store = json.loads(path.read_text(encoding="utf-8"))
    decl = json.loads((ARTIFACTS.parent / "configs" / "phase17_f1_predeclaration.json").read_text(encoding="utf-8"))
    pre, post = store["pre"], store["post"]
    cells, class_changes = [], []
    for case, p in pre["d2_primary"].items():
        q = post["d2_primary"][case]
        for pair in ("GC", "GM"):
            a, b = p["pairs"][pair], q["pairs"][pair]
            if not a.get("n_rows"):
                cells.append(dict(case=case, pair=pair, n_rows=0, note="no dual-visible rows (pre and post)"))
                continue
            row = dict(case=case, pair=pair, n_rows=a["n_rows"],
                       range_f_pre=p["range_only"]["f_perp"], range_f_post=q["range_only"]["f_perp"],
                       comb_f_pre=a["f_perp"], comb_f_post=b["f_perp"],
                       delta_pre=a["delta_f_perp"], delta_post=b["delta_f_perp"],
                       rel_class_pre=a["relative_class"], rel_class_post=b["relative_class"],
                       abs_class_pre=a["absolute_class"], abs_class_post=b["absolute_class"])
            if (row["rel_class_pre"], row["abs_class_pre"]) != (row["rel_class_post"], row["abs_class_post"]):
                class_changes.append("%s/%s" % (case, pair))
            cells.append(row)
    canon_pre, canon_post = pre["d2_primary"]["CANON"], post["d2_primary"]["CANON"]
    canonical = {}
    for block, a, b in (("range_only", canon_pre["range_only"], canon_post["range_only"]),
                        ("range_plus_ddor_GC_5nrad", canon_pre["pairs"]["GC"], canon_post["pairs"]["GC"])):
        canonical[block] = {k: dict(pre=a[k], post=b[k], rel_change=(b[k] - a[k]) / a[k] if a[k] else None)
                            for k in KEYS}
        canonical[block]["design_singular_values"] = dict(pre=a["design_singular_values"],
                                                          post=b["design_singular_values"])
    geo_rule = decl["geo_spot_check"]
    geo, escalate = {}, False
    for cid, g_pre_decl in geo_rule["pre_f1_values"].items():
        a, b = pre["geo_spot"][cid], post["geo_spot"][cid]
        dd = abs(b["d"] - a["d"])
        df = abs(b["f_perp"] - a["f_perp"])
        trig = (a["class_full"] != b["class_full"] or dd >= 0.5 * g_pre_decl["threshold_margin"] or df >= 0.01)
        escalate |= trig
        geo[cid] = dict(f_pre=a["f_perp"], f_post=b["f_perp"], d_pre=a["d"], d_post=b["d"],
                        class_pre=a["class_full"], class_post=b["class_full"],
                        threshold_margin_pre=g_pre_decl["threshold_margin"], abs_delta_d=dd, abs_delta_f=df,
                        escalation_triggered=trig)
    numbers_moved = any(abs(canonical[b]["f_perp"]["post"] - canonical[b]["f_perp"]["pre"]) >= 5e-7
                        for b in canonical)
    label = ("SCIENTIFIC_CLASSIFICATION_CHANGED" if class_changes else
             "NUMERICAL_BASELINES_UPDATED_CLASSES_PRESERVED" if numbers_moved else
             "NO_MATERIAL_RECLASSIFICATION")
    store["comparison"] = dict(d2_primary_cells=cells, d2_cells_requalified=len(cells),
                               d2_class_changes=class_changes, canonical=canonical, geo_spot=geo,
                               GEO_ESCALATED_TO_FULL_GRID=escalate,
                               F1_SCIENCE_MATERIALITY_CLASS=label,
                               authoritative_precision_note="studies/phase17/reports/phase17_authoritative_status.md quotes f_perp "
                                                            "to 6 decimals; any move >= 5e-7 changes a quoted digit")
    path.write_text(json.dumps(store, indent=1, default=float), encoding="utf-8")
    print("D2 cells", len(cells), "class changes", class_changes)
    for b in canonical:
        print(b, {k: "%.6g -> %.6g" % (v["pre"], v["post"]) for k, v in canonical[b].items()
                  if k != "design_singular_values"})
    for cid, g in geo.items():
        print("GEO", cid, "%.6f -> %.6f  %s -> %s  |dd| %.1e (0.5 margin %.1e)" % (
            g["f_pre"], g["f_post"], g["class_pre"], g["class_post"], g["abs_delta_d"],
            0.5 * g["threshold_margin_pre"]))
    print("GEO escalation:", escalate, "| label:", label)


if __name__ == "__main__":
    main()
