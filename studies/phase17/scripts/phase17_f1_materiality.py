"""PHASE 17-F1 - science-materiality requalification of the SRP variational correction.

REQUALIFICATION ONLY, NO NEW SCIENCE.  The same script runs twice: once on the
entering dynamics (``--tag pre``) and once after the SRP position gradient is
added to the variational equations (``--tag post``).  Nothing but
``src/lunar_od/dynamics.py`` / ``src/lunar_od/srp.py`` differs between the two runs, so
every difference is attributable to the variational correction.

Population (predeclared in studies/phase17/configs/phase17_f1_predeclaration.json):
  * D2 PRIMARY: the 9 predeclared D2 geometries x {Goldstone-Canberra,
    Goldstone-Madrid}, production visibility rows, dense angular 5 nrad,
    classes by the D2 rules (relative_gain_class, absolute_class);
  * canonical detail: range-only and GC 5 nrad full metric set;
  * GEO spot check: the 5 frozen cases, range-only, GEO classes against the
    post-run G0_consistent f_ref.

Output: studies/phase17/results/phase17_f1_science_materiality.json, key = tag.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ARTIFACTS = HERE.parents[2] / "studies" / "phase17" / "results"
OUT = ARTIFACTS / "phase17_f1_science_materiality.json"

D2_KEYS = ("n_obs", "f_perp", "theta_k_deg", "conditional_k_information", "i_kk", "k_column_norm",
           "sigma_k", "sigma_k_frac", "design_singular_values", "design_condition",
           "weakest_mode_k_component", "max_abs_k_state_correlation")


def _init() -> None:
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels(None, clear=True)
    warnings.simplefilter("ignore")
    import phase17_r1od2_core as D2
    D2.install_legacy_earth_guard()


def d2_geometry(g: dict) -> dict:
    import phase17_r1od2_core as D2
    scn = D2.Scenario(g["case_id"], g["family"], float(g["et0"]), tuple(g["x0"]),
                      float(g["duration_s"]), float(g["k_srp"]), g["shadow_model"])
    nom = D2.propagate_nom48(scn)
    rb = D2.build_range_block(scn, nom)
    m_range = D2.info_metrics([(rb.h_x0, rb.h_k, rb.w)])
    out = dict(case=scn.case_id, range_only={k: m_range[k] for k in D2_KEYS}, pairs={},
               nom48_state_end=nom[-1, :6].tolist(), s_k_end=nom[-1, 42:48].tolist())
    for pair in ("GC", "GM"):
        rows = D2.build_ddor_rows(scn, rb, pair, D2.ROW_RULE_PRODUCTION_VISIBILITY)
        if rows.n == 0:
            out["pairs"][pair] = dict(n_rows=0)
            continue
        w5 = D2.ddor_weights(rows, D2.SEMANTICS_ANGULAR, 5e-9)
        m = D2.info_metrics([(rb.h_x0, rb.h_k, rb.w), (rows.h_x0, rows.h_k, w5)])
        d = m["f_perp"] - m_range["f_perp"]
        out["pairs"][pair] = dict(n_rows=int(rows.n), delta_f_perp=d,
                                  relative_class=D2.relative_gain_class(d),
                                  absolute_class=D2.absolute_class(m["f_perp"]),
                                  **{k: m[k] for k in D2_KEYS})
    return out


def geo_case(case: dict) -> dict:
    import phase17_geo_core as G
    arc = G.build_geo_range_arc(np.asarray(case["x0"], float), case["epoch_et"], case["duration_s"],
                                label=case["id"], station_names=tuple(case["stations"]),
                                k_srp=case["k_srp"], shadow_model=case["shadow_model"])
    m = G.information_metrics(arc.h_x0, arc.h_k, arc.w, k_truth=case["k_srp"])
    return dict(id=case["id"], **{k: v for k, v in m.items() if not isinstance(v, list)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=("pre", "post"))
    args = ap.parse_args()
    import phase17_geo_campaign as GC
    decl = json.loads((ARTIFACTS.parent / "configs" / "phase17_f1_predeclaration.json").read_text(encoding="utf-8"))
    d2grid = json.loads((ARTIFACTS.parent / "configs" / "phase17_r1od2_predeclared_grid.json").read_text(encoding="utf-8"))
    geogrid = json.loads((ARTIFACTS.parent / "configs" / "phase17_geo_predeclared_grid.json").read_text(encoding="utf-8"))
    sel = decl["geo_spot_check"]["selected"]
    geo_cases = [c for c in geogrid["cases"] if c["id"] in set(sel.values())]
    assert len(geo_cases) == len(set(sel.values())), "a frozen GEO case is missing from the grid"

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(d2grid["geometries"]) + len(geo_cases),
                             initializer=_init) as ex:
        f_d2 = [ex.submit(d2_geometry, g) for g in d2grid["geometries"]]
        f_geo = [ex.submit(geo_case, c) for c in geo_cases]
        d2 = [f.result() for f in f_d2]
        geo = [f.result() for f in f_geo]
    f_ref = next(g for g in geo if g["id"] == "G0_consistent")["f_perp"]
    for g in geo:
        g["d"] = g["f_perp"] - f_ref
        g["class_full"] = GC.classify(g["d"], g["f_perp"])
    import subprocess
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=HERE.parents[2]).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short", "lunar_od"], capture_output=True, text=True,
                           cwd=HERE.parents[2]).stdout.strip()
    result = dict(head=head, lunar_od_uncommitted=dirty, elapsed_s=time.time() - t0,
                  d2_primary={r["case"]: r for r in d2}, geo_spot={g["id"]: g for g in geo})
    store = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    store[args.tag] = result
    OUT.write_text(json.dumps(store, indent=1, default=float), encoding="utf-8")
    for r in d2:
        for pair, p in r["pairs"].items():
            if p.get("n_rows"):
                print("%-28s %s range=%.6f comb=%.6f d=%+.6f %s %s sig=%.4f%%" % (
                    r["case"], pair, r["range_only"]["f_perp"], p["f_perp"], p["delta_f_perp"],
                    p["relative_class"], p["absolute_class"], 100 * p["sigma_k_frac"]))
    for g in geo:
        print("GEO %-16s f=%.6f d=%+.6f %s" % (g["id"], g["f_perp"], g["d"], g["class_full"]))
    print("elapsed %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
