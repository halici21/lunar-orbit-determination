"""PHASE 17-R1O-D2 s32/s33 - production-fidelity dynamics spot check.

ANALYSIS SPACE ONLY.  No new force model: the existing production propagator
(`propagate_state_with_k_sensitivity`) with the capabilities it already has --
GL1800F truncated 50x50 in MOON_PA_DE440 (the campaign's own loaded model, with
the matched analytic Pines STM opt-in), Earth and Sun point-mass third-body
gravity from SPICE at the scenario epoch, and the same cannonball SRP.

The measurement side is unchanged (canonical Earth, production range and ΔDOR,
PRODUCTION_VISIBILITY rows).  Only the dynamics -- hence the truth, the STM and
dx/dK, i.e. the state-sensitivity basis K is compared against -- change.

Cases (predeclared): CANON GC, O2_FULL_SUN_HIGH_BETA GC, and the primary-
population case with the smallest dense-5-nrad delta (selected by rule from
studies/phase17/results/phase17_r1od2_results.json).
"""
from __future__ import annotations

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
SIGMA_OPS_S = 0.048e-9
NMAX = 50


def _init() -> None:
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels(None, clear=True)
    warnings.simplefilter("ignore")
    import phase17_r1od2_core as D2
    D2.install_legacy_earth_guard()


def propagate_fidelity(scn) -> np.ndarray:
    import od_gravity_covariance_campaign as C  # noqa: N811
    import phase17_geo_core as G
    from lunar_od.constants import MU_EARTH_M3S2, MU_MOON_M3S2, MU_SUN_M3S2
    from lunar_od.dynamics import propagate_state_with_k_sensitivity
    from lunar_od.srp import SRPOptions

    t_grid = scn.t_grid
    return propagate_state_with_k_sensitivity(
        t_grid, scn.x0_array, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
        lambda t: G.earth_state_mci(scn.et0 + float(t))[:3],
        lambda t: G.sun_mci(scn.et0 + float(t)),
        srp=SRPOptions(k_srp_m2_per_kg=scn.k_srp, shadow_model=scn.shadow_model),
        rtol=1e-12, atol=1e-13, j2_moon=0.0,
        harmonic_model=C._M[NMAX], harmonic_epoch_et0=scn.et0, harmonic_stm_opt_in=True)


def run_case(spec) -> dict:
    import phase17_r1od2_core as D2

    t0 = time.time()
    g, pair = spec
    scn = D2.Scenario(g["case_id"], g["family"], float(g["et0"]), tuple(g["x0"]),
                      float(g["duration_s"]), float(g["k_srp"]), g["shadow_model"])
    out = dict(case_id=scn.case_id, pair=pair)
    for model, prop in (("SIMPLIFIED", D2.propagate_nom48), ("FIDELITY_GL50_EARTH_SUN", propagate_fidelity)):
        nom = prop(scn)
        rb = D2.build_range_block(scn, nom)
        rows = D2.build_ddor_rows(scn, rb, pair, D2.ROW_RULE_PRODUCTION_VISIBILITY)
        rblock = (rb.h_x0, rb.h_k, rb.w)
        m_r = D2.info_metrics([rblock])
        res = dict(range_f_perp=m_r["f_perp"], range_sigma_k_frac=m_r["sigma_k_frac"],
                   range_obs=rb.n_obs, n_rows=rows.n, nom48_final_pos=nom[-1, :3].tolist(),
                   k_sens_final_norm=float(np.linalg.norm(nom[-1, 42:45])))
        if rows.n:
            w5 = D2.ddor_weights(rows, D2.SEMANTICS_ANGULAR, 5e-9)
            m_c = D2.info_metrics([rblock, (rows.h_x0, rows.h_k, w5)])
            d = m_c["f_perp"] - m_r["f_perp"]
            res.update(dense_f_perp=m_c["f_perp"], dense_delta=d,
                       dense_class=D2.relative_gain_class(d), dense_sigma_k_frac=m_c["sigma_k_frac"],
                       absolute_class=D2.absolute_class(m_c["f_perp"]),
                       weakest_mode_k_component=m_c["weakest_mode_k_component"],
                       design_condition=m_c["design_condition"])
            sel = D2.ops_selection(rows, "OPS_EVERY_WINDOW", 0.5 * float(rb.t_grid[-1]))
            if sel.size:
                m_o = D2.info_metrics([rblock, (rows.h_x0[sel], rows.h_k[sel],
                                                np.full(sel.size, 1 / SIGMA_OPS_S ** 2))])
                res.update(ops_delta=m_o["f_perp"] - m_r["f_perp"],
                           ops_class=D2.relative_gain_class(m_o["f_perp"] - m_r["f_perp"]))
            sub = D2.subset(rows, np.arange(rows.n))
            nuis = D2.session_nuisance_columns(rb.n_obs, sub)
            m_b4 = D2.info_metrics([rblock, (rows.h_x0, rows.h_k, w5)], nuisance=nuis,
                                   nuisance_prior_sigma=0.033e-9)
            res.update(b4_delta=m_b4["f_perp"] - m_r["f_perp"])
        out[model] = res
        if model == "SIMPLIFIED":
            simple_final = nom[:, :3]
        else:
            out["max_position_difference_m"] = float(np.max(np.linalg.norm(nom[:, :3] - simple_final, axis=1)))
    s, f = out["SIMPLIFIED"], out["FIDELITY_GL50_EARTH_SUN"]
    out["class_preserved"] = s.get("dense_class") == f.get("dense_class")
    out["margin_to_nearest_threshold"] = float(min(abs(f.get("dense_delta", 0) - t)
                                                   for t in (0.05, 0.02, -0.02)))
    out["elapsed_s"] = time.time() - t0
    return out


def main() -> None:
    grid = json.loads((ARTIFACTS.parent / "configs" / "phase17_r1od2_predeclared_grid.json").read_text())
    res = json.loads((ARTIFACTS / "phase17_r1od2_results.json").read_text())["geometries"]
    geos = {g["case_id"]: g for g in grid["geometries"]}
    prim = [(cid, p, r["pairs"][p]["dense_angular"]["5"]["delta_f_perp"])
            for cid, r in res.items() for p in ("GC", "GM") if "dense_angular" in r["pairs"][p]]
    weakest = min(prim, key=lambda x: x[2])
    specs = [(geos["CANON"], "GC"), (geos["O2_FULL_SUN_HIGH_BETA"], "GC"), (geos[weakest[0]], weakest[1])]
    print("weakest primary case by rule:", weakest)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=3, initializer=_init) as ex:
        out = list(ex.map(run_case, specs))
    print("done in %.0f s" % (time.time() - t0))
    for o in out:
        s, f = o["SIMPLIFIED"], o["FIDELITY_GL50_EARTH_SUN"]
        print("%-28s %s  simplified: range %.4f comb %.4f d %+.4f %s | fidelity: range %.4f comb %.4f d %+.4f %s | "
              "ops %s->%s | B4 %+.4f->%+.4f | max |dr| %.1f km"
              % (o["case_id"], o["pair"], s["range_f_perp"], s.get("dense_f_perp", float("nan")),
                 s.get("dense_delta", 0), s.get("dense_class"), f["range_f_perp"],
                 f.get("dense_f_perp", float("nan")), f.get("dense_delta", 0), f.get("dense_class"),
                 s.get("ops_class"), f.get("ops_class"), s.get("b4_delta", 0), f.get("b4_delta", 0),
                 o["max_position_difference_m"] / 1e3))
    preserved = all(o["class_preserved"] for o in out)
    near = any(o["margin_to_nearest_threshold"] < 0.005 for o in out)
    effect = "PRESERVES_CLASS" if preserved and not near else (
        "INCONCLUSIVE" if near else "CHANGES_CLASS")
    summary = dict(force_model="GL1800F %dx%d MOON_PA_DE440 + Earth/Sun point-mass third body + "
                               "cannonball SRP (conical penumbra); harmonic STM opt-in" % (NMAX, NMAX),
                   weakest_rule_selection=dict(case=weakest[0], pair=weakest[1], delta=weakest[2]),
                   cases=out, PRODUCTION_FIDELITY_DYNAMICS_EFFECT=effect)
    print("PRODUCTION_FIDELITY_DYNAMICS_EFFECT =", effect)
    # Phase 17-F1: ``--out NAME`` re-runs the SAME predeclared cases after the SRP
    # variational correction without overwriting the D2 evidence.
    name = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "phase17_r1od2_fidelity.json"
    (ARTIFACTS / name).write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
