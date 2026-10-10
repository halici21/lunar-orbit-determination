"""PHASE 17-R1O-D2 - run the predeclared robustness grid (D0-D6, surrogate bridge).

ANALYSIS SPACE ONLY.  Reads artifacts/phase17_r1od2_predeclared_grid.json and
refuses to run if its LF-normalized SHA-256 differs from the committed value.
One worker per geometry; every worker installs the legacy-Earth guard.

Outputs: artifacts/phase17_r1od2_results.json (+ CSV summaries).
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ARTIFACTS = HERE.parent / "artifacts"
GRID_PATH = ARTIFACTS / "phase17_r1od2_predeclared_grid.json"
GRID_SHA256_LF = "2a99c50e95c49bc4c1837787c328bcc2537fa653a2e933bf5a9ea39221c07c11"
SIGMA_NOMINAL_SYS_S = 0.033e-9
SIGMA_STRESS_S = 1.0e-9
SIGMA_OPS_S = 0.048e-9


def load_grid() -> dict:
    text = GRID_PATH.read_text(encoding="utf-8")
    sha = hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()
    if sha != GRID_SHA256_LF:
        raise SystemExit("predeclared grid hash mismatch: %s" % sha)
    return json.loads(text)


def _init() -> None:
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels(None, clear=True)
    warnings.simplefilter("ignore")
    import phase17_r1od2_core as D2
    D2.install_legacy_earth_guard()


def _clean(d):
    if isinstance(d, dict):
        return {str(k): _clean(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_clean(v) for v in d]
    if isinstance(d, np.ndarray):
        return d.tolist()
    if isinstance(d, np.generic):
        return d.item()
    return d


def _brief(m: dict) -> dict:
    keys = ("n_obs", "n_nuisance", "f_perp", "theta_k_deg", "conditional_k_information", "i_kk",
            "k_column_norm", "sigma_k", "sigma_k_frac", "design_condition", "design_rank",
            "design_rank_margin", "weakest_mode_k_component", "max_abs_k_state_correlation",
            "max_abs_k_nuisance_correlation", "design_singular_values",
            "weakest_right_singular_vector", "k_state_correlations")
    return {k: m[k] for k in keys}


def evaluate_pair(D2, scn, rb, rows, m_range, grid, *, do_quasar: bool, do_surrogate: bool) -> dict:
    rblock = (rb.h_x0, rb.h_k, rb.w)
    f0 = m_range["f_perp"]
    out = dict(pair=rows.pair, row_rule=rows.row_rule, n_rows=rows.n, n_candidates=rows.n_candidates,
               n_sessions=rows.n_sessions, cond_max=rows.cond_max,
               geometry=D2.pair_geometry(rows, rb))
    if rows.n == 0:
        return out

    def comb(w, sel=None, nuis=None, prior=None):
        hx = rows.h_x0 if sel is None else rows.h_x0[sel]
        hk = rows.h_k if sel is None else rows.h_k[sel]
        m = D2.info_metrics([rblock, (hx, hk, w)], nuisance=nuis, nuisance_prior_sigma=prior)
        d = m["f_perp"] - f0
        return dict(**_brief(m), delta_f_perp=d, relative_class=D2.relative_gain_class(d),
                    historical_r1od_class=D2.historical_r1od_class(d),
                    absolute_class=D2.absolute_class(m["f_perp"]),
                    sigma_ratio_vs_range=m["sigma_k"] / m_range["sigma_k"])

    # ---------- noise / semantics (D5) ----------
    dense_ang = {}
    for nrad in grid["realizations"]["DENSE_ANGULAR"]["sigmas_nrad"]:
        dense_ang[str(nrad)] = comb(D2.ddor_weights(rows, D2.SEMANTICS_ANGULAR, nrad * 1e-9))
    dense_del = {}
    for ns in grid["realizations"]["DENSE_DELAY"]["sigmas_ns"]:
        dense_del[str(ns)] = comb(D2.ddor_weights(rows, D2.SEMANTICS_DELAY, ns * 1e-9))
    out["dense_angular"] = dense_ang
    out["dense_delay"] = dense_del
    w5 = D2.ddor_weights(rows, D2.SEMANTICS_ANGULAR, 5e-9)
    out["ddor_only_dense_5nrad"] = _brief(D2.info_metrics([(rows.h_x0, rows.h_k, w5)]))
    # equivalent delay sigma implied by 5 nrad on this pair/geometry (s23 audit)
    out["implied_delay_sigma_ns_at_5nrad"] = dict(
        median=float(np.median(np.maximum(rows.b_perp, 1.0) * 5e-9 / D2.C_LIGHT) * 1e9),
        min=float(np.min(np.maximum(rows.b_perp, 1.0) * 5e-9 / D2.C_LIGHT) * 1e9),
        max=float(np.max(np.maximum(rows.b_perp, 1.0) * 5e-9 / D2.C_LIGHT) * 1e9))

    # ---------- operational sessions ----------
    t_split = 0.5 * float(rb.t_grid[-1])
    ops = {}
    ops_rows = {}
    for mode in ("OPS_EVERY_WINDOW", "OPS_TWO_SESSIONS"):
        sel = D2.ops_selection(rows, mode, t_split)
        ops_rows[mode] = D2.subset(rows, sel)
        if sel.size == 0:
            ops[mode] = dict(n_points=0)
            continue
        ops[mode] = dict(n_points=int(sel.size), n_sessions=int(ops_rows[mode].n_sessions),
                         **comb(np.full(sel.size, 1.0 / SIGMA_OPS_S ** 2), sel=sel))
    out["ops"] = ops

    # ---------- bias / nuisance (D6) ----------
    def bias_suite(sub, w):
        n_r = rb.n_obs
        blocks = [rblock, (sub.h_x0, sub.h_k, w)]
        res = dict(B0=comb_generic(blocks))
        ones = np.concatenate([np.zeros(n_r), np.ones(sub.n)])
        dk_per_s = D2.omitted_bias_k_shift(blocks, ones)
        res["B1"] = dict(dk_per_ns=dk_per_s * 1e-9,
                         k_shift_frac_nominal=abs(dk_per_s * SIGMA_NOMINAL_SYS_S) / 0.01,
                         k_shift_frac_stress_1ns=abs(dk_per_s * SIGMA_STRESS_S) / 0.01)
        per = []
        for s in range(sub.n_sessions):
            v = np.concatenate([np.zeros(n_r), (sub.session == s).astype(float)])
            per.append(D2.omitted_bias_k_shift(blocks, v) * SIGMA_NOMINAL_SYS_S)
        res["B2"] = dict(k_shift_rss_frac=float(np.sqrt(np.sum(np.square(per)))) / 0.01,
                         n_sessions=sub.n_sessions)
        nuis = D2.session_nuisance_columns(n_r, sub)
        for tag, prior in (("B3", None), ("B4", SIGMA_NOMINAL_SYS_S)):
            m = D2.info_metrics(blocks, nuisance=nuis, nuisance_prior_sigma=prior)
            d = m["f_perp"] - f0
            res[tag] = dict(**_brief(m), delta_f_perp=d, relative_class=D2.relative_gain_class(d),
                            absolute_class=D2.absolute_class(m["f_perp"]),
                            sigma_inflation_vs_B0=m["sigma_k"] / res["B0"]["sigma_k"],
                            retained_delta_fraction=(d / res["B0"]["delta_f_perp"]
                                                     if res["B0"]["delta_f_perp"] != 0 else float("nan")))
        b3 = res["B3"]
        res["survives_calibration_freedom"] = bool(
            b3["delta_f_perp"] >= 0.02 and b3["retained_delta_fraction"] >= 0.5)
        return res

    def comb_generic(blocks):
        m = D2.info_metrics(blocks)
        d = m["f_perp"] - f0
        return dict(**_brief(m), delta_f_perp=d, relative_class=D2.relative_gain_class(d),
                    absolute_class=D2.absolute_class(m["f_perp"]))

    out["bias"] = dict(DENSE_5NRAD=bias_suite(rows, w5))
    if ops["OPS_EVERY_WINDOW"].get("n_points", 0):
        sub = ops_rows["OPS_EVERY_WINDOW"]
        out["bias"]["OPS_EVERY_WINDOW"] = bias_suite(sub, np.full(sub.n, 1.0 / SIGMA_OPS_S ** 2))

    # ---------- surrogate bridge (s31) ----------
    if do_surrogate:
        sx, sk, sw = D2.surrogate_rows(rb, rows, 5e-9)
        ms = D2.info_metrics([rblock, (sx, sk, sw)])
        ms_pw = D2.info_metrics([rblock, (sx, sk, w5)])
        out["surrogate"] = dict(
            f_perp_combined_surrogate_weights=ms["f_perp"],
            f_perp_combined_production_weights=ms_pw["f_perp"],
            f_perp_combined_production=dense_ang["5"]["f_perp"],
            abs_diff=abs(ms["f_perp"] - dense_ang["5"]["f_perp"]),
            rel_diff=abs(ms["f_perp"] - dense_ang["5"]["f_perp"]) / dense_ang["5"]["f_perp"],
            abs_diff_same_weights=abs(ms_pw["f_perp"] - dense_ang["5"]["f_perp"]),
            max_rel_row_diff_state=float(np.max(np.abs(sx - rows.h_x0)) / np.max(np.abs(rows.h_x0))),
            max_rel_row_diff_k=float(np.max(np.abs(sk - rows.h_k)) / np.max(np.abs(rows.h_k))))

    # ---------- quasar systematics (D1) ----------
    if do_quasar:
        qg = grid["quasar_grid"]
        qres = []
        for real in ("DENSE_5NRAD", "OPS_EVERY_WINDOW"):
            if real == "DENSE_5NRAD":
                sub, w = rows, w5
            else:
                sub = ops_rows["OPS_EVERY_WINDOW"]
                if sub.n == 0:
                    continue
                w = np.full(sub.n, 1.0 / SIGMA_OPS_S ** 2)
            blocks = [rblock, (sub.h_x0, sub.h_k, w)]
            xf_rows = rb.xf[sub.idx]
            sig_formal = D2.info_metrics(blocks)["sigma_k"]
            for sep in qg["separations_rad"]:
                for pa in qg["position_angles_deg"]:
                    quas = D2.session_quasars(sub, sep, np.radians(pa))
                    vecs = D2.systematic_delay_vectors(sub, xf_rows, quas)
                    grp = {}
                    for g, vs in vecs.items():
                        shifts = [D2.omitted_bias_k_shift(
                            blocks, np.concatenate([np.zeros(rb.n_obs), v])) for v in vs]
                        grp[g] = float(np.sqrt(np.sum(np.square(shifts))))
                    tot = float(np.sqrt(sum(v ** 2 for v in grp.values())))
                    dvec = sub.s_sc_geo - quas[sub.session]
                    bhat = sub.baseline - np.sum(sub.baseline * sub.s_sc_geo, axis=1)[:, None] * sub.s_sc_geo
                    bhat /= np.linalg.norm(bhat, axis=1)[:, None]
                    qres.append(dict(realization=real, sep_rad=sep, pa_deg=pa,
                                     k_bias_rss_by_group_frac={k: v / 0.01 for k, v in grp.items()},
                                     k_bias_rss_total_frac=tot / 0.01,
                                     k_bias_over_formal_sigma=tot / sig_formal,
                                     formal_sigma_k_frac=sig_formal / 0.01,
                                     separation_along_projected_baseline_mean_rad=float(
                                         np.mean(np.abs(np.sum(dvec * bhat, axis=1))))))
        out["quasar_systematics"] = qres
    return out


def run_geometry(g: dict) -> dict:
    import phase17_r1od2_core as D2

    t0 = time.time()
    grid = load_grid()
    scn = D2.Scenario(g["case_id"], g["family"], float(g["et0"]), tuple(g["x0"]),
                      float(g["duration_s"]), float(g["k_srp"]), g["shadow_model"])
    nom = D2.propagate_nom48(scn)
    rb = D2.build_range_block(scn, nom)
    audit = D2.earth_epoch_audit(scn, rb)
    m_range = D2.info_metrics([(rb.h_x0, rb.h_k, rb.w)])
    res = dict(case_id=scn.case_id, family=scn.family, source=g["source"],
               descriptors=D2.geometry_descriptors(scn, rb), earth_audit=audit,
               range_only=_brief(m_range), pairs={})
    for pair in ("GC", "GM", "MC"):
        rows = D2.build_ddor_rows(scn, rb, pair, D2.ROW_RULE_PRODUCTION_VISIBILITY)
        res["pairs"][pair] = evaluate_pair(
            D2, scn, rb, rows, m_range, grid, do_quasar=(scn.case_id == "CANON" and pair != "MC"),
            do_surrogate=(pair != "MC"))
    if scn.case_id == "CANON":
        hist = {}
        for pair in ("GC", "GM", "MC"):
            rows_h = D2.build_ddor_rows(scn, rb, pair, D2.ROW_RULE_ELEVATION_ONLY)
            hist[pair] = evaluate_pair(D2, scn, rb, rows_h, m_range, grid, do_quasar=False,
                                       do_surrogate=(pair == "GC"))
        res["elevation_only_historical"] = hist
    res["elapsed_s"] = time.time() - t0
    return _clean(res)


def main() -> None:
    grid = load_grid()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(grid["geometries"]), initializer=_init) as ex:
        results = list(ex.map(run_geometry, grid["geometries"]))
    print("done in %.0f s" % (time.time() - t0))
    out = dict(phase="17-R1O-D2", grid_sha256_lf=GRID_SHA256_LF,
               geometries={r["case_id"]: r for r in results})
    (ARTIFACTS / "phase17_r1od2_results.json").write_text(json.dumps(out, indent=1, default=float))
    rows = []
    for r in results:
        for pair, p in r["pairs"].items():
            if "dense_angular" not in p:
                continue
            d5 = p["dense_angular"]["5"]
            op = p["ops"]["OPS_EVERY_WINDOW"]
            rows.append(dict(case=r["case_id"], family=r["family"], pair=pair, n_rows=p["n_rows"],
                             n_sessions=p["n_sessions"],
                             range_f_perp=r["range_only"]["f_perp"], combined_f_perp=d5["f_perp"],
                             delta_f_perp=d5["delta_f_perp"], relative_class=d5["relative_class"],
                             absolute_class=d5["absolute_class"], sigma_k_frac=d5["sigma_k_frac"],
                             range_sigma_k_frac=r["range_only"]["sigma_k_frac"],
                             ops_delta=op.get("delta_f_perp"), ops_class=op.get("relative_class"),
                             b3_dense_delta=p["bias"]["DENSE_5NRAD"]["B3"]["delta_f_perp"],
                             survives_dense=p["bias"]["DENSE_5NRAD"]["survives_calibration_freedom"],
                             survives_ops=p["bias"].get("OPS_EVERY_WINDOW", {}).get(
                                 "survives_calibration_freedom")))
    with (ARTIFACTS / "phase17_r1od2_case_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for x in rows:
        print("  %-28s %s n=%3d s=%d range=%.4f comb=%.4f d=%+.4f %-22s sig=%.2f%% | ops d=%s %s | B3 d=%+.4f surv=%s/%s"
              % (x["case"], x["pair"], x["n_rows"], x["n_sessions"], x["range_f_perp"],
                 x["combined_f_perp"], x["delta_f_perp"], x["relative_class"],
                 100 * x["sigma_k_frac"],
                 ("%+.4f" % x["ops_delta"]) if x["ops_delta"] is not None else "-",
                 x["ops_class"], x["b3_dense_delta"], x["survives_dense"], x["survives_ops"]))


if __name__ == "__main__":
    main()
