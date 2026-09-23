"""PHASE 17-GEO - run the predeclared geometry grid (G0-G5 + eclipse counterfactual).

ANALYSIS SPACE ONLY.  Reads artifacts/phase17_geo_predeclared_grid.json and refuses
to run unless its SHA-256 equals the frozen v2 hash below: cases cannot be moved
after results exist.  Per case it builds the production two-way range arc,
checks physical validity, records geometry/eclipse/tracking descriptors, and
computes the QR-only information metrics.  Design matrices are cached under
results/phase17_geo_cache/ (gitignored) so the matched-count, classification and
figure stages never re-propagate.

Usage:  python examples/phase17_geo_campaign.py [--workers 8] [--only ID ...]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
ARTIFACTS = ROOT / "artifacts"
CACHE = ROOT / "results" / "phase17_geo_cache"
GRID_FILE = ARTIFACTS / "phase17_geo_predeclared_grid.json"
GRID_V2_SHA256 = "2ba87a471c6b303e7d39a0607daed60bca693b21cb6075642ed7b64e4c7e754d"


def _init_worker():
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels(None, clear=True)


def _passes(obs: np.ndarray, n_stations: int) -> dict:
    """Station passes = runs of consecutive grid indices per station."""
    out = {}
    for s in range(1, n_stations + 1):
        idx = np.sort(obs[obs[:, 2] == s, 3].astype(int))
        if idx.size == 0:
            out[s] = []
            continue
        breaks = np.where(np.diff(idx) > 1)[0]
        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks, [idx.size - 1]])
        out[s] = [int(idx[e] - idx[b] + 1) for b, e in zip(starts, ends)]
    return out


def run_case(case: dict) -> dict:
    import phase17_geo_core as G
    from phase17_r1m_core import build_range_arc

    t0 = time.time()
    cid = case["id"]
    et = case["epoch_et"]
    x0 = np.asarray(case["x0"], float)
    try:
        if case["fixture"] == "PUBLISHED_R1M":
            fx = build_range_arc(0.0, case["duration_s"], label=cid,
                                 station_filter=tuple(case["stations"]), cadence_s=90.0)
            arc = G.GeoArc(cid, fx.t_grid, fx.nom48, fx.obs, fx.pass_geo, fx.h_x0,
                           fx.h_k, fx.w, tuple(case["stations"]))
        else:
            arc = G.build_geo_range_arc(x0, et, case["duration_s"], label=cid,
                                        station_names=tuple(case["stations"]),
                                        k_srp=case["k_srp"], shadow_model=case["shadow_model"])
    except Exception as exc:  # recorded, never silently replaced (s47)
        return dict(id=cid, stage=case["stage"], valid=False,
                    invalid_reason=f"arc build failed: {exc!r}", trace=traceback.format_exc())

    nom, t = arc.nom48, arc.t_grid
    alt = G.altitude_stats(nom)
    reasons = []
    if not np.all(np.isfinite(nom)):
        reasons.append("non-finite propagated state")
    if alt["alt_min_km"] <= 5.0:
        reasons.append(f"min altitude {alt['alt_min_km']:.2f} km <= 5 km")
    el0 = G.elements_from_state(nom[0, :6])
    if arc.n_obs == 0:
        reasons.append("no visible observations")

    # geometry descriptors along the arc (osculating plane at start/mid/end)
    idx3 = (0, len(t) // 2, len(t) - 1)
    betas, views = [], []
    for i in idx3:
        h = np.array(G.elements_from_state(nom[i, :6])["h_hat_mci"])
        betas.append(G.beta_deg(h, et + t[i]))
        views.append(G.earth_view_deg(h, et + t[i]))
    sh = G.srp_history(nom, t, et)
    ecl = G.eclipse_stats(sh["nu"], t)
    lit = sh["nu"] > 0.999
    rtn = np.abs(sh["g_rtn"][lit]) if lit.any() else np.zeros((1, 3))
    rtn_mean = rtn.mean(axis=0)
    rtn_share = (rtn_mean / rtn_mean.sum()).tolist() if rtn_mean.sum() > 0 else [float("nan")] * 3
    if not np.all(np.isfinite(sh["nu"])):
        reasons.append("non-finite illumination")
    s_k_norm = np.linalg.norm(nom[:, 42:45], axis=1)
    passes = _passes(arc.obs, len(arc.stations)) if arc.n_obs else {}
    pass_lengths = [p for v in passes.values() for p in v]

    res = dict(
        id=cid, stage=case["stage"], fixture=case["fixture"], valid=not reasons,
        invalid_reason="; ".join(reasons), epoch_et=et,
        duration_h=case["duration_s"] / 3600.0, k_srp=case["k_srp"],
        shadow_model=case["shadow_model"], stations=";".join(case["stations"]),
        inc_deg=el0["inc_deg"], raan_deg=el0["raan_deg"], ecc=el0["e"],
        period_s=el0["period_s"], revolutions=case["duration_s"] / el0["period_s"],
        beta_start_deg=betas[0], beta_mid_deg=betas[1], beta_end_deg=betas[2],
        earth_view_start_deg=views[0], earth_view_mid_deg=views[1], earth_view_end_deg=views[2],
        **alt, **ecl,
        srp_rtn_share_R=rtn_share[0], srp_rtn_share_T=rtn_share[1], srp_rtn_share_N=rtn_share[2],
        s_k_pos_norm_end_m=float(s_k_norm[-1]), s_k_pos_norm_max_m=float(s_k_norm.max()),
        n_passes=len(pass_lengths),
        mean_pass_samples=float(np.mean(pass_lengths)) if pass_lengths else 0.0,
        runtime_s=time.time() - t0,
    )
    for key in ("beta_target", "altitude_nominal_km", "normalization", "inc_target",
                "partner", "counterfactual_of", "role"):
        if key in case:
            res[key] = case[key]
    if arc.n_obs:
        m = G.information_metrics(arc.h_x0, arc.h_k, arc.w, k_truth=case["k_srp"])
        res.update({f"full_{k}": v for k, v in m.items()
                    if not isinstance(v, list)})
        res["full_k_state_correlations"] = m["k_state_correlations"]
        res["full_design_singular_values"] = m["design_singular_values"]
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            CACHE / f"{cid}.npz", h_x0=arc.h_x0, h_k=arc.h_k, w=arc.w, obs=arc.obs, t=t,
            nu=sh["nu"], g_rtn=sh["g_rtn"], g_norm=sh["g_norm"], s_k_norm=s_k_norm,
            r=nom[:, :3], v=nom[:, 3:6])
    return res


def classify(d: float, f: float) -> str:
    """Predeclared thresholds (grid 'thresholds')."""
    if d >= 0.25 and f >= 0.50:
        return "STRONG_DIRECTION_GAIN"
    if d >= 0.10 and f >= 0.35:
        return "MODERATE_DIRECTION_GAIN"
    if d <= -0.10:
        return "DIRECTION_LOSS"
    if abs(d) >= 0.02:
        return "MARGINAL_DIRECTION_CHANGE_" + ("GAIN" if d > 0 else "LOSS")
    return "MAGNITUDE_ONLY"


def matched_pass(results: list[dict]) -> None:
    import phase17_geo_core as G
    by_stage: dict[str, list[dict]] = {}
    for r in results:
        if r.get("valid") and "full_f_perp" in r:
            by_stage.setdefault(r["stage"], []).append(r)
    for stage, rs in by_stage.items():
        n_match = min(r["full_n_obs"] for r in rs)
        for r in rs:
            z = np.load(CACHE / f"{r['id']}.npz")
            sel = G.matched_indices(z["h_k"].size, n_match)
            m = G.information_metrics(z["h_x0"][sel], z["h_k"][sel], z["w"][sel],
                                      k_truth=r["k_srp"])
            r["matched_n_obs"] = int(sel.size)
            r["matched_stage_n_match"] = int(n_match)
            for k in ("f_perp", "theta_k_deg", "conditional_k_information", "sigma_k",
                      "sigma_k_frac", "k_column_norm", "max_abs_k_state_correlation",
                      "weakest_mode_k_component", "smallest_design_sv"):
                r[f"matched_{k}"] = m[k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    text = GRID_FILE.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != GRID_V2_SHA256:
        raise SystemExit(f"grid SHA-256 {sha} != frozen v2 {GRID_V2_SHA256}: refusing (s19)")
    grid = json.loads(text)
    cases = [c for c in grid["cases"] if "invalid" not in c]
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only)]
    print(f"grid v{grid['grid_version']} sha256 {sha} OK -- {len(cases)} cases, "
          f"{args.workers} workers", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(run_case, c): c["id"] for c in cases}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            tag = (f"f_perp={r['full_f_perp']:.4f} sigK/K={100 * r['full_sigma_k_frac']:.3f}% "
                   f"n={r['full_n_obs']}" if "full_f_perp" in r else r.get("invalid_reason", ""))
            print(f"  [{len(results):2d}/{len(cases)}] {r['id']:26s} valid={r['valid']} "
                  f"{tag}  ({r.get('runtime_s', 0):.0f}s)", flush=True)

    order = {c["id"]: i for i, c in enumerate(grid["cases"])}
    results.sort(key=lambda r: order.get(r["id"], 999))
    matched_pass(results)

    ref = next((r for r in results if r["id"] == "G0_consistent"), None)
    if ref is not None:
        f_ref = ref["full_f_perp"]
        fm_ref = ref.get("matched_f_perp")
        for r in results:
            if "full_f_perp" not in r:
                continue
            base = f_ref
            if r["stage"] == "CF":
                src = next(x for x in results if x["id"] == r["counterfactual_of"])
                base = src["full_f_perp"]
            r["delta_f_perp_full"] = r["full_f_perp"] - base
            r["class_full"] = classify(r["delta_f_perp_full"], r["full_f_perp"])
            if "matched_f_perp" in r and fm_ref is not None and r["stage"] != "CF":
                # matched values are stage-internal; the class uses the stage's own
                # matched G0-equivalent where one exists, else the full-data reference
                r["delta_f_perp_matched_vs_ref_full"] = r["matched_f_perp"] - f_ref
                r["class_matched"] = classify(r["delta_f_perp_matched_vs_ref_full"],
                                              r["matched_f_perp"])

    ARTIFACTS.mkdir(exist_ok=True)
    out_json = ARTIFACTS / "phase17_geo_results.json"
    out_json.write_text(json.dumps(dict(grid_sha256=sha, results=results), indent=1, default=float))
    scalar_keys = []
    for r in results:
        for k, v in r.items():
            if not isinstance(v, (list, dict)) and k not in scalar_keys and k != "trace":
                scalar_keys.append(k)
    with (ARTIFACTS / "phase17_geo_information_metrics.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=scalar_keys, extrasaction="ignore")
        wr.writeheader()
        for r in results:
            wr.writerow({k: r.get(k, "") for k in scalar_keys})
    print("wrote", out_json)


if __name__ == "__main__":
    main()
