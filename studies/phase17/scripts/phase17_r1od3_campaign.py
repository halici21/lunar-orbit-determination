"""PHASE 17-R1O-D3 - predeclared DeltaDOR tracking-design characterization: campaign.

Stage A  24 uniform synodic epochs x {GC_ONLY, GM_ONLY, GC_PLUS_GM} x deterministic
         schedule patterns x {C0, C1 x0.5/1/2, C2}, canonical orbit.
Stage B  6 representative orbits x epochs {0, 6, 12, 18} (+ window centre by rule),
         primary patterns, C0 / C1 nominal.
Stage C  4 rule-selected high-fidelity cases.

Every rule comes from artifacts/phase17_r1od3_predeclared_grid.json (hash-checked).
``--catalog PATH`` enables the DSN-107F coordinate overlay (else UNRESOLVED).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase17_r1od3_core as D3  # noqa: E402

ARTIFACTS = D3.ARTIFACTS
N_WORKERS = 15


def _init() -> None:
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels(None, clear=True)
    warnings.simplefilter("ignore")
    D3.D2.install_legacy_earth_guard()


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


# ======================================================================
# One geometry (epoch x orbit x dynamics)
# ======================================================================
def run_geometry(spec: dict) -> dict:
    """spec: label, k, x0, orbit, dynamics ('SIMPLIFIED'|'FIDELITY'), cals, patterns ('ALL'|'PRIMARY')."""
    D2 = D3.D2
    grid = D3.load_grid()
    t0 = time.time()
    et0 = D3.epoch_et(spec["k"], grid)
    _, t_orbit = D2.M.campaign_epoch()
    scn = D2.Scenario(spec["label"], "d3", et0, tuple(spec["x0"]), 15.0 * t_orbit)
    if spec["dynamics"] == "FIDELITY":
        from phase17_r1od2_fidelity import propagate_fidelity
        nom = propagate_fidelity(scn)
    else:
        nom = D2.propagate_nom48(scn)
    rb = D2.build_range_block(scn, nom)
    audit = D2.earth_epoch_audit(scn, rb)
    m_range = D2.info_metrics([(rb.h_x0, rb.h_k, rb.w)])
    f0 = m_range["f_perp"]
    rows = {p: D2.build_ddor_rows(scn, rb, p, D2.ROW_RULE_PRODUCTION_VISIBILITY) for p in ("GC", "GM")}
    sessions = {p: D3.build_sessions(rows[p], grid) for p in ("GC", "GM")}
    t_mid = 0.5 * float(rb.t_grid[-1])
    scheds = D3.build_schedules(spec["k"], rows, sessions, t_mid)
    if spec["patterns"] == "PRIMARY":
        scheds = {key: v for key, v in scheds.items() if key[1] == D3.PRIMARY_PATTERN[key[0]]}

    # -------- contract checks (no K information involved)
    contract = dict(earth_audit_passed=bool(audit["passed"]),
                    rows_dual_visible={p: D3.pair_row_contract(rows[p], rb, p) for p in rows},
                    n_rows={p: rows[p].n for p in rows})
    np_ok, own_ok, dbl = True, True, False
    for key, s in scheds.items():
        if s is None:
            continue
        for p in ("GC", "GM"):
            sel = s.baseline == p
            if sel.any():
                own_ok &= bool(np.array_equal(s.h_x0[sel], rows[p].h_x0[s.parent_row[sel]]))
                own_ok &= bool(np.array_equal(s.h_k[sel], rows[p].h_k[s.parent_row[sel]]))
        for j, sess in enumerate(s.sessions):
            r = s.parent_row[s.session == j]
            np_ok &= bool(set(r.tolist()) == set(sess.np_index) and set(sess.np_index) <= set(sess.row_index))
            np_ok &= bool(np.all((rows[sess.pair].t_used[list(sess.np_index)] >= sess.start)
                                 & (rows[sess.pair].t_used[list(sess.np_index)] <= sess.end)))
        bc = D3.bias_columns(rb.n_obs, s)
        own_ok &= bool(np.all(bc[:rb.n_obs] == 0) and np.all(bc[rb.n_obs:].sum(axis=1) == 1))
        dbl |= D3.double_booked(s)
    contract.update(normal_points_attached=np_ok, ownership_consistent=own_ok, joint_double_booked=dbl)

    # -------- metrics
    cells = {}
    rebuild = []
    for (strategy, pattern), s in scheds.items():
        key = "%s|%s" % (strategy, pattern)
        if s is None:
            cells[key] = dict(feasible=False)
            continue
        c = dict(feasible=True, n_np=s.n, n_sessions=s.n_sessions, session_hours=s.session_hours,
                 station_hours=2.0 * s.session_hours, raw_conflicts=s.raw_conflicts, gm_dropped=s.gm_dropped,
                 sessions=[dict(pair=x.pair, window=[x.window_start, x.window_end], interval=[x.start, x.end],
                                np_t=[float(rows[x.pair].t_used[i]) for i in x.np_index],
                                s_geo_mid=rows[x.pair].s_sc_geo[x.np_index[len(x.np_index) // 2]].tolist())
                           for x in s.sessions],
                 projected_baseline_km={p: D3.projected_baseline_km(rows[p], s.parent_row[s.baseline == p])
                                        for p in ("GC", "GM") if np.any(s.baseline == p)},
                 cal={})
        for cal in spec["cals"]:
            c["cal"][cal] = D3.schedule_metrics(rb, s, cal, grid, f0)
        base = c["cal"].get("C0_IDEAL")
        for cal, m in c["cal"].items():
            if base and cal != "C0_IDEAL":
                m["sigma_k_inflation_vs_C0"] = m["sigma_k"] / base["sigma_k"]
                m["delta_loss_vs_C0"] = base["delta_f_perp"] - m["delta_f_perp"]
                m["retained_fraction_vs_C0"] = (m["delta_f_perp"] / base["delta_f_perp"]
                                                if abs(base["delta_f_perp"]) > 1e-12 else float("nan"))
        if pattern == D3.PRIMARY_PATTERN[strategy]:
            for cal in ("C0_IDEAL", D3.C1_NOMINAL, "C2_FREE"):
                if cal in c["cal"]:
                    f_ind = D3.independent_f_perp(rb, s, cal, grid)
                    rebuild.append(dict(cell=key, cal=cal, reported=c["cal"][cal]["f_perp"], independent=f_ind,
                                        abs_diff=abs(f_ind - c["cal"][cal]["f_perp"])))
        cells[key] = c

    # -------- predictors / diagnostics (dense rows, no schedule)
    diag = dict(geometry=D2.geometry_descriptors(scn, rb), range_only=dict(
        f_perp=f0, sigma_k_frac=m_range["sigma_k_frac"], n_obs=int(rb.n_obs),
        design_singular_values=m_range["design_singular_values"],
        weakest_mode_k_component=m_range["weakest_mode_k_component"]), pairs={})
    for p in ("GC", "GM"):
        r = rows[p]
        d = dict(n_dense_rows=r.n, n_windows=r.n_sessions, n_sessions=len(sessions[p]),
                 windows=[dict(start=x.window_start, end=x.window_end) for x in sessions[p]])
        if r.n >= 7:
            d["ddor_only_dense_delay_f_perp"] = D2.info_metrics([(r.h_x0, r.h_k, D3.delay_weight(r.n, grid))])["f_perp"]
            # IMPLEMENTATION_BUG_FIX (D3, logged in the task state): pair_geometry's own
            # "n_sessions"/"n_rows" (RAW windows) must not overwrite the qualifying-session count
            d.update({k_: v_ for k_, v_ in D2.pair_geometry(r, rb).items() if k_ not in d and k_ != "n_sessions"})
        if r.n and spec["k"] == 0 and spec["orbit"] == "CANON" and spec["dynamics"] == "SIMPLIFIED" and p == "GC":
            w5 = D2.ddor_weights(r, D2.SEMANTICS_ANGULAR, 5e-9)
            diag["canonical_dense_angular_gc_f_perp"] = D2.info_metrics([(rb.h_x0, rb.h_k, rb.w),
                                                                        (r.h_x0, r.h_k, w5)])["f_perp"]
        diag["pairs"][p] = d
    return _clean(dict(label=spec["label"], k=spec["k"], orbit=spec["orbit"], dynamics=spec["dynamics"],
                       et0=et0, range_f_perp=f0, contract=contract, cells=cells, diagnostics=diag,
                       rebuild=rebuild, elapsed_s=time.time() - t0))


# ======================================================================
# Analysis helpers
# ======================================================================
def primary_delta(res: dict, strategy: str, cal: str) -> float | None:
    c = res["cells"].get("%s|%s" % (strategy, D3.PRIMARY_PATTERN[strategy]))
    if not c or not c.get("feasible") or cal not in c["cal"]:
        return None
    return c["cal"][cal]["delta_f_perp"]


def windows(flags: list[bool], min_len: int) -> list[tuple[int, int]]:
    """Maximal runs of True of length >= min_len, no wraparound: (first, last)."""
    out, start = [], None
    for i, f in enumerate(flags + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None
    return out


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def strength(rho: float) -> str:
    a = abs(rho) if np.isfinite(rho) else 0.0
    return "STRONG" if a >= 0.7 else "MODERATE" if a >= 0.5 else "WEAK" if a >= 0.3 else "NONE"


def parse_catalog(path: Path) -> np.ndarray | None:
    """DSN-107F Table 3 coordinates -> (N, 3) J2000/ICRF unit vectors (coordinates only)."""
    import subprocess
    txt_path = path.with_suffix(".txt")
    if not txt_path.exists():
        subprocess.run(["pdftotext", "-layout", str(path), str(txt_path)], check=True)
    txt = txt_path.read_text(encoding="utf-8", errors="replace")
    pat = re.compile(r"(?m)^\s*(?:\S+\s+){0,3}?(\d{1,4})\s+(\d{2})\s+(\d{2})\s+(\d{1,2}\.\d{6,})\s+"
                     r"([+-]\d{2})\s+(\d{2})\s+(\d{1,2}\.\d{5,})\s")
    src = {}
    for m in pat.finditer(txt):
        i, h, mi, s, d, dm, ds = m.groups()
        ra = math.radians(15 * (int(h) + int(mi) / 60 + float(s) / 3600))
        dec = math.radians((-1 if d.startswith("-") else 1) * (abs(int(d)) + int(dm) / 60 + float(ds) / 3600))
        src[int(i)] = (math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec))
    return np.array(list(src.values())) if src else None


# ======================================================================
# Main
# ======================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=None)
    args = ap.parse_args()
    grid = D3.load_grid()
    t_start = time.time()
    import phase17_r1m_core as M
    canon_x0 = M.campaign_initial_state().tolist()
    d2grid = json.loads((ARTIFACTS / "phase17_r1od2_predeclared_grid.json").read_text(encoding="utf-8"))
    orbit_x0 = {"CANON": canon_x0}
    for g in d2grid["geometries"]:
        if g["case_id"] in grid["stage_B_orbit_robustness"]["orbits"]:
            orbit_x0[g["case_id"]] = g["x0"]
    assert set(orbit_x0) == set(grid["stage_B_orbit_robustness"]["orbits"]), orbit_x0.keys()
    n_ep = grid["stage_A_epoch_grid"]["count"]

    # ---------------------------------------------------------------- Stage A
    specs = [dict(label="A_E%02d" % k, k=k, x0=canon_x0, orbit="CANON", dynamics="SIMPLIFIED",
                  cals=list(D3.CALIBRATIONS), patterns="ALL") for k in range(n_ep)]
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init) as ex:
        stage_a = list(ex.map(run_geometry, specs))
    print("Stage A done in %.0f s" % (time.time() - t_start), flush=True)

    strategies = ("GC_ONLY", "GM_ONLY", "GC_PLUS_GM")
    delta = {s: {c: [primary_delta(r, s, c) for r in stage_a] for c in D3.CALIBRATIONS} for s in strategies}
    strong = {s: {c: [d is not None and d > 0.05 for d in delta[s][c]] for c in D3.CALIBRATIONS}
              for s in strategies}
    minlen = grid["tracking_window"]["min_consecutive"]
    win = {s: {c: windows(strong[s][c], minlen) for c in D3.CALIBRATIONS} for s in strategies}
    longest = None
    for s in strategies:
        for w in win[s][D3.C1_NOMINAL]:
            if longest is None or (w[1] - w[0]) > (longest[1][1] - longest[1][0]):
                longest = (s, w)
    window_centre = None if longest is None else (longest[1][0] + longest[1][1]) // 2

    # ---------------------------------------------------------------- Stage B
    b_epochs = sorted({0, 6, 12, 18} | ({window_centre} if window_centre is not None else set()))
    specs_b = [dict(label="B_%s_E%02d" % (o, k), k=k, x0=x0, orbit=o, dynamics="SIMPLIFIED",
                    cals=["C0_IDEAL", D3.C1_NOMINAL], patterns="PRIMARY")
               for o, x0 in orbit_x0.items() if o != "CANON" for k in b_epochs]
    # ---------------------------------------------------------------- Stage C (rule-selected)
    gc1 = delta["GC_ONLY"][D3.C1_NOMINAL]
    valid = [(k, d) for k, d in enumerate(gc1) if d is not None]
    hf2 = (longest[0], longest[1][0]) if longest else ("GC_ONLY", max(valid, key=lambda kv: kv[1])[0])
    hf3 = ("GC_ONLY", min(valid, key=lambda kv: kv[1])[0])
    hf_cases = {"HF1": ("GC_ONLY", 0), "HF2": hf2, "HF3": hf3, "HF4": ("GC_PLUS_GM", 0)}
    specs_c = [dict(label="C_E%02d_FID" % k, k=k, x0=canon_x0, orbit="CANON", dynamics="FIDELITY",
                    cals=["C0_IDEAL", D3.C1_NOMINAL], patterns="PRIMARY")
               for k in sorted({k for _, k in hf_cases.values()})]
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init) as ex:
        fut_c = [ex.submit(run_geometry, s) for s in specs_c]
        stage_b = list(ex.map(run_geometry, specs_b))
        stage_c = [f.result() for f in fut_c]
    print("Stages B/C done in %.0f s" % (time.time() - t_start), flush=True)

    analyse(grid, stage_a, stage_b, stage_c, delta, strong, win, longest, window_centre, hf_cases,
            b_epochs, args.catalog, time.time() - t_start)


def analyse(grid, stage_a, stage_b, stage_c, delta, strong, win, longest, window_centre, hf_cases,
            b_epochs, catalog, elapsed) -> None:
    strategies = ("GC_ONLY", "GM_ONLY", "GC_PLUS_GM")
    n_ep = len(stage_a)
    phase = [15.0 * k for k in range(n_ep)]
    spacing_d = grid["stage_A_epoch_grid"]["spacing_s"] / 86400.0
    cls = lambda d: None if d is None else ("STRONG" if d > 0.05 else "MODEST" if d > 0.02 else  # noqa: E731
                                            "MAGNITUDE_ONLY" if d >= -0.02 else "DIRECTION_LOSS")

    # ---------------- contracts and rebuild
    contracts_ok = all(r["contract"]["earth_audit_passed"] and all(r["contract"]["rows_dual_visible"].values())
                       and r["contract"]["normal_points_attached"] and r["contract"]["ownership_consistent"]
                       and not r["contract"]["joint_double_booked"] for r in stage_a + stage_b + stage_c)
    rebuild = [x for r in stage_a + stage_b + stage_c for x in r["rebuild"]]
    rebuild_max = max(x["abs_diff"] for x in rebuild)
    canon = stage_a[0]
    cont = dict(range_f_perp=canon["range_f_perp"],
                dense_angular_gc_f_perp=canon["diagnostics"].get("canonical_dense_angular_gc_f_perp"),
                authoritative_range=0.091502, authoritative_combined=0.172343)
    cont["reproduced"] = bool(abs(cont["range_f_perp"] - 0.0915024012983624) < 1e-6
                              and abs(cont["dense_angular_gc_f_perp"] - 0.172343) < 1e-6)

    # ---------------- CSVs
    with (ARTIFACTS / "phase17_r1od3_epoch_geometry.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["k", "synodic_phase_deg", "et0", "sun_moon_earth_deg", "earth_los_vs_srp_direction_deg",
                    "beta_deg", "earth_view_deg", "range_obs", "range_f_perp", "range_sigma_k_frac",
                    "gc_dense_rows", "gc_windows", "gc_sessions", "gm_dense_rows", "gm_windows", "gm_sessions",
                    "gc_ddor_only_dense_f_perp", "gm_ddor_only_dense_f_perp",
                    "gc_abs_cos_baseline_vs_k", "gm_abs_cos_baseline_vs_k"])
        for r in stage_a:
            g, dg = r["diagnostics"]["geometry"], r["diagnostics"]["pairs"]
            w.writerow([r["k"], 15.0 * r["k"], r["et0"], g["sun_moon_earth_deg"], g["earth_los_vs_srp_direction_deg"],
                        g["beta_deg"], g["earth_view_deg"], r["diagnostics"]["range_only"]["n_obs"], r["range_f_perp"],
                        r["diagnostics"]["range_only"]["sigma_k_frac"],
                        dg["GC"]["n_dense_rows"], dg["GC"]["n_windows"], dg["GC"]["n_sessions"],
                        dg["GM"]["n_dense_rows"], dg["GM"]["n_windows"], dg["GM"]["n_sessions"],
                        dg["GC"].get("ddor_only_dense_delay_f_perp"), dg["GM"].get("ddor_only_dense_delay_f_perp"),
                        dg["GC"].get("abs_cos_baseline_vs_k_plane_of_sky_mean"),
                        dg["GM"].get("abs_cos_baseline_vs_k_plane_of_sky_mean")])
    with (ARTIFACTS / "phase17_r1od3_sessions.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["k", "strategy", "pattern", "session", "pair", "window_start_s", "window_end_s",
                    "session_start_s", "session_end_s", "np_times_s", "raw_conflicts", "gm_dropped"])
        for r in stage_a:
            for key, c in r["cells"].items():
                if not c.get("feasible"):
                    continue
                st, pt = key.split("|")
                for j, s in enumerate(c["sessions"]):
                    w.writerow([r["k"], st, pt, j, s["pair"], s["window"][0], s["window"][1], s["interval"][0],
                                s["interval"][1], ";".join("%.0f" % t for t in s["np_t"]),
                                c["raw_conflicts"], c["gm_dropped"]])
    prim_fields = ["k", "synodic_phase_deg", "strategy", "pattern", "calibration", "feasible", "n_range_obs",
                   "n_normal_points", "n_sessions", "session_hours", "station_hours", "range_f_perp",
                   "combined_f_perp", "delta_f_perp", "relative_class", "absolute_class", "theta_k_deg",
                   "conditional_k_information", "sigma_k", "sigma_k_frac", "k_column_norm", "design_condition",
                   "weakest_mode_k_component", "design_singular_values"]
    bias_fields = ["k", "strategy", "pattern", "calibration", "n_sessions", "delta_f_perp", "bias_sigma_ns",
                   "max_abs_k_bias_corr", "sigma_k_inflation_vs_C0", "delta_loss_vs_C0", "retained_fraction_vs_C0"]
    with (ARTIFACTS / "phase17_r1od3_primary_results.csv").open("w", newline="") as fp, \
            (ARTIFACTS / "phase17_r1od3_bias_results.csv").open("w", newline="") as fb:
        wp, wb = csv.writer(fp), csv.writer(fb)
        wp.writerow(prim_fields)
        wb.writerow(bias_fields)
        for r in stage_a:
            for key, c in r["cells"].items():
                st, pt = key.split("|")
                if not c.get("feasible"):
                    wp.writerow([r["k"], 15.0 * r["k"], st, pt, "", False] + [""] * (len(prim_fields) - 6))
                    continue
                for cal, m in c["cal"].items():
                    wp.writerow([r["k"], 15.0 * r["k"], st, pt, cal, True,
                                 r["diagnostics"]["range_only"]["n_obs"], c["n_np"], c["n_sessions"],
                                 c["session_hours"], c["station_hours"], r["range_f_perp"], m["f_perp"],
                                 m["delta_f_perp"], m["relative_class"], m["absolute_class"], m["theta_k_deg"],
                                 m["conditional_k_information"], m["sigma_k"], m["sigma_k_frac"],
                                 m["k_column_norm"], m["design_condition"], m["weakest_mode_k_component"],
                                 ";".join("%.6g" % v for v in m["design_singular_values"])])
                    if cal != "C0_IDEAL":
                        wb.writerow([r["k"], st, pt, cal, c["n_sessions"], m["delta_f_perp"],
                                     ";".join("%.4g" % v for v in m.get("bias_sigma_ns", [])),
                                     m.get("max_abs_k_bias_corr"), m.get("sigma_k_inflation_vs_C0"),
                                     m.get("delta_loss_vs_C0"), m.get("retained_fraction_vs_C0")])
    with (ARTIFACTS / "phase17_r1od3_joint_baseline_results.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["k", "synodic_phase_deg"] + ["%s_%s" % (s, c) for s in strategies for c in
                                                   ("C0_IDEAL", D3.C1_NOMINAL, "C2_FREE")]
                   + ["joint_raw_conflicts", "joint_gm_dropped", "joint_minus_gc_C1"])
        for k, r in enumerate(stage_a):
            jc = r["cells"].get("GC_PLUS_GM|ALL_GC_PLUS_ALL_GM", {})
            dj, dg = delta["GC_PLUS_GM"][D3.C1_NOMINAL][k], delta["GC_ONLY"][D3.C1_NOMINAL][k]
            w.writerow([k, 15.0 * k] + [delta[s][c][k] for s in strategies for c in
                                        ("C0_IDEAL", D3.C1_NOMINAL, "C2_FREE")]
                       + [jc.get("raw_conflicts"), jc.get("gm_dropped"),
                          (dj - dg) if dj is not None and dg is not None else None])

    # ---------------- counts, windows
    counts = {s: {lab: sum(cls(d) == lab for d in delta[s][D3.C1_NOMINAL])
                  for lab in ("STRONG", "MODEST", "MAGNITUDE_ONLY", "DIRECTION_LOSS")} for s in strategies}
    counts_c0 = {s: sum(cls(d) == "STRONG" for d in delta[s]["C0_IDEAL"]) for s in strategies}
    counts_c2 = {s: sum(cls(d) == "STRONG" for d in delta[s]["C2_FREE"]) for s in strategies}
    feas = {s: sum(d is not None for d in delta[s][D3.C1_NOMINAL]) for s in strategies}

    # ---------------- joint baseline effect (predeclared order)
    diffs = [dj - dg for dj, dg in zip(delta["GC_PLUS_GM"][D3.C1_NOMINAL], delta["GC_ONLY"][D3.C1_NOMINAL])
             if dj is not None and dg is not None]
    med_diff = float(np.median(diffs)) if diffs else float("nan")
    gc_strong_eps = [k for k, f in enumerate(strong["GC_ONLY"][D3.C1_NOMINAL]) if f]
    med_diff_on_gc_strong = (float(np.median([delta["GC_PLUS_GM"][D3.C1_NOMINAL][k] - delta["GC_ONLY"][D3.C1_NOMINAL][k]
                                              for k in gc_strong_eps if delta["GC_PLUS_GM"][D3.C1_NOMINAL][k] is not None]))
                             if gc_strong_eps else float("nan"))
    lw = lambda s: max([b - a + 1 for a, b in win[s][D3.C1_NOMINAL]], default=0)  # noqa: E731

    def retained(s, k):
        d0, d2 = delta[s]["C0_IDEAL"][k], delta[s]["C2_FREE"][k]
        return d2 / d0 if d0 is not None and d2 is not None and d0 > 0.02 else None
    ret_diff = [retained("GC_PLUS_GM", k) - retained("GC_ONLY", k) for k in range(n_ep)
                if retained("GC_PLUS_GM", k) is not None and retained("GC_ONLY", k) is not None]
    med_ret = float(np.median(ret_diff)) if ret_diff else float("nan")
    cj, cg = counts["GC_PLUS_GM"]["STRONG"], counts["GC_ONLY"]["STRONG"]
    if cj < cg and med_diff < -0.01:
        jbe = "DEGRADES_DIRECTION"
    elif lw("GC_PLUS_GM") > lw("GC_ONLY") or cj >= cg + 2:
        jbe = "BROADENS_WINDOW"
    elif np.isfinite(med_diff_on_gc_strong) and med_diff_on_gc_strong >= 0.01:
        jbe = "STRENGTHENS_EXISTING_WINDOW"
    elif np.isfinite(med_ret) and med_ret >= 0.10 and not (med_diff >= 0.01):
        jbe = "CALIBRATION_HELP_ONLY"
    else:
        jbe = "NO_MATERIAL_BENEFIT"

    # ---------------- calibration effects (GC_ONLY primary)
    r10 = [delta["GC_ONLY"][D3.C1_NOMINAL][k] / delta["GC_ONLY"]["C0_IDEAL"][k] for k in range(n_ep)
           if delta["GC_ONLY"]["C0_IDEAL"][k] is not None and delta["GC_ONLY"]["C0_IDEAL"][k] > 0.02]
    r21 = [delta["GC_ONLY"]["C2_FREE"][k] / delta["GC_ONLY"][D3.C1_NOMINAL][k] for k in range(n_ep)
           if delta["GC_ONLY"][D3.C1_NOMINAL][k] is not None and delta["GC_ONLY"][D3.C1_NOMINAL][k] > 0.02]
    m10 = float(np.median(r10)) if r10 else float("nan")
    m21 = float(np.median(r21)) if r21 else float("nan")
    lab = lambda m, pre: ("%s_RETAINS_MOST" % pre if m > 0.75 else "%s_RETAINS_PART" % pre if m >= 0.25  # noqa: E731
                          else "%s_REMOVES_MOST" % pre) if np.isfinite(m) else "NOT_EVALUABLE"
    cal_effects = dict(IDEAL_VS_NOMINAL=dict(label=lab(m10, "NOMINAL"), median_ratio=m10, n=len(r10)),
                       NOMINAL_VS_FREE=dict(label=lab(m21, "FREE"), median_ratio=m21, n=len(r21)))
    prior_bracket = {c: {s: counts_c for s, counts_c in
                         ((s, sum(cls(d) == "STRONG" for d in delta[s][c])) for s in strategies)}
                     for c in D3.CALIBRATIONS}

    # ---------------- session placement / count (C1 nominal), cost trade-offs
    placement = {}
    for s, pats in (("GC_ONLY", D3.PATTERNS_SINGLE), ("GM_ONLY", D3.PATTERNS_SINGLE),
                    ("GC_PLUS_GM", tuple(D3.PATTERNS_JOINT))):
        for p in pats:
            ds, hrs, ns = [], [], []
            for r in stage_a:
                c = r["cells"].get("%s|%s" % (s, p), {})
                if c.get("feasible"):
                    ds.append(c["cal"][D3.C1_NOMINAL]["delta_f_perp"]); hrs.append(c["session_hours"])
                    ns.append(c["n_sessions"])
            placement["%s|%s" % (s, p)] = dict(
                feasible_epochs=len(ds), strong_epochs=int(sum(d > 0.05 for d in ds)),
                median_delta=float(np.median(ds)) if ds else None, max_delta=float(max(ds)) if ds else None,
                median_session_hours=float(np.median(hrs)) if hrs else None,
                median_sessions=float(np.median(ns)) if ns else None,
                longest_window=max([b - a + 1 for a, b in windows([d is not None and d > 0.05 for d in
                                    [(r["cells"].get("%s|%s" % (s, p), {}).get("cal", {}).get(D3.C1_NOMINAL, {})
                                      .get("delta_f_perp")) for r in stage_a]], 3)], default=0))

    # ---------------- predictors (predeclared set, Spearman)
    def pred_rows(strategy):
        pair = "GC" if strategy == "GC_ONLY" else "GM"
        out = []
        for k, r in enumerate(stage_a):
            c = r["cells"].get("%s|ALL_FEASIBLE_SESSIONS" % strategy, {})
            dg = r["diagnostics"]["pairs"][pair]
            g = r["diagnostics"]["geometry"]
            out.append(dict(response=delta[strategy][D3.C1_NOMINAL][k],
                            ddor_only_f_perp=dg.get("ddor_only_dense_delay_f_perp"),
                            range_f_perp=r["range_f_perp"],
                            projected_baseline_km=(c.get("projected_baseline_km") or {}).get(pair),
                            sun_moon_earth_deg=g["sun_moon_earth_deg"],
                            earth_los_vs_srp_direction_deg=g["earth_los_vs_srp_direction_deg"],
                            n_feasible_sessions=dg["n_sessions"],
                            n_normal_points=c.get("n_np", 0),
                            abs_cos_baseline_vs_k=dg.get("abs_cos_baseline_vs_k_plane_of_sky_mean")))
        return out
    names = ("ddor_only_f_perp", "range_f_perp", "projected_baseline_km", "sun_moon_earth_deg",
             "earth_los_vs_srp_direction_deg", "n_feasible_sessions", "n_normal_points", "abs_cos_baseline_vs_k")
    pred = {}
    pooled = pred_rows("GC_ONLY") + pred_rows("GM_ONLY")
    for nm in names:
        f = lambda rows: spearman([x[nm] if x[nm] is not None else np.nan for x in rows],  # noqa: E731
                                  [x["response"] if x["response"] is not None else np.nan for x in rows])
        rho_p = f(pooled)
        pred[nm] = dict(rho_gc=f(pred_rows("GC_ONLY")), rho_gm=f(pred_rows("GM_ONLY")), rho_pooled=rho_p,
                        strength=strength(rho_p))
    best = max(pred, key=lambda n: abs(pred[n]["rho_pooled"]) if np.isfinite(pred[n]["rho_pooled"]) else -1)
    best_label = best if pred[best]["strength"] in ("STRONG", "MODERATE") else "NONE_ADEQUATE"
    (ARTIFACTS / "phase17_r1od3_predictor_analysis.json").write_text(json.dumps(_clean(dict(
        response="delta_f_perp, C1 nominal, ALL_FEASIBLE_SESSIONS, per epoch",
        candidates=pred, best_by_abs_pooled_rho=best, BEST_SUPPORTED_PREDICTOR=best_label,
        PREDICTOR_STRENGTH=pred[best]["strength"], data=dict(GC=pred_rows("GC_ONLY"), GM=pred_rows("GM_ONLY")))),
        indent=1), encoding="utf-8")

    # ---------------- Stage B orbit robustness
    orb = {}
    for r in stage_b:
        for s in strategies:
            c = r["cells"].get("%s|%s" % (s, D3.PRIMARY_PATTERN[s]), {})
            orb.setdefault(r["orbit"], {}).setdefault("E%02d" % r["k"], {})[s] = dict(
                range_f_perp=r["range_f_perp"],
                delta_C0=c.get("cal", {}).get("C0_IDEAL", {}).get("delta_f_perp") if c.get("feasible") else None,
                delta_C1=c.get("cal", {}).get(D3.C1_NOMINAL, {}).get("delta_f_perp") if c.get("feasible") else None,
                n_sessions=c.get("n_sessions"))
    # canonical orbit rows for the same epochs come from Stage A
    for k in b_epochs:
        for s in strategies:
            orb.setdefault("CANON", {}).setdefault("E%02d" % k, {})[s] = dict(
                range_f_perp=stage_a[k]["range_f_perp"], delta_C0=delta[s]["C0_IDEAL"][k],
                delta_C1=delta[s][D3.C1_NOMINAL][k])
    if longest is None:
        orbit_class = "NO_WINDOW_TO_TEST"
        kept = None
    else:
        s_w = longest[0]
        kept = sum(1 for o in orb if o != "CANON"
                   and (orb[o]["E%02d" % window_centre][s_w]["delta_C1"] or -1) > 0.05)
        orbit_class = "PRESERVES" if kept >= 4 else "PARTIAL" if kept >= 2 else "FAILS"
    strong_by_orbit = {o: {e: {s: cls(v[s]["delta_C1"]) for s in strategies} for e, v in ev.items()}
                       for o, ev in orb.items()}
    (ARTIFACTS / "phase17_r1od3_orbit_robustness.json").write_text(json.dumps(_clean(dict(
        epochs=b_epochs, window_centre=window_centre, window_strategy=longest[0] if longest else None,
        orbits_keeping_strong_at_window_centre=kept, ORBIT_ROBUSTNESS_CLASS=orbit_class,
        results=orb, relative_class_C1=strong_by_orbit)), indent=1), encoding="utf-8")

    # ---------------- Stage C high fidelity
    hf = {}
    changed = False
    for name, (s, k) in hf_cases.items():
        r = next(x for x in stage_c if x["k"] == k)
        c = r["cells"].get("%s|%s" % (s, D3.PRIMARY_PATTERN[s]), {})
        d_hf = c.get("cal", {}).get(D3.C1_NOMINAL, {}).get("delta_f_perp") if c.get("feasible") else None
        d_si = delta[s][D3.C1_NOMINAL][k]
        same = (cls(d_hf) == cls(d_si)) or (d_hf is not None and d_si is not None and abs(d_hf - d_si) < 0.005)
        changed |= not same
        hf[name] = dict(strategy=s, k=k, delta_C1_simplified=d_si, delta_C1_fidelity=d_hf,
                        class_simplified=cls(d_si), class_fidelity=cls(d_hf), preserved=same,
                        range_f_simplified=stage_a[k]["range_f_perp"], range_f_fidelity=r["range_f_perp"],
                        delta_C0_fidelity=c.get("cal", {}).get("C0_IDEAL", {}).get("delta_f_perp")
                        if c.get("feasible") else None)
    hf_class = "CHANGES_PRIMARY_INTERPRETATION" if changed else "PRESERVES_PRIMARY_INTERPRETATION"
    (ARTIFACTS / "phase17_r1od3_high_fidelity.json").write_text(json.dumps(_clean(dict(
        cases=hf, TRACKING_DESIGN_HIGH_FIDELITY_CLASS=hf_class,
        force_model=grid["stage_C_high_fidelity"]["force_model"])), indent=1), encoding="utf-8")

    # ---------------- real-source overlay
    overlay = dict(REAL_SOURCE_AVAILABILITY="UNRESOLVED", reason="catalog not supplied")
    if catalog and Path(catalog).exists():
        import hashlib
        cat = parse_catalog(Path(catalog))
        if cat is not None:
            seps = []
            for r in stage_a:
                for s in strategies:
                    c = r["cells"].get("%s|%s" % (s, D3.PRIMARY_PATTERN[s]), {})
                    for sess in c.get("sessions", []) if c.get("feasible") else []:
                        u = np.asarray(sess["s_geo_mid"], float)
                        seps.append(float(np.degrees(np.arccos(np.clip(np.max(cat @ u), -1, 1)))))
            within = sum(x <= math.degrees(0.1) for x in seps)
            overlay = dict(catalog="DSN 810-005 Module 107 Rev. F Table 3 (ICRF3-SX), coordinates only",
                           catalog_pdf_sha256=hashlib.sha256(Path(catalog).read_bytes()).hexdigest(),
                           n_sources_parsed=int(cat.shape[0]), n_primary_sessions=len(seps),
                           sessions_with_source_within_0p1_rad=within,
                           nearest_source_sep_deg=dict(median=float(np.median(seps)), max=float(max(seps))),
                           REAL_SOURCE_AVAILABILITY=("COORDINATE_AVAILABLE_ALL_SESSIONS" if within == len(seps)
                                                     else "COORDINATE_AVAILABLE_%d_OF_%d" % (within, len(seps))),
                           not_assessed=["flux", "SNR", "structure", "Ka/X interchangeability"])

    # ---------------- decision tree
    robust = [(s, w) for s in strategies for w in win[s][D3.C1_NOMINAL]]
    has_window = bool(robust)
    robust_ok = has_window and orbit_class == "PRESERVES" and hf_class == "PRESERVES_PRIMARY_INTERPRETATION"
    c1_strong_total = sum(counts[s]["STRONG"] for s in strategies)
    c0_strong_total = sum(counts_c0.values())
    c0_window = any(win[s]["C0_IDEAL"] for s in strategies)
    if robust_ok:
        primary = "DDOR_TRACKING_DESIGN_HAS_ROBUST_FAVORABLE_WINDOWS"
    elif c1_strong_total >= 2:
        primary = "DDOR_TRACKING_GAIN_REMAINS_CONFINED_TO_NARROW_GEOMETRIC_WINDOWS"
    elif c0_window and c1_strong_total <= 0.10 * c0_strong_total:
        primary = "DDOR_TRACKING_GEOMETRY_EXISTS_BUT_IS_CALIBRATION_LIMITED"
    else:
        primary = "DDOR_TRACKING_DESIGN_DOES_NOT_GENERALIZE_THE_D2_GAIN"

    # absolute identifiability
    abs_labels = [r["cells"][key]["cal"][D3.C1_NOMINAL]["absolute_class"] for r in stage_a
                  for key in ("GC_ONLY|ALL_FEASIBLE_SESSIONS", "GM_ONLY|ALL_FEASIBLE_SESSIONS",
                              "GC_PLUS_GM|ALL_GC_PLUS_ALL_GM") if r["cells"].get(key, {}).get("feasible")]
    weakest = [r["cells"][key]["cal"][D3.C1_NOMINAL]["weakest_mode_k_component"] for r in stage_a
               for key in ("GC_ONLY|ALL_FEASIBLE_SESSIONS", "GM_ONLY|ALL_FEASIBLE_SESSIONS",
                           "GC_PLUS_GM|ALL_GC_PLUS_ALL_GM") if r["cells"].get(key, {}).get("feasible")]
    abs_counts = {lab_: abs_labels.count(lab_) for lab_ in ("STRONG_K_SEPARATION", "MODERATE_K_SEPARATION",
                                                              "WEAK_K_SEPARATION")}
    abs_class = ("STRONG_K_SEPARATION" if abs_counts["STRONG_K_SEPARATION"] else
                 "MODERATE_K_SEPARATION" if abs_counts["MODERATE_K_SEPARATION"] else "WEAK_K_SEPARATION")

    rw = None
    if longest is not None:
        s_w, (a, b) = longest
        rw = dict(strategy=s_w, start_k=a, end_k=b, start_phase_deg=15.0 * a, end_phase_deg=15.0 * b,
                  grid_width=b - a + 1, approx_duration_days=(b - a + 1) * spacing_d)

    summary = dict(
        phase="17-R1O-D3", grid_sha256=grid["canonical_sha256"], elapsed_s=elapsed,
        contracts_all_passed=contracts_ok, independent_rebuild=dict(n=len(rebuild), max_abs_diff=rebuild_max),
        canonical_continuity=cont,
        primary_epoch_count=n_ep, synodic_phase_spacing_deg=15.0, spacing_days=spacing_d,
        feasible_epochs=feas, counts_C1_nominal=counts, strong_counts_C0=counts_c0, strong_counts_C2=counts_c2,
        strong_counts_by_calibration=prior_bracket,
        windows_C1_nominal={s: win[s][D3.C1_NOMINAL] for s in strategies},
        windows_C0={s: win[s]["C0_IDEAL"] for s in strategies},
        windows_C2={s: win[s]["C2_FREE"] for s in strategies},
        longest_window=rw, window_centre=window_centre,
        ROBUST_WINDOW_FOUND=bool(robust_ok),
        JOINT_BASELINE_EFFECT=jbe, joint_stats=dict(median_joint_minus_gc_C1=med_diff,
                                                    median_on_gc_strong=med_diff_on_gc_strong,
                                                    median_C2_retained_diff=med_ret,
                                                    longest_window_gc=lw("GC_ONLY"), longest_window_joint=lw("GC_PLUS_GM")),
        calibration_effects=cal_effects, placement_and_count=placement,
        BEST_SUPPORTED_PREDICTOR=best_label, PREDICTOR_STRENGTH=pred[best]["strength"],
        best_predictor_rho=pred[best]["rho_pooled"],
        ABSOLUTE_K_IDENTIFIABILITY_CLASS=abs_class, absolute_counts=abs_counts,
        K_WEAKEST_MODE_FRACTION=dict(min=float(min(weakest)), median=float(np.median(weakest))),
        ORBIT_ROBUSTNESS_CLASS=orbit_class, TRACKING_DESIGN_HIGH_FIDELITY_CLASS=hf_class,
        hf_cases={k: list(v) for k, v in hf_cases.items()},
        real_source_overlay=overlay, PRIMARY_CLASS=primary,
        NONLINEAR_ESTIMATOR_CONFIRMATION_JUSTIFIED="YES" if primary.endswith("ROBUST_FAVORABLE_WINDOWS") else "NO",
        delta_series={s: {c: delta[s][c] for c in D3.CALIBRATIONS} for s in strategies},
    )
    (ARTIFACTS / "phase17_r1od3_summary.json").write_text(json.dumps(_clean(summary), indent=1), encoding="utf-8")
    # raw per-geometry results kept outside git-tracked artifacts would lose evidence; keep compact copy
    (ARTIFACTS / "phase17_r1od3_raw_results.json").write_text(json.dumps(_clean(dict(
        stage_a=stage_a, stage_b=stage_b, stage_c=stage_c)), separators=(",", ":")), encoding="utf-8")
    figures(stage_a, delta, spacing_d)
    print(json.dumps(_clean({k: summary[k] for k in (
        "contracts_all_passed", "independent_rebuild", "canonical_continuity", "counts_C1_nominal",
        "strong_counts_C0", "strong_counts_C2", "windows_C1_nominal", "windows_C0", "longest_window",
        "JOINT_BASELINE_EFFECT", "joint_stats", "calibration_effects", "BEST_SUPPORTED_PREDICTOR",
        "PREDICTOR_STRENGTH", "ABSOLUTE_K_IDENTIFIABILITY_CLASS", "ORBIT_ROBUSTNESS_CLASS",
        "TRACKING_DESIGN_HIGH_FIDELITY_CLASS", "PRIMARY_CLASS", "real_source_overlay")}), indent=1))


def figures(stage_a, delta, spacing_d) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ph = [15.0 * r["k"] for r in stage_a]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for ax, s in zip(axes, ("GC_ONLY", "GM_ONLY", "GC_PLUS_GM")):
        for c, st in (("C0_IDEAL", "o-"), (D3.C1_NOMINAL, "s-"), ("C2_FREE", "^-")):
            ys = [np.nan if v is None else v for v in delta[s][c]]
            ax.plot(ph, ys, st, ms=4, label=c)
        ax.axhline(0.05, color="k", lw=0.8, ls="--")
        ax.axhline(-0.02, color="grey", lw=0.8, ls=":")
        ax.set_ylabel("delta f_perp")
        ax.set_title(s + " (primary pattern)")
        ax.legend(fontsize=7)
    axes[-1].set_xlabel("synodic phase from canonical epoch [deg] (%.2f d per step)" % spacing_d)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "phase17_r1od3_delta_vs_phase.png", dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for s, mk in (("GC_ONLY", "o"), ("GM_ONLY", "s"), ("GC_PLUS_GM", "^")):
        xs, ys = [], []
        for r in stage_a:
            for key, c in r["cells"].items():
                if key.startswith(s + "|") and c.get("feasible"):
                    xs.append(c["session_hours"]); ys.append(c["cal"][D3.C1_NOMINAL]["delta_f_perp"])
        ax.scatter(xs, ys, marker=mk, s=12, label=s, alpha=0.7)
    ax.axhline(0.05, color="k", lw=0.8, ls="--")
    ax.set_xlabel("session-hours in schedule")
    ax.set_ylabel("delta f_perp (C1 nominal)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "phase17_r1od3_delta_vs_session_hours.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
