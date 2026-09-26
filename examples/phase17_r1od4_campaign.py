"""PHASE 17-R1O-D4 - Goldstone-Madrid narrow-window confirmatory robustness qualification.

Every rule comes from artifacts/phase17_r1od4_predeclaration.json (hash-pinned,
committed before any D4 metric).  GM_ONLY; FIRST_SESSION (primary) and
ALL_FEASIBLE_SESSIONS (reference); 22 phases 60-165 deg; 5 orbits; SIMPLIFIED and
HIGH_FIDELITY dynamics for every cell; C0 / C1 half, nominal, double / C2.

The session, stacking and information machinery is phase17_r1od3_core, unchanged.
``--catalog PATH`` enables the DSN-107F coordinate overlay.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
PRE_PATH = ARTIFACTS / "phase17_r1od4_predeclaration.json"
PREDECLARATION_SHA256 = "3a48bc212f2b5b0b26f15953dfd6c2e2cc230a2fdf042cb9cd1e77772c8b9084"
CANONICAL_ET0 = 857806356.99998
SYNODIC_S = 2551442.8896
CAL_MAP = {"C0_IDEAL": "C0_IDEAL", "C1_HALF": "C1_x0.5", "C1_NOMINAL": "C1_x1.0", "C1_DOUBLE": "C1_x2.0",
           "C2_FREE": "C2_FREE"}
SCHEDULES = ("FIRST_SESSION", "ALL_FEASIBLE_SESSIONS")
N_WORKERS = 15


class PredeclarationMismatch(SystemExit):
    """The D4 predeclaration on disk is not the committed one."""


def load_predeclaration(path: Path = PRE_PATH) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in d.items() if k != "canonical_sha256"}
    got = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if not (got == d.get("canonical_sha256") == PREDECLARATION_SHA256):
        raise PredeclarationMismatch("D4 predeclaration hash mismatch: file %s stored %s pinned %s"
                                     % (got, d.get("canonical_sha256"), PREDECLARATION_SHA256))
    return d


def phase_et(phase_deg: float) -> float:
    """The predeclared convention: identical to D3's k * S / 24 at phase = 15 k."""
    return CANONICAL_ET0 + (phase_deg / 15.0) * SYNODIC_S / 24


def windows(flags: list, min_len: int) -> list:
    """Maximal runs of True of length >= min_len: (first_index, last_index)."""
    out, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None
    return out


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
# One cell: orbit x phase x dynamics
# ======================================================================
def run_cell(spec: dict) -> dict:
    D2 = D3.D2
    grid = D3.load_grid()
    t0 = time.time()
    et0 = phase_et(spec["phase"])
    _, t_orbit = D2.M.campaign_epoch()
    scn = D2.Scenario("D4_%s_%s_%05.1f" % (spec["orbit"], spec["dynamics"], spec["phase"]), "d4", et0,
                      tuple(spec["x0"]), 15.0 * t_orbit)
    if spec["dynamics"] == "HIGH_FIDELITY":
        from phase17_r1od2_fidelity import propagate_fidelity
        nom = propagate_fidelity(scn)
    else:
        nom = D2.propagate_nom48(scn)
    rb = D2.build_range_block(scn, nom)
    audit = D2.earth_epoch_audit(scn, rb)
    m_range = D2.info_metrics([(rb.h_x0, rb.h_k, rb.w)])
    f0 = m_range["f_perp"]
    rows = D2.build_ddor_rows(scn, rb, "GM", D2.ROW_RULE_PRODUCTION_VISIBILITY)
    sessions = D3.build_sessions(rows, grid)
    t_mid = 0.5 * float(rb.t_grid[-1])
    ia, ib = (D2.DSN3.index(n) for n in D2.PAIRS["GM"])

    import phase17_geo_core as G

    def geo_at(i):
        """(spacecraft geocentric declination, Sun-Earth-spacecraft angle) at row i of the GM rows."""
        s = rows.s_sc_geo[i]
        t = float(rows.t_used[i])
        sun_geo = G.sun_mci(et0 + t) - rb.earth.pos([t])[0]
        ses = math.degrees(math.acos(float(np.clip(np.dot(sun_geo / np.linalg.norm(sun_geo), s), -1, 1))))
        return math.degrees(math.asin(float(np.clip(s[2], -1, 1)))), ses

    cells = {}
    for sched_name in SCHEDULES:
        chosen = D3.select_pattern(sessions, sched_name, t_mid)
        if chosen is None:
            cells[sched_name] = dict(feasible=False)
            continue
        s = D3.stack("GM_ONLY", sched_name, -1, chosen, {"GM": rows})
        # contract: every normal point dual-visible, owned once, bias columns exact
        np_rows = rows.idx[s.parent_row]
        vis_ok = bool(np.all(rb.vis[np_rows, ia] & rb.vis[np_rows, ib]))
        own_ok = bool(np.array_equal(s.h_x0, rows.h_x0[s.parent_row]) and np.array_equal(s.h_k, rows.h_k[s.parent_row]))
        for j, sess in enumerate(s.sessions):
            own_ok &= bool(set(s.parent_row[s.session == j].tolist()) == set(sess.np_index))
        bc = D3.bias_columns(rb.n_obs, s)
        own_ok &= bool(np.all(bc[:rb.n_obs] == 0) and np.all(bc[rb.n_obs:].sum(axis=1) == 1))
        c = dict(feasible=True, n_np=s.n, n_sessions=s.n_sessions, session_hours=s.session_hours,
                 np_visible=vis_ok, ownership_ok=own_ok,
                 sessions=[dict(window=[x.window_start, x.window_end], interval=[x.start, x.end],
                                np_t=[float(rows.t_used[i]) for i in x.np_index],
                                s_geo_mid=rows.s_sc_geo[x.np_index[len(x.np_index) // 2]].tolist())
                           for x in s.sessions],
                 projected_baseline_km=D3.projected_baseline_km(rows, s.parent_row), cal={}, rebuild={})
        for cname, core in CAL_MAP.items():
            m = D3.schedule_metrics(rb, s, core, grid, f0)
            c["cal"][cname] = m
            c["rebuild"][cname] = abs(D3.independent_f_perp(rb, s, core, grid) - m["f_perp"])
        first = s.sessions[0]
        sess_rows = np.asarray(first.row_index, int)
        if sess_rows.size >= 7:
            c["first_session_ddor_only_f_perp"] = D2.info_metrics(
                [(rows.h_x0[sess_rows], rows.h_k[sess_rows], D3.delay_weight(sess_rows.size, grid))])["f_perp"]
        dec, ses = geo_at(first.np_index[1])
        c["first_midpoint_sc_declination_deg"] = dec
        c["first_midpoint_sun_earth_sc_deg"] = ses
        cells[sched_name] = c

    geo = D2.geometry_descriptors(scn, rb)
    qual_hours = float(sum(x.window_end - x.window_start for x in sessions) / 3600.0)
    diag = dict(range_f_perp=f0, range_n_obs=int(rb.n_obs), range_sigma_k_frac=m_range["sigma_k_frac"],
                gm_dense_rows=int(rows.n), gm_rows_candidates=int(rows.n_candidates), gm_rows_converged=int(rows.n_converged),
                gm_visibility_windows=int(rows.n_sessions), gm_qualified_sessions=len(sessions),
                gm_qualified_dual_visibility_hours=qual_hours,
                moon_geocentric_declination_deg=geo["moon_geocentric_declination_deg"],
                sun_moon_earth_deg=geo["sun_moon_earth_deg"],
                earth_audit_passed=bool(audit["passed"]),
                all_gm_rows_dual_visible=D3.pair_row_contract(rows, rb, "GM"))
    if rows.n >= 7:
        diag["gm_dense_ddor_only_f_perp"] = D2.info_metrics(
            [(rows.h_x0, rows.h_k, D3.delay_weight(rows.n, grid))])["f_perp"]
    return _clean(dict(orbit=spec["orbit"], phase=spec["phase"], dynamics=spec["dynamics"], et0=et0,
                       diag=diag, cells=cells, elapsed_s=time.time() - t0))


# ======================================================================
# Analysis
# ======================================================================
def delta(res, sched, cal):
    c = res["cells"].get(sched, {})
    return c["cal"][cal]["delta_f_perp"] if c.get("feasible") else None


def is_strong(d):
    return d is not None and d > 0.05


def cls(d):
    if d is None:
        return "INFEASIBLE"
    return D3.D2.relative_gain_class(d)


def window_report(phases, deltas, pre):
    n = pre["window"]["required_consecutive_points"]
    ws = windows([is_strong(d) for d in deltas], n)
    lo, hi = pre["window"]["overlap_region_deg"]
    out = []
    for a, b in ws:
        vals = deltas[a:b + 1]
        out.append(dict(start_deg=phases[a], end_deg=phases[b], width_deg=phases[b] - phases[a],
                        n_points=b - a + 1, duration_days=(phases[b] - phases[a]) / 360.0 * SYNODIC_S / 86400.0,
                        max_delta=max(vals), median_delta=float(np.median(vals)),
                        overlaps_candidate=bool(phases[a] <= hi and phases[b] >= lo)))
    return out


def r2_status(phases, nominal_window, deltas, pre):
    """Fate of a nominal window W under another calibration (labels operationalised here,
    names from the predeclaration): PERSISTS = every W point STRONG; SHRINKS = a >= 7-point
    window still intersects W but no longer covers it; FRAGMENTS = no such window but
    >= 2 STRONG points remain inside W; DISAPPEARS = otherwise."""
    i0, i1 = phases.index(nominal_window["start_deg"]), phases.index(nominal_window["end_deg"])
    flags = [is_strong(d) for d in deltas]
    if all(flags[i0:i1 + 1]):
        return "PERSISTS"
    ws = windows(flags, pre["window"]["required_consecutive_points"])
    if any(a <= i1 and b >= i0 for a, b in ws):
        return "SHRINKS"
    if sum(flags[i0:i1 + 1]) >= 2:
        return "FRAGMENTS"
    return "DISAPPEARS"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--analyse-only", action="store_true", help="re-run the analysis on the stored raw results")
    args = ap.parse_args()
    pre = load_predeclaration()
    D3.load_grid()
    t_start = time.time()
    phases = pre["phase_grid"]["phases_deg"]
    order = ["CANON", "O1_LOW_BETA_ECLIPSE_RICH", "O2_FULL_SUN_HIGH_BETA", "O3_HIGH_ALTITUDE_500KM",
             "O5_FAVOURABLE_GEO_MAX_FPERP"]
    if args.analyse_only:
        results = json.loads((ARTIFACTS / "phase17_r1od4_raw_results.json").read_text(encoding="utf-8"))
        analyse(pre, results, phases, order, args.catalog, None)
        return
    d2 = json.loads((ARTIFACTS / "phase17_r1od2_predeclared_grid.json").read_text(encoding="utf-8"))
    orbits = {}
    for g in d2["geometries"]:
        if g["case_id"] in pre["orbits"]["set"]:
            h = hashlib.sha256(json.dumps(g["x0"]).encode()).hexdigest()
            assert h == pre["orbits"]["set"][g["case_id"]]["x0_sha256"], g["case_id"]
            orbits[g["case_id"]] = g["x0"]
    assert list(sorted(orbits)) == sorted(pre["orbits"]["set"])
    import phase17_r1m_core as M
    assert np.array_equal(np.asarray(orbits["CANON"]), M.campaign_initial_state())
    specs = [dict(orbit=o, x0=orbits[o], phase=p, dynamics=dyn)
             for dyn in ("HIGH_FIDELITY", "SIMPLIFIED") for o in order for p in phases]
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init) as ex:
        results = list(ex.map(run_cell, specs))
    print("campaign done in %.0f s" % (time.time() - t_start), flush=True)
    (ARTIFACTS / "phase17_r1od4_raw_results.json").write_text(json.dumps(results, separators=(",", ":")),
                                                               encoding="utf-8")
    analyse(pre, results, phases, order, args.catalog, time.time() - t_start)


def analyse(pre, results, phases, order, catalog, elapsed) -> None:
    by = {(r["orbit"], r["dynamics"], r["phase"]): r for r in results}
    series = lambda o, dyn, sched, cal: [delta(by[(o, dyn, p)], sched, cal) for p in phases]  # noqa: E731

    # ---------------- verification
    contract_ok = all(r["diag"]["earth_audit_passed"] and r["diag"]["all_gm_rows_dual_visible"]
                      and all(c.get("np_visible", True) and c.get("ownership_ok", True) for c in r["cells"].values())
                      for r in results)
    rebuild = [v for r in results for c in r["cells"].values() if c.get("feasible") for v in c["rebuild"].values()]
    et_diff = max(abs(phase_et(p) - (CANONICAL_ET0 + p / 360.0 * SYNODIC_S)) for p in phases)
    d3rows = list(csv.DictReader((ARTIFACTS / "phase17_r1od3_primary_results.csv").open()))
    anchor_diffs = []
    for p in pre["phase_grid"]["d3_anchors_deg"]:
        k = int(round(p / 15.0))
        for sched in SCHEDULES:
            for cname, core in CAL_MAP.items():
                ref = [r for r in d3rows if r["strategy"] == "GM_ONLY" and r["pattern"] == sched
                       and int(r["k"]) == k and r["calibration"] == core]
                mine = delta(by[("CANON", "SIMPLIFIED", p)], sched, cname)
                if ref and ref[0]["delta_f_perp"] != "":
                    anchor_diffs.append(dict(phase=p, schedule=sched, cal=cname, d3=float(ref[0]["delta_f_perp"]),
                                             d4=mine, abs_diff=abs(mine - float(ref[0]["delta_f_perp"]))))
                else:
                    anchor_diffs.append(dict(phase=p, schedule=sched, cal=cname, d3=None, d4=mine,
                                             abs_diff=0.0 if mine is None else float("inf")))
    anchors_ok = all(x["abs_diff"] <= 1e-12 for x in anchor_diffs)
    rb_ok = max(rebuild) <= 1e-10
    (ARTIFACTS / "phase17_r1od4_independent_rebuild.json").write_text(json.dumps(_clean(dict(
        A_predeclaration_hash=PREDECLARATION_SHA256, B_phase_to_et_max_abs_diff_s=et_diff,
        C_contracts_all_pass=contract_ok, DE_ownership_all_pass=contract_ok,
        F_rebuild=dict(n=len(rebuild), max_abs_f_perp_diff=max(rebuild), gate=rb_ok),
        G_d3_anchors=dict(n=len(anchor_diffs), max_abs_diff=max(x["abs_diff"] for x in anchor_diffs),
                          reproduced=anchors_ok, rows=anchor_diffs))), indent=1), encoding="utf-8")

    # ---------------- windows
    win = {}
    for o in order:
        for dyn in ("HIGH_FIDELITY", "SIMPLIFIED"):
            for sched in SCHEDULES:
                for cname in CAL_MAP:
                    win[(o, dyn, sched, cname)] = window_report(phases, series(o, dyn, sched, cname), pre)
    ok_w = lambda o, dyn, sched, cal: [w for w in win[(o, dyn, sched, cal)] if w["overlaps_candidate"]]  # noqa: E731
    r0 = ok_w("CANON", "HIGH_FIDELITY", "FIRST_SESSION", "C1_NOMINAL")
    r1 = {o: ok_w(o, "HIGH_FIDELITY", "FIRST_SESSION", "C1_NOMINAL") for o in order}
    n_orb = sum(1 for o in order if r1[o])
    orbit_class = ("ORBIT_ROBUST" if n_orb >= 4 else "ORBIT_DEPENDENT" if n_orb >= 2 else
                   "CANONICAL_ONLY" if n_orb == 1 else "NO_CONFIRMED_WINDOW")
    cal_w = {c: ok_w("CANON", "HIGH_FIDELITY", "FIRST_SESSION", c) for c in CAL_MAP}
    if cal_w["C1_NOMINAL"] and cal_w["C1_DOUBLE"]:
        cal_class = "CALIBRATION_ROBUST"
    elif cal_w["C1_NOMINAL"]:
        cal_class = "CALIBRATION_SENSITIVE"
    elif cal_w["C1_HALF"] or cal_w["C0_IDEAL"]:
        cal_class = "CALIBRATION_FRAGILE"
    else:
        cal_class = "NO_WINDOW_ANY_CALIBRATION"
    r2 = []
    for o in order:
        nominal = ok_w(o, "HIGH_FIDELITY", "FIRST_SESSION", "C1_NOMINAL")
        basis = nominal or ([dict(w, basis=c) for c in ("C0_IDEAL", "C1_HALF")
                             for w in ok_w(o, "HIGH_FIDELITY", "FIRST_SESSION", c)] if o == "CANON" else [])
        for w in basis:
            r2.append(dict(orbit=o, basis=w.get("basis", "C1_NOMINAL"), start_deg=w["start_deg"], end_deg=w["end_deg"],
                           fate={c: r2_status(phases, w, series(o, "HIGH_FIDELITY", "FIRST_SESSION", c), pre)
                                 for c in CAL_MAP},
                           windows_under={c: [(x["start_deg"], x["end_deg"]) for x in
                                              ok_w(o, "HIGH_FIDELITY", "FIRST_SESSION", c)] for c in CAL_MAP}))
    simp = ok_w("CANON", "SIMPLIFIED", "FIRST_SESSION", "C1_NOMINAL")
    if bool(simp) != bool(r0):
        fid_class = "FIDELITY_CHANGES_EXISTENCE"
    elif simp and r0:
        shift = max(abs(simp[0]["start_deg"] - r0[0]["start_deg"]), abs(simp[0]["end_deg"] - r0[0]["end_deg"]))
        fid_class = "FIDELITY_STABLE" if shift <= 5.0 else "FIDELITY_SHIFTS_WINDOW"
    else:
        fid_class = "FIDELITY_STABLE"
    hf_s = series("CANON", "HIGH_FIDELITY", "FIRST_SESSION", "C1_NOMINAL")
    si_s = series("CANON", "SIMPLIFIED", "FIRST_SESSION", "C1_NOMINAL")
    # descriptive (not a decision input): split the HF - SIMPLIFIED change of delta into the
    # range-only baseline change and the combined (range + DDOR) change
    fid_decomp = []
    for p in phases:
        rs, rh = by[("CANON", "SIMPLIFIED", p)], by[("CANON", "HIGH_FIDELITY", p)]
        if rs["cells"]["FIRST_SESSION"].get("feasible") and rh["cells"]["FIRST_SESSION"].get("feasible"):
            cs = rs["cells"]["FIRST_SESSION"]["cal"]["C1_NOMINAL"]["f_perp"]
            ch = rh["cells"]["FIRST_SESSION"]["cal"]["C1_NOMINAL"]["f_perp"]
            fs, fh = rs["diag"]["range_f_perp"], rh["diag"]["range_f_perp"]
            fid_decomp.append(dict(phase=p, range_f_simplified=fs, range_f_hf=fh, combined_f_simplified=cs,
                                   combined_f_hf=ch, d_range=fh - fs, d_combined=ch - cs,
                                   d_delta=(ch - fh) - (cs - fs)))
    class_changes = [dict(phase=p, simplified=cls(a), high_fidelity=cls(b)) for p, a, b in zip(phases, si_s, hf_s)
                     if cls(a) != cls(b)]

    # ---------------- confirmatory success and decision
    success = dict(
        c1_canonical_hf_window=bool(win[("CANON", "HIGH_FIDELITY", "FIRST_SESSION", "C1_NOMINAL")]),
        c2_overlaps_candidate=bool(r0), c3_orbits_ge_4=n_orb >= 4, c4_contract_valid=contract_ok,
        c5_not_c0_only=bool(r0), c6_rebuild=rb_ok, c7_predeclaration_unchanged=True)
    if all(success.values()):
        primary = "GM_NARROW_WINDOW_CONFIRMED_AND_ROBUST"
    elif r0:
        primary = "GM_WINDOW_EXISTS_BUT_IS_ORBIT_GEOMETRY_DEPENDENT"
    elif bool(simp) != bool(r0):
        primary = "GM_WINDOW_IS_DYNAMICS_FIDELITY_FRAGILE"
    elif cal_w["C0_IDEAL"] or cal_w["C1_HALF"]:
        primary = "GM_WINDOW_IS_CALIBRATION_FRAGILE"
    else:
        primary = "D3_SECONDARY_GM_WINDOW_DOES_NOT_CONFIRM"

    # ---------------- mechanism (canonical HF)
    mech_rows = []
    for p in phases:
        r = by[("CANON", "HIGH_FIDELITY", p)]
        f, a = r["cells"]["FIRST_SESSION"], r["cells"]["ALL_FEASIBLE_SESSIONS"]
        if not (f.get("feasible") and a.get("feasible")):
            continue
        mf, ma = f["cal"]["C1_NOMINAL"], a["cal"]["C1_NOMINAL"]
        row = dict(phase=p, n_sessions_all=a["n_sessions"], delta_first=mf["delta_f_perp"], delta_all=ma["delta_f_perp"],
                   d_diff=ma["delta_f_perp"] - mf["delta_f_perp"], ikk_first=mf["i_kk"], ikk_all=ma["i_kk"],
                   iperp_first=mf["conditional_k_information"], iperp_all=ma["conditional_k_information"],
                   theta_first=mf["theta_k_deg"], theta_all=ma["theta_k_deg"],
                   kbias_corr_first=mf.get("max_abs_k_bias_corr"), kbias_corr_all=ma.get("max_abs_k_bias_corr"),
                   projected_baseline_first_km=f["projected_baseline_km"], projected_baseline_all_km=a["projected_baseline_km"],
                   first_session_ddor_only_f_perp=f.get("first_session_ddor_only_f_perp"),
                   gm_dense_ddor_only_f_perp=r["diag"].get("gm_dense_ddor_only_f_perp"),
                   first_sc_declination_deg=f["first_midpoint_sc_declination_deg"],
                   first_sun_earth_sc_deg=f["first_midpoint_sun_earth_sc_deg"])
        if a["n_sessions"] > 1 and ma["i_kk"] > mf["i_kk"]:
            row["ikk_ratio"] = ma["i_kk"] / mf["i_kk"]
            row["added_orth_frac"] = (ma["conditional_k_information"] - mf["conditional_k_information"]) / (ma["i_kk"] - mf["i_kk"])
            row["first_orth_frac"] = mf["f_perp"] ** 2
        mech_rows.append(row)
    multi = [x for x in mech_rows if "ikk_ratio" in x]
    md = float(np.median([x["d_diff"] for x in multi])) if multi else float("nan")
    mr = float(np.median([x["ikk_ratio"] for x in multi])) if multi else float("nan")
    mao = float(np.median([x["added_orth_frac"] for x in multi])) if multi else float("nan")
    mfo = float(np.median([x["first_orth_frac"] for x in multi])) if multi else float("nan")
    if multi and md < -0.01 and mr > 1.1 and mao < mfo:
        mech_class = "LATER_SESSIONS_DILUTE_DIRECTION"
    elif multi and md > 0.01:
        mech_class = "LATER_SESSIONS_ADD_DIRECTION"
    else:
        mech_class = "NO_SYSTEMATIC_DIFFERENCE"
    (ARTIFACTS / "phase17_r1od4_mechanism.json").write_text(json.dumps(_clean(dict(
        population="CANON, HIGH_FIDELITY, C1_NOMINAL; phases where ALL has > 1 session",
        n_multi=len(multi), median_d_diff=md, median_ikk_ratio=mr, median_added_orth_frac=mao,
        median_first_orth_frac=mfo, FIRST_VS_ALL_MECHANISM_CLASS=mech_class, rows=mech_rows)), indent=1),
        encoding="utf-8")

    # ---------------- literature geometry overlay (canonical HF)
    lit = []
    for p in phases:
        r = by[("CANON", "HIGH_FIDELITY", p)]
        f = r["cells"]["FIRST_SESSION"]
        good = (f.get("feasible") and f["first_midpoint_sun_earth_sc_deg"] > 10.0
                and f["first_midpoint_sc_declination_deg"] > -15.0)
        lit.append(dict(phase=p, moon_dec_deg=r["diag"]["moon_geocentric_declination_deg"],
                        sc_dec_first_deg=f.get("first_midpoint_sc_declination_deg"),
                        sun_earth_sc_first_deg=f.get("first_midpoint_sun_earth_sc_deg"),
                        gm_qualified_hours=r["diag"]["gm_qualified_dual_visibility_hours"],
                        dsn_good_gm_geometry=bool(good), strong_first_c1=is_strong(delta(r, "FIRST_SESSION", "C1_NOMINAL"))))
    n_good = sum(x["dsn_good_gm_geometry"] for x in lit)
    n_strong = sum(x["strong_first_c1"] for x in lit)
    n_both = sum(x["dsn_good_gm_geometry"] and x["strong_first_c1"] for x in lit)

    # ---------------- source overlay
    overlay = dict(REAL_SOURCE_COORDINATE_COVERAGE="UNRESOLVED", reason="catalog not supplied")
    if catalog and Path(catalog).exists():
        import phase17_r1od3_campaign as C3
        cat = C3.parse_catalog(Path(catalog))
        seps = []
        for r in results:
            f = r["cells"]["FIRST_SESSION"]
            if f.get("feasible"):
                u = np.asarray(f["sessions"][0]["s_geo_mid"], float)
                seps.append(float(np.degrees(np.arccos(np.clip(np.max(cat @ u), -1, 1)))))
        within = sum(x <= math.degrees(0.1) for x in seps)
        overlay = dict(catalog_pdf_sha256=hashlib.sha256(Path(catalog).read_bytes()).hexdigest(),
                       n_sources=int(cat.shape[0]), n_first_sessions=len(seps), within_0p1_rad=within,
                       nearest_sep_deg=dict(median=float(np.median(seps)), max=float(max(seps))),
                       REAL_SOURCE_COORDINATE_COVERAGE=("COORDINATE_SOURCE_AVAILABLE" if within == len(seps)
                                                        else "NO_COORDINATE_SOURCE_WITHIN_LIMIT (%d of %d missing)"
                                                        % (len(seps) - within, len(seps))))

    # ---------------- absolute separation
    abs_all = [r["cells"][s]["cal"]["C1_NOMINAL"]["absolute_class"] for r in results for s in SCHEDULES
               if r["cells"][s].get("feasible")]
    abs_class = ("STRONG_K_SEPARATION" if "STRONG_K_SEPARATION" in abs_all else
                 "MODERATE_K_SEPARATION" if "MODERATE_K_SEPARATION" in abs_all else "WEAK_K_SEPARATION")
    max_comb = max(r["cells"][s]["cal"]["C1_NOMINAL"]["f_perp"] for r in results for s in SCHEDULES
                   if r["cells"][s].get("feasible"))

    # ---------------- CSV artifacts
    with (ARTIFACTS / "phase17_r1od4_epoch_grid.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["phase_deg", "et0", "d3_k_equivalent", "d3_anchor", "moon_geocentric_dec_deg",
                    "sc_dec_first_deg", "sun_earth_sc_first_deg", "gm_visibility_windows", "gm_qualified_sessions",
                    "gm_qualified_hours", "dsn_good_gm_geometry"])
        for x in lit:
            r = by[("CANON", "HIGH_FIDELITY", x["phase"])]
            w.writerow([x["phase"], phase_et(x["phase"]), x["phase"] / 15.0 if x["phase"] % 15 == 0 else "",
                        x["phase"] in pre["phase_grid"]["d3_anchors_deg"], x["moon_dec_deg"], x["sc_dec_first_deg"],
                        x["sun_earth_sc_first_deg"], r["diag"]["gm_visibility_windows"],
                        r["diag"]["gm_qualified_sessions"], x["gm_qualified_hours"], x["dsn_good_gm_geometry"]])
    with (ARTIFACTS / "phase17_r1od4_sessions.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["orbit", "dynamics", "phase_deg", "schedule", "session", "window_start_s", "window_end_s",
                    "session_start_s", "session_end_s", "np_times_s", "np_visible", "ownership_ok"])
        for r in results:
            for sched, c in r["cells"].items():
                if c.get("feasible"):
                    for j, s in enumerate(c["sessions"]):
                        w.writerow([r["orbit"], r["dynamics"], r["phase"], sched, j, s["window"][0], s["window"][1],
                                    s["interval"][0], s["interval"][1], ";".join("%.0f" % t for t in s["np_t"]),
                                    c["np_visible"], c["ownership_ok"]])
    with (ARTIFACTS / "phase17_r1od4_canonical_fidelity.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["phase_deg", "schedule", "calibration", "simplified_delta", "simplified_class",
                    "high_fidelity_delta", "high_fidelity_class", "simplified_range_f", "high_fidelity_range_f"])
        for p in phases:
            for sched in SCHEDULES:
                for cname in CAL_MAP:
                    a, b = delta(by[("CANON", "SIMPLIFIED", p)], sched, cname), delta(by[("CANON", "HIGH_FIDELITY", p)], sched, cname)
                    w.writerow([p, sched, cname, a, cls(a), b, cls(b), by[("CANON", "SIMPLIFIED", p)]["diag"]["range_f_perp"],
                                by[("CANON", "HIGH_FIDELITY", p)]["diag"]["range_f_perp"]])
    with (ARTIFACTS / "phase17_r1od4_orbit_robustness.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["orbit", "dynamics", "phase_deg", "range_f_perp", "range_n_obs", "gm_qualified_sessions",
                    "first_delta_C1", "first_class_C1", "all_delta_C1", "all_class_C1", "first_combined_f_C1",
                    "first_absolute_class_C1"])
        for o in order:
            for dyn in ("HIGH_FIDELITY", "SIMPLIFIED"):
                for p in phases:
                    r = by[(o, dyn, p)]
                    f = r["cells"]["FIRST_SESSION"]
                    df, da = delta(r, "FIRST_SESSION", "C1_NOMINAL"), delta(r, "ALL_FEASIBLE_SESSIONS", "C1_NOMINAL")
                    w.writerow([o, dyn, p, r["diag"]["range_f_perp"], r["diag"]["range_n_obs"],
                                r["diag"]["gm_qualified_sessions"], df, cls(df), da, cls(da),
                                f["cal"]["C1_NOMINAL"]["f_perp"] if f.get("feasible") else None,
                                f["cal"]["C1_NOMINAL"]["absolute_class"] if f.get("feasible") else None])
    fields = ["n_obs", "f_perp", "delta_f_perp", "relative_class", "absolute_class", "theta_k_deg",
              "conditional_k_information", "i_kk", "sigma_k", "sigma_k_frac", "k_column_norm", "design_condition",
              "weakest_mode_k_component"]
    with (ARTIFACTS / "phase17_r1od4_calibration.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["orbit", "dynamics", "phase_deg", "schedule", "calibration", "range_f_perp", "n_np", "n_sessions",
                    "session_hours"] + fields + ["design_singular_values", "bias_sigma_ns", "max_abs_k_bias_corr",
                                                 "rebuild_abs_diff"])
        for r in results:
            for sched, c in r["cells"].items():
                if not c.get("feasible"):
                    w.writerow([r["orbit"], r["dynamics"], r["phase"], sched, "", r["diag"]["range_f_perp"], 0, 0, 0]
                               + [""] * (len(fields) + 4))
                    continue
                for cname, m in c["cal"].items():
                    w.writerow([r["orbit"], r["dynamics"], r["phase"], sched, cname, r["diag"]["range_f_perp"], c["n_np"],
                                c["n_sessions"], c["session_hours"]] + [m[k] for k in fields]
                               + [";".join("%.6g" % v for v in m["design_singular_values"]),
                                  ";".join("%.4g" % v for v in m.get("bias_sigma_ns", [])), m.get("max_abs_k_bias_corr"),
                                  c["rebuild"][cname]])

    counts = {}
    for o in order:
        for dyn in ("HIGH_FIDELITY", "SIMPLIFIED"):
            for sched in SCHEDULES:
                s = series(o, dyn, sched, "C1_NOMINAL")
                counts["%s|%s|%s" % (o, dyn, sched)] = {lab: sum(cls(d) == lab for d in s) for lab in
                                                         ("STRONG_DIRECTION_GAIN", "MODEST_DIRECTION_GAIN",
                                                          "MAGNITUDE_ONLY", "DIRECTION_LOSS", "INFEASIBLE")}
    longest_runs = {k: max([b - a + 1 for a, b in windows([is_strong(d) for d in series(*k.split("|"), "C1_NOMINAL")], 1)],
                           default=0) for k in counts}
    if elapsed is None:
        prev = ARTIFACTS / "phase17_r1od4_summary.json"
        elapsed = json.loads(prev.read_text(encoding="utf-8")).get("elapsed_s") if prev.exists() else None
    summary = dict(
        phase="17-R1O-D4", predeclaration_sha256=PREDECLARATION_SHA256, elapsed_s=elapsed,
        literature_status=pre["literature"]["status"],
        verification=dict(contracts=contract_ok, rebuild_max=max(rebuild), rebuild_gate=rb_ok, anchors=anchors_ok,
                          phase_et_max_diff_s=et_diff),
        R0_canonical_hf_c1_first=r0,
        canonical_simplified_c1_first=simp,
        R1_orbit_windows={o: r1[o] for o in order}, orbit_window_count=n_orb, ORBIT_ROBUSTNESS_CLASS=orbit_class,
        R2_calibration_windows_canonical_hf_first=cal_w, R2_window_fates=r2, CALIBRATION_ROBUSTNESS_CLASS=cal_class,
        R3_fidelity=dict(FIDELITY_CLASS=fid_class, class_changes_first_c1=class_changes,
                         descriptive_decomposition_canon_first_c1=dict(
                             rms_d_range=float(np.sqrt(np.mean([x["d_range"] ** 2 for x in fid_decomp]))),
                             rms_d_combined=float(np.sqrt(np.mean([x["d_combined"] ** 2 for x in fid_decomp]))),
                             rms_d_delta=float(np.sqrt(np.mean([x["d_delta"] ** 2 for x in fid_decomp]))),
                             corr_d_delta_vs_minus_d_range=float(np.corrcoef([x["d_delta"] for x in fid_decomp],
                                                                             [-x["d_range"] for x in fid_decomp])[0, 1]),
                             rows=fid_decomp)),
        all_windows={"|".join(k): v for k, v in win.items() if v},
        counts_c1=counts, longest_strong_run_points_c1=longest_runs,
        canonical_hf_first_c1_series=dict(zip([str(p) for p in phases], hf_s)),
        canonical_simplified_first_c1_series=dict(zip([str(p) for p in phases], si_s)),
        canonical_hf_max_delta_first_c1=max([d for d in hf_s if d is not None], default=None),
        confirmatory_success=success, PRIMARY_CLASS=primary,
        D3_SECONDARY_HYPOTHESIS_CONFIRMED=primary == "GM_NARROW_WINDOW_CONFIRMED_AND_ROBUST",
        NONLINEAR_ESTIMATOR_CONFIRMATION_JUSTIFIED="YES" if primary == "GM_NARROW_WINDOW_CONFIRMED_AND_ROBUST" else "NO",
        DDOR_TRACKING_DESIGN_BRANCH_STATUS=("CLOSED_FOR_CURRENT_K_SRP_FORMULATION"
                                            if primary == "D3_SECONDARY_GM_WINDOW_DOES_NOT_CONFIRM" else
                                            "OPEN_NARROW_GM_DESIGN" if primary.startswith("GM_NARROW") else
                                            "NOT_CONFIRMED_FRAGILITY_DOCUMENTED"),
        FIRST_VS_ALL_MECHANISM_CLASS=mech_class,
        literature_overlay=dict(n_phases=len(lit), dsn_good_gm_geometry=n_good, strong_first_c1=n_strong,
                                strong_and_good=n_both, rows=lit),
        source_overlay=overlay, ABSOLUTE_K_IDENTIFIABILITY_CLASS=abs_class, max_combined_f_perp_c1=max_comb)
    (ARTIFACTS / "phase17_r1od4_summary.json").write_text(json.dumps(_clean(summary), indent=1), encoding="utf-8")
    figures(phases, series, order, mech_rows)
    print(json.dumps(_clean({k: summary[k] for k in (
        "verification", "R0_canonical_hf_c1_first", "canonical_simplified_c1_first", "orbit_window_count",
        "ORBIT_ROBUSTNESS_CLASS", "CALIBRATION_ROBUSTNESS_CLASS", "R3_fidelity", "longest_strong_run_points_c1",
        "canonical_hf_max_delta_first_c1", "PRIMARY_CLASS", "FIRST_VS_ALL_MECHANISM_CLASS",
        "ABSOLUTE_K_IDENTIFIABILITY_CLASS", "source_overlay")}), indent=1))


def figures(phases, series, order, mech_rows) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nan = lambda s: [np.nan if v is None else v for v in s]  # noqa: E731
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for ax, sched in zip(axes, SCHEDULES):
        for dyn, st in (("SIMPLIFIED", "o--"), ("HIGH_FIDELITY", "s-")):
            for cal, col in (("C0_IDEAL", "tab:blue"), ("C1_NOMINAL", "tab:orange"), ("C2_FREE", "tab:green")):
                ax.plot(phases, nan(series("CANON", dyn, sched, cal)), st, color=col, ms=4,
                        label="%s %s" % (dyn.split("_")[0][:4], cal))
        ax.axhline(0.05, color="k", lw=0.8, ls="--")
        ax.axvspan(75, 150, color="grey", alpha=0.08)
        ax.set_title("CANON, GM %s" % sched)
        ax.set_ylabel("delta f_perp")
        ax.legend(fontsize=6, ncol=3)
    axes[-1].set_xlabel("synodic phase [deg] (5 deg = %.2f d)" % (5 / 360 * SYNODIC_S / 86400))
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "phase17_r1od4_canonical_phase_map.png", dpi=110)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, dyn in zip(axes, ("HIGH_FIDELITY", "SIMPLIFIED")):
        m = np.array([nan(series(o, dyn, "FIRST_SESSION", "C1_NOMINAL")) for o in order])
        im = ax.imshow(m, aspect="auto", cmap="RdBu_r", vmin=-0.1, vmax=0.1,
                       extent=[phases[0] - 2.5, phases[-1] + 2.5, len(order) - 0.5, -0.5])
        for i in range(len(order)):
            for j, p in enumerate(phases):
                if np.isfinite(m[i, j]) and m[i, j] > 0.05:
                    ax.plot(p, i, "k*", ms=5)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=7)
        ax.set_title("GM FIRST_SESSION, C1 nominal, %s (* = STRONG)" % dyn, fontsize=8)
        ax.set_xlabel("synodic phase [deg]")
    fig.colorbar(im, ax=axes, label="delta f_perp")
    fig.savefig(ARTIFACTS / "phase17_r1od4_orbit_phase_heatmap.png", dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for cal, mk in (("C0_IDEAL", "o"), ("C1_HALF", "v"), ("C1_NOMINAL", "s"), ("C1_DOUBLE", "^"), ("C2_FREE", "x")):
        ax.plot(phases, nan(series("CANON", "HIGH_FIDELITY", "FIRST_SESSION", cal)), mk + "-", ms=4, label=cal)
    ax.axhline(0.05, color="k", lw=0.8, ls="--")
    ax.set_xlabel("synodic phase [deg]")
    ax.set_ylabel("delta f_perp")
    ax.set_title("CANON HIGH_FIDELITY GM FIRST_SESSION by calibration")
    ax.legend(fontsize=7, ncol=5)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "phase17_r1od4_calibration_map.png", dpi=110)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ph = [x["phase"] for x in mech_rows]
    axes[0].plot(ph, [x["delta_first"] for x in mech_rows], "s-", label="FIRST_SESSION")
    axes[0].plot(ph, [x["delta_all"] for x in mech_rows], "o-", label="ALL_FEASIBLE_SESSIONS")
    axes[0].axhline(0.05, color="k", lw=0.8, ls="--")
    axes[0].set_ylabel("delta f_perp")
    axes[0].legend(fontsize=7)
    m = [x for x in mech_rows if "ikk_ratio" in x]
    axes[1].plot([x["phase"] for x in m], [x["ikk_ratio"] for x in m], "k^-")
    axes[1].set_ylabel("I_kk(ALL) / I_kk(FIRST)")
    axes[2].plot([x["phase"] for x in m], [x["added_orth_frac"] for x in m], "o-", label="added sessions: dI_perp / dI_kk")
    axes[2].plot([x["phase"] for x in m], [x["first_orth_frac"] for x in m], "s-", label="first session: f_perp^2")
    axes[2].set_ylabel("orthogonal fraction")
    axes[2].set_xlabel("synodic phase [deg]")
    axes[2].legend(fontsize=7)
    axes[0].set_title("CANON HIGH_FIDELITY C1 nominal: FIRST vs ALL session-signature decomposition", fontsize=9)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "phase17_r1od4_first_vs_all_mechanism.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
