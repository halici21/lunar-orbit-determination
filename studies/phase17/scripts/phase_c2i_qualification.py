"""Phase C2-I: qualification of the station-time relation in relativistic_light_time (G1 bit-identity, G4, G5,
G8, G9).

Predeclared in studies/phase17/configs/phase_c2i_station_time_implementation_predeclaration.json (+ amendment A1).
G2/G3: phase_c2i_station_time_gates.py; G6/G7: phase_c2i_estimator_gates.py; G1's full test suite runs separately.
Writes studies/phase17/results/phase_c2i_qualification.json.

    python studies/phase17/scripts/phase_c2i_qualification.py [--workers 10]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")

import phase17_kp2_campaign as P  # noqa: E402
import phase17_kp2_core as K2  # noqa: E402
import phase_c2_time_scale_survey as SV  # noqa: E402

ROOT = HERE.parents[2]
OUT = P.ARTIFACTS / "phase_c2i_qualification.json"
BASE_COMMIT = "d24e380"
TC = 60.0
FD_CELLS = (("CANON", 0), ("O2_FULL_SUN_HIGH_BETA", 12), ("O3_HIGH_ALTITUDE_500KM", 0))
FD_ROWS = 20
FD_STEPS = {"pos": (1.0, 10.0, 100.0), "vel": (1e-3, 1e-2, 1e-1), "k": (1e-5, 1e-4, 1e-3, 3e-3, 1e-2)}   # A1
CLOCK_STEPS = (1e-4, 1e-3, 1e-2)
SIGMA_CLOCK_S = 1e-6


def _base_module():
    """relativistic_light_time.py as of the branch base, imported as lunar_od._rlt_base."""
    name = "lunar_od._rlt_base"
    if name in sys.modules:
        return sys.modules[name]
    import lunar_od  # noqa: F401
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASE_COMMIT}:src/lunar_od/relativistic_light_time.py"],
                         check=True, capture_output=True).stdout
    path = Path(tempfile.mkdtemp()) / "_rlt_base.py"
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "lunar_od"
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _od_mapping(g, dy_r, dy_d):
    """The C1/C2 survey's linear bias mapping (max |dtheta/sigma| over the 6 states and K) for one cell."""
    ev = K2.evaluate(g["fx"], g["X"], g["PHI"], g["S"])
    nr, nd = dy_r.size, dy_d.size
    out = {}
    for design in ("RANGE_ONLY", "RANGE_DOPPLER_FREE_BIAS"):
        use_d = design != "RANGE_ONLY"
        a_, _ = K2._design(ev, g["fx"], "M0", "FREE", use_d)
        for nl, sig in SV.NOISE.items():
            w = np.concatenate([np.full(nr, 1 / sig["range"])] + ([np.full(nd, 1 / sig["doppler"])] if use_d else []))
            prior = np.zeros((6, a_.shape[1]))
            prior[:, :6] = np.diag(1 / K2.PRIOR_SD)
            aw = np.vstack([a_ * w[:, None], prior])
            scale = np.linalg.norm(aw, axis=0)
            _, r = np.linalg.qr(aw / scale)
            sigma = np.sqrt(np.sum(np.linalg.inv(r) ** 2, axis=1)) / scale
            dy = np.concatenate([dy_r] + ([dy_d] if use_d else []))
            dth = np.linalg.lstsq(aw / scale, np.concatenate([dy * w, np.zeros(6)]), rcond=None)[0] / scale
            ratio = np.abs(dth[:7] / sigma[:7])
            out[f"{design}|{nl}"] = dict(max_state_over_sigma=float(ratio[:6].max()), k_over_sigma=float(ratio[6]))
    return out


def _sites(fx):
    from lunar_od.station_time import Site
    return [Site.from_itrf_m(s.r_ecef_m) for s in fx.stations]


# ======================================================================
# G1 (bit identity), G4 (survey parity), G9 (OD impact) per cell
# ======================================================================
def run_cell(spec) -> dict:
    orbit, k = spec
    from lunar_od import relativistic_light_time as R
    from lunar_od.radiometrics import _interp_state
    from lunar_od.station_time import Site

    t_start = time.time()
    B = _base_module()
    survey = SV.run_cell(spec, return_rows=True)
    rows_sv = survey["rows"]
    g = SV.build_cell(orbit, k)
    rb, rows, fx, et0 = g["rb"], g["rows"], g["fx"], g["et0"]
    t_grid = rb.t_grid
    sites = _sites(fx)
    eph_new, eph_base = R.BarycentricEphemeris(et0), B.BarycentricEphemeris(et0)
    sc_new, sc_base = R.SpacecraftHistory(t_grid, rb.nom48[:, :6]), B.SpacecraftHistory(t_grid, rb.nom48[:, :6])
    models = {"DSN": (R.DSN_LIGHT_TIME_MODEL, B.DSN_LIGHT_TIME_MODEL), "DEFAULT": (R.LightTimeModel(), B.LightTimeModel())}

    # ---- two-way range ----
    nr = rb.obs.shape[0]
    g1 = {m: dict(range_max_abs=0.0, doppler_max_abs=0.0, ddor_max_abs=0.0) for m in models}
    dsn_r, c2_r = np.zeros(nr), np.zeros(nr)
    for i, (t3, _, sid, _) in enumerate(rb.obs[:, :4]):
        st = g["range_st"][int(sid) - 1]
        for m, (mn, mb) in models.items():
            vn = R.two_way_range_m(R.solve_two_way(float(t3), st, sc_new, eph_new, mn, site=sites[int(sid) - 1]))
            vb = B.two_way_range_m(B.solve_two_way(float(t3), st, sc_base, eph_base, mb))
            g1[m]["range_max_abs"] = max(g1[m]["range_max_abs"], abs(vn - vb))
            if m == "DSN":
                dsn_r[i] = vn
        c2_r[i] = R.two_way_range_m(R.solve_two_way(float(t3), st, sc_new, eph_new, R.DSN_C2_LIGHT_TIME_MODEL,
                                                    site=sites[int(sid) - 1]))
    sv_r = np.asarray(rows_sv["range"]["T2_RATE"]) + np.asarray(rows_sv["range"]["T3_RATE"])
    g4_r = float(np.max(np.abs((c2_r - dsn_r) - sv_r)))

    # ---- counted Doppler ----
    nd = rows.n
    dsn_d, c2_d = np.zeros(nd), np.zeros(nd)
    for i in range(nd):
        s, tm = int(rows.station[i]), float(rows.t_mid[i])
        st = g["dop_st"][s]
        for m, (mn, mb) in models.items():
            vn = R.counted_doppler(tm, TC, st, sc_new, eph_new, mn, site=sites[s])[0]
            vb = B.counted_doppler(tm, TC, st, sc_base, eph_base, mb)[0]
            g1[m]["doppler_max_abs"] = max(g1[m]["doppler_max_abs"], abs(vn - vb))
            if m == "DSN":
                dsn_d[i] = vn
        c2_d[i] = R.counted_doppler(tm, TC, st, sc_new, eph_new, R.DSN_C2_LIGHT_TIME_MODEL, site=sites[s])[0]
    sv_d = np.asarray(rows_sv["doppler"]["T2_RATE"]) + np.asarray(rows_sv["doppler"]["T3_RATE"])
    g4_d = float(np.max(np.abs((c2_d - dsn_d) - sv_d)))

    # ---- delta-DOR (CANON k0) ----
    g4_dd = None
    if g["ddor"]:
        import phase17_r1od2_core as D2
        dr, (na, nb) = g["ddor"]["rows"], g["ddor"]["names"]
        by = D2.stations_by_name()
        site_a, site_b = Site.from_itrf_m(by[na].r_ecef_m), Site.from_itrf_m(by[nb].r_ecef_m)
        a_fn = lambda t: _interp_state(t_grid, dr.hist_a, t)  # noqa: E731
        b_fn = lambda t: _interp_state(t_grid, dr.hist_b, t)  # noqa: E731
        n = len(dr.t_used)
        dsn_dd, c2_dd = np.zeros(n), np.zeros(n)
        for i, t_obs in enumerate(dr.t_used):
            t_obs = float(t_obs)
            for m, (mn, mb) in models.items():
                vn = R.delta_dor_station_time(t_obs, a_fn, b_fn, sc_new, eph_new, mn, sites=(site_a, site_b)).delay_s
                vb = B.solve_delta_dor(t_obs, a_fn(t_obs), b_fn, sc_base, eph_base, mb).differential_delay_s
                g1[m]["ddor_max_abs"] = max(g1[m]["ddor_max_abs"], abs(vn - vb))
                if m == "DSN":
                    dsn_dd[i] = vn
            c2_dd[i] = R.delta_dor_station_time(t_obs, a_fn, b_fn, sc_new, eph_new, R.DSN_C2_LIGHT_TIME_MODEL,
                                                sites=(site_a, site_b)).delay_s
        sv_dd = np.asarray(rows_sv["ddor"]["T3_DIFFERENTIAL"]) + np.asarray(rows_sv["ddor"]["TT_INTERVAL_INFO"])
        g4_dd = dict(max_abs_s=float(np.max(np.abs((c2_dd - dsn_dd) - sv_dd))),
                     change_ns=dict(min=float((c2_dd - dsn_dd).min() * 1e9), max=float((c2_dd - dsn_dd).max() * 1e9)))

    return dict(cell=f"{orbit}:{k}", n_range=nr, n_doppler=nd, elapsed_s=time.time() - t_start, g1=g1,
                g4=dict(range_max_abs_m=g4_r, doppler_max_abs_mps=g4_d, ddor=g4_dd),
                change=dict(range_max_abs_m=float(np.max(np.abs(c2_r - dsn_r))),
                            doppler_max_abs_mps=float(np.max(np.abs(c2_d - dsn_d)))),
                g9=_od_mapping(g, c2_r - dsn_r, c2_d - dsn_d))


# ======================================================================
# G4 epoch parts: UTC tag conversion vs the SPICE chain, against the C2 survey's reference values
# ======================================================================
def g4_epoch_parts() -> dict:
    from lunar_od import station_time as S
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels(None, clear=True)
    r = np.load(P.ARTIFACTS / "phase_c2i_erfa_reference.npz", allow_pickle=False)
    ref = json.loads((P.ARTIFACTS / "phase_c2_erfa_reference.json").read_text(encoding="utf-8"))["cells"]
    names = [str(x) for x in r["site_names"]]
    pos = json.loads((P.ARTIFACTS / "phase_c2_survey_epochs.json").read_text(encoding="utf-8"))[
        "station_positions_itrf93_m"]
    # (et0, station, t) -> (geo, topo); cells sharing an epoch carry identical values for a shared tag
    val = {}
    for c in ref.values():
        for name, rec in c["stations"].items():
            for t, geo, topo in zip(rec["t"], rec["geo"], rec["topo"]):
                val[(float(c["et0"]), name, float(t))] = (geo, topo)
    worst = 0.0
    n = 0
    for e0 in np.unique(r["g3c_et0"]):
        for k_, name in enumerate(names):
            sel = np.flatnonzero((r["g3c_et0"] == e0) & (r["g3c_site"] == k_))
            if not sel.size:
                continue
            tags = [(int(r["g3c_y"][i]), int(r["g3c_m"][i]), int(r["g3c_d"][i]), float(r["g3c_sod"][i])) for i in sel]
            site = S.Site.from_itrf_m(pos[name])
            diff = S.utc_to_tdb_seconds(tags, float(e0), site) - S.spice_utc_chain_tdb_seconds(tags, float(e0))
            for j, i in enumerate(sel):
                t = float(r["g3c_t"][i])
                geo, topo = val[(float(e0), name, t)]
                expect = geo - SV.spice_tdb_minus_tt(float(e0), t) + topo
                worst = max(worst, abs(diff[j] - expect))
                n += 1
    return dict(n=n, max_abs_s=worst, limit_s=1e-10, pass_=bool(worst <= 1e-10))


# ======================================================================
# G5: analytic partials vs central finite differences (state re-propagated; clock offsets)
# ======================================================================
def run_fd(spec) -> dict:
    orbit, k = spec
    from lunar_od import relativistic_light_time as R
    from lunar_od.radiometrics import _interp_state
    from lunar_od.station_time import Site

    g = SV.build_cell(orbit, k)
    rb, rows, fx, et0 = g["rb"], g["rows"], g["fx"], g["et0"]
    t_grid = rb.t_grid
    sites = _sites(fx)
    model = R.DSN_C2_LIGHT_TIME_MODEL
    eph = R.BarycentricEphemeris(et0)
    aug = K2.nom48(g["X"], g["PHI"], g["S"][0] + g["S"][1] + g["S"][2])
    sc_aug = R.SpacecraftHistory(t_grid, aug)
    rng_idx = np.linspace(0, rb.obs.shape[0] - 1, FD_ROWS).astype(int)
    dop_idx = np.linspace(0, rows.n - 1, FD_ROWS).astype(int)
    dd = None
    if g["ddor"]:
        import phase17_r1od2_core as D2
        dr, (na, nb) = g["ddor"]["rows"], g["ddor"]["names"]
        by = D2.stations_by_name()
        dd = dict(rows=dr, sites=(Site.from_itrf_m(by[na].r_ecef_m), Site.from_itrf_m(by[nb].r_ecef_m)),
                  idx=np.linspace(0, len(dr.t_used) - 1, FD_ROWS).astype(int),
                  a=lambda t: _interp_state(t_grid, dr.hist_a, t), b=lambda t: _interp_state(t_grid, dr.hist_b, t))

    def rng_val(sc, i, b=0.0):
        t3, sid = float(rb.obs[i, 0]), int(rb.obs[i, 2]) - 1
        return R.two_way_range_m(R.solve_two_way(t3, g["range_st"][sid], sc, eph, model, site=sites[sid],
                                                 clock_offset_s=b))

    def dop_val(sc, i, b=0.0):
        s = int(rows.station[i])
        return R.counted_doppler(float(rows.t_mid[i]), TC, g["dop_st"][s], sc, eph, model, site=sites[s],
                                 clock_offset_s=b)[0]

    def dd_val(sc, i, ba=0.0, bb=0.0):
        t_obs = float(dd["rows"].t_used[i])
        return R.delta_dor_station_time(t_obs, dd["a"], dd["b"], sc, eph, model, sites=dd["sites"],
                                        clock_offsets_s=(ba, bb)).delay_s

    def evaluate(sc):
        out = dict(range=[rng_val(sc, i) for i in rng_idx], doppler=[dop_val(sc, i) for i in dop_idx])
        if dd:
            out["ddor"] = [dd_val(sc, i) for i in dd["idx"]]
        return {kk: np.asarray(v, float) for kk, v in out.items()}

    an = dict(range=[], doppler=[], ddor=[])
    cl = dict(range=[], doppler=[], ddor_a=[], ddor_b=[])
    for i in rng_idx:
        t3, sid = float(rb.obs[i, 0]), int(rb.obs[i, 2]) - 1
        sol = R.solve_two_way(t3, g["range_st"][sid], sc_aug, eph, model, site=sites[sid])
        an["range"].append(R.two_way_range_partial(sol, sc_aug))
        cl["range"].append(R.two_way_range_clock_partial(sol))
    for i in dop_idx:
        s = int(rows.station[i])
        _, p, _, c = R.counted_doppler(float(rows.t_mid[i]), TC, g["dop_st"][s], sc_aug, eph, model, site=sites[s],
                                       with_partial=True, with_clock_partial=True)
        an["doppler"].append(p)
        cl["doppler"].append(c)
    if dd:
        for i in dd["idx"]:
            st = R.delta_dor_station_time(float(dd["rows"].t_used[i]), dd["a"], dd["b"], sc_aug, eph, model,
                                          sites=dd["sites"])
            an["ddor"].append(R.delta_dor_partial(st.solution, sc_aug))
            ca, cb = R.delta_dor_clock_partials(st.solution)
            cl["ddor_a"].append(ca)
            cl["ddor_b"].append(cb)
    an = {kk: np.asarray(v, float).reshape(-1, 7) for kk, v in an.items() if len(v)}

    # state + K
    n_steps = max(len(v) for v in FD_STEPS.values())
    fd = {kk: np.full((n_steps,) + v.shape, np.nan) for kk, v in an.items()}
    for j in range(7):
        kind = "pos" if j < 3 else ("vel" if j < 6 else "k")
        for s_i, h in enumerate(FD_STEPS[kind]):
            vals = []
            for sign in (+1, -1):
                dx = np.zeros(6)
                ks = np.full(3, K2.K0)
                if j < 6:
                    dx[j] = sign * h
                else:
                    ks = ks + sign * h
                P.K._init()
                x0 = np.asarray(P.x0s()[orbit], float)
                Xp, _, _ = K2.propagate_segmented(g["scn"], x0 + dx, ks, "SIMPLIFIED")
                vals.append(evaluate(R.SpacecraftHistory(t_grid, Xp)))
            for kk in fd:
                fd[kk][s_i, :, j] = (vals[0][kk] - vals[1][kk]) / (2.0 * h)
    result = {}
    for kk, a in an.items():
        scale = np.maximum(np.abs(a), 1e-6 * np.max(np.abs(a), axis=1, keepdims=True))
        rel = np.nanmin(np.abs(fd[kk] - a[None]) / scale[None], axis=0)
        worst = np.unravel_index(int(np.argmax(rel)), rel.shape)
        result[kk] = dict(max_rel=float(rel.max()), median_rel=float(np.median(rel)), n_rows=int(a.shape[0]),
                          worst_row_param=[int(worst[0]), int(worst[1])])

    # clock offsets: analytic vs central difference in b, best over CLOCK_STEPS
    sc6 = R.SpacecraftHistory(t_grid, rb.nom48[:, :6])
    clock = {}

    def best_rel(analytic, f_of_b):
        best = np.inf
        for h in CLOCK_STEPS:
            fdv = (f_of_b(h) - f_of_b(-h)) / (2 * h)
            best = min(best, abs(fdv - analytic) / max(abs(analytic), 1e-300))
        return best
    clock["range"] = max(best_rel(cl["range"][j], lambda b, i=i: rng_val(sc6, i, b)) for j, i in enumerate(rng_idx))
    clock["doppler"] = max(best_rel(cl["doppler"][j], lambda b, i=i: dop_val(sc6, i, b))
                           for j, i in enumerate(dop_idx))
    if dd:
        clock["ddor_a"] = max(best_rel(cl["ddor_a"][j], lambda b, i=i: dd_val(sc6, i, ba=b))
                              for j, i in enumerate(dd["idx"]))
        clock["ddor_b"] = max(best_rel(cl["ddor_b"][j], lambda b, i=i: dd_val(sc6, i, bb=b))
                              for j, i in enumerate(dd["idx"]))
    return dict(cell=f"{orbit}:{k}", fd=result, clock_max_rel=clock)


# ======================================================================
# G8: consider covariance (CANON k0, RANGE_DOPPLER_FREE_BIAS, DSN_REALISTIC, sigma_b = 1 us)
# ======================================================================
def run_consider() -> dict:
    from lunar_od import relativistic_light_time as R
    from lunar_od.consider_covariance import consider_covariance

    g = SV.build_cell("CANON", 0)
    rb, rows, fx, et0 = g["rb"], g["rows"], g["fx"], g["et0"]
    sites = _sites(fx)
    model = R.DSN_C2_LIGHT_TIME_MODEL
    eph = R.BarycentricEphemeris(et0)
    sc = R.SpacecraftHistory(rb.t_grid, rb.nom48[:, :6])
    hr = np.array([R.two_way_range_clock_partial(R.solve_two_way(float(t3), g["range_st"][int(sid) - 1], sc, eph,
                                                                 model, site=sites[int(sid) - 1]))
                   for t3, _, sid, _ in rb.obs[:, :4]])
    hd = np.array([R.counted_doppler(float(rows.t_mid[i]), TC, g["dop_st"][int(rows.station[i])], sc, eph, model,
                                     site=sites[int(rows.station[i])], with_clock_partial=True)[3]
                   for i in range(rows.n)])
    ev = K2.evaluate(g["fx"], g["X"], g["PHI"], g["S"])
    a_, _ = K2._design(ev, g["fx"], "M0", "FREE", True)
    sig = SV.NOISE["DSN_REALISTIC"]
    w = np.concatenate([np.full(hr.size, 1 / sig["range"]), np.full(hd.size, 1 / sig["doppler"])])
    prior = np.zeros((6, a_.shape[1]))
    prior[:, :6] = np.diag(1 / K2.PRIOR_SD)
    a_x = np.vstack([a_ * w[:, None], prior])
    a_c = np.concatenate([np.concatenate([hr, hd]) * w, np.zeros(6)])[:, None]
    res = consider_covariance(a_x, a_c, np.array([[SIGMA_CLOCK_S ** 2]]))
    zero = consider_covariance(a_x, a_c, np.zeros((1, 1)))
    response = np.linalg.lstsq(a_x, a_c[:, 0], rcond=None)[0]
    s = res.sensitivity[:, 0]
    rel_a = float(np.max(np.abs(s + response)) / np.max(np.abs(response)))
    infl = res.p_consider - res.p_x
    ev_ = np.linalg.eigvalsh(0.5 * (infl + infl.T))
    sigma = res.sigma
    ratio = np.abs(s[:7] * SIGMA_CLOCK_S) / sigma[:7]
    survey = json.loads((P.ARTIFACTS / "phase_c2_time_scale_survey.json").read_text(encoding="utf-8"))
    sv_cell = next(c for c in survey["cells"] if c["cell"] == "CANON:0")
    sv_od = sv_cell["od"]["RANGE_DOPPLER_FREE_BIAS|DSN_REALISTIC|T4_OFFSET_1e-06"]
    sv_ratio = max(sv_od["max_state_over_sigma"], sv_od["k_over_sigma"])
    rel_c = abs(float(ratio.max()) - sv_ratio) / sv_ratio
    inflation = (res.sigma_consider[:7] / sigma[:7] - 1.0)
    return dict(
        a_sensitivity_vs_bias_response_rel=rel_a, a_pass=bool(rel_a <= 1e-8),
        b_symmetric=bool(np.array_equal(res.p_consider, res.p_consider.T)),
        b_psd_min_over_max=float(ev_.min() / max(ev_.max(), 1e-300)),
        b_zero_pcc_bitwise=bool(np.array_equal(zero.p_consider, zero.p_x)),
        b_pass=bool(np.array_equal(res.p_consider, res.p_consider.T) and ev_.min() >= -1e-12 * ev_.max()
                    and np.array_equal(zero.p_consider, zero.p_x)),
        c_max_ratio_1us=float(ratio.max()), c_survey_ratio=sv_ratio, c_rel=rel_c, c_pass=bool(rel_c <= 0.10),
        sigma_inflation_max_fraction=float(inflation.max()), sigma_inflation=inflation.tolist(),
        n_range=int(hr.size), n_doppler=int(hd.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        f_cells = ex.map(run_cell, list(P.CELLS))
        f_fd = ex.map(run_fd, list(FD_CELLS))
        f_con = ex.submit(run_consider)
        cells, fds, con = list(f_cells), list(f_fd), f_con.result()
    epoch = g4_epoch_parts()
    canon = next(c for c in cells if c["cell"] == "CANON:0")
    gates = dict(
        G1_bit_identity=dict(
            passed=all(v == 0.0 for c in cells for m in c["g1"].values() for v in m.values()),
            max_abs={m: {kk: max(c["g1"][m][kk] for c in cells) for kk in ("range_max_abs", "doppler_max_abs",
                                                                              "ddor_max_abs")} for m in ("DSN", "DEFAULT")},
            note="full test suite: run separately"),
        G4=dict(range_max_abs_m=max(c["g4"]["range_max_abs_m"] for c in cells),
                doppler_max_abs_mps=max(c["g4"]["doppler_max_abs_mps"] for c in cells),
                ddor_max_abs_s=canon["g4"]["ddor"]["max_abs_s"], epoch_parts=epoch),
        G5=dict(range=max(f["fd"]["range"]["max_rel"] for f in fds),
                doppler=max(f["fd"]["doppler"]["max_rel"] for f in fds),
                ddor=next(f for f in fds if "ddor" in f["fd"])["fd"]["ddor"]["max_rel"],
                clock_range=max(f["clock_max_rel"]["range"] for f in fds),
                clock_doppler=max(f["clock_max_rel"]["doppler"] for f in fds),
                clock_ddor_a=next(f for f in fds if "ddor_a" in f["clock_max_rel"])["clock_max_rel"]["ddor_a"],
                clock_ddor_b=next(f for f in fds if "ddor_b" in f["clock_max_rel"])["clock_max_rel"]["ddor_b"]),
        G8=con,
    )
    g4, g5 = gates["G4"], gates["G5"]
    g4["passed"] = bool(g4["range_max_abs_m"] <= 1e-5 and g4["doppler_max_abs_mps"] <= 1e-9
                        and g4["ddor_max_abs_s"] <= 1e-12 and epoch["pass_"])
    g5["passed"] = bool(g5["range"] <= 1e-6 and g5["doppler"] <= 1e-5 and g5["ddor"] <= 1e-6
                        and g5["clock_range"] <= 1e-6 and g5["clock_doppler"] <= 1e-5
                        and g5["clock_ddor_a"] <= 1e-6 and g5["clock_ddor_b"] <= 1e-6)
    gates["G8"]["passed"] = bool(con["a_pass"] and con["b_pass"] and con["c_pass"])
    keys = [kk for kk in cells[0]["g9"]]
    gates["G9_informational"] = {kk: max(max(c["g9"][kk]["max_state_over_sigma"], c["g9"][kk]["k_over_sigma"])
                                         for c in cells) for kk in keys}
    out = dict(predeclaration="studies/phase17/configs/phase_c2i_station_time_implementation_predeclaration.json",
               base_commit=BASE_COMMIT, gates=gates, cells=cells, fd=fds, elapsed_s=time.time() - t0)
    OUT.write_text(json.dumps(P.K._clean(out), indent=1), encoding="utf-8")
    print(json.dumps(P.K._clean(gates), indent=1))


if __name__ == "__main__":
    main()
