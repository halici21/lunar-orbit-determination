"""Phase C1-I: qualification of src/lunar_od/relativistic_light_time.py (gates G2, G3, G5, G8).

Predeclared in studies/phase17/configs/phase_c1i_light_time_implementation_predeclaration.json (+ amendment A1).
Writes studies/phase17/results/phase_c1i_qualification.json. G1, G4, G6, G7, G9 run separately.

    python studies/phase17/scripts/phase_c1i_qualification.py [--workers 10]
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
warnings.simplefilter("ignore")

import phase17_kp2_campaign as P  # noqa: E402
import phase17_kp2_core as K2  # noqa: E402
import phase_c1_light_time_survey as SV  # noqa: E402

OUT = P.ARTIFACTS / "phase_c1i_qualification.json"
EXACT_STATIONS = "--exact-stations" in sys.argv     # post-hoc diagnostic: see the C1-I report
FD_CELLS = (("CANON", 0), ("O2_FULL_SUN_HIGH_BETA", 12), ("O3_HIGH_ALTITUDE_500KM", 0))
FD_ROWS = 20
FD_STEPS = {"pos": (1.0, 10.0, 100.0), "vel": (1e-3, 1e-2, 1e-1), "k": (1e-5, 1e-4, 1e-3)}
if "--wide-k-steps" in sys.argv:        # post-hoc adjudication evidence: the predeclared K steps sit in the FD noise
    FD_STEPS = {"pos": (1.0, 10.0, 100.0, 100.0), "vel": (1e-3, 1e-2, 1e-1, 1e-1), "k": (1e-4, 1e-3, 3e-3, 1e-2)}
TC = 60.0


def _cell(orbit, k, ks=None, dx=None):
    P.K._init()
    x0 = np.asarray(P.x0s()[orbit], float)
    scn = P.K.scenario(orbit, x0.tolist(), k)
    ks = np.full(3, K2.K0) if ks is None else ks
    X, PHI, S = K2.propagate_segmented(scn, x0 + (0.0 if dx is None else dx), ks, "SIMPLIFIED")
    return scn, x0, X, PHI, S


def _models():
    from lunar_od.relativistic_light_time import LightTimeModel
    return {
        "MOON": LightTimeModel(),
        "SSB": LightTimeModel(frame="SSB"),
        **{f"SSB+{b}": LightTimeModel(frame="SSB", shapiro_bodies=(b,)) for b in ("SUN", "EARTH", "MOON", "JUPITER")},
        "SSB+TT": LightTimeModel(frame="SSB", interval_time_scale="TT"),
    }


def _feedback_prediction(sol, shapiro_d, shapiro_u):
    """Linear change of the round trip when per-leg delays S_d, S_u enter G inside the fixed point, minus
    their plain sum (amendment A1): -dt1 - (S_d + S_u) with dy = solve(G_y, -dG), dG = -[S_d, S_u]."""
    c = SV.C
    d, u = sol.downlink, sol.uplink
    los_d = (d.rx[:3] - d.tx[:3]) / np.linalg.norm(d.rx[:3] - d.tx[:3])
    los_u = (u.rx[:3] - u.tx[:3]) / np.linalg.norm(u.rx[:3] - u.tx[:3])
    g_y = np.array([[0.0, -1.0 + float(los_d / c @ d.tx[3:6])],
                    [-1.0 + float(los_u / c @ u.tx[3:6]), 1.0 + float(-los_u / c @ d.tx[3:6])]])
    dy = np.linalg.solve(g_y, np.array([shapiro_d, shapiro_u]))
    return 0.5 * c * (-dy[0] - (shapiro_d + shapiro_u))


def run_cell(spec) -> dict:
    orbit, k = spec
    import phase17_r1o_cd1_core as CD
    import phase17_r1od2_core as D2
    from lunar_od import relativistic_light_time as R
    from lunar_od.radiometrics import _interp_state, resolve_counted_doppler_station_state_provider
    from lunar_od.two_way_range import _station_state_providers_for_pass

    t_start = time.time()
    scn, x0, X, PHI, S = _cell(orbit, k)
    fx, zr, zd = K2.build_fixed(scn, X, PHI, S, P.K.stations())
    rb, rows, pg = fx.rb, fx.rows, fx.rb.pass_geo
    t_grid = rb.t_grid
    sc = R.SpacecraftHistory(t_grid, rb.nom48[:, :6])
    eph = R.BarycentricEphemeris(float(pg.et0_s))
    models = _models()
    provs = _station_state_providers_for_pass(pg)
    cfg = CD.doppler_config(TC)
    dprov = {s: resolve_counted_doppler_station_state_provider(cfg, fx.stations[s], t_grid, pg.earth_pos_mci_m,
                                                               pg.earth_vel_mci_mps, et0_s=pg.et0_s)
             for s in set(int(x) for x in rows.station)}

    def st_fn_raw(p):
        return lambda t: np.asarray(p.state(t), float)

    def st_fn(p):
        raw = lambda t: np.asarray(p.state(t), float)  # noqa: E731
        return (R.exact_epoch_station_state(raw, pg.et0_s, t_grid, pg.earth_pos_mci_m, pg.earth_vel_mci_mps)
                if EXACT_STATIONS else raw)

    # ---------------- two-way range: G2 (MOON parity), G3 (per-term vs survey), G8 ----------------
    sv_frames = {"MOON": SV.Frame("MOON", pg.et0_s), "SSB": SV.Frame("SSB", pg.et0_s)}
    nr = rb.obs.shape[0]
    val = {m: np.zeros(nr) for m in models}
    two_cycle_rows = []
    g3 = {t: dict(max_abs=0.0) for t in ("FRAME", "TIME_SCALE", "SHAPIRO_SUN", "SHAPIRO_EARTH", "SHAPIRO_MOON",
                                         "SHAPIRO_JUPITER")}
    a1 = {b: dict(max_abs_diff=0.0, worst_ratio_error=0.0, n_checked=0) for b in ("SUN", "EARTH", "MOON", "JUPITER")}
    for i, (t3, _, sid, _) in enumerate(rb.obs[:, :4]):
        stf = st_fn(provs[int(sid) - 1])
        sols = {}
        for name, m in models.items():
            sols[name] = R.solve_two_way(float(t3), stf, sc, eph, m)
            val[name][i] = R.two_way_range_m(sols[name])
            if sols[name].two_cycle:
                two_cycle_rows.append((i, name))
        # survey (independent code), same row; in --exact-stations mode with the survey's own station correction
        sv_st = (SV.exact_station(st_fn_raw(provs[int(sid) - 1]), pg.et0_s, t_grid, pg.earth_pos_mci_m,
                                  pg.earth_vel_mci_mps) if EXACT_STATIONS else stf)
        st_pos = lambda t, f=sv_st: f(t)[:3]  # noqa: E731
        sc_pos = lambda t: _interp_state(t_grid, rb.nom48[:, :6], t)[:3]  # noqa: E731
        s_moon = SV.two_way(sv_frames["MOON"], float(t3), st_pos, sc_pos, False)
        s_ssb = SV.two_way(sv_frames["SSB"], float(t3), st_pos, sc_pos, True)
        sv_terms = {"FRAME": 0.5 * SV.C * ((s_ssb["down"] + s_ssb["up"]) - (s_moon["down"] + s_moon["up"])),
                    "TIME_SCALE": -0.5 * SV.C * (s_ssb["down"] + s_ssb["up"]) * SV.tdb_minus_tt_rate(pg.et0_s, float(t3))}
        mod_terms = {"FRAME": val["SSB"][i] - val["MOON"][i], "TIME_SCALE": val["SSB+TT"][i] - val["SSB"][i]}
        for b in ("SUN", "EARTH", "MOON", "JUPITER"):
            sv_b = 0.5 * SV.C * s_ssb["shapiro"][b]
            mod_b = val[f"SSB+{b}"][i] - val["SSB"][i]
            sv_terms["SHAPIRO_" + b], mod_terms["SHAPIRO_" + b] = sv_b, mod_b
            # amendment A1: the difference must be the Shapiro feedback
            sd = sols["SSB+" + b].downlink.shapiro_s
            su = sols["SSB+" + b].uplink.shapiro_s
            pred = _feedback_prediction(sols["SSB"], sd, su)
            diff = mod_b - sv_b
            a1[b]["max_abs_diff"] = max(a1[b]["max_abs_diff"], abs(diff))
            if abs(pred) >= 1e-5:
                a1[b]["n_checked"] += 1
                a1[b]["worst_ratio_error"] = max(a1[b]["worst_ratio_error"], abs(diff - pred) / abs(pred))
        for t in g3:
            g3[t]["max_abs"] = max(g3[t]["max_abs"], abs(mod_terms[t] - sv_terms[t]))

    row_821_residual = None
    if (orbit, k) == ("CANON", 0):
        s821 = R.solve_two_way(float(rb.obs[821, 0]), st_fn(provs[int(rb.obs[821, 2]) - 1]), sc, eph, models["SSB"])
        row_821_residual = max(abs(lg.newtonian_s - float(np.linalg.norm(lg.rx[:3] - lg.tx[:3])) / SV.C)
                               for lg in (s821.downlink, s821.uplink))
        # the residual of the converged epoch pair: re-evaluate both legs at the returned epochs
        d_, u_ = s821.downlink, s821.uplink
        rx = R._to_frame(models["SSB"], eph, d_.t_rx_s, provs[int(rb.obs[821, 2]) - 1].state(d_.t_rx_s))
        tx = R._to_frame(models["SSB"], eph, d_.t_tx_s, sc.state(d_.t_tx_s))
        row_821_residual = abs((d_.t_rx_s - d_.t_tx_s) - float(np.linalg.norm(rx[:3] - tx[:3])) / SV.C)

    # ---------------- counted Doppler ----------------
    nd = rows.n
    dval = {m: np.zeros(nd) for m in ("MOON", "SSB", "SSB+SUN", "SSB+EARTH", "SSB+TT")}
    g3d = dict(FRAME=0.0, SHAPIRO_SUN=0.0, SHAPIRO_EARTH=0.0, TIME_SCALE=0.0)
    kd = SV.C / (2.0 * TC)
    for i in range(nd):
        stf = st_fn(dprov[int(rows.station[i])])
        tm = float(rows.t_mid[i])
        for name in dval:
            dval[name][i] = R.counted_doppler(tm, TC, stf, sc, eph, models[name])[0]
        sv_st = (SV.exact_station(st_fn_raw(dprov[int(rows.station[i])]), pg.et0_s, t_grid, pg.earth_pos_mci_m,
                                  pg.earth_vel_mci_mps) if EXACT_STATIONS else stf)
        st_pos = lambda t, f=sv_st: f(t)[:3]  # noqa: E731
        sc_pos = lambda t: _interp_state(t_grid, rb.nom48[:, :6], t)[:3]  # noqa: E731
        ends = {f: [SV.two_way(sv_frames[f], tm + s * TC / 2, st_pos, sc_pos, f == "SSB") for s in (-1, 1)]
                for f in sv_frames}
        tau = {f: [e["down"] + e["up"] for e in ends[f]] for f in ends}
        sv_d = dict(FRAME=kd * ((tau["SSB"][1] - tau["SSB"][0]) - (tau["MOON"][1] - tau["MOON"][0])),
                    SHAPIRO_SUN=kd * (ends["SSB"][1]["shapiro"]["SUN"] - ends["SSB"][0]["shapiro"]["SUN"]),
                    SHAPIRO_EARTH=kd * (ends["SSB"][1]["shapiro"]["EARTH"] - ends["SSB"][0]["shapiro"]["EARTH"]),
                    TIME_SCALE=-kd * (tau["SSB"][1] - tau["SSB"][0]) * SV.tdb_minus_tt_rate(pg.et0_s, tm))
        mod_d = dict(FRAME=dval["SSB"][i] - dval["MOON"][i], SHAPIRO_SUN=dval["SSB+SUN"][i] - dval["SSB"][i],
                     SHAPIRO_EARTH=dval["SSB+EARTH"][i] - dval["SSB"][i], TIME_SCALE=dval["SSB+TT"][i] - dval["SSB"][i])
        for kk in g3d:
            g3d[kk] = max(g3d[kk], abs(mod_d[kk] - sv_d[kk]))

    # ---------------- delta-DOR (CANON k0) ----------------
    ddor = None
    if (orbit, k) == ("CANON", 0):
        dr = D2.build_ddor_rows(scn, rb, "GC", D2.ROW_RULE_PRODUCTION_VISIBILITY)
        b_fn = lambda t: _interp_state(t_grid, dr.hist_b, t)  # noqa: E731
        ds = {m: np.zeros(dr.n_converged) for m in ("MOON", "SSB")}
        g3_dd = 0.0
        for i, t_obs in enumerate(dr.t_used):
            a_state = _interp_state(t_grid, dr.hist_a, float(t_obs))
            sv_ds = {}
            for m in ds:
                ds[m][i] = R.solve_delta_dor(float(t_obs), a_state, b_fn, sc, eph, models[m]).differential_delay_s
                fr = sv_frames[m]
                fr.set_anchor(float(t_obs))
                sc_pos = lambda t: _interp_state(t_grid, rb.nom48[:, :6], t)[:3]  # noqa: E731
                ta, ttx, rtx, _ = SV.leg_backward(fr, float(t_obs), a_state[:3] + fr.offset(float(t_obs)), sc_pos)
                tb, _, _, _ = SV.leg_forward(fr, ttx, rtx, lambda t: b_fn(t)[:3])
                sv_ds[m] = tb - ta
            g3_dd = max(g3_dd, abs((ds["SSB"][i] - ds["MOON"][i]) - (sv_ds["SSB"] - sv_ds["MOON"])))
        ddor = dict(n=int(dr.n_converged), g2_max_s=float(np.max(np.abs(ds["MOON"] - dr.d_s))),
                    g3_frame_max_s=g3_dd, frame_max_ns=float(np.max(np.abs(ds["SSB"] - ds["MOON"])) * 1e9))

    return dict(cell=f"{orbit}:{k}", n_range=nr, n_doppler=nd, elapsed_s=time.time() - t_start,
                g2=dict(range_max_m=float(np.max(np.abs(val["MOON"] - zr))),
                        doppler_max_mps=float(np.max(np.abs(dval["MOON"] - zd))),
                        ddor_max_s=(ddor["g2_max_s"] if ddor else None)),
                g3_range_m={t: v["max_abs"] for t, v in g3.items()},
                a1_shapiro=a1,
                g3_doppler_mps=g3d,
                doppler_terms_mps=dict(FRAME=float(np.max(np.abs(dval["SSB"] - dval["MOON"]))),
                                       SUN=float(np.max(np.abs(dval["SSB+SUN"] - dval["SSB"]))),
                                       EARTH=float(np.max(np.abs(dval["SSB+EARTH"] - dval["SSB"]))),
                                       TT=float(np.max(np.abs(dval["SSB+TT"] - dval["SSB"])))),
                ddor=ddor, two_cycle_rows=two_cycle_rows, row_821_residual_s=row_821_residual)


# ======================================================================
# G5: analytic partials vs central finite differences (full chain, re-propagated)
# ======================================================================
def run_fd(spec) -> dict:
    orbit, k = spec
    import phase17_r1od2_core as D2
    from lunar_od import relativistic_light_time as R
    from lunar_od.radiometrics import _interp_state, resolve_counted_doppler_station_state_provider
    import phase17_r1o_cd1_core as CD
    from lunar_od.two_way_range import _station_state_providers_for_pass

    scn, x0, X, PHI, S = _cell(orbit, k)
    fx, _, _ = K2.build_fixed(scn, X, PHI, S, P.K.stations())
    rb, rows, pg = fx.rb, fx.rows, fx.rb.pass_geo
    t_grid = rb.t_grid
    model = R.DSN_LIGHT_TIME_MODEL
    eph = R.BarycentricEphemeris(float(pg.et0_s))
    aug = K2.nom48(X, PHI, S[0] + S[1] + S[2])
    sc_aug = R.SpacecraftHistory(t_grid, aug)
    provs = _station_state_providers_for_pass(pg)
    cfg = CD.doppler_config(TC)
    dprov = {s: resolve_counted_doppler_station_state_provider(cfg, fx.stations[s], t_grid, pg.earth_pos_mci_m,
                                                               pg.earth_vel_mci_mps, et0_s=pg.et0_s)
             for s in range(len(fx.stations))}
    rng_idx = np.linspace(0, rb.obs.shape[0] - 1, FD_ROWS).astype(int)
    dop_idx = np.linspace(0, rows.n - 1, FD_ROWS).astype(int)
    ddor_rows = (D2.build_ddor_rows(scn, rb, "GC", D2.ROW_RULE_PRODUCTION_VISIBILITY)
                 if (orbit, k) == ("CANON", 0) else None)
    dd_idx = (np.linspace(0, ddor_rows.n_converged - 1, FD_ROWS).astype(int) if ddor_rows is not None else [])

    def st(p):
        raw = lambda t: np.asarray(p.state(t), float)  # noqa: E731
        return (R.exact_epoch_station_state(raw, pg.et0_s, t_grid, pg.earth_pos_mci_m, pg.earth_vel_mci_mps)
                if EXACT_STATIONS else raw)

    def evaluate(sc):
        out = dict(range=[], doppler=[], one_way=[], ddor=[])
        for i in rng_idx:
            t3, sid = float(rb.obs[i, 0]), int(rb.obs[i, 2]) - 1
            out["range"].append(R.two_way_range_m(R.solve_two_way(t3, st(provs[sid]), sc, eph, model)))
            out["one_way"].append(R.one_way_range(t3, st(provs[sid]), sc, eph, model)[0])
        for i in dop_idx:
            out["doppler"].append(R.counted_doppler(float(rows.t_mid[i]), TC, st(dprov[int(rows.station[i])]), sc,
                                                    eph, model)[0])
        for i in dd_idx:
            t_obs = float(ddor_rows.t_used[i])
            a_state = _interp_state(t_grid, ddor_rows.hist_a, t_obs)
            b_fn = lambda t: _interp_state(t_grid, ddor_rows.hist_b, t)  # noqa: E731
            out["ddor"].append(R.solve_delta_dor(t_obs, a_state, b_fn, sc, eph, model).differential_delay_s)
        return {kk: np.asarray(v, float) for kk, v in out.items()}

    # analytic
    an = dict(range=[], doppler=[], one_way=[], ddor=[])
    for i in rng_idx:
        t3, sid = float(rb.obs[i, 0]), int(rb.obs[i, 2]) - 1
        an["range"].append(R.two_way_range_partial(R.solve_two_way(t3, st(provs[sid]), sc_aug, eph, model), sc_aug))
        an["one_way"].append(R.one_way_range(t3, st(provs[sid]), sc_aug, eph, model, with_partial=True)[1])
    for i in dop_idx:
        an["doppler"].append(R.counted_doppler(float(rows.t_mid[i]), TC, st(dprov[int(rows.station[i])]), sc_aug,
                                               eph, model, with_partial=True)[1])
    for i in dd_idx:
        t_obs = float(ddor_rows.t_used[i])
        a_state = _interp_state(t_grid, ddor_rows.hist_a, t_obs)
        b_fn = lambda t: _interp_state(t_grid, ddor_rows.hist_b, t)  # noqa: E731
        an["ddor"].append(R.delta_dor_partial(R.solve_delta_dor(t_obs, a_state, b_fn, sc_aug, eph, model), sc_aug))
    an = {kk: np.asarray(v, float).reshape(-1, 7) for kk, v in an.items() if len(v)}

    # finite differences: 6 state components + constant K, three steps each
    fd = {kk: np.zeros((len(FD_STEPS["pos"]),) + v.shape) for kk, v in an.items()}
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
                _, _, Xp, _, _ = _cell(orbit, k, ks=ks, dx=dx)
                vals.append(evaluate(R.SpacecraftHistory(t_grid, Xp)))
            for kk in fd:
                fd[kk][s_i, :, j] = (vals[0][kk] - vals[1][kk]) / (2.0 * h)
    result = {}
    for kk, a in an.items():
        scale = np.maximum(np.abs(a), 1e-6 * np.max(np.abs(a), axis=1, keepdims=True))
        rel = np.min(np.abs(fd[kk] - a[None]) / scale[None], axis=0)      # best step per row and parameter
        row_rel = np.min(np.abs(fd[kk] - a[None]), axis=0) / np.max(np.abs(a), axis=1, keepdims=True)
        worst = np.unravel_index(int(np.argmax(rel)), rel.shape)
        result[kk] = dict(max_rel=float(rel.max()), median_rel=float(np.median(rel)), n_rows=int(a.shape[0]),
                          worst_row_param=[int(worst[0]), int(worst[1])],
                          max_rel_row_norm=float(row_rel.max()))
    return dict(cell=f"{orbit}:{k}", fd=result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--exact-stations", action="store_true")
    ap.add_argument("--wide-k-steps", action="store_true")
    a = ap.parse_args()
    with ProcessPoolExecutor(a.workers) as ex:
        cells = list(ex.map(run_cell, list(P.CELLS)))
        fds = list(ex.map(run_fd, list(FD_CELLS)))
    canon = next(c for c in cells if c["cell"] == "CANON:0")
    gates = dict(
        G2=dict(range=max(c["g2"]["range_max_m"] for c in cells) <= 1e-6,
                doppler=max(c["g2"]["doppler_max_mps"] for c in cells) <= 1e-7,
                ddor=canon["g2"]["ddor_max_s"] <= 1e-12),
        G3=dict(frame_time_scale_range=max(max(c["g3_range_m"]["FRAME"], c["g3_range_m"]["TIME_SCALE"]) for c in cells) <= 1e-6,
                shapiro_range_A1=all(c["a1_shapiro"][b]["max_abs_diff"] <= 2e-3 and c["a1_shapiro"][b]["worst_ratio_error"] <= 0.2
                                     for c in cells for b in c["a1_shapiro"]),
                doppler=max(max(c["g3_doppler_mps"].values()) for c in cells) <= 1e-8,
                ddor=canon["ddor"]["g3_frame_max_s"] <= 1e-12),
        G5=dict(range=max(f["fd"]["range"]["max_rel"] for f in fds) <= 1e-6,
                one_way=max(f["fd"]["one_way"]["max_rel"] for f in fds) <= 1e-6,
                doppler=max(f["fd"]["doppler"]["max_rel"] for f in fds) <= 1e-5,
                ddor=next(f for f in fds if "ddor" in f["fd"])["fd"]["ddor"]["max_rel"] <= 1e-6),
        G8=dict(row_821_converged_without_two_cycle=not any(r[0] == 821 for r in canon["two_cycle_rows"]),
                row_821_residual_ok=canon["row_821_residual_s"] <= 1e-12,
                two_cycle_rows_all_cells=[(c["cell"], c["two_cycle_rows"]) for c in cells if c["two_cycle_rows"]]),
    )
    out = dict(predeclaration="studies/phase17/configs/phase_c1i_light_time_implementation_predeclaration.json",
               gates=gates, cells=cells, fd=fds)
    path = OUT.with_name("phase_c1i_qualification_exact_stations.json") if EXACT_STATIONS else OUT
    if "--wide-k-steps" in sys.argv:
        path = path.with_name(path.stem + "_wide_k.json")
    path.write_text(json.dumps(P.K._clean(out), indent=1), encoding="utf-8")
    print(json.dumps(P.K._clean(gates), indent=1))


if __name__ == "__main__":
    main()
