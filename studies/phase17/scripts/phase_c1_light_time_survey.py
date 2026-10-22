"""Phase C1: barycentric light time and Shapiro delay - materiality survey (analysis only).

Predeclared in studies/phase17/configs/phase_c1_light_time_predeclaration.json (committed before any number).
Writes studies/phase17/results/phase_c1_light_time_survey.json.

For every KP2 cell (SIMPLIFIED dynamics) the survey re-solves the production two-way range, counted Doppler,
delta-DOR (CANON k0 only) and one-way downlink light time with an independent fixed-point solver, first in
the Moon-centred frame (must reproduce production: gates S1-S3), then in the solar-system barycentric frame,
and adds Moyer's Shapiro delay per body and the TT/TDB interval conversion. Differences are the terms; their
OD impact is the linear bias mapping on the KP2 designs.

    python studies/phase17/scripts/phase_c1_light_time_survey.py [--workers 10] [--cells CANON:0,...]
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

OUT = P.ARTIFACTS / "phase_c1_light_time_survey_erratum2.json"   # earlier records are kept unchanged
C = 299792458.0
TOL_S = 1e-15
MAX_ITER = 60
BODIES = {"SUN": "10", "EARTH": "399", "MOON": "301", "JUPITER": "5"}
NOISE = {"PROJECT": dict(range=5.0, doppler=1e-4, ddor_rad=5e-9),
         "DSN_REALISTIC": dict(range=1.0, doppler=3e-5, ddor_rad=5e-9)}


# ======================================================================
# geometry in a chosen frame
# ======================================================================
class Frame:
    """Endpoint positions in the Moon-centred (offset 0) or barycentric (Moon's SSB motion added) frame.

    Erratum 2: barycentric positions (~1.5e11 m) carry ~2e-5 m of rounding per component, a ~1e-13 s floor
    under every light time. Positions are therefore taken relative to the Moon's SSB position at an ANCHOR
    epoch set per solve (the observable's receive / transmit tag); the Moon's motion from the anchor is a
    second-order Taylor expansion (SPICE velocity, acceleration by central difference). Within a 2.6 s two-way
    chain the third-order term is ~2e-8 m. Shapiro bodies are read relative to the Moon and moved the same way.
    """

    def __init__(self, name: str, et0: float):
        import spiceypy as spice
        self.name, self.et0, self._spice = name, float(et0), spice
        self._anchor = None

    def set_anchor(self, t_anchor: float) -> None:
        if self.name == "MOON":
            return
        et = self.et0 + t_anchor

        def v(e):
            return np.asarray(self._spice.spkezr("MOON", e, "J2000", "NONE", "SSB")[0][3:]) * 1e3
        self._anchor = (float(t_anchor), v(et), (v(et + 10.0) - v(et - 10.0)) / 20.0)

    def offset(self, t: float) -> np.ndarray:
        """M(t) - M(t_anchor) (zero in the Moon-centred frame)."""
        if self.name == "MOON":
            return np.zeros(3)
        ta, v, a = self._anchor
        dt = t - ta
        return v * dt + 0.5 * a * dt * dt

    def body(self, code: str, t: float) -> np.ndarray:
        """Body position relative to M(t_anchor): (B - M)(t) from SPICE plus the Moon's motion since the anchor."""
        rel = np.asarray(self._spice.spkpos(code, self.et0 + t, "J2000", "NONE", "MOON")[0]) * 1e3
        return rel + self.offset(t)


def exact_station(state_fn, et0: float, t_grid, earth_pos, earth_vel):
    """Erratum 2: the production station provider evaluates Earth rotation at the rounded et0 + t (~4.5e-5 m
    steps). Move the station by its Earth-relative velocity x the exact rounding error."""
    from lunar_od.radiometrics import _interp_state
    earth = np.hstack([np.asarray(earth_pos, float), np.asarray(earth_vel, float)])

    def f(t):
        s = np.array(state_fn(t), float)[:6]
        err = t - ((et0 + t) - et0)
        s[:3] += (s[3:6] - _interp_state(t_grid, earth, t)[3:6]) * err
        return s
    return f


def gm(code: str) -> float:
    import spiceypy as spice
    return float(spice.bodvrd(code, "GM", 1)[1][0]) * 1e9


def shapiro_s(frame: Frame, r_tx: np.ndarray, r_rx: np.ndarray, t_rx: float) -> dict:
    """Moyer (2000) eq. 8-38 per body, (1 + gamma) = 2, body position at the leg's receive epoch [s]."""
    r12 = float(np.linalg.norm(r_rx - r_tx))
    out = {}
    for name, code in BODIES.items():
        b = frame.body(code, t_rx)
        r1, r2 = float(np.linalg.norm(r_tx - b)), float(np.linalg.norm(r_rx - b))
        out[name] = 2.0 * gm(code) / C ** 3 * np.log((r1 + r2 + r12) / (r1 + r2 - r12))
    return out


def leg_backward(frame, t_rx, r_rx, sc_pos):
    """Receiver fixed at (t_rx, r_rx); solve t_tx with t_rx - t_tx = |r_rx - r_tx(t_tx)| / c."""
    tau = float(np.linalg.norm(r_rx - (sc_pos(t_rx) + frame.offset(t_rx)))) / C
    for _ in range(MAX_ITER):
        t_tx = t_rx - tau
        new = float(np.linalg.norm(r_rx - (sc_pos(t_tx) + frame.offset(t_tx)))) / C
        if abs(new - tau) < TOL_S:
            tau = new
            break
        tau = new
    t_tx = t_rx - tau
    r_tx = sc_pos(t_tx) + frame.offset(t_tx)
    return tau, t_tx, r_tx, abs(tau - float(np.linalg.norm(r_rx - r_tx)) / C)


def leg_uplink(frame, t2, r2, st_pos):
    """Spacecraft fixed at (t2, r2); solve station transmit t1 with t2 - t1 = |r2 - r_st(t1)| / c."""
    tau = float(np.linalg.norm(r2 - (st_pos(t2) + frame.offset(t2)))) / C
    for _ in range(MAX_ITER):
        t1 = t2 - tau
        new = float(np.linalg.norm(r2 - (st_pos(t1) + frame.offset(t1)))) / C
        if abs(new - tau) < TOL_S:
            tau = new
            break
        tau = new
    t1 = t2 - tau
    r1 = st_pos(t1) + frame.offset(t1)
    return tau, t1, r1, abs(tau - float(np.linalg.norm(r2 - r1)) / C)


def leg_forward(frame, t_tx, r_tx, st_pos):
    """Transmitter fixed at (t_tx, r_tx); solve station receive t_rx with t_rx - t_tx = |r_st(t_rx) - r_tx| / c."""
    tau = float(np.linalg.norm(st_pos(t_tx) + frame.offset(t_tx) - r_tx)) / C
    for _ in range(MAX_ITER):
        t_rx = t_tx + tau
        new = float(np.linalg.norm(st_pos(t_rx) + frame.offset(t_rx) - r_tx)) / C
        if abs(new - tau) < TOL_S:
            tau = new
            break
        tau = new
    t_rx = t_tx + tau
    r_rx = st_pos(t_rx) + frame.offset(t_rx)
    return tau, t_rx, r_rx, abs(tau - float(np.linalg.norm(r_rx - r_tx)) / C)


def two_way(frame, t3, st_pos, sc_pos, with_shapiro):
    """Receive-tagged two-way chain, zero transponder delay. Returns light times, Shapiro per body, residuals."""
    frame.set_anchor(t3)
    r3 = st_pos(t3) + frame.offset(t3)
    td, t2, r2, res_d = leg_backward(frame, t3, r3, sc_pos)
    tu, t1, r1, res_u = leg_uplink(frame, t2, r2, st_pos)
    sh = {}
    if with_shapiro:
        sd, su = shapiro_s(frame, r2, r3, t3), shapiro_s(frame, r1, r2, t2)
        sh = {b: sd[b] + su[b] for b in BODIES}
        sh_down = sd
    else:
        sh_down = {}
    return dict(down=td, up=tu, shapiro=sh, shapiro_down=sh_down, res=max(res_d, res_u))


def tdb_minus_tt_rate(et0: float, t: float) -> float:
    """Erratum 1: analytic derivative of SPICE's TDB - TT = K sin(M + EB sin M) model (the original central
    difference of unitim was dominated by ulp(et) ~ 1.2e-7 s quantisation)."""
    import spiceypy as spice
    k, eb = spice.gdpool("DELTET/K", 0, 1)[0], spice.gdpool("DELTET/EB", 0, 1)[0]
    m0, m1 = spice.gdpool("DELTET/M", 0, 2)
    m = m0 + m1 * (et0 + t)
    return float(k * np.cos(m + eb * np.sin(m)) * (1.0 + eb * np.cos(m)) * m1)


# ======================================================================
# one cell
# ======================================================================
def run_cell(spec) -> dict:
    orbit, k = spec
    import phase17_r1o_cd1_core as CD
    import phase17_r1od2_core as D2
    from lunar_od.radiometrics import _interp_state, resolve_counted_doppler_station_state_provider
    from lunar_od.two_way_range import _station_state_providers_for_pass

    t_start = time.time()
    P.K._init()
    x0 = np.asarray(P.x0s()[orbit], float)
    scn = P.K.scenario(orbit, x0.tolist(), k)
    X, PHI, S = K2.propagate_segmented(scn, x0, np.full(3, K2.K0), "SIMPLIFIED")
    fx, zr, zd = K2.build_fixed(scn, X, PHI, S, P.K.stations())
    rb, rows = fx.rb, fx.rows
    t_grid, x6 = rb.t_grid, rb.nom48[:, :6]
    pg = rb.pass_geo
    et0 = float(pg.et0_s)
    frames = {"MOON": Frame("MOON", et0), "SSB": Frame("SSB", et0)}

    def sc_pos(t):
        return _interp_state(t_grid, x6, t)[:3]

    # ---------------- two-way range and one-way downlink ----------------
    provs = _station_state_providers_for_pass(pg)
    nr = rb.obs.shape[0]
    rng = {f: np.zeros(nr) for f in frames}
    one = {f: np.zeros(nr) for f in frames}
    sh_r = {b: np.zeros(nr) for b in BODIES}
    ts_r = np.zeros(nr)
    res_max = 0.0
    res_rows = []  # (observable, frame, row, residual) for every solve above 1e-12 s
    for i, (t3, _, sid, _) in enumerate(rb.obs[:, :4]):
        st = provs[int(sid) - 1]

        exact = exact_station(lambda t, st=st: np.asarray(st.state(t), float), et0, t_grid, pg.earth_pos_mci_m,
                              pg.earth_vel_mci_mps)

        def st_pos(t, f=exact):
            return f(t)[:3]
        for f, fr in frames.items():
            sol = two_way(fr, float(t3), st_pos, sc_pos, with_shapiro=(f == "SSB"))
            rng[f][i] = 0.5 * C * (sol["down"] + sol["up"])
            one[f][i] = C * sol["down"]
            res_max = max(res_max, sol["res"])
            if sol["res"] > 1e-12:
                res_rows.append(("range", f, i, sol["res"]))
            if f == "SSB":
                for b in BODIES:
                    sh_r[b][i] = 0.5 * C * sol["shapiro"][b]
                ts_r[i] = -0.5 * C * (sol["down"] + sol["up"]) * tdb_minus_tt_rate(et0, float(t3))

    # ---------------- counted Doppler ----------------
    cfg = CD.doppler_config(60.0)
    tc = 60.0
    dprov = {s: resolve_counted_doppler_station_state_provider(cfg, fx.stations[s], t_grid, pg.earth_pos_mci_m,
                                                               pg.earth_vel_mci_mps, et0_s=et0)
             for s in set(int(x) for x in rows.station)}
    nd = rows.n
    dop = {f: np.zeros(nd) for f in frames}
    sh_d = {b: np.zeros(nd) for b in BODIES}
    ts_d = np.zeros(nd)
    for i in range(nd):
        s, tm = int(rows.station[i]), float(rows.t_mid[i])

        exact = exact_station(lambda t, p=dprov[s]: np.asarray(p.state(t), float), et0, t_grid,
                              pg.earth_pos_mci_m, pg.earth_vel_mci_mps)

        def st_pos(t, f=exact):
            return f(t)[:3]
        for f, fr in frames.items():
            a = two_way(fr, tm - tc / 2, st_pos, sc_pos, with_shapiro=(f == "SSB"))
            b = two_way(fr, tm + tc / 2, st_pos, sc_pos, with_shapiro=(f == "SSB"))
            dop[f][i] = C / (2 * tc) * ((b["down"] + b["up"]) - (a["down"] + a["up"]))
            res_max = max(res_max, a["res"], b["res"])
            if max(a["res"], b["res"]) > 1e-12:
                res_rows.append(("doppler", f, i, max(a["res"], b["res"])))
            if f == "SSB":
                for body in BODIES:
                    sh_d[body][i] = C / (2 * tc) * (b["shapiro"][body] - a["shapiro"][body])
                ts_d[i] = -dop["SSB"][i] * tdb_minus_tt_rate(et0, tm)

    # ---------------- delta-DOR (CANON k0, Goldstone-Canberra) ----------------
    ddor = None
    if (orbit, k) == ("CANON", 0):
        dr = D2.build_ddor_rows(scn, rb, "GC", D2.ROW_RULE_PRODUCTION_VISIBILITY)

        def hist_pos(h):
            return lambda t: _interp_state(t_grid, h, t)[:3]
        a_pos, b_pos = hist_pos(dr.hist_a), hist_pos(dr.hist_b)
        ds = {f: np.zeros(dr.n_converged) for f in frames}
        sh_dd = {b: np.zeros(dr.n_converged) for b in BODIES}
        for i, t_obs in enumerate(dr.t_used):
            t_obs = float(t_obs)
            for f, fr in frames.items():
                fr.set_anchor(t_obs)
                ra = a_pos(t_obs) + fr.offset(t_obs)          # station A frozen at t_obs (production contract)
                tau_a, t_tx, r_tx, ra_res = leg_backward(fr, t_obs, ra, sc_pos)
                tau_b, t_b, r_b, rb_res = leg_forward(fr, t_tx, r_tx, b_pos)
                ds[f][i] = tau_b - tau_a                       # local delays: t_B - t_obs without epoch rounding
                res_max = max(res_max, ra_res, rb_res)
                if max(ra_res, rb_res) > 1e-12:
                    res_rows.append(("ddor", f, i, max(ra_res, rb_res)))
                if f == "SSB":
                    sa, sb = shapiro_s(fr, r_tx, ra, t_obs), shapiro_s(fr, r_tx, r_b, t_b)
                    for body in BODIES:
                        sh_dd[body][i] = sb[body] - sa[body]
        to_rad = C / np.maximum(dr.b_perp, 1.0)
        ddor = dict(n=int(dr.n_converged),
                    parity_max_s=float(np.max(np.abs(ds["MOON"] - dr.d_s))),
                    frame_max_ns=float(np.max(np.abs(ds["SSB"] - ds["MOON"])) * 1e9),
                    frame_max_nrad=float(np.max(np.abs((ds["SSB"] - ds["MOON"]) * to_rad)) * 1e9),
                    shapiro_max_nrad={b: float(np.max(np.abs(sh_dd[b] * to_rad)) * 1e9) for b in BODIES})

    # ---------------- OD impact (linear bias mapping) ----------------
    ev = K2.evaluate(fx, X, PHI, S)
    terms_r = dict(FRAME=rng["SSB"] - rng["MOON"], TIME_SCALE=ts_r,
                   **{"SHAPIRO_" + b: sh_r[b] for b in BODIES})
    terms_d = dict(FRAME=dop["SSB"] - dop["MOON"], TIME_SCALE=ts_d,
                   **{"SHAPIRO_" + b: sh_d[b] for b in BODIES})
    od = {}
    for design in ("RANGE_ONLY", "RANGE_DOPPLER_FREE_BIAS"):
        use_d = design != "RANGE_ONLY"
        a, q = K2._design(ev, fx, "M0", "FREE", use_d)
        for nl, sig in NOISE.items():
            w = np.concatenate([np.full(nr, 1 / sig["range"])] + ([np.full(nd, 1 / sig["doppler"])] if use_d else []))
            prior = np.zeros((6, a.shape[1]))
            prior[:, :6] = np.diag(1 / K2.PRIOR_SD)
            aw = np.vstack([a * w[:, None], prior])
            scale = np.linalg.norm(aw, axis=0)
            _, r = np.linalg.qr(aw / scale)
            rinv = np.linalg.inv(r)
            sigma = np.sqrt(np.sum(rinv ** 2, axis=1)) / scale
            for term in terms_r:
                dy = np.concatenate([terms_r[term]] + ([terms_d[term]] if use_d else []))
                dth = np.linalg.lstsq(aw / scale, np.concatenate([dy * w, np.zeros(6)]), rcond=None)[0] / scale
                ratio = np.abs(dth[:7] / sigma[:7])
                od[f"{design}|{nl}|{term}"] = dict(max_state_over_sigma=float(ratio[:6].max()),
                                                    k_over_sigma=float(ratio[6]))

    def mx(v):
        return float(np.max(np.abs(v)))
    return dict(
        cell=f"{orbit}:{k}", n_range=nr, n_doppler=nd, elapsed_s=time.time() - t_start,
        parity=dict(range_max_m=mx(rng["MOON"] - zr), doppler_max_mps=mx(dop["MOON"] - zd),
                    ddor_max_s=(ddor["parity_max_s"] if ddor else None)),
        convergence_max_residual_s=res_max, convergence_rows_over_1e12=res_rows,
        range_m={t: dict(min=float(v.min()), max=float(v.max()), maxabs=mx(v)) for t, v in terms_r.items()},
        doppler_mps={t: dict(min=float(v.min()), max=float(v.max()), maxabs=mx(v)) for t, v in terms_d.items()},
        one_way_frame_m=dict(min=float((one["SSB"] - one["MOON"]).min()), max=float((one["SSB"] - one["MOON"]).max())),
        ddor=ddor, od=od)


# ======================================================================
# population, gates, classification
# ======================================================================
def classify(cells: list[dict]) -> dict:
    terms = list(cells[0]["range_m"])
    worst = lambda key, t: max(c[key][t]["maxabs"] for c in cells)  # noqa: E731
    dsn = NOISE["DSN_REALISTIC"]
    out = {}
    for t in terms:
        obs_r = worst("range_m", t) / dsn["range"]
        obs_d = worst("doppler_mps", t) / dsn["doppler"]
        od = max(max(v["max_state_over_sigma"], v["k_over_sigma"])
                 for c in cells for kk, v in c["od"].items() if kk.endswith("|" + t))
        ddor_frac = None
        dd = [c["ddor"] for c in cells if c["ddor"]]
        if dd:
            val = dd[0]["frame_max_nrad"] if t == "FRAME" else dd[0]["shapiro_max_nrad"].get(t.replace("SHAPIRO_", ""))
            ddor_frac = None if val is None else val / (dsn["ddor_rad"] * 1e9)
        material_obs = max(obs_r, obs_d, ddor_frac or 0.0) >= 0.1
        out[t] = dict(range_over_sigma_dsn=obs_r, doppler_over_sigma_dsn=obs_d, ddor_over_sigma=ddor_frac,
                      od_max_over_sigma=od, OBSERVABLE_MATERIAL=material_obs, OD_MATERIAL=od >= 0.1,
                      decision="IMPLEMENT" if (material_obs or od >= 0.1) else "DOCUMENT_ONLY")
    return out


def gates(cells: list[dict]) -> dict:
    return dict(
        S1_parity_range=max(c["parity"]["range_max_m"] for c in cells) <= 1e-6,
        S2_parity_doppler=max(c["parity"]["doppler_max_mps"] for c in cells) <= 1e-7,
        S3_parity_ddor=all(c["parity"]["ddor_max_s"] <= 1e-12 for c in cells if c["parity"]["ddor_max_s"] is not None),
        S4_convergence=max(c["convergence_max_residual_s"] for c in cells) <= 1e-12,
        values=dict(range_max_m=max(c["parity"]["range_max_m"] for c in cells),
                    doppler_max_mps=max(c["parity"]["doppler_max_mps"] for c in cells),
                    residual_max_s=max(c["convergence_max_residual_s"] for c in cells)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--cells", default="")
    a = ap.parse_args()
    cells = ([tuple((s.split(":")[0], int(s.split(":")[1]))) for s in a.cells.split(",")] if a.cells
             else list(P.CELLS))
    with ProcessPoolExecutor(min(a.workers, len(cells))) as ex:
        res = list(ex.map(run_cell, cells))
    out = dict(predeclaration="studies/phase17/configs/phase_c1_light_time_predeclaration.json",
               cells=res, gates=gates(res), classification=classify(res))
    path = OUT if not a.cells else OUT.with_name("phase_c1_light_time_survey_partial.json")
    path.write_text(json.dumps(P.K._clean(out), indent=1), encoding="utf-8")
    print(json.dumps(dict(gates=out["gates"], classification=out["classification"]), indent=1))


if __name__ == "__main__":
    main()
