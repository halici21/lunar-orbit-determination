"""Phase C2: time scales (UTC/TAI/TT/TDB) and station clocks - materiality survey (analysis only).

Predeclared in studies/phase17/configs/phase_c2_time_scale_predeclaration.json (committed before any number;
amendment A1 fixes the delta-DOR propagation before any delta-DOR number). Two steps:

    python studies/phase17/scripts/phase_c2_time_scale_survey.py --export-epochs
        writes studies/phase17/results/phase_c2_survey_epochs.json (every station tag epoch of the population)
    <agent-sandbox>/c2-erfa-reference/.venv/Scripts/python.exe -I erfa_reference.py <epochs> <reference>
        writes studies/phase17/results/phase_c2_erfa_reference.json and prints its sha256
    python studies/phase17/scripts/phase_c2_time_scale_survey.py --reference-sha256 <sha> [--workers 10]
        writes studies/phase17/results/phase_c2_time_scale_survey.json

Baseline: relativistic_light_time.DSN_LIGHT_TIME_MODEL (C1-I). A station-time error that moves the true TDB
of a tag by dt changes an observable by dt * dh/dt; an error in d(dt)/dt scales the station-measured
interval. dh/dt is a central difference of the baseline with h = 1 s. Counted Doppler is propagated per
count-interval endpoint; delta-DOR per station (A1). OD impact is the C1 linear bias mapping.
"""
from __future__ import annotations

import argparse
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
warnings.simplefilter("ignore")

import phase17_kp2_campaign as P  # noqa: E402
import phase17_kp2_core as K2  # noqa: E402

OUT = P.ARTIFACTS / "phase_c2_time_scale_survey.json"
EPOCHS = P.ARTIFACTS / "phase_c2_survey_epochs.json"
REFERENCE = P.ARTIFACTS / "phase_c2_erfa_reference.json"
C = 299792458.0
TC_S = 60.0
H_DIFF_S = 1.0
DT_LINEARITY_S = 1e-2
NOISE = {"PROJECT": dict(range=5.0, doppler=1e-4, ddor_rad=5e-9),
         "DSN_REALISTIC": dict(range=1.0, doppler=3e-5, ddor_rad=5e-9)}
CLOCK_OFFSETS_S = (1e-8, 1e-7, 1e-6)
CLOCK_FREQUENCIES = (1e-13, 1e-12)
DDOR_CELL = ("CANON", 0)
S5_OFFSET_S = 1e-3          # parity probe offset: far above ulp(et0) so the production epoch shift is resolved
S5_ROWS = 40


def _term_names():
    names = ["T1_SIZE", "T1C_SPICE_CHAIN", "T2_EPOCH", "T2_RATE", "T2", "T3_EPOCH", "T3_RATE", "T3"]
    names += [f"T4_OFFSET_{x:.0e}" for x in CLOCK_OFFSETS_S] + [f"T4_FREQ_{x:.0e}" for x in CLOCK_FREQUENCIES]
    return names


TERMS = _term_names()


# ======================================================================
# SPICE side of the time relation (the baseline)
# ======================================================================
def spice_tdb_minus_tt(et0: float, t: float) -> float:
    """SPICE's DELTET model TDB - TT = K sin(E), E = M + EB sin M (analytic; the et0 + t rounding moves M by
    M1 * 1.2e-7 s ~ 2e-14 rad, i.e. < 1e-16 s in the value)."""
    import spiceypy as spice
    k, eb = spice.gdpool("DELTET/K", 0, 1)[0], spice.gdpool("DELTET/EB", 0, 1)[0]
    m0, m1 = spice.gdpool("DELTET/M", 0, 2)
    m = m0 + m1 * (et0 + t)
    return float(k * np.sin(m + eb * np.sin(m)))


def spice_tdb_minus_utc(et0: float, t: float) -> float:
    import spiceypy as spice
    return float(spice.deltet(et0 + t, "ET"))


# ======================================================================
# one cell: geometry
# ======================================================================
def build_cell(orbit: str, k: int):
    from lunar_od import relativistic_light_time as R
    from lunar_od.radiometrics import resolve_counted_doppler_station_state_provider
    from lunar_od.two_way_range import _station_state_providers_for_pass
    import phase17_r1o_cd1_core as CD

    P.K._init()
    x0 = np.asarray(P.x0s()[orbit], float)
    scn = P.K.scenario(orbit, x0.tolist(), k)
    X, PHI, S = K2.propagate_segmented(scn, x0, np.full(3, K2.K0), "SIMPLIFIED")
    fx, zr, zd = K2.build_fixed(scn, X, PHI, S, P.K.stations())
    rb, rows = fx.rb, fx.rows
    pg = rb.pass_geo
    et0 = float(pg.et0_s)

    def exact(p):
        return R.exact_epoch_station_state(lambda t, p=p: np.asarray(p.state(t), float), et0, pg.t_s,
                                           pg.earth_pos_mci_m, pg.earth_vel_mci_mps)
    range_st = [exact(p) for p in _station_state_providers_for_pass(pg)]
    cfg = CD.doppler_config(TC_S)
    dop_st = {s: exact(resolve_counted_doppler_station_state_provider(cfg, fx.stations[s], rb.t_grid,
                                                                      pg.earth_pos_mci_m, pg.earth_vel_mci_mps,
                                                                      et0_s=et0))
              for s in set(int(x) for x in rows.station)}
    ddor = None
    if (orbit, k) == DDOR_CELL:
        import phase17_r1od2_core as D2
        dr = D2.build_ddor_rows(scn, rb, "GC", D2.ROW_RULE_PRODUCTION_VISIBILITY)
        ddor = dict(rows=dr, names=D2.PAIRS["GC"])
    return dict(scn=scn, X=X, PHI=PHI, S=S, fx=fx, rb=rb, rows=rows, pg=pg, et0=et0, zd=zd, cfg=cfg,
                range_st=range_st, dop_st=dop_st, ddor=ddor,
                names=[s.name for s in fx.stations])


def export_cell(spec) -> tuple[str, dict]:
    orbit, k = spec
    g = build_cell(orbit, k)
    per = {}
    for t3, _, sid, _ in g["rb"].obs[:, :4]:
        per.setdefault(g["names"][int(sid) - 1], set()).add(float(t3))
    for s, tm in zip(g["rows"].station, g["rows"].t_mid):
        for t in (float(tm) - TC_S / 2, float(tm) + TC_S / 2):
            per.setdefault(g["names"][int(s)], set()).add(t)
    if g["ddor"]:
        for name in g["ddor"]["names"]:
            per.setdefault(name, set()).update(float(t) for t in g["ddor"]["rows"].t_used)
    return f"{orbit}:{k}", dict(et0=g["et0"], stations={n: sorted(v) for n, v in per.items()})


# ======================================================================
# reference lookup
# ======================================================================
class Reference:
    def __init__(self, cell_record: dict):
        self.cell = cell_record
        self.idx = {n: {t: i for i, t in enumerate(r["t"])} for n, r in cell_record["stations"].items()}
        self.misses = 0

    def get(self, station: str, t: float) -> dict:
        r = self.cell["stations"][station]
        i = self.idx[station].get(float(t))
        if i is None:
            self.misses += 1
            raise KeyError(f"{station} t={t!r} not in the ERFA reference")
        return {k: r[k][i] for k in ("geo", "topo", "geo_rate", "topo_rate", "tai_utc", "tdb_utc")}


# ======================================================================
# one cell: terms
# ======================================================================
def run_cell(spec) -> dict:
    orbit, k = spec
    from lunar_od import relativistic_light_time as R
    from lunar_od.radiometrics import _interp_state, two_way_counted_doppler_observable
    from dataclasses import replace

    t_start = time.time()
    ref_all = json.loads(REFERENCE.read_text(encoding="utf-8"))
    g = build_cell(orbit, k)
    cell = f"{orbit}:{k}"
    ref = Reference(ref_all["cells"][cell])
    et0, rb, rows, names = g["et0"], g["rb"], g["rows"], g["names"]
    t_grid = rb.t_grid
    t_ref = float(t_grid[0])
    model = R.DSN_LIGHT_TIME_MODEL
    eph = R.BarycentricEphemeris(et0)
    sc = R.SpacecraftHistory(t_grid, rb.nom48[:, :6])
    two_cycles = 0

    def rt(t, st):
        nonlocal two_cycles
        sol = R.solve_two_way(float(t), st, sc, eph, model)
        two_cycles += int(sol.two_cycle)
        return sol

    def spice(t):
        return spice_tdb_minus_tt(et0, t), eph.tdb_minus_tt_rate(t)

    # ---------------- two-way range ----------------
    nr = rb.obs.shape[0]
    tr = {n: np.zeros(nr) for n in TERMS}
    lin_r = np.zeros(nr)
    s1_r, s3_r = [], []
    for i, (t3, _, sid, _) in enumerate(rb.obs[:, :4]):
        t3, st, name = float(t3), g["range_st"][int(sid) - 1], names[int(sid) - 1]
        sol = rt(t3, st)
        h0, round_trip = R.two_way_range_m(sol), sol.round_trip_s
        hdot = (R.two_way_range_m(rt(t3 + H_DIFF_S, st)) - R.two_way_range_m(rt(t3 - H_DIFF_S, st))) / (2 * H_DIFF_S)
        direct = R.two_way_range_m(rt(t3 + DT_LINEARITY_S, st)) - h0
        lin_r[i] = abs(direct - DT_LINEARITY_S * hdot)
        e = ref.get(name, t3)
        d_sp, r_sp = spice(t3)
        s3_r.append(abs(e["geo"] - d_sp))
        s1_r.append((e["tai_utc"], spice_tdb_minus_utc(et0, t3) - 32.184 - d_sp))
        tr["T1_SIZE"][i] = e["tdb_utc"] * hdot
        tr["T1C_SPICE_CHAIN"][i] = ((e["tdb_utc"] - e["geo"]) - (spice_tdb_minus_utc(et0, t3) - d_sp)) * hdot
        tr["T2_EPOCH"][i] = (e["geo"] - d_sp) * hdot
        tr["T2_RATE"][i] = -0.5 * C * round_trip * (e["geo_rate"] - r_sp)
        tr["T3_EPOCH"][i] = e["topo"] * hdot
        tr["T3_RATE"][i] = -0.5 * C * round_trip * e["topo_rate"]
        for off in CLOCK_OFFSETS_S:
            tr[f"T4_OFFSET_{off:.0e}"][i] = -off * hdot
        for f in CLOCK_FREQUENCIES:
            tr[f"T4_FREQ_{f:.0e}"][i] = -f * (t3 - t_ref) * hdot + 0.5 * C * round_trip * f
    tr["T2"] = tr["T2_EPOCH"] + tr["T2_RATE"]
    tr["T3"] = tr["T3_EPOCH"] + tr["T3_RATE"]

    # ---------------- counted Doppler (per count-interval endpoint) ----------------
    nd = rows.n
    td = {n: np.zeros(nd) for n in TERMS}
    lin_d = np.zeros(nd)
    kk = C / (2 * TC_S)
    for i in range(nd):
        s, tm = int(rows.station[i]), float(rows.t_mid[i])
        st, name = g["dop_st"][s], names[s]
        y0 = R.counted_doppler(tm, TC_S, st, sc, eph, model)[0]
        ydot = (R.counted_doppler(tm + H_DIFF_S, TC_S, st, sc, eph, model)[0]
                - R.counted_doppler(tm - H_DIFF_S, TC_S, st, sc, eph, model)[0]) / (2 * H_DIFF_S)
        lin_d[i] = abs(R.counted_doppler(tm + DT_LINEARITY_S, TC_S, st, sc, eph, model)[0] - y0
                       - DT_LINEARITY_S * ydot)
        ends = []
        for t in (tm - TC_S / 2, tm + TC_S / 2):
            rtt = rt(t, st).round_trip_s
            rdot = (rt(t + H_DIFF_S, st).round_trip_s - rt(t - H_DIFF_S, st).round_trip_s) / (2 * H_DIFF_S)
            e = ref.get(name, t)
            d_sp, r_sp = spice(t)
            ends.append(dict(t=t, rt=rtt, rdot=rdot, e=e, d_sp=d_sp, r_sp=r_sp,
                             sp_utc=spice_tdb_minus_utc(et0, t)))
        a, b = ends

        def epoch(da, db):
            return kk * (b["rdot"] * db - a["rdot"] * da)

        def rate(ra, rb_):
            return -kk * (b["rt"] * rb_ - a["rt"] * ra)
        td["T1_SIZE"][i] = epoch(a["e"]["tdb_utc"], b["e"]["tdb_utc"])
        chain = [(x["e"]["tdb_utc"] - x["e"]["geo"]) - (x["sp_utc"] - x["d_sp"]) for x in ends]
        td["T1C_SPICE_CHAIN"][i] = epoch(*chain)
        td["T2_EPOCH"][i] = epoch(a["e"]["geo"] - a["d_sp"], b["e"]["geo"] - b["d_sp"])
        td["T2_RATE"][i] = rate(a["e"]["geo_rate"] - a["r_sp"], b["e"]["geo_rate"] - b["r_sp"])
        td["T3_EPOCH"][i] = epoch(a["e"]["topo"], b["e"]["topo"])
        td["T3_RATE"][i] = rate(a["e"]["topo_rate"], b["e"]["topo_rate"])
        for off in CLOCK_OFFSETS_S:
            td[f"T4_OFFSET_{off:.0e}"][i] = -epoch(off, off)
        for f in CLOCK_FREQUENCIES:
            td[f"T4_FREQ_{f:.0e}"][i] = -epoch(f * (a["t"] - t_ref), f * (b["t"] - t_ref)) + y0 * f
    td["T2"] = td["T2_EPOCH"] + td["T2_RATE"]
    td["T3"] = td["T3_EPOCH"] + td["T3_RATE"]

    # ---------------- delta-DOR (amendment A1: per-station tags) ----------------
    ddor = None
    if g["ddor"]:
        dr, (na, nb) = g["ddor"]["rows"], g["ddor"]["names"]
        b_fn = lambda t: _interp_state(t_grid, dr.hist_b, t)  # noqa: E731

        def delay(t):
            a_state = _interp_state(t_grid, dr.hist_a, float(t))
            return R.solve_delta_dor(float(t), a_state, b_fn, sc, eph, model)
        n = len(dr.t_used)
        to_rad = C / np.maximum(np.asarray(dr.b_perp, float)[:n], 1.0)
        dd = {n_: np.zeros(n) for n_ in ("T1_SIZE", "T2_EPOCH", "T3_EPOCH", "T3_DIFFERENTIAL", "T3",
                                         "TT_INTERVAL_INFO")
              + tuple(f"T4_OFFSET_{x:.0e}" for x in CLOCK_OFFSETS_S)
              + tuple(f"T4_FREQ_{x:.0e}" for x in CLOCK_FREQUENCIES)}
        for i, t_obs in enumerate(dr.t_used):
            t_obs = float(t_obs)
            sol = delay(t_obs)
            dly, t_b = sol.differential_delay_s, sol.leg_b.t_rx_s
            ddot = (delay(t_obs + H_DIFF_S).differential_delay_s
                    - delay(t_obs - H_DIFF_S).differential_delay_s) / (2 * H_DIFF_S)
            ea, eb = ref.get(na, t_obs), ref.get(nb, t_obs)
            d_sp, _ = spice(t_obs)
            topo_b = eb["topo"] + eb["topo_rate"] * (t_b - t_obs)
            dd["T1_SIZE"][i] = ea["tdb_utc"] * ddot
            dd["T2_EPOCH"][i] = (ea["geo"] - d_sp) * ddot
            dd["T3_EPOCH"][i] = ea["topo"] * ddot
            dd["T3_DIFFERENTIAL"][i] = -(topo_b - ea["topo"])
            dd["TT_INTERVAL_INFO"][i] = -ea["geo_rate"] * dly
            for off in CLOCK_OFFSETS_S:
                dd[f"T4_OFFSET_{off:.0e}"][i] = off
            for f in CLOCK_FREQUENCIES:
                dd[f"T4_FREQ_{f:.0e}"][i] = f * (t_obs - t_ref) + f * dly
        dd["T2"] = dd["T2_EPOCH"]
        dd["T3"] = dd["T3_EPOCH"] + dd["T3_DIFFERENTIAL"]
        ddor = dict(n=n, b_perp_min_m=float(np.min(np.asarray(dr.b_perp, float)[:n])),
                    max_nrad={k_: float(np.max(np.abs(v * to_rad)) * 1e9) for k_, v in dd.items()},
                    max_ns={k_: float(np.max(np.abs(v)) * 1e9) for k_, v in dd.items()},
                    t3_differential_ns=dict(min=float(dd["T3_DIFFERENTIAL"].min() * 1e9),
                                            max=float(dd["T3_DIFFERENTIAL"].max() * 1e9),
                                            spread=float(np.ptp(dd["T3_DIFFERENTIAL"]) * 1e9)))

    # ---------------- S5: production clock parity (informational, CANON:0) ----------------
    s5 = None
    if (orbit, k) == DDOR_CELL:
        pgt, fx, X = g["pg"], g["fx"], g["X"]
        cfg_off = replace(g["cfg"], station_clock_offset_s=S5_OFFSET_S)
        tail = lambda cfg: (pgt.earth_pos_mci_m, pgt.earth_vel_mci_mps, pgt.x_j2000_to_itrf93, cfg)  # noqa: E731
        from lunar_od.radiometrics import resolve_counted_doppler_station_state_provider
        rel = []
        for i in range(min(S5_ROWS, nd)):
            s, tm = int(rows.station[i]), float(rows.t_mid[i])
            prov = resolve_counted_doppler_station_state_provider(g["cfg"], fx.stations[s], t_grid,
                                                                  pgt.earth_pos_mci_m, pgt.earth_vel_mci_mps,
                                                                  et0_s=et0)

            def yp(t, cfg):
                return two_way_counted_doppler_observable(t, fx.stations[s], t_grid, X, *tail(cfg), et0_s=et0,
                                                          station_state_provider=prov)
            shifted = yp(tm, cfg_off) - yp(tm, g["cfg"])
            expect = S5_OFFSET_S * (yp(tm + H_DIFF_S, g["cfg"]) - yp(tm - H_DIFF_S, g["cfg"])) / (2 * H_DIFF_S)
            rel.append(abs(shifted - expect) / max(abs(expect), 1e-12))
        s5 = dict(offset_s=S5_OFFSET_S, rows=len(rel), max_relative_diff=float(max(rel)),
                  median_relative_diff=float(np.median(rel)))

    # ---------------- OD impact (linear bias mapping, as C1) ----------------
    ev = K2.evaluate(g["fx"], g["X"], g["PHI"], g["S"])
    od = {}
    for design in ("RANGE_ONLY", "RANGE_DOPPLER_FREE_BIAS"):
        use_d = design != "RANGE_ONLY"
        a_, _ = K2._design(ev, g["fx"], "M0", "FREE", use_d)
        for nl, sig in NOISE.items():
            w = np.concatenate([np.full(nr, 1 / sig["range"])] + ([np.full(nd, 1 / sig["doppler"])] if use_d else []))
            prior = np.zeros((6, a_.shape[1]))
            prior[:, :6] = np.diag(1 / K2.PRIOR_SD)
            aw = np.vstack([a_ * w[:, None], prior])
            scale = np.linalg.norm(aw, axis=0)
            _, r = np.linalg.qr(aw / scale)
            sigma = np.sqrt(np.sum(np.linalg.inv(r) ** 2, axis=1)) / scale
            for term in TERMS:
                dy = np.concatenate([tr[term]] + ([td[term]] if use_d else []))
                dth = np.linalg.lstsq(aw / scale, np.concatenate([dy * w, np.zeros(6)]), rcond=None)[0] / scale
                ratio = np.abs(dth[:7] / sigma[:7])
                od[f"{design}|{nl}|{term}"] = dict(max_state_over_sigma=float(ratio[:6].max()),
                                                    k_over_sigma=float(ratio[6]))

    def stats(v):
        return dict(min=float(v.min()), max=float(v.max()), maxabs=float(np.max(np.abs(v))))
    tai_utc = sorted({round(x[0], 9) for x in s1_r})
    return dict(
        cell=cell, et0=et0, n_range=nr, n_doppler=nd, elapsed_s=time.time() - t_start,
        gates=dict(S1_tai_utc_erfa=tai_utc,
                   S1_tai_utc_spice=sorted({round(x[1], 6) for x in s1_r}),
                   S2_range_max_m=float(lin_r.max()), S2_doppler_max_mps=float(lin_d.max()),
                   S3_max_s=float(max(s3_r)), S4_two_cycles=two_cycles, S6_reference_misses=ref.misses),
        range_m={t: stats(v) for t, v in tr.items()},
        doppler_mps={t: stats(v) for t, v in td.items()},
        ddor=ddor, s5=s5, od=od)


# ======================================================================
# population, gates, classification
# ======================================================================
def classify(cells: list[dict]) -> dict:
    dsn = NOISE["DSN_REALISTIC"]
    dd = next((c["ddor"] for c in cells if c["ddor"]), None)
    out = {}
    for t in TERMS:
        obs_r = max(c["range_m"][t]["maxabs"] for c in cells) / dsn["range"]
        obs_d = max(c["doppler_mps"][t]["maxabs"] for c in cells) / dsn["doppler"]
        od = max(max(v["max_state_over_sigma"], v["k_over_sigma"])
                 for c in cells for kk, v in c["od"].items() if kk.endswith("|" + t))
        ddor_frac = None
        if dd and t in dd["max_nrad"]:
            ddor_frac = dd["max_nrad"][t] / (dsn["ddor_rad"] * 1e9)
        ddor_counts = ddor_frac if not t.startswith("T4") else None   # T4 delta-DOR: CANCELLED_BY_CALIBRATION
        material_obs = max(obs_r, obs_d, ddor_counts or 0.0) >= 0.1
        decision = "IMPLEMENT" if (material_obs or od >= 0.1) else "DOCUMENT_ONLY"
        if t.startswith("T1_SIZE"):
            decision = "CONVENTION_REAL_DATA"
        out[t] = dict(range_over_sigma_dsn=obs_r, doppler_over_sigma_dsn=obs_d, ddor_over_sigma=ddor_frac,
                      od_max_over_sigma=od, OBSERVABLE_MATERIAL=material_obs, OD_MATERIAL=od >= 0.1,
                      decision=decision)
    if dd:
        out["T3_DDOR_DIFFERENTIAL"] = dict(ddor_over_sigma=dd["max_nrad"]["T3_DIFFERENTIAL"] / 5.0,
                                           note="delta-DOR only; enters T3's delta-DOR column")
    thresholds = {}
    for fam, levels in (("T4_OFFSET", CLOCK_OFFSETS_S), ("T4_FREQ", CLOCK_FREQUENCIES)):
        lv = levels[-1]
        key = f"{fam}_{lv:.0e}"
        o = out[key]
        thresholds[fam] = dict(
            range_obs=lv * 0.1 / o["range_over_sigma_dsn"] if o["range_over_sigma_dsn"] else None,
            doppler_obs=lv * 0.1 / o["doppler_over_sigma_dsn"] if o["doppler_over_sigma_dsn"] else None,
            od=lv * 0.1 / o["od_max_over_sigma"] if o["od_max_over_sigma"] else None,
            note="level at which the term reaches 0.1 sigma (linear in the level), from the largest grid level")
    out["T4_THRESHOLDS"] = thresholds
    out["T4_DDOR"] = "CANCELLED_BY_CALIBRATION (fixed in advance)"
    out["T6_SPACECRAFT_TIME"] = "OUT_OF_SCOPE (fixed in advance)"
    return out


def gates(cells: list[dict], reference_ok: bool) -> dict:
    g = [c["gates"] for c in cells]
    erfa_ls = sorted({x for v in g for x in v["S1_tai_utc_erfa"]})
    spice_ls = sorted({x for v in g for x in v["S1_tai_utc_spice"]})
    s1 = len(erfa_ls) == 1 and len(spice_ls) == 1 and abs(erfa_ls[0] - spice_ls[0]) < 1e-6 \
        and all(len(v["S1_tai_utc_erfa"]) == 1 for v in g)
    return dict(
        S1_leap_seconds=s1, S1_values=dict(erfa=erfa_ls, spice=spice_ls),
        S2_linearity=max(v["S2_range_max_m"] for v in g) <= 1e-3 and max(v["S2_doppler_max_mps"] for v in g) <= 1e-6,
        S2_values=dict(range_max_m=max(v["S2_range_max_m"] for v in g),
                       doppler_max_mps=max(v["S2_doppler_max_mps"] for v in g)),
        S3_spice_chain_sanity=max(v["S3_max_s"] for v in g) <= 1e-4, S3_max_s=max(v["S3_max_s"] for v in g),
        S4_convergence=True, S4_two_cycles=sum(v["S4_two_cycles"] for v in g),
        S4_note="every solve returned (production raises LightTimeConvergenceError otherwise)",
        S5_production_clock_parity=next((c["s5"] for c in cells if c["s5"]), None),
        S6_reference_record=reference_ok and all(v["S6_reference_misses"] == 0 for v in g))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--cells", default="")
    ap.add_argument("--export-epochs", action="store_true")
    ap.add_argument("--reference-sha256", default="")
    a = ap.parse_args()
    cells = ([tuple((s.split(":")[0], int(s.split(":")[1]))) for s in a.cells.split(",")] if a.cells
             else list(P.CELLS))
    if a.export_epochs:
        with ProcessPoolExecutor(min(a.workers, len(cells))) as ex:
            res = dict(ex.map(export_cell, cells))
        P.K._init()
        pos = {s.name: [float(v) for v in np.asarray(s.r_ecef_m, float)] for s in P.K.stations()}
        EPOCHS.write_text(json.dumps(dict(cells=res, station_positions_itrf93_m=pos), indent=0), encoding="utf-8")
        print(json.dumps(dict(epochs=str(EPOCHS), sha256=sha256(EPOCHS),
                              n={c: sum(len(v) for v in r["stations"].values()) for c, r in res.items()})))
        return
    ref_sha = sha256(REFERENCE)
    ref_meta = json.loads(REFERENCE.read_text(encoding="utf-8"))["meta"]
    reference_ok = ref_sha == a.reference_sha256 and ref_meta["input_sha256"] == sha256(EPOCHS)
    with ProcessPoolExecutor(min(a.workers, len(cells))) as ex:
        res = list(ex.map(run_cell, cells))
    out = dict(predeclaration="studies/phase17/configs/phase_c2_time_scale_predeclaration.json",
               reference=dict(file=REFERENCE.name, sha256=ref_sha, expected_sha256=a.reference_sha256,
                              epochs_sha256=sha256(EPOCHS), meta=ref_meta),
               cells=res, gates=gates(res, reference_ok), classification=classify(res))
    path = OUT if not a.cells else OUT.with_name("phase_c2_time_scale_survey_partial.json")
    path.write_text(json.dumps(P.K._clean(out), indent=1), encoding="utf-8")
    print(json.dumps(dict(gates=out["gates"], classification=out["classification"]), indent=1))


if __name__ == "__main__":
    main()
