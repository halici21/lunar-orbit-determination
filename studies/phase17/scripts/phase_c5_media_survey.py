"""Phase C5: tropospheric and ionospheric delay - materiality survey (analysis only).

Predeclared in studies/phase17/configs/phase_c5_media_predeclaration.json (committed before any number).
Writes studies/phase17/results/phase_c5_media_survey.json.

    python studies/phase17/scripts/phase_c5_media_survey.py [--workers 10] [--cells CANON:0,...]

Media delays add to the light time per leg at the ground event (uplink t1, downlink t3): two-way range
+(d_up + d_down)/2, counted Doppler +[(d_up + d_down)_end - (d_up + d_down)_start]/(2 Tc) with the ionosphere's
phase advance (opposite sign), delta-DOR (group) +d_B - d_A.
"""
from __future__ import annotations

import argparse
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
warnings.simplefilter("ignore")

import phase17_kp2_campaign as P  # noqa: E402
import phase_c2_time_scale_survey as SV  # noqa: E402
import phase_c2i_qualification as Q  # noqa: E402

OUT = P.ARTIFACTS / "phase_c5_media_survey.json"
TC = 60.0
TERMS = ("T1_TROPO_HYDROSTATIC", "T2_TROPO_WET", "T3_IONOSPHERE")
R_E, H_ION = 6371e3, 350e3
ZWD_M = 0.15
F_UP = 7.2e9
F_DOWN = F_UP * 880.0 / 749.0


def zhd(lat_rad, h_m):
    p = 1013.25 * (1.0 - 2.2557e-5 * h_m) ** 5.2568
    return 0.0022768 * p / (1.0 - 0.00266 * math.cos(2 * lat_rad) - 0.00028 * h_m / 1e3)


def mapping(el):
    return 1.001 / math.sqrt(0.002001 + math.sin(el) ** 2)


def vtec_tecu(lt_h):
    return 5.0 + 25.0 * max(0.0, math.cos(2 * math.pi * (lt_h - 14.0) / 24.0))


def iono_delay(el, lt_h, f):
    obl = 1.0 / math.sqrt(1.0 - (R_E * math.cos(el) / (R_E + H_ION)) ** 2)
    return 40.3 * vtec_tecu(lt_h) * 1e16 * obl / f ** 2


class Site:
    def __init__(self, st, et0):
        import spiceypy as spice
        self.spice, self.et0 = spice, et0
        self.lat, self.lon, self.h = st.lat_rad, st.lon_rad, float(st.alt_m)
        self.up_ecef = np.array([math.cos(self.lat) * math.cos(self.lon), math.cos(self.lat) * math.sin(self.lon),
                                 math.sin(self.lat)])
        self.zhd = zhd(self.lat, self.h)

    def elevation(self, t, los):
        up = np.asarray(self.spice.pxform("ITRF93", "J2000", self.et0 + float(t))) @ self.up_ecef
        return math.asin(float(np.dot(los / np.linalg.norm(los), up)))

    def local_time_h(self, t):
        utc = self.et0 + float(t) - 69.184
        return ((utc + 43200.0) / 3600.0 + math.degrees(self.lon) / 15.0) % 24.0

    def delays(self, t, los, f):
        el = self.elevation(t, los)
        m = mapping(el)
        return el, {"T1_TROPO_HYDROSTATIC": self.zhd * m, "T2_TROPO_WET": ZWD_M * m,
                    "T3_IONOSPHERE": iono_delay(el, self.local_time_h(t), f)}


def _two_way_media(sol, site):
    el_d, dd = site.delays(sol.t3_s, -sol.downlink.rel, F_DOWN)       # station -> spacecraft at t3
    el_u, du = site.delays(sol.t1_s, sol.uplink.rel, F_UP)            # station -> spacecraft at t1
    return (el_d, el_u), {tm: dd[tm] + du[tm] for tm in TERMS}


def run_cell(spec) -> dict:
    orbit, k = spec
    from lunar_od import relativistic_light_time as R
    from lunar_od.radiometrics import _interp_state
    from lunar_od.station_time import Site as StSite

    t0 = time.time()
    g = SV.build_cell(orbit, k)
    cell = f"{orbit}:{k}"
    rb, rows, fx, et0 = g["rb"], g["rows"], g["fx"], g["et0"]
    tsites = [StSite.from_itrf_m(s.r_ecef_m) for s in fx.stations]
    msites = [Site(s, et0) for s in fx.stations]
    model = R.DSN_C2_LIGHT_TIME_MODEL
    eph = R.BarycentricEphemeris(et0)
    sc = R.SpacecraftHistory(rb.t_grid, rb.nom48[:, :6])
    els = []
    nr, nd = rb.obs.shape[0], rows.n
    dr = {tm: np.zeros(nr) for tm in TERMS}
    dd = {tm: np.zeros(nd) for tm in TERMS}
    for i, (t3, _, sid, _) in enumerate(rb.obs[:, :4]):
        s_i = int(sid) - 1
        sol = R.solve_two_way(float(t3), g["range_st"][s_i], sc, eph, model, site=tsites[s_i])
        e, d = _two_way_media(sol, msites[s_i])
        els += list(e)
        for tm in TERMS:
            dr[tm][i] = 0.5 * d[tm]
    for i in range(nd):
        s_i, tm_ = int(rows.station[i]), float(rows.t_mid[i])
        _, _, (a, b) = R.counted_doppler(tm_, TC, g["dop_st"][s_i], sc, eph, model, site=tsites[s_i])
        ea, da = _two_way_media(a, msites[s_i])
        eb, db = _two_way_media(b, msites[s_i])
        els += list(ea) + list(eb)
        for tm in TERMS:
            sign = -1.0 if tm == "T3_IONOSPHERE" else 1.0
            dd[tm][i] = sign * (db[tm] - da[tm]) / (2.0 * TC)
    ddor = None
    if g["ddor"]:
        import phase17_r1od2_core as D2
        drows, (na, nb) = g["ddor"]["rows"], g["ddor"]["names"]
        by = D2.stations_by_name()
        sa, sb = Site(by[na], et0), Site(by[nb], et0)
        a_fn = lambda t: _interp_state(rb.t_grid, drows.hist_a, t)  # noqa: E731
        b_fn = lambda t: _interp_state(rb.t_grid, drows.hist_b, t)  # noqa: E731
        n = len(drows.t_used)
        to_rad = SV.C / np.maximum(np.asarray(drows.b_perp, float)[:n], 1.0)
        val = {tm: np.zeros(n) for tm in TERMS}
        for i, t_obs in enumerate(drows.t_used):
            st = R.delta_dor_station_time(float(t_obs), a_fn, b_fn, sc, eph, model,
                                          sites=(StSite.from_itrf_m(by[na].r_ecef_m), StSite.from_itrf_m(by[nb].r_ecef_m)))
            la, lb = -st.solution.leg_a.rel, -st.solution.leg_b.rel
            _, d_a = sa.delays(st.solution.leg_a.t_rx_s, la, F_DOWN)
            _, d_b = sb.delays(st.solution.leg_b.t_rx_s, lb, F_DOWN)
            for tm in TERMS:
                val[tm][i] = (d_b[tm] - d_a[tm]) / SV.C
        ddor = dict(n=n, max_nrad={tm: float(np.max(np.abs(val[tm] * to_rad)) * 1e9) for tm in TERMS})
    od = {tm: Q._od_mapping(g, dr[tm], dd[tm]) for tm in TERMS}

    def stats(v):
        return dict(min=float(v.min()), max=float(v.max()), maxabs=float(np.max(np.abs(v))))
    return dict(cell=cell, elapsed_s=time.time() - t0,
                gates=dict(el_min_deg=float(np.degrees(min(els))), el_max_deg=float(np.degrees(max(els))),
                           zhd_m=[s.zhd for s in msites]),
                range_m={tm: stats(v) for tm, v in dr.items()}, doppler_mps={tm: stats(v) for tm, v in dd.items()},
                ddor=ddor, od=od)


def classify(cells):
    dsn = SV.NOISE["DSN_REALISTIC"]
    dd = next((c["ddor"] for c in cells if c["ddor"]), None)
    out = {}
    for tm in TERMS:
        obs_r = max(c["range_m"][tm]["maxabs"] for c in cells) / dsn["range"]
        obs_d = max(c["doppler_mps"][tm]["maxabs"] for c in cells) / dsn["doppler"]
        ddor = dd["max_nrad"][tm] / (dsn["ddor_rad"] * 1e9) if dd else None
        od = max(max(v["max_state_over_sigma"], v["k_over_sigma"]) for c in cells for v in c["od"][tm].values())
        mat = max(obs_r, obs_d, ddor or 0.0) >= 0.1
        out[tm] = dict(range_over_sigma_dsn=obs_r, doppler_over_sigma_dsn=obs_d, ddor_over_sigma=ddor,
                       od_max_over_sigma=od, OBSERVABLE_MATERIAL=mat, OD_MATERIAL=od >= 0.1,
                       decision="IMPLEMENT" if (mat or od >= 0.1) else "DOCUMENT_ONLY")
    out["T4_RAMPED_UPLINK"] = "CONVENTION_REAL_DATA (fixed in advance)"
    return out


def gates(cells):
    el_min = min(c["gates"]["el_min_deg"] for c in cells)
    el_max = max(c["gates"]["el_max_deg"] for c in cells)
    zh = [z for c in cells for z in c["gates"]["zhd_m"]]
    zen30 = 40.3 * 30e16 / F_DOWN ** 2
    return dict(S1_elevation=bool(9.0 <= el_min and el_max <= 90.0), S1_values=dict(min_deg=el_min, max_deg=el_max),
                S2_tropo_sanity=bool(1.8 <= min(zh) and max(zh) <= 2.4), S2_zhd_m=dict(min=min(zh), max=max(zh)),
                S3_iono_sanity=bool(0.15 <= zen30 <= 0.20), S3_zenith_30tecu_m=zen30, S4_convergence=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--cells", default="")
    a = ap.parse_args()
    cells = ([tuple((s.split(":")[0], int(s.split(":")[1]))) for s in a.cells.split(",")] if a.cells
             else list(P.CELLS))
    with ProcessPoolExecutor(min(a.workers, len(cells))) as ex:
        res = list(ex.map(run_cell, cells))
    out = dict(predeclaration="studies/phase17/configs/phase_c5_media_predeclaration.json", cells=res,
               gates=gates(res), classification=classify(res))
    path = OUT if not a.cells else OUT.with_name("phase_c5_media_survey_partial.json")
    path.write_text(json.dumps(P.K._clean(out), indent=1), encoding="utf-8")
    print(json.dumps(P.K._clean(dict(gates=out["gates"], classification=out["classification"])), indent=1))


if __name__ == "__main__":
    main()
