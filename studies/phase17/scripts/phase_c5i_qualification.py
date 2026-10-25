"""Phase C5-I: qualification of media.py and the media hooks of relativistic_light_time (G1-G4, G6, G7).

Predeclared in studies/phase17/configs/phase_c5i_media_predeclaration.json. G5: phase_c5i_estimator_gates.py;
the full test suite (G1) runs separately. Writes studies/phase17/results/phase_c5i_qualification.json.

    python studies/phase17/scripts/phase_c5i_qualification.py [--workers 10]
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
import phase_c2_time_scale_survey as SV  # noqa: E402
import phase_c2i_qualification as Q  # noqa: E402
import phase_c5_media_survey as S5  # noqa: E402

ROOT = HERE.parents[2]
REF = P.ARTIFACTS / "phase_c5i_reference.npz"
OUT = P.ARTIFACTS / "phase_c5i_qualification.json"
BASE_COMMIT = "f5bceaf"
TC = 60.0
G6_CELLS = (("CANON", 0), ("O2_FULL_SUN_HIGH_BETA", 12), ("O3_HIGH_ALTITUDE_500KM", 0))


def _base_module():
    name = "lunar_od._rlt_base_c5"
    if name in sys.modules:
        return sys.modules[name]
    import lunar_od  # noqa: F401
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASE_COMMIT}:src/lunar_od/relativistic_light_time.py"],
                         check=True, capture_output=True).stdout
    path = Path(tempfile.mkdtemp()) / "_rlt_base_c5.py"
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "lunar_od"
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def survey_model():
    from lunar_od.media import MediaModel
    return MediaModel(troposphere=True, zhd_m=None, zwd_m=S5.ZWD_M, mapping="BLACK_EISNER",
                      vtec_tecu="DEFAULT_CLIMATOLOGY", layer_height_m=S5.H_ION, earth_radius_m=S5.R_E,
                      uplink_hz=S5.F_UP, downlink_hz=S5.F_DOWN)


def g2_g3() -> dict:
    from lunar_od.media import niell_mapping, single_layer_mapping
    r = np.load(REF, allow_pickle=False)
    d = 0.0
    for e, la, h, doy, mh, mw in r["g2"]:
        a, b = niell_mapping(e, la, h, doy)
        d = max(d, abs(a - mh), abs(b - mw))
    g3 = 0.0
    for hh in (350000, 450000):
        m = np.array([single_layer_mapping(e, hh, 6371e3) for e in r["g3_el"]])
        g3 = max(g3, float(np.max(np.abs(m / r[f"g3_{hh}"] - 1.0))))
    return dict(G2_niell=dict(max_abs=float(d), n=int(r["g2"].shape[0]), limit=1e-9, pass_=bool(d <= 1e-9)),
                G3_single_layer=dict(max_rel=g3, limit=1e-12, pass_=bool(g3 <= 1e-12)))


def run_cell(spec) -> dict:
    orbit, k = spec
    from lunar_od import relativistic_light_time as R
    from lunar_od.media import MediaModel, MediaSite
    from lunar_od.radiometrics import _interp_state
    from lunar_od.station_time import Site

    t0 = time.time()
    B = _base_module()
    g = SV.build_cell(orbit, k)
    cell = f"{orbit}:{k}"
    rb, rows, fx, et0 = g["rb"], g["rows"], g["fx"], g["et0"]
    sites = [Site.from_itrf_m(s.r_ecef_m) for s in fx.stations]
    msites = [MediaSite.from_station(s) for s in fx.stations]
    ssites = [S5.Site(s, et0) for s in fx.stations]
    model = R.DSN_C2_LIGHT_TIME_MODEL
    eph, eph_b = R.BarycentricEphemeris(et0), B.BarycentricEphemeris(et0)
    sc, sc_b = R.SpacecraftHistory(rb.t_grid, rb.nom48[:, :6]), B.SpacecraftHistory(rb.t_grid, rb.nom48[:, :6])
    sm, dm = survey_model(), MediaModel()
    g1 = dict(range=0.0, doppler=0.0, ddor=0.0)
    g4 = dict(range=0.0, doppler=0.0, ddor=0.0)
    chg_r, chg_d = np.zeros(rb.obs.shape[0]), np.zeros(rows.n)
    for i, (t3, _, sid, _) in enumerate(rb.obs[:, :4]):
        s_i = int(sid) - 1
        st = g["range_st"][s_i]
        sol = R.solve_two_way(float(t3), st, sc, eph, model, site=sites[s_i])
        base = B.two_way_range_m(B.solve_two_way(float(t3), st, sc_b, eph_b, B.DSN_C2_LIGHT_TIME_MODEL, site=sites[s_i]))
        new = R.two_way_range_m(sol)
        g1["range"] = max(g1["range"], abs(new - base))
        mod = R.two_way_range_m(R.solve_two_way(float(t3), st, sc, eph, model, site=sites[s_i], media=sm,
                                                media_site=msites[s_i])) - new
        _, sv = S5._two_way_media(sol, ssites[s_i])
        g4["range"] = max(g4["range"], abs(mod - 0.5 * sum(sv.values())))
        chg_r[i] = R.two_way_range_m(R.solve_two_way(float(t3), st, sc, eph, model, site=sites[s_i], media=dm,
                                                     media_site=msites[s_i])) - new
    for i in range(rows.n):
        s_i, tm = int(rows.station[i]), float(rows.t_mid[i])
        st = g["dop_st"][s_i]
        v, _, (a, b) = R.counted_doppler(tm, TC, st, sc, eph, model, site=sites[s_i])
        base = B.counted_doppler(tm, TC, st, sc_b, eph_b, B.DSN_C2_LIGHT_TIME_MODEL, site=sites[s_i])[0]
        g1["doppler"] = max(g1["doppler"], abs(v - base))
        mod = R.counted_doppler(tm, TC, st, sc, eph, model, site=sites[s_i], media=sm, media_site=msites[s_i])[0] - v
        _, da = S5._two_way_media(a, ssites[s_i])
        _, db = S5._two_way_media(b, ssites[s_i])
        sv = sum((-1.0 if tm_ == "T3_IONOSPHERE" else 1.0) * (db[tm_] - da[tm_]) for tm_ in S5.TERMS) / (2 * TC)
        g4["doppler"] = max(g4["doppler"], abs(mod - sv))
        chg_d[i] = R.counted_doppler(tm, TC, st, sc, eph, model, site=sites[s_i], media=dm,
                                     media_site=msites[s_i])[0] - v
    if g["ddor"]:
        import phase17_r1od2_core as D2
        drows, (na, nb) = g["ddor"]["rows"], g["ddor"]["names"]
        by = D2.stations_by_name()
        sa, sb = Site.from_itrf_m(by[na].r_ecef_m), Site.from_itrf_m(by[nb].r_ecef_m)
        ma, mb = MediaSite.from_station(by[na]), MediaSite.from_station(by[nb])
        xa, xb = S5.Site(by[na], et0), S5.Site(by[nb], et0)
        a_fn = lambda t: _interp_state(rb.t_grid, drows.hist_a, t)  # noqa: E731
        b_fn = lambda t: _interp_state(rb.t_grid, drows.hist_b, t)  # noqa: E731
        for t_obs in drows.t_used:
            t_obs = float(t_obs)
            st = R.delta_dor_station_time(t_obs, a_fn, b_fn, sc, eph, model, sites=(sa, sb))
            base = B.delta_dor_station_time(t_obs, a_fn, b_fn, sc_b, eph_b, B.DSN_C2_LIGHT_TIME_MODEL, sites=(sa, sb))
            g1["ddor"] = max(g1["ddor"], abs(st.delay_s - base.delay_s))
            mod = R.delta_dor_station_time(t_obs, a_fn, b_fn, sc, eph, model, sites=(sa, sb), media=sm,
                                           media_sites=(ma, mb)).delay_s - st.delay_s
            _, d_a = xa.delays(st.solution.leg_a.t_rx_s, -st.solution.leg_a.rel, S5.F_DOWN)
            _, d_b = xb.delays(st.solution.leg_b.t_rx_s, -st.solution.leg_b.rel, S5.F_DOWN)
            sv = sum(d_b[tm_] - d_a[tm_] for tm_ in S5.TERMS) / SV.C
            g4["ddor"] = max(g4["ddor"], abs(mod - sv))
    return dict(cell=cell, g1=g1, g4=g4, elapsed_s=time.time() - t0,
                change=dict(range_max_abs_m=float(np.max(np.abs(chg_r))), doppler_max_abs_mps=float(np.max(np.abs(chg_d)))),
                g7=Q._od_mapping(g, chg_r, chg_d))


def g6_cell(spec) -> dict:
    orbit, k = spec
    from lunar_od import relativistic_light_time as R
    from lunar_od.media import MediaModel, MediaSite
    from lunar_od.station_time import Site
    g = SV.build_cell(orbit, k)
    rb, fx, et0 = g["rb"], g["fx"], g["et0"]
    dm = MediaModel()
    eph = R.BarycentricEphemeris(et0)
    sc = R.SpacecraftHistory(rb.t_grid, rb.nom48[:, :6])
    worst = 0.0
    h = 1000.0
    for i in np.linspace(0, rb.obs.shape[0] - 1, 20).astype(int):
        t3, sid = float(rb.obs[i, 0]), int(rb.obs[i, 2]) - 1
        sol = R.solve_two_way(t3, g["range_st"][sid], sc, eph, R.DSN_C2_LIGHT_TIME_MODEL,
                              site=Site.from_itrf_m(fx.stations[sid].r_ecef_m))
        ms = MediaSite.from_station(fx.stations[sid])
        for leg, ep, los0, up in ((sol.downlink, sol.downlink.t_rx_s, -sol.downlink.rel, False),
                                  (sol.uplink, sol.uplink.t_tx_s, sol.uplink.rel, True)):
            grad = np.zeros(3)
            for j in range(3):
                dv = np.zeros(3)
                dv[j] = h
                gp = dm.leg_delays(ms, et0 + ep, los0 + dv, uplink=up)[0]
                gm = dm.leg_delays(ms, et0 + ep, los0 - dv, uplink=up)[0]
                grad[j] = (gp - gm) / (2 * h)
            worst = max(worst, float(np.linalg.norm(grad)))
    return dict(cell=f"{orbit}:{k}", max_gradient=worst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run_cell, list(P.CELLS)))
        g6 = list(ex.map(g6_cell, list(G6_CELLS)))
    gates = g2_g3()
    gates["G1_bit_identity"] = dict(passed=all(v == 0.0 for c in res for v in c["g1"].values()),
                                    max_abs={kk: max(c["g1"][kk] for c in res) for kk in ("range", "doppler", "ddor")})
    g4 = dict(range_m=max(c["g4"]["range"] for c in res), doppler_mps=max(c["g4"]["doppler"] for c in res),
              ddor_s=max(c["g4"]["ddor"] for c in res))
    g4["pass_"] = bool(g4["range_m"] <= 1e-6 and g4["doppler_mps"] <= 3e-9 and g4["ddor_s"] <= 1e-15)
    gates["G4_survey_parity"] = g4
    w = max(x["max_gradient"] for x in g6)
    gates["G6_partial_neglect"] = dict(max_gradient=w, limit=1e-6, pass_=bool(w <= 1e-6), cells=g6)
    keys = list(res[0]["g7"])
    gates["G7_informational"] = dict(
        od={kk: max(max(c["g7"][kk]["max_state_over_sigma"], c["g7"][kk]["k_over_sigma"]) for c in res) for kk in keys},
        change_range_m=max(c["change"]["range_max_abs_m"] for c in res),
        change_doppler_mps=max(c["change"]["doppler_max_abs_mps"] for c in res))
    out = dict(predeclaration="studies/phase17/configs/phase_c5i_media_predeclaration.json", base_commit=BASE_COMMIT,
               gates=gates, cells=res, elapsed_s=time.time() - t0)
    OUT.write_text(json.dumps(P.K._clean(out), indent=1), encoding="utf-8")
    print(json.dumps(P.K._clean({k: v for k, v in gates.items() if k != "G6_partial_neglect"} |
                                {"G6": {"max_gradient": w, "pass_": w <= 1e-6}}), indent=1))


if __name__ == "__main__":
    main()
