"""Phase F2l: the LOLA reference orbit scored against the 2009-260..263 range and Doppler, no fit.

Predeclaration: studies/phase17/configs/phase_f2l_lola_orbit_scoring_predeclaration.json.  The measurement code is
predict() of phase_f2_lro_od.py with the spacecraft history passed in, so the reference ephemeris and our own orbit
go through identical light-time, media and Doppler code.  One process per case.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_f2_lro_od as M  # noqa: E402
M.use_historical_numerics()  # merge of main (G4e-R/TRF): keep the numerics of this phase's records
import phase_f2k_tight_tolerance as K  # noqa: E402

OUT = HERE.parent / "results" / "phase_f2l_lola_scoring.json"
DAYS = M.F2J_FIT_DAYS + M.F2J_PREDICT_DAYS
CASES = [("LOLA", d) for d in DAYS] + [("F2k", M.F2J_PREDICT_DAYS[0])]


def _stats(r):
    r = np.asarray(r, float)
    return dict(n=int(r.size), rms=float(np.sqrt(np.mean(r ** 2))), mean=float(r.mean()),
                rms_about_mean=float(r.std()))


def _groups(pads, t):
    """Indices per pass: same PAD, gaps up to F1's PASS_GAP_S."""
    out = []
    for pad in sorted(set(pads)):
        idx = np.array([i for i, q in enumerate(pads) if q == pad], dtype=int)
        idx = idx[np.argsort(t[idx])]
        out.extend((pad, g) for g in np.split(idx, np.flatnonzero(np.diff(t[idx]) > M.F.PASS_GAP_S) + 1))
    return out


def score(pb, p, day: str, orbit: str, intervals: bool = False, freq: bool = False) -> dict:
    """predict() for one day, returning the residuals; the history is the LOLA ephemeris or our propagation."""
    import spiceypy as spice
    from lunar_od import station_time as ST
    from lunar_od.ephemeris import sample_moon_centered_ephemeris
    from lunar_od.media import MediaModel
    import phase_f1c_lro_doppler as D
    R, s = pb.R, pb.setup
    rows = M.F.read_day(pb.data, day)[0]
    t3 = np.array([ST.utc_to_tdb_seconds([r.utc.strftime("%Y-%m-%dT%H:%M:%S.%f")], s.et0, s.sites[q])[0]
                   for q, r in rows])
    drows = D.read_doppler(pb.data, day)
    drows.sort(key=lambda x: (x[0], x[1].utc))
    td = np.array([ST.utc_to_tdb_seconds([r.utc.strftime("%Y-%m-%dT%H:%M:%S.%f")], s.et0, s.sites[q])[0]
                   for q, r in drows])
    grid = np.arange(pb.t_epoch, max(t3.max(), td.max()) + 10.0 + M.GRID_S, M.GRID_S)
    pb.eph = sample_moon_centered_ephemeris(s.et0, np.arange(pb.t_epoch - 600.0, grid[-1] + 600.0, 60.0))
    if orbit == "LOLA":
        hist = np.array([spice.spkezr(M.F.LRO, s.et0 + t, "J2000", "NONE", "MOON")[0] for t in grid]) * 1e3
    else:
        hist = pb.propagate_p(p, t_eval=grid)
    sc = R.SpacecraftHistory(grid, hist)
    res_r = []
    for (q, r), t in zip(rows, t3):
        sol = R.solve_two_way(float(t), s.tide[q], sc, s.eph, R.DSN_C2_LIGHT_TIME_MODEL, site=s.sites[q],
                              media=pb.media, media_site=s.media_sites[q])
        res_r.append(M.F.C * r.rtlt_s / 2.0 - R.two_way_range_m(sol) - p[7 + M.PADS.index(q)])
    res_r = np.array(res_r)
    res_d, td_mid, d_pad, d_utc, d_ft = [], [], [], [], []
    for i, j in D.intervals(drows, td):
        q, ra = drows[i]
        rb = drows[j][1]
        a = (rb.doppler_count_raw - ra.doppler_count_raw) / (rb.utc - ra.utc).total_seconds()
        ft = ra.transmit_frequency_hz
        obs = -M.F.C * (a - D.BIAS_HZ) / (2.0 * ft * D.TURNAROUND * D.M_MULT)
        if abs(obs) > M.MAX_DOPPLER_MPS:             # F2e A1
            continue
        val, _, _ = R.counted_doppler(0.5 * (td[i] + td[j]), td[j] - td[i], s.tide[q], sc, s.eph,
                                      R.DSN_C2_LIGHT_TIME_MODEL, site=s.sites[q],
                                      media=MediaModel(uplink_hz=ft, downlink_hz=ft * D.TURNAROUND),
                                      media_site=s.media_sites[q])
        res_d.append(obs - val - p[10])
        td_mid.append(0.5 * (td[i] + td[j]))
        d_pad.append(q)
        d_utc.append(ra.utc.isoformat())
        d_ft.append(float(ft))
    res_d, td_mid = np.array(res_d), np.array(td_mid)
    r_pass = [dict(pad=int(rows[g[0]][0]), t_start_h=float(t3[g].min() / 3600.0), **_stats(res_r[g]))
              for g in M.F.passes(rows, t3)]
    d_pass = [dict(pad=int(q), t_start_h=float(td_mid[g].min() / 3600.0), **_stats(res_d[g] * 1e3))
              for q, g in _groups(d_pad, td_mid)]
    out = dict(range_m=_stats(res_r), doppler_mmps=_stats(res_d * 1e3), range_passes=r_pass, doppler_passes=d_pass)
    if intervals:   # F2o: every Doppler interval, in pass order
        out["doppler_intervals"] = [dict(pad=int(q), utc=d_utc[k], t_mid_s=float(td_mid[k]),
                                         resid_mmps=float(res_d[k] * 1e3), pass_index=n)
                                    for n, (q, g) in enumerate(_groups(d_pad, td_mid)) for k in g]
        if freq:    # F2q: the transmit frequency the model used for each interval
            for x, k in zip(out["doppler_intervals"], [k for _, g in _groups(d_pad, td_mid) for k in g]):
                x["transmit_hz"] = d_ft[k]
    return out


def case(key, intervals: bool = False, run=None, freq: bool = False):
    import warnings
    warnings.simplefilter("ignore")
    orbit, day = key
    t0 = time.time()
    M.set_tolerance(*M.F2K_TOL)
    p = K.point_vector(run or json.loads(M.OUT_F2K.read_text(encoding="utf-8"))["runs"]["S1"])
    pb = M.Problem(300, doppler=True, days=M.F2J_FIT_DAYS, n_emp=len(M.EMP_NAMES))
    out = score(pb, p, day, orbit, intervals, freq)
    out["elapsed_s"] = time.time() - t0
    return key, out


def main():
    with ProcessPoolExecutor(len(CASES)) as ex:
        res = dict(ex.map(case, CASES))
    stored = json.loads(M.OUT_F2K.read_text(encoding="utf-8"))["runs"]["S1"]["prediction"]
    own = res[("F2k", DAYS[-1])]
    repro = max(abs(own["range_m"]["rms"] / stored["range_rms_m"] - 1),
                abs(own["doppler_mmps"]["rms"] / stored["doppler_rms_mmps"] - 1))
    lola = {d: res[("LOLA", d)] for d in DAYS}
    d263 = lola[DAYS[-1]]["doppler_mmps"]["rms"]
    control = [lola[d]["doppler_mmps"]["rms"] for d in DAYS[:-1]]
    if repro > 1e-6:
        verdict = "NOT_CONCLUDED (reproduction check failed)"
    elif d263 >= 10.0 and max(control) <= 5.0:
        verdict = "H_DATA"
    elif d263 <= 5.0:
        verdict = "H_ORBIT"
    else:
        verdict = "MIXED"
    out = dict(predeclaration="studies/phase17/configs/phase_f2l_lola_orbit_scoring_predeclaration.json",
               reproduction_relative=repro, reproduction_PASS=repro <= 1e-6, verdict=verdict,
               lola_doppler_rms_mmps={d.rsplit("/", 1)[1]: lola[d]["doppler_mmps"]["rms"] for d in DAYS},
               lola_range_rms_m={d.rsplit("/", 1)[1]: lola[d]["range_m"]["rms"] for d in DAYS},
               lola={d.rsplit("/", 1)[1]: v for d, v in lola.items()}, f2k_263=own)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("reproduction_relative", "verdict", "lola_doppler_rms_mmps",
                                          "lola_range_rms_m")}, indent=1))


if __name__ == "__main__":
    main()
