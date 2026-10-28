"""Phase F2n: why does PAD 103's 2009-263 10:50-11:20 UTC Doppler pass scatter by 69 mm/s?

Predeclaration: studies/phase17/configs/phase_f2n_bad_doppler_pass_diagnostic_predeclaration.json.  Residuals of
every 5 s counted-Doppler interval against the LOLA ephemeris (predict()'s measurement code, as in F2l), with the
raw UTDF fields of the records that bound each interval, for the bad pass and two control passes.
"""
from __future__ import annotations

import collections
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_f2_lro_od as M  # noqa: E402
import phase_f2k_tight_tolerance as K  # noqa: E402

OUT = HERE.parent / "results" / "phase_f2n_bad_pass_diagnostic.json"
FIG = HERE.parent / "results" / "phase_f2n_bad_pass_residuals.png"
UTC = timezone.utc
PASSES = {
 "bad_263_1050": ("trk/lro_no_01/2009263", datetime(2009, 9, 20, 10, 45, tzinfo=UTC), datetime(2009, 9, 20, 11, 25, tzinfo=UTC)),
 "control_263_0704": ("trk/lro_no_01/2009263", datetime(2009, 9, 20, 7, 0, tzinfo=UTC), datetime(2009, 9, 20, 8, 0, tzinfo=UTC)),
 "control_262_0825": ("trk/lro_no_01/2009262", datetime(2009, 9, 19, 8, 20, tzinfo=UTC), datetime(2009, 9, 19, 9, 20, tzinfo=UTC)),
}
PAD = 103
BIG = 20.0  # mm/s


def series(pb, p, day, t_lo, t_hi):
    import spiceypy as spice
    from lunar_od import station_time as ST
    from lunar_od.ephemeris import sample_moon_centered_ephemeris
    from lunar_od.media import MediaModel
    import phase_f1c_lro_doppler as D
    R, s = pb.R, pb.setup
    drows = [(q, r) for q, r in D.read_doppler(pb.data, day) if q == PAD and t_lo <= r.utc <= t_hi]
    drows.sort(key=lambda x: x[1].utc)
    td = np.array([ST.utc_to_tdb_seconds([r.utc.strftime("%Y-%m-%dT%H:%M:%S.%f")], s.et0, s.sites[q])[0]
                   for q, r in drows])
    grid = np.arange(td.min() - 600.0, td.max() + 600.0, M.GRID_S)
    pb.eph = sample_moon_centered_ephemeris(s.et0, np.arange(grid[0] - 600.0, grid[-1] + 600.0, 60.0))
    hist = np.array([spice.spkezr(M.F.LRO, s.et0 + t, "J2000", "NONE", "MOON")[0] for t in grid]) * 1e3
    sc = R.SpacecraftHistory(grid, hist)
    rec = []
    for i, j in D.intervals(drows, td):
        q, ra = drows[i]
        rb = drows[j][1]
        dt = (rb.utc - ra.utc).total_seconds()
        dn = rb.doppler_count_raw - ra.doppler_count_raw
        a = dn / dt
        ft = ra.transmit_frequency_hz
        k = 2.0 * ft * D.TURNAROUND * D.M_MULT / M.F.C
        obs = -(a - D.BIAS_HZ) / k
        if abs(obs) > M.MAX_DOPPLER_MPS:
            continue
        val, _, _ = R.counted_doppler(0.5 * (td[i] + td[j]), td[j] - td[i], s.tide[q], sc, s.eph,
                                      R.DSN_C2_LIGHT_TIME_MODEL, site=s.sites[q],
                                      media=MediaModel(uplink_hz=ft, downlink_hz=ft * D.TURNAROUND),
                                      media_site=s.media_sites[q])
        r = obs - val - p[10]
        rec.append(dict(utc=ra.utc.isoformat(), t_rel_s=float(td[i] - td[0]), dt_s=dt, dn=int(dn), rate_hz=a,
                        rate_minus_expected_hz=float(r * k), resid_mmps=float(r * 1e3),
                        us_a=int(ra.microseconds), us_b=int(rb.microseconds), ft_hz=ft))
    return drows, rec


def summarize(drows, rec):
    r = np.array([x["resid_mmps"] for x in rec])
    t = np.array([x["t_rel_s"] for x in rec])
    ac1 = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if r.size > 2 else float("nan")
    tt = np.array([(x.utc - drows[0][1].utc).total_seconds() for _, x in drows])
    gaps = [(float(tt[i]), float(tt[i + 1] - tt[i])) for i in range(len(tt) - 1) if tt[i + 1] - tt[i] > 60.0]
    big = np.abs(r) > BIG
    seg = {}
    if gaps:
        cut = gaps[np.argmax([g[1] for g in gaps])][0]
        for name, m in (("before_gap", t <= cut), ("after_gap", t > cut)):
            seg[name] = dict(n=int(m.sum()), rms=float(np.sqrt(np.mean(r[m] ** 2))) if m.any() else None,
                             n_big=int((big & m).sum()))
    return dict(n_records=len(drows), n_intervals=int(r.size), rms_mmps=float(np.sqrt(np.mean(r ** 2))),
                mean_mmps=float(r.mean()), std_mmps=float(r.std()), lag1_autocorr=ac1,
                frac_big=float(big.mean()), n_big=int(big.sum()),
                record_spacing_s=dict(collections.Counter(np.round(np.diff(tt), 3)).most_common(6)),
                microseconds_values=dict(collections.Counter(int(x.microseconds) for _, x in drows).most_common(6)),
                transmit_frequencies_hz=sorted({x.transmit_frequency_hz for _, x in drows}),
                gaps_over_60s=gaps, segments=seg)


def main():
    import warnings
    warnings.simplefilter("ignore")
    M.set_tolerance(*M.F2K_TOL)
    p = K.point_vector(json.loads(M.OUT_F2K.read_text(encoding="utf-8"))["runs"]["S1"])
    pb = M.Problem(300, doppler=True, days=M.F2J_FIT_DAYS, n_emp=len(M.EMP_NAMES))
    out = dict(predeclaration="studies/phase17/configs/phase_f2n_bad_doppler_pass_diagnostic_predeclaration.json",
               passes={})
    recs = {}
    for name, (day, lo, hi) in PASSES.items():
        drows, rec = series(pb, p, day, lo, hi)
        recs[name] = rec
        out["passes"][name] = dict(summary=summarize(drows, rec), intervals=rec)
    OUT.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(recs), 1, figsize=(9, 7), sharey=True)
    for a, (name, rec) in zip(ax, recs.items()):
        a.plot([x["t_rel_s"] / 60.0 for x in rec], [x["resid_mmps"] for x in rec], ".", ms=3)
        a.set_ylim(-25.0, 15.0)
        for x in rec:
            if x["resid_mmps"] > 15.0:
                a.annotate(f"{x['resid_mmps']:.0f} mm/s ({x['utc'][11:19]} UTC)", (x["t_rel_s"] / 60.0, 15.0),
                           xytext=(4, -12), textcoords="offset points", fontsize=8, arrowprops=dict(arrowstyle="->"))
        a.set_title(f"PAD {PAD} {name}", fontsize=9)
        a.set_ylabel("Doppler residual vs LOLA (mm/s)", fontsize=8)
    ax[-1].set_xlabel("minutes from first record")
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    print(json.dumps({k: v["summary"] for k, v in out["passes"].items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
