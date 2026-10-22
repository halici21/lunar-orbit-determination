"""Phase C2-I: POST-HOC diagnostics for G4 (counted Doppler) and G5 (clock-offset partials). Informational; they
do not change any gate verdict. Written after phase_c2i_qualification.py reported both FAIL as computed.

  D3  counted-Doppler station-time change, term by term, against the exact interval difference
      D_site(t3) - D_site(t1) of station_time: (a) the module (round trip x midpoint rate, A1) and (b) the C2
      survey's form (round trip x rate at the receive epoch); plus the evaluation floor of a counted-Doppler
      value, c/(2 Tc) x ulp(round trip), which bounds any difference of two module Doppler values.
  D4  clock-offset partials vs a central difference in the offset on the G5 rows, per row: analytic, best
      |fd - analytic| over the offset steps, and whether the row is the arc's last range row.

    python studies/phase17/scripts/phase_c2i_g4_g5_diagnostics.py
"""
from __future__ import annotations

import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")

import phase17_kp2_campaign as P  # noqa: E402
import phase_c2i_qualification as Q  # noqa: E402
import phase_c2_time_scale_survey as SV  # noqa: E402

OUT = P.ARTIFACTS / "phase_c2i_g4_g5_diagnostics.json"
TC = 60.0
D3_CELLS = (("CANON", 0), ("O5_FAVOURABLE_GEO_MAX_FPERP", 0))
STEPS = (1e-4, 1e-3, 1e-2, 1e-1)


def d3(spec) -> dict:
    from lunar_od import relativistic_light_time as R
    from lunar_od import station_time as ST
    g = SV.build_cell(*spec)
    rb, rows, fx, et0 = g["rb"], g["rows"], g["fx"], g["et0"]
    sites = Q._sites(fx)
    eph = R.BarycentricEphemeris(et0)
    sc = R.SpacecraftHistory(rb.t_grid, rb.nom48[:, :6])
    k = R.C_LIGHT_MPS / (2 * TC)
    w = dict(module_vs_exact=0.0, receive_epoch_form_vs_exact=0.0, value_difference_vs_exact=0.0)
    floor = 0.0
    for i in range(rows.n):
        s, tm = int(rows.station[i]), float(rows.t_mid[i])
        st, site = g["dop_st"][s], sites[s]
        c2 = R.counted_doppler(tm, TC, st, sc, eph, R.DSN_C2_LIGHT_TIME_MODEL, site=site)
        ds = R.counted_doppler(tm, TC, st, sc, eph, R.DSN_LIGHT_TIME_MODEL)
        mod, exa, rec = [], [], []
        for sol_c2, sol_ds in zip(c2[2], ds[2]):
            ex = float(ST.tdb_minus_tt(et0, sol_c2.t3_s, site) - ST.tdb_minus_tt(et0, sol_c2.t1_s, site))
            spice = sol_ds.round_trip_s * sol_ds.tdb_minus_tt_rate
            mod.append(sol_c2.round_trip_s * sol_c2.tdb_minus_tt_rate - spice)
            exa.append(ex - spice)
            rec.append(sol_c2.round_trip_s * float(ST.tdb_minus_tt_rate(et0, sol_c2.t3_s, site)) - spice)
            floor = max(floor, k * float(np.spacing(sol_c2.round_trip_s)))
        e = -k * (exa[1] - exa[0])
        w["module_vs_exact"] = max(w["module_vs_exact"], abs(-k * (mod[1] - mod[0]) - e))
        w["receive_epoch_form_vs_exact"] = max(w["receive_epoch_form_vs_exact"], abs(-k * (rec[1] - rec[0]) - e))
        w["value_difference_vs_exact"] = max(w["value_difference_vs_exact"], abs((c2[0] - ds[0]) - e))
    return dict(cell=f"{spec[0]}:{spec[1]}", n=int(rows.n), doppler_mps=w,
                evaluation_floor_per_value_mps=floor)


def d4(spec) -> dict:
    from lunar_od import relativistic_light_time as R
    g = SV.build_cell(*spec)
    rb, rows, fx = g["rb"], g["rows"], g["fx"]
    sites = Q._sites(fx)
    m = R.DSN_C2_LIGHT_TIME_MODEL
    eph = R.BarycentricEphemeris(g["et0"])
    sc = R.SpacecraftHistory(rb.t_grid, rb.nom48[:, :6])
    out = dict(range=[], doppler=[])
    n_r = rb.obs.shape[0]
    for i in np.linspace(0, n_r - 1, Q.FD_ROWS).astype(int):
        t3, sid = float(rb.obs[i, 0]), int(rb.obs[i, 2]) - 1

        def f(b):
            return R.two_way_range_m(R.solve_two_way(t3, g["range_st"][sid], sc, eph, m, site=sites[sid],
                                                     clock_offset_s=b))
        an = R.two_way_range_clock_partial(R.solve_two_way(t3, g["range_st"][sid], sc, eph, m, site=sites[sid]))
        best = min(abs((f(h) - f(-h)) / (2 * h) - an) for h in STEPS)
        out["range"].append(dict(row=int(i), last_row=bool(i == n_r - 1), analytic=an, best_abs_diff=best))
    for i in np.linspace(0, rows.n - 1, Q.FD_ROWS).astype(int):
        s, tm = int(rows.station[i]), float(rows.t_mid[i])

        def f(b):
            return R.counted_doppler(tm, TC, g["dop_st"][s], sc, eph, m, site=sites[s], clock_offset_s=b)[0]
        an = R.counted_doppler(tm, TC, g["dop_st"][s], sc, eph, m, site=sites[s], with_clock_partial=True)[3]
        best = min(abs((f(h) - f(-h)) / (2 * h) - an) for h in STEPS)
        out["doppler"].append(dict(row=int(i), last_row=bool(i == rows.n - 1), analytic=an, best_abs_diff=best))
    summ = {}
    for kk, v in out.items():
        rel = [x["best_abs_diff"] / max(abs(x["analytic"]), 1e-300) for x in v]
        rel_not_last = [r for r, x in zip(rel, v) if not x["last_row"]]
        row_norm = max(x["best_abs_diff"] for x in v) / max(abs(x["analytic"]) for x in v)
        summ[kk] = dict(max_rel=max(rel), max_rel_excluding_last_row=max(rel_not_last),
                        worst_row=v[int(np.argmax(rel))], max_abs_over_max_partial=row_norm)
    return dict(cell=f"{spec[0]}:{spec[1]}", rows=out, summary=summ)


def main():
    with ProcessPoolExecutor(5) as ex:
        r3 = list(ex.map(d3, D3_CELLS))
        r4 = list(ex.map(d4, list(Q.FD_CELLS)))
    out = dict(post_hoc=True, D3=r3, D4=r4)
    OUT.write_text(json.dumps(P.K._clean(out), indent=1), encoding="utf-8")
    print(json.dumps(P.K._clean(dict(D3=r3, D4=[dict(cell=x["cell"], summary=x["summary"]) for x in r4])), indent=1))


if __name__ == "__main__":
    main()
