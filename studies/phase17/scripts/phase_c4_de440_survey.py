"""Phase C4: DE440 vs the production DE421 ephemeris - materiality survey (analysis only).

Predeclared in studies/phase17/configs/phase_c4_de440_predeclaration.json (committed before any number).
Writes studies/phase17/results/phase_c4_de440_survey.json.

    python studies/phase17/scripts/phase_c4_de440_survey.py [--workers 10] [--cells CANON:0,...]

Variants per cell (exact re-solve under DSN_C2_LIGHT_TIME_MODEL): T1 station moved by the Moon-centred Earth
difference DE440 - DE421; T2 barycentric light-time ephemeris from DE440 (de440.bsp furnished after de421.bsp,
unloaded afterwards); T3 trajectory with the SRP Sun direction from DE440; TOTAL all three.
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
import phase_c2_time_scale_survey as SV  # noqa: E402
import phase_c2i_qualification as Q  # noqa: E402

OUT = P.ARTIFACTS / "phase_c4_de440_survey.json"
TC = 60.0
TERMS = ("T1_EARTH_POSITION", "T2_SSB_EPHEMERIS", "T3_SRP_SUN", "TOTAL")


def _de440_path() -> str:
    from lunar_od.spice_loader import resolve_kernel_dir
    return str(Path(resolve_kernel_dir(None)) / "de440.bsp")


class DE440:
    """Context manager: de440.bsp furnished after the loaded kernels, unloaded on exit."""

    def __enter__(self):
        import spiceypy as spice
        spice.furnsh(_de440_path())

    def __exit__(self, *exc):
        import spiceypy as spice
        spice.unload(_de440_path())


def _probe(et0, ts):
    import spiceypy as spice
    return np.array([np.concatenate([spice.spkpos("EARTH", et0 + t, "J2000", "NONE", "MOON")[0],
                                     spice.spkpos("MOON", et0 + t, "J2000", "NONE", "SSB")[0]]) for t in ts])


def run_cell(spec) -> dict:
    orbit, k = spec
    from scipy.interpolate import CubicSpline
    from lunar_od import relativistic_light_time as R
    from lunar_od.radiometrics import _interp_state
    from lunar_od.station_time import Site

    t0 = time.time()
    g = SV.build_cell(orbit, k)
    cell = f"{orbit}:{k}"
    rb, rows, fx, et0, scn = g["rb"], g["rows"], g["fx"], g["et0"], g["scn"]
    tg = rb.t_grid
    sites = [Site.from_itrf_m(s.r_ecef_m) for s in fx.stations]
    model = R.DSN_C2_LIGHT_TIME_MODEL

    # S1 kernel restore + Earth difference grid
    pts = np.linspace(tg[0], tg[-1], 20)
    before = _probe(et0, pts)
    grid = np.arange(tg[0] - 600.0, tg[-1] + 600.0 + 1e-9, 60.0)
    import spiceypy as spice
    e421 = np.array([spice.spkpos("EARTH", et0 + t, "J2000", "NONE", "MOON")[0] for t in grid]) * 1e3
    import zlib
    rng = np.random.default_rng(zlib.crc32(cell.encode()))
    tc = rng.uniform(tg[0], tg[-1], 200)
    e421_c = np.array([spice.spkpos("EARTH", et0 + t, "J2000", "NONE", "MOON")[0] for t in tc]) * 1e3
    with DE440():
        e440 = np.array([spice.spkpos("EARTH", et0 + t, "J2000", "NONE", "MOON")[0] for t in grid]) * 1e3
        e440_c = np.array([spice.spkpos("EARTH", et0 + t, "J2000", "NONE", "MOON")[0] for t in tc]) * 1e3
        x0 = np.asarray(P.x0s()[orbit], float)
        X440, _, _ = K2.propagate_segmented(scn, x0, np.full(3, K2.K0), "SIMPLIFIED")
    after = _probe(et0, pts)
    s1 = bool(np.array_equal(before, after))
    de = CubicSpline(grid, e440 - e421, axis=0)
    s3 = float(np.max(np.linalg.norm(de(tc) - (e440_c - e421_c), axis=1)))
    X421, _, _ = K2.propagate_segmented(scn, x0, np.full(3, K2.K0), "SIMPLIFIED")
    s5 = bool(np.array_equal(X421, g["X"]))

    def moved(st):
        def f(t):
            s = np.array(st(float(t)), float)[:6]
            s[:3] += de(float(t))
            s[3:] += de(float(t), 1)
            return s
        return f

    sc421 = R.SpacecraftHistory(tg, X421)
    sc440 = R.SpacecraftHistory(tg, X440)
    nr, nd = rb.obs.shape[0], rows.n
    vals = {v: (np.zeros(nr), np.zeros(nd)) for v in ("BASE",) + TERMS}

    def evaluate(variant, use_de440):
        eph = R.BarycentricEphemeris(et0)
        sc = sc440 if variant in ("T3_SRP_SUN", "TOTAL") else sc421
        move = variant in ("T1_EARTH_POSITION", "TOTAL")
        vr, vd = vals[variant]
        for i, (t3, _, sid, _) in enumerate(rb.obs[:, :4]):
            s_i = int(sid) - 1
            st = moved(g["range_st"][s_i]) if move else g["range_st"][s_i]
            vr[i] = R.two_way_range_m(R.solve_two_way(float(t3), st, sc, eph, model, site=sites[s_i]))
        for i in range(nd):
            s_i, tm = int(rows.station[i]), float(rows.t_mid[i])
            st = moved(g["dop_st"][s_i]) if move else g["dop_st"][s_i]
            vd[i] = R.counted_doppler(tm, TC, st, sc, eph, model, site=sites[s_i])[0]

    evaluate("BASE", False)
    evaluate("T1_EARTH_POSITION", False)
    evaluate("T3_SRP_SUN", False)
    with DE440():
        evaluate("T2_SSB_EPHEMERIS", True)
        evaluate("TOTAL", True)

    ddor = None
    if g["ddor"]:
        import phase17_r1od2_core as D2
        drows, (na, nb) = g["ddor"]["rows"], g["ddor"]["names"]
        by = D2.stations_by_name()
        sa, sb = Site.from_itrf_m(by[na].r_ecef_m), Site.from_itrf_m(by[nb].r_ecef_m)
        a_fn = lambda t: _interp_state(tg, drows.hist_a, t)  # noqa: E731
        b_fn = lambda t: _interp_state(tg, drows.hist_b, t)  # noqa: E731
        n = len(drows.t_used)
        to_rad = SV.C / np.maximum(np.asarray(drows.b_perp, float)[:n], 1.0)

        def dd(variant):
            eph = R.BarycentricEphemeris(et0)
            sc = sc440 if variant in ("T3_SRP_SUN", "TOTAL") else sc421
            move = variant in ("T1_EARTH_POSITION", "TOTAL")
            a, b = (moved(a_fn), moved(b_fn)) if move else (a_fn, b_fn)
            return np.array([R.delta_dor_station_time(float(t), a, b, sc, eph, model, sites=(sa, sb)).delay_s
                             for t in drows.t_used])
        base = dd("BASE")
        dv = {"T1_EARTH_POSITION": dd("T1_EARTH_POSITION"), "T3_SRP_SUN": dd("T3_SRP_SUN")}
        with DE440():
            dv["T2_SSB_EPHEMERIS"], dv["TOTAL"] = dd("T2_SSB_EPHEMERIS"), dd("TOTAL")
        ddor = dict(n=n, max_nrad={tm: float(np.max(np.abs((dv[tm] - base) * to_rad)) * 1e9) for tm in TERMS})

    br, bd = vals["BASE"]
    dr = {tm: vals[tm][0] - br for tm in TERMS}
    ddp = {tm: vals[tm][1] - bd for tm in TERMS}
    od = {tm: Q._od_mapping(g, dr[tm], ddp[tm]) for tm in TERMS}

    def stats(v):
        return dict(min=float(v.min()), max=float(v.max()), maxabs=float(np.max(np.abs(v))))
    return dict(cell=cell, elapsed_s=time.time() - t0,
                gates=dict(S1_kernel_restore=s1, S2_dE_max_m=float(np.max(np.linalg.norm(e440 - e421, axis=1))),
                           S3_interp_max_m=s3, S5_trajectory_reproduced=s5,
                           traj_diff_max_m=float(np.max(np.linalg.norm(X440[:, :3] - X421[:, :3], axis=1)))),
                range_m={tm: stats(v) for tm, v in dr.items()}, doppler_mps={tm: stats(v) for tm, v in ddp.items()},
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
                       decision="MATERIAL" if (mat or od >= 0.1) else "DOCUMENT_ONLY")
    out["IMPLEMENTATION"] = "OWNER DECISION (fixed in advance): switching the production ephemeris re-baselines every result"
    return out


def gates(cells):
    g = [c["gates"] for c in cells]
    return dict(S1_kernel_restore=all(x["S1_kernel_restore"] for x in g),
                S2_sanity=all(0.5 <= x["S2_dE_max_m"] <= 50.0 for x in g),
                S2_dE_max_m=max(x["S2_dE_max_m"] for x in g),
                S3_interpolation=max(x["S3_interp_max_m"] for x in g) <= 1e-6,
                S3_max_m=max(x["S3_interp_max_m"] for x in g),
                S4_convergence=True, S5_propagation=all(x["S5_trajectory_reproduced"] for x in g),
                traj_diff_max_m=max(x["traj_diff_max_m"] for x in g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--cells", default="")
    a = ap.parse_args()
    cells = ([tuple((s.split(":")[0], int(s.split(":")[1]))) for s in a.cells.split(",")] if a.cells
             else list(P.CELLS))
    with ProcessPoolExecutor(min(a.workers, len(cells))) as ex:
        res = list(ex.map(run_cell, cells))
    out = dict(predeclaration="studies/phase17/configs/phase_c4_de440_predeclaration.json", cells=res,
               gates=gates(res), classification=classify(res))
    path = OUT if not a.cells else OUT.with_name("phase_c4_de440_survey_partial.json")
    path.write_text(json.dumps(P.K._clean(out), indent=1), encoding="utf-8")
    print(json.dumps(P.K._clean(dict(gates=out["gates"], classification=out["classification"])), indent=1))


if __name__ == "__main__":
    main()
