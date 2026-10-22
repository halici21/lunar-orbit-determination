"""Phase C3: terrestrial reference frame, EOP and station displacements - materiality survey (analysis only).

Predeclared in studies/phase17/configs/phase_c3_station_frame_predeclaration.json (committed before any number).

    python studies/phase17/scripts/phase_c3_station_frame_survey.py --export-grid
        writes studies/phase17/results/phase_c3_station_grid.json (cell spans, station ECEF vectors)
    <agent-sandbox>/orekit-gmat-oracle: ..\\.venv\\Scripts\\python.exe c3_station_reference.py <grid> <reference.npz>
        (copy: studies/phase17/scripts/phase_c3_station_reference_sandbox.py)
    python studies/phase17/scripts/phase_c3_station_frame_survey.py --reference-sha256 <sha> [--workers 10]
        writes studies/phase17/results/phase_c3_station_frame_survey.json

Each term is an Earth-relative station position perturbation dr(t) (J2000/GCRF axes) from the Orekit reference
(variants A, B, TIDE, POLE) and the production station S(t). Every observable is re-solved exactly with the
perturbed station providers under the C2-I model; OD impact is the C2 linear bias mapping.
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
import phase_c2_time_scale_survey as SV  # noqa: E402
import phase_c2i_qualification as Q  # noqa: E402

GRID = P.ARTIFACTS / "phase_c3_station_grid.json"
REFERENCE = P.ARTIFACTS / "phase_c3_station_reference.npz"
OUT = P.ARTIFACTS / "phase_c3_station_frame_survey.json"
TC = 60.0
TERMS = ("T2A_ITRF_REALIZATION", "T2B_EOP_AND_PN_MODEL", "T2_TOTAL", "T3_SOLID_TIDE", "T4_POLE_TIDE")
NOISE = SV.NOISE


def export_cell(spec):
    g = SV.build_cell(*spec)
    t = g["rb"].t_grid
    return f"{spec[0]}:{spec[1]}", dict(et0=g["et0"], t_min=float(t[0]), t_max=float(t[-1]))


def _earth_relative(g, name_index, t):
    """Production Earth-relative J2000 station position S(t) [m] (exact-epoch provider minus Earth state)."""
    from lunar_od.radiometrics import _interp_state
    pg = g["pg"]
    earth = np.hstack([np.asarray(pg.earth_pos_mci_m, float), np.asarray(pg.earth_vel_mci_mps, float)])
    st = g["range_st"][name_index]
    return np.array([np.asarray(st(float(x)))[:3] - _interp_state(g["rb"].t_grid, earth, float(x))[:3] for x in t])


def _perturbations(g, ref, cell, names):
    """CubicSpline dr(t) per term and station (Earth-relative, J2000 axes)."""
    from scipy.interpolate import CubicSpline
    out = {}
    s4 = 0.0
    for k, name in enumerate(names):
        key = f"{cell}|{name}"
        t = ref[key + "|t"]
        grid = ref[key + "|grid"]                       # (n, 4, 3): A, B, TIDE, POLE
        s = _earth_relative(g, k, t)
        terms = {"T2A_ITRF_REALIZATION": grid[:, 0] - grid[:, 1], "T2B_EOP_AND_PN_MODEL": grid[:, 1] - s,
                 "T2_TOTAL": grid[:, 0] - s, "T3_SOLID_TIDE": grid[:, 2], "T4_POLE_TIDE": grid[:, 3]}
        out[name] = {tm: CubicSpline(t, v, axis=0) for tm, v in terms.items()}
        # S4: interpolation of the reference-only terms against direct values (S is not interpolated)
        tc, chk = ref[key + "|tc"], ref[key + "|check"]
        for tm, direct in (("T2A_ITRF_REALIZATION", chk[:, 0] - chk[:, 1]), ("T3_SOLID_TIDE", chk[:, 2]),
                           ("T4_POLE_TIDE", chk[:, 3])):
            s4 = max(s4, float(np.max(np.linalg.norm(out[name][tm](tc) - direct, axis=1))))
        sp_b = CubicSpline(t, grid[:, 1], axis=0)
        s_c = _earth_relative(g, k, tc)
        s4 = max(s4, float(np.max(np.linalg.norm((sp_b(tc) - s_c) - out[name]["T2B_EOP_AND_PN_MODEL"](tc), axis=1))))
    return out, s4


def _perturbed(st, spline):
    def f(t):
        s = np.array(st(float(t)), float)[:6]
        s[:3] += spline(float(t))
        s[3:] += spline(float(t), 1)
        return s
    return f


def run_cell(spec) -> dict:
    orbit, k = spec
    from lunar_od import relativistic_light_time as R
    from lunar_od.radiometrics import _interp_state
    from lunar_od.station_time import Site
    import spiceypy as spice

    t_start = time.time()
    ref = np.load(REFERENCE, allow_pickle=False)
    g = SV.build_cell(orbit, k)
    cell = f"{orbit}:{k}"
    rb, rows, fx, et0 = g["rb"], g["rows"], g["fx"], g["et0"]
    names = g["names"]
    sites = [Site.from_itrf_m(s.r_ecef_m) for s in fx.stations]
    model = R.DSN_C2_LIGHT_TIME_MODEL
    eph = R.BarycentricEphemeris(et0)
    sc = R.SpacecraftHistory(rb.t_grid, rb.nom48[:, :6])
    sp, s4 = _perturbations(g, ref, cell, names)

    # S1: production station vs pxform(ITRF93 -> J2000) r_ecef
    tt = np.linspace(rb.t_grid[0], rb.t_grid[-1], 200)
    s1 = 0.0
    for kk, st in enumerate(fx.stations):
        s = _earth_relative(g, kk, tt)
        direct = np.array([np.asarray(spice.pxform("ITRF93", "J2000", et0 + float(x))) @ np.asarray(st.r_ecef_m, float)
                           for x in tt])
        s1 = max(s1, float(np.max(np.linalg.norm(s - direct, axis=1))))

    station_mag = {n: {tm: float(np.max(np.linalg.norm(sp[n][tm](tt), axis=1))) for tm in TERMS} for n in names}
    tide_peak = {n: float(np.max(np.linalg.norm(sp[n]["T3_SOLID_TIDE"](tt), axis=1))) for n in names}

    nr, nd = rb.obs.shape[0], rows.n
    dr = {tm: np.zeros(nr) for tm in TERMS}
    dd = {tm: np.zeros(nd) for tm in TERMS}
    for i, (t3, _, sid, _) in enumerate(rb.obs[:, :4]):
        s_i = int(sid) - 1
        st, site = g["range_st"][s_i], sites[s_i]
        base = R.two_way_range_m(R.solve_two_way(float(t3), st, sc, eph, model, site=site))
        for tm in TERMS:
            dr[tm][i] = R.two_way_range_m(R.solve_two_way(float(t3), _perturbed(st, sp[names[s_i]][tm]), sc, eph,
                                                          model, site=site)) - base
    for i in range(nd):
        s_i, tm_ = int(rows.station[i]), float(rows.t_mid[i])
        st, site = g["dop_st"][s_i], sites[s_i]
        base = R.counted_doppler(tm_, TC, st, sc, eph, model, site=site)[0]
        for tm in TERMS:
            dd[tm][i] = R.counted_doppler(tm_, TC, _perturbed(st, sp[names[s_i]][tm]), sc, eph, model, site=site)[0] - base

    ddor = None
    if g["ddor"]:
        import phase17_r1od2_core as D2
        drows, (na, nb) = g["ddor"]["rows"], g["ddor"]["names"]
        by = D2.stations_by_name()
        site_a, site_b = Site.from_itrf_m(by[na].r_ecef_m), Site.from_itrf_m(by[nb].r_ecef_m)
        a_fn = lambda t: _interp_state(rb.t_grid, drows.hist_a, t)  # noqa: E731
        b_fn = lambda t: _interp_state(rb.t_grid, drows.hist_b, t)  # noqa: E731
        n = len(drows.t_used)
        to_rad = SV.C / np.maximum(np.asarray(drows.b_perp, float)[:n], 1.0)
        val = {tm: np.zeros(n) for tm in TERMS}
        for i, t_obs in enumerate(drows.t_used):
            t_obs = float(t_obs)
            base = R.delta_dor_station_time(t_obs, a_fn, b_fn, sc, eph, model, sites=(site_a, site_b)).delay_s
            for tm in TERMS:
                val[tm][i] = R.delta_dor_station_time(t_obs, _perturbed(a_fn, sp[na][tm]), _perturbed(b_fn, sp[nb][tm]),
                                                      sc, eph, model, sites=(site_a, site_b)).delay_s - base
        ddor = dict(n=n, max_nrad={tm: float(np.max(np.abs(val[tm] * to_rad)) * 1e9) for tm in TERMS},
                    max_ns={tm: float(np.max(np.abs(val[tm])) * 1e9) for tm in TERMS})

    od = {tm: Q._od_mapping(g, dr[tm], dd[tm]) for tm in TERMS}

    def stats(v):
        return dict(min=float(v.min()), max=float(v.max()), maxabs=float(np.max(np.abs(v))))
    return dict(cell=cell, n_range=nr, n_doppler=nd, elapsed_s=time.time() - t_start,
                gates=dict(S1_max_m=s1, S4_max_m=s4, tide_peak_m=tide_peak),
                station_dr_max_m=station_mag,
                range_m={tm: stats(v) for tm, v in dr.items()}, doppler_mps={tm: stats(v) for tm, v in dd.items()},
                ddor=ddor, od=od)


def classify(cells):
    dsn = NOISE["DSN_REALISTIC"]
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
    out.update(T1_STATION_COORDINATES="CONVENTION_REAL_DATA (fixed in advance)",
               T5_OCEAN_LOADING="NOT_MEASURED (needs BLQ coefficients; owner input)",
               T6_PLATE_MOTION="CONVENTION_REAL_DATA (fixed in advance)",
               T7_EOP_FINAL_VS_PREDICTED="NOT_MEASURABLE at 2027; final EOP required for real data")
    return out


def gates(cells, reference_ok):
    s1 = max(c["gates"]["S1_max_m"] for c in cells)
    s4 = max(c["gates"]["S4_max_m"] for c in cells)
    t2b = max(c["station_dr_max_m"][n]["T2B_EOP_AND_PN_MODEL"] for c in cells for n in c["station_dr_max_m"])
    t4 = max(c["station_dr_max_m"][n]["T4_POLE_TIDE"] for c in cells for n in c["station_dr_max_m"])
    peaks = [p for c in cells for p in c["gates"]["tide_peak_m"].values()]
    return dict(S1_spice_station=s1 <= 1e-3, S1_max_m=s1,
                S2_sanity=bool(t2b <= 2.0 and 0.05 <= min(peaks) and max(peaks) <= 0.6 and t4 <= 0.05),
                S2_values=dict(t2b_max_m=t2b, tide_peak_min_m=min(peaks), tide_peak_max_m=max(peaks), pole_max_m=t4),
                S3_phase_b_informational={c["cell"]: {n: c["station_dr_max_m"][n]["T2_TOTAL"]
                                                      for n in c["station_dr_max_m"]} for c in cells[:2]},
                S4_interpolation=s4 <= 1e-4, S4_max_m=s4, S5_reference_record=reference_ok,
                S6_convergence=True, S6_note="every solve returned (production raises otherwise)")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--export-grid", action="store_true")
    ap.add_argument("--reference-sha256", default="")
    ap.add_argument("--cells", default="")
    a = ap.parse_args()
    cells = ([tuple((s.split(":")[0], int(s.split(":")[1]))) for s in a.cells.split(",")] if a.cells
             else list(P.CELLS))
    if a.export_grid:
        with ProcessPoolExecutor(min(a.workers, len(cells))) as ex:
            res = dict(ex.map(export_cell, cells))
        P.K._init()
        pos = {s.name: [float(v) for v in np.asarray(s.r_ecef_m, float)] for s in P.K.stations()}
        GRID.write_text(json.dumps(dict(cells=res, stations=pos), indent=1), encoding="utf-8")
        print(json.dumps(dict(grid=str(GRID), sha256=sha256(GRID))))
        return
    meta = json.loads(str(np.load(REFERENCE, allow_pickle=False)["meta"]))
    ref_ok = sha256(REFERENCE) == a.reference_sha256 and meta["grid_sha256"] == sha256(GRID)
    with ProcessPoolExecutor(min(a.workers, len(cells))) as ex:
        res = list(ex.map(run_cell, cells))
    out = dict(predeclaration="studies/phase17/configs/phase_c3_station_frame_predeclaration.json",
               reference=dict(file=REFERENCE.name, sha256=sha256(REFERENCE), meta=meta),
               cells=res, gates=gates(res, ref_ok), classification=classify(res))
    path = OUT if not a.cells else OUT.with_name("phase_c3_station_frame_survey_partial.json")
    path.write_text(json.dumps(P.K._clean(out), indent=1), encoding="utf-8")
    print(json.dumps(P.K._clean(dict(gates=out["gates"], classification=out["classification"])), indent=1))


if __name__ == "__main__":
    main()
