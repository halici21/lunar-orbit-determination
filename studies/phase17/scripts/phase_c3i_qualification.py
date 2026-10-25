"""Phase C3-I: qualification of station_frames.py, solid_tide.py and the station-displacement wrapper.

Predeclared in studies/phase17/configs/phase_c3i_station_displacement_predeclaration.json. Gates G1 (bit
identity; the full suite runs separately), G2 (Helmert vs Orekit), G3 (solid tide vs Orekit), G4 (survey
parity), G6 (EOP prediction flag), G7 (informational OD impact). G5: phase_c3i_estimator_gates.py.
Writes studies/phase17/results/phase_c3i_qualification.json.

    python studies/phase17/scripts/phase_c3i_qualification.py --reference-sha256 <sha> [--workers 10]
"""
from __future__ import annotations

import argparse
import hashlib
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
import phase_c3_station_frame_survey as S3  # noqa: E402

ROOT = HERE.parents[2]
REF = P.ARTIFACTS / "phase_c3i_reference.npz"
OUT = P.ARTIFACTS / "phase_c3i_qualification.json"
BASE_COMMIT = "1354c0a"
TC = 60.0


def _base_module():
    name = "lunar_od._rlt_base_c3"
    if name in sys.modules:
        return sys.modules[name]
    import lunar_od  # noqa: F401
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASE_COMMIT}:src/lunar_od/relativistic_light_time.py"],
                         check=True, capture_output=True).stdout
    path = Path(tempfile.mkdtemp()) / "_rlt_base_c3.py"
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "lunar_od"
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def g2() -> dict:
    from lunar_od.station_frames import helmert_parameters, itrf2020_to_itrf93
    r = np.load(REF, allow_pickle=False)
    worst, scale = 0.0, 0.0
    for i, p in enumerate(r["g2_sites"]):
        for j, et in enumerate(r["g2_et"]):
            mine = itrf2020_to_itrf93(p, float(et), include_scale=False) - p
            worst = max(worst, float(np.linalg.norm(mine - r["g2_offset"][i, j])))
            scale = max(scale, float(np.linalg.norm(helmert_parameters(float(et))[1] * p)))
    return dict(max_abs_m=worst, limit_m=1e-4, pass_=bool(worst <= 1e-4), scale_part_max_m=scale,
                n=int(r["g2_offset"].shape[0] * r["g2_offset"].shape[1]))


def g3_cell(cell: str) -> dict:
    from lunar_od.solid_tide import solid_tide_at_et
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels(None, clear=True)
    r = np.load(REF, allow_pickle=False)
    grid = json.loads(S3.GRID.read_text(encoding="utf-8"))
    et0 = float(grid["cells"][cell]["et0"])
    worst, n = 0.0, 0
    for name in [str(x) for x in r["site_names"]]:
        t, ref = r[f"g3|{cell}|{name}|t"], r[f"g3|{cell}|{name}|disp"]
        p = np.asarray(grid["stations"][name], float)
        for i, ti in enumerate(t):
            worst = max(worst, float(np.linalg.norm(solid_tide_at_et(p, et0 + float(ti)) - ref[i])))
            n += 1
    return dict(cell=cell, max_abs_m=worst, n=n)


def run_cell(spec) -> dict:
    orbit, k = spec
    from lunar_od import relativistic_light_time as R
    from lunar_od.radiometrics import _interp_state
    from lunar_od.station_time import Site

    t0 = time.time()
    B = _base_module()
    ref3 = np.load(S3.REFERENCE, allow_pickle=False)
    g = SV.build_cell(orbit, k)
    cell = f"{orbit}:{k}"
    rb, rows, fx, et0 = g["rb"], g["rows"], g["fx"], g["et0"]
    names = g["names"]
    sites = [Site.from_itrf_m(s.r_ecef_m) for s in fx.stations]
    sp, _ = S3._perturbations(g, ref3, cell, names)
    model = R.DSN_C2_LIGHT_TIME_MODEL
    eph, eph_b = R.BarycentricEphemeris(et0), B.BarycentricEphemeris(et0)
    sc, sc_b = R.SpacecraftHistory(rb.t_grid, rb.nom48[:, :6]), B.SpacecraftHistory(rb.t_grid, rb.nom48[:, :6])
    disp_kw = dict(solid_tide=True, coordinate_frame="ITRF2020", include_scale=False)

    def survey_pert(st, name):
        def f(t):
            s = np.array(st(float(t)), float)[:6]
            for tm in ("T2A_ITRF_REALIZATION", "T3_SOLID_TIDE"):
                s[:3] += sp[name][tm](float(t))
                s[3:] += sp[name][tm](float(t), 1)
            return s
        return f

    g1 = dict(range=0.0, doppler=0.0, ddor=0.0)
    g4 = dict(range=0.0, doppler=0.0, ddor=0.0)
    chg_r, chg_d = np.zeros(rb.obs.shape[0]), np.zeros(rows.n)
    for i, (t3, _, sid, _) in enumerate(rb.obs[:, :4]):
        s_i = int(sid) - 1
        st, site = g["range_st"][s_i], sites[s_i]
        new = R.two_way_range_m(R.solve_two_way(float(t3), st, sc, eph, model, site=site))
        base = B.two_way_range_m(B.solve_two_way(float(t3), st, sc_b, eph_b, B.DSN_C2_LIGHT_TIME_MODEL, site=site))
        g1["range"] = max(g1["range"], abs(new - base))
        mod = R.two_way_range_m(R.solve_two_way(float(t3), R.displaced_station_state(st, et0, fx.stations[s_i].r_ecef_m,
                                                                                    **disp_kw), sc, eph, model, site=site))
        sv = R.two_way_range_m(R.solve_two_way(float(t3), survey_pert(st, names[s_i]), sc, eph, model, site=site))
        g4["range"] = max(g4["range"], abs(mod - sv))
        full = R.two_way_range_m(R.solve_two_way(float(t3), R.displaced_station_state(st, et0, fx.stations[s_i].r_ecef_m,
                                                                                     solid_tide=True,
                                                                                     coordinate_frame="ITRF2020"),
                                                 sc, eph, model, site=site))
        chg_r[i] = full - new
    for i in range(rows.n):
        s_i, tm = int(rows.station[i]), float(rows.t_mid[i])
        st, site = g["dop_st"][s_i], sites[s_i]
        new = R.counted_doppler(tm, TC, st, sc, eph, model, site=site)[0]
        base = B.counted_doppler(tm, TC, st, sc_b, eph_b, B.DSN_C2_LIGHT_TIME_MODEL, site=site)[0]
        g1["doppler"] = max(g1["doppler"], abs(new - base))
        dsp = R.displaced_station_state(st, et0, fx.stations[s_i].r_ecef_m, **disp_kw)
        mod = R.counted_doppler(tm, TC, dsp, sc, eph, model, site=site)[0]
        sv = R.counted_doppler(tm, TC, survey_pert(st, names[s_i]), sc, eph, model, site=site)[0]
        g4["doppler"] = max(g4["doppler"], abs(mod - sv))
        dfull = R.displaced_station_state(st, et0, fx.stations[s_i].r_ecef_m, solid_tide=True,
                                          coordinate_frame="ITRF2020")
        chg_d[i] = R.counted_doppler(tm, TC, dfull, sc, eph, model, site=site)[0] - new
    if g["ddor"]:
        import phase17_r1od2_core as D2
        drows, (na, nb) = g["ddor"]["rows"], g["ddor"]["names"]
        by = D2.stations_by_name()
        sa, sb = Site.from_itrf_m(by[na].r_ecef_m), Site.from_itrf_m(by[nb].r_ecef_m)
        a_fn = lambda t: _interp_state(rb.t_grid, drows.hist_a, t)  # noqa: E731
        b_fn = lambda t: _interp_state(rb.t_grid, drows.hist_b, t)  # noqa: E731
        for t_obs in drows.t_used:
            t_obs = float(t_obs)
            new = R.delta_dor_station_time(t_obs, a_fn, b_fn, sc, eph, model, sites=(sa, sb)).delay_s
            base = B.delta_dor_station_time(t_obs, a_fn, b_fn, sc_b, eph_b, B.DSN_C2_LIGHT_TIME_MODEL,
                                            sites=(sa, sb)).delay_s
            g1["ddor"] = max(g1["ddor"], abs(new - base))
            mod = R.delta_dor_station_time(t_obs, R.displaced_station_state(a_fn, et0, by[na].r_ecef_m, **disp_kw),
                                           R.displaced_station_state(b_fn, et0, by[nb].r_ecef_m, **disp_kw),
                                           sc, eph, model, sites=(sa, sb)).delay_s
            sv = R.delta_dor_station_time(t_obs, survey_pert(a_fn, na), survey_pert(b_fn, nb), sc, eph, model,
                                          sites=(sa, sb)).delay_s
            g4["ddor"] = max(g4["ddor"], abs(mod - sv))
    return dict(cell=cell, g1=g1, g4=g4, elapsed_s=time.time() - t0,
                change=dict(range_max_abs_m=float(np.max(np.abs(chg_r))), doppler_max_abs_mps=float(np.max(np.abs(chg_d)))),
                g7=Q._od_mapping(g, chg_r, chg_d))


def g6() -> dict:
    import spiceypy as spice
    from lunar_od.spice_loader import load_spice_kernels
    from lunar_od.station_frames import is_predicted_earth_orientation, predicted_earth_orientation_start_et
    load_spice_kernels(None, clear=True)
    grid = json.loads(S3.GRID.read_text(encoding="utf-8"))
    survey = all(is_predicted_earth_orientation(float(c["et0"]) + x) for c in grid["cells"].values()
                 for x in (c["t_min"], c["t_max"]))
    past = is_predicted_earth_orientation(float(spice.str2et("2025-03-01T00:00:00")))
    return dict(start_et=predicted_earth_orientation_start_et(), survey_epochs_predicted=survey,
                past_2025_03_predicted=past, pass_=bool(survey and not past))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--reference-sha256", required=True)
    a = ap.parse_args()
    t0 = time.time()
    ref_ok = hashlib.sha256(REF.read_bytes()).hexdigest() == a.reference_sha256
    cells = list(P.CELLS)
    with ProcessPoolExecutor(a.workers) as ex:
        f3 = list(ex.map(g3_cell, [f"{o}:{k}" for o, k in cells]))
        res = list(ex.map(run_cell, cells))
    gates = dict(
        reference_ok=ref_ok,
        G1_bit_identity=dict(passed=all(v == 0.0 for c in res for v in c["g1"].values()),
                             max_abs={kk: max(c["g1"][kk] for c in res) for kk in ("range", "doppler", "ddor")}),
        G2_helmert=g2(),
        G3_solid_tide=dict(max_abs_m=max(x["max_abs_m"] for x in f3), n=sum(x["n"] for x in f3), limit_m=1e-3,
                           pass_=bool(max(x["max_abs_m"] for x in f3) <= 1e-3)),
        G4_survey_parity=dict(range_m=max(c["g4"]["range"] for c in res), doppler_mps=max(c["g4"]["doppler"] for c in res),
                              ddor_s=max(c["g4"]["ddor"] for c in res)),
        G6_eop_flag=g6(),
    )
    g4 = gates["G4_survey_parity"]
    g4["pass_"] = bool(g4["range_m"] <= 2e-3 and g4["doppler_mps"] <= 2e-7 and g4["ddor_s"] <= 1e-11)
    keys = list(res[0]["g7"])
    gates["G7_informational"] = dict(
        od={kk: max(max(c["g7"][kk]["max_state_over_sigma"], c["g7"][kk]["k_over_sigma"]) for c in res) for kk in keys},
        change_range_m=max(c["change"]["range_max_abs_m"] for c in res),
        change_doppler_mps=max(c["change"]["doppler_max_abs_mps"] for c in res))
    out = dict(predeclaration="studies/phase17/configs/phase_c3i_station_displacement_predeclaration.json",
               base_commit=BASE_COMMIT, gates=gates, cells=res, g3_cells=f3, elapsed_s=time.time() - t0)
    OUT.write_text(json.dumps(P.K._clean(out), indent=1), encoding="utf-8")
    print(json.dumps(P.K._clean(gates), indent=1))


if __name__ == "__main__":
    main()
