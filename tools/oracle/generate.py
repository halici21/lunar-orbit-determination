"""Generate the pinned WP5.1 Orekit oracle fixtures (``tests/fixtures/external_oracle``).

Two steps, in this order (README.md has the full recipe):

1. ``python tools/oracle/generate.py lunar-od``      (repository environment, no Orekit)
   writes ``scenario.json``, runs the lunar_od integrator ladder (self-convergence only),
   writes ``lunar_od_settings.json`` and the shared spacecraft trajectory
   ``shared_trajectory_d1a.csv``.
2. ``<python with orekit-jpype> tools/oracle/generate.py orekit --orekit-data DIR --jpl-dir DIR --jar JAR``
   with ``PYTHONPATH=<repo>/src`` (lunar_od is used only for the kernel loader, the
   GL1800F parser and SPICE preconditions): preconditions, the Orekit integrator ladder,
   ``orekit_dynamics_<case>.csv``, the station track ``orekit_station_track.csv``,
   ``orekit_measurements.csv`` and ``provenance.json`` (versions, SHA-256 of every input
   and output file).

No comparison between the tools is made here: ``studies/phase17/scripts/wp5_1_orekit_oracle_gates.py``
and ``tests/test_external_oracle.py`` do that from the pinned files.
Predeclaration: ``studies/phase17/configs/wp5_1_orekit_oracle_predeclaration.json``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenario import (DYNAMICS_CASES, FIXTURE_DIR, ROOT, RUN_DIR, STATION_GRID, T_MEAS, TC_S,  # noqa: E402
                      TRAJ_GRID, build_scenario, ephemeris_of, gravity_tab, harmonic_degree, kernel_dir,
                      load_kernels, load_scenario, output_grid, read_csv, sha256, write_csv)

FLOOR_M = 1e-4  # settings_selection.rule: next rung changes |r| by <= 0.1 mm
STATE_HEADER = ["t_tdb_s", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _commit_info() -> dict:
    return {"commit": _git("rev-parse", "HEAD"),
            "tree_clean_outside_fixtures": not [ln for ln in _git("status", "--porcelain").splitlines()
                                                if "tests/fixtures/external_oracle" not in ln
                                                and "outputs/" not in ln]}


def _ladder(run, rungs, name: str, log: dict):
    """Self-convergence: accept rung k when |run(k+1) - run(k)| <= FLOOR_M; returns (index, states)."""
    cache = {}

    def get(i):
        if i not in cache:
            t0 = time.perf_counter()
            cache[i] = run(rungs[i])
            log.setdefault("runtime_s", {})[str(i)] = round(time.perf_counter() - t0, 2)
        return cache[i]

    log["floors_m"] = {}
    for i in range(len(rungs) - 1):
        d = float(np.max(np.linalg.norm(get(i)[:, :3] - get(i + 1)[:, :3], axis=1)))
        log["floors_m"][f"{i}->{i + 1}"] = d
        print(f"  {name}: rung {i} vs {i + 1}: {d * 1e3:.4f} mm", flush=True)
        if d <= FLOOR_M:
            log.update(accepted=i, integrator_limited=False)
            return i, get(i)
    last = len(rungs) - 1
    log.update(accepted=last, integrator_limited=True)
    return last, get(last)


# ---------------------------------------------------------------------------
# step 1: lunar_od
# ---------------------------------------------------------------------------
def step_lunar_od(cases) -> None:
    import lunar_od_side as L

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    sc = build_scenario()
    (FIXTURE_DIR / "scenario.json").write_text(json.dumps(sc, indent=1) + "\n", encoding="utf-8")
    out_path = FIXTURE_DIR / "lunar_od_settings.json"
    settings = json.loads(out_path.read_text(encoding="utf-8")) if out_path.is_file() else {"cases": {}}
    for case in cases:
        log: dict = {}
        idx, _ = _ladder(lambda s, c=case: L.propagate_aligned(c, sc, time_arg="TT", setting=s)[1],
                         L.LADDER, f"lunar_od {case}", log)
        settings["cases"][case] = {"setting": L.LADDER[idx].as_dict(), "ladder": log,
                                   "rungs": [r.as_dict() for r in L.LADDER]}
    settings["rule"] = "accept rung k when max_samples |r(k+1) - r(k)| <= 1e-4 m (predeclaration settings_selection)"
    settings["generated"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    settings["lunar_od_commit"] = _commit_info()
    out_path.write_text(json.dumps(settings, indent=1) + "\n", encoding="utf-8")

    s_d1a = L.Setting.from_dict(settings["cases"]["D1a"]["setting"])
    _, traj, _ = L.propagate_aligned("D1a", sc, time_arg="TT", setting=s_d1a, t_out=TRAJ_GRID)
    write_csv(FIXTURE_DIR / "shared_trajectory_d1a.csv", STATE_HEADER, np.column_stack([TRAJ_GRID, traj]),
              "WP5.1 shared spacecraft trajectory (input to both measurement solvers).\n"
              "lunar_od D1a force model, TT-aligned adapter, setting " + s_d1a.label() + ".\n"
              "Moon-centred J2000/ICRF axes; t = TDB s since 2025-03-01T00:00:00 TDB.")
    print("lunar_od step done")


# ---------------------------------------------------------------------------
# step 2: Orekit
# ---------------------------------------------------------------------------
OREKIT_RUNGS = tuple({"abs_pos_m": 1e-8, "abs_vel_mps": 1e-11, "rel": 1e-14, "max_step_s": h}
                     for h in (300.0, 60.0, 20.0, 10.0, 5.0, 2.0, 1.0))
MASS_KG, AREA_M2 = 1000.0, 10.0


def _orekit_propagate(case: str, sc: dict, rung: dict) -> np.ndarray:
    import orekit_session as O
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
    from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, ThirdBodyAttraction
    from org.orekit.forces.radiation import (ConicallyShadowedLightFluxModel, IsotropicRadiationSingleCoefficient,
                                             RadiationPressureModel)
    from org.orekit.orbits import CartesianOrbit, OrbitType
    from org.orekit.propagation import SpacecraftState
    from org.orekit.propagation.numerical import NumericalPropagator
    from org.orekit.utils import PVCoordinates, TimeStampedPVCoordinates

    eph = ephemeris_of(case)
    frame = O.moon_icrf_frame(eph)
    x0 = sc["state0_mci_j2000_m_mps"]
    pv = TimeStampedPVCoordinates(O.tdb_date(0.0), PVCoordinates(Vector3D(*x0[:3]), Vector3D(*x0[3:])))
    orbit = CartesianOrbit(pv, frame, float(sc["gm_m3_s2"]["MOON"]))
    atol = [rung["abs_pos_m"]] * 3 + [rung["abs_vel_mps"]] * 3 + [1e-3]
    rtol = [rung["rel"]] * 7
    prop = NumericalPropagator(DormandPrince853Integrator(1e-3, rung["max_step_s"], atol, rtol))
    prop.setOrbitType(OrbitType.CARTESIAN)
    prop.setInitialState(SpacecraftState(orbit, MASS_KG))
    # central term: NumericalPropagator adds NewtonianAttraction(mu of the orbit)
    prop.addForceModel(ThirdBodyAttraction(O.gm_body("EARTH", eph)))
    prop.addForceModel(ThirdBodyAttraction(O.gm_body("SUN", eph)))
    n = harmonic_degree(case)
    if n is not None:
        prop.addForceModel(HolmesFeatherstoneAttractionModel(O.moon_pa_frame(eph), O.gl1800f_provider(n, gravity_tab())))
    if case == "D3":
        s = sc["srp"]
        flux = ConicallyShadowedLightFluxModel(s["P1AU_N_m2"] * s["AU_m"] ** 2, s["R_sun_m"], O.bodies(eph)["SUN"],
                                               s["R_occ_m"])
        cr = sc["k_srp_m2_per_kg"] * MASS_KG / AREA_M2
        prop.addForceModel(RadiationPressureModel(flux, IsotropicRadiationSingleCoefficient(AREA_M2, cr)))
    t = output_grid()
    x = np.empty((t.size, 6))
    for i, ti in enumerate(t):
        st = prop.propagate(O.tdb_date(ti)) if ti > 0 else prop.getInitialState()
        x[i] = O.pv_array(st.getPVCoordinates(frame))
    return x


def _preconditions(sc: dict) -> dict:
    import orekit_session as O
    import spiceypy as spice
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.orekit.time import TimeScalesFactory

    res: dict = {}
    t_chk = np.linspace(0.0, sc["arc_s"], 25)
    for eph in ("DE440", "DE421"):
        load_kernels(eph)
        f = O.moon_icrf_frame(eph)
        worst = {"EARTH": 0.0, "SUN": 0.0}
        for t in t_chk:
            d = O.tdb_date(t)
            for b in worst:
                p = O.vec(O.bodies(eph)[b].getPosition(d, f))
                s = np.asarray(spice.spkpos(b, O.et_of(d), "J2000", "NONE", "MOON")[0]) * 1e3
                worst[b] = max(worst[b], float(np.linalg.norm(p - s)))
        res[f"a_reader_vs_cspice_{eph}_m"] = worst
    load_kernels("DE440")
    worst = 0.0
    for t in t_chk:
        d = O.tdb_date(t)
        tr = O.moon_icrf_frame("DE440").getTransformTo(O.moon_pa_frame("DE440"), d)
        m = np.column_stack([O.vec(tr.transformVector(e)) for e in (Vector3D.PLUS_I, Vector3D.PLUS_J, Vector3D.PLUS_K)])
        worst = max(worst, float(np.abs(m - np.asarray(spice.pxform("J2000", "MOON_PA_DE440", O.et_of(d)))).max()))
    res["b_pck_euler_vs_pxform_max_element"] = worst
    rate = max(float(O.gcrf().getTransformTo(O.moon_icrf_frame("DE440"), O.tdb_date(t)).getRotationRate().getNorm())
               for t in (0.0, sc["arc_s"]))
    res["c_moon_icrf_rotation_rate_rad_s"] = rate
    tt, tdb = TimeScalesFactory.getTT(), TimeScalesFactory.getTDB()
    diff = []
    for t in output_grid():
        d = O.tdb_date(t)
        orekit = tdb.offsetFromTAI(d).toDouble() - tt.offsetFromTAI(d).toDouble()
        et = O.et_of(d)
        diff.append(orekit - (et - spice.unitim(et, "TDB", "TT")))
    res["d_tdb_minus_tt_orekit_minus_spice_s"] = {"mean": float(np.mean(diff)),
                                                   "change_over_arc": float(np.max(diff) - np.min(diff))}
    from jpype import JArray, JDouble
    from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
    from org.orekit.forces.gravity.potential import GravityFieldFactory, TideSystem
    from org.orekit.orbits import CartesianOrbit
    from org.orekit.propagation import SpacecraftState
    from org.orekit.utils import PVCoordinates, TimeStampedPVCoordinates

    c = JArray(JDouble, 2)([JArray(JDouble)([1.0]), JArray(JDouble)([0.0, 0.0]), JArray(JDouble)([0.0, 0.0, 0.0])])
    s = JArray(JDouble, 2)([JArray(JDouble)([0.0]), JArray(JDouble)([0.0, 0.0]), JArray(JDouble)([0.0, 0.0, 0.0])])
    hf = HolmesFeatherstoneAttractionModel(O.moon_pa_frame("DE440"), GravityFieldFactory.getNormalizedProvider(
        1738.0e3, float(sc["gm_m3_s2"]["MOON"]), TideSystem.UNKNOWN, c, s))
    x0 = sc["state0_mci_j2000_m_mps"]
    st = SpacecraftState(CartesianOrbit(TimeStampedPVCoordinates(O.tdb_date(0.0), PVCoordinates(
        Vector3D(*x0[:3]), Vector3D(*x0[3:]))), O.moon_icrf_frame("DE440"), float(sc["gm_m3_s2"]["MOON"])))
    res["e_holmes_featherstone_c00_only_accel_mps2"] = float(np.linalg.norm(O.vec(hf.acceleration(st, hf.getParameters()))))
    return res


def _orekit_measurements(sc: dict) -> tuple[np.ndarray, dict]:
    import orekit_session as O
    from java.util import ArrayList
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.orekit.bodies import GeodeticPoint, OneAxisEllipsoid
    from org.orekit.estimation.measurements import GroundStation, ObservableSatellite, Range
    from org.orekit.frames import FramesFactory, TopocentricFrame
    from org.orekit.propagation import SpacecraftState
    from org.orekit.utils import (AbsolutePVCoordinates, CartesianDerivativesFilter, Constants, IERSConventions,
                                  PVCoordinates, TimeStampedPVCoordinates,
                                  TimeStampedPVCoordinatesHermiteInterpolator)

    _, traj = read_csv(FIXTURE_DIR / "shared_trajectory_d1a.csv")
    grid, states = traj[:, 0], traj[:, 1:]
    if not np.array_equal(grid, TRAJ_GRID):
        raise ValueError("shared trajectory grid differs from scenario.TRAJ_GRID")
    mf = O.moon_icrf_frame("DE440")
    icrf = FramesFactory.getICRF()
    samples = [TimeStampedPVCoordinates(O.tdb_date(t), Vector3D(*x[:3]), Vector3D(*x[3:])) for t, x in zip(grid, states)]
    interp = TimeStampedPVCoordinatesHermiteInterpolator(8, CartesianDerivativesFilter.USE_PV)

    def sc_pv(date):
        t = float(date.durationFrom(O.tdb_date(0.0)))  # TAI-uniform seconds; only selects neighbours
        i = int(np.clip(np.searchsorted(grid, t) - 4, 0, grid.size - 8))
        lst = ArrayList()
        for k in range(i, i + 8):
            lst.add(samples[k])
        return interp.interpolate(date, lst)

    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, False)  # full EOP (not simple)
    earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS, Constants.WGS84_EARTH_FLATTENING, itrf)
    s = sc["station"]
    topo = TopocentricFrame(earth, GeodeticPoint(np.radians(s["lat_deg"]), np.radians(s["lon_deg"]), s["alt_m"]),
                            s["name"])
    station = GroundStation(topo)
    for drv in (station.getPrimeMeridianOffsetDriver(), station.getPrimeMeridianDriftDriver(),
                station.getPolarOffsetXDriver(), station.getPolarDriftXDriver(),
                station.getPolarOffsetYDriver(), station.getPolarDriftYDriver(),
                station.getClockOffsetDriver(), station.getClockDriftDriver(), station.getClockAccelerationDriver()):
        drv.setReferenceDate(O.tdb_date(0.0))  # all stay at their zero default
    sat = ObservableSatellite(0)

    st_track = np.array([O.pv_array(topo.getTransformTo(mf, O.tdb_date(t)).transformPVCoordinates(PVCoordinates.ZERO))
                         for t in STATION_GRID])

    def state_at(date, frame_name):
        pv = sc_pv(date)
        if frame_name == "MOON":
            return SpacecraftState(AbsolutePVCoordinates(mf, pv))
        return SpacecraftState(AbsolutePVCoordinates(icrf, mf.getTransformTo(icrf, date).transformPVCoordinates(pv)))

    iters = {"max_iterations": 0, "max_last_update_s": 0.0}

    def rng(t, two_way, frame_name):
        m = Range(station, two_way, O.tdb_date(t), 0.0, 1.0, 1.0, sat)
        k_sat = 1 if two_way else 0
        date = O.tdb_date(t)
        for it in range(1, 6):
            est = m.estimateWithoutDerivatives([state_at(date, frame_name)])
            new = est.getParticipants()[k_sat].getDate()
            upd = abs(float(new.durationFrom(date)))
            date = new
            if upd < 1e-9:
                break
        else:
            raise RuntimeError(f"Orekit transit epoch did not settle at t={t} ({frame_name}, two_way={two_way})")
        iters["max_iterations"] = max(iters["max_iterations"], it)
        iters["max_last_update_s"] = max(iters["max_last_update_s"], upd)
        est = m.estimateWithoutDerivatives([state_at(date, frame_name)])
        # the satellite participant must be the spacecraft on the shared trajectory
        p_sat = est.getParticipants()[k_sat]
        ref = sc_pv(p_sat.getDate())
        p_ref = O.vec(ref.getPosition()) if frame_name == "MOON" else O.vec(
            mf.getTransformTo(icrf, p_sat.getDate()).transformPVCoordinates(ref).getPosition())
        if np.linalg.norm(O.vec(p_sat.getPosition()) - p_ref) > 1e-6:
            raise RuntimeError("satellite participant is not the shared trajectory")
        return float(est.getEstimatedValue()[0])

    cols = {}
    for fr in ("MOON", "SSB"):
        r2 = lambda t, fr=fr: rng(t, True, fr)  # noqa: E731
        cols[f"two_way_{fr.lower()}"] = np.array([r2(t) for t in T_MEAS])
        cols[f"cd_{fr.lower()}"] = np.array([(r2(t + TC_S / 2) - r2(t - TC_S / 2)) / TC_S for t in T_MEAS])
    cols["one_way_moon"] = np.array([rng(t, False, "MOON") for t in T_MEAS])
    return st_track, {"cols": cols, "transit_epoch_iteration": iters}


def _versions() -> dict:
    import importlib.metadata as md

    v = {"python": platform.python_version(), "platform": platform.platform()}
    for p in ("orekit-jpype", "JPype1", "numpy", "scipy", "jplephem", "spiceypy", "numba"):
        try:
            v[p] = md.version(p)
        except md.PackageNotFoundError:
            v[p] = None
    from java.lang import System

    v["java"] = System.getProperty("java.version")
    v["java_vendor"] = System.getProperty("java.vendor")
    import orekit_jpype

    jars = sorted(p.name for p in (Path(orekit_jpype.__file__).parent / "jars").glob("*.jar"))
    v["orekit_jars"] = jars
    return v


def step_orekit(args) -> None:
    import orekit_session as O

    sc = load_scenario()
    O.start(Path(args.orekit_data), Path(args.jpl_dir), Path(args.jar), sc)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"preconditions": _preconditions(sc)}
    print(json.dumps(report["preconditions"], indent=1), flush=True)
    # Precondition (d) failed on 2026-11-25 (126 ns > 50 ns), so the TT-aligned lunar_od adapter
    # is given Orekit's own TDB->TT mapping, pinned at the 600 s output samples (predeclaration).
    write_csv(FIXTURE_DIR / "orekit_time_map.csv", ["t_tdb_s", "s_tt_s"],
              np.column_stack([output_grid(), [float(O.tdb_date(t).durationFrom(O.tdb_date(0.0)))
                                               for t in output_grid()]]),
              "WP5.1 Orekit time map: s = TAI/TT seconds elapsed between Orekit's TDB dates t0 and t0 + t.\n"
              "Orekit's TDB model, as used by its propagator; the TT-aligned lunar_od adapter inverts it.")

    report["dynamics"] = {}
    for case in args.cases:
        log: dict = {}
        idx, x = _ladder(lambda r, c=case: _orekit_propagate(c, sc, r), OREKIT_RUNGS, f"Orekit {case}", log)
        report["dynamics"][case] = {"setting": OREKIT_RUNGS[idx], "ladder": log}
        write_csv(FIXTURE_DIR / f"orekit_dynamics_{case}.csv", STATE_HEADER, np.column_stack([output_grid(), x]),
                  f"WP5.1 Orekit reference trajectory, case {case} (see provenance.json).\n"
                  "Moon-centred ICRF axes; t = TDB s since 2025-03-01T00:00:00 TDB; Orekit integrates in TAI/TT s.")

    if args.measurements:
        st_track, meas = _orekit_measurements(sc)
        report["measurements"] = {"transit_epoch_iteration": meas["transit_epoch_iteration"]}
        cols = meas["cols"]
        d = np.abs(cols["two_way_ssb"] - cols["two_way_moon"])
        report["preconditions"]["f_ssb_minus_moon_two_way_m"] = {"min": float(d.min()), "max": float(d.max())}
        write_csv(FIXTURE_DIR / "orekit_station_track.csv", STATE_HEADER, np.column_stack([STATION_GRID, st_track]),
                  "WP5.1 Orekit Canberra DSN station track (input to both measurement solvers).\n"
                  "WGS84 geodetic point in ITRF (IERS 2010 conventions, full EOP from orekit-data),\n"
                  "expressed in Moon-centred ICRF axes; t = TDB s since 2025-03-01T00:00:00 TDB.")
        names = ["two_way_moon", "cd_moon", "two_way_ssb", "cd_ssb", "one_way_moon"]
        write_csv(FIXTURE_DIR / "orekit_measurements.csv", ["t_receive_tdb_s"] + [f"{n}_m" if "cd" not in n else
                                                                                   f"{n}_mps" for n in names],
                  np.column_stack([T_MEAS] + [cols[n] for n in names]),
                  "WP5.1 Orekit measurements on the shared trajectory and station track.\n"
                  "Range (two-way / one-way, Newtonian, zero delay); cd = (R2(t+30) - R2(t-30)) / 60 s.\n"
                  "moon: state in the Moon-centred ICRF frame; ssb: state in Orekit's ICRF (SSB) frame.")
    (RUN_DIR / "orekit_step.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    _write_provenance(args, report)
    print("Orekit step done")


def _write_provenance(args, report: dict) -> None:
    from scenario import load_predeclaration

    prov_path = FIXTURE_DIR / "provenance.json"
    old = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.is_file() else {}
    kd = kernel_dir()
    inputs = {
        "orekit_data_dir_files": _dir_digest(Path(args.orekit_data)),
        "jpl_DE440_binary": {"name": "linux_p1550p2650.440", "sha256": sha256(Path(args.jpl_dir) / "linux_p1550p2650.440")},
        "jpl_DE421_binary": {"name": "lnxp1900p2053.421", "sha256": sha256(Path(args.jpl_dir) / "lnxp1900p2053.421")},
        "oracle_helpers_jar_sources": {p.name: sha256(p) for p in sorted((Path(__file__).parent / "jhelpers" / "oracle").glob("*.java"))},
        "gl1800f_tab": {"name": gravity_tab().name, "sha256": sha256(gravity_tab())},
    }
    for k in ("de440.bsp", "de421.bsp", "moon_pa_de440_200625.bpc", "moon_de440_220930.txt", "naif0012.tls.txt"):
        inputs[k] = sha256(kd / k)
    dyn = dict(old.get("orekit_dynamics", {}))
    dyn.update(report["dynamics"])
    prov = {
        "work_package": "F5 WP5.1",
        "predeclaration": "studies/phase17/configs/wp5_1_orekit_oracle_predeclaration.json",
        "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "generator": "tools/oracle/generate.py (lunar-od step, then orekit step)",
        "repository": _commit_info(),
        "versions": _versions(),
        "inputs": inputs,
        "alignment": load_predeclaration()["alignment"],
        "orekit_dynamics": dyn,
        "orekit_dynamics_force_models": {
            "central": "NewtonianAttraction(mu = scenario GM Moon) from the CartesianOrbit",
            "third_body": "ThirdBodyAttraction(GmOverrideBody(Earth|Sun, scenario GM)), positions from Orekit's JPL reader",
            "harmonics": "HolmesFeatherstoneAttractionModel(MOON_PA_DE440 via PckEulerProvider, GL1800F n x n, file GM and radius)",
            "srp": "RadiationPressureModel(ConicallyShadowedLightFluxModel(P1AU*AU^2, R_sun, Sun, R_occ), "
                   "IsotropicRadiationSingleCoefficient(A=10 m^2, C_R=K*m/A), m=1000 kg)",
            "integrator": "DormandPrince853, minStep 1e-3 s, per-case maxStep/tolerances in orekit_dynamics",
            "frame": "Moon-centred, ICRF axes (BodyCenteredIcrfProvider on GCRF)",
        },
        "orekit_measurements": report.get("measurements", old.get("orekit_measurements")),
        "preconditions": {**old.get("preconditions", {}), **report["preconditions"]},
    }
    files = {}
    for p in sorted(FIXTURE_DIR.iterdir()):
        if p.name != "provenance.json":
            files[p.name] = sha256(p)
    prov["files_sha256"] = files
    prov_path.write_text(json.dumps(prov, indent=1) + "\n", encoding="utf-8")


def _dir_digest(d: Path) -> dict:
    """SHA-256 over the sorted (relative path, file SHA-256) list of a data directory."""
    import hashlib

    h = hashlib.sha256()
    n = 0
    for p in sorted(x for x in d.rglob("*") if x.is_file()):
        h.update(f"{p.relative_to(d).as_posix()}\0{sha256(p)}\n".encode())
        n += 1
    return {"files": n, "sha256_of_listing": h.hexdigest()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    a = sub.add_parser("lunar-od")
    a.add_argument("--cases", nargs="*", default=list(DYNAMICS_CASES))
    b = sub.add_parser("orekit")
    b.add_argument("--orekit-data", required=True)
    b.add_argument("--jpl-dir", required=True)
    b.add_argument("--jar", default=str(Path(__file__).parent / "jhelpers" / "build" / "oracle-helpers.jar"))
    b.add_argument("--cases", nargs="*", default=list(DYNAMICS_CASES))
    b.add_argument("--no-measurements", dest="measurements", action="store_false")
    args = ap.parse_args(argv)
    if args.step == "lunar-od":
        step_lunar_od(args.cases)
    else:
        step_orekit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
