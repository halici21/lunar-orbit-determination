"""WP5.1 gate evaluation from the pinned Orekit fixtures (no Orekit import).

Shared by ``studies/phase17/scripts/wp5_1_orekit_oracle_gates.py`` (writes the results file)
and ``tests/test_external_oracle.py`` (asserts the predeclared thresholds).
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

import lunar_od_side as L
from scenario import FIXTURE_DIR, STATION_GRID, T_MEAS, TC_S, TRAJ_GRID, load_predeclaration, read_csv, sha256

THRESHOLDS = {  # parsed from the predeclaration text so the test cannot loosen them alone
    "O_D_m": 1e-3, "O_L_m": 1e-4, "O_M2_m": 1e-3, "O_M3_mps": 1e-6, "L_M3_mps": 1e-8,
}


def check_thresholds_match_predeclaration() -> None:
    g = load_predeclaration()["gates"]
    expected = {"O_D_DYNAMICS": "< 1 mm", "O_L_LINK": "<= 0.1 mm", "O_M2": "< 1 mm", "O_M3": "< 1 um/s",
                "L_M3": "<= 1e-8 m/s"}
    for key, text in expected.items():
        if text not in g[key]:
            raise AssertionError(f"predeclaration gate {key} no longer states '{text}'")


@lru_cache(maxsize=None)
def scenario() -> dict:
    return json.loads((FIXTURE_DIR / "scenario.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def provenance() -> dict:
    return json.loads((FIXTURE_DIR / "provenance.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def lunar_od_settings() -> dict:
    return json.loads((FIXTURE_DIR / "lunar_od_settings.json").read_text(encoding="utf-8"))


def fixture_hash_mismatches() -> list[str]:
    want = provenance()["files_sha256"]
    bad = [n for n, h in want.items() if not (FIXTURE_DIR / n).is_file() or sha256(FIXTURE_DIR / n) != h]
    extra = [p.name for p in FIXTURE_DIR.iterdir() if p.name not in want and p.name != "provenance.json"]
    return bad + [f"unlisted:{n}" for n in extra]


def orekit_trajectory(case: str) -> np.ndarray:
    return read_csv(FIXTURE_DIR / f"orekit_dynamics_{case}.csv")[1]


def orekit_time_map() -> tuple[np.ndarray, np.ndarray]:
    data = read_csv(FIXTURE_DIR / "orekit_time_map.csv")[1]
    return data[:, 0], data[:, 1]


def setting(case: str) -> L.Setting:
    return L.Setting.from_dict(lunar_od_settings()["cases"][case]["setting"])


def _rtn(r: np.ndarray, v: np.ndarray, d: np.ndarray) -> np.ndarray:
    rh = r / np.linalg.norm(r)
    nh = np.cross(r, v)
    nh /= np.linalg.norm(nh)
    return np.array([d @ rh, d @ np.cross(nh, rh), d @ nh])


def dynamics_case(case: str, *, link: bool = True, info: bool = False) -> dict:
    """O_D and (optionally) O_L for one case; distances in metres.

    ``info`` adds the D2 DIRECT-rotation production run and the adapter with SPICE's TDB-TT model
    (the size of the precondition O_P(d) effect); neither is gated.
    """
    sc = scenario()
    ref = orekit_trajectory(case)
    s = setting(case)
    t, x_tt, n_ev = L.propagate_aligned(case, sc, time_arg="TT_OREKIT", setting=s, time_map=orekit_time_map())
    if not np.array_equal(t, ref[:, 0]):
        raise AssertionError("output grids differ")
    dr = np.linalg.norm(x_tt[:, :3] - ref[:, 1:4], axis=1)
    dv = np.linalg.norm(x_tt[:, 3:] - ref[:, 4:7], axis=1)
    out = {"setting": s.as_dict(), "shadow_events": n_ev,
           "O_D_max_dr_m": float(dr.max()), "dr_end_m": float(dr[-1]), "O_D_max_dv_mps": float(dv.max()),
           "rtn_end_m": _rtn(ref[-1, 1:4], ref[-1, 4:7], x_tt[-1, :3] - ref[-1, 1:4]).tolist()}
    if link:
        _, x_tdb, _ = L.propagate_aligned(case, sc, time_arg="TDB", setting=s)
        out["I_TIME_tdb_minus_tt_end_m"] = float(np.linalg.norm(x_tdb[-1, :3] - x_tt[-1, :3]))
        out["I_TIME_rtn_end_m"] = _rtn(x_tt[-1, :3], x_tt[-1, 3:], x_tdb[-1, :3] - x_tt[-1, :3]).tolist()
        link_setting = s if not np.isfinite(s.max_step) else L.Setting(s.rtol, s.atol)
        x_prod = L.propagate_production(case, sc, setting=link_setting)
        out["O_L_setting"] = link_setting.as_dict()
        out["O_L_max_dr_m"] = float(np.max(np.linalg.norm(x_prod[:, :3] - x_tdb[:, :3], axis=1)))
        if info and case.startswith("D2"):
            x_dir = L.propagate_production(case, sc, setting=link_setting, rotation="direct")
            out["I_LINK_direct_rotation_max_dr_m"] = float(np.max(np.linalg.norm(x_dir[:, :3] - x_tdb[:, :3], axis=1)))
    if info:
        _, x_sp, _ = L.propagate_aligned(case, sc, time_arg="TT", setting=s)
        out["I_TIMEMAP_spice_tt_minus_orekit_tt_max_dr_m"] = float(np.max(np.linalg.norm(x_sp[:, :3] - x_tt[:, :3], axis=1)))
        out["I_TIMEMAP_O_D_with_spice_tt_max_dr_m"] = float(np.max(np.linalg.norm(x_sp[:, :3] - ref[:, 1:4], axis=1)))
        if not case.startswith("D2"):  # the reference kernel's Python harmonics are too slow for D2
            _, x_ref, _ = L.propagate_aligned(case, sc, time_arg="TT_OREKIT", setting=s, time_map=orekit_time_map(),
                                              kernel="reference")
            out["I_KERNEL_f3body_moon_minus_production_max_dr_m"] = float(
                np.max(np.linalg.norm(x_ref[:, :3] - x_tt[:, :3], axis=1)))
    return out


def shared_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, traj = read_csv(FIXTURE_DIR / "shared_trajectory_d1a.csv")
    _, st = read_csv(FIXTURE_DIR / "orekit_station_track.csv")
    if not (np.array_equal(traj[:, 0], TRAJ_GRID) and np.array_equal(st[:, 0], STATION_GRID)):
        raise AssertionError("pinned grids differ from scenario.py")
    return traj[:, 0], traj[:, 1:], st[:, 0], st[:, 1:]


def orekit_measurements() -> dict:
    header, data = read_csv(FIXTURE_DIR / "orekit_measurements.csv")
    if not np.array_equal(data[:, 0], T_MEAS):
        raise AssertionError("measurement epochs differ from scenario.py")
    return {h.rsplit("_", 1)[0]: data[:, i] for i, h in enumerate(header) if i}


def measurement_gates() -> dict:
    tg, traj, sg, st = shared_inputs()
    lod = L.measurements(scenario(), tg, traj, sg, st, T_MEAS, TC_S)
    ore = orekit_measurements()
    out = {}
    for k in ("two_way_moon", "cd_moon", "two_way_ssb", "cd_ssb", "one_way_moon"):
        d = lod[k] - ore[k]
        out[k] = {"max_abs": float(np.abs(d).max()), "rms": float(np.sqrt(np.mean(d ** 2))), "mean": float(d.mean())}
    out["L_M3_max_abs_mps"] = float(np.abs(lod["cd_ssb_production"] - lod["cd_ssb"]).max())
    return out


def station_information() -> dict:
    """I_STATION: lunar_od's own station chain minus the pinned Orekit track."""
    _, _, sg, st = shared_inputs()
    own = L.own_station_track(scenario(), sg)
    d = np.linalg.norm(own[:, :3] - st[:, :3], axis=1)
    return {"max_m": float(d.max()), "min_m": float(d.min())}
