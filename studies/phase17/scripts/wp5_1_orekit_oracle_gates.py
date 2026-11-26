"""F5 WP5.1: evaluate every predeclared gate against the pinned Orekit fixtures.

    python studies/phase17/scripts/wp5_1_orekit_oracle_gates.py [--trial-dir <agent-sandbox/orekit-gmat-oracle/out>]

Writes ``studies/phase17/results/wp5_1_orekit_oracle.json``. No Orekit is needed; the
fixtures come from ``tools/oracle/generate.py``. ``--trial-dir`` adds the information item
I_TRIAL (difference to the 2026-10-21 trial's stored Orekit ref3 trajectories).
Predeclaration: ``studies/phase17/configs/wp5_1_orekit_oracle_predeclaration.json``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "oracle"))

import compare as C  # noqa: E402
from scenario import DYNAMICS_CASES  # noqa: E402

OUT = ROOT / "studies" / "phase17" / "results" / "wp5_1_orekit_oracle.json"


def _git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial-dir", type=Path)
    ap.add_argument("--cases", nargs="*", default=list(DYNAMICS_CASES))
    a = ap.parse_args()
    C.check_thresholds_match_predeclaration()
    prov = C.provenance()
    res = {
        "work_package": "F5 WP5.1",
        "predeclaration": "studies/phase17/configs/wp5_1_orekit_oracle_predeclaration.json",
        "run": {"date": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                "commit": _git("rev-parse", "HEAD"), "dirty_paths": _git("status", "--porcelain").splitlines()},
        "thresholds": C.THRESHOLDS,
        "O_F_fixture_hash_mismatches": C.fixture_hash_mismatches(),
        "O_P_preconditions": prov["preconditions"],
        "settings": {"lunar_od": {c: C.lunar_od_settings()["cases"][c] for c in a.cases},
                     "orekit": {c: prov["orekit_dynamics"][c] for c in a.cases}},
        "dynamics": {},
    }
    for case in a.cases:
        t0 = time.perf_counter()
        d = C.dynamics_case(case, info=True)
        d["runtime_s"] = round(time.perf_counter() - t0, 1)
        d["O_D_pass"] = d["O_D_max_dr_m"] < C.THRESHOLDS["O_D_m"]
        d["O_L_pass"] = d["O_L_max_dr_m"] <= C.THRESHOLDS["O_L_m"]
        res["dynamics"][case] = d
        print(case, {k: v for k, v in d.items() if k.startswith(("O_", "I_TIME_tdb"))}, flush=True)
    m = C.measurement_gates()
    m["O_M2_pass"] = {k: m[k]["max_abs"] < C.THRESHOLDS["O_M2_m"] for k in ("two_way_moon", "two_way_ssb")}
    m["O_M3_pass"] = {k: m[k]["max_abs"] < C.THRESHOLDS["O_M3_mps"] for k in ("cd_moon", "cd_ssb")}
    m["L_M3_pass"] = m["L_M3_max_abs_mps"] <= C.THRESHOLDS["L_M3_mps"]
    res["measurements"] = m
    print(json.dumps(m, indent=1), flush=True)
    res["I_STATION"] = C.station_information()
    if a.trial_dir:
        trial = {}
        for case in a.cases:
            p = a.trial_dir / "orekit" / f"{case}_ref3.npz"
            if p.is_file():
                z = np.load(p)
                trial[case] = float(np.max(np.linalg.norm(z["x"][:, :3] - C.orekit_trajectory(case)[:, 1:4], axis=1)))
        res["I_TRIAL_orekit_now_minus_trial_ref3_max_dr_m"] = trial
    import importlib.metadata as md
    import platform

    res["run"]["environment"] = {"python": platform.python_version(),
                                 **{p: md.version(p) for p in ("numpy", "scipy", "numba", "spiceypy")}}
    OUT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print("wrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
