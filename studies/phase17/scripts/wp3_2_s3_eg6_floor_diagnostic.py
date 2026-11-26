"""F3 WP3.2 S3: exploratory gate set E_G6-A (written after the E_G6 FAIL, before it runs; amendment log 2026-11-26).

E_G6 as predeclared FAILs: the 6- vs 42/48-state positions differ by 1.15e-5 m (i), 1.00e-5 m (iv) and 1.28e-5 m
(v, after the post-result epoch fix) against 1e-5 m; the Earth-shadow effect is 0.56 m in (i)-(iii) against
> 1 m. Hypothesis: the position differences are the DOP853 integration floor of the three paths at rtol 1e-13,
present without the Earth shadow too (as S2's L_G6); the effect is small because the TERM orbit at K = 0.05
spends only part of the 6 h arc in the Earth's penumbra (the predeclared expectation of 10-40 m was wrong).

E_G6-A, the five predeclared cases with the same settings:
- A1: in every case, max position difference with the Earth shadow on <= 2 x the same quantity with it off
  (earth_shadow=None, everything else unchanged), and Phi <= 1e-8 with it on;
- A2: case (v) completes on the default grail_pa_de440 J2 (the epoch fix) with effect > 1 m.
The predeclared E_G6 result stays FAIL.

usage: python wp3_2_s3_eg6_floor_diagnostic.py OUT.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for extra in (ROOT, ROOT / "studies" / "phase17" / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import lunar_od.dynamics as dyn  # noqa: E402
from lunar_od.lunar_j2_model import use_lunar_j2_model  # noqa: E402
from tests import test_wp3_2_earth_shadow as M  # noqa: E402


def differences(case, earth_shadow):
    t, x0, base, kw6, kw_var, _ = M._path_setup(case)
    kw6, kw_var = dict(kw6, earth_shadow=earth_shadow, **M.TIGHT), dict(kw_var, earth_shadow=earth_shadow, **M.TIGHT)
    p6 = dyn.propagate_state(t, x0, *base, **kw6)
    runs = {"42": dyn.propagate_augmented_state(t, M.G._aug0(x0), *base, **kw_var),
            "48": dyn.propagate_state_with_k_sensitivity(t, x0, *base, **kw_var)}
    scale = np.max(np.abs(runs["48"][:, 6:42]))
    return (max(float(np.max(np.abs(r[:, 0:3] - p6[:, 0:3]))) for r in runs.values()),
            max(float(np.max(np.abs(r[:, 6:42] - runs["48"][:, 6:42])) / scale) for r in runs.values()))


def run():
    out = {}
    for case in M.PATH_CASES:
        if case == "v_e2_grail_events":
            on, off = differences(case, M.ES), differences(case, None)
        else:
            with use_lunar_j2_model("iau_mean_pole_legacy"):
                on, off = differences(case, M.ES), differences(case, None)
        _, _, effect = (M.path_differences(case) if case == "v_e2_grail_events" else _legacy(M.path_differences, case))
        out[case] = dict(position_on_m=on[0], position_off_m=off[0], ratio=on[0] / off[0], phi_on=on[1],
                         effect_m=effect, A1_pass=on[0] <= 2.0 * off[0] and on[1] <= 1e-8)
    out["A1_pass"] = all(v["A1_pass"] for v in out.values() if isinstance(v, dict))
    out["A2_pass"] = out["v_e2_grail_events"]["effect_m"] > 1.0
    return out


def _legacy(fn, case):
    with use_lunar_j2_model("iau_mean_pole_legacy"):
        return fn(case)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--", "src", "tests"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    result = dict(study="F3 WP3.2 S3 E_G6-A (exploratory)", commit=commit, status=status, **run())
    print(json.dumps(result, indent=1, default=float))
    Path(sys.argv[1]).write_text(json.dumps(result, indent=1, sort_keys=True, default=float) + "\n", encoding="utf-8")
