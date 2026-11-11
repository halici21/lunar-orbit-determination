"""F3 WP3.2 S2: the measured values behind gates L_G2-L_G7 (the tests assert them; this records them).

Predeclaration: studies/phase17/configs/wp3_2_s2_lunar_radiation_predeclaration.json (da5a4a0).
Reuses the geometry, oracles and metrics of tests/test_wp3_2_lunar_radiation.py, so the recorded numbers are
the ones the tests compare with their thresholds. Also records, as information only: the integration floor of
the 6-state propagation with and without LRP, and the kernel's cost per call.

usage: python wp3_2_s2_gate_values.py LABEL OUT.json
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for extra in (ROOT, ROOT / "studies" / "phase17" / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import lunar_od.dynamics as dyn  # noqa: E402
import lunar_od.lunar_radiation as lr  # noqa: E402
from lunar_od.constants import J2_MOON_UNNORMALIZED  # noqa: E402
from tests import test_wp3_2_lunar_radiation as M  # noqa: E402

THRESHOLDS = dict(l_g2=1e-12, l_g3_closed=1e-10, l_g3_transverse=1e-12, l_g3_c=3e-4, l_g4_cs=1e-10,
                  l_g4_ref=1e-12, l_g4_py=1e-12, l_g5a=1e-7, l_g5a_ratio=10.0, l_g5b=1e-6, l_g6_pos=1e-5,
                  l_g6_phi=1e-8, l_g6_effect=0.1)


def l_g2():
    worst, worst_py = M.c6_parity(M.make_samples())
    return dict(c6_parity=worst, python_fallback_parity=worst_py)


def l_g3():
    closed = {f"{int(a / 1e3)}km": M.closed_form_errors(a) for a in M.ALTITUDES_AB}
    errors = M.accuracy_errors()
    return dict(closed_forms=closed,
                adaptive=[dict(altitude_km=a / 1e3, sun_nadir_deg=b, rel_error=e) for a, b, e in errors],
                adaptive_worst=max(e for _, _, e in errors))


def l_g4():
    cs, ref, py = M.gradient_errors(M.make_samples())
    return dict(complex_step=cs, reference_vs_c6=ref, python_fallback=py)


def l_g5():
    cols = M.make_rk4_columns()
    gated = M.G._stm_err(cols, M.G._phi(M.gated_stm()))
    control = M.G._stm_err(cols, M.G._phi(M.no_lrp_gradient_replica()))
    return dict(a_stm_gated=gated, a_stm_control_no_lrp_gradient=control, a_ratio=control / gated,
                b_k_sensitivity=M.k_sensitivity_error())


def l_g6():
    out = {}
    for harmonics in (False, True):
        for tide in (False, True):
            pos, phi, effect = M.path_differences(harmonics, tide)
            out[f"harmonics_{harmonics}_tide_{tide}"] = dict(position_vs_6_state_m=pos, phi_vs_48_relative=phi,
                                                             lrp_effect_6h_m=effect)
    return out


def l_g7():
    cuts_lrp, cuts_free, message = M.shadow_event_values()
    same = cuts_lrp.size == cuts_free.size
    shift = np.abs(cuts_lrp - cuts_free) if same else np.array([np.nan])
    return dict(n_cuts=int(cuts_lrp.size), n_cuts_free=int(cuts_free.size), max_cut_shift_s=float(np.max(shift)),
                contact_time_tol_s=float(dyn._CONTACT_TIME_TOL_S),
                cuts_beyond_tolerance=int(np.sum(shift > dyn._CONTACT_TIME_TOL_S)), control_message=message)


def integration_floor():
    """Information: 6-state DOP853, rtol 1e-12 and 3e-14 against 1e-13, with and without LRP (6 h, J2)."""
    out = {}
    for name, lrp in (("without_lrp", None), ("with_lrp", M.LRP)):
        runs = {}
        for rtol in (1e-12, 1e-13, 3e-14):
            runs[rtol] = dyn.propagate_state(M.T6H, M.X0, *M.BASE, j2_moon=J2_MOON_UNNORMALIZED, srp=M.NOS,
                                             lunar_radiation=lrp, method="DOP853", rtol=rtol, atol=0.1 * rtol)
        out[name] = {f"rtol_{rtol:g}_vs_1e-13_m": float(np.max(np.linalg.norm(
            runs[rtol][:, :3] - runs[1e-13][:, :3], axis=1))) for rtol in (1e-12, 3e-14)}
    return out


def cost():
    r = np.array([1.8374e6, 1e3, 2e3])
    s = np.array([1.5e11, 1e9, 0.0])
    out = {}
    for name, fn in (("value_us", lr.lunar_radiation_acceleration_per_k),
                     ("value_and_gradient_us", lr.lunar_radiation_acceleration_and_gradient_per_k)):
        fn(r, s, M.LRP)
        t0 = time.perf_counter()
        for _ in range(2000):
            fn(r, s, M.LRP)
        out[name] = (time.perf_counter() - t0) / 2000 * 1e6
    return out


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc}"


def verdicts(result):
    t = result["thresholds"]
    g3 = result["L_G3"]
    closed_ok = all(e["night"] <= t["l_g3_closed"] and e["subsolar_radial"] <= t["l_g3_closed"]
                    and e["subsolar_transverse"] <= t["l_g3_transverse"]
                    and e["night_reference"] <= t["l_g3_closed"] and e["subsolar_reference"] <= t["l_g3_closed"]
                    for e in g3["closed_forms"].values())
    g4, g5, g7 = result["L_G4"], result["L_G5"], result["L_G7"]
    g6 = {name: ("PASS" if max(c["position_vs_6_state_m"].values()) <= t["l_g6_pos"]
                 and max(c["phi_vs_48_relative"].values()) <= t["l_g6_phi"]
                 and c["lrp_effect_6h_m"] > t["l_g6_effect"] else "FAIL")
          for name, c in result["L_G6"].items()}
    return {
        "L_G2": "PASS" if max(result["L_G2"].values()) <= t["l_g2"] else "FAIL",
        "L_G3": "PASS" if closed_ok and g3["adaptive_worst"] <= t["l_g3_c"] else "FAIL",
        "L_G4": "PASS" if (g4["complex_step"] <= t["l_g4_cs"] and g4["reference_vs_c6"] <= t["l_g4_ref"]
                           and g4["python_fallback"] <= t["l_g4_py"]) else "FAIL",
        "L_G5": "PASS" if (g5["a_stm_gated"] <= t["l_g5a"] and g5["a_ratio"] >= t["l_g5a_ratio"]
                           and g5["b_k_sensitivity"] <= t["l_g5b"]) else "FAIL",
        "L_G6": "PASS" if all(v == "PASS" for v in g6.values()) else "FAIL",
        "L_G6_cases": g6,
        "L_G7": "PASS" if (g7["n_cuts"] == g7["n_cuts_free"] > 20 and g7["cuts_beyond_tolerance"] > 0
                           and g7["control_message"] and "Shadow contact inconsistent" in g7["control_message"])
        else "FAIL",
    }


def main(label, path):
    import numba
    import scipy
    result = dict(predeclaration="studies/phase17/configs/wp3_2_s2_lunar_radiation_predeclaration.json (da5a4a0)",
                  label=label, commit=_git("rev-parse", "HEAD"), src_status=_git("status", "--porcelain", "--", "src"),
                  python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                  numba=numba.__version__, kernel_compiled=hasattr(lr._kernel, "py_func"))
    for name, fn in (("L_G2", l_g2), ("L_G3", l_g3), ("L_G4", l_g4), ("L_G5", l_g5), ("L_G6", l_g6),
                     ("L_G7", l_g7), ("integration_floor", integration_floor), ("cost", cost)):
        t0 = time.perf_counter()
        result[name] = fn()
        print(name, json.dumps(result[name])[:600], f"({time.perf_counter() - t0:.1f} s)", flush=True)
    result["thresholds"] = THRESHOLDS
    result["verdicts"] = verdicts(result)
    print(json.dumps(result["verdicts"]))
    Path(path).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
