"""F3 WP3.2 S4: the measured values behind gates LT_G2-LT_G7 and LTB_G3 (the tests assert them; this records them).

Predeclaration: studies/phase17/configs/wp3_2_s4_srp_light_time_predeclaration.json (cc7a454, errata e7e261d
before any result).
Reuses the geometry, oracles and metrics of tests/test_wp3_2_srp_light_time.py and
tests/test_wp3_2_srp_light_time_transport.py, so the recorded numbers are the ones the tests compare with their
thresholds. Also records, as information only: the LT_G5(a) STM control without the package's gradients, the
LT_G3(b) angle after the predicted v_sun correction, WP3.1's ray_crossing_check, the light-time shift of each
lunar cut, and the cost per wrapper evaluation.

usage: python wp3_2_s4_gate_values.py LABEL OUT.json
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
from lunar_od.lunar_j2_model import use_lunar_j2_model  # noqa: E402
from tests import test_wp3_2_srp_light_time as M  # noqa: E402
from tests import test_wp3_2_srp_light_time_transport as B  # noqa: E402

THRESHOLDS = dict(lt_g2_min_count=300, lt_g3a=1e-7, lt_g3a_precondition=4 * float(np.finfo(float).eps),
                  lt_g3a_control_ratio=100.0, lt_g3b_rad=1.5e-7, lt_g3b_precondition_m=1.0,
                  lt_g3b_control_ratio=100.0, lt_g3c_m=0.1, lt_g4_pos=1e-6, lt_g4_full_light=1e-5, lt_g4_vel=1e-6,
                  lt_g5a=1e-7, lt_g5b=1e-6, lt_g5b_ratio=10.0, lt_g6_floor_factor=2.0, lt_g6_floor_add_m=1e-6,
                  lt_g6_phi=1e-8, lt_g6_effect_m=1e-4, lt_g7_shift_s=(0.02, 0.5), lt_g7_earth_cuts=2,
                  lt_g7_time_tol_s=dyn._CONTACT_TIME_TOL_S, ltb_g3_dr=1e-3, ltb_g3_dv=1e-6, ltb_g3_ratio=100.0)


def lt_g2():
    mismatches, counts = M.wp3_1_parity(M.make_samples())
    return dict(mismatches=len(mismatches), counts=counts)


def lt_g3():
    sm = M.make_samples()
    return dict(a=M.lorentz_values(sm), b=M.spice_direction_values(), c=M.occulter_values(sm),
                c_information_wp3_1_ray_crossing_check=M.WD.ray_crossing_check())


def lt_g4():
    return dict(abc=M.gradient_values(M.make_samples()), d=M.composed_gradient_values())


def lt_g5():
    cols = M.make_rk4_columns()
    gated, control = M.stm_values(cols)
    k_gated, k_control = M.k_sensitivity_values()
    return dict(a_stm_gated=gated, a_stm_control_no_lt_gradients_information=control, a_ratio=control / gated,
                b_k_sensitivity=k_gated, b_k_control_no_lt_dfdk=k_control, b_ratio=k_control / k_gated)


def lt_g6():
    out = {}
    for case in M.PATH_CASES:
        v = M.path_values(case)
        on, off = max(v["pos_on"].values()), max(v["pos_off"].values())
        ok = on <= 2.0 * off + 1e-6 and max(v["phi_on"].values()) <= 1e-8 and v["effect_m"] >= 1e-4
        out[case] = dict(v, max_on_m=on, max_off_m=off, on_over_off=on / off if off else None,
                         verdict="PASS" if ok else "FAIL")
    return out


def lt_g7():
    v = M.shadow_event_values()
    return dict(n_cuts_lt=len(v["cuts_lt"]), n_cuts_geo=len(v["cuts_geo"]), max_offset_s=max(v["offsets"]),
                min_distance_to_geometric_cut_s=min(v["nearest_geo"]), shifts_s=v["shifts"],
                control_message=v["control_message"], n_cuts_with_earth_shadow=len(v["cuts_es"]),
                labels_with_earth_shadow=v["labels_es"],
                n_earth_cuts=sum(lab == "earth" for lab, _ in v["labels_es"]))


def ltb_g3():
    case = B.make_light_time_truth_case()
    out = {}
    for solver in ("bls_lm", "srif"):
        dr, dv = B.light_time_recovery(solver, case, B.LT)
        out[solver] = dict(dr0_m=dr, dv0_mps=dv)
    dr_free, dv_free = B.light_time_recovery("bls_lm", case, None)
    out["bls_lm"].update(dr0_without_lt_m=dr_free, dv0_without_lt_mps=dv_free, ratio=dr_free / out["bls_lm"]["dr0_m"])
    return out


def cost():
    rng = np.random.default_rng(3)
    rhs6 = dyn._with_srp_light_time_acceleration(lambda t, y: np.zeros(6), M.earth, M.sun, M.PEN, M.LT, None,
                                                 state_size=6)
    rhs48 = dyn._with_srp_light_time_acceleration(lambda t, y: np.zeros(48), M.earth, M.sun, M.PEN, M.LT, None,
                                                  state_size=48)
    y = rng.normal(size=48)
    y[:6] = M.PLANE
    out = {}
    for name, rhs in (("value_6", rhs6), ("value_and_gradients_48", rhs48)):
        t0 = time.perf_counter()
        for k in range(2000):
            rhs(600.0 + k, y[:6] if name == "value_6" else y)
        out[name + "_us"] = (time.perf_counter() - t0) / 2000 * 1e6
    return out


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc}"


def main(label, out_path):
    payload = dict(study="F3 WP3.2 S4 gate values", label=label, commit=_git("rev-parse", "HEAD"),
                   src_status=_git("status", "--porcelain", "--", "src"), python=platform.python_version(),
                   numpy=np.__version__, thresholds=THRESHOLDS)
    with use_lunar_j2_model("iau_mean_pole_legacy"):
        for name, fn in (("LT_G2", lt_g2), ("LT_G3", lt_g3), ("LT_G4", lt_g4), ("LT_G5", lt_g5), ("LT_G6", lt_g6),
                         ("LT_G7", lt_g7), ("LTB_G3", ltb_g3), ("cost", cost)):
            t0 = time.perf_counter()
            payload[name] = fn()
            print(f"{name}: {time.perf_counter() - t0:.0f} s", flush=True)
            Path(out_path).write_text(json.dumps(payload, indent=1, sort_keys=True, default=float) + "\n",
                                      encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
