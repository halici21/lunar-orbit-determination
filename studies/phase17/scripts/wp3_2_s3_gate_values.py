"""F3 WP3.2 S3: the measured values behind gates E_G2-E_G7 and EB_G3 (the tests assert them; this records them).

Predeclaration: studies/phase17/configs/wp3_2_s3_earth_shadow_predeclaration.json (a5954f1, errata 6ebbed7 and
15468af before any result).
Reuses the geometry, oracles and metrics of tests/test_wp3_2_earth_shadow.py and
tests/test_wp3_2_earth_shadow_transport.py, so the recorded numbers are the ones the tests compare with their
thresholds. Also records, as information only: the E_G7 D2 position error at rtol 1e-11 with and without the
Earth contacts in the cut list, the size of the Earth-shadow effect, and the cost per wrapper evaluation.

usage: python wp3_2_s3_gate_values.py LABEL OUT.json
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
from tests import test_wp3_2_earth_shadow as M  # noqa: E402
from tests import test_wp3_2_earth_shadow_transport as B  # noqa: E402

THRESHOLDS = dict(e_g3a=2.5e-4, e_g3a_precondition=1e-9, e_g3b_s=0.01, e_g4=1e-6, e_g5a=1e-7, e_g5_ratio=10.0,
                  e_g5b=1e-6, e_g6_pos=1e-5, e_g6_phi=1e-8, e_g6_effect=1.0, e_g7_earth_cuts=4,
                  e_g7_time_tol_s=dyn._CONTACT_TIME_TOL_S, eb_g3_dr=1e-3, eb_g3_dv=1e-6, eb_g3_ratio=100.0)


def e_g2():
    mismatches, counts = M.wp3_1_parity(M.make_samples())
    return dict(mismatches=len(mismatches), counts=counts)


def e_g3():
    a = M.accuracy_errors(M.make_samples())
    b = {name: M.spice_contact_comparison(name) for name in ("E2", "E3")}
    return dict(a=a, b={k: dict(worst_s=v["worst_s"], counts_equal=v["counts_equal"],
                               n_any=len(v["roots"]["ANY"]), n_full=len(v["roots"]["FULL"]),
                               roots=v["roots"], spice=v["spice"]) for k, v in b.items()})


def e_g4():
    return M.gradient_errors(M.make_samples())


def e_g5():
    cols = M.make_rk4_columns()
    gated, control = M.stm_values(cols)
    k_gated, k_control = M.k_sensitivity_values()
    return dict(a_stm_gated=gated, a_stm_control_no_es_gradient=control, a_ratio=control / gated,
                b_k_sensitivity=k_gated, b_k_control_no_es_dfdk=k_control, b_ratio=k_control / k_gated)


def e_g6():
    out = {}
    for case in M.PATH_CASES:
        pos, phi, effect = M.path_differences(case)
        out[case] = dict(position_m=pos, phi_rel=phi, earth_shadow_effect_m=effect,
                         verdict="PASS" if max(pos.values()) <= 1e-5 and max(phi.values()) <= 1e-8 and effect > 1.0
                         else "FAIL")
    return out


def e_g7():
    v = M.shadow_event_values()
    labels = [lab for lab, _ in v["labels_es"]]
    info = e_g7_floor_information()
    return dict(n_cuts_with_es=len(v["cuts_es"]), n_cuts_without_es=len(v["cuts_free"]),
                n_earth_cuts=sum(lab.startswith("earth") for lab in labels),
                labels_with_es=v["labels_es"], max_offset_s=max(off for _, off in v["labels_es"]),
                earth_in_free_list=sum(lab.startswith("earth") for lab, _ in v["labels_free"]),
                max_lunar_cut_shift_s=v["max_lunar_shift"], control_message=v["control_message"],
                earth_only=dict(cuts=list(map(float, v["cuts_only_earth"])), labels=v["labels_only_earth"]),
                information=info)


def e_g7_floor_information():
    """Information: D2 at rtol 1e-11 against rtol 1e-13 on the same cuts, with and without the Earth contacts."""
    x0 = M.PLANE
    cuts_es = dyn.shadow_contact_times(M.T6H, x0, *M.BASE, srp=M.PEN, earth_shadow=M.ES, **M.PRE)
    cuts_lunar = dyn.shadow_contact_times(M.T6H, x0, *M.BASE, srp=M.PEN, **M.PRE)
    rhs, sun_lookup = dyn._build_state_rhs(M.T6H, *M.BASE, M.J2, 0.0, "indirect", None, None, None, M.PEN,
                                           dyn._legacy_j2_rotation, dyn.MOON_R_M, earth_shadow=M.ES)
    out = {}
    for name, cuts in (("earth_and_lunar_cuts", cuts_es), ("lunar_cuts_only_wp3_1", cuts_lunar)):
        runs = {}
        for rtol, atol in ((1e-11, 1e-12), (1e-13, 1e-14)):
            runs[rtol] = dyn._integrate_segmented(M.T6H, x0, rhs, rtol, atol, "DOP853", list(cuts), name)
        out[name] = float(np.max(np.abs(runs[1e-11][:, :3] - runs[1e-13][:, :3])))
    return out


def eb_g3():
    case = B.make_earth_shadow_truth_case()
    out = {}
    for solver in ("bls_lm", "srif"):
        dr, dv = B.earth_shadow_recovery(solver, case, B.ES)
        dr_free, dv_free = B.earth_shadow_recovery(solver, case, None)
        out[solver] = dict(dr0_m=dr, dv0_mps=dv, dr0_without_es_m=dr_free, dv0_without_es_mps=dv_free,
                           ratio=dr_free / dr)
    return out


def cost():
    rng = np.random.default_rng(3)
    rhs6 = dyn._with_earth_shadow_acceleration(lambda t, y: np.zeros(6), M.earth, M.sun, M.PEN, M.ES, state_size=6)
    rhs48 = dyn._with_earth_shadow_acceleration(lambda t, y: np.zeros(48), M.earth, M.sun, M.PEN, M.ES, state_size=48)
    y = rng.normal(size=48)
    y[:3] = M.TERM[:3]
    out = {}
    for name, rhs in (("value_6", rhs6), ("value_and_gradient_48", rhs48)):
        t0 = time.perf_counter()
        for k in range(2000):
            rhs(3600.0 + k, y[:6] if name == "value_6" else y)          # the Earth's penumbra at the Moon
        out[name + "_us"] = (time.perf_counter() - t0) / 2000 * 1e6
    return out


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc}"


def main(label, out_path):
    payload = dict(study="F3 WP3.2 S3 gate values", label=label, commit=_git("rev-parse", "HEAD"),
                   src_status=_git("status", "--porcelain", "--", "src"), python=platform.python_version(),
                   numpy=np.__version__, thresholds=THRESHOLDS)
    with use_lunar_j2_model("iau_mean_pole_legacy"):
        for name, fn in (("E_G2", e_g2), ("E_G3", e_g3), ("E_G4", e_g4), ("E_G5", e_g5), ("E_G6", e_g6),
                         ("E_G7", e_g7), ("EB_G3", eb_g3), ("cost", cost)):
            t0 = time.perf_counter()
            payload[name] = fn()
            print(f"{name}: {time.perf_counter() - t0:.0f} s", flush=True)
            Path(out_path).write_text(json.dumps(payload, indent=1, sort_keys=True, default=float) + "\n",
                                      encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
