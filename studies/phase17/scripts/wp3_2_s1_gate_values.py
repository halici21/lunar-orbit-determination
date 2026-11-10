"""F3 WP3.2 S1: the measured values behind gates T_G2-T_G7 (the tests assert them; this records them).

Predeclaration: studies/phase17/configs/wp3_2_s1_lunar_tide_predeclaration.json (bdafb53).
Reuses the geometry, oracles and metrics of tests/test_wp3_2_lunar_tide.py, so the recorded numbers
are the ones the tests compare with their thresholds (B_G3 from tests/test_wp3_2_lunar_tide_transport.py).
Also records, as information only, the
tide's trajectory effect and the wrapper's cost per RHS call.

usage: python wp3_2_s1_gate_values.py OUT.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for extra in (ROOT, ROOT / "studies" / "phase17" / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import lunar_od.dynamics as dyn  # noqa: E402
import lunar_od.lunar_tide as lt  # noqa: E402
from lunar_od.constants import J2_MOON_UNNORMALIZED, MU_EARTH_M3S2, MU_SUN_M3S2  # noqa: E402
from tests import test_wp3_2_lunar_tide as M  # noqa: E402


def t_g2_g3_g4():
    import wp3_1_force_terms as W
    from lunar_od.gravity_harmonics import spherical_harmonic_acceleration, spherical_harmonic_gravity_gradient
    r, d_e, d_s, rot = M.make_samples()
    worst = dict(g2_acc=0.0, g2_grad=0.0, g2_sum=0.0, g3_study=0.0, g4_complex_step=0.0, g4_trace=0.0)
    symmetric = True
    for i in range(r.shape[0]):
        d = {"EARTH": d_e[i], "SUN": d_s[i]}
        a_sum, g_sum, sa_max, sg_max = np.zeros(3), np.zeros((3, 3)), 0.0, 0.0
        for body, gm, n, k in M._terms():
            model = M._iers_model(rot[i] @ d[body], gm, n, k, M.TIDE.radius_m)
            a_sh = spherical_harmonic_acceleration(r[i], model, rot[i])
            g_sh = spherical_harmonic_gravity_gradient(r[i], model, rot[i])
            rn = np.linalg.norm(r[i])
            amp = k * gm * M.TIDE.radius_m ** (2 * n + 1) / np.linalg.norm(d[body]) ** (n + 1)
            sa, sg = amp * rn ** -(n + 2), amp * rn ** -(n + 3)
            a_cf = lt.tide_degree_acceleration(r[i], d[body], gm, n, k, M.TIDE.radius_m)
            g_cf = lt.tide_degree_gravity_gradient(r[i], d[body], gm, n, k, M.TIDE.radius_m)
            worst["g2_acc"] = max(worst["g2_acc"], float(np.linalg.norm(a_cf - a_sh) / sa))
            worst["g2_grad"] = max(worst["g2_grad"], float(np.linalg.norm(g_cf - g_sh) / sg))
            a_sum, g_sum, sa_max, sg_max = a_sum + a_sh, g_sum + g_sh, max(sa_max, sa), max(sg_max, sg)
            f = (lambda x, b=body, gm=gm, n=n, k=k: lt.tide_degree_acceleration(x, d[b], gm, n, k, M.TIDE.radius_m))
            cs = M._complex_step(f, r[i])
            worst["g4_complex_step"] = max(worst["g4_complex_step"], float(np.linalg.norm(g_cf - cs) / np.linalg.norm(cs)))
            worst["g4_trace"] = max(worst["g4_trace"], float(abs(np.trace(g_cf)) / np.linalg.norm(g_cf)))
            symmetric &= bool(np.array_equal(g_cf, g_cf.T))
        a_tot = lt.tide_acceleration(r[i], d_e[i], d_s[i], MU_EARTH_M3S2, MU_SUN_M3S2, M.TIDE)
        g_tot = lt.tide_gravity_gradient(r[i], d_e[i], d_s[i], MU_EARTH_M3S2, MU_SUN_M3S2, M.TIDE)
        worst["g2_sum"] = max(worst["g2_sum"], float(np.linalg.norm(a_tot - a_sum) / sa_max),
                              float(np.linalg.norm(g_tot - g_sum) / sg_max))
        study = W.tide_acceleration(r[i], d_e[i], d_s[i], MU_EARTH_M3S2, MU_SUN_M3S2)
        worst["g3_study"] = max(worst["g3_study"], float(np.linalg.norm(a_tot - study) / np.linalg.norm(study)))
        cs = M._complex_step(lambda x: lt.tide_acceleration(x, d_e[i], d_s[i], MU_EARTH_M3S2, MU_SUN_M3S2, M.TIDE),
                             r[i])
        worst["g4_complex_step"] = max(worst["g4_complex_step"], float(np.linalg.norm(g_tot - cs) / np.linalg.norm(cs)))
        worst["g4_trace"] = max(worst["g4_trace"], float(abs(np.trace(g_tot)) / np.linalg.norm(g_tot)))
        symmetric &= bool(np.array_equal(g_tot, g_tot.T))
    worst["g4_symmetric_exactly"] = symmetric
    return worst


def t_g5():
    cols = M.make_rk4_columns()
    hist = dyn.propagate_augmented_state(M.T6H, M._aug0(M.X0), *M.BASE, j2_moon=J2_MOON_UNNORMALIZED,
                                         lunar_tide=M.TIDE, **M.TIGHT)
    gated = M._stm_err(cols, M._phi(hist))
    control = M._stm_err(cols, M._phi(M._no_tide_gradient_replica()))
    k, dk = 0.01, 1e-3
    srp = lambda kk: M.SRPOptions(k_srp_m2_per_kg=kk, shadow_model="NO_SHADOW")  # noqa: E731
    fd = (M._rk4(M._production_rhs6(srp(k + dk)), M.X0, M.T6H)
          - M._rk4(M._production_rhs6(srp(k - dk)), M.X0, M.T6H)) / (2 * dk)
    s_k = dyn.propagate_state_with_k_sensitivity(M.T6H, M.X0, *M.BASE, srp=M.NOSHADOW, j2_moon=J2_MOON_UNNORMALIZED,
                                                 lunar_tide=M.TIDE, **M.TIGHT)[:, 42:48]
    return dict(a_stm_gated=gated, a_stm_control_no_tide_gradient=control, a_ratio=control / gated,
                b_k_sensitivity=float(np.max(np.abs(fd - s_k)) / np.max(np.abs(s_k))))


def t_g6():
    out = {}
    for harmonics in (False, True):
        kw = dict(srp=M.NOSHADOW, lunar_tide=M.TIDE, **M.TIGHT)
        if harmonics:
            kw.update(harmonic_model=M._zonal_model(), harmonic_rotation=M._rot_x(0.03))
            kw_var = dict(kw, harmonic_stm_opt_in=True)
        else:
            kw.update(j2_moon=J2_MOON_UNNORMALIZED)
            kw_var = kw
        p6 = dyn.propagate_state(M.T6H, M.X0, *M.BASE, **kw)
        runs = {"42": dyn.propagate_augmented_state(M.T6H, M._aug0(M.X0), *M.BASE, **kw_var),
                "48": dyn.propagate_state_with_k_sensitivity(M.T6H, M.X0, *M.BASE, **kw_var)}
        if not harmonics:
            runs["42_reference"] = M._reference(
                lambda: dyn.propagate_augmented_state(M.T6H, M._aug0(M.X0), *M.BASE, **kw_var))
        scale = np.max(np.abs(runs["48"][:, 6:42]))
        free = dyn.propagate_state(M.T6H, M.X0, *M.BASE, **dict(kw, lunar_tide=None))
        out["harmonic" if harmonics else "j2"] = dict(
            position_vs_6_state_m={k: float(np.max(np.abs(v[:, 0:3] - p6[:, 0:3]))) for k, v in runs.items()},
            phi_vs_48_relative={k: float(np.max(np.abs(v[:, 6:42] - runs["48"][:, 6:42])) / scale)
                                for k, v in runs.items()},
            tide_effect_6h_m=float(np.max(np.linalg.norm(free[:, 0:3] - p6[:, 0:3], axis=1))))
    return out


def t_g7():
    pre = dict(srp=M.PENUMBRA, rtol=1e-11, atol=1e-12, j2_moon=J2_MOON_UNNORMALIZED)
    cuts_tide = np.array(dyn.shadow_contact_times(M.T1D, M.X0_ECLIPSED, *M.BASE, lunar_tide=M.TIDE, **pre))
    cuts_free = np.array(dyn.shadow_contact_times(M.T1D, M.X0_ECLIPSED, *M.BASE, **pre))
    rhs, sun_lookup = dyn._build_state_rhs(M.T1D, *M.BASE, J2_MOON_UNNORMALIZED, 0.0, "indirect", None, None, None,
                                           M.PENUMBRA, lunar_tide=M.TIDE)
    try:
        dyn._integrate_segmented(M.T1D, M.X0_ECLIPSED, rhs, 1e-11, 1e-12, "DOP853", list(cuts_free), "control",
                                 contact_check=dyn._shadow_contact_check(sun_lookup, M.PENUMBRA))
        control = "no error"
    except RuntimeError as exc:
        control = str(exc)
    p_tide = dyn.propagate_state(M.T1D, M.X0_ECLIPSED, *M.BASE, lunar_tide=M.TIDE, **M.D2)
    p_free = dyn.propagate_state(M.T1D, M.X0_ECLIPSED, *M.BASE, **M.D2)
    shift = cuts_tide - cuts_free
    return dict(n_cuts=int(cuts_tide.size), n_cuts_free=int(cuts_free.size),
                max_cut_shift_s=float(np.max(np.abs(shift))), cuts_beyond_tolerance=int(np.sum(np.abs(shift) > 1e-3)),
                contact_tolerance_s=dyn._CONTACT_TIME_TOL_S, control_message=control,
                tide_effect_1d_m=float(np.max(np.linalg.norm(p_tide[:, 0:3] - p_free[:, 0:3], axis=1))))


def b_g3():
    from tests import test_wp3_2_lunar_tide_transport as B
    t = M.T6H
    truth = dyn.propagate_state(t, M.X0, *M.BASE, j2_moon=J2_MOON_UNNORMALIZED, lunar_tide=M.TIDE,
                                method="DOP853", rtol=1e-12, atol=1e-13)
    stations = tuple(B.T._synthetic_position_station(*q) for q in [(0., 0., 0.), (0., 90., 0.), (45., -30., 500.)])
    pg = B.PassGeometry(t_s=t, earth_pos_mci_m=np.zeros((t.size, 3)), earth_vel_mci_mps=np.zeros((t.size, 3)),
                        x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t.size, axis=0), stations=stations,
                        measurement_type="position")
    case = (t, pg, B.T._build_clean_position_observations(t, truth, pg))
    out = {}
    for solver in ("bls_lm", "srif"):
        dr, dv = B._solve(solver, case, M.TIDE)
        dr_free, dv_free = B._solve(solver, case, None)
        out[solver] = dict(dr0_m=dr, dv0_m_s=dv, dr0_without_tide_m=dr_free, dv0_without_tide_m_s=dv_free,
                           contrast=dr_free / dr if dr > 0 else float("inf"))
    return out


def cost_per_call(n=20000):
    rhs_off = M._production_rhs6(tide=None)
    rhs_on = M._production_rhs6()
    srp_on = M._production_rhs6(srp=M.NOSHADOW, tide=None)
    out = {}
    for name, f in (("off", rhs_off), ("tide", rhs_on), ("srp_only", srp_on)):
        f(0.0, M.X0)
        t0 = time.perf_counter()
        for i in range(n):
            f(float(i), M.X0)
        out[name + "_us"] = (time.perf_counter() - t0) / n * 1e6
    return out


def main(path):
    import platform
    import numba
    import scipy
    result = {"predeclaration": "studies/phase17/configs/wp3_2_s1_lunar_tide_predeclaration.json (bdafb53)",
              "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
              "numba": numba.__version__, "fast_dynamics": bool(dyn._FAST_DYNAMICS)}
    for name, fn in (("T_G2_G3_G4", t_g2_g3_g4), ("T_G5", t_g5), ("T_G6", t_g6), ("T_G7", t_g7),
                     ("B_G3", b_g3), ("wrapper_cost_6_state_rhs", cost_per_call)):
        t0 = time.perf_counter()
        result[name] = fn()
        print(name, json.dumps(result[name]), f"({time.perf_counter() - t0:.1f} s)", flush=True)
    thresholds = dict(g2=1e-12, g3=1e-13, g4_cs=1e-13, g4_tr=1e-12, g5a=1e-7, g5b=1e-6, g6_pos=1e-5, g6_phi=1e-8)
    g = result["T_G2_G3_G4"]
    verdicts = {
        "T_G2": max(g["g2_acc"], g["g2_grad"], g["g2_sum"]) <= thresholds["g2"],
        "T_G3": g["g3_study"] <= thresholds["g3"],
        "T_G4": g["g4_complex_step"] <= thresholds["g4_cs"] and g["g4_trace"] <= thresholds["g4_tr"]
        and g["g4_symmetric_exactly"],
        "T_G5": result["T_G5"]["a_stm_gated"] <= thresholds["g5a"] and result["T_G5"]["a_ratio"] >= 10
        and result["T_G5"]["b_k_sensitivity"] <= thresholds["g5b"],
        "T_G6": all(max(v["position_vs_6_state_m"].values()) <= thresholds["g6_pos"]
                    and max(v["phi_vs_48_relative"].values()) <= thresholds["g6_phi"] and v["tide_effect_6h_m"] > 1.0
                    for v in result["T_G6"].values()),
        "T_G7": result["T_G7"]["n_cuts"] == result["T_G7"]["n_cuts_free"]
        and result["T_G7"]["max_cut_shift_s"] > 1e-3 and "Shadow contact inconsistent" in result["T_G7"]["control_message"],
        "B_G3": all(v["dr0_m"] < 1e-3 and v["dv0_m_s"] < 1e-6 and v["contrast"] >= 100 for v in result["B_G3"].values()),
    }
    result["thresholds"] = thresholds
    result["verdicts"] = {k: "PASS" if v else "FAIL" for k, v in verdicts.items()}
    print(json.dumps(result["verdicts"]))
    Path(path).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
