"""PHASE 17-F1 - STM and K-sensitivity qualification against independent re-propagation.

Setup traceable to Phase 17-HARDEN D05 (examples/phase17_harden_range_jacobian.py):
canonical campaign x0 and epoch, GEO/D2 canonical force model (Moon point mass +
lunar J2 + cannonball SRP, conical penumbra), 90 s grid, 4 orbits.

MODELS UNDER TEST (48-state [x; Phi; dx/dK], VODE Adams, same rtol as the oracle)
  NEW  production propagate_state_with_k_sensitivity (A includes d a_SRP / d r)
  OLD  analysis-only replica of the pre-F1 variational equations: identical
       trajectory RHS (f3body_moon with srp), A WITHOUT the SRP gradient.  Not a
       production option; it exists only as the negative control.

ORACLE (never uses Phi or S_K): 6-state production propagate_state
  STM  column j:  (x(t; x0 + h e_j) - x(t; x0 - h e_j)) / 2h,
       h_pos in {1, 10, 100} m, h_vel in {1e-3, 1e-2, 1e-1} m/s
  S_K:           (x(t; K + dK) - x(t; K - dK)) / 2dK,  dK in {1e-4, 1e-3, 3e-3, 1e-2}
                 (1e-2 added after the first sweep proved noise-limited; K - dK >= 0 bounds it)
  tolerance ladder rtol in {1e-13, 1e-14, 3e-15} (atol = 0.1 rtol)
  error of a column = max over grid of |FD - analytic| / max|analytic|, taken
  separately for position and velocity rows, the larger reported.

Variants: SRP_CONICAL_PENUMBRA (science), SRP_NO_SHADOW, NO_SRP_K0 (STM floor).
K-sensitivity cases: CANON_4ORB (penumbra-rich), FULL_SUN_4ORB (D2
O2_FULL_SUN_HIGH_BETA geometry), CANON_15ORB (the Phase-17 canonical arc).

FIXED-STEP ORACLE (added after the adaptive K oracle proved noise-limited):
  the adaptive VODE step pattern differs between the K + dK and K - dK runs,
  leaving ~5e-5 m of noise, while a K perturbation moves the orbit by only
  ~S_K dK <= 4 m (dK <= 0.01 because K - dK >= 0).  A fixed-step classical RK4
  on the PRODUCTION force (f3body_moon with srp; no Phi, no S_K) has an
  identical step sequence for every run, so its discrete solution is smooth in
  K and x0 and differencing it carries no step-pattern noise.  Its own
  discretisation error is shown to converge (dt = 2 s and 1 s).

Outputs: artifacts/phase17_f1_stm_oracle.json, artifacts/phase17_f1_k_sensitivity_oracle.json
"""
from __future__ import annotations

import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ARTIFACTS = HERE.parent / "artifacts"
RTOLS = (1e-13, 1e-14, 3e-15)
STEPS = {"pos": (1.0, 10.0, 100.0), "vel": (1e-3, 1e-2, 1e-1)}
#: 1e-4 .. 3e-3 were the first sweep; it was roundoff/integration-noise limited
#: (error ~ 1/dK and ~ rtol: a K perturbation moves the orbit by only ~metres), so
#: the sweep was extended to 1e-2, the largest step allowed by K - dK >= 0.
DKS = (1e-4, 1e-3, 3e-3, 1e-2)
K_RTOLS = (1e-13, 1e-14, 3e-15)
VARIANTS = (("SRP_CONICAL_PENUMBRA", 0.01, "CONICAL_PENUMBRA"),
            ("SRP_NO_SHADOW", 0.01, "NO_SHADOW"),
            ("NO_SRP_K0", 0.0, "CONICAL_PENUMBRA"))
GATE = 1e-6
COLS = "x0 y0 z0 vx0 vy0 vz0".split()


def _init() -> None:
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels(None, clear=True)
    warnings.simplefilter("ignore")


def _case(name):
    import phase17_r1m_core as M
    et0, t_orbit = M.campaign_epoch()
    x0 = M.campaign_initial_state()
    if name == "FULL_SUN_4ORB":
        g = next(g for g in json.loads((ARTIFACTS / "phase17_r1od2_predeclared_grid.json")
                                       .read_text(encoding="utf-8"))["geometries"]
                 if g["case_id"] == "O2_FULL_SUN_HIGH_BETA")
        et0, x0 = float(g["et0"]), np.asarray(g["x0"], float)
    orbits = 15.0 if name == "CANON_15ORB" else 4.0
    return et0, np.asarray(x0, float), np.arange(0.0, orbits * t_orbit + 45.0, 90.0)


def _providers(et0):
    import phase17_geo_core as G
    return G.mu_moon(), (lambda _t: np.array([384_400e3, 0.0, 0.0])), (lambda t: G.sun_mci(et0 + float(t)))


def prop6(args):
    case, k, shadow, rtol, x0 = args
    from lunar_od.constants import J2_MOON_UNNORMALIZED
    from lunar_od.dynamics import propagate_state
    from lunar_od.srp import SRPOptions
    et0, _, t = _case(case)
    mu, earth, sun = _providers(et0)
    return propagate_state(t, np.asarray(x0, float), mu, 0.0, 0.0, earth, sun,
                           srp=SRPOptions(k_srp_m2_per_kg=k, shadow_model=shadow),
                           rtol=rtol, atol=0.1 * rtol, j2_moon=J2_MOON_UNNORMALIZED)


def prop48(args):
    """NEW = production; OLD = the pre-F1 variational equations (negative control)."""
    model, case, k, shadow, rtol = args
    import lunar_od.dynamics as dyn
    from lunar_od.constants import J2_MOON_UNNORMALIZED
    from lunar_od.srp import SRPOptions, srp_acceleration_kernel
    et0, x0, t = _case(case)
    mu, earth, sun = _providers(et0)
    srp = SRPOptions(k_srp_m2_per_kg=k, shadow_model=shadow)
    if model == "NEW":
        return dyn.propagate_state_with_k_sensitivity(t, x0, mu, 0.0, 0.0, earth, sun, srp=srp,
                                                      rtol=rtol, atol=0.1 * rtol,
                                                      j2_moon=J2_MOON_UNNORMALIZED)

    def rhs(t_s, y):
        x, phi, s_k = y[:6], y[6:42].reshape((6, 6), order="F"), y[42:48]
        r_earth, r_sun = earth(t_s), sun(t_s)
        x_dot = dyn.f3body_moon(x, mu, 0.0, 0.0, r_earth, r_sun, j2_moon=J2_MOON_UNNORMALIZED, srp=srp)
        a = dyn.dynamics_jacobian_a_matrix(x, mu, 0.0, 0.0, r_earth, r_sun, j2_moon=J2_MOON_UNNORMALIZED)
        df_dk = np.zeros(6)
        df_dk[3:] = srp_acceleration_kernel(x[:3], r_sun, srp)
        return np.concatenate([x_dot, (a @ phi).reshape(-1, order="F"), a @ s_k + df_dk])

    y0 = np.concatenate([x0, np.eye(6).reshape(-1, order="F"), np.zeros(6)])
    return dyn._propagate_vode(t, y0, rhs, rtol, 0.1 * rtol)


FIXED_DTS = (2.0, 1.0)
FIXED_DKS = (1e-4, 1e-3, 1e-2)


def prop6_rk4(args):
    """Fixed-step classical RK4 on the production force; output on the 90 s grid."""
    case, k, shadow, dt, x0 = args
    import lunar_od.dynamics as dyn
    from lunar_od.constants import J2_MOON_UNNORMALIZED
    from lunar_od.srp import SRPOptions
    et0, _, t = _case(case)
    mu, earth, sun = _providers(et0)
    srp = SRPOptions(k_srp_m2_per_kg=k, shadow_model=shadow)
    r_e = earth(0.0)

    def f(tt, y):
        return dyn.f3body_moon(y, mu, 0.0, 0.0, r_e, sun(tt), j2_moon=J2_MOON_UNNORMALIZED, srp=srp)

    n_sub = int(round(90.0 / dt))
    y = np.asarray(x0, float).copy()
    out = np.empty((t.size, 6))
    out[0] = y
    tt = 0.0
    for i in range(1, t.size):
        for _ in range(n_sub):
            k1 = f(tt, y)
            k2 = f(tt + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = f(tt + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = f(tt + dt, y + dt * k3)
            y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            tt += dt
        out[i] = y
    return out


def _col_err(fd, an):
    out = {}
    for tag, sl in (("pos", slice(0, 3)), ("vel", slice(3, 6))):
        den = float(np.max(np.abs(an[:, sl])))
        out[tag] = float(np.max(np.abs(fd[:, sl] - an[:, sl])) / den) if den > 0 else 0.0
    out["max"] = max(out["pos"], out["vel"])
    return out


def main() -> None:
    jobs6, jobs48 = {}, {}
    et0, x0c, _ = _case("CANON_4ORB")
    for name, k, shadow in VARIANTS:
        for rtol in RTOLS:
            jobs48[("NEW", "CANON_4ORB", k, shadow, rtol)] = None
            jobs48[("OLD", "CANON_4ORB", k, shadow, rtol)] = None
            for j in range(6):
                for h in STEPS["pos" if j < 3 else "vel"]:
                    for sgn in (1, -1):
                        x = x0c.copy()
                        x[j] += sgn * h
                        jobs6[("STM", name, rtol, j, h, sgn)] = ("CANON_4ORB", k, shadow, rtol, tuple(x))
    for case in ("CANON_4ORB", "FULL_SUN_4ORB", "CANON_15ORB"):
        _, x0, _ = _case(case)
        for rtol in K_RTOLS:
            jobs48[("NEW", case, 0.01, "CONICAL_PENUMBRA", rtol)] = None
            jobs48[("OLD", case, 0.01, "CONICAL_PENUMBRA", rtol)] = None
            for dk in DKS:
                for sgn in (1, -1):
                    jobs6[("K", case, rtol, dk, sgn)] = (case, 0.01 + sgn * dk, "CONICAL_PENUMBRA", rtol,
                                                          tuple(x0))
            jobs6[("K", case, rtol, 0.0, 0)] = (case, 0.01, "CONICAL_PENUMBRA", rtol, tuple(x0))
    jobs_rk = {}
    for case in ("CANON_4ORB", "FULL_SUN_4ORB", "CANON_15ORB"):
        _, x0, _ = _case(case)
        for dt in FIXED_DTS:
            for dk in FIXED_DKS:
                for sgn in (1, -1):
                    jobs_rk[("K", case, dt, dk, sgn)] = (case, 0.01 + sgn * dk, "CONICAL_PENUMBRA", dt,
                                                         tuple(x0))
    for dt in FIXED_DTS:        # STM cross-check, penumbra case, one step per column
        for j in range(6):
            h = 10.0 if j < 3 else 1e-2
            for sgn in (1, -1):
                x = x0c.copy()
                x[j] += sgn * h
                jobs_rk[("STM", dt, j, sgn)] = ("CANON_4ORB", 0.01, "CONICAL_PENUMBRA", dt, tuple(x))
    with ProcessPoolExecutor(max_workers=15, initializer=_init) as ex:
        k48 = list(jobs48)
        r48 = dict(zip(k48, ex.map(prop48, k48)))
        k6 = list(jobs6)
        r6 = dict(zip(k6, ex.map(prop6, [jobs6[k] for k in k6], chunksize=4)))
        krk = list(jobs_rk)
        rrk = dict(zip(krk, ex.map(prop6_rk4, [jobs_rk[k] for k in krk])))

    # ---------------- STM
    stm = dict(phase="17-F1", setup="HARDEN D05: canonical x0/epoch, 4 orbits, 90 s grid",
               rtols=RTOLS, steps=STEPS, gate=GATE, variants={})
    for name, k, shadow in VARIANTS:
        v = dict(models={})
        for model in ("NEW", "OLD"):
            cols = {}
            for j, c in enumerate(COLS):
                cols[c] = {}
                for rtol in RTOLS:
                    an = r48[(model, "CANON_4ORB", k, shadow, rtol)]
                    phi = np.array([row.reshape((6, 6), order="F") for row in an[:, 6:42]])
                    for h in STEPS["pos" if j < 3 else "vel"]:
                        fd = (r6[("STM", name, rtol, j, h, 1)] - r6[("STM", name, rtol, j, h, -1)]) / (2 * h)
                        cols[c]["rtol=%g,h=%g" % (rtol, h)] = _col_err(fd, phi[:, :, j])
                best = min(cols[c].items(), key=lambda kv: kv[1]["max"])
                cols[c]["best"] = dict(setting=best[0], **best[1])
            v["models"][model] = dict(columns=cols,
                                      max_best_rel_err=max(cols[c]["best"]["max"] for c in COLS))
        stm["variants"][name] = v
    new_pen = stm["variants"]["SRP_CONICAL_PENUMBRA"]["models"]["NEW"]["max_best_rel_err"]
    old_pen = stm["variants"]["SRP_CONICAL_PENUMBRA"]["models"]["OLD"]["max_best_rel_err"]
    floor = max(stm["variants"][n]["models"]["NEW"]["max_best_rel_err"] for n in ("SRP_NO_SHADOW", "NO_SRP_K0"))
    # tolerance convergence of the NEW penumbra error (best step per rtol)
    conv = {}
    for rtol in RTOLS:
        conv["%g" % rtol] = {m: max(min(stm["variants"]["SRP_CONICAL_PENUMBRA"]["models"][m]["columns"][c][
            "rtol=%g,h=%g" % (rtol, h)]["max"] for h in STEPS["pos" if j < 3 else "vel"])
            for j, c in enumerate(COLS)) for m in ("NEW", "OLD")}
    rk_stm = {}
    for model in ("NEW", "OLD"):
        an = r48[(model, "CANON_4ORB", 0.01, "CONICAL_PENUMBRA", 3e-15)]
        phi = np.array([row.reshape((6, 6), order="F") for row in an[:, 6:42]])
        rk_stm[model] = {"dt=%g" % dt: max(_col_err((rrk[("STM", dt, j, 1)] - rrk[("STM", dt, j, -1)])
                                                    / (2 * (10.0 if j < 3 else 1e-2)), phi[:, :, j])["max"]
                                           for j in range(6)) for dt in FIXED_DTS}
    stm["fixed_step_rk4_cross_check"] = rk_stm
    stm.update(MAX_STM_FD_RELERR=new_pen, OLD_MODEL_MAX_STM_FD_RELERR=old_pen,
               STM_ORACLE_FLOOR=floor, tolerance_convergence=conv,
               old_over_new=old_pen / new_pen,
               F1_STM_GATE="PASS" if new_pen <= GATE else "FAIL",
               OLD_VARIATIONAL_APPROXIMATION_FAILURE_REPRODUCED=bool(old_pen >= 10 * new_pen and old_pen > GATE))
    (ARTIFACTS / "phase17_f1_stm_oracle.json").write_text(json.dumps(stm, indent=1), encoding="utf-8")

    # ---------------- K sensitivity
    ks = dict(phase="17-F1", dks=DKS, rtols=K_RTOLS, gate=GATE, cases={})
    for case in ("CANON_4ORB", "FULL_SUN_4ORB", "CANON_15ORB"):
        c = dict(models={})
        for model in ("NEW", "OLD"):
            sweep = {}
            for rtol in K_RTOLS:
                an = r48[(model, case, 0.01, "CONICAL_PENUMBRA", rtol)][:, 42:48]
                x_c = r6[("K", case, rtol, 0.0, 0)]
                for dk in DKS:
                    xp, xm = r6[("K", case, rtol, dk, 1)], r6[("K", case, rtol, dk, -1)]
                    fd = (xp - xm) / (2 * dk)
                    e = _col_err(fd, an)
                    # K-nonlinearity of the trajectory: forward vs backward difference
                    nonlin = float(np.max(np.abs((xp - x_c) - (x_c - xm))[:, :3])
                                   / np.max(np.abs(xp - xm)[:, :3]))
                    sweep["rtol=%g,dK=%g" % (rtol, dk)] = dict(**e, abs_err_pos_m_per_unitK=float(
                        np.max(np.abs(fd[:, :3] - an[:, :3]))), fwd_bwd_asymmetry=nonlin)
            best = min(sweep.items(), key=lambda kv: kv[1]["max"])
            c["models"][model] = dict(sweep=sweep, best=dict(setting=best[0], **best[1]),
                                      s_k_pos_max_m_per_unitK=float(np.max(np.abs(an[:, :3]))))
        ks["cases"][case] = c
    for case in ks["cases"]:
        for model in ("NEW", "OLD"):
            an = r48[(model, case, 0.01, "CONICAL_PENUMBRA", 3e-15)][:, 42:48]
            sw = {}
            for dt in FIXED_DTS:
                for dk in FIXED_DKS:
                    fd = (rrk[("K", case, dt, dk, 1)] - rrk[("K", case, dt, dk, -1)]) / (2 * dk)
                    sw["dt=%g,dK=%g" % (dt, dk)] = _col_err(fd, an)
            best = min(sw.items(), key=lambda kv: kv[1]["max"])
            ks["cases"][case]["models"][model]["fixed_step_rk4"] = dict(
                sweep=sw, best=dict(setting=best[0], **best[1]))
            adaptive = ks["cases"][case]["models"][model]["best"]
            ks["cases"][case]["models"][model]["best_resolved"] = (
                dict(oracle="FIXED_STEP_RK4", **best[1], setting=best[0])
                if best[1]["max"] < adaptive["max"] else dict(oracle="ADAPTIVE_VODE", **adaptive))
    new_k = max(ks["cases"][c]["models"]["NEW"]["best_resolved"]["max"] for c in ks["cases"])
    ks.update(MAX_K_SENSITIVITY_FD_RELERR=new_k,
              OLD_MODEL_MAX_K_SENSITIVITY_FD_RELERR=max(ks["cases"][c]["models"]["OLD"]["best_resolved"]["max"]
                                                        for c in ks["cases"]),
              F1_K_SENSITIVITY_GATE="PASS" if new_k <= GATE else "FAIL")
    (ARTIFACTS / "phase17_f1_k_sensitivity_oracle.json").write_text(json.dumps(ks, indent=1), encoding="utf-8")

    for name in stm["variants"]:
        for model in ("NEW", "OLD"):
            m = stm["variants"][name]["models"][model]
            print("%-22s %s max best %.2e  " % (name, model, m["max_best_rel_err"]) + " ".join(
                "%s:%.1e(%s)" % (c, m["columns"][c]["best"]["max"], m["columns"][c]["best"]["setting"]) for c in COLS))
    print("tolerance convergence (penumbra, best step):", json.dumps(conv))
    print("STM gate", stm["F1_STM_GATE"], "new %.2e old %.2e floor %.2e old/new %.0f" % (new_pen, old_pen, floor, old_pen / new_pen))
    for case, c in ks["cases"].items():
        for model in ("NEW", "OLD"):
            b = c["models"][model]["best"]
            r = c["models"][model]["fixed_step_rk4"]
            print("K %-14s %s adaptive best %.2e (%s) | RK4 " % (case, model, b["max"], b["setting"])
                  + " ".join("%s:%.1e" % (kk, vv["max"]) for kk, vv in r["sweep"].items()))
    print("STM RK4 cross-check:", json.dumps(rk_stm))
    print("K gate", ks["F1_K_SENSITIVITY_GATE"], "max %.2e" % new_k)


if __name__ == "__main__":
    main()
