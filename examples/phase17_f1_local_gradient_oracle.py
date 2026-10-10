"""PHASE 17-F1 - local SRP position-gradient oracle.

Independent oracle: central finite differences (3-point, 2nd order AND 5-point,
4th order) of the PRODUCTION acceleration
``srp_acceleration_with_lunar_shadow`` (and of the production illumination for
the nu-gradient), never of the new gradient code.  Predeclared step sweep
(metres, every point, every axis): 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4.  Every
step's error is stored; the gate uses the best (order, step) and requires the
sweep to follow the FD error model (roundoff falling ~1/h, then truncation
rising).  The 4th-order oracle was added after the first run: next to the
penumbra contacts nu ~ eps^(3/2) is C1 but not C2, so the 2nd-order oracle's
truncation (h^2 * curvature) and the acos-cancellation roundoff of production nu
cross at ~1e-5 there -- an ORACLE floor.  At nu ~ 0.005 / 0.995 even the
4th-order sweep stays roundoff-dominated (error ~ 1/h) down to ~1e-6, so a
second, difference-free oracle is used as well: the fundamental theorem of
calculus along +/-50 m and +/-200 m axis segments that stay inside one smooth
regime (Gauss-Legendre integral of the analytic gradient vs the difference of
production accelerations 2L apart).  The gate is the best-resolved independent
oracle; both are stored.  The 1e-6 gate is unchanged.

Points (two Sun directions, two altitudes behind the Moon):
  FULL_SUN_SUNWARD, FULL_SUN_ANTISUNWARD (outside the cone),
  PENUMBRA_INTERIOR nu ~ 0.1 / 0.25 / 0.5 / 0.75 / 0.9,
  SHADOW_ENTRY_NEIGHBORHOOD nu ~ 0.005, SHADOW_EXIT_NEIGHBORHOOD nu ~ 0.995,
  UMBRA_INTERIOR, ANNULAR (4e8 m, completeness only), EXACT contacts
  (external / internal tangency, finiteness + one-sided behaviour),
  K in {0, 0.01, 0.037}, and NO_SHADOW.

Output: artifacts/phase17_f1_local_gradient_oracle.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ARTIFACTS = HERE.parent / "artifacts"
STEPS = (1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4)
GATE = 1e-6


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _fd(fun, r, h, order=2):
    """Central difference of order 2 (3-point) or 4 (5-point)."""
    g = np.zeros((3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        if order == 2:
            g[:, j] = (fun(r + e) - fun(r - e)) / (2.0 * h)
        else:
            g[:, j] = (-fun(r + 2 * e) + 8 * fun(r + e) - 8 * fun(r - e) + fun(r - 2 * e)) / (12.0 * h)
    return g


def _error_model_consistent(errs):
    """The sweep must look like an FD error curve: falling (roundoff, ~1/h) to a
    single minimum and then rising (truncation), or falling throughout when the
    function is smooth on the sweep scale (roundoff-limited).  An erratic curve
    means the oracle, not the analytic value, is unresolved."""
    i = int(np.argmin(errs))
    falling = all(errs[j] >= errs[j + 1] * 0.5 for j in range(i))
    rising = all(errs[j + 1] >= errs[j] * 0.5 for j in range(i, len(errs) - 1))
    return bool(falling and rising)


def _integral_oracle(acc, grad, regime, r, half_lengths=(50.0, 200.0), nodes=32):
    """Fundamental-theorem oracle, independent of finite differencing:
    int_{-L}^{L} G(r + s e_j) e_j ds  ==  a(r + L e_j) - a(r - L e_j).
    The right side differences production accelerations 2L apart (no roundoff
    amplification); the left side integrates the analytic gradient with
    Gauss-Legendre quadrature.  A segment is used only if it stays inside one
    smooth regime (same production regime label at both ends and at every node)."""
    x, w = np.polynomial.legendre.leggauss(nodes)
    best = None
    for big_l in half_lengths:
        errs = []
        for j in range(3):
            e = np.zeros(3)
            e[j] = 1.0
            pts = [r + big_l * t * e for t in x]
            if any(regime(p) != regime(r) for p in pts + [r + big_l * e, r - big_l * e]):
                errs = None
                break
            lhs = big_l * sum(wi * (grad(p) @ e) for wi, p in zip(w, pts))
            rhs = acc(r + big_l * e) - acc(r - big_l * e)
            den = float(np.linalg.norm(rhs))
            errs.append(float(np.linalg.norm(lhs - rhs) / den) if den > 0 else 0.0)
        if errs is not None:
            cand = dict(half_length_m=big_l, rel_err=max(errs))
            if best is None or cand["rel_err"] < best["rel_err"]:
                best = cand
    return best


def _find_nu(target, axis_pt, n, sun, opts, lo, hi):
    from lunar_od.srp import _illumination_for
    f = lambda rho: _illumination_for(axis_pt + rho * n, sun, opts) - target  # noqa: E731
    a, b = lo, hi
    for _ in range(200):
        m = 0.5 * (a + b)
        if (f(a) < 0) == (f(m) < 0):
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def build_points():
    from lunar_od.constants import AU_M, R_MOON_M
    from lunar_od.srp import SRPOptions, apparent_radii_and_separation
    pts = []
    for sun_dir in (_unit([-0.83, 0.52, 0.19]), _unit([0.31, -0.12, 0.94])):
        sun = sun_dir * AU_M * 1.0167
        n = _unit(np.cross(sun_dir, [0.2, 0.9, -0.4]))
        for dist in (R_MOON_M + 100e3, R_MOON_M + 500e3):
            axis_pt = -dist * sun_dir
            opts = SRPOptions(k_srp_m2_per_kg=0.01)
            # locate the penumbra band along n by bisection on production nu
            rho_um = _find_nu(1e-12, axis_pt, n, sun, opts, 0.0, 3 * R_MOON_M)      # umbra edge
            rho_fl = _find_nu(1 - 1e-12, axis_pt, n, sun, opts, rho_um, 3 * R_MOON_M)  # light edge
            tag = "sun%s_alt%dkm" % ("A" if sun_dir[2] < 0.5 else "B", round((dist - R_MOON_M) / 1e3))
            pts.append(("FULL_SUN_SUNWARD", tag, dist * sun_dir + 0.3 * dist * n, sun))
            pts.append(("FULL_SUN_ANTISUNWARD", tag, axis_pt + (rho_fl + 50e3) * n, sun))
            pts.append(("UMBRA_INTERIOR", tag, axis_pt + 0.5 * rho_um * n, sun))
            for nu in (0.1, 0.25, 0.5, 0.75, 0.9):
                rho = _find_nu(nu, axis_pt, n, sun, opts, rho_um, rho_fl)
                pts.append(("PENUMBRA_INTERIOR", tag + "_nu%.2f" % nu, axis_pt + rho * n, sun))
            pts.append(("SHADOW_ENTRY_NEIGHBORHOOD", tag + "_nu0.005",
                        axis_pt + _find_nu(0.005, axis_pt, n, sun, opts, rho_um, rho_fl) * n, sun))
            pts.append(("SHADOW_EXIT_NEIGHBORHOOD", tag + "_nu0.995",
                        axis_pt + _find_nu(0.995, axis_pt, n, sun, opts, rho_um, rho_fl) * n, sun))
            pts.append(("EXACT_INTERNAL_CONTACT", tag, axis_pt + rho_um * n, sun))
            pts.append(("EXACT_EXTERNAL_CONTACT", tag, axis_pt + rho_fl * n, sun))
        # annular: far enough that the Moon's disk is smaller than the Sun's
        far = -4.0e8 * sun_dir
        a_s, a_o, *_ = apparent_radii_and_separation(far, sun, np.zeros(3), R_MOON_M)
        pts.append(("ANNULAR", "sun%s_4e8m" % ("A" if sun_dir[2] < 0.5 else "B"),
                    far + 0.3 * (a_s - a_o) * 4.0e8 * n, sun))
    return pts


def evaluate(kind, tag, r, sun, k, shadow):
    from lunar_od.srp import (SRPOptions, _illumination_and_gradient, _illumination_for,
                              srp_acceleration_position_gradient, srp_acceleration_with_lunar_shadow)
    opts = SRPOptions(k_srp_m2_per_kg=k, shadow_model=shadow)
    g_an = srp_acceleration_position_gradient(r, sun, opts)
    finite = bool(np.all(np.isfinite(g_an)))
    nu, grad_nu = _illumination_and_gradient(r, sun, opts)
    rec = dict(kind=kind, tag=tag, k=k, shadow=shadow, nu=nu, nu_production=_illumination_for(r, sun, opts),
               g_an_norm=float(np.linalg.norm(g_an)), finite=finite, sweep={})
    acc = lambda x: srp_acceleration_with_lunar_shadow(x, sun, opts)  # noqa: E731
    best = None
    for order in (2, 4):
        errs = []
        for h in STEPS:
            g_fd = _fd(acc, r, h, order)
            den = max(float(np.linalg.norm(g_an)), float(np.linalg.norm(g_fd)))
            e = float(np.linalg.norm(g_fd - g_an) / den) if den > 0 else 0.0
            rec["sweep"]["o%d_%g" % (order, h)] = dict(rel_err=e, fd_norm=float(np.linalg.norm(g_fd)))
            errs.append(e)
        i = int(np.argmin(errs))
        cand = dict(order=order, best_step_m=STEPS[i], best_rel_err=errs[i],
                    error_model_consistent=_error_model_consistent(errs))
        rec["order%d" % order] = cand
        if best is None or cand["best_rel_err"] < best["best_rel_err"]:
            best = cand
    from lunar_od.srp import illumination_fraction

    def regime(x):
        if shadow == "NO_SHADOW" or float(np.dot(x, sun)) > 0.0:
            return "LIT_CONSTANT"
        return illumination_fraction(x, sun, np.zeros(3), opts.occulting_body_radius_m)[1]
    integ = _integral_oracle(acc, lambda x: srp_acceleration_position_gradient(x, sun, opts),
                             regime, r) if np.linalg.norm(g_an) > 0 else None
    rec["integral_oracle"] = integ
    rec["fd_best_rel_err"] = best["best_rel_err"]
    if integ is not None and integ["rel_err"] < best["best_rel_err"]:
        best = dict(order="INTEGRAL", best_step_m=integ["half_length_m"], best_rel_err=integ["rel_err"],
                    error_model_consistent=True)
    rec["best_order"] = best["order"]
    rec["best_step_m"] = best["best_step_m"]
    rec["best_rel_err"] = best["best_rel_err"]
    rec["plateau"] = best["error_model_consistent"]
    # independent nu-gradient check (production nu, not the new code)
    nu_fd = [np.linalg.norm(_fd(lambda x: np.array([_illumination_for(x, sun, opts)] * 3), r, h, o)[0]
                            - grad_nu) / max(np.linalg.norm(grad_nu), 1e-300)
             for h in STEPS for o in (2, 4)]
    rec["grad_nu_norm"] = float(np.linalg.norm(grad_nu))
    nu_integ = None
    if np.linalg.norm(grad_nu) > 0:
        nu_integ = _integral_oracle(lambda x: np.full(3, _illumination_for(x, sun, opts)),
                                    lambda x: np.tile(_illumination_and_gradient(x, sun, opts)[1], (3, 1)),
                                    regime, r)
    rec["grad_nu_integral_oracle"] = nu_integ
    rec["grad_nu_best_rel_err"] = (float(min(min(nu_fd), nu_integ["rel_err"] if nu_integ else np.inf))
                                   if np.linalg.norm(grad_nu) > 0 else 0.0)
    return rec


def main() -> None:
    from lunar_od.constants import AU_M
    recs = []
    for kind, tag, r, sun in build_points():
        for k in (0.0, 0.01, 0.037):
            recs.append(evaluate(kind, tag, r, sun, k, "CONICAL_PENUMBRA"))
        recs.append(evaluate(kind, tag, r, sun, 0.01, "NO_SHADOW"))
    # gates
    well_defined = [x for x in recs if not x["kind"].startswith("EXACT") and x["k"] > 0
                    and x["g_an_norm"] > 0]
    exact_zero = [x for x in recs if x["k"] == 0.0 or (x["kind"] == "UMBRA_INTERIOR"
                                                         and x["shadow"] == "CONICAL_PENUMBRA")]

    def gate(sel):
        return bool(sel) and all(x["best_rel_err"] <= GATE and x["plateau"] and x["finite"] for x in sel)
    by = lambda *kinds: [x for x in well_defined if x["kind"] in kinds and x["shadow"] == "CONICAL_PENUMBRA"]  # noqa: E731
    gates = dict(
        FULL_SUN_GRADIENT_GATE=gate(by("FULL_SUN_SUNWARD", "FULL_SUN_ANTISUNWARD")),
        PENUMBRA_GRADIENT_GATE=gate(by("PENUMBRA_INTERIOR", "SHADOW_ENTRY_NEIGHBORHOOD",
                                       "SHADOW_EXIT_NEIGHBORHOOD")),
        ANNULAR_GRADIENT_GATE=gate(by("ANNULAR")),
        UMBRA_GRADIENT_GATE=all(x["g_an_norm"] == 0.0 and x["sweep"]["o2_1"]["fd_norm"] == 0.0
                                for x in recs if x["kind"] == "UMBRA_INTERIOR"
                                and x["shadow"] == "CONICAL_PENUMBRA"),
        K_ZERO_CONTROL=all(x["g_an_norm"] == 0.0 for x in recs if x["k"] == 0.0),
        NO_SHADOW_CONTROL=gate([x for x in well_defined if x["shadow"] == "NO_SHADOW"])
        and all(x["grad_nu_norm"] == 0.0 for x in recs if x["shadow"] == "NO_SHADOW"),
        NU_GRADIENT_GATE=all(x["grad_nu_best_rel_err"] <= GATE for x in well_defined
                             if x["grad_nu_norm"] > 0),
        SHADOW_BOUNDARY_GATE=all(x["finite"] and x["nu"] == x["nu_production"]
                                 for x in recs if x["kind"].startswith("EXACT")),
        NU_CONSISTENT_WITH_PRODUCTION=all(x["nu"] == x["nu_production"] for x in recs),
    )
    gates["LOCAL_SRP_GRADIENT_GATE"] = all(gates.values())
    boundary = [dict(kind=x["kind"], tag=x["tag"], nu=x["nu"], g_an_norm=x["g_an_norm"],
                     grad_nu_norm=x["grad_nu_norm"], best_rel_err=x["best_rel_err"],
                     best_step_m=x["best_step_m"])
                for x in recs if x["kind"].startswith("EXACT") and x["k"] == 0.01
                and x["shadow"] == "CONICAL_PENUMBRA"]
    out = dict(phase="17-F1", steps_m=STEPS, gate_rel_err=GATE, n_points=len(recs),
               au_m=AU_M, gates=gates, boundary_characterisation=boundary,
               max_best_rel_err_well_defined=max(x["best_rel_err"] for x in well_defined),
               records=recs)
    (ARTIFACTS / "phase17_f1_local_gradient_oracle.json").write_text(json.dumps(out, indent=1),
                                                                     encoding="utf-8")
    for x in recs:
        if x["k"] == 0.01:
            print("%-26s %-22s %-16s nu=%.4f |G|=%.3e best=%.2e @%gm plateau=%s" % (
                x["kind"], x["tag"], x["shadow"], x["nu"], x["g_an_norm"], x["best_rel_err"],
                x["best_step_m"], x["plateau"]))
    print(json.dumps(gates, indent=1))
    print("max best rel err (well-defined):", out["max_best_rel_err_well_defined"])


if __name__ == "__main__":
    main()
