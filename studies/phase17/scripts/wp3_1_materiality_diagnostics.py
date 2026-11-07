"""F3 WP3.1-R diagnostics after the A2 result (exploratory; the survey's decisions are not changed by them).

1. SRP_LIGHT_TIME. A2 did not cut the integration at the light-time-shifted shadow contacts. Here the baseline
   and the term are integrated with max_step 1 s (every kink inside a 1 s step), and the term is split into
   its aberration part (production shadow geometry, Poynting-Robertson force) and its shadow light-time part
   (Sun at emission, Moon at the ray crossing, no velocity in the force).
2. EARTH_J2 in O1 x E1 (response 0.108, signal 16x the floor): baseline and term at the floor tolerance
   (1e-13, 1e-14) of gate G4, profile D2.

The design, rows and fits are the survey's (wp3_1_materiality_survey, profile D2).

    python wp3_1_materiality_diagnostics.py part.json O3xE3        # one SRP_LIGHT_TIME cell, or EARTH_J2
    python wp3_1_materiality_diagnostics.py combine part1.json ...  # -> results/wp3_1_materiality_diagnostics.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wp3_1_force_terms as W  # noqa: E402
import wp3_1_materiality_survey as S  # noqa: E402
from lunar_od.constants import SPEED_OF_LIGHT_M_S  # noqa: E402
from lunar_od.srp import solar_pressure_at, srp_acceleration_with_lunar_shadow  # noqa: E402

OUT = S.ROOT / "studies" / "phase17" / "results" / "wp3_1_materiality_diagnostics.json"
MAX_STEP_S = 1.0
SRP_CELLS = ("O3xE3", "O3xE1", "O1xE1", "O2xE1")
ZERO = np.zeros(3)


def aberration_only(r_sc, v_sc, r_sun, v_sun, v_moon, srp, c=SPEED_OF_LIGHT_M_S):
    """K P nu [(1 - v.u/c) u - v/c] - a_SRP with production's geometric Sun and Moon (no light time)."""
    r_sc, r_sun = np.asarray(r_sc, float), np.asarray(r_sun, float)
    a_srp = srp_acceleration_with_lunar_shadow(r_sc, r_sun, srp)
    nu = W.lunar_illumination(r_sc, r_sun, ZERO, srp)
    if nu == 0.0:
        return -a_srp
    rel = r_sc - r_sun
    d = float(np.linalg.norm(rel))
    u = rel / d
    v = np.asarray(v_sc, float) + np.asarray(v_moon, float) - np.asarray(v_sun, float)
    return srp.require_k() * ((solar_pressure_at(d) * nu) * ((1.0 - float(np.dot(v, u)) / c) * u - v / c)) - a_srp


def light_time_only(r_sc, v_sc, r_sun, v_sun, v_moon, srp, c=SPEED_OF_LIGHT_M_S):
    """K P nu_LT u_LT - a_SRP: Sun at emission, Moon at the ray crossing, no velocity in the force."""
    r_sc = np.asarray(r_sc, float)
    a_srp = srp_acceleration_with_lunar_shadow(r_sc, r_sun, srp)
    r_emit, r_occ = W.light_time_geometry(r_sc, r_sun, v_sun, v_moon, c)
    nu = W.lunar_illumination(r_sc, r_emit, r_occ, srp)
    if nu == 0.0:
        return -a_srp
    rel = r_sc - r_emit
    d = float(np.linalg.norm(rel))
    return srp.require_k() * ((solar_pressure_at(d) * nu) * (rel / d)) - a_srp


def _ssb_term(fn):
    def term(t, x, ctx):
        et = ctx.et0 + t
        return fn(x[:3], x[3:6], ctx.sun_pos(t), W.ssb_velocity("SUN", et), W.ssb_velocity("MOON", et), ctx.srp)
    return term


VARIANTS = {"FULL": "SRP_LIGHT_TIME", "ABERRATION_ONLY": _ssb_term(aberration_only),
            "LIGHT_TIME_ONLY": _ssb_term(light_time_only)}


def integrate_small_steps(rhs, x0):
    from scipy.integrate import solve_ivp
    sol = solve_ivp(rhs, (float(S.T_GRID[0]), float(S.T_GRID[-1])), np.asarray(x0, float), method="DOP853",
                    t_eval=S.T_GRID, rtol=S.RTOL, atol=S.ATOL, max_step=MAX_STEP_S)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def fits(h, dy) -> dict:
    out = {f"{d}/{n}": S.fit(h, dy, *S.NOISE[n], d == "RANGE_RANGE_RATE") for d in S.DESIGNS for n in S.NOISE}
    return dict(response=max(v["response"] for v in out.values()),
                by_design={k: v["response"] for k, v in out.items()},
                argmax={k: v["argmax"] for k, v in out.items()})


def setup(name: str, epochs: dict):
    orbit, epoch = name.split("x")
    if epoch not in epochs:
        epochs[epoch] = S.epoch_et0(epoch)
    cell = S.build_cell(orbit, epoch, *epochs[epoch])
    x_base, _ = S.run_base(cell)
    obs, pass_geo, _ = S.measurement_rows(cell, x_base)
    h, _ = S.design(cell, x_base, obs, pass_geo)
    return cell, x_base, obs, pass_geo, h


def srp_light_time(name: str, epochs: dict) -> dict:
    t0 = time.time()
    cell, x_base, obs, pass_geo, h = setup(name, epochs)
    y_base = S.observables(x_base, obs, pass_geo)
    x_term, _ = S.run_term(cell, "SRP_LIGHT_TIME", 1.0)              # the survey's own run, reproduced
    survey = fits(h, S.observables(x_term, obs, pass_geo) - y_base)
    rhs, _ = S.closure(cell)
    x_ref = integrate_small_steps(rhs, cell.x0)
    out = dict(survey_reproduced=survey["response"],
               small_step_baseline_vs_survey_m=float(np.max(np.linalg.norm(x_ref[:, :3] - x_base[:, :3], axis=1))))
    y_ref = S.observables(x_ref, obs, pass_geo)
    for label, term in VARIANTS.items():
        rhs, sun = S.closure(cell)
        x = integrate_small_steps(W.with_term(rhs, term, S.context(cell, sun), 1.0), cell.x0)
        dr = np.linalg.norm(x[:, :3] - x_ref[:, :3], axis=1)
        out[label] = dict(max_dr_m=float(dr.max()), final_dr_m=float(dr[-1]),
                          **fits(h, S.observables(x, obs, pass_geo) - y_ref))
        print(f"[{name}] {label}: max|dr| {dr.max():.3e} m, response {out[label]['response']:.3f}", flush=True)
    out["wall_s"] = time.time() - t0
    return out


def earth_j2_fine(epochs: dict) -> dict:
    cell, x_base, obs, pass_geo, h = setup("O1xE1", epochs)
    y_base = S.observables(x_base, obs, pass_geo)
    x_term, _ = S.run_term(cell, "EARTH_J2", 1.0)
    survey = fits(h, S.observables(x_term, obs, pass_geo) - y_base)
    rhs, sun = S.closure(cell)
    x_ref, _ = S.integrate(rhs, sun, cell.srp, cell.x0, *S.FLOOR_TOL)
    rhs, sun = S.closure(cell)
    x_fine, _ = S.integrate(W.with_term(rhs, "EARTH_J2", S.context(cell, sun), 1.0), sun, cell.srp, cell.x0,
                            *S.FLOOR_TOL)
    dr = np.linalg.norm(x_fine[:, :3] - x_ref[:, :3], axis=1)
    fine = fits(h, S.observables(x_fine, obs, pass_geo) - S.observables(x_ref, obs, pass_geo))
    return dict(survey_reproduced=survey["response"], tolerance=list(S.FLOOR_TOL), max_dr_m=float(dr.max()), **fine)


def main(argv: list) -> None:
    if argv[1] == "combine":
        parts = [json.loads(Path(p).read_text(encoding="utf-8")) for p in argv[2:]]
        result = dict(status="exploratory diagnostics after the A2 result; no decision changed", profile="D2",
                      max_step_s=MAX_STEP_S, SRP_LIGHT_TIME={})
        for p in parts:
            result["SRP_LIGHT_TIME"].update(p.get("SRP_LIGHT_TIME", {}))
            if "EARTH_J2_O1xE1_fine_tolerance" in p:
                result["EARTH_J2_O1xE1_fine_tolerance"] = p["EARTH_J2_O1xE1_fine_tolerance"]
        if sorted(result["SRP_LIGHT_TIME"]) != sorted(SRP_CELLS) or "EARTH_J2_O1xE1_fine_tolerance" not in result:
            raise ValueError("a diagnostic part is missing")
        OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
        return
    from lunar_od.spice_loader import load_spice_kernels
    S.PROFILE = "D2"
    load_spice_kernels()
    epochs = {}
    which = argv[2]
    if which == "EARTH_J2":
        part = {"EARTH_J2_O1xE1_fine_tolerance": earth_j2_fine(epochs)}
    elif which in SRP_CELLS:
        part = {"SRP_LIGHT_TIME": {which: srp_light_time(which, epochs)}}
    else:
        raise SystemExit(__doc__)
    Path(argv[1]).write_text(json.dumps(part, indent=1), encoding="utf-8")
    print(json.dumps(part, indent=1)[:2000])


if __name__ == "__main__":
    main(sys.argv)
