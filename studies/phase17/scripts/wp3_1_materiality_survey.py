"""F3 WP3.1-R: materiality survey of the force terms production lacks (REQ-DYN-007, analysis only).

Predeclaration: studies/phase17/configs/wp3_1_materiality_predeclaration.json (a93dbc7, amendment A1 399fdc8).
Nine cells (3 orbits x 3 epochs), 1-day arcs at 60 s. Each term is added alone to the production RHS
(wp3_1_force_terms.with_term). Its nonlinear trajectory difference gives a range / range-rate signature on the
baseline's visible rows; the linear OD impact is the prior-regularised least-squares fit of that signature by
(x0, K_SRP), scored as max_i |delta_theta_i / sigma_i| (OD_MATERIAL at >= 0.1).

Two integration profiles. P0 is the predeclared one: a single solve_ivp DOP853 run (propagate_state with
shadow_events False). D2 is amendment A2: production's consistent profile (PropagationProfile(): DOP853 with
shadow_events True), which restarts the integrator at every lunar penumbra contact, for every propagation.

    python wp3_1_materiality_survey.py run P0 E1 part_e1.json   # the three cells of one epoch
    python wp3_1_materiality_survey.py combine P0 part_e1.json part_e2.json part_e3.json
    python wp3_1_materiality_survey.py smoke P0 out.json        # 1 h arc, O1 x E1, software check only
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import wp3_1_force_terms as W  # noqa: E402

PRE = ROOT / "studies" / "phase17" / "configs" / "wp3_1_materiality_predeclaration.json"
OUT = {"P0": ROOT / "studies" / "phase17" / "results" / "wp3_1_materiality_survey.json",
       "D2": ROOT / "studies" / "phase17" / "results" / "wp3_1_materiality_survey_a2.json"}
PROFILES = ("P0", "D2")
PROFILE = "P0"
RTOL, ATOL = 1e-12, 1e-13
FLOOR_TOL = (1e-13, 1e-14)
FLOOR_MAX_M = 1e-3
T_GRID = np.arange(0.0, 86400.0 + 30.0, 60.0)
EPH_STEP_S, EPH_MARGIN_S = 60.0, 600.0
ALTITUDE_M = 100e3
STATIONS = ("Goldstone DSN", "Madrid DSN", "Canberra DSN")
NOISE = {"PROJECT": (5.0, 1e-4), "DSN_REALISTIC": (1.0, 3e-5)}
DESIGNS = ("RANGE_ONLY", "RANGE_RANGE_RATE")
PRIOR_SD = np.array([100.0, 100.0, 100.0, 0.1, 0.1, 0.1])
PARAMS = ("x", "y", "z", "vx", "vy", "vz", "K_SRP")
THRESHOLD = 0.1
FD_STEPS = (1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3, 1e-4)
G5_CELL, G5_MAX = "O1xE1", 1e-4
ORBITS = ("O1", "O2", "O3")
EPOCHS = ("E1", "E2", "E3")
ECLIPSE_DATES = {"E2": "2026-03-03", "E3": "2026-08-28"}
RUN_TERMS = W.TERMS + W.REPORTED
G1_TESTS = ("tests/test_wp3_1_force_terms.py", "tests/test_phase_c6_lunar_radiation.py",
            "tests/test_phase_c7_dynamics_candidates.py")
#: G1 test-name keywords per term; a failure in a shared test (parameters, wrapper) touches every term
G1_KEYWORDS = {"EARTH_SHADOW": ("earth_shadow",), "SRP_LIGHT_TIME": ("light_time", "aberration", "sun_line",
               "moon_displacement"), "EARTH_J2": ("earth_j2", "itrf93"), "TIDE": ("tide",),
               "LRP": ("test_phase_c6", "lrp"), "PN": ("pn_",), "PLANETS": ("planet", "third_body")}
SHARED_G1 = ("declared_parameters", "with_term")


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


# ======================================================================
# population
# ======================================================================
def anti_sun_angle(et: float) -> float:
    """Angle between the geocentric Moon and the anti-Sun direction (geometric DE440, Earth-centred J2000), rad."""
    import spiceypy as spice
    m = np.asarray(spice.spkpos("MOON", float(et), "J2000", "NONE", "EARTH")[0], dtype=float)
    s = -np.asarray(spice.spkpos("SUN", float(et), "J2000", "NONE", "EARTH")[0], dtype=float)
    return math.atan2(float(np.linalg.norm(np.cross(m, s))), float(np.dot(m, s)))


def greatest_eclipse_et(date: str) -> float:
    """t_ge rule: 60 s grid from 00:00 UTC of the day before to 00:00 UTC of the day after the date, then
    bounded minimisation +/- 60 s around the grid minimum to 1 s."""
    import spiceypy as spice
    from scipy.optimize import minimize_scalar
    day = dt.date.fromisoformat(date)
    lo = float(spice.str2et(f"{day - dt.timedelta(days=1)}T00:00:00"))
    hi = float(spice.str2et(f"{day + dt.timedelta(days=1)}T00:00:00"))
    grid = np.arange(lo, hi + 0.5, 60.0)
    k = int(np.argmin([anti_sun_angle(e) for e in grid]))
    res = minimize_scalar(anti_sun_angle, bounds=(grid[k] - 60.0, grid[k] + 60.0), method="bounded",
                          options={"xatol": 1.0})
    return float(res.x)


def epoch_et0(epoch: str) -> tuple[float, dict]:
    import spiceypy as spice
    if epoch == "E1":
        et0 = float(spice.str2et("2027-03-02T00:00:00"))
        return et0, dict(utc0=spice.et2utc(et0, "ISOC", 3))
    t_ge = greatest_eclipse_et(ECLIPSE_DATES[epoch])
    et0 = t_ge - 43200.0
    return et0, dict(utc0=spice.et2utc(et0, "ISOC", 3), t_ge_et=t_ge, t_ge_utc=spice.et2utc(t_ge, "ISOC", 1),
                     anti_sun_angle_deg_at_t_ge=math.degrees(anti_sun_angle(t_ge)))


def initial_state(orbit: str, et0: float, sun0, mu_moon: float) -> tuple[np.ndarray, dict]:
    from lunar_od.constants import R_MOON_M
    from lunar_od.lunar_frames import moon_pa_de440_rotation_at_et
    p = moon_pa_de440_rotation_at_et(et0)[2]                 # MOON_PA_DE440 z-axis in J2000
    s = _unit(sun0)
    if orbit == "O1":
        n = _unit(np.cross(p, s))
    elif orbit == "O2":
        n = _unit(s - float(np.dot(s, p)) * p)
    else:
        n = p
    c = np.cross(n, p)
    r_hat = _unit(c) if float(np.linalg.norm(c)) > 0.1 else _unit(np.cross(n, [1.0, 0.0, 0.0]))
    r0 = R_MOON_M + ALTITUDE_M
    x0 = np.concatenate([r0 * r_hat, math.sqrt(mu_moon / r0) * np.cross(n, r_hat)])
    return x0, dict(pole=p.tolist(), normal=n.tolist(), beta_deg=math.degrees(math.asin(float(np.dot(n, s)))))


@dataclass
class Cell:
    name: str
    epoch: str
    et0: float
    x0: np.ndarray
    eph: object
    mus: tuple
    srp: object
    gm_planets: dict
    info: dict


def build_cell(orbit: str, epoch: str, et0: float, epoch_info: dict) -> Cell:
    from lunar_od.ephemeris import sample_moon_centered_ephemeris
    from lunar_od.ephemeris_generation import gravitational_parameters
    from lunar_od.srp import SRPOptions
    mus = gravitational_parameters("DE440")
    t_eph = np.arange(-EPH_MARGIN_S, float(T_GRID[-1]) + EPH_MARGIN_S + 0.5, EPH_STEP_S)
    eph = sample_moon_centered_ephemeris(et0, t_eph)
    x0, orbit_info = initial_state(orbit, et0, eph.sun_position(0.0), mus[0])
    return Cell(f"{orbit}x{epoch}", epoch, et0, x0, eph, mus, SRPOptions(k_srp_m2_per_kg=W.K_SRP_M2_PER_KG),
                W.planet_gm(), dict(epoch_info, **orbit_info))


# ======================================================================
# dynamics
# ======================================================================
def closure(cell: Cell):
    """A fresh production 6-state closure (and its Sun lookup), as propagate_state builds it."""
    from lunar_od import dynamics as D
    return D._build_state_rhs(T_GRID, *cell.mus, cell.eph.earth_position, cell.eph.sun_position,
                              0.0, 0.0, "indirect", None, None, None, cell.srp)


def integrate(rhs, sun, srp, x0, rtol: float = RTOL, atol: float = ATOL):
    """propagate_state's DOP853 path under PROFILE: P0 one solve_ivp run, D2 the shadow-event segmented run.

    Returns (states, count): the RHS evaluations (P0) or the penumbra cuts (D2). In D2 the contact pre-pass
    integrates the RHS it is given, so a term run is cut at its own trajectory's contacts.
    """
    from lunar_od import dynamics as D
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    if PROFILE == "D2":
        cuts = D._shadow_contact_times(rhs, T_GRID, x0, sun, srp, rtol, atol)
        states = D._integrate_segmented(T_GRID, x0, rhs, rtol, atol, "DOP853", cuts, "WP3.1-R propagation",
                                        contact_check=D._shadow_contact_check(sun, srp))
        return states, len(cuts)
    from scipy.integrate import solve_ivp
    sol = solve_ivp(rhs, (float(T_GRID[0]), float(T_GRID[-1])), x0, method="DOP853", t_eval=T_GRID,
                    rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"propagation failed: {sol.message}")
    return sol.y.T, int(sol.nfev)


def context(cell: Cell, sun) -> W.Context:
    return W.Context(cell.et0, *cell.mus, cell.eph.earth_position, sun, cell.srp, cell.gm_planets)


def run_base(cell: Cell, rtol: float = RTOL, atol: float = ATOL):
    rhs, sun = closure(cell)
    return integrate(rhs, sun, cell.srp, cell.x0, rtol, atol)


def run_term(cell: Cell, term: str, scale: float):
    rhs, sun = closure(cell)
    return integrate(W.with_term(rhs, term, context(cell, sun), scale), sun, cell.srp, cell.x0)


def production(cell: Cell, x0, k_srp: float) -> np.ndarray:
    from lunar_od.dynamics import propagate_state
    from lunar_od.srp import SRPOptions
    return propagate_state(T_GRID, x0, *cell.mus, cell.eph.earth_position, cell.eph.sun_position,
                           rtol=RTOL, atol=ATOL, method="DOP853", srp=SRPOptions(k_srp_m2_per_kg=k_srp),
                           shadow_events=PROFILE == "D2")


# ======================================================================
# measurements and OD impact
# ======================================================================
def measurement_rows(cell: Cell, states):
    import spiceypy as spice
    from lunar_od.config import range_rate_stations
    from lunar_od.constants import R_MOON_M
    from lunar_od.measurements import generate_range_rate_measurements
    from lunar_od.visibility import (VisibilityConfig, analyze_visibility_gap_with_transforms,
                                     sample_j2000_to_itrf93_transforms)
    by_name = {s.name: s for s in range_rate_stations()}
    stations = tuple(by_name[n] for n in STATIONS)
    xf = sample_j2000_to_itrf93_transforms(cell.et0, T_GRID)
    cfg = VisibilityConfig(r_moon_mean_m=R_MOON_M, earth_rotation_rad_s=7.292115e-5,
                           epoch_utc=spice.et2utc(cell.et0, "ISOC", 3), min_elevation_deg=10.0)
    _, _, vis, _ = analyze_visibility_gap_with_transforms(T_GRID, states, stations, cell.eph.earth_position,
                                                          xf, 0.0, cfg)
    vis = np.asarray(vis, dtype=bool).reshape(T_GRID.size, len(stations))
    obs, pass_geo = generate_range_rate_measurements(T_GRID, states, stations, vis, cell.eph.earth_position,
                                                     cell.eph.earth_velocity, cell.et0, noise=False)
    return obs, pass_geo, vis


def observables(states, obs, pass_geo) -> np.ndarray:
    """Production instantaneous (range, range-rate) values on fixed rows (values, not the zeroed residuals)."""
    from lunar_od.measurements import compute_range_rate_residuals
    return np.asarray(compute_range_rate_residuals(states, obs, pass_geo)[1], dtype=float)[:, :2]


def design(cell: Cell, states, obs, pass_geo):
    """(n, 2, 7) range / range-rate rows of H_tilde [Phi | dx/dK] along the baseline, and the 48-state run."""
    from lunar_od.dynamics import propagate_state_with_k_sensitivity
    from lunar_od.measurements import compute_range_rate_residuals_analytic
    nom = propagate_state_with_k_sensitivity(T_GRID, cell.x0, *cell.mus, cell.eph.earth_position,
                                             cell.eph.sun_position, srp=cell.srp, rtol=RTOL, atol=ATOL,
                                             method="DOP853", shadow_events=PROFILE == "D2")
    ht = np.asarray(compute_range_rate_residuals_analytic(states, obs, pass_geo)[2], dtype=float)
    ht = ht.reshape(-1, 4, 6)[:, :2, :]
    k = obs[:, 6].astype(int) - 1
    aug = np.stack([np.hstack([row[6:42].reshape((6, 6), order="F"), row[42:48, None]]) for row in nom[k]])
    return np.einsum("nij,njk->nik", ht, aug), nom


def fit(h, dy, sigma_range: float, sigma_rr: float, use_rr: bool) -> dict:
    """QR of the stacked whitened design with the x0 prior (no normal matrix): delta_theta, sigma, z."""
    from scipy.linalg import solve_triangular
    a_rows, b_rows = [h[:, 0, :] / sigma_range], [dy[:, 0] / sigma_range]
    if use_rr:
        a_rows.append(h[:, 1, :] / sigma_rr)
        b_rows.append(dy[:, 1] / sigma_rr)
    n_meas = int(sum(b.size for b in b_rows))
    prior = np.hstack([np.diag(1.0 / PRIOR_SD), np.zeros((6, 1))])
    a = np.vstack(a_rows + [prior])
    b = np.concatenate(b_rows + [np.zeros(6)])
    col = np.linalg.norm(a, axis=0)
    q, r = np.linalg.qr(a / col)
    theta = solve_triangular(r, q.T @ b) / col
    sigma = np.linalg.norm(solve_triangular(r, np.eye(7)), axis=1) / col
    z = theta / sigma
    post = b[:n_meas] - a[:n_meas] @ theta
    return dict(delta_theta=theta.tolist(), sigma=sigma.tolist(), z=z.tolist(), response=float(np.max(np.abs(z))),
                argmax=PARAMS[int(np.argmax(np.abs(z)))], rows=n_meas, cond_scaled=float(np.linalg.cond(r)),
                prefit_rms_sigma=float(np.sqrt(np.mean(b[:n_meas] ** 2))),
                postfit_rms_sigma=float(np.sqrt(np.mean(post ** 2))))


def g5_design_check(cell: Cell, obs, pass_geo, h) -> dict:
    """Central differences of production propagation + measurement against every design column."""
    rel = {}
    for j, step in enumerate(FD_STEPS):
        ys = []
        for sign in (1.0, -1.0):
            x0, k = cell.x0.copy(), W.K_SRP_M2_PER_KG
            if j < 6:
                x0[j] += sign * step
            else:
                k += sign * step
            ys.append(observables(production(cell, x0, k), obs, pass_geo))
        fd = (ys[0] - ys[1]) / (2.0 * step)
        for t, name in ((0, "range"), (1, "range_rate")):
            rel[f"{PARAMS[j]}/{name}"] = float(np.linalg.norm(fd[:, t] - h[:, t, j]) / np.linalg.norm(h[:, t, j]))
    worst = max(rel.values())
    return dict(relative=rel, worst=worst, PASS=bool(worst <= G5_MAX), steps=list(FD_STEPS))


# ======================================================================
# one cell
# ======================================================================
def run_cell(cell: Cell, do_g5: bool) -> dict:
    t_start = time.time()
    zero = np.zeros(3)
    x_base, count = run_base(cell)
    g3 = bool(np.array_equal(production(cell, cell.x0, W.K_SRP_M2_PER_KG), x_base))
    x_fine, _ = run_base(cell, *FLOOR_TOL)
    floor = float(np.max(np.linalg.norm(x_fine[:, :3] - x_base[:, :3], axis=1)))
    obs, pass_geo, vis = measurement_rows(cell, x_base)
    h, nom = design(cell, x_base, obs, pass_geo)
    y_base = observables(x_base, obs, pass_geo)
    sun, earth = cell.eph.sun_position(T_GRID), cell.eph.earth_position(T_GRID)
    nu_moon = np.array([W.lunar_illumination(x[:3], s, zero, cell.srp) for x, s in zip(x_base, sun)])
    nu_earth_sc = np.array([W.earth_illumination(x[:3], s, e) for x, s, e in zip(x_base, sun, earth)])
    nu_earth_moon = np.array([W.earth_illumination(zero, s, e) for s, e in zip(sun, earth)])
    out = dict(cell=cell.name, epoch=cell.epoch, et0=cell.et0, x0=cell.x0.tolist(), info=cell.info,
               G3=g3, floor_m=floor, G4=bool(floor <= FLOOR_MAX_M),
               **{"baseline_nfev" if PROFILE == "P0" else "baseline_cuts": count},
               rows=int(obs.shape[0]), rows_per_station=np.sum(vis, axis=0).tolist(),
               visible_epoch_fraction=float(np.mean(np.any(vis, axis=1))),
               nom48_vs_baseline_m=float(np.max(np.linalg.norm(nom[:, :3] - x_base[:, :3], axis=1))),
               lunar_shadow=dict(fraction_nu_below_1=float(np.mean(nu_moon < 1.0)),
                                 fraction_umbra=float(np.mean(nu_moon == 0.0))),
               earth_shadow_sc=dict(fraction_nu_below_1=float(np.mean(nu_earth_sc < 1.0)),
                                    min_nu=float(nu_earth_sc.min())),
               earth_shadow_moon_centre=dict(min_nu=float(nu_earth_moon.min()),
                                             any_umbra=bool(np.any(nu_earth_moon == 0.0)),
                                             any_below_half=bool(np.any(nu_earth_moon < 0.5)),
                                             all_lit=bool(np.all(nu_earth_moon == 1.0)),
                                             epochs_below_1=int(np.sum(nu_earth_moon < 1.0))))
    print(f"[{PROFILE} {cell.name}] baseline count {count}, G3 {g3}, floor {floor:.3e} m, rows {obs.shape[0]}",
          flush=True)
    terms = {}
    for term in RUN_TERMS:
        t_term = time.time()
        x_term, count_term = run_term(cell, term, 1.0)
        x_null, _ = run_term(cell, term, 0.0)
        dr = np.linalg.norm(x_term[:, :3] - x_base[:, :3], axis=1)
        dy = observables(x_term, obs, pass_geo) - y_base
        acc = W.along(term, T_GRID, x_base, context(cell, cell.eph.sun_position))
        acc_n = np.linalg.norm(acc, axis=1)
        fits = {f"{d}/{n}": fit(h, dy, *NOISE[n], d == "RANGE_RANGE_RATE") for d in DESIGNS for n in NOISE}
        best = max(fits, key=lambda k: fits[k]["response"])
        terms[term] = dict(G2=bool(np.array_equal(x_null, x_base)), max_dr_m=float(dr.max()),
                           final_dr_m=float(dr[-1]), below_floor=bool(dr.max() < 10.0 * floor),
                           max_dy_range_m=float(np.max(np.abs(dy[:, 0]))),
                           max_dy_range_rate_mps=float(np.max(np.abs(dy[:, 1]))),
                           accel_rms_mps2=float(np.sqrt(np.mean(acc_n ** 2))), accel_max_mps2=float(acc_n.max()),
                           response=fits[best]["response"], response_at=best, fits=fits, count=count_term,
                           wall_s=time.time() - t_term)
        print(f"[{PROFILE} {cell.name}] {term}: max|dr| {dr.max():.3e} m, response {fits[best]['response']:.3e} "
              f"({best}), G2 {terms[term]['G2']}, {time.time() - t_term:.0f} s", flush=True)
    out["terms"] = terms
    if do_g5:
        out["G5"] = g5_design_check(cell, obs, pass_geo, h)
        print(f"[{PROFILE} {cell.name}] G5 worst {out['G5']['worst']:.3e}", flush=True)
    out["wall_s"] = time.time() - t_start
    return out


def run(epoch: str, out_path: str, orbits=ORBITS, smoke: bool = False) -> None:
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels()
    et0, epoch_info = epoch_et0(epoch)
    cells = []
    for orbit in orbits:
        cell = build_cell(orbit, epoch, et0, epoch_info)
        cells.append(run_cell(cell, do_g5=(cell.name == G5_CELL)))
        Path(out_path).write_text(json.dumps(dict(profile=PROFILE, epoch=epoch, smoke=smoke,
                                                  arc_s=float(T_GRID[-1]), cells=cells), indent=1),
                                  encoding="utf-8")


# ======================================================================
# gates and decisions
# ======================================================================
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def g1_unit() -> dict:
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rfE", *G1_TESTS],
                          cwd=ROOT, capture_output=True, text=True)
    failed = [ln.split(" ", 1)[1] for ln in proc.stdout.splitlines() if ln.startswith(("FAILED ", "ERROR "))]
    per_term = {}
    for term, keys in G1_KEYWORDS.items():
        hit = [f for f in failed if any(k in f for k in keys + SHARED_G1)]
        per_term[term] = not hit
    summary = [ln for ln in proc.stdout.splitlines() if " passed" in ln or " failed" in ln]
    return dict(PASS=proc.returncode == 0, per_term=per_term, failed=failed,
                summary=summary[-1] if summary else proc.stdout[-500:], tests=list(G1_TESTS))


def decide(cells: list, g1: dict) -> dict:
    g2 = all(t["G2"] for c in cells for t in c["terms"].values())
    g3 = all(c["G3"] for c in cells)
    g5_cell = [c for c in cells if c["cell"] == G5_CELL]
    g5 = bool(g5_cell and g5_cell[0]["G5"]["PASS"])
    by_epoch = {e: [c for c in cells if c["epoch"] == e] for e in EPOCHS}
    g6 = dict(E1_all_lit=all(c["earth_shadow_moon_centre"]["all_lit"] for c in by_epoch["E1"]),
              E2_umbra=any(c["earth_shadow_moon_centre"]["any_umbra"] for c in by_epoch["E2"]),
              E3_below_half=any(c["earth_shadow_moon_centre"]["any_below_half"] for c in by_epoch["E3"]))
    g6["PASS"] = all(g6.values())
    common = g2 and g3 and g5
    decisions, summary = {}, {}
    for term in W.TERMS:
        rows = []
        for c in cells:
            if term == "EARTH_SHADOW" and c["epoch"] == "E1":
                continue                                  # zero by construction outside eclipses
            t = c["terms"][term]
            inconclusive = (not c["G4"]) and t["max_dr_m"] < 100.0 * c["floor_m"]
            rows.append(dict(cell=c["cell"], response=t["response"], at=t["response_at"], max_dr_m=t["max_dr_m"],
                             below_floor=t["below_floor"], inconclusive_in_cell=inconclusive))
        usable = [r for r in rows if not r["inconclusive_in_cell"]]
        top = max(usable, key=lambda r: r["response"]) if usable else None
        material = bool(top and top["response"] >= THRESHOLD)
        if not (g1["per_term"][term] and common) or (term == "EARTH_SHADOW" and not g6["PASS"]):
            decision = "WP3_1_INCONCLUSIVE"
        elif material:
            decision = "IMPLEMENT"
        elif len(usable) < len(rows):
            decision = "WP3_1_INCONCLUSIVE"
        else:
            decision = "DOCUMENT_ONLY"
        if term == "EARTH_SHADOW" and decision != "WP3_1_INCONCLUSIVE":
            decision += " (eclipse arcs)"
        decisions[term] = decision
        summary[term] = dict(OD_MATERIAL=material, max_response=top["response"] if top else None,
                             at=f"{top['cell']} {top['at']}" if top else None,
                             max_dr_m=max(r["max_dr_m"] for r in rows),
                             accel_rms_mps2_max=max(c["terms"][term]["accel_rms_mps2"] for c in cells),
                             cells=rows)
    reported = {k: dict(max_response=max(c["terms"][k]["response"] for c in cells),
                        max_dr_m=max(c["terms"][k]["max_dr_m"] for c in cells)) for k in W.REPORTED}
    gates = dict(G1_UNIT=g1["PASS"], G2_NULL=g2, G3_PARITY=g3, G4_FLOOR={c["cell"]: c["G4"] for c in cells},
                 G5_DESIGN=g5, G6_ECLIPSE=g6["PASS"])
    return dict(gates=gates, G1=g1, G6=g6, decisions=decisions, summary=summary, reported=reported)


def combine(parts: list) -> None:
    import scipy
    loaded = [json.loads(Path(p).read_text(encoding="utf-8")) for p in parts]
    if any(p.get("smoke") for p in loaded):
        raise ValueError("a smoke part cannot enter the survey result")
    if any(p.get("profile") != PROFILE for p in loaded):
        raise ValueError(f"every part must be profile {PROFILE}")
    cells = [c for p in loaded for c in p["cells"]]
    names = sorted(c["cell"] for c in cells)
    if names != sorted(f"{o}x{e}" for o in ORBITS for e in EPOCHS):
        raise ValueError(f"the nine cells are required, got {names}")
    g1 = g1_unit()
    status = ("as predeclared (shadow_events False)" if PROFILE == "P0" else
              "amendment A2, exploratory: production PropagationProfile() (DOP853, shadow_events True)")
    result = dict(predeclaration=str(PRE.relative_to(ROOT)).replace("\\", "/"), predeclaration_sha256=_sha256(PRE),
                  profile=PROFILE, status=status,
                  commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                        text=True).stdout.strip(),
                  environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                  threshold=THRESHOLD, noise=NOISE, prior_sd=PRIOR_SD.tolist(), parameters=PARAMS,
                  **decide(cells, g1), cells=cells)
    OUT[PROFILE].write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("profile", "gates", "decisions")}, indent=1))


def main(argv: list) -> None:
    global T_GRID, PROFILE
    mode, PROFILE = argv[1], argv[2]
    if PROFILE not in PROFILES:
        raise SystemExit(__doc__)
    if mode == "run":
        run(argv[3], argv[4])
    elif mode == "smoke":
        T_GRID = np.arange(0.0, 3600.0 + 30.0, 60.0)
        run("E1", argv[3], orbits=("O1",), smoke=True)
    elif mode == "combine":
        combine(argv[3:])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
