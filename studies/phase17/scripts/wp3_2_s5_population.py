"""F3 WP3.2 S5: the hold-out population of the replication (geometry only, no OD number).

Predeclaration: studies/phase17/configs/wp3_2_s5_replication_predeclaration.json (section "population", the
erratum before any result, and gate S5_G6). The draws come from one seeded generator in the predeclared order:

1. two nominal epochs (no Earth shadow on the Moon centre during the arc, > 3 days from each other and from
   every catalogue eclipse);
2. two TOTAL eclipses, then two NON_TOTAL eclipses, from the catalogue of 2027-04-01 .. 2029-12-31;
3. the four eclipse arc offsets (t0 = t_ge - offset, offset in [6 h, 18 h]);
4. for each epoch and beta stratum: |beta|, sign(beta), node angle Omega, argument of latitude u, altitude.

    python wp3_2_s5_population.py            # -> studies/phase17/results/wp3_2_s5_population.json
"""
from __future__ import annotations

import datetime as dt
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

PRE = ROOT / "studies" / "phase17" / "configs" / "wp3_2_s5_replication_predeclaration.json"
OUT = ROOT / "studies" / "phase17" / "results" / "wp3_2_s5_population.json"

SEED = 20261127
WINDOW_UTC = ("2027-04-01T00:00:00", "2029-12-31T00:00:00")
NOMINAL_END_UTC = "2029-12-30T00:00:00"
STRATA_DEG = ((0.0, 30.0), (30.0, 60.0), (60.0, 90.0))
ALTITUDE_RANGE_M = (50e3, 200e3)
ECLIPSE_OFFSET_RANGE_S = (6 * 3600.0, 18 * 3600.0)
ARC_S, STEP_S = 86400.0, 60.0
MIN_SEPARATION_S = 3 * 86400.0
#: WP3.1's population (studies/phase17/reports/wp3_1_materiality_survey.md section 1), excluded by S5_G6
WP3_1_EPOCHS_UTC = ("2027-03-02T00:00:00", "2026-03-02T23:34:14", "2026-08-27T16:13:28")
WP3_1_ECLIPSE_DATES = ("2026-03-03", "2026-08-28")
WP3_1_ALTITUDE_M = 100e3
EPOCH_KINDS = {"H1": "NOMINAL", "H2": "NOMINAL", "H3": "TOTAL", "H4": "TOTAL", "H5": "NON_TOTAL",
               "H6": "NON_TOTAL"}
CATALOGUE_GRID_S = 3600.0
CATALOGUE_MAX_ANGLE_DEG = 2.0          # a full moon farther than this from the anti-Sun point cannot be eclipsed


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


# ======================================================================
# geometry (pure)
# ======================================================================
def orbit_normal(beta: float, omega: float, s, p) -> np.ndarray:
    """n = sin(beta) s + cos(beta) (cos(Omega) e1 + sin(Omega) e2), e1 = unit(p - (p.s) s), e2 = s x e1."""
    s, p = _unit(s), _unit(p)
    e1 = _unit(p - float(np.dot(p, s)) * s)
    e2 = np.cross(s, e1)
    return _unit(math.sin(beta) * s + math.cos(beta) * (math.cos(omega) * e1 + math.sin(omega) * e2))


def initial_state(n, p, u: float, altitude_m: float, mu_moon: float) -> np.ndarray:
    """Circular state: r_hat = cos(u) r_hat0 + sin(u) (n x r_hat0), r_hat0 = unit(n x p) (or n x x_J2000)."""
    from lunar_od.constants import R_MOON_M
    n, p = _unit(n), _unit(p)
    c = np.cross(n, p)
    r_hat0 = _unit(c) if float(np.linalg.norm(c)) > 0.1 else _unit(np.cross(n, [1.0, 0.0, 0.0]))
    r_hat = math.cos(u) * r_hat0 + math.sin(u) * np.cross(n, r_hat0)
    r0 = R_MOON_M + float(altitude_m)
    return np.concatenate([r0 * r_hat, math.sqrt(mu_moon / r0) * np.cross(n, r_hat)])


def draw_cell(rng, stratum: int) -> dict:
    """One cell's draws, in the predeclared order: |beta|, sign(beta), Omega, u, altitude."""
    lo, hi = STRATA_DEG[stratum]
    abs_beta = float(rng.uniform(lo, hi))
    sign = 1.0 if float(rng.random()) < 0.5 else -1.0
    omega = float(rng.uniform(0.0, 2 * math.pi))
    u = float(rng.uniform(0.0, 2 * math.pi))
    altitude = float(rng.uniform(*ALTITUDE_RANGE_M))
    return dict(beta_deg=sign * abs_beta, omega_rad=omega, u_rad=u, altitude_m=altitude)


# ======================================================================
# independence (gate S5_G6, pure)
# ======================================================================
def independence(pop: dict) -> dict:
    epochs, cells = pop["epochs"], pop["cells"]
    start = dt.datetime.fromisoformat(WINDOW_UTC[0])
    checks = {}
    checks["epochs_in_window"] = all(dt.datetime.fromisoformat(e["t0_utc"][:19]) >= start for e in epochs.values())
    wp31 = {dt.datetime.fromisoformat(u) for u in WP3_1_EPOCHS_UTC}
    checks["no_wp3_1_epoch"] = all(dt.datetime.fromisoformat(e["t0_utc"][:19]) not in wp31 for e in epochs.values())
    checks["no_wp3_1_eclipse"] = all(e.get("t_ge_utc") is None or e["t_ge_utc"][:10] not in WP3_1_ECLIPSE_DATES
                                     for e in epochs.values())
    checks["altitudes_differ_from_100_km"] = all(abs(c["altitude_m"] - WP3_1_ALTITUDE_M) > 1.0 for c in cells)
    ets = sorted(float(e["et0"]) for e in epochs.values())
    checks["epochs_pairwise_over_3_days"] = all(b - a > MIN_SEPARATION_S for a, b in zip(ets, ets[1:]))
    kinds = sorted(e["kind"] for e in epochs.values())
    checks["epoch_kinds"] = kinds == sorted(EPOCH_KINDS.values())
    checks["eighteen_cells"] = len(cells) == 18 and len({c["name"] for c in cells}) == 18
    return dict(checks=checks, PASS=all(checks.values()))


# ======================================================================
# SPICE parts
# ======================================================================
def _moon_centred(name: str, et: float) -> np.ndarray:
    import spiceypy as spice
    return np.asarray(spice.spkpos(name, float(et), "J2000", "NONE", "MOON")[0], dtype=float) * 1e3


def moon_centre_nu_e(et: float) -> float:
    import wp3_1_force_terms as W
    return W.earth_illumination(np.zeros(3), _moon_centred("SUN", et), _moon_centred("EARTH", et))


def refine_t_ge(et_guess: float) -> float:
    """WP3.1's t_ge rule around a guess: 60 s grid over +/- 1 day, bounded minimisation +/- 60 s to 1 s."""
    from scipy.optimize import minimize_scalar
    from wp3_1_materiality_survey import anti_sun_angle
    grid = np.arange(et_guess - 86400.0, et_guess + 86400.0 + 0.5, 60.0)
    k = int(np.argmin([anti_sun_angle(e) for e in grid]))
    res = minimize_scalar(anti_sun_angle, bounds=(grid[k] - 60.0, grid[k] + 60.0), method="bounded",
                          options={"xatol": 1.0})
    return float(res.x)


def eclipse_catalogue() -> list:
    import spiceypy as spice
    from wp3_1_materiality_survey import anti_sun_angle
    lo, hi = (float(spice.str2et(u)) for u in WINDOW_UTC)
    grid = np.arange(lo, hi + 0.5, CATALOGUE_GRID_S)
    ang = np.array([anti_sun_angle(e) for e in grid])
    out = []
    for i in range(1, ang.size - 1):
        if not (ang[i] <= ang[i - 1] and ang[i] < ang[i + 1]):
            continue
        if math.degrees(ang[i]) > CATALOGUE_MAX_ANGLE_DEG:
            continue
        t_ge = refine_t_ge(float(grid[i]))
        nu = moon_centre_nu_e(t_ge)
        kind = "TOTAL" if nu == 0.0 else ("NON_TOTAL" if nu < 1.0 else None)
        out.append(dict(t_ge_et=t_ge, t_ge_utc=spice.et2utc(t_ge, "ISOC", 1),
                        anti_sun_angle_deg=math.degrees(anti_sun_angle(t_ge)), nu_e_moon_centre=nu, kind=kind))
    return out


def arc_moon_nu(et0: float) -> np.ndarray:
    return np.array([moon_centre_nu_e(et0 + t) for t in np.arange(0.0, ARC_S + 0.5, STEP_S)])


def draw_nominal(rng, chosen: list, eclipse_ets: list, log: list) -> float:
    import spiceypy as spice
    lo, hi = float(spice.str2et(WINDOW_UTC[0])), float(spice.str2et(NOMINAL_END_UTC))
    while True:
        et0 = lo + math.floor(float(rng.uniform(0.0, hi - lo)))
        utc = spice.et2utc(et0, "ISOC", 0)
        if any(abs(et0 - c) <= MIN_SEPARATION_S for c in chosen):
            log.append(dict(t0_utc=utc, rejected="within 3 days of a chosen epoch"))
            continue
        if any(abs(et0 - e) <= MIN_SEPARATION_S for e in eclipse_ets):
            log.append(dict(t0_utc=utc, rejected="within 3 days of a catalogue eclipse"))
            continue
        nu = arc_moon_nu(et0)
        if float(nu.min()) < 1.0:
            log.append(dict(t0_utc=utc, rejected=f"Moon centre nu_E {nu.min():.3f} < 1 in the arc"))
            continue
        log.append(dict(t0_utc=utc, accepted=True))
        return et0


def build() -> dict:
    import spiceypy as spice
    from lunar_od.ephemeris_generation import gravitational_parameters
    from lunar_od.lunar_frames import moon_pa_de440_rotation_at_et
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels()
    rng = np.random.default_rng(SEED)
    catalogue = eclipse_catalogue()
    eclipse_ets = [c["t_ge_et"] for c in catalogue if c["kind"]]
    log: list = []
    nominal = []
    for _ in range(2):
        nominal.append(draw_nominal(rng, nominal, eclipse_ets, log))
    nominal.sort()
    totals = [c for c in catalogue if c["kind"] == "TOTAL"]
    partials = [c for c in catalogue if c["kind"] == "NON_TOTAL"]
    pick_t = sorted(int(i) for i in rng.choice(len(totals), size=2, replace=False))
    pick_p = sorted(int(i) for i in rng.choice(len(partials), size=2, replace=False))
    chosen = [totals[i] for i in pick_t] + [partials[i] for i in pick_p]
    offsets = [float(round(rng.uniform(*ECLIPSE_OFFSET_RANGE_S))) for _ in chosen]
    epochs = {}
    for name, et0 in zip(("H1", "H2"), nominal):
        nu = arc_moon_nu(et0)
        epochs[name] = dict(kind="NOMINAL", et0=et0, t0_utc=spice.et2utc(et0, "ISOC", 3), t_ge_utc=None,
                            moon_centre_min_nu_e=float(nu.min()))
    for name, ecl, off in zip(("H3", "H4", "H5", "H6"), chosen, offsets):
        et0 = ecl["t_ge_et"] - off
        nu = arc_moon_nu(et0)
        epochs[name] = dict(kind=ecl["kind"], et0=et0, t0_utc=spice.et2utc(et0, "ISOC", 3), t_ge_et=ecl["t_ge_et"],
                            t_ge_utc=ecl["t_ge_utc"], offset_s=off, moon_centre_nu_e_at_t_ge=ecl["nu_e_moon_centre"],
                            moon_centre_min_nu_e=float(nu.min()))
    mu_moon = gravitational_parameters("DE440")[0]
    cells = []
    for name in ("H1", "H2", "H3", "H4", "H5", "H6"):
        et0 = epochs[name]["et0"]
        p = moon_pa_de440_rotation_at_et(et0)[2]
        s = _unit(_moon_centred("SUN", et0))
        for k in range(3):
            d = draw_cell(rng, k)
            n = orbit_normal(math.radians(d["beta_deg"]), d["omega_rad"], s, p)
            x0 = initial_state(n, p, d["u_rad"], d["altitude_m"], mu_moon)
            cells.append(dict(name=f"{name}xS{k + 1}", epoch=name, stratum=f"S{k + 1}", **d, normal=n.tolist(),
                              pole=p.tolist(), sun_unit=s.tolist(),
                              beta_check_deg=math.degrees(math.asin(float(np.dot(n, s)))),
                              inclination_to_lunar_equator_deg=math.degrees(math.acos(float(np.dot(n, p)))),
                              x0=x0.tolist()))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    pop = dict(predeclaration=str(PRE.relative_to(ROOT)).replace("\\", "/"), seed=SEED, commit=commit,
               status="geometry only, generated before any S5 survey result", window_utc=WINDOW_UTC,
               catalogue=catalogue, nominal_draws=log, picks=dict(total=pick_t, non_total=pick_p),
               epochs=epochs, cells=cells)
    pop["S5_G6_INDEPENDENCE"] = independence(pop)
    return pop


def main() -> None:
    pop = build()
    OUT.write_text(json.dumps(pop, indent=1), encoding="utf-8")
    for k, e in pop["epochs"].items():
        print(k, e["kind"], e["t0_utc"], e.get("t_ge_utc"), f"min nu_E {e['moon_centre_min_nu_e']:.3f}")
    print("S5_G6", pop["S5_G6_INDEPENDENCE"])


if __name__ == "__main__":
    main()
