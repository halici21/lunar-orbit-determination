"""WP2.5: implicit_light_time vs analytic_first_order_light_time on 1 h, 6 h and 24 h arcs.

Three DSN complexes, 10 deg mask plus lunar occultation, 60 s cadence. Per arc length:
noise-free Jacobian error against Richardson (position profile), then seeded noisy BLS-LM fits for
one_way_light_time position and for range-rate with the apparent_one_way companion, both models
interleaved on the same data. Run from the worktree.
"""
import json
from dataclasses import replace
import sys
import time
from pathlib import Path

import numpy as np
import spiceypy as spice

from lunar_od.config import Station
from lunar_od.dynamics import propagate_augmented_state
from lunar_od.ephemeris import sample_moon_centered_ephemeris
from lunar_od.ephemeris_generation import gravitational_parameters
from lunar_od.geometry import ecef2razel_sez
from lunar_od.spice_loader import load_spice_kernels
from lunar_od.visibility import is_occulted_by_moon
import lunar_od.estimators as est
from lunar_od.measurements import (compute_position_residuals, compute_position_residuals_analytic,
                                   generate_position_measurements, generate_range_rate_measurements,
                                   position_initial_state_jacobian_from_augmented_history)

LENGTHS_H = [float(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ("1", "6", "24"))]
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("long_arc_compare.json")
IMPLICIT, FIRST = "implicit_light_time", "analytic_first_order_light_time"
STATIONS = (Station("Goldstone", 35.4267, -116.8900, 1000.0, (0, 0, 0), 2.0, 1e-4, 1e-4),
            Station("Canberra", -35.4014, 148.9817, 690.0, (0, 0, 0), 2.0, 1e-4, 1e-4),
            Station("Madrid", 40.4314, -4.2481, 834.0, (0, 0, 0), 2.0, 1e-4, 1e-4))
R_MOON = 1737.4e3
load_spice_kernels(None, clear=False)
et0 = spice.str2et("2027-03-02T00:00:00")
gm = gravitational_parameters()
r0 = R_MOON + 100e3
x0 = np.array([r0, 30e3, -20e3, -15.0, np.sqrt(gm[0] / r0), 4.0])
start = x0 + np.array([80.0, -60.0, 40.0, 0.02, -0.01, 0.015])


def setup(hours):
    t = np.arange(0.0, hours * 3600.0 + 1.0, 60.0)
    eph = sample_moon_centered_ephemeris(et0, np.arange(-600.0, t[-1] + 601.0, 60.0))

    def propagate(x, augmented=False):
        h = propagate_augmented_state(t, np.concatenate([x, np.eye(6).reshape(-1, order="F")]), *gm,
                                      eph.earth_position, eph.sun_position, rtol=1e-12, atol=1e-13)
        return h if augmented else h[:, :6]

    aug = propagate(x0, True)
    earth = eph.earth_position(t)
    vis = np.zeros((t.size, len(STATIONS)), dtype=bool)
    for k in range(t.size):
        rot = np.asarray(spice.pxform("J2000", "ITRF93", et0 + t[k]))
        for j, st in enumerate(STATIONS):
            el = ecef2razel_sez(rot @ (aug[k, :3] - earth[k]) - st.r_ecef_m, st.lat_rad, st.lon_rad)[1]
            if el < np.radians(10.0):
                continue
            station_mci = earth[k] + rot.T @ st.r_ecef_m
            vis[k, j] = not bool(np.asarray(is_occulted_by_moon(station_mci, aug[k:k + 1, :3], np.zeros(3), R_MOON))[0])
    return dict(t=t, eph=eph, propagate=propagate, aug=aug, vis=vis)


def position(a, model, noise=False, rng=None):
    return generate_position_measurements(a["t"], a["aug"][:, :6], STATIONS, a["vis"], a["eph"].earth_position,
                                          a["eph"].earth_velocity, et0, noise=noise, rng=rng, arc_id=1,
                                          measurement_model_profile="one_way_light_time", jacobian_model=model)[:2]


def range_rate(a, model, rng):
    return generate_range_rate_measurements(a["t"], a["aug"][:, :6], STATIONS, a["vis"], a["eph"].earth_position,
                                            a["eph"].earth_velocity, et0, noise=True, rng=rng, arc_id=1,
                                            companion_geometry="apparent_one_way", jacobian_model=model)


def jacobian_error(a):
    obs, geo = position(a, IMPLICIT)

    def central(p, v):
        d = np.zeros((obs.shape[0], 3, 6))
        for j in range(6):
            e = np.zeros(6)
            e[j] = p if j < 3 else v
            delta = (compute_position_residuals(a["propagate"](x0 + e), obs, geo)[1]
                     - compute_position_residuals(a["propagate"](x0 - e), obs, geo)[1]).reshape(-1, 3)
            delta[:, 1] = (delta[:, 1] + np.pi) % (2 * np.pi) - np.pi
            d[:, :, j] = delta / (2 * e[j])
        return d

    ref = (4 * central(10.0, 0.01) - central(20.0, 0.02)) / 3
    out = {"rows": int(obs.shape[0])}
    for model in (IMPLICIT, FIRST):
        g = replace(geo, jacobian_model=model)
        _, _, tilde = compute_position_residuals_analytic(a["aug"][:, :6], obs, g)
        h = np.asarray(position_initial_state_jacobian_from_augmented_history(obs, a["aug"], tilde, g)).reshape(-1, 3, 6)
        rel = np.linalg.norm(h - ref, axis=2) / np.linalg.norm(ref, axis=2)
        out[model] = {"max": float(rel.max()), "median": float(np.median(rel))}
    return out


def fit(kind, a, obs, geo):
    run = getattr(est, f"estimate_{kind}_bls_lm")
    t0 = time.perf_counter()
    x, status, stats = run(a["t"], obs, start, geo, *gm, a["eph"].earth_position, a["eph"].sun_position,
                           rtol=1e-12, atol=1e-13, return_posterior=True)
    wall = time.perf_counter() - t0
    e = np.asarray(x[:6]) - x0
    cov = np.asarray(stats.posterior_covariance)[:6, :6]
    return {"pos_err_m": float(np.linalg.norm(e[:3])), "vel_err_mps": float(np.linalg.norm(e[3:])),
            "nees": float(e @ np.linalg.solve(cov, e)), "sigma_pos_rss_m": float(np.sqrt(np.trace(cov[:3, :3]))),
            "iterations": int(stats.iterations), "wall_s": wall, "status": str(status)}


def main():
    result = {"lengths_h": LENGTHS_H, "n_seeds": N_SEEDS, "stations": [s.name for s in STATIONS],
              "cadence_s": 60, "mask": "10 deg elevation and lunar occultation", "per_length": {}}
    for hours in LENGTHS_H:
        a = setup(hours)
        entry = {"visible_rows": int(a["vis"].sum()), "jacobian_vs_richardson": jacobian_error(a)}
        print(hours, "h jacobian", json.dumps(entry), flush=True)
        for kind in ("position", "range_rate"):
            rows = {IMPLICIT: [], FIRST: []}
            for seed in range(N_SEEDS):
                order = (IMPLICIT, FIRST) if seed % 2 == 0 else (FIRST, IMPLICIT)
                for model in order:
                    rng = np.random.default_rng(2000 + seed)
                    obs, geo = position(a, model, True, rng) if kind == "position" else range_rate(a, model, rng)
                    rows[model].append(fit(kind, a, obs, geo))
                print(hours, kind, seed, {m: round(rows[m][-1]["pos_err_m"], 3) for m in rows}, flush=True)
            entry[kind] = rows
        result["per_length"][str(hours)] = entry
        OUT.write_text(json.dumps(result, indent=1))
    print("done")


if __name__ == "__main__":
    main()
