"""WP4.2-R survey before any gate: BLS-LM with and without the early-iteration loosening (monkeypatch), no code change.

Arcs 6 h and 24 h (three DSN complexes, 60 s, mask + occultation); measurement types position (one_way_light_time),
range_rate (apparent_one_way companion) and two_way_range; noise-free and seed 2000. Records position error,
status, iterations, final cost against the cost at the truth.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ARGS = sys.argv[1:]
sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).parent))
import wp2_5_long_arc_compare as L  # noqa: E402
from lunar_od.measurements import compute_position_residuals, compute_range_rate_residuals, measurement_sigma_vector  # noqa: E402
from lunar_od.two_way_range import compute_two_way_range_residuals, generate_two_way_range_measurements  # noqa: E402

HOURS = [float(h) for h in ARGS[0].split(",")] if ARGS else [6.0, 24.0]
OUT = Path(ARGS[1]) if len(ARGS) > 1 else Path(__file__).with_name("wp42r_survey.json")
original = L.est._adaptive_tol
out = {}


def data(a, kind, noise):
    rng = np.random.default_rng(2000)
    if kind == "position":
        return L.position(a, L.IMPLICIT, noise, rng)
    if kind == "range_rate":
        return L.generate_range_rate_measurements(a["t"], a["aug"][:, :6], L.STATIONS, a["vis"], a["eph"].earth_position,
                                                  a["eph"].earth_velocity, L.et0, noise=noise, rng=rng, arc_id=1,
                                                  companion_geometry="apparent_one_way")
    return generate_two_way_range_measurements(a["t"], a["aug"][:, :6], L.STATIONS, a["vis"], a["eph"].earth_position,
                                               a["eph"].earth_velocity, L.et0, noise=noise, rng=rng, arc_id=1)[:2]


def cost(a, kind, x, obs, geo):
    f = {"position": compute_position_residuals, "range_rate": compute_range_rate_residuals,
         "two_way_range": compute_two_way_range_residuals}[kind]
    r = np.asarray(f(a["propagate"](x), obs, geo)[0]) / measurement_sigma_vector(obs, geo)
    return float(r @ r)


for hours in HOURS:
    a = L.setup(hours)
    for kind in ("position", "range_rate", "two_way_range"):
        for noise in (False, True):
            obs, geo = data(a, kind, noise)
            run = getattr(L.est, f"estimate_{kind}_bls_lm")
            for loosening in ("on", "off"):
                L.est._adaptive_tol = original if loosening == "on" else (lambda i, m, r, at: (r, at))
                t0 = time.perf_counter()
                x, status, stats = run(a["t"], obs, L.start, geo, *L.gm, a["eph"].earth_position,
                                       a["eph"].sun_position, rtol=1e-12, atol=1e-13)
                row = {"pos_err_m": float(np.linalg.norm(np.asarray(x[:3]) - L.x0[:3])), "status": str(status),
                       "detail": str(getattr(stats, "termination_detail", "")), "iterations": int(stats.iterations),
                       "cost": cost(a, kind, x, obs, geo), "cost_truth": cost(a, kind, L.x0, obs, geo),
                       "rows": int(obs.shape[0]), "wall_s": time.perf_counter() - t0}
                key = f"{hours}h/{kind}/{'noisy' if noise else 'clean'}/loosening_{loosening}"
                out[key] = row
                print(key, json.dumps(row), flush=True)
            L.est._adaptive_tol = original
            OUT.write_text(json.dumps(out, indent=1))
