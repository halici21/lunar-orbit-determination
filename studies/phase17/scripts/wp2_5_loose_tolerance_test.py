"""Controlled test: the 24 h BLS-LM fits with the early-iteration tolerance loosening switched off (monkeypatch)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import wp2_5_long_arc_compare as L  # noqa: E402
from lunar_od.dynamics import propagate_augmented_state  # noqa: E402
from lunar_od.measurements import compute_position_residuals, compute_range_rate_residuals, measurement_sigma_vector  # noqa: E402

a = L.setup(24.0)
out = {}
hist = {}
for r_tol, a_tol in ((1e-8, 1e-9), (1e-10, 1e-11), (1e-12, 1e-13)):
    hist[r_tol] = propagate_augmented_state(a["t"], np.concatenate([L.x0, np.eye(6).reshape(-1, order="F")]), *L.gm,
                                            a["eph"].earth_position, a["eph"].sun_position, rtol=r_tol, atol=a_tol)[:, :3]
out["propagation_difference_vs_rtol_1e-12_max_m"] = {str(k): float(np.max(np.linalg.norm(v - hist[1e-12], axis=1)))
                                                     for k, v in hist.items() if k != 1e-12}


def cost(kind, x, obs, geo):
    f = compute_position_residuals if kind == "position" else compute_range_rate_residuals
    r = np.asarray(f(a["propagate"](x), obs, geo)[0]) / measurement_sigma_vector(obs, geo)
    return float(r @ r)


original = L.est._adaptive_tol
for loosening in ("off", "on"):
    L.est._adaptive_tol = (lambda i, m, r, at: (r, at)) if loosening == "off" else original
    for kind in ("range_rate", "position"):
        for model in (L.IMPLICIT, L.FIRST):
            obs, geo = (L.position(a, model, True, np.random.default_rng(2000)) if kind == "position"
                        else L.range_rate(a, model, np.random.default_rng(2000)))
            run = getattr(L.est, f"estimate_{kind}_bls_lm")
            x, status, stats = run(a["t"], obs, L.start, geo, *L.gm, a["eph"].earth_position, a["eph"].sun_position,
                                   rtol=1e-12, atol=1e-13, return_posterior=True)
            e = np.asarray(x[:6]) - L.x0
            cov = np.asarray(stats.posterior_covariance)[:6, :6]
            row = {"pos_err_m": float(np.linalg.norm(e[:3])), "vel_err_mps": float(np.linalg.norm(e[3:])),
                   "nees": float(e @ np.linalg.solve(cov, e)), "status": str(status),
                   "iterations": int(stats.iterations), "cost": cost(kind, x, obs, geo),
                   "cost_truth": cost(kind, L.x0, obs, geo)}
            out[f"loosening_{loosening}/{kind}/{model}"] = row
            print(loosening, kind, model, json.dumps(row), flush=True)
L.est._adaptive_tol = original
Path("studies/phase17/results/wp2_5_loose_tolerance_test.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out["propagation_difference_vs_rtol_1e-12_max_m"]))
