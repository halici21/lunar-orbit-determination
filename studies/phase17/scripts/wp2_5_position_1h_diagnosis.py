"""Why the 1 h one_way_light_time position fits end at ~1.9 km: weak geometry or early stop? Run from the worktree."""
import inspect
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "tests")
import test_wp2_5_measurement_defaults as T  # noqa: E402
import lunar_od.estimators as est  # noqa: E402
from lunar_od.measurements import compute_position_residuals, generate_position_measurements, measurement_sigma_vector  # noqa: E402

arc = inspect.unwrap(T.arc)()
eph, t = arc["eph"], arc["t"]
truth = arc["aug"][:, :6]
start = arc["x0"] + np.array([80.0, -60.0, 40.0, 0.02, -0.01, 0.015])
out = {}


def cost(x, obs, geo):
    r, _ = compute_position_residuals(arc["propagate"](x), obs, geo)
    w = np.asarray(r) / measurement_sigma_vector(obs, geo)
    return float(w @ w)


def run(x_start, obs, geo, **kw):
    x, status, stats = est.estimate_position_bls_lm(t, obs, x_start, geo, *arc["gm"], eph.earth_position,
                                                    eph.sun_position, rtol=1e-12, atol=1e-13, return_posterior=True, **kw)
    cov = np.asarray(stats.posterior_covariance)[:6, :6]
    e = np.asarray(x[:6]) - arc["x0"]
    return x, {"pos_err_m": float(np.linalg.norm(e[:3])), "nees": float(e @ np.linalg.solve(cov, e)),
               "status": str(status), "iterations": int(stats.iterations), "cost": cost(x, obs, geo)}


for seed in (1001, 1002, 1009, 1004):
    obs, geo = generate_position_measurements(t, truth, T.STATIONS, arc["vis"], eph.earth_position, eph.earth_velocity,
                                              arc["et0"], noise=True, rng=np.random.default_rng(seed), arc_id=1,
                                              measurement_model_profile="one_way_light_time")[:2]
    row = {"rows": int(obs.shape[0]), "cost_at_truth": cost(arc["x0"], obs, geo)}
    x1, row["default_60_iter"] = run(start, obs, geo)
    _, row["continued_300_iter"] = run(x1, obs, geo, max_iter=300)
    _, row["from_truth"] = run(arc["x0"], obs, geo, max_iter=300)
    _, row["nielsen_300_iter"] = run(start, obs, geo, max_iter=300, damping_schedule="nielsen")
    out[seed] = row
    print(seed, json.dumps(row), flush=True)

# formal geometry: eigen-decomposition of the posterior at the truth
_, stats = est.estimate_position_bls_lm(t, obs, arc["x0"], geo, *arc["gm"], eph.earth_position, eph.sun_position,
                                        rtol=1e-12, atol=1e-13, return_posterior=True, max_iter=1)[1:]
cov = np.asarray(stats.posterior_covariance)[:6, :6]
vals, vecs = np.linalg.eigh(cov[:3, :3])
out["position_sigma_principal_m"] = np.sqrt(vals).tolist()
out["rows_note"] = "one station (Canberra) visible, 30 s cadence"
Path("studies/phase17/results/wp2_5_position_1h_diagnosis.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if not isinstance(k, int)}))
