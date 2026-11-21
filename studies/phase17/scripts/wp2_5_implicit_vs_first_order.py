"""WP2.5: implicit_light_time vs analytic_first_order_light_time on noisy data, same seeds, interleaved order.

Cases: one_way_light_time position (BLS-LM, SRIF), range-rate with the apparent_one_way companion (BLS-LM).
Metrics per fit: position / velocity error against the truth, NEES with the BLS posterior, posterior sigmas,
iterations, wall time. Run from the worktree; reuses the WP2.5 test arc.
"""
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "tests")
import test_wp2_5_measurement_defaults as T  # noqa: E402
import lunar_od.estimators as est  # noqa: E402
from lunar_od.measurements import generate_position_measurements, generate_range_rate_measurements  # noqa: E402

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
arc = inspect.unwrap(T.arc)()
eph, t = arc["eph"], arc["t"]
truth = arc["aug"][:, :6]
start = arc["x0"] + np.array([80.0, -60.0, 40.0, 0.02, -0.01, 0.015])
MODELS = (T.IMPLICIT, T.FIRST_ORDER)


def data(case, seed, model):
    rng = np.random.default_rng(seed)
    if case.startswith("position"):
        obs, geo = generate_position_measurements(t, truth, T.STATIONS, arc["vis"], eph.earth_position, eph.earth_velocity,
                                                  arc["et0"], noise=True, rng=rng, arc_id=1,
                                                  measurement_model_profile="one_way_light_time",
                                                  jacobian_model=model)[:2]
    else:
        vis = np.zeros_like(arc["vis"])
        vis[::2] = arc["vis"][::2]
        obs, geo = generate_range_rate_measurements(t, truth, T.STATIONS, vis, eph.earth_position, eph.earth_velocity,
                                                    arc["et0"], noise=True, rng=rng, arc_id=1,
                                                    companion_geometry="apparent_one_way", jacobian_model=model)
    assert geo.jacobian_model == model
    return obs, geo


def fit(case, obs, geo):
    kind = "position" if case.startswith("position") else "range_rate"
    estimator = "srif" if case.endswith("srif") else "bls_lm"
    run = getattr(est, f"estimate_{kind}_{estimator}")
    kwargs = dict(rtol=1e-12, atol=1e-13)
    if estimator == "bls_lm":
        kwargs["return_posterior"] = True
    t0 = time.perf_counter()
    x_hat, status, stats = run(t, obs, start, geo, *arc["gm"], eph.earth_position, eph.sun_position, **kwargs)
    wall = time.perf_counter() - t0
    err = np.asarray(x_hat[:6]) - arc["x0"]
    row = {"pos_err_m": float(np.linalg.norm(err[:3])), "vel_err_mps": float(np.linalg.norm(err[3:])),
           "iterations": int(getattr(stats, "iterations", -1)), "wall_s": wall, "status": str(status)}
    cov = getattr(stats, "posterior_covariance", None)
    if cov is not None:
        cov = np.asarray(cov)[:6, :6]
        row["nees"] = float(err @ np.linalg.solve(cov, err))
        row["sigma"] = np.sqrt(np.diag(cov)).tolist()
    return row


cases = ("position_bls_lm", "position_srif", "range_rate_apparent_bls_lm")
raw = {c: {m: [] for m in MODELS} for c in cases}
for case in cases:
    for seed in range(N_SEEDS):
        order = MODELS if seed % 2 == 0 else MODELS[::-1]
        for model in order:
            obs, geo = data(case, 1000 + seed, model)
            raw[case][model].append(fit(case, obs, geo))
        print(case, seed, {m: round(raw[case][m][-1]["pos_err_m"], 4) for m in MODELS}, flush=True)


def summary(rows):
    out = {}
    for key in ("pos_err_m", "vel_err_mps", "iterations", "wall_s", "nees"):
        v = np.array([r[key] for r in rows if key in r], dtype=float)
        if v.size:
            out[key] = {"median": float(np.median(v)), "mean": float(np.mean(v)), "p95": float(np.percentile(v, 95)),
                        "max": float(np.max(v))}
    if rows and "sigma" in rows[0]:
        out["sigma_median"] = np.median(np.array([r["sigma"] for r in rows]), axis=0).tolist()
    out["status_counts"] = {k: sum(r["status"] == k for r in rows) for k in sorted({r["status"] for r in rows})}
    return out


result = {"n_seeds": N_SEEDS, "start_offset": "80, -60, 40 m; 0.02, -0.01, 0.015 m/s",
          "noise": "2 m range, 1e-4 rad angles, 1e-4 m/s range rate; seeds 1000..",
          "note": "wall times measured while a full test regression ran on the same machine; compare ratios only",
          "summary": {c: {m: summary(raw[c][m]) for m in MODELS} for c in cases}}
paired = {}
for c in cases:
    imp, fo = raw[c][T.IMPLICIT], raw[c][T.FIRST_ORDER]
    paired[c] = {
        "pos_err_diff_median_m": float(np.median([a["pos_err_m"] - b["pos_err_m"] for a, b in zip(imp, fo)])),
        "pos_err_diff_max_abs_m": float(np.max([abs(a["pos_err_m"] - b["pos_err_m"]) for a, b in zip(imp, fo)])),
        "wall_ratio_median": float(np.median([a["wall_s"] / b["wall_s"] for a, b in zip(imp, fo)])),
        "iteration_ratio_median": float(np.median([a["iterations"] / max(b["iterations"], 1) for a, b in zip(imp, fo)])),
    }
    if "sigma" in imp[0]:
        paired[c]["sigma_ratio_median_first_order_over_implicit"] = np.median(
            [np.array(b["sigma"]) / np.array(a["sigma"]) for a, b in zip(imp, fo)], axis=0).tolist()
result["paired"] = paired
result["raw"] = raw
Path("studies/phase17/results/wp2_5_implicit_vs_first_order.json").write_text(json.dumps(result, indent=1))
print(json.dumps({"summary": result["summary"], "paired": paired}, indent=1))
