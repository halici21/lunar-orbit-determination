"""Phase D3: EKF, SR-UKF, EKF+RTS and BLS-LM on the same synthetic lunar arcs.

Predeclaration: studies/phase17/configs/phase_d3_ekf_rts_predeclaration.json.
Every estimator gets the same truth, observations, noise realisation, x_guess and P0.
"""
from __future__ import annotations

import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
warnings.simplefilter("ignore")

from lunar_od import (  # noqa: E402
    PassGeometry,
    UnscentedTransformConfig,
    compute_position_residuals_analytic,
    compute_range_rate_residuals_analytic,
    normalized_estimation_error_squared,
    propagate_augmented_state,
    run_lunar_ukf,
)
from lunar_od.ekf import run_lunar_ekf  # noqa: E402
from lunar_od.estimators import estimate_position_bls_lm, estimate_range_rate_bls_lm  # noqa: E402
from lunar_od.geometry import wrap_to_pi  # noqa: E402
from tests.test_filters import _synthetic_position_station, _synthetic_range_rate_station  # noqa: E402

OUT = HERE.parent / "results" / "phase_d3_ekf_rts_comparison.json"
MU_MOON = 4902.800066e9
TOL = dict(rtol=1e-12, atol=1e-13)
MEASUREMENTS = ("position", "range_rate")
SCALES = {"S20m": (20.0, 0.02), "S2km": (2000.0, 2.0), "S20km": (20000.0, 20.0)}
SEEDS = 50
NEES_BAND = (4.39, 7.61)


def _earth(t):
    return np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))


def _sun(t):
    return np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))


def arc(measurement_type):
    r0 = 1737.4e3 + 100e3
    x0 = np.array([r0, 30e3, -20e3, -15.0, math.sqrt(MU_MOON / r0), 4.0])
    t = np.arange(0.0, 3601.0, 60.0)
    truth = propagate_augmented_state(t, np.concatenate([x0, np.eye(6).reshape(-1, order="F")]),
                                      MU_MOON, 0.0, 0.0, _earth, _sun, **TOL)[:, :6]
    station = _synthetic_position_station if measurement_type == "position" else _synthetic_range_rate_station
    geo = PassGeometry(t_s=t, earth_pos_mci_m=np.zeros((t.size, 3)), earth_vel_mci_mps=np.zeros((t.size, 3)),
                       x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t.size, axis=0),
                       stations=(station(0.0, 0.0, 0.0), station(0.0, 90.0, 0.0), station(45.0, -30.0, 500.0)),
                       measurement_type=measurement_type)
    ncol = 6 if measurement_type == "position" else 7
    rows = [[ti] + [0.0] * (ncol - 3) + [sid, k] for k, ti in enumerate(t, start=1) for sid in (1, 2, 3)]
    obs = np.asarray(rows, dtype=float)
    if measurement_type == "position":
        obs[:, 1:4] = compute_position_residuals_analytic(truth, obs, geo)[1]
    else:
        obs[:, 1:5] = compute_range_rate_residuals_analytic(truth, obs, geo)[1]
    return x0, t, truth, geo, obs


def noisy(obs, measurement_type, rng):
    obs = obs.copy()
    n = obs.shape[0]
    obs[:, 1] += rng.normal(0.0, 1.0, n)
    if measurement_type == "position":
        obs[:, 2] = wrap_to_pi(obs[:, 2] + rng.normal(0.0, 1e-5, n))
        obs[:, 3] = wrap_to_pi(obs[:, 3] + rng.normal(0.0, 1e-5, n))
    else:
        obs[:, 2] += rng.normal(0.0, 1e-4, n)
        obs[:, 3] = wrap_to_pi(obs[:, 3] + rng.normal(0.0, 1e-5, n))
        obs[:, 4] = wrap_to_pi(obs[:, 4] + rng.normal(0.0, 1e-5, n))
    return obs


def _timed(fn):
    t0 = time.perf_counter()
    try:
        return fn(), time.perf_counter() - t0, None
    except Exception as exc:  # reported, not hidden
        return None, time.perf_counter() - t0, f"{type(exc).__name__}: {exc}"[:300]


def one_seed(measurement_type, mi, si, scale, seed, case):
    x_true0, t, truth, geo, clean = case
    rng = np.random.default_rng([20261030, mi, si, seed])
    obs = noisy(clean, measurement_type, rng)
    p0 = np.diag([scale[0] ** 2] * 3 + [scale[1] ** 2] * 3)
    x_guess = x_true0 + rng.multivariate_normal(np.zeros(6), p0)
    common = (t, obs, x_guess, p0, geo, MU_MOON, 0.0, 0.0, _earth, _sun)
    bls_fn = estimate_position_bls_lm if measurement_type == "position" else estimate_range_rate_bls_lm
    ekf, t_ekf, e_ekf = _timed(lambda: run_lunar_ekf(*common, smooth=True, **TOL))
    ukf, t_ukf, e_ukf = _timed(lambda: run_lunar_ukf(*common, config=UnscentedTransformConfig(alpha=0.35), **TOL))
    bls, t_bls, e_bls = _timed(lambda: bls_fn(t, obs, x_guess, geo, MU_MOON, 0.0, 0.0, _earth, _sun,
                                               prior_covariance=p0, return_posterior=True, max_iter=60, **TOL))

    def entry(x, p, ref, elapsed, error, extra=None):
        if error is not None or x is None or p is None or not np.all(np.isfinite(x)):
            return dict(ok=False, error=error, elapsed_s=elapsed)
        d = np.asarray(x) - ref
        return dict(ok=True, x=list(map(float, x)), sigma=list(map(float, np.sqrt(np.diag(p)))),
                    pos_err_m=float(np.linalg.norm(d[:3])), vel_err_mps=float(np.linalg.norm(d[3:6])),
                    nees=float(normalized_estimation_error_squared(x, ref, p)), elapsed_s=elapsed, **(extra or {}))

    out = {}
    out["EKF"] = entry(None if ekf is None else ekf.final_state, None if ekf is None else ekf.final_covariance,
                       truth[-1], t_ekf, e_ekf,
                       None if ekf is None else dict(mean_nis=float(np.mean(ekf.normalized_innovation_squared))))
    out["SRUKF"] = entry(None if ukf is None else ukf.final_state, None if ukf is None else ukf.final_covariance,
                         truth[-1], t_ukf, e_ukf,
                         None if ukf is None else dict(mean_nis=float(np.mean(ukf.normalized_innovation_squared))))
    out["EKF_RTS"] = entry(None if ekf is None else ekf.smoothed.state_estimates[0],
                           None if ekf is None else ekf.smoothed.covariances[0], truth[0], t_ekf, e_ekf)
    out["BLS"] = entry(None if bls is None else bls[0], None if bls is None else bls[2].posterior_covariance,
                       truth[0], t_bls, e_bls, None if bls is None else dict(status=bls[1]))
    return out


def summarise(runs):
    summary = {}
    for name in ("EKF", "SRUKF", "EKF_RTS", "BLS"):
        ok = [r[name] for r in runs if r[name]["ok"]]
        summary[name] = dict(
            n_ok=len(ok), n_failed=len(runs) - len(ok),
            pos_rms_m=float(np.sqrt(np.mean([o["pos_err_m"] ** 2 for o in ok]))) if ok else None,
            vel_rms_mps=float(np.sqrt(np.mean([o["vel_err_mps"] ** 2 for o in ok]))) if ok else None,
            mean_nees=float(np.mean([o["nees"] for o in ok])) if ok else None,
            mean_nis=float(np.mean([o["mean_nis"] for o in ok])) if ok and "mean_nis" in ok[0] else None,
            mean_elapsed_s=float(np.mean([o["elapsed_s"] for o in ok])) if ok else None)
    if summary["BLS"]["n_ok"]:
        statuses = [r["BLS"].get("status") for r in runs if r["BLS"]["ok"]]
        summary["BLS"]["statuses"] = {s: statuses.count(s) for s in sorted(set(statuses))}
    return summary


def agreement(runs, a, b):
    """max over seeds and components of |x_a - x_b| / sigma_b."""
    worst = 0.0
    for r in runs:
        if not (r[a]["ok"] and r[b]["ok"]):
            return None
        worst = max(worst, float(np.max(np.abs(np.subtract(r[a]["x"], r[b]["x"])) / np.asarray(r[b]["sigma"]))))
    return worst


def main():
    results = {}
    for mi, measurement_type in enumerate(MEASUREMENTS):
        case = arc(measurement_type)
        for si, (label, scale) in enumerate(SCALES.items()):
            runs = [one_seed(measurement_type, mi, si, scale, seed, case) for seed in range(SEEDS)]
            results[f"{measurement_type}/{label}"] = dict(
                summary=summarise(runs),
                max_ekf_vs_ukf_sigma=agreement(runs, "EKF", "SRUKF"),
                max_rts_vs_bls_sigma=agreement(runs, "EKF_RTS", "BLS"),
                runs=runs)
            s = results[f"{measurement_type}/{label}"]
            print(measurement_type, label, json.dumps({k: (v["pos_rms_m"], v["mean_nees"], v["n_failed"])
                                                       for k, v in s["summary"].items()}),
                  s["max_ekf_vs_ukf_sigma"], s["max_rts_vs_bls_sigma"], flush=True)
    gates = {}
    g3 = [results[f"{m}/S20m"] for m in MEASUREMENTS]
    gates["G3_linear_regime_agreement"] = all(
        r["max_ekf_vs_ukf_sigma"] is not None and r["max_ekf_vs_ukf_sigma"] <= 0.1
        and r["max_rts_vs_bls_sigma"] is not None and r["max_rts_vs_bls_sigma"] <= 0.1 for r in g3)
    gates["G4_consistency"] = all(
        r["summary"][name]["n_failed"] == 0 and NEES_BAND[0] <= r["summary"][name]["mean_nees"] <= NEES_BAND[1]
        for r in g3 for name in ("EKF", "SRUKF", "EKF_RTS", "BLS"))
    out = dict(predeclaration="studies/phase17/configs/phase_d3_ekf_rts_predeclaration.json",
               gates_computed_here=gates, nees_band=NEES_BAND, seeds=SEEDS, scales=SCALES, tolerances=TOL,
               results=results)
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(gates, indent=1))


if __name__ == "__main__":
    main()
