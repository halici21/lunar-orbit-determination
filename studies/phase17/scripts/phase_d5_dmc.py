"""Phase D5: dynamic model compensation on the iterated extended Kalman smoother.

Predeclaration: studies/phase17/configs/phase_d5_dmc_predeclaration.json.
Per cell (measurement type x truth case) and seed, IS, IS_DMC and EKF_RTS_DMC get the same
truth, observations, noise realisation, x_guess and P0 on [r, v].
"""
from __future__ import annotations

import json
import math
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
warnings.simplefilter("ignore")

from lunar_od import (  # noqa: E402
    PassGeometry,
    compute_position_residuals_analytic,
    compute_range_rate_residuals_analytic,
    normalized_estimation_error_squared,
    propagate_state,
)
from lunar_od.dmc import DMCOptions  # noqa: E402
from lunar_od.ekf import run_lunar_ekf, run_lunar_iterated_smoother  # noqa: E402
from lunar_od.geometry import wrap_to_pi  # noqa: E402
from tests.test_filters import _synthetic_position_station, _synthetic_range_rate_station  # noqa: E402

OUT = HERE.parent / "results" / "phase_d5_dmc.json"
MU_MOON = 4902.800066e9
TOL = dict(rtol=1e-12, atol=1e-13)
MEASUREMENTS = ("position", "range_rate")
CASES = ("C0_matched", "M1_along_track", "M2_j2")
PUSH_MPS2 = 2e-6
J2_TRUTH = 2.03e-6
DMC = DMCOptions(sigma_mps2=5e-6, tau_s=1800.0)
SCALE = (2000.0, 2.0)
SEEDS = 20
DURATION_S = 3.0 * 3600.0
NEES_BAND = (4.19, 8.18)


def _earth(t):
    return np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))


def _sun(t):
    return np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))


def truth_trajectory(case, x0, t):
    if case == "C0_matched":
        return propagate_state(t, x0, MU_MOON, 0.0, 0.0, _earth, _sun, **TOL)
    if case == "M2_j2":
        return propagate_state(t, x0, MU_MOON, 0.0, 0.0, _earth, _sun, j2_moon=J2_TRUTH, **TOL)

    def rhs(_t, y):
        r, v = y[:3], y[3:]
        return np.concatenate([v, -MU_MOON * r / np.linalg.norm(r) ** 3 + PUSH_MPS2 * v / np.linalg.norm(v)])

    return solve_ivp(rhs, (t[0], t[-1]), x0, method="DOP853", t_eval=t, **TOL).y.T


def arc(measurement_type, case):
    r0 = 1737.4e3 + 100e3
    x0 = np.array([r0, 30e3, -20e3, -15.0, math.sqrt(MU_MOON / r0), 4.0])
    t = np.arange(0.0, DURATION_S + 1.0, 60.0)
    truth = truth_trajectory(case, x0, t)
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


def metrics(result, t, truth, elapsed, error, extra=None):
    if error is not None:
        return dict(ok=False, error=error, elapsed_s=elapsed)
    sm = result.smoothed
    ref = truth[np.searchsorted(t, result.t_update_s)]
    d = sm.state_estimates[:, :3] - ref[:, :3]
    x0, p0 = sm.state_estimates[0], sm.covariances[0]
    out = dict(ok=bool(np.all(np.isfinite(sm.state_estimates))),
               pos_rms_m=float(np.sqrt(np.mean(np.sum(d**2, axis=1)))),
               pos_max_m=float(np.max(np.linalg.norm(d, axis=1))),
               first_pos_err_m=float(np.linalg.norm(x0[:3] - truth[0, :3])),
               nees_first=float(normalized_estimation_error_squared(x0[:6], truth[0], p0[:6, :6])),
               elapsed_s=elapsed, **(extra or {}))
    if sm.state_estimates.shape[1] >= 9:
        v = ref[:, 3:6] / np.linalg.norm(ref[:, 3:6], axis=1)[:, None]
        out["mean_w_along_mps2"] = float(np.mean(np.sum(sm.state_estimates[:, 6:9] * v, axis=1)))
    return out


def one_seed(job):
    mi, ci, seed = job
    measurement_type, case = MEASUREMENTS[mi], CASES[ci]
    x_true0, t, truth, geo, clean = arc(measurement_type, case)
    rng = np.random.default_rng([20261031, mi, ci, seed])
    obs = noisy(clean, measurement_type, rng)
    p0 = np.diag([SCALE[0] ** 2] * 3 + [SCALE[1] ** 2] * 3)
    x_guess = x_true0 + rng.multivariate_normal(np.zeros(6), p0)
    p9 = np.zeros((9, 9))
    p9[:6, :6] = p0
    p9[6:, 6:] = DMC.sigma_mps2**2 * np.eye(3)
    x9 = np.concatenate([x_guess, np.zeros(3)])
    args6 = (t, obs, x_guess, p0, geo, MU_MOON, 0.0, 0.0, _earth, _sun)
    args9 = (t, obs, x9, p9, geo, MU_MOON, 0.0, 0.0, _earth, _sun)
    it, t_it, e_it = _timed(lambda: run_lunar_iterated_smoother(*args6, **TOL))
    itd, t_itd, e_itd = _timed(lambda: run_lunar_iterated_smoother(*args9, dmc=DMC, **TOL))
    ekd, t_ekd, e_ekd = _timed(lambda: run_lunar_ekf(*args9, dmc=DMC, smooth=True, **TOL))
    return dict(
        cell=f"{measurement_type}/{case}", seed=seed,
        IS=metrics(None if it is None else it.result, t, truth, t_it, e_it,
                   None if it is None else dict(status=it.status, iterations=it.iterations)),
        IS_DMC=metrics(None if itd is None else itd.result, t, truth, t_itd, e_itd,
                       None if itd is None else dict(status=itd.status, iterations=itd.iterations)),
        EKF_RTS_DMC=metrics(ekd, t, truth, t_ekd, e_ekd))


def summarise(runs):
    out = {}
    for name in ("IS", "IS_DMC", "EKF_RTS_DMC"):
        ok = [r[name] for r in runs if r[name]["ok"]]
        s = dict(n_ok=len(ok), n_failed=len(runs) - len(ok))
        if ok:
            s.update(pos_rms_m=float(np.sqrt(np.mean([o["pos_rms_m"] ** 2 for o in ok]))),
                     mean_nees_first=float(np.mean([o["nees_first"] for o in ok])),
                     mean_elapsed_s=float(np.mean([o["elapsed_s"] for o in ok])))
            if "status" in ok[0]:
                statuses = [o["status"] for o in ok]
                s["statuses"] = {k: statuses.count(k) for k in sorted(set(statuses))}
                s["max_iterations"] = max(o["iterations"] for o in ok)
            if "mean_w_along_mps2" in ok[0]:
                s["mean_w_along_mps2"] = float(np.mean([o["mean_w_along_mps2"] for o in ok]))
        out[name] = s
    return out


def main():
    jobs = [(mi, ci, seed) for mi in range(len(MEASUREMENTS)) for ci in range(len(CASES)) for seed in range(SEEDS)]
    with Pool(12) as pool:
        runs = pool.map(one_seed, jobs, chunksize=1)
    results = {}
    for m in MEASUREMENTS:
        for c in CASES:
            cell = f"{m}/{c}"
            cell_runs = [r for r in runs if r["cell"] == cell]
            results[cell] = dict(summary=summarise(cell_runs), runs=cell_runs)
            print(cell, json.dumps(results[cell]["summary"]), flush=True)

    def s(cell, name):
        return results[cell]["summary"][name]

    def in_band(v):
        return v is not None and NEES_BAND[0] <= v <= NEES_BAND[1]

    gates = {}
    gates["G3_matched_control"] = all(
        s(f"{m}/C0_matched", "IS_DMC")["n_failed"] == 0
        and s(f"{m}/C0_matched", "IS_DMC")["pos_rms_m"] <= 2.0 * s(f"{m}/C0_matched", "IS")["pos_rms_m"]
        and in_band(s(f"{m}/C0_matched", "IS_DMC")["mean_nees_first"]) for m in MEASUREMENTS)
    mis = [f"{m}/{c}" for m in MEASUREMENTS for c in CASES[1:]]
    gates["G4_mismodel_accuracy"] = all(
        s(c, "IS_DMC")["n_failed"] == 0 and s(c, "IS_DMC")["pos_rms_m"] <= 0.5 * s(c, "IS")["pos_rms_m"] for c in mis)
    gates["G5_mismodel_consistency"] = all(in_band(s(c, "IS_DMC").get("mean_nees_first")) for c in mis)
    gates["G6_convergence"] = all(
        s(cell, "IS_DMC")["n_failed"] == 0 and s(cell, "IS_DMC")["statuses"] == {"Converged": SEEDS}
        for cell in results)
    out = dict(predeclaration="studies/phase17/configs/phase_d5_dmc_predeclaration.json",
               gates_computed_here=gates, nees_band=NEES_BAND, seeds=SEEDS, prior_scale=SCALE,
               dmc=dict(sigma_mps2=DMC.sigma_mps2, tau_s=DMC.tau_s), push_mps2=PUSH_MPS2, j2_truth=J2_TRUTH,
               duration_s=DURATION_S, tolerances=TOL, results=results)
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(gates, indent=1))


if __name__ == "__main__":
    main()
