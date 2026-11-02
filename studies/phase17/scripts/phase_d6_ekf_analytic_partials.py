"""Phase D6: analytic measurement partials for the EKF against central differences.

Predeclaration: studies/phase17/configs/phase_d6_ekf_analytic_partials_predeclaration.json.
Single process, one BLAS thread (G4 compares wall times).
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_var] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
warnings.simplefilter("ignore")

from lunar_od import normalized_estimation_error_squared  # noqa: E402
from lunar_od.ekf import (  # noqa: E402
    DEFAULT_JACOBIAN_STEPS,
    finite_difference_measurement_jacobian,
    run_lunar_ekf,
    run_lunar_iterated_smoother,
)
from lunar_od.ekf_partials import analytic_measurement_jacobian  # noqa: E402
from lunar_od.filters import _measurement_context, _measurement_residual, _resolve_ukf_bias_config  # noqa: E402
from tests.test_ekf_partials import counted_doppler_arc, richardson_jacobian  # noqa: E402
from tests.test_ekf_smoother import MU_MOON, _add_noise, _arc, _earth, _sun  # noqa: E402

OUT = HERE.parent / "results" / "phase_d6_ekf_analytic_partials.json"
TOL = dict(rtol=1e-12, atol=1e-13)
TYPES = ("position", "range_rate", "counted_doppler")
SEEDS = 20
P0 = np.diag([20.0**2] * 3 + [0.02**2] * 3)
NEES_BAND = (4.19, 8.18)
ORACLE_EPOCHS = (600.0, 1200.0, 1800.0, 2400.0, 3000.0)
IS_SEEDS = 5


def case(kind):
    if kind == "counted_doppler":
        return "range_rate", counted_doppler_arc()
    return kind, _arc(kind, duration_s=3601.0)


def jacobian_check(kind, mtype, arc):
    x_true0, t_pass, truth, geo, clean = arc
    rng = np.random.default_rng([20261031, 6, TYPES.index(kind), 0])
    _ = _add_noise(clean, mtype, rng)
    dx = rng.multivariate_normal(np.zeros(6), P0)
    time_col = 5 if mtype == "position" else 6
    bias_cfg = _resolve_ukf_bias_config(mtype, 6, len(geo.stations), None)
    residual = _measurement_residual(mtype)
    rows = []
    for t in ORACLE_EPOCHS:
        for row in clean[clean[:, 0] == t]:
            x = truth[int(row[time_col]) - 1] + dx
            oracle = richardson_jacobian(mtype, row, geo, x)
            h_an = analytic_measurement_jacobian(mtype, row, geo, x, MU_MOON, 0.0, 0.0, _earth, _sun, **TOL)
            _, _, fn, _, _ = _measurement_context(mtype, row, geo, MU_MOON, 0.0, 0.0, _earth, _sun,
                                                  TOL["rtol"], TOL["atol"], bias_cfg)
            h_fd = finite_difference_measurement_jacobian(fn, x, DEFAULT_JACOBIAN_STEPS, residual)
            norm = np.linalg.norm(oracle, axis=1)
            rows.append(dict(t=float(t), station=int(row[time_col - 1]),
                             analytic_rel=list(map(float, np.linalg.norm(h_an - oracle, axis=1) / norm)),
                             fd_rel=list(map(float, np.linalg.norm(h_fd - oracle, axis=1) / norm))))
    return rows


def one_seed(kind, mtype, arc, seed):
    x_true0, t_pass, truth, geo, clean = arc
    rng = np.random.default_rng([20261031, 6, TYPES.index(kind), seed])
    obs = _add_noise(clean, mtype, rng)
    x_guess = x_true0 + rng.multivariate_normal(np.zeros(6), P0)
    args = (t_pass, obs, x_guess, P0, geo, MU_MOON, 0.0, 0.0, _earth, _sun)
    out = {}
    time_col = 5 if mtype == "position" else 6
    for name, option in (("FD", "finite_difference"), ("ANALYTIC", "analytic")):
        t0 = time.perf_counter()
        res = run_lunar_ekf(*args, smooth=True, measurement_jacobian=option, **TOL)
        elapsed = time.perf_counter() - t0
        final_truth = truth[int(obs[res.obs_indices[-1], time_col]) - 1]
        out[name] = dict(x=list(map(float, res.final_state)), sigma=list(map(float, np.sqrt(np.diag(res.final_covariance)))),
                         x_s0=list(map(float, res.smoothed.state_estimates[0])),
                         sigma_s0=list(map(float, np.sqrt(np.diag(res.smoothed.covariances[0])))),
                         nees=float(normalized_estimation_error_squared(res.final_state, final_truth,
                                                                        res.final_covariance)),
                         elapsed_s=elapsed)
    fd, an = out["FD"], out["ANALYTIC"]
    out["final_diff_sigma"] = float(np.max(np.abs(np.subtract(an["x"], fd["x"])) / np.asarray(fd["sigma"])))
    out["smoothed_diff_sigma"] = float(np.max(np.abs(np.subtract(an["x_s0"], fd["x_s0"])) / np.asarray(fd["sigma_s0"])))
    return out


def iterated_smoother_on_counted(arc):
    """Reported only: the iterated smoother with analytic partials (first observation at 60 s)."""
    x_true0, t_pass, truth, geo, clean = arc
    first = float(clean[0, 0])
    t_arc = t_pass[t_pass >= first]
    i0 = int(np.searchsorted(t_pass, first))
    rows = []
    for seed in range(IS_SEEDS):
        rng = np.random.default_rng([20261031, 6, 99, seed])
        obs = _add_noise(clean, "range_rate", rng)
        x_guess = truth[i0] + rng.multivariate_normal(np.zeros(6), P0)
        entry = {}
        for name, option in (("FD", "finite_difference"), ("ANALYTIC", "analytic")):
            t0 = time.perf_counter()
            it = run_lunar_iterated_smoother(t_arc, obs, x_guess, P0, geo, MU_MOON, 0.0, 0.0, _earth, _sun,
                                             measurement_jacobian=option, **TOL)
            entry[name] = dict(status=it.status, iterations=it.iterations, elapsed_s=time.perf_counter() - t0,
                               x0=list(map(float, it.result.smoothed.state_estimates[0])),
                               nees0=float(normalized_estimation_error_squared(
                                   it.result.smoothed.state_estimates[0], truth[i0], it.result.smoothed.covariances[0])))
        sigma = np.sqrt(np.diag(it.result.smoothed.covariances[0]))
        entry["x0_diff_sigma"] = float(np.max(np.abs(np.subtract(entry["ANALYTIC"]["x0"], entry["FD"]["x0"])) / sigma))
        rows.append(entry)
    return rows


def main():
    results = {}
    for kind in TYPES:
        mtype, arc = case(kind)
        jac = jacobian_check(kind, mtype, arc)
        runs = [one_seed(kind, mtype, arc, seed) for seed in range(SEEDS)]
        summary = dict(
            max_final_diff_sigma=max(r["final_diff_sigma"] for r in runs),
            max_smoothed_diff_sigma=max(r["smoothed_diff_sigma"] for r in runs),
            mean_nees_fd=float(np.mean([r["FD"]["nees"] for r in runs])),
            mean_nees_analytic=float(np.mean([r["ANALYTIC"]["nees"] for r in runs])),
            median_elapsed_fd_s=float(np.median([r["FD"]["elapsed_s"] for r in runs])),
            median_elapsed_analytic_s=float(np.median([r["ANALYTIC"]["elapsed_s"] for r in runs])),
            max_analytic_rel=[max(j["analytic_rel"][i] for j in jac) for i in range(len(jac[0]["analytic_rel"]))],
            max_fd_rel=[max(j["fd_rel"][i] for j in jac) for i in range(len(jac[0]["fd_rel"]))])
        results[kind] = dict(summary=summary, jacobian=jac, runs=runs)
        if kind == "counted_doppler":
            results[kind]["iterated_smoother"] = iterated_smoother_on_counted(arc)
        print(kind, json.dumps(summary), flush=True)

    def g1(kind):
        rel = results[kind]["summary"]["max_analytic_rel"]
        if kind == "counted_doppler":
            return rel[1] <= 1e-4 and max(rel[0], rel[2], rel[3]) <= 1e-6
        return max(rel) <= 1e-6

    gates = dict(
        G1_jacobian=all(g1(k) for k in TYPES),
        G2_filter_equivalence=all(results[k]["summary"]["max_final_diff_sigma"] <= 0.01
                                  and results[k]["summary"]["max_smoothed_diff_sigma"] <= 0.01 for k in TYPES),
        G3_consistency=all(NEES_BAND[0] <= results[k]["summary"]["mean_nees_analytic"] <= NEES_BAND[1] for k in TYPES),
        G4_cost=results["counted_doppler"]["summary"]["median_elapsed_analytic_s"]
        <= results["counted_doppler"]["summary"]["median_elapsed_fd_s"] / 3.0,
        G5_invariance="tests/test_ekf_smoother.py and tests/test_dmc.py run separately (see report)")
    out = dict(predeclaration="studies/phase17/configs/phase_d6_ekf_analytic_partials_predeclaration.json",
               gates_computed_here=gates, seeds=SEEDS, nees_band=NEES_BAND, tolerances=TOL, results=results)
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(gates, indent=1))


if __name__ == "__main__":
    main()
