"""Phase D4: iterated EKF + RTS smoother on the D3 arcs, against BLS-LM; D3 rerun for invariance.

Predeclaration: studies/phase17/configs/phase_d4_iterated_smoother_predeclaration.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import phase_d3_ekf_rts_comparison as D3  # noqa: E402
from lunar_od import normalized_estimation_error_squared  # noqa: E402
from lunar_od.ekf import run_lunar_iterated_smoother  # noqa: E402

OUT = HERE.parent / "results" / "phase_d4_iterated_smoother.json"


def _strip_times(obj):
    if isinstance(obj, dict):
        return {k: _strip_times(v) for k, v in obj.items() if k not in ("elapsed_s", "mean_elapsed_s")}
    if isinstance(obj, list):
        return [_strip_times(v) for v in obj]
    return obj


def d3_invariance():
    """G1: every D3 run recomputed on this code equals the committed D3 record (times excluded)."""
    record = json.loads(D3.OUT.read_text(encoding="utf-8"))["results"]
    mismatches = []
    for mi, measurement_type in enumerate(D3.MEASUREMENTS):
        case = D3.arc(measurement_type)
        for si, (label, scale) in enumerate(D3.SCALES.items()):
            for seed in range(D3.SEEDS):
                now = _strip_times(json.loads(json.dumps(D3.one_seed(measurement_type, mi, si, scale, seed, case))))
                then = _strip_times(record[f"{measurement_type}/{label}"]["runs"][seed])
                if now != then:
                    mismatches.append(f"{measurement_type}/{label}/{seed}")
    return mismatches


def one_seed(measurement_type, mi, si, scale, seed, case, bls_run):
    x_true0, t, truth, geo, clean = case
    rng = np.random.default_rng([20261030, mi, si, seed])     # D3's stream: same noise, start and prior
    obs = D3.noisy(clean, measurement_type, rng)
    p0 = np.diag([scale[0] ** 2] * 3 + [scale[1] ** 2] * 3)
    x_guess = x_true0 + rng.multivariate_normal(np.zeros(6), p0)
    t0 = time.perf_counter()
    it = run_lunar_iterated_smoother(t, obs, x_guess, p0, geo, D3.MU_MOON, 0.0, 0.0, D3._earth, D3._sun, **D3.TOL)
    elapsed = time.perf_counter() - t0
    x = it.result.smoothed.state_estimates[0]
    p = it.result.smoothed.covariances[0]
    bls_sigma = np.asarray(bls_run["sigma"])
    return dict(status=it.status, iterations=it.iterations, halvings=[h["halvings"] for h in it.history],
                x=list(map(float, x)), sigma=list(map(float, np.sqrt(np.diag(p)))),
                pos_err_m=float(np.linalg.norm(x[:3] - truth[0, :3])),
                vel_err_mps=float(np.linalg.norm(x[3:6] - truth[0, 3:6])),
                nees=float(normalized_estimation_error_squared(x, truth[0], p)),
                max_vs_bls_sigma=float(np.max(np.abs(x - np.asarray(bls_run["x"])) / bls_sigma)),
                elapsed_s=elapsed, bls_elapsed_s=bls_run["elapsed_s"])


def main():
    d3 = json.loads(D3.OUT.read_text(encoding="utf-8"))["results"]
    results = {}
    for mi, measurement_type in enumerate(D3.MEASUREMENTS):
        case = D3.arc(measurement_type)
        for si, (label, scale) in enumerate(D3.SCALES.items()):
            key = f"{measurement_type}/{label}"
            runs = [one_seed(measurement_type, mi, si, scale, seed, case, d3[key]["runs"][seed]["BLS"])
                    for seed in range(D3.SEEDS)]
            statuses = [r["status"] for r in runs]
            results[key] = dict(
                statuses={s: statuses.count(s) for s in sorted(set(statuses))},
                max_vs_bls_sigma=max(r["max_vs_bls_sigma"] for r in runs),
                mean_nees=float(np.mean([r["nees"] for r in runs])),
                pos_rms_m=float(np.sqrt(np.mean([r["pos_err_m"] ** 2 for r in runs]))),
                iterations=dict(mean=float(np.mean([r["iterations"] for r in runs])),
                                max=max(r["iterations"] for r in runs)),
                total_halvings=int(sum(sum(r["halvings"]) for r in runs)),
                mean_elapsed_s=float(np.mean([r["elapsed_s"] for r in runs])),
                bls_mean_elapsed_s=float(np.mean([r["bls_elapsed_s"] for r in runs])),
                runs=runs)
            s = results[key]
            print(key, s["statuses"], round(s["max_vs_bls_sigma"], 5), round(s["mean_nees"], 3),
                  s["iterations"], s["total_halvings"], round(s["mean_elapsed_s"], 3), flush=True)
    mismatches = d3_invariance()
    print("D3 mismatches:", len(mismatches), mismatches[:5], flush=True)
    gates = dict(
        G1_d3_invariance=not mismatches,
        G2_bls_agreement=all(r["statuses"] == {"Converged": D3.SEEDS} and r["max_vs_bls_sigma"] <= 0.1
                             for r in results.values()),
        G3_consistency=all(D3.NEES_BAND[0] <= r["mean_nees"] <= D3.NEES_BAND[1] for r in results.values()))
    out = dict(predeclaration="studies/phase17/configs/phase_d4_iterated_smoother_predeclaration.json",
               gates_computed_here=gates, d3_mismatches=mismatches, results=results)
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(gates, indent=1))


if __name__ == "__main__":
    main()
