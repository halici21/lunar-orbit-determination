"""Run the predeclared, bounded, paired Level-0 estimator comparison."""
from __future__ import annotations
import argparse
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.quickstart_od import CONFIG, build_case, sha256, source_identity
from lunar_od import (compute_position_residuals_analytic, estimate_position_bls_lm,
                      estimate_position_srif, propagate_augmented_state,
                      run_lunar_ukf, UnscentedTransformConfig)

PLAN = ROOT / "examples/configs/benchmark.json"
ESTIMATORS = ("BLS-LM", "SRIF", "SR-UKF")


def distribution(values):
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if not data.size:
        return {"n": 0}
    return {"n": int(data.size), "mean": float(data.mean()),
            "std": float(data.std(ddof=1)) if len(data) > 1 else None,
            "min": float(data.min()), "p5": float(np.percentile(data, 5)),
            "median": float(np.median(data)), "p95": float(np.percentile(data, 95)),
            "max": float(data.max())}


def diagnostics(values):
    data = np.asarray(values, dtype=float)
    return {"mean": np.mean(data, axis=0).tolist(),
            "rms": np.sqrt(np.mean(data ** 2, axis=0)).tolist(),
            "lag1_acquisition_order": [float(np.corrcoef(data[:-1, i], data[1:, i])[0, 1])
                                       if np.std(data[:, i]) > 0 else None
                                       for i in range(data.shape[1])]}


def run_trial(seed, noise_scale, plan, base, output):
    config = {**base, "seed": seed,
              "sigma_range_m": base["sigma_range_m"] * noise_scale,
              "sigma_angle_rad": base["sigma_angle_rad"] * noise_scale}
    times, truth, geo, obs, propagate, earth, sun = build_case(config)
    std = np.asarray(plan["initial_std_m_mps"])
    initial = truth[0] + np.random.default_rng(np.random.SeedSequence([seed, 1])).normal(size=6) * std
    prior = np.diag(std ** 2)
    sigma = np.array([config["sigma_range_m"], config["sigma_angle_rad"], config["sigma_angle_rad"]])
    input_hash = hashlib.sha256(b"".join(x.tobytes() for x in (times, truth, obs, initial, prior))).hexdigest()
    arrays = {"times_s": times, "truth_m_mps": truth, "observations": obs,
              "initial_m_mps": initial, "prior_covariance": prior, "measurement_sigma_m_rad_rad": sigma}
    rows = []
    for label in ESTIMATORS:
        started = perf_counter()
        row = {"seed": seed, "noise_scale": noise_scale, "estimator": label,
               "input_sha256": input_hash, "observations": len(obs), "success": False,
               "solver_terminated": False, "outlier": False, "failure": ""}
        key = label.replace("-", "_")
        try:
            if label == "SR-UKF":
                result = run_lunar_ukf(times, obs.copy(), initial.copy(), prior.copy(), geo,
                    config["mu_moon_m3_s2"], 0., 0., earth, sun,
                    process_noise=None, covariance_form="square_root",
                    config=UnscentedTransformConfig(**plan["ukf"]),
                    rtol=config["rtol"], atol=config["atol"])
                state, covariance = result.final_state[:6], result.final_covariance[:6, :6]
                reason = "AllUpdates" if len(result.t_update_s) == len(obs) else "IncompleteUpdates"
                whitened = result.innovations / sigma
                arrays.update(ukf_innovations=result.innovations,
                              ukf_innovation_covariances=result.innovation_covariances,
                              ukf_nis=result.normalized_innovation_squared,
                              ukf_states=result.state_estimates,
                              ukf_covariances=result.covariances,
                              ukf_times_s=result.t_update_s,
                              ukf_accepted=result.accepted_updates)
                row["mean_nis"] = float(np.mean(result.normalized_innovation_squared))
                row["accepted_fraction"] = float(np.mean(result.accepted_updates))
            else:
                estimator = estimate_position_bls_lm if label == "BLS-LM" else estimate_position_srif
                estimate, reason, stats = estimator(times, obs.copy(), initial.copy(), geo,
                    config["mu_moon_m3_s2"], 0., 0., earth, sun,
                    prior_covariance=prior.copy(), return_posterior=True,
                    max_iter=config["max_iter"], rtol=config["rtol"], atol=config["atol"])
                augmented = np.r_[estimate[:6], np.eye(6).reshape(-1, order="F")]
                history = propagate_augmented_state(times, augmented, config["mu_moon_m3_s2"],
                    0., 0., earth, sun, rtol=config["rtol"], atol=config["atol"])
                state = history[-1, :6]
                phi = history[-1, 6:42].reshape(6, 6, order="F")
                covariance = phi @ stats.posterior_covariance[:6, :6] @ phi.T
                residuals = compute_position_residuals_analytic(history[:, :6], obs, geo)[0].reshape(-1, 3)
                whitened = residuals / sigma
                arrays[key + "_postfit_residuals"] = residuals
                row["iterations"] = int(stats.iterations)
                row["design_condition_number"] = float(stats.condition_number)
            row["runtime_s"] = perf_counter() - started
            error = state - truth[-1]
            pos, vel = float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))
            np.linalg.cholesky(covariance)
            nees = float(error @ np.linalg.solve(covariance, error))
            if not np.isfinite(np.r_[state, covariance.ravel(), nees]).all():
                raise ValueError("Non-finite final state/covariance/NEES")
            row.update(stop_reason=reason, solver_terminated=reason in {"AllUpdates", "Converged", "J-Stab"},
                       position_error_m=pos, velocity_error_mps=vel, nees=nees,
                       covariance_condition_number=float(np.linalg.cond(covariance)),
                       success=pos < plan["success_position_m"] and vel < plan["success_velocity_mps"],
                       outlier=pos > plan["outlier_position_m"])
            arrays[key + "_final_state"] = state
            arrays[key + "_final_covariance"] = covariance
            row["diagnostics"] = diagnostics(whitened)
        except Exception as exc:
            row.update(success=False, solver_terminated=False,
                       failure=f"{type(exc).__name__}: {exc}", runtime_s=perf_counter() - started,
                       stop_reason="Exception")
        rows.append(row)
    assert input_hash == hashlib.sha256(b"".join(x.tobytes() for x in (times, truth, obs, initial, prior))).hexdigest()
    np.savez_compressed(output / f"seed-{seed}-noise-{noise_scale:g}.npz", **arrays)
    return rows


def summarize(rows, plan):
    groups = {}
    for scale in plan["noise_scales"]:
        for label in ESTIMATORS:
            group = [r for r in rows if r["noise_scale"] == scale and r["estimator"] == label]
            n = len(group)
            nees = [r["nees"] for r in group if "nees" in r]
            groups[f"noise={scale:g}/{label}"] = {
                "attempted": n, "successes": sum(r["success"] for r in group),
                "success_fraction": sum(r["success"] for r in group) / n if n else None,
                "failure_fraction": sum(not r["success"] for r in group) / n if n else None,
                "exceptions": [r for r in group if r["failure"]],
                "solver_termination_fraction": sum(r["solver_terminated"] for r in group) / n if n else None,
                "outlier_fraction": sum(r["outlier"] for r in group) / n if n else None,
                **{field: distribution([r[field] for r in group if field in r]) for field in
                   ("position_error_m", "velocity_error_mps", "runtime_s", "nees")},
                "mean_nees_95_interval": (chi2.ppf([.025, .975], 6 * len(nees)) / len(nees)).tolist() if nees else None,
            }
    pairs = {}
    for scale in plan["noise_scales"]:
        by_key = {(r["seed"], r["estimator"]): r for r in rows if r["noise_scale"] == scale}
        for other in ("SRIF", "SR-UKF"):
            differences = [by_key[(seed, other)]["position_error_m"] - by_key[(seed, "BLS-LM")]["position_error_m"]
                           for seed in plan["seeds"] if (seed, other) in by_key and (seed, "BLS-LM") in by_key
                           and "position_error_m" in by_key[(seed, other)] and "position_error_m" in by_key[(seed, "BLS-LM")]]
            pairs[f"noise={scale:g}/{other}-minus-BLS-LM"] = distribution(differences)
    return {"groups": groups, "paired_position_difference_m": pairs}


def run(plan_path, output, pilot=False):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    base = json.loads(CONFIG.read_text(encoding="utf-8"))
    if not plan["seeds"] or len(set(plan["seeds"])) != len(plan["seeds"]):
        raise ValueError("Declare nonempty, unique seeds")
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    rows = []
    timed_out = False
    scales, seeds = (plan["noise_scales"][:1], plan["seeds"][:1]) if pilot else (plan["noise_scales"], plan["seeds"])
    for scale in scales:
        for seed in seeds:
            if perf_counter() - started >= plan["max_elapsed_s"]:
                timed_out = True
                break
            rows.extend(run_trial(seed, scale, plan, base, output))
            print(f"noise={scale:g} seed={seed}: " + ", ".join(
                f"{r['estimator']}={r.get('position_error_m', 'failed')} m" for r in rows[-3:]), flush=True)
        if timed_out:
            break
    (output / "trials.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    columns = sorted(set().union(*(row.keys() for row in rows)) - {"diagnostics"}) if rows else ["seed"]
    with (output / "trials.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    source = source_identity()
    source["source_sha256"]["examples/benchmark_estimators.py"] = sha256(Path(__file__))
    source["source_sha256"]["docs/research/paired-benchmark-plan.md"] = sha256(ROOT / "docs/research/paired-benchmark-plan.md")
    summary = {"schema": "lunar-od-paired-benchmark-result-v1", "pilot": pilot,
        "complete": not pilot and not timed_out and len(rows) == len(plan["seeds"]) * len(plan["noise_scales"]) * 3,
        "timed_out": timed_out, "truth_level": 0, "elapsed_s": perf_counter() - started,
        "plan": plan, "plan_sha256": sha256(plan_path), "base_config": base,
        "base_config_sha256": sha256(CONFIG), "source": source,
        "environment": {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
                        **{p: version(p) for p in ("numpy", "scipy")}},
        "analysis": summarize(rows, plan),
        "artifact_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    result = run(args.plan, args.output, args.pilot)
    if not args.pilot and not result["complete"]:
        raise SystemExit("Incomplete campaign; inspect saved partial results")
