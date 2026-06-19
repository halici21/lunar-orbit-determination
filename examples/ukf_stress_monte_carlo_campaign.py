"""Run a reproducible UKF stress Monte Carlo campaign from one scenario JSON.

Example:

    python python_port/examples/ukf_stress_monte_carlo_campaign.py config.json --trials 100 --max-workers 4
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_od import (  # noqa: E402
    MonteCarloCase,
    load_scenario_config_json,
    run_monte_carlo_campaign,
    write_monte_carlo_trials_csv,
)
from run_scenario_config import run_configured_scenario  # noqa: E402


def main(argv=None) -> int:
    args = _parse_args(argv)
    config = load_scenario_config_json(args.config)
    if config.estimator_type != "ukf":
        raise ValueError("UKF stress campaign requires estimator_type='ukf'.")

    cases = _campaign_cases(args.earth_position_bias_m, args.cold_start_scale)
    output_dir = Path(args.output_dir or config.output_dir)
    trials_csv = output_dir / f"{config.name}_ukf_stress_trials.csv"
    summary_csv = output_dir / f"{config.name}_ukf_stress_summary.csv"
    if args.dry_run:
        print(f"Cases: {[case.label for case in cases]}")
        print(f"Trials per case: {args.trials}")
        print(f"Max workers: {args.max_workers}")
        print(f"Would write {trials_csv}")
        print(f"Would write {summary_csv}")
        return 0

    campaign = run_monte_carlo_campaign(
        cases,
        args.trials,
        lambda case, spec, rng: _trial_metrics(config, case, spec.seed),
        base_seed=args.base_seed,
        max_workers=args.max_workers,
        continue_on_error=args.continue_on_error,
    )
    flat_trials = tuple(trial for case_result in campaign for trial in case_result.trials)
    write_monte_carlo_trials_csv(flat_trials, trials_csv)
    _write_case_summary_csv(campaign, summary_csv)
    print(f"Wrote {trials_csv}")
    print(f"Wrote {summary_csv}")
    return 0


def _campaign_cases(bias_csv: str, cold_scale_csv: str) -> tuple[MonteCarloCase, ...]:
    biases = _float_list(bias_csv)
    cold_scales = _float_list(cold_scale_csv)
    cases = []
    for bias in biases:
        for cold_scale in cold_scales:
            cases.append(
                MonteCarloCase(
                    f"earth_dx_{bias:g}m_cold_{cold_scale:g}x",
                    {
                        "earth_position_bias_m": bias,
                        "cold_start_scale": cold_scale,
                    },
                )
            )
    return tuple(cases)


def _trial_metrics(config, case: MonteCarloCase, seed: int) -> dict:
    scenario = run_configured_scenario(
        config,
        earth_position_bias_m=(float(case.parameters["earth_position_bias_m"]), 0.0, 0.0),
        measurement_seed=seed,
        cold_start_seed=seed + 1_000_000,
        cold_start_scale=float(case.parameters["cold_start_scale"]),
    )
    final_errors = np.asarray([arc.final_position_error_m for arc in scenario.arc_results], dtype=float)
    mean_nis = np.asarray([arc.ukf_mean_nis for arc in scenario.arc_results], dtype=float)
    accepted = np.asarray([arc.ukf_accepted_update_fraction for arc in scenario.arc_results], dtype=float)
    elapsed = np.asarray([arc.ukf_elapsed_s for arc in scenario.arc_results], dtype=float)
    measurement_calls = np.asarray(
        [arc.ukf_measurement_function_evaluations for arc in scenario.arc_results],
        dtype=float,
    )
    unique_measurements = np.asarray(
        [arc.ukf_unique_measurement_model_evaluations for arc in scenario.arc_results],
        dtype=float,
    )
    return {
        "success": bool(scenario.success_fraction >= 0.999),
        "final_position_error_m": _finite_mean(final_errors),
        "max_final_position_error_m": _finite_max(final_errors),
        "mean_nis": _finite_mean(mean_nis),
        "mean_accepted_update_fraction": _finite_mean(accepted),
        "ukf_elapsed_s": _finite_sum(elapsed),
        "measurement_model_reduction": _measurement_reduction(measurement_calls, unique_measurements),
        "metadata": {
            "case": case.label,
            "earth_position_bias_m": str(case.parameters["earth_position_bias_m"]),
            "cold_start_scale": str(case.parameters["cold_start_scale"]),
        },
    }


def _write_case_summary_csv(campaign, output_path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case",
                "num_trials",
                "success_fraction",
                "metric",
                "count",
                "mean",
                "std",
                "median",
                "p05",
                "p95",
                "min",
                "max",
            ]
        )
        for case_result in campaign:
            for metric in case_result.summary.metric_summaries:
                writer.writerow(
                    [
                        case_result.case.label,
                        case_result.summary.num_trials,
                        case_result.summary.success_fraction,
                        metric.metric_name,
                        metric.count,
                        metric.mean,
                        metric.std,
                        metric.median,
                        metric.p05,
                        metric.p95,
                        metric.min_value,
                        metric.max_value,
                    ]
                )
    return output_path


def _float_list(csv_text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in csv_text.split(",") if item.strip())
    if not values:
        raise ValueError("At least one numeric value is required.")
    return values


def _finite_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def _finite_max(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.max(values)) if values.size else float("nan")


def _finite_sum(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.sum(values)) if values.size else float("nan")


def _measurement_reduction(calls: np.ndarray, unique: np.ndarray) -> float:
    call_sum = _finite_sum(calls)
    unique_sum = _finite_sum(unique)
    if not np.isfinite(call_sum) or call_sum <= 0.0:
        return float("nan")
    return float(1.0 - unique_sum / call_sum)


def _parse_args(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="UKF scenario JSON path.")
    parser.add_argument("--trials", type=int, default=25, help="Trials per stress case.")
    parser.add_argument("--base-seed", type=int, default=20260609)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--earth-position-bias-m", default="0,100,1000")
    parser.add_argument("--cold-start-scale", default="0.5,1,2")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
