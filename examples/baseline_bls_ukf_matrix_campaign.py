"""Run compact BLS-vs-UKF comparison matrices.

This script builds on ``baseline_bls_ukf_comparison.py`` and produces two
thesis-facing campaign surfaces:

- Gaussian Monte Carlo over multiple measurement/cold-start seeds.
- Initial-error scale sweep over 0.5x / 1x / 2x / 4x style cases.

Run from the project root, for example:

    python python_port/examples/baseline_bls_ukf_matrix_campaign.py --campaign gaussian_mc --num-seeds 20
    python python_port/examples/baseline_bls_ukf_matrix_campaign.py --campaign initial_sweep
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lunar_od import (  # noqa: E402
    UKFAdaptiveConfig,
    UnscentedTransformConfig,
    chi_square_nis_gate,
    normalized_estimation_error_squared,
    run_batch_arc_sequence,
)

from baseline_bls_ukf_comparison import (  # noqa: E402
    _build_experiment,
    _cold_start_bank,
    _plot_runtime_success_consistency,
    _safe_max,
    _safe_percentile,
    _scenario_reduced_chi_square,
)


def main() -> None:
    args = _parse_args()
    out_dir = Path("python_port") / "results" / "baseline_bls_ukf_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = _suffix(args)
    trials_path = out_dir / f"baseline_matrix_{suffix}_trials.csv"
    summary_path = out_dir / f"baseline_matrix_{suffix}_summary.csv"
    trial_rows = []
    for case_label, error_scale, seed_index in _case_grid(args):
        print(f"Running {case_label} seed_index={seed_index} error_scale={error_scale:g}...")
        trial_rows.extend(_run_case(args, case_label, error_scale, seed_index))
        _write_trials_csv(trial_rows, trials_path)
        _write_summary_csv(trial_rows, summary_path)

    trials_csv = _write_trials_csv(trial_rows, trials_path)
    summary_csv = _write_summary_csv(trial_rows, summary_path)
    plot_path = _plot_campaign_summary(trial_rows, out_dir / f"baseline_matrix_{suffix}_summary.png")
    print(f"Wrote {trials_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {plot_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=("gaussian_mc", "initial_sweep"), default="gaussian_mc")
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--duration-days", type=float, default=1.0)
    parser.add_argument("--sample-step-s", type=float, default=600.0)
    parser.add_argument("--arc-duration-h", type=float, default=2.0)
    parser.add_argument("--arc-stride-h", type=float, default=6.0)
    parser.add_argument("--max-arcs", type=int)
    parser.add_argument("--start-modes", nargs="+", choices=("cold", "hot", "formal"), default=("cold",))
    parser.add_argument("--error-scales", nargs="+", type=float, default=(0.5, 1.0, 2.0, 4.0))
    parser.add_argument("--cold-sigma-pos-m", type=float, default=150.0)
    parser.add_argument("--cold-sigma-vel-mps", type=float, default=0.05)
    parser.add_argument("--handoff-sigma-pos-m", type=float, default=0.1)
    parser.add_argument("--handoff-sigma-vel-mps", type=float, default=1e-4)
    parser.add_argument("--sigma-range-m", type=float, default=5.0)
    parser.add_argument("--sigma-angle-rad", type=float, default=1e-5)
    parser.add_argument("--strict-outlier-gate", action="store_true")
    parser.add_argument("--outlier-gate-sigma", type=float, default=3.0)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--rtol", type=float, default=1e-11)
    parser.add_argument("--atol", type=float, default=1e-12)
    return parser.parse_args()


def _case_grid(args: argparse.Namespace):
    if args.campaign == "gaussian_mc":
        for seed_index in range(int(args.num_seeds)):
            yield "gaussian_mc", 1.0, seed_index
    else:
        for scale in args.error_scales:
            for seed_index in range(int(args.num_seeds)):
                yield f"initial_error_{scale:g}x", float(scale), seed_index


def _run_case(args: argparse.Namespace, case_label: str, error_scale: float, seed_index: int) -> list[dict]:
    baseline_args = argparse.Namespace(
        arc_source="regular",
        duration_days=args.duration_days,
        sample_step_s=args.sample_step_s,
        ephem_step_s=3600.0,
        arc_duration_h=args.arc_duration_h,
        arc_stride_h=args.arc_stride_h,
        max_arcs=args.max_arcs,
        start_modes=tuple(args.start_modes),
        cold_sigma_pos_m=args.cold_sigma_pos_m * error_scale,
        cold_sigma_vel_mps=args.cold_sigma_vel_mps * error_scale,
        handoff_sigma_pos_m=args.handoff_sigma_pos_m,
        handoff_sigma_vel_mps=args.handoff_sigma_vel_mps,
        batch_estimator="bls_lm",
        seed=args.seed + 1000 * seed_index + 17,
        measurement_noise=True,
        measurement_seed=args.seed + 1000 * seed_index + 23,
        sigma_range_m=args.sigma_range_m,
        sigma_angle_rad=args.sigma_angle_rad,
        strict_outlier_gate=args.strict_outlier_gate,
        outlier_gate_sigma=args.outlier_gate_sigma,
        visibility_network="dsn4",
        min_elevation_deg=5.0,
        max_gap_s=1800.0,
        min_visibility_samples=4,
        max_iter=args.max_iter,
        rtol=args.rtol,
        atol=args.atol,
    )
    experiment = _build_experiment(baseline_args)
    arcs = experiment["arcs"]
    if args.max_arcs is not None:
        arcs = arcs[: args.max_arcs]
    cold_bank = _cold_start_bank(
        len(arcs),
        sigma_pos_m=baseline_args.cold_sigma_pos_m,
        sigma_vel_mps=baseline_args.cold_sigma_vel_mps,
        seed=baseline_args.seed,
    )

    rows = []
    for start_mode in baseline_args.start_modes:
        for estimator_type in ("bls_lm", "ukf"):
            scenario, runtime_s = _run_scenario(baseline_args, experiment, arcs, cold_bank, start_mode, estimator_type)
            rows.append(_trial_row(args.campaign, case_label, error_scale, seed_index, scenario, arcs, runtime_s))
    return rows


def _run_scenario(args, experiment, arcs, cold_bank, start_mode: str, estimator_type: str):
    start = perf_counter()
    scenario = run_batch_arc_sequence(
        arcs,
        "position",
        start_mode,
        estimator_type,
        experiment["mu_moon"],
        experiment["mu_earth"],
        experiment["mu_sun"],
        experiment["get_earth_pos"],
        experiment["get_sun_pos"],
        cold_start_bank=cold_bank,
        label=f"{estimator_type.upper()} {start_mode}",
        max_iter=args.max_iter,
        rtol=args.rtol,
        atol=args.atol,
        process_noise_covariance=np.diag([0.01**2] * 3 + [1e-5**2] * 3)
        if estimator_type == "ukf"
        else (
            np.diag([args.handoff_sigma_pos_m**2] * 3 + [args.handoff_sigma_vel_mps**2] * 3)
            if start_mode == "formal"
            else None
        ),
        ukf_transform_config=UnscentedTransformConfig(alpha=0.35) if estimator_type == "ukf" else None,
        ukf_adaptive_config=_ukf_adaptive_config(args) if estimator_type == "ukf" else None,
        ukf_covariance_form="square_root",
    )
    return scenario, perf_counter() - start


def _ukf_adaptive_config(args) -> UKFAdaptiveConfig:
    if not args.strict_outlier_gate:
        return UKFAdaptiveConfig()
    return UKFAdaptiveConfig(
        nis_gate=chi_square_nis_gate(3, sigma=args.outlier_gate_sigma),
        component_nis_gate=args.outlier_gate_sigma**2,
        component_gate_mode="conditional",
    )


def _trial_row(
    campaign: str,
    case_label: str,
    error_scale: float,
    seed_index: int,
    scenario,
    arcs,
    runtime_s: float,
) -> dict:
    final_pos = scenario.final_position_errors_m
    final_vel = np.asarray([arc.final_velocity_error_mps for arc in scenario.arc_results], dtype=float)
    conditions = np.asarray([arc.stats.condition_number for arc in scenario.arc_results], dtype=float)
    reduced_chi_square = _scenario_reduced_chi_square(scenario)
    nees_values = _scenario_nees(scenario, arcs)
    final_nees = float(nees_values[-1]) if nees_values.size else float("nan")
    consistency = (
        _safe_percentile([arc.ukf_normalized_mean_nis for arc in scenario.arc_results], 50)
        if scenario.estimator_type == "ukf"
        else _safe_percentile(reduced_chi_square, 50)
    )
    return {
        "campaign": campaign,
        "case": case_label,
        "error_scale": float(error_scale),
        "seed_index": int(seed_index),
        "scenario": scenario.label,
        "estimator_type": scenario.estimator_type,
        "start_mode": scenario.start_mode,
        "num_arcs": len(scenario.arc_results),
        "median_final_position_error_m": _safe_percentile(final_pos, 50),
        "p95_final_position_error_m": _safe_percentile(final_pos, 95),
        "max_final_position_error_m": _safe_max(final_pos),
        "median_final_velocity_error_mps": _safe_percentile(final_vel, 50),
        "p95_final_velocity_error_mps": _safe_percentile(final_vel, 95),
        "final_nees_6d": final_nees,
        "median_arc_nees_6d": _safe_percentile(nees_values, 50),
        "median_normalized_nis": _safe_percentile(
            [arc.ukf_normalized_mean_nis for arc in scenario.arc_results],
            50,
        ),
        "median_reduced_chi_square": _safe_percentile(reduced_chi_square, 50),
        "operational_success_fraction": scenario.operational_success_fraction,
        "algorithmic_success_fraction": scenario.algorithmic_success_fraction,
        "median_consistency": consistency,
        "median_condition_number": _safe_percentile(conditions, 50),
        "runtime_s": float(runtime_s),
    }


def _scenario_nees(scenario, arcs) -> np.ndarray:
    truth_by_arc = {int(arc.arc_id): arc for arc in arcs}
    values = []
    for result in scenario.arc_results:
        covariance = result.posterior_covariance
        truth_arc = truth_by_arc.get(int(result.arc_id))
        if covariance is None or truth_arc is None:
            continue
        truth_index = -1 if scenario.estimator_type == "ukf" else 0
        truth = np.asarray(truth_arc.truth_state_history_mci[truth_index, :6], dtype=float)
        covariance_6d = np.asarray(covariance, dtype=float)[:6, :6]
        try:
            nees = normalized_estimation_error_squared(result.estimated_state[:6], truth, covariance_6d)
        except (ValueError, np.linalg.LinAlgError):
            nees = float("nan")
        if np.isfinite(nees):
            values.append(float(nees))
    return np.asarray(values, dtype=float)


def _ensemble_nees_bounds(num_trials: int, state_dimension: int = 6, confidence: float = 0.95) -> tuple[float, float]:
    if num_trials <= 0:
        return float("nan"), float("nan")
    tail = (1.0 - float(confidence)) / 2.0
    degrees_of_freedom = int(num_trials) * int(state_dimension)
    return (
        float(chi2.ppf(tail, degrees_of_freedom) / num_trials),
        float(chi2.ppf(1.0 - tail, degrees_of_freedom) / num_trials),
    )


def _write_trials_csv(rows: list[dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _write_summary_csv(rows: list[dict], output_path: Path) -> Path:
    grouped = {}
    for row in rows:
        key = (row["case"], row["estimator_type"], row["start_mode"])
        grouped.setdefault(key, []).append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case",
                "estimator_type",
                "start_mode",
                "num_trials",
                "median_of_median_final_position_error_m",
                "p95_of_median_final_position_error_m",
                "max_of_max_final_position_error_m",
                "median_of_median_final_velocity_error_mps",
                "p95_of_median_final_velocity_error_mps",
                "median_operational_success_fraction",
                "median_consistency",
                "mean_final_nees_6d",
                "median_final_nees_6d",
                "nees_95_lower",
                "nees_95_upper",
                "nees_95_consistent",
                "median_normalized_nis",
                "median_reduced_chi_square",
                "median_runtime_s",
            ]
        )
        for key, values in sorted(grouped.items()):
            final_nees = np.asarray([item["final_nees_6d"] for item in values], dtype=float)
            final_nees = final_nees[np.isfinite(final_nees)]
            nees_lower, nees_upper = _ensemble_nees_bounds(int(final_nees.size))
            mean_final_nees = float(np.mean(final_nees)) if final_nees.size else float("nan")
            writer.writerow(
                [
                    key[0],
                    key[1],
                    key[2],
                    len(values),
                    _safe_percentile([item["median_final_position_error_m"] for item in values], 50),
                    _safe_percentile([item["median_final_position_error_m"] for item in values], 95),
                    _safe_max([item["max_final_position_error_m"] for item in values]),
                    _safe_percentile([item["median_final_velocity_error_mps"] for item in values], 50),
                    _safe_percentile([item["median_final_velocity_error_mps"] for item in values], 95),
                    _safe_percentile([item["operational_success_fraction"] for item in values], 50),
                    _safe_percentile([item["median_consistency"] for item in values], 50),
                    mean_final_nees,
                    _safe_percentile(final_nees, 50),
                    nees_lower,
                    nees_upper,
                    bool(nees_lower <= mean_final_nees <= nees_upper) if final_nees.size else False,
                    _safe_percentile([item["median_normalized_nis"] for item in values], 50),
                    _safe_percentile([item["median_reduced_chi_square"] for item in values], 50),
                    _safe_percentile([item["runtime_s"] for item in values], 50),
                ]
            )
    return output_path


def _plot_campaign_summary(rows: list[dict], output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    grouped = {}
    for row in rows:
        key = f"{row['case']} / {row['estimator_type']} / {row['start_mode']}"
        grouped.setdefault(key, []).append(row)
    labels = list(grouped.keys())
    x = np.arange(len(labels))
    median_errors = np.asarray(
        [_safe_percentile([row["median_final_position_error_m"] for row in grouped[label]], 50) for label in labels],
        dtype=float,
    )
    p95_errors = np.asarray(
        [_safe_percentile([row["median_final_position_error_m"] for row in grouped[label]], 95) for label in labels],
        dtype=float,
    )
    success = np.asarray(
        [_safe_percentile([row["operational_success_fraction"] for row in grouped[label]], 50) for label in labels],
        dtype=float,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.5), constrained_layout=True)
    axes[0].bar(x - 0.18, median_errors, width=0.36, label="median", color="#2563eb")
    axes[0].bar(x + 0.18, p95_errors, width=0.36, label="p95", color="#f59e0b")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("final position error [m]")
    axes[0].set_title("Campaign error summary")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=28, ha="right")
    axes[0].grid(True, axis="y", which="both", alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].bar(x, success, color="#16a34a")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("fraction")
    axes[1].set_title("Operational success")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=28, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _suffix(args: argparse.Namespace) -> str:
    days = str(args.duration_days).replace(".", "p")
    modes = "-".join(args.start_modes)
    if args.campaign == "gaussian_mc":
        return f"gaussian_mc_{args.num_seeds}seed_{days}d_{modes}"
    scales = "-".join(str(scale).replace(".", "p") for scale in args.error_scales)
    return f"initial_sweep_{args.num_seeds}seed_{days}d_{modes}_{scales}"


if __name__ == "__main__":
    main()
