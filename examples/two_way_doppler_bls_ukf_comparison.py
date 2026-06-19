"""Compare BLS-LM and SR-UKF on identical two-way counted-Doppler arcs.

The campaign uses SPICE-backed truth, Earth-station visibility, and the same
measurement and initialization realizations for both estimators.

Run from the project root:

    python python_port/examples/two_way_doppler_bls_ukf_comparison.py --noise
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lunar_od import (  # noqa: E402
    RangeRatePhysicsConfig,
    UKFAdaptiveConfig,
    UnscentedTransformConfig,
    VisibilityConfig,
    analyze_visibility_gap_with_transforms,
    build_measurement_arcs,
    chi_square_nis_gate,
    load_spice_kernels,
    make_cold_start_bank,
    propagate_state,
    propagate_truth_with_ephemeris,
    range_rate_stations,
    run_batch_arc_sequence,
    sample_j2000_to_itrf93_transforms,
    sample_moon_centered_ephemeris,
    write_scenario_summary_csv,
    write_visibility_summary_csv,
)

from baseline_bls_ukf_comparison import (  # noqa: E402
    _plot_runtime_success_consistency,
    _safe_max,
    _safe_percentile,
    _scenario_reduced_chi_square,
)


def main() -> None:
    args = _parse_args()
    context = _build_experiment(args)
    arcs = context["arcs"]
    if args.max_arcs is not None:
        arcs = _select_representative_arcs(arcs, args.max_arcs)
    if not arcs:
        raise RuntimeError("No OD-ready two-way Doppler arcs were produced.")

    cold_bank = make_cold_start_bank(
        len(arcs),
        sigma_pos_m=args.cold_sigma_pos_m,
        sigma_vel_mps=args.cold_sigma_vel_mps,
        seed=args.seed,
    )
    scenarios = []
    runtimes: dict[str, float] = {}
    for start_mode in args.start_modes:
        for estimator_type in ("bls_lm", "ukf"):
            label = f"Two-way {estimator_type.upper()} {start_mode}"
            print(f"Running {label} on {len(arcs)} arcs...")
            started = perf_counter()
            scenario = run_batch_arc_sequence(
                arcs,
                "range_rate",
                start_mode,
                estimator_type,
                context["mu_moon"],
                context["mu_earth"],
                context["mu_sun"],
                context["earth_position"],
                context["sun_position"],
                cold_start_bank=cold_bank,
                label=label,
                max_iter=args.max_iter,
                tol_cost_stability=args.tol_cost_stability,
                rtol=args.rtol,
                atol=args.atol,
                process_noise_covariance=_process_noise(args, estimator_type, start_mode),
                ukf_transform_config=(
                    UnscentedTransformConfig(alpha=args.ukf_alpha)
                    if estimator_type == "ukf"
                    else None
                ),
                ukf_adaptive_config=(
                    _ukf_adaptive_config(args)
                    if estimator_type == "ukf"
                    else None
                ),
                ukf_covariance_form="square_root",
            )
            runtimes[label] = perf_counter() - started
            scenarios.append(scenario)

    out_dir = Path("python_port") / "results" / "two_way_doppler_bls_ukf"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = _suffix(args, len(arcs))
    detail_csv = write_scenario_summary_csv(
        scenarios,
        out_dir / f"two_way_bls_ukf_{suffix}_arc_summary.csv",
    )
    aggregate_csv = _write_aggregate_csv(
        scenarios,
        runtimes,
        arcs,
        context,
        args,
        out_dir / f"two_way_bls_ukf_{suffix}_aggregate.csv",
    )
    visibility_csv = write_visibility_summary_csv(
        context["t_eval_s"],
        [station.name for station in context["stations"]],
        context["vis_mask_raw"],
        context["net_vis_filled"],
        context["seg_starts"],
        context["seg_ends"],
        out_dir / f"two_way_bls_ukf_{suffix}_visibility.csv",
    )
    comparison_png = _plot_aligned_error_summary(
        scenarios,
        arcs,
        context,
        args,
        out_dir / f"two_way_bls_ukf_{suffix}_errors.png",
        title="Two-Way Counted Doppler: BLS-LM vs SR-UKF",
    )
    runtime_png = _plot_runtime_success_consistency(
        scenarios,
        runtimes,
        out_dir / f"two_way_bls_ukf_{suffix}_runtime_success.png",
        title="Two-Way Counted Doppler Runtime and Consistency",
    )

    for scenario in scenarios:
        errors, _ = _aligned_arc_end_errors(scenario, arcs, context, args)
        print(
            f"{scenario.label}: median={_safe_percentile(errors, 50):.3f} m, "
            f"p95={_safe_percentile(errors, 95):.3f} m, "
            f"operational_success={scenario.operational_success_fraction:.3f}, "
            f"runtime={runtimes[scenario.label]:.3f} s"
        )
    print(f"Wrote {aggregate_csv}")
    print(f"Wrote {detail_csv}")
    print(f"Wrote {visibility_csv}")
    print(f"Wrote {comparison_png}")
    print(f"Wrote {runtime_png}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-h", type=float, default=24.0)
    parser.add_argument("--sample-step-s", type=float, default=300.0)
    parser.add_argument("--ephemeris-step-s", type=float, default=1800.0)
    parser.add_argument("--count-interval-s", type=float, default=60.0)
    parser.add_argument("--local-state-model", choices=("ode", "taylor3"), default="taylor3")
    parser.add_argument("--max-gap-s", type=float, default=1800.0)
    parser.add_argument("--min-elevation-deg", type=float, default=5.0)
    parser.add_argument("--min-samples", type=int, default=4)
    parser.add_argument("--max-arcs", type=int, default=4)
    parser.add_argument("--start-modes", nargs="+", choices=("cold", "hot", "formal"), default=("cold",))
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--cold-sigma-pos-m", type=float, default=150.0)
    parser.add_argument("--cold-sigma-vel-mps", type=float, default=0.05)
    parser.add_argument("--handoff-sigma-pos-m", type=float, default=0.1)
    parser.add_argument("--handoff-sigma-vel-mps", type=float, default=1e-4)
    parser.add_argument("--strict-outlier-gate", action="store_true")
    parser.add_argument("--outlier-gate-sigma", type=float, default=3.0)
    parser.add_argument("--ukf-alpha", type=float, default=0.35)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--tol-cost-stability", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-11)
    parser.add_argument(
        "--stations",
        nargs="+",
        default=("Goldstone DSN", "Madrid DSN", "Canberra DSN", "ITU Ayazaga"),
    )
    return parser.parse_args()


def _build_experiment(args: argparse.Namespace) -> dict:
    fixture = json.loads(
        (Path("python_port") / "fixtures" / "spice_snapshots.json").read_text(encoding="utf-8")
    )
    initial = fixture["initial_state"]
    constants = fixture["constants"]
    epoch_utc = fixture["epoch_utc"]
    x0_mci = np.asarray(initial["state_mci_j2000_m_mps"], dtype=float)
    mu_moon = float(initial["mu_moon_m3_s2"])
    mu_earth = float(np.asarray(constants["mu_earth_km3_s2"]).reshape(-1)[0] * 1e9)
    mu_sun = float(np.asarray(constants["mu_sun_km3_s2"]).reshape(-1)[0] * 1e9)
    duration_s = float(args.duration_h) * 3600.0
    t_eval_s = np.arange(0.0, duration_s + args.sample_step_s, args.sample_step_s)
    t_ephem_s = np.arange(0.0, duration_s + args.ephemeris_step_s, args.ephemeris_step_s)

    import spiceypy as spice

    load_spice_kernels()
    try:
        et0 = float(spice.str2et(epoch_utc))
        ephemeris = sample_moon_centered_ephemeris(et0, t_ephem_s)
        xforms = sample_j2000_to_itrf93_transforms(et0, t_eval_s)
        truth = propagate_truth_with_ephemeris(
            t_eval_s,
            x0_mci,
            mu_moon,
            mu_earth,
            mu_sun,
            ephemeris,
            rtol=args.rtol,
            atol=args.atol,
        )
        station_by_name = {station.name: station for station in range_rate_stations()}
        stations = tuple(station_by_name[name] for name in args.stations)
        visibility_config = VisibilityConfig(
            r_moon_mean_m=float(initial["r_moon_mean_m"]),
            earth_rotation_rad_s=7.292115e-5,
            epoch_utc=epoch_utc,
            min_elevation_deg=args.min_elevation_deg,
        )
        seg_starts, seg_ends, vis_mask_raw, net_vis_filled = analyze_visibility_gap_with_transforms(
            t_eval_s,
            truth,
            stations,
            ephemeris.earth_position,
            xforms,
            args.max_gap_s,
            visibility_config,
        )
        physics = RangeRatePhysicsConfig(
            mode="two_way_counted_doppler",
            count_interval_s=args.count_interval_s,
            output_unit="mps_equivalent",
            local_state_model=args.local_state_model,
        )
        arcs = build_measurement_arcs(
            "range_rate",
            t_eval_s,
            truth,
            seg_starts,
            seg_ends,
            vis_mask_raw,
            stations,
            ephemeris.earth_position,
            ephemeris.earth_velocity,
            et0,
            noise=args.noise,
            rng=np.random.default_rng(args.seed + 1),
            min_samples=args.min_samples,
            range_rate_physics=physics,
        )
        arcs = _trim_two_way_boundary_observations(arcs, physics)
    finally:
        spice.kclear()

    return {
        "mu_moon": mu_moon,
        "mu_earth": mu_earth,
        "mu_sun": mu_sun,
        "earth_position": ephemeris.earth_position,
        "sun_position": ephemeris.sun_position,
        "t_eval_s": t_eval_s,
        "stations": stations,
        "seg_starts": seg_starts,
        "seg_ends": seg_ends,
        "vis_mask_raw": vis_mask_raw,
        "net_vis_filled": net_vis_filled,
        "arcs": arcs,
    }


def _trim_two_way_boundary_observations(arcs, physics: RangeRatePhysicsConfig):
    """Drop epochs whose counted-Doppler interval extends outside the arc history."""
    trimmed = []
    for arc in arcs:
        light_time_margin_s = 5.0
        margin_s = 0.5 * float(physics.count_interval_s) + light_time_margin_s
        lower = float(arc.t_pass_s[0]) + margin_s
        upper = float(arc.t_pass_s[-1]) - margin_s
        keep = (arc.obs_data[:, 0] >= lower) & (arc.obs_data[:, 0] <= upper)
        obs_data = np.asarray(arc.obs_data[keep], dtype=float)
        if obs_data.shape[0] < 4:
            continue
        trimmed.append(replace(arc, obs_data=obs_data))
    return tuple(trimmed)


def _select_representative_arcs(arcs, max_arcs: int):
    arcs = tuple(arcs)
    if len(arcs) <= max_arcs:
        return arcs
    order = np.argsort([arc.t_pass_s[-1] - arc.t_pass_s[0] for arc in arcs])[::-1]
    selected = sorted(order[:max_arcs])
    return tuple(arcs[int(index)] for index in selected)


def _process_noise(args, estimator_type: str, start_mode: str) -> np.ndarray | None:
    if estimator_type == "ukf":
        return np.diag([0.01**2] * 3 + [1e-5**2] * 3)
    if start_mode == "formal":
        return np.diag(
            [args.handoff_sigma_pos_m**2] * 3
            + [args.handoff_sigma_vel_mps**2] * 3
        )
    return None


def _ukf_adaptive_config(args) -> UKFAdaptiveConfig:
    if not args.strict_outlier_gate:
        return UKFAdaptiveConfig()
    return UKFAdaptiveConfig(
        nis_gate=chi_square_nis_gate(4, sigma=args.outlier_gate_sigma),
        component_nis_gate=args.outlier_gate_sigma**2,
        component_gate_mode="conditional",
    )


def _write_aggregate_csv(
    scenarios,
    runtimes: dict[str, float],
    arcs,
    context: dict,
    args: argparse.Namespace,
    output_path: Path,
) -> Path:
    fieldnames = [
        "scenario",
        "estimator_type",
        "start_mode",
        "range_rate_physics",
        "count_interval_s",
        "num_arcs",
        "algorithmic_success_fraction",
        "operational_success_fraction",
        "arc_end_accuracy_success_fraction",
        "median_final_position_error_m",
        "p95_final_position_error_m",
        "max_final_position_error_m",
        "median_final_velocity_error_mps",
        "p95_final_velocity_error_mps",
        "median_consistency",
        "median_condition_number",
        "median_iteration_or_update_count",
        "median_accepted_update_fraction",
        "process_function_evaluations",
        "measurement_function_evaluations",
        "unique_measurement_model_evaluations",
        "runtime_s",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for scenario in scenarios:
            final_position, final_velocity = _aligned_arc_end_errors(
                scenario,
                arcs,
                context,
                args,
            )
            reduced_chi_square = _scenario_reduced_chi_square(scenario)
            consistency = (
                _safe_percentile([arc.ukf_normalized_mean_nis for arc in scenario.arc_results], 50)
                if scenario.estimator_type == "ukf"
                else _safe_percentile(reduced_chi_square, 50)
            )
            writer.writerow(
                {
                    "scenario": scenario.label,
                    "estimator_type": scenario.estimator_type,
                    "start_mode": scenario.start_mode,
                    "range_rate_physics": scenario.range_rate_physics,
                    "count_interval_s": scenario.count_interval_s,
                    "num_arcs": len(scenario.arc_results),
                    "algorithmic_success_fraction": scenario.algorithmic_success_fraction,
                    "operational_success_fraction": scenario.operational_success_fraction,
                    "arc_end_accuracy_success_fraction": float(np.mean(final_position <= 100.0)),
                    "median_final_position_error_m": _safe_percentile(final_position, 50),
                    "p95_final_position_error_m": _safe_percentile(final_position, 95),
                    "max_final_position_error_m": _safe_max(final_position),
                    "median_final_velocity_error_mps": _safe_percentile(final_velocity, 50),
                    "p95_final_velocity_error_mps": _safe_percentile(final_velocity, 95),
                    "median_consistency": consistency,
                    "median_condition_number": _safe_percentile(
                        [arc.stats.condition_number for arc in scenario.arc_results],
                        50,
                    ),
                    "median_iteration_or_update_count": _safe_percentile(
                        [arc.stats.iterations for arc in scenario.arc_results],
                        50,
                    ),
                    "median_accepted_update_fraction": _safe_percentile(
                        [arc.ukf_accepted_update_fraction for arc in scenario.arc_results],
                        50,
                    ),
                    "process_function_evaluations": sum(
                        arc.ukf_process_function_evaluations for arc in scenario.arc_results
                    ),
                    "measurement_function_evaluations": sum(
                        arc.ukf_measurement_function_evaluations for arc in scenario.arc_results
                    ),
                    "unique_measurement_model_evaluations": sum(
                        arc.ukf_unique_measurement_model_evaluations for arc in scenario.arc_results
                    ),
                    "runtime_s": runtimes[scenario.label],
                }
            )
    return output_path


def _aligned_arc_end_errors(
    scenario,
    arcs,
    context: dict,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    arc_by_id = {int(arc.arc_id): arc for arc in arcs}
    position_errors = []
    velocity_errors = []
    for result in scenario.arc_results:
        arc = arc_by_id[int(result.arc_id)]
        if scenario.estimator_type == "ukf":
            final_measurement_time = float(np.max(arc.obs_data[:, 0]))
            estimate_at_end = propagate_state(
                [final_measurement_time, float(arc.t_pass_s[-1])],
                np.asarray(result.estimated_state, dtype=float)[:6],
                context["mu_moon"],
                context["mu_earth"],
                context["mu_sun"],
                context["earth_position"],
                context["sun_position"],
                rtol=args.rtol,
                atol=args.atol,
            )[-1]
        else:
            estimate_at_end = propagate_state(
                [float(arc.t_pass_s[0]), float(arc.t_pass_s[-1])],
                np.asarray(result.estimated_state, dtype=float)[:6],
                context["mu_moon"],
                context["mu_earth"],
                context["mu_sun"],
                context["earth_position"],
                context["sun_position"],
                rtol=args.rtol,
                atol=args.atol,
            )[-1]
        truth_at_end = np.asarray(arc.truth_state_history_mci[-1, :6], dtype=float)
        position_errors.append(float(np.linalg.norm(estimate_at_end[:3] - truth_at_end[:3])))
        velocity_errors.append(float(np.linalg.norm(estimate_at_end[3:6] - truth_at_end[3:6])))
    return np.asarray(position_errors, dtype=float), np.asarray(velocity_errors, dtype=float)


def _plot_aligned_error_summary(
    scenarios,
    arcs,
    context: dict,
    args: argparse.Namespace,
    output_path: Path,
    *,
    title: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    labels = [scenario.label for scenario in scenarios]
    aligned = [
        _aligned_arc_end_errors(scenario, arcs, context, args)
        for scenario in scenarios
    ]
    x = np.arange(len(scenarios))
    median_position = [_safe_percentile(values[0], 50) for values in aligned]
    p95_position = [_safe_percentile(values[0], 95) for values in aligned]
    median_velocity = [_safe_percentile(values[1], 50) for values in aligned]
    p95_velocity = [_safe_percentile(values[1], 95) for values in aligned]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    fig.suptitle(f"{title} (errors aligned at arc end)")
    axes[0].bar(x - 0.18, median_position, width=0.36, label="median")
    axes[0].bar(x + 0.18, p95_position, width=0.36, label="p95")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("arc-end position error [m]")
    axes[0].legend()
    axes[1].bar(x - 0.18, median_velocity, width=0.36, label="median")
    axes[1].bar(x + 0.18, p95_velocity, width=0.36, label="p95")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("arc-end velocity error [m/s]")
    axes[1].legend()
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.grid(True, axis="y", which="both", alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _suffix(args: argparse.Namespace, num_arcs: int) -> str:
    noise = "noisy" if args.noise else "clean"
    modes = "-".join(args.start_modes)
    suffix = (
        f"{args.duration_h:g}h_step{args.sample_step_s:g}s_tc{args.count_interval_s:g}s_"
        f"{args.local_state_model}_{noise}_{modes}_{num_arcs}arc"
    )
    if args.strict_outlier_gate:
        suffix += "_strictgate"
    return suffix.replace(".", "p")


if __name__ == "__main__":
    main()
