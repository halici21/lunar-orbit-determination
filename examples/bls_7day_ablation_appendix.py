"""Run a three-day BLS-LM ablation campaign for the thesis appendix.

The campaign uses the first chronological OD-ready SPICE visibility arcs that
satisfy a predefined duration and sample-count rule for:

- range/azimuth/elevation observations,
- geometric range-rate observations,
- simplified two-way counted-Doppler observations,
- cold, hot, and formal starts,
- standard and robust range-rate BLS-LM under injected outliers.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_od import (  # noqa: E402
    RangeRatePhysicsConfig,
    VisibilityConfig,
    analyze_visibility_gap_with_transforms,
    build_measurement_arcs,
    load_spice_kernels,
    make_cold_start_bank,
    propagate_truth_with_ephemeris,
    range_rate_stations,
    run_batch_arc_sequence,
    sample_j2000_to_itrf93_transforms,
    sample_moon_centered_ephemeris,
    write_scenario_summary_csv,
)


def main() -> None:
    args = _parse_args()
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
    duration_s = args.duration_days * 86400.0
    t_eval_s = np.arange(0.0, duration_s + args.sample_step_s, args.sample_step_s)
    t_ephem_s = np.arange(0.0, duration_s + args.ephemeris_step_s, args.ephemeris_step_s)

    import spiceypy as spice

    load_spice_kernels()
    try:
        et0 = float(spice.str2et(epoch_utc))
        print(f"Sampling {args.duration_days:g}-day SPICE geometry ({t_eval_s.size} state epochs)...")
        ephemeris = sample_moon_centered_ephemeris(et0, t_ephem_s)
        transforms = sample_j2000_to_itrf93_transforms(et0, t_eval_s)
        states = propagate_truth_with_ephemeris(
            t_eval_s,
            x0_mci,
            mu_moon,
            mu_earth,
            mu_sun,
            ephemeris,
            rtol=1e-10,
            atol=1e-11,
        )

        station_map = {station.name: station for station in range_rate_stations()}
        station_names = ["ITU Ayazaga", "Goldstone DSN", "Madrid DSN", "Canberra DSN"]
        stations = [station_map[name] for name in station_names]
        visibility = VisibilityConfig(
            r_moon_mean_m=float(initial["r_moon_mean_m"]),
            earth_rotation_rad_s=7.292115e-5,
            epoch_utc=epoch_utc,
            min_elevation_deg=5.0,
        )
        starts, ends, raw_masks, _ = analyze_visibility_gap_with_transforms(
            t_eval_s,
            states,
            stations,
            ephemeris.earth_position,
            transforms,
            args.max_gap_s,
            visibility,
        )

        rng_position = np.random.default_rng(args.seed)
        rng_geometric = np.random.default_rng(args.seed + 1)
        rng_two_way = np.random.default_rng(args.seed + 2)
        arc_sets = {
            "Position": build_measurement_arcs(
                "position",
                t_eval_s,
                states,
                starts,
                ends,
                raw_masks,
                stations,
                ephemeris.earth_position,
                ephemeris.earth_velocity,
                et0,
                noise=True,
                rng=rng_position,
                min_samples=args.min_samples,
            ),
            "Geometric RR": build_measurement_arcs(
                "range_rate",
                t_eval_s,
                states,
                starts,
                ends,
                raw_masks,
                stations,
                ephemeris.earth_position,
                ephemeris.earth_velocity,
                et0,
                noise=True,
                rng=rng_geometric,
                min_samples=args.min_samples,
                range_rate_physics=RangeRatePhysicsConfig(mode="geometric_instantaneous"),
            ),
            "Two-way Doppler": build_measurement_arcs(
                "range_rate",
                t_eval_s,
                states,
                starts,
                ends,
                raw_masks,
                stations,
                ephemeris.earth_position,
                ephemeris.earth_velocity,
                et0,
                noise=True,
                rng=rng_two_way,
                min_samples=args.min_samples,
                range_rate_physics=RangeRatePhysicsConfig(
                    mode="two_way_counted_doppler",
                    count_interval_s=args.count_interval_s,
                    output_unit="mps_equivalent",
                ),
            ),
        }
    finally:
        spice.kclear()

    selected = _select_common_arcs(
        arc_sets,
        max_arcs=args.max_arcs,
        min_duration_s=args.min_arc_duration_min * 60.0,
    )
    print("Chronological OD-ready arc IDs:", [arc.arc_id for arc in selected["Position"]])
    for arc in selected["Position"]:
        print(
            f"  arc {arc.arc_id}: "
            f"{(arc.t_pass_s[-1] - arc.t_pass_s[0]) / 60.0:.1f} min, "
            f"{arc.t_pass_s.size} epochs, {arc.obs_data.shape[0]} observations"
        )
    cold_bank = make_cold_start_bank(
        len(selected["Position"]),
        sigma_pos_m=250.0,
        sigma_vel_mps=0.05,
        seed=args.seed + 10,
    )

    scenarios = []
    runtimes: dict[str, float] = {}
    for model_label, arcs in selected.items():
        measurement_type = "position" if model_label == "Position" else "range_rate"
        for start_mode in ("cold", "hot", "formal"):
            label = f"{model_label} / {start_mode}"
            print(f"Running {label} over {len(arcs)} arcs...")
            started = time.perf_counter()
            scenario = run_batch_arc_sequence(
                arcs,
                measurement_type,
                start_mode,
                "bls_lm",
                mu_moon,
                mu_earth,
                mu_sun,
                ephemeris.earth_position,
                ephemeris.sun_position,
                cold_start_bank=cold_bank,
                label=label,
                max_iter=args.max_iter,
                tol_cost_stability=1e-8,
                rtol=1e-10,
                atol=1e-11,
            )
            runtimes[label] = time.perf_counter() - started
            scenarios.append(scenario)

    contaminated = _inject_outliers(
        selected["Geometric RR"],
        station_names=station_names,
        outlier_fraction=args.outlier_fraction,
        outlier_sigma=args.outlier_amplitude_sigma,
        seed=args.seed + 20,
    )
    robust_scenarios = []
    for robust in (False, True):
        label = "Outlier standard" if not robust else "Outlier robust rejection"
        print(f"Running {label}...")
        started = time.perf_counter()
        scenario = run_batch_arc_sequence(
            contaminated,
            "range_rate",
            "cold",
            "bls_lm",
            mu_moon,
            mu_earth,
            mu_sun,
            ephemeris.earth_position,
            ephemeris.sun_position,
            cold_start_bank=cold_bank,
            label=label,
            max_iter=args.max_iter,
            tol_cost_stability=1e-8,
            rtol=1e-10,
            atol=1e-11,
            robust_outlier_rejection=robust,
        )
        runtimes[label] = time.perf_counter() - started
        robust_scenarios.append(scenario)

    out_dir = Path("python_port") / "results" / "bls_3day_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_scenarios = scenarios + robust_scenarios
    detail_csv = write_scenario_summary_csv(all_scenarios, out_dir / "bls_3day_ablation_detail.csv")
    aggregate_rows = _aggregate(all_scenarios, runtimes)
    aggregate_csv = _write_aggregate(aggregate_rows, out_dir / "bls_3day_ablation_aggregate.csv")
    arc_csv = _write_arc_manifest(selected, out_dir / "bls_3day_arc_manifest.csv")
    plot_paths = [
        _plot_measurement_and_start(scenarios, out_dir / "bls_measurement_start_ablation.png"),
        _plot_runtime_and_success(aggregate_rows, out_dir / "bls_runtime_success_ablation.png"),
        _plot_numerical_diagnostics(scenarios, out_dir / "bls_numerical_diagnostics.png"),
        _plot_robustness(robust_scenarios, out_dir / "bls_outlier_rejection_ablation.png"),
    ]

    print(f"Wrote {detail_csv}")
    print(f"Wrote {aggregate_csv}")
    print(f"Wrote {arc_csv}")
    for path in plot_paths:
        print(f"Wrote {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-days", type=float, default=3.0)
    parser.add_argument("--sample-step-s", type=float, default=600.0)
    parser.add_argument("--ephemeris-step-s", type=float, default=3600.0)
    parser.add_argument("--max-gap-s", type=float, default=1800.0)
    parser.add_argument("--min-samples", type=int, default=6)
    parser.add_argument("--min-arc-duration-min", type=float, default=50.0)
    parser.add_argument("--max-arcs", type=int, default=12)
    parser.add_argument("--max-iter", type=int, default=12)
    parser.add_argument("--count-interval-s", type=float, default=60.0)
    parser.add_argument("--outlier-fraction", type=float, default=0.04)
    parser.add_argument("--outlier-amplitude-sigma", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=260609)
    return parser.parse_args()


def _select_common_arcs(
    arc_sets: dict[str, tuple],
    *,
    max_arcs: int,
    min_duration_s: float,
) -> dict[str, tuple]:
    maps = {label: {arc.arc_id: arc for arc in arcs} for label, arcs in arc_sets.items()}
    common_ids = set.intersection(*(set(mapping) for mapping in maps.values()))
    if not common_ids:
        raise RuntimeError("No common OD-ready arcs were produced for all measurement models.")
    position_map = maps["Position"]
    eligible_ids = [
        arc_id
        for arc_id in common_ids
        if position_map[arc_id].t_pass_s[-1] - position_map[arc_id].t_pass_s[0]
        >= min_duration_s
    ]
    selected_ids = sorted(
        eligible_ids,
        key=lambda arc_id: position_map[arc_id].t_pass_s[0],
    )[:max_arcs]
    if not selected_ids:
        raise RuntimeError("No common arcs satisfied the predefined OD-readiness rule.")
    return {
        label: tuple(mapping[arc_id] for arc_id in selected_ids)
        for label, mapping in maps.items()
    }


def _write_arc_manifest(arc_sets: dict[str, tuple], path: Path) -> Path:
    position_arcs = arc_sets["Position"]
    geometric_map = {arc.arc_id: arc for arc in arc_sets["Geometric RR"]}
    two_way_map = {arc.arc_id: arc for arc in arc_sets["Two-way Doppler"]}
    fields = [
        "arc_id",
        "start_h",
        "end_h",
        "duration_min",
        "state_epochs",
        "position_observations",
        "geometric_rr_observations",
        "two_way_observations",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for arc in position_arcs:
            writer.writerow(
                {
                    "arc_id": arc.arc_id,
                    "start_h": float(arc.t_pass_s[0] / 3600.0),
                    "end_h": float(arc.t_pass_s[-1] / 3600.0),
                    "duration_min": float((arc.t_pass_s[-1] - arc.t_pass_s[0]) / 60.0),
                    "state_epochs": int(arc.t_pass_s.size),
                    "position_observations": int(arc.obs_data.shape[0]),
                    "geometric_rr_observations": int(
                        geometric_map[arc.arc_id].obs_data.shape[0]
                    ),
                    "two_way_observations": int(two_way_map[arc.arc_id].obs_data.shape[0]),
                }
            )
    return path


def _inject_outliers(
    arcs: tuple,
    *,
    station_names: list[str],
    outlier_fraction: float,
    outlier_sigma: float,
    seed: int,
) -> tuple:
    rng = np.random.default_rng(seed)
    station_map = {station.name: station for station in range_rate_stations()}
    result = []
    for arc in arcs:
        obs = np.asarray(arc.obs_data, dtype=float).copy()
        count = max(1, int(np.ceil(outlier_fraction * obs.shape[0])))
        indices = rng.choice(obs.shape[0], size=min(count, obs.shape[0]), replace=False)
        for idx in indices:
            station_idx = int(obs[idx, 5]) - 1
            station = station_map[station_names[station_idx]]
            component = int(rng.integers(0, 2))
            sign = -1.0 if rng.random() < 0.5 else 1.0
            if component == 0:
                obs[idx, 1] += sign * outlier_sigma * station.sigma_range_m
            else:
                obs[idx, 2] += sign * outlier_sigma * station.sigma_range_rate_mps
        result.append(replace(arc, obs_data=obs))
    return tuple(result)


def _aggregate(scenarios, runtimes: dict[str, float]) -> list[dict]:
    rows = []
    for scenario in scenarios:
        errors = scenario.final_position_errors_m
        iterations = np.array([arc.stats.iterations for arc in scenario.arc_results], dtype=float)
        conditions = np.array([arc.stats.condition_number for arc in scenario.arc_results], dtype=float)
        rejected = np.array([arc.stats.rejected_components for arc in scenario.arc_results], dtype=float)
        active = np.array([arc.stats.active_weight_fraction for arc in scenario.arc_results], dtype=float)
        rows.append(
            {
                "scenario": scenario.label,
                "measurement_type": scenario.measurement_type,
                "range_rate_physics": scenario.range_rate_physics,
                "start_mode": scenario.start_mode,
                "num_arcs": len(scenario.arc_results),
                "median_final_position_error_m": float(np.median(errors)),
                "p95_final_position_error_m": float(np.percentile(errors, 95)),
                "max_final_position_error_m": float(np.max(errors)),
                "median_iterations": float(np.median(iterations)),
                "median_condition_number": float(np.nanmedian(conditions)),
                "algorithmic_success_fraction": scenario.algorithmic_success_fraction,
                "operational_success_fraction": scenario.operational_success_fraction,
                "median_rejected_components": float(np.median(rejected)),
                "median_active_weight_fraction": float(np.median(active)),
                "runtime_s": runtimes[scenario.label],
            }
        )
    return rows


def _write_aggregate(rows: list[dict], path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _scenario_map(scenarios) -> dict[tuple[str, str], object]:
    result = {}
    for scenario in scenarios:
        model = scenario.label.split(" / ")[0]
        result[(model, scenario.start_mode)] = scenario
    return result


def _plot_measurement_and_start(scenarios, path: Path) -> Path:
    models = ("Position", "Geometric RR", "Two-way Doppler")
    starts = ("cold", "hot", "formal")
    colors = {"cold": "#64748b", "hot": "#2563eb", "formal": "#0f766e"}
    mapping = _scenario_map(scenarios)
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.8), sharex=True, constrained_layout=True)
    fig.suptitle("Three-Day BLS-LM Measurement and Initialization Ablation", fontweight="bold")
    for ax, start in zip(axes, starts):
        for model in models:
            scenario = mapping[(model, start)]
            ax.plot(
                [arc.arc_id for arc in scenario.arc_results],
                scenario.final_position_errors_m,
                marker="o",
                linewidth=1.5,
                label=model,
            )
        ax.set_yscale("log")
        ax.set_ylabel("final position error [m]")
        ax.set_title(f"{start.capitalize()} initialization")
        ax.grid(True, which="both", alpha=0.25)
    axes[-1].set_xlabel("visibility arc ID")
    axes[0].legend(ncol=3, fontsize=8)
    fig.savefig(path, dpi=210)
    plt.close(fig)
    return path


def _plot_runtime_and_success(rows: list[dict], path: Path) -> Path:
    rows = [row for row in rows if not row["scenario"].startswith("Outlier")]
    labels = [row["scenario"].replace(" / ", "\n") for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.0), constrained_layout=True)
    axes[0].bar(x, [row["runtime_s"] for row in rows], color="#4c78a8")
    axes[0].set_ylabel("wall-clock runtime [s]")
    axes[0].set_title("Computational cost")
    axes[0].grid(axis="y", alpha=0.25)
    width = 0.38
    axes[1].bar(
        x - width / 2,
        [row["algorithmic_success_fraction"] for row in rows],
        width,
        label="algorithmic",
        color="#f58518",
    )
    axes[1].bar(
        x + width / 2,
        [row["operational_success_fraction"] for row in rows],
        width,
        label="operational",
        color="#54a24b",
    )
    axes[1].set_ylim(0.0, 1.08)
    axes[1].set_ylabel("success fraction")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("BLS-LM Runtime and Success by Configuration", fontweight="bold")
    fig.savefig(path, dpi=210)
    plt.close(fig)
    return path


def _plot_numerical_diagnostics(scenarios, path: Path) -> Path:
    labels = [scenario.label.replace(" / ", "\n") for scenario in scenarios]
    iterations = [np.median([arc.stats.iterations for arc in scenario.arc_results]) for scenario in scenarios]
    conditions = [
        np.nanmedian([arc.stats.condition_number for arc in scenario.arc_results])
        for scenario in scenarios
    ]
    x = np.arange(len(scenarios))
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.2), constrained_layout=True)
    axes[0].bar(x, iterations, color="#b279a2")
    axes[0].set_ylabel("median iterations")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, np.maximum(conditions, 1.0), color="#e45756")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("median condition number")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].grid(axis="y", which="both", alpha=0.25)
    fig.suptitle("BLS-LM Iteration and Conditioning Diagnostics", fontweight="bold")
    fig.savefig(path, dpi=210)
    plt.close(fig)
    return path


def _plot_robustness(scenarios, path: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.8), sharex=True, constrained_layout=True)
    for scenario in scenarios:
        axes[0].plot(
            [arc.arc_id for arc in scenario.arc_results],
            scenario.final_position_errors_m,
            marker="o",
            linewidth=1.6,
            label=scenario.label,
        )
        axes[1].plot(
            [arc.arc_id for arc in scenario.arc_results],
            [1.0 - arc.stats.active_weight_fraction for arc in scenario.arc_results],
            marker="s",
            linewidth=1.4,
            label=scenario.label,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("final position error [m]")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.25)
    axes[1].set_ylabel("rejected component fraction")
    axes[1].set_xlabel("visibility arc ID")
    axes[1].grid(True, alpha=0.25)
    fig.suptitle("BLS-LM Robust Component Rejection under Injected Outliers", fontweight="bold")
    fig.savefig(path, dpi=210)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
