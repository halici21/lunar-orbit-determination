"""Run a compact thesis factorial OD comparison.

The matrix is:

- estimator: baseline BLS/LM vs SRIF/QR
- start: cold vs state hot-start
- measurement: position-only vs range-rate
- network: single station vs multi station

This is intentionally shorter than the long 12 h reports so the full
factorial matrix can be regenerated during development.

Run from the project root:

    python python_port/examples/thesis_factorial_report.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_od import (  # noqa: E402
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
    thesis_network_by_name,
    thesis_seed_for,
    write_scenario_summary_csv,
)
from lunar_od.thesis_matrix import (  # noqa: E402
    THESIS_ATOL,
    THESIS_COLD_START_SIGMA_POS_M,
    THESIS_COLD_START_SIGMA_VEL_MPS,
    THESIS_DURATION_H,
    THESIS_EPHEMERIS_STEP_S,
    THESIS_FACTORIAL_CASES,
    THESIS_MAX_GAP_S,
    THESIS_MAX_ITER,
    THESIS_MIN_ELEVATION_DEG,
    THESIS_NETWORKS,
    THESIS_RTOL,
    THESIS_SAMPLE_STEP_S,
)


def main() -> None:
    fixture_path = Path("python_port") / "fixtures" / "spice_snapshots.json"
    if not fixture_path.is_file():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    initial = fixture["initial_state"]
    constants = fixture["constants"]
    epoch_utc = fixture["epoch_utc"]

    x0_mci = np.asarray(initial["state_mci_j2000_m_mps"], dtype=float)
    mu_moon = float(initial["mu_moon_m3_s2"])
    mu_earth = float(np.asarray(constants["mu_earth_km3_s2"]).reshape(-1)[0] * 1e9)
    mu_sun = float(np.asarray(constants["mu_sun_km3_s2"]).reshape(-1)[0] * 1e9)

    t_eval_s = np.arange(0.0, THESIS_DURATION_H * 3600.0 + THESIS_SAMPLE_STEP_S, THESIS_SAMPLE_STEP_S)
    t_ephem_s = np.arange(0.0, THESIS_DURATION_H * 3600.0 + THESIS_EPHEMERIS_STEP_S, THESIS_EPHEMERIS_STEP_S)

    all_stations_by_name = {station.name: station for station in range_rate_stations()}

    import spiceypy as spice

    load_spice_kernels()
    try:
        et0 = float(spice.str2et(epoch_utc))
        ephemeris = sample_moon_centered_ephemeris(et0, t_ephem_s)
        x_j2000_to_itrf93 = sample_j2000_to_itrf93_transforms(et0, t_eval_s)

        print(f"Propagating {THESIS_DURATION_H:.1f} h compact truth trajectory with {t_eval_s.size} samples...")
        state_history = propagate_truth_with_ephemeris(
            t_eval_s,
            x0_mci,
            mu_moon,
            mu_earth,
            mu_sun,
            ephemeris,
            rtol=THESIS_RTOL,
            atol=THESIS_ATOL,
        )

        config = VisibilityConfig(
            r_moon_mean_m=float(initial["r_moon_mean_m"]),
            earth_rotation_rad_s=7.292115e-5,
            epoch_utc=epoch_utc,
            min_elevation_deg=THESIS_MIN_ELEVATION_DEG,
        )

        scenarios = []
        for network in THESIS_NETWORKS:
            stations = [all_stations_by_name[name] for name in network.station_names]
            seg_starts, seg_ends, vis_mask_raw, _ = analyze_visibility_gap_with_transforms(
                t_eval_s,
                state_history,
                stations,
                ephemeris.earth_position,
                x_j2000_to_itrf93,
                THESIS_MAX_GAP_S,
                config,
            )
            print(f"{network.name}: {len(seg_starts)} SPICE visibility arcs")

            for measurement_type in sorted({case.measurement_type for case in THESIS_FACTORIAL_CASES if case.network == network.name}):
                arcs = build_measurement_arcs(
                    measurement_type,
                    t_eval_s,
                    state_history,
                    seg_starts,
                    seg_ends,
                    vis_mask_raw,
                    stations,
                    ephemeris.earth_position,
                    ephemeris.earth_velocity,
                    et0,
                    noise=False,
                    min_samples=4,
                )
                if not arcs:
                    print(f"  skipping {network.name}/{measurement_type}: no measurement arcs")
                    continue

                cold_bank = make_cold_start_bank(
                    len(arcs),
                    sigma_pos_m=THESIS_COLD_START_SIGMA_POS_M,
                    sigma_vel_mps=THESIS_COLD_START_SIGMA_VEL_MPS,
                    seed=thesis_seed_for(network.name, measurement_type),
                )
                for case in _cases_for(network.name, measurement_type):
                    label = case.label
                    print(f"  running {label} over {len(arcs)} arcs")
                    scenario = run_batch_arc_sequence(
                        arcs,
                        case.measurement_type,
                        case.start_mode,
                        case.estimator_type,
                        mu_moon,
                        mu_earth,
                        mu_sun,
                        ephemeris.earth_position,
                        ephemeris.sun_position,
                        cold_start_bank=cold_bank,
                        label=label,
                        max_iter=THESIS_MAX_ITER,
                        rtol=THESIS_RTOL,
                        atol=THESIS_ATOL,
                    )
                    scenarios.append(scenario)
    finally:
        spice.kclear()

    out_dir = Path("python_port") / "results"
    detail_csv = write_scenario_summary_csv(scenarios, out_dir / "thesis_factorial_detail.csv")
    aggregate_rows = _aggregate_scenarios(scenarios)
    aggregate_csv = _write_aggregate_csv(aggregate_rows, out_dir / "thesis_factorial_aggregate.csv")
    fig_path = _plot_factorial_aggregate(aggregate_rows, out_dir / "thesis_factorial_summary.png")

    print(f"Wrote {fig_path}")
    print(f"Wrote {aggregate_csv}")
    print(f"Wrote {detail_csv}")


def _cases_for(network_name: str, measurement_type: str):
    thesis_network_by_name(network_name)
    return tuple(
        case
        for case in THESIS_FACTORIAL_CASES
        if case.network == network_name and case.measurement_type == measurement_type
    )


def _aggregate_scenarios(scenarios) -> list[dict]:
    rows = []
    for scenario in scenarios:
        initial = scenario.initial_position_errors_m
        final = scenario.final_position_errors_m
        costs = np.array([result.stats.final_cost for result in scenario.arc_results], dtype=float)
        conds = np.array([result.stats.condition_number for result in scenario.arc_results], dtype=float)
        iterations = np.array([result.stats.iterations for result in scenario.arc_results], dtype=float)
        label_parts = scenario.label.split()
        network = label_parts[0]
        rows.append(
            {
                "network": network,
                "measurement_type": scenario.measurement_type,
                "estimator_type": scenario.estimator_type,
                "start_mode": scenario.start_mode,
                "num_arcs": len(scenario.arc_results),
                "median_initial_position_error_m": _safe_median(initial),
                "median_final_position_error_m": _safe_median(final),
                "max_final_position_error_m": float(np.max(final)) if final.size else float("nan"),
                "improved_fraction": float(np.mean(final < initial)) if initial.size else 0.0,
                "median_iterations": _safe_median(iterations),
                "median_final_cost": _safe_median(costs),
                "median_condition_number": _safe_median(conds),
            }
        )
    return rows


def _write_aggregate_csv(rows: list[dict], output_path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "network",
        "measurement_type",
        "estimator_type",
        "start_mode",
        "num_arcs",
        "median_initial_position_error_m",
        "median_final_position_error_m",
        "max_final_position_error_m",
        "improved_fraction",
        "median_iterations",
        "median_final_cost",
        "median_condition_number",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _plot_factorial_aggregate(rows: list[dict], output_path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    panels = [
        ("single", "position"),
        ("single", "range_rate"),
        ("multi", "position"),
        ("multi", "range_rate"),
    ]
    case_order = [
        ("bls_lm", "cold", "BLS cold"),
        ("bls_lm", "hot", "BLS hot"),
        ("srif", "cold", "SRIF cold"),
        ("srif", "hot", "SRIF hot"),
        ("ukf", "cold", "UKF cold"),
        ("ukf", "hot", "UKF hot"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=True)
    fig.suptitle("Compact Thesis Factorial Matrix", fontsize=13, fontweight="bold")
    for ax, (network, measurement_type) in zip(axes.flat, panels):
        panel_rows = {
            (row["estimator_type"], row["start_mode"]): row
            for row in rows
            if row["network"] == network and row["measurement_type"] == measurement_type
        }
        labels = [case_label for _, _, case_label in case_order]
        values = [
            panel_rows.get((estimator, start), {}).get("median_final_position_error_m", np.nan)
            for estimator, start, _ in case_order
        ]
        colors = ["#64748b", "#2563eb", "#0f766e", "#16a34a", "#9333ea", "#a855f7"]
        ax.bar(np.arange(len(labels)), np.maximum(values, 1e-12), color=colors)
        ax.set_title(f"{network} / {measurement_type}")
        ax.set_yscale("log")
        ax.set_ylabel("median final pos error [m]")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(True, axis="y", which="both", alpha=0.25)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _safe_median(values) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


if __name__ == "__main__":
    main()
