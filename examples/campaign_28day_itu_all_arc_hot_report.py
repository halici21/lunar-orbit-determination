"""Run hot-start clean RR OD over all 28-day ITU-only OD-ready arcs.

This complements ``campaign_28day_itu_report.py``.  The earlier report keeps
runtime modest by estimating representative arcs only; this report estimates
every ITU visibility arc with at least four visible samples using hot-start
handoff.

Run from the project root:

    python python_port/examples/campaign_28day_itu_all_arc_hot_report.py
"""

from __future__ import annotations

import csv
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
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
    run_srif_arc_sequence,
    sample_j2000_to_itrf93_transforms,
    sample_moon_centered_ephemeris,
    write_scenario_summary_csv,
)


def main() -> None:
    args = _parse_args()
    fixture_path = Path("python_port") / "fixtures" / "spice_snapshots.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    initial = fixture["initial_state"]
    constants = fixture["constants"]
    epoch_utc = fixture["epoch_utc"]

    x0_mci = np.asarray(initial["state_mci_j2000_m_mps"], dtype=float)
    mu_moon = float(initial["mu_moon_m3_s2"])
    mu_earth = float(np.asarray(constants["mu_earth_km3_s2"]).reshape(-1)[0] * 1e9)
    mu_sun = float(np.asarray(constants["mu_sun_km3_s2"]).reshape(-1)[0] * 1e9)

    duration_h = 28.0 * 24.0
    sample_step_s = 600.0
    ephem_step_s = 3600.0
    max_gap_s = 30.0 * 60.0
    min_elevation_deg = 5.0

    t_eval_s = np.arange(0.0, duration_h * 3600.0 + sample_step_s, sample_step_s)
    t_ephem_s = np.arange(0.0, duration_h * 3600.0 + ephem_step_s, ephem_step_s)

    import spiceypy as spice

    load_spice_kernels()
    try:
        et0 = float(spice.str2et(epoch_utc))
        print(f"Sampling SPICE ephemerides for {duration_h:.1f} h...")
        ephemeris = sample_moon_centered_ephemeris(et0, t_ephem_s)
        x_j2000_to_itrf93 = sample_j2000_to_itrf93_transforms(et0, t_eval_s)

        print(f"Propagating {duration_h:.1f} h truth trajectory with {t_eval_s.size} samples...")
        state_history = propagate_truth_with_ephemeris(
            t_eval_s,
            x0_mci,
            mu_moon,
            mu_earth,
            mu_sun,
            ephemeris,
            rtol=1e-10,
            atol=1e-11,
        )

        station = {item.name: item for item in range_rate_stations()}["ITU Ayazaga"]
        stations = [station]
        config = VisibilityConfig(
            r_moon_mean_m=float(initial["r_moon_mean_m"]),
            earth_rotation_rad_s=7.292115e-5,
            epoch_utc=epoch_utc,
            min_elevation_deg=min_elevation_deg,
        )
        seg_starts, seg_ends, vis_mask_raw, _ = analyze_visibility_gap_with_transforms(
            t_eval_s,
            state_history,
            stations,
            ephemeris.earth_position,
            x_j2000_to_itrf93,
            max_gap_s,
            config,
        )

        print(f"Building clean ITU-only RR measurements for {len(seg_starts)} visibility arcs...")
        arcs = build_measurement_arcs(
            "range_rate",
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
    finally:
        spice.kclear()

    print(
        f"Running clean hot-start SRIF over all {len(arcs)} ITU OD-ready arcs "
        f"with max_iter={args.max_iter}..."
    )
    cold_bank = make_cold_start_bank(len(arcs), sigma_pos_m=250.0, sigma_vel_mps=0.04, seed=240628)
    scenario = run_srif_arc_sequence(
        arcs,
        "range_rate",
        "hot",
        mu_moon,
        mu_earth,
        mu_sun,
        ephemeris.earth_position,
        ephemeris.sun_position,
        cold_start_bank=cold_bank,
        label=f"ITU 28d all-arcs hot maxiter{args.max_iter}",
        max_iter=args.max_iter,
        rtol=args.rtol,
        atol=args.atol,
    )

    out_dir = Path("python_port") / "results"
    suffix = f"maxiter{args.max_iter}"
    detail_csv = write_scenario_summary_csv([scenario], out_dir / f"campaign_28day_itu_all_arc_hot_{suffix}_rr_od_summary.csv")
    arc_csv = _write_arc_csv(t_eval_s, arcs, scenario, out_dir / f"campaign_28day_itu_all_arc_hot_{suffix}_errors.csv")
    aggregate_csv = _write_aggregate_csv(scenario, out_dir / f"campaign_28day_itu_all_arc_hot_{suffix}_aggregate.csv")
    png_path = _plot_all_arc_hot_errors(scenario, out_dir / f"campaign_28day_itu_all_arc_hot_{suffix}_errors.png")

    print(_scenario_line(scenario))
    print(f"Wrote {aggregate_csv}")
    print(f"Wrote {arc_csv}")
    print(f"Wrote {detail_csv}")
    print(f"Wrote {png_path}")


def _parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--max-iter", type=int, default=6, help="SRIF iterations per arc; use 20 for a slower full run.")
    parser.add_argument("--rtol", type=float, default=1e-9, help="Estimator propagation relative tolerance.")
    parser.add_argument("--atol", type=float, default=1e-10, help="Estimator propagation absolute tolerance.")
    return parser.parse_args()


def _write_arc_csv(t_eval_s, arcs, scenario, output_path: Path) -> Path:
    by_arc = {result.arc_id: result for result in scenario.arc_results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "arc_id",
                "start_h",
                "end_h",
                "arc_span_h",
                "visible_samples",
                "num_observations",
                "initial_position_error_m",
                "final_position_error_m",
                "final_velocity_error_mps",
                "condition_number",
                "rank",
                "iterations",
                "stop_reason",
            ]
        )
        for arc in arcs:
            result = by_arc[arc.arc_id]
            start_h = float(t_eval_s[arc.start_idx]) / 3600.0
            end_h = float(t_eval_s[arc.end_idx]) / 3600.0
            writer.writerow(
                [
                    arc.arc_id,
                    start_h,
                    end_h,
                    end_h - start_h,
                    int(arc.end_idx - arc.start_idx + 1),
                    result.num_observations,
                    result.initial_position_error_m,
                    result.final_position_error_m,
                    result.final_velocity_error_mps,
                    result.stats.condition_number,
                    result.stats.rank,
                    result.stats.iterations,
                    result.stop_reason,
                ]
            )
    return output_path


def _write_aggregate_csv(scenario, output_path: Path) -> Path:
    errors = scenario.final_position_errors_m
    initial = scenario.initial_position_errors_m
    conditions = np.array([result.stats.condition_number for result in scenario.arc_results], dtype=float)
    rows = [
        ("num_arcs", len(scenario.arc_results)),
        ("median_initial_position_error_m", _safe_percentile(initial, 50)),
        ("median_final_position_error_m", _safe_percentile(errors, 50)),
        ("p75_final_position_error_m", _safe_percentile(errors, 75)),
        ("p90_final_position_error_m", _safe_percentile(errors, 90)),
        ("p95_final_position_error_m", _safe_percentile(errors, 95)),
        ("max_final_position_error_m", float(np.max(errors)) if errors.size else float("nan")),
        ("last_final_position_error_m", float(errors[-1]) if errors.size else float("nan")),
        ("improved_fraction", float(np.mean(errors < initial)) if errors.size else float("nan")),
        ("success_fraction", scenario.success_fraction),
        ("median_condition_number", _safe_percentile(conditions, 50)),
        ("p90_condition_number", _safe_percentile(conditions, 90)),
        ("max_condition_number", float(np.max(conditions)) if conditions.size else float("nan")),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)
    return output_path


def _plot_all_arc_hot_errors(scenario, output_path: Path) -> Path:
    arc_ids = np.array([result.arc_id for result in scenario.arc_results], dtype=float)
    final_errors = scenario.final_position_errors_m
    initial_errors = scenario.initial_position_errors_m
    durations_h = np.array([(result.end_idx - result.start_idx) * 600.0 / 3600.0 for result in scenario.arc_results], dtype=float)
    conditions = np.array([result.stats.condition_number for result in scenario.arc_results], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0), constrained_layout=True)
    fig.suptitle("28-Day ITU-only all OD-ready arcs: hot-start clean geometric RR")

    ax = axes[0, 0]
    ax.semilogy(arc_ids, np.maximum(final_errors, 1e-12), marker="o", markersize=3.2, linewidth=1.2)
    ax.set_xlabel("visibility arc id")
    ax.set_ylabel("final position error [m]")
    ax.set_title("Final position error over all estimated arcs")
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[0, 1]
    bins = np.logspace(np.floor(np.log10(max(np.min(final_errors[final_errors > 0]), 1e-12))), np.ceil(np.log10(max(np.max(final_errors), 1e-10))), 28)
    ax.hist(np.maximum(final_errors, 1e-12), bins=bins, color="#2f6f9f", edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("final position error [m]")
    ax.set_ylabel("arc count")
    ax.set_title("Error distribution")

    ax = axes[1, 0]
    sc = ax.scatter(
        durations_h,
        np.maximum(final_errors, 1e-12),
        c=np.log10(np.maximum(conditions, 1.0)),
        cmap="viridis",
        s=46,
        edgecolor="black",
        linewidth=0.25,
    )
    ax.set_yscale("log")
    ax.set_xlabel("arc duration [h]")
    ax.set_ylabel("final position error [m]")
    ax.set_title("Duration vs error, color = log10(condition)")
    ax.grid(True, which="both", alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("log10 condition number")

    ax = axes[1, 1]
    ax.loglog(np.maximum(conditions, 1.0), np.maximum(final_errors, 1e-12), marker="o", linestyle="", markersize=4.2)
    ax.set_xlabel("condition number")
    ax.set_ylabel("final position error [m]")
    ax.set_title("Conditioning vs error")
    ax.grid(True, which="both", alpha=0.25)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _scenario_line(scenario) -> str:
    final = scenario.final_position_errors_m
    return (
        f"{scenario.label}: arcs={len(scenario.arc_results)}, "
        f"median final={_safe_percentile(final, 50):.3g} m, "
        f"p90 final={_safe_percentile(final, 90):.3g} m, "
        f"p95 final={_safe_percentile(final, 95):.3g} m, "
        f"max final={float(np.max(final)) if final.size else float('nan'):.3g} m"
    )


def _safe_percentile(values, percentile: float) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, percentile))


if __name__ == "__main__":
    main()
