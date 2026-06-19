"""Run a 28-day ITU-only visibility and clean RR OD sanity report.

This report answers two campaign-level questions for the ITU Ayazaga station:

- How much 28-day visibility is available, and how long are the arcs?
- On representative ITU-only arcs, how do clean geometric RR SRIF errors and
  conditioning behave for cold and hot starts?

Run from the project root:

    python python_port/examples/campaign_28day_itu_report.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_od import (  # noqa: E402
    VisibilityConfig,
    analyze_visibility_gap_with_transforms,
    build_measurement_arcs,
    load_spice_kernels,
    make_cold_start_bank,
    plot_scenario_comparison,
    plot_visibility_analysis,
    propagate_truth_with_ephemeris,
    range_rate_stations,
    run_srif_arc_sequence,
    sample_j2000_to_itrf93_transforms,
    sample_moon_centered_ephemeris,
    write_scenario_summary_csv,
    write_visibility_summary_csv,
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
        seg_starts, seg_ends, vis_mask_raw, net_vis_filled = analyze_visibility_gap_with_transforms(
            t_eval_s,
            state_history,
            stations,
            ephemeris.earth_position,
            x_j2000_to_itrf93,
            max_gap_s,
            config,
        )
        visibility_row = _write_visibility_outputs(
            t_eval_s,
            vis_mask_raw,
            net_vis_filled,
            seg_starts,
            seg_ends,
            Path("python_port") / "results",
        )
        print(_visibility_line(visibility_row))

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

    max_od_arcs = 16
    od_arcs = _select_representative_arcs(arcs, max_od_arcs)
    print(f"Running clean ITU-only RR SRIF over {len(od_arcs)} representative arcs from {len(arcs)} total arcs...")
    cold_bank = make_cold_start_bank(len(od_arcs), sigma_pos_m=250.0, sigma_vel_mps=0.04, seed=240628)
    scenarios = [
        run_srif_arc_sequence(
            od_arcs,
            "range_rate",
            "cold",
            mu_moon,
            mu_earth,
            mu_sun,
            ephemeris.earth_position,
            ephemeris.sun_position,
            cold_start_bank=cold_bank,
            label="ITU 28d cold",
            max_iter=20,
            rtol=1e-10,
            atol=1e-11,
        ),
        run_srif_arc_sequence(
            od_arcs,
            "range_rate",
            "hot",
            mu_moon,
            mu_earth,
            mu_sun,
            ephemeris.earth_position,
            ephemeris.sun_position,
            cold_start_bank=cold_bank,
            label="ITU 28d hot",
            max_iter=20,
            rtol=1e-10,
            atol=1e-11,
        ),
    ]

    out_dir = Path("python_port") / "results"
    detail_csv = write_scenario_summary_csv(scenarios, out_dir / "campaign_28day_itu_rr_od_summary.csv")
    comparison_png = plot_scenario_comparison(
        scenarios,
        out_dir / "campaign_28day_itu_rr_od_comparison.png",
        title="28-Day ITU-Only Clean RR SRIF",
    )
    aggregate_csv = _write_aggregate_csv(visibility_row, scenarios, out_dir / "campaign_28day_itu_summary.csv")
    arc_csv = _write_selected_arc_csv(t_eval_s, od_arcs, scenarios, out_dir / "campaign_28day_itu_selected_arcs.csv")

    print("OD summary:")
    for scenario in scenarios:
        print(_scenario_line(scenario))
    print(f"Wrote {aggregate_csv}")
    print(f"Wrote {arc_csv}")
    print(f"Wrote {detail_csv}")
    print(f"Wrote {comparison_png}")


def _write_visibility_outputs(t_eval_s, vis_mask_raw, net_vis_filled, seg_starts, seg_ends, out_dir) -> dict:
    station_names = ["ITU Ayazaga"]
    plot_visibility_analysis(
        t_eval_s,
        station_names,
        vis_mask_raw,
        net_vis_filled,
        seg_starts,
        seg_ends,
        out_dir / "campaign_28day_itu_visibility_analysis.png",
        title="28-Day ITU-Only Visibility",
    )
    write_visibility_summary_csv(
        t_eval_s,
        station_names,
        vis_mask_raw,
        net_vis_filled,
        seg_starts,
        seg_ends,
        out_dir / "campaign_28day_itu_visibility_summary.csv",
    )
    raw_network = np.any(vis_mask_raw, axis=1)
    step_h = _sample_step_h(t_eval_s)
    durations_h = np.array(
        [(float(t_eval_s[end]) - float(t_eval_s[start])) / 3600.0 for start, end in zip(seg_starts, seg_ends)],
        dtype=float,
    )
    visible_samples = np.array([int(end - start + 1) for start, end in zip(seg_starts, seg_ends)], dtype=int)
    return {
        "num_arcs": int(len(seg_starts)),
        "raw_visible_fraction": float(np.mean(raw_network)),
        "visible_sample_hours": float(np.sum(raw_network) * step_h),
        "arc_span_total_h": float(np.sum(durations_h)),
        "median_arc_span_h": _safe_median(durations_h),
        "mean_arc_span_h": float(np.mean(durations_h)) if durations_h.size else float("nan"),
        "min_arc_span_h": float(np.min(durations_h)) if durations_h.size else float("nan"),
        "max_arc_span_h": float(np.max(durations_h)) if durations_h.size else float("nan"),
        "median_visible_samples": _safe_median(visible_samples),
        "duration_histogram": dict(Counter(np.round(durations_h, 6))),
    }


def _write_aggregate_csv(visibility_row: dict, scenarios, output_path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "name", "metric", "value"])
        for key, value in visibility_row.items():
            if key == "duration_histogram":
                for duration_h, count in sorted(value.items()):
                    writer.writerow(["visibility_duration_histogram", "ITU Ayazaga", f"{float(duration_h):.6f}_h", count])
            else:
                writer.writerow(["visibility", "ITU Ayazaga", key, value])
        for scenario in scenarios:
            initial = scenario.initial_position_errors_m
            final = scenario.final_position_errors_m
            improved_fraction = float(np.mean(final < initial)) if final.size else 0.0
            writer.writerow(["od", scenario.label, "num_arcs", len(scenario.arc_results)])
            writer.writerow(["od", scenario.label, "median_initial_position_error_m", _safe_median(initial)])
            writer.writerow(["od", scenario.label, "median_final_position_error_m", _safe_median(final)])
            writer.writerow(["od", scenario.label, "max_final_position_error_m", float(np.max(final)) if final.size else float("nan")])
            writer.writerow(["od", scenario.label, "last_final_position_error_m", float(final[-1]) if final.size else float("nan")])
            writer.writerow(["od", scenario.label, "improved_fraction", improved_fraction])
            writer.writerow(["od", scenario.label, "success_fraction", scenario.success_fraction])
    return output_path


def _write_selected_arc_csv(t_eval_s, od_arcs, scenarios, output_path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_label = {scenario.label: {result.arc_id: result for result in scenario.arc_results} for scenario in scenarios}
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "arc_id",
                "start_h",
                "end_h",
                "arc_span_h",
                "visible_samples",
                "scenario",
                "num_observations",
                "initial_position_error_m",
                "final_position_error_m",
                "condition_number",
                "rank",
                "iterations",
                "stop_reason",
            ]
        )
        for arc in od_arcs:
            start_h = float(t_eval_s[arc.start_idx]) / 3600.0
            end_h = float(t_eval_s[arc.end_idx]) / 3600.0
            span_h = end_h - start_h
            visible_samples = int(arc.end_idx - arc.start_idx + 1)
            for label, result_by_arc in by_label.items():
                result = result_by_arc[arc.arc_id]
                writer.writerow(
                    [
                        arc.arc_id,
                        start_h,
                        end_h,
                        span_h,
                        visible_samples,
                        label,
                        result.num_observations,
                        result.initial_position_error_m,
                        result.final_position_error_m,
                        result.stats.condition_number,
                        result.stats.rank,
                        result.stats.iterations,
                        result.stop_reason,
                    ]
                )
    return output_path


def _select_representative_arcs(arcs, max_arcs: int):
    arcs = tuple(arcs)
    if len(arcs) <= max_arcs:
        return arcs
    indices = np.unique(np.linspace(0, len(arcs) - 1, max_arcs, dtype=int))
    return tuple(arcs[int(index)] for index in indices)


def _sample_step_h(t_eval_s) -> float:
    t_eval_s = np.asarray(t_eval_s, dtype=float)
    if t_eval_s.size < 2:
        return 0.0
    return float(np.median(np.diff(t_eval_s)) / 3600.0)


def _safe_median(values) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def _visibility_line(row: dict) -> str:
    duration_bits = ", ".join(
        f"{float(duration_h):.2f}h:{count}" for duration_h, count in sorted(row["duration_histogram"].items())
    )
    return (
        f"ITU 28d visibility: arcs={row['num_arcs']}, "
        f"visible={row['visible_sample_hours']:.2f} h, "
        f"fraction={row['raw_visible_fraction']:.3f}, "
        f"median_arc={row['median_arc_span_h']:.2f} h, "
        f"hist=[{duration_bits}]"
    )


def _scenario_line(scenario) -> str:
    initial = scenario.initial_position_errors_m
    final = scenario.final_position_errors_m
    improved_fraction = float(np.mean(final < initial)) if final.size else 0.0
    return (
        f"{scenario.label}: arcs={len(scenario.arc_results)}, "
        f"median initial={_safe_median(initial):.3g} m, "
        f"median final={_safe_median(final):.3g} m, "
        f"max final={float(np.max(final)) if final.size else float('nan'):.3g} m, "
        f"last final={float(final[-1]) if final.size else float('nan'):.3g} m, "
        f"improved={improved_fraction:.3f}, success={scenario.success_fraction:.3f}"
    )


if __name__ == "__main__":
    main()
