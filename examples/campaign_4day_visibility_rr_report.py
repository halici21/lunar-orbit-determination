"""Run a 4-day visibility and clean RR OD campaign sanity report.

This is intentionally a medium-fidelity campaign check: it uses the existing
geometric range-rate observable, SPICE Earth/Sun ephemerides, SPICE
J2000->ITRF93 station geometry, and SRIF arc handoff modes.

Run from the project root:

    python python_port/examples/campaign_4day_visibility_rr_report.py
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

    duration_h = 96.0
    sample_step_s = 600.0
    ephem_step_s = 1800.0
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

        config = VisibilityConfig(
            r_moon_mean_m=float(initial["r_moon_mean_m"]),
            earth_rotation_rad_s=7.292115e-5,
            epoch_utc=epoch_utc,
            min_elevation_deg=min_elevation_deg,
        )

        stations_by_name = {station.name: station for station in range_rate_stations()}
        visibility_cases = {
            "campaign_4day_single_canberra": ["Canberra DSN"],
            "campaign_4day_multi_dsn_itu": ["Goldstone DSN", "Madrid DSN", "Canberra DSN", "ITU Ayazaga"],
            "campaign_4day_multi_extended": [
                "Goldstone DSN",
                "Madrid DSN",
                "Canberra DSN",
                "Daejeon KGS",
                "Dongara KGS",
                "Malargue ESA",
                "New Norcia ESA",
                "ITU Ayazaga",
            ],
        }

        out_dir = Path("python_port") / "results"
        visibility_rows = []
        multi_visibility = None
        multi_stations = None
        for case_name, station_names in visibility_cases.items():
            stations = [stations_by_name[name] for name in station_names]
            seg_starts, seg_ends, vis_mask_raw, net_vis_filled = analyze_visibility_gap_with_transforms(
                t_eval_s,
                state_history,
                stations,
                ephemeris.earth_position,
                x_j2000_to_itrf93,
                max_gap_s,
                config,
            )
            row = _write_visibility_case(
                case_name,
                station_names,
                t_eval_s,
                vis_mask_raw,
                net_vis_filled,
                seg_starts,
                seg_ends,
                out_dir,
            )
            visibility_rows.append(row)
            print(_visibility_line(row))

            if case_name == "campaign_4day_multi_dsn_itu":
                multi_visibility = (seg_starts, seg_ends, vis_mask_raw)
                multi_stations = stations

        if multi_visibility is None or multi_stations is None:
            raise RuntimeError("Multi-station campaign visibility case was not built.")

        seg_starts, seg_ends, vis_mask_raw = multi_visibility
        print(f"Building clean RR measurements for {len(seg_starts)} campaign arcs...")
        arcs = build_measurement_arcs(
            "range_rate",
            t_eval_s,
            state_history,
            seg_starts,
            seg_ends,
            vis_mask_raw,
            multi_stations,
            ephemeris.earth_position,
            ephemeris.earth_velocity,
            et0,
            noise=False,
            min_samples=4,
        )
    finally:
        spice.kclear()

    max_od_arcs = 12
    od_arcs = _select_representative_arcs(arcs, max_od_arcs)
    print(f"Running clean RR SRIF campaign over {len(od_arcs)} representative arcs from {len(arcs)} total arcs...")
    cold_bank = make_cold_start_bank(len(od_arcs), sigma_pos_m=250.0, sigma_vel_mps=0.04, seed=240604)
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
            label="Campaign cold",
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
            label="Campaign hot",
            max_iter=20,
            rtol=1e-10,
            atol=1e-11,
        ),
        run_srif_arc_sequence(
            od_arcs,
            "range_rate",
            "sqrt_formal",
            mu_moon,
            mu_earth,
            mu_sun,
            ephemeris.earth_position,
            ephemeris.sun_position,
            cold_start_bank=cold_bank,
            label="Campaign sqrt_formal",
            max_iter=20,
            rtol=1e-10,
            atol=1e-11,
        ),
    ]

    out_dir = Path("python_port") / "results"
    scenario_csv = write_scenario_summary_csv(scenarios, out_dir / "campaign_4day_rr_od_summary.csv")
    scenario_png = plot_scenario_comparison(
        scenarios,
        out_dir / "campaign_4day_rr_od_comparison.png",
        title="4-Day Campaign Clean RR SRIF",
    )
    aggregate_csv = _write_campaign_aggregate(scenarios, visibility_rows, out_dir / "campaign_4day_summary.csv")

    print("OD summary:")
    for scenario in scenarios:
        print(_scenario_line(scenario))
    print(f"Wrote {aggregate_csv}")
    print(f"Wrote {scenario_csv}")
    print(f"Wrote {scenario_png}")


def _write_visibility_case(
    case_name,
    station_names,
    t_eval_s,
    vis_mask_raw,
    net_vis_filled,
    seg_starts,
    seg_ends,
    out_dir,
) -> dict:
    plot_visibility_analysis(
        t_eval_s,
        station_names,
        vis_mask_raw,
        net_vis_filled,
        seg_starts,
        seg_ends,
        out_dir / f"{case_name}_visibility_analysis.png",
        title=f"{case_name.replace('_', ' ').title()} Visibility",
    )
    write_visibility_summary_csv(
        t_eval_s,
        station_names,
        vis_mask_raw,
        net_vis_filled,
        seg_starts,
        seg_ends,
        out_dir / f"{case_name}_visibility_summary.csv",
    )
    raw_network = np.any(vis_mask_raw, axis=1)
    durations_h = np.array(
        [
            (float(t_eval_s[end_idx]) - float(t_eval_s[start_idx])) / 3600.0
            for start_idx, end_idx in zip(seg_starts, seg_ends)
        ],
        dtype=float,
    )
    return {
        "case": case_name,
        "num_stations": len(station_names),
        "num_arcs": int(len(seg_starts)),
        "raw_visible_fraction": float(np.mean(raw_network)),
        "filled_visible_fraction": float(np.mean(net_vis_filled)),
        "raw_visible_hours": float(np.sum(raw_network) * _sample_step_h(t_eval_s)),
        "filled_visible_hours": float(np.sum(net_vis_filled) * _sample_step_h(t_eval_s)),
        "median_arc_duration_h": _safe_median(durations_h),
        "max_arc_duration_h": float(np.max(durations_h)) if durations_h.size else float("nan"),
    }


def _select_representative_arcs(arcs, max_arcs: int):
    arcs = tuple(arcs)
    if len(arcs) <= max_arcs:
        return arcs
    indices = np.unique(np.linspace(0, len(arcs) - 1, max_arcs, dtype=int))
    return tuple(arcs[int(index)] for index in indices)


def _write_campaign_aggregate(scenarios, visibility_rows, output_path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "name", "metric", "value"])
        for row in visibility_rows:
            for key, value in row.items():
                if key != "case":
                    writer.writerow(["visibility", row["case"], key, value])
        for scenario in scenarios:
            final = scenario.final_position_errors_m
            initial = scenario.initial_position_errors_m
            improved_fraction = float(np.mean(final < initial)) if final.size else 0.0
            writer.writerow(["od", scenario.label, "num_arcs", len(scenario.arc_results)])
            writer.writerow(["od", scenario.label, "median_initial_position_error_m", _safe_median(initial)])
            writer.writerow(["od", scenario.label, "median_final_position_error_m", _safe_median(final)])
            writer.writerow(["od", scenario.label, "max_final_position_error_m", float(np.max(final)) if final.size else float("nan")])
            writer.writerow(["od", scenario.label, "final_arc_position_error_m", float(final[-1]) if final.size else float("nan")])
            writer.writerow(["od", scenario.label, "improved_fraction", improved_fraction])
            writer.writerow(["od", scenario.label, "success_fraction", scenario.success_fraction])
    return output_path


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
    return (
        f"{row['case']}: arcs={row['num_arcs']}, "
        f"raw={row['raw_visible_fraction']:.3f} ({row['raw_visible_hours']:.2f} h), "
        f"filled={row['filled_visible_fraction']:.3f} ({row['filled_visible_hours']:.2f} h), "
        f"median_arc={row['median_arc_duration_h']:.2f} h"
    )


def _scenario_line(scenario) -> str:
    final = scenario.final_position_errors_m
    initial = scenario.initial_position_errors_m
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
