"""Run a short SPICE-backed two-way counted Doppler OD campaign.

This is a final sanity check for the two-way counted Doppler path:

- SPICE Earth/Sun ephemerides
- SPICE J2000 -> ITRF93 station geometry for visibility/measurements
- multi-station visibility arcs
- two-way counted Doppler range-rate measurements
- SRIF cold and hot-start OD

Run from the project root:

    python python_port/examples/quick_two_way_spice_campaign.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_od import (  # noqa: E402
    RangeRatePhysicsConfig,
    VisibilityConfig,
    analyze_visibility_gap_with_transforms,
    build_measurement_arcs,
    load_spice_kernels,
    make_cold_start_bank,
    plot_scenario_comparison,
    propagate_truth_with_ephemeris,
    range_rate_stations,
    run_srif_arc_sequence,
    sample_j2000_to_itrf93_transforms,
    sample_moon_centered_ephemeris,
    write_scenario_summary_csv,
    write_visibility_summary_csv,
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

    duration_s = args.duration_h * 3600.0
    t_eval_s = np.arange(0.0, duration_s + args.sample_step_s, args.sample_step_s)
    t_ephem_s = np.arange(0.0, duration_s + args.ephemeris_step_s, args.ephemeris_step_s)

    import spiceypy as spice

    load_spice_kernels()
    try:
        et0 = float(spice.str2et(epoch_utc))
        print(f"Sampling SPICE ephemerides for {args.duration_h:g} h...")
        ephemeris = sample_moon_centered_ephemeris(et0, t_ephem_s)
        x_j2000_to_itrf93 = sample_j2000_to_itrf93_transforms(et0, t_eval_s)

        print(f"Propagating truth trajectory at {t_eval_s.size} samples...")
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

        stations_by_name = {station.name: station for station in range_rate_stations()}
        stations = [stations_by_name[name] for name in args.stations]
        visibility_config = VisibilityConfig(
            r_moon_mean_m=float(initial["r_moon_mean_m"]),
            earth_rotation_rad_s=7.292115e-5,
            epoch_utc=epoch_utc,
            min_elevation_deg=args.min_elevation_deg,
        )
        seg_starts, seg_ends, vis_mask_raw, net_vis_filled = analyze_visibility_gap_with_transforms(
            t_eval_s,
            state_history,
            stations,
            ephemeris.earth_position,
            x_j2000_to_itrf93,
            args.max_gap_s,
            visibility_config,
        )
        print(f"Visibility arcs: {len(seg_starts)}")

        rr_physics = RangeRatePhysicsConfig(
            mode="two_way_counted_doppler",
            count_interval_s=args.count_interval_s,
            output_unit="mps_equivalent",
        )
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
            noise=args.noise,
            rng=np.random.default_rng(args.seed),
            min_samples=args.min_samples,
            range_rate_physics=rr_physics,
        )
    finally:
        spice.kclear()

    od_arcs = _select_representative_arcs(arcs, args.max_arcs)
    if not od_arcs:
        raise RuntimeError("No OD-ready two-way arcs were produced. Try a longer duration or lower min-samples.")
    print(f"Running two-way SRIF on {len(od_arcs)} representative arcs...")

    cold_bank = make_cold_start_bank(len(od_arcs), sigma_pos_m=200.0, sigma_vel_mps=0.035, seed=args.seed)
    scenarios = [
        run_srif_arc_sequence(
            od_arcs,
            "range_rate",
            start_mode,
            mu_moon,
            mu_earth,
            mu_sun,
            ephemeris.earth_position,
            ephemeris.sun_position,
            cold_start_bank=cold_bank,
            label=f"Two-way {start_mode}",
            max_iter=args.max_iter,
            tol_cost_stability=args.tol_cost_stability,
            rtol=1e-10,
            atol=1e-11,
        )
        for start_mode in ("cold", "hot")
    ]

    out_dir = Path("python_port") / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = write_scenario_summary_csv(scenarios, out_dir / "quick_two_way_spice_od_summary.csv")
    plot_png = plot_scenario_comparison(
        scenarios,
        out_dir / "quick_two_way_spice_od_comparison.png",
        title="Quick SPICE Two-Way Counted Doppler SRIF",
    )
    visibility_csv = write_visibility_summary_csv(
        t_eval_s,
        [station.name for station in stations],
        vis_mask_raw,
        net_vis_filled,
        seg_starts,
        seg_ends,
        out_dir / "quick_two_way_spice_visibility_summary.csv",
    )
    aggregate_csv = _write_aggregate_csv(scenarios, out_dir / "quick_two_way_spice_summary.csv")

    for scenario in scenarios:
        final = scenario.final_position_errors_m
        print(
            f"{scenario.label}: arcs={len(scenario.arc_results)}, "
            f"median_final_pos={np.median(final):.3f} m, "
            f"algorithmic_success={scenario.algorithmic_success_fraction:.2f}, "
            f"operational_success={scenario.operational_success_fraction:.2f}"
        )
    print(f"Wrote {aggregate_csv}")
    print(f"Wrote {detail_csv}")
    print(f"Wrote {visibility_csv}")
    print(f"Wrote {plot_png}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-h", type=float, default=24.0)
    parser.add_argument("--sample-step-s", type=float, default=300.0)
    parser.add_argument("--ephemeris-step-s", type=float, default=1800.0)
    parser.add_argument("--count-interval-s", type=float, default=60.0)
    parser.add_argument("--max-gap-s", type=float, default=1800.0)
    parser.add_argument("--min-elevation-deg", type=float, default=5.0)
    parser.add_argument("--min-samples", type=int, default=4)
    parser.add_argument("--max-arcs", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--tol-cost-stability", type=float, default=1e-8)
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--seed", type=int, default=240608)
    parser.add_argument(
        "--stations",
        nargs="+",
        default=["Goldstone DSN", "Madrid DSN", "Canberra DSN", "ITU Ayazaga"],
    )
    return parser.parse_args()


def _select_representative_arcs(arcs, max_arcs: int):
    arcs = tuple(arcs)
    if len(arcs) <= max_arcs:
        return arcs
    order = np.argsort([-arc.obs_data.shape[0] for arc in arcs])
    selected = sorted(order[:max_arcs], key=lambda idx: arcs[int(idx)].start_idx)
    return tuple(arcs[int(idx)] for idx in selected)


def _write_aggregate_csv(scenarios, output_path: Path) -> Path:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scenario",
                "range_rate_physics",
                "count_interval_s",
                "num_arcs",
                "success_fraction",
                "algorithmic_success_fraction",
                "operational_success_fraction",
                "median_final_position_error_m",
                "max_final_position_error_m",
            ]
        )
        for scenario in scenarios:
            final = scenario.final_position_errors_m
            writer.writerow(
                [
                    scenario.label,
                    scenario.range_rate_physics,
                    scenario.count_interval_s,
                    len(scenario.arc_results),
                    scenario.success_fraction,
                    scenario.algorithmic_success_fraction,
                    scenario.operational_success_fraction,
                    float(np.median(final)) if final.size else float("nan"),
                    float(np.max(final)) if final.size else float("nan"),
                ]
            )
    return output_path


if __name__ == "__main__":
    main()
