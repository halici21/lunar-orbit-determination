"""Run RR SRIF OD on long SPICE-ITRF visibility arcs.

This report uses the same long visibility/truth chain as the position-only
report, but generates `[range, range_rate, az, el]` observations.

Run from the project root:

    python python_port/examples/long_visibility_rr_od_report.py
"""

from __future__ import annotations

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
    propagate_truth_with_ephemeris,
    range_rate_stations,
    run_srif_arc_sequence,
    sample_j2000_to_itrf93_transforms,
    sample_moon_centered_ephemeris,
    write_scenario_summary_csv,
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

    duration_h = 12.0
    sample_step_s = 120.0
    ephem_step_s = 600.0
    max_gap_s = 20.0 * 60.0
    min_elevation_deg = 5.0

    t_eval_s = np.arange(0.0, duration_h * 3600.0 + sample_step_s, sample_step_s)
    t_ephem_s = np.arange(0.0, duration_h * 3600.0 + ephem_step_s, ephem_step_s)

    station_names = ["Goldstone DSN", "Madrid DSN", "Canberra DSN", "ITU Ayazaga"]
    stations_by_name = {station.name: station for station in range_rate_stations()}
    stations = [stations_by_name[name] for name in station_names]

    import spiceypy as spice

    load_spice_kernels()
    try:
        et0 = float(spice.str2et(epoch_utc))
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
        seg_starts, seg_ends, vis_mask_raw, _ = analyze_visibility_gap_with_transforms(
            t_eval_s,
            state_history,
            stations,
            ephemeris.earth_position,
            x_j2000_to_itrf93,
            max_gap_s,
            config,
        )

        print(f"Building range-rate measurements for {len(seg_starts)} SPICE-visibility arcs...")
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
            min_samples=8,
        )
    finally:
        spice.kclear()

    cold_bank = make_cold_start_bank(
        len(arcs),
        sigma_pos_m=250.0,
        sigma_vel_mps=0.04,
        seed=240605,
    )

    print(f"Running cold-start RR SRIF over {len(arcs)} arcs...")
    cold = run_srif_arc_sequence(
        arcs,
        "range_rate",
        "cold",
        mu_moon,
        mu_earth,
        mu_sun,
        ephemeris.earth_position,
        ephemeris.sun_position,
        cold_start_bank=cold_bank,
        label="Cold start",
        max_iter=20,
        rtol=1e-10,
        atol=1e-11,
    )
    print("Running state hot-start RR SRIF...")
    hot = run_srif_arc_sequence(
        arcs,
        "range_rate",
        "hot",
        mu_moon,
        mu_earth,
        mu_sun,
        ephemeris.earth_position,
        ephemeris.sun_position,
        cold_start_bank=cold_bank,
        label="State hot-start",
        max_iter=20,
        rtol=1e-10,
        atol=1e-11,
    )

    out_dir = Path("python_port") / "results"
    csv_path = write_scenario_summary_csv([cold, hot], out_dir / "long_visibility_rr_od_summary.csv")
    fig_path = plot_scenario_comparison(
        [cold, hot],
        out_dir / "long_visibility_rr_od_comparison.png",
        title="Long SPICE-Visibility RR SRIF Cold vs State Hot-Start",
    )

    print(_summary_line(cold))
    print(_summary_line(hot))
    print(f"Wrote {fig_path}")
    print(f"Wrote {csv_path}")


def _summary_line(scenario):
    initial = scenario.initial_position_errors_m
    final = scenario.final_position_errors_m
    improved_fraction = float(np.mean(final < initial)) if initial.size else 0.0
    return (
        f"{scenario.label}: arcs={len(scenario.arc_results)}, "
        f"median initial={np.median(initial):.3g} m, "
        f"median final={np.median(final):.3g} m, "
        f"improved_fraction={improved_fraction:.3f}"
    )


if __name__ == "__main__":
    main()
