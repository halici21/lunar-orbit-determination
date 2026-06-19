"""Generate a longer SPICE-based visibility analysis report.

This script uses the MATLAB-exported initial-state fixture as the golden
starting point, then propagates a longer truth trajectory with Python dynamics
and SPICE Earth/Sun ephemerides.

Run from the project root:

    python python_port/examples/long_visibility_report.py
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
    load_spice_kernels,
    plot_visibility_analysis,
    propagate_truth_with_ephemeris,
    range_rate_stations,
    sample_j2000_to_itrf93_transforms,
    sample_moon_centered_ephemeris,
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

    duration_h = 12.0
    sample_step_s = 120.0
    ephem_step_s = 600.0
    max_gap_s = 20.0 * 60.0
    min_elevation_deg = 5.0

    t_eval_s = np.arange(0.0, duration_h * 3600.0 + sample_step_s, sample_step_s)
    t_ephem_s = np.arange(0.0, duration_h * 3600.0 + ephem_step_s, ephem_step_s)

    import spiceypy as spice

    load_spice_kernels()
    try:
        et0 = float(spice.str2et(epoch_utc))
        ephemeris = sample_moon_centered_ephemeris(et0, t_ephem_s)
        x_j2000_to_itrf93 = sample_j2000_to_itrf93_transforms(et0, t_eval_s)
    finally:
        spice.kclear()

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
    cases = {
        "long_single_canberra": ["Canberra DSN"],
        "long_multi_dsn_itu": ["Goldstone DSN", "Madrid DSN", "Canberra DSN", "ITU Ayazaga"],
        "long_multi_extended": [
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
    for case_name, station_names in cases.items():
        _run_case(
            case_name,
            station_names,
            t_eval_s,
            state_history,
            ephemeris.earth_position,
            x_j2000_to_itrf93,
            max_gap_s,
            config,
            stations_by_name,
            out_dir,
        )


def _run_case(
    case_name,
    station_names,
    t_eval_s,
    state_history,
    earth_position,
    x_j2000_to_itrf93,
    max_gap_s,
    config,
    stations_by_name,
    out_dir,
):
    stations = [stations_by_name[name] for name in station_names]
    seg_starts, seg_ends, vis_mask_raw, net_vis_filled = analyze_visibility_gap_with_transforms(
        t_eval_s,
        state_history,
        stations,
        earth_position,
        x_j2000_to_itrf93,
        max_gap_s,
        config,
    )

    png_path = plot_visibility_analysis(
        t_eval_s,
        station_names,
        vis_mask_raw,
        net_vis_filled,
        seg_starts,
        seg_ends,
        out_dir / f"{case_name}_visibility_analysis.png",
        title=f"{case_name.replace('_', ' ').title()} SPICE-ITRF Visibility Analysis",
    )
    csv_path = write_visibility_summary_csv(
        t_eval_s,
        station_names,
        vis_mask_raw,
        net_vis_filled,
        seg_starts,
        seg_ends,
        out_dir / f"{case_name}_visibility_summary.csv",
    )

    raw_network = np.any(vis_mask_raw, axis=1)
    raw_fraction = float(np.mean(raw_network))
    filled_fraction = float(np.mean(net_vis_filled))
    print(
        f"{case_name}: {len(seg_starts)} arcs, raw fraction={raw_fraction:.3f}, "
        f"filled fraction={filled_fraction:.3f}"
    )
    print(f"Wrote {png_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
