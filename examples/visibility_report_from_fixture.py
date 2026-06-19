"""Generate visibility analysis figures from the MATLAB SPICE fixture.

Run from the project root:

    python python_port/examples/visibility_report_from_fixture.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_od import (  # noqa: E402
    VisibilityConfig,
    analyze_visibility_gap,
    plot_visibility_analysis,
    range_rate_stations,
    write_visibility_summary_csv,
)


def main() -> None:
    fixture_path = Path("python_port") / "fixtures" / "spice_snapshots.json"
    if not fixture_path.is_file():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    initial = fixture["initial_state"]
    truth = fixture["truth_propagation"]
    visibility = fixture["visibility"]

    t_eval_s = np.asarray(visibility["t_eval_s"], dtype=float)
    state_history = np.asarray(truth["state_history_mci_m_mps"], dtype=float)
    t_ephem_s = np.asarray(truth["t_ephem_s"], dtype=float)
    earth_pos_grid_m = np.asarray(truth["earth_pos_grid_m"], dtype=float)
    earth_interp = PchipInterpolator(t_ephem_s, earth_pos_grid_m, axis=0)

    config = VisibilityConfig(
        r_moon_mean_m=float(initial["r_moon_mean_m"]),
        earth_rotation_rad_s=7.292115e-5,
        epoch_utc=fixture["epoch_utc"],
        min_elevation_deg=float(visibility["min_elevation_deg"]),
    )
    stations_by_name = {station.name: station for station in range_rate_stations()}
    out_dir = Path("python_port") / "results"

    _run_case(
        "single",
        visibility["single_station_names"],
        t_eval_s,
        state_history,
        earth_interp,
        float(visibility["max_gap_s"]),
        config,
        stations_by_name,
        out_dir,
    )
    _run_case(
        "multi",
        visibility["multi_station_names"],
        t_eval_s,
        state_history,
        earth_interp,
        float(visibility["max_gap_s"]),
        config,
        stations_by_name,
        out_dir,
    )


def _run_case(case_name, station_names, t_eval_s, state_history, earth_interp, max_gap_s, config, stations_by_name, out_dir):
    stations = [stations_by_name[name] for name in station_names]
    seg_starts, seg_ends, vis_mask_raw, net_vis_filled = analyze_visibility_gap(
        t_eval_s,
        state_history,
        stations,
        earth_interp,
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
        out_dir / f"visibility_{case_name}_analysis.png",
        title=f"{case_name.title()}-Station Visibility Analysis",
    )
    csv_path = write_visibility_summary_csv(
        t_eval_s,
        station_names,
        vis_mask_raw,
        net_vis_filled,
        seg_starts,
        seg_ends,
        out_dir / f"visibility_{case_name}_summary.csv",
    )
    print(f"Wrote {png_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
