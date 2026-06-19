"""Generate a visible cold-vs-hot SRIF demo report.

Run from the project root:

    python python_port/examples/synthetic_hot_start_report.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_od import (
    PassGeometry,
    PreparedArc,
    Station,
    compute_position_residuals_analytic,
    plot_scenario_comparison,
    propagate_augmented_state,
    run_srif_arc_sequence,
    write_scenario_summary_csv,
)


def main() -> None:
    mu_moon = 4902.800066e9
    r0norm = 1737.4e3 + 100e3
    x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
    t_all_s = np.arange(0.0, 3301.0, 60.0)

    get_earth_pos = lambda t: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
    get_sun_pos = lambda t: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))

    x_aug0 = np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")])
    x_aug_truth = propagate_augmented_state(
        t_all_s,
        x_aug0,
        mu_moon,
        0.0,
        0.0,
        get_earth_pos,
        get_sun_pos,
        rtol=1e-12,
        atol=1e-13,
    )
    x_truth = x_aug_truth[:, :6]

    stations = (
        _synthetic_station("Equator 0", 0.0, 0.0, 0.0),
        _synthetic_station("Equator 90E", 0.0, 90.0, 0.0),
        _synthetic_station("Midlat West", 45.0, -30.0, 500.0),
        _synthetic_station("South East", -35.0, 150.0, 600.0),
    )

    arcs = (
        _build_position_arc(1, 0, 12, t_all_s, x_truth, stations),
        _build_position_arc(2, 20, 32, t_all_s, x_truth, stations),
        _build_position_arc(3, 40, 52, t_all_s, x_truth, stations),
    )
    cold_bank = (
        np.array([80.0, -70.0, 40.0, 0.04, -0.025, 0.015]),
        np.array([900.0, -650.0, 380.0, 0.28, -0.19, 0.12]),
        np.array([-700.0, 520.0, -300.0, -0.22, 0.16, -0.10]),
    )

    cold = run_srif_arc_sequence(
        arcs,
        "position",
        "cold",
        mu_moon,
        0.0,
        0.0,
        get_earth_pos,
        get_sun_pos,
        cold_start_bank=cold_bank,
        label="Cold start",
        max_iter=50,
        rtol=1e-12,
        atol=1e-13,
    )
    hot = run_srif_arc_sequence(
        arcs,
        "position",
        "hot",
        mu_moon,
        0.0,
        0.0,
        get_earth_pos,
        get_sun_pos,
        cold_start_bank=cold_bank,
        label="State hot-start",
        max_iter=50,
        rtol=1e-12,
        atol=1e-13,
    )

    out_dir = Path("python_port") / "results"
    csv_path = write_scenario_summary_csv([cold, hot], out_dir / "synthetic_hot_start_summary.csv")
    fig_path = plot_scenario_comparison(
        [cold, hot],
        out_dir / "synthetic_hot_start_comparison.png",
        title="Synthetic Position-Only SRIF Cold vs State Hot-Start",
    )

    print(f"Wrote {fig_path}")
    print(f"Wrote {csv_path}")


def _synthetic_station(name: str, lat_deg: float, lon_deg: float, alt_m: float) -> Station:
    return Station(
        name=name,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        alt_m=alt_m,
        color_rgb=(0.0, 0.0, 0.0),
        sigma_range_m=1.0,
        sigma_angle_rad=1e-5,
    )


def _build_position_arc(arc_id, start_idx, end_idx, t_all_s, x_truth, stations):
    t_pass_s = np.asarray(t_all_s[start_idx : end_idx + 1], dtype=float)
    x_pass = np.asarray(x_truth[start_idx : end_idx + 1, :], dtype=float)
    pass_geo = PassGeometry(
        t_s=t_pass_s,
        earth_pos_mci_m=np.zeros((t_pass_s.size, 3)),
        earth_vel_mci_mps=np.zeros((t_pass_s.size, 3)),
        x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass_s.size, axis=0),
        stations=stations,
        measurement_type="position",
    )
    rows = []
    for time_idx, t_s in enumerate(t_pass_s, start=1):
        for station_id in range(1, len(stations) + 1):
            rows.append([t_s, 0.0, 0.0, 0.0, station_id, time_idx, arc_id])
    obs_data = np.asarray(rows, dtype=float)
    _, h_meas, _ = compute_position_residuals_analytic(x_pass, obs_data, pass_geo)
    obs_data[:, 1:4] = h_meas
    return PreparedArc(
        arc_id=arc_id,
        start_idx=start_idx,
        end_idx=end_idx,
        t_pass_s=t_pass_s,
        truth_state_history_mci=x_pass,
        obs_data=obs_data,
        pass_geo=pass_geo,
    )


if __name__ == "__main__":
    main()
