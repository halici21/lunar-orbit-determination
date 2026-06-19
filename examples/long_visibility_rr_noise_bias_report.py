"""Run noisy/biased RR SRIF OD on long SPICE-ITRF visibility arcs.

This report extends ``long_visibility_rr_od_report.py`` by injecting
station-specific measurement biases and Gaussian measurement noise. It then
compares a state-only estimator against a station-full bias solve-for.

Run from the project root:

    python python_port/examples/long_visibility_rr_noise_bias_report.py
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import replace
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
    base_stations = [stations_by_name[name] for name in station_names]
    true_bias_blocks = _true_rr_station_biases(station_names)
    biased_stations = [
        replace(station, bias=tuple(true_bias_blocks[idx, :]))
        for idx, station in enumerate(base_stations)
    ]

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
            biased_stations,
            ephemeris.earth_position,
            x_j2000_to_itrf93,
            max_gap_s,
            config,
        )

        print(f"Building noisy/biased RR measurements for {len(seg_starts)} SPICE-visibility arcs...")
        arcs = build_measurement_arcs(
            "range_rate",
            t_eval_s,
            state_history,
            seg_starts,
            seg_ends,
            vis_mask_raw,
            biased_stations,
            ephemeris.earth_position,
            ephemeris.earth_velocity,
            et0,
            noise=True,
            rng=np.random.default_rng(240606),
            min_samples=8,
        )
    finally:
        spice.kclear()

    cold_bank = make_cold_start_bank(
        len(arcs),
        sigma_pos_m=250.0,
        sigma_vel_mps=0.04,
        seed=240607,
    )
    zero_bias0 = np.zeros(4 * len(biased_stations), dtype=float)

    common_kwargs = dict(
        mu_moon_m3_s2=mu_moon,
        mu_earth_m3_s2=mu_earth,
        mu_sun_m3_s2=mu_sun,
        get_earth_pos=ephemeris.earth_position,
        get_sun_pos=ephemeris.sun_position,
        cold_start_bank=cold_bank,
        max_iter=16,
        rtol=1e-10,
        atol=1e-11,
    )

    print(f"Running noisy/biased RR SRIF over {len(arcs)} arcs...")
    scenarios = [
        run_srif_arc_sequence(
            arcs,
            "range_rate",
            "cold",
            **common_kwargs,
            label="Cold no-bias model",
        ),
        run_srif_arc_sequence(
            arcs,
            "range_rate",
            "hot",
            **common_kwargs,
            label="Hot no-bias model",
        ),
        run_srif_arc_sequence(
            arcs,
            "range_rate",
            "cold",
            **common_kwargs,
            bias_mode="station_full",
            initial_bias=zero_bias0,
            label="Cold station-bias solve",
        ),
        run_srif_arc_sequence(
            arcs,
            "range_rate",
            "hot",
            **common_kwargs,
            bias_mode="station_full",
            initial_bias=zero_bias0,
            label="Hot station-bias solve",
        ),
    ]

    out_dir = Path("python_port") / "results"
    csv_path = write_scenario_summary_csv(
        scenarios,
        out_dir / "long_visibility_rr_noise_bias_summary.csv",
    )
    fig_path = plot_scenario_comparison(
        scenarios,
        out_dir / "long_visibility_rr_noise_bias_comparison.png",
        title="Noisy/Biased Long SPICE-Visibility RR SRIF",
    )
    bias_csv_path = _write_rr_bias_recovery_csv(
        scenarios[2:],
        true_bias_blocks,
        station_names,
        out_dir / "long_visibility_rr_noise_bias_recovery.csv",
    )
    bias_fig_path = _plot_rr_bias_recovery(
        scenarios[2:],
        true_bias_blocks,
        station_names,
        out_dir / "long_visibility_rr_noise_bias_recovery.png",
    )

    print("Injected station biases:")
    for station_name, block in zip(station_names, true_bias_blocks):
        print(
            f"  {station_name}: range={block[0]:+.3f} m, "
            f"rr={block[1]:+.3e} m/s, "
            f"az={np.rad2deg(block[2]):+.6f} deg, "
            f"el={np.rad2deg(block[3]):+.6f} deg"
        )
    for scenario in scenarios:
        print(_summary_line(scenario))

    print(f"Wrote {fig_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {bias_fig_path}")
    print(f"Wrote {bias_csv_path}")


def _true_rr_station_biases(station_names: list[str]) -> np.ndarray:
    bias_by_name = {
        "Goldstone DSN": [8.0, 3.0e-5, np.deg2rad(0.0006), np.deg2rad(-0.0004)],
        "Madrid DSN": [-6.0, -2.5e-5, np.deg2rad(-0.0005), np.deg2rad(0.0003)],
        "Canberra DSN": [10.0, 1.5e-5, np.deg2rad(0.0004), np.deg2rad(0.0005)],
        "ITU Ayazaga": [35.0, -1.2e-4, np.deg2rad(0.0020), np.deg2rad(-0.0015)],
    }
    return np.asarray([bias_by_name[name] for name in station_names], dtype=float)


def _write_rr_bias_recovery_csv(scenarios, true_bias_blocks, station_names, output_path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    true_bias_blocks = np.asarray(true_bias_blocks, dtype=float)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scenario",
                "arc_id",
                "station",
                "observed_in_arc",
                "true_range_bias_m",
                "estimated_range_bias_m",
                "range_bias_error_m",
                "true_rr_bias_mps",
                "estimated_rr_bias_mps",
                "rr_bias_error_mps",
                "true_az_bias_deg",
                "estimated_az_bias_deg",
                "az_bias_error_deg",
                "true_el_bias_deg",
                "estimated_el_bias_deg",
                "el_bias_error_deg",
            ]
        )
        for scenario in scenarios:
            for result in scenario.arc_results:
                est_blocks = _estimated_rr_bias_blocks(result.estimated_bias, len(station_names))
                observed = _observed_station_ids(result)
                for station_idx, station_name in enumerate(station_names):
                    true_block = true_bias_blocks[station_idx, :]
                    est_block = est_blocks[station_idx, :]
                    err = est_block - true_block
                    writer.writerow(
                        [
                            scenario.label,
                            result.arc_id,
                            station_name,
                            int(station_idx in observed),
                            true_block[0],
                            est_block[0],
                            err[0],
                            true_block[1],
                            est_block[1],
                            err[1],
                            np.rad2deg(true_block[2]),
                            np.rad2deg(est_block[2]),
                            np.rad2deg(err[2]),
                            np.rad2deg(true_block[3]),
                            np.rad2deg(est_block[3]),
                            np.rad2deg(err[3]),
                        ]
                    )
    return output_path


def _plot_rr_bias_recovery(scenarios, true_bias_blocks, station_names, output_path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    true_bias_blocks = np.asarray(true_bias_blocks, dtype=float)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 8.0), constrained_layout=True)
    fig.suptitle("Station-Full RR Bias Recovery", fontsize=13, fontweight="bold")

    palette = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    for idx, scenario in enumerate(scenarios):
        color = palette[idx % len(palette)]
        arc_ids = []
        range_rmse = []
        rr_rmse = []
        angle_rmse_deg = []

        for result in scenario.arc_results:
            est_blocks = _estimated_rr_bias_blocks(result.estimated_bias, len(station_names))
            observed = sorted(_observed_station_ids(result))
            if not observed:
                continue
            err = est_blocks[observed, :] - true_bias_blocks[observed, :]
            arc_ids.append(result.arc_id)
            range_rmse.append(float(np.sqrt(np.mean(err[:, 0] ** 2))))
            rr_rmse.append(float(np.sqrt(np.mean(err[:, 1] ** 2))))
            angle_rmse_deg.append(float(np.rad2deg(np.sqrt(np.mean(err[:, 2:4] ** 2)))))

        label = scenario.label
        axes[0].semilogy(arc_ids, np.maximum(range_rmse, 1e-12), marker="o", color=color, label=label)
        axes[1].semilogy(arc_ids, np.maximum(rr_rmse, 1e-14), marker="s", color=color, label=label)
        axes[2].semilogy(arc_ids, np.maximum(angle_rmse_deg, 1e-12), marker="^", color=color, label=label)

    axes[0].set_ylabel("range bias RMSE [m]")
    axes[1].set_ylabel("RR bias RMSE [m/s]")
    axes[2].set_ylabel("angle bias RMSE [deg]")
    axes[2].set_xlabel("Arc")

    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _estimated_rr_bias_blocks(estimated_bias: np.ndarray, num_stations: int) -> np.ndarray:
    estimated_bias = np.asarray(estimated_bias, dtype=float).reshape(-1)
    if estimated_bias.size != 4 * num_stations:
        return np.zeros((num_stations, 4), dtype=float)
    return estimated_bias.reshape(num_stations, 4)


def _observed_station_ids(result) -> set[int]:
    return set(int(idx) for idx in getattr(result, "observed_station_ids", ()))


def _summary_line(scenario) -> str:
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
