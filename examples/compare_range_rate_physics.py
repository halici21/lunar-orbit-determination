"""Compare instantaneous geometric RR and simplified two-way counted Doppler.

Run from the project root:

    python python_port/examples/compare_range_rate_physics.py

Outputs:

- python_port/results/range_rate_physics_comparison.csv
- python_port/results/range_rate_physics_residual_mismatch.csv
- python_port/results/range_rate_physics_comparison_summary.csv
- python_port/results/range_rate_physics_comparison.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_od import (  # noqa: E402
    compute_range_rate_residuals,
    RangeRatePhysicsConfig,
    generate_range_rate_measurements,
    load_spice_kernels,
    measurement_sigma_vector,
    propagate_truth_with_ephemeris,
    range_rate_stations,
    sample_moon_centered_ephemeris,
)


def main() -> None:
    args = _parse_args()
    fixture_path = Path("python_port") / "fixtures" / "spice_snapshots.json"
    if not fixture_path.is_file():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    initial = fixture["initial_state"]
    constants = fixture["constants"]
    x0_mci = np.asarray(initial["state_mci_j2000_m_mps"], dtype=float)
    mu_moon = float(initial["mu_moon_m3_s2"])
    mu_earth = float(np.asarray(constants["mu_earth_km3_s2"]).reshape(-1)[0] * 1e9)
    mu_sun = float(np.asarray(constants["mu_sun_km3_s2"]).reshape(-1)[0] * 1e9)

    stations_by_name = {station.name: station for station in range_rate_stations()}
    stations = [stations_by_name[name] for name in args.stations]

    duration_s = args.duration_h * 3600.0
    step_ratio = args.sample_step_s / args.propagation_step_s
    if abs(step_ratio - round(step_ratio)) > 1e-9:
        raise ValueError("sample-step-s must be an integer multiple of propagation-step-s.")
    sample_stride = int(round(step_ratio))
    t_eval_s = np.arange(0.0, duration_s + args.propagation_step_s, args.propagation_step_s)
    if t_eval_s.size < 4:
        raise ValueError("duration/sample step must produce at least four samples.")
    t_ephem_s = np.arange(0.0, duration_s + args.ephemeris_step_s, args.ephemeris_step_s)
    vis_mask = np.zeros((t_eval_s.size, len(stations)), dtype=bool)
    max_half_count_s = 0.5 * max(args.count_interval_s)
    measurement_indices = np.arange(0, t_eval_s.size, sample_stride, dtype=int)
    measurement_indices = measurement_indices[
        (t_eval_s[measurement_indices] >= max_half_count_s)
        & (t_eval_s[measurement_indices] <= duration_s - max_half_count_s)
    ]
    vis_mask[measurement_indices, :] = True

    import spiceypy as spice

    load_spice_kernels()
    try:
        et0 = float(spice.str2et(fixture["epoch_utc"]))
        ephemeris = sample_moon_centered_ephemeris(et0, t_ephem_s)
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
        geometric_obs, geometric_pass_geo = generate_range_rate_measurements(
            t_eval_s,
            state_history,
            stations,
            vis_mask,
            ephemeris.earth_position,
            ephemeris.earth_velocity,
            et0,
            noise=False,
            range_rate_physics="geometric",
        )
        two_way_cases = []
        for count_interval_s in args.count_interval_s:
            two_way_config = RangeRatePhysicsConfig(
                mode="two_way_counted_doppler",
                count_interval_s=count_interval_s,
                output_unit="mps_equivalent",
            )
            two_way_obs, two_way_pass_geo = generate_range_rate_measurements(
                t_eval_s,
                state_history,
                stations,
                vis_mask,
                ephemeris.earth_position,
                ephemeris.earth_velocity,
                et0,
                noise=False,
                range_rate_physics=two_way_config,
                bias_rr_mps=args.bias_rr_mps,
            )
            two_way_cases.append((count_interval_s, two_way_obs, two_way_pass_geo))
    finally:
        spice.kclear()

    rows = []
    mismatch_rows = []
    for count_interval_s, two_way_obs, two_way_pass_geo in two_way_cases:
        rows.extend(_comparison_rows(geometric_obs, two_way_obs, stations, count_interval_s))
        mismatch_rows.extend(
            _mismatch_rows(
                state_history,
                two_way_obs,
                geometric_pass_geo,
                two_way_pass_geo,
                stations,
                count_interval_s,
            )
        )
    out_dir = Path("python_port") / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = _write_detail_csv(rows, out_dir / f"{args.output_prefix}_comparison.csv")
    mismatch_csv = _write_mismatch_csv(mismatch_rows, out_dir / f"{args.output_prefix}_residual_mismatch.csv")
    summary = _summary_rows(rows, mismatch_rows)
    summary_csv = _write_summary_csv(summary, out_dir / f"{args.output_prefix}_comparison_summary.csv")
    plot_png = _plot_comparison(rows, summary, out_dir / f"{args.output_prefix}_comparison.png")

    print("Range-rate physics comparison")
    print(f"Count intervals: {', '.join(f'{item:g}' for item in args.count_interval_s)} s")
    print(f"Injected RR bias: {args.bias_rr_mps:.6g} m/s")
    for row in summary:
        print(
            f"Tc={row['count_interval_s']} s: diff_rms={row['difference_rms_mps']} m/s, "
            f"mismatch_rms={row['two_way_obs_with_geometric_model_rr_residual_rms_sigma']} sigma"
        )
    print(f"Wrote {detail_csv}")
    print(f"Wrote {mismatch_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {plot_png}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count-interval-s",
        type=float,
        nargs="+",
        default=[1.0, 60.0, 300.0],
        help="One or more two-way Doppler count intervals in seconds.",
    )
    parser.add_argument("--duration-h", type=float, default=4.0, help="Dense comparison trajectory duration in hours.")
    parser.add_argument("--sample-step-s", type=float, default=60.0, help="Measurement sample step in seconds.")
    parser.add_argument(
        "--propagation-step-s",
        type=float,
        default=1.0,
        help="Dense state/transform grid step used by the light-time model.",
    )
    parser.add_argument("--ephemeris-step-s", type=float, default=1800.0, help="SPICE ephemeris sampling step in seconds.")
    parser.add_argument(
        "--bias-rr-mps",
        type=float,
        default=0.0,
        help="Constant m/s-equivalent bias injected into the two-way Doppler component.",
    )
    parser.add_argument(
        "--output-prefix",
        default="range_rate_physics",
        help="Prefix for result filenames in python_port/results.",
    )
    parser.add_argument(
        "--stations",
        nargs="+",
        default=["ITU Ayazaga", "Goldstone DSN"],
        help="Station names to include in the comparison.",
    )
    return parser.parse_args()


def _comparison_rows(
    geometric_obs: np.ndarray,
    two_way_obs: np.ndarray,
    stations,
    count_interval_s: float,
) -> list[dict]:
    if geometric_obs.shape != two_way_obs.shape:
        raise ValueError("geometric and two-way observation arrays must have the same shape.")
    rows = []
    for idx in range(geometric_obs.shape[0]):
        station_idx = int(geometric_obs[idx, 5]) - 1
        geometric_rr = float(geometric_obs[idx, 2])
        two_way_rr = float(two_way_obs[idx, 2])
        rows.append(
            {
                "obs_index": idx + 1,
                "count_interval_s": float(count_interval_s),
                "t_s": float(geometric_obs[idx, 0]),
                "t_h": float(geometric_obs[idx, 0]) / 3600.0,
                "station": stations[station_idx].name,
                "range_m": float(geometric_obs[idx, 1]),
                "geometric_rr_mps": geometric_rr,
                "two_way_counted_rr_mps_equiv": two_way_rr,
                "rr_difference_mps": two_way_rr - geometric_rr,
                "az_rad": float(geometric_obs[idx, 3]),
                "el_rad": float(geometric_obs[idx, 4]),
            }
        )
    return rows


def _mismatch_rows(
    state_history: np.ndarray,
    two_way_obs: np.ndarray,
    geometric_pass_geo,
    two_way_pass_geo,
    stations,
    count_interval_s: float,
) -> list[dict]:
    wrong_residuals, _ = compute_range_rate_residuals(state_history, two_way_obs, geometric_pass_geo)
    correct_residuals, _ = compute_range_rate_residuals(state_history, two_way_obs, two_way_pass_geo)
    sigma = measurement_sigma_vector(two_way_obs, geometric_pass_geo, "range_rate")
    wrong_blocks = wrong_residuals.reshape(-1, 4)
    correct_blocks = correct_residuals.reshape(-1, 4)
    sigma_blocks = sigma.reshape(-1, 4)
    rows = []
    for idx in range(two_way_obs.shape[0]):
        station_idx = int(two_way_obs[idx, 5]) - 1
        rows.append(
            {
                "obs_index": idx + 1,
                "count_interval_s": float(count_interval_s),
                "t_s": float(two_way_obs[idx, 0]),
                "t_h": float(two_way_obs[idx, 0]) / 3600.0,
                "station": stations[station_idx].name,
                "two_way_obs_rr_mps_equiv": float(two_way_obs[idx, 2]),
                "rr_residual_if_geometric_model_mps": float(wrong_blocks[idx, 1]),
                "rr_residual_if_two_way_model_mps": float(correct_blocks[idx, 1]),
                "rr_residual_if_geometric_model_sigma": float(wrong_blocks[idx, 1] / sigma_blocks[idx, 1]),
                "rr_residual_if_two_way_model_sigma": float(correct_blocks[idx, 1] / sigma_blocks[idx, 1]),
            }
        )
    return rows


def _summary_rows(rows: list[dict], mismatch_rows: list[dict]) -> list[dict]:
    summary = []
    for count_interval_s in sorted({float(row["count_interval_s"]) for row in rows}):
        case_rows = [row for row in rows if float(row["count_interval_s"]) == count_interval_s]
        case_mismatch_rows = [row for row in mismatch_rows if float(row["count_interval_s"]) == count_interval_s]
        diff = np.array([row["rr_difference_mps"] for row in case_rows], dtype=float)
        geometric = np.array([row["geometric_rr_mps"] for row in case_rows], dtype=float)
        two_way = np.array([row["two_way_counted_rr_mps_equiv"] for row in case_rows], dtype=float)
        wrong_rr_residual = np.array(
            [row["rr_residual_if_geometric_model_mps"] for row in case_mismatch_rows],
            dtype=float,
        )
        correct_rr_residual = np.array(
            [row["rr_residual_if_two_way_model_mps"] for row in case_mismatch_rows],
            dtype=float,
        )
        wrong_rr_sigma = np.array(
            [row["rr_residual_if_geometric_model_sigma"] for row in case_mismatch_rows],
            dtype=float,
        )
        summary.append(
            {
                "count_interval_s": f"{count_interval_s:.6g}",
                "num_observations": str(len(case_rows)),
                "geometric_rr_mean_mps": f"{np.mean(geometric):.12e}",
                "two_way_rr_mean_mps_equiv": f"{np.mean(two_way):.12e}",
                "difference_mean_mps": f"{np.mean(diff):.12e}",
                "difference_rms_mps": f"{np.sqrt(np.mean(diff**2)):.12e}",
                "difference_max_abs_mps": f"{np.max(np.abs(diff)):.12e}",
                "difference_median_mps": f"{np.median(diff):.12e}",
                "two_way_obs_with_geometric_model_rr_residual_rms_mps": f"{np.sqrt(np.mean(wrong_rr_residual**2)):.12e}",
                "two_way_obs_with_geometric_model_rr_residual_max_abs_mps": f"{np.max(np.abs(wrong_rr_residual)):.12e}",
                "two_way_obs_with_geometric_model_rr_residual_rms_sigma": f"{np.sqrt(np.mean(wrong_rr_sigma**2)):.12e}",
                "two_way_obs_with_two_way_model_rr_residual_rms_mps": f"{np.sqrt(np.mean(correct_rr_residual**2)):.12e}",
            }
        )
    return summary


def _write_detail_csv(rows: list[dict], output_path: Path) -> Path:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _write_summary_csv(rows: list[dict], output_path: Path) -> Path:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _write_mismatch_csv(rows: list[dict], output_path: Path) -> Path:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _plot_comparison(rows: list[dict], summary: list[dict], output_path: Path) -> Path:
    stations = sorted({row["station"] for row in rows})
    colors = ["#2f6f9f", "#b83232", "#5b8e3e", "#9a5b2f", "#6b4c9a"]
    count_intervals = sorted({float(row["count_interval_s"]) for row in rows})
    display_tc = 60.0 if 60.0 in count_intervals else count_intervals[0]

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 8.0), constrained_layout=True)
    fig.suptitle("Range-rate physics comparison: geometric vs two-way counted Doppler")

    for station_idx, station in enumerate(stations):
        station_rows = [
            row for row in rows if row["station"] == station and float(row["count_interval_s"]) == display_tc
        ]
        t_h = np.array([row["t_h"] for row in station_rows], dtype=float)
        geometric = np.array([row["geometric_rr_mps"] for row in station_rows], dtype=float)
        two_way = np.array([row["two_way_counted_rr_mps_equiv"] for row in station_rows], dtype=float)
        color = colors[station_idx % len(colors)]
        axes[0].plot(t_h, geometric, color=color, linewidth=1.5, label=f"{station} geometric")
        axes[0].plot(
            t_h,
            two_way,
            color=color,
            linestyle="--",
            linewidth=1.3,
            label=f"{station} two-way Tc={display_tc:g}s",
        )

    primary_station = "ITU Ayazaga" if "ITU Ayazaga" in stations else stations[0]
    for interval_idx, count_interval_s in enumerate(count_intervals):
        case_rows = [
            row
            for row in rows
            if row["station"] == primary_station and float(row["count_interval_s"]) == count_interval_s
        ]
        t_h = np.array([row["t_h"] for row in case_rows], dtype=float)
        diff = np.array([row["rr_difference_mps"] for row in case_rows], dtype=float)
        axes[1].plot(
            t_h,
            diff,
            color=colors[interval_idx % len(colors)],
            linewidth=1.3,
            label=f"{primary_station} Tc={count_interval_s:g}s",
        )

    axes[0].set_ylabel("range-rate [m/s]")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].axhline(0.0, color="#222222", linewidth=0.8)
    axes[1].set_ylabel("two-way - geometric [m/s]")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)

    metric_lines = ["Tc [s] | RMS diff [m/s] | max diff [m/s] | wrong-model RMS [sigma] | correct-model RMS [m/s]"]
    for row in summary:
        metric_lines.append(
            f"{row['count_interval_s']} | {float(row['difference_rms_mps']):.3e} | "
            f"{float(row['difference_max_abs_mps']):.3e} | "
            f"{float(row['two_way_obs_with_geometric_model_rr_residual_rms_sigma']):.3e} | "
            f"{float(row['two_way_obs_with_two_way_model_rr_residual_rms_mps']):.3e}"
        )
    metric_text = "\n".join(metric_lines)
    axes[2].axis("off")
    axes[2].text(
        0.02,
        0.95,
        metric_text,
        va="top",
        ha="left",
        fontsize=10,
        linespacing=1.45,
        bbox={"facecolor": "#f5f7f8", "edgecolor": "#c7d0d5", "boxstyle": "round,pad=0.45"},
    )
    axes[2].set_title("Summary")
    axes[1].set_xlabel("arc time [h]")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    main()
