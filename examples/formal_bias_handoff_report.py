"""Run formal handoff with an augmented range-rate bias solve-for.

This report injects a global `[range, range_rate, az, el]` bias into compact
SPICE visibility arcs. It compares state hot-start against formal handoff with
the full augmented covariance, including state-bias cross covariance.

Run from the project root:

    python python_port/examples/formal_bias_handoff_report.py
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
    plot_scenario_comparison,
    propagate_truth_with_ephemeris,
    range_rate_stations,
    run_batch_arc_sequence,
    sample_j2000_to_itrf93_transforms,
    sample_moon_centered_ephemeris,
    write_scenario_summary_csv,
)


TRUE_GLOBAL_RR_BIAS = np.array([12.0, 6.0e-5, np.deg2rad(0.0008), np.deg2rad(-0.0006)])


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

    duration_h = 4.0
    sample_step_s = 240.0
    ephem_step_s = 600.0
    max_gap_s = 20.0 * 60.0

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
            min_elevation_deg=5.0,
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
        arcs = list(
            build_measurement_arcs(
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
        )
    finally:
        spice.kclear()

    for arc in arcs:
        arc.obs_data[:, 1:5] += TRUE_GLOBAL_RR_BIAS.reshape(1, 4)

    cold_bank = tuple(_shared_cold_perturbation(idx) for idx in range(len(arcs)))
    scenarios = []
    for start_mode in ("hot", "formal"):
        label = f"RR global-bias {start_mode}"
        print(f"Running {label} over {len(arcs)} arcs")
        scenarios.append(
            run_batch_arc_sequence(
                tuple(arcs),
                "range_rate",
                start_mode,
                "srif",
                mu_moon,
                mu_earth,
                mu_sun,
                ephemeris.earth_position,
                ephemeris.sun_position,
                cold_start_bank=cold_bank,
                initial_bias=np.zeros(4),
                label=label,
                max_iter=12,
                rtol=1e-10,
                atol=1e-11,
            )
        )

    out_dir = Path("python_port") / "results"
    summary_csv = write_scenario_summary_csv(scenarios, out_dir / "formal_bias_handoff_summary.csv")
    comparison_png = plot_scenario_comparison(
        scenarios,
        out_dir / "formal_bias_handoff_comparison.png",
        title="Formal Handoff with Global RR Bias Solve-For",
    )
    bias_csv = _write_bias_recovery_csv(scenarios, out_dir / "formal_bias_handoff_bias_recovery.csv")
    bias_png = _plot_bias_recovery(scenarios, out_dir / "formal_bias_handoff_bias_recovery.png")

    print(
        "Injected global RR bias: "
        f"range={TRUE_GLOBAL_RR_BIAS[0]:+.3f} m, "
        f"rr={TRUE_GLOBAL_RR_BIAS[1]:+.3e} m/s, "
        f"az={np.rad2deg(TRUE_GLOBAL_RR_BIAS[2]):+.6f} deg, "
        f"el={np.rad2deg(TRUE_GLOBAL_RR_BIAS[3]):+.6f} deg"
    )
    for scenario in scenarios:
        final = scenario.final_position_errors_m
        print(f"{scenario.label}: median final={np.median(final):.3g} m")
    print(f"Wrote {comparison_png}")
    print(f"Wrote {bias_png}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {bias_csv}")


def _shared_cold_perturbation(index: int) -> np.ndarray:
    rng = np.random.default_rng(241000 + index)
    return np.concatenate([180.0 * rng.standard_normal(3), 0.03 * rng.standard_normal(3)])


def _write_bias_recovery_csv(scenarios, output_path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scenario",
                "arc_id",
                "range_bias_error_m",
                "rr_bias_error_mps",
                "az_bias_error_deg",
                "el_bias_error_deg",
                "bias_sigma_rss",
                "state_bias_cross_norm",
            ]
        )
        for scenario in scenarios:
            for result in scenario.arc_results:
                err = result.estimated_bias - TRUE_GLOBAL_RR_BIAS
                writer.writerow(
                    [
                        scenario.label,
                        result.arc_id,
                        err[0],
                        err[1],
                        np.rad2deg(err[2]),
                        np.rad2deg(err[3]),
                        _bias_sigma(result.posterior_covariance),
                        _state_bias_cross_norm(result.posterior_covariance),
                    ]
                )
    return output_path


def _plot_bias_recovery(scenarios, output_path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.6), constrained_layout=True)
    fig.suptitle("Global RR Bias Recovery with Formal Handoff", fontsize=13, fontweight="bold")
    for scenario in scenarios:
        arc_ids = np.array([result.arc_id for result in scenario.arc_results], dtype=float)
        bias_errors = np.array([result.estimated_bias - TRUE_GLOBAL_RR_BIAS for result in scenario.arc_results])
        range_bias_error = np.abs(bias_errors[:, 0])
        angle_bias_error_deg = np.rad2deg(np.sqrt(np.mean(bias_errors[:, 2:4] ** 2, axis=1)))
        axes[0].semilogy(arc_ids, np.maximum(range_bias_error, 1e-12), marker="o", label=scenario.label)
        axes[1].semilogy(arc_ids, np.maximum(angle_bias_error_deg, 1e-12), marker="s", label=scenario.label)

    axes[0].set_ylabel("range bias abs error [m]")
    axes[1].set_ylabel("angle bias RMSE [deg]")
    axes[1].set_xlabel("Arc")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _bias_sigma(covariance) -> float:
    if covariance is None or covariance.shape[0] <= 6:
        return float("nan")
    return float(np.sqrt(np.sum(np.clip(np.diag(covariance)[6:], 0.0, None))))


def _state_bias_cross_norm(covariance) -> float:
    if covariance is None or covariance.shape[0] <= 6:
        return float("nan")
    return float(np.linalg.norm(covariance[:6, 6:], ord="fro"))


if __name__ == "__main__":
    main()
