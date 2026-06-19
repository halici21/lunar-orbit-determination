"""Run a compact formal and square-root formal handoff report.

This report compares state hot-start, covariance/information formal handoff,
and square-root information handoff. The sqrt_formal mode carries the previous
arc posterior SRIF R factor through the STM without converting to covariance
when no process noise is applied.

Run from the project root:

    python python_port/examples/formal_handoff_report.py
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
    run_batch_arc_sequence,
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

        scenarios = []
        for measurement_type in ("position", "range_rate"):
            arcs = build_measurement_arcs(
                measurement_type,
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
            print(f"{measurement_type}: {len(arcs)} SPICE visibility arcs")
            cold_bank = make_cold_start_bank(
                len(arcs),
                sigma_pos_m=180.0,
                sigma_vel_mps=0.03,
                seed=240810 + len(measurement_type),
            )
            for start_mode in ("hot", "formal", "sqrt_formal"):
                label = f"{measurement_type} {start_mode}"
                print(f"  running {label}")
                scenarios.append(
                    run_batch_arc_sequence(
                        arcs,
                        measurement_type,
                        start_mode,
                        "srif",
                        mu_moon,
                        mu_earth,
                        mu_sun,
                        ephemeris.earth_position,
                        ephemeris.sun_position,
                        cold_start_bank=cold_bank,
                        label=label,
                        max_iter=10,
                        rtol=1e-10,
                        atol=1e-11,
                    )
                )
    finally:
        spice.kclear()

    out_dir = Path("python_port") / "results"
    summary_csv = write_scenario_summary_csv(scenarios, out_dir / "formal_handoff_summary.csv")
    comparison_png = plot_scenario_comparison(
        scenarios,
        out_dir / "formal_handoff_comparison.png",
        title="SRIF Hot-Start vs Formal and Square-Root Handoff",
    )
    covariance_png = _plot_formal_covariance(scenarios, out_dir / "formal_handoff_covariance.png")

    for scenario in scenarios:
        initial = scenario.initial_position_errors_m
        final = scenario.final_position_errors_m
        print(
            f"{scenario.label}: arcs={len(scenario.arc_results)}, "
            f"median initial={np.median(initial):.3g} m, "
            f"median final={np.median(final):.3g} m"
        )

    print(f"Wrote {comparison_png}")
    print(f"Wrote {covariance_png}")
    print(f"Wrote {summary_csv}")


def _plot_formal_covariance(scenarios, output_path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    formal_scenarios = [scenario for scenario in scenarios if scenario.start_mode in {"formal", "sqrt_formal"}]
    fig, axes = plt.subplots(1, len(formal_scenarios), figsize=(11.0, 4.2), constrained_layout=True)
    if len(formal_scenarios) == 1:
        axes = [axes]
    fig.suptitle("Formal Handoff Position Error and 1-Sigma RSS", fontsize=13, fontweight="bold")

    for ax, scenario in zip(axes, formal_scenarios):
        arc_ids = np.array([result.arc_id for result in scenario.arc_results], dtype=float)
        final_error = np.array([result.final_position_error_m for result in scenario.arc_results], dtype=float)
        prior_sigma = np.array(
            [_covariance_rss_sigma(result.prior_covariance, slice(0, 3)) for result in scenario.arc_results],
            dtype=float,
        )
        posterior_sigma = np.array(
            [_covariance_rss_sigma(result.posterior_covariance, slice(0, 3)) for result in scenario.arc_results],
            dtype=float,
        )
        ax.semilogy(arc_ids, np.maximum(final_error, 1e-12), marker="o", label="actual final error")
        ax.semilogy(arc_ids, np.maximum(prior_sigma, 1e-12), marker="s", label="prior 1-sigma RSS")
        ax.semilogy(arc_ids, np.maximum(posterior_sigma, 1e-12), marker="^", label="posterior 1-sigma RSS")
        ax.set_title(f"{scenario.measurement_type} {scenario.start_mode}")
        ax.set_xlabel("Arc")
        ax.set_ylabel("m")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _covariance_rss_sigma(covariance, component_slice: slice) -> float:
    if covariance is None:
        return float("nan")
    cov = np.asarray(covariance, dtype=float)
    diag = np.clip(np.diag(cov)[component_slice], 0.0, None)
    return float(np.sqrt(np.sum(diag)))


if __name__ == "__main__":
    main()
