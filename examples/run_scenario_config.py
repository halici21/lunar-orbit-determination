"""Run one Lunar OD scenario JSON config end-to-end.

Run from the project root:

    python python_port/examples/run_scenario_config.py python_port/results/my.scenario.json
    python python_port/examples/run_scenario_config.py config.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_od import (  # noqa: E402
    VisibilityConfig,
    build_measurement_arcs,
    load_scenario_config_json,
    load_spice_kernels,
    make_cold_start_bank,
    plot_scenario_comparison,
    perturb_moon_centered_ephemeris,
    propagate_truth_with_ephemeris,
    range_rate_stations,
    sample_j2000_to_itrf93_transforms,
    sample_moon_centered_ephemeris,
    scenario_config_summary,
    scenario_range_rate_physics_config,
    scenario_ukf_configs,
    thesis_network_by_name,
    thesis_seed_for,
    write_scenario_summary_csv,
)
from lunar_od.thesis_matrix import (  # noqa: E402
    THESIS_COLD_START_SIGMA_POS_M,
    THESIS_COLD_START_SIGMA_VEL_MPS,
    THESIS_EPHEMERIS_STEP_S,
    THESIS_MAX_GAP_S,
    THESIS_MIN_ELEVATION_DEG,
)


def main(argv=None) -> int:
    args = _parse_args(argv)
    config = load_scenario_config_json(args.config)
    output_dir = Path(args.output_dir or config.output_dir)
    summary_csv = output_dir / f"{config.name}_summary.csv"
    comparison_png = output_dir / f"{config.name}_comparison.png"

    print(scenario_config_summary(config))
    if args.dry_run:
        print(f"Would write {summary_csv}")
        print(f"Would write {comparison_png}")
        return 0

    scenario = run_configured_scenario(config)
    summary_csv = write_scenario_summary_csv([scenario], summary_csv)
    comparison_png = plot_scenario_comparison([scenario], comparison_png, title=config.name)
    print(f"Wrote {summary_csv}")
    print(f"Wrote {comparison_png}")
    return 0


def run_configured_scenario(
    config,
    *,
    earth_position_bias_m=(0.0, 0.0, 0.0),
    earth_velocity_bias_mps=(0.0, 0.0, 0.0),
    sun_position_bias_m=(0.0, 0.0, 0.0),
    measurement_seed: int | None = None,
    cold_start_seed: int | None = None,
    cold_start_scale: float = 1.0,
):
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

    t_eval_s = np.arange(0.0, config.duration_h * 3600.0 + config.sample_step_s, config.sample_step_s)
    t_ephem_s = np.arange(0.0, config.duration_h * 3600.0 + THESIS_EPHEMERIS_STEP_S, THESIS_EPHEMERIS_STEP_S)
    all_stations_by_name = {station.name: station for station in range_rate_stations()}
    network = thesis_network_by_name(config.network)
    stations = [all_stations_by_name[name] for name in network.station_names]

    import spiceypy as spice

    load_spice_kernels()
    try:
        et0 = float(spice.str2et(epoch_utc))
        ephemeris = sample_moon_centered_ephemeris(et0, t_ephem_s)
        estimator_ephemeris = perturb_moon_centered_ephemeris(
            ephemeris,
            earth_position_bias_m=earth_position_bias_m,
            earth_velocity_bias_mps=earth_velocity_bias_mps,
            sun_position_bias_m=sun_position_bias_m,
        )
        x_j2000_to_itrf93 = sample_j2000_to_itrf93_transforms(et0, t_eval_s)
        truth = propagate_truth_with_ephemeris(
            t_eval_s,
            x0_mci,
            mu_moon,
            mu_earth,
            mu_sun,
            ephemeris,
            rtol=config.rtol,
            atol=config.atol,
        )
        visibility_config = VisibilityConfig(
            r_moon_mean_m=float(initial["r_moon_mean_m"]),
            earth_rotation_rad_s=7.292115e-5,
            epoch_utc=epoch_utc,
            min_elevation_deg=THESIS_MIN_ELEVATION_DEG,
        )
        from lunar_od import analyze_visibility_gap_with_transforms, run_batch_arc_sequence

        seg_starts, seg_ends, vis_mask_raw, _ = analyze_visibility_gap_with_transforms(
            t_eval_s,
            truth,
            stations,
            ephemeris.earth_position,
            x_j2000_to_itrf93,
            THESIS_MAX_GAP_S,
            visibility_config,
        )
        arcs = build_measurement_arcs(
            config.measurement_type,
            t_eval_s,
            truth,
            seg_starts,
            seg_ends,
            vis_mask_raw,
            stations,
            ephemeris.earth_position,
            ephemeris.earth_velocity,
            et0,
            noise=config.noise,
            rng=None if measurement_seed is None else np.random.default_rng(int(measurement_seed)),
            min_samples=4,
            range_rate_physics=scenario_range_rate_physics_config(config),
            apply_light_time=config.apply_light_time,
            apply_stellar_aberration=config.apply_stellar_aberration,
            stellar_aberration_model=config.stellar_aberration_model,
        )
        arcs = tuple(_with_estimator_ephemeris(arc, estimator_ephemeris) for arc in arcs)
        cold_bank = make_cold_start_bank(
            len(arcs),
            THESIS_COLD_START_SIGMA_POS_M * float(cold_start_scale),
            THESIS_COLD_START_SIGMA_VEL_MPS * float(cold_start_scale),
            seed=(
                int(cold_start_seed)
                if cold_start_seed is not None
                else thesis_seed_for(config.network, config.measurement_type)
            ),
        )
        ukf_transform_config, ukf_adaptive_config = scenario_ukf_configs(config)
        return run_batch_arc_sequence(
            arcs,
            config.measurement_type,
            config.start_mode,
            config.estimator_type,
            mu_moon,
            mu_earth,
            mu_sun,
            estimator_ephemeris.earth_position,
            estimator_ephemeris.sun_position,
            cold_start_bank=cold_bank,
            bias_mode=config.bias_mode,
            initial_bias=_initial_bias(config, len(stations)),
            label=config.name,
            process_noise_covariance=config.ukf_acceleration_psd_m2_s3,
            max_iter=config.max_iter,
            rtol=config.rtol,
            atol=config.atol,
            ukf_transform_config=ukf_transform_config,
            ukf_adaptive_config=ukf_adaptive_config,
            ukf_covariance_form=config.ukf_covariance_form,
            ukf_process_noise_model=config.ukf_process_noise_model,
            ukf_auto_bias_constraints=config.ukf_auto_bias_constraints,
            ukf_bias_freeze_relative_information=config.ukf_bias_freeze_relative_information,
            ukf_bias_regularize_relative_information=config.ukf_bias_regularize_relative_information,
            ukf_bias_regularization_std=config.ukf_bias_regularization_std,
        )
    finally:
        spice.kclear()


def _with_estimator_ephemeris(arc, ephemeris):
    pass_geo = replace(
        arc.pass_geo,
        earth_pos_mci_m=np.asarray(ephemeris.earth_position(arc.pass_geo.t_s), dtype=float),
        earth_vel_mci_mps=np.asarray(ephemeris.earth_velocity(arc.pass_geo.t_s), dtype=float),
    )
    return replace(arc, pass_geo=pass_geo)


def _initial_bias(config, num_stations: int) -> np.ndarray:
    if config.bias_mode is None:
        return np.zeros(0, dtype=float)
    measurement_dim = 3 if config.measurement_type == "position" else 4
    if config.bias_mode == "global":
        return np.zeros(measurement_dim, dtype=float)
    if config.bias_mode == "station_angles":
        return np.zeros(2 * num_stations, dtype=float)
    if config.bias_mode == "station_full":
        return np.zeros(measurement_dim * num_stations, dtype=float)
    raise ValueError(f"Unsupported bias_mode: {config.bias_mode}")


def _parse_args(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one normalized Lunar OD scenario config.")
    parser.add_argument("config", help="Scenario JSON path.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print expected outputs without running OD.")
    parser.add_argument("--output-dir", help="Override the config output_dir.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
