"""Run the baseline synthetic BLS-vs-UKF comparison.

Baseline definition:

- Moon-centered inertial truth trajectory
- lunar two-body dynamics plus fixed Earth/Sun third-body points
- synthetic range/azimuth/elevation measurements
- four ground stations
- optional Gaussian measurement noise and no model mismatch
- identical cold-start perturbations for BLS-LM and UKF

Run from the project root:

    python python_port/examples/baseline_bls_ukf_comparison.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_od import (  # noqa: E402
    PassGeometry,
    PreparedArc,
    Station,
    UKFAdaptiveConfig,
    UnscentedTransformConfig,
    VisibilityConfig,
    analyze_visibility_gap_with_transforms,
    build_measurement_arcs,
    chi_square_nis_gate,
    compute_position_residuals_analytic,
    load_spice_kernels,
    plot_scenario_comparison,
    position_only_stations,
    propagate_augmented_state,
    propagate_truth_with_ephemeris,
    run_batch_arc_sequence,
    sample_j2000_to_itrf93_transforms,
    sample_moon_centered_ephemeris,
    write_scenario_summary_csv,
)


def main() -> None:
    args = _parse_args()
    experiment = _build_experiment(args)
    mu_moon = experiment["mu_moon"]
    mu_earth = experiment["mu_earth"]
    mu_sun = experiment["mu_sun"]
    get_earth_pos = experiment["get_earth_pos"]
    get_sun_pos = experiment["get_sun_pos"]
    arcs = experiment["arcs"]
    if args.max_arcs is not None:
        arcs = arcs[: args.max_arcs]
    cold_bank = _cold_start_bank(
        len(arcs),
        sigma_pos_m=args.cold_sigma_pos_m,
        sigma_vel_mps=args.cold_sigma_vel_mps,
        seed=args.seed,
    )

    scenarios = []
    runtimes: dict[str, float] = {}
    ukf_adaptive_config = _ukf_adaptive_config(args)

    for start_mode in args.start_modes:
        if start_mode == "sqrt_formal" and args.batch_estimator != "srif":
            print("Skipping sqrt_formal for batch_estimator != srif.")
            continue

        start = perf_counter()
        bls = run_batch_arc_sequence(
            arcs,
            "position",
            start_mode,
            args.batch_estimator,
            mu_moon,
            mu_earth,
            mu_sun,
            get_earth_pos,
            get_sun_pos,
            cold_start_bank=cold_bank,
            label=f"Baseline {args.batch_estimator.upper()} {start_mode}",
            max_iter=args.max_iter,
            rtol=args.rtol,
            atol=args.atol,
            process_noise_covariance=np.diag([args.handoff_sigma_pos_m**2] * 3 + [args.handoff_sigma_vel_mps**2] * 3)
            if start_mode in {"formal", "sqrt_formal"}
            else None,
        )
        runtimes[bls.label] = perf_counter() - start
        scenarios.append(bls)

        if start_mode == "sqrt_formal":
            continue
        start = perf_counter()
        ukf = run_batch_arc_sequence(
            arcs,
            "position",
            start_mode,
            "ukf",
            mu_moon,
            mu_earth,
            mu_sun,
            get_earth_pos,
            get_sun_pos,
            cold_start_bank=cold_bank,
            label=f"Baseline UKF {start_mode}",
            process_noise_covariance=np.diag([0.01**2] * 3 + [1e-5**2] * 3),
            ukf_transform_config=UnscentedTransformConfig(alpha=0.35),
            ukf_adaptive_config=ukf_adaptive_config,
            ukf_covariance_form="square_root",
            rtol=args.rtol,
            atol=args.atol,
        )
        runtimes[ukf.label] = perf_counter() - start
        scenarios.append(ukf)

    suffix = _filename_suffix(args)
    out_dir = Path("python_port") / "results" / "baseline_bls_ukf"
    summary_csv = write_scenario_summary_csv(scenarios, out_dir / f"baseline_bls_ukf_{suffix}_arc_summary.csv")
    aggregate_csv = _write_aggregate_csv(
        scenarios,
        runtimes,
        cold_bank,
        out_dir / f"baseline_bls_ukf_{suffix}_aggregate.csv",
    )
    figure_path = plot_scenario_comparison(
        scenarios,
        out_dir / f"baseline_bls_ukf_{suffix}_comparison.png",
        title=(
            f"Baseline Position OD: BLS-LM vs UKF "
            f"({args.arc_source}, {args.duration_days:g} days, {_noise_label(args)})"
        ),
    )
    runtime_figure = _plot_runtime_success_consistency(
        scenarios,
        runtimes,
        out_dir / f"baseline_bls_ukf_{suffix}_runtime_success_consistency.png",
        title=f"Baseline BLS-vs-UKF summary ({args.arc_source}, {args.duration_days:g} days)",
    )
    error_summary_figure = _plot_error_summary(
        scenarios,
        out_dir / f"baseline_bls_ukf_{suffix}_median_p95_max_errors.png",
        title=f"Median / P95 / Max final error ({args.arc_source}, {args.duration_days:g} days)",
    )

    print("Baseline BLS-vs-UKF comparison")
    print(
        f"arc_source={args.arc_source}, duration_days={args.duration_days:g}, sample_step_s={args.sample_step_s:g}, "
        f"arc_duration_h={args.arc_duration_h:g}, arc_stride_h={args.arc_stride_h:g}, "
        f"arcs={len(arcs)}, noise={_noise_label(args)}"
    )
    for scenario in scenarios:
        final = scenario.final_position_errors_m
        reduced_chi_square = _scenario_reduced_chi_square(scenario)
        print(
            f"{scenario.label}: arcs={len(scenario.arc_results)}, "
            f"median_final_pos={np.median(final):.6g} m, "
            f"max_final_pos={np.max(final):.6g} m, "
            f"operational_success={scenario.operational_success_fraction:.2f}, "
            f"median_reduced_chi_square={np.nanmedian(reduced_chi_square):.6g}, "
            f"median_ukf_norm_nis={_nanmedian_arc_attr(scenario, 'ukf_normalized_mean_nis'):.6g}, "
            f"runtime={runtimes[scenario.label]:.3f} s"
        )
    print(f"Wrote {aggregate_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {figure_path}")
    print(f"Wrote {runtime_figure}")
    print(f"Wrote {error_summary_figure}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arc-source", choices=("regular", "spice_visibility"), default="regular")
    parser.add_argument("--duration-days", type=float, default=7.0)
    parser.add_argument("--sample-step-s", type=float, default=600.0)
    parser.add_argument("--ephem-step-s", type=float, default=3600.0)
    parser.add_argument("--arc-duration-h", type=float, default=2.0)
    parser.add_argument("--arc-stride-h", type=float, default=6.0)
    parser.add_argument("--max-arcs", type=int)
    parser.add_argument(
        "--start-modes",
        nargs="+",
        choices=("cold", "hot", "formal", "sqrt_formal"),
        default=("cold",),
    )
    parser.add_argument("--cold-sigma-pos-m", type=float, default=150.0)
    parser.add_argument("--cold-sigma-vel-mps", type=float, default=0.05)
    parser.add_argument("--handoff-sigma-pos-m", type=float, default=0.1)
    parser.add_argument("--handoff-sigma-vel-mps", type=float, default=1e-4)
    parser.add_argument("--batch-estimator", choices=("srif", "bls_lm"), default="bls_lm")
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--measurement-noise", action="store_true")
    parser.add_argument("--measurement-seed", type=int, default=20260610)
    parser.add_argument("--sigma-range-m", type=float, default=5.0)
    parser.add_argument("--sigma-angle-rad", type=float, default=1e-5)
    parser.add_argument("--strict-outlier-gate", action="store_true")
    parser.add_argument("--outlier-gate-sigma", type=float, default=3.0)
    parser.add_argument("--visibility-network", choices=("dsn4", "itu", "all"), default="dsn4")
    parser.add_argument("--min-elevation-deg", type=float, default=5.0)
    parser.add_argument("--max-gap-s", type=float, default=1800.0)
    parser.add_argument("--min-visibility-samples", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--rtol", type=float, default=1e-11)
    parser.add_argument("--atol", type=float, default=1e-12)
    return parser.parse_args()


def _ukf_adaptive_config(args: argparse.Namespace) -> UKFAdaptiveConfig:
    if not args.strict_outlier_gate:
        return UKFAdaptiveConfig()
    return UKFAdaptiveConfig(
        nis_gate=chi_square_nis_gate(3, sigma=args.outlier_gate_sigma),
        component_nis_gate=args.outlier_gate_sigma**2,
        component_gate_mode="conditional",
    )


def _build_experiment(args: argparse.Namespace) -> dict:
    if args.arc_source == "regular":
        return _regular_experiment(args)
    return _spice_visibility_experiment(args)


def _regular_experiment(args: argparse.Namespace) -> dict:
    mu_moon = 4902.800066e9
    t_all_s, x_truth, get_earth_pos, get_sun_pos = _truth_history(
        mu_moon,
        duration_days=args.duration_days,
        sample_step_s=args.sample_step_s,
    )
    stations = (
        _synthetic_station("Equator 0", 0.0, 0.0, 0.0, args.sigma_range_m, args.sigma_angle_rad),
        _synthetic_station("Equator 90E", 0.0, 90.0, 0.0, args.sigma_range_m, args.sigma_angle_rad),
        _synthetic_station("Midlat West", 45.0, -30.0, 500.0, args.sigma_range_m, args.sigma_angle_rad),
        _synthetic_station("South East", -35.0, 150.0, 600.0, args.sigma_range_m, args.sigma_angle_rad),
    )
    arcs = _build_baseline_arcs(
        t_all_s,
        x_truth,
        stations,
        sample_step_s=args.sample_step_s,
        arc_duration_h=args.arc_duration_h,
        arc_stride_h=args.arc_stride_h,
        measurement_noise=args.measurement_noise,
        seed=args.measurement_seed,
    )
    return {
        "mu_moon": mu_moon,
        "mu_earth": 0.0,
        "mu_sun": 0.0,
        "get_earth_pos": get_earth_pos,
        "get_sun_pos": get_sun_pos,
        "arcs": arcs,
    }


def _spice_visibility_experiment(args: argparse.Namespace) -> dict:
    fixture_path = Path("python_port") / "fixtures" / "spice_snapshots.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    initial = fixture["initial_state"]
    constants = fixture["constants"]
    epoch_utc = fixture["epoch_utc"]

    x0_mci = np.asarray(initial["state_mci_j2000_m_mps"], dtype=float)
    mu_moon = float(initial["mu_moon_m3_s2"])
    mu_earth = float(np.asarray(constants["mu_earth_km3_s2"]).reshape(-1)[0] * 1e9)
    mu_sun = float(np.asarray(constants["mu_sun_km3_s2"]).reshape(-1)[0] * 1e9)
    duration_h = float(args.duration_days) * 24.0
    t_eval_s = np.arange(0.0, duration_h * 3600.0 + float(args.sample_step_s), float(args.sample_step_s))
    t_ephem_s = np.arange(0.0, duration_h * 3600.0 + float(args.ephem_step_s), float(args.ephem_step_s))

    import spiceypy as spice

    load_spice_kernels()
    try:
        et0 = float(spice.str2et(epoch_utc))
        ephemeris = sample_moon_centered_ephemeris(et0, t_ephem_s)
        xforms = sample_j2000_to_itrf93_transforms(et0, t_eval_s)
        state_history = propagate_truth_with_ephemeris(
            t_eval_s,
            x0_mci,
            mu_moon,
            mu_earth,
            mu_sun,
            ephemeris,
            rtol=args.rtol,
            atol=args.atol,
        )
        stations = _visibility_stations(args)
        config = VisibilityConfig(
            r_moon_mean_m=float(initial["r_moon_mean_m"]),
            earth_rotation_rad_s=7.292115e-5,
            epoch_utc=epoch_utc,
            min_elevation_deg=args.min_elevation_deg,
        )
        seg_starts, seg_ends, vis_mask_raw, _ = analyze_visibility_gap_with_transforms(
            t_eval_s,
            state_history,
            stations,
            ephemeris.earth_position,
            xforms,
            args.max_gap_s,
            config,
        )
        arcs = build_measurement_arcs(
            "position",
            t_eval_s,
            state_history,
            seg_starts,
            seg_ends,
            vis_mask_raw,
            stations,
            ephemeris.earth_position,
            ephemeris.earth_velocity,
            et0,
            noise=args.measurement_noise,
            rng=np.random.default_rng(args.measurement_seed),
            min_samples=args.min_visibility_samples,
        )
    finally:
        spice.kclear()

    return {
        "mu_moon": mu_moon,
        "mu_earth": mu_earth,
        "mu_sun": mu_sun,
        "get_earth_pos": ephemeris.earth_position,
        "get_sun_pos": ephemeris.sun_position,
        "arcs": arcs,
    }


def _visibility_stations(args: argparse.Namespace) -> tuple[Station, ...]:
    stations_by_name = {station.name: station for station in position_only_stations()}
    if args.visibility_network == "itu":
        names = ("ITU Ayazaga",)
    elif args.visibility_network == "dsn4":
        names = ("ITU Ayazaga", "Goldstone DSN", "Madrid DSN", "Canberra DSN")
    else:
        return tuple(position_only_stations())
    return tuple(
        _station_with_noise(stations_by_name[name], args.sigma_range_m, args.sigma_angle_rad)
        for name in names
    )


def _station_with_noise(station: Station, sigma_range_m: float, sigma_angle_rad: float) -> Station:
    return Station(
        name=station.name,
        lat_deg=station.lat_deg,
        lon_deg=station.lon_deg,
        alt_m=station.alt_m,
        color_rgb=station.color_rgb,
        sigma_range_m=float(sigma_range_m),
        sigma_angle_rad=float(sigma_angle_rad),
        sigma_range_rate_mps=station.sigma_range_rate_mps,
        bias=station.bias,
    )


def _truth_history(mu_moon: float, *, duration_days: float, sample_step_s: float):
    r0norm = 1737.4e3 + 100e3
    x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
    duration_s = float(duration_days) * 86400.0
    t_all_s = np.arange(0.0, duration_s + float(sample_step_s), float(sample_step_s))
    get_earth_pos = lambda t: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
    get_sun_pos = lambda t: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
    x_aug0 = np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")])
    truth = propagate_augmented_state(
        t_all_s,
        x_aug0,
        mu_moon,
        0.0,
        0.0,
        get_earth_pos,
        get_sun_pos,
        rtol=1e-11,
        atol=1e-12,
    )[:, :6]
    return t_all_s, truth, get_earth_pos, get_sun_pos


def _synthetic_station(
    name: str,
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    sigma_range_m: float,
    sigma_angle_rad: float,
) -> Station:
    return Station(
        name=name,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        alt_m=alt_m,
        color_rgb=(0.0, 0.0, 0.0),
        sigma_range_m=float(sigma_range_m),
        sigma_angle_rad=float(sigma_angle_rad),
    )


def _build_position_arc(arc_id, start_idx, end_idx, t_all_s, x_truth, stations, *, rng=None):
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
    if rng is not None:
        obs_data[:, 1:4] += _position_measurement_noise(obs_data, pass_geo, rng)
    return PreparedArc(
        arc_id=arc_id,
        start_idx=start_idx,
        end_idx=end_idx,
        t_pass_s=t_pass_s,
        truth_state_history_mci=x_pass,
        obs_data=obs_data,
        pass_geo=pass_geo,
    )


def _build_baseline_arcs(
    t_all_s: np.ndarray,
    x_truth: np.ndarray,
    stations,
    *,
    sample_step_s: float,
    arc_duration_h: float,
    arc_stride_h: float,
    measurement_noise: bool,
    seed: int,
) -> tuple[PreparedArc, ...]:
    arc_span_steps = max(1, int(round(float(arc_duration_h) * 3600.0 / float(sample_step_s))))
    stride_steps = max(1, int(round(float(arc_stride_h) * 3600.0 / float(sample_step_s))))
    arcs = []
    arc_id = 1
    rng = np.random.default_rng(seed) if measurement_noise else None
    for start_idx in range(0, t_all_s.size - arc_span_steps, stride_steps):
        end_idx = start_idx + arc_span_steps
        if end_idx >= t_all_s.size:
            break
        arcs.append(_build_position_arc(arc_id, start_idx, end_idx, t_all_s, x_truth, stations, rng=rng))
        arc_id += 1
    return tuple(arcs)


def _position_measurement_noise(obs_data: np.ndarray, pass_geo: PassGeometry, rng: np.random.Generator) -> np.ndarray:
    noise = np.zeros((obs_data.shape[0], 3), dtype=float)
    for obs_idx in range(obs_data.shape[0]):
        station = pass_geo.stations[int(obs_data[obs_idx, 4]) - 1]
        noise[obs_idx, :] = [
            station.sigma_range_m * rng.standard_normal(),
            station.sigma_angle_rad * rng.standard_normal(),
            station.sigma_angle_rad * rng.standard_normal(),
        ]
    return noise


def _cold_start_bank(num_arcs: int, *, sigma_pos_m: float, sigma_vel_mps: float, seed: int):
    rng = np.random.default_rng(seed)
    return tuple(
        np.concatenate(
            [
                float(sigma_pos_m) * rng.standard_normal(3),
                float(sigma_vel_mps) * rng.standard_normal(3),
            ]
        )
        for _ in range(num_arcs)
    )


def _write_aggregate_csv(scenarios, runtimes: dict[str, float], cold_bank, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scenario",
                "estimator_type",
                "start_mode",
                "num_arcs",
                "algorithmic_success_fraction",
                "operational_success_fraction",
                "median_initial_position_error_m",
                "p95_initial_position_error_m",
                "median_final_position_error_m",
                "p95_final_position_error_m",
                "max_final_position_error_m",
                "median_final_velocity_error_mps",
                "p95_final_velocity_error_mps",
                "max_final_velocity_error_mps",
                "median_final_cost",
                "median_reduced_chi_square",
                "p95_reduced_chi_square",
                "median_condition_number",
                "p95_condition_number",
                "max_condition_number",
                "median_iteration_or_update_count",
                "p95_iteration_or_update_count",
                "max_iteration_or_update_count",
                "median_active_weight_fraction",
                "median_rejected_component_fraction",
                "max_rejected_component_fraction",
                "arc_to_arc_handoff_success_fraction",
                "median_handoff_initial_position_error_m",
                "p95_handoff_initial_position_error_m",
                "median_ukf_normalized_mean_nis",
                "median_ukf_innovation_mean_abs_lag1",
                "median_ukf_accepted_update_fraction",
                "runtime_s",
            ]
        )
        for scenario in scenarios:
            final_pos = scenario.final_position_errors_m
            initial_pos = scenario.initial_position_errors_m
            final_vel = np.array([arc.final_velocity_error_mps for arc in scenario.arc_results], dtype=float)
            reduced_chi_square = _scenario_reduced_chi_square(scenario)
            conditions = np.asarray([arc.stats.condition_number for arc in scenario.arc_results], dtype=float)
            iterations = np.asarray([arc.stats.iterations for arc in scenario.arc_results], dtype=float)
            active_weight = np.asarray([arc.stats.active_weight_fraction for arc in scenario.arc_results], dtype=float)
            rejected_fraction = _scenario_rejected_component_fraction(scenario)
            handoff_success, handoff_initial = _handoff_metrics(scenario, cold_bank)
            writer.writerow(
                [
                    scenario.label,
                    scenario.estimator_type,
                    scenario.start_mode,
                    len(scenario.arc_results),
                    scenario.algorithmic_success_fraction,
                    scenario.operational_success_fraction,
                    _safe_percentile(initial_pos, 50),
                    _safe_percentile(initial_pos, 95),
                    _safe_percentile(final_pos, 50),
                    _safe_percentile(final_pos, 95),
                    _safe_max(final_pos),
                    _safe_percentile(final_vel, 50),
                    _safe_percentile(final_vel, 95),
                    _safe_max(final_vel),
                    _nanmedian_arc_stat(scenario, "final_cost"),
                    _safe_percentile(reduced_chi_square, 50),
                    _safe_percentile(reduced_chi_square, 95),
                    _safe_percentile(conditions, 50),
                    _safe_percentile(conditions, 95),
                    _safe_max(conditions),
                    _safe_percentile(iterations, 50),
                    _safe_percentile(iterations, 95),
                    _safe_max(iterations),
                    _safe_percentile(active_weight, 50),
                    _safe_percentile(rejected_fraction, 50),
                    _safe_max(rejected_fraction),
                    handoff_success,
                    _safe_percentile(handoff_initial, 50),
                    _safe_percentile(handoff_initial, 95),
                    _nanmedian_arc_attr(scenario, "ukf_normalized_mean_nis"),
                    _nanmedian_arc_attr(scenario, "ukf_innovation_mean_abs_lag1"),
                    _nanmedian_arc_attr(scenario, "ukf_accepted_update_fraction"),
                    float(runtimes[scenario.label]),
                ]
            )
    return output_path


def _scenario_reduced_chi_square(scenario) -> np.ndarray:
    values = []
    for arc in scenario.arc_results:
        measurement_dimension = 3 if scenario.measurement_type == "position" else 4
        num_components = int(arc.num_observations) * measurement_dimension
        num_solve_for = 6 + int(np.asarray(arc.estimated_bias).size)
        dof = num_components - num_solve_for
        if dof > 0 and np.isfinite(arc.stats.final_cost):
            values.append(float(arc.stats.final_cost) / float(dof))
        else:
            values.append(float("nan"))
    return np.asarray(values, dtype=float)


def _scenario_rejected_component_fraction(scenario) -> np.ndarray:
    values = []
    for arc in scenario.arc_results:
        measurement_dimension = 3 if scenario.measurement_type == "position" else 4
        num_components = int(arc.num_observations) * measurement_dimension
        if num_components > 0:
            values.append(float(arc.stats.rejected_components) / float(num_components))
        else:
            values.append(float("nan"))
    return np.asarray(values, dtype=float)


def _handoff_metrics(scenario, cold_bank) -> tuple[float, np.ndarray]:
    if scenario.start_mode == "cold" or len(scenario.arc_results) <= 1:
        return float("nan"), np.asarray([], dtype=float)
    cold_position_errors = np.asarray([np.linalg.norm(np.asarray(item, dtype=float).reshape(6)[:3]) for item in cold_bank])
    handoff_initial = np.asarray([arc.initial_position_error_m for arc in scenario.arc_results[1:]], dtype=float)
    usable = min(handoff_initial.size, max(cold_position_errors.size - 1, 0))
    if usable == 0:
        return float("nan"), handoff_initial
    improved = handoff_initial[:usable] <= cold_position_errors[1 : 1 + usable]
    operational = np.asarray([arc.operational_success for arc in scenario.arc_results[1 : 1 + usable]], dtype=bool)
    return float(np.mean(improved & operational)), handoff_initial


def _nanmedian_arc_stat(scenario, field: str) -> float:
    values = np.asarray([getattr(arc.stats, field) for arc in scenario.arc_results], dtype=float)
    return _nanmedian(values)


def _nanmedian_arc_attr(scenario, field: str) -> float:
    values = np.asarray([getattr(arc, field) for arc in scenario.arc_results], dtype=float)
    return _nanmedian(values)


def _nanmedian(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _safe_percentile(values, percentile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, percentile))


def _safe_max(values) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def _plot_runtime_success_consistency(scenarios, runtimes: dict[str, float], output_path: Path, *, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [scenario.label for scenario in scenarios]
    x = np.arange(len(labels))
    runtime = np.asarray([runtimes[scenario.label] for scenario in scenarios], dtype=float)
    success = np.asarray([scenario.operational_success_fraction for scenario in scenarios], dtype=float)
    consistency = np.asarray(
        [
            _nanmedian_arc_attr(scenario, "ukf_normalized_mean_nis")
            if scenario.estimator_type == "ukf"
            else _safe_percentile(_scenario_reduced_chi_square(scenario), 50)
            for scenario in scenarios
        ],
        dtype=float,
    )
    conditions = np.asarray(
        [
            _safe_percentile([arc.stats.condition_number for arc in scenario.arc_results], 50)
            for scenario in scenarios
        ],
        dtype=float,
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.8), constrained_layout=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, values, ylabel, panel_title, logy in (
        (axes[0, 0], runtime, "s", "Runtime vs scenario", True),
        (axes[0, 1], consistency, "normalized statistic", "NIS / residual consistency", False),
        (axes[1, 0], success, "fraction", "Operational success fraction", False),
        (axes[1, 1], conditions, "condition number", "Median covariance/normal condition", True),
    ):
        ax.bar(x, values, color="#2563eb")
        ax.set_title(panel_title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right")
        if logy:
            ax.set_yscale("log")
        if panel_title == "NIS / residual consistency":
            ax.axhline(1.0, color="#dc2626", linestyle="--", linewidth=1.2, label="ideal ~1")
            ax.legend(fontsize=8)
        if panel_title == "Operational success fraction":
            ax.set_ylim(0.0, 1.05)
        ax.grid(True, axis="y", which="both", alpha=0.25)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_error_summary(scenarios, output_path: Path, *, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [scenario.label for scenario in scenarios]
    x = np.arange(len(labels))
    width = 0.24
    final_pos = [scenario.final_position_errors_m for scenario in scenarios]
    final_vel = [
        np.asarray([arc.final_velocity_error_mps for arc in scenario.arc_results], dtype=float)
        for scenario in scenarios
    ]
    pos_median = np.asarray([_safe_percentile(values, 50) for values in final_pos], dtype=float)
    pos_p95 = np.asarray([_safe_percentile(values, 95) for values in final_pos], dtype=float)
    pos_max = np.asarray([_safe_max(values) for values in final_pos], dtype=float)
    vel_median = np.asarray([_safe_percentile(values, 50) for values in final_vel], dtype=float)
    vel_p95 = np.asarray([_safe_percentile(values, 95) for values in final_vel], dtype=float)
    vel_max = np.asarray([_safe_max(values) for values in final_vel], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), constrained_layout=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    for ax, median, p95, max_values, ylabel, panel_title in (
        (axes[0], pos_median, pos_p95, pos_max, "m", "Final position error"),
        (axes[1], vel_median, vel_p95, vel_max, "m/s", "Final velocity error"),
    ):
        ax.bar(x - width, median, width=width, label="median", color="#2563eb")
        ax.bar(x, p95, width=width, label="p95", color="#f59e0b")
        ax.bar(x + width, max_values, width=width, label="max", color="#dc2626")
        ax.set_title(panel_title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_yscale("log")
        ax.grid(True, axis="y", which="both", alpha=0.25)
        ax.legend(fontsize=8)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _filename_suffix(args: argparse.Namespace) -> str:
    days = str(args.duration_days).replace(".", "p")
    step = str(int(args.sample_step_s))
    arc_h = str(args.arc_duration_h).replace(".", "p")
    stride_h = str(args.arc_stride_h).replace(".", "p")
    noise = "noisy" if args.measurement_noise else "clean"
    modes = "-".join(args.start_modes)
    source = "vis" if args.arc_source == "spice_visibility" else "regular"
    base = f"{source}_{days}d_step{step}s_arc{arc_h}h_stride{stride_h}h_{noise}_{modes}"
    if args.max_arcs is None:
        return base
    return f"{base}_max{args.max_arcs}"


def _noise_label(args: argparse.Namespace) -> str:
    if not args.measurement_noise:
        return "clean"
    return f"Gaussian range={args.sigma_range_m:g} m, angle={args.sigma_angle_rad:g} rad"


if __name__ == "__main__":
    main()
