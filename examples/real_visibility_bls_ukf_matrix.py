"""Run a real-visibility BLS-vs-SR-UKF comparison matrix.

This is the fragmented-tracking counterpart of the synthetic baseline:

- SPICE-backed Moon-centered truth propagation
- elevation mask and lunar occultation visibility
- single-station vs DSN-like network
- gap stitching off/on
- offline BLS-LM vs sequential SR-UKF over identical arcs

Run from the project root:

    python python_port/examples/real_visibility_bls_ukf_matrix.py --duration-days 7 --measurement-noise
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lunar_od import (  # noqa: E402
    Station,
    UKFAdaptiveConfig,
    UnscentedTransformConfig,
    VisibilityConfig,
    analyze_visibility_gap_with_transforms,
    build_measurement_arcs,
    chi_square_nis_gate,
    load_spice_kernels,
    make_cold_start_bank,
    position_only_stations,
    propagate_truth_with_ephemeris,
    run_batch_arc_sequence,
    sample_j2000_to_itrf93_transforms,
    sample_moon_centered_ephemeris,
    write_scenario_summary_csv,
    write_visibility_summary_csv,
)

from baseline_bls_ukf_comparison import (  # noqa: E402
    _plot_error_summary,
    _plot_runtime_success_consistency,
    _safe_max,
    _safe_percentile,
    _scenario_reduced_chi_square,
)


def main() -> None:
    args = _parse_args()
    context = _build_spice_context(args)
    out_dir = Path("python_port") / "results" / "real_visibility_bls_ukf"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = []
    runtimes: dict[str, float] = {}
    visibility_rows = []
    for network_name, stations in _network_cases(args):
        for gap_label, max_gap_s in _gap_cases(args):
            case_label = f"{network_name}_{gap_label}"
            print(f"Building visibility case {case_label}...")
            arcs, visibility_csv = _build_case_arcs(args, context, stations, case_label, max_gap_s, out_dir)
            visibility_rows.append(_visibility_case_row(case_label, network_name, gap_label, max_gap_s, arcs, visibility_csv))
            if args.max_arcs_per_case is not None:
                arcs = _select_representative_arcs(arcs, args.max_arcs_per_case)
            if not arcs:
                print(f"Skipping {case_label}: no OD-ready arcs.")
                continue

            cold_bank = make_cold_start_bank(
                len(arcs),
                sigma_pos_m=args.cold_sigma_pos_m,
                sigma_vel_mps=args.cold_sigma_vel_mps,
                seed=args.seed + _stable_case_offset(case_label),
            )
            for estimator_type in ("bls_lm", "ukf"):
                label = f"{case_label} {estimator_type.upper()} {args.start_mode}"
                print(f"Running {label} on {len(arcs)} arcs...")
                start = perf_counter()
                scenario = run_batch_arc_sequence(
                    arcs,
                    "position",
                    args.start_mode,
                    estimator_type,
                    context["mu_moon"],
                    context["mu_earth"],
                    context["mu_sun"],
                    context["earth_position"],
                    context["sun_position"],
                    cold_start_bank=cold_bank,
                    label=label,
                    max_iter=args.max_iter,
                    rtol=args.rtol,
                    atol=args.atol,
                    process_noise_covariance=_process_noise(args, estimator_type),
                    ukf_transform_config=UnscentedTransformConfig(alpha=args.ukf_alpha) if estimator_type == "ukf" else None,
                    ukf_adaptive_config=_ukf_adaptive_config(args) if estimator_type == "ukf" else None,
                    ukf_covariance_form="square_root",
                )
                runtimes[scenario.label] = perf_counter() - start
                scenarios.append(scenario)

    suffix = _suffix(args)
    detail_csv = write_scenario_summary_csv(scenarios, out_dir / f"real_visibility_{suffix}_arc_summary.csv")
    aggregate_csv = _write_aggregate_csv(scenarios, runtimes, out_dir / f"real_visibility_{suffix}_aggregate.csv")
    visibility_csv = _write_visibility_cases_csv(visibility_rows, out_dir / f"real_visibility_{suffix}_visibility_cases.csv")
    runtime_png = _plot_runtime_success_consistency(
        scenarios,
        runtimes,
        out_dir / f"real_visibility_{suffix}_runtime_success_consistency.png",
        title=f"Real visibility BLS-vs-SR-UKF ({args.duration_days:g} days)",
    )
    error_png = _plot_error_summary(
        scenarios,
        out_dir / f"real_visibility_{suffix}_median_p95_max_errors.png",
        title=f"Real visibility median / p95 / max error ({args.duration_days:g} days)",
    )

    print(f"Wrote {aggregate_csv}")
    print(f"Wrote {detail_csv}")
    print(f"Wrote {visibility_csv}")
    print(f"Wrote {runtime_png}")
    print(f"Wrote {error_png}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-days", type=float, default=7.0)
    parser.add_argument("--sample-step-s", type=float, default=900.0)
    parser.add_argument("--ephem-step-s", type=float, default=3600.0)
    parser.add_argument("--measurement-noise", action="store_true")
    parser.add_argument("--measurement-seed", type=int, default=20260610)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--sigma-range-m", type=float, default=5.0)
    parser.add_argument("--sigma-angle-rad", type=float, default=1e-5)
    parser.add_argument("--networks", nargs="+", choices=("single_itu", "dsn3", "dsn4_itu", "all"), default=("single_itu", "dsn3"))
    parser.add_argument("--gap-modes", nargs="+", choices=("off", "on"), default=("off", "on"))
    parser.add_argument("--max-gap-s", type=float, default=1800.0)
    parser.add_argument("--min-elevation-deg", type=float, default=5.0)
    parser.add_argument("--min-visibility-samples", type=int, default=4)
    parser.add_argument("--max-arcs-per-case", type=int)
    parser.add_argument("--start-mode", choices=("cold", "hot", "formal"), default="formal")
    parser.add_argument("--cold-sigma-pos-m", type=float, default=150.0)
    parser.add_argument("--cold-sigma-vel-mps", type=float, default=0.05)
    parser.add_argument("--handoff-sigma-pos-m", type=float, default=0.1)
    parser.add_argument("--handoff-sigma-vel-mps", type=float, default=1e-4)
    parser.add_argument("--strict-outlier-gate", action="store_true")
    parser.add_argument("--outlier-gate-sigma", type=float, default=3.0)
    parser.add_argument("--ukf-alpha", type=float, default=0.35)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-11)
    return parser.parse_args()


def _build_spice_context(args: argparse.Namespace) -> dict:
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
        print(f"Sampling SPICE ephemeris and transforms for {duration_h:.1f} h...")
        ephemeris = sample_moon_centered_ephemeris(et0, t_ephem_s)
        xforms = sample_j2000_to_itrf93_transforms(et0, t_eval_s)
        print(f"Propagating truth at {t_eval_s.size} samples...")
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
    finally:
        spice.kclear()

    return {
        "initial": initial,
        "epoch_utc": epoch_utc,
        "et0": et0,
        "t_eval_s": t_eval_s,
        "state_history": state_history,
        "xforms": xforms,
        "earth_position": ephemeris.earth_position,
        "earth_velocity": ephemeris.earth_velocity,
        "sun_position": ephemeris.sun_position,
        "mu_moon": mu_moon,
        "mu_earth": mu_earth,
        "mu_sun": mu_sun,
    }


def _network_cases(args: argparse.Namespace):
    by_name = {station.name: station for station in position_only_stations()}
    definitions = {
        "single_itu": ("ITU Ayazaga",),
        "dsn3": ("Goldstone DSN", "Madrid DSN", "Canberra DSN"),
        "dsn4_itu": ("ITU Ayazaga", "Goldstone DSN", "Madrid DSN", "Canberra DSN"),
        "all": tuple(by_name),
    }
    for network in args.networks:
        yield network, tuple(_station_with_noise(by_name[name], args) for name in definitions[network])


def _station_with_noise(station: Station, args: argparse.Namespace) -> Station:
    return Station(
        name=station.name,
        lat_deg=station.lat_deg,
        lon_deg=station.lon_deg,
        alt_m=station.alt_m,
        color_rgb=station.color_rgb,
        sigma_range_m=args.sigma_range_m,
        sigma_angle_rad=args.sigma_angle_rad,
        sigma_range_rate_mps=station.sigma_range_rate_mps,
        bias=station.bias,
    )


def _gap_cases(args: argparse.Namespace):
    for mode in args.gap_modes:
        yield mode, 0.0 if mode == "off" else float(args.max_gap_s)


def _build_case_arcs(args, context, stations, case_label: str, max_gap_s: float, out_dir: Path):
    config = VisibilityConfig(
        r_moon_mean_m=float(context["initial"]["r_moon_mean_m"]),
        earth_rotation_rad_s=7.292115e-5,
        epoch_utc=context["epoch_utc"],
        min_elevation_deg=args.min_elevation_deg,
    )
    seg_starts, seg_ends, vis_mask_raw, net_vis_filled = analyze_visibility_gap_with_transforms(
        context["t_eval_s"],
        context["state_history"],
        stations,
        context["earth_position"],
        context["xforms"],
        max_gap_s,
        config,
    )
    visibility_csv = write_visibility_summary_csv(
        context["t_eval_s"],
        [station.name for station in stations],
        vis_mask_raw,
        net_vis_filled,
        seg_starts,
        seg_ends,
        out_dir / f"real_visibility_{case_label}_visibility.csv",
    )
    import spiceypy as spice

    load_spice_kernels()
    try:
        arcs = build_measurement_arcs(
            "position",
            context["t_eval_s"],
            context["state_history"],
            seg_starts,
            seg_ends,
            vis_mask_raw,
            stations,
            context["earth_position"],
            context["earth_velocity"],
            context["et0"],
            noise=args.measurement_noise,
            rng=np.random.default_rng(args.measurement_seed + _stable_case_offset(case_label)),
            min_samples=args.min_visibility_samples,
        )
    finally:
        spice.kclear()
    return arcs, visibility_csv


def _select_representative_arcs(arcs, max_arcs: int):
    arcs = tuple(arcs)
    if len(arcs) <= max_arcs:
        return arcs
    order = np.argsort([arc.t_pass_s[-1] - arc.t_pass_s[0] for arc in arcs])[::-1]
    selected = sorted(order[:max_arcs])
    return tuple(arcs[int(idx)] for idx in selected)


def _process_noise(args: argparse.Namespace, estimator_type: str) -> np.ndarray | None:
    if estimator_type == "ukf":
        return np.diag([0.01**2] * 3 + [1e-5**2] * 3)
    if args.start_mode == "formal":
        return np.diag([args.handoff_sigma_pos_m**2] * 3 + [args.handoff_sigma_vel_mps**2] * 3)
    return None


def _ukf_adaptive_config(args: argparse.Namespace) -> UKFAdaptiveConfig:
    if not args.strict_outlier_gate:
        return UKFAdaptiveConfig()
    return UKFAdaptiveConfig(
        nis_gate=chi_square_nis_gate(3, sigma=args.outlier_gate_sigma),
        component_nis_gate=args.outlier_gate_sigma**2,
        component_gate_mode="conditional",
    )


def _visibility_case_row(case_label, network_name, gap_label, max_gap_s, arcs, visibility_csv):
    durations_h = np.asarray([(arc.t_pass_s[-1] - arc.t_pass_s[0]) / 3600.0 for arc in arcs], dtype=float)
    observations = np.asarray([arc.obs_data.shape[0] for arc in arcs], dtype=float)
    return {
        "case": case_label,
        "network": network_name,
        "gap_mode": gap_label,
        "max_gap_s": float(max_gap_s),
        "num_od_ready_arcs": len(arcs),
        "median_arc_duration_h": _safe_percentile(durations_h, 50),
        "p95_arc_duration_h": _safe_percentile(durations_h, 95),
        "max_arc_duration_h": _safe_max(durations_h),
        "median_observations_per_arc": _safe_percentile(observations, 50),
        "visibility_csv": str(visibility_csv),
    }


def _write_visibility_cases_csv(rows: list[dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _write_aggregate_csv(scenarios, runtimes: dict[str, float], output_path: Path) -> Path:
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
                "median_final_position_error_m",
                "p95_final_position_error_m",
                "max_final_position_error_m",
                "median_final_velocity_error_mps",
                "p95_final_velocity_error_mps",
                "median_consistency",
                "median_condition_number",
                "p95_condition_number",
                "median_iteration_or_update_count",
                "median_accepted_update_fraction",
                "median_rejected_component_fraction",
                "runtime_s",
            ]
        )
        for scenario in scenarios:
            final_pos = scenario.final_position_errors_m
            final_vel = np.asarray([arc.final_velocity_error_mps for arc in scenario.arc_results], dtype=float)
            reduced_chi_square = _scenario_reduced_chi_square(scenario)
            consistency = (
                _safe_percentile([arc.ukf_normalized_mean_nis for arc in scenario.arc_results], 50)
                if scenario.estimator_type == "ukf"
                else _safe_percentile(reduced_chi_square, 50)
            )
            conditions = np.asarray([arc.stats.condition_number for arc in scenario.arc_results], dtype=float)
            iterations = np.asarray([arc.stats.iterations for arc in scenario.arc_results], dtype=float)
            rejected = _rejected_fraction(scenario)
            writer.writerow(
                [
                    scenario.label,
                    scenario.estimator_type,
                    scenario.start_mode,
                    len(scenario.arc_results),
                    scenario.algorithmic_success_fraction,
                    scenario.operational_success_fraction,
                    _safe_percentile(final_pos, 50),
                    _safe_percentile(final_pos, 95),
                    _safe_max(final_pos),
                    _safe_percentile(final_vel, 50),
                    _safe_percentile(final_vel, 95),
                    consistency,
                    _safe_percentile(conditions, 50),
                    _safe_percentile(conditions, 95),
                    _safe_percentile(iterations, 50),
                    _safe_percentile([arc.ukf_accepted_update_fraction for arc in scenario.arc_results], 50),
                    _safe_percentile(rejected, 50),
                    runtimes[scenario.label],
                ]
            )
    return output_path


def _rejected_fraction(scenario) -> np.ndarray:
    values = []
    for arc in scenario.arc_results:
        components = int(arc.num_observations) * (3 if scenario.measurement_type == "position" else 4)
        values.append(float("nan") if components <= 0 else float(arc.stats.rejected_components) / float(components))
    return np.asarray(values, dtype=float)


def _stable_case_offset(label: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(label))


def _suffix(args: argparse.Namespace) -> str:
    days = str(args.duration_days).replace(".", "p")
    networks = "-".join(args.networks)
    gaps = "-".join(args.gap_modes)
    noise = "noisy" if args.measurement_noise else "clean"
    suffix = f"{days}d_step{int(args.sample_step_s)}s_{networks}_{gaps}_{noise}_{args.start_mode}"
    if args.max_arcs_per_case is not None:
        suffix += f"_max{args.max_arcs_per_case}"
    if args.strict_outlier_gate:
        suffix += "_strictgate"
    return suffix


if __name__ == "__main__":
    main()
