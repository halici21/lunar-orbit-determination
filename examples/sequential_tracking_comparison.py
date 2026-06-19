"""Compare offline BLS arc solutions against online UKF update history.

This report separates two different questions:

- BLS: after an arc closes, what batch solution did the full arc produce?
- UKF: as each visible measurement arrives, how does the sequential estimate move?

The plot places UKF state error as a time history, BLS arc-end solutions as
markers, and visibility arc spans as shaded regions.

Run from the project root:

    python python_port/examples/sequential_tracking_comparison.py --duration-days 1 --measurement-noise
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lunar_od import (  # noqa: E402
    UKFAdaptiveConfig,
    UnscentedTransformConfig,
    chi_square_nis_gate,
    propagate_augmented_state,
    propagate_state,
    run_batch_arc_sequence,
    run_lunar_ukf,
)

from baseline_bls_ukf_comparison import _build_experiment, _cold_start_bank  # noqa: E402


def main() -> None:
    args = _parse_args()
    experiment = _build_experiment(args)
    arcs = experiment["arcs"]
    if args.max_arcs is not None:
        arcs = arcs[: args.max_arcs]
    if not arcs:
        raise RuntimeError("No arcs were produced for the selected configuration.")

    cold_bank = _cold_start_bank(
        len(arcs),
        sigma_pos_m=args.cold_sigma_pos_m,
        sigma_vel_mps=args.cold_sigma_vel_mps,
        seed=args.seed,
    )

    print(f"Running offline BLS over {len(arcs)} arcs...")
    bls_scenario = run_batch_arc_sequence(
        arcs,
        "position",
        args.bls_start_mode,
        "bls_lm",
        experiment["mu_moon"],
        experiment["mu_earth"],
        experiment["mu_sun"],
        experiment["get_earth_pos"],
        experiment["get_sun_pos"],
        cold_start_bank=cold_bank,
        label=f"BLS-LM offline {args.bls_start_mode}",
        max_iter=args.max_iter,
        rtol=args.rtol,
        atol=args.atol,
        process_noise_covariance=np.diag([args.handoff_sigma_pos_m**2] * 3 + [args.handoff_sigma_vel_mps**2] * 3)
        if args.bls_start_mode == "formal"
        else None,
    )
    bls_rows = _bls_arc_rows(args, experiment, arcs, bls_scenario)

    print(f"Running online UKF over {len(arcs)} arcs...")
    ukf_rows = _run_online_ukf(args, experiment, arcs, cold_bank)

    suffix = _suffix(args)
    out_dir = Path("python_port") / "results" / "sequential_tracking"
    ukf_csv = _write_dict_csv(ukf_rows, out_dir / f"sequential_tracking_{suffix}_ukf_updates.csv")
    bls_csv = _write_dict_csv(bls_rows, out_dir / f"sequential_tracking_{suffix}_bls_arc_markers.csv")
    plot_path = _plot_tracking(args, arcs, ukf_rows, bls_rows, out_dir / f"sequential_tracking_{suffix}.png")

    print(f"UKF updates={len(ukf_rows)}, BLS arcs={len(bls_rows)}")
    print(f"Wrote {ukf_csv}")
    print(f"Wrote {bls_csv}")
    print(f"Wrote {plot_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arc-source", choices=("regular", "spice_visibility"), default="regular")
    parser.add_argument("--duration-days", type=float, default=1.0)
    parser.add_argument("--sample-step-s", type=float, default=600.0)
    parser.add_argument("--ephem-step-s", type=float, default=3600.0)
    parser.add_argument("--arc-duration-h", type=float, default=2.0)
    parser.add_argument("--arc-stride-h", type=float, default=6.0)
    parser.add_argument("--max-arcs", type=int)
    parser.add_argument("--measurement-noise", action="store_true")
    parser.add_argument("--measurement-seed", type=int, default=20260610)
    parser.add_argument("--sigma-range-m", type=float, default=5.0)
    parser.add_argument("--sigma-angle-rad", type=float, default=1e-5)
    parser.add_argument("--strict-outlier-gate", action="store_true")
    parser.add_argument("--outlier-gate-sigma", type=float, default=3.0)
    parser.add_argument("--cold-sigma-pos-m", type=float, default=150.0)
    parser.add_argument("--cold-sigma-vel-mps", type=float, default=0.05)
    parser.add_argument("--ukf-initial-pos-sigma-m", type=float)
    parser.add_argument("--ukf-initial-vel-sigma-mps", type=float)
    parser.add_argument("--ukf-process-pos-sigma-m", type=float, default=0.01)
    parser.add_argument("--ukf-process-vel-sigma-mps", type=float, default=1e-5)
    parser.add_argument("--ukf-gap-accel-psd", type=float, default=0.0)
    parser.add_argument("--ukf-gap-covariance-inflation", type=float, default=1.0)
    parser.add_argument("--ukf-alpha", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--bls-start-mode", choices=("cold", "hot", "formal"), default="cold")
    parser.add_argument("--handoff-sigma-pos-m", type=float, default=0.1)
    parser.add_argument("--handoff-sigma-vel-mps", type=float, default=1e-4)
    parser.add_argument("--visibility-network", choices=("dsn4", "itu", "all"), default="dsn4")
    parser.add_argument("--min-elevation-deg", type=float, default=5.0)
    parser.add_argument("--max-gap-s", type=float, default=1800.0)
    parser.add_argument("--min-visibility-samples", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--rtol", type=float, default=1e-11)
    parser.add_argument("--atol", type=float, default=1e-12)
    return parser.parse_args()


def _run_online_ukf(args: argparse.Namespace, experiment: dict, arcs, cold_bank) -> list[dict]:
    rows: list[dict] = []
    previous_state = None
    previous_covariance = None
    previous_time_s = None
    update_counter = 1

    for arc_index, arc in enumerate(arcs):
        if previous_state is None:
            x0 = np.asarray(arc.truth_state_history_mci[0, :6], dtype=float) + np.asarray(cold_bank[arc_index])
            p0 = _default_ukf_covariance(args)
        else:
            x0, p0 = _propagate_state_and_covariance_to(
                previous_state,
                previous_covariance,
                previous_time_s,
                float(arc.t_pass_s[0]),
                experiment,
                args,
            )

        result = run_lunar_ukf(
            arc.t_pass_s,
            arc.obs_data,
            x0,
            p0,
            arc.pass_geo,
            experiment["mu_moon"],
            experiment["mu_earth"],
            experiment["mu_sun"],
            experiment["get_earth_pos"],
            experiment["get_sun_pos"],
            measurement_type="position",
            process_noise=np.diag([args.ukf_process_pos_sigma_m**2] * 3 + [args.ukf_process_vel_sigma_mps**2] * 3),
            covariance_form="square_root",
            adaptive_config=_ukf_adaptive_config(args),
            config=UnscentedTransformConfig(alpha=args.ukf_alpha),
            rtol=args.rtol,
            atol=args.atol,
        )

        for local_idx, obs_idx in enumerate(result.obs_indices):
            obs_row = arc.obs_data[int(obs_idx), :]
            time_index = int(obs_row[5]) - 1
            truth = np.asarray(arc.truth_state_history_mci[time_index, :6], dtype=float)
            state = result.state_estimates[local_idx, :6]
            covariance = result.covariances[local_idx]
            rows.append(
                {
                    "update_id": update_counter,
                    "arc_id": arc.arc_id,
                    "t_s": float(result.t_update_s[local_idx]),
                    "t_h": float(result.t_update_s[local_idx] / 3600.0),
                    "station_id": int(obs_row[4]),
                    "time_index_1based": int(obs_row[5]),
                    "position_error_m": float(np.linalg.norm(state[:3] - truth[:3])),
                    "velocity_error_mps": float(np.linalg.norm(state[3:6] - truth[3:6])),
                    "nis": float(result.normalized_innovation_squared[local_idx]),
                    "normalized_nis": float(result.normalized_innovation_squared[local_idx] / 3.0),
                    "accepted": bool(result.accepted_updates[local_idx]),
                    "accepted_component_fraction": float(np.mean(result.accepted_components[local_idx])),
                    "covariance_condition_number": float(np.linalg.cond(covariance)),
                }
            )
            update_counter += 1

        previous_state = result.final_state[:6].copy()
        previous_covariance = result.final_covariance.copy()
        previous_time_s = float(result.t_update_s[-1])

    return rows


def _ukf_adaptive_config(args: argparse.Namespace) -> UKFAdaptiveConfig:
    if not args.strict_outlier_gate:
        return UKFAdaptiveConfig()
    return UKFAdaptiveConfig(
        nis_gate=chi_square_nis_gate(3, sigma=args.outlier_gate_sigma),
        component_nis_gate=args.outlier_gate_sigma**2,
        component_gate_mode="conditional",
    )


def _bls_arc_rows(args: argparse.Namespace, experiment: dict, arcs, bls_scenario) -> list[dict]:
    rows = []
    by_arc = {result.arc_id: result for result in bls_scenario.arc_results}
    for arc in arcs:
        result = by_arc[arc.arc_id]
        start_t = float(arc.t_pass_s[0])
        end_t = float(arc.t_pass_s[-1])
        estimated_end_state = _propagate_state_to(result.estimated_state, start_t, end_t, experiment, args)
        truth_end_state = np.asarray(arc.truth_state_history_mci[-1, :6], dtype=float)
        rows.append(
            {
                "arc_id": arc.arc_id,
                "start_s": start_t,
                "end_s": end_t,
                "start_h": start_t / 3600.0,
                "end_h": end_t / 3600.0,
                "num_observations": result.num_observations,
                "start_epoch_position_error_m": result.final_position_error_m,
                "arc_end_position_error_m": float(np.linalg.norm(estimated_end_state[:3] - truth_end_state[:3])),
                "arc_end_velocity_error_mps": float(np.linalg.norm(estimated_end_state[3:6] - truth_end_state[3:6])),
                "iterations": result.stats.iterations,
                "condition_number": result.stats.condition_number,
                "stop_reason": result.stop_reason,
                "operational_success": result.operational_success,
            }
        )
    return rows


def _propagate_state_to(state, t0_s: float | None, t1_s: float, experiment: dict, args: argparse.Namespace) -> np.ndarray:
    state = np.asarray(state, dtype=float).reshape(-1)[:6]
    if t0_s is None or np.isclose(float(t0_s), float(t1_s)):
        return state.copy()
    return propagate_state(
        [float(t0_s), float(t1_s)],
        state,
        experiment["mu_moon"],
        experiment["mu_earth"],
        experiment["mu_sun"],
        experiment["get_earth_pos"],
        experiment["get_sun_pos"],
        rtol=args.rtol,
        atol=args.atol,
    )[-1, :]


def _default_ukf_covariance(args: argparse.Namespace) -> np.ndarray:
    pos_sigma = args.ukf_initial_pos_sigma_m if args.ukf_initial_pos_sigma_m is not None else args.cold_sigma_pos_m
    vel_sigma = args.ukf_initial_vel_sigma_mps if args.ukf_initial_vel_sigma_mps is not None else args.cold_sigma_vel_mps
    return np.diag([pos_sigma**2] * 3 + [vel_sigma**2] * 3)


def _propagate_covariance_across_gap(
    covariance: np.ndarray,
    previous_time_s: float | None,
    next_time_s: float,
    args: argparse.Namespace,
) -> np.ndarray:
    """Apply optional stochastic growth across a measurement gap.

    Deterministic covariance mapping through the dynamics is performed by
    `_propagate_state_and_covariance_to`; this helper only adds configured
    inflation or white-acceleration growth.
    """
    p0 = np.asarray(covariance, dtype=float).copy()
    if args.ukf_gap_covariance_inflation != 1.0:
        p0 *= float(args.ukf_gap_covariance_inflation)
    if previous_time_s is None:
        return p0
    dt = max(0.0, float(next_time_s) - float(previous_time_s))
    if dt <= 0.0 or args.ukf_gap_accel_psd <= 0.0:
        return p0
    q = float(args.ukf_gap_accel_psd)
    q_gap = np.zeros((6, 6), dtype=float)
    q_gap[:3, :3] = (dt**3 / 3.0) * q * np.eye(3)
    q_gap[:3, 3:6] = (dt**2 / 2.0) * q * np.eye(3)
    q_gap[3:6, :3] = q_gap[:3, 3:6]
    q_gap[3:6, 3:6] = dt * q * np.eye(3)
    return p0 + q_gap


def _propagate_state_and_covariance_to(
    state: np.ndarray,
    covariance: np.ndarray,
    t0_s: float | None,
    t1_s: float,
    experiment: dict,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(state, dtype=float).reshape(-1)[:6]
    covariance = np.asarray(covariance, dtype=float)
    if t0_s is None or np.isclose(float(t0_s), float(t1_s)):
        p1 = _propagate_covariance_across_gap(covariance, t0_s, t1_s, args)
        return state.copy(), _symmetrize_matrix(p1)

    x_aug0 = np.concatenate([state, np.eye(6).reshape(-1, order="F")])
    x_aug = propagate_augmented_state(
        [float(t0_s), float(t1_s)],
        x_aug0,
        experiment["mu_moon"],
        experiment["mu_earth"],
        experiment["mu_sun"],
        experiment["get_earth_pos"],
        experiment["get_sun_pos"],
        rtol=args.rtol,
        atol=args.atol,
    )
    phi = x_aug[-1, 6:].reshape((6, 6), order="F")
    transition = np.eye(covariance.shape[0], dtype=float)
    transition[:6, :6] = phi
    p1 = transition @ covariance @ transition.T
    p1 = _propagate_covariance_across_gap(p1, t0_s, t1_s, args)
    return x_aug[-1, :6].copy(), _symmetrize_matrix(p1)


def _symmetrize_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def _write_dict_csv(rows: list[dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _plot_tracking(args: argparse.Namespace, arcs, ukf_rows: list[dict], bls_rows: list[dict], output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    t_h = np.asarray([row["t_h"] for row in ukf_rows], dtype=float)
    pos_error = np.asarray([row["position_error_m"] for row in ukf_rows], dtype=float)
    vel_error = np.asarray([row["velocity_error_mps"] for row in ukf_rows], dtype=float)
    nis = np.asarray([row["normalized_nis"] for row in ukf_rows], dtype=float)
    accepted = np.asarray([row["accepted"] for row in ukf_rows], dtype=bool)
    cond = np.asarray([row["covariance_condition_number"] for row in ukf_rows], dtype=float)
    bls_t_h = np.asarray([row["end_h"] for row in bls_rows], dtype=float)
    bls_pos = np.asarray([row["arc_end_position_error_m"] for row in bls_rows], dtype=float)
    bls_vel = np.asarray([row["arc_end_velocity_error_mps"] for row in bls_rows], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(13.5, 11.0), sharex=True, constrained_layout=True)
    source_label = "SPICE visibility" if args.arc_source == "spice_visibility" else "prescribed arcs"
    fig.suptitle(
        f"Time-Sequential Tracking over {args.duration_days:g} Days ({source_label})",
        fontsize=13,
        fontweight="bold",
    )
    for ax in axes:
        for arc in arcs:
            ax.axvspan(float(arc.t_pass_s[0]) / 3600.0, float(arc.t_pass_s[-1]) / 3600.0, color="#dbeafe", alpha=0.35)
        ax.grid(True, which="both", alpha=0.25)

    axes[0].semilogy(t_h, np.maximum(pos_error, 1e-12), color="#2563eb", linewidth=1.6, label="UKF after each update")
    axes[0].semilogy(bls_t_h, np.maximum(bls_pos, 1e-12), "o", color="#dc2626", label="BLS available at arc end")
    axes[0].set_ylabel("position error [m]")
    axes[0].set_title("Online UKF trajectory vs offline BLS arc-end markers")
    axes[0].legend(fontsize=8)

    axes[1].semilogy(t_h, np.maximum(vel_error, 1e-15), color="#0f766e", linewidth=1.6, label="UKF")
    axes[1].semilogy(bls_t_h, np.maximum(bls_vel, 1e-15), "o", color="#ea580c", label="BLS arc-end")
    axes[1].set_ylabel("velocity error [m/s]")
    axes[1].legend(fontsize=8)

    axes[2].plot(t_h[accepted], nis[accepted], ".", color="#16a34a", label="accepted")
    if np.any(~accepted):
        axes[2].plot(t_h[~accepted], nis[~accepted], "x", color="#dc2626", label="gated")
    axes[2].axhline(1.0, color="#334155", linestyle="--", linewidth=1.0, label="ideal normalized NIS")
    axes[2].set_ylabel("normalized NIS")
    axes[2].legend(fontsize=8)

    axes[3].semilogy(t_h, np.maximum(cond, 1.0), color="#7c3aed", linewidth=1.5)
    axes[3].set_ylabel("covariance cond.")
    axes[3].set_xlabel("time [h]")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _suffix(args: argparse.Namespace) -> str:
    source = "vis" if args.arc_source == "spice_visibility" else "regular"
    noise = "noisy" if args.measurement_noise else "clean"
    days = str(args.duration_days).replace(".", "p")
    step = str(int(args.sample_step_s))
    suffix = f"{source}_{days}d_step{step}s_{noise}_{args.bls_start_mode}"
    if args.ukf_gap_accel_psd > 0.0:
        suffix += f"_gapq{args.ukf_gap_accel_psd:g}".replace(".", "p").replace("-", "m")
    if args.ukf_gap_covariance_inflation != 1.0:
        suffix += f"_gapinfl{args.ukf_gap_covariance_inflation:g}".replace(".", "p")
    if args.ukf_initial_pos_sigma_m is not None or args.ukf_initial_vel_sigma_mps is not None:
        pos_sigma = args.ukf_initial_pos_sigma_m if args.ukf_initial_pos_sigma_m is not None else args.cold_sigma_pos_m
        vel_sigma = args.ukf_initial_vel_sigma_mps if args.ukf_initial_vel_sigma_mps is not None else args.cold_sigma_vel_mps
        suffix += f"_p0{pos_sigma:g}m_v{vel_sigma:g}".replace(".", "p")
    if args.ukf_process_pos_sigma_m != 0.01 or args.ukf_process_vel_sigma_mps != 1e-5:
        suffix += f"_qstep{args.ukf_process_pos_sigma_m:g}m_v{args.ukf_process_vel_sigma_mps:g}".replace(".", "p")
    if args.max_arcs is not None:
        suffix += f"_max{args.max_arcs}"
    return suffix


if __name__ == "__main__":
    main()
