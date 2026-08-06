"""Plotting and tabular reporting helpers for Lunar OD scenarios."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from .diagnostics import analyze_convergence, analyze_state_bias_correlation
from .force_contract import manifest_canonical_bytes, manifest_sha256
from .radiometrics import COUNTED_DOPPLER_MODEL_VERSION

# Owner Addendum 03 Decision 4: the R3 measurement-provenance columns are a
# separate, append-only segment. They are never folded into R1_COLUMNS and
# never inserted before the R1 segment.
R3_MEASUREMENT_PROVENANCE_COLUMNS = (
    "station_state_method",
    "counted_doppler_model_version",
)
from .scenarios import ScenarioResult


def write_force_model_manifest(manifest, output_path) -> Path:
    """Write the canonical force-model manifest bytes for one run bundle.

    Written from the canonical encoding, so the same run inputs produce the
    same bytes (and therefore the same SHA-256) in any process or working
    directory. The manifest never carries an absolute path; scientific
    identity comes from its own hash, not from the file name.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest_canonical_bytes(manifest)
    # Atomic write: fully materialize a sibling temp file, then replace, so a
    # reader never observes a partial manifest and the on-disk bytes are exactly
    # the canonical bytes the SHA is computed from.
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    tmp_path.write_bytes(payload)
    os.replace(tmp_path, output_path)
    return output_path


def force_model_manifest_sha256(manifest) -> str:
    """Return ``sha256:<hex>`` of the canonical manifest bytes."""
    return manifest_sha256(manifest)


def write_scenario_summary_csv(scenarios: Sequence[ScenarioResult], output_path) -> Path:
    """Write per-arc scenario metrics to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scenario",
                "estimator_type",
                "measurement_type",
                "measurement_model_profile",
                "range_rate_physics",
                "companion_geometry",
                "jacobian_model",
                "range_jacobian_model",
                "line_of_sight_jacobian_model",
                "angle_jacobian_model",
                "aberration_jacobian_model",
                "angle_jacobian_matches_full_residual_physics",
                "observer_velocity_epoch",
                "observer_velocity_frame",
                "observer_velocity_reference_center",
                "aberration_local_jacobian_input",
                "aberration_local_jacobian_space",
                "aberration_local_jacobian_step",
                "count_interval_s",
                "start_mode",
                "arc_id",
                "num_observations",
                "initial_position_error_m",
                "initial_velocity_error_mps",
                "final_position_error_m",
                "final_velocity_error_mps",
                "iterations",
                "final_cost",
                "condition_number",
                "rank",
                "rejected_components",
                "active_weight_fraction",
                "ukf_mean_nis",
                "ukf_max_nis",
                "ukf_accepted_update_fraction",
                "ukf_final_process_noise_scale",
                "ukf_innovation_mean_abs_lag1",
                "ukf_innovation_max_abs_lag1",
                "ukf_normalized_mean_nis",
                "ukf_nis_upper_consistent",
                "ukf_elapsed_s",
                "ukf_process_function_evaluations",
                "ukf_unique_dynamic_propagations",
                "ukf_dynamic_propagation_cache_hits",
                "ukf_measurement_function_evaluations",
                "ukf_unique_measurement_model_evaluations",
                "ukf_measurement_model_cache_hits",
                "ukf_frozen_state_indices",
                "ukf_regularized_state_indices",
                "prior_position_sigma_rss_m",
                "posterior_position_sigma_rss_m",
                "prior_velocity_sigma_rss_mps",
                "posterior_velocity_sigma_rss_mps",
                "prior_bias_sigma_rss",
                "posterior_bias_sigma_rss",
                "prior_state_bias_cross_norm",
                "posterior_state_bias_cross_norm",
                "prior_num_bias_states",
                "posterior_num_bias_states",
                "prior_max_state_bias_corr",
                "posterior_max_state_bias_corr",
                "prior_state_rss_sigma",
                "posterior_state_rss_sigma",
                "prior_bias_rss_sigma",
                "posterior_bias_rss_sigma",
                "prior_sqrt_information_condition",
                "posterior_sqrt_information_condition",
                "posterior_sqrt_information_mismatch_rel",
                "stop_reason",
                "convergence_category",
                "converged",
                "converged_by_step_norm",
                "converged_by_cost_stability",
                "max_iter_reached",
                "singular_or_ill_conditioned",
                "rank_deficient",
                "outlier_rejected",
                "finite_final_cost",
                "algorithmic_success",
                "operational_success",
                "operational_category",
                "final_error_acceptable",
                "condition_acceptable",
                "history_domain_dropped_measurements",
                "history_domain_position_drops",
                "history_domain_range_rate_drops",
                "history_domain_required_pre_roll_s",
                "history_domain_required_post_roll_s",
                "history_domain_all_measurement_arcs_empty",
                # R0B-2 force-model provenance (append-only scalars; the
                # canonical payload lives in the side JSON manifest, never here).
                "force_contract_schema_version",
                "truth_force_fingerprint",
                "estimator_force_fingerprint",
                "force_model_match",
                "explicit_force_model_mismatch",
                "force_model_mismatch_reason",
                "force_contract_manifest_sha256",
                "posterior_force_role_status",
                "observability_force_role_status",
                "posterior_covariance_stm_mode",
                "posterior_covariance_rank",
                "posterior_covariance_condition_number",
                "posterior_covariance_min_eigenvalue",
                "posterior_covariance_finite",
                "observability_rank",
                "observability_condition_number",
                "observability_singular_value_min",
                "observability_singular_value_max",
                "observability_finite",
                "condition_number_available",
                "derivative_validation_profile",
                # R3 (appended last, following the R0B/R1 column convention):
                # which station site-state strategy executed, and the
                # counted-Doppler model version that produced the observable.
                *R3_MEASUREMENT_PROVENANCE_COLUMNS,
            ]
        )
        for scenario in scenarios:
            for result in scenario.arc_results:
                convergence = analyze_convergence(
                    result.stop_reason,
                    stats=result.stats,
                    expected_rank=6 + int(np.asarray(result.estimated_bias).size),
                )
                writer.writerow(
                    [
                        scenario.label,
                        scenario.estimator_type,
                        scenario.measurement_type,
                        scenario.measurement_model_profile,
                        scenario.range_rate_physics,
                        scenario.companion_geometry,
                        scenario.jacobian_model,
                        scenario.range_jacobian_model,
                        scenario.line_of_sight_jacobian_model,
                        scenario.angle_jacobian_model,
                        scenario.aberration_jacobian_model,
                        scenario.angle_jacobian_matches_full_residual_physics,
                        scenario.observer_velocity_epoch,
                        scenario.observer_velocity_frame,
                        scenario.observer_velocity_reference_center,
                        scenario.aberration_local_jacobian_input,
                        scenario.aberration_local_jacobian_space,
                        scenario.aberration_local_jacobian_step,
                        scenario.count_interval_s,
                        scenario.start_mode,
                        result.arc_id,
                        result.num_observations,
                        result.initial_position_error_m,
                        result.initial_velocity_error_mps,
                        result.final_position_error_m,
                        result.final_velocity_error_mps,
                        result.stats.iterations,
                        result.stats.final_cost,
                        result.stats.condition_number,
                        result.stats.rank,
                        result.stats.rejected_components,
                        result.stats.active_weight_fraction,
                        result.ukf_mean_nis,
                        result.ukf_max_nis,
                        result.ukf_accepted_update_fraction,
                        result.ukf_final_process_noise_scale,
                        result.ukf_innovation_mean_abs_lag1,
                        result.ukf_innovation_max_abs_lag1,
                        result.ukf_normalized_mean_nis,
                        result.ukf_nis_upper_consistent,
                        result.ukf_elapsed_s,
                        result.ukf_process_function_evaluations,
                        result.ukf_unique_dynamic_propagations,
                        result.ukf_dynamic_propagation_cache_hits,
                        result.ukf_measurement_function_evaluations,
                        result.ukf_unique_measurement_model_evaluations,
                        result.ukf_measurement_model_cache_hits,
                        " ".join(map(str, result.ukf_frozen_state_indices)),
                        " ".join(map(str, result.ukf_regularized_state_indices)),
                        _covariance_rss_sigma(result.prior_covariance, slice(0, 3)),
                        _covariance_rss_sigma(result.posterior_covariance, slice(0, 3)),
                        _covariance_rss_sigma(result.prior_covariance, slice(3, 6)),
                        _covariance_rss_sigma(result.posterior_covariance, slice(3, 6)),
                        _bias_covariance_rss_sigma(result.prior_covariance),
                        _bias_covariance_rss_sigma(result.posterior_covariance),
                        _state_bias_cross_norm(result.prior_covariance),
                        _state_bias_cross_norm(result.posterior_covariance),
                        _state_bias_num_bias(result.prior_covariance),
                        _state_bias_num_bias(result.posterior_covariance),
                        _state_bias_max_abs_correlation(result.prior_covariance),
                        _state_bias_max_abs_correlation(result.posterior_covariance),
                        _state_bias_state_rss_sigma(result.prior_covariance),
                        _state_bias_state_rss_sigma(result.posterior_covariance),
                        _state_bias_bias_rss_sigma(result.prior_covariance),
                        _state_bias_bias_rss_sigma(result.posterior_covariance),
                        _sqrt_information_condition(result.prior_sqrt_information),
                        _sqrt_information_condition(result.posterior_sqrt_information),
                        _sqrt_information_mismatch_rel(
                            result.posterior_sqrt_information,
                            result.stats.posterior_information,
                        ),
                        result.stop_reason,
                        convergence.category,
                        convergence.converged,
                        convergence.converged_by_step_norm,
                        convergence.converged_by_cost_stability,
                        convergence.max_iter_reached,
                        convergence.singular_or_ill_conditioned,
                        convergence.rank_deficient,
                        convergence.outlier_rejected,
                        convergence.finite_final_cost,
                        result.algorithmic_success,
                        result.operational_success,
                        result.operational_category,
                        result.final_error_acceptable,
                        result.condition_acceptable,
                        scenario.history_domain_dropped_measurements,
                        scenario.history_domain_position_drops,
                        scenario.history_domain_range_rate_drops,
                        scenario.history_domain_required_pre_roll_s,
                        scenario.history_domain_required_post_roll_s,
                        scenario.history_domain_all_measurement_arcs_empty,
                        scenario.force_contract_schema_version,
                        scenario.truth_force_fingerprint,
                        scenario.estimator_force_fingerprint,
                        scenario.force_model_match,
                        scenario.explicit_force_model_mismatch,
                        scenario.force_model_mismatch_reason,
                        scenario.force_contract_manifest_sha256,
                        scenario.posterior_force_role_status,
                        scenario.observability_force_role_status,
                        scenario.posterior_covariance_stm_mode,
                        scenario.posterior_covariance_rank,
                        scenario.posterior_covariance_condition_number,
                        scenario.posterior_covariance_min_eigenvalue,
                        scenario.posterior_covariance_finite,
                        scenario.observability_rank,
                        scenario.observability_condition_number,
                        scenario.observability_singular_value_min,
                        scenario.observability_singular_value_max,
                        scenario.observability_finite,
                        result.condition_number_available,
                        scenario.derivative_validation_profile,
                        scenario.station_state_method,
                        COUNTED_DOPPLER_MODEL_VERSION,
                    ]
                )

    return output_path


def plot_scenario_comparison(scenarios: Sequence[ScenarioResult], output_path, *, title: str = "") -> Path:
    """Create a compact cold/hot-start comparison figure."""
    if not scenarios:
        raise ValueError("At least one scenario is required.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=True)
    fig.suptitle(title or "Lunar OD SRIF Scenario Comparison", fontsize=13, fontweight="bold")

    ax_final = axes[0, 0]
    ax_initial = axes[0, 1]
    ax_success = axes[1, 0]
    ax_iters = axes[1, 1]

    palette = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]
    labels = []
    final_medians = []
    initial_medians = []
    improved_values = []

    for idx, scenario in enumerate(scenarios):
        color = palette[idx % len(palette)]
        label = scenario.label or f"{scenario.measurement_type} {scenario.start_mode}"
        labels.append(label)

        arc_ids = np.array([result.arc_id for result in scenario.arc_results], dtype=float)
        final_errors = scenario.final_position_errors_m
        initial_errors = scenario.initial_position_errors_m

        if arc_ids.size:
            ax_final.semilogy(arc_ids, np.maximum(final_errors, 1e-12), marker="o", linewidth=1.8, color=color, label=label)
            ax_initial.semilogy(
                arc_ids,
                np.maximum(initial_errors, 1e-12),
                marker="s",
                linewidth=1.5,
                color=color,
                label=label,
            )

        final_medians.append(_safe_median(final_errors))
        initial_medians.append(_safe_median(initial_errors))
        improved_values.append(_improved_fraction(initial_errors, final_errors))

    ax_final.set_title("Final Position Error")
    ax_final.set_xlabel("Arc")
    ax_final.set_ylabel("m")
    ax_final.grid(True, which="both", alpha=0.25)
    ax_final.legend(fontsize=8)

    ax_initial.set_title("Initial Position Error")
    ax_initial.set_xlabel("Arc")
    ax_initial.set_ylabel("m")
    ax_initial.grid(True, which="both", alpha=0.25)
    ax_initial.legend(fontsize=8)

    x = np.arange(len(labels))
    width = 0.36
    ax_success.bar(x, improved_values, width=0.55, color="#334155", label="improved arc fraction")
    ax_success.set_title("Improved Arc Fraction")
    ax_success.set_xticks(x)
    ax_success.set_xticklabels(labels, rotation=20, ha="right")
    ax_success.set_ylim(0.0, 1.05)
    ax_success.grid(True, axis="y", alpha=0.25)

    ax_iters.bar(x - width / 2, initial_medians, width=width, color="#94a3b8", label="median initial pos err")
    ax_iters.bar(x + width / 2, final_medians, width=width, color="#0f766e", label="median final pos err")
    ax_iters.set_title("Median Initial vs Final Error")
    ax_iters.set_xticks(x)
    ax_iters.set_xticklabels(labels, rotation=20, ha="right")
    ax_iters.set_yscale("log")
    ax_iters.set_ylabel("m")
    ax_iters.grid(True, axis="y", which="both", alpha=0.25)
    ax_iters.legend(fontsize=8)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def write_visibility_summary_csv(
    t_s,
    station_names: Sequence[str],
    vis_mask_raw,
    net_vis_filled,
    seg_starts,
    seg_ends,
    output_path,
) -> Path:
    """Write station/network visibility and arc-duration metrics to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t_s = np.asarray(t_s, dtype=float).reshape(-1)
    vis_mask_raw = _as_bool_mask(vis_mask_raw, len(t_s), len(station_names))
    net_vis_filled = np.asarray(net_vis_filled, dtype=bool).reshape(-1)
    seg_starts = np.asarray(seg_starts, dtype=int).reshape(-1)
    seg_ends = np.asarray(seg_ends, dtype=int).reshape(-1)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "name", "value"])
        for station_idx, station_name in enumerate(station_names):
            writer.writerow(["station_visible_samples", station_name, int(np.sum(vis_mask_raw[:, station_idx]))])
            writer.writerow(["station_visible_fraction", station_name, float(np.mean(vis_mask_raw[:, station_idx]))])

        raw_network = np.any(vis_mask_raw, axis=1)
        writer.writerow(["network_raw_visible_samples", "network", int(np.sum(raw_network))])
        writer.writerow(["network_raw_visible_fraction", "network", float(np.mean(raw_network))])
        writer.writerow(["network_filled_visible_samples", "network", int(np.sum(net_vis_filled))])
        writer.writerow(["network_filled_visible_fraction", "network", float(np.mean(net_vis_filled))])
        writer.writerow(["num_arcs", "network", int(seg_starts.size)])

        writer.writerow([])
        writer.writerow(["arc_id", "start_s", "end_s", "duration_s", "raw_measurement_samples"])
        for arc_id, (start_idx, end_idx) in enumerate(zip(seg_starts, seg_ends), start=1):
            duration_s = float(t_s[end_idx] - t_s[start_idx]) if end_idx > start_idx else 0.0
            raw_samples = int(np.sum(raw_network[start_idx : end_idx + 1]))
            writer.writerow([arc_id, float(t_s[start_idx]), float(t_s[end_idx]), duration_s, raw_samples])

    return output_path


def plot_visibility_analysis(
    t_s,
    station_names: Sequence[str],
    vis_mask_raw,
    net_vis_filled,
    seg_starts,
    seg_ends,
    output_path,
    *,
    title: str = "",
) -> Path:
    """Plot station-level visibility, network visibility, and arc durations."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t_s = np.asarray(t_s, dtype=float).reshape(-1)
    station_names = tuple(station_names)
    vis_mask_raw = _as_bool_mask(vis_mask_raw, len(t_s), len(station_names))
    net_vis_filled = np.asarray(net_vis_filled, dtype=bool).reshape(-1)
    seg_starts = np.asarray(seg_starts, dtype=int).reshape(-1)
    seg_ends = np.asarray(seg_ends, dtype=int).reshape(-1)
    raw_network = np.any(vis_mask_raw, axis=1)
    t_hours = t_s / 3600.0

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11.0, 7.6),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.2, 1.2, 1.3]},
    )
    fig.suptitle(title or "Visibility Analysis", fontsize=13, fontweight="bold")

    ax_mask, ax_network, ax_arcs = axes

    if t_hours.size > 1:
        extent = [float(t_hours[0]), float(t_hours[-1]), -0.5, len(station_names) - 0.5]
    else:
        extent = [0.0, 1.0, -0.5, len(station_names) - 0.5]
    ax_mask.imshow(
        vis_mask_raw.T.astype(float),
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=extent,
        cmap=ListedColormap(["#f8fafc", "#2563eb"]),
    )
    ax_mask.set_yticks(np.arange(len(station_names)))
    ax_mask.set_yticklabels(station_names)
    ax_mask.set_title("Raw Station Visibility")
    ax_mask.set_ylabel("Station")
    ax_mask.grid(False)

    ax_network.step(t_hours, raw_network.astype(int), where="post", color="#64748b", linewidth=1.5, label="raw network")
    ax_network.step(
        t_hours,
        net_vis_filled.astype(int),
        where="post",
        color="#dc2626",
        linewidth=1.8,
        label="gap-filled network",
    )
    for start_idx, end_idx in zip(seg_starts, seg_ends):
        ax_network.axvspan(t_hours[start_idx], t_hours[end_idx], color="#fde68a", alpha=0.25)
    ax_network.set_title("Network Visibility")
    ax_network.set_ylabel("visible")
    ax_network.set_ylim(-0.08, 1.15)
    ax_network.set_yticks([0, 1])
    ax_network.grid(True, axis="x", alpha=0.25)
    ax_network.legend(fontsize=8, loc="upper right")

    arc_ids = np.arange(1, seg_starts.size + 1)
    durations_min = np.array(
        [
            (float(t_s[end_idx]) - float(t_s[start_idx])) / 60.0 if end_idx > start_idx else 0.0
            for start_idx, end_idx in zip(seg_starts, seg_ends)
        ],
        dtype=float,
    )
    ax_arcs.bar(arc_ids, durations_min, color="#0f766e")
    ax_arcs.set_title("Gap-Filled Arc Durations")
    ax_arcs.set_xlabel("Arc")
    ax_arcs.set_ylabel("min")
    ax_arcs.grid(True, axis="y", alpha=0.25)

    for ax in axes[:2]:
        ax.set_xlabel("Time since epoch [h]")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _safe_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def _improved_fraction(initial_errors: np.ndarray, final_errors: np.ndarray) -> float:
    initial_errors = np.asarray(initial_errors, dtype=float)
    final_errors = np.asarray(final_errors, dtype=float)
    if initial_errors.size == 0:
        return 0.0
    return float(np.mean(final_errors < initial_errors))


def _covariance_rss_sigma(covariance, component_slice: slice) -> float:
    if covariance is None:
        return float("nan")
    cov = np.asarray(covariance, dtype=float)
    if cov.shape[0] < component_slice.stop:
        return float("nan")
    diag = np.clip(np.diag(cov)[component_slice], 0.0, None)
    return float(np.sqrt(np.sum(diag)))


def _bias_covariance_rss_sigma(covariance) -> float:
    if covariance is None:
        return float("nan")
    cov = np.asarray(covariance, dtype=float)
    if cov.shape[0] <= 6:
        return float("nan")
    diag = np.clip(np.diag(cov)[6:], 0.0, None)
    return float(np.sqrt(np.sum(diag)))


def _state_bias_cross_norm(covariance) -> float:
    if covariance is None:
        return float("nan")
    cov = np.asarray(covariance, dtype=float)
    if cov.shape[0] <= 6:
        return float("nan")
    return float(np.linalg.norm(cov[:6, 6:], ord="fro"))


def _state_bias_num_bias(covariance) -> int:
    diagnostics = _state_bias_correlation_diagnostics(covariance)
    return 0 if diagnostics is None else diagnostics.num_bias


def _state_bias_max_abs_correlation(covariance) -> float:
    diagnostics = _state_bias_correlation_diagnostics(covariance)
    return float("nan") if diagnostics is None else diagnostics.max_abs_correlation


def _state_bias_state_rss_sigma(covariance) -> float:
    diagnostics = _state_bias_correlation_diagnostics(covariance)
    return float("nan") if diagnostics is None else diagnostics.state_rss_sigma


def _state_bias_bias_rss_sigma(covariance) -> float:
    diagnostics = _state_bias_correlation_diagnostics(covariance)
    return float("nan") if diagnostics is None else diagnostics.bias_rss_sigma


def _state_bias_correlation_diagnostics(covariance):
    if covariance is None:
        return None
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] <= 6:
        return None
    return analyze_state_bias_correlation(cov, num_state=6)

def _sqrt_information_condition(sqrt_information) -> float:
    if sqrt_information is None:
        return float("nan")
    return float(np.linalg.cond(np.asarray(sqrt_information, dtype=float)))


def _sqrt_information_mismatch_rel(sqrt_information, information) -> float:
    if sqrt_information is None or information is None:
        return float("nan")
    sqrt_information = np.asarray(sqrt_information, dtype=float)
    information = np.asarray(information, dtype=float)
    rebuilt = sqrt_information.T @ sqrt_information
    denom = max(float(np.linalg.norm(information)), np.finfo(float).eps)
    return float(np.linalg.norm(rebuilt - information) / denom)


def _as_bool_mask(vis_mask_raw, n_steps: int, n_stations: int) -> np.ndarray:
    mask = np.asarray(vis_mask_raw, dtype=bool)
    if mask.ndim == 1:
        mask = mask.reshape(-1, 1)
    if mask.shape != (n_steps, n_stations):
        raise ValueError("vis_mask_raw must have shape (len(t_s), len(station_names)).")
    return mask
