"""Scenario runners for arc-by-arc Lunar OD experiments."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .dynamics import make_fast_sigma_propagator, propagate_augmented_state, propagate_state
from .diagnostics import analyze_convergence, analyze_innovation_whiteness
from .estimators import (
    EstimatorStats,
    estimate_position_bls_lm,
    estimate_position_srif,
    estimate_range_rate_bls_lm,
    estimate_range_rate_srif,
    estimate_two_way_range_bls_lm,
    estimate_two_way_range_srif,
)
from .filters import UKFAdaptiveConfig, UnscentedTransformConfig, assess_ukf_operational_stability, run_lunar_ukf
from .history_domain import (
    HistoryDomainDropRecord,
    HistoryDomainError,
    summarize_history_domain_drops,
)
from .measurements import (
    PassGeometry,
    generate_position_measurements,
    generate_range_rate_measurements,
    measurement_model_metadata,
)
from .radiometrics import (
    DEFAULT_STATION_STATE_METHOD,
    RangeRatePhysicsConfig,
    range_rate_physics_config,
)
from .two_way_range import TwoWayRangeConfig, generate_two_way_range_measurements

MeasurementType = Literal["position", "range_rate", "two_way_range"]
StartMode = Literal["cold", "hot", "formal", "sqrt_formal"]
EstimatorType = Literal["srif", "bls_lm", "ukf"]


@dataclass(frozen=True)
class PreparedArc:
    arc_id: int
    start_idx: int
    end_idx: int
    t_pass_s: np.ndarray
    truth_state_history_mci: np.ndarray
    obs_data: np.ndarray
    pass_geo: PassGeometry


@dataclass(frozen=True)
class ArcResult:
    arc_id: int
    start_idx: int
    end_idx: int
    num_observations: int
    initial_position_error_m: float
    initial_velocity_error_mps: float
    final_position_error_m: float
    final_velocity_error_mps: float
    stop_reason: str
    stats: EstimatorStats
    estimated_state: np.ndarray
    estimated_bias: np.ndarray
    observed_station_ids: tuple[int, ...] = ()
    ukf_mean_nis: float = float("nan")
    ukf_max_nis: float = float("nan")
    ukf_accepted_update_fraction: float = float("nan")
    ukf_final_process_noise_scale: float = float("nan")
    ukf_innovation_mean_abs_lag1: float = float("nan")
    ukf_innovation_max_abs_lag1: float = float("nan")
    ukf_normalized_mean_nis: float = float("nan")
    ukf_nis_upper_consistent: bool | None = None
    ukf_elapsed_s: float = float("nan")
    ukf_process_function_evaluations: int = 0
    ukf_unique_dynamic_propagations: int = 0
    ukf_dynamic_propagation_cache_hits: int = 0
    ukf_measurement_function_evaluations: int = 0
    ukf_unique_measurement_model_evaluations: int = 0
    ukf_measurement_model_cache_hits: int = 0
    ukf_frozen_state_indices: tuple[int, ...] = ()
    ukf_regularized_state_indices: tuple[int, ...] = ()
    ukf_stability_passed: bool | None = None
    ukf_min_covariance_eigenvalue: float = float("nan")
    ukf_max_covariance_condition_number: float = float("nan")
    ukf_robust_reweighted_fraction: float = float("nan")
    prior_covariance: np.ndarray | None = None
    posterior_covariance: np.ndarray | None = None
    prior_sqrt_information: np.ndarray | None = None
    posterior_sqrt_information: np.ndarray | None = None

    @property
    def algorithmic_success(self) -> bool:
        convergence = analyze_convergence(
            self.stop_reason,
            stats=self.stats,
            expected_rank=6 + int(np.asarray(self.estimated_bias).size),
        )
        return convergence.converged

    @property
    def final_error_acceptable(self) -> bool:
        return bool(np.isfinite(self.final_position_error_m) and self.final_position_error_m <= 100.0)

    @property
    def condition_acceptable(self) -> bool:
        condition = float(self.stats.condition_number)
        return bool(np.isfinite(condition) and condition <= 1e14)

    @property
    def condition_number_available(self) -> bool:
        """Whether a condition diagnostic was computed, including infinity."""
        return bool(not np.isnan(float(self.stats.condition_number)))

    @property
    def operational_success(self) -> bool:
        convergence = analyze_convergence(
            self.stop_reason,
            stats=self.stats,
            expected_rank=6 + int(np.asarray(self.estimated_bias).size),
        )
        ukf_stable = True if self.ukf_stability_passed is None else bool(self.ukf_stability_passed)
        return bool(
            self.final_error_acceptable
            and self.condition_acceptable
            and convergence.finite_final_cost
            and not convergence.rank_deficient
            and ukf_stable
        )

    @property
    def operational_category(self) -> str:
        if not self.operational_success:
            return "operational_failure"
        if self.algorithmic_success:
            return "accuracy_converged"
        convergence = analyze_convergence(
            self.stop_reason,
            stats=self.stats,
            expected_rank=6 + int(np.asarray(self.estimated_bias).size),
        )
        if convergence.max_iter_reached:
            return "max_iter_acceptable"
        if convergence.singular_or_ill_conditioned:
            return "accuracy_stable"
        return "accuracy_stable"


@dataclass(frozen=True)
class ScenarioResult:
    label: str
    measurement_type: MeasurementType
    start_mode: StartMode
    arc_results: tuple[ArcResult, ...]
    estimator_type: EstimatorType = "srif"
    range_rate_physics: str = "geometric_instantaneous"
    count_interval_s: float = 60.0
    # R3: the station site-state strategy this run actually executed, resolved
    # alongside the range-rate physics so every result records its own method.
    station_state_method: str = DEFAULT_STATION_STATE_METHOD
    measurement_model_profile: str = "geometric_instantaneous"
    companion_geometry: str = "instantaneous"
    jacobian_model: str = "analytic_exact_geometric"
    range_jacobian_model: str = "analytic_exact_geometric"
    line_of_sight_jacobian_model: str = "analytic_exact_geometric"
    angle_jacobian_model: str = "analytic_exact_geometric"
    aberration_jacobian_model: str = "not_applied"
    angle_jacobian_matches_full_residual_physics: bool = True
    observer_velocity_epoch: str = "not_applicable"
    observer_velocity_frame: str = "not_applicable"
    observer_velocity_reference_center: str = "not_applicable"
    aberration_local_jacobian_input: str = "not_applicable"
    aberration_local_jacobian_space: str = "not_applicable"
    aberration_local_jacobian_step: float = float("nan")
    history_domain_dropped_measurements: int = 0
    history_domain_position_drops: int = 0
    history_domain_range_rate_drops: int = 0
    history_domain_required_pre_roll_s: float = 0.0
    history_domain_required_post_roll_s: float = 0.0
    history_domain_all_measurement_arcs_empty: bool = False
    history_domain_drop_records: tuple[dict, ...] = ()
    # R0B-2 force-model provenance (append-only).
    force_contract_schema_version: str = ""
    truth_force_fingerprint: str = ""
    estimator_force_fingerprint: str = ""
    force_model_match: bool = True
    explicit_force_model_mismatch: bool = False
    force_model_mismatch_reason: str = ""
    force_contract_manifest_sha256: str = ""
    posterior_force_role_status: str = ""
    observability_force_role_status: str = ""
    posterior_covariance_stm_mode: str = ""
    posterior_covariance_rank: int = 0
    posterior_covariance_condition_number: float = float("nan")
    posterior_covariance_min_eigenvalue: float = float("nan")
    posterior_covariance_finite: bool = False
    observability_rank: int = 0
    observability_condition_number: float = float("nan")
    observability_singular_value_min: float = float("nan")
    observability_singular_value_max: float = float("nan")
    observability_finite: bool = False
    derivative_validation_profile: str = ""

    @property
    def algorithmic_success_fraction(self) -> float:
        if not self.arc_results:
            return 0.0
        successes = sum(result.algorithmic_success for result in self.arc_results)
        return successes / len(self.arc_results)

    @property
    def operational_success_fraction(self) -> float:
        if not self.arc_results:
            return 0.0
        successes = sum(result.operational_success for result in self.arc_results)
        return successes / len(self.arc_results)

    @property
    def success_fraction(self) -> float:
        return self.operational_success_fraction

    @property
    def final_position_errors_m(self) -> np.ndarray:
        return np.array([result.final_position_error_m for result in self.arc_results], dtype=float)

    @property
    def initial_position_errors_m(self) -> np.ndarray:
        return np.array([result.initial_position_error_m for result in self.arc_results], dtype=float)


def make_cold_start_bank(
    num_arcs: int,
    sigma_pos_m: float,
    sigma_vel_mps: float,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, ...]:
    """Create reproducible 6-state cold-start perturbations."""
    if num_arcs < 0:
        raise ValueError("num_arcs must be non-negative.")
    rng = rng or np.random.default_rng(seed)
    return tuple(
        np.concatenate(
            [
                sigma_pos_m * rng.standard_normal(3),
                sigma_vel_mps * rng.standard_normal(3),
            ]
        )
        for _ in range(num_arcs)
    )


def build_measurement_arcs(
    measurement_type: MeasurementType,
    t_sim_s: ArrayLike,
    state_history_mci: ArrayLike,
    seg_starts: ArrayLike,
    seg_ends: ArrayLike,
    vis_mask_raw: ArrayLike,
    stations: Sequence,
    get_earth_pos: Callable[[ArrayLike], ArrayLike],
    get_earth_vel: Callable[[ArrayLike], ArrayLike],
    et0: float,
    *,
    noise: bool = True,
    rng: np.random.Generator | None = None,
    min_samples: int = 2,
    range_rate_physics: RangeRatePhysicsConfig | str | None = None,
    count_interval_s: float | None = None,
    apply_light_time: bool = False,
    apply_stellar_aberration: bool = False,
    stellar_aberration_model: str = "local_mci",
    measurement_model_profile: str | None = None,
    companion_geometry: str = "instantaneous",
    jacobian_model: str | None = None,
    two_way_range: TwoWayRangeConfig | None = None,
) -> tuple[PreparedArc, ...]:
    """Build per-arc observation packages from visibility segmentation."""
    t_sim_s = np.asarray(t_sim_s, dtype=float).reshape(-1)
    state_history_mci = np.asarray(state_history_mci, dtype=float)
    seg_starts = np.asarray(seg_starts, dtype=int).reshape(-1)
    seg_ends = np.asarray(seg_ends, dtype=int).reshape(-1)
    vis_mask_raw = np.asarray(vis_mask_raw, dtype=bool)

    if seg_starts.shape != seg_ends.shape:
        raise ValueError("seg_starts and seg_ends must have the same shape.")
    if state_history_mci.shape[0] != t_sim_s.size:
        raise ValueError("state_history_mci row count must match t_sim_s.")
    if vis_mask_raw.shape != (t_sim_s.size, len(stations)):
        raise ValueError("vis_mask_raw must have shape (len(t_sim_s), len(stations)).")
    rr_physics = _resolve_range_rate_physics(range_rate_physics, count_interval_s)

    arcs: list[PreparedArc] = []
    history_domain_drop_records: list[HistoryDomainDropRecord] = []
    for arc_number, (start_idx, end_idx) in enumerate(zip(seg_starts, seg_ends), start=1):
        if end_idx < start_idx:
            continue
        idx = np.arange(start_idx, end_idx + 1)
        if idx.size < min_samples:
            continue

        t_pass_s = t_sim_s[idx]
        x_pass = state_history_mci[idx, :]
        vis_pass = vis_mask_raw[idx, :]
        if not np.any(vis_pass):
            continue

        if measurement_type == "position":
            obs_data, pass_geo, _ = generate_position_measurements(
                t_pass_s,
                x_pass,
                stations,
                vis_pass,
                get_earth_pos,
                get_earth_vel,
                et0,
                noise=noise,
                rng=rng,
                arc_id=arc_number,
                apply_light_time=apply_light_time,
                apply_stellar_aberration=apply_stellar_aberration,
                stellar_aberration_model=stellar_aberration_model,
                measurement_model_profile=measurement_model_profile,
                jacobian_model=jacobian_model,
            )
        elif measurement_type == "range_rate":
            obs_data, pass_geo = generate_range_rate_measurements(
                t_pass_s,
                x_pass,
                stations,
                vis_pass,
                get_earth_pos,
                get_earth_vel,
                et0,
                noise=noise,
                rng=rng,
                arc_id=arc_number,
                range_rate_physics=rr_physics,
                companion_geometry=companion_geometry,
                jacobian_model=jacobian_model,
            )
        elif measurement_type == "two_way_range":
            obs_data, pass_geo = generate_two_way_range_measurements(
                t_pass_s,
                x_pass,
                stations,
                vis_pass,
                get_earth_pos,
                get_earth_vel,
                et0,
                noise=noise,
                rng=rng,
                arc_id=arc_number,
                config=two_way_range,
            )
        else:
            raise ValueError(f"Unsupported measurement_type: {measurement_type}")

        pass_metadata = _attached_measurement_metadata(pass_geo)
        history_domain_drop_records.extend(
            _history_domain_records_from_metadata(pass_metadata)
        )
        if obs_data.shape[0] == 0:
            continue
        arcs.append(
            PreparedArc(
                arc_id=arc_number,
                start_idx=int(start_idx),
                end_idx=int(end_idx),
                t_pass_s=t_pass_s,
                truth_state_history_mci=x_pass,
                obs_data=obs_data,
                pass_geo=pass_geo,
            )
        )

    if history_domain_drop_records:
        aggregate = summarize_history_domain_drops(history_domain_drop_records)
        dropped_count = int(aggregate["history_domain_dropped_measurements"])
        aggregate.update(
            {
                "history_domain_position_drops": (
                    dropped_count if measurement_type == "position" else 0
                ),
                "history_domain_range_rate_drops": (
                    dropped_count if measurement_type == "range_rate" else 0
                ),
                "history_domain_all_measurement_arcs_empty": not arcs,
            }
        )
        if not arcs:
            error = HistoryDomainError.from_drop_summary(
                history_domain_drop_records,
                model_context=f"{measurement_type}_measurement_arc_build",
                consumer="build_measurement_arcs",
            )
            error.history_domain_total_candidates = dropped_count
            error.history_domain_dropped_measurements = dropped_count
            error.history_domain_position_drops = int(
                aggregate["history_domain_position_drops"]
            )
            error.history_domain_range_rate_drops = int(
                aggregate["history_domain_range_rate_drops"]
            )
            error.history_domain_required_pre_roll_s = float(
                aggregate["history_domain_required_pre_roll_s"]
            )
            error.history_domain_required_post_roll_s = float(
                aggregate["history_domain_required_post_roll_s"]
            )
            error.history_domain_all_measurement_arcs_empty = True
            raise error
        arcs = [_with_history_domain_aggregate(arc, aggregate) for arc in arcs]

    return tuple(arcs)


def _attached_measurement_metadata(pass_geo: PassGeometry | None) -> dict:
    if pass_geo is None:
        return {}
    if pass_geo.measurement_type == "two_way_range":
        return measurement_model_metadata(pass_geo)
    metadata = measurement_model_metadata(pass_geo)
    if pass_geo.measurement_metadata is not None:
        metadata.update(pass_geo.measurement_metadata)
    return metadata


def _history_domain_records_from_metadata(
    metadata: dict,
) -> tuple[HistoryDomainDropRecord, ...]:
    records = []
    for record in metadata.get("history_domain_drop_records", ()):
        if isinstance(record, HistoryDomainDropRecord):
            records.append(record)
        else:
            records.append(HistoryDomainDropRecord(**dict(record)))
    return tuple(records)


def _with_history_domain_aggregate(
    arc: PreparedArc,
    aggregate: dict,
) -> PreparedArc:
    metadata = _attached_measurement_metadata(arc.pass_geo)
    metadata.update(aggregate)
    return replace(
        arc,
        pass_geo=replace(arc.pass_geo, measurement_metadata=metadata),
    )


def _history_domain_scenario_fields(
    metadata: dict,
    measurement_type: MeasurementType,
) -> dict:
    dropped = int(metadata.get("history_domain_dropped_measurements", 0))
    records = tuple(
        (
            dict(record)
            if not isinstance(record, HistoryDomainDropRecord)
            else record.as_dict()
        )
        for record in metadata.get("history_domain_drop_records", ())
    )
    return {
        "history_domain_dropped_measurements": dropped,
        "history_domain_position_drops": int(
            metadata.get(
                "history_domain_position_drops",
                dropped if measurement_type == "position" else 0,
            )
        ),
        "history_domain_range_rate_drops": int(
            metadata.get(
                "history_domain_range_rate_drops",
                dropped if measurement_type == "range_rate" else 0,
            )
        ),
        "history_domain_required_pre_roll_s": float(
            metadata.get("history_domain_required_pre_roll_s", 0.0)
        ),
        "history_domain_required_post_roll_s": float(
            metadata.get("history_domain_required_post_roll_s", 0.0)
        ),
        "history_domain_all_measurement_arcs_empty": bool(
            metadata.get("history_domain_all_measurement_arcs_empty", False)
        ),
        "history_domain_drop_records": records,
    }


def _resolve_range_rate_physics(
    range_rate_physics: RangeRatePhysicsConfig | str | None,
    count_interval_s: float | None,
) -> RangeRatePhysicsConfig:
    cfg = range_rate_physics_config(range_rate_physics)
    if count_interval_s is None:
        return cfg
    return RangeRatePhysicsConfig(
        mode=cfg.mode,
        count_interval_s=float(count_interval_s),
        uplink_frequency_hz=cfg.uplink_frequency_hz,
        turnaround_ratio=cfg.turnaround_ratio,
        output_unit=cfg.output_unit,
        light_speed_mps=cfg.light_speed_mps,
        light_time_tolerance_s=cfg.light_time_tolerance_s,
        light_time_max_iter=cfg.light_time_max_iter,
        light_time_equation_tolerance_s=cfg.light_time_equation_tolerance_s,
        local_state_model=cfg.local_state_model,
        station_clock_offset_s=cfg.station_clock_offset_s,
        station_clock_drift=cfg.station_clock_drift,
        clock_reference_time_s=cfg.clock_reference_time_s,
        transponder_delay_s=cfg.transponder_delay_s,
    )


def run_srif_arc_sequence(
    prepared_arcs: Sequence[PreparedArc],
    measurement_type: MeasurementType,
    start_mode: StartMode,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    *,
    cold_start_bank: Sequence[ArrayLike] | None = None,
    bias_mode: str | None = None,
    initial_bias: ArrayLike | None = None,
    label: str = "",
    max_iter: int = 40,
    tol_cost_stability: float = 1e-8,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    robust_outlier_rejection: bool = False,
    process_noise_covariance: ArrayLike | None = None,
    parallel: bool = False,
) -> ScenarioResult:
    """Run SRIF over prepared arcs with cold, hot, covariance-formal, or sqrt-formal handoff."""
    return run_batch_arc_sequence(
        prepared_arcs,
        measurement_type,
        start_mode,
        "srif",
        mu_moon_m3_s2,
        mu_earth_m3_s2,
        mu_sun_m3_s2,
        get_earth_pos,
        get_sun_pos,
        cold_start_bank=cold_start_bank,
        bias_mode=bias_mode,
        initial_bias=initial_bias,
        label=label,
        max_iter=max_iter,
        tol_cost_stability=tol_cost_stability,
        rtol=rtol,
        atol=atol,
        robust_outlier_rejection=robust_outlier_rejection,
        process_noise_covariance=process_noise_covariance,
        parallel=parallel,
    )


def _run_arc_worker(args: tuple) -> "ArcResult":
    arc, kwargs = args
    return run_batch_arc_sequence(
        (arc,),
        kwargs["measurement_type"],
        "cold",
        kwargs["estimator_type"],
        kwargs["mu_moon_m3_s2"],
        kwargs["mu_earth_m3_s2"],
        kwargs["mu_sun_m3_s2"],
        kwargs["get_earth_pos"],
        kwargs["get_sun_pos"],
        cold_start_bank=(kwargs["cold_pert"],),
        bias_mode=kwargs["bias_mode"],
        initial_bias=kwargs["initial_bias"],
        label=kwargs["label"],
        max_iter=kwargs["max_iter"],
        tol_cost_stability=kwargs["tol_cost_stability"],
        bls_lambda0=kwargs["bls_lambda0"],
        rtol=kwargs["rtol"],
        atol=kwargs["atol"],
        robust_outlier_rejection=kwargs["robust_outlier_rejection"],
        process_noise_covariance=kwargs["process_noise_covariance"],
        ukf_transform_config=kwargs["ukf_transform_config"],
        ukf_adaptive_config=kwargs["ukf_adaptive_config"],
        ukf_covariance_form=kwargs["ukf_covariance_form"],
        ukf_process_noise_model=kwargs["ukf_process_noise_model"],
        ukf_auto_bias_constraints=kwargs["ukf_auto_bias_constraints"],
        ukf_bias_freeze_relative_information=kwargs["ukf_bias_freeze_relative_information"],
        ukf_bias_regularize_relative_information=kwargs["ukf_bias_regularize_relative_information"],
        ukf_bias_regularization_std=kwargs["ukf_bias_regularization_std"],
        j2_moon=kwargs["j2_moon"],
    ).arc_results[0]


def run_batch_arc_sequence(
    prepared_arcs: Sequence[PreparedArc],
    measurement_type: MeasurementType,
    start_mode: StartMode,
    estimator_type: EstimatorType,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    *,
    cold_start_bank: Sequence[ArrayLike] | None = None,
    bias_mode: str | None = None,
    initial_bias: ArrayLike | None = None,
    label: str = "",
    max_iter: int = 40,
    tol_cost_stability: float = 1e-8,
    bls_lambda0: float = 1e-2,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    robust_outlier_rejection: bool = False,
    process_noise_covariance: ArrayLike | None = None,
    ukf_transform_config: UnscentedTransformConfig | None = None,
    ukf_adaptive_config: UKFAdaptiveConfig | None = None,
    ukf_covariance_form: str = "square_root",
    ukf_process_noise_model: str = "discrete",
    ukf_auto_bias_constraints: bool = False,
    ukf_bias_freeze_relative_information: float = 1e-12,
    ukf_bias_regularize_relative_information: float = 1e-5,
    ukf_bias_regularization_std: float = 1.0,
    ephemeris=None,
    j2_moon: float = 0.0,
    parallel: bool = False,
) -> ScenarioResult:
    """Run BLS/LM, SRIF, or UKF over prepared arcs with cold, hot, or formal starts."""
    _fast_sigma = None
    if ephemeris is not None and estimator_type == "ukf":
        _fast_sigma = make_fast_sigma_propagator(
            ephemeris, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
            j2_moon=j2_moon,
        )

    if start_mode not in {"cold", "hot", "formal", "sqrt_formal"}:
        raise ValueError("start_mode must be 'cold', 'hot', 'formal', or 'sqrt_formal'.")
    if measurement_type not in {"position", "range_rate", "two_way_range"}:
        raise ValueError(
            "measurement_type must be 'position', 'range_rate', or 'two_way_range'."
        )
    if estimator_type not in {"srif", "bls_lm", "ukf"}:
        raise ValueError("estimator_type must be 'srif', 'bls_lm', or 'ukf'.")
    if measurement_type == "two_way_range" and estimator_type == "ukf":
        raise ValueError(
            "measurement_type='two_way_range' is not supported by the UKF in M3; "
            "use estimator_type='bls_lm' or 'srif'."
        )
    if start_mode == "sqrt_formal" and estimator_type != "srif":
        raise ValueError("sqrt_formal handoff is only supported by the SRIF estimators.")
    bias_mode = _normalize_bias_mode(bias_mode)

    arcs = tuple(prepared_arcs)
    if cold_start_bank is None:
        cold_start_bank = tuple(np.zeros(6) for _ in arcs)
    if len(cold_start_bank) < len(arcs):
        raise ValueError("cold_start_bank must contain at least one perturbation per arc.")

    bias0 = np.asarray(initial_bias if initial_bias is not None else [], dtype=float).reshape(-1)
    results: list[ArcResult] = []
    observability_summaries: list[object] = []
    previous_estimate: np.ndarray | None = None
    previous_t0: float | None = None
    previous_covariance: np.ndarray | None = None
    previous_sqrt_information: np.ndarray | None = None
    process_noise = None if process_noise_covariance is None else np.asarray(process_noise_covariance, dtype=float)
    needs_posterior_handoff = start_mode in {"formal", "sqrt_formal"}

    if parallel:
        if start_mode != "cold":
            raise ValueError("parallel=True is only valid for start_mode='cold'.")
        from concurrent.futures import ProcessPoolExecutor
        import os as _os
        _base_kwargs: dict = {
            "measurement_type": measurement_type,
            "estimator_type": estimator_type,
            "mu_moon_m3_s2": mu_moon_m3_s2,
            "mu_earth_m3_s2": mu_earth_m3_s2,
            "mu_sun_m3_s2": mu_sun_m3_s2,
            "get_earth_pos": get_earth_pos,
            "get_sun_pos": get_sun_pos,
            "bias_mode": bias_mode,
            "initial_bias": initial_bias,
            "label": label,
            "max_iter": max_iter,
            "tol_cost_stability": tol_cost_stability,
            "bls_lambda0": bls_lambda0,
            "rtol": rtol,
            "atol": atol,
            "robust_outlier_rejection": robust_outlier_rejection,
            "process_noise_covariance": process_noise_covariance,
            "ukf_transform_config": ukf_transform_config,
            "ukf_adaptive_config": ukf_adaptive_config,
            "ukf_covariance_form": ukf_covariance_form,
            "ukf_process_noise_model": ukf_process_noise_model,
            "ukf_auto_bias_constraints": ukf_auto_bias_constraints,
            "ukf_bias_freeze_relative_information": ukf_bias_freeze_relative_information,
            "ukf_bias_regularize_relative_information": ukf_bias_regularize_relative_information,
            "ukf_bias_regularization_std": ukf_bias_regularization_std,
            "j2_moon": j2_moon,
        }
        _tasks = [
            (arc, {**_base_kwargs, "cold_pert": np.asarray(cold_start_bank[arc_idx], dtype=float).reshape(6)})
            for arc_idx, arc in enumerate(arcs)
        ]
        with ProcessPoolExecutor(max_workers=_os.cpu_count()) as _pool:
            _par_results = list(_pool.map(_run_arc_worker, _tasks))
        _rr_physics = RangeRatePhysicsConfig()
        if arcs and measurement_type == "range_rate":
            _rr_physics = range_rate_physics_config(arcs[0].pass_geo.range_rate_physics)
        _pass_geo0 = arcs[0].pass_geo if arcs else None
        _measurement_meta = _attached_measurement_metadata(_pass_geo0)
        _history_domain_fields = _history_domain_scenario_fields(
            _measurement_meta,
            measurement_type,
        )
        return ScenarioResult(
            label=label,
            measurement_type=measurement_type,
            start_mode=start_mode,
            arc_results=tuple(_par_results),
            estimator_type=estimator_type,
            range_rate_physics=_rr_physics.mode,
            count_interval_s=_rr_physics.count_interval_s,
            station_state_method=_rr_physics.station_state_method,
            measurement_model_profile=(
                "geometric_instantaneous"
                if _pass_geo0 is None
                else _pass_geo0.measurement_model_profile
            ),
            companion_geometry=(
                "instantaneous"
                if _pass_geo0 is None
                else _pass_geo0.companion_geometry
            ),
            jacobian_model=(
                "analytic_exact_geometric"
                if _pass_geo0 is None
                else _pass_geo0.jacobian_model
            ),
            range_jacobian_model=_measurement_meta.get(
                "range_jacobian_model", "analytic_exact_geometric"
            ),
            line_of_sight_jacobian_model=_measurement_meta.get(
                "line_of_sight_jacobian_model", "analytic_exact_geometric"
            ),
            angle_jacobian_model=_measurement_meta.get(
                "angle_jacobian_model", "analytic_exact_geometric"
            ),
            aberration_jacobian_model=_measurement_meta.get(
                "aberration_jacobian_model", "not_applied"
            ),
            angle_jacobian_matches_full_residual_physics=bool(
                _measurement_meta.get("angle_jacobian_matches_full_residual_physics", True)
            ),
            observer_velocity_epoch=_measurement_meta.get(
                "observer_velocity_epoch", "not_applicable"
            ),
            observer_velocity_frame=_measurement_meta.get(
                "observer_velocity_frame", "not_applicable"
            ),
            observer_velocity_reference_center=_measurement_meta.get(
                "observer_velocity_reference_center", "not_applicable"
            ),
            aberration_local_jacobian_input=_measurement_meta.get(
                "aberration_local_jacobian_input", "not_applicable"
            ),
            aberration_local_jacobian_space=_measurement_meta.get(
                "aberration_local_jacobian_space", "not_applicable"
            ),
            aberration_local_jacobian_step=float(
                _measurement_meta.get("aberration_local_jacobian_step", float("nan"))
            ),
            **_history_domain_fields,
            **_r1_numerical_provenance_fields(
                _par_results,
                estimator_type=estimator_type,
                observability_summaries=(),
            ),
        )

    for arc_index, arc in enumerate(arcs):
        x_true0 = np.asarray(arc.truth_state_history_mci[0, :6], dtype=float)
        cold_pert = np.asarray(cold_start_bank[arc_index], dtype=float).reshape(6)

        prior_covariance = None
        prior_sqrt_information = None
        if start_mode == "formal" and previous_estimate is not None and previous_t0 is not None:
            if previous_covariance is not None:
                x_dyn0, prior_covariance = _propagate_formal_handoff(
                    previous_estimate[:6],
                    previous_covariance,
                    previous_t0,
                    float(arc.t_pass_s[0]),
                    mu_moon_m3_s2,
                    mu_earth_m3_s2,
                    mu_sun_m3_s2,
                    get_earth_pos,
                    get_sun_pos,
                    process_noise,
                    rtol=rtol,
                    atol=atol,
                    j2_moon=j2_moon,
                )
            else:
                x_dyn0 = _propagate_hot_start(
                    previous_estimate[:6],
                    previous_t0,
                    float(arc.t_pass_s[0]),
                    mu_moon_m3_s2,
                    mu_earth_m3_s2,
                    mu_sun_m3_s2,
                    get_earth_pos,
                    get_sun_pos,
                    rtol=rtol,
                    atol=atol,
                    j2_moon=j2_moon,
                )
            b0 = previous_estimate[6:] if previous_estimate.size > 6 else bias0
        elif start_mode == "sqrt_formal" and previous_estimate is not None and previous_t0 is not None:
            if previous_sqrt_information is not None:
                x_dyn0, prior_sqrt_information, prior_covariance = _propagate_sqrt_information_handoff(
                    previous_estimate[:6],
                    previous_sqrt_information,
                    previous_t0,
                    float(arc.t_pass_s[0]),
                    mu_moon_m3_s2,
                    mu_earth_m3_s2,
                    mu_sun_m3_s2,
                    get_earth_pos,
                    get_sun_pos,
                    process_noise,
                    rtol=rtol,
                    atol=atol,
                    j2_moon=j2_moon,
                )
            else:
                x_dyn0 = _propagate_hot_start(
                    previous_estimate[:6],
                    previous_t0,
                    float(arc.t_pass_s[0]),
                    mu_moon_m3_s2,
                    mu_earth_m3_s2,
                    mu_sun_m3_s2,
                    get_earth_pos,
                    get_sun_pos,
                    rtol=rtol,
                    atol=atol,
                    j2_moon=j2_moon,
                )
            b0 = previous_estimate[6:] if previous_estimate.size > 6 else bias0
        elif start_mode == "hot" and previous_estimate is not None and previous_t0 is not None:
            x_dyn0 = _propagate_hot_start(
                previous_estimate[:6],
                previous_t0,
                float(arc.t_pass_s[0]),
                mu_moon_m3_s2,
                mu_earth_m3_s2,
                mu_sun_m3_s2,
                get_earth_pos,
                get_sun_pos,
                rtol=rtol,
                atol=atol,
                j2_moon=j2_moon,
            )
            b0 = previous_estimate[6:] if previous_estimate.size > 6 else bias0
        else:
            x_dyn0 = x_true0 + cold_pert
            b0 = bias0

        x_nominal = np.concatenate([x_dyn0, b0]) if b0.size else x_dyn0.copy()
        initial_position_error = float(np.linalg.norm(x_nominal[:3] - x_true0[:3]))
        initial_velocity_error = float(np.linalg.norm(x_nominal[3:6] - x_true0[3:6]))

        if estimator_type == "ukf":
            station_col = 4 if measurement_type == "position" else 5
            p0 = prior_covariance if prior_covariance is not None else _ukf_default_initial_covariance(
                x_nominal.size,
                measurement_type,
                bias_mode,
                len(arc.pass_geo.stations),
            )
            frozen_state_indices: tuple[int, ...] = ()
            regularization_std_by_state: dict[int, float] = {}
            regularized_state_indices: tuple[int, ...] = ()
            if ukf_auto_bias_constraints and bias_mode is not None:
                from .observability import (
                    BiasObservabilityPolicy,
                    analyze_augmented_arc_observability,
                    decide_bias_state_handling,
                )

                observability = analyze_augmented_arc_observability(
                    arc,
                    measurement_type,
                    mu_moon_m3_s2,
                    mu_earth_m3_s2,
                    mu_sun_m3_s2,
                    get_earth_pos,
                    get_sun_pos,
                    bias_mode=bias_mode,
                    x0_mci=x_nominal[:6],
                    rtol=rtol,
                    atol=atol,
                    j2_moon=j2_moon,
                )
                observability_summaries.append(observability)
                decision = decide_bias_state_handling(
                    observability,
                    policy=BiasObservabilityPolicy(
                        freeze_relative_information=ukf_bias_freeze_relative_information,
                        regularize_relative_information=ukf_bias_regularize_relative_information,
                        regularization_std=ukf_bias_regularization_std,
                    ),
                )
                frozen_state_indices = decision.frozen_state_indices
                regularized_state_indices = decision.regularized_state_indices
                regularization_std_by_state = decision.regularization_std_by_state
            ukf_result = run_lunar_ukf(
                arc.t_pass_s,
                arc.obs_data,
                x_nominal,
                p0,
                arc.pass_geo,
                mu_moon_m3_s2,
                mu_earth_m3_s2,
                mu_sun_m3_s2,
                get_earth_pos,
                get_sun_pos,
                measurement_type=measurement_type,
                bias_mode=bias_mode,
                process_noise=process_noise,
                process_noise_model=ukf_process_noise_model,
                covariance_form=ukf_covariance_form,
                adaptive_config=ukf_adaptive_config,
                config=ukf_transform_config or UnscentedTransformConfig(alpha=0.35),
                frozen_state_indices=frozen_state_indices,
                regularization_std_by_state=regularization_std_by_state,
                rtol=rtol,
                atol=atol,
                fast_sigma_propagator=_fast_sigma,
                j2_moon=j2_moon,
            )
            x_est = ukf_result.final_state.copy()
            stop_reason = "Converged" if np.all(ukf_result.accepted_updates) else "Gated"
            posterior_covariance = ukf_result.final_covariance.copy()
            posterior_sqrt_information = None
            stats = EstimatorStats(
                iterations=int(ukf_result.t_update_s.size),
                final_cost=float(np.nansum(ukf_result.normalized_innovation_squared)),
                position_step_norm_m=float(np.linalg.norm(x_est[:3] - x_nominal[:3])),
                velocity_step_norm_mps=float(np.linalg.norm(x_est[3:6] - x_nominal[3:6])),
                condition_number=float(np.linalg.cond(posterior_covariance)),
                rank=int(np.linalg.matrix_rank(posterior_covariance)),
                rejected_components=int(
                    np.size(ukf_result.accepted_components) - np.count_nonzero(ukf_result.accepted_components)
                ),
                active_weight_fraction=float(np.mean(ukf_result.accepted_components))
                if ukf_result.accepted_components.size
                else 1.0,
                posterior_covariance=posterior_covariance,
            )
            ukf_mean_nis = float(np.mean(ukf_result.normalized_innovation_squared)) if ukf_result.normalized_innovation_squared.size else float("nan")
            ukf_max_nis = float(np.max(ukf_result.normalized_innovation_squared)) if ukf_result.normalized_innovation_squared.size else float("nan")
            ukf_accepted_fraction = float(np.mean(ukf_result.accepted_updates)) if ukf_result.accepted_updates.size else float("nan")
            ukf_final_q_scale = float(ukf_result.process_noise_scales[-1]) if ukf_result.process_noise_scales.size else float("nan")
            whiteness = analyze_innovation_whiteness(
                ukf_result.innovations,
                ukf_result.innovation_covariances,
            )
            ukf_mean_abs_lag1 = whiteness.mean_abs_lag1_autocorrelation
            ukf_max_abs_lag1 = whiteness.max_abs_lag1_autocorrelation
            measurement_dimension = 3 if measurement_type == "position" else 4
            ukf_normalized_mean_nis = ukf_mean_nis / measurement_dimension
            ukf_nis_upper_consistent = bool(ukf_normalized_mean_nis <= 3.0)
            ukf_elapsed_s = ukf_result.performance.elapsed_s
            ukf_process_evaluations = ukf_result.performance.process_function_evaluations
            ukf_unique_propagations = ukf_result.performance.unique_dynamic_propagations
            ukf_cache_hits = ukf_result.performance.dynamic_propagation_cache_hits
            ukf_measurement_evaluations = ukf_result.performance.measurement_function_evaluations
            ukf_unique_measurement_evaluations = ukf_result.performance.unique_measurement_model_evaluations
            ukf_measurement_cache_hits = ukf_result.performance.measurement_model_cache_hits
            ukf_frozen_indices = frozen_state_indices
            ukf_regularized_indices = regularized_state_indices
            ukf_stability = assess_ukf_operational_stability(ukf_result)
            ukf_stability_passed = ukf_stability.stable
            ukf_min_covariance_eigenvalue = ukf_stability.min_covariance_eigenvalue
            ukf_max_covariance_condition = ukf_stability.max_covariance_condition_number
            ukf_robust_reweighted_fraction = ukf_stability.robust_reweighted_fraction
            final_obs_row = arc.obs_data[int(np.argmax(arc.obs_data[:, 0]))]
            time_index_column = 5 if measurement_type == "position" else 6
            final_truth_index = int(final_obs_row[time_index_column]) - 1
            truth_compare_state = np.asarray(
                arc.truth_state_history_mci[final_truth_index, :6],
                dtype=float,
            )
            handoff_epoch_s = float(final_obs_row[0])
        elif measurement_type == "position":
            station_col = 4
            estimator = estimate_position_srif if estimator_type == "srif" else estimate_position_bls_lm
            _lm_kw = {"lambda0": bls_lambda0} if estimator_type == "bls_lm" else {}
            x_est, stop_reason, stats = estimator(
                arc.t_pass_s,
                arc.obs_data,
                x_nominal,
                arc.pass_geo,
                mu_moon_m3_s2,
                mu_earth_m3_s2,
                mu_sun_m3_s2,
                get_earth_pos,
                get_sun_pos,
                max_iter=max_iter,
                tol_cost_stability=tol_cost_stability,
                rtol=rtol,
                atol=atol,
                j2_moon=j2_moon,
                bias_mode=bias_mode,
                **_lm_kw,
                prior_covariance=None if start_mode == "sqrt_formal" else prior_covariance,
                prior_sqrt_information=prior_sqrt_information,
                return_posterior=needs_posterior_handoff,
            )
            ukf_mean_nis = float("nan")
            ukf_max_nis = float("nan")
            ukf_accepted_fraction = float("nan")
            ukf_final_q_scale = float("nan")
            ukf_mean_abs_lag1 = float("nan")
            ukf_max_abs_lag1 = float("nan")
            ukf_normalized_mean_nis = float("nan")
            ukf_nis_upper_consistent = None
            ukf_elapsed_s = float("nan")
            ukf_process_evaluations = 0
            ukf_unique_propagations = 0
            ukf_cache_hits = 0
            ukf_measurement_evaluations = 0
            ukf_unique_measurement_evaluations = 0
            ukf_measurement_cache_hits = 0
            ukf_frozen_indices = ()
            ukf_regularized_indices = ()
            ukf_stability_passed = None
            ukf_min_covariance_eigenvalue = float("nan")
            ukf_max_covariance_condition = float("nan")
            ukf_robust_reweighted_fraction = float("nan")
            truth_compare_state = x_true0
            handoff_epoch_s = float(arc.t_pass_s[0])
        elif measurement_type == "two_way_range":
            station_col = 2
            estimator = (
                estimate_two_way_range_srif
                if estimator_type == "srif"
                else estimate_two_way_range_bls_lm
            )
            _lm_kw = {"lambda0": bls_lambda0} if estimator_type == "bls_lm" else {}
            x_est, stop_reason, stats = estimator(
                arc.t_pass_s,
                arc.obs_data,
                x_nominal,
                arc.pass_geo,
                mu_moon_m3_s2,
                mu_earth_m3_s2,
                mu_sun_m3_s2,
                get_earth_pos,
                get_sun_pos,
                max_iter=max_iter,
                tol_cost_stability=tol_cost_stability,
                rtol=rtol,
                atol=atol,
                j2_moon=j2_moon,
                bias_mode=bias_mode,
                robust_outlier_rejection=robust_outlier_rejection,
                **_lm_kw,
                prior_covariance=None if start_mode == "sqrt_formal" else prior_covariance,
                prior_sqrt_information=prior_sqrt_information,
                return_posterior=needs_posterior_handoff,
            )
            ukf_mean_nis = float("nan")
            ukf_max_nis = float("nan")
            ukf_accepted_fraction = float("nan")
            ukf_final_q_scale = float("nan")
            ukf_mean_abs_lag1 = float("nan")
            ukf_max_abs_lag1 = float("nan")
            ukf_normalized_mean_nis = float("nan")
            ukf_nis_upper_consistent = None
            ukf_elapsed_s = float("nan")
            ukf_process_evaluations = 0
            ukf_unique_propagations = 0
            ukf_cache_hits = 0
            ukf_measurement_evaluations = 0
            ukf_unique_measurement_evaluations = 0
            ukf_measurement_cache_hits = 0
            ukf_frozen_indices = ()
            ukf_regularized_indices = ()
            ukf_stability_passed = None
            ukf_min_covariance_eigenvalue = float("nan")
            ukf_max_covariance_condition = float("nan")
            ukf_robust_reweighted_fraction = float("nan")
            truth_compare_state = x_true0
            handoff_epoch_s = float(arc.t_pass_s[0])
        else:
            station_col = 5
            estimator = estimate_range_rate_srif if estimator_type == "srif" else estimate_range_rate_bls_lm
            _lm_kw = {"lambda0": bls_lambda0} if estimator_type == "bls_lm" else {}
            x_est, stop_reason, stats = estimator(
                arc.t_pass_s,
                arc.obs_data,
                x_nominal,
                arc.pass_geo,
                mu_moon_m3_s2,
                mu_earth_m3_s2,
                mu_sun_m3_s2,
                get_earth_pos,
                get_sun_pos,
                max_iter=max_iter,
                tol_cost_stability=tol_cost_stability,
                rtol=rtol,
                atol=atol,
                j2_moon=j2_moon,
                bias_mode=bias_mode,
                robust_outlier_rejection=robust_outlier_rejection,
                **_lm_kw,
                prior_covariance=None if start_mode == "sqrt_formal" else prior_covariance,
                prior_sqrt_information=prior_sqrt_information,
                return_posterior=needs_posterior_handoff,
            )
            ukf_mean_nis = float("nan")
            ukf_max_nis = float("nan")
            ukf_accepted_fraction = float("nan")
            ukf_final_q_scale = float("nan")
            ukf_mean_abs_lag1 = float("nan")
            ukf_max_abs_lag1 = float("nan")
            ukf_normalized_mean_nis = float("nan")
            ukf_nis_upper_consistent = None
            ukf_elapsed_s = float("nan")
            ukf_process_evaluations = 0
            ukf_unique_propagations = 0
            ukf_cache_hits = 0
            ukf_measurement_evaluations = 0
            ukf_unique_measurement_evaluations = 0
            ukf_measurement_cache_hits = 0
            ukf_frozen_indices = ()
            ukf_regularized_indices = ()
            ukf_stability_passed = None
            ukf_min_covariance_eigenvalue = float("nan")
            ukf_max_covariance_condition = float("nan")
            ukf_robust_reweighted_fraction = float("nan")
            truth_compare_state = x_true0
            handoff_epoch_s = float(arc.t_pass_s[0])

        final_position_error = float(np.linalg.norm(x_est[:3] - truth_compare_state[:3]))
        final_velocity_error = float(np.linalg.norm(x_est[3:6] - truth_compare_state[3:6]))
        observed_station_ids = tuple(sorted(set((arc.obs_data[:, station_col].astype(int) - 1).tolist())))
        if estimator_type != "ukf" and stats.posterior_covariance is not None:
            posterior_covariance = np.asarray(stats.posterior_covariance, dtype=float).copy()
        elif estimator_type != "ukf":
            posterior_covariance = None
        if estimator_type != "ukf" and stats.posterior_sqrt_information is not None:
            posterior_sqrt_information = np.asarray(stats.posterior_sqrt_information, dtype=float).copy()
        elif estimator_type != "ukf":
            posterior_sqrt_information = None
        results.append(
            ArcResult(
                arc_id=arc.arc_id,
                start_idx=arc.start_idx,
                end_idx=arc.end_idx,
                num_observations=int(arc.obs_data.shape[0]),
                initial_position_error_m=initial_position_error,
                initial_velocity_error_mps=initial_velocity_error,
                final_position_error_m=final_position_error,
                final_velocity_error_mps=final_velocity_error,
                stop_reason=stop_reason,
                stats=stats,
                estimated_state=x_est[:6].copy(),
                estimated_bias=x_est[6:].copy(),
                observed_station_ids=observed_station_ids,
                ukf_mean_nis=ukf_mean_nis,
                ukf_max_nis=ukf_max_nis,
                ukf_accepted_update_fraction=ukf_accepted_fraction,
                ukf_final_process_noise_scale=ukf_final_q_scale,
                ukf_innovation_mean_abs_lag1=ukf_mean_abs_lag1,
                ukf_innovation_max_abs_lag1=ukf_max_abs_lag1,
                ukf_normalized_mean_nis=ukf_normalized_mean_nis,
                ukf_nis_upper_consistent=ukf_nis_upper_consistent,
                ukf_elapsed_s=ukf_elapsed_s,
                ukf_process_function_evaluations=ukf_process_evaluations,
                ukf_unique_dynamic_propagations=ukf_unique_propagations,
                ukf_dynamic_propagation_cache_hits=ukf_cache_hits,
                ukf_measurement_function_evaluations=ukf_measurement_evaluations,
                ukf_unique_measurement_model_evaluations=ukf_unique_measurement_evaluations,
                ukf_measurement_model_cache_hits=ukf_measurement_cache_hits,
                ukf_frozen_state_indices=ukf_frozen_indices,
                ukf_regularized_state_indices=ukf_regularized_indices,
                ukf_stability_passed=ukf_stability_passed,
                ukf_min_covariance_eigenvalue=ukf_min_covariance_eigenvalue,
                ukf_max_covariance_condition_number=ukf_max_covariance_condition,
                ukf_robust_reweighted_fraction=ukf_robust_reweighted_fraction,
                prior_covariance=None if prior_covariance is None else prior_covariance.copy(),
                posterior_covariance=posterior_covariance,
                prior_sqrt_information=None
                if prior_sqrt_information is None
                else prior_sqrt_information.copy(),
                posterior_sqrt_information=posterior_sqrt_information,
            )
        )

        previous_estimate = x_est.copy()
        previous_t0 = handoff_epoch_s
        if start_mode == "formal":
            previous_covariance = posterior_covariance
        elif start_mode == "sqrt_formal":
            previous_sqrt_information = posterior_sqrt_information
            previous_covariance = posterior_covariance

    result_rr_physics = RangeRatePhysicsConfig()
    if arcs and measurement_type == "range_rate":
        result_rr_physics = range_rate_physics_config(arcs[0].pass_geo.range_rate_physics)
    pass_geo0 = arcs[0].pass_geo if arcs else None
    measurement_meta = _attached_measurement_metadata(pass_geo0)
    history_domain_fields = _history_domain_scenario_fields(
        measurement_meta,
        measurement_type,
    )

    return ScenarioResult(
        label=label,
        measurement_type=measurement_type,
        start_mode=start_mode,
        arc_results=tuple(results),
        estimator_type=estimator_type,
        range_rate_physics=result_rr_physics.mode,
        count_interval_s=result_rr_physics.count_interval_s,
        measurement_model_profile=(
            "geometric_instantaneous"
            if pass_geo0 is None
            else pass_geo0.measurement_model_profile
        ),
        companion_geometry=(
            "instantaneous"
            if pass_geo0 is None
            else pass_geo0.companion_geometry
        ),
        jacobian_model=(
            "analytic_exact_geometric"
            if pass_geo0 is None
            else pass_geo0.jacobian_model
        ),
        range_jacobian_model=measurement_meta.get(
            "range_jacobian_model", "analytic_exact_geometric"
        ),
        line_of_sight_jacobian_model=measurement_meta.get(
            "line_of_sight_jacobian_model", "analytic_exact_geometric"
        ),
        angle_jacobian_model=measurement_meta.get(
            "angle_jacobian_model", "analytic_exact_geometric"
        ),
        aberration_jacobian_model=measurement_meta.get(
            "aberration_jacobian_model", "not_applied"
        ),
        angle_jacobian_matches_full_residual_physics=bool(
            measurement_meta.get("angle_jacobian_matches_full_residual_physics", True)
        ),
        observer_velocity_epoch=measurement_meta.get(
            "observer_velocity_epoch", "not_applicable"
        ),
        observer_velocity_frame=measurement_meta.get(
            "observer_velocity_frame", "not_applicable"
        ),
        observer_velocity_reference_center=measurement_meta.get(
            "observer_velocity_reference_center", "not_applicable"
        ),
        aberration_local_jacobian_input=measurement_meta.get(
            "aberration_local_jacobian_input", "not_applicable"
        ),
        aberration_local_jacobian_space=measurement_meta.get(
            "aberration_local_jacobian_space", "not_applicable"
        ),
        aberration_local_jacobian_step=float(
            measurement_meta.get("aberration_local_jacobian_step", float("nan"))
        ),
        **history_domain_fields,
        **_r1_numerical_provenance_fields(
            results,
            estimator_type=estimator_type,
            observability_summaries=observability_summaries,
        ),
    )


def _r1_numerical_provenance_fields(
    arc_results: Sequence[ArcResult],
    *,
    estimator_type: EstimatorType,
    observability_summaries: Sequence[object],
) -> dict[str, object]:
    covariance_arrays = [
        np.asarray(result.posterior_covariance, dtype=float)
        for result in arc_results
        if result.posterior_covariance is not None
    ]
    covariance_finite = bool(
        covariance_arrays and all(np.all(np.isfinite(value)) for value in covariance_arrays)
    )
    if covariance_finite:
        covariance_ranks = [int(np.linalg.matrix_rank(value)) for value in covariance_arrays]
        covariance_conditions = [float(np.linalg.cond(value)) for value in covariance_arrays]
        covariance_min_eigenvalues = [
            float(np.min(np.linalg.eigvalsh(0.5 * (value + value.T))))
            for value in covariance_arrays
        ]
        covariance_rank = min(covariance_ranks)
        covariance_condition = max(covariance_conditions)
        covariance_min_eigenvalue = min(covariance_min_eigenvalues)
    else:
        covariance_rank = 0
        covariance_condition = float("nan")
        covariance_min_eigenvalue = float("nan")

    observability_finite = bool(
        observability_summaries
        and all(
            np.all(np.isfinite(np.asarray(summary.weighted_jacobian, dtype=float)))
            and np.all(np.isfinite(np.asarray(summary.fisher_information, dtype=float)))
            and np.all(np.isfinite(np.asarray(summary.singular_values, dtype=float)))
            for summary in observability_summaries
        )
    )
    if observability_finite:
        observability_rank = min(int(summary.rank) for summary in observability_summaries)
        observability_condition = max(
            float(summary.condition_number) for summary in observability_summaries
        )
        observability_singular_min = min(
            float(np.min(summary.singular_values)) for summary in observability_summaries
        )
        observability_singular_max = max(
            float(np.max(summary.singular_values)) for summary in observability_summaries
        )
    else:
        observability_rank = 0
        observability_condition = float("nan")
        observability_singular_min = float("nan")
        observability_singular_max = float("nan")

    return {
        "posterior_covariance_stm_mode": (
            "ukf_sigma_point" if estimator_type == "ukf" else "analytic_variational"
        ),
        "posterior_covariance_rank": covariance_rank,
        "posterior_covariance_condition_number": covariance_condition,
        "posterior_covariance_min_eigenvalue": covariance_min_eigenvalue,
        "posterior_covariance_finite": covariance_finite,
        "observability_rank": observability_rank,
        "observability_condition_number": observability_condition,
        "observability_singular_value_min": observability_singular_min,
        "observability_singular_value_max": observability_singular_max,
        "observability_finite": observability_finite,
        "derivative_validation_profile": "r1.fd-sweep.v1",
    }


def _ukf_default_initial_covariance(
    state_size: int,
    measurement_type: MeasurementType,
    bias_mode: str | None,
    num_stations: int,
) -> np.ndarray:
    covariance = np.zeros((state_size, state_size), dtype=float)
    covariance[:6, :6] = np.diag([100.0**2] * 3 + [0.1**2] * 3)
    if state_size <= 6:
        return covariance

    bias_size = state_size - 6
    if measurement_type == "position":
        block = np.array([100.0**2, np.deg2rad(0.05) ** 2, np.deg2rad(0.05) ** 2], dtype=float)
        station_block = np.array([100.0**2, np.deg2rad(0.05) ** 2, np.deg2rad(0.05) ** 2], dtype=float)
        angle_block = np.array([np.deg2rad(0.05) ** 2, np.deg2rad(0.05) ** 2], dtype=float)
    else:
        block = np.array([100.0**2, 1e-2**2, np.deg2rad(0.05) ** 2, np.deg2rad(0.05) ** 2], dtype=float)
        station_block = block
        angle_block = np.array([np.deg2rad(0.05) ** 2, np.deg2rad(0.05) ** 2], dtype=float)

    mode = "global_full" if bias_mode in {None, "global", "global_full"} else str(bias_mode)
    if mode == "station_angles":
        diag = np.tile(angle_block, num_stations)
    elif mode == "station_full":
        diag = np.tile(station_block, num_stations)
    else:
        diag = block
    if diag.size != bias_size:
        diag = np.full(bias_size, 1.0, dtype=float)
    covariance[6:, 6:] = np.diag(diag)
    return covariance


def _normalize_bias_mode(bias_mode: str | None) -> str | None:
    if bias_mode is None:
        return None
    normalized = bias_mode.lower()
    if normalized == "global":
        return "global_full"
    return normalized


def _propagate_hot_start(
    x_prev: np.ndarray,
    t_prev_s: float,
    t_next_s: float,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    *,
    rtol: float,
    atol: float,
    j2_moon: float = 0.0,
) -> np.ndarray:
    if np.isclose(t_prev_s, t_next_s):
        return x_prev.copy()
    return propagate_state(
        [t_prev_s, t_next_s],
        x_prev,
        mu_moon_m3_s2,
        mu_earth_m3_s2,
        mu_sun_m3_s2,
        get_earth_pos,
        get_sun_pos,
        rtol=rtol,
        atol=atol,
        j2_moon=j2_moon,
    )[-1, :]


def _propagate_formal_handoff(
    x_prev: np.ndarray,
    p_prev: np.ndarray,
    t_prev_s: float,
    t_next_s: float,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    process_noise_covariance: np.ndarray | None,
    *,
    rtol: float,
    atol: float,
    j2_moon: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    p_prev = _symmetrize(np.asarray(p_prev, dtype=float))
    if np.isclose(t_prev_s, t_next_s):
        p_next = p_prev.copy()
        if process_noise_covariance is not None:
            p_next = _add_process_noise(p_next, process_noise_covariance)
        return x_prev.copy(), _symmetrize(p_next)

    x_aug0 = np.concatenate([x_prev, np.eye(6).reshape(-1, order="F")])
    x_aug = propagate_augmented_state(
        [t_prev_s, t_next_s],
        x_aug0,
        mu_moon_m3_s2,
        mu_earth_m3_s2,
        mu_sun_m3_s2,
        get_earth_pos,
        get_sun_pos,
        rtol=rtol,
        atol=atol,
        j2_moon=j2_moon,
    )
    phi = x_aug[-1, 6:].reshape((6, 6), order="F")
    transition = np.eye(p_prev.shape[0], dtype=float)
    transition[:6, :6] = phi
    p_next = transition @ p_prev @ transition.T
    if process_noise_covariance is not None:
        p_next = _add_process_noise(p_next, process_noise_covariance)
    return x_aug[-1, :6].copy(), _symmetrize(p_next)


def _propagate_sqrt_information_handoff(
    x_prev: np.ndarray,
    r_prev: np.ndarray,
    t_prev_s: float,
    t_next_s: float,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    process_noise_covariance: np.ndarray | None,
    *,
    rtol: float,
    atol: float,
    j2_moon: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_prev = np.asarray(r_prev, dtype=float)
    if r_prev.ndim != 2 or r_prev.shape[0] != r_prev.shape[1] or r_prev.shape[0] < 6:
        raise ValueError("sqrt information handoff factor must be square with at least 6 states.")

    if np.isclose(t_prev_s, t_next_s):
        x_next = x_prev.copy()
        transition = np.eye(r_prev.shape[0], dtype=float)
    else:
        x_aug0 = np.concatenate([x_prev, np.eye(6).reshape(-1, order="F")])
        x_aug = propagate_augmented_state(
            [t_prev_s, t_next_s],
            x_aug0,
            mu_moon_m3_s2,
            mu_earth_m3_s2,
            mu_sun_m3_s2,
            get_earth_pos,
            get_sun_pos,
            rtol=rtol,
            atol=atol,
            j2_moon=j2_moon,
        )
        phi = x_aug[-1, 6:].reshape((6, 6), order="F")
        transition = np.eye(r_prev.shape[0], dtype=float)
        transition[:6, :6] = phi
        x_next = x_aug[-1, :6].copy()

    if process_noise_covariance is None:
        propagated_rows = np.linalg.solve(transition.T, r_prev.T).T
        r_next = _upper_triangular_qr_factor(propagated_rows)
        p_next = _covariance_from_sqrt_information(r_next)
        return x_next, r_next, p_next

    r_next = _sqrt_information_time_update(r_prev, transition, process_noise_covariance)
    p_next = _covariance_from_sqrt_information(r_next)
    return x_next, r_next, p_next


def _sqrt_information_time_update(
    r_prev: np.ndarray,
    transition: np.ndarray,
    process_noise_covariance: np.ndarray,
) -> np.ndarray:
    r_prev = np.asarray(r_prev, dtype=float)
    transition = np.asarray(transition, dtype=float)
    process_noise = np.asarray(process_noise_covariance, dtype=float)
    n_state = r_prev.shape[0]

    if transition.shape != (n_state, n_state):
        raise ValueError("transition must match the sqrt information factor size.")

    if process_noise.shape == (n_state, n_state):
        r_next = _full_sqrt_information_time_update(r_prev, transition, process_noise)
    elif process_noise.shape == (6, 6) and n_state == 6:
        r_next = _full_sqrt_information_time_update(r_prev, transition, process_noise)
    elif process_noise.shape == (6, 6) and n_state > 6:
        r_next = _state_noise_augmented_sqrt_information_time_update(r_prev, transition, process_noise)
    else:
        raise ValueError("process_noise_covariance must be 6x6 or match the handoff covariance size.")

    return r_next


def _full_sqrt_information_time_update(
    r_prev: np.ndarray,
    transition: np.ndarray,
    process_noise_covariance: np.ndarray,
) -> np.ndarray:
    n_state = r_prev.shape[0]
    q_sqrt_info = _sqrt_information_from_covariance_strict(process_noise_covariance)
    rows = np.vstack(
        [
            np.hstack([r_prev, np.zeros((n_state, n_state), dtype=float)]),
            np.hstack([-q_sqrt_info @ transition, q_sqrt_info]),
        ]
    )
    r_joint = _upper_triangular_qr_factor(rows)
    return _orient_upper_triangular(r_joint[n_state:, n_state:])


def _state_noise_augmented_sqrt_information_time_update(
    r_prev: np.ndarray,
    transition: np.ndarray,
    state_process_noise_covariance: np.ndarray,
) -> np.ndarray:
    n_aug = r_prev.shape[0]
    n_dyn = 6
    n_bias = n_aug - n_dyn
    if np.linalg.norm(transition[:n_dyn, n_dyn:], ord="fro") > 1e-10:
        raise ValueError("6x6 state process noise assumes dynamics do not depend on bias states.")
    if np.linalg.norm(transition[n_dyn:, :n_dyn], ord="fro") > 1e-10:
        raise ValueError("6x6 state process noise assumes bias states are deterministic.")
    if not np.allclose(transition[n_dyn:, n_dyn:], np.eye(n_bias), atol=1e-10, rtol=1e-10):
        raise ValueError("6x6 state process noise assumes identity bias handoff.")

    q_sqrt_info = _sqrt_information_from_covariance_strict(state_process_noise_covariance)
    phi = transition[:n_dyn, :n_dyn]

    prior_rows = np.hstack(
        [
            r_prev[:, :n_dyn],
            np.zeros((n_aug, n_dyn), dtype=float),
            r_prev[:, n_dyn:],
        ]
    )
    process_rows = np.hstack(
        [
            -q_sqrt_info @ phi,
            q_sqrt_info,
            np.zeros((n_dyn, n_bias), dtype=float),
        ]
    )
    r_joint = _upper_triangular_qr_factor(np.vstack([prior_rows, process_rows]))
    return _orient_upper_triangular(r_joint[n_dyn:, n_dyn:])


def _add_process_noise(covariance: np.ndarray, process_noise_covariance: np.ndarray) -> np.ndarray:
    process_noise = np.asarray(process_noise_covariance, dtype=float)
    covariance = np.asarray(covariance, dtype=float).copy()
    if process_noise.shape == covariance.shape:
        covariance += process_noise
    elif process_noise.shape == (6, 6):
        covariance[:6, :6] += process_noise
    else:
        raise ValueError("process_noise_covariance must be 6x6 or match the handoff covariance size.")
    return covariance


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _covariance_from_sqrt_information(r_info: np.ndarray) -> np.ndarray:
    r_info = np.asarray(r_info, dtype=float)
    identity = np.eye(r_info.shape[0], dtype=float)
    try:
        inv_r = np.linalg.solve(r_info, identity)
        return _symmetrize(inv_r @ inv_r.T)
    except np.linalg.LinAlgError:
        info = _symmetrize(r_info.T @ r_info)
        return _safe_covariance_from_information(info)


def _sqrt_information_from_covariance(covariance: np.ndarray) -> np.ndarray:
    information = _safe_information_from_covariance(covariance)
    try:
        return np.linalg.cholesky(information).T
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(_symmetrize(information))
        max_val = float(np.max(np.abs(vals))) if vals.size else 1.0
        floor = max(max_val * 1e-14, np.finfo(float).eps)
        vals = np.clip(vals, floor, None)
        return _upper_triangular_qr_factor(np.sqrt(vals)[:, None] * vecs.T)


def _sqrt_information_from_covariance_strict(covariance: np.ndarray) -> np.ndarray:
    covariance = _symmetrize(np.asarray(covariance, dtype=float))
    try:
        chol = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "sqrt_formal process-noise update requires a positive-definite process-noise covariance. "
            "Use None for deterministic handoff, or provide a strictly positive covariance."
        ) from exc
    identity = np.eye(covariance.shape[0], dtype=float)
    return np.linalg.solve(chol, identity)


def _safe_information_from_covariance(covariance: np.ndarray) -> np.ndarray:
    cov = _symmetrize(np.asarray(covariance, dtype=float))
    vals, vecs = np.linalg.eigh(cov)
    max_val = float(np.max(np.abs(vals))) if vals.size else 1.0
    floor = max(max_val * 1e-14, np.finfo(float).eps)
    vals = np.clip(vals, floor, None)
    return _symmetrize((vecs / vals) @ vecs.T)


def _safe_covariance_from_information(information: np.ndarray) -> np.ndarray:
    info = _symmetrize(np.asarray(information, dtype=float))
    vals, vecs = np.linalg.eigh(info)
    max_val = float(np.max(np.abs(vals))) if vals.size else 1.0
    floor = max(max_val * 1e-14, np.finfo(float).eps)
    vals = np.clip(vals, floor, None)
    return _symmetrize((vecs / vals) @ vecs.T)


def _upper_triangular_qr_factor(rows: np.ndarray) -> np.ndarray:
    _, r_factor = np.linalg.qr(np.asarray(rows, dtype=float), mode="reduced")
    n_cols = rows.shape[1]
    r_factor = r_factor[:n_cols, :n_cols]
    return _orient_upper_triangular(r_factor)


def _orient_upper_triangular(r_factor: np.ndarray) -> np.ndarray:
    signs = np.where(np.diag(r_factor) < 0.0, -1.0, 1.0)
    return signs[:, None] * r_factor
