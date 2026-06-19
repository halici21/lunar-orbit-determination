"""Empirical adaptive-Q/R selection using consistency and accuracy metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike

from .filters import (
    LunarUKFResult,
    UKFAdaptiveConfig,
    assess_ukf_operational_stability,
    normalized_estimation_error_squared,
)
from .monte_carlo import MonteCarloCaseResult


@dataclass(frozen=True)
class AdaptiveTuningCase:
    label: str
    config: UKFAdaptiveConfig


@dataclass(frozen=True)
class AdaptiveTuningEvaluation:
    label: str
    mean_nis: float
    measurement_dimension: int
    rms_position_error_m: float
    mean_nees: float | None = None
    state_dimension: int | None = None
    failure_fraction: float = 0.0
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.measurement_dimension <= 0:
            raise ValueError("measurement_dimension must be positive.")
        if self.state_dimension is not None and self.state_dimension <= 0:
            raise ValueError("state_dimension must be positive when provided.")
        if (self.mean_nees is None) != (self.state_dimension is None):
            raise ValueError("mean_nees and state_dimension must be provided together.")
        if self.rms_position_error_m < 0.0:
            raise ValueError("rms_position_error_m must be non-negative.")
        if not (0.0 <= self.failure_fraction <= 1.0):
            raise ValueError("failure_fraction must be between zero and one.")


@dataclass(frozen=True)
class AdaptiveTuningObjective:
    nis_weight: float = 1.0
    nees_weight: float = 1.0
    position_error_weight: float = 1.0
    failure_weight: float = 4.0
    position_error_scale_m: float = 100.0

    def __post_init__(self) -> None:
        weights = (
            self.nis_weight,
            self.nees_weight,
            self.position_error_weight,
            self.failure_weight,
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("Adaptive tuning weights must be non-negative.")
        if self.position_error_scale_m <= 0.0:
            raise ValueError("position_error_scale_m must be positive.")


@dataclass(frozen=True)
class AdaptiveTuningSelection:
    best_label: str
    best_score: float
    scores: Mapping[str, float]


@dataclass(frozen=True)
class AdaptiveTuningTradeoff:
    candidate_label: str
    reference_label: str
    nis_consistency_improvement_fraction: float
    position_error_increase_fraction: float
    nees_increase_fraction: float | None
    failure_fraction_delta: float
    nis_consistency_improved: bool
    position_accuracy_degraded: bool
    nees_consistency_degraded: bool
    reliability_degraded: bool

    @property
    def flagged(self) -> bool:
        return self.nis_consistency_improved and (
            self.position_accuracy_degraded
            or self.nees_consistency_degraded
            or self.reliability_degraded
        )


def adaptive_tuning_score(
    evaluation: AdaptiveTuningEvaluation,
    objective: AdaptiveTuningObjective | None = None,
) -> float:
    """Score consistency and accuracy without allowing NIS alone to dominate."""
    objective = objective or AdaptiveTuningObjective()
    nis_penalty = _log_ratio_penalty(evaluation.mean_nis, evaluation.measurement_dimension)
    nees_penalty = 0.0
    if evaluation.mean_nees is not None and evaluation.state_dimension is not None:
        nees_penalty = _log_ratio_penalty(evaluation.mean_nees, evaluation.state_dimension)
    position_penalty = np.log1p(
        evaluation.rms_position_error_m / objective.position_error_scale_m
    )
    return float(
        objective.nis_weight * nis_penalty
        + objective.nees_weight * nees_penalty
        + objective.position_error_weight * position_penalty
        + objective.failure_weight * evaluation.failure_fraction
    )


def analyze_adaptive_tuning_tradeoffs(
    evaluations: Sequence[AdaptiveTuningEvaluation],
    *,
    reference_label: str | None = None,
    min_nis_improvement_fraction: float = 0.05,
    max_position_error_increase_fraction: float = 0.10,
    max_nees_increase_fraction: float = 0.10,
    max_failure_fraction_delta: float = 0.05,
) -> tuple[AdaptiveTuningTradeoff, ...]:
    """Flag candidates that buy better NIS consistency by degrading accuracy."""
    if not evaluations:
        return ()
    labels = [evaluation.label for evaluation in evaluations]
    if len(labels) != len(set(labels)):
        raise ValueError("Adaptive tuning evaluation labels must be unique.")
    if min_nis_improvement_fraction < 0.0:
        raise ValueError("min_nis_improvement_fraction must be non-negative.")
    if max_position_error_increase_fraction < 0.0:
        raise ValueError("max_position_error_increase_fraction must be non-negative.")
    if max_nees_increase_fraction < 0.0:
        raise ValueError("max_nees_increase_fraction must be non-negative.")
    if max_failure_fraction_delta < 0.0:
        raise ValueError("max_failure_fraction_delta must be non-negative.")

    by_label = {evaluation.label: evaluation for evaluation in evaluations}
    if reference_label is not None and reference_label not in by_label:
        raise ValueError(f"Unknown adaptive tuning reference label: {reference_label!r}.")
    reference = by_label[reference_label] if reference_label is not None else evaluations[0]
    tradeoffs = []
    reference_nis_penalty = _log_ratio_penalty(reference.mean_nis, reference.measurement_dimension)
    reference_nees_penalty = (
        _log_ratio_penalty(reference.mean_nees, reference.state_dimension)
        if reference.mean_nees is not None and reference.state_dimension is not None
        else None
    )
    for candidate in evaluations:
        if candidate.label == reference.label:
            continue
        candidate_nis_penalty = _log_ratio_penalty(candidate.mean_nis, candidate.measurement_dimension)
        nis_improvement = _relative_penalty_improvement(reference_nis_penalty, candidate_nis_penalty)
        position_increase = _relative_increase(
            reference.rms_position_error_m,
            candidate.rms_position_error_m,
        )

        nees_increase = None
        nees_degraded = False
        if (
            reference_nees_penalty is not None
            and candidate.mean_nees is not None
            and candidate.state_dimension is not None
        ):
            candidate_nees_penalty = _log_ratio_penalty(candidate.mean_nees, candidate.state_dimension)
            nees_increase = _relative_increase(reference_nees_penalty, candidate_nees_penalty)
            nees_degraded = bool(nees_increase > max_nees_increase_fraction)

        failure_delta = float(candidate.failure_fraction - reference.failure_fraction)
        tradeoffs.append(
            AdaptiveTuningTradeoff(
                candidate_label=candidate.label,
                reference_label=reference.label,
                nis_consistency_improvement_fraction=nis_improvement,
                position_error_increase_fraction=position_increase,
                nees_increase_fraction=nees_increase,
                failure_fraction_delta=failure_delta,
                nis_consistency_improved=bool(nis_improvement >= min_nis_improvement_fraction),
                position_accuracy_degraded=bool(position_increase > max_position_error_increase_fraction),
                nees_consistency_degraded=nees_degraded,
                reliability_degraded=bool(failure_delta > max_failure_fraction_delta),
            )
        )
    return tuple(tradeoffs)


def select_adaptive_tuning(
    evaluations: Sequence[AdaptiveTuningEvaluation],
    objective: AdaptiveTuningObjective | None = None,
) -> AdaptiveTuningSelection:
    """Select the lowest finite empirical score from a Monte Carlo campaign."""
    labels = [evaluation.label for evaluation in evaluations]
    if len(labels) != len(set(labels)):
        raise ValueError("Adaptive tuning evaluation labels must be unique.")
    scores = {
        evaluation.label: adaptive_tuning_score(evaluation, objective)
        for evaluation in evaluations
    }
    finite_scores = [(score, label) for label, score in scores.items() if np.isfinite(score)]
    if not finite_scores:
        raise ValueError("No finite adaptive tuning scores are available.")
    best_score, best_label = min(finite_scores)
    return AdaptiveTuningSelection(
        best_label=best_label,
        best_score=float(best_score),
        scores=scores,
    )


def adaptive_evaluations_from_campaign(
    case_results: Sequence[MonteCarloCaseResult],
    *,
    measurement_dimension: int,
    state_dimension: int | None = None,
    nis_metric: str = "mean_nis",
    nees_metric: str = "final_nees",
    position_error_metric: str = "final_position_error_m",
) -> tuple[AdaptiveTuningEvaluation, ...]:
    """Convert trial-level campaign metrics into comparable tuning evaluations."""
    evaluations = []
    for case_result in case_results:
        successful = [trial for trial in case_result.trials if trial.success]
        nis_values = _finite_metric_values(successful, nis_metric)
        position_errors = _finite_metric_values(successful, position_error_metric)
        nees_values = _finite_metric_values(successful, nees_metric)
        mean_nees = float(np.mean(nees_values)) if nees_values.size else None
        evaluation_state_dimension = state_dimension if mean_nees is not None else None
        evaluations.append(
            AdaptiveTuningEvaluation(
                label=case_result.case.label,
                mean_nis=float(np.mean(nis_values)) if nis_values.size else float("nan"),
                measurement_dimension=measurement_dimension,
                rms_position_error_m=float(np.sqrt(np.mean(position_errors**2)))
                if position_errors.size
                else float("inf"),
                mean_nees=mean_nees,
                state_dimension=evaluation_state_dimension,
                failure_fraction=1.0 - case_result.summary.success_fraction,
                metadata={"num_trials": str(case_result.summary.num_trials)},
            )
        )
    return tuple(evaluations)


def adaptive_evaluation_from_ukf_result(
    label: str,
    result: LunarUKFResult,
    truth_state_history: ArrayLike,
    *,
    measurement_dimension: int | None = None,
    state_dimension: int | None = None,
    include_nees: bool = True,
    failure_fraction_from_rejections: bool = True,
    metadata: Mapping[str, str] | None = None,
) -> AdaptiveTuningEvaluation:
    """Create a tuning evaluation directly from a completed UKF run and truth."""
    states = np.asarray(result.state_estimates, dtype=float)
    covariances = np.asarray(result.covariances, dtype=float)
    truth = _coerce_truth_history(truth_state_history, states.shape[0])
    if truth.shape[1] < 3 or states.shape[1] < 3:
        raise ValueError("state estimates and truth must contain at least position components.")
    if covariances.shape[:2] != (states.shape[0], states.shape[1]):
        raise ValueError("result covariance history must match state estimate history.")

    measurement_dim = measurement_dimension or _infer_measurement_dimension(result)
    pos_errors = states[:, :3] - truth[:, :3]
    rms_position_error = float(np.sqrt(np.mean(np.sum(pos_errors**2, axis=1))))

    evaluation_state_dimension = None
    mean_nees = None
    if include_nees:
        evaluation_state_dimension = state_dimension or min(states.shape[1], truth.shape[1])
        if evaluation_state_dimension <= 0:
            raise ValueError("state_dimension must be positive.")
        nees_values = []
        for state, truth_state, covariance in zip(states, truth, covariances):
            nees_values.append(
                normalized_estimation_error_squared(
                    state[:evaluation_state_dimension],
                    truth_state[:evaluation_state_dimension],
                    covariance[:evaluation_state_dimension, :evaluation_state_dimension],
                )
            )
        finite_nees = np.asarray(nees_values, dtype=float)
        finite_nees = finite_nees[np.isfinite(finite_nees)]
        mean_nees = float(np.mean(finite_nees)) if finite_nees.size else float("nan")

    accepted_fraction = (
        float(np.mean(result.accepted_updates)) if result.accepted_updates.size else 0.0
    )
    failure_fraction = 1.0 - accepted_fraction if failure_fraction_from_rejections else 0.0
    stability = assess_ukf_operational_stability(result)
    merged_metadata = {
        "num_updates": str(int(states.shape[0])),
        "accepted_update_fraction": f"{accepted_fraction:.12g}",
        "max_nis": f"{stability.max_normalized_innovation_squared:.12g}",
        "stable": str(stability.stable),
        "robust_reweighted_fraction": f"{stability.robust_reweighted_fraction:.12g}",
    }
    if result.measurement_noise_scales.size:
        merged_metadata["mean_measurement_noise_scale"] = f"{float(np.mean(result.measurement_noise_scales)):.12g}"
    if result.process_noise_scales.size:
        merged_metadata["final_process_noise_scale"] = f"{float(result.process_noise_scales[-1]):.12g}"
    if metadata:
        merged_metadata.update(metadata)

    nis_values = np.asarray(result.normalized_innovation_squared, dtype=float)
    finite_nis = nis_values[np.isfinite(nis_values)]
    return AdaptiveTuningEvaluation(
        label=label,
        mean_nis=float(np.mean(finite_nis)) if finite_nis.size else float("nan"),
        measurement_dimension=measurement_dim,
        rms_position_error_m=rms_position_error,
        mean_nees=mean_nees,
        state_dimension=evaluation_state_dimension,
        failure_fraction=failure_fraction,
        metadata=merged_metadata,
    )


def adaptive_evaluations_from_ukf_results(
    labeled_results: Mapping[str, LunarUKFResult] | Sequence[tuple[str, LunarUKFResult]],
    truth_state_history: ArrayLike,
    *,
    measurement_dimension: int | None = None,
    state_dimension: int | None = None,
    include_nees: bool = True,
    failure_fraction_from_rejections: bool = True,
) -> tuple[AdaptiveTuningEvaluation, ...]:
    """Convert labeled UKF results into comparable adaptive tuning evaluations."""
    items = labeled_results.items() if isinstance(labeled_results, Mapping) else labeled_results
    return tuple(
        adaptive_evaluation_from_ukf_result(
            label,
            result,
            truth_state_history,
            measurement_dimension=measurement_dimension,
            state_dimension=state_dimension,
            include_nees=include_nees,
            failure_fraction_from_rejections=failure_fraction_from_rejections,
        )
        for label, result in items
    )


def _log_ratio_penalty(value: float, expected: int) -> float:
    if not np.isfinite(value) or value <= 0.0:
        return float("inf")
    return float(abs(np.log(value / float(expected))))


def _relative_penalty_improvement(reference_penalty: float, candidate_penalty: float) -> float:
    if not np.isfinite(reference_penalty) or not np.isfinite(candidate_penalty):
        return 0.0
    if reference_penalty <= 0.0:
        return 0.0
    return float((reference_penalty - candidate_penalty) / reference_penalty)


def _relative_increase(reference_value: float, candidate_value: float) -> float:
    if not np.isfinite(reference_value) or not np.isfinite(candidate_value):
        return 0.0
    if reference_value <= 0.0:
        return float("inf") if candidate_value > reference_value else 0.0
    return float((candidate_value - reference_value) / reference_value)


def _finite_metric_values(trials, metric_name: str) -> np.ndarray:
    values = np.asarray(
        [trial.metrics[metric_name] for trial in trials if metric_name in trial.metrics],
        dtype=float,
    )
    return values[np.isfinite(values)]


def _coerce_truth_history(truth_state_history: ArrayLike, num_updates: int) -> np.ndarray:
    truth = np.asarray(truth_state_history, dtype=float)
    if truth.ndim == 1:
        return np.repeat(truth.reshape(1, -1), num_updates, axis=0)
    if truth.ndim != 2:
        raise ValueError("truth_state_history must be a state vector or a 2-D history.")
    if truth.shape[0] != num_updates:
        raise ValueError("truth_state_history length must match the UKF update history.")
    return truth


def _infer_measurement_dimension(result: LunarUKFResult) -> int:
    if result.innovations.ndim == 2 and result.innovations.shape[1] > 0:
        return int(result.innovations.shape[1])
    if result.predicted_measurements.ndim == 2 and result.predicted_measurements.shape[1] > 0:
        return int(result.predicted_measurements.shape[1])
    raise ValueError("measurement_dimension could not be inferred from the UKF result.")
