"""Process-noise sweep helpers for formal handoff tuning."""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProcessNoiseCase:
    label: str
    covariance: np.ndarray | None
    sigma_pos_m: float | None = None
    sigma_vel_mps: float | None = None


@dataclass(frozen=True)
class ProcessNoiseEvaluation:
    label: str
    metrics: Mapping[str, float]
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ProcessNoiseSelection:
    metric_name: str
    best_label: str
    best_metric_value: float
    minimize: bool


Q_TUNING_DEFAULT_CASES: tuple[ProcessNoiseCase, ...] = (
    ProcessNoiseCase("Q0", None, None, None),
    ProcessNoiseCase("Qsmall", np.diag([0.1**2] * 3 + [1e-5**2] * 3), 0.1, 1e-5),
    ProcessNoiseCase("Qmedium", np.diag([1.0**2] * 3 + [1e-4**2] * 3), 1.0, 1e-4),
    ProcessNoiseCase("Qlarge", np.diag([10.0**2] * 3 + [1e-3**2] * 3), 10.0, 1e-3),
)


SweepEvaluator = Callable[[ProcessNoiseCase], Mapping[str, Any] | ProcessNoiseEvaluation]


def state_process_noise_covariance(sigma_pos_m: float, sigma_vel_mps: float) -> np.ndarray:
    """Build a 6x6 diagonal state process-noise covariance."""
    if sigma_pos_m <= 0.0 or sigma_vel_mps <= 0.0:
        raise ValueError("sigma_pos_m and sigma_vel_mps must be positive.")
    return np.diag([float(sigma_pos_m) ** 2] * 3 + [float(sigma_vel_mps) ** 2] * 3)


def default_process_noise_cases(*, include_zero: bool = True) -> tuple[ProcessNoiseCase, ...]:
    """Return the canonical process-noise grid used by the handoff report."""
    if include_zero:
        return Q_TUNING_DEFAULT_CASES
    return Q_TUNING_DEFAULT_CASES[1:]


def run_process_noise_sweep(
    cases: Sequence[ProcessNoiseCase],
    evaluator: SweepEvaluator,
) -> tuple[ProcessNoiseEvaluation, ...]:
    """Evaluate each process-noise case with a caller-provided function."""
    _validate_case_labels(cases)
    results: list[ProcessNoiseEvaluation] = []
    for case in cases:
        raw = evaluator(case)
        results.append(_coerce_evaluation(case, raw))
    return tuple(results)


def select_best_process_noise(
    evaluations: Sequence[ProcessNoiseEvaluation],
    metric_name: str,
    *,
    minimize: bool = True,
) -> ProcessNoiseSelection:
    """Select the best process-noise case by one finite metric."""
    candidates = []
    for evaluation in evaluations:
        if metric_name not in evaluation.metrics:
            continue
        value = float(evaluation.metrics[metric_name])
        if np.isfinite(value):
            candidates.append((value, evaluation.label))
    if not candidates:
        raise ValueError(f"No finite metric values found for {metric_name!r}.")
    value, label = min(candidates) if minimize else max(candidates)
    return ProcessNoiseSelection(
        metric_name=metric_name,
        best_label=label,
        best_metric_value=float(value),
        minimize=minimize,
    )


def write_process_noise_sweep_csv(evaluations: Sequence[ProcessNoiseEvaluation], output_path) -> Path:
    """Write process-noise sweep evaluations to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluations = tuple(evaluations)
    metric_names = sorted({name for evaluation in evaluations for name in evaluation.metrics})
    metadata_names = sorted({name for evaluation in evaluations for name in evaluation.metadata})

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label"] + [f"metric_{name}" for name in metric_names] + [f"metadata_{name}" for name in metadata_names])
        for evaluation in evaluations:
            writer.writerow(
                [evaluation.label]
                + [evaluation.metrics.get(name, "") for name in metric_names]
                + [evaluation.metadata.get(name, "") for name in metadata_names]
            )
    return output_path


def _validate_case_labels(cases: Sequence[ProcessNoiseCase]) -> None:
    labels = [case.label for case in cases]
    if len(labels) != len(set(labels)):
        raise ValueError("Process-noise case labels must be unique.")


def _coerce_evaluation(
    case: ProcessNoiseCase,
    raw: Mapping[str, Any] | ProcessNoiseEvaluation,
) -> ProcessNoiseEvaluation:
    if isinstance(raw, ProcessNoiseEvaluation):
        return raw
    if not isinstance(raw, Mapping):
        raise TypeError("sweep evaluator must return a mapping or ProcessNoiseEvaluation.")
    metadata_raw = raw.get("metadata", {})
    if metadata_raw is None:
        metadata_raw = {}
    if not isinstance(metadata_raw, Mapping):
        raise TypeError("metadata must be a mapping when provided.")
    metrics = {str(name): float(value) for name, value in raw.items() if name != "metadata"}
    metadata = {str(name): str(value) for name, value in metadata_raw.items()}
    return ProcessNoiseEvaluation(label=case.label, metrics=metrics, metadata=metadata)
