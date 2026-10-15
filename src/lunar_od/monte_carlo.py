"""Seeded Monte Carlo trial helpers and aggregation."""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MonteCarloTrialSpec:
    trial_id: int
    seed: int


@dataclass(frozen=True)
class MonteCarloTrialResult:
    trial_id: int
    seed: int
    success: bool
    metrics: Mapping[str, float]
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MonteCarloMetricSummary:
    metric_name: str
    count: int
    mean: float
    std: float
    median: float
    p05: float
    p95: float
    min_value: float
    max_value: float


@dataclass(frozen=True)
class MonteCarloSummary:
    num_trials: int
    success_count: int
    success_fraction: float
    metric_summaries: tuple[MonteCarloMetricSummary, ...]


@dataclass(frozen=True)
class MonteCarloCase:
    label: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MonteCarloCaseResult:
    case: MonteCarloCase
    trials: tuple[MonteCarloTrialResult, ...]
    summary: MonteCarloSummary


TrialFunction = Callable[[MonteCarloTrialSpec, np.random.Generator], Mapping[str, Any] | MonteCarloTrialResult]
CampaignTrialFunction = Callable[
    [MonteCarloCase, MonteCarloTrialSpec, np.random.Generator],
    Mapping[str, Any] | MonteCarloTrialResult,
]
ProgressFunction = Callable[[int, int, str, int], None]


def make_trial_specs(num_trials: int, base_seed: int = 0) -> tuple[MonteCarloTrialSpec, ...]:
    """Create deterministic one-based trial specs."""
    if num_trials < 0:
        raise ValueError("num_trials must be non-negative.")
    seed0 = int(base_seed)
    modulus = 2**32 - 1
    return tuple(
        MonteCarloTrialSpec(trial_id=trial_id, seed=(seed0 + trial_id - 1) % modulus)
        for trial_id in range(1, num_trials + 1)
    )


def run_monte_carlo_trials(
    num_trials: int,
    trial_fn: TrialFunction,
    *,
    base_seed: int = 0,
    continue_on_error: bool = False,
) -> tuple[MonteCarloTrialResult, ...]:
    """Run a seeded trial callback and collect metric rows."""
    results: list[MonteCarloTrialResult] = []
    for spec in make_trial_specs(num_trials, base_seed):
        rng = np.random.default_rng(spec.seed)
        try:
            raw_result = trial_fn(spec, rng)
            results.append(_coerce_trial_result(spec, raw_result))
        except Exception as exc:
            if not continue_on_error:
                raise
            results.append(
                MonteCarloTrialResult(
                    trial_id=spec.trial_id,
                    seed=spec.seed,
                    success=False,
                    metrics={},
                    metadata={"error": f"{type(exc).__name__}: {exc}"},
                )
            )
    return tuple(results)


def run_monte_carlo_campaign(
    cases: Sequence[MonteCarloCase],
    num_trials: int,
    trial_fn: CampaignTrialFunction,
    *,
    base_seed: int = 0,
    max_workers: int = 1,
    continue_on_error: bool = False,
    progress_fn: ProgressFunction | None = None,
) -> tuple[MonteCarloCaseResult, ...]:
    """Run a deterministic case-by-trial grid, optionally in parallel.

    Thread workers are used because campaign callbacks commonly call NumPy and
    SciPy routines that release the GIL, while avoiding process pickling limits
    for local ephemeris and station callbacks.
    """
    cases = tuple(cases)
    _validate_campaign(cases, num_trials, max_workers)
    jobs = [
        (case_idx, case, spec)
        for case_idx, case in enumerate(cases)
        for spec in make_trial_specs(num_trials, _case_seed_base(base_seed, case_idx, num_trials))
    ]
    total = len(jobs)
    completed = 0
    collected: dict[tuple[int, int], MonteCarloTrialResult] = {}

    def execute(case_idx: int, case: MonteCarloCase, spec: MonteCarloTrialSpec):
        rng = np.random.default_rng(spec.seed)
        try:
            result = _coerce_trial_result(spec, trial_fn(case, spec, rng))
        except Exception as exc:
            if not continue_on_error:
                raise
            result = MonteCarloTrialResult(
                trial_id=spec.trial_id,
                seed=spec.seed,
                success=False,
                metrics={},
                metadata={"error": f"{type(exc).__name__}: {exc}"},
            )
        metadata = {"case": case.label, **result.metadata}
        return case_idx, MonteCarloTrialResult(
            trial_id=result.trial_id,
            seed=result.seed,
            success=result.success,
            metrics=result.metrics,
            metadata=metadata,
        )

    if max_workers == 1:
        for case_idx, case, spec in jobs:
            result_case_idx, result = execute(case_idx, case, spec)
            collected[(result_case_idx, result.trial_id)] = result
            completed += 1
            if progress_fn is not None:
                progress_fn(completed, total, case.label, result.trial_id)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(execute, case_idx, case, spec): (case_idx, case, spec)
                for case_idx, case, spec in jobs
            }
            for future in as_completed(futures):
                case_idx, case, _ = futures[future]
                result_case_idx, result = future.result()
                collected[(result_case_idx, result.trial_id)] = result
                completed += 1
                if progress_fn is not None:
                    progress_fn(completed, total, case.label, result.trial_id)

    case_results = []
    for case_idx, case in enumerate(cases):
        trials = tuple(collected[(case_idx, trial_id)] for trial_id in range(1, num_trials + 1))
        case_results.append(
            MonteCarloCaseResult(
                case=case,
                trials=trials,
                summary=summarize_monte_carlo_trials(trials),
            )
        )
    return tuple(case_results)


def summarize_monte_carlo_trials(
    results: Sequence[MonteCarloTrialResult],
    *,
    metric_names: Sequence[str] | None = None,
) -> MonteCarloSummary:
    """Aggregate finite numeric metrics over Monte Carlo trial results."""
    results = tuple(results)
    success_count = sum(1 for result in results if result.success)
    if metric_names is None:
        names = sorted({name for result in results for name in result.metrics})
    else:
        names = list(metric_names)

    summaries = tuple(_summarize_metric(name, results) for name in names)
    return MonteCarloSummary(
        num_trials=len(results),
        success_count=success_count,
        success_fraction=0.0 if not results else success_count / len(results),
        metric_summaries=summaries,
    )


def write_monte_carlo_trials_csv(results: Sequence[MonteCarloTrialResult], output_path) -> Path:
    """Write per-trial Monte Carlo metrics and metadata to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = tuple(results)
    metric_names = sorted({name for result in results for name in result.metrics})
    metadata_names = sorted({name for result in results for name in result.metadata})

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["trial_id", "seed", "success"]
            + [f"metric_{name}" for name in metric_names]
            + [f"metadata_{name}" for name in metadata_names]
        )
        for result in results:
            writer.writerow(
                [result.trial_id, result.seed, result.success]
                + [result.metrics.get(name, "") for name in metric_names]
                + [result.metadata.get(name, "") for name in metadata_names]
            )
    return output_path


def write_monte_carlo_summary_csv(summary: MonteCarloSummary, output_path) -> Path:
    """Write aggregate Monte Carlo metric summaries to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["num_trials", summary.num_trials])
        writer.writerow(["success_count", summary.success_count])
        writer.writerow(["success_fraction", summary.success_fraction])
        writer.writerow([])
        writer.writerow(["metric", "count", "mean", "std", "median", "p05", "p95", "min", "max"])
        for metric in summary.metric_summaries:
            writer.writerow(
                [
                    metric.metric_name,
                    metric.count,
                    metric.mean,
                    metric.std,
                    metric.median,
                    metric.p05,
                    metric.p95,
                    metric.min_value,
                    metric.max_value,
                ]
            )
    return output_path


def _coerce_trial_result(
    spec: MonteCarloTrialSpec,
    raw_result: Mapping[str, Any] | MonteCarloTrialResult,
) -> MonteCarloTrialResult:
    if isinstance(raw_result, MonteCarloTrialResult):
        return raw_result
    if not isinstance(raw_result, Mapping):
        raise TypeError("trial_fn must return a mapping or MonteCarloTrialResult.")

    success = bool(raw_result.get("success", True))
    metadata_raw = raw_result.get("metadata", {})
    if metadata_raw is None:
        metadata_raw = {}
    if not isinstance(metadata_raw, Mapping):
        raise TypeError("metadata must be a mapping when provided.")

    metrics = {
        str(name): float(value)
        for name, value in raw_result.items()
        if name not in {"success", "metadata"}
    }
    metadata = {str(name): str(value) for name, value in metadata_raw.items()}
    return MonteCarloTrialResult(
        trial_id=spec.trial_id,
        seed=spec.seed,
        success=success,
        metrics=metrics,
        metadata=metadata,
    )


def _validate_campaign(cases: Sequence[MonteCarloCase], num_trials: int, max_workers: int) -> None:
    if num_trials < 0:
        raise ValueError("num_trials must be non-negative.")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive.")
    labels = [case.label for case in cases]
    if len(labels) != len(set(labels)):
        raise ValueError("Monte Carlo case labels must be unique.")


def _case_seed_base(base_seed: int, case_idx: int, num_trials: int) -> int:
    return (int(base_seed) + case_idx * max(1, num_trials)) % (2**32 - 1)


def _summarize_metric(metric_name: str, results: Sequence[MonteCarloTrialResult]) -> MonteCarloMetricSummary:
    values = np.array(
        [float(result.metrics[metric_name]) for result in results if metric_name in result.metrics],
        dtype=float,
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return MonteCarloMetricSummary(
            metric_name=metric_name,
            count=0,
            mean=float("nan"),
            std=float("nan"),
            median=float("nan"),
            p05=float("nan"),
            p95=float("nan"),
            min_value=float("nan"),
            max_value=float("nan"),
        )
    return MonteCarloMetricSummary(
        metric_name=metric_name,
        count=int(values.size),
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        median=float(np.median(values)),
        p05=float(np.percentile(values, 5.0)),
        p95=float(np.percentile(values, 95.0)),
        min_value=float(np.min(values)),
        max_value=float(np.max(values)),
    )
