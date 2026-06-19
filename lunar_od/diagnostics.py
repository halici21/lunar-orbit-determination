"""Residual and boundary-jump diagnostics for OD experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class ResidualDiagnostics:
    count: int
    dof: int
    mean: float
    rms: float
    std: float
    max_abs: float
    whitened_rms: float
    chi_square: float
    reduced_chi_square: float
    mahalanobis_norm: float
    lag1_autocorrelation: float


@dataclass(frozen=True)
class BoundaryJumpDiagnostics:
    count: int
    jump_norms: np.ndarray
    rms_jump: float
    median_jump: float
    max_jump: float


@dataclass(frozen=True)
class StateBiasCorrelationDiagnostics:
    num_state: int
    num_bias: int
    correlation_matrix: np.ndarray
    max_abs_correlation: float
    state_rss_sigma: float
    bias_rss_sigma: float


@dataclass(frozen=True)
class ConsistencyDiagnostics:
    dimension: int
    statistic: float
    normalized_statistic: float
    sigma_norm: float
    whitened: np.ndarray


@dataclass(frozen=True)
class InnovationWhitenessDiagnostics:
    count: int
    dimension: int
    whitened_innovations: np.ndarray
    component_lag1_autocorrelation: np.ndarray
    mean_abs_lag1_autocorrelation: float
    max_abs_lag1_autocorrelation: float


@dataclass(frozen=True)
class ConvergenceDiagnostics:
    stop_reason: str
    category: str
    converged: bool
    converged_by_step_norm: bool
    converged_by_cost_stability: bool
    max_iter_reached: bool
    singular_or_ill_conditioned: bool
    rank_deficient: bool
    outlier_rejected: bool
    finite_final_cost: bool


def analyze_convergence(
    stop_reason: str,
    *,
    stats=None,
    rank: int | None = None,
    expected_rank: int | None = None,
    condition_number: float | None = None,
    condition_threshold: float = 1e14,
    rejected_components: int | None = None,
    final_cost: float | None = None,
) -> ConvergenceDiagnostics:
    """Classify estimator termination into report-friendly flags.

    The raw estimators currently emit compact stop strings such as
    ``Converged``, ``J-Stab``, ``MaxIter`` and ``Singular``. This helper keeps
    that raw reason but also exposes stable boolean columns for CSV reporting.
    ``stats`` may be an ``EstimatorStats`` instance or any object with matching
    attributes.
    """
    reason = str(stop_reason)
    if stats is not None:
        rank = getattr(stats, "rank", rank)
        condition_number = getattr(stats, "condition_number", condition_number)
        rejected_components = getattr(stats, "rejected_components", rejected_components)
        final_cost = getattr(stats, "final_cost", final_cost)

    normalized = reason.strip().lower().replace("_", "-")
    converged_by_step = normalized == "converged"
    converged_by_cost = normalized in {"j-stab", "cost-stable", "cost_stable"}
    max_iter = normalized in {"maxiter", "max-iter", "maximum-iterations"}
    singular_reason = normalized in {"singular", "rank-deficient", "ill-conditioned"}

    cond_value = float("nan") if condition_number is None else float(condition_number)
    ill_conditioned = bool(np.isfinite(cond_value) and cond_value > condition_threshold)
    rank_deficient = False
    if expected_rank is not None and rank is not None:
        rank_deficient = int(rank) < int(expected_rank)

    singular_or_ill_conditioned = bool(singular_reason or ill_conditioned or rank_deficient)
    converged = bool((converged_by_step or converged_by_cost) and not singular_or_ill_conditioned)
    if converged_by_step:
        category = "converged_step"
    elif converged_by_cost:
        category = "converged_cost_stability"
    elif max_iter:
        category = "max_iter"
    elif singular_or_ill_conditioned:
        category = "singular_or_ill_conditioned"
    else:
        category = "unknown"

    cost_value = float("nan") if final_cost is None else float(final_cost)
    return ConvergenceDiagnostics(
        stop_reason=reason,
        category=category,
        converged=converged,
        converged_by_step_norm=converged_by_step,
        converged_by_cost_stability=converged_by_cost,
        max_iter_reached=bool(max_iter),
        singular_or_ill_conditioned=singular_or_ill_conditioned,
        rank_deficient=rank_deficient,
        outlier_rejected=bool((0 if rejected_components is None else int(rejected_components)) > 0),
        finite_final_cost=bool(np.isfinite(cost_value)),
    )

def analyze_residuals(
    residuals: ArrayLike,
    *,
    sigma: ArrayLike | None = None,
    covariance: ArrayLike | None = None,
    num_solve_for: int = 0,
) -> ResidualDiagnostics:
    """Compute scalar and whitened residual diagnostics.

    Provide either per-component ``sigma`` or a full residual covariance.
    If neither is provided, whitened metrics use the raw residual vector.
    """
    residual = np.asarray(residuals, dtype=float).reshape(-1)
    if residual.size == 0:
        return _empty_residual_diagnostics()
    if sigma is not None and covariance is not None:
        raise ValueError("Provide either sigma or covariance, not both.")

    whitened = _whiten_residuals(residual, sigma=sigma, covariance=covariance)
    count = int(residual.size)
    dof = max(count - int(num_solve_for), 0)
    chi_square = float(np.dot(whitened, whitened))
    return ResidualDiagnostics(
        count=count,
        dof=dof,
        mean=float(np.mean(residual)),
        rms=float(np.sqrt(np.mean(residual**2))),
        std=float(np.std(residual)),
        max_abs=float(np.max(np.abs(residual))),
        whitened_rms=float(np.sqrt(np.mean(whitened**2))),
        chi_square=chi_square,
        reduced_chi_square=float("nan") if dof == 0 else chi_square / dof,
        mahalanobis_norm=float(np.sqrt(chi_square)),
        lag1_autocorrelation=_lag1_autocorrelation(whitened),
    )


def analyze_innovation_consistency(innovation: ArrayLike, innovation_covariance: ArrayLike) -> ConsistencyDiagnostics:
    """Compute normalized innovation squared (NIS) diagnostics.

    ``innovation`` follows the project residual convention, observed minus
    predicted measurement. ``innovation_covariance`` is the predicted
    measurement covariance ``S`` from an EKF/UKF update.
    """
    return _quadratic_consistency(innovation, innovation_covariance)


def analyze_estimation_error_consistency(state_error: ArrayLike, state_covariance: ArrayLike) -> ConsistencyDiagnostics:
    """Compute normalized estimation error squared (NEES) diagnostics.

    ``state_error`` should be estimated state minus truth state for synthetic
    cases where truth is available. ``state_covariance`` is the filter/posterior
    state covariance in the same state ordering.
    """
    return _quadratic_consistency(state_error, state_covariance)


def analyze_innovation_whiteness(
    innovations: ArrayLike,
    innovation_covariances: ArrayLike,
) -> InnovationWhitenessDiagnostics:
    """Whiten an innovation sequence and summarize lag-1 temporal correlation."""
    values = np.asarray(innovations, dtype=float)
    covariances = np.asarray(innovation_covariances, dtype=float)
    if values.ndim != 2:
        raise ValueError("innovations must be a 2-D array.")
    if covariances.shape != (values.shape[0], values.shape[1], values.shape[1]):
        raise ValueError("innovation_covariances must have shape (N, M, M).")

    whitened = np.zeros_like(values)
    for idx in range(values.shape[0]):
        whitened[idx, :] = _whiten_residuals(values[idx], sigma=None, covariance=covariances[idx])
    correlations = np.array(
        [_lag1_autocorrelation(whitened[:, col]) for col in range(values.shape[1])],
        dtype=float,
    )
    finite = np.abs(correlations[np.isfinite(correlations)])
    return InnovationWhitenessDiagnostics(
        count=int(values.shape[0]),
        dimension=int(values.shape[1]),
        whitened_innovations=whitened,
        component_lag1_autocorrelation=correlations,
        mean_abs_lag1_autocorrelation=float(np.mean(finite)) if finite.size else float("nan"),
        max_abs_lag1_autocorrelation=float(np.max(finite)) if finite.size else float("nan"),
    )


def analyze_boundary_jumps(previous_values: ArrayLike, next_values: ArrayLike) -> BoundaryJumpDiagnostics:
    """Summarize Euclidean jumps between adjacent arc boundary values."""
    previous = _as_2d(previous_values)
    next_ = _as_2d(next_values)
    if previous.shape != next_.shape:
        raise ValueError("previous_values and next_values must have matching shapes.")
    if previous.size == 0:
        return BoundaryJumpDiagnostics(
            count=0,
            jump_norms=np.zeros(0, dtype=float),
            rms_jump=float("nan"),
            median_jump=float("nan"),
            max_jump=float("nan"),
        )

    jump_norms = np.linalg.norm(next_ - previous, axis=1)
    return BoundaryJumpDiagnostics(
        count=int(jump_norms.size),
        jump_norms=jump_norms,
        rms_jump=float(np.sqrt(np.mean(jump_norms**2))),
        median_jump=float(np.median(jump_norms)),
        max_jump=float(np.max(jump_norms)),
    )


def analyze_state_bias_correlation(covariance: ArrayLike, *, num_state: int = 6) -> StateBiasCorrelationDiagnostics:
    """Summarize normalized covariance coupling between dynamic state and bias states."""
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    if num_state <= 0 or num_state >= cov.shape[0]:
        raise ValueError("num_state must leave at least one bias state.")

    diag = np.diag(cov)
    if np.any(diag < 0.0):
        raise ValueError("covariance diagonal must be non-negative.")
    sigma = np.sqrt(np.clip(diag, 0.0, None))
    denom = sigma[:num_state, None] * sigma[None, num_state:]
    cross = cov[:num_state, num_state:]
    correlation = np.divide(cross, denom, out=np.zeros_like(cross), where=denom > 0.0)
    return StateBiasCorrelationDiagnostics(
        num_state=int(num_state),
        num_bias=int(cov.shape[0] - num_state),
        correlation_matrix=correlation,
        max_abs_correlation=float(np.max(np.abs(correlation))) if correlation.size else float("nan"),
        state_rss_sigma=float(np.sqrt(np.sum(sigma[:num_state] ** 2))),
        bias_rss_sigma=float(np.sqrt(np.sum(sigma[num_state:] ** 2))),
    )


def _quadratic_consistency(vector: ArrayLike, covariance: ArrayLike) -> ConsistencyDiagnostics:
    residual = np.asarray(vector, dtype=float).reshape(-1)
    if residual.size == 0:
        raise ValueError("consistency vector must not be empty.")
    whitened = _whiten_residuals(residual, sigma=None, covariance=covariance)
    statistic = float(np.dot(whitened, whitened))
    dimension = int(residual.size)
    return ConsistencyDiagnostics(
        dimension=dimension,
        statistic=statistic,
        normalized_statistic=statistic / dimension,
        sigma_norm=float(np.sqrt(statistic)),
        whitened=whitened,
    )


def _whiten_residuals(
    residual: np.ndarray,
    *,
    sigma: ArrayLike | None,
    covariance: ArrayLike | None,
) -> np.ndarray:
    if sigma is not None:
        sigma_vec = np.asarray(sigma, dtype=float).reshape(-1)
        if sigma_vec.shape != residual.shape:
            raise ValueError("sigma must match residuals shape.")
        if np.any(sigma_vec <= 0.0):
            raise ValueError("sigma values must be positive.")
        return residual / sigma_vec

    if covariance is not None:
        cov = np.asarray(covariance, dtype=float)
        if cov.shape != (residual.size, residual.size):
            raise ValueError("covariance must be square with residual dimension.")
        try:
            chol = np.linalg.cholesky(cov)
            return np.linalg.solve(chol, residual)
        except np.linalg.LinAlgError as exc:
            raise ValueError("covariance must be positive definite.") from exc

    return residual.copy()


def _lag1_autocorrelation(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size < 2:
        return float("nan")
    centered = values - float(np.mean(values))
    denom = float(np.dot(centered, centered))
    if denom <= np.finfo(float).eps:
        return float("nan")
    return float(np.dot(centered[:-1], centered[1:]) / denom)


def _as_2d(values: ArrayLike) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("values must be one- or two-dimensional.")
    return array


def _empty_residual_diagnostics() -> ResidualDiagnostics:
    return ResidualDiagnostics(
        count=0,
        dof=0,
        mean=float("nan"),
        rms=float("nan"),
        std=float("nan"),
        max_abs=float("nan"),
        whitened_rms=float("nan"),
        chi_square=float("nan"),
        reduced_chi_square=float("nan"),
        mahalanobis_norm=float("nan"),
        lag1_autocorrelation=float("nan"),
    )
