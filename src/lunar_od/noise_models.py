"""Seeded multivariate measurement-noise models for OD robustness campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

NoiseModel = Literal["gaussian", "correlated_gaussian", "ar1", "student_t"]


@dataclass(frozen=True)
class MeasurementNoiseConfig:
    model: NoiseModel = "gaussian"
    ar1_coefficient: float = 0.0
    student_t_dof: float = 5.0
    # WP4.5 (EST-15): when set, "ar1" is a first-order Gauss-Markov process per
    # station in TIME, rho_k = exp(-dt_k / tau), instead of a chain over rows.
    correlation_time_s: float | None = None

    def __post_init__(self) -> None:
        if self.model not in {"gaussian", "correlated_gaussian", "ar1", "student_t"}:
            raise ValueError(f"Unsupported measurement noise model: {self.model}")
        if not (-1.0 < self.ar1_coefficient < 1.0):
            raise ValueError("ar1_coefficient must be strictly between -1 and 1.")
        if self.student_t_dof <= 2.0:
            raise ValueError("student_t_dof must be greater than 2 for finite covariance.")
        if self.correlation_time_s is not None and not (
            np.isfinite(self.correlation_time_s) and self.correlation_time_s > 0.0
        ):
            raise ValueError("correlation_time_s must be finite and positive.")


def generate_measurement_noise(
    num_samples: int,
    covariance: ArrayLike,
    *,
    rng: np.random.Generator,
    config: MeasurementNoiseConfig | None = None,
    times: ArrayLike | None = None,
    groups: ArrayLike | None = None,
) -> np.ndarray:
    """Generate zero-mean noise with the requested nominal stationary covariance.

    For ``model="ar1"`` with ``correlation_time_s`` set, ``times`` (s, non-decreasing)
    is required and ``groups`` (e.g. station ids, default one group) separates
    independent processes; without ``correlation_time_s`` the legacy row-index AR(1)
    with ``ar1_coefficient`` is used.
    """
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative.")
    cfg = config or MeasurementNoiseConfig()
    covariance_array = np.asarray(covariance, dtype=float)
    if covariance_array.ndim == 1:
        covariance_array = np.diag(covariance_array)
    if covariance_array.ndim != 2 or covariance_array.shape[0] != covariance_array.shape[1]:
        raise ValueError("covariance must be square or a diagonal vector.")
    covariance_array = 0.5 * (covariance_array + covariance_array.T)
    root = _covariance_root(covariance_array)
    dimension = covariance_array.shape[0]
    if num_samples == 0:
        return np.zeros((0, dimension), dtype=float)

    if cfg.model in {"gaussian", "correlated_gaussian"}:
        return rng.standard_normal((num_samples, dimension)) @ root.T
    if cfg.model == "student_t":
        gaussian = rng.standard_normal((num_samples, dimension)) @ root.T
        chi_square = rng.chisquare(cfg.student_t_dof, size=num_samples)
        covariance_scale = np.sqrt((cfg.student_t_dof - 2.0) / chi_square)
        return gaussian * covariance_scale[:, None]

    if cfg.correlation_time_s is not None:
        rho, first = _gauss_markov_links(times, groups, cfg.correlation_time_s, num_samples)
        innovations = rng.standard_normal((num_samples, dimension)) @ root.T
        values = np.zeros_like(innovations)
        for idx in range(num_samples):
            prev = first[idx]
            if prev < 0:
                values[idx] = innovations[idx]
            else:
                values[idx] = rho[idx] * values[prev] + np.sqrt(1.0 - rho[idx] ** 2) * innovations[idx]
        return values

    innovations = rng.standard_normal((num_samples, dimension)) @ root.T
    values = np.zeros_like(innovations)
    rho = cfg.ar1_coefficient
    values[0] = innovations[0]
    innovation_scale = np.sqrt(1.0 - rho**2)
    for idx in range(1, num_samples):
        values[idx] = rho * values[idx - 1] + innovation_scale * innovations[idx]
    return values


def _covariance_root(covariance: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.min(eigenvalues) < -1e-12:
        raise ValueError("covariance must be positive semidefinite.")
    return (eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))) @ eigenvectors.T


def _gauss_markov_links(
    times: ArrayLike | None, groups: ArrayLike | None, tau: float, num_samples: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Per sample: rho to the previous sample of the same group, and that sample's index
    (-1 for the first of a group)."""
    if times is None:
        raise ValueError("time-based AR(1) noise needs the sample times.")
    t = np.asarray(times, dtype=float).reshape(-1)
    if num_samples is not None and t.size != num_samples:
        raise ValueError("times must have one entry per sample.")
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) < 0.0):
        raise ValueError("times must be finite and non-decreasing.")
    g = np.zeros(t.size, dtype=int) if groups is None else np.asarray(groups).reshape(-1)
    if g.size != t.size:
        raise ValueError("groups must have one entry per sample.")
    rho = np.zeros(t.size)
    prev = np.full(t.size, -1, dtype=int)
    last: dict = {}
    for idx, key in enumerate(g.tolist()):
        if key in last:
            j = last[key]
            dt = t[idx] - t[j]
            if dt <= 0.0:
                raise ValueError("two samples of one group share an epoch; rho would be 1.")
            prev[idx] = j
            rho[idx] = np.exp(-dt / tau)
        last[key] = idx
    return rho, prev


def gauss_markov_covariance(times: ArrayLike, groups: ArrayLike | None, tau: float) -> np.ndarray:
    """Unit-variance covariance exp(-|t_i - t_j| / tau) within a group, 0 across groups."""
    t = np.asarray(times, dtype=float).reshape(-1)
    g = np.zeros(t.size, dtype=int) if groups is None else np.asarray(groups).reshape(-1)
    same = g[:, None] == g[None, :]
    return np.where(same, np.exp(-np.abs(t[:, None] - t[None, :]) / tau), 0.0)


def gauss_markov_whitening_matrix(times: ArrayLike, groups: ArrayLike | None, tau: float) -> np.ndarray:
    """Prais-Winsten whitening W (lower triangular) with W Sigma W^T = I for the
    unit-variance Gauss-Markov covariance of :func:`gauss_markov_covariance`.

    Row k: e_k for the first sample of a group, else
    (e_k - rho_k e_prev) / sqrt(1 - rho_k^2).  Apply as W @ (residual / sigma).
    """
    rho, prev = _gauss_markov_links(times, groups, tau)
    w = np.eye(rho.size)
    has_prev = prev >= 0
    scale = 1.0 / np.sqrt(1.0 - rho[has_prev] ** 2)
    rows = np.flatnonzero(has_prev)
    w[rows, rows] = scale
    w[rows, prev[has_prev]] = -rho[has_prev] * scale
    return w


def ar1_variance_inflation(times: ArrayLike, groups: ArrayLike | None, tau: float) -> float:
    """n Var(mean) under the unit-variance Gauss-Markov covariance: the factor by which
    white weighting understates the variance of an average (effective sample size
    n / factor).  O(n): 1^T Sigma 1 = n + 2 sum_i c_i with c_i = rho_i (1 + c_prev)."""
    rho, prev = _gauss_markov_links(times, groups, tau)
    c = np.zeros(rho.size)
    for idx in range(rho.size):
        if prev[idx] >= 0:
            c[idx] = rho[idx] * (1.0 + c[prev[idx]])
    return float((rho.size + 2.0 * c.sum()) / rho.size)
