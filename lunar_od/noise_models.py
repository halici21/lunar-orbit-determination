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

    def __post_init__(self) -> None:
        if self.model not in {"gaussian", "correlated_gaussian", "ar1", "student_t"}:
            raise ValueError(f"Unsupported measurement noise model: {self.model}")
        if not (-1.0 < self.ar1_coefficient < 1.0):
            raise ValueError("ar1_coefficient must be strictly between -1 and 1.")
        if self.student_t_dof <= 2.0:
            raise ValueError("student_t_dof must be greater than 2 for finite covariance.")


def generate_measurement_noise(
    num_samples: int,
    covariance: ArrayLike,
    *,
    rng: np.random.Generator,
    config: MeasurementNoiseConfig | None = None,
) -> np.ndarray:
    """Generate zero-mean noise with the requested nominal stationary covariance."""
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
