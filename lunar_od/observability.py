"""Observability and conditioning helpers for Lunar OD arcs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .dynamics import propagate_augmented_state
from .measurements import (
    PassGeometry,
    compute_position_residuals_analytic,
    compute_range_rate_residuals_analytic,
    measurement_sigma_vector,
)
from .radiometrics import (
    RangeRatePhysicsConfig,
    range_rate_physics_config,
    two_way_counted_doppler_initial_state_jacobian,
)
from .scenarios import PreparedArc

MeasurementType = Literal["position", "range_rate"]


@dataclass(frozen=True)
class ObservabilityResult:
    measurement_type: MeasurementType
    num_observations: int
    num_parameters: int
    weighted_jacobian: np.ndarray
    fisher_information: np.ndarray
    singular_values: np.ndarray
    information_eigenvalues: np.ndarray
    rank: int
    condition_number: float
    weakest_direction: np.ndarray

    @property
    def rank_deficiency(self) -> int:
        return self.num_parameters - self.rank


@dataclass(frozen=True)
class ArcCombinationObservability:
    arc_ids: tuple[int, ...]
    observability: ObservabilityResult


@dataclass(frozen=True)
class BiasObservabilityPolicy:
    freeze_relative_information: float = 1e-12
    regularize_relative_information: float = 1e-5
    regularization_std: float = 1.0

    def __post_init__(self) -> None:
        if self.freeze_relative_information < 0.0:
            raise ValueError("freeze_relative_information must be non-negative.")
        if self.regularize_relative_information < self.freeze_relative_information:
            raise ValueError(
                "regularize_relative_information must not be below freeze_relative_information."
            )
        if self.regularization_std <= 0.0:
            raise ValueError("regularization_std must be positive.")


@dataclass(frozen=True)
class BiasObservabilityDecision:
    conditional_information: np.ndarray
    relative_information: np.ndarray
    active_state_indices: tuple[int, ...]
    regularized_state_indices: tuple[int, ...]
    frozen_state_indices: tuple[int, ...]
    regularization_std_by_state: dict[int, float]


def decide_bias_state_handling(
    observability: ObservabilityResult,
    *,
    dynamic_state_size: int = 6,
    policy: BiasObservabilityPolicy | None = None,
) -> BiasObservabilityDecision:
    """Classify bias states using information conditional on the dynamic state."""
    policy = policy or BiasObservabilityPolicy()
    if not (0 < dynamic_state_size <= observability.num_parameters):
        raise ValueError("dynamic_state_size must be within the parameter vector.")
    num_bias = observability.num_parameters - dynamic_state_size
    if num_bias == 0:
        return BiasObservabilityDecision(
            conditional_information=np.zeros(0),
            relative_information=np.zeros(0),
            active_state_indices=(),
            regularized_state_indices=(),
            frozen_state_indices=(),
            regularization_std_by_state={},
        )

    fisher = observability.fisher_information
    f_xx = fisher[:dynamic_state_size, :dynamic_state_size]
    f_xb = fisher[:dynamic_state_size, dynamic_state_size:]
    f_bb = fisher[dynamic_state_size:, dynamic_state_size:]
    conditional = _symmetrize(f_bb - f_xb.T @ np.linalg.pinv(f_xx) @ f_xb)
    information = np.clip(np.diag(conditional), 0.0, None)
    scale = max(float(np.max(information)), np.finfo(float).tiny)
    relative = information / scale

    active = []
    regularized = []
    frozen = []
    for bias_idx, relative_value in enumerate(relative):
        state_idx = dynamic_state_size + bias_idx
        if relative_value <= policy.freeze_relative_information:
            frozen.append(state_idx)
        elif relative_value <= policy.regularize_relative_information:
            regularized.append(state_idx)
        else:
            active.append(state_idx)
    return BiasObservabilityDecision(
        conditional_information=information,
        relative_information=relative,
        active_state_indices=tuple(active),
        regularized_state_indices=tuple(regularized),
        frozen_state_indices=tuple(frozen),
        regularization_std_by_state={
            state_idx: policy.regularization_std for state_idx in regularized
        },
    )


def analyze_arc_observability(
    prepared_arc: PreparedArc,
    measurement_type: MeasurementType,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    *,
    x0_mci: ArrayLike | None = None,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    rank_tol: float | None = None,
) -> ObservabilityResult:
    """Analyze initial-state observability for one prepared measurement arc."""
    x0 = (
        np.asarray(x0_mci, dtype=float).reshape(6)
        if x0_mci is not None
        else np.asarray(prepared_arc.truth_state_history_mci[0, :6], dtype=float).reshape(6)
    )
    return analyze_initial_state_observability(
        measurement_type,
        prepared_arc.t_pass_s,
        prepared_arc.obs_data,
        prepared_arc.pass_geo,
        x0,
        mu_moon_m3_s2,
        mu_earth_m3_s2,
        mu_sun_m3_s2,
        get_earth_pos,
        get_sun_pos,
        rtol=rtol,
        atol=atol,
        rank_tol=rank_tol,
    )


def analyze_augmented_arc_observability(
    prepared_arc: PreparedArc,
    measurement_type: MeasurementType,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    *,
    bias_mode: str,
    x0_mci: ArrayLike | None = None,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    rank_tol: float | None = None,
) -> ObservabilityResult:
    """Analyze joint initial-state and measurement-bias observability."""
    x0 = (
        np.asarray(x0_mci, dtype=float).reshape(6)
        if x0_mci is not None
        else np.asarray(prepared_arc.truth_state_history_mci[0, :6], dtype=float).reshape(6)
    )
    h_state = build_initial_state_jacobian(
        measurement_type,
        prepared_arc.t_pass_s,
        prepared_arc.obs_data,
        prepared_arc.pass_geo,
        x0,
        mu_moon_m3_s2,
        mu_earth_m3_s2,
        mu_sun_m3_s2,
        get_earth_pos,
        get_sun_pos,
        rtol=rtol,
        atol=atol,
    )
    h_bias = build_measurement_bias_jacobian(
        measurement_type,
        prepared_arc.obs_data,
        len(prepared_arc.pass_geo.stations),
        bias_mode,
    )
    sigma = measurement_sigma_vector(prepared_arc.obs_data, prepared_arc.pass_geo, measurement_type)
    weighted = np.hstack([h_state, h_bias]) / sigma[:, None]
    return summarize_weighted_jacobian(
        measurement_type,
        int(prepared_arc.obs_data.shape[0]),
        weighted,
        rank_tol=rank_tol,
    )


def build_measurement_bias_jacobian(
    measurement_type: MeasurementType,
    obs_data: ArrayLike,
    num_stations: int,
    bias_mode: str,
) -> np.ndarray:
    """Return the exact additive measurement-bias Jacobian."""
    measurement_type = _normalize_measurement_type(measurement_type)
    obs = np.asarray(obs_data, dtype=float)
    block_size = 3 if measurement_type == "position" else 4
    station_col = 4 if measurement_type == "position" else 5
    mode = str(bias_mode).lower()
    mode = {"global": "global_full", "station": "station_full"}.get(mode, mode)

    if mode == "global_full":
        jacobian = np.zeros((obs.shape[0] * block_size, block_size), dtype=float)
        for obs_idx in range(obs.shape[0]):
            row0 = obs_idx * block_size
            jacobian[row0 : row0 + block_size, :] = np.eye(block_size)
        return jacobian
    if mode == "station_angles":
        bias_block_size = 2
    elif mode == "station_full":
        bias_block_size = block_size
    else:
        raise ValueError("bias_mode must be 'global_full', 'station_angles', or 'station_full'.")

    jacobian = np.zeros((obs.shape[0] * block_size, bias_block_size * num_stations), dtype=float)
    for obs_idx in range(obs.shape[0]):
        station_id = int(obs[obs_idx, station_col]) - 1
        if station_id < 0 or station_id >= num_stations:
            raise ValueError("Observation station id is out of range.")
        row0 = obs_idx * block_size
        col0 = station_id * bias_block_size
        if mode == "station_angles":
            jacobian[row0 + block_size - 2 : row0 + block_size, col0 : col0 + 2] = np.eye(2)
        else:
            jacobian[row0 : row0 + block_size, col0 : col0 + block_size] = np.eye(block_size)
    return jacobian


def analyze_initial_state_observability(
    measurement_type: MeasurementType,
    t_pass_s: ArrayLike,
    obs_data: ArrayLike,
    pass_geo: PassGeometry,
    x0_mci: ArrayLike,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    *,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    rank_tol: float | None = None,
) -> ObservabilityResult:
    """Build a whitened initial-state Jacobian and Fisher information summary."""
    h_initial = build_initial_state_jacobian(
        measurement_type,
        t_pass_s,
        obs_data,
        pass_geo,
        x0_mci,
        mu_moon_m3_s2,
        mu_earth_m3_s2,
        mu_sun_m3_s2,
        get_earth_pos,
        get_sun_pos,
        rtol=rtol,
        atol=atol,
    )
    sigma = measurement_sigma_vector(obs_data, pass_geo, measurement_type)
    weighted_jacobian = h_initial / sigma[:, None]
    fisher_information = _symmetrize(weighted_jacobian.T @ weighted_jacobian)
    return summarize_weighted_jacobian(
        measurement_type,
        int(np.asarray(obs_data).shape[0]),
        weighted_jacobian,
        fisher_information=fisher_information,
        rank_tol=rank_tol,
    )


def build_initial_state_jacobian(
    measurement_type: MeasurementType,
    t_pass_s: ArrayLike,
    obs_data: ArrayLike,
    pass_geo: PassGeometry,
    x0_mci: ArrayLike,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    *,
    rtol: float = 1e-11,
    atol: float = 1e-12,
) -> np.ndarray:
    """Return the measurement Jacobian with respect to the arc initial state."""
    measurement_type = _normalize_measurement_type(measurement_type)
    t_pass_s = np.asarray(t_pass_s, dtype=float).reshape(-1)
    obs_data = np.asarray(obs_data, dtype=float)
    x0 = np.asarray(x0_mci, dtype=float).reshape(6)

    x_aug0 = np.concatenate([x0, np.eye(6).reshape(-1, order="F")])
    x_aug_hist = propagate_augmented_state(
        t_pass_s,
        x_aug0,
        mu_moon_m3_s2,
        mu_earth_m3_s2,
        mu_sun_m3_s2,
        get_earth_pos,
        get_sun_pos,
        rtol=rtol,
        atol=atol,
    )

    if measurement_type == "position":
        _, _, h_tilde = compute_position_residuals_analytic(x_aug_hist[:, :6], obs_data, pass_geo)
        block_size = 3
        time_col = 5
    elif range_rate_physics_config(pass_geo.range_rate_physics).mode == "geometric_instantaneous":
        _, _, h_tilde = compute_range_rate_residuals_analytic(x_aug_hist[:, :6], obs_data, pass_geo)
        block_size = 4
        time_col = 6
    else:
        return _build_two_way_range_rate_initial_state_jacobian(obs_data, pass_geo, x_aug_hist)

    h_initial = np.zeros((obs_data.shape[0] * block_size, 6), dtype=float)
    for obs_idx in range(obs_data.shape[0]):
        row0 = obs_idx * block_size
        time_idx = int(obs_data[obs_idx, time_col]) - 1
        phi_k = x_aug_hist[time_idx, 6:].reshape((6, 6), order="F")
        h_initial[row0 : row0 + block_size, :] = h_tilde[row0 : row0 + block_size, :] @ phi_k
    return h_initial


def _build_two_way_range_rate_initial_state_jacobian(
    obs_data: np.ndarray,
    pass_geo: PassGeometry,
    x_aug_hist: np.ndarray,
) -> np.ndarray:
    rr_physics = range_rate_physics_config(pass_geo.range_rate_physics)
    geometric_pass_geo = replace(pass_geo, range_rate_physics=RangeRatePhysicsConfig())
    _, _, h_tilde = compute_range_rate_residuals_analytic(x_aug_hist[:, :6], obs_data, geometric_pass_geo)

    h_initial = np.zeros((obs_data.shape[0] * 4, 6), dtype=float)
    for obs_idx in range(obs_data.shape[0]):
        row0 = obs_idx * 4
        time_idx = int(obs_data[obs_idx, 6]) - 1
        phi_k = x_aug_hist[time_idx, 6:].reshape((6, 6), order="F")
        h_initial[row0 : row0 + 4, :] = h_tilde[row0 : row0 + 4, :] @ phi_k
        station_id = int(obs_data[obs_idx, 5]) - 1
        h_initial[row0 + 1, :] = two_way_counted_doppler_initial_state_jacobian(
            float(pass_geo.t_s[time_idx]),
            pass_geo.stations[station_id],
            pass_geo.t_s,
            x_aug_hist,
            pass_geo.earth_pos_mci_m,
            pass_geo.earth_vel_mci_mps,
            pass_geo.x_j2000_to_itrf93,
            rr_physics,
        )
    return h_initial


def summarize_weighted_jacobian(
    measurement_type: MeasurementType,
    num_observations: int,
    weighted_jacobian: ArrayLike,
    *,
    fisher_information: ArrayLike | None = None,
    rank_tol: float | None = None,
) -> ObservabilityResult:
    """Summarize rank, singular values, and Fisher eigenstructure."""
    measurement_type = _normalize_measurement_type(measurement_type)
    h_white = np.asarray(weighted_jacobian, dtype=float)
    if h_white.ndim != 2:
        raise ValueError("weighted_jacobian must be a 2-D array.")

    fisher = (
        _symmetrize(np.asarray(fisher_information, dtype=float))
        if fisher_information is not None
        else _symmetrize(h_white.T @ h_white)
    )
    if fisher.shape != (h_white.shape[1], h_white.shape[1]):
        raise ValueError("fisher_information shape must match weighted_jacobian columns.")

    singular_values = np.linalg.svd(h_white, compute_uv=False)
    eigenvalues, eigenvectors = np.linalg.eigh(fisher)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    tol = _rank_tolerance(singular_values, h_white.shape, rank_tol)
    rank = int(np.sum(singular_values > tol))
    num_parameters = int(h_white.shape[1])
    if rank < num_parameters or singular_values.size == 0:
        condition_number = float("inf")
    else:
        condition_number = float(singular_values[0] / singular_values[-1])

    weakest_direction = eigenvectors[:, 0].copy() if eigenvectors.size else np.zeros(num_parameters)
    return ObservabilityResult(
        measurement_type=measurement_type,
        num_observations=int(num_observations),
        num_parameters=num_parameters,
        weighted_jacobian=h_white,
        fisher_information=fisher,
        singular_values=singular_values,
        information_eigenvalues=eigenvalues,
        rank=rank,
        condition_number=condition_number,
        weakest_direction=weakest_direction,
    )


def summarize_arc_observability_combinations(
    arc_results: Sequence[tuple[int, ObservabilityResult]],
    *,
    max_combination_size: int | None = None,
    rank_tol: float | None = None,
) -> tuple[ArcCombinationObservability, ...]:
    """Summarize how stacking different arc Jacobian combinations changes observability."""
    arc_results = tuple(arc_results)
    if not arc_results:
        return ()
    measurement_type = arc_results[0][1].measurement_type
    num_parameters = arc_results[0][1].num_parameters
    for _, result in arc_results:
        if result.measurement_type != measurement_type:
            raise ValueError("All arc observability results must share measurement_type.")
        if result.num_parameters != num_parameters:
            raise ValueError("All arc observability results must have the same number of parameters.")

    max_size = len(arc_results) if max_combination_size is None else int(max_combination_size)
    if max_size <= 0:
        raise ValueError("max_combination_size must be positive.")
    max_size = min(max_size, len(arc_results))

    summaries: list[ArcCombinationObservability] = []
    for size in range(1, max_size + 1):
        for combo in combinations(arc_results, size):
            arc_ids = tuple(int(item[0]) for item in combo)
            weighted = np.vstack([item[1].weighted_jacobian for item in combo])
            fisher = _symmetrize(weighted.T @ weighted)
            num_observations = sum(item[1].num_observations for item in combo)
            summaries.append(
                ArcCombinationObservability(
                    arc_ids=arc_ids,
                    observability=summarize_weighted_jacobian(
                        measurement_type,
                        num_observations,
                        weighted,
                        fisher_information=fisher,
                        rank_tol=rank_tol,
                    ),
                )
            )
    return tuple(summaries)


def _rank_tolerance(singular_values: np.ndarray, shape: tuple[int, int], rank_tol: float | None) -> float:
    if rank_tol is not None:
        if rank_tol < 0.0:
            raise ValueError("rank_tol must be non-negative.")
        return float(rank_tol)
    if singular_values.size == 0:
        return 0.0
    return float(max(shape) * np.finfo(float).eps * singular_values[0])


def _normalize_measurement_type(measurement_type: str) -> MeasurementType:
    normalized = str(measurement_type).lower()
    if normalized not in {"position", "range_rate"}:
        raise ValueError("measurement_type must be 'position' or 'range_rate'.")
    return normalized  # type: ignore[return-value]


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)
