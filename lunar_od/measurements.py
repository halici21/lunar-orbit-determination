"""Synthetic measurement generation and residual helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .geometry import ecef2razel_sez, ecef2sez_dcm, wrap_to_pi
from .accelerated import apply_stm_to_jacobian, geometric_range_rate_observables, position_observables
from .radiometrics import (
    RangeRatePhysicsConfig,
    _interp_state_transition_position,
    instantaneous_geometric_range_rate,
    range_rate_physics_config,
    two_way_counted_doppler_observable,
    interp_state_history,
)

C_LIGHT_MPS = 299792458.0
ANGLE_JACOBIAN_MIN_HORIZONTAL_UNIT_NORM = 1.0e-6
ABERRATION_LOCAL_JACOBIAN_STEP = 1.0e-5

MEASUREMENT_MODEL_PROFILES = (
    "geometric_instantaneous",
    "one_way_light_time",
    "one_way_light_time_aberrated_local_mci",
    "one_way_light_time_aberrated_spice_ssb",
    "two_way_counted_doppler",
)
POSITION_MEASUREMENT_MODEL_PROFILES = MEASUREMENT_MODEL_PROFILES[:4]
COMPANION_GEOMETRIES = ("instantaneous", "apparent_one_way")
JACOBIAN_MODELS = (
    "analytic_exact_geometric",
    "analytic_first_order_light_time",
    "implicit_light_time",
    "finite_difference_reference",
)


@dataclass(frozen=True)
class PassGeometry:
    t_s: np.ndarray
    earth_pos_mci_m: np.ndarray
    earth_vel_mci_mps: np.ndarray
    x_j2000_to_itrf93: np.ndarray
    stations: tuple
    measurement_type: str
    range_rate_physics: RangeRatePhysicsConfig | None = None
    apply_light_time: bool = False
    apply_stellar_aberration: bool = False
    stellar_aberration_model: str = "local_mci"
    earth_vel_ssb_j2000_mps: np.ndarray | None = None
    measurement_model_profile: str = "geometric_instantaneous"
    companion_geometry: str = "instantaneous"
    jacobian_model: str = "analytic_exact_geometric"
    measurement_metadata: dict | None = None
    # Two-way range (M3) transport: SPICE ET of scenario t=0 for exact
    # event-epoch sxform evaluation, and the TwoWayRangeConfig in use.
    # Both stay None for all other measurement types.
    et0_s: float | None = None
    two_way_range: object | None = None


@dataclass(frozen=True)
class LightTimeSolution:
    range_m: float
    light_time_s: float
    transmit_time_s: float
    iterations: int
    converged: bool
    target_position_m: np.ndarray


@dataclass(frozen=True)
class OneWayLightTimeSensitivity:
    d_light_time_d_state: np.ndarray
    d_range_d_state: np.ndarray
    d_transmit_time_d_state: np.ndarray
    spacecraft_position_sensitivity: np.ndarray
    spacecraft_velocity_mps: np.ndarray
    denominator_mps: float
    condition_metric: float


@dataclass(frozen=True)
class OneWayLightTimeInitialStateSensitivity:
    station_position_rx_m: np.ndarray
    spacecraft_position_tx_m: np.ndarray
    spacecraft_velocity_tx_mps: np.ndarray
    line_of_sight_m: np.ndarray
    unit_line_of_sight: np.ndarray
    range_m: float
    phi_r_tx: np.ndarray
    d_tau_dx0: np.ndarray
    d_range_dx0: np.ndarray
    j_los_dx0: np.ndarray
    j_unit_los_dx0: np.ndarray
    denominator_mps: float
    condition_metric: float


@dataclass(frozen=True)
class _StellarAberrationLocalJacobian:
    apparent_unit_los: np.ndarray
    local_jacobian: np.ndarray
    tangent_basis: np.ndarray
    step: float
    norm_error: float


class MeasurementJacobianError(ValueError):
    """Raised when a requested measurement Jacobian is physically undefined."""


def normalize_measurement_model_profile(
    measurement_model_profile: str | None = None,
    *,
    apply_light_time: bool = False,
    apply_stellar_aberration: bool = False,
    stellar_aberration_model: str = "local_mci",
    allow_two_way: bool = False,
) -> tuple[str, bool, bool, str]:
    """Resolve profile and legacy light-time booleans into one effective model."""
    if stellar_aberration_model not in ("local_mci", "spice_ssb"):
        raise ValueError("stellar_aberration_model must be 'local_mci' or 'spice_ssb'.")

    if measurement_model_profile is None:
        if apply_stellar_aberration and not apply_light_time:
            raise ValueError(
                "apply_stellar_aberration=True requires apply_light_time=True "
                "(stellar aberration is applied on top of the light-time solution)."
            )
        if apply_stellar_aberration:
            measurement_model_profile = f"one_way_light_time_aberrated_{stellar_aberration_model}"
        elif apply_light_time:
            measurement_model_profile = "one_way_light_time"
        else:
            measurement_model_profile = "geometric_instantaneous"

    if measurement_model_profile not in MEASUREMENT_MODEL_PROFILES:
        raise ValueError(
            f"measurement_model_profile must be one of {MEASUREMENT_MODEL_PROFILES}; "
            f"got {measurement_model_profile!r}."
        )
    if measurement_model_profile == "two_way_counted_doppler" and not allow_two_way:
        raise ValueError(
            "measurement_model_profile='two_way_counted_doppler' is valid only for "
            "range-rate measurement metadata; use range_rate_physics to select Doppler physics."
        )

    if measurement_model_profile == "geometric_instantaneous":
        return measurement_model_profile, False, False, stellar_aberration_model
    if measurement_model_profile == "one_way_light_time":
        return measurement_model_profile, True, False, stellar_aberration_model
    if measurement_model_profile == "one_way_light_time_aberrated_local_mci":
        return measurement_model_profile, True, True, "local_mci"
    if measurement_model_profile == "one_way_light_time_aberrated_spice_ssb":
        return measurement_model_profile, True, True, "spice_ssb"
    return measurement_model_profile, apply_light_time, apply_stellar_aberration, stellar_aberration_model


def normalize_companion_geometry(companion_geometry: str | None = None) -> str:
    """Return the effective range/angle companion geometry mode."""
    value = "instantaneous" if companion_geometry is None else companion_geometry
    if value not in COMPANION_GEOMETRIES:
        raise ValueError(f"companion_geometry must be one of {COMPANION_GEOMETRIES}; got {value!r}.")
    return value


def normalize_jacobian_model(
    jacobian_model: str | None = None,
    *,
    measurement_model_profile: str = "geometric_instantaneous",
    companion_geometry: str = "instantaneous",
) -> str:
    """Return the declared Jacobian model for metadata and validation."""
    if jacobian_model is None:
        if measurement_model_profile == "geometric_instantaneous" and companion_geometry == "instantaneous":
            jacobian_model = "analytic_exact_geometric"
        else:
            jacobian_model = "analytic_first_order_light_time"
    if jacobian_model not in JACOBIAN_MODELS:
        raise ValueError(f"jacobian_model must be one of {JACOBIAN_MODELS}; got {jacobian_model!r}.")
    if (
        jacobian_model == "implicit_light_time"
        and measurement_model_profile == "geometric_instantaneous"
        and companion_geometry == "instantaneous"
    ):
        raise ValueError(
            "jacobian_model='implicit_light_time' requires a light-time corrected "
            "measurement profile or apparent companion geometry."
        )
    return jacobian_model


def measurement_model_metadata(
    pass_geo: "PassGeometry",
    *,
    noise_enabled: bool | None = None,
    noise_seed: int | None = None,
) -> dict:
    """Build traceable measurement-physics metadata from a pass geometry."""
    if pass_geo.measurement_type == "two_way_range":
        # Two-way range metadata is built by the generation path in
        # lunar_od.two_way_range; this helper only transports it.
        if pass_geo.measurement_metadata is not None:
            return dict(pass_geo.measurement_metadata)
        raise ValueError(
            "two_way_range pass geometry carries its metadata from generation; "
            "none was attached."
        )
    rr = range_rate_physics_config(pass_geo.range_rate_physics)
    profile_light_time = False
    profile_stellar = False
    stellar_model = pass_geo.stellar_aberration_model
    if pass_geo.measurement_model_profile != "geometric_instantaneous":
        _profile, profile_light_time, profile_stellar, profile_model = normalize_measurement_model_profile(
            pass_geo.measurement_model_profile,
            allow_two_way=pass_geo.measurement_type == "range_rate",
        )
        if profile_stellar:
            stellar_model = profile_model
    use_light_time = bool(pass_geo.apply_light_time or profile_light_time)
    use_stellar = bool(pass_geo.apply_stellar_aberration or profile_stellar)
    station_sigmas = []
    for station in pass_geo.stations:
        station_sigmas.append(
            {
                "name": getattr(station, "name", None),
                "sigma_range_m": float(getattr(station, "sigma_range_m", np.nan)),
                "sigma_angle_rad": float(getattr(station, "sigma_angle_rad", np.nan)),
                "sigma_range_rate_mps": (
                    None
                    if getattr(station, "sigma_range_rate_mps", None) is None
                    else float(getattr(station, "sigma_range_rate_mps"))
                ),
            }
        )
    metadata = {
        "measurement_type": pass_geo.measurement_type,
        "measurement_model_profile": pass_geo.measurement_model_profile,
        "range_rate_physics": rr.mode,
        "companion_geometry": pass_geo.companion_geometry,
        "apply_light_time": bool(pass_geo.apply_light_time),
        "apply_stellar_aberration": bool(pass_geo.apply_stellar_aberration),
        "stellar_aberration_model": stellar_model,
        "light_time_tolerance_s": float(rr.light_time_tolerance_s),
        "light_time_max_iter": int(rr.light_time_max_iter),
        "count_interval_s": float(rr.count_interval_s),
        "uplink_frequency_hz": float(rr.uplink_frequency_hz),
        "turnaround_ratio": float(rr.turnaround_ratio),
        "station_clock_offset_s": float(rr.station_clock_offset_s),
        "station_clock_drift": float(rr.station_clock_drift),
        "transponder_delay_s": float(rr.transponder_delay_s),
        "noise_enabled": None if noise_enabled is None else bool(noise_enabled),
        "noise_seed": noise_seed,
        "station_sigmas": station_sigmas,
        "jacobian_model": pass_geo.jacobian_model,
        "station_position_epoch": "receive",
        "station_velocity_epoch": "receive",
        "station_velocity_model": "sxform",
        "frame_transformation_epoch": "receive",
        "troposphere_model": "none",
        "ionosphere_model": "none",
        "solar_plasma_model": "none",
    }
    if use_light_time:
        metadata["spacecraft_position_epoch"] = "transmit"
        metadata["spacecraft_velocity_epoch"] = "transmit"
        metadata["measurement_tag_epoch"] = "receive"
    else:
        metadata["spacecraft_position_epoch"] = "receive"
        metadata["spacecraft_velocity_epoch"] = "receive"
        metadata["measurement_tag_epoch"] = "receive"
    if pass_geo.jacobian_model == "implicit_light_time":
        metadata.update(
            {
                "light_time_sensitivity": "enabled",
                "jacobian_spacecraft_epoch": "transmit",
                "range_jacobian_model": "implicit_light_time",
                "initial_state_sensitivity_epoch": "transmit",
                "azimuth_residual_model": "wrapped",
            }
        )
        if use_stellar:
            metadata.update(
                {
                    "line_of_sight_jacobian_model": "implicit_light_time_chain_rule",
                    "angle_jacobian_model": "hybrid_apparent_chain_rule",
                    "aberration_jacobian_model": "local_central_finite_difference",
                    "angle_jacobian_matches_full_residual_physics": True,
                    "angle_jacobian_includes_light_time_sensitivity": True,
                    "angle_jacobian_includes_los_normalization": True,
                    "angle_jacobian_includes_frame_chain_rule": True,
                    "angle_singularity_policy": "raise",
                    "angle_horizontal_unit_norm_threshold": (
                        ANGLE_JACOBIAN_MIN_HORIZONTAL_UNIT_NORM
                    ),
                    "observer_velocity_epoch": "receive",
                    "observer_velocity_frame": "J2000",
                    "observer_velocity_reference_center": (
                        "SSB" if stellar_model == "spice_ssb" else "MOON"
                    ),
                    "aberration_local_jacobian_input": "unit_cn_los",
                    "aberration_local_jacobian_space": "tangent",
                    "aberration_local_jacobian_step": ABERRATION_LOCAL_JACOBIAN_STEP,
                }
            )
        else:
            metadata.update(
                {
                    "line_of_sight_jacobian_model": "implicit_light_time_chain_rule",
                    "angle_jacobian_model": "implicit_light_time_chain_rule",
                    "aberration_jacobian_model": "not_applied",
                    "angle_jacobian_matches_full_residual_physics": True,
                    "angle_jacobian_includes_light_time_sensitivity": True,
                    "angle_jacobian_includes_los_normalization": True,
                    "angle_jacobian_includes_frame_chain_rule": True,
                    "angle_singularity_policy": "raise",
                    "angle_horizontal_unit_norm_threshold": (
                        ANGLE_JACOBIAN_MIN_HORIZONTAL_UNIT_NORM
                    ),
                }
            )
    return metadata


def solve_one_way_light_time(
    receive_time_s: float,
    observer_position_m: ArrayLike,
    get_target_position_m,
    *,
    light_speed_mps: float = C_LIGHT_MPS,
    tolerance_s: float = 1e-12,
    max_iter: int = 10,
) -> LightTimeSolution:
    """Iterate one-way geometric light-time from target transmit to receive time."""
    observer_position_m = np.asarray(observer_position_m, dtype=float).reshape(3)
    if light_speed_mps <= 0.0:
        raise ValueError("light_speed_mps must be positive.")
    if tolerance_s <= 0.0:
        raise ValueError("tolerance_s must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    receive_time_s = float(receive_time_s)
    target_position = np.asarray(get_target_position_m(receive_time_s), dtype=float).reshape(3)
    light_time_s = float(np.linalg.norm(target_position - observer_position_m) / light_speed_mps)
    converged = False

    for iteration in range(1, max_iter + 1):
        transmit_time_s = receive_time_s - light_time_s
        target_position = np.asarray(get_target_position_m(transmit_time_s), dtype=float).reshape(3)
        new_light_time_s = float(np.linalg.norm(target_position - observer_position_m) / light_speed_mps)
        if abs(new_light_time_s - light_time_s) <= tolerance_s:
            light_time_s = new_light_time_s
            converged = True
            break
        light_time_s = new_light_time_s
    else:
        iteration = max_iter

    transmit_time_s = receive_time_s - light_time_s
    range_m = light_time_s * light_speed_mps
    return LightTimeSolution(
        range_m=range_m,
        light_time_s=light_time_s,
        transmit_time_s=transmit_time_s,
        iterations=iteration,
        converged=converged,
        target_position_m=target_position,
    )


def one_way_light_time_range_sensitivity(
    receive_time_s: float,
    station,
    t_grid_s: ArrayLike,
    state_history_mci: ArrayLike,
    earth_pos_mci_rx: ArrayLike,
    x_j2k_itrf_rx: ArrayLike,
    *,
    light_speed_mps: float = C_LIGHT_MPS,
    tolerance_s: float = 1e-12,
    max_iter: int = 10,
) -> tuple[LightTimeSolution, OneWayLightTimeSensitivity]:
    """Return one-way range and its implicit local-state sensitivity.

    The receive epoch and station state are fixed.  The derivative is with
    respect to a local spacecraft state at the receive epoch.  At fixed
    transmit time, the position sensitivity is approximated as
    ``[I, (t_t - t_r) I]``; the implicit dependence of transmit time on that
    state is then included exactly for this local constant-velocity contract.
    This is not a replacement for evaluating a full dynamical STM at the
    transmit epoch.
    """
    t_grid_s = np.asarray(t_grid_s, dtype=float)
    state_history_mci = np.asarray(state_history_mci, dtype=float)
    earth_pos_mci_rx = np.asarray(earth_pos_mci_rx, dtype=float).reshape(3)
    x_rx = np.asarray(x_j2k_itrf_rx, dtype=float)
    station_ecef = np.asarray(station.r_ecef_m, dtype=float).reshape(3)
    station_ecef_state = np.concatenate([station_ecef, np.zeros(3)])
    station_rel_state_j2000 = np.linalg.solve(x_rx, station_ecef_state)
    station_mci_rx = earth_pos_mci_rx + station_rel_state_j2000[:3]

    solution = solve_one_way_light_time(
        receive_time_s,
        station_mci_rx,
        lambda t: interp_state_history(t_grid_s, state_history_mci, t)[:3],
        light_speed_mps=light_speed_mps,
        tolerance_s=tolerance_s,
        max_iter=max_iter,
    )
    if not solution.converged:
        raise RuntimeError(
            f"One-way light-time did not converge in {max_iter} iterations at "
            f"receive time {float(receive_time_s):.16g} s."
        )

    state_tx = interp_state_history(t_grid_s, state_history_mci, solution.transmit_time_s)
    rho_vec = state_tx[:3] - station_mci_rx
    range_m = float(np.linalg.norm(rho_vec))
    if range_m <= 0.0:
        raise ValueError("One-way light-time range must be positive for Jacobian evaluation.")
    rho_hat = rho_vec / range_m
    velocity_tx = np.asarray(state_tx[3:6], dtype=float)

    dt_tx_rx = float(solution.transmit_time_s - receive_time_s)
    position_sensitivity = np.zeros((3, 6), dtype=float)
    position_sensitivity[:, :3] = np.eye(3)
    position_sensitivity[:, 3:] = dt_tx_rx * np.eye(3)

    denominator_mps = float(light_speed_mps + np.dot(rho_hat, velocity_tx))
    denominator_floor = np.finfo(float).eps * max(float(light_speed_mps), 1.0) * 16.0
    if abs(denominator_mps) <= denominator_floor:
        raise RuntimeError(
            "Implicit light-time Jacobian is singular because c + rho_hat dot v is near zero."
        )

    d_light_time_d_state = (rho_hat @ position_sensitivity) / denominator_mps
    d_range_d_state = float(light_speed_mps) * d_light_time_d_state
    sensitivity = OneWayLightTimeSensitivity(
        d_light_time_d_state=np.asarray(d_light_time_d_state, dtype=float),
        d_range_d_state=np.asarray(d_range_d_state, dtype=float),
        d_transmit_time_d_state=-np.asarray(d_light_time_d_state, dtype=float),
        spacecraft_position_sensitivity=position_sensitivity,
        spacecraft_velocity_mps=velocity_tx,
        denominator_mps=denominator_mps,
        condition_metric=abs(denominator_mps) / float(light_speed_mps),
    )
    return solution, sensitivity


def _station_relative_state_j2000_at_receive_epoch(
    station,
    x_j2k_itrf_rx: ArrayLike,
) -> np.ndarray:
    """Return the fixed station state in J2000 axes at the receive epoch."""
    x_rx = np.asarray(x_j2k_itrf_rx, dtype=float)
    station_ecef = np.asarray(station.r_ecef_m, dtype=float).reshape(3)
    station_ecef_state = np.concatenate([station_ecef, np.zeros(3)])
    return np.linalg.solve(x_rx, station_ecef_state)


def _station_position_mci_at_receive_epoch(
    station,
    earth_pos_mci_rx: ArrayLike,
    x_j2k_itrf_rx: ArrayLike,
) -> np.ndarray:
    earth_pos_mci_rx = np.asarray(earth_pos_mci_rx, dtype=float).reshape(3)
    station_rel_state_j2000 = _station_relative_state_j2000_at_receive_epoch(
        station, x_j2k_itrf_rx
    )
    return earth_pos_mci_rx + station_rel_state_j2000[:3]


def _observer_velocity_j2000_at_receive_epoch(
    station,
    x_j2k_itrf_rx: ArrayLike,
    reference_center_velocity_j2000_mps: ArrayLike,
    *,
    station_relative_state_j2000: ArrayLike | None = None,
) -> np.ndarray:
    """Return station observer velocity at receive time in J2000 axes, m/s.

    The supplied reference-center velocity is Moon-relative for ``local_mci``
    or SSB-relative for ``spice_ssb``.  The station's inertial velocity from
    Earth rotation is added through the receive-epoch 6x6 state transform.
    The observer state is fixed with respect to the spacecraft initial state,
    so its derivative with respect to ``x0`` is zero in the current OD model.
    """
    reference_velocity = np.asarray(
        reference_center_velocity_j2000_mps, dtype=float
    ).reshape(3)
    station_rel_state = (
        _station_relative_state_j2000_at_receive_epoch(station, x_j2k_itrf_rx)
        if station_relative_state_j2000 is None
        else np.asarray(station_relative_state_j2000, dtype=float).reshape(6)
    )
    return reference_velocity + station_rel_state[3:]


def one_way_light_time_initial_state_sensitivity(
    receive_time_s: float,
    station,
    t_grid_s: ArrayLike,
    state_history_mci: ArrayLike,
    phi_history: ArrayLike,
    earth_pos_mci_rx: ArrayLike,
    x_j2k_itrf_rx: ArrayLike,
    *,
    light_speed_mps: float = C_LIGHT_MPS,
    tolerance_s: float = 1e-12,
    max_iter: int = 10,
) -> tuple[LightTimeSolution, OneWayLightTimeSensitivity, OneWayLightTimeInitialStateSensitivity]:
    """Build implicit one-way LOS sensitivities with respect to the arc initial state."""
    solution, local_range_sensitivity = one_way_light_time_range_sensitivity(
        receive_time_s,
        station,
        t_grid_s,
        state_history_mci,
        earth_pos_mci_rx,
        x_j2k_itrf_rx,
        light_speed_mps=light_speed_mps,
        tolerance_s=tolerance_s,
        max_iter=max_iter,
    )
    phi_r_tx = _interp_state_transition_position(
        t_grid_s, phi_history, solution.transmit_time_s
    )
    spacecraft_state_tx = interp_state_history(
        t_grid_s, state_history_mci, solution.transmit_time_s
    )
    station_position_rx = _station_position_mci_at_receive_epoch(
        station, earth_pos_mci_rx, x_j2k_itrf_rx
    )
    line_of_sight_m = spacecraft_state_tx[:3] - station_position_rx
    range_m = float(np.linalg.norm(line_of_sight_m))
    if range_m <= 0.0:
        raise MeasurementJacobianError(
            "One-way apparent LOS range must be positive for Jacobian evaluation."
        )
    unit_line_of_sight = line_of_sight_m / range_m
    spacecraft_velocity_tx = np.asarray(spacecraft_state_tx[3:6], dtype=float)

    # Reuse the tested M2.1 range kernel: its position coefficients are
    # rho_hat/(c + rho_hat.v). Mapping those coefficients with Phi_r(t_t,t0)
    # gives the implicit light-time derivative with respect to x0.
    d_tau_dx0 = local_range_sensitivity.d_light_time_d_state[:3] @ phi_r_tx
    d_range_dx0 = float(light_speed_mps) * d_tau_dx0
    j_los_dx0 = phi_r_tx - np.outer(spacecraft_velocity_tx, d_tau_dx0)
    perpendicular_projection = np.eye(3) - np.outer(unit_line_of_sight, unit_line_of_sight)
    j_unit_los_dx0 = (perpendicular_projection @ j_los_dx0) / range_m

    initial_sensitivity = OneWayLightTimeInitialStateSensitivity(
        station_position_rx_m=station_position_rx,
        spacecraft_position_tx_m=np.asarray(spacecraft_state_tx[:3], dtype=float),
        spacecraft_velocity_tx_mps=spacecraft_velocity_tx,
        line_of_sight_m=line_of_sight_m,
        unit_line_of_sight=unit_line_of_sight,
        range_m=range_m,
        phi_r_tx=np.asarray(phi_r_tx, dtype=float),
        d_tau_dx0=np.asarray(d_tau_dx0, dtype=float),
        d_range_dx0=np.asarray(d_range_dx0, dtype=float),
        j_los_dx0=np.asarray(j_los_dx0, dtype=float),
        j_unit_los_dx0=np.asarray(j_unit_los_dx0, dtype=float),
        denominator_mps=local_range_sensitivity.denominator_mps,
        condition_metric=local_range_sensitivity.condition_metric,
    )
    return solution, local_range_sensitivity, initial_sensitivity


def _unit_line_of_sight_sensitivity(
    line_of_sight_m: np.ndarray,
    j_los: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    line_of_sight_m = np.asarray(line_of_sight_m, dtype=float).reshape(3)
    j_los = np.asarray(j_los, dtype=float)
    range_m = float(np.linalg.norm(line_of_sight_m))
    if range_m <= 0.0:
        raise MeasurementJacobianError(
            "One-way apparent LOS range must be positive for Jacobian evaluation."
        )
    unit_line_of_sight = line_of_sight_m / range_m
    perpendicular_projection = np.eye(3) - np.outer(unit_line_of_sight, unit_line_of_sight)
    j_unit_los = (perpendicular_projection @ j_los) / range_m
    return unit_line_of_sight, j_unit_los, range_m


def _position_measurement_jacobian_from_unit_los(
    d_range_dx: ArrayLike,
    unit_line_of_sight_mci: ArrayLike,
    j_unit_los_dx: ArrayLike,
    station,
    x_j2k_itrf_rx: ArrayLike,
    *,
    horizontal_unit_norm_threshold: float = ANGLE_JACOBIAN_MIN_HORIZONTAL_UNIT_NORM,
) -> np.ndarray:
    """Map range and unit-LOS sensitivities to [range, azimuth, elevation]."""
    if horizontal_unit_norm_threshold <= 0.0:
        raise ValueError("horizontal_unit_norm_threshold must be positive.")
    unit_line_of_sight_mci = np.asarray(unit_line_of_sight_mci, dtype=float).reshape(3)
    j_unit_los_dx = np.asarray(j_unit_los_dx, dtype=float)
    if j_unit_los_dx.ndim != 2 or j_unit_los_dx.shape[0] != 3:
        raise ValueError("j_unit_los_dx must have shape (3, state_count).")
    d_range_dx = np.asarray(d_range_dx, dtype=float).reshape(-1)
    if d_range_dx.size != j_unit_los_dx.shape[1]:
        raise ValueError("d_range_dx and j_unit_los_dx must use the same state count.")

    c_sez_ecef = ecef2sez_dcm(station.lat_rad, station.lon_rad)
    c_ecef_mci = np.asarray(x_j2k_itrf_rx, dtype=float)[:3, :3]
    c_sez_mci = c_sez_ecef @ c_ecef_mci
    unit_los_sez = c_sez_mci @ unit_line_of_sight_mci
    j_unit_los_sez = c_sez_mci @ j_unit_los_dx

    south, east, zenith = unit_los_sez
    horizontal2 = float(south * south + east * east)
    horizontal = float(np.sqrt(horizontal2))
    if horizontal < horizontal_unit_norm_threshold:
        raise MeasurementJacobianError(
            "Azimuth Jacobian is undefined or ill-conditioned near zenith: "
            f"horizontal unit-LOS norm {horizontal:.6e} is below "
            f"{horizontal_unit_norm_threshold:.6e}."
        )

    d_az_dsez = np.array([east / horizontal2, -south / horizontal2, 0.0])
    unit_norm2 = float(np.dot(unit_los_sez, unit_los_sez))
    d_el_dsez = np.array(
        [
            -(zenith * south) / (unit_norm2 * horizontal),
            -(zenith * east) / (unit_norm2 * horizontal),
            horizontal / unit_norm2,
        ]
    )
    block = np.zeros((3, d_range_dx.size), dtype=float)
    block[0, :] = d_range_dx
    block[1, :] = d_az_dsez @ j_unit_los_sez
    block[2, :] = d_el_dsez @ j_unit_los_sez
    return block


def one_way_light_time_position_initial_state_jacobian(
    receive_time_s: float,
    station,
    t_grid_s: ArrayLike,
    state_history_mci: ArrayLike,
    phi_history: ArrayLike,
    earth_pos_mci_rx: ArrayLike,
    x_j2k_itrf_rx: ArrayLike,
    *,
    light_speed_mps: float = C_LIGHT_MPS,
    tolerance_s: float = 1e-12,
    max_iter: int = 10,
    horizontal_unit_norm_threshold: float = ANGLE_JACOBIAN_MIN_HORIZONTAL_UNIT_NORM,
    apply_stellar: bool = False,
    observer_reference_velocity_j2000_mps: ArrayLike | None = None,
) -> tuple[LightTimeSolution, OneWayLightTimeInitialStateSensitivity, np.ndarray]:
    """Return an implicit initial-state [range, azimuth, elevation] Jacobian.

    With stellar aberration enabled, the validated analytic CN unit-LOS
    sensitivity is passed through a local tangent-space derivative of the
    production aberration transform.  The observer velocity is fixed with
    respect to the spacecraft initial state; station/clock/EOP solve-for
    extensions must revisit that assumption.
    """
    solution, _local_range, sensitivity = one_way_light_time_initial_state_sensitivity(
        receive_time_s,
        station,
        t_grid_s,
        state_history_mci,
        phi_history,
        earth_pos_mci_rx,
        x_j2k_itrf_rx,
        light_speed_mps=light_speed_mps,
        tolerance_s=tolerance_s,
        max_iter=max_iter,
    )
    direction_unit_los = sensitivity.unit_line_of_sight
    j_direction_dx0 = sensitivity.j_unit_los_dx0
    if apply_stellar:
        if observer_reference_velocity_j2000_mps is None:
            raise MeasurementJacobianError(
                "Stellar-aberration initial-state Jacobian requires the receive-epoch "
                "observer reference-center velocity in J2000 axes."
            )
        observer_velocity = _observer_velocity_j2000_at_receive_epoch(
            station,
            x_j2k_itrf_rx,
            observer_reference_velocity_j2000_mps,
        )
        aberration = _stellar_aberration_local_jacobian(
            sensitivity.unit_line_of_sight,
            observer_velocity,
            light_speed_mps=light_speed_mps,
        )
        direction_unit_los = aberration.apparent_unit_los
        j_direction_dx0 = aberration.local_jacobian @ sensitivity.j_unit_los_dx0

    block = _position_measurement_jacobian_from_unit_los(
        sensitivity.d_range_dx0,
        direction_unit_los,
        j_direction_dx0,
        station,
        x_j2k_itrf_rx,
        horizontal_unit_norm_threshold=horizontal_unit_norm_threshold,
    )
    return solution, sensitivity, block


def one_way_light_time_position_local_state_jacobian(
    receive_time_s: float,
    station,
    t_grid_s: ArrayLike,
    state_history_mci: ArrayLike,
    earth_pos_mci_rx: ArrayLike,
    x_j2k_itrf_rx: ArrayLike,
    *,
    light_speed_mps: float = C_LIGHT_MPS,
    tolerance_s: float = 1e-12,
    max_iter: int = 10,
    horizontal_unit_norm_threshold: float = ANGLE_JACOBIAN_MIN_HORIZONTAL_UNIT_NORM,
) -> tuple[LightTimeSolution, OneWayLightTimeSensitivity, np.ndarray]:
    """Return a local receive-state [range, azimuth, elevation] Jacobian."""
    solution, sensitivity = one_way_light_time_range_sensitivity(
        receive_time_s,
        station,
        t_grid_s,
        state_history_mci,
        earth_pos_mci_rx,
        x_j2k_itrf_rx,
        light_speed_mps=light_speed_mps,
        tolerance_s=tolerance_s,
        max_iter=max_iter,
    )
    spacecraft_state_tx = interp_state_history(
        t_grid_s, state_history_mci, solution.transmit_time_s
    )
    station_position_rx = _station_position_mci_at_receive_epoch(
        station, earth_pos_mci_rx, x_j2k_itrf_rx
    )
    line_of_sight_m = spacecraft_state_tx[:3] - station_position_rx
    j_los_dlocal = sensitivity.spacecraft_position_sensitivity - np.outer(
        sensitivity.spacecraft_velocity_mps, sensitivity.d_light_time_d_state
    )
    unit_los, j_unit_los_dlocal, _range_m = _unit_line_of_sight_sensitivity(
        line_of_sight_m, j_los_dlocal
    )
    block = _position_measurement_jacobian_from_unit_los(
        sensitivity.d_range_d_state,
        unit_los,
        j_unit_los_dlocal,
        station,
        x_j2k_itrf_rx,
        horizontal_unit_norm_threshold=horizontal_unit_norm_threshold,
    )
    return solution, sensitivity, block


def one_way_light_time_range_initial_state_jacobian(
    receive_time_s: float,
    station,
    t_grid_s: ArrayLike,
    state_history_mci: ArrayLike,
    phi_history: ArrayLike,
    earth_pos_mci_rx: ArrayLike,
    x_j2k_itrf_rx: ArrayLike,
    *,
    light_speed_mps: float = C_LIGHT_MPS,
    tolerance_s: float = 1e-12,
    max_iter: int = 10,
) -> tuple[LightTimeSolution, OneWayLightTimeSensitivity, np.ndarray]:
    """Map the implicit one-way range sensitivity to the initial state.

    Unlike the local Jacobian helper, this function evaluates the propagated
    position sensitivity ``Phi_r(t_t, t_0)`` at the converged transmit epoch.
    """
    solution, sensitivity, initial_sensitivity = one_way_light_time_initial_state_sensitivity(
        receive_time_s,
        station,
        t_grid_s,
        state_history_mci,
        phi_history,
        earth_pos_mci_rx,
        x_j2k_itrf_rx,
        light_speed_mps=light_speed_mps,
        tolerance_s=tolerance_s,
        max_iter=max_iter,
    )
    return solution, sensitivity, initial_sensitivity.d_range_dx0


def position_initial_state_jacobian_from_augmented_history(
    obs_data: ArrayLike,
    x_aug_hist: ArrayLike,
    h_tilde: ArrayLike,
    pass_geo: PassGeometry,
) -> np.ndarray:
    """Map local position Jacobians to the arc initial state exactly once.

    Geometric and first-order modes retain the receive-epoch STM mapping.  The
    implicit CN mode replaces each full [range, azimuth, elevation] block with
    the transmit-epoch initial-state helper.  CN+S additionally chains the
    local stellar-aberration derivative before the receive-frame angle map.
    """
    obs_data = np.asarray(obs_data, dtype=float)
    x_aug_hist = np.asarray(x_aug_hist, dtype=float)
    h_tilde = np.asarray(h_tilde, dtype=float)
    h_initial = apply_stm_to_jacobian(obs_data, x_aug_hist, h_tilde, 3, 5)
    if getattr(pass_geo, "jacobian_model", "analytic_exact_geometric") != "implicit_light_time":
        return h_initial

    profile_stellar = False
    stellar_model = getattr(pass_geo, "stellar_aberration_model", "local_mci")
    profile = getattr(pass_geo, "measurement_model_profile", "geometric_instantaneous")
    if profile != "geometric_instantaneous":
        _profile, _light_time, profile_stellar, profile_model = normalize_measurement_model_profile(
            profile
        )
        if profile_stellar:
            stellar_model = profile_model
    use_stellar = bool(getattr(pass_geo, "apply_stellar_aberration", False) or profile_stellar)
    observer_reference_velocity = None
    if use_stellar:
        if stellar_model == "spice_ssb":
            observer_reference_velocity = getattr(
                pass_geo, "earth_vel_ssb_j2000_mps", None
            )
            if observer_reference_velocity is None:
                raise MeasurementJacobianError(
                    "stellar_aberration_model='spice_ssb' requires "
                    "pass_geo.earth_vel_ssb_j2000_mps for the Jacobian."
                )
        else:
            observer_reference_velocity = getattr(pass_geo, "earth_vel_mci_mps", None)
            if observer_reference_velocity is None:
                raise MeasurementJacobianError(
                    "stellar_aberration_model='local_mci' requires "
                    "pass_geo.earth_vel_mci_mps for the Jacobian."
                )
        observer_reference_velocity = np.asarray(observer_reference_velocity, dtype=float)
    phi_history = np.stack(
        [row.reshape((6, 6), order="F") for row in x_aug_hist[:, 6:]], axis=0
    )
    for obs_idx in range(obs_data.shape[0]):
        time_idx = int(obs_data[obs_idx, 5]) - 1
        station_idx = int(obs_data[obs_idx, 4]) - 1
        common_args = (
            float(obs_data[obs_idx, 0]),
            pass_geo.stations[station_idx],
            pass_geo.t_s,
            x_aug_hist[:, :6],
            phi_history,
            pass_geo.earth_pos_mci_m[time_idx],
            pass_geo.x_j2000_to_itrf93[time_idx],
        )
        row0 = 3 * obs_idx
        _solution, _sensitivity, position_initial = (
            one_way_light_time_position_initial_state_jacobian(
                *common_args,
                apply_stellar=use_stellar,
                observer_reference_velocity_j2000_mps=(
                    None
                    if observer_reference_velocity is None
                    else observer_reference_velocity[time_idx]
                ),
            )
        )
        # position_initial already contains Phi_r(t_t,t0), including the CN+S
        # direction chain when active; another STM would double-map it.
        h_initial[row0 : row0 + 3, :] = position_initial
    return h_initial


def apply_stellar_aberration(
    rho_vec_inertial: ArrayLike,
    observer_velocity_inertial: ArrayLike,
    *,
    light_speed_mps: float = C_LIGHT_MPS,
) -> np.ndarray:
    """Reception-case Newtonian stellar aberration (SPICE ``+S`` model).

    Rotate the light-time corrected inertial line-of-sight vector
    ``rho_vec_inertial`` *toward* the observer's inertial velocity by the
    aberration angle ``phi`` where ``sin(phi) = (v / c) * sin(w)`` and ``w`` is
    the angle between the line of sight and the observer velocity. The rotation
    axis is ``h = rho x v_obs`` and the rotation is evaluated with the Rodrigues
    formula. The returned vector has the same magnitude as the input (a pure
    rotation), so range derived from it is unchanged.

    ``observer_velocity_inertial`` must be expressed in the same axes as
    ``rho_vec_inertial`` (here J2000); do not pass an ECEF velocity. The helper
    itself is frame-agnostic -- the caller selects the inertial reference: the
    SPICE-like ``+S`` correction uses the observer velocity relative to the
    solar-system barycentre, while the cheaper local-MCI model uses the velocity
    relative to the Moon. Both are self-consistent when the identical model is
    used in generation and prediction, so neither introduces an estimator bias.
    """
    r = np.asarray(rho_vec_inertial, dtype=float).reshape(3)
    v = np.asarray(observer_velocity_inertial, dtype=float).reshape(3)
    r_norm = float(np.linalg.norm(r))
    v_norm = float(np.linalg.norm(v))
    if r_norm == 0.0 or v_norm == 0.0:
        return r.copy()
    h = np.cross(r, v)
    h_norm = float(np.linalg.norm(h))
    if h_norm == 0.0:
        return r.copy()  # line of sight parallel to velocity: sin(w) = 0
    k_hat = h / h_norm
    sin_w = h_norm / (r_norm * v_norm)
    sin_phi = float(np.clip((v_norm / light_speed_mps) * sin_w, -1.0, 1.0))
    phi = float(np.arcsin(sin_phi))
    cos_phi = float(np.cos(phi))
    # Rodrigues rotation of r about unit axis k_hat by +phi (toward v_obs).
    return (
        r * cos_phi
        + np.cross(k_hat, r) * sin_phi
        + k_hat * float(np.dot(k_hat, r)) * (1.0 - cos_phi)
    )


def _stellar_aberration_tangent_basis(unit_los_inertial: ArrayLike) -> np.ndarray:
    """Construct a deterministic orthonormal basis tangent to a unit LOS."""
    unit_los = np.asarray(unit_los_inertial, dtype=float).reshape(3)
    if not np.all(np.isfinite(unit_los)):
        raise MeasurementJacobianError("Stellar-aberration LOS must be finite.")
    unit_norm = float(np.linalg.norm(unit_los))
    if unit_norm <= 0.0 or abs(unit_norm - 1.0) > 1.0e-12:
        raise MeasurementJacobianError(
            "Stellar-aberration local Jacobian requires a unit input LOS."
        )
    reference_axis = np.eye(3)[int(np.argmin(np.abs(unit_los)))]
    tangent_1 = np.cross(unit_los, reference_axis)
    tangent_1 /= np.linalg.norm(tangent_1)
    tangent_2 = np.cross(unit_los, tangent_1)
    tangent_2 /= np.linalg.norm(tangent_2)
    return np.column_stack([tangent_1, tangent_2])


def _stellar_aberration_local_jacobian(
    unit_los_inertial: ArrayLike,
    observer_velocity_inertial: ArrayLike,
    *,
    light_speed_mps: float = C_LIGHT_MPS,
) -> _StellarAberrationLocalJacobian:
    """Differentiate the production aberration transform on the unit sphere.

    Normalized central perturbations are applied along two tangent directions.
    The ``2*h`` denominator is the derivative with respect to the tangent
    coordinate at zero; normalization changes the curve only at second order.
    The returned ambient 3x3 operator is defined by its action on tangent
    perturbations, which is the only action used by the M2.2 unit-LOS
    initial-state sensitivity.
    """
    unit_los = np.asarray(unit_los_inertial, dtype=float).reshape(3)
    velocity = np.asarray(observer_velocity_inertial, dtype=float).reshape(3)
    if not np.all(np.isfinite(velocity)):
        raise MeasurementJacobianError("Stellar-aberration observer velocity must be finite.")
    light_speed = float(light_speed_mps)
    if not np.isfinite(light_speed) or light_speed <= 0.0:
        raise MeasurementJacobianError("Stellar-aberration light speed must be finite and positive.")
    if float(np.linalg.norm(velocity)) >= light_speed:
        raise MeasurementJacobianError(
            "Stellar-aberration observer speed must be below the speed of light."
        )

    tangent_basis = _stellar_aberration_tangent_basis(unit_los)
    step = ABERRATION_LOCAL_JACOBIAN_STEP

    apparent_raw = apply_stellar_aberration(
        unit_los, velocity, light_speed_mps=light_speed
    )
    apparent_norm = float(np.linalg.norm(apparent_raw))
    if not np.isfinite(apparent_norm) or apparent_norm <= 0.0:
        raise MeasurementJacobianError(
            "Stellar-aberration transform returned an invalid apparent LOS."
        )
    apparent_unit = apparent_raw / apparent_norm

    directional_derivatives = np.empty((3, 2), dtype=float)
    for column in range(2):
        tangent = tangent_basis[:, column]
        unit_plus = unit_los + step * tangent
        unit_minus = unit_los - step * tangent
        unit_plus /= np.linalg.norm(unit_plus)
        unit_minus /= np.linalg.norm(unit_minus)
        apparent_plus = apply_stellar_aberration(
            unit_plus, velocity, light_speed_mps=light_speed
        )
        apparent_minus = apply_stellar_aberration(
            unit_minus, velocity, light_speed_mps=light_speed
        )
        apparent_plus /= np.linalg.norm(apparent_plus)
        apparent_minus /= np.linalg.norm(apparent_minus)
        directional_derivatives[:, column] = (
            apparent_plus - apparent_minus
        ) / (2.0 * step)

    local_jacobian = directional_derivatives @ tangent_basis.T
    return _StellarAberrationLocalJacobian(
        apparent_unit_los=apparent_unit,
        local_jacobian=local_jacobian,
        tangent_basis=tangent_basis,
        step=step,
        norm_error=abs(apparent_norm - 1.0),
    )


def _apparent_position_observable(
    receive_time_s: float,
    station,
    t_grid_s: ArrayLike,
    state_history_mci: ArrayLike,
    earth_pos_mci_rx: ArrayLike,
    x_j2k_itrf_rx: ArrayLike,
    *,
    observer_earth_vel_rx: ArrayLike | None = None,
    apply_stellar: bool = False,
    light_speed_mps: float = C_LIGHT_MPS,
    tolerance_s: float = 1e-12,
    max_iter: int = 10,
) -> tuple[np.ndarray, float, float, int]:
    """Apparent (one-way light-time corrected) [range, az, el] for one receive epoch.

    The spacecraft position is evaluated at the transmit time ``t_t = t_r - tau``;
    the Earth centre, station, and topocentric SEZ frame are evaluated at the
    receive time ``t_r``. ``earth_pos_mci_rx`` and ``x_j2k_itrf_rx`` are the
    receive-epoch values (already indexed, not interpolated). The transmit-time
    spacecraft state is cubic-Hermite interpolated from ``state_history_mci``;
    linear extrapolation is used when ``t_t`` falls just before the grid start
    (e.g. at the first receive epoch).

    When ``apply_stellar`` is set, the converged-light-time inertial line of
    sight is additionally rotated by the reception-case stellar aberration
    correction (:func:`apply_stellar_aberration`) before az/el/range are formed.
    This requires ``observer_earth_vel_rx``: the Earth-centre velocity (in J2000
    axes) appropriate for the chosen aberration frame -- velocity relative to the
    Moon for the local-MCI model, or relative to the SSB for the SPICE-like model.
    The station's Earth-rotation velocity (from the 6x6 transform) is added to it.
    The light-time solution itself is unchanged.
    """
    earth_pos_mci_rx = np.asarray(earth_pos_mci_rx, dtype=float).reshape(3)
    x_rx = np.asarray(x_j2k_itrf_rx, dtype=float)
    r_rot_rx = x_rx[:3, :3]
    station_ecef = np.asarray(station.r_ecef_m, dtype=float).reshape(3)
    # Station inertial (MCI) state at the receive time: Earth centre plus the
    # ITRF93->J2000 transform of the (fixed) station ECEF state (6x6 solve). The
    # velocity part carries the inertial velocity due to Earth rotation.
    station_ecef_state = np.concatenate([station_ecef, np.zeros(3)])
    station_rel_state_j2000 = np.linalg.solve(x_rx, station_ecef_state)
    station_rel_j2000 = station_rel_state_j2000[:3]
    station_mci_rx = earth_pos_mci_rx + station_rel_j2000

    solution = solve_one_way_light_time(
        receive_time_s,
        station_mci_rx,
        lambda t: interp_state_history(t_grid_s, state_history_mci, t)[:3],
        light_speed_mps=light_speed_mps,
        tolerance_s=tolerance_s,
        max_iter=max_iter,
    )
    transmit_time_s = solution.transmit_time_s
    r_sc_tt = interp_state_history(t_grid_s, state_history_mci, transmit_time_s)[:3]

    if apply_stellar:
        if observer_earth_vel_rx is None:
            raise ValueError("apply_stellar=True requires observer_earth_vel_rx.")
        # Observer (station) velocity is evaluated at receive time in J2000 axes
        # and includes both reference-center translation and Earth rotation.
        observer_vel_inertial = _observer_velocity_j2000_at_receive_epoch(
            station,
            x_rx,
            observer_earth_vel_rx,
            station_relative_state_j2000=station_rel_state_j2000,
        )
        # CN-corrected inertial line of sight, then stellar aberration rotation.
        rho_inertial = apply_stellar_aberration(
            r_sc_tt - station_mci_rx,
            observer_vel_inertial,
            light_speed_mps=light_speed_mps,
        )
        rho_ecef = r_rot_rx @ rho_inertial
    else:
        # Apparent line of sight: spacecraft at transmit time, Earth/frame/station
        # at receive time. ||rho_ecef|| equals the converged light-time range.
        rho_ecef = r_rot_rx @ (r_sc_tt - earth_pos_mci_rx) - station_ecef

    az_rad, el_rad, range_m = ecef2razel_sez(rho_ecef, station.lat_rad, station.lon_rad)
    z = np.array([range_m, az_rad, el_rad], dtype=float)
    return z, transmit_time_s, solution.light_time_s, int(solution.iterations)


def _apparent_position_rowwise(
    state_history_mci: np.ndarray,
    obs_data: np.ndarray,
    pass_geo: "PassGeometry",
    *,
    apply_stellar_aberration: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Row-by-row apparent observables for an obs_data block.

    Returns ``(h_meas, r_transmit_mci)`` where ``h_meas`` is (n_obs, 3) and
    ``r_transmit_mci`` is the (n_obs, 3) spacecraft position evaluated at each
    transmit time (used by the approximate light-time analytic Jacobian).

    When stellar aberration is active (``pass_geo.apply_stellar_aberration`` or
    the ``apply_stellar_aberration`` override), the apparent line of sight is
    rotated by the reception-case stellar aberration correction; ``r_transmit``
    is unaffected (it is the transmit-time spacecraft position).
    """
    use_stellar = bool(
        getattr(pass_geo, "apply_stellar_aberration", False)
        if apply_stellar_aberration is None
        else apply_stellar_aberration
    )
    model = getattr(pass_geo, "stellar_aberration_model", "local_mci")
    profile = getattr(pass_geo, "measurement_model_profile", "geometric_instantaneous")
    if profile != "geometric_instantaneous":
        _profile, _light_time, profile_stellar, profile_model = (
            normalize_measurement_model_profile(profile)
        )
        if profile_stellar:
            use_stellar = True
            model = profile_model
    if use_stellar and model == "spice_ssb":
        earth_vel = getattr(pass_geo, "earth_vel_ssb_j2000_mps", None)
        if earth_vel is None:
            raise ValueError(
                "stellar_aberration_model='spice_ssb' requires "
                "pass_geo.earth_vel_ssb_j2000_mps (regenerate with the SSB model)."
            )
    else:
        earth_vel = getattr(pass_geo, "earth_vel_mci_mps", None)
    n_obs = obs_data.shape[0]
    h_meas = np.zeros((n_obs, 3), dtype=float)
    r_tx = np.zeros((n_obs, 3), dtype=float)
    for i in range(n_obs):
        station_id = int(obs_data[i, 4]) - 1
        time_idx = int(obs_data[i, 5]) - 1
        receive_time_s = float(obs_data[i, 0])
        z, transmit_time_s, _lt, _it = _apparent_position_observable(
            receive_time_s,
            pass_geo.stations[station_id],
            pass_geo.t_s,
            state_history_mci,
            pass_geo.earth_pos_mci_m[time_idx],
            pass_geo.x_j2000_to_itrf93[time_idx],
            observer_earth_vel_rx=(
                earth_vel[time_idx] if (use_stellar and earth_vel is not None) else None
            ),
            apply_stellar=use_stellar,
        )
        h_meas[i] = z
        r_tx[i] = interp_state_history(pass_geo.t_s, state_history_mci, transmit_time_s)[:3]
    return h_meas, r_tx


def generate_position_measurements(
    t_pass_s: ArrayLike,
    state_history_mci: ArrayLike,
    stations,
    vis_mask_raw: ArrayLike,
    get_earth_pos,
    get_earth_vel,
    et0: float,
    *,
    noise: bool = True,
    rng: np.random.Generator | None = None,
    arc_id: int | None = None,
    apply_light_time: bool = False,
    apply_stellar_aberration: bool = False,
    stellar_aberration_model: str = "local_mci",
    measurement_model_profile: str | None = None,
    jacobian_model: str | None = None,
    noise_seed: int | None = None,
) -> tuple[np.ndarray, PassGeometry, np.ndarray]:
    """Generate range/azimuth/elevation measurements.

    ObsData follows MATLAB's convention:
    `[t, range, az, el, station_id_1based, time_index_1based, optional_arc_id]`.
    """
    import spiceypy as spice

    t_pass_s = np.asarray(t_pass_s, dtype=float).reshape(-1)
    state_history_mci = np.asarray(state_history_mci, dtype=float)
    vis_mask_raw = np.asarray(vis_mask_raw, dtype=bool)
    if vis_mask_raw.ndim == 1:
        vis_mask_raw = vis_mask_raw.reshape(-1, 1)

    n_steps = t_pass_s.size
    if state_history_mci.shape[0] != n_steps or state_history_mci.shape[1] < 3:
        raise ValueError("state_history_mci must have shape (N, >=3).")
    (
        measurement_model_profile,
        apply_light_time,
        apply_stellar_aberration,
        stellar_aberration_model,
    ) = normalize_measurement_model_profile(
        measurement_model_profile,
        apply_light_time=apply_light_time,
        apply_stellar_aberration=apply_stellar_aberration,
        stellar_aberration_model=stellar_aberration_model,
    )
    jacobian_model = normalize_jacobian_model(
        jacobian_model,
        measurement_model_profile=measurement_model_profile,
        companion_geometry="instantaneous",
    )
    if apply_light_time and state_history_mci.shape[1] < 6:
        raise ValueError(
            "apply_light_time=True requires a 6-state (position+velocity) history."
        )
    if vis_mask_raw.shape != (n_steps, len(stations)):
        raise ValueError("vis_mask_raw must have shape (N, num_stations).")

    r_earth_mci = _ensure_n_by_3(get_earth_pos(t_pass_s), n_steps, "r_earth_mci")
    v_earth_mci = _ensure_n_by_3(get_earth_vel(t_pass_s), n_steps, "v_earth_mci")
    xforms = np.zeros((n_steps, 6, 6), dtype=float)
    # SPICE-like CN+S uses the observer velocity relative to the SSB (J2000 axes).
    need_ssb = apply_stellar_aberration and stellar_aberration_model == "spice_ssb"
    v_earth_ssb_j2000 = np.zeros((n_steps, 3), dtype=float) if need_ssb else None

    include_arc_id = arc_id is not None
    num_cols = 7 if include_arc_id else 6
    total_measurements = int(np.sum(vis_mask_raw))
    obs_data = np.zeros((total_measurements, num_cols), dtype=float)
    clean_obs_data = np.zeros((total_measurements, num_cols), dtype=float)
    rng = rng or np.random.default_rng()

    obs_counter = 0
    for k, t_s in enumerate(t_pass_s):
        x_j2k_itrf = np.asarray(spice.sxform("J2000", "ITRF93", float(et0 + t_s)), dtype=float)
        xforms[k, :, :] = x_j2k_itrf
        if need_ssb:
            # Earth-centre state w.r.t. the SSB in J2000 axes; spkezr returns
            # km / (km/s), so scale velocity to m/s. MCI axes are J2000-aligned,
            # hence this velocity is already in the LOS vector basis.
            state_e, _lt_e = spice.spkezr("EARTH", float(et0 + t_s), "J2000", "NONE", "SSB")
            v_earth_ssb_j2000[k, :] = np.asarray(state_e[3:6], dtype=float) * 1000.0

        active_station_cols = np.where(vis_mask_raw[k, :])[0]
        if active_station_cols.size == 0:
            continue

        r_sat_mci = state_history_mci[k, :3]
        dr_eci = r_sat_mci - r_earth_mci[k, :]
        r_sat_ecef = x_j2k_itrf[:3, :3] @ dr_eci

        for station_col in active_station_cols:
            station = stations[station_col]
            if apply_light_time:
                if apply_stellar_aberration:
                    obs_earth_vel = (
                        v_earth_ssb_j2000[k]
                        if stellar_aberration_model == "spice_ssb"
                        else v_earth_mci[k]
                    )
                else:
                    obs_earth_vel = None
                z_clean, _t_t, _lt, _it = _apparent_position_observable(
                    float(t_s), station, t_pass_s, state_history_mci,
                    r_earth_mci[k], x_j2k_itrf,
                    observer_earth_vel_rx=obs_earth_vel,
                    apply_stellar=apply_stellar_aberration,
                )
            else:
                rho_vec_ecef = r_sat_ecef - station.r_ecef_m
                az_rad, el_rad, range_m = ecef2razel_sez(rho_vec_ecef, station.lat_rad, station.lon_rad)
                z_clean = np.array([range_m, az_rad, el_rad], dtype=float)

            bias_vec = np.asarray(getattr(station, "bias", np.zeros(3)), dtype=float).reshape(-1)
            if bias_vec.size != 3:
                bias_vec = np.zeros(3)
            if noise:
                noise_vec = np.array(
                    [
                        station.sigma_range_m * rng.standard_normal(),
                        station.sigma_angle_rad * rng.standard_normal(),
                        station.sigma_angle_rad * rng.standard_normal(),
                    ],
                    dtype=float,
                )
            else:
                noise_vec = np.zeros(3)
            z_noisy = z_clean + noise_vec + bias_vec

            station_id_1based = station_col + 1
            time_index_1based = k + 1
            row_noisy = [t_s, z_noisy[0], z_noisy[1], z_noisy[2], station_id_1based, time_index_1based]
            row_clean = [t_s, z_clean[0], z_clean[1], z_clean[2], station_id_1based, time_index_1based]
            if include_arc_id:
                row_noisy.append(float(arc_id))
                row_clean.append(float(arc_id))
            obs_data[obs_counter, :] = row_noisy
            clean_obs_data[obs_counter, :] = row_clean
            obs_counter += 1

    pass_geo = PassGeometry(
        t_s=t_pass_s,
        earth_pos_mci_m=r_earth_mci,
        earth_vel_mci_mps=v_earth_mci,
        x_j2000_to_itrf93=xforms,
        stations=tuple(stations),
        measurement_type="position",
        apply_light_time=apply_light_time,
        apply_stellar_aberration=apply_stellar_aberration,
        stellar_aberration_model=stellar_aberration_model,
        earth_vel_ssb_j2000_mps=v_earth_ssb_j2000,
        measurement_model_profile=measurement_model_profile,
        companion_geometry="instantaneous",
        jacobian_model=jacobian_model,
    )
    object.__setattr__(
        pass_geo,
        "measurement_metadata",
        measurement_model_metadata(pass_geo, noise_enabled=noise, noise_seed=noise_seed),
    )
    return obs_data[:obs_counter, :], pass_geo, clean_obs_data[:obs_counter, :]


def compute_position_residuals(
    state_history_mci: ArrayLike,
    obs_data: ArrayLike,
    pass_geo: PassGeometry,
    *,
    apply_light_time: bool = False,
    apply_stellar_aberration: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute position-only observed-minus-computed residuals.

    When ``pass_geo.apply_light_time`` (or the ``apply_light_time`` override) is
    set, the computed measurements use the one-way light-time corrected apparent
    geometry, row by row, matching :func:`generate_position_measurements`.
    Otherwise the Numba instantaneous fast path is used unchanged. The residual
    sign convention (observed - computed) and angle wrapping are unchanged."""
    state_history_mci = np.asarray(state_history_mci, dtype=float)
    obs_data = np.asarray(obs_data, dtype=float)
    profile_lt = False
    profile_stellar = False
    if getattr(pass_geo, "measurement_model_profile", "geometric_instantaneous") != "geometric_instantaneous":
        _profile, profile_lt, profile_stellar, _model = normalize_measurement_model_profile(
            getattr(pass_geo, "measurement_model_profile", "geometric_instantaneous")
        )
    use_stellar = bool(
        getattr(pass_geo, "apply_stellar_aberration", False) or profile_stellar
        if apply_stellar_aberration is None
        else apply_stellar_aberration
    )
    use_light_time = bool(
        getattr(pass_geo, "apply_light_time", False) or profile_lt or apply_light_time or use_stellar
    )
    if not use_light_time:
        h_meas = position_observables(
            state_history_mci,
            obs_data,
            pass_geo.earth_pos_mci_m,
            pass_geo.x_j2000_to_itrf93,
            pass_geo.stations,
        )
    else:
        h_meas, _ = _apparent_position_rowwise(
            state_history_mci, obs_data, pass_geo, apply_stellar_aberration=use_stellar
        )

    diff_raw = obs_data[:, 1:4] - h_meas
    diff_raw[:, 1] = wrap_to_pi(diff_raw[:, 1])
    diff_raw[:, 2] = wrap_to_pi(diff_raw[:, 2])
    _zero_tiny_position_residuals(diff_raw)
    residuals = diff_raw.T.reshape(-1, order="F")
    return residuals, h_meas


def _instantaneous_range_rate_companion(
    state_sat_mci: np.ndarray,
    earth_pos_mci: np.ndarray,
    xform_j2000_to_itrf93: np.ndarray,
    station,
) -> tuple[float, float, float]:
    dr_eci = state_sat_mci[:3] - earth_pos_mci
    r_sat_ecef = xform_j2000_to_itrf93[:3, :3] @ dr_eci
    rho_vec_ecef = r_sat_ecef - station.r_ecef_m
    az_rad, el_rad, range_m = ecef2razel_sez(rho_vec_ecef, station.lat_rad, station.lon_rad)
    return float(range_m), float(az_rad), float(el_rad)


def _range_rate_companion_observable(
    receive_time_s: float,
    station,
    t_grid_s: np.ndarray,
    state_history_mci: np.ndarray,
    earth_pos_mci: np.ndarray,
    xform_j2000_to_itrf93: np.ndarray,
    *,
    companion_geometry: str,
) -> tuple[float, float, float]:
    companion_geometry = normalize_companion_geometry(companion_geometry)
    if companion_geometry == "instantaneous":
        z = _instantaneous_range_rate_companion(
            interp_state_history(t_grid_s, state_history_mci, receive_time_s),
            earth_pos_mci,
            xform_j2000_to_itrf93,
            station,
        )
        return z
    z, _t_t, _lt, _it = _apparent_position_observable(
        receive_time_s,
        station,
        t_grid_s,
        state_history_mci,
        earth_pos_mci,
        xform_j2000_to_itrf93,
        apply_stellar=False,
    )
    return float(z[0]), float(z[1]), float(z[2])


def generate_range_rate_measurements(
    t_pass_s: ArrayLike,
    state_history_mci: ArrayLike,
    stations,
    vis_mask_raw: ArrayLike,
    get_earth_pos,
    get_earth_vel,
    et0: float,
    *,
    bias_range_m: float = 0.0,
    bias_rr_mps: float = 0.0,
    bias_az_rad: float = 0.0,
    bias_el_rad: float = 0.0,
    noise: bool = True,
    rng: np.random.Generator | None = None,
    arc_id: int | None = None,
    range_rate_physics: RangeRatePhysicsConfig | str | None = None,
    companion_geometry: str = "instantaneous",
    measurement_model_profile: str | None = None,
    jacobian_model: str | None = None,
    noise_seed: int | None = None,
) -> tuple[np.ndarray, PassGeometry]:
    """Generate range/range-rate/azimuth/elevation measurements."""
    import spiceypy as spice

    t_pass_s = np.asarray(t_pass_s, dtype=float).reshape(-1)
    state_history_mci = np.asarray(state_history_mci, dtype=float)
    vis_mask_raw = np.asarray(vis_mask_raw, dtype=bool)
    if vis_mask_raw.ndim == 1:
        vis_mask_raw = vis_mask_raw.reshape(-1, 1)

    n_steps = t_pass_s.size
    if state_history_mci.shape[0] != n_steps or state_history_mci.shape[1] != 6:
        raise ValueError("state_history_mci must have shape (N, 6).")
    if vis_mask_raw.shape != (n_steps, len(stations)):
        raise ValueError("vis_mask_raw must have shape (N, num_stations).")
    rr_physics = range_rate_physics_config(range_rate_physics)
    companion_geometry = normalize_companion_geometry(companion_geometry)
    if measurement_model_profile is None:
        if rr_physics.mode == "two_way_counted_doppler":
            measurement_model_profile = "two_way_counted_doppler"
        elif companion_geometry == "apparent_one_way":
            measurement_model_profile = "one_way_light_time"
        else:
            measurement_model_profile = "geometric_instantaneous"
    measurement_model_profile, _lt, _stellar, _stellar_model = normalize_measurement_model_profile(
        measurement_model_profile,
        allow_two_way=True,
    )
    jacobian_model = normalize_jacobian_model(
        jacobian_model,
        measurement_model_profile=measurement_model_profile,
        companion_geometry=companion_geometry,
    )

    r_earth_mci = _ensure_n_by_3(get_earth_pos(t_pass_s), n_steps, "r_earth_mci")
    v_earth_mci = _ensure_n_by_3(get_earth_vel(t_pass_s), n_steps, "v_earth_mci")
    xforms = np.zeros((n_steps, 6, 6), dtype=float)
    for k, t_s in enumerate(t_pass_s):
        xforms[k, :, :] = np.asarray(spice.sxform("J2000", "ITRF93", float(et0 + t_s)), dtype=float)

    include_arc_id = arc_id is not None
    num_cols = 8 if include_arc_id else 7
    total_measurements = int(np.sum(vis_mask_raw))
    obs_data = np.zeros((total_measurements, num_cols), dtype=float)
    rng = rng or np.random.default_rng()

    obs_counter = 0
    for k, t_s in enumerate(t_pass_s):
        active_station_cols = np.where(vis_mask_raw[k, :])[0]
        if active_station_cols.size == 0:
            continue

        state_sat_mci = state_history_mci[k, :]
        state_earth_mci = np.concatenate([r_earth_mci[k, :], v_earth_mci[k, :]])
        state_sat_eci = state_sat_mci - state_earth_mci
        state_sat_ecef = xforms[k, :, :] @ state_sat_eci
        r_sat_ecef = state_sat_ecef[:3]
        v_sat_ecef = state_sat_ecef[3:]

        for station_col in active_station_cols:
            station = stations[station_col]
            rho_vec_ecef = r_sat_ecef - station.r_ecef_m
            rho_dot_ecef = v_sat_ecef

            if companion_geometry == "instantaneous":
                range_ideal = float(np.linalg.norm(rho_vec_ecef))
                az_ideal, el_ideal, _ = ecef2razel_sez(rho_vec_ecef, station.lat_rad, station.lon_rad)
            else:
                range_ideal, az_ideal, el_ideal = _range_rate_companion_observable(
                    float(t_s),
                    station,
                    t_pass_s,
                    state_history_mci,
                    r_earth_mci[k],
                    xforms[k, :, :],
                    companion_geometry=companion_geometry,
                )
            if rr_physics.mode == "geometric_instantaneous":
                rr_ideal = instantaneous_geometric_range_rate(rho_vec_ecef, rho_dot_ecef)
            else:
                rr_ideal = two_way_counted_doppler_observable(
                    float(t_s),
                    station,
                    t_pass_s,
                    state_history_mci,
                    r_earth_mci,
                    v_earth_mci,
                    xforms,
                    rr_physics,
                )

            station_bias = np.asarray(getattr(station, "bias", []), dtype=float).reshape(-1)
            if station_bias.size == 4:
                bias_vec = station_bias
            else:
                bias_vec = np.array([bias_range_m, bias_rr_mps, bias_az_rad, bias_el_rad], dtype=float)

            if noise:
                noise_vec = np.array(
                    [
                        station.sigma_range_m * rng.standard_normal(),
                        station.sigma_range_rate_mps * rng.standard_normal(),
                        station.sigma_angle_rad * rng.standard_normal(),
                        station.sigma_angle_rad * rng.standard_normal(),
                    ],
                    dtype=float,
                )
            else:
                noise_vec = np.zeros(4)

            z = np.array([range_ideal, rr_ideal, az_ideal, el_ideal], dtype=float) + bias_vec + noise_vec
            station_id_1based = station_col + 1
            time_index_1based = k + 1
            row = [t_s, z[0], z[1], z[2], z[3], station_id_1based, time_index_1based]
            if include_arc_id:
                row.append(float(arc_id))
            obs_data[obs_counter, :] = row
            obs_counter += 1

    pass_geo = PassGeometry(
        t_s=t_pass_s,
        earth_pos_mci_m=r_earth_mci,
        earth_vel_mci_mps=v_earth_mci,
        x_j2000_to_itrf93=xforms,
        stations=tuple(stations),
        measurement_type="range_rate",
        range_rate_physics=rr_physics,
        measurement_model_profile=measurement_model_profile,
        companion_geometry=companion_geometry,
        jacobian_model=jacobian_model,
    )
    object.__setattr__(
        pass_geo,
        "measurement_metadata",
        measurement_model_metadata(pass_geo, noise_enabled=noise, noise_seed=noise_seed),
    )
    return obs_data[:obs_counter, :], pass_geo


def compute_range_rate_residuals(
    state_history_mci: ArrayLike,
    obs_data: ArrayLike,
    pass_geo: PassGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute range-rate observed-minus-computed residuals."""
    state_history_mci = np.asarray(state_history_mci, dtype=float)
    obs_data = np.asarray(obs_data, dtype=float)
    n_obs = obs_data.shape[0]
    h_meas = np.zeros((n_obs, 4), dtype=float)
    rr_physics = range_rate_physics_config(pass_geo.range_rate_physics)
    companion_geometry = normalize_companion_geometry(
        getattr(pass_geo, "companion_geometry", "instantaneous")
    )

    if rr_physics.mode == "geometric_instantaneous" and companion_geometry == "instantaneous":
        h_meas = geometric_range_rate_observables(
            state_history_mci,
            obs_data,
            pass_geo.earth_pos_mci_m,
            pass_geo.earth_vel_mci_mps,
            pass_geo.x_j2000_to_itrf93,
            pass_geo.stations,
        )
        diff_raw = obs_data[:, 1:5] - h_meas
        diff_raw[:, 2] = wrap_to_pi(diff_raw[:, 2])
        diff_raw[:, 3] = wrap_to_pi(diff_raw[:, 3])
        _zero_tiny_range_rate_residuals(diff_raw)
        residuals = diff_raw.T.reshape(-1, order="F")
        return residuals, h_meas

    for obs_idx in range(n_obs):
        station_id = int(obs_data[obs_idx, 5]) - 1
        time_idx = int(obs_data[obs_idx, 6]) - 1
        station = pass_geo.stations[station_id]

        state_sat_mci = state_history_mci[time_idx, :]
        state_earth_mci = np.concatenate(
            [pass_geo.earth_pos_mci_m[time_idx, :], pass_geo.earth_vel_mci_mps[time_idx, :]]
        )
        state_sat_eci = state_sat_mci - state_earth_mci
        state_sat_ecef = pass_geo.x_j2000_to_itrf93[time_idx, :, :] @ state_sat_eci
        v_sat_ecef = state_sat_ecef[3:]

        r_rel_ecef = state_sat_ecef[:3] - station.r_ecef_m
        v_rel_ecef = v_sat_ecef
        if companion_geometry == "instantaneous":
            range_val = float(np.linalg.norm(r_rel_ecef))
            az_rad, el_rad, _ = ecef2razel_sez(r_rel_ecef, station.lat_rad, station.lon_rad)
        else:
            range_val, az_rad, el_rad = _range_rate_companion_observable(
                float(pass_geo.t_s[time_idx]),
                station,
                pass_geo.t_s,
                state_history_mci,
                pass_geo.earth_pos_mci_m[time_idx, :],
                pass_geo.x_j2000_to_itrf93[time_idx, :, :],
                companion_geometry=companion_geometry,
            )
        if rr_physics.mode == "geometric_instantaneous":
            rr_val = instantaneous_geometric_range_rate(r_rel_ecef, v_rel_ecef)
        else:
            rr_val = two_way_counted_doppler_observable(
                float(pass_geo.t_s[time_idx]),
                station,
                pass_geo.t_s,
                state_history_mci,
                pass_geo.earth_pos_mci_m,
                pass_geo.earth_vel_mci_mps,
                pass_geo.x_j2000_to_itrf93,
                rr_physics,
            )
        h_meas[obs_idx, :] = [range_val, rr_val, az_rad, el_rad]

    diff_raw = obs_data[:, 1:5] - h_meas
    diff_raw[:, 2] = wrap_to_pi(diff_raw[:, 2])
    diff_raw[:, 3] = wrap_to_pi(diff_raw[:, 3])
    _zero_tiny_range_rate_residuals(diff_raw)
    residuals = diff_raw.T.reshape(-1, order="F")
    return residuals, h_meas


def _zero_tiny_position_residuals(diff_raw: np.ndarray) -> None:
    """Suppress insignificant arithmetic roundoff in synthetic closure tests."""
    if diff_raw.size == 0:
        return
    diff_raw[np.abs(diff_raw[:, 0]) < 1e-7, 0] = 0.0
    diff_raw[np.abs(diff_raw[:, 1]) < 1e-14, 1] = 0.0
    diff_raw[np.abs(diff_raw[:, 2]) < 1e-14, 2] = 0.0


def _zero_tiny_range_rate_residuals(diff_raw: np.ndarray) -> None:
    """Suppress insignificant arithmetic roundoff in synthetic closure tests."""
    if diff_raw.size == 0:
        return
    diff_raw[np.abs(diff_raw[:, 0]) < 1e-7, 0] = 0.0
    diff_raw[np.abs(diff_raw[:, 1]) < 1e-10, 1] = 0.0
    diff_raw[np.abs(diff_raw[:, 2]) < 1e-14, 2] = 0.0
    diff_raw[np.abs(diff_raw[:, 3]) < 1e-14, 3] = 0.0


def compute_position_residuals_analytic(
    state_history_mci: ArrayLike,
    obs_data: ArrayLike,
    pass_geo: PassGeometry,
    *,
    apply_light_time: bool = False,
    apply_stellar_aberration: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Position-only residuals and local analytic 3x6 H_tilde blocks.

    Light-time corrected residuals are exact for the implemented model.  The
    default light-time Jacobian remains the legacy first-order approximation.
    With ``jacobian_model='implicit_light_time'``, the returned block remains a
    local receive-epoch state Jacobian.  For CN geometry its range and angle rows
    include implicit light-time and LOS-normalization sensitivity.  When stellar
    aberration is active, only the range row is implicit and the angle rows keep
    the documented first-order approximation."""
    residuals, h_meas = compute_position_residuals(
        state_history_mci, obs_data, pass_geo,
        apply_light_time=apply_light_time,
        apply_stellar_aberration=apply_stellar_aberration,
    )
    state_history_mci = np.asarray(state_history_mci, dtype=float)
    obs_data = np.asarray(obs_data, dtype=float)
    n_obs = obs_data.shape[0]

    station_ids = obs_data[:, 4].astype(int) - 1    # (n_obs,)
    time_idxs   = obs_data[:, 5].astype(int) - 1    # (n_obs,)

    n_stations = len(pass_geo.stations)
    c_sez_all = np.stack(
        [ecef2sez_dcm(pass_geo.stations[i].lat_rad, pass_geo.stations[i].lon_rad)
         for i in range(n_stations)]
    )  # (n_stations, 3, 3)
    r_ecef_all = np.stack(
        [pass_geo.stations[i].r_ecef_m for i in range(n_stations)]
    )  # (n_stations, 3)

    profile_lt = False
    profile_stellar = False
    if getattr(pass_geo, "measurement_model_profile", "geometric_instantaneous") != "geometric_instantaneous":
        _profile, profile_lt, profile_stellar, _model = normalize_measurement_model_profile(
            getattr(pass_geo, "measurement_model_profile", "geometric_instantaneous")
        )
    use_stellar = bool(
        getattr(pass_geo, "apply_stellar_aberration", False) or profile_stellar
        if apply_stellar_aberration is None
        else apply_stellar_aberration
    )
    use_light_time = bool(
        getattr(pass_geo, "apply_light_time", False) or profile_lt or apply_light_time or use_stellar
    )
    if use_light_time:
        # Apparent geometry: spacecraft evaluated at each transmit time. The H
        # block below reuses the instantaneous formula at this apparent LOS
        # (first-order receive-epoch STM approximation; the stellar aberration
        # rotation derivative is neglected per the first-stage design).
        _, r_sat = _apparent_position_rowwise(
            state_history_mci, obs_data, pass_geo, apply_stellar_aberration=use_stellar
        )
    else:
        r_sat = state_history_mci[time_idxs, :3]                    # (n_obs, 3)
    dr_eci = r_sat - pass_geo.earth_pos_mci_m[time_idxs, :]         # (n_obs, 3)
    R_j2k  = pass_geo.x_j2000_to_itrf93[time_idxs, :3, :3]          # (n_obs, 3, 3)

    dr_ecef  = np.einsum('nij,nj->ni', R_j2k, dr_eci)               # (n_obs, 3)
    rho_ecef = dr_ecef - r_ecef_all[station_ids]                     # (n_obs, 3)

    c_sez = c_sez_all[station_ids]                                    # (n_obs, 3, 3)
    rho_sez = np.einsum('nij,nj->ni', c_sez, rho_ecef)              # (n_obs, 3)

    south, east, zenith = rho_sez[:, 0], rho_sez[:, 1], rho_sez[:, 2]
    range_m = np.linalg.norm(rho_sez, axis=1)                        # (n_obs,)
    rho_h2  = south**2 + east**2                                      # (n_obs,)
    rho2    = range_m**2                                              # (n_obs,)
    rho_h   = np.sqrt(np.maximum(rho_h2, 1e-12))

    valid    = range_m > 1e-9
    rng_safe = np.where(valid, range_m, 1.0)
    rho2_safe = np.where(valid, rho2, 1.0)
    az_valid  = rho_h2 >= 1e-12
    rho_h2_safe = np.where(az_valid, rho_h2, 1.0)

    d_range = np.where(valid[:, None], rho_sez / rng_safe[:, None], 0.0)

    d_az = np.zeros((n_obs, 3))
    d_az[:, 0] = np.where(az_valid,  east  / rho_h2_safe, 0.0)
    d_az[:, 1] = np.where(az_valid, -south / rho_h2_safe, 0.0)

    d_el = np.column_stack([
        np.where(valid, -(zenith * south) / (rho2_safe * rho_h), 0.0),
        np.where(valid, -(zenith * east)  / (rho2_safe * rho_h), 0.0),
        np.where(valid,  rho_h / rho2_safe,                      0.0),
    ])

    d_meas_dsez = np.stack([d_range, d_az, d_el], axis=1)           # (n_obs, 3, 3)
    dsez_drmci  = np.einsum('nij,njk->nik', c_sez, R_j2k)           # (n_obs, 3, 3)
    dmeas_drmci = np.einsum('nij,njk->nik', d_meas_dsez, dsez_drmci) # (n_obs, 3, 3)

    h_tilde = np.zeros((3 * n_obs, 6), dtype=float)
    h_tilde[:, :3] = dmeas_drmci.reshape(3 * n_obs, 3)

    jacobian_model = getattr(pass_geo, "jacobian_model", "analytic_exact_geometric")
    if jacobian_model == "implicit_light_time":
        if not use_light_time:
            raise ValueError(
                "jacobian_model='implicit_light_time' requires light-time corrected position geometry."
            )
        for obs_idx in range(n_obs):
            time_idx = int(time_idxs[obs_idx])
            station = pass_geo.stations[int(station_ids[obs_idx])]
            if use_stellar:
                _solution, sensitivity = one_way_light_time_range_sensitivity(
                    float(obs_data[obs_idx, 0]),
                    station,
                    pass_geo.t_s,
                    state_history_mci,
                    pass_geo.earth_pos_mci_m[time_idx],
                    pass_geo.x_j2000_to_itrf93[time_idx],
                )
                h_tilde[3 * obs_idx, :] = sensitivity.d_range_d_state
            else:
                _solution, _sensitivity, local_block = (
                    one_way_light_time_position_local_state_jacobian(
                        float(obs_data[obs_idx, 0]),
                        station,
                        pass_geo.t_s,
                        state_history_mci,
                        pass_geo.earth_pos_mci_m[time_idx],
                        pass_geo.x_j2000_to_itrf93[time_idx],
                    )
                )
                h_tilde[3 * obs_idx : 3 * obs_idx + 3, :] = local_block

    return residuals, h_meas, h_tilde


def compute_range_rate_residuals_analytic(
    state_history_mci: ArrayLike,
    obs_data: ArrayLike,
    pass_geo: PassGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RR residuals and local analytic 4x6 H_tilde blocks."""
    rr_physics = range_rate_physics_config(pass_geo.range_rate_physics)
    companion_geometry = normalize_companion_geometry(
        getattr(pass_geo, "companion_geometry", "instantaneous")
    )
    if rr_physics.mode != "geometric_instantaneous":
        raise NotImplementedError(
            "Analytic H_tilde is currently implemented for geometric_instantaneous RR only; "
            "use compute_range_rate_residuals or UKF-style numerical propagation for two_way_counted_doppler."
        )
    residuals, h_meas = compute_range_rate_residuals(state_history_mci, obs_data, pass_geo)
    state_history_mci = np.asarray(state_history_mci, dtype=float)
    obs_data = np.asarray(obs_data, dtype=float)
    n_obs = obs_data.shape[0]

    station_ids = obs_data[:, 5].astype(int) - 1    # (n_obs,)
    time_idxs   = obs_data[:, 6].astype(int) - 1    # (n_obs,)

    n_stations = len(pass_geo.stations)
    c_sez_all = np.stack(
        [ecef2sez_dcm(pass_geo.stations[i].lat_rad, pass_geo.stations[i].lon_rad)
         for i in range(n_stations)]
    )  # (n_stations, 3, 3)
    r_ecef_all = np.stack(
        [pass_geo.stations[i].r_ecef_m for i in range(n_stations)]
    )  # (n_stations, 3)

    state_sat  = state_history_mci[time_idxs, :]                     # (n_obs, 6)
    earth_pos  = pass_geo.earth_pos_mci_m[time_idxs, :]              # (n_obs, 3)
    earth_vel  = pass_geo.earth_vel_mci_mps[time_idxs, :]            # (n_obs, 3)
    r_eci      = state_sat[:, :3] - earth_pos                        # (n_obs, 3)
    v_eci      = state_sat[:, 3:] - earth_vel                        # (n_obs, 3)
    r_comp_eci = r_eci.copy()
    if companion_geometry == "apparent_one_way":
        for obs_idx in range(n_obs):
            station = pass_geo.stations[station_ids[obs_idx]]
            receive_time_s = float(obs_data[obs_idx, 0])
            z_app, transmit_time_s, _lt, _it = _apparent_position_observable(
                receive_time_s,
                station,
                pass_geo.t_s,
                state_history_mci,
                pass_geo.earth_pos_mci_m[time_idxs[obs_idx], :],
                pass_geo.x_j2000_to_itrf93[time_idxs[obs_idx], :, :],
                apply_stellar=False,
            )
            _ = z_app
            r_tx_mci = interp_state_history(pass_geo.t_s, state_history_mci, transmit_time_s)[:3]
            r_comp_eci[obs_idx, :] = r_tx_mci - pass_geo.earth_pos_mci_m[time_idxs[obs_idx], :]

    R_j2k  = pass_geo.x_j2000_to_itrf93[time_idxs, :3, :3]          # (n_obs, 3, 3)
    dR_j2k = pass_geo.x_j2000_to_itrf93[time_idxs, 3:6, :3]         # (n_obs, 3, 3)

    r_ecef   = np.einsum('nij,nj->ni', R_j2k, r_eci)                # (n_obs, 3)
    r_comp_ecef = np.einsum('nij,nj->ni', R_j2k, r_comp_eci)        # (n_obs, 3)
    v_ecef   = (np.einsum('nij,nj->ni', dR_j2k, r_eci)
                + np.einsum('nij,nj->ni', R_j2k,  v_eci))           # (n_obs, 3)
    rho_ecef = r_ecef - r_ecef_all[station_ids]                      # (n_obs, 3)
    rho_comp_ecef = r_comp_ecef - r_ecef_all[station_ids]            # (n_obs, 3)
    v_rel    = v_ecef                                                 # (n_obs, 3)

    range_m  = np.linalg.norm(rho_ecef, axis=1)                     # (n_obs,)
    valid    = range_m >= 1e-3
    rng_safe = np.where(valid, range_m, 1.0)
    range_comp_m = np.linalg.norm(rho_comp_ecef, axis=1)
    valid_comp = range_comp_m >= 1e-3
    rng_comp_safe = np.where(valid_comp, range_comp_m, 1.0)

    u_hat = rho_ecef / rng_safe[:, None]                             # (n_obs, 3)
    rr    = np.einsum('ni,ni->n', rho_ecef, v_rel) / rng_safe       # (n_obs,)

    c_sez   = c_sez_all[station_ids]                                  # (n_obs, 3, 3)
    rho_sez = np.einsum('nij,nj->ni', c_sez, rho_comp_ecef)         # (n_obs, 3)

    south, east, zenith = rho_sez[:, 0], rho_sez[:, 1], rho_sez[:, 2]
    rho_h2    = south**2 + east**2
    rho2      = rng_comp_safe**2
    rho_h     = np.sqrt(np.maximum(rho_h2, 1e-12))
    az_valid  = rho_h2 >= 1e-12
    rho_h2_safe = np.where(az_valid, rho_h2, 1.0)

    d_range_sez = np.where(valid_comp[:, None], rho_sez / rng_comp_safe[:, None], 0.0)

    d_az_sez = np.zeros((n_obs, 3))
    d_az_sez[:, 0] = np.where(az_valid & valid_comp,  east  / rho_h2_safe, 0.0)
    d_az_sez[:, 1] = np.where(az_valid & valid_comp, -south / rho_h2_safe, 0.0)

    d_el_sez = np.column_stack([
        np.where(valid_comp, -(zenith * south) / (rho2 * rho_h), 0.0),
        np.where(valid_comp, -(zenith * east)  / (rho2 * rho_h), 0.0),
        np.where(valid_comp,  rho_h / rho2,                      0.0),
    ])

    dsez_drmci    = np.einsum('nij,njk->nik', c_sez, R_j2k)         # (n_obs, 3, 3)
    d_range_drmci = np.einsum('ni,nij->nj', d_range_sez, dsez_drmci) # (n_obs, 3)
    d_az_drmci    = np.einsum('ni,nij->nj', d_az_sez,    dsez_drmci) # (n_obs, 3)
    d_el_drmci    = np.einsum('ni,nij->nj', d_el_sez,    dsez_drmci) # (n_obs, 3)

    v_orth       = v_rel - rr[:, None] * u_hat                       # (n_obs, 3)
    drr_drmci    = (np.einsum('ni,nij->nj', v_orth / rng_safe[:, None], R_j2k)
                    + np.einsum('ni,nij->nj', u_hat, dR_j2k))        # (n_obs, 3)
    drr_drmci    = np.where(valid[:, None], drr_drmci, 0.0)
    drr_dvmci    = np.einsum('ni,nij->nj', u_hat, R_j2k)             # (n_obs, 3)
    drr_dvmci    = np.where(valid[:, None], drr_dvmci, 0.0)

    h_tilde = np.zeros((4 * n_obs, 6), dtype=float)
    rows = np.arange(n_obs)
    h_tilde[rows * 4,     :3] = d_range_drmci
    h_tilde[rows * 4 + 1, :3] = drr_drmci
    h_tilde[rows * 4 + 1, 3:] = drr_dvmci
    h_tilde[rows * 4 + 2, :3] = d_az_drmci
    h_tilde[rows * 4 + 3, :3] = d_el_drmci

    return residuals, h_meas, h_tilde


def measurement_sigma_vector(
    obs_data: ArrayLike,
    pass_geo: PassGeometry,
    measurement_type: str | None = None,
) -> np.ndarray:
    """Return per-component measurement standard deviations in residual order."""
    measurement_type = (measurement_type or pass_geo.measurement_type).lower()
    obs_data = np.asarray(obs_data, dtype=float)
    if measurement_type == "position":
        sigma = np.zeros(obs_data.shape[0] * 3, dtype=float)
        for obs_idx in range(obs_data.shape[0]):
            station = pass_geo.stations[int(obs_data[obs_idx, 4]) - 1]
            row0 = obs_idx * 3
            sigma[row0 : row0 + 3] = [
                station.sigma_range_m,
                station.sigma_angle_rad,
                station.sigma_angle_rad,
            ]
    elif measurement_type == "range_rate":
        sigma = np.zeros(obs_data.shape[0] * 4, dtype=float)
        for obs_idx in range(obs_data.shape[0]):
            station = pass_geo.stations[int(obs_data[obs_idx, 5]) - 1]
            if station.sigma_range_rate_mps is None:
                raise ValueError("range-rate measurements require station.sigma_range_rate_mps.")
            row0 = obs_idx * 4
            sigma[row0 : row0 + 4] = [
                station.sigma_range_m,
                station.sigma_range_rate_mps,
                station.sigma_angle_rad,
                station.sigma_angle_rad,
            ]
    elif measurement_type == "two_way_range":
        # M3 policy: the scalar two-way range reuses station.sigma_range_m;
        # metadata records two_way_range_noise_source accordingly.
        sigma = np.zeros(obs_data.shape[0], dtype=float)
        for obs_idx in range(obs_data.shape[0]):
            station = pass_geo.stations[int(obs_data[obs_idx, 2]) - 1]
            sigma[obs_idx] = station.sigma_range_m
    else:
        raise ValueError(
            "measurement_type must be 'position', 'range_rate', or 'two_way_range'."
        )

    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("Measurement sigmas must be finite and positive.")
    return sigma


def measurement_covariance_matrix(
    obs_data: ArrayLike,
    pass_geo: PassGeometry,
    measurement_type: str | None = None,
) -> np.ndarray:
    """Return diagonal measurement covariance R in residual-vector order."""
    sigma = measurement_sigma_vector(obs_data, pass_geo, measurement_type)
    return np.diag(sigma**2)


def _ensure_n_by_3(value: ArrayLike, n_rows: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape == (3, n_rows):
        array = array.T
    if array.shape != (n_rows, 3):
        raise ValueError(f"{name} must have shape (N, 3) or (3, N).")
    return array


def _range_az_el_partials_sez(rho_sez: ArrayLike) -> np.ndarray:
    rho_sez = np.asarray(rho_sez, dtype=float).reshape(3)
    south, east, zenith = rho_sez
    range_m = float(np.linalg.norm(rho_sez))
    if range_m < 1e-9:
        return np.zeros((3, 3), dtype=float)

    rho_h2 = south**2 + east**2
    rho2 = range_m**2
    d_range_dsez = (rho_sez / range_m).reshape(1, 3)

    if rho_h2 < 1e-12:
        d_az_dsez = np.zeros((1, 3), dtype=float)
    else:
        d_az_dsez = np.array([[east / rho_h2, -south / rho_h2, 0.0]], dtype=float)

    rho_h = float(np.sqrt(max(rho_h2, 1e-12)))
    d_el_dsez = np.array(
        [[-(zenith * south) / (rho2 * rho_h), -(zenith * east) / (rho2 * rho_h), rho_h / rho2]],
        dtype=float,
    )

    return np.vstack([d_range_dsez, d_az_dsez, d_el_dsez])
