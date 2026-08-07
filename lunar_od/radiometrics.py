"""Radiometric observable models for range-rate-like measurements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike

from .history_domain import normalize_supported_epoch

C_LIGHT_MPS = 299792458.0
DEFAULT_X_BAND_UPLINK_HZ = 7.2e9
DEFAULT_X_BAND_TURNAROUND_RATIO = 880.0 / 749.0
TAYLOR3_MAX_COUNT_INTERVAL_S = 60.0

RangeRatePhysicsMode = Literal["geometric_instantaneous", "two_way_counted_doppler"]

# R3: how the counted-Doppler station site state is transformed out of the
# Earth body-fixed frame. The exact method evaluates the J2000->ITRF93 state
# transform at the true event epoch; the legacy method interpolates the
# pre-sampled transform grid element-wise, which is not a rotation between
# grid nodes (the S-L term the accepted R2 decision requires production to
# remove).
StationStateMethod = Literal[
    "exact_event_epoch_sxform",
    "legacy_interpolated_transform_grid",
]
EXACT_EVENT_EPOCH_STATION_METHOD = "exact_event_epoch_sxform"
LEGACY_INTERPOLATED_STATION_METHOD = "legacy_interpolated_transform_grid"
STATION_STATE_METHODS = (
    EXACT_EVENT_EPOCH_STATION_METHOD,
    LEGACY_INTERPOLATED_STATION_METHOD,
)
DEFAULT_STATION_STATE_METHOD = EXACT_EVENT_EPOCH_STATION_METHOD

# Provenance labels for the two policies R3 deliberately does NOT change.
# Earth ephemeris stays linear so the exact production path reproduces R2
# model S exactly and S-L isolates the site transform alone (R3-P22).
COUNTED_DOPPLER_EARTH_EPHEMERIS_METHOD = "linear_grid_interpolation"
COUNTED_DOPPLER_SPACECRAFT_INTERPOLATION_METHOD = "cubic_hermite"
COUNTED_DOPPLER_MODEL_VERSION = "r3.counted-doppler.exact-station.v1"
STATION_STATE_SOURCE_FRAME = "J2000"
STATION_STATE_TARGET_FRAME = "ITRF93"
# Truthful station_velocity_model provenance labels (the pre-R3 literal
# "sxform" overclaimed the interpolated path).
EXACT_STATION_VELOCITY_MODEL = "exact_event_epoch_sxform"
LEGACY_STATION_VELOCITY_MODEL = "interpolated_sxform_grid"


@dataclass(frozen=True)
class RangeRatePhysicsConfig:
    """Select the physics used for the second RR observable component.

    The default mode reproduces the existing instantaneous geometric line-of-sight
    range-rate. The two-way mode returns a range-rate equivalent derived from a
    constant-uplink counted Doppler observable over ``count_interval_s``.
    """

    mode: RangeRatePhysicsMode = "geometric_instantaneous"
    count_interval_s: float = 60.0
    uplink_frequency_hz: float = DEFAULT_X_BAND_UPLINK_HZ
    turnaround_ratio: float = DEFAULT_X_BAND_TURNAROUND_RATIO
    output_unit: Literal["mps_equivalent", "hz"] = "mps_equivalent"
    light_speed_mps: float = C_LIGHT_MPS
    light_time_tolerance_s: float = 1e-10
    light_time_max_iter: int = 20
    local_state_model: Literal["ode", "taylor3"] = "ode"
    station_clock_offset_s: float = 0.0
    station_clock_drift: float = 0.0
    clock_reference_time_s: float = 0.0
    transponder_delay_s: float = 0.0
    # FA-03A (P0B-1): equation-residual half of the dual convergence
    # criterion (per leg, seconds). Kept last to preserve positional callers.
    light_time_equation_tolerance_s: float = 1e-11
    # R3: station site-state strategy. Kept last (after the FA-03A field) so
    # every existing positional caller keeps working unchanged. The default
    # is the exact event-epoch transform: this is the intended, disclosed
    # consequence of the accepted R2 decision
    # EXACT_STATION_TRANSFORM_UPGRADE_REQUIRED. Selecting the legacy value is
    # an explicit opt-in compatibility mode, never an automatic fallback.
    station_state_method: StationStateMethod = DEFAULT_STATION_STATE_METHOD

    def __post_init__(self) -> None:
        normalized = _normalize_range_rate_mode(self.mode)
        object.__setattr__(self, "mode", normalized)
        if self.station_state_method not in STATION_STATE_METHODS:
            raise ValueError(
                "station_state_method must be one of "
                f"{STATION_STATE_METHODS}; got {self.station_state_method!r}."
            )
        if self.count_interval_s <= 0.0:
            raise ValueError("count_interval_s must be positive.")
        if self.uplink_frequency_hz <= 0.0:
            raise ValueError("uplink_frequency_hz must be positive.")
        if self.turnaround_ratio <= 0.0:
            raise ValueError("turnaround_ratio must be positive.")
        if self.output_unit not in {"mps_equivalent", "hz"}:
            raise ValueError("output_unit must be 'mps_equivalent' or 'hz'.")
        if self.light_speed_mps <= 0.0:
            raise ValueError("light_speed_mps must be positive.")
        if self.light_time_tolerance_s <= 0.0:
            raise ValueError("light_time_tolerance_s must be positive.")
        if self.light_time_max_iter <= 0:
            raise ValueError("light_time_max_iter must be positive.")
        if (
            not np.isfinite(self.light_time_equation_tolerance_s)
            or self.light_time_equation_tolerance_s <= 0.0
        ):
            raise ValueError(
                "light_time_equation_tolerance_s must be finite and positive."
            )
        if self.local_state_model not in {"ode", "taylor3"}:
            raise ValueError("local_state_model must be 'ode' or 'taylor3'.")
        if self.local_state_model == "taylor3" and self.count_interval_s > TAYLOR3_MAX_COUNT_INTERVAL_S:
            raise ValueError(
                "local_state_model='taylor3' is limited to count_interval_s <= "
                f"{TAYLOR3_MAX_COUNT_INTERVAL_S:g}."
            )
        if not np.isfinite(self.station_clock_offset_s):
            raise ValueError("station_clock_offset_s must be finite.")
        if not np.isfinite(self.station_clock_drift):
            raise ValueError("station_clock_drift must be finite.")
        if not np.isfinite(self.clock_reference_time_s):
            raise ValueError("clock_reference_time_s must be finite.")
        if self.transponder_delay_s < 0.0 or not np.isfinite(self.transponder_delay_s):
            raise ValueError("transponder_delay_s must be finite and non-negative.")
        if self.mode == "two_way_counted_doppler" and self.transponder_delay_s != 0.0:
            # P0A safety gate: the legacy solver keeps a single spacecraft
            # bounce state, so a nonzero delay would use physically
            # inconsistent uplink geometry (r_sc at the downlink transmit
            # epoch reused for the uplink leg). The fixed scalar delay term
            # itself cancels in the endpoint RTLT difference, but nonzero
            # delay still affects counted Doppler through the t2u/t2d
            # separation, spacecraft motion during the delay, and the changed
            # uplink/downlink event geometry — which this model cannot
            # represent.
            raise ValueError(
                "Nonzero transponder delay is not supported by the legacy "
                "single-bounce counted-Doppler model. Use zero delay or a "
                "future four-event counted-Doppler model. M3 two-way range "
                "(TwoWayRangeConfig) nonzero-delay support is unaffected."
            )

    @property
    def exact_event_epoch_enabled(self) -> bool:
        """True when the site transform is evaluated at the true event epoch."""
        return self.station_state_method == EXACT_EVENT_EPOCH_STATION_METHOD

    @property
    def legacy_compatibility_mode(self) -> bool:
        """True when the run deliberately opted into the pre-R3 station path."""
        return self.station_state_method == LEGACY_INTERPOLATED_STATION_METHOD

    @property
    def station_velocity_model(self) -> str:
        """Truthful provenance label for the station velocity actually used."""
        return (
            EXACT_STATION_VELOCITY_MODEL
            if self.exact_event_epoch_enabled
            else LEGACY_STATION_VELOCITY_MODEL
        )


@dataclass(frozen=True)
class RoundTripLightTimeSolution:
    """Round-trip light-time solution for a two-way coherent observable."""

    receive_time_s: float
    transmit_time_s: float
    transponder_time_s: float
    round_trip_light_time_s: float
    uplink_light_time_s: float
    downlink_light_time_s: float
    iterations: int
    converged: bool
    # FA-03A: independently evaluated per-leg equation residuals (seconds) at
    # the returned events; `converged` is True only when the update tolerance
    # AND both residual tolerances hold.
    uplink_equation_residual_s: float = float("nan")
    downlink_equation_residual_s: float = float("nan")
    uplink_update_converged: bool = False
    downlink_update_converged: bool = False
    uplink_iterations: int = 0
    downlink_iterations: int = 0
    uplink_update_residual_s: float = float("nan")
    downlink_update_residual_s: float = float("nan")


class RoundTripLightTimeConvergenceError(RuntimeError):
    """Raised when a round-trip light-time result fails its strict policy."""


class StationStateEvaluationError(RuntimeError):
    """Raised when an exact event-epoch station state cannot be evaluated.

    R3 failure contract F14 (the defining rule): there is NO fallback path
    from a failed exact evaluation to the legacy interpolated transform grid,
    under any condition. Every raise site below therefore fails closed.
    """


@dataclass
class CountedDopplerStationStateProvider:
    """Exact event-epoch station-state source for counted Doppler.

    Shape-compatible with ``two_way_range.TwoWayStationStateProvider`` — it
    exposes ``state(t_s)`` — but deliberately NOT imported from it. Two
    reasons, both frozen by the R3 architecture:

    * production ``radiometrics`` must not depend on ``two_way_range`` (M3) or
      on the R2 reference module;
    * M3's provider interpolates the Earth ephemeris with cubic Hermite, while
      counted Doppler and R2 model S both require LINEAR Earth interpolation so
      that the S-L difference isolates the site transform alone (R3-P22).

    The cache is keyed on the exact IEEE-754 float64 event epoch: no tolerance
    and no rounding, so it is provably result-neutral (R3-P19a). It lives for
    one measurement evaluation and is released with the provider.
    """

    state_fn: Callable[[float], np.ndarray]
    station_state_method: str = EXACT_EVENT_EPOCH_STATION_METHOD
    earth_ephemeris_method: str = COUNTED_DOPPLER_EARTH_EPHEMERIS_METHOD
    source_frame: str = STATION_STATE_SOURCE_FRAME
    target_frame: str = STATION_STATE_TARGET_FRAME
    center: str = "moon"
    cache_enabled: bool = True
    station_name: str = "<unnamed>"
    _sxform_counter: list[int] = field(default_factory=lambda: [0])
    _cache: dict[float, np.ndarray] = field(default_factory=dict)

    def state(self, t_s: float) -> np.ndarray:
        key = float(t_s)
        if self.cache_enabled:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        state = np.asarray(self.state_fn(key), dtype=float).reshape(6)
        if not np.all(np.isfinite(state)):
            raise StationStateEvaluationError(
                "Exact event-epoch station state is non-finite at "
                f"t={key:.16g} s for station {self.station_name!r} "
                f"({self.source_frame}->{self.target_frame}); no legacy "
                "fallback was used."
            )
        state.flags.writeable = False
        if self.cache_enabled:
            self._cache[key] = state
        return state

    @property
    def exact_sxform_call_count(self) -> int:
        """Number of exact sxform evaluations actually performed (R3-P19)."""
        return int(self._sxform_counter[0])

    @property
    def cached_epoch_count(self) -> int:
        return len(self._cache)


def make_exact_counted_doppler_station_state_provider(
    station,
    et0_s: float,
    t_grid_s: ArrayLike,
    earth_pos_mci_m: ArrayLike,
    earth_vel_mci_mps: ArrayLike,
    *,
    sxform_fn: Callable[[str, str, float], ArrayLike] | None = None,
    cache_enabled: bool = True,
) -> CountedDopplerStationStateProvider:
    """Build the exact event-epoch counted-Doppler station-state provider.

    The site transform is evaluated with ``spice.sxform`` at the true event
    epoch ``et0_s + t_s``. The Earth ephemeris keeps the LEGACY linear grid
    interpolation on purpose (R3-P22). ``sxform_fn`` exists only for
    deterministic SPICE-free fixtures and must obey the same J2000-to-ITRF93
    state-transform convention.
    """
    t_grid = np.asarray(t_grid_s, dtype=float).reshape(-1)
    earth_pos = np.asarray(earth_pos_mci_m, dtype=float)
    earth_vel = np.asarray(earth_vel_mci_mps, dtype=float)
    station_name = str(getattr(station, "name", "<unnamed>"))
    if t_grid.size < 2 or np.any(np.diff(t_grid) <= 0.0):
        raise ValueError("t_grid_s must contain at least two strictly increasing epochs.")
    if earth_pos.shape != (t_grid.size, 3) or earth_vel.shape != (t_grid.size, 3):
        raise ValueError("earth ephemeris histories must have shape (N, 3) matching t_grid_s.")
    if et0_s is None or not np.isfinite(float(et0_s)):
        # F11: the exact method cannot convert scenario time to ET without a
        # finite et0_s, and legacy is a deliberate opt-in, never a fallback.
        raise ValueError(
            "et0_s must be finite to evaluate the exact event-epoch station "
            "transform; supply et0_s on the pass geometry. Legacy "
            f"station_state_method={LEGACY_INTERPOLATED_STATION_METHOD!r} is a "
            "deliberate opt-in compatibility mode, not an automatic fallback."
        )
    # F05: malformed fixed station site, rejected at construction, not per call.
    r_ecef = np.asarray(getattr(station, "r_ecef_m", None), dtype=float).reshape(-1)
    if r_ecef.size != 3 or not np.all(np.isfinite(r_ecef)):
        raise ValueError(
            f"Station {station_name!r} has a malformed r_ecef_m: expected three "
            f"finite values, got {r_ecef!r}."
        )

    if sxform_fn is None:
        import spiceypy as spice

        sxform_fn = spice.sxform

    station_ecef_state = np.concatenate([r_ecef.reshape(3), np.zeros(3)])
    counter = [0]
    et0 = float(et0_s)

    def state_fn(t_s: float) -> np.ndarray:
        t = float(t_s)
        earth_pos_t = _interp_vector(t_grid, earth_pos, t)
        earth_vel_t = _interp_vector(t_grid, earth_vel, t)
        # F06: a non-finite Earth history must fail closed, never silently
        # produce a station state.
        if not np.all(np.isfinite(earth_pos_t)):
            raise StationStateEvaluationError(
                f"Earth position history is non-finite at t={t:.16g} s "
                f"(station {station_name!r}); no legacy fallback was used."
            )
        if not np.all(np.isfinite(earth_vel_t)):
            raise StationStateEvaluationError(
                f"Earth velocity history is non-finite at t={t:.16g} s "
                f"(station {station_name!r}); no legacy fallback was used."
            )
        earth_state = np.concatenate([earth_pos_t, earth_vel_t])
        try:
            raw_xform = sxform_fn(
                STATION_STATE_SOURCE_FRAME, STATION_STATE_TARGET_FRAME, et0 + t
            )
        except StationStateEvaluationError:
            raise
        except Exception as exc:  # F01/F02/F03: kernels, frames, coverage
            raise StationStateEvaluationError(
                "Exact event-epoch station transform failed for station "
                f"{station_name!r} at t={t:.16g} s "
                f"(ET {et0 + t:.16g}, et0_s={et0:.16g}), frame pair "
                f"{STATION_STATE_SOURCE_FRAME}->{STATION_STATE_TARGET_FRAME}: "
                f"{type(exc).__name__}: {exc}. No legacy fallback was used."
            ) from exc
        xform = np.asarray(raw_xform, dtype=float)
        # F04: never solve with a non-finite or wrongly shaped transform.
        if xform.shape != (6, 6) or not np.all(np.isfinite(xform)):
            raise StationStateEvaluationError(
                "Exact event-epoch station transform is invalid for station "
                f"{station_name!r} at t={t:.16g} s (ET {et0 + t:.16g}): shape "
                f"{xform.shape}, finite={bool(np.all(np.isfinite(xform)))}, "
                f"frame pair {STATION_STATE_SOURCE_FRAME}->"
                f"{STATION_STATE_TARGET_FRAME}. No legacy fallback was used."
            )
        counter[0] += 1
        station_rel_j2000 = np.linalg.solve(xform, station_ecef_state)
        return earth_state + station_rel_j2000

    return CountedDopplerStationStateProvider(
        state_fn=state_fn,
        cache_enabled=bool(cache_enabled),
        station_name=station_name,
        _sxform_counter=counter,
    )


def resolve_counted_doppler_station_state_provider(
    config: RangeRatePhysicsConfig,
    station,
    t_grid_s: ArrayLike,
    earth_pos_mci_m: ArrayLike,
    earth_vel_mci_mps: ArrayLike,
    *,
    et0_s: float | None = None,
    sxform_fn: Callable[[str, str, float], ArrayLike] | None = None,
    cache_enabled: bool = True,
) -> CountedDopplerStationStateProvider | None:
    """Resolve the station strategy once per measurement evaluation.

    Returns ``None`` for the legacy interpolated-grid strategy, in which case
    the unchanged ``_station_state_mci`` helpers are used.
    """
    if not config.exact_event_epoch_enabled:
        return None
    return make_exact_counted_doppler_station_state_provider(
        station,
        et0_s,
        t_grid_s,
        earth_pos_mci_m,
        earth_vel_mci_mps,
        sxform_fn=sxform_fn,
        cache_enabled=cache_enabled,
    )


def _require_exact_provider_contract(provider: CountedDopplerStationStateProvider) -> None:
    """F07: the provider must declare the MCI J2000-aligned, Moon-centred contract."""
    declared = (
        provider.station_state_method,
        provider.source_frame,
        provider.target_frame,
        provider.center,
    )
    expected = (
        EXACT_EVENT_EPOCH_STATION_METHOD,
        STATION_STATE_SOURCE_FRAME,
        STATION_STATE_TARGET_FRAME,
        "moon",
    )
    if declared != expected:
        raise StationStateEvaluationError(
            "Counted-Doppler station provider declares "
            f"{declared!r} but the counted-Doppler contract expects "
            f"{expected!r} (MCI J2000-aligned, Moon-centred). No legacy "
            "fallback was used."
        )


def _normalize_exact_station_epoch(
    t_grid_s: ArrayLike,
    t_s: float,
    *,
    endpoint_label: str,
    event_label: str,
    consumer: str,
) -> float:
    """Apply the unchanged FA-03B closed-support policy to the Earth lookups.

    The exact path still interpolates the Earth ephemeris on ``t_grid_s``, so
    those lookups keep exactly the legacy guard (F08). It deliberately does
    NOT normalize against the transform grid, because the exact path never
    reads ``x_j2000_to_itrf93`` at all (R3-P24).
    """
    earth_pos_epoch = _normalize_counted_history_epoch(
        t_grid_s,
        t_s,
        history_name="earth_position_mci",
        endpoint_label=endpoint_label,
        event_label=event_label,
        consumer=consumer,
    )
    _normalize_counted_history_epoch(
        t_grid_s,
        t_s,
        history_name="earth_velocity_mci",
        endpoint_label=endpoint_label,
        event_label=event_label,
        consumer=consumer,
    )
    return earth_pos_epoch


def _counted_station_state(
    t_s: float,
    station,
    t_grid_s: ArrayLike,
    earth_pos_mci_m: ArrayLike,
    earth_vel_mci_mps: ArrayLike,
    x_j2000_to_itrf93: ArrayLike,
    *,
    provider: CountedDopplerStationStateProvider | None,
    endpoint_label: str,
    event_label: str,
    consumer: str,
) -> np.ndarray:
    """Return the MCI station state through the resolved R3 strategy."""
    if provider is None:
        return _station_state_mci(
            t_s,
            station,
            t_grid_s,
            earth_pos_mci_m,
            earth_vel_mci_mps,
            x_j2000_to_itrf93,
            endpoint_label=endpoint_label,
            event_label=event_label,
            consumer=consumer,
        )
    _require_exact_provider_contract(provider)
    epoch = _normalize_exact_station_epoch(
        t_grid_s,
        t_s,
        endpoint_label=endpoint_label,
        event_label=event_label,
        consumer=consumer,
    )
    return provider.state(epoch)


def _counted_station_state_and_velocity(
    t_s: float,
    station,
    t_grid_s: ArrayLike,
    earth_pos_mci_m: ArrayLike,
    earth_vel_mci_mps: ArrayLike,
    x_j2000_to_itrf93: ArrayLike,
    *,
    provider: CountedDopplerStationStateProvider | None,
    endpoint_label: str,
    event_label: str,
    consumer: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (station state, the velocity used as ``vg1`` in the Jacobian).

    Legacy keeps the interpolation-slope velocity exactly as before — an
    internal inconsistency of model L that R3 deliberately preserves rather
    than silently repairs. Exact mode uses the true station velocity
    ``state[3:6]``, which is what R2 model S uses.
    """
    if provider is None:
        state, slope = _station_state_mci_with_time_slope(
            t_s,
            station,
            t_grid_s,
            earth_pos_mci_m,
            earth_vel_mci_mps,
            x_j2000_to_itrf93,
            endpoint_label=endpoint_label,
            event_label=event_label,
            consumer=consumer,
        )
        return state, np.asarray(slope[:3], dtype=float)
    state = _counted_station_state(
        t_s,
        station,
        t_grid_s,
        earth_pos_mci_m,
        earth_vel_mci_mps,
        x_j2000_to_itrf93,
        provider=provider,
        endpoint_label=endpoint_label,
        event_label=event_label,
        consumer=consumer,
    )
    return state, np.asarray(state[3:6], dtype=float)


def range_rate_physics_config(config: RangeRatePhysicsConfig | str | None) -> RangeRatePhysicsConfig:
    """Normalize a user-provided range-rate physics selector."""
    if config is None:
        return RangeRatePhysicsConfig()
    if isinstance(config, RangeRatePhysicsConfig):
        return config
    if isinstance(config, str):
        return RangeRatePhysicsConfig(mode=_normalize_range_rate_mode(config))
    raise TypeError("range_rate_physics must be None, a string, or RangeRatePhysicsConfig.")


def _require_round_trip_light_time_convergence(
    solution: RoundTripLightTimeSolution,
    config: RangeRatePhysicsConfig,
    *,
    endpoint_label: str,
) -> None:
    if solution.converged:
        return
    c = float(config.light_speed_mps)
    raise RoundTripLightTimeConvergenceError(
        f"Two-way light-time {endpoint_label} did not converge at receive time "
        f"{solution.receive_time_s:.16g} s: update convergence "
        f"[uplink {solution.uplink_update_converged} "
        f"({solution.uplink_iterations} iterations, residual "
        f"{solution.uplink_update_residual_s:.3e} s), downlink "
        f"{solution.downlink_update_converged} "
        f"({solution.downlink_iterations} iterations, residual "
        f"{solution.downlink_update_residual_s:.3e} s)] versus tolerance "
        f"{config.light_time_tolerance_s:.3e} s; equation residuals "
        f"[uplink {solution.uplink_equation_residual_s:.3e} s / "
        f"{c * solution.uplink_equation_residual_s:.3e} m, downlink "
        f"{solution.downlink_equation_residual_s:.3e} s / "
        f"{c * solution.downlink_equation_residual_s:.3e} m] versus tolerance "
        f"{config.light_time_equation_tolerance_s:.3e} s; refusing to use "
        "the last iterate."
    )


def instantaneous_geometric_range_rate(r_rel_m: ArrayLike, v_rel_mps: ArrayLike) -> float:
    """Return line-of-sight range-rate from relative position and velocity."""
    r_rel_m = np.asarray(r_rel_m, dtype=float).reshape(3)
    v_rel_mps = np.asarray(v_rel_mps, dtype=float).reshape(3)
    range_m = float(np.linalg.norm(r_rel_m))
    if range_m < 1e-9:
        return 0.0
    return float(np.dot(r_rel_m, v_rel_mps) / range_m)


def two_way_counted_doppler_observable(
    receive_mid_time_s: float,
    station,
    t_grid_s: ArrayLike,
    state_history_mci: ArrayLike,
    earth_pos_mci_m: ArrayLike,
    earth_vel_mci_mps: ArrayLike,
    x_j2000_to_itrf93: ArrayLike,
    config: RangeRatePhysicsConfig | str | None = None,
    *,
    et0_s: float | None = None,
    station_state_provider: CountedDopplerStationStateProvider | None = None,
) -> float:
    """Compute simplified two-way counted Doppler or its m/s equivalent.

    This controlled DSN-like model assumes a constant uplink frequency and fixed
    coherent turnaround ratio. Optional station clock and transponder-delay
    errors can be enabled for mismatch campaigns. Media corrections are not yet
    modeled.

    R3: the station site state is produced by the strategy selected in
    ``config.station_state_method``. The provider is resolved ONCE here and
    shared by both count endpoints.
    """
    cfg = range_rate_physics_config(config)
    if cfg.mode != "two_way_counted_doppler":
        raise ValueError("two_way_counted_doppler_observable requires two_way_counted_doppler mode.")

    if station_state_provider is None:
        station_state_provider = resolve_counted_doppler_station_state_provider(
            cfg,
            station,
            t_grid_s,
            earth_pos_mci_m,
            earth_vel_mci_mps,
            et0_s=et0_s,
        )

    half_tc = 0.5 * cfg.count_interval_s
    t_start = float(receive_mid_time_s) - half_tc
    t_end = float(receive_mid_time_s) + half_tc
    receive_start = _clock_corrected_receive_time(t_start, cfg)
    receive_end = _clock_corrected_receive_time(t_end, cfg)
    start_solution = solve_two_way_light_time(
        receive_start,
        station,
        t_grid_s,
        state_history_mci,
        earth_pos_mci_m,
        earth_vel_mci_mps,
        x_j2000_to_itrf93,
        cfg,
        endpoint_label="count-start endpoint",
        station_state_provider=station_state_provider,
    )
    _require_round_trip_light_time_convergence(
        start_solution, cfg, endpoint_label="count-start endpoint"
    )
    end_solution = solve_two_way_light_time(
        receive_end,
        station,
        t_grid_s,
        state_history_mci,
        earth_pos_mci_m,
        earth_vel_mci_mps,
        x_j2000_to_itrf93,
        cfg,
        endpoint_label="count-end endpoint",
        station_state_provider=station_state_provider,
    )
    _require_round_trip_light_time_convergence(
        end_solution, cfg, endpoint_label="count-end endpoint"
    )
    rho_start = start_solution.round_trip_light_time_s
    rho_end = end_solution.round_trip_light_time_s
    rho_rate = (rho_end - rho_start) / cfg.count_interval_s
    doppler_hz = cfg.turnaround_ratio * cfg.uplink_frequency_hz * rho_rate
    if cfg.output_unit == "hz":
        return float(doppler_hz)
    return float(cfg.light_speed_mps * rho_rate / 2.0)


def two_way_counted_doppler_initial_state_jacobian(
    receive_mid_time_s: float,
    station,
    t_grid_s: ArrayLike,
    augmented_state_history_mci: ArrayLike,
    earth_pos_mci_m: ArrayLike,
    earth_vel_mci_mps: ArrayLike,
    x_j2000_to_itrf93: ArrayLike,
    config: RangeRatePhysicsConfig | str | None = None,
    *,
    et0_s: float | None = None,
    station_state_provider: CountedDopplerStationStateProvider | None = None,
) -> np.ndarray:
    """Return analytic counted-Doppler partials with respect to arc initial state.

    R3: one station-state provider is resolved here and shared by both count
    endpoints, so the Jacobian consumes exactly the station states the
    observable consumed (R3-P08).
    """
    cfg = range_rate_physics_config(config)
    if cfg.mode != "two_way_counted_doppler":
        raise ValueError("two_way_counted_doppler_initial_state_jacobian requires two_way_counted_doppler mode.")

    if station_state_provider is None:
        station_state_provider = resolve_counted_doppler_station_state_provider(
            cfg,
            station,
            t_grid_s,
            earth_pos_mci_m,
            earth_vel_mci_mps,
            et0_s=et0_s,
        )

    half_tc = 0.5 * cfg.count_interval_s
    receive_start = _clock_corrected_receive_time(float(receive_mid_time_s) - half_tc, cfg)
    receive_end = _clock_corrected_receive_time(float(receive_mid_time_s) + half_tc, cfg)
    d_tau_start = round_trip_light_time_initial_state_jacobian(
        receive_start,
        station,
        t_grid_s,
        augmented_state_history_mci,
        earth_pos_mci_m,
        earth_vel_mci_mps,
        x_j2000_to_itrf93,
        cfg,
        endpoint_label="count-start Jacobian endpoint",
        station_state_provider=station_state_provider,
    )
    d_tau_end = round_trip_light_time_initial_state_jacobian(
        receive_end,
        station,
        t_grid_s,
        augmented_state_history_mci,
        earth_pos_mci_m,
        earth_vel_mci_mps,
        x_j2000_to_itrf93,
        cfg,
        endpoint_label="count-end Jacobian endpoint",
        station_state_provider=station_state_provider,
    )
    if cfg.output_unit == "hz":
        scale = cfg.turnaround_ratio * cfg.uplink_frequency_hz / cfg.count_interval_s
    else:
        scale = cfg.light_speed_mps / (2.0 * cfg.count_interval_s)
    return scale * (d_tau_end - d_tau_start)


def round_trip_light_time_initial_state_jacobian(
    receive_time_s: float,
    station,
    t_grid_s: ArrayLike,
    augmented_state_history_mci: ArrayLike,
    earth_pos_mci_m: ArrayLike,
    earth_vel_mci_mps: ArrayLike,
    x_j2000_to_itrf93: ArrayLike,
    config: RangeRatePhysicsConfig | str | None = None,
    *,
    endpoint_label: str = "Jacobian endpoint",
    et0_s: float | None = None,
    station_state_provider: CountedDopplerStationStateProvider | None = None,
) -> np.ndarray:
    """Return d(round-trip light-time)/d(initial spacecraft state)."""
    cfg = range_rate_physics_config(config)
    x_aug = np.asarray(augmented_state_history_mci, dtype=float)
    if x_aug.ndim != 2 or x_aug.shape[1] < 42:
        raise ValueError("augmented_state_history_mci must contain state plus 6x6 STM columns.")

    # R3-P08: one resolved provider for the solve AND for the post-solve
    # station re-queries, so the Jacobian can never consume interpolated
    # station states while the observable consumed exact ones.
    if station_state_provider is None and et0_s is not None:
        station_state_provider = resolve_counted_doppler_station_state_provider(
            cfg,
            station,
            t_grid_s,
            earth_pos_mci_m,
            earth_vel_mci_mps,
            et0_s=et0_s,
        )

    solution = solve_two_way_light_time(
        receive_time_s,
        station,
        t_grid_s,
        x_aug[:, :6],
        earth_pos_mci_m,
        earth_vel_mci_mps,
        x_j2000_to_itrf93,
        cfg,
        endpoint_label=endpoint_label,
        station_state_provider=station_state_provider,
    )
    _require_round_trip_light_time_convergence(
        solution, cfg, endpoint_label=endpoint_label
    )
    t1 = solution.transmit_time_s
    t2 = solution.transponder_time_s
    t3 = solution.receive_time_s

    station_rx_state = _counted_station_state(
        t3,
        station,
        t_grid_s,
        earth_pos_mci_m,
        earth_vel_mci_mps,
        x_j2000_to_itrf93,
        provider=station_state_provider,
        endpoint_label=endpoint_label,
        event_label="downlink",
        consumer="round_trip_light_time_initial_state_jacobian",
    )
    # Legacy keeps the interpolation-slope velocity; exact mode substitutes the
    # true station velocity state[3:6], which is what R2 model S uses.
    station_tx_state, station_tx_velocity = _counted_station_state_and_velocity(
        t1,
        station,
        t_grid_s,
        earth_pos_mci_m,
        earth_vel_mci_mps,
        x_j2000_to_itrf93,
        provider=station_state_provider,
        endpoint_label=endpoint_label,
        event_label="uplink",
        consumer="round_trip_light_time_initial_state_jacobian",
    )
    sc_t2_state = _guarded_spacecraft_state(
        t_grid_s,
        x_aug[:, :6],
        t2,
        endpoint_label=endpoint_label,
        event_label="downlink",
        consumer="round_trip_light_time_initial_state_jacobian",
    )
    phi_history = np.array([row.reshape((6, 6), order="F") for row in x_aug[:, 6:]], dtype=float)
    phi_position_t2 = _guarded_spacecraft_stm_position(
        t_grid_s,
        phi_history,
        t2,
        endpoint_label=endpoint_label,
        event_label="downlink",
        consumer="round_trip_light_time_initial_state_jacobian",
    )

    r2 = sc_t2_state[:3]
    v2 = sc_t2_state[3:6]
    g3 = station_rx_state[:3]
    g1 = station_tx_state[:3]
    vg1 = station_tx_velocity
    a2 = phi_position_t2

    rho_down = r2 - g3
    rho_up = r2 - g1
    norm_down = float(np.linalg.norm(rho_down))
    norm_up = float(np.linalg.norm(rho_up))
    if norm_down < 1e-9 or norm_up < 1e-9:
        return np.zeros(6, dtype=float)

    u_down = rho_down / norm_down
    u_up = rho_up / norm_up
    c = cfg.light_speed_mps

    dt2_dx = -((u_down.reshape(1, 3) @ a2).reshape(6)) / (c + float(np.dot(u_down, v2)))
    dt1_denom = -1.0 + float(np.dot(u_up, vg1)) / c
    dt1_dx = (
        ((u_up.reshape(1, 3) @ a2).reshape(6) / c)
        - (1.0 - float(np.dot(u_up, v2)) / c) * dt2_dx
    ) / dt1_denom

    return -dt1_dx


def solve_two_way_light_time(
    receive_time_s: float,
    station,
    t_grid_s: ArrayLike,
    state_history_mci: ArrayLike,
    earth_pos_mci_m: ArrayLike,
    earth_vel_mci_mps: ArrayLike,
    x_j2000_to_itrf93: ArrayLike,
    config: RangeRatePhysicsConfig | str | None = None,
    *,
    endpoint_label: str = "round-trip endpoint",
    et0_s: float | None = None,
    station_state_provider: CountedDopplerStationStateProvider | None = None,
) -> RoundTripLightTimeSolution:
    """Solve station-spacecraft-station geometric round-trip light-time."""
    cfg = range_rate_physics_config(config)
    receive_time_s = float(receive_time_s)
    if station_state_provider is None and et0_s is not None:
        # Convenience for direct callers that supply et0_s; the production
        # counted-Doppler entry points resolve once and thread the provider in
        # (frozen architecture section 3). A direct caller that supplies
        # neither keeps the legacy strategy: this is NOT the F14 fallback
        # case, which concerns a FAILED exact evaluation.
        station_state_provider = resolve_counted_doppler_station_state_provider(
            cfg,
            station,
            t_grid_s,
            earth_pos_mci_m,
            earth_vel_mci_mps,
            et0_s=et0_s,
        )
    station_rx_state = _counted_station_state(
        receive_time_s,
        station,
        t_grid_s,
        earth_pos_mci_m,
        earth_vel_mci_mps,
        x_j2000_to_itrf93,
        provider=station_state_provider,
        endpoint_label=endpoint_label,
        event_label="downlink",
        consumer="solve_two_way_light_time",
    )

    sc_rx_state = _guarded_spacecraft_state(
        t_grid_s,
        state_history_mci,
        receive_time_s,
        endpoint_label=endpoint_label,
        event_label="downlink",
        consumer="solve_two_way_light_time",
    )
    downlink_lt = float(np.linalg.norm(sc_rx_state[:3] - station_rx_state[:3]) / cfg.light_speed_mps)
    downlink_converged = False
    iteration_count = 0
    downlink_update_residual_s = float("inf")
    t2 = receive_time_s - downlink_lt

    for iteration_count in range(1, cfg.light_time_max_iter + 1):
        sc_t2_state = _guarded_spacecraft_state(
            t_grid_s,
            state_history_mci,
            t2,
            endpoint_label=endpoint_label,
            event_label="downlink",
            consumer="solve_two_way_light_time",
        )
        new_downlink_lt = float(np.linalg.norm(sc_t2_state[:3] - station_rx_state[:3]) / cfg.light_speed_mps)
        new_t2 = receive_time_s - new_downlink_lt
        downlink_update_residual_s = abs(new_t2 - t2)
        if downlink_update_residual_s <= cfg.light_time_tolerance_s:
            t2 = new_t2
            downlink_lt = new_downlink_lt
            downlink_converged = True
            break
        t2 = new_t2
        downlink_lt = new_downlink_lt

    sc_t2_state = _guarded_spacecraft_state(
        t_grid_s,
        state_history_mci,
        t2,
        endpoint_label=endpoint_label,
        event_label="downlink",
        consumer="solve_two_way_light_time",
    )
    t1 = t2 - cfg.transponder_delay_s - downlink_lt
    uplink_converged = False
    uplink_update_residual_s = float("inf")
    for uplink_iter in range(1, cfg.light_time_max_iter + 1):
        station_tx_state = _counted_station_state(
            t1,
            station,
            t_grid_s,
            earth_pos_mci_m,
            earth_vel_mci_mps,
            x_j2000_to_itrf93,
            provider=station_state_provider,
            endpoint_label=endpoint_label,
            event_label="uplink",
            consumer="solve_two_way_light_time",
        )
        uplink_lt = float(np.linalg.norm(sc_t2_state[:3] - station_tx_state[:3]) / cfg.light_speed_mps)
        new_t1 = t2 - cfg.transponder_delay_s - uplink_lt
        uplink_update_residual_s = abs(new_t1 - t1)
        if uplink_update_residual_s <= cfg.light_time_tolerance_s:
            t1 = new_t1
            uplink_converged = True
            break
        t1 = new_t1
    else:
        uplink_lt = float(np.linalg.norm(sc_t2_state[:3] - station_tx_state[:3]) / cfg.light_speed_mps)

    # FA-03A per-leg equation residuals, evaluated fresh at the returned
    # events: the spacecraft state at the final t2 was re-interpolated above;
    # the uplink station is re-queried at the final t1 (the loop's last
    # station_tx_state can lag t1 by one update).
    downlink_equation_residual_s = abs(
        (receive_time_s - t2)
        - float(
            np.linalg.norm(sc_t2_state[:3] - station_rx_state[:3]) / cfg.light_speed_mps
        )
    )
    station_tx_final = _counted_station_state(
        t1,
        station,
        t_grid_s,
        earth_pos_mci_m,
        earth_vel_mci_mps,
        x_j2000_to_itrf93,
        provider=station_state_provider,
        endpoint_label=endpoint_label,
        event_label="uplink",
        consumer="solve_two_way_light_time",
    )
    uplink_equation_residual_s = abs(
        (t2 - cfg.transponder_delay_s - t1)
        - float(
            np.linalg.norm(sc_t2_state[:3] - station_tx_final[:3]) / cfg.light_speed_mps
        )
    )
    converged_all = bool(
        downlink_converged
        and uplink_converged
        and downlink_equation_residual_s <= cfg.light_time_equation_tolerance_s
        and uplink_equation_residual_s <= cfg.light_time_equation_tolerance_s
    )

    return RoundTripLightTimeSolution(
        receive_time_s=receive_time_s,
        transmit_time_s=float(t1),
        transponder_time_s=float(t2),
        round_trip_light_time_s=float(receive_time_s - t1),
        uplink_light_time_s=float(uplink_lt),
        downlink_light_time_s=float(downlink_lt),
        iterations=int(iteration_count + uplink_iter),
        converged=converged_all,
        uplink_equation_residual_s=float(uplink_equation_residual_s),
        downlink_equation_residual_s=float(downlink_equation_residual_s),
        uplink_update_converged=uplink_converged,
        downlink_update_converged=downlink_converged,
        uplink_iterations=int(uplink_iter),
        downlink_iterations=int(iteration_count),
        uplink_update_residual_s=float(uplink_update_residual_s),
        downlink_update_residual_s=float(downlink_update_residual_s),
    )


def _station_state_mci(
    t_s: float,
    station,
    t_grid_s: ArrayLike,
    earth_pos_mci_m: ArrayLike,
    earth_vel_mci_mps: ArrayLike,
    x_j2000_to_itrf93: ArrayLike,
    *,
    endpoint_label: str = "round-trip endpoint",
    event_label: str,
    consumer: str,
) -> np.ndarray:
    earth_pos_epoch = _normalize_counted_history_epoch(
        t_grid_s,
        t_s,
        history_name="earth_position_mci",
        endpoint_label=endpoint_label,
        event_label=event_label,
        consumer=consumer,
    )
    earth_vel_epoch = _normalize_counted_history_epoch(
        t_grid_s,
        t_s,
        history_name="earth_velocity_mci",
        endpoint_label=endpoint_label,
        event_label=event_label,
        consumer=consumer,
    )
    transform_epoch = _normalize_counted_history_epoch(
        t_grid_s,
        t_s,
        history_name="j2000_to_itrf93_state_transform",
        endpoint_label=endpoint_label,
        event_label=event_label,
        consumer=consumer,
    )
    earth_state = np.concatenate(
        [
            _interp_vector(t_grid_s, earth_pos_mci_m, earth_pos_epoch),
            _interp_vector(t_grid_s, earth_vel_mci_mps, earth_vel_epoch),
        ]
    )
    xform = _interp_matrix(t_grid_s, x_j2000_to_itrf93, transform_epoch)
    station_ecef_state = np.concatenate([np.asarray(station.r_ecef_m, dtype=float).reshape(3), np.zeros(3)])
    station_rel_j2000 = np.linalg.solve(xform, station_ecef_state)
    return earth_state + station_rel_j2000


def _clock_corrected_receive_time(t_s: float, config: RangeRatePhysicsConfig) -> float:
    return float(
        t_s
        + config.station_clock_offset_s
        + config.station_clock_drift * (t_s - config.clock_reference_time_s)
    )


def _station_state_mci_with_time_slope(
    t_s: float,
    station,
    t_grid_s: ArrayLike,
    earth_pos_mci_m: ArrayLike,
    earth_vel_mci_mps: ArrayLike,
    x_j2000_to_itrf93: ArrayLike,
    *,
    endpoint_label: str = "round-trip endpoint",
    event_label: str,
    consumer: str,
) -> tuple[np.ndarray, np.ndarray]:
    earth_pos_epoch = _normalize_counted_history_epoch(
        t_grid_s,
        t_s,
        history_name="earth_position_mci",
        endpoint_label=endpoint_label,
        event_label=event_label,
        consumer=consumer,
    )
    earth_vel_epoch = _normalize_counted_history_epoch(
        t_grid_s,
        t_s,
        history_name="earth_velocity_mci",
        endpoint_label=endpoint_label,
        event_label=event_label,
        consumer=consumer,
    )
    transform_epoch = _normalize_counted_history_epoch(
        t_grid_s,
        t_s,
        history_name="j2000_to_itrf93_state_transform",
        endpoint_label=endpoint_label,
        event_label=event_label,
        consumer=consumer,
    )
    earth_pos, earth_pos_slope = _interp_array_and_slope(
        t_grid_s, earth_pos_mci_m, earth_pos_epoch
    )
    earth_vel, earth_vel_slope = _interp_array_and_slope(
        t_grid_s, earth_vel_mci_mps, earth_vel_epoch
    )
    xform, xform_slope = _interp_array_and_slope(
        t_grid_s, x_j2000_to_itrf93, transform_epoch
    )
    station_ecef_state = np.concatenate([np.asarray(station.r_ecef_m, dtype=float).reshape(3), np.zeros(3)])
    station_rel_j2000 = np.linalg.solve(xform, station_ecef_state)
    station_rel_slope = -np.linalg.solve(xform, xform_slope @ station_rel_j2000)
    return (
        np.concatenate([earth_pos, earth_vel]) + station_rel_j2000,
        np.concatenate([earth_pos_slope, earth_vel_slope]) + station_rel_slope,
    )


def _normalize_counted_history_epoch(
    t_grid_s: ArrayLike,
    requested_epoch_s: float,
    *,
    history_name: str,
    endpoint_label: str,
    event_label: str,
    consumer: str,
) -> float:
    """Normalize one counted production lookup onto closed history support."""
    t_grid = np.asarray(t_grid_s, dtype=float).reshape(-1)
    if t_grid.size == 0:
        raise ValueError("t_grid_s must contain at least one history epoch.")
    return normalize_supported_epoch(
        requested_epoch_s,
        float(t_grid[0]),
        float(t_grid[-1]),
        history_name=history_name,
        model_context="two_way_counted_doppler",
        consumer=consumer,
        endpoint_label=endpoint_label,
        event_label=event_label,
    )


def _guarded_spacecraft_state(
    t_grid_s: ArrayLike,
    state_history_mci: ArrayLike,
    requested_epoch_s: float,
    *,
    endpoint_label: str,
    event_label: str,
    consumer: str,
) -> np.ndarray:
    normalized = _normalize_counted_history_epoch(
        t_grid_s,
        requested_epoch_s,
        history_name="spacecraft_state",
        endpoint_label=endpoint_label,
        event_label=event_label,
        consumer=consumer,
    )
    return _interp_state(t_grid_s, state_history_mci, normalized)


def _guarded_spacecraft_stm_position(
    t_grid_s: ArrayLike,
    phi_history: ArrayLike,
    requested_epoch_s: float,
    *,
    endpoint_label: str,
    event_label: str,
    consumer: str,
) -> np.ndarray:
    normalized = _normalize_counted_history_epoch(
        t_grid_s,
        requested_epoch_s,
        history_name="spacecraft_stm",
        endpoint_label=endpoint_label,
        event_label=event_label,
        consumer=consumer,
    )
    return _interp_state_transition_position(t_grid_s, phi_history, normalized)


def _interp_state(t_grid_s: ArrayLike, state_history: ArrayLike, t_s: float) -> np.ndarray:
    t_grid = np.asarray(t_grid_s, dtype=float).reshape(-1)
    states = np.asarray(state_history, dtype=float)
    if states.ndim != 2 or states.shape[0] != t_grid.size or states.shape[1] < 6:
        raise ValueError("state_history must have shape (N, 6+) matching t_grid_s.")
    t = float(t_s)
    if t <= t_grid[0] or t >= t_grid[-1]:
        return _interp_vector(t_grid, states, t)

    i1 = int(np.searchsorted(t_grid, t, side="right"))
    i0 = i1 - 1
    h = float(t_grid[i1] - t_grid[i0])
    if h <= 0.0:
        raise ValueError("t_grid_s must be strictly increasing.")
    s = (t - float(t_grid[i0])) / h
    r0 = states[i0, :3]
    v0 = states[i0, 3:6]
    r1 = states[i1, :3]
    v1 = states[i1, 3:6]

    position = (
        (2.0 * s**3 - 3.0 * s**2 + 1.0) * r0
        + (s**3 - 2.0 * s**2 + s) * h * v0
        + (-2.0 * s**3 + 3.0 * s**2) * r1
        + (s**3 - s**2) * h * v1
    )
    velocity = (
        (6.0 * s**2 - 6.0 * s) * r0 / h
        + (3.0 * s**2 - 4.0 * s + 1.0) * v0
        + (-6.0 * s**2 + 6.0 * s) * r1 / h
        + (3.0 * s**2 - 2.0 * s) * v1
    )
    result = states[i0].copy()
    result[:3] = position
    result[3:6] = velocity
    return result


def interp_state_history(t_grid_s: ArrayLike, state_history: ArrayLike, t_s: float) -> np.ndarray:
    """Cubic-Hermite (position+velocity) interpolation of a 6-state history with
    linear extrapolation outside the grid. Shared helper for light-time models."""
    return _interp_state(t_grid_s, state_history, t_s)


def _interp_state_transition_position(
    t_grid_s: ArrayLike,
    phi_history: ArrayLike,
    t_s: float,
) -> np.ndarray:
    t_grid = np.asarray(t_grid_s, dtype=float).reshape(-1)
    phi = np.asarray(phi_history, dtype=float)
    if phi.shape != (t_grid.size, 6, 6):
        raise ValueError("phi_history must have shape (N, 6, 6) matching t_grid_s.")
    t = float(t_s)
    if t <= t_grid[0] or t >= t_grid[-1]:
        return _interp_matrix(t_grid, phi, t)[:3, :]

    i1 = int(np.searchsorted(t_grid, t, side="right"))
    i0 = i1 - 1
    h = float(t_grid[i1] - t_grid[i0])
    if h <= 0.0:
        raise ValueError("t_grid_s must be strictly increasing.")
    s = (t - float(t_grid[i0])) / h
    return (
        (2.0 * s**3 - 3.0 * s**2 + 1.0) * phi[i0, :3, :]
        + (s**3 - 2.0 * s**2 + s) * h * phi[i0, 3:6, :]
        + (-2.0 * s**3 + 3.0 * s**2) * phi[i1, :3, :]
        + (s**3 - s**2) * h * phi[i1, 3:6, :]
    )


def _interp_array_and_slope(t_grid_s: ArrayLike, values: ArrayLike, t_s: float) -> tuple[np.ndarray, np.ndarray]:
    t_grid = np.asarray(t_grid_s, dtype=float).reshape(-1)
    array = np.asarray(values, dtype=float)
    if array.shape[0] != t_grid.size:
        raise ValueError("values must have the same number of rows as t_grid_s.")
    if t_grid.size < 2:
        raise ValueError("At least two time samples are required for interpolation.")

    t = float(t_s)
    if t <= t_grid[0]:
        i0, i1 = 0, 1
    elif t >= t_grid[-1]:
        i0, i1 = t_grid.size - 2, t_grid.size - 1
    else:
        i1 = int(np.searchsorted(t_grid, t, side="right"))
        i0 = i1 - 1

    dt = t_grid[i1] - t_grid[i0]
    if dt == 0.0:
        raise ValueError("t_grid_s must be strictly increasing.")
    frac = (t - t_grid[i0]) / dt
    slope = (array[i1] - array[i0]) / dt
    value = array[i0] + frac * (array[i1] - array[i0])
    return value, slope


def _interp_vector(t_grid_s: ArrayLike, values: ArrayLike, t_s: float) -> np.ndarray:
    t_grid_s = np.asarray(t_grid_s, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape[0] != t_grid_s.size:
        raise ValueError("values must have the same number of rows as t_grid_s.")
    return np.array([_interp_1d_linear_extrap(t_grid_s, values[:, col], float(t_s)) for col in range(values.shape[1])])


def _interp_matrix(t_grid_s: ArrayLike, values: ArrayLike, t_s: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    original_shape = values.shape[1:]
    flat = values.reshape(values.shape[0], -1)
    return _interp_vector(t_grid_s, flat, t_s).reshape(original_shape)


def _interp_1d_linear_extrap(t_grid_s: np.ndarray, y: np.ndarray, t_s: float) -> float:
    if t_grid_s.size < 2:
        raise ValueError("At least two time samples are required for interpolation.")
    if t_s <= t_grid_s[0]:
        i0, i1 = 0, 1
    elif t_s >= t_grid_s[-1]:
        i0, i1 = t_grid_s.size - 2, t_grid_s.size - 1
    else:
        return float(np.interp(t_s, t_grid_s, y))
    dt = t_grid_s[i1] - t_grid_s[i0]
    if dt == 0.0:
        raise ValueError("t_grid_s must be strictly increasing.")
    slope = (y[i1] - y[i0]) / dt
    return float(y[i0] + slope * (t_s - t_grid_s[i0]))


def _normalize_range_rate_mode(mode: str) -> RangeRatePhysicsMode:
    normalized = mode.strip().lower().replace("-", "_")
    aliases = {
        "geometric": "geometric_instantaneous",
        "instantaneous": "geometric_instantaneous",
        "geometric_rr": "geometric_instantaneous",
        "geometric_range_rate": "geometric_instantaneous",
        "geometric_instantaneous": "geometric_instantaneous",
        "two_way": "two_way_counted_doppler",
        "twoway": "two_way_counted_doppler",
        "two_way_doppler": "two_way_counted_doppler",
        "counted_doppler": "two_way_counted_doppler",
        "two_way_counted": "two_way_counted_doppler",
        "two_way_counted_doppler": "two_way_counted_doppler",
    }
    if normalized not in aliases:
        raise ValueError("range-rate physics mode must be geometric_instantaneous or two_way_counted_doppler.")
    return aliases[normalized]  # type: ignore[return-value]
