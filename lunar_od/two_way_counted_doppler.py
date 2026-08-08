"""R4 (CD-4): four-event counted Doppler with a constant transponder delay.

This module productionises the already accepted four-event model-F capability
for explicit nonzero constant transponder delay.  This is a capability
expansion, not a claim that the preceding R2 campaign established an
operational need for nonzero delay.

Why this is a separate module
-----------------------------
R3 froze the rule that production :mod:`lunar_od.radiometrics` must not depend
on :mod:`lunar_od.two_way_range` (M3) or on the R2 reference module.  The R4
architecture keeps that boundary and puts the composition here instead, so
``radiometrics`` stays free of the M3 import while R4 still reuses the accepted
event solver rather than mirroring a second one.

Composition boundary (frozen, ``m3_composition_boundary.csv``)
--------------------------------------------------------------
COMPOSED from M3, read-only:

* :func:`lunar_od.two_way_range.solve_two_way_range_events`
* :func:`lunar_od.two_way_range.two_way_range_event_sensitivity`
* :func:`lunar_od.two_way_range.two_way_range_from_solution`

DELIBERATELY NOT COMPOSED:

* ``make_exact_sxform_station_state_provider`` -- M3's provider *factory*
  interpolates the Earth-centre ephemeris with **cubic Hermite**, while counted
  Doppler (R3, R2 model S and R4) requires **linear** Earth interpolation so the
  S-L difference isolates the site transform alone.  R4 injects the accepted R3
  provider instead, through :class:`FourEventStationStateAdapter`.
* ``two_way_range_measurement_metadata`` -- hard-codes
  ``cubic_hermite_grid_interpolation``; emitting it for an R4 run would be a
  false provenance string.
* ``generate_two_way_range_measurements``, ``compute_two_way_range_residuals``,
  ``two_way_range_nominal_and_initial_jacobian`` -- M3 two-way *range*
  measurement semantics that counted Doppler does not want.

Physics
-------
Four events ``t1`` (ground transmit), ``t2u`` (spacecraft uplink receive),
``t2d`` (spacecraft downlink transmit), ``t3`` (ground receive), with
``t2d - t2u = delta_0`` and ``t1 < t2u <= t2d < t3``.  At ``delta_0 = 0`` the
chain collapses to the accepted R3 single-bounce model, which is the R4-P08
reduction gate.

The constant-delay sensitivity is **not zero**.  A constant delay shifts the
uplink chain backwards, but the shift magnitude depends on the local geometry,
so the two count endpoints do not shift identically; the counted observable's
``c / (2 Tc)`` scaling then restores that ``O(v/c)`` difference to an ``O(1)``
term.  See :func:`four_event_counted_doppler_delay_sensitivity`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .radiometrics import (
    COUNTED_DOPPLER_EARTH_EPHEMERIS_METHOD,
    COUNTED_DOPPLER_SPACECRAFT_INTERPOLATION_METHOD,
    CountedDopplerStationStateProvider,
    RangeRatePhysicsConfig,
    _clock_corrected_receive_time,
    _interp_state,
    _normalize_exact_station_epoch,
    _require_exact_provider_contract,
    range_rate_physics_config,
    resolve_counted_doppler_station_state_provider,
)
from .two_way_range import (
    TwoWayRangeConfig,
    TwoWayRangeEventSolution,
    solve_two_way_range_events,
    two_way_range_event_sensitivity,
    two_way_range_from_solution,
)

__all__ = [
    "FourEventCountedDopplerError",
    "FourEventStationStateAdapter",
    "four_event_counted_doppler_delay_sensitivity",
    "four_event_counted_doppler_endpoints",
    "four_event_counted_doppler_initial_state_jacobian",
    "four_event_counted_doppler_measurement_metadata",
    "four_event_counted_doppler_observable",
    "four_event_range_config",
]

# The R4 model always uses the delay-calibrated half-round-trip convention: the
# constant delay is removed from the reported range, and it cancels exactly in
# the endpoint difference that forms the counted observable.
FOUR_EVENT_RANGE_CONVENTION = "delay_calibrated_half_round_trip"


class FourEventCountedDopplerError(RuntimeError):
    """Raised when the four-event counted-Doppler path cannot be evaluated.

    R4-F11: there is no fallback.  The four-event model never degrades to the
    R3 single-bounce model, the legacy interpolated transform grid, a relaxed
    tolerance or a cached neighbouring solution; the originating error
    propagates.
    """


class FourEventStationStateAdapter:
    """Duck-typed M3 station provider backed by the accepted R3 provider.

    M3's solver needs ``state(t)``, ``station_state_method`` and
    ``earth_ephemeris_method``.  The accepted R3
    :class:`~lunar_od.radiometrics.CountedDopplerStationStateProvider` already
    supplies all three and carries ``linear_grid_interpolation``.  This adapter
    adds the two R3 policies that live in ``radiometrics._counted_station_state``
    and that M3 knows nothing about:

    * the F07 exact-provider contract check, and
    * the FA-03B closed-support epoch normalisation used for the Earth lookups.

    Applying them here is what makes the R4 path reduce to R3 exactly at zero
    delay: both consume the identical station states at the identical epochs.
    """

    def __init__(
        self,
        provider: CountedDopplerStationStateProvider,
        t_grid_s: ArrayLike,
        *,
        endpoint_label: str = "four-event endpoint",
        consumer: str = "four_event_counted_doppler",
    ) -> None:
        if provider is None:
            raise FourEventCountedDopplerError(
                "The four-event counted-Doppler model requires the exact "
                "event-epoch station provider; none was resolved. No legacy "
                "interpolated transform-grid fallback is permitted."
            )
        _require_exact_provider_contract(provider)
        self._provider = provider
        self._t_grid_s = t_grid_s
        self._endpoint_label = endpoint_label
        self._consumer = consumer

    @property
    def station_state_method(self) -> str:
        return self._provider.station_state_method

    @property
    def earth_ephemeris_method(self) -> str:
        return self._provider.earth_ephemeris_method

    @property
    def exact_sxform_call_count(self) -> int:
        return int(self._provider.exact_sxform_call_count)

    @property
    def provider(self) -> CountedDopplerStationStateProvider:
        return self._provider

    def state(self, t_s: float) -> np.ndarray:
        epoch = _normalize_exact_station_epoch(
            self._t_grid_s,
            float(t_s),
            endpoint_label=self._endpoint_label,
            event_label="four-event station",
            consumer=self._consumer,
        )
        return self._provider.state(epoch)


def four_event_range_config(config: RangeRatePhysicsConfig) -> TwoWayRangeConfig:
    """Map the counted-Doppler physics config onto the accepted M3 event solver.

    Only the event-solver scalars cross the boundary: the delay, the light
    speed, both halves of the dual convergence criterion and the iteration cap.
    Nothing about the station, Earth or spacecraft policy is taken from M3.
    """
    return TwoWayRangeConfig(
        transponder_delay_s=float(config.transponder_delay_s),
        convention=FOUR_EVENT_RANGE_CONVENTION,
        light_speed_mps=float(config.light_speed_mps),
        tolerance_s=float(config.light_time_tolerance_s),
        equation_tolerance_s=float(config.light_time_equation_tolerance_s),
        max_iter=int(config.light_time_max_iter),
    )


def _require_four_event_mode(config: RangeRatePhysicsConfig, caller: str) -> None:
    if config.mode != "two_way_counted_doppler":
        raise ValueError(f"{caller} requires two_way_counted_doppler mode.")
    if not config.four_event_enabled:
        raise ValueError(
            f"{caller} requires counted_doppler_model='four_event_delay'; got "
            f"{config.counted_doppler_model!r}."
        )


def _resolve_adapter(
    config: RangeRatePhysicsConfig,
    station,
    t_grid_s: ArrayLike,
    earth_pos_mci_m: ArrayLike,
    earth_vel_mci_mps: ArrayLike,
    *,
    et0_s: float | None,
    station_state_provider: CountedDopplerStationStateProvider | None,
    endpoint_label: str,
) -> FourEventStationStateAdapter:
    if isinstance(station_state_provider, FourEventStationStateAdapter):
        return station_state_provider
    provider = station_state_provider
    if provider is None:
        provider = resolve_counted_doppler_station_state_provider(
            config,
            station,
            t_grid_s,
            earth_pos_mci_m,
            earth_vel_mci_mps,
            et0_s=et0_s,
        )
    return FourEventStationStateAdapter(
        provider, t_grid_s, endpoint_label=endpoint_label
    )


def _count_endpoint_epochs(
    receive_mid_time_s: float, config: RangeRatePhysicsConfig
) -> tuple[float, float]:
    """Count endpoints on the GROUND RECEIVE epochs, unchanged from R3 (R4-P03)."""
    half_tc = 0.5 * config.count_interval_s
    t_start = float(receive_mid_time_s) - half_tc
    t_end = float(receive_mid_time_s) + half_tc
    return (
        _clock_corrected_receive_time(t_start, config),
        _clock_corrected_receive_time(t_end, config),
    )


def four_event_counted_doppler_endpoints(
    receive_mid_time_s: float,
    station,
    t_grid_s: ArrayLike,
    state_history_mci: ArrayLike,
    earth_pos_mci_m: ArrayLike,
    earth_vel_mci_mps: ArrayLike,
    x_j2000_to_itrf93: ArrayLike = None,
    config: RangeRatePhysicsConfig | str | None = None,
    *,
    et0_s: float | None = None,
    station_state_provider: CountedDopplerStationStateProvider | None = None,
) -> tuple[float, float, TwoWayRangeEventSolution, TwoWayRangeEventSolution, FourEventStationStateAdapter]:
    """Solve both count endpoints as independent four-event chains.

    Two endpoints x four events = eight physical events, matching the accepted
    R2 model-F decomposition.  One station provider is resolved here and shared
    by both endpoints, so the observable and the Jacobian provably consume the
    same station states (the R3-P08 discipline).
    """
    cfg = range_rate_physics_config(config)
    _require_four_event_mode(cfg, "four_event_counted_doppler_endpoints")
    adapter = _resolve_adapter(
        cfg,
        station,
        t_grid_s,
        earth_pos_mci_m,
        earth_vel_mci_mps,
        et0_s=et0_s,
        station_state_provider=station_state_provider,
        endpoint_label="four-event counted-Doppler endpoint",
    )
    range_config = four_event_range_config(cfg)
    receive_start, receive_end = _count_endpoint_epochs(receive_mid_time_s, cfg)
    start_solution = solve_two_way_range_events(
        receive_start, adapter, t_grid_s, state_history_mci, range_config
    )
    end_solution = solve_two_way_range_events(
        receive_end, adapter, t_grid_s, state_history_mci, range_config
    )
    return receive_start, receive_end, start_solution, end_solution, adapter


def _delay_calibrated_light_time_s(
    solution: TwoWayRangeEventSolution, config: RangeRatePhysicsConfig
) -> float:
    """Delay-calibrated round-trip light time in seconds.

    ``rho = (t3 - t1) - delta_0``.  At ``delta_0 == 0`` this is bitwise the raw
    round-trip light time, which is exactly what R3 returns, so the R4 path
    reduces to R3 bitwise at zero delay.
    """
    return float(solution.round_trip_light_time_s) - float(config.transponder_delay_s)


def _observable_from_light_time_difference(
    rho_start_s: float, rho_end_s: float, config: RangeRatePhysicsConfig
) -> float:
    """Frozen counted-Doppler observable, in R3's exact arithmetic association.

    R3 forms ``rho_rate = (rho_end - rho_start) / Tc`` and then scales.  The
    association is itself observable at the ULP level for short count
    intervals, so it is reproduced here verbatim rather than refactored into a
    mathematically equivalent form.
    """
    rho_rate = (rho_end_s - rho_start_s) / config.count_interval_s
    if config.output_unit == "hz":
        return float(config.turnaround_ratio * config.uplink_frequency_hz * rho_rate)
    return float(config.light_speed_mps * rho_rate / 2.0)


def _light_time_rate_scale(config: RangeRatePhysicsConfig) -> float:
    """Scale applied to a light-time *sensitivity* difference (R3 association)."""
    if config.output_unit == "hz":
        return config.turnaround_ratio * config.uplink_frequency_hz / config.count_interval_s
    return config.light_speed_mps / (2.0 * config.count_interval_s)


def four_event_counted_doppler_observable(
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
    """Evaluate R4 four-event counted Doppler at one receive-tagged midpoint.

    ``x_j2000_to_itrf93`` is accepted for signature compatibility with the R3
    entry point and is deliberately never read: the four-event model requires
    the exact event-epoch station transform and must not touch the legacy
    interpolated transform grid.
    """
    cfg = range_rate_physics_config(config)
    _require_four_event_mode(cfg, "four_event_counted_doppler_observable")
    _, _, start_solution, end_solution, _ = four_event_counted_doppler_endpoints(
        receive_mid_time_s,
        station,
        t_grid_s,
        state_history_mci,
        earth_pos_mci_m,
        earth_vel_mci_mps,
        x_j2000_to_itrf93,
        cfg,
        et0_s=et0_s,
        station_state_provider=station_state_provider,
    )
    return _observable_from_light_time_difference(
        _delay_calibrated_light_time_s(start_solution, cfg),
        _delay_calibrated_light_time_s(end_solution, cfg),
        cfg,
    )


def four_event_counted_doppler_initial_state_jacobian(
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
    """Analytic R4 partials with respect to the arc initial state.

    The event epochs come from the causal nested solve; the sensitivity comes
    from M3's structurally different uniform 3x3 implicit system evaluated on
    the converged solution.  Keeping the two structures different is the point
    -- it is what the finite-difference and independent-root gates test.

    ``dt1/dx0`` already embeds the STM through the interpolated
    position-sensitivity blocks at the two *distinct* spacecraft epochs ``t2u``
    and ``t2d``; it must not be propagated again downstream.
    """
    cfg = range_rate_physics_config(config)
    _require_four_event_mode(cfg, "four_event_counted_doppler_initial_state_jacobian")
    x_aug = np.asarray(augmented_state_history_mci, dtype=float)
    if x_aug.ndim != 2 or x_aug.shape[1] < 42:
        raise ValueError(
            "augmented_state_history_mci must contain state plus 6x6 STM columns."
        )
    _, _, start_solution, end_solution, adapter = four_event_counted_doppler_endpoints(
        receive_mid_time_s,
        station,
        t_grid_s,
        x_aug[:, :6],
        earth_pos_mci_m,
        earth_vel_mci_mps,
        x_j2000_to_itrf93,
        cfg,
        et0_s=et0_s,
        station_state_provider=station_state_provider,
    )
    range_config = four_event_range_config(cfg)
    start_sensitivity = two_way_range_event_sensitivity(
        start_solution, adapter, t_grid_s, x_aug, range_config
    )
    end_sensitivity = two_way_range_event_sensitivity(
        end_solution, adapter, t_grid_s, x_aug, range_config
    )
    # d(rho)/dx0 in seconds is -dt1/dx0: the constant delay and the fixed t3
    # both drop out.  Scaling then mirrors the R3 Jacobian association.
    d_tau_start = -np.asarray(start_sensitivity.dt1_dx0, dtype=float)
    d_tau_end = -np.asarray(end_sensitivity.dt1_dx0, dtype=float)
    return _light_time_rate_scale(cfg) * (d_tau_end - d_tau_start)


def _endpoint_delay_light_time_sensitivity(
    solution: TwoWayRangeEventSolution,
    adapter: FourEventStationStateAdapter,
    t_grid_s: ArrayLike,
    state_history_mci: ArrayLike,
    config: RangeRatePhysicsConfig,
) -> float:
    """``d(rho)/d(delta_0)`` in seconds per second at one fixed receive epoch.

    From the implicit system at fixed ``t3``::

        dt2d/d(delta_0) = 0                      (G_d contains no delay)
        dt2u/d(delta_0) = -1                     (transponder relation)
        dt1 /d(delta_0) = -(c - u.vs(t2u)) / (c - u.vg(t1))

    so with ``rho = (t3 - t1) - delta_0``::

        d(rho)/d(delta_0) = -dt1/d(delta_0) - 1
                          = -u.(vs(t2u) - vg(t1)) / (c - u.vg(t1))

    The final closed form is used rather than the two-term difference: the two
    terms are each close to 1 and differ at ``O(v/c)``, so subtracting them
    directly would cancel away most of the significant digits.
    """
    t_grid = np.asarray(t_grid_s, dtype=float).reshape(-1)
    states = np.asarray(state_history_mci, dtype=float)
    c = float(config.light_speed_mps)

    station_tx = adapter.state(solution.t1_s)
    sc_t2u = _interp_state(t_grid, states, solution.t2u_s)
    rho_vec = sc_t2u[:3] - station_tx[:3]
    rho_u = float(np.linalg.norm(rho_vec))
    if not np.isfinite(rho_u) or rho_u <= 0.0:
        raise FourEventCountedDopplerError(
            "Four-event delay sensitivity requires a positive uplink range; got "
            f"{rho_u!r} m at t1={solution.t1_s!r} s, t2u={solution.t2u_s!r} s."
        )
    u_up = rho_vec / rho_u
    numerator = float(u_up @ (sc_t2u[3:6] - station_tx[3:6]))
    denominator = c - float(u_up @ station_tx[3:6])
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise FourEventCountedDopplerError(
            "Four-event delay sensitivity denominator (c - u.v_station) must be "
            f"finite and positive; got {denominator!r}."
        )
    return -numerator / denominator


def four_event_counted_doppler_delay_sensitivity(
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
    """Analytic ``dH/d(delta_0)`` for the constant transponder delay.

    This term is **not** zero.  The direct additive delay cancels between the
    two count endpoints, but the geometric contribution does not: the endpoint
    shifts differ at ``O(v/c)`` and the counted observable's ``c / (2 Tc)``
    scaling cancels that ``1/c``, leaving

        dH/d(delta_0) ~= -0.5 * (uplink range acceleration)

    which is order ``0.5 (m/s)/s`` for lunar geometry.  Returned in the
    configured output unit per second of delay.

    ``delta_0`` remains a configuration parameter: R4 does not estimate it, and
    this sensitivity is never appended to a solve-for state vector.
    """
    cfg = range_rate_physics_config(config)
    _require_four_event_mode(cfg, "four_event_counted_doppler_delay_sensitivity")
    _, _, start_solution, end_solution, adapter = four_event_counted_doppler_endpoints(
        receive_mid_time_s,
        station,
        t_grid_s,
        state_history_mci,
        earth_pos_mci_m,
        earth_vel_mci_mps,
        x_j2000_to_itrf93,
        cfg,
        et0_s=et0_s,
        station_state_provider=station_state_provider,
    )
    d_rho_start = _endpoint_delay_light_time_sensitivity(
        start_solution, adapter, t_grid_s, state_history_mci, cfg
    )
    d_rho_end = _endpoint_delay_light_time_sensitivity(
        end_solution, adapter, t_grid_s, state_history_mci, cfg
    )
    return float(_light_time_rate_scale(cfg) * (d_rho_end - d_rho_start))


def four_event_counted_doppler_measurement_metadata(
    config: RangeRatePhysicsConfig,
    adapter: FourEventStationStateAdapter | None = None,
) -> dict:
    """Truthful R4 provenance.

    Every label is derived from what the code actually did.  In particular the
    Earth-centre interpolation is reported as the **linear** policy the R4 path
    really uses -- M3's ``two_way_range_measurement_metadata`` would report
    ``cubic_hermite_grid_interpolation`` here, which would be false.
    """
    cfg = range_rate_physics_config(config)
    _require_four_event_mode(cfg, "four_event_counted_doppler_measurement_metadata")
    earth_method = (
        adapter.earth_ephemeris_method
        if adapter is not None
        else COUNTED_DOPPLER_EARTH_EPHEMERIS_METHOD
    )
    station_method = (
        adapter.station_state_method
        if adapter is not None
        else cfg.station_state_method
    )
    return {
        "counted_doppler_model": cfg.counted_doppler_model,
        "counted_doppler_model_version": cfg.counted_doppler_model_version,
        "event_model": cfg.event_model,
        "physical_event_count": 8,
        "station_state_method": station_method,
        "station_velocity_model": cfg.station_velocity_model,
        "earth_ephemeris_method": earth_method,
        "spacecraft_state_interpolation_method": (
            COUNTED_DOPPLER_SPACECRAFT_INTERPOLATION_METHOD
        ),
        "range_convention": FOUR_EVENT_RANGE_CONVENTION,
        "transponder_delay_s": float(cfg.transponder_delay_s),
        "transponder_delay_model": "constant_scalar",
        "transponder_delay_is_solve_for": False,
        "count_interval_s": float(cfg.count_interval_s),
        "count_interval_reference": "ground_receive_epochs",
    }


def four_event_endpoint_ranges_m(
    start_solution: TwoWayRangeEventSolution,
    end_solution: TwoWayRangeEventSolution,
    config: RangeRatePhysicsConfig,
) -> tuple[float, float]:
    """Endpoint ranges in metres through M3's accepted convention helper.

    Provided for model-F parity reporting (R4-P09).  The production observable
    is formed in light-time space so that it reduces to R3 bitwise at zero
    delay; this metre-space view is the same quantity in model F's units.
    """
    range_config = four_event_range_config(config)
    return (
        float(two_way_range_from_solution(start_solution, range_config)),
        float(two_way_range_from_solution(end_solution, range_config)),
    )
