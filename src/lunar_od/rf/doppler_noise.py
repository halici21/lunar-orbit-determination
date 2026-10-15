"""Thermal contribution to two-way counted-Doppler measurement noise.

The formulation follows DSN 810-005 Module 202 Rev E:

    Eq (12)  sigma_f = 2 f_C sigma_V / c            two-way, f_C the downlink
    Eq (13)  sigma_V^2 = sigma_VN^2 + sigma_VF^2 + sigma_VS^2
    Eq (25)  sigma_VNU^2 ~ (1/2) (c / (2 pi f_C T))^2 G^2 / rho_TR * B_L/B_TR
    Eq (26)  sigma_VND^2 = (1/2) (c / (2 pi f_C T))^2 / rho_L

``sigma_V`` is the error in the rate of change of the ONE-WAY range, which is
exactly what the project's counted-Doppler observable returns in
``mps_equivalent`` mode. The factor of one half in Eq (25) and (26) is that
one-way convention, not an extra safety margin.

What this module does NOT do is combine anything beyond Layer 1. Media,
station-location and Earth-orientation errors are correlated or bias-like over an
observation cadence and cannot be root-sum-squared into a white sigma; they are
handled outside a diagonal measurement covariance.
"""

from __future__ import annotations

import math

from .link_budget import SPEED_OF_LIGHT_M_S, build_two_way_link_budget
from .types import TwoWayDopplerNoiseResult, TwoWayLinkConfig

__all__ = [
    "HYDROGEN_MASER_SIGMA_60S_HIGH_MPS",
    "HYDROGEN_MASER_SIGMA_60S_LOW_MPS",
    "compute_counted_doppler_thermal_noise",
    "compute_short_term_doppler_noise_budget",
    "cn0_for_thermal_sigma_dbhz",
    "frequency_standard_sigma_mps",
    "thermal_sigma_one_leg_mps",
]

#: Frequency-standard contribution at a 60 s count, carried forward from the
#: Phase-9H qualification. Derived from the DSN 810-005 Module 304 Rev B
#: Figure 3 hydrogen-maser band read at tau = 60 s, mapped through Eq (12).
#: The interval spans two limits: the low end assumes the uplink source and the
#: receiving local oscillator are fully correlated, the high end that they are
#: independent. The independence argument in 202E rests on a long round trip;
#: a lunar round trip is only about 2.6 s, so both limits are retained.
HYDROGEN_MASER_SIGMA_60S_LOW_MPS = 4.497e-7
HYDROGEN_MASER_SIGMA_60S_HIGH_MPS = 2.120e-6


def frequency_standard_sigma_mps(
    allan_deviation: float, *, independent_ends: bool = True
) -> float:
    """Map a fractional frequency instability onto ``sigma_V``.

    From Eq (12), ``sigma_V = c sigma_f / (2 f_C)``; a fractional instability
    ``sigma_y`` gives ``sigma_f = f_C sigma_y`` per contributing source, so the
    carrier frequency cancels and only the correlation assumption remains.
    """
    if allan_deviation < 0.0:
        raise ValueError("allan_deviation must be non-negative.")
    factor = math.sqrt(2.0) if independent_ends else 1.0
    return SPEED_OF_LIGHT_M_S * factor * allan_deviation / 2.0


def thermal_sigma_one_leg_mps(
    count_interval_s: float,
    carrier_loop_snr: float,
    downlink_frequency_hz: float,
    *,
    gain_factor: float = 1.0,
    bandwidth_ratio: float = 1.0,
) -> float:
    """DSN 810-005 202E Eq (25)/(26) for a single leg.

    ``gain_factor`` carries the turnaround ratio on the uplink leg and is unity
    on the downlink. ``bandwidth_ratio`` is ``B_L / B_TR``, capped at one: in the
    regime where the ground loop is at least as wide as the transponder loop, all
    of the uplink noise the transponder tracks is also tracked on the ground.
    """
    if count_interval_s <= 0.0:
        raise ValueError("count_interval_s must be positive.")
    if carrier_loop_snr <= 0.0:
        raise ValueError("carrier_loop_snr must be positive.")
    if downlink_frequency_hz <= 0.0:
        raise ValueError("downlink_frequency_hz must be positive.")
    phase_to_rate = SPEED_OF_LIGHT_M_S / (
        2.0 * math.pi * downlink_frequency_hz * count_interval_s
    )
    variance = (
        0.5
        * phase_to_rate**2
        * gain_factor**2
        / carrier_loop_snr
        * min(bandwidth_ratio, 1.0)
    )
    return math.sqrt(variance)


def cn0_for_thermal_sigma_dbhz(
    sigma_mps: float,
    count_interval_s: float,
    loop_bandwidth_hz: float,
    downlink_frequency_hz: float,
) -> float:
    """Inverse: the ``C/N0`` whose downlink thermal noise equals ``sigma_mps``.

    Useful without any spacecraft hardware, because it converts a Doppler-noise
    target into a link requirement.
    """
    if sigma_mps <= 0.0:
        raise ValueError("sigma_mps must be positive.")
    phase_to_rate = SPEED_OF_LIGHT_M_S / (
        2.0 * math.pi * downlink_frequency_hz * count_interval_s
    )
    rho = 0.5 * phase_to_rate**2 / sigma_mps**2
    return 10.0 * math.log10(rho * loop_bandwidth_hz)


def compute_counted_doppler_thermal_noise(
    config: TwoWayLinkConfig,
    uplink_range_m: float,
    downlink_range_m: float,
    elevation_deg: float,
) -> TwoWayDopplerNoiseResult:
    """Both thermal legs plus the frequency-standard component.

    Returns a structured PARAMETRIC result rather than a bare NaN when the
    spacecraft side is unspecified, and names exactly what is missing.
    """
    uplink, downlink = build_two_way_link_budget(
        config, uplink_range_m, downlink_range_m, elevation_deg
    )
    f_down = config.downlink_frequency_hz
    tc = config.count_interval_s

    sigma_down = None
    if downlink.carrier_loop_snr is not None:
        sigma_down = thermal_sigma_one_leg_mps(
            tc, downlink.carrier_loop_snr, f_down
        )

    sigma_up = None
    bandwidth_ratio = 1.0
    if config.transponder_carrier_loop is not None:
        bandwidth_ratio = (
            config.ground_carrier_loop.loop_bandwidth_hz
            / config.transponder_carrier_loop.loop_bandwidth_hz
        )
    if uplink.carrier_loop_snr is not None:
        sigma_up = thermal_sigma_one_leg_mps(
            tc,
            uplink.carrier_loop_snr,
            f_down,
            gain_factor=config.turnaround_ratio,
            bandwidth_ratio=bandwidth_ratio,
        )

    sigma_total = None
    if sigma_up is not None and sigma_down is not None:
        sigma_total = math.hypot(sigma_up, sigma_down)
    elif sigma_down is not None:
        sigma_total = None  # a one-sided total would understate the noise

    missing = tuple(dict.fromkeys(uplink.missing_inputs + downlink.missing_inputs))

    white_low = white_high = None
    dominant = "UNRESOLVED_WITHOUT_LINK_INPUTS"
    if sigma_total is not None:
        white_low = math.hypot(sigma_total, HYDROGEN_MASER_SIGMA_60S_LOW_MPS)
        white_high = math.hypot(sigma_total, HYDROGEN_MASER_SIGMA_60S_HIGH_MPS)
        if sigma_total > HYDROGEN_MASER_SIGMA_60S_HIGH_MPS:
            dominant = "THERMAL"
        elif sigma_total < HYDROGEN_MASER_SIGMA_60S_LOW_MPS:
            dominant = "FREQUENCY_STANDARD"
        else:
            dominant = "COMPARABLE"

    notes = (
        "Layer 1 only: media, station-location and Earth-orientation errors are "
        "correlated or bias-like over the observation cadence and are excluded "
        "from this white budget by construction.",
    )
    if sigma_total is None and sigma_down is not None:
        notes = notes + (
            "the downlink thermal term is numeric but the uplink is not; a "
            "downlink-only figure would understate two-way Doppler noise, so no "
            "total is reported.",
        )

    status = "NUMERIC" if sigma_total is not None else "PARAMETRIC"
    return TwoWayDopplerNoiseResult(
        count_interval_s=tc,
        downlink_frequency_hz=f_down,
        uplink=uplink,
        downlink=downlink,
        sigma_thermal_uplink_mps=sigma_up,
        sigma_thermal_downlink_mps=sigma_down,
        sigma_thermal_total_mps=sigma_total,
        sigma_frequency_standard_low_mps=HYDROGEN_MASER_SIGMA_60S_LOW_MPS,
        sigma_frequency_standard_high_mps=HYDROGEN_MASER_SIGMA_60S_HIGH_MPS,
        sigma_short_term_white_low_mps=white_low,
        sigma_short_term_white_high_mps=white_high,
        dominant_white_component=dominant,
        status=status,
        missing_inputs=missing,
        notes=notes,
    )


def compute_short_term_doppler_noise_budget(
    config: TwoWayLinkConfig,
    uplink_range_m: float,
    downlink_range_m: float,
    elevation_deg: float,
) -> TwoWayDopplerNoiseResult:
    """Alias kept for callers that want the intent spelled out.

    The name matters: this is a SHORT-TERM WHITE budget, not a total operational
    Doppler uncertainty, and the two must not be conflated.
    """
    return compute_counted_doppler_thermal_noise(
        config, uplink_range_m, downlink_range_m, elevation_deg
    )
