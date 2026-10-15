"""Reference ground-terminal configurations built from public DSN documentation.

Every entry is labelled ``REFERENCE_SCENARIO``. The distinction matters: this
project's stations are navigation tracking SITES, defined by latitude, longitude
and altitude. Being named "Canberra" does not say which antenna at that complex
is tracking, and the difference between a 34-m and a 70-m dish is several
decibels. These configurations therefore validate arithmetic and support
inverse design; they are not mission truth, and promoting one to
``MISSION_CONFIGURATION`` requires an actual mission decision.

Source for every number below:

    DSN 810-005 Module 104 Rev L, "34-m BWG Stations Telecommunications
    Interfaces", Jet Propulsion Laboratory
    https://deepspace.jpl.nasa.gov/dsndocs/810-005/104/104L.pdf
        Table A-2  X-band vacuum gain and antenna-microwave noise temperature
                   parameters, referenced to the feedhorn aperture
        Table A-5  zenith atmospheric attenuation
        Section 2  all BWG antennas carry a 20 kW X-band transmitter

The DIPLEXED configurations are used because coherent two-way tracking requires
the station to transmit and receive at once.
"""

from __future__ import annotations

from .environment import zenith_attenuation_db
from .types import (
    CarrierLoopConfig,
    GroundTerminalConfig,
    LinkEnvironmentConfig,
    SpacecraftRadioConfig,
    TwoWayLinkConfig,
)

__all__ = ["DSN_34M_BWG_REFERENCE_TERMINALS", "reference_two_way_link"]

_SOURCE = "DSN 810-005 104L Table A-2 and A-5; 20 kW X-band transmitter per Section 2"

DSN_34M_BWG_REFERENCE_TERMINALS: dict[str, GroundTerminalConfig] = {
    "Goldstone": GroundTerminalConfig(
        name="DSS-24 (X-Only, MASER-1, Diplexed)",
        site="Goldstone",
        g0_transmit_dbi=66.88,
        g0_receive_dbi=68.24,
        gain_curvature_per_deg2=0.000027,
        reference_elevation_deg=51.50,
        t1_k=30.39,
        t2_k=2.9,
        temperature_decay_per_deg=0.11,
        transmit_power_w=20e3,
        scenario_class="REFERENCE_SCENARIO",
        provenance=_SOURCE,
    ),
    "Canberra": GroundTerminalConfig(
        name="DSS-35 (X/Ka, HEMT-1, RCP, Diplexed)",
        site="Canberra",
        g0_transmit_dbi=66.99,
        g0_receive_dbi=68.35,
        gain_curvature_per_deg2=0.000045,
        reference_elevation_deg=45.00,
        t1_k=14.7,
        t2_k=0.0,
        temperature_decay_per_deg=0.00,
        transmit_power_w=20e3,
        scenario_class="REFERENCE_SCENARIO",
        provenance=_SOURCE,
    ),
}

#: X-band uplink and turnaround ratio as frozen in the project's radiometric
#: configuration; repeated here so the RF package does not import the observable.
REFERENCE_UPLINK_HZ = 7.2e9
REFERENCE_TURNAROUND = 880.0 / 749.0


def reference_two_way_link(
    site: str,
    *,
    weather_cd: float = 0.50,
    ground_loop_bandwidth_hz: float = 1.0,
    transponder_loop_bandwidth_hz: float | None = None,
    spacecraft_eirp_dbw: float | None = None,
    spacecraft_g_over_t_db_per_k: float | None = None,
    lunar_background_k: float = 0.0,
    count_interval_s: float = 60.0,
) -> TwoWayLinkConfig:
    """Assemble a reference two-way configuration for one DSN site.

    The spacecraft arguments default to ``None`` on purpose. With them unset the
    link model reports PARAMETRIC and names the missing inputs, which is the
    honest state for this project; supplying them turns the same code into a
    numeric evaluation or an inverse-design study.

    ``ground_loop_bandwidth_hz`` has no project-frozen value either. One hertz is
    a neutral reference point, not a mission parameter, and results should be
    reported against it explicitly.
    """
    try:
        terminal = DSN_34M_BWG_REFERENCE_TERMINALS[site]
    except KeyError:
        raise ValueError(
            f"no reference terminal for site {site!r}; known sites are "
            f"{sorted(DSN_34M_BWG_REFERENCE_TERMINALS)}."
        ) from None

    environment = LinkEnvironmentConfig(
        weather_cumulative_distribution=weather_cd,
        zenith_attenuation_db=zenith_attenuation_db(site, weather_cd),
        background_temperature_k=lunar_background_k,
        provenance="DSN 810-005 104L Table A-5, X-band",
    )
    ground_loop = CarrierLoopConfig(
        loop_bandwidth_hz=ground_loop_bandwidth_hz,
        carrier_mode="residual",
        provenance="REFERENCE value; no project-frozen carrier-loop bandwidth exists",
    )
    transponder_loop = (
        CarrierLoopConfig(
            loop_bandwidth_hz=transponder_loop_bandwidth_hz,
            carrier_mode="residual",
            provenance="REFERENCE value; no project-frozen transponder loop exists",
        )
        if transponder_loop_bandwidth_hz is not None
        else None
    )
    spacecraft = SpacecraftRadioConfig(
        eirp_dbw=spacecraft_eirp_dbw,
        g_over_t_db_per_k=spacecraft_g_over_t_db_per_k,
        scenario_class="PARAMETRIC",
        provenance="project contains no spacecraft RF hardware",
    )
    return TwoWayLinkConfig(
        uplink_frequency_hz=REFERENCE_UPLINK_HZ,
        turnaround_ratio=REFERENCE_TURNAROUND,
        ground_terminal=terminal,
        spacecraft=spacecraft,
        environment=environment,
        ground_carrier_loop=ground_loop,
        transponder_carrier_loop=transponder_loop,
        count_interval_s=count_interval_s,
    )
