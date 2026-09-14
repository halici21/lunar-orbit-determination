"""Atmosphere, ground-terminal gain and system noise temperature.

The formulation is the one published in DSN 810-005 Module 104 Rev L,
Appendix A, so that every number a caller sees can be traced to a table rather
than to a fitted curve of unknown origin:

    G(theta)   = G0 - G1 (theta - theta_ref)^2 - Azen / sin(theta)      (A-1)
    Top        = TAMW + Tsky                                            (A2)
    TAMW       = T1 + T2 exp(-a theta)                                  (A3)
    Tsky       = Tatm + Tcmb / L                                        (A4, A9)
    A          = Azen / sin(theta)                                      (A5)
    L          = 10^(A/10)                                              (A6)
    Tp         = 255 + 25 CD                                            (A7)
    Tatm       = Tp (1 - 1/L)                                           (A8)

Equation (A-1) is stated for elevations between 6 and 90 degrees; outside that
range the model is not evaluated rather than silently extrapolated.
"""

from __future__ import annotations

import math

from .types import GroundTerminalConfig, LinkEnvironmentConfig

__all__ = [
    "MIN_MODEL_ELEVATION_DEG",
    "T_CMB_K",
    "X_BAND_ZENITH_ATTENUATION_DB",
    "atmospheric_loss_db",
    "ground_terminal_gain_dbi",
    "sky_temperature_k",
    "system_temperature_k",
    "zenith_attenuation_db",
]

#: Cosmic microwave background temperature, K.
T_CMB_K = 2.725

#: DSN 810-005 104L Eq (A-1) is tabulated over this elevation range.
MIN_MODEL_ELEVATION_DEG = 6.0
MAX_MODEL_ELEVATION_DEG = 90.0

#: DSN 810-005 104L Table A-5, X-band zenith atmospheric attenuation, dB,
#: keyed by site and weather cumulative distribution.
X_BAND_ZENITH_ATTENUATION_DB: dict[str, dict[float, float]] = {
    "Goldstone": {0.00: 0.037, 0.25: 0.039, 0.50: 0.040, 0.90: 0.047},
    "Canberra": {0.00: 0.039, 0.25: 0.044, 0.50: 0.046, 0.90: 0.058},
    "Madrid": {0.00: 0.038, 0.25: 0.042, 0.50: 0.045, 0.90: 0.055},
}


def zenith_attenuation_db(site: str, weather_cd: float) -> float:
    """Interpolate DSN 810-005 104L Table A-5 in weather cumulative distribution.

    Interpolation is linear between tabulated points and clamped outside them;
    the table is not extrapolated beyond CD = 0.90.
    """
    try:
        table = X_BAND_ZENITH_ATTENUATION_DB[site]
    except KeyError:
        raise ValueError(
            f"no tabulated X-band zenith attenuation for site {site!r}; "
            f"known sites are {sorted(X_BAND_ZENITH_ATTENUATION_DB)}."
        ) from None
    points = sorted(table)
    if weather_cd <= points[0]:
        return table[points[0]]
    if weather_cd >= points[-1]:
        return table[points[-1]]
    for low, high in zip(points, points[1:]):
        if low <= weather_cd <= high:
            fraction = (weather_cd - low) / (high - low)
            return table[low] + fraction * (table[high] - table[low])
    raise ValueError(f"could not bracket weather_cd={weather_cd!r}.")


def _check_elevation(elevation_deg: float) -> None:
    if not MIN_MODEL_ELEVATION_DEG <= elevation_deg <= MAX_MODEL_ELEVATION_DEG:
        raise ValueError(
            "the DSN 810-005 104L gain and temperature model is stated for "
            f"{MIN_MODEL_ELEVATION_DEG}-{MAX_MODEL_ELEVATION_DEG} deg elevation; "
            f"got {elevation_deg!r}."
        )


def atmospheric_loss_db(zenith_db: float, elevation_deg: float) -> float:
    """DSN 810-005 104L Eq (A5): a flat-slab airmass, Azen / sin(elevation)."""
    _check_elevation(elevation_deg)
    return zenith_db / math.sin(math.radians(elevation_deg))


def ground_terminal_gain_dbi(
    terminal: GroundTerminalConfig,
    elevation_deg: float,
    zenith_db: float,
    *,
    transmit: bool = False,
) -> float:
    """DSN 810-005 104L Eq (A-1), including the atmospheric term."""
    _check_elevation(elevation_deg)
    if transmit:
        if terminal.g0_transmit_dbi is None:
            raise ValueError(
                f"terminal {terminal.name!r} has no transmit gain; it cannot be "
                "used for an uplink."
            )
        g0 = terminal.g0_transmit_dbi
    else:
        g0 = terminal.g0_receive_dbi
    offset = elevation_deg - terminal.reference_elevation_deg
    return (
        g0
        - terminal.gain_curvature_per_deg2 * offset * offset
        - atmospheric_loss_db(zenith_db, elevation_deg)
    )


def sky_temperature_k(zenith_db: float, elevation_deg: float, weather_cd: float) -> float:
    """DSN 810-005 104L Eq (A4)-(A9)."""
    loss_db = atmospheric_loss_db(zenith_db, elevation_deg)
    loss = 10.0 ** (loss_db / 10.0)
    physical_temperature = 255.0 + 25.0 * weather_cd
    atmosphere = physical_temperature * (1.0 - 1.0 / loss)
    return atmosphere + T_CMB_K / loss


def system_temperature_k(
    terminal: GroundTerminalConfig,
    environment: LinkEnvironmentConfig,
    elevation_deg: float,
) -> float:
    """DSN 810-005 104L Eq (A2)-(A3), plus any in-beam background.

    ``environment.background_temperature_k`` is where a warm body in the beam -
    for lunar tracking, the Moon itself - enters. It is additive and defaults to
    zero, so a caller must opt into it deliberately.
    """
    _check_elevation(elevation_deg)
    antenna_microwave = terminal.t1_k + terminal.t2_k * math.exp(
        -terminal.temperature_decay_per_deg * elevation_deg
    )
    sky = sky_temperature_k(
        environment.zenith_attenuation_db,
        elevation_deg,
        environment.weather_cumulative_distribution,
    )
    return antenna_microwave + sky + environment.background_temperature_k
