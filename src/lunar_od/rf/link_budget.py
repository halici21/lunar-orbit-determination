"""Link-budget arithmetic and the two-way coherent tracking chain.

Conventions, stated once so they cannot drift:

* powers in dBW, gains in dBi, ``G/T`` in dB/K, ``C/N0`` in dB-Hz
* ``C/N0`` carries a residual hertz and is NOT dimensionless
* the carrier-loop SNR is dimensionless and is NOT a dB-Hz quantity

That last distinction is the one worth guarding. ``rho_L = (C/N0) / B_L`` mixes
a dB-Hz quantity with a bandwidth, and comparing a loop SNR against a C/N0
directly is a category error rather than a rounding difference.
"""

from __future__ import annotations

import math

from .environment import (
    ground_terminal_gain_dbi,
    system_temperature_k,
    zenith_attenuation_db,
)
from .types import (
    AntennaConfig,
    CarrierLoopConfig,
    LinkBudgetResult,
    TwoWayLinkConfig,
)

__all__ = [
    "BOLTZMANN_DBW_PER_K_PER_HZ",
    "SPEED_OF_LIGHT_M_S",
    "antenna_gain_dbi",
    "build_two_way_link_budget",
    "carrier_loop_snr",
    "cn0_dbhz",
    "dbw_to_watts",
    "eirp_dbw",
    "free_space_loss_db",
    "lock_status",
    "noise_spectral_density_dbw_per_hz",
    "required_eirp_dbw",
    "wavelength_m",
    "watts_to_dbw",
]

SPEED_OF_LIGHT_M_S = 299792458.0

#: 10 log10(k) with k = 1.380649e-23 J/K exactly, in dBW/K/Hz.
BOLTZMANN_DBW_PER_K_PER_HZ = 10.0 * math.log10(1.380649e-23)


def watts_to_dbw(power_w: float) -> float:
    if power_w <= 0.0:
        raise ValueError("power_w must be positive to convert to dBW.")
    return 10.0 * math.log10(power_w)


def dbw_to_watts(power_dbw: float) -> float:
    return 10.0 ** (power_dbw / 10.0)


def wavelength_m(frequency_hz: float) -> float:
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive.")
    return SPEED_OF_LIGHT_M_S / frequency_hz


def antenna_gain_dbi(antenna: AntennaConfig, frequency_hz: float) -> float:
    """Aperture gain, ``G = eta (pi D / lambda)^2``.

    Doubling the diameter or the frequency each add 6.0206 dB.
    """
    lam = wavelength_m(frequency_hz)
    gain_linear = antenna.aperture_efficiency * (math.pi * antenna.diameter_m / lam) ** 2
    return 10.0 * math.log10(gain_linear)


def eirp_dbw(
    transmit_power_w: float, transmit_gain_dbi: float, transmit_loss_db: float = 0.0
) -> float:
    return watts_to_dbw(transmit_power_w) + transmit_gain_dbi - transmit_loss_db


def free_space_loss_db(range_m: float, frequency_hz: float) -> float:
    """``20 log10(4 pi R / lambda)``. Pure geometry: no configuration required."""
    if range_m <= 0.0:
        raise ValueError("range_m must be positive.")
    return 20.0 * math.log10(4.0 * math.pi * range_m / wavelength_m(frequency_hz))


def noise_spectral_density_dbw_per_hz(system_temperature: float) -> float:
    if system_temperature <= 0.0:
        raise ValueError("system_temperature must be positive.")
    return BOLTZMANN_DBW_PER_K_PER_HZ + 10.0 * math.log10(system_temperature)


def cn0_dbhz(
    eirp: float,
    range_m: float,
    frequency_hz: float,
    receive_gain_dbi: float,
    system_temperature_kelvin: float,
    extra_loss_db: float = 0.0,
) -> float:
    """Carrier power over noise density, via received power."""
    received = (
        eirp
        - free_space_loss_db(range_m, frequency_hz)
        - extra_loss_db
        + receive_gain_dbi
    )
    return received - noise_spectral_density_dbw_per_hz(system_temperature_kelvin)


def cn0_dbhz_via_g_over_t(
    eirp: float,
    range_m: float,
    frequency_hz: float,
    receive_gain_dbi: float,
    system_temperature_kelvin: float,
    extra_loss_db: float = 0.0,
) -> float:
    """The same quantity through ``G/T``.

    Kept as a separate route on purpose: agreement between the two is a cheap
    standing check against a sign slip or a double-counted temperature.
    """
    g_over_t = receive_gain_dbi - 10.0 * math.log10(system_temperature_kelvin)
    return (
        eirp
        - free_space_loss_db(range_m, frequency_hz)
        - extra_loss_db
        + g_over_t
        - BOLTZMANN_DBW_PER_K_PER_HZ
    )


def required_eirp_dbw(
    target_cn0_dbhz: float,
    range_m: float,
    frequency_hz: float,
    receive_gain_dbi: float,
    system_temperature_kelvin: float,
    extra_loss_db: float = 0.0,
) -> float:
    """Inverse design: the EIRP that would deliver a target ``C/N0``.

    This is what makes the model useful without spacecraft hardware - it answers
    "what would the spacecraft have to radiate?" instead of refusing to answer.
    """
    return (
        target_cn0_dbhz
        + noise_spectral_density_dbw_per_hz(system_temperature_kelvin)
        + free_space_loss_db(range_m, frequency_hz)
        + extra_loss_db
        - receive_gain_dbi
    )


def carrier_loop_snr(cn0: float, loop: CarrierLoopConfig) -> float:
    """``rho_L = (C/N0) / B_L``, dimensionless.

    Only the residual-carrier case is implemented. A suppressed carrier incurs a
    squaring loss whose form depends on the modulation, and inventing one would
    put an unsourced factor into every downstream Doppler sigma.
    """
    if loop.carrier_mode != "residual":
        raise NotImplementedError(
            "only the residual-carrier relation rho_L = (C/N0)/B_L is sourced "
            f"here; carrier_mode={loop.carrier_mode!r} needs a squaring-loss "
            "model from an authoritative reference."
        )
    return (10.0 ** (cn0 / 10.0)) / loop.loop_bandwidth_hz


def lock_status(rho: float | None, loop: CarrierLoopConfig) -> str:
    """Classify tracking viability.

    A Doppler precision figure computed below the tracking threshold is
    meaningless, so the regime is reported alongside every result.
    """
    if rho is None:
        return "UNKNOWN_DUE_TO_MISSING_INPUT"
    snr_db = 10.0 * math.log10(rho) if rho > 0.0 else float("-inf")
    if snr_db >= loop.lock_threshold_db:
        return "LOCK_ROBUST"
    if snr_db >= loop.marginal_threshold_db:
        return "LOCK_MARGINAL"
    return "LOCK_NOT_SUPPORTED"


def _one_leg(
    direction: str,
    config: TwoWayLinkConfig,
    frequency: float,
    range_m: float,
    elevation_deg: float,
    eirp: float | None,
    receive_gain: float | None,
    tsys: float | None,
    loop: CarrierLoopConfig | None,
    missing: tuple[str, ...],
) -> LinkBudgetResult:
    env = config.environment
    zenith = env.zenith_attenuation_db
    atmospheric = zenith / math.sin(math.radians(elevation_deg))
    extra = env.pointing_loss_db + env.polarization_loss_db + env.other_loss_db
    fspl = free_space_loss_db(range_m, frequency)

    received = cn0 = rho = rho_db = n0 = g_over_t = None
    if eirp is not None and receive_gain is not None and tsys is not None:
        received = eirp - fspl - atmospheric - extra + receive_gain
        n0 = noise_spectral_density_dbw_per_hz(tsys)
        g_over_t = receive_gain - 10.0 * math.log10(tsys)
        cn0 = received - n0
        direct = cn0_dbhz(
            eirp, range_m, frequency, receive_gain, tsys, atmospheric + extra
        )
        alternate = cn0_dbhz_via_g_over_t(
            eirp, range_m, frequency, receive_gain, tsys, atmospheric + extra
        )
        if abs(direct - alternate) > 1e-9:
            raise AssertionError(
                "the two C/N0 formulations disagree by "
                f"{direct - alternate:.3e} dB; this indicates a unit or "
                "double-counting defect."
            )
        if loop is not None:
            rho = carrier_loop_snr(cn0, loop)
            rho_db = 10.0 * math.log10(rho) if rho > 0.0 else float("-inf")

    status = "NUMERIC" if cn0 is not None else "PARAMETRIC"
    return LinkBudgetResult(
        direction=direction,
        frequency_hz=frequency,
        wavelength_m=wavelength_m(frequency),
        range_m=range_m,
        elevation_deg=elevation_deg,
        eirp_dbw=eirp,
        free_space_loss_db=fspl,
        atmospheric_loss_db=atmospheric,
        pointing_loss_db=env.pointing_loss_db,
        other_loss_db=env.polarization_loss_db + env.other_loss_db,
        receive_gain_dbi=receive_gain,
        received_carrier_power_dbw=received,
        system_temperature_k=tsys,
        g_over_t_db_per_k=g_over_t,
        noise_spectral_density_dbw_per_hz=n0,
        cn0_dbhz=cn0,
        carrier_loop_snr=rho,
        carrier_loop_snr_db=rho_db,
        lock_status=lock_status(rho, loop) if loop is not None else "UNKNOWN_DUE_TO_MISSING_INPUT",
        status=status,
        missing_inputs=missing,
    )


def build_two_way_link_budget(
    config: TwoWayLinkConfig,
    uplink_range_m: float,
    downlink_range_m: float,
    elevation_deg: float,
) -> tuple[LinkBudgetResult, LinkBudgetResult]:
    """Both legs of a coherent two-way link.

    The legs are kept separate throughout. Collapsing them would lose the
    uplink, and a weak uplink degrades two-way Doppler even when the downlink is
    strong - the transponder faithfully retransmits whatever phase noise it
    locked onto.
    """
    terminal = config.ground_terminal
    env = config.environment
    ground_extra = env.pointing_loss_db + env.polarization_loss_db + env.other_loss_db

    # ---- uplink: ground transmits, spacecraft receives
    uplink_missing: list[str] = []
    uplink_eirp = None
    if terminal.transmit_power_w is not None and terminal.g0_transmit_dbi is not None:
        transmit_gain = ground_terminal_gain_dbi(
            terminal, elevation_deg, env.zenith_attenuation_db, transmit=True
        )
        uplink_eirp = eirp_dbw(terminal.transmit_power_w, transmit_gain)
    else:
        uplink_missing.append("ground transmit power or transmit gain")

    spacecraft_g_over_t = config.spacecraft.g_over_t_db_per_k
    if spacecraft_g_over_t is None:
        uplink_missing.append("spacecraft receive G/T")

    uplink_gain = uplink_tsys = None
    if spacecraft_g_over_t is not None:
        # G/T is supplied as a single figure; split it against a nominal 1 K so
        # the shared arithmetic still applies without inventing a temperature.
        uplink_gain = spacecraft_g_over_t
        uplink_tsys = 1.0

    uplink = _one_leg(
        "uplink",
        config,
        config.uplink_frequency_hz,
        uplink_range_m,
        elevation_deg,
        uplink_eirp,
        uplink_gain,
        uplink_tsys,
        config.transponder_carrier_loop,
        tuple(uplink_missing),
    )

    # ---- downlink: spacecraft transmits, ground receives
    downlink_missing: list[str] = []
    spacecraft_eirp = config.spacecraft.eirp_dbw
    if spacecraft_eirp is None:
        if (
            config.spacecraft.transmit_power_w is not None
            and config.spacecraft.antenna is not None
        ):
            spacecraft_eirp = eirp_dbw(
                config.spacecraft.transmit_power_w,
                antenna_gain_dbi(
                    config.spacecraft.antenna, config.downlink_frequency_hz
                ),
                config.spacecraft.transmit_loss_db,
            )
        else:
            downlink_missing.append("spacecraft EIRP")

    receive_gain = ground_terminal_gain_dbi(
        terminal, elevation_deg, env.zenith_attenuation_db
    )
    tsys = system_temperature_k(terminal, env, elevation_deg)

    downlink = _one_leg(
        "downlink",
        config,
        config.downlink_frequency_hz,
        downlink_range_m,
        elevation_deg,
        spacecraft_eirp,
        receive_gain,
        tsys,
        config.ground_carrier_loop,
        tuple(downlink_missing),
    )
    return uplink, downlink
