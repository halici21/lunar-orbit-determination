"""Typed, unit-explicit configuration objects for the RF link model.

Every field name carries its unit, so a bare ``power`` or ``gain`` cannot slip
through. Nothing here is imported by the estimator: RF performance is a separate
concern from radiometric observable geometry, and the two are kept apart.

Externally sourced configurations carry ``provenance`` so a published number can
always be traced back to its document, revision and table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "SCENARIO_CLASSES",
    "AntennaConfig",
    "CarrierLoopConfig",
    "GroundTerminalConfig",
    "LinkEnvironmentConfig",
    "LinkBudgetResult",
    "SpacecraftRadioConfig",
    "TwoWayDopplerNoiseResult",
    "TwoWayLinkConfig",
]

#: A configuration is either a frozen mission value or a labelled reference.
#: Reference scenarios validate arithmetic; they are never mission truth.
SCENARIO_CLASSES = ("MISSION_CONFIGURATION", "REFERENCE_SCENARIO", "PARAMETRIC")


@dataclass(frozen=True)
class AntennaConfig:
    """A parabolic aperture described from its physical dimensions."""

    diameter_m: float
    aperture_efficiency: float
    provenance: str = ""

    def __post_init__(self) -> None:
        if self.diameter_m <= 0.0:
            raise ValueError(f"diameter_m must be positive; got {self.diameter_m!r}.")
        if not 0.0 < self.aperture_efficiency <= 1.0:
            raise ValueError(
                "aperture_efficiency must lie in (0, 1]; got "
                f"{self.aperture_efficiency!r}."
            )


@dataclass(frozen=True)
class GroundTerminalConfig:
    """A DSN-style ground terminal.

    The gain and noise-temperature parameters follow the DSN 810-005 Module 104
    formulation: gain falls off quadratically about the elevation at which the
    reflector panels were aligned, and the antenna-microwave temperature decays
    exponentially with elevation.

    A tracking SITE is not a terminal. Latitude and longitude say nothing about
    dish diameter, so station coordinates live elsewhere and are never conflated
    with this.
    """

    name: str
    site: str
    g0_transmit_dbi: float | None
    g0_receive_dbi: float
    gain_curvature_per_deg2: float
    reference_elevation_deg: float
    t1_k: float
    t2_k: float
    temperature_decay_per_deg: float
    transmit_power_w: float | None
    scenario_class: Literal["MISSION_CONFIGURATION", "REFERENCE_SCENARIO", "PARAMETRIC"]
    provenance: str = ""

    def __post_init__(self) -> None:
        if self.scenario_class not in SCENARIO_CLASSES:
            raise ValueError(
                f"scenario_class must be one of {SCENARIO_CLASSES}; "
                f"got {self.scenario_class!r}."
            )
        if self.t1_k < 0.0 or self.t2_k < 0.0:
            raise ValueError("noise temperature parameters must be non-negative.")
        if self.transmit_power_w is not None and self.transmit_power_w <= 0.0:
            raise ValueError("transmit_power_w must be positive when supplied.")


@dataclass(frozen=True)
class SpacecraftRadioConfig:
    """Spacecraft radio side.

    This project has never frozen spacecraft RF hardware, so ``eirp_dbw`` and
    ``g_over_t_db_per_k`` may both be ``None``. That is a supported state: the
    link model then reports PARAMETRIC and names what is missing, rather than
    inventing a transmitter.
    """

    eirp_dbw: float | None = None
    g_over_t_db_per_k: float | None = None
    transmit_power_w: float | None = None
    antenna: AntennaConfig | None = None
    transmit_loss_db: float = 0.0
    scenario_class: Literal[
        "MISSION_CONFIGURATION", "REFERENCE_SCENARIO", "PARAMETRIC"
    ] = "PARAMETRIC"
    provenance: str = ""

    @property
    def missing_inputs(self) -> tuple[str, ...]:
        missing: list[str] = []
        if self.eirp_dbw is None and (
            self.transmit_power_w is None or self.antenna is None
        ):
            missing.append("spacecraft EIRP (or transmit power plus antenna)")
        if self.g_over_t_db_per_k is None:
            missing.append("spacecraft receive G/T")
        return tuple(missing)


@dataclass(frozen=True)
class CarrierLoopConfig:
    """Carrier-tracking loop.

    ``loop_bandwidth_hz`` is never defaulted silently: the loop SNR scales
    inversely with it, so an invented bandwidth would invent a Doppler noise.
    """

    loop_bandwidth_hz: float
    carrier_mode: Literal["residual", "suppressed"] = "residual"
    lock_threshold_db: float = 10.0
    marginal_threshold_db: float = 6.0
    provenance: str = ""

    def __post_init__(self) -> None:
        if self.loop_bandwidth_hz <= 0.0:
            raise ValueError("loop_bandwidth_hz must be positive.")
        if self.marginal_threshold_db > self.lock_threshold_db:
            raise ValueError("marginal_threshold_db cannot exceed lock_threshold_db.")


@dataclass(frozen=True)
class LinkEnvironmentConfig:
    """Atmosphere, weather and in-beam background.

    ``weather_cumulative_distribution`` follows the DSN convention: 0.00 is the
    driest condition tabulated, 0.90 a high-availability design case.
    """

    weather_cumulative_distribution: float
    zenith_attenuation_db: float
    background_temperature_k: float = 0.0
    pointing_loss_db: float = 0.0
    polarization_loss_db: float = 0.0
    other_loss_db: float = 0.0
    provenance: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.weather_cumulative_distribution <= 1.0:
            raise ValueError("weather_cumulative_distribution must lie in [0, 1].")
        if self.zenith_attenuation_db < 0.0:
            raise ValueError("zenith_attenuation_db must be non-negative.")
        if self.background_temperature_k < 0.0:
            raise ValueError("background_temperature_k must be non-negative.")


@dataclass(frozen=True)
class TwoWayLinkConfig:
    """A complete coherent two-way tracking configuration."""

    uplink_frequency_hz: float
    turnaround_ratio: float
    ground_terminal: GroundTerminalConfig
    spacecraft: SpacecraftRadioConfig
    environment: LinkEnvironmentConfig
    ground_carrier_loop: CarrierLoopConfig
    transponder_carrier_loop: CarrierLoopConfig | None = None
    count_interval_s: float = 60.0

    @property
    def downlink_frequency_hz(self) -> float:
        return self.turnaround_ratio * self.uplink_frequency_hz

    def __post_init__(self) -> None:
        if self.uplink_frequency_hz <= 0.0:
            raise ValueError("uplink_frequency_hz must be positive.")
        if self.turnaround_ratio <= 0.0:
            raise ValueError("turnaround_ratio must be positive.")
        if self.count_interval_s <= 0.0:
            raise ValueError("count_interval_s must be positive.")


@dataclass(frozen=True)
class LinkBudgetResult:
    """One direction of the link, with every intermediate term retained.

    Returning the whole chain rather than a single float is deliberate: a link
    budget that cannot be audited line by line cannot be reviewed.
    """

    direction: Literal["uplink", "downlink"]
    frequency_hz: float
    wavelength_m: float
    range_m: float
    elevation_deg: float
    eirp_dbw: float | None
    free_space_loss_db: float
    atmospheric_loss_db: float
    pointing_loss_db: float
    other_loss_db: float
    receive_gain_dbi: float | None
    received_carrier_power_dbw: float | None
    system_temperature_k: float | None
    g_over_t_db_per_k: float | None
    noise_spectral_density_dbw_per_hz: float | None
    cn0_dbhz: float | None
    carrier_loop_snr: float | None
    carrier_loop_snr_db: float | None
    lock_status: str
    status: Literal["NUMERIC", "PARAMETRIC", "BLOCKED_INPUT"]
    missing_inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class TwoWayDopplerNoiseResult:
    """Thermal and frequency-standard contributions to counted-Doppler noise.

    ``sigma_short_term_white_mps`` is populated only when every Layer-1 term the
    configuration requires is itself numeric. A frequency-standard value alone is
    a component, not a total, and this object refuses to present it as one.
    """

    count_interval_s: float
    downlink_frequency_hz: float
    uplink: LinkBudgetResult
    downlink: LinkBudgetResult
    sigma_thermal_uplink_mps: float | None
    sigma_thermal_downlink_mps: float | None
    sigma_thermal_total_mps: float | None
    sigma_frequency_standard_low_mps: float
    sigma_frequency_standard_high_mps: float
    sigma_short_term_white_low_mps: float | None
    sigma_short_term_white_high_mps: float | None
    dominant_white_component: str
    status: Literal["NUMERIC", "PARAMETRIC", "BLOCKED_INPUT"]
    missing_inputs: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)
