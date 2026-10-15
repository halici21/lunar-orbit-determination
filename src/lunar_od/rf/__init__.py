"""Radio-frequency link performance and measurement-noise modelling.

This package is OPT-IN. Nothing in the estimator or the radiometric observable
path imports it, and requesting no RF model leaves existing behaviour - including
the legacy ``sigma_range_rate_mps`` weighting - numerically unchanged.

It exists because Phase 9 could not qualify the thermal contribution to
counted-Doppler noise without a link budget, and this project has never had one.
The package supplies that, along with an inverse mode that answers "what would
the spacecraft need to radiate?" when the hardware is unspecified - which, for
this project, it is.

Reference configurations built from public DSN documentation are labelled
``REFERENCE_SCENARIO``. They validate arithmetic; they are not mission truth.
"""

from __future__ import annotations

from .doppler_noise import (
    HYDROGEN_MASER_SIGMA_60S_HIGH_MPS,
    HYDROGEN_MASER_SIGMA_60S_LOW_MPS,
    cn0_for_thermal_sigma_dbhz,
    compute_counted_doppler_thermal_noise,
    compute_short_term_doppler_noise_budget,
    frequency_standard_sigma_mps,
    thermal_sigma_one_leg_mps,
)
from .environment import (
    X_BAND_ZENITH_ATTENUATION_DB,
    atmospheric_loss_db,
    ground_terminal_gain_dbi,
    system_temperature_k,
    zenith_attenuation_db,
)
from .link_budget import (
    antenna_gain_dbi,
    build_two_way_link_budget,
    carrier_loop_snr,
    cn0_dbhz,
    cn0_dbhz_via_g_over_t,
    dbw_to_watts,
    eirp_dbw,
    free_space_loss_db,
    lock_status,
    required_eirp_dbw,
    watts_to_dbw,
    wavelength_m,
)
from .reference import (
    DSN_34M_BWG_REFERENCE_TERMINALS,
    reference_two_way_link,
)
from .types import (
    AntennaConfig,
    CarrierLoopConfig,
    GroundTerminalConfig,
    LinkBudgetResult,
    LinkEnvironmentConfig,
    SpacecraftRadioConfig,
    TwoWayDopplerNoiseResult,
    TwoWayLinkConfig,
)

__all__ = [
    "DSN_34M_BWG_REFERENCE_TERMINALS",
    "HYDROGEN_MASER_SIGMA_60S_HIGH_MPS",
    "HYDROGEN_MASER_SIGMA_60S_LOW_MPS",
    "X_BAND_ZENITH_ATTENUATION_DB",
    "AntennaConfig",
    "CarrierLoopConfig",
    "GroundTerminalConfig",
    "LinkBudgetResult",
    "LinkEnvironmentConfig",
    "SpacecraftRadioConfig",
    "TwoWayDopplerNoiseResult",
    "TwoWayLinkConfig",
    "antenna_gain_dbi",
    "atmospheric_loss_db",
    "build_two_way_link_budget",
    "carrier_loop_snr",
    "cn0_dbhz",
    "cn0_dbhz_via_g_over_t",
    "cn0_for_thermal_sigma_dbhz",
    "compute_counted_doppler_thermal_noise",
    "compute_short_term_doppler_noise_budget",
    "dbw_to_watts",
    "eirp_dbw",
    "free_space_loss_db",
    "frequency_standard_sigma_mps",
    "ground_terminal_gain_dbi",
    "lock_status",
    "reference_two_way_link",
    "required_eirp_dbw",
    "system_temperature_k",
    "thermal_sigma_one_leg_mps",
    "watts_to_dbw",
    "wavelength_m",
    "zenith_attenuation_db",
]
