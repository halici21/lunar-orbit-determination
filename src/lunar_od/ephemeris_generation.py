"""One ephemeris generation per run: planetary ephemeris and its GMs (WP1.2).

DYN-01 / X-01 / CFG-04: before WP1.2 the runtime propagated with the DE440 planetary
ephemeris but took GM(Moon, Earth, Sun) from gm_de431 (the MATLAB snapshot), a
1.07e-8 relative difference in GM(Moon) (~3 m/day along track in low lunar orbit)
that synthetic tests cannot see because truth and estimator share it.  A run now
takes its GMs from the selected generation: DE440 -> gm_de440, DE421 (historical
profile) -> gm_de431, the values every DE421-era record used.
"""

from __future__ import annotations

import math

from .constants import (
    MU_EARTH_DE431_M3S2,
    MU_EARTH_DE440_M3S2,
    MU_MOON_DE431_M3S2,
    MU_MOON_DE440_M3S2,
    MU_SUN_DE431_M3S2,
    MU_SUN_DE440_M3S2,
)
from .spice_loader import planetary_ephemeris

#: (GM Moon, GM Earth, GM Sun) in m^3/s^2 for each planetary-ephemeris generation.
GRAVITATIONAL_PARAMETERS = {
    "DE440": (MU_MOON_DE440_M3S2, MU_EARTH_DE440_M3S2, MU_SUN_DE440_M3S2),
    "DE421": (MU_MOON_DE431_M3S2, MU_EARTH_DE431_M3S2, MU_SUN_DE431_M3S2),
}


def gravitational_parameters(ephemeris: str | None = None) -> tuple[float, float, float]:
    """``(mu_moon, mu_earth, mu_sun)`` of the selected generation (argument, else the
    ``LUNAR_OD_PLANETARY_EPHEMERIS`` environment variable, else DE440)."""
    return GRAVITATIONAL_PARAMETERS[planetary_ephemeris(ephemeris)]


def check_generation_consistency(
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    ephemeris: str | None = None,
) -> str:
    """Fail closed when the GMs handed to a run belong to another generation.

    Returns the generation name.  The comparison is exact (rel 1e-15): every GM a
    run uses comes from :data:`GRAVITATIONAL_PARAMETERS`, so any other value is a
    mixed configuration, not round-off.
    """
    name = planetary_ephemeris(ephemeris)
    for label, given, expected in zip(("Moon", "Earth", "Sun"),
                                      (mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2),
                                      GRAVITATIONAL_PARAMETERS[name]):
        if not math.isclose(float(given), expected, rel_tol=1e-15, abs_tol=0.0):
            raise ValueError(
                f"GM({label}) = {float(given)!r} m^3/s^2 is not the {name} value {expected!r}; "
                f"the planetary ephemeris and the GMs must come from one generation "
                f"(select the ephemeris with LUNAR_OD_PLANETARY_EPHEMERIS)."
            )
    return name
