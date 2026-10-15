"""Centralized physical constants for the lunar OD force model.

Single source of truth for gravitational parameters, reference radii, and zonal
J2 coefficients. This is a leaf module: it imports nothing from ``lunar_od`` and
must stay free of side effects so it can be used from both the pure-Python and
the Numba code paths (plain floats only).

Unit contract
-------------
All quantities are SI: gravitational parameters in m^3/s^2, radii in metres,
J2 coefficients dimensionless.

J2 convention
-------------
J2 coefficients here are **unnormalized** zonal coefficients (J2 = -C20).
They are NOT the geodesy-normalized C-bar(2,0); to convert use
J2 = -sqrt(5) * C-bar(2,0).

Pairing rule
------------
A J2 coefficient is only meaningful together with the reference radius of the
gravity model that produced it (the J2 perturbation scales as J2 * R_ref^2).
The (J2, R_ref) pairs below are kept from a single model each and must not be
mixed across models.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Gravitational parameters (GM), m^3/s^2
# Source: SPICE gm_de431 (GM constants distributed with DE431); values match the
# kernel pool exactly so they reproduce the current SPICE-based runtime numbers.
# NOTE (Phase 2 scope): these are DEFINED here but the production runtime still
# obtains GM from the SPICE pool; they are not yet wired into propagation.
# ---------------------------------------------------------------------------
MU_MOON_M3S2:  float = 4902.800066163796e9       # gm_de431  GM(Moon)  = 4902.800066163796 km^3/s^2
MU_EARTH_M3S2: float = 398600.4354360959e9       # gm_de431  GM(Earth) = 398600.4354360959 km^3/s^2
MU_SUN_M3S2:   float = 132712440041.9393e9       # gm_de431  GM(Sun)   = 132712440041.9393 km^3/s^2

# ---------------------------------------------------------------------------
# Reference radii, metres
# ---------------------------------------------------------------------------
# Moon mean radius (pck00010). This is the reference radius paired with the
# lunar J2 coefficient used by the propagator. Equals the legacy MOON_R_M.
R_MOON_M: float = 1_737_400.0                    # pck00010 Moon mean radius

# Earth J2 gravity reference radius (EGM96). This is the reference radius that
# pairs with J2_EARTH_UNNORMALIZED below (same gravity model). It is a gravity
# reference radius, NOT a body shape radius: distinct from geometry.WGS84_A_M
# (6378137.0 m, station/ellipsoid geometry) and from the pck00010 Earth shape
# radius (6378136.6 m, used for occultation/eclipse geometry).
R_EARTH_J2_REF_M: float = 6_378_136.3            # EGM96 gravity reference radius (pairs with J2_EARTH_UNNORMALIZED)

# ---------------------------------------------------------------------------
# Zonal J2 coefficients (unnormalized, dimensionless)
# ---------------------------------------------------------------------------
# Moon J2 (IAU/GRAIL). Unnormalized. Pairs with R_MOON_M. Equals legacy MOON_J2.
J2_MOON_UNNORMALIZED: float = 2.0346e-4          # IAU/GRAIL lunar J2 (unnormalized)

# Earth J2 (EGM96). Unnormalized: J2 = -sqrt(5) * C-bar(2,0) with
# C-bar(2,0) = -4.84165372e-4. Pairs with R_EARTH_J2_REF_M (EGM96 ref radius).
# Defined for Phase 5 (Earth-J2) preparation only; not yet wired into any
# propagation path in Phase 2.
J2_EARTH_UNNORMALIZED: float = 1.08262668e-3     # EGM96 Earth J2 (unnormalized)

# ---------------------------------------------------------------------------
# Solar radiation constants (Phase 16, for the opt-in cannonball SRP force)
#
# Consumed only by lunar_od.srp, which is itself opt-in. Defining them here
# rather than inside that module keeps one canonical declaration per number, so
# a second SRP consumer cannot introduce a slightly different solar constant.
# ---------------------------------------------------------------------------
# Speed of light in vacuum, exact by SI definition. This MUST equal
# measurements.C_LIGHT_MPS, which predates this block and is the measurement
# domain's own declaration; a test asserts the two agree so they cannot drift.
# It is repeated rather than imported because this module has no dependencies
# and the measurement stack must not be pulled into the force model.
SPEED_OF_LIGHT_M_S: float = 299_792_458.0        # SI definition, exact

# Total solar irradiance at 1 AU. IAU 2015 Resolution B3 nominal value.
SOLAR_IRRADIANCE_1AU_W_M2: float = 1361.0        # IAU 2015 B3 nominal TSI

# Radiation pressure at 1 AU: P = S / c, N/m^2. The reflectivity coefficient
# C_R carries the optical behaviour, so this is the total-absorption reference.
SOLAR_PRESSURE_1AU_N_M2: float = SOLAR_IRRADIANCE_1AU_W_M2 / SPEED_OF_LIGHT_M_S

# Astronomical unit, exact by IAU 2012 Resolution B2.
AU_M: float = 1.495978707e11                     # IAU 2012 B2, exact

# Mean solar radius (IAU 2015 B3 nominal). Used for the apparent angular radius
# of the solar disk in the conical shadow model, NOT for any gravity term.
R_SUN_M: float = 6.957e8                         # IAU 2015 B3 nominal solar radius
