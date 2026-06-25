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
