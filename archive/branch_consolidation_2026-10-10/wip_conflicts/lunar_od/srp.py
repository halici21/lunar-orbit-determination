"""Cannonball solar radiation pressure with a conical lunar shadow.

OPT-IN. Nothing in this module runs unless a caller passes an :class:`SRPOptions`
to a propagation entry point. With ``srp=None`` — the default everywhere — the
production force model is exactly what it was before this module existed.

THE COEFFICIENT IS ALWAYS THE CALLER'S
--------------------------------------
The controlling quantity is the cannonball coefficient

    K_SRP = C_R * A/m        [m^2/kg]

and this module never obtains it from anywhere but its caller. That is not a
stylistic choice. The project's frozen reference spacecraft
(``LTB-IRIS-DSN34X-v1``) reports ``K_SRP`` as UNKNOWN, because neither its
reflectivity nor its illuminated area is public, and the Phase 15 configuration
layer raises rather than producing a number for it. A default here — the Phase 13
parametric 0.01, a screening-envelope endpoint, a midpoint — would quietly
convert "nobody knows" into "this is the value", which is the one failure this
whole line of work exists to prevent.

So enabling SRP without a finite K is an error, not a fallback.

C_R AND A/m ARE NOT SEPARATELY REQUIRED
---------------------------------------
Only their product is physically identifiable from tracking data when the area
is unknown, so the runtime takes the product. A caller who happens to know both
can pass ``SRPOptions.from_cr_and_area_to_mass(...)``, which multiplies them and
records that it did.

RELATION TO THE PHASE 13 REFERENCE
----------------------------------
The physics below is an independent implementation of the model Phase 13
qualified, not an import of it. The Phase 13 campaign oracle stays frozen and
unmodified, and the production tests compare against it. If production imported
the oracle, or the oracle imported production, the parity test would be
comparing something to itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .constants import (
    AU_M,
    R_MOON_M,
    R_SUN_M,
    SOLAR_PRESSURE_1AU_N_M2,
)

__all__ = [
    "K_SRP_SOURCES",
    "SHADOW_MODELS",
    "SRPConfigurationError",
    "SRPOptions",
    "apparent_radii_and_separation",
    "illumination_fraction",
    "solar_pressure_at",
    "srp_acceleration",
    "srp_acceleration_kernel",
    "srp_acceleration_with_lunar_shadow",
    "srp_partial_wrt_k_srp",
]


class SRPConfigurationError(ValueError):
    """An SRP configuration that cannot be evaluated as stated.

    Raised instead of substituting a coefficient. The message names what is
    missing, because the caller is the only party that can supply it.
    """


#: Where a runtime coefficient came from. This is provenance, not physics: the
#: force is identical whichever label is attached. It exists so a result can
#: distinguish "0.01 because Phase 13 picked a screening value" from "0.01
#: because somebody measured a spacecraft", which are not the same claim.
K_SRP_SOURCES: tuple[str, ...] = (
    "CALLER_SUPPLIED_FIXED",
    "CAMPAIGN_PARAMETRIC",
    "PUBLIC_SOURCED",
    "OWNER_FROZEN_REFERENCE",
    "SCREENING_ENVELOPE_ENDPOINT",
)

#: ``CONICAL_PENUMBRA`` is the qualified production model. ``NO_SHADOW`` exists
#: for oracle fixtures and sensitivity work. There is deliberately no binary
#: cylindrical option here: Phase 13 measured its state-level error as small but
#: its illumination step as a full unit discontinuity, which is exactly what a
#: variational or numerical path must not encounter.
SHADOW_MODELS: tuple[str, ...] = ("CONICAL_PENUMBRA", "NO_SHADOW")


@dataclass(frozen=True)
class SRPOptions:
    """Explicit runtime SRP configuration.

    Invalid states are hard to construct on purpose:

    - ``SRPOptions()`` raises. Enabled is the default, and enabled without a
      coefficient has no meaning.
    - ``SRPOptions(enabled=False)`` is legal and needs no coefficient.
    - ``srp=None`` at a propagation entry point is the disabled default and
      never touches this class at all.

    ``k_srp_m2_per_kg`` is validated for finiteness and sign only. It is
    deliberately NOT range-checked against the Phase 15 screening envelope: that
    envelope excludes the solar arrays and is not a bound, so rejecting a value
    for lying outside it would enforce a limit the evidence does not support.
    """

    k_srp_m2_per_kg: float | None = None
    enabled: bool = True
    shadow_model: Literal["CONICAL_PENUMBRA", "NO_SHADOW"] = "CONICAL_PENUMBRA"
    occulting_body_radius_m: float = R_MOON_M
    k_srp_source: str = "CALLER_SUPPLIED_FIXED"
    provenance: str = ""

    def __post_init__(self) -> None:
        if self.shadow_model not in SHADOW_MODELS:
            raise SRPConfigurationError(
                f"shadow_model must be one of {SHADOW_MODELS}; got "
                f"{self.shadow_model!r}"
            )
        if self.k_srp_source not in K_SRP_SOURCES:
            raise SRPConfigurationError(
                f"k_srp_source must be one of {K_SRP_SOURCES}; got "
                f"{self.k_srp_source!r}"
            )

        if not self.enabled:
            return

        k = self.k_srp_m2_per_kg
        if k is None:
            raise SRPConfigurationError(
                "SRP is enabled but no k_srp_m2_per_kg was supplied. There is no "
                "default: the frozen reference spacecraft reports K_SRP as "
                "UNKNOWN, and the Phase 15 screening envelope is not a bound and "
                "must not be sampled as one. Pass an explicit coefficient, or "
                "leave SRP disabled."
            )
        if isinstance(k, bool) or not isinstance(k, (int, float)):
            raise SRPConfigurationError(
                f"k_srp_m2_per_kg must be a real number in m^2/kg; got "
                f"{type(k).__name__}"
            )
        k = float(k)
        if not math.isfinite(k):
            raise SRPConfigurationError(
                f"k_srp_m2_per_kg must be finite; got {k!r}"
            )
        if k < 0.0:
            raise SRPConfigurationError(
                f"k_srp_m2_per_kg must be non-negative; got {k}. A negative "
                "coefficient would point the radiation force at the Sun."
            )
        if self.occulting_body_radius_m <= 0.0:
            raise SRPConfigurationError(
                "occulting_body_radius_m must be positive; got "
                f"{self.occulting_body_radius_m}"
            )
        object.__setattr__(self, "k_srp_m2_per_kg", k)

    @classmethod
    def from_cr_and_area_to_mass(
        cls,
        c_r: float,
        area_to_mass_m2_per_kg: float,
        **kwargs,
    ) -> "SRPOptions":
        """Build from a separately known reflectivity and area-to-mass ratio.

        The product is what the force uses; the two factors are recorded in
        ``provenance`` so a later reader can see the coefficient was composed
        rather than supplied whole.
        """
        for name, value in (("c_r", c_r),
                            ("area_to_mass_m2_per_kg", area_to_mass_m2_per_kg)):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise SRPConfigurationError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise SRPConfigurationError(f"{name} must be finite")
        note = kwargs.pop("provenance", "")
        composed = f"K_SRP = C_R * A/m = {float(c_r)!r} * {float(area_to_mass_m2_per_kg)!r}"
        return cls(
            k_srp_m2_per_kg=float(c_r) * float(area_to_mass_m2_per_kg),
            provenance=f"{composed}; {note}".strip("; "),
            **kwargs,
        )

    @property
    def is_active(self) -> bool:
        """True when this configuration will actually contribute acceleration."""
        return bool(self.enabled) and self.k_srp_m2_per_kg is not None

    def require_k(self) -> float:
        if not self.is_active:
            raise SRPConfigurationError(
                "SRP is not active; no coefficient is available"
            )
        return float(self.k_srp_m2_per_kg)


# ======================================================================
# physics
# ======================================================================
def solar_pressure_at(d_sun_m: float) -> float:
    """Radiation pressure at a distance from the Sun, N/m^2.

    ``P(d) = P_1AU * (AU / d)^2``. The true instantaneous distance is used
    everywhere; Phase 13 measured a 2.079% magnitude bias over the reference arc
    from holding it at one astronomical unit, so a fixed-1-AU shortcut is not
    available here.
    """
    d = float(d_sun_m)
    if not math.isfinite(d) or d <= 0.0:
        raise SRPConfigurationError(
            f"Sun distance must be finite and positive; got {d_sun_m!r}"
        )
    return SOLAR_PRESSURE_1AU_N_M2 * (AU_M / d) ** 2


def srp_acceleration(
    r_sc_m: ArrayLike,
    r_sun_m: ArrayLike,
    k_srp_m2_per_kg: float,
    illumination: float = 1.0,
) -> np.ndarray:
    """Cannonball SRP acceleration, m/s^2, in the frame of the inputs.

        a = P(d) * K_SRP * nu * u,     u = (r_sc - r_sun) / |r_sc - r_sun|

    ``u`` points from the Sun to the spacecraft, so the force is anti-solar.
    Both positions must share an origin — in the production Moon-centered path
    that origin is the Moon.
    """
    r_sc = np.asarray(r_sc_m, dtype=float).reshape(3)
    r_sun = np.asarray(r_sun_m, dtype=float).reshape(3)
    rel = r_sc - r_sun                       # Sun -> spacecraft
    d = float(np.linalg.norm(rel))
    if d <= 0.0:
        raise SRPConfigurationError("spacecraft and Sun are coincident")
    nu = float(illumination)
    if nu == 0.0:
        return np.zeros(3)
    return (solar_pressure_at(d) * float(k_srp_m2_per_kg) * nu) * (rel / d)


def _safe_asin(x: float) -> float:
    return math.asin(max(-1.0, min(1.0, float(x))))


def _safe_acos(x: float) -> float:
    return math.acos(max(-1.0, min(1.0, float(x))))


def apparent_radii_and_separation(
    r_sc_m: ArrayLike,
    r_sun_m: ArrayLike,
    r_occ_m: ArrayLike,
    r_occ_body_m: float,
) -> tuple[float, float, float, float, float]:
    """Apparent angular radii of Sun and occultor, and their separation.

    Returns ``(alpha_sun, alpha_occ, gamma, d_sun, d_occ)`` in radians and
    metres. All three position vectors share one origin.
    """
    r_sc = np.asarray(r_sc_m, dtype=float).reshape(3)
    to_sun = np.asarray(r_sun_m, dtype=float).reshape(3) - r_sc
    to_occ = np.asarray(r_occ_m, dtype=float).reshape(3) - r_sc
    d_sun = float(np.linalg.norm(to_sun))
    d_occ = float(np.linalg.norm(to_occ))
    if d_sun <= 0.0 or d_occ <= 0.0:
        raise SRPConfigurationError(
            "spacecraft coincides with the Sun or the occulting body"
        )
    alpha_s = _safe_asin(R_SUN_M / d_sun)
    alpha_o = _safe_asin(float(r_occ_body_m) / d_occ)
    gamma = _safe_acos(float(np.dot(to_sun, to_occ)) / (d_sun * d_occ))
    return alpha_s, alpha_o, gamma, d_sun, d_occ


def _circle_overlap_area(r1: float, r2: float, d: float) -> float:
    """Area of the lens where two disks overlap.

    Written to stay well-conditioned at both contact conditions, ``d = r1 + r2``
    and ``d = |r1 - r2|``, because those are exactly the geometries a penumbra
    transition sweeps through.
    """
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    d1 = (d * d + r1 * r1 - r2 * r2) / (2.0 * d)
    d2 = d - d1
    a1 = r1 * r1 * _safe_acos(d1 / r1) - d1 * math.sqrt(max(r1 * r1 - d1 * d1, 0.0))
    a2 = r2 * r2 * _safe_acos(d2 / r2) - d2 * math.sqrt(max(r2 * r2 - d2 * d2, 0.0))
    return a1 + a2


def illumination_fraction(
    r_sc_m: ArrayLike,
    r_sun_m: ArrayLike,
    r_occ_m: ArrayLike,
    r_occ_body_m: float,
) -> tuple[float, str]:
    """Visible fraction of the solar disk, ``nu`` in [0, 1], and the regime.

    Conical: the occultor and the Sun are treated as disks on the sky and ``nu``
    is one minus their fractional overlap. It varies continuously through the
    penumbra, which is why production uses it — a binary model steps by a full
    unit at contact, and a discontinuity of that size is what a variational path
    must never meet.
    """
    a_s, a_o, gamma, _, _ = apparent_radii_and_separation(
        r_sc_m, r_sun_m, r_occ_m, r_occ_body_m
    )
    if gamma >= a_s + a_o:
        return 1.0, "FULL_LIGHT"
    if a_o >= a_s and gamma <= a_o - a_s:
        return 0.0, "UMBRA"
    if a_s > a_o and gamma <= a_s - a_o:
        return 1.0 - (a_o / a_s) ** 2, "ANNULAR"
    lens = _circle_overlap_area(a_s, a_o, gamma)
    return max(0.0, 1.0 - lens / (math.pi * a_s * a_s)), "PENUMBRA"


#: The Moon sits at the origin of the production Moon-centered frame.
_MOON_CENTRE = np.zeros(3)


def _illumination_for(
    r_sc: np.ndarray, r_sun: np.ndarray, options: SRPOptions
) -> float:
    """Illumination factor under this configuration's shadow model."""
    if options.shadow_model == "NO_SHADOW":
        return 1.0
    # Exact early exit, not an approximation: the Moon's shadow lies entirely
    # anti-sunward of its centre, so a spacecraft on the sunward side of the
    # plane through the centre cannot be occulted by it. Roughly half of a lunar
    # orbit qualifies, and skipping the angular geometry there costs nothing in
    # fidelity — the parity sweep against the Phase 13 oracle covers both
    # branches.
    if float(np.dot(r_sc, r_sun)) > 0.0:
        return 1.0
    nu, _ = illumination_fraction(
        r_sc, r_sun, _MOON_CENTRE, options.occulting_body_radius_m
    )
    return nu


def srp_acceleration_kernel(
    r_sc_m: ArrayLike,
    r_moon_sun_m: ArrayLike,
    options: SRPOptions,
) -> np.ndarray:
    """The K-independent part of the cannonball acceleration.

    Units are m/s^2 per (m^2/kg): the acceleration this configuration would
    produce at ``K_SRP = 1``.

    Writing the force as

        a_SRP = K_SRP * g(r, t),     g = P(d) * nu * u

    makes ``g`` exactly the partial derivative of the acceleration with respect
    to the coefficient. It is computed here from the geometry directly and never
    as ``a / K``, because that quotient is undefined at ``K = 0`` while the
    derivative itself is perfectly well defined — the model is linear in K, so
    the sensitivity at zero coefficient is the same finite vector as anywhere
    else. That edge case matters: an estimator may legitimately start a solve-for
    from ``K = 0``.

    Returns zero when SRP is disabled, because then the force is not in the
    model at all and there is nothing to differentiate.
    """
    if not options.enabled:
        return np.zeros(3)
    r_sc = np.asarray(r_sc_m, dtype=float).reshape(3)
    r_sun = np.asarray(r_moon_sun_m, dtype=float).reshape(3)
    nu = _illumination_for(r_sc, r_sun, options)
    if nu == 0.0:                       # umbra: no sunlight, no sensitivity
        return np.zeros(3)
    rel = r_sc - r_sun
    d = float(np.linalg.norm(rel))
    if d <= 0.0:
        raise SRPConfigurationError("spacecraft and Sun are coincident")
    return (solar_pressure_at(d) * nu) * (rel / d)


#: Alias naming the quantity by what it is used for. Same function.
srp_partial_wrt_k_srp = srp_acceleration_kernel


def srp_acceleration_with_lunar_shadow(
    r_sc_m: ArrayLike,
    r_moon_sun_m: ArrayLike,
    options: SRPOptions,
) -> np.ndarray:
    """Production entry point: SRP acceleration in Moon-centered inertial axes.

    ``r_sc_m`` and ``r_moon_sun_m`` are both referred to the Moon's centre,
    which is the origin the propagation already uses for the third-body Sun
    term — so no second Sun provider, frame or unit convention is introduced.

    Evaluated as ``K * kernel`` so that the force and its coefficient derivative
    can never drift apart: there is one geometry path, used by both.

    Earth shadow is not evaluated. Phase 13 measured a 118.5 deg margin to first
    contact of the Earth disk over the reference arc, so it is not material
    there; supporting a second occultor is a generalisation, recorded as future
    scope rather than added here for completeness.
    """
    if not options.is_active:
        return np.zeros(3)
    k = options.require_k()
    if k == 0.0:
        return np.zeros(3)
    return k * srp_acceleration_kernel(r_sc_m, r_moon_sun_m, options)
