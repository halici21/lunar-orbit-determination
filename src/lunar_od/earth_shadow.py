"""Opt-in shadow of the Earth on the spacecraft's solar radiation pressure (F3 WP3.2 S3).

Predeclaration: studies/phase17/configs/wp3_2_s3_earth_shadow_predeclaration.json.

The model is WP3.1's EARTH_SHADOW term (``studies/phase17/scripts/wp3_1_force_terms.py``), unchanged, so production
equals the term WP3.1-R found material (200 sigma on the eclipse arcs):

- the SRP acceleration is multiplied by ``nu_E``, the visible fraction of the solar disk with the Earth as the
  occulter, under production's conical model (:func:`lunar_od.srp.illumination_fraction`: Sun and Earth as uniform
  disks on the sky with angular radii ``asin(R / d)``);
- the Earth is a sphere of radius 6378.137 km (WGS-84 equatorial); ``nu_E = 1`` on the sunward side of the plane
  through the Earth's centre normal to the Sun direction (the exact early exit of the lunar shadow);
- the lunar and the Earth occulters multiply: ``a = K P(d) nu_M nu_E u``.

Production adds the increment ``da = (nu_E - 1) a_SRP`` to the existing SRP, ``a_SRP`` the SRP with the lunar
shadow of ``srp.shadow_model``. Its position gradient is ``(nu_E - 1) G_SRP + a_SRP (grad nu_E)^T`` and its
part of ``df/dK`` is ``(nu_E - 1) g_SRP``. ``grad nu_E`` is the exact derivative of the conical formula,
continuous across both contacts. Geometric positions, no atmosphere, no limb darkening, no flattening.

All positions are Moon-centred inertial, in metres.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import R_SUN_M
from .srp import (
    SRPOptions,
    _circle_overlap_area,
    _safe_acos,
    apparent_radii_and_separation,
    illumination_fraction,
    srp_acceleration_kernel,
    srp_acceleration_position_gradient,
    srp_acceleration_with_lunar_shadow,
)

__all__ = [
    "EARTH_SHADOW_PROVENANCE",
    "EarthShadowOptions",
    "R_EARTH_SHADOW_M",
    "earth_illumination",
    "earth_illumination_and_gradient",
    "earth_shadow_acceleration",
    "earth_shadow_acceleration_and_gradient",
    "earth_shadow_contact_functions",
    "require_earth_shadow_options",
]

#: WGS-84 equatorial radius, the occulting sphere of the WP3.1 model.
R_EARTH_SHADOW_M = 6_378_137.0

EARTH_SHADOW_PROVENANCE = (
    "WP3.1 EARTH_SHADOW model (studies/phase17/scripts/wp3_1_force_terms.py): conical Earth shadow, spherical "
    "Earth of 6378.137 km, occulters multiplied with the lunar shadow; found material by WP3.1-R on eclipse arcs "
    "(200 sigma); no atmosphere, flattening, limb darkening or light time"
)


@dataclass(frozen=True, kw_only=True)
class EarthShadowOptions:
    """Explicit opt-in configuration of the Earth's shadow on the SRP (WP3.1 defaults)."""

    radius_m: float = R_EARTH_SHADOW_M
    provenance: str = EARTH_SHADOW_PROVENANCE

    def __post_init__(self) -> None:
        r = self.radius_m
        if isinstance(r, bool) or not isinstance(r, (int, float)) or not math.isfinite(r):
            raise ValueError(f"radius_m must be a finite real number; got {r!r}.")
        if r <= 0.0:
            raise ValueError(f"radius_m must be > 0; got {r}.")
        if not isinstance(self.provenance, str):
            raise ValueError(f"provenance must be a str; got {type(self.provenance).__name__}.")


def require_earth_shadow_options(value) -> EarthShadowOptions:
    if not isinstance(value, EarthShadowOptions):
        raise TypeError(f"earth_shadow must be EarthShadowOptions or None; got {type(value).__name__}.")
    return value


def _vector(value, name: str) -> np.ndarray:
    v = np.asarray(value, dtype=float).reshape(-1)
    if v.size != 3 or not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be a finite 3-vector.")
    return v


def _inputs(r_sc, r_sun, r_earth, options):
    options = require_earth_shadow_options(options)
    r_sc = _vector(r_sc, "r_sc")
    r_sun = _vector(r_sun, "r_sun")
    r_earth = _vector(r_earth, "r_earth")
    if float(np.linalg.norm(r_sc - r_earth)) <= options.radius_m:
        raise ValueError("the spacecraft is at or inside the Earth shadow sphere.")
    return r_sc, r_sun, r_earth, options


def earth_illumination(r_sc, r_sun, r_earth, options: EarthShadowOptions) -> float:
    """``nu_E``: visible fraction of the solar disk with the Earth as the occulter (WP3.1's expression)."""
    r_sc, r_sun, r_earth, options = _inputs(r_sc, r_sun, r_earth, options)
    if float(np.dot(r_sc - r_earth, r_sun - r_earth)) > 0.0:
        return 1.0
    return illumination_fraction(r_sc, r_sun, r_earth, options.radius_m)[0]


def _illumination_gradient(r_sc, r_sun, r_occ, radius) -> np.ndarray:
    """Spacecraft-position gradient (1/m) of the conical ``nu`` with an occulter of ``radius`` at ``r_occ``.

    ``srp._illumination_and_gradient``'s derivation with the occulter moved from the origin to ``r_occ``; branch
    for branch the regime tests of :func:`lunar_od.srp.illumination_fraction`, so the gradient belongs to the
    value it returns.  Zero in full light and umbra.
    """
    zero = np.zeros(3)
    a_s, a_o, gamma, d_s, d_o = apparent_radii_and_separation(r_sc, r_sun, r_occ, radius)
    if gamma >= a_s + a_o:
        return zero
    if a_o >= a_s and gamma <= a_o - a_s:
        return zero
    e_s = (r_sun - r_sc) / d_s
    e_o = (r_occ - r_sc) / d_o
    grad_as = (R_SUN_M / (d_s * math.sqrt(max(d_s * d_s - R_SUN_M * R_SUN_M, 0.0)))) * e_s
    grad_ao = (radius / (d_o * math.sqrt(max(d_o * d_o - radius * radius, 0.0)))) * e_o
    if a_s > a_o and gamma <= a_s - a_o:
        ratio = a_o / a_s
        return -2.0 * ratio * (grad_ao * a_s - a_o * grad_as) / (a_s * a_s)
    lens = _circle_overlap_area(a_s, a_o, gamma)
    if 1.0 - lens / (math.pi * a_s * a_s) <= 0.0:          # the max(0, .) guard of illumination_fraction
        return zero
    c = float(np.dot(e_s, e_o))
    sin_g = math.sin(gamma)
    grad_gamma = ((e_o - c * e_s) / d_s + (e_s - c * e_o) / d_o) / sin_g if sin_g > 0.0 else zero
    d1 = (gamma * gamma + a_s * a_s - a_o * a_o) / (2.0 * gamma)
    d2 = gamma - d1
    chord_half = math.sqrt(max(a_s * a_s - d1 * d1, 0.0))
    dl_das = 2.0 * a_s * _safe_acos(d1 / a_s)
    dl_dao = 2.0 * a_o * _safe_acos(d2 / a_o)
    dl_dgamma = -2.0 * chord_half
    grad_lens = dl_das * grad_as + dl_dao * grad_ao + dl_dgamma * grad_gamma
    area_sun = math.pi * a_s * a_s
    return -grad_lens / area_sun + (2.0 * lens / (area_sun * a_s)) * grad_as


def earth_illumination_and_gradient(r_sc, r_sun, r_earth, options: EarthShadowOptions):
    """``(nu_E, grad nu_E)``: the value of :func:`earth_illumination` and its exact position gradient (1/m)."""
    r_sc, r_sun, r_earth, options = _inputs(r_sc, r_sun, r_earth, options)
    if float(np.dot(r_sc - r_earth, r_sun - r_earth)) > 0.0:
        return 1.0, np.zeros(3)
    nu = illumination_fraction(r_sc, r_sun, r_earth, options.radius_m)[0]
    if nu == 1.0 or nu == 0.0:
        return nu, np.zeros(3)
    return nu, _illumination_gradient(r_sc, r_sun, r_earth, float(options.radius_m))


def earth_shadow_contact_functions(r_sc, r_sun, r_earth, options: EarthShadowOptions) -> tuple[float, float]:
    """Outer and inner contact functions of the Earth's shadow, rad: ``gamma - (a_s + a_E)``, ``gamma - |a_s - a_E|``.

    Smooth along a trajectory and zero exactly at the penumbra (outer) and umbra/annular (inner) contacts.
    """
    options = require_earth_shadow_options(options)
    a_s, a_o, gamma, _, _ = apparent_radii_and_separation(r_sc, r_sun, r_earth, options.radius_m)
    return gamma - (a_s + a_o), gamma - abs(a_s - a_o)


def earth_shadow_acceleration(r_sc, r_sun, r_earth, srp: SRPOptions, options: EarthShadowOptions) -> np.ndarray:
    """The Earth-shadow increment ``(nu_E - 1) a_SRP`` (m/s^2), WP3.1's ``earth_shadow_acceleration``."""
    nu = earth_illumination(r_sc, r_sun, r_earth, options)
    return srp_acceleration_with_lunar_shadow(r_sc, r_sun, srp) * (nu - 1.0)


def earth_shadow_acceleration_and_gradient(r_sc, r_sun, r_earth, srp: SRPOptions, options: EarthShadowOptions):
    """``(nu_E, da, d(da)/dr, (nu_E - 1) g_SRP)``: the increment, its position gradient and its part of df/dK.

    ``nu_E = 1`` returns the increment and both derivatives as zeros without evaluating the SRP.
    """
    nu, grad_nu = earth_illumination_and_gradient(r_sc, r_sun, r_earth, options)
    if nu == 1.0:
        return nu, np.zeros(3), np.zeros((3, 3)), np.zeros(3)
    a_srp = srp_acceleration_with_lunar_shadow(r_sc, r_sun, srp)
    da = a_srp * (nu - 1.0)
    gradient = (nu - 1.0) * srp_acceleration_position_gradient(r_sc, r_sun, srp) + np.outer(a_srp, grad_nu)
    return nu, da, gradient, (nu - 1.0) * srp_acceleration_kernel(r_sc, r_sun, srp)
