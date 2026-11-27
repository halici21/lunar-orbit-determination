"""Opt-in SRP light-time package (F3 WP3.2 S4): the Poynting-Robertson force with the Sun at emission and the
lunar shadow with the Moon at the ray crossing.

Predeclaration: studies/phase17/configs/wp3_2_s4_srp_light_time_predeclaration.json.

The model is WP3.1's ray-crossing SRP_LIGHT_TIME term (``studies/phase17/scripts/wp3_1_materiality_diagnostics.py``,
``full_ray_crossing``), unchanged, so production equals the term WP3.1-R re-evaluated (0.109 sigma in O2xE1):

- the Sun at emission ``r_emit = r_sun - v_sun tau``, ``tau = |r_sc - r_sun| / c``;
- the light direction ``u = (r_sc - r_emit) / d``, ``d = |r_sc - r_emit|``;
- the occulting Moon where the ray passed it, ``r_occ = -v_moon delta``, ``delta = (r_sc . u) / c``;
- ``nu_LT`` the production lunar illumination (``srp.shadow_model``) with the Sun at ``r_emit`` and the Moon at
  ``r_occ``;
- ``a_LT = K P(d) nu_LT [(1 - v.u/c) u - v/c]`` (first order in v/c), ``v = v_sc + v_moon - v_sun``.

``v_sun`` and ``v_moon`` are the SSB velocities of the Sun and the Moon (J2000, geometric), ``v_sc`` the
Moon-centred spacecraft velocity. Production adds the increment ``da = a_LT - a_SRP`` to the existing SRP. The
increment depends on the spacecraft velocity as well as on its position, so the variational equations get a
velocity gradient ``G_v = -K P nu_LT (u u^T + I) / c`` beside the position gradient.

Known limits (predeclaration ``physics.known_model_limits``): WP3.1's ``v`` is relative to the Sun while ``u``
comes from the retarded Sun, which counts the Sun's SSB motion twice (<= ~5e-8 rad in the force direction); the
(v/c)^2 terms, the source-motion intensity factor, light bending and the Earth shadow's light time are left out.

All positions are Moon-centred inertial (J2000), in metres; velocities in m/s.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .constants import R_SUN_M, SPEED_OF_LIGHT_M_S
from .srp import (
    SRPOptions,
    _circle_overlap_area,
    _safe_acos,
    apparent_radii_and_separation,
    illumination_fraction,
    solar_pressure_at,
    srp_acceleration_kernel,
    srp_acceleration_position_gradient,
    srp_acceleration_with_lunar_shadow,
)

__all__ = [
    "SRPLightTimeOptions",
    "SRP_LIGHT_TIME_PROVENANCE",
    "SSBVelocityTable",
    "light_time_contact_functions",
    "light_time_geometry",
    "light_time_illumination",
    "require_srp_light_time_options",
    "sample_ssb_velocities",
    "srp_light_time_increment",
    "srp_light_time_increment_and_jacobians",
]

SRP_LIGHT_TIME_PROVENANCE = (
    "WP3.1 SRP_LIGHT_TIME, ray-crossing occulter (studies/phase17/scripts/wp3_1_materiality_diagnostics.py "
    "full_ray_crossing): Poynting-Robertson force first order in v/c with the Sun at emission, v relative to the "
    "Sun, lunar shadow with the Moon at the ray crossing delta = (r.u)/c; WP3.1-R decision 2026-11-08 (one package)"
)

_C = SPEED_OF_LIGHT_M_S


def _vector(value, name: str) -> np.ndarray:
    v = np.asarray(value, dtype=float).reshape(-1)
    if v.size != 3 or not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be a finite 3-vector.")
    return v


# ======================================================================
# SSB velocities of the Sun and the Moon
# ======================================================================
@dataclass(frozen=True, kw_only=True)
class SSBVelocityTable:
    """Sampled SSB velocities of the Sun and the Moon (J2000, m/s) on the propagation time axis ``t_s``.

    PCHIP per component (as the Sun position of ``MoonCenteredEphemeris``). Queries more than
    ``extrapolation_margin_s`` (10 s, ``MoonCenteredEphemeris``'s default) outside the grid raise
    ``EphemerisCoverageError``.
    """

    t_s: np.ndarray
    sun_vel_mps: np.ndarray
    moon_vel_mps: np.ndarray
    et0_s: float | None = None
    extrapolation_margin_s: float = 10.0
    _interp: object = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        from scipy.interpolate import PchipInterpolator

        t = np.asarray(self.t_s, dtype=float).reshape(-1)
        if t.size < 2 or not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0.0):
            raise ValueError("t_s must hold at least two finite, strictly increasing epochs.")
        sun = np.asarray(self.sun_vel_mps, dtype=float)
        moon = np.asarray(self.moon_vel_mps, dtype=float)
        for name, arr in (("sun_vel_mps", sun), ("moon_vel_mps", moon)):
            if arr.ndim != 2 or arr.shape != (t.size, 3) or not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be a finite ({t.size}, 3) array.")
        margin = self.extrapolation_margin_s
        if isinstance(margin, bool) or not math.isfinite(float(margin)) or float(margin) < 0.0:
            raise ValueError("extrapolation_margin_s must be a finite number >= 0.")
        object.__setattr__(self, "t_s", t)
        object.__setattr__(self, "sun_vel_mps", sun)
        object.__setattr__(self, "moon_vel_mps", moon)
        object.__setattr__(self, "extrapolation_margin_s", float(margin))
        object.__setattr__(self, "_interp", PchipInterpolator(t, np.hstack([sun, moon]), axis=0))

    def velocities(self, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        """``(v_sun, v_moon)``, SSB velocities (m/s) at the propagation time ``t_s``."""
        from .ephemeris import EphemerisCoverageError

        t = float(t_s)
        lo = self.t_s[0] - self.extrapolation_margin_s
        hi = self.t_s[-1] + self.extrapolation_margin_s
        if not math.isfinite(t) or t < lo or t > hi:
            raise EphemerisCoverageError(
                f"SSB velocity table queried at t={t!r} s outside [{lo!r}, {hi!r}] s")
        row = self._interp(t)
        return row[0:3], row[3:6]


def sample_ssb_velocities(et0: float, t_s) -> SSBVelocityTable:
    """Sample the SSB velocities of the Sun and the Moon with the loaded SPICE kernels.

    ``spkezr('SUN' | 'MOON', et0 + t, 'J2000', 'NONE', 'SOLAR SYSTEM BARYCENTER')``, km/s -> m/s.
    """
    import spiceypy as spice

    t_s = np.asarray(t_s, dtype=float).reshape(-1)
    sun, moon = [], []
    for t in t_s:
        et = float(et0 + t)
        sun.append(np.asarray(spice.spkezr("SUN", et, "J2000", "NONE", "SOLAR SYSTEM BARYCENTER")[0][3:6]) * 1e3)
        moon.append(np.asarray(spice.spkezr("MOON", et, "J2000", "NONE", "SOLAR SYSTEM BARYCENTER")[0][3:6]) * 1e3)
    return SSBVelocityTable(t_s=t_s, sun_vel_mps=np.vstack(sun), moon_vel_mps=np.vstack(moon), et0_s=float(et0))


@dataclass(frozen=True, kw_only=True)
class SRPLightTimeOptions:
    """Explicit opt-in configuration of the SRP light-time package (WP3.1's ray-crossing model)."""

    ssb_velocities: SSBVelocityTable
    provenance: str = SRP_LIGHT_TIME_PROVENANCE

    def __post_init__(self) -> None:
        if not isinstance(self.ssb_velocities, SSBVelocityTable):
            raise TypeError(
                f"ssb_velocities must be an SSBVelocityTable; got {type(self.ssb_velocities).__name__}.")
        if not isinstance(self.provenance, str):
            raise ValueError(f"provenance must be a str; got {type(self.provenance).__name__}.")


def require_srp_light_time_options(value) -> SRPLightTimeOptions:
    if not isinstance(value, SRPLightTimeOptions):
        raise TypeError(f"srp_light_time must be SRPLightTimeOptions or None; got {type(value).__name__}.")
    return value


# ======================================================================
# the term (WP3.1's expressions, in WP3.1's order)
# ======================================================================
def light_time_geometry(r_sc, r_sun, v_sun, v_moon):
    """``(r_emit, r_occ)``: the Sun at emission and the Moon at the ray crossing (WP3.1 ray_crossing_geometry)."""
    r_sc, r_sun = np.asarray(r_sc, float), np.asarray(r_sun, float)
    r_emit = r_sun - np.asarray(v_sun, float) * (float(np.linalg.norm(r_sc - r_sun)) / _C)
    u = (r_sc - r_emit) / float(np.linalg.norm(r_sc - r_emit))
    return r_emit, -(np.asarray(v_moon, float) * (float(np.dot(r_sc, u)) / _C))


def _lunar_illumination(r_sc, r_emit, r_occ, srp: SRPOptions) -> float:
    """WP3.1's lunar_illumination: ``srp._illumination_for`` with the Moon at ``r_occ`` (same early exit)."""
    if srp.shadow_model == "NO_SHADOW":
        return 1.0
    if float(np.dot(r_sc - r_occ, r_emit - r_occ)) > 0.0:
        return 1.0
    return illumination_fraction(r_sc, r_emit, r_occ, srp.occulting_body_radius_m)[0]


def _inputs(r_sc, v_sc, r_sun, v_sun, v_moon):
    return (_vector(r_sc, "r_sc"), _vector(v_sc, "v_sc"), _vector(r_sun, "r_sun"),
            _vector(v_sun, "v_sun"), _vector(v_moon, "v_moon"))


def light_time_illumination(r_sc, r_sun, v_sun, v_moon, srp: SRPOptions) -> float:
    """``nu_LT``: the lunar illumination with the Sun at emission and the Moon at the ray crossing."""
    r_sc = _vector(r_sc, "r_sc")
    r_emit, r_occ = light_time_geometry(r_sc, _vector(r_sun, "r_sun"), _vector(v_sun, "v_sun"),
                                        _vector(v_moon, "v_moon"))
    return _lunar_illumination(r_sc, r_emit, r_occ, srp)


def light_time_contact_functions(r_sc, r_sun, v_sun, v_moon, radius: float) -> tuple[float, float]:
    """Outer and inner light-time lunar contact functions, rad: ``gamma - (a_s + a_M)``, ``gamma - |a_s - a_M|``.

    The Sun at emission and the Moon at the ray crossing; smooth along a trajectory.
    """
    r_sc = _vector(r_sc, "r_sc")
    r_emit, r_occ = light_time_geometry(r_sc, _vector(r_sun, "r_sun"), _vector(v_sun, "v_sun"),
                                        _vector(v_moon, "v_moon"))
    a_s, a_o, gamma, _, _ = apparent_radii_and_separation(r_sc, r_emit, r_occ, radius)
    return gamma - (a_s + a_o), gamma - abs(a_s - a_o)


def srp_light_time_increment(r_sc, v_sc, r_sun, v_sun, v_moon, srp: SRPOptions) -> np.ndarray:
    """``da = a_LT - a_SRP`` (m/s^2): WP3.1's ``full_ray_crossing``, expression for expression."""
    r_sc, v_sc, r_sun, v_sun, v_moon = _inputs(r_sc, v_sc, r_sun, v_sun, v_moon)
    a_srp = srp_acceleration_with_lunar_shadow(r_sc, r_sun, srp)
    r_emit, r_occ = light_time_geometry(r_sc, r_sun, v_sun, v_moon)
    nu = _lunar_illumination(r_sc, r_emit, r_occ, srp)
    if nu == 0.0:
        return -a_srp
    rel = r_sc - r_emit
    d = float(np.linalg.norm(rel))
    u = rel / d
    v = v_sc + v_moon - v_sun
    return srp.require_k() * ((solar_pressure_at(d) * nu) * ((1.0 - float(np.dot(v, u)) / _C) * u - v / _C)) - a_srp


# ----------------------------------------------------------------------
# derivatives
# ----------------------------------------------------------------------
def _illumination_partials(r_sc, r_sun, r_occ, radius):
    """``(nu, dnu/dr_sun, dnu/dr_occ)`` of the conical ``nu`` (1/m), spacecraft held fixed.

    ``srp._illumination_and_gradient``'s derivation, with the apparent-radius and separation partials taken with
    respect to the Sun's and the occulter's positions separately (the spacecraft-position gradient is minus their
    sum). Branch for branch the regime tests of ``illumination_fraction``.
    """
    zero = np.zeros(3)
    a_s, a_o, gamma, d_s, d_o = apparent_radii_and_separation(r_sc, r_sun, r_occ, radius)
    if gamma >= a_s + a_o:
        return 1.0, zero, zero
    if a_o >= a_s and gamma <= a_o - a_s:
        return 0.0, zero, zero
    e_s = (r_sun - r_sc) / d_s
    e_o = (r_occ - r_sc) / d_o
    das_dsun = -(R_SUN_M / (d_s * math.sqrt(max(d_s * d_s - R_SUN_M * R_SUN_M, 0.0)))) * e_s
    dao_docc = -(radius / (d_o * math.sqrt(max(d_o * d_o - radius * radius, 0.0)))) * e_o \
        if d_o > radius else zero
    if a_s > a_o and gamma <= a_s - a_o:
        ratio = a_o / a_s
        return (1.0 - ratio * ratio,
                -2.0 * ratio * (-a_o * das_dsun) / (a_s * a_s),
                -2.0 * ratio * (dao_docc * a_s) / (a_s * a_s))
    lens = _circle_overlap_area(a_s, a_o, gamma)
    nu = 1.0 - lens / (math.pi * a_s * a_s)
    if nu <= 0.0:                        # the max(0, .) guard of illumination_fraction
        return 0.0, zero, zero
    c = float(np.dot(e_s, e_o))
    sin_g = math.sin(gamma)
    dg_dsun = -(e_o - c * e_s) / (d_s * sin_g) if sin_g > 0.0 else zero
    dg_docc = -(e_s - c * e_o) / (d_o * sin_g) if sin_g > 0.0 else zero
    d1 = (gamma * gamma + a_s * a_s - a_o * a_o) / (2.0 * gamma)
    d2 = gamma - d1
    chord_half = math.sqrt(max(a_s * a_s - d1 * d1, 0.0))
    dl_das = 2.0 * a_s * _safe_acos(d1 / a_s)
    dl_dao = 2.0 * a_o * _safe_acos(d2 / a_o)
    dl_dgamma = -2.0 * chord_half
    area_sun = math.pi * a_s * a_s
    d_sun = -(dl_das * das_dsun + dl_dgamma * dg_dsun) / area_sun + (2.0 * lens / (area_sun * a_s)) * das_dsun
    d_occ = -(dl_dao * dao_docc + dl_dgamma * dg_docc) / area_sun
    return nu, d_sun, d_occ


def srp_light_time_increment_and_jacobians(r_sc, v_sc, r_sun, v_sun, v_moon, srp: SRPOptions):
    """``(da, G_r, G_v, da_per_K)``: the increment, its position and velocity gradients and its part of df/dK.

    ``da`` is :func:`srp_light_time_increment` bit for bit. ``G_r = d(a_LT)/dr - G_SRP``,
    ``G_v = d(a_LT)/dv_sc = -K P nu (u u^T + I)/c``, ``da_per_K = g_LT - g_SRP`` (K-free kernels).
    """
    r_sc, v_sc, r_sun, v_sun, v_moon = _inputs(r_sc, v_sc, r_sun, v_sun, v_moon)
    k = srp.require_k() if srp.is_active else 0.0
    a_srp = srp_acceleration_with_lunar_shadow(r_sc, r_sun, srp)
    g_srp = srp_acceleration_kernel(r_sc, r_sun, srp)
    grad_srp = srp_acceleration_position_gradient(r_sc, r_sun, srp)
    r_emit, r_occ = light_time_geometry(r_sc, r_sun, v_sun, v_moon)
    nu = _lunar_illumination(r_sc, r_emit, r_occ, srp)
    if nu == 0.0:
        return -a_srp, -grad_srp, np.zeros((3, 3)), -g_srp
    rel = r_sc - r_emit
    d = float(np.linalg.norm(rel))
    u = rel / d
    v = v_sc + v_moon - v_sun
    vu = float(np.dot(v, u))
    p = solar_pressure_at(d)
    g = (1.0 - vu / _C) * u - v / _C
    da = k * ((p * nu) * g) - a_srp
    g_lt = (p * nu) * g

    eye = np.eye(3)
    w = (r_sc - r_sun) / float(np.linalg.norm(r_sc - r_sun))
    demit_dr = -np.outer(v_sun, w) / _C
    jac = eye - demit_dr                                       # d(rel)/dr
    du_dr = (eye - np.outer(u, u)) @ jac / d
    dd_dr = u @ jac
    ddelta_dr = (u + r_sc @ du_dr) / _C
    docc_dr = -np.outer(v_moon, ddelta_dr)
    grad_nu = np.zeros(3)
    if srp.shadow_model != "NO_SHADOW" and float(np.dot(r_sc - r_occ, r_emit - r_occ)) <= 0.0:
        _, p_sun, p_occ = _illumination_partials(r_sc, r_emit, r_occ, float(srp.occulting_body_radius_m))
        grad_nu = -(p_sun + p_occ) + demit_dr.T @ p_sun + docc_dr.T @ p_occ
    dg_dr = ((1.0 - vu / _C) * eye - np.outer(u, v) / _C) @ du_dr
    dglt_dr = (-2.0 * p / d) * nu * np.outer(g, dd_dr) + p * np.outer(g, grad_nu) + (p * nu) * dg_dr
    dglt_dv = -(p * nu / _C) * (np.outer(u, u) + eye)
    return da, k * dglt_dr - grad_srp, k * dglt_dv, g_lt - g_srp
