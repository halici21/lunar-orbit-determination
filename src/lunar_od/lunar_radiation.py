"""Opt-in lunar albedo and thermal-IR radiation pressure (LRP) on a cannonball (F3 WP3.2 S2).

Predeclaration: studies/phase17/configs/wp3_2_s2_lunar_radiation_predeclaration.json.

The model is Phase C6's (``studies/phase17/scripts/phase_c6_lunar_radiation.py``), unchanged, so production
equals the term WP3.1-R found material (165 sigma on 1-day 100 km arcs):

- the Moon is a uniform Lambertian sphere of radius R;
- each element of the cap visible from the spacecraft emits the radiance
  ``L = [A F cos+(psi) + eps sigma T^4] / pi``, with ``T = max(T_ss cos+(psi)^(1/4), T_night)``,
  ``eps sigma T_ss^4 = (1 - A) F`` and F the solar flux at the Moon;
- the cannonball absorbs from every direction with the SRP convention:
  ``a = K / c * sum L cos(e) dA / d^2 * u``, u the unit vector from the element to the spacecraft.

The cap is integrated by C6's rule: Gauss-Legendre rings in x with ``theta = theta_max x^2`` around the nadir,
times uniform azimuth sectors, in C6's nadir frame. K is K_SRP (one cannonball), so the acceleration is K times
the kernel ``g(r, r_sun)`` that these functions return, and ``d a / dK = g``.

The gradient is the exact derivative of the discrete sum, by forward-mode differentiation through the node
geometry. The frame seed and the active radiance branches are piecewise constant, so their derivative is zero.
The gradient is not symmetric, because the radiation force is not conservative.

Moon-centred inertial axes, SI units. The kernels are loops compiled by numba when it is installed; otherwise the
same code runs as Python (correct, and slow).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .accelerated import _optional_njit
from .constants import R_MOON_M, SPEED_OF_LIGHT_M_S
from .srp import solar_pressure_at

#: W m^-2 K^-4, CODATA 2018 (exact); the value C6 uses.
STEFAN_BOLTZMANN_W_M2_K4 = 5.670374419e-8

C6_PROVENANCE = (
    "Phase C6 lunar radiation model (studies/phase17/scripts/phase_c6_lunar_radiation.py): uniform Lambertian "
    "sphere, Bond albedo 0.12, emissivity 0.95, radiative-equilibrium dayside, 100 K night floor, 60 x 72 cap "
    "quadrature; found material by WP3.1-R (165 sigma); K_LRP = K_SRP"
)


@dataclass(frozen=True, kw_only=True)
class LunarRadiationOptions:
    """Explicit opt-in configuration of the lunar radiation pressure (C6 defaults)."""

    albedo: float = 0.12
    emissivity: float = 0.95
    night_temperature_k: float = 100.0
    n_theta: int = 60
    n_phi: int = 72
    radius_m: float = R_MOON_M
    provenance: str = C6_PROVENANCE

    def __post_init__(self) -> None:
        for name in ("albedo", "emissivity", "night_temperature_k", "radius_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number; got {value!r}.")
        if not 0.0 <= self.albedo < 1.0:
            raise ValueError(f"albedo must lie in [0, 1); got {self.albedo}.")
        if not 0.0 < self.emissivity <= 1.0:
            raise ValueError(f"emissivity must lie in (0, 1]; got {self.emissivity}.")
        if self.night_temperature_k < 0.0:
            raise ValueError(f"night_temperature_k must be >= 0; got {self.night_temperature_k}.")
        if self.radius_m <= 0.0:
            raise ValueError(f"radius_m must be > 0; got {self.radius_m}.")
        for name, low in (("n_theta", 4), ("n_phi", 8)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < low:
                raise ValueError(f"{name} must be an integer >= {low}; got {value!r}.")


def require_lunar_radiation_options(value) -> LunarRadiationOptions:
    if not isinstance(value, LunarRadiationOptions):
        raise TypeError(f"lunar_radiation must be LunarRadiationOptions or None; got {type(value).__name__}.")
    return value


@lru_cache(maxsize=8)
def _nodes(n_theta: int, n_phi: int):
    """C6's nodes: Gauss-Legendre x in (0, 1) with weights, and the azimuth cosines and sines."""
    x, wx = np.polynomial.legendre.leggauss(n_theta)
    x, wx = 0.5 * (x + 1.0), 0.5 * wx
    phi = 2.0 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
    out = (x, wx, np.cos(phi), np.sin(phi))
    for a in out:
        a.setflags(write=False)
    return out


def _vector(value, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=float).reshape(-1)
    if out.shape != (3,) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be a finite 3-vector.")
    return out


@_optional_njit(cache=True)
def _matmul3(a, b):
    out = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            out[i, j] = a[i, 0] * b[0, j] + a[i, 1] * b[1, j] + a[i, 2] * b[2, j]
    return out


@_optional_njit(cache=True)
def _cross_matrix(v):
    out = np.zeros((3, 3))
    out[0, 1], out[0, 2] = -v[2], v[1]
    out[1, 0], out[1, 2] = v[2], -v[0]
    out[2, 0], out[2, 1] = -v[1], v[0]
    return out


@_optional_njit(cache=True)
def _kernel(r_sc, s_hat, flux, albedo, emissivity, t_night, radius, x, wx, cphi, sphi, want_gradient):
    """C6's sum for one spacecraft position: (g, dg/dr) per unit K, dg/dr left zero unless requested."""
    acc = np.zeros(3)
    grad = np.zeros((3, 3))
    r = math.sqrt(r_sc[0] * r_sc[0] + r_sc[1] * r_sc[1] + r_sc[2] * r_sc[2])
    z = r_sc / r
    seed = np.zeros(3)
    if abs(z[0]) < 0.9:
        seed[0] = 1.0
    else:
        seed[1] = 1.0
    c = np.array([z[1] * seed[2] - z[2] * seed[1], z[2] * seed[0] - z[0] * seed[2], z[0] * seed[1] - z[1] * seed[0]])
    c_norm = math.sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2])
    e1 = c / c_norm
    e2 = np.array([z[1] * e1[2] - z[2] * e1[1], z[2] * e1[0] - z[0] * e1[2], z[0] * e1[1] - z[1] * e1[0]])
    theta_max = math.acos(radius / r)
    t_ss = ((1.0 - albedo) * flux / (emissivity * STEFAN_BOLTZMANN_W_M2_K4)) ** 0.25
    es_t4 = emissivity * STEFAN_BOLTZMANN_W_M2_K4 * t_ss ** 4
    dphi = 2.0 * math.pi / cphi.size

    # derivatives with respect to r_sc (rows: component, columns: r_sc) of the frame and theta_max
    dz = np.zeros((3, 3))
    de1 = np.zeros((3, 3))
    de2 = np.zeros((3, 3))
    dtmax = np.zeros(3)
    if want_gradient:
        for a in range(3):
            for b in range(3):
                dz[a, b] = ((1.0 if a == b else 0.0) - z[a] * z[b]) / r
        dc = -_matmul3(_cross_matrix(seed), dz)            # c = z x seed = -[seed]x z
        proj = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                proj[a, b] = ((1.0 if a == b else 0.0) - e1[a] * e1[b]) / c_norm
        de1 = _matmul3(proj, dc)
        de2 = _matmul3(_cross_matrix(z), de1) - _matmul3(_cross_matrix(e1), dz)   # e2 = z x e1
        dtmax = (radius / (r * math.sqrt(r * r - radius * radius))) * z

    n = np.empty(3)
    d = np.empty(3)
    u = np.empty(3)
    p = np.empty(3)
    dn = np.empty((3, 3))
    dd = np.empty((3, 3))
    du = np.empty((3, 3))
    dth = np.empty(3)
    ddth = np.empty(3)
    ddist = np.empty(3)
    dq = np.empty(3)
    for i in range(x.size):
        theta = theta_max * x[i] * x[i]
        dtheta = theta_max * 2.0 * x[i] * wx[i]
        ct = math.cos(theta)
        st = math.sin(theta)
        da = radius * radius * st * dtheta * dphi
        if want_gradient:
            for b in range(3):
                dth[b] = x[i] * x[i] * dtmax[b]
                ddth[b] = 2.0 * x[i] * wx[i] * dtmax[b]
        for j in range(cphi.size):
            for k in range(3):
                p[k] = cphi[j] * e1[k] + sphi[j] * e2[k]
                n[k] = ct * z[k] + st * p[k]
                d[k] = r_sc[k] - radius * n[k]
            dist = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
            for k in range(3):
                u[k] = d[k] / dist
            cos_e = n[0] * u[0] + n[1] * u[1] + n[2] * u[2]
            cos_e_active = cos_e >= 0.0
            if not cos_e_active:
                cos_e = 0.0
            w = da / (dist * dist)
            cos_psi = n[0] * s_hat[0] + n[1] * s_hat[1] + n[2] * s_hat[2]
            lit = cos_psi > 0.0
            if not lit:
                cos_psi = 0.0
            t_day = t_ss * cos_psi ** 0.25
            thermal_day = t_day > t_night
            temp = t_day if thermal_day else t_night
            lum = albedo * flux * cos_psi / math.pi + emissivity * STEFAN_BOLTZMANN_W_M2_K4 * temp ** 4 / math.pi
            q = lum * cos_e * w
            for k in range(3):
                acc[k] += q * u[k]
            if not want_gradient:
                continue
            for a in range(3):
                for b in range(3):
                    dp_ab = cphi[j] * de1[a, b] + sphi[j] * de2[a, b]
                    dn[a, b] = (-st * dth[b]) * z[a] + ct * dz[a, b] + (ct * dth[b]) * p[a] + st * dp_ab
                    dd[a, b] = (1.0 if a == b else 0.0) - radius * dn[a, b]
            for b in range(3):
                ddist[b] = u[0] * dd[0, b] + u[1] * dd[1, b] + u[2] * dd[2, b]
            for a in range(3):
                for b in range(3):
                    du[a, b] = (dd[a, b] - u[a] * ddist[b]) / dist
            for b in range(3):
                dcos_e = 0.0
                if cos_e_active:
                    dcos_e = (u[0] * dn[0, b] + u[1] * dn[1, b] + u[2] * dn[2, b]
                              + n[0] * du[0, b] + n[1] * du[1, b] + n[2] * du[2, b])
                dda = radius * radius * dphi * (dtheta * ct * dth[b] + st * ddth[b])
                dw = dda / (dist * dist) - (2.0 * da / (dist * dist * dist)) * ddist[b]
                dcos_psi = 0.0
                if lit:
                    dcos_psi = s_hat[0] * dn[0, b] + s_hat[1] * dn[1, b] + s_hat[2] * dn[2, b]
                dlum = albedo * flux / math.pi * dcos_psi
                if thermal_day:
                    dlum += es_t4 / math.pi * dcos_psi
                dq[b] = (cos_e * w) * dlum + (lum * w) * dcos_e + (lum * cos_e) * dw
            for a in range(3):
                for b in range(3):
                    grad[a, b] += u[a] * dq[b] + q * du[a, b]
    return acc / SPEED_OF_LIGHT_M_S, grad / SPEED_OF_LIGHT_M_S


def _evaluate(r_sc, r_sun, options: LunarRadiationOptions, want_gradient: bool):
    options = require_lunar_radiation_options(options)
    r = _vector(r_sc, "r_sc_m")
    s = _vector(r_sun, "r_moon_sun_m")
    if np.linalg.norm(r) <= options.radius_m:
        raise ValueError("the spacecraft must be above the lunar radiation sphere (|r| > radius_m).")
    d_sun = float(np.linalg.norm(s))
    flux = solar_pressure_at(d_sun) * SPEED_OF_LIGHT_M_S
    x, wx, cphi, sphi = _nodes(options.n_theta, options.n_phi)
    return _kernel(r, s / d_sun, flux, float(options.albedo), float(options.emissivity),
                   float(options.night_temperature_k), float(options.radius_m), x, wx, cphi, sphi,
                   bool(want_gradient))


def lunar_radiation_acceleration_per_k(r_sc, r_sun, options: LunarRadiationOptions) -> np.ndarray:
    """g(r, r_sun), m/s^2 per (m^2/kg): the LRP acceleration is K_SRP times this."""
    return _evaluate(r_sc, r_sun, options, False)[0]


def lunar_radiation_gradient_per_k(r_sc, r_sun, options: LunarRadiationOptions) -> np.ndarray:
    """dg/dr (3, 3), s^-2 per (m^2/kg): the exact derivative of the discrete sum."""
    return _evaluate(r_sc, r_sun, options, True)[1]


def lunar_radiation_acceleration_and_gradient_per_k(r_sc, r_sun, options: LunarRadiationOptions):
    """(g, dg/dr) from one evaluation."""
    return _evaluate(r_sc, r_sun, options, True)
