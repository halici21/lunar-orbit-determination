"""Phase C6: lunar albedo + thermal-IR radiation pressure on a cannonball (study-level model; dynamics.py unchanged).

Predeclaration: studies/phase17/configs/phase_c6_lunar_radiation_survey_predeclaration.json.

The Moon is a uniform sphere. The cap visible from the spacecraft is cut into rings around the nadir (polar angle
theta at the Moon's centre, midpoints from 0 to acos(R/r)) and azimuth sectors. Each element is a Lambertian source
with radiance L (W m^-2 sr^-1); a cannonball with the SRP convention feels a = K * sum L cos(e) dA / (c d^2) * u,
u the unit vector from the element to the spacecraft and e the emission angle.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunar_od.constants import R_MOON_M, SPEED_OF_LIGHT_M_S
from lunar_od.srp import solar_pressure_at

STEFAN_BOLTZMANN = 5.670374419e-8      # W m^-2 K^-4 (CODATA 2018, exact)


@dataclass(frozen=True)
class LunarRadiation:
    albedo: float = 0.12
    emissivity: float = 0.95
    t_night_k: float = 100.0
    n_theta: int = 60
    n_phi: int = 72
    radius_m: float = R_MOON_M
    thermal: bool = True
    reflected: bool = True


def _cap(r_sc: np.ndarray, model: LunarRadiation):
    """Element normals (n, m, 3), unit vectors element -> spacecraft, cos(emission), dA / d^2, for each epoch."""
    r = np.linalg.norm(r_sc, axis=1)
    z = r_sc / r[:, None]
    seed = np.where(np.abs(z[:, :1]) < 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
    e1 = np.cross(z, seed)
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(z, e1)
    theta_max = np.arccos(model.radius_m / r)                                  # (n,)
    # A1: Gauss-Legendre in x with theta = theta_max x^2 (clusters the rings under the nadir, where a 50 km
    # altitude makes the integrand peak; the predeclared midpoint rule missed G_UNIT (a) by 1.7e-3)
    x, wx = np.polynomial.legendre.leggauss(model.n_theta)
    x, wx = 0.5 * (x + 1.0), 0.5 * wx
    theta = theta_max[:, None] * x[None, :] ** 2                               # (n, nt)
    dtheta = theta_max[:, None] * 2.0 * x[None, :] * wx[None, :]
    phi = 2.0 * np.pi * (np.arange(model.n_phi) + 0.5) / model.n_phi
    st, ct = np.sin(theta), np.cos(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    # normals (n, nt, np, 3)
    nrm = (ct[:, :, None, None] * z[:, None, None, :]
           + st[:, :, None, None] * (cp[None, None, :, None] * e1[:, None, None, :]
                                     + sp[None, None, :, None] * e2[:, None, None, :]))
    d = r_sc[:, None, None, :] - model.radius_m * nrm
    dist = np.linalg.norm(d, axis=-1)
    u = d / dist[..., None]
    cos_e = np.einsum("ntpk,ntpk->ntp", nrm, u)
    da = model.radius_m ** 2 * st * dtheta * (2.0 * np.pi / model.n_phi)
    w = da[:, :, None] / dist ** 2
    return nrm, u, np.clip(cos_e, 0.0, None), w


def radiance(nrm: np.ndarray, r_moon_sun: np.ndarray, model: LunarRadiation) -> np.ndarray:
    """Albedo + thermal radiance of each element, W m^-2 sr^-1 (shape of nrm without the last axis)."""
    d_sun = np.linalg.norm(r_moon_sun, axis=1)
    s_hat = r_moon_sun / d_sun[:, None]
    flux = np.array([solar_pressure_at(float(x)) for x in d_sun]) * SPEED_OF_LIGHT_M_S      # W m^-2
    shape = (-1,) + (1,) * (nrm.ndim - 2)
    cos_psi = np.clip(np.einsum("n...k,nk->n...", nrm, s_hat), 0.0, None)
    out = np.zeros(nrm.shape[:-1])
    if model.reflected:
        out += model.albedo * flux.reshape(shape) * cos_psi / np.pi
    if model.thermal:
        t_ss = ((1.0 - model.albedo) * flux / (model.emissivity * STEFAN_BOLTZMANN)) ** 0.25
        temp = np.maximum(t_ss.reshape(shape) * cos_psi ** 0.25, model.t_night_k)
        out += model.emissivity * STEFAN_BOLTZMANN * temp ** 4 / np.pi
    return out


def acceleration_kernel(r_sc, r_moon_sun, model: LunarRadiation = LunarRadiation(), chunk: int = 512,
                        radiance_fn=None) -> np.ndarray:
    """Lunar radiation acceleration per unit K (m/s^2 per m^2/kg), Moon-centred J2000, for (n, 3) or (3,) inputs.

    ``radiance_fn(nrm, r_moon_sun)`` replaces the albedo/thermal radiance (unit tests use a uniform source).
    """
    r_sc = np.atleast_2d(np.asarray(r_sc, float))
    r_moon_sun = np.atleast_2d(np.asarray(r_moon_sun, float))
    if r_moon_sun.shape[0] == 1 and r_sc.shape[0] > 1:
        r_moon_sun = np.repeat(r_moon_sun, r_sc.shape[0], axis=0)
    out = np.empty_like(r_sc)
    for i in range(0, r_sc.shape[0], chunk):
        sl = slice(i, i + chunk)
        nrm, u, cos_e, w = _cap(r_sc[sl], model)
        lum = radiance(nrm, r_moon_sun[sl], model) if radiance_fn is None else radiance_fn(nrm, r_moon_sun[sl])
        out[sl] = np.einsum("ntp,ntpk->nk", lum * cos_e * w, u) / SPEED_OF_LIGHT_M_S
    return out
