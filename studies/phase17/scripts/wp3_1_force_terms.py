"""F3 WP3.1-R: force terms production lacks, each added alone to the production 6-state RHS (study level).

Predeclaration: studies/phase17/configs/wp3_1_materiality_predeclaration.json (a93dbc7, amendment A1 399fdc8).
dynamics.py is unchanged: ``with_term`` adds ``scale * [0, 0, 0, a(t, x)]`` after the production closure, so
scale 0 runs the production force model through the same wrapper (gate G2). Moon-centred J2000, SI. The
geometry functions take (3,) vectors; the term functions take (t, x, ctx) and call SPICE where they need it.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from lunar_od.constants import J2_EARTH_UNNORMALIZED, R_EARTH_J2_REF_M, SPEED_OF_LIGHT_M_S
from lunar_od.force_models import body_j2_acceleration
from lunar_od.srp import SRPOptions, illumination_fraction, solar_pressure_at, srp_acceleration_with_lunar_shadow

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import phase_c6_lunar_radiation as C6  # noqa: E402
import phase_c7_dynamics_candidates as C7  # noqa: E402

TERMS = ("TIDE", "LRP", "EARTH_SHADOW", "PN", "PLANETS", "SRP_LIGHT_TIME", "EARTH_J2")
#: reported with EARTH_J2, no decision of its own: the epoch's pole minus production's identity (DYN-19)
REPORTED = ("EARTH_J2_POLE",)
K_SRP_M2_PER_KG = 0.010                     # KP2 K0
K_LRP_M2_PER_KG = K_SRP_M2_PER_KG           # one cannonball
LRP_MODEL = C6.LunarRadiation()             # albedo 0.12, emissivity 0.95, night 100 K, cap 60 x 72
TIDE_DEGREES = ((2, C7.K2_GL1800F), (3, C7.K3_GL1800F))
TIDE_RADIUS_M = C7.R_REF_GL1800F_M
R_EARTH_SHADOW_M = 6_378_137.0              # WGS-84 equatorial
PLANET_GM_CODES = {"VENUS": "299", "JUPITER BARYCENTER": "5"}


@dataclass(frozen=True)
class Context:
    """What a term needs besides (t, x): the epoch and the baseline's GMs, providers and SRP."""

    et0: float
    mu_moon: float
    mu_earth: float
    mu_sun: float
    earth_pos: Callable[[float], np.ndarray]
    sun_pos: Callable[[float], np.ndarray]   # inside a run: that closure's own Sun lookup
    srp: SRPOptions
    gm_planets: dict | None = None


def _vec(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(3)


# ======================================================================
# geometry (no SPICE)
# ======================================================================
def tide_acceleration(r_sc, r_earth, r_sun, mu_earth: float, mu_sun: float) -> np.ndarray:
    """TIDE: elastic tide of the Moon raised by the Earth and the Sun, degrees 2 and 3 (C7 model)."""
    r = _vec(r_sc)[None, :]
    out = np.zeros(3)
    for d, mu in ((r_earth, mu_earth), (r_sun, mu_sun)):
        for n, k in TIDE_DEGREES:
            out = out + C7.tide_acceleration(r, _vec(d)[None, :], mu, n, k, TIDE_RADIUS_M)[0]
    return out


def lrp_acceleration(r_sc, r_sun) -> np.ndarray:
    """LRP: lunar albedo + thermal-IR pressure on the cannonball (C6 kernel times K_LRP)."""
    return K_LRP_M2_PER_KG * C6.acceleration_kernel(_vec(r_sc)[None, :], _vec(r_sun)[None, :], LRP_MODEL)[0]


def earth_illumination(r_sc, r_sun, r_earth, radius: float = R_EARTH_SHADOW_M) -> float:
    """nu_E: visible fraction of the solar disk with the Earth as the occulter (production conical model).

    1 on the sunward side of the plane through the Earth's centre normal to the Sun direction, the same
    exact early exit as the production lunar shadow.
    """
    r_sc, r_sun, r_earth = _vec(r_sc), _vec(r_sun), _vec(r_earth)
    if float(np.dot(r_sc - r_earth, r_sun - r_earth)) > 0.0:
        return 1.0
    return illumination_fraction(r_sc, r_sun, r_earth, radius)[0]


def earth_shadow_acceleration(r_sc, r_sun, r_earth, srp: SRPOptions) -> np.ndarray:
    """EARTH_SHADOW: a_SRP (production, lunar shadow) * (nu_E - 1); the two occulters multiply."""
    return srp_acceleration_with_lunar_shadow(r_sc, r_sun, srp) * (earth_illumination(r_sc, r_sun, r_earth) - 1.0)


def pn_acceleration(r_sc, v_sc, mu_moon: float) -> np.ndarray:
    """PN: Schwarzschild term of the Moon (C7 model)."""
    return C7.pn_acceleration(_vec(r_sc)[None, :], _vec(v_sc)[None, :], mu_moon)[0]


def planets_acceleration(r_sc, r_planets: dict, gm_planets: dict) -> np.ndarray:
    """PLANETS: Venus and the Jupiter barycentre as third bodies (indirect form, C7 model)."""
    r = _vec(r_sc)[None, :]
    out = np.zeros(3)
    for name in PLANET_GM_CODES:
        out = out + C7.third_body_acceleration(r, _vec(r_planets[name])[None, :], gm_planets[name])[0]
    return out


def lunar_illumination(r_sc, r_sun, r_occ, srp: SRPOptions) -> float:
    """srp._illumination_for with the Moon at r_occ instead of the origin (same early exit, same order)."""
    if srp.shadow_model == "NO_SHADOW":
        return 1.0
    if float(np.dot(r_sc - r_occ, r_sun - r_occ)) > 0.0:
        return 1.0
    return illumination_fraction(r_sc, r_sun, r_occ, srp.occulting_body_radius_m)[0]


def light_time_geometry(r_sc, r_sun, v_sun, v_moon, c: float = SPEED_OF_LIGHT_M_S):
    """(the Sun at emission, the Moon where the ray passed it), first order, in the frame at the Moon at t.

    tau = |r_sc - r_sun| / c, delta = |r_sc| / c; v_sun, v_moon are SSB velocities.
    """
    r_sc, r_sun = _vec(r_sc), _vec(r_sun)
    tau = float(np.linalg.norm(r_sc - r_sun)) / c
    delta = float(np.linalg.norm(r_sc)) / c
    return r_sun - _vec(v_sun) * tau, -(_vec(v_moon) * delta)


def srp_light_time_acceleration(r_sc, v_sc, r_sun, v_sun, v_moon, srp: SRPOptions,
                                c: float = SPEED_OF_LIGHT_M_S) -> np.ndarray:
    """a_LT = K P(d) nu_LT [(1 - v.u/c) u - v/c] (Poynting-Robertson, first order in v/c).

    u and d run from the Sun at emission to the spacecraft, nu_LT has the Moon at its ray-crossing position,
    v = v_sc + v_moon - v_sun. With c = inf every correction is an exact zero and the operations are those of
    srp_acceleration_kernel, so a_LT equals the production a_SRP.
    """
    if not srp.is_active:
        return np.zeros(3)
    k = srp.require_k()
    if k == 0.0:
        return np.zeros(3)
    r_sc = _vec(r_sc)
    r_emit, r_occ = light_time_geometry(r_sc, r_sun, v_sun, v_moon, c)
    nu = lunar_illumination(r_sc, r_emit, r_occ, srp)
    if nu == 0.0:
        return np.zeros(3)
    rel = r_sc - r_emit
    d = float(np.linalg.norm(rel))
    u = rel / d
    v = _vec(v_sc) + _vec(v_moon) - _vec(v_sun)
    return k * ((solar_pressure_at(d) * nu) * ((1.0 - float(np.dot(v, u)) / c) * u - v / c))


def srp_light_time_term(r_sc, v_sc, r_sun, v_sun, v_moon, srp: SRPOptions,
                        c: float = SPEED_OF_LIGHT_M_S) -> np.ndarray:
    """SRP_LIGHT_TIME: a_LT - a_SRP (production)."""
    return (srp_light_time_acceleration(r_sc, v_sc, r_sun, v_sun, v_moon, srp, c)
            - srp_acceleration_with_lunar_shadow(r_sc, r_sun, srp))


def earth_j2_parts(r_sc, r_earth, mu_earth: float, c_j2000_to_bf, j2: float = J2_EARTH_UNNORMALIZED):
    """The two parts of the indirect Earth J2 (f3body_moon's form): at the spacecraft, at the Moon."""
    r_sc, r_earth = _vec(r_sc), _vec(r_earth)
    return (body_j2_acceleration(r_sc - r_earth, mu_earth, R_EARTH_J2_REF_M, j2, c_j2000_to_bf),
            body_j2_acceleration(-r_earth, mu_earth, R_EARTH_J2_REF_M, j2, c_j2000_to_bf))


def earth_j2_acceleration(r_sc, r_earth, mu_earth: float, c_j2000_to_bf,
                          j2: float = J2_EARTH_UNNORMALIZED) -> np.ndarray:
    """EARTH_J2: indirect Earth J2 with the rotation c_j2000_to_bf (ITRF93 at the epoch in the survey)."""
    a_sc, a_moon = earth_j2_parts(r_sc, r_earth, mu_earth, c_j2000_to_bf, j2)
    return a_sc - a_moon


# ======================================================================
# SPICE inputs
# ======================================================================
def body_position(name: str, et: float) -> np.ndarray:
    import spiceypy as spice
    return np.asarray(spice.spkpos(name, float(et), "J2000", "NONE", "MOON")[0], dtype=float) * 1e3


def ssb_velocity(name: str, et: float) -> np.ndarray:
    import spiceypy as spice
    state = spice.spkezr(name, float(et), "J2000", "NONE", "SOLAR SYSTEM BARYCENTER")[0]
    return np.asarray(state[3:6], dtype=float) * 1e3


def itrf93_rotation(et: float) -> np.ndarray:
    import spiceypy as spice
    return np.asarray(spice.pxform("J2000", "ITRF93", float(et)), dtype=float)


def planet_gm() -> dict:
    """GM of Venus and the Jupiter barycentre from the loaded GM kernel (gm_de440.tpc), m^3/s^2."""
    import spiceypy as spice
    return {name: float(spice.bodvrd(code, "GM", 1)[1][0]) * 1e9 for name, code in PLANET_GM_CODES.items()}


# ======================================================================
# term functions a(t, x, ctx)
# ======================================================================
def _tide(t, x, ctx: Context):
    return tide_acceleration(x[:3], ctx.earth_pos(t), ctx.sun_pos(t), ctx.mu_earth, ctx.mu_sun)


def _lrp(t, x, ctx: Context):
    return lrp_acceleration(x[:3], ctx.sun_pos(t))


def _earth_shadow(t, x, ctx: Context):
    return earth_shadow_acceleration(x[:3], ctx.sun_pos(t), ctx.earth_pos(t), ctx.srp)


def _pn(t, x, ctx: Context):
    return pn_acceleration(x[:3], x[3:6], ctx.mu_moon)


def _planets(t, x, ctx: Context):
    et = ctx.et0 + t
    return planets_acceleration(x[:3], {name: body_position(name, et) for name in PLANET_GM_CODES}, ctx.gm_planets)


def _srp_light_time(t, x, ctx: Context):
    et = ctx.et0 + t
    return srp_light_time_term(x[:3], x[3:6], ctx.sun_pos(t), ssb_velocity("SUN", et), ssb_velocity("MOON", et),
                               ctx.srp)


def _earth_j2(t, x, ctx: Context):
    return earth_j2_acceleration(x[:3], ctx.earth_pos(t), ctx.mu_earth, itrf93_rotation(ctx.et0 + t))


def _earth_j2_pole(t, x, ctx: Context):
    r_earth = ctx.earth_pos(t)
    return (earth_j2_acceleration(x[:3], r_earth, ctx.mu_earth, itrf93_rotation(ctx.et0 + t))
            - earth_j2_acceleration(x[:3], r_earth, ctx.mu_earth, np.eye(3)))


TERM_FUNCTIONS = {"TIDE": _tide, "LRP": _lrp, "EARTH_SHADOW": _earth_shadow, "PN": _pn, "PLANETS": _planets,
                  "SRP_LIGHT_TIME": _srp_light_time, "EARTH_J2": _earth_j2, "EARTH_J2_POLE": _earth_j2_pole}


def with_term(rhs, term, ctx: Context | None, scale: float = 1.0):
    """``rhs(t, x) + scale * [0, 0, 0, term(t, x, ctx)]``: the production closure runs first and unchanged.

    ``term`` is a name in TERM_FUNCTIONS or a function. The term is evaluated at scale 0 as well, so G2
    checks the wrapper, not only the closure.
    """
    fn = TERM_FUNCTIONS[term] if isinstance(term, str) else term
    scale = float(scale)

    def rhs_term(t_s, state):
        out = np.array(rhs(t_s, state), dtype=float, copy=True)
        out[3:6] += scale * np.asarray(fn(float(t_s), np.asarray(state, dtype=float), ctx), dtype=float)
        return out

    return rhs_term


def along(term, t_grid, states, ctx: Context) -> np.ndarray:
    """The term's acceleration (N, 3) at the states of a trajectory."""
    fn = TERM_FUNCTIONS[term] if isinstance(term, str) else term
    return np.array([fn(float(t), np.asarray(x, dtype=float), ctx) for t, x in zip(t_grid, states)])
