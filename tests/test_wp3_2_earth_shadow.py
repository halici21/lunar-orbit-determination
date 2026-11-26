"""F3 WP3.2 S3: the opt-in Earth shadow on the SRP in production (gates E_G1(a), E_G2-E_G8).

Predeclaration: studies/phase17/configs/wp3_2_s3_earth_shadow_predeclaration.json (a5954f1, errata 6ebbed7 and
15468af before any result).

Oracles never use the code under test:
- the WP3.1 study term, the term WP3.1 measured (E_G2, and the finite differences of E_G4);
- an independent spherical-cap integral of the hidden solar disk and SPICE gfoclt on the real eclipses (E_G3);
- a fixed-step RK4 re-propagation of the PRODUCTION Earth-shadow-on RHS, with the identical step sequence for
  every perturbed run (E_G5, the F1 pattern).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

import lunar_od.dynamics as dyn
import lunar_od.earth_shadow as es
from lunar_od.constants import AU_M, J2_MOON_GRAIL_UNNORMALIZED, J2_MOON_UNNORMALIZED, R_MOON_M, R_SUN_M
from lunar_od.earth_shadow import EarthShadowOptions
from lunar_od.lunar_j2_model import use_lunar_j2_model
from lunar_od.lunar_tide import LunarTideOptions
from lunar_od.srp import SRPOptions, apparent_radii_and_separation, illumination_fraction, srp_acceleration_kernel
from tests import test_wp3_2_lunar_tide as G


@pytest.fixture(autouse=True, scope="module")
def _legacy_lunar_j2_model():
    """The synthetic gates run on the legacy mean-pole lunar J2 (predeclaration ``lunar_j2``), as S1/S2; the
    real-E2 cases pass the default grail_pa_de440 model explicitly."""
    with use_lunar_j2_model("iau_mean_pole_legacy"):
        yield


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "studies" / "phase17" / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "studies" / "phase17" / "scripts"))

import wp3_1_force_terms as W  # noqa: E402
import wp3_2_s3_geometry as GEO  # noqa: E402

ES = EarthShadowOptions()
TIDE = LunarTideOptions(static_field_convention="TIDE_FREE")
EPS = np.finfo(float).eps
J2 = J2_MOON_UNNORMALIZED
K = 0.05
K_VAR = 0.2                                  # E_G5 only (predeclaration)
NOS = SRPOptions(k_srp_m2_per_kg=K, shadow_model="NO_SHADOW")
PEN = SRPOptions(k_srp_m2_per_kg=K)
NOS_VAR = SRPOptions(k_srp_m2_per_kg=K_VAR, shadow_model="NO_SHADOW")
earth, sun = GEO.earth, GEO.sun_e
BASE, T6H, TERM, PLANE = GEO.BASE_E, GEO.T6H, GEO.TERM, GEO.PLANE
TIGHT = dict(method="DOP853", rtol=1e-13, atol=1e-14)


def _bits(x) -> bytes:
    return np.ascontiguousarray(np.asarray(x, dtype=float)).tobytes()


# ----------------------------------------------------------------------
# E_G1(a): off is the identity
# ----------------------------------------------------------------------
@pytest.mark.parametrize("size", [6, 42, 48])
def test_earth_shadow_wrapper_returns_the_rhs_itself_when_off(size):
    def rhs(_t, y):
        return y
    for srp in (None, NOS, SRPOptions(enabled=False)):
        assert dyn._with_earth_shadow_acceleration(rhs, earth, sun, srp, None, state_size=size) is rhs
    assert dyn._share_earth_shadow_lookup(earth, None) is earth
    assert dyn._earth_contacts(earth, None) is None


# ----------------------------------------------------------------------
# samples (seed 20261125; the Sun near the Earth direction seen from the Moon, erratum 6ebbed7)
# ----------------------------------------------------------------------
def make_samples(n=2000, seed=20261125):
    rng = np.random.default_rng(seed)
    e_hat = G._unit(rng, n)
    r_earth = e_hat * rng.uniform(356_000e3, 407_000e3, size=n)[:, None]
    q = G._unit(rng, n)
    p = q - np.sum(q * e_hat, axis=1)[:, None] * e_hat
    p /= np.linalg.norm(p, axis=1)[:, None]
    delta = np.radians(rng.uniform(0.0, 1.5, size=n))[:, None]
    r_sun = (e_hat * np.cos(delta) + p * np.sin(delta)) * rng.uniform(0.983 * AU_M, 1.017 * AU_M, size=n)[:, None]
    r_sc = G._unit(rng, n) * rng.uniform(R_MOON_M + 20e3, R_MOON_M + 3000e3, size=n)[:, None]
    return r_sc, r_sun, r_earth


@pytest.fixture(scope="module")
def samples():
    return make_samples()


# ----------------------------------------------------------------------
# E_G2: production equals the WP3.1 study term, bit for bit
# ----------------------------------------------------------------------
def wp3_1_parity(samples):
    r, s, e = samples
    srps = (SRPOptions(k_srp_m2_per_kg=0.01), SRPOptions(k_srp_m2_per_kg=0.01, shadow_model="NO_SHADOW"))
    mismatches, counts = [], dict(full_light=0, penumbra=0, umbra=0)
    for i in range(r.shape[0]):
        want = W.earth_illumination(r[i], s[i], e[i])
        got = es.earth_illumination(r[i], s[i], e[i], ES)
        nu_g, _ = es.earth_illumination_and_gradient(r[i], s[i], e[i], ES)
        if _bits(got) != _bits(want) or _bits(nu_g) != _bits(want):
            mismatches.append((i, "nu"))
        for srp in srps:
            if _bits(es.earth_shadow_acceleration(r[i], s[i], e[i], srp, ES)) != _bits(
                    W.earth_shadow_acceleration(r[i], s[i], e[i], srp)):
                mismatches.append((i, srp.shadow_model))
        counts["full_light" if want == 1.0 else "umbra" if want == 0.0 else "penumbra"] += 1
    return mismatches, counts


def test_production_equals_the_wp3_1_term_bit_for_bit(samples):
    mismatches, counts = wp3_1_parity(samples)
    assert counts["penumbra"] >= 300 and counts["umbra"] >= 300, counts        # sample-design precondition
    assert not mismatches, mismatches[:10]


def test_defaults_are_the_wp3_1_model():
    assert ES.radius_m == W.R_EARTH_SHADOW_M == 6_378_137.0


# ----------------------------------------------------------------------
# E_G3(a): independent spherical-cap oracle
# ----------------------------------------------------------------------
def _hidden_arc(rho, a_o, gamma):
    """Azimuth (rad) of the ring of angular radius ``rho`` about the Sun centre inside the occulter cap."""
    if rho <= 0.0 or gamma <= 0.0:
        return 2.0 * math.pi if max(rho, gamma) < a_o else 0.0
    # cos a_o - cos rho cos gamma, written without cancellation
    num = (math.sin(0.5 * (a_o + rho + gamma)) * math.sin(0.5 * (rho + gamma - a_o))
           + math.sin(0.5 * (a_o + rho - gamma)) * math.sin(0.5 * (rho - gamma - a_o)))
    c = num / (math.sin(rho) * math.sin(gamma))
    if c >= 1.0:
        return 0.0
    if c <= -1.0:
        return 2.0 * math.pi
    return 2.0 * math.acos(c)


def spherical_fraction(a_s, a_o, gamma):
    """1 - (solid angle of the solar cap of radius a_s hidden by the cap of radius a_o at separation gamma)
    / (solid angle of the solar cap): exact spherical geometry of uniform disks."""
    if gamma >= a_s + a_o:
        return 1.0
    points = [p for p in (abs(gamma - a_o), gamma + a_o) if 0.0 < p < a_s]
    hidden, _ = quad(lambda rho: math.sin(rho) * _hidden_arc(rho, a_o, gamma), 0.0, a_s,
                     points=points or None, epsabs=1e-15, epsrel=1e-13, limit=400)
    return 1.0 - hidden / (4.0 * math.pi * math.sin(0.5 * a_s) ** 2)


def oracle_angles(r_sc, r_sun, r_occ, radius):
    to_s, to_o = r_sun - r_sc, r_occ - r_sc
    d_s, d_o = float(np.linalg.norm(to_s)), float(np.linalg.norm(to_o))
    gamma = math.atan2(float(np.linalg.norm(np.cross(to_s, to_o))), float(np.dot(to_s, to_o)))
    return math.asin(R_SUN_M / d_s), math.asin(radius / d_o), gamma


def _planar(a_s, a_o, gamma):
    if gamma >= a_s + a_o:
        return 1.0
    if a_o >= a_s and gamma <= a_o - a_s:
        return 0.0
    if a_s > a_o and gamma <= a_s - a_o:
        return 1.0 - (a_o / a_s) ** 2
    d1 = (gamma * gamma + a_s * a_s - a_o * a_o) / (2.0 * gamma)
    d2 = gamma - d1
    lens = (a_s * a_s * math.acos(d1 / a_s) - d1 * math.sqrt(max(a_s * a_s - d1 * d1, 0.0))
            + a_o * a_o * math.acos(d2 / a_o) - d2 * math.sqrt(max(a_o * a_o - d2 * d2, 0.0)))
    return 1.0 - lens / (math.pi * a_s * a_s)


def accuracy_errors(samples):
    r, s, e = samples
    worst, worst_pen, n_pen, precondition, n_pre = 0.0, 0.0, 0, 0.0, 0
    for i in range(r.shape[0]):
        nu = es.earth_illumination(r[i], s[i], e[i], ES)
        if float(np.dot(r[i] - e[i], s[i] - e[i])) > 0.0:
            ref = 1.0
        else:
            a_s, a_o, gamma = oracle_angles(r[i], s[i], e[i], ES.radius_m)
            ref = spherical_fraction(a_s, a_o, gamma)
            if 0.0 < nu < 1.0 and n_pre < 50:
                k = 1e-3
                precondition = max(precondition, abs(spherical_fraction(k * a_s, k * a_o, k * gamma)
                                                     - _planar(k * a_s, k * a_o, k * gamma)))
                n_pre += 1
        err = abs(nu - ref)
        worst = max(worst, err)
        if 0.0 < nu < 1.0:
            n_pen += 1
            worst_pen = max(worst_pen, err)
    return dict(worst=worst, worst_penumbra=worst_pen, n_penumbra=n_pen, precondition=precondition,
                n_precondition=n_pre)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="E_G3(a) FAIL as predeclared: oracle quad epsabs floor in the scaled limit (amendment log)")
def test_illumination_matches_the_spherical_cap_oracle(samples):
    v = accuracy_errors(samples)
    assert v["n_precondition"] == 50 and v["precondition"] <= 1e-9, v
    assert v["worst"] <= 2.5e-4, v


# ----------------------------------------------------------------------
# E_G3(b): SPICE gfoclt on the real E2 and E3 eclipses
# ----------------------------------------------------------------------
def spice_contact_comparison(name):
    import spiceypy as spice
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels()
    t_ge = GEO.greatest_eclipse_et(name)
    a, b = t_ge - 6 * 3600.0, t_ge + 6 * 3600.0
    saved = {key: list(spice.gdpool(key, 0, 3)) for key in ("BODY399_RADII", "BODY10_RADII")}
    spice.pdpool("BODY399_RADII", [ES.radius_m / 1e3] * 3)
    spice.pdpool("BODY10_RADII", [R_SUN_M / 1e3] * 3)
    try:
        boundaries = {}
        for kind in ("ANY", "FULL"):
            cnfine = spice.cell_double(2)
            spice.wninsd(a, b, cnfine)
            result = spice.cell_double(200)
            spice.gfoclt(kind, "EARTH", "ELLIPSOID", "IAU_EARTH", "SUN", "ELLIPSOID", "IAU_SUN", "NONE", "MOON",
                         60.0, cnfine, result)
            edges = []
            for k in range(spice.wncard(result)):
                lo, hi = spice.wnfetd(result, k)
                edges += [x for x in (lo, hi) if a < x < b]
            boundaries[kind] = sorted(edges)
    finally:
        for key, value in saved.items():
            spice.pdpool(key, value)

    def f(u, which):
        et = t_ge + u
        s = np.asarray(spice.spkpos("SUN", et, "J2000", "NONE", "MOON")[0], float) * 1e3
        e = np.asarray(spice.spkpos("EARTH", et, "J2000", "NONE", "MOON")[0], float) * 1e3
        return es.earth_shadow_contact_functions(np.zeros(3), s, e, ES)[which]

    roots = {}
    grid = np.arange(-6 * 3600.0, 6 * 3600.0 + 1.0, 60.0)
    for kind, which in (("ANY", 0), ("FULL", 1)):
        vals = [f(u, which) for u in grid]
        roots[kind] = [t_ge + brentq(f, grid[k], grid[k + 1], args=(which,), xtol=1e-9)
                       for k in range(grid.size - 1) if vals[k] * vals[k + 1] < 0.0]
    counts_equal = all(len(roots[k]) == len(boundaries[k]) for k in roots)
    worst = max((abs(x - y) for k in roots for x, y in zip(roots[k], boundaries[k])), default=float("inf"))
    return dict(roots=roots, spice=boundaries, counts_equal=counts_equal, worst_s=worst)


@pytest.mark.parametrize("name", ["E2", "E3"])
def test_earth_contacts_match_spice_gfoclt(name):
    v = spice_contact_comparison(name)
    assert v["counts_equal"] and len(v["roots"]["ANY"]) >= 2, v
    assert v["worst_s"] <= 0.01, v


# ----------------------------------------------------------------------
# E_G4: gradients against finite differences of the WP3.1 term
# ----------------------------------------------------------------------
def _fd4(f, x, h=100.0):
    cols = []
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        cols.append((8.0 * (np.asarray(f(x + e)) - np.asarray(f(x - e)))
                     - (np.asarray(f(x + 2 * e)) - np.asarray(f(x - 2 * e)))) / (12.0 * h))
    return np.stack(cols, axis=-1)


def _margins(r, s, e):
    earth_f = es.earth_shadow_contact_functions(r, s, e, ES)
    a, b, g, _, _ = apparent_radii_and_separation(r, s, np.zeros(3), R_MOON_M)
    return min(abs(x) for x in earth_f), min(abs(g - a - b), abs(g - abs(a - b)))


def gradient_errors(samples):
    r, s, e = samples
    worst_nu, worst_nos, worst_pen, n, n_pen = 0.0, 0.0, 0.0, 0, 0
    nos, pen = SRPOptions(k_srp_m2_per_kg=0.01, shadow_model="NO_SHADOW"), SRPOptions(k_srp_m2_per_kg=0.01)
    for i in range(r.shape[0]):
        nu, grad = es.earth_illumination_and_gradient(r[i], s[i], e[i], ES)
        earth_margin, lunar_margin = _margins(r[i], s[i], e[i])
        if not (0.0 < nu < 1.0) or earth_margin < 1e-4:
            continue
        n += 1
        fd = _fd4(lambda x: W.earth_illumination(x, s[i], e[i]), r[i])
        worst_nu = max(worst_nu, float(np.linalg.norm(grad - fd) / np.linalg.norm(fd)))
        for srp, lunar_ok in ((nos, True), (pen, lunar_margin >= 1e-2)):
            if not lunar_ok:
                continue
            _, _, g_es, _ = es.earth_shadow_acceleration_and_gradient(r[i], s[i], e[i], srp, ES)
            fd_a = _fd4(lambda x: W.earth_shadow_acceleration(x, s[i], e[i], srp), r[i])
            err = float(np.linalg.norm(g_es - fd_a) / np.linalg.norm(fd_a))
            if srp is nos:
                worst_nos = max(worst_nos, err)
            else:
                worst_pen = max(worst_pen, err)
                n_pen += 1
    return dict(n=n, n_conical=n_pen, grad_nu=worst_nu, grad_da_no_shadow=worst_nos, grad_da_conical=worst_pen)


def test_gradients_match_finite_differences_of_the_wp3_1_term(samples):
    v = gradient_errors(samples)
    assert v["n"] >= 50 and v["n_conical"] >= 10, v
    assert v["grad_nu"] <= 1e-6, v
    assert v["grad_da_no_shadow"] <= 1e-6, v
    assert v["grad_da_conical"] <= 1e-6, v


# ----------------------------------------------------------------------
# E_G5: variational equations against a fixed-step RK4 re-propagation
# ----------------------------------------------------------------------
H_POS, H_VEL = 3.0, 3e-3


def _production_rhs6(srp=NOS_VAR, earth_shadow=ES):
    rhs, _ = dyn._build_state_rhs(T6H, *BASE, J2, 0.0, "indirect", None, None, None, srp,
                                  dyn._legacy_j2_rotation, dyn.MOON_R_M, earth_shadow=earth_shadow)
    return rhs


def make_rk4_columns():
    rhs = _production_rhs6()
    cols = []
    for j in range(6):
        h = H_POS if j < 3 else H_VEL
        e = np.zeros(6)
        e[j] = h
        cols.append((G._rk4(rhs, TERM + e, T6H) - G._rk4(rhs, TERM - e, T6H)) / (2 * h))
    return np.stack(cols, axis=-1)


@pytest.fixture(scope="module")
def rk4_columns():
    return make_rk4_columns()


def _replica_rhs48(include_gradient: bool, include_dfdk: bool):
    """Analysis-only 48-state replica; the negative controls drop the Earth shadow's gradient or df/dK part."""
    mus = BASE[:3]

    def rhs(t, y):
        x, phi, s_k = y[:6], y[6:42].reshape((6, 6), order="F"), y[42:48]
        r_e, r_s = earth(t), sun(t)
        x_dot = np.array(dyn.f3body_moon(x, *mus, r_e, r_s, j2_moon=J2, srp=NOS_VAR,
                                         moon_j2_rotation=dyn._legacy_j2_rotation(t), moon_j2_radius_m=dyn.MOON_R_M),
                         dtype=float)
        a = np.array(dyn.dynamics_jacobian_a_matrix(x, *mus, r_e, r_s, j2_moon=J2, srp=NOS_VAR,
                                                    moon_j2_rotation=dyn._legacy_j2_rotation(t),
                                                    moon_j2_radius_m=dyn.MOON_R_M), dtype=float)
        _nu, da, g_es, dfdk_es = es.earth_shadow_acceleration_and_gradient(x[:3], r_s, r_e, NOS_VAR, ES)
        x_dot[3:6] += da
        if include_gradient:
            a[3:6, 0:3] += g_es
        df_dk = np.zeros(6)
        df_dk[3:] = srp_acceleration_kernel(x[:3], r_s, NOS_VAR) + (dfdk_es if include_dfdk else 0.0)
        return np.concatenate([x_dot, (a @ phi).reshape(-1, order="F"), a @ s_k + df_dk])
    return rhs


def _replica(include_gradient, include_dfdk):
    y0 = np.concatenate([G._aug0(TERM), np.zeros(6)])
    sol = solve_ivp(_replica_rhs48(include_gradient, include_dfdk), (T6H[0], T6H[-1]), y0, method="DOP853",
                    t_eval=T6H, rtol=1e-13, atol=1e-14)
    assert sol.success
    return sol.y.T


def gated_stm():
    return dyn.propagate_augmented_state(T6H, G._aug0(TERM), *BASE, j2_moon=J2, srp=NOS_VAR, earth_shadow=ES, **TIGHT)


def stm_values(rk4_columns):
    gated = G._stm_err(rk4_columns, G._phi(gated_stm()))
    control = G._stm_err(rk4_columns, G._phi(_replica(include_gradient=False, include_dfdk=True)))
    return gated, control


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="E_G5(a) FAIL as predeclared: negative-control ratio 5.9 < 10 (amendment log)")
def test_earth_shadow_stm_matches_repropagation_and_omitting_its_gradient_does_not(rk4_columns):
    gated, control = stm_values(rk4_columns)
    assert gated <= 1e-7, gated
    assert control >= 10 * gated, (control, gated)


def k_sensitivity_values():
    dk = 0.02
    plus = G._rk4(_production_rhs6(SRPOptions(k_srp_m2_per_kg=K_VAR + dk, shadow_model="NO_SHADOW")), TERM, T6H)
    minus = G._rk4(_production_rhs6(SRPOptions(k_srp_m2_per_kg=K_VAR - dk, shadow_model="NO_SHADOW")), TERM, T6H)
    fd = (plus - minus) / (2 * dk)
    s_k = dyn.propagate_state_with_k_sensitivity(T6H, TERM, *BASE, srp=NOS_VAR, j2_moon=J2, earth_shadow=ES,
                                                 **TIGHT)[:, 42:48]
    control = _replica(include_gradient=True, include_dfdk=False)[:, 42:48]
    scale = np.max(np.abs(s_k))
    return float(np.max(np.abs(fd - s_k)) / scale), float(np.max(np.abs(fd - control)) / scale)


def test_earth_shadow_k_sensitivity_matches_k_repropagation():
    gated, control = k_sensitivity_values()
    assert gated <= 1e-6, gated
    assert control >= 10 * gated, (control, gated)


def _structure_states(rng):
    """Random states over the eclipse window, plus states in both the lunar and the Earth penumbra."""
    states = []
    for _ in range(40):
        t = float(rng.uniform(0.0, T6H[-1]))
        y = G._unit(rng, 1)[0] * rng.uniform(R_MOON_M + 20e3, R_MOON_M + 500e3)
        states.append((t, y))
    # both penumbrae: scan the time for the Moon centre in the Earth's penumbra, then the lunar shadow edge
    for t in np.linspace(0.3 * 3600.0, 1.6 * 3600.0, 6):
        s_hat = sun(t) / np.linalg.norm(sun(t))
        p = np.cross(s_hat, [0.0, 0.0, 1.0])
        p /= np.linalg.norm(p)
        for theta in np.radians(np.arange(65.0, 80.0, 1e-3)):
            r = (R_MOON_M + 100e3) * (-math.cos(theta) * s_hat + math.sin(theta) * p)
            if 0.0 < illumination_fraction(r, sun(t), np.zeros(3), R_MOON_M)[0] < 1.0:
                states.append((float(t), r))
                break
    return states


@pytest.mark.parametrize("size", [6, 42, 48])
@pytest.mark.parametrize("srp", [NOS, PEN], ids=["no_shadow", "conical"])
def test_wrapper_adds_exactly_the_earth_shadow_terms_and_nothing_else(size, srp):
    rng = np.random.default_rng(31 + size)

    def base(t, y):
        return np.sin(np.arange(size) + t) * (1.0 + np.abs(y)) * 1e-3 + 1.45 * (np.arange(size) % 6 >= 3)
    wrapped = dyn._with_earth_shadow_acceleration(base, earth, sun, srp, ES, state_size=size)
    n_earth_penumbra = n_both = 0
    for t, r in _structure_states(rng):
        y = rng.normal(size=size)
        y[:3] = r
        y[3:6] = rng.normal(size=3) * 1.6e3
        off, on = base(t, y), wrapped(t, y)
        nu, da, g_es, dfdk = es.earth_shadow_acceleration_and_gradient(y[:3], sun(t), earth(t), srp, ES)
        nu_m = illumination_fraction(y[:3], sun(t), np.zeros(3), R_MOON_M)[0]
        n_earth_penumbra += 0.0 < nu < 1.0
        n_both += (0.0 < nu < 1.0) and (0.0 < nu_m < 1.0) and srp is PEN
        diff = on - off
        assert not diff[0:3].any()
        np.testing.assert_allclose(diff[3:6], da, rtol=0, atol=4 * EPS * np.abs(off[3:6]).max())
        if size == 6:
            assert not diff[6:].any()
        if size >= 42:
            phi = y[6:42].reshape((6, 6), order="F")
            dphi = diff[6:42].reshape((6, 6), order="F")
            assert not dphi[0:3, :].any()
            np.testing.assert_allclose(dphi[3:6, :], g_es @ phi[0:3, :], rtol=0,
                                       atol=4 * EPS * np.abs(off[6:42]).max())
        if size == 48:
            assert not diff[42:45].any()
            np.testing.assert_allclose(diff[45:48], g_es @ y[42:45] + dfdk, rtol=0,
                                       atol=4 * EPS * np.abs(off[42:48]).max())
    assert n_earth_penumbra >= 5, n_earth_penumbra
    if srp is PEN:
        assert n_both >= 2, n_both


def test_state_only_propagation_never_evaluates_an_earth_shadow_gradient(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("evaluated the Earth-shadow gradient")
    monkeypatch.setattr(es, "earth_illumination_and_gradient", boom)
    monkeypatch.setattr(es, "earth_shadow_acceleration_and_gradient", boom)
    monkeypatch.setattr(es, "_illumination_gradient", boom)
    t = np.array([0.0, 3600.0])                       # the Moon centre enters the Earth's penumbra
    dyn.propagate_state(t, TERM, *BASE, srp=NOS, earth_shadow=ES)
    with pytest.raises(AssertionError, match="Earth-shadow gradient"):     # the patch is live
        dyn.propagate_augmented_state(t, G._aug0(TERM), *BASE, srp=NOS, earth_shadow=ES, method="DOP853")


# ----------------------------------------------------------------------
# E_G6: 6-, 42- and 48-state paths agree
# ----------------------------------------------------------------------
PATH_CASES = ("i_j2", "ii_j2_tide", "iii_harmonics", "iv_conical_events", "v_e2_grail_events")


def _path_setup(case):
    if case == "v_e2_grail_events":
        from lunar_od.spice_loader import load_spice_kernels
        load_spice_kernels()
        _et0, t, eph, mus, x0 = GEO.e2_case()
        base = (*mus, eph.earth_position, eph.sun_position)
        kw = dict(srp=PEN, shadow_events=True, j2_moon=J2_MOON_GRAIL_UNNORMALIZED, lunar_j2_model="grail_pa_de440")
        return t, x0, base, kw, kw, True
    if case == "iv_conical_events":
        kw = dict(srp=PEN, shadow_events=True, j2_moon=J2)
        return T6H, PLANE, BASE, kw, kw, True
    kw = dict(srp=NOS)
    if case == "ii_j2_tide":
        kw["lunar_tide"] = TIDE
    if case == "iii_harmonics":
        kw.update(harmonic_model=G._zonal_model(), harmonic_rotation=G._rot_x(0.03))
        return T6H, TERM, BASE, kw, dict(kw, harmonic_stm_opt_in=True), False
    kw["j2_moon"] = J2
    return T6H, TERM, BASE, kw, kw, True


def path_differences(case):
    t, x0, base, kw6, kw_var, reference = _path_setup(case)
    kw6, kw_var = dict(kw6, earth_shadow=ES, **TIGHT), dict(kw_var, earth_shadow=ES, **TIGHT)
    p6 = dyn.propagate_state(t, x0, *base, **kw6)
    runs = {"42": dyn.propagate_augmented_state(t, G._aug0(x0), *base, **kw_var),
            "48": dyn.propagate_state_with_k_sensitivity(t, x0, *base, **kw_var)}
    if reference:
        runs["42_reference"] = G._reference(lambda: dyn.propagate_augmented_state(t, G._aug0(x0), *base, **kw_var))
    scale = np.max(np.abs(runs["48"][:, 6:42]))
    p6_free = dyn.propagate_state(t, x0, *base, **dict(kw6, earth_shadow=None))
    return ({name: float(np.max(np.abs(run[:, 0:3] - p6[:, 0:3]))) for name, run in runs.items()},
            {name: float(np.max(np.abs(run[:, 6:42] - runs["48"][:, 6:42])) / scale) for name, run in runs.items()},
            float(np.max(np.abs(p6_free[:, 0:3] - p6[:, 0:3]))))


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="E_G6 FAIL as predeclared in all five cases (WP3.2 S3 report; amendment log)")
@pytest.mark.parametrize("case", PATH_CASES)
def test_six_42_and_48_state_paths_carry_the_same_earth_shadow(case):
    pos, phi, effect = path_differences(case)
    assert max(pos.values()) <= 1e-5, pos
    assert max(phi.values()) <= 1e-8, phi
    assert effect > 1.0, effect                       # the Earth shadow is really in all of them


# ----------------------------------------------------------------------
# E_G7: event-segmented (D2) propagation with the Earth contacts
# ----------------------------------------------------------------------
D2 = dict(method="DOP853", shadow_events=True, rtol=1e-11, atol=1e-12, j2_moon=J2)
PRE = dict(rtol=1e-11, atol=1e-12, j2_moon=J2)


def classify_cuts(cuts, x0, srp, earth_shadow):
    """``[(label, offset_s)]`` of each cut on the 6-state D2 trajectory: the contact (``lunar_outer``,
    ``lunar_inner``, ``earth_outer``, ``earth_inner``) with the smallest time offset."""
    if not len(cuts):
        return []
    t = np.array([0.0, *cuts])
    states = dyn.propagate_state(t, x0, *BASE, srp=srp, earth_shadow=earth_shadow, **D2)
    out = []
    for tc, state in zip(t[1:], states[1:]):
        offsets = dyn._contact_time_offsets(float(tc), state, sun, srp, (earth, ES))
        label = min(offsets, key=lambda k: offsets[k])
        out.append((label, offsets[label]))
    return out


def shadow_event_values():
    x0 = PLANE
    for call in (lambda: dyn.propagate_state(T6H, x0, *BASE, srp=PEN, earth_shadow=ES, **D2),
                 lambda: dyn.propagate_augmented_state(T6H, G._aug0(x0), *BASE, srp=PEN, earth_shadow=ES, **D2),
                 lambda: dyn.propagate_state_with_k_sensitivity(T6H, x0, *BASE, srp=PEN, earth_shadow=ES, **D2)):
        assert np.all(np.isfinite(call()))
    cuts_es = np.array(dyn.shadow_contact_times(T6H, x0, *BASE, srp=PEN, earth_shadow=ES, **PRE))
    cuts_free = np.array(dyn.shadow_contact_times(T6H, x0, *BASE, srp=PEN, **PRE))
    labels_es = classify_cuts(cuts_es, x0, PEN, ES)
    labels_free = classify_cuts(cuts_free, x0, PEN, None)
    lunar_es = np.array([c for c, (lab, _) in zip(cuts_es, labels_es) if lab.startswith("lunar")])
    shifts = [float(np.min(np.abs(cuts_free - c))) for c in lunar_es]
    rhs, sun_lookup = dyn._build_state_rhs(T6H, *BASE, J2, 0.0, "indirect", None, None, None, PEN,
                                           dyn._legacy_j2_rotation, dyn.MOON_R_M, earth_shadow=ES)
    message = None
    try:
        dyn._integrate_segmented(T6H, x0, rhs, 1e-11, 1e-12, "DOP853", list(cuts_free), "control",
                                 contact_check=dyn._shadow_contact_check(sun_lookup, PEN))
    except RuntimeError as exc:
        message = str(exc)
    cuts_only_earth = np.array(dyn.shadow_contact_times(T6H, TERM, *BASE, srp=NOS, earth_shadow=ES, **PRE))
    labels_only_earth = classify_cuts(cuts_only_earth, TERM, NOS, ES)
    for call in (lambda: dyn.propagate_state(T6H, TERM, *BASE, srp=NOS, earth_shadow=ES, **D2),
                 lambda: dyn.propagate_augmented_state(T6H, G._aug0(TERM), *BASE, srp=NOS, earth_shadow=ES, **D2),
                 lambda: dyn.propagate_state_with_k_sensitivity(T6H, TERM, *BASE, srp=NOS, earth_shadow=ES, **D2)):
        assert np.all(np.isfinite(call()))
    return dict(cuts_es=cuts_es, cuts_free=cuts_free, labels_es=labels_es, labels_free=labels_free,
                max_lunar_shift=max(shifts) if shifts else 0.0, control_message=message,
                cuts_only_earth=cuts_only_earth, labels_only_earth=labels_only_earth)


def test_shadow_events_cut_at_the_earth_contacts():
    v = shadow_event_values()
    tol = dyn._CONTACT_TIME_TOL_S
    assert all(off <= tol for _, off in v["labels_es"]), v["labels_es"]
    n_earth = sum(lab.startswith("earth") for lab, _ in v["labels_es"])
    assert n_earth >= 4, v["labels_es"]
    assert not any(lab.startswith("earth") for lab, _ in v["labels_free"]), v["labels_free"]
    assert v["max_lunar_shift"] > tol, v["max_lunar_shift"]
    assert v["control_message"] is not None and "Shadow contact inconsistent" in v["control_message"]
    assert len(v["labels_only_earth"]) >= 2, v["labels_only_earth"]
    assert all(lab.startswith("earth") and off <= tol for lab, off in v["labels_only_earth"]), v["labels_only_earth"]


# ----------------------------------------------------------------------
# E_G8: fail closed
# ----------------------------------------------------------------------
@pytest.mark.parametrize("radius", [0.0, -6378137.0, float("nan"), float("inf"), True, "6378137.0"])
def test_invalid_earth_shadow_options_raise(radius):
    with pytest.raises(ValueError):
        EarthShadowOptions(radius_m=radius)


def test_options_are_keyword_only_and_provenance_is_text():
    with pytest.raises(TypeError):
        EarthShadowOptions(6378137.0)
    with pytest.raises(ValueError):
        EarthShadowOptions(provenance=1)


def _entry_points(earth_shadow, srp):
    t = np.array([0.0, 60.0])
    kw = dict(earth_shadow=earth_shadow, srp=srp)
    return [
        lambda: dyn.propagate_state(t, TERM, *BASE, **kw),
        lambda: dyn.propagate_state(t, TERM, *BASE, shadow_events=True, method="DOP853", **kw),
        lambda: dyn.propagate_augmented_state(t, G._aug0(TERM), *BASE, **kw),
        lambda: dyn.propagate_augmented_state(t, G._aug0(TERM), *BASE, shadow_events=True, method="DOP853", **kw),
        lambda: dyn.propagate_state_with_k_sensitivity(t, TERM, *BASE, **kw),
        lambda: dyn.shadow_contact_times(t, TERM, *BASE, **kw),
        lambda: dyn.propagate_truth_with_ephemeris(t, TERM, *BASE[:3], GEO.SyntheticEclipseEphemeris(), **kw),
    ]


@pytest.mark.parametrize("srp", [None, SRPOptions(enabled=False)])
def test_earth_shadow_without_active_srp_raises_everywhere(srp):
    for call in _entry_points(ES, srp):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("bad", [True, "WGS84", {"radius_m": 6378137.0}])
def test_an_earth_shadow_that_is_not_options_raises_before_integrating(bad):
    for call in _entry_points(bad, NOS):
        with pytest.raises(TypeError, match="EarthShadowOptions"):
            call()


def test_a_spacecraft_inside_the_earth_sphere_raises():
    r_e = np.array([384_400e3, 0.0, 0.0])
    with pytest.raises(ValueError, match="inside the Earth shadow sphere"):
        es.earth_illumination(r_e + [1000.0, 0.0, 0.0], 2 * r_e, r_e, ES)
