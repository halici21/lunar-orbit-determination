"""F3 WP3.2 S2: the opt-in lunar albedo/IR radiation pressure in production (gates L_G1(a), L_G2-L_G8).

Predeclaration: studies/phase17/configs/wp3_2_s2_lunar_radiation_predeclaration.json (da5a4a0).

Oracles never use the code under test:
- the C6 study kernel, the term WP3.1 measured (L_G2);
- closed forms, a 1-D quadrature and an independent adaptive cap integral in a Sun-aligned frame (L_G3);
- complex-step derivatives of a complex-safe re-implementation of the discrete rule (L_G4);
- a fixed-step RK4 re-propagation of the PRODUCTION LRP-on RHS, with the identical step sequence for every
  perturbed run (L_G5, the F1 pattern).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad, quad_vec, solve_ivp

import lunar_od.dynamics as dyn
import lunar_od.lunar_radiation as lr
from lunar_od.constants import AU_M, J2_MOON_UNNORMALIZED, R_MOON_M, SPEED_OF_LIGHT_M_S
from lunar_od.lunar_radiation import LunarRadiationOptions
from lunar_od.lunar_tide import LunarTideOptions
from lunar_od.srp import SRPOptions, solar_pressure_at
from tests import test_wp3_2_lunar_tide as G

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "studies" / "phase17" / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "studies" / "phase17" / "scripts"))

LRP = LunarRadiationOptions()
TIDE = LunarTideOptions(static_field_convention="TIDE_FREE")
EPS = np.finfo(float).eps
SB = lr.STEFAN_BOLTZMANN_W_M2_K4
K = 0.05                                     # 5x the physical prior (predeclaration L_G5)
NOS = SRPOptions(k_srp_m2_per_kg=K, shadow_model="NO_SHADOW")
PEN = SRPOptions(k_srp_m2_per_kg=K)
earth, sun, BASE, X0, T6H, TIGHT = G.earth, G.sun, G.BASE, G.X0, G.T6H, G.TIGHT


class _Ephemeris:
    earth_position = staticmethod(earth)
    sun_position = staticmethod(sun)


# ----------------------------------------------------------------------
# L_G1(a): off is the identity
# ----------------------------------------------------------------------
@pytest.mark.parametrize("size", [6, 42, 48])
def test_lunar_radiation_wrapper_returns_the_rhs_itself_when_off(size):
    def rhs(_t, y):
        return y
    for srp in (None, NOS, SRPOptions(enabled=False)):
        assert dyn._with_lunar_radiation_acceleration(rhs, sun, srp, None, state_size=size) is rhs


# ----------------------------------------------------------------------
# samples (seed 20261110)
# ----------------------------------------------------------------------
def make_samples(n=200, seed=20261110):
    rng = np.random.default_rng(seed)
    r = G._unit(rng, n) * rng.uniform(R_MOON_M + 20e3, R_MOON_M + 3000e3, size=n)[:, None]
    s = G._unit(rng, n) * rng.uniform(0.983 * AU_M, 1.017 * AU_M, size=n)[:, None]
    return r, s


@pytest.fixture(scope="module")
def samples():
    return make_samples()


# ----------------------------------------------------------------------
# L_G2: production equals the C6 study kernel (the WP3.1 term divided by K)
# ----------------------------------------------------------------------
def _production(r, s, gradient=False, python=False, o=LRP):
    kernel = lr._kernel.py_func if python and hasattr(lr._kernel, "py_func") else lr._kernel
    d = float(np.linalg.norm(s))
    x, wx, cphi, sphi = lr._nodes(o.n_theta, o.n_phi)
    return kernel(np.asarray(r, float), s / d, solar_pressure_at(d) * SPEED_OF_LIGHT_M_S, o.albedo, o.emissivity,
                  o.night_temperature_k, o.radius_m, x, wx, cphi, sphi, gradient)


def c6_parity(samples, n_fallback=40):
    import phase_c6_lunar_radiation as C6
    import wp3_1_force_terms as W
    r, s = samples
    worst = worst_py = 0.0
    for i in range(r.shape[0]):
        want = C6.acceleration_kernel(r[i], s[i])[0]
        got = lr.lunar_radiation_acceleration_per_k(r[i], s[i], LRP)
        worst = max(worst, float(np.linalg.norm(got - want) / np.linalg.norm(want)))
        if i < n_fallback:          # the Python fallback is slow
            fallback = _production(r[i], s[i], python=True)[0]
            worst_py = max(worst_py, float(np.linalg.norm(fallback - want) / np.linalg.norm(want)))
        np.testing.assert_array_equal(W.lrp_acceleration(r[i], s[i]), W.K_LRP_M2_PER_KG * want)
    return worst, worst_py


def test_production_equals_the_c6_study_kernel_and_the_wp3_1_term(samples):
    worst, worst_py = c6_parity(samples)
    assert worst <= 1e-12, worst
    assert worst_py <= 1e-12, worst_py


def test_defaults_are_the_c6_model():
    import phase_c6_lunar_radiation as C6
    m = C6.LunarRadiation()
    assert (LRP.albedo, LRP.emissivity, LRP.night_temperature_k, LRP.n_theta, LRP.n_phi, LRP.radius_m) == (
        m.albedo, m.emissivity, m.t_night_k, m.n_theta, m.n_phi, m.radius_m)
    assert SB == C6.STEFAN_BOLTZMANN
    assert "K_LRP = K_SRP" in LRP.provenance


# ----------------------------------------------------------------------
# L_G3: accuracy against closed forms and an independent adaptive integral
# ----------------------------------------------------------------------
ALTITUDES_AB = (20e3, 100e3, 1000e3, 3000e3)


def _flux(r_sun):
    return solar_pressure_at(float(np.linalg.norm(r_sun))) * SPEED_OF_LIGHT_M_S


def _radiance(cos_psi, flux, o=LRP):
    """C6's radiance, written independently (scalar)."""
    cp = max(cos_psi, 0.0)
    t_ss = ((1.0 - o.albedo) * flux / (o.emissivity * SB)) ** 0.25
    temp = max(t_ss * cp ** 0.25, o.night_temperature_k)
    return (o.albedo * flux * cp + o.emissivity * SB * temp ** 4) / math.pi


def _subsolar_radial_1d(r, flux, o=LRP):
    """Spacecraft on the Sun direction: the radial component as a 1-D integral over the polar angle."""
    big_r = o.radius_m

    def f(theta):
        dist = math.sqrt(r * r + big_r * big_r - 2.0 * r * big_r * math.cos(theta))
        cos_e = (r * math.cos(theta) - big_r) / dist
        u_z = (r - big_r * math.cos(theta)) / dist
        return (_radiance(math.cos(theta), flux, o) * cos_e * u_z * 2.0 * math.pi * big_r ** 2 * math.sin(theta)
                / dist ** 2)
    val, _ = quad(f, 0.0, math.acos(big_r / r), epsabs=0.0, epsrel=1e-13, limit=200)
    return val / SPEED_OF_LIGHT_M_S


def _adaptive_reference(r_sc, r_sun, o=LRP, epsrel=1e-11):
    """Independent cap integral: Sun-aligned azimuth, adaptive in both angles, broken at the radiance kinks."""
    r_sc = np.asarray(r_sc, float)
    r = float(np.linalg.norm(r_sc))
    z = r_sc / r
    s = np.asarray(r_sun, float) / np.linalg.norm(r_sun)
    flux = _flux(r_sun)
    sz = float(s @ z)
    perp = s - sz * z
    sx = float(np.linalg.norm(perp))
    if sx > 1e-12:
        xh = perp / sx
    else:
        xh = np.cross(z, [1.0, 0.0, 0.0] if abs(z[0]) < 0.9 else [0.0, 1.0, 0.0])
        xh /= np.linalg.norm(xh)
        sx = 0.0
    big_r = o.radius_m
    t_max = math.acos(big_r / r)
    t_ss = ((1.0 - o.albedo) * flux / (o.emissivity * SB)) ** 0.25
    kinks = (0.0, (o.night_temperature_k / t_ss) ** 4)          # cos(psi) at both radiance kinks

    def f(theta, phi):
        st, ct, sp, cp = math.sin(theta), math.cos(theta), math.sin(phi), math.cos(phi)
        n = (st * cp, st * sp, ct)
        d = (-big_r * n[0], -big_r * n[1], r - big_r * n[2])
        dist = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
        cos_e = max((n[0] * d[0] + n[1] * d[1] + n[2] * d[2]) / dist, 0.0)
        q = _radiance(st * cp * sx + ct * sz, flux, o) * cos_e * big_r ** 2 * st / dist ** 3
        return np.array([q * d[0], q * d[2]])                   # the y part cancels by the mirror symmetry

    def inner(theta):
        pts = []
        den = math.sin(theta) * sx
        for c in kinks:
            if den > 0.0:
                arg = (c - math.cos(theta) * sz) / den
                if -1.0 < arg < 1.0:
                    pts.append(math.acos(arg))
        val = quad_vec(lambda ph: f(theta, ph), 0.0, math.pi, epsrel=epsrel, norm="max",
                       points=sorted(pts) or None)[0]
        return 2.0 * val

    beta = math.acos(max(-1.0, min(1.0, sz)))
    tangent = sorted({t for c in kinks for g in (math.acos(c),)
                      for t in (abs(beta - g), beta + g, 2.0 * math.pi - beta - g) if 0.0 < t < t_max})
    out = quad_vec(inner, 0.0, t_max, epsrel=epsrel, norm="max", points=tangent or None)[0]
    return (out[0] * xh + out[1] * z) / SPEED_OF_LIGHT_M_S


def _axis_case(altitude, sunward):
    s = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8]) * AU_M
    r = (1.0 if sunward else -1.0) * s / np.linalg.norm(s) * (R_MOON_M + altitude)
    return r, s


def closed_form_errors(altitude):
    r, s = _axis_case(altitude, sunward=False)
    rn = np.linalg.norm(r)
    want = LRP.emissivity * SB * LRP.night_temperature_k ** 4 * (R_MOON_M / rn) ** 2 / SPEED_OF_LIGHT_M_S * r / rn
    night = float(np.linalg.norm(lr.lunar_radiation_acceleration_per_k(r, s, LRP) - want) / np.linalg.norm(want))
    night_ref = float(np.linalg.norm(_adaptive_reference(r, s) - want) / np.linalg.norm(want))
    r, s = _axis_case(altitude, sunward=True)
    rn = np.linalg.norm(r)
    radial = _subsolar_radial_1d(rn, _flux(s))
    got = lr.lunar_radiation_acceleration_per_k(r, s, LRP)
    sub = float(abs(got @ r / rn - radial) / abs(radial))
    sub_transverse = float(np.linalg.norm(got - (got @ r / rn) * r / rn) / np.linalg.norm(got))
    sub_ref = float(np.linalg.norm(_adaptive_reference(r, s) - radial * r / rn) / abs(radial))
    return dict(night=night, night_reference=night_ref, subsolar_radial=sub, subsolar_transverse=sub_transverse,
                subsolar_reference=sub_ref)


@pytest.mark.parametrize("altitude", ALTITUDES_AB)
def test_axis_cases_match_the_closed_form_and_the_1d_integral(altitude):
    e = closed_form_errors(altitude)
    assert e["night"] <= 1e-10, e
    assert e["subsolar_radial"] <= 1e-10, e
    assert e["subsolar_transverse"] <= 1e-12, e
    assert e["night_reference"] <= 1e-10, e            # precondition on the adaptive reference
    assert e["subsolar_reference"] <= 1e-10, e


def make_accuracy_grid(seed=20261110):
    rng = np.random.default_rng(seed)
    cases = []
    for altitude in (20e3, 100e3, 1000e3):
        for beta_deg in (0.0, 45.0, 80.0, 90.0, 100.0, 135.0, 180.0):
            z = G._unit(rng, 1)[0]
            a = np.cross(z, [1.0, 0.0, 0.0] if abs(z[0]) < 0.9 else [0.0, 1.0, 0.0])
            a /= np.linalg.norm(a)
            b = np.cross(z, a)
            alpha = rng.uniform(0.0, 2.0 * math.pi)
            beta = math.radians(beta_deg)
            s_hat = math.cos(beta) * z + math.sin(beta) * (math.cos(alpha) * a + math.sin(alpha) * b)
            cases.append((altitude, beta_deg, z * (R_MOON_M + altitude), s_hat * AU_M))
    return cases


def accuracy_errors():
    out = []
    for altitude, beta_deg, r, s in make_accuracy_grid():
        ref = _adaptive_reference(r, s)
        got = lr.lunar_radiation_acceleration_per_k(r, s, LRP)
        out.append((altitude, beta_deg, float(np.linalg.norm(got - ref) / np.linalg.norm(ref))))
    return out


def test_quadrature_error_stays_below_the_materiality_requirement():
    errors = accuracy_errors()
    worst = max(e for _, _, e in errors)
    assert worst <= 3e-4, errors


# ----------------------------------------------------------------------
# L_G4: forward-mode gradient against complex step of an independent re-implementation
# ----------------------------------------------------------------------
def _complex_rule(r_sc, r_sun, o=LRP):
    """C6's discrete rule, complex-safe (branches chosen on real parts), written independently."""
    x, wx = np.polynomial.legendre.leggauss(o.n_theta)
    x, wx = 0.5 * (x + 1.0), 0.5 * wx
    phi = 2.0 * np.pi * (np.arange(o.n_phi) + 0.5) / o.n_phi
    r = np.sqrt(np.sum(r_sc * r_sc))
    z = r_sc / r
    seed = np.array([1.0, 0.0, 0.0]) if abs(z[0].real) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(z, seed)
    e1 = e1 / np.sqrt(np.sum(e1 * e1))
    e2 = np.cross(z, e1)
    big_r = o.radius_m
    theta = np.arccos(big_r / r) * x ** 2
    dtheta = np.arccos(big_r / r) * 2.0 * x * wx
    nrm = (np.cos(theta)[:, None, None] * z
           + np.sin(theta)[:, None, None] * (np.cos(phi)[None, :, None] * e1 + np.sin(phi)[None, :, None] * e2))
    d = r_sc - big_r * nrm
    dist = np.sqrt(np.sum(d * d, axis=-1))
    u = d / dist[..., None]
    cos_e = np.sum(nrm * u, axis=-1)
    cos_e = np.where(cos_e.real < 0.0, 0.0, cos_e)
    w = (big_r ** 2 * np.sin(theta) * dtheta * (2.0 * np.pi / o.n_phi))[:, None] / dist ** 2
    s_hat = np.asarray(r_sun, float) / np.linalg.norm(r_sun)
    flux = _flux(r_sun)
    cos_psi = np.sum(nrm * s_hat, axis=-1)
    lit = cos_psi.real > 0.0
    cos_psi = np.where(lit, cos_psi, 0.0)
    t_ss = ((1.0 - o.albedo) * flux / (o.emissivity * SB)) ** 0.25
    t_day = np.where(lit, t_ss * np.where(lit, cos_psi, 1.0) ** 0.25, 0.0)
    temp = np.where(t_day.real > o.night_temperature_k, t_day, o.night_temperature_k)
    lum = (o.albedo * flux * cos_psi + o.emissivity * SB * temp ** 4) / np.pi
    return np.sum((lum * cos_e * w)[..., None] * u, axis=(0, 1)) / SPEED_OF_LIGHT_M_S


def _complex_step_jacobian(r_sc, r_sun, h=1e-30):
    cols = []
    for k in range(3):
        rc = np.asarray(r_sc, complex).copy()
        rc[k] += 1j * h
        cols.append(_complex_rule(rc, r_sun).imag / h)
    return np.stack(cols, axis=1)


def gradient_errors(samples, n_fallback=40):
    import phase_c6_lunar_radiation as C6
    r, s = samples
    worst_cs = worst_ref = worst_py = 0.0
    for i in range(r.shape[0]):
        a, g = lr.lunar_radiation_acceleration_and_gradient_per_k(r[i], s[i], LRP)
        cs = _complex_step_jacobian(r[i], s[i])
        worst_cs = max(worst_cs, float(np.linalg.norm(g - cs) / np.linalg.norm(cs)))
        ref = _complex_rule(np.asarray(r[i], complex), s[i]).real
        c6 = C6.acceleration_kernel(r[i], s[i])[0]
        worst_ref = max(worst_ref, float(np.linalg.norm(ref - c6) / np.linalg.norm(c6)))
        np.testing.assert_array_equal(a, lr.lunar_radiation_acceleration_per_k(r[i], s[i], LRP))
        if hasattr(lr._kernel, "py_func") and i < n_fallback:
            g_py = _production(r[i], s[i], gradient=True, python=True)[1]
            worst_py = max(worst_py, float(np.linalg.norm(g_py - g) / np.linalg.norm(g)))
    return worst_cs, worst_ref, worst_py


def test_gradient_is_the_exact_derivative_of_the_discrete_rule(samples):
    worst_cs, worst_ref, worst_py = gradient_errors(samples)
    assert worst_cs <= 1e-10, worst_cs
    assert worst_ref <= 1e-12, worst_ref
    assert worst_py <= 1e-12, worst_py


# ----------------------------------------------------------------------
# L_G5: variational equations against a fixed-step RK4 re-propagation
# ----------------------------------------------------------------------
H_POS, H_VEL = 3.0, 3e-3


def _production_rhs6(srp=NOS, lrp=LRP):
    rhs, _ = dyn._build_state_rhs(T6H, *BASE, J2_MOON_UNNORMALIZED, 0.0, "indirect", None, None, None, srp,
                                  lunar_radiation=lrp)
    return rhs


def make_rk4_columns():
    rhs = _production_rhs6()
    cols = []
    for j in range(6):
        h = H_POS if j < 3 else H_VEL
        e = np.zeros(6)
        e[j] = h
        cols.append((G._rk4(rhs, X0 + e, T6H) - G._rk4(rhs, X0 - e, T6H)) / (2 * h))
    return np.stack(cols, axis=-1)


@pytest.fixture(scope="module")
def rk4_columns():
    return make_rk4_columns()


def no_lrp_gradient_replica():
    """Analysis-only negative control: LRP in the trajectory, none in the variational block."""
    def base(t, y):
        return np.array(dyn.ode_fun_v3(t, y, *BASE, J2_MOON_UNNORMALIZED, 0.0, "indirect"), dtype=float)
    with_srp = dyn._with_srp_acceleration(base, sun, NOS, state_size=42)

    def rhs(t, y):
        out = with_srp(t, y)
        out[3:6] += K * lr.lunar_radiation_acceleration_per_k(y[:3], sun(t), LRP)
        return out
    sol = solve_ivp(rhs, (T6H[0], T6H[-1]), G._aug0(X0), method="DOP853", t_eval=T6H, rtol=1e-13, atol=1e-14)
    assert sol.success
    return sol.y.T


def gated_stm():
    return dyn.propagate_augmented_state(T6H, G._aug0(X0), *BASE, j2_moon=J2_MOON_UNNORMALIZED, srp=NOS,
                                         lunar_radiation=LRP, **TIGHT)


def test_lrp_stm_matches_repropagation_and_omitting_its_gradient_does_not(rk4_columns):
    gated = G._stm_err(rk4_columns, G._phi(gated_stm()))
    control = G._stm_err(rk4_columns, G._phi(no_lrp_gradient_replica()))
    assert gated <= 1e-7, gated
    assert control >= 10 * gated, (control, gated)


def k_sensitivity_error():
    dk = 5e-3
    plus = G._rk4(_production_rhs6(SRPOptions(k_srp_m2_per_kg=K + dk, shadow_model="NO_SHADOW")), X0, T6H)
    minus = G._rk4(_production_rhs6(SRPOptions(k_srp_m2_per_kg=K - dk, shadow_model="NO_SHADOW")), X0, T6H)
    fd = (plus - minus) / (2 * dk)
    s_k = dyn.propagate_state_with_k_sensitivity(T6H, X0, *BASE, srp=NOS, j2_moon=J2_MOON_UNNORMALIZED,
                                                 lunar_radiation=LRP, **TIGHT)[:, 42:48]
    return float(np.max(np.abs(fd - s_k)) / np.max(np.abs(s_k)))


def test_lrp_k_sensitivity_matches_k_repropagation():
    err = k_sensitivity_error()
    assert err <= 1e-6, err


@pytest.mark.parametrize("size", [6, 42, 48])
def test_wrapper_adds_exactly_the_lrp_terms_and_nothing_else(size):
    rng = np.random.default_rng(11 + size)

    def base(t, y):
        return np.sin(np.arange(size) + t) * (1.0 + np.abs(y)) * 1e-3 + 1.45 * (np.arange(size) % 6 >= 3)
    wrapped = dyn._with_lunar_radiation_acceleration(base, sun, NOS, LRP, state_size=size)
    for _ in range(10):
        t = float(rng.uniform(0, G.DAY_S))
        y = rng.normal(size=size)
        y[:3] = G._unit(rng, 1)[0] * rng.uniform(R_MOON_M + 20e3, R_MOON_M + 500e3)
        y[3:6] = rng.normal(size=3) * 1.6e3
        off, on = base(t, y), wrapped(t, y)
        g, grad = lr.lunar_radiation_acceleration_and_gradient_per_k(y[:3], sun(t), LRP)
        diff = on - off
        assert not diff[0:3].any()
        np.testing.assert_allclose(diff[3:6], K * g, rtol=0, atol=4 * EPS * np.abs(off[3:6]).max())
        if size >= 42:
            phi = y[6:42].reshape((6, 6), order="F")
            dphi = diff[6:42].reshape((6, 6), order="F")
            assert not dphi[0:3, :].any()
            np.testing.assert_allclose(dphi[3:6, :], K * grad @ phi[0:3, :], rtol=0,
                                       atol=4 * EPS * np.abs(off[6:42]).max())
        if size == 48:
            assert not diff[42:45].any()
            np.testing.assert_allclose(diff[45:48], K * grad @ y[42:45] + g, rtol=0,
                                       atol=4 * EPS * np.abs(off[42:48]).max())


def test_state_only_propagation_never_evaluates_the_lrp_gradient(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("evaluated the LRP gradient")
    monkeypatch.setattr(lr, "lunar_radiation_acceleration_and_gradient_per_k", boom)
    monkeypatch.setattr(lr, "lunar_radiation_gradient_per_k", boom)
    t = np.array([0.0, 60.0])
    dyn.propagate_state(t, X0, *BASE, srp=NOS, lunar_radiation=LRP)
    with pytest.raises(AssertionError, match="LRP gradient"):         # the patch is live
        dyn.propagate_augmented_state(t, G._aug0(X0), *BASE, srp=NOS, lunar_radiation=LRP, method="DOP853")


# ----------------------------------------------------------------------
# L_G6: 6-, 42- and 48-state paths agree, with and without harmonics, with and without the tide
# ----------------------------------------------------------------------
def path_differences(harmonics, tide):
    kw = dict(srp=NOS, lunar_radiation=LRP, **TIGHT)
    if tide:
        kw["lunar_tide"] = TIDE
    if harmonics:
        kw.update(harmonic_model=G._zonal_model(), harmonic_rotation=G._rot_x(0.03))
        kw_var = dict(kw, harmonic_stm_opt_in=True)
    else:
        kw.update(j2_moon=J2_MOON_UNNORMALIZED)
        kw_var = kw
    p6 = dyn.propagate_state(T6H, X0, *BASE, **kw)
    p42 = dyn.propagate_augmented_state(T6H, G._aug0(X0), *BASE, **kw_var)
    p48 = dyn.propagate_state_with_k_sensitivity(T6H, X0, *BASE, **kw_var)
    runs = {"42": p42, "48": p48}
    if not harmonics:
        runs["42_reference"] = G._reference(
            lambda: dyn.propagate_augmented_state(T6H, G._aug0(X0), *BASE, **kw_var))
    scale = np.max(np.abs(p48[:, 6:42]))
    p6_free = dyn.propagate_state(T6H, X0, *BASE, **dict(kw, lunar_radiation=None))
    return ({name: float(np.max(np.abs(run[:, 0:3] - p6[:, 0:3]))) for name, run in runs.items()},
            {name: float(np.max(np.abs(run[:, 6:42] - p48[:, 6:42])) / scale) for name, run in runs.items()},
            float(np.max(np.abs(p6_free[:, 0:3] - p6[:, 0:3]))))


@pytest.mark.parametrize("tide", [False, True])
@pytest.mark.parametrize("harmonics", [False, True])
def test_six_42_and_48_state_paths_carry_the_same_lrp(harmonics, tide):
    pos, phi, effect = path_differences(harmonics, tide)
    assert max(pos.values()) <= 1e-5, pos
    assert max(phi.values()) <= 1e-8, phi
    assert effect > 0.1, effect                    # LRP is really in all of them


# ----------------------------------------------------------------------
# L_G7: event-segmented (D2) propagation with LRP
# ----------------------------------------------------------------------
D2 = dict(srp=PEN, method="DOP853", shadow_events=True, rtol=1e-11, atol=1e-12, j2_moon=J2_MOON_UNNORMALIZED)


def shadow_event_values():
    x0 = G.X0_ECLIPSED
    for call in (lambda: dyn.propagate_state(G.T1D, x0, *BASE, lunar_radiation=LRP, **D2),
                 lambda: dyn.propagate_augmented_state(G.T1D, G._aug0(x0), *BASE, lunar_radiation=LRP, **D2),
                 lambda: dyn.propagate_state_with_k_sensitivity(G.T1D, x0, *BASE, lunar_radiation=LRP, **D2)):
        assert np.all(np.isfinite(call()))
    pre = dict(srp=PEN, rtol=1e-11, atol=1e-12, j2_moon=J2_MOON_UNNORMALIZED)
    cuts_lrp = np.array(dyn.shadow_contact_times(G.T1D, x0, *BASE, lunar_radiation=LRP, **pre))
    cuts_free = np.array(dyn.shadow_contact_times(G.T1D, x0, *BASE, **pre))
    rhs, sun_lookup = dyn._build_state_rhs(G.T1D, *BASE, J2_MOON_UNNORMALIZED, 0.0, "indirect", None, None, None,
                                           PEN, lunar_radiation=LRP)
    message = None
    try:
        dyn._integrate_segmented(G.T1D, x0, rhs, 1e-11, 1e-12, "DOP853", list(cuts_free), "control",
                                 contact_check=dyn._shadow_contact_check(sun_lookup, PEN))
    except RuntimeError as exc:
        message = str(exc)
    return cuts_lrp, cuts_free, message


def test_shadow_events_with_lrp_are_consistent_and_lrp_reaches_the_pre_pass():
    cuts_lrp, cuts_free, message = shadow_event_values()
    assert cuts_lrp.size == cuts_free.size > 20
    assert np.max(np.abs(cuts_lrp - cuts_free)) > dyn._CONTACT_TIME_TOL_S
    assert message is not None and "Shadow contact inconsistent" in message


# ----------------------------------------------------------------------
# L_G8: fail closed
# ----------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    dict(albedo=-0.01), dict(albedo=1.0), dict(albedo=float("nan")), dict(albedo=True),
    dict(emissivity=0.0), dict(emissivity=1.01), dict(emissivity=float("inf")),
    dict(night_temperature_k=-1.0), dict(night_temperature_k=float("nan")),
    dict(n_theta=3), dict(n_theta=60.0), dict(n_theta=True), dict(n_phi=7), dict(n_phi="72"),
    dict(radius_m=0.0), dict(radius_m=-R_MOON_M), dict(radius_m=float("inf")),
])
def test_invalid_lunar_radiation_options_raise(kwargs):
    with pytest.raises(ValueError):
        LunarRadiationOptions(**kwargs)


def test_options_are_keyword_only():
    with pytest.raises(TypeError):
        LunarRadiationOptions(0.12)


def _entry_points(lunar_radiation, srp):
    t = np.array([0.0, 60.0])
    kw = dict(lunar_radiation=lunar_radiation, srp=srp)
    return [
        lambda: dyn.propagate_state(t, X0, *BASE, **kw),
        lambda: dyn.propagate_augmented_state(t, G._aug0(X0), *BASE, **kw),
        lambda: dyn.propagate_state_with_k_sensitivity(t, X0, *BASE, **kw),
        lambda: dyn.shadow_contact_times(t, X0, *BASE, **kw),
        lambda: dyn.propagate_truth_with_ephemeris(t, X0, *BASE[:3], _Ephemeris(), **kw),
    ]


@pytest.mark.parametrize("srp", [None, SRPOptions(enabled=False)])
def test_lunar_radiation_without_active_srp_raises_everywhere(srp):
    for call in _entry_points(LRP, srp):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("bad", [True, "C6", {"albedo": 0.12}])
def test_a_lunar_radiation_that_is_not_options_raises_before_integrating(bad):
    for call in _entry_points(bad, NOS):
        with pytest.raises(TypeError, match="LunarRadiationOptions"):
            call()


def test_a_spacecraft_inside_the_sphere_raises():
    with pytest.raises(ValueError, match="above the lunar radiation sphere"):
        lr.lunar_radiation_acceleration_per_k([R_MOON_M - 1.0, 0.0, 0.0], [AU_M, 0.0, 0.0], LRP)
