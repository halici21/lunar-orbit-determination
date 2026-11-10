"""F3 WP3.2 S1: the opt-in lunar solid tide in production (gates T_G1(a), T_G2-T_G8).

Predeclaration: studies/phase17/configs/wp3_2_s1_lunar_tide_predeclaration.json (bdafb53).

Oracles never use the code under test: the IERS 2010 eq. 6.6 coefficient form evaluated by the
production spherical-harmonic engine (T_G2), the WP3.1 study term (T_G3), complex-step derivatives
of the production acceleration (T_G4), and a fixed-step RK4 re-propagation of the PRODUCTION
tide-on RHS, identical step sequence for every perturbed run (T_G5, the F1 pattern).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import lunar_od.dynamics as dyn
import lunar_od.lunar_tide as lt
from lunar_od.constants import (
    AU_M, J2_MOON_UNNORMALIZED, MU_EARTH_M3S2, MU_MOON_M3S2, MU_SUN_M3S2, R_MOON_M,
)
from lunar_od.gravity_harmonics import (
    SphericalHarmonicGravityModel,
    cbar_n0_from_unnormalized_jn,
    spherical_harmonic_acceleration,
    spherical_harmonic_gravity_gradient,
)
from lunar_od.lunar_tide import LunarTideOptions
from lunar_od.srp import SRPOptions

ROOT = Path(__file__).resolve().parents[1]
TIDE = LunarTideOptions(static_field_convention="TIDE_FREE")
EPS = np.finfo(float).eps

# ----------------------------------------------------------------------
# synthetic geometry (no SPICE): Earth and Sun on circles, 23.44 deg tilt
# ----------------------------------------------------------------------
DAY_S = 86400.0
TILT = math.radians(23.44)
W_EARTH = 2.0 * math.pi / (27.321661 * DAY_S)
W_SUN = 2.0 * math.pi / (365.25636 * DAY_S)


def _circle(radius, omega, phase, t):
    th = omega * float(t) + phase
    return radius * np.array([math.cos(th), math.sin(th) * math.cos(TILT), math.sin(th) * math.sin(TILT)])


def earth(t):
    return _circle(384_400e3, W_EARTH, 0.0, t)


def sun(t):
    return _circle(AU_M, W_SUN, 1.0, t) + earth(t)


def _orbit(sun_in_plane: bool) -> np.ndarray:
    """100 km circular polar orbit; with the Sun in its plane it is eclipsed every revolution."""
    a = R_MOON_M + 100e3
    z = np.array([0.0, 0.0, 1.0])
    s = sun(0.0) / np.linalg.norm(sun(0.0))
    h = np.cross(s, z) if sun_in_plane else np.cross(np.cross(s, z), z)
    h /= np.linalg.norm(h)
    return np.concatenate([a * z, math.sqrt(MU_MOON_M3S2 / a) * np.cross(h, z)])


X0 = _orbit(sun_in_plane=False)
X0_ECLIPSED = _orbit(sun_in_plane=True)
T6H = np.arange(0.0, 6 * 3600.0 + 1.0, 300.0)
BASE = (MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2, earth, sun)
TIGHT = dict(method="DOP853", rtol=1e-13, atol=1e-14)
NOSHADOW = SRPOptions(k_srp_m2_per_kg=0.01, shadow_model="NO_SHADOW")


def _aug0(x0):
    return np.concatenate([x0, np.eye(6).reshape(-1, order="F")])


def _phi(hist):
    return np.array([row.reshape((6, 6), order="F") for row in hist[:, 6:42]])


def _reference(call):
    saved = dyn._FAST_DYNAMICS
    dyn._FAST_DYNAMICS = False
    try:
        return call()
    finally:
        dyn._FAST_DYNAMICS = saved


# ----------------------------------------------------------------------
# T_G1(a): off is the identity
# ----------------------------------------------------------------------
@pytest.mark.parametrize("size", [6, 42, 48])
def test_tide_wrapper_returns_the_rhs_itself_when_off(size):
    def rhs(_t, y):
        return y
    assert dyn._with_tide_acceleration(rhs, earth, sun, MU_EARTH_M3S2, MU_SUN_M3S2, None, state_size=size) is rhs
    assert dyn._share_tide_lookups(earth, sun, None) == (earth, sun)
    assert dyn._share_tide_lookups(earth, sun, None)[0] is earth


# ----------------------------------------------------------------------
# samples for T_G2-T_G4 (seed 20261108)
# ----------------------------------------------------------------------
def _unit(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1)[:, None]


def _rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def make_samples():
    rng = np.random.default_rng(20261108)
    n = 200
    r = _unit(rng, n) * rng.uniform(R_MOON_M + 20e3, R_MOON_M + 3000e3, n)[:, None]
    d_e = _unit(rng, n) * rng.uniform(356_000e3, 407_000e3, n)[:, None]
    d_s = _unit(rng, n) * rng.uniform(0.983 * AU_M, 1.017 * AU_M, n)[:, None]
    rot = [_rotation(rng) for _ in range(n)]
    return r, d_e, d_s, rot


@pytest.fixture(scope="module")
def samples():
    return make_samples()


def _terms():
    """(body index, GM, degree, k) of the default options, in their summation order."""
    gms = {"EARTH": MU_EARTH_M3S2, "SUN": MU_SUN_M3S2}
    return [(b, gms[b], n, k) for b in TIDE.raising_bodies for n, k in TIDE.love_numbers]


# ----------------------------------------------------------------------
# T_G2: IERS 2010 eq. 6.6 through the production spherical-harmonic engine
# ----------------------------------------------------------------------
def _pbar(n, m, x):
    """Fully normalised associated Legendre function (geodesy convention, no Condon-Shortley phase)."""
    u = math.sqrt(max(0.0, 1.0 - x * x))
    p = {(2, 0): 0.5 * (3 * x * x - 1), (2, 1): 3 * x * u, (2, 2): 3 * u * u,
         (3, 0): 0.5 * (5 * x ** 3 - 3 * x), (3, 1): 1.5 * (5 * x * x - 1) * u,
         (3, 2): 15 * x * u * u, (3, 3): 15 * u ** 3}[(n, m)]
    norm = math.sqrt((1 if m == 0 else 2) * (2 * n + 1) * math.factorial(n - m) / math.factorial(n + m))
    return norm * p


def _iers_model(d_bf, gm_body, n, k, radius):
    dn = float(np.linalg.norm(d_bf))
    lat = math.asin(d_bf[2] / dn)
    lon = math.atan2(d_bf[1], d_bf[0])
    cbar = np.zeros((4, 4))
    sbar = np.zeros((4, 4))
    scale = k / (2 * n + 1) * (gm_body / MU_MOON_M3S2) * (radius / dn) ** (n + 1)
    for m in range(n + 1):
        cbar[n, m] = scale * _pbar(n, m, math.sin(lat)) * math.cos(m * lon)
        sbar[n, m] = scale * _pbar(n, m, math.sin(lat)) * math.sin(m * lon)
    return SphericalHarmonicGravityModel(mu_m3_s2=MU_MOON_M3S2, r_ref_m=radius, cbar=cbar, sbar=sbar,
                                         nmax=3, mmax=3, frame="ORACLE")


def test_closed_form_matches_iers_coefficients_through_the_production_harmonic_engine(samples):
    r, d_e, d_s, rot = samples
    worst_a = worst_g = worst_sum = 0.0
    for i in range(r.shape[0]):
        d = {"EARTH": d_e[i], "SUN": d_s[i]}
        a_sum, g_sum = np.zeros(3), np.zeros((3, 3))
        scale_max_a = scale_max_g = 0.0
        for body, gm, n, k in _terms():
            model = _iers_model(rot[i] @ d[body], gm, n, k, TIDE.radius_m)
            a_sh = spherical_harmonic_acceleration(r[i], model, rot[i])
            g_sh = spherical_harmonic_gravity_gradient(r[i], model, rot[i])
            rn = np.linalg.norm(r[i])
            amp = k * gm * TIDE.radius_m ** (2 * n + 1) / np.linalg.norm(d[body]) ** (n + 1)
            sa, sg = amp * rn ** -(n + 2), amp * rn ** -(n + 3)
            a_cf = lt.tide_degree_acceleration(r[i], d[body], gm, n, k, TIDE.radius_m)
            g_cf = lt.tide_degree_gravity_gradient(r[i], d[body], gm, n, k, TIDE.radius_m)
            worst_a = max(worst_a, np.linalg.norm(a_cf - a_sh) / sa)
            worst_g = max(worst_g, np.linalg.norm(g_cf - g_sh) / sg)
            a_sum, g_sum = a_sum + a_sh, g_sum + g_sh
            scale_max_a, scale_max_g = max(scale_max_a, sa), max(scale_max_g, sg)
        a_tot = lt.tide_acceleration(r[i], d_e[i], d_s[i], MU_EARTH_M3S2, MU_SUN_M3S2, TIDE)
        g_tot = lt.tide_gravity_gradient(r[i], d_e[i], d_s[i], MU_EARTH_M3S2, MU_SUN_M3S2, TIDE)
        worst_sum = max(worst_sum, np.linalg.norm(a_tot - a_sum) / scale_max_a,
                        np.linalg.norm(g_tot - g_sum) / scale_max_g)
    assert worst_a <= 1e-12, worst_a
    assert worst_g <= 1e-12, worst_g
    assert worst_sum <= 1e-12, worst_sum


# ----------------------------------------------------------------------
# T_G3: the term WP3.1 found material
# ----------------------------------------------------------------------
def test_production_equals_the_wp3_1_study_term(samples):
    scripts = ROOT / "studies" / "phase17" / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import wp3_1_force_terms as W
    r, d_e, d_s, _ = samples
    worst = 0.0
    for i in range(r.shape[0]):
        study = W.tide_acceleration(r[i], d_e[i], d_s[i], MU_EARTH_M3S2, MU_SUN_M3S2)
        prod = lt.tide_acceleration(r[i], d_e[i], d_s[i], MU_EARTH_M3S2, MU_SUN_M3S2, TIDE)
        worst = max(worst, np.linalg.norm(prod - study) / np.linalg.norm(study))
    assert worst <= 1e-13, worst


# ----------------------------------------------------------------------
# T_G4: analytic gradient vs complex step; symmetry and trace
# ----------------------------------------------------------------------
def _complex_step(f, r, h=1e-30):
    jac = np.zeros((3, 3))
    for j in range(3):
        e = np.zeros(3, dtype=complex)
        e[j] = 1j * h
        jac[:, j] = np.imag(f(r.astype(complex) + e)) / h
    return jac


def test_gradient_matches_complex_step_and_is_symmetric_and_traceless(samples):
    r, d_e, d_s, _ = samples
    worst_cs = worst_tr = 0.0
    for i in range(r.shape[0]):
        d = {"EARTH": d_e[i], "SUN": d_s[i]}
        cases = [(lambda x, b=b, gm=gm, n=n, k=k: lt.tide_degree_acceleration(x, d[b], gm, n, k, TIDE.radius_m),
                  lt.tide_degree_gravity_gradient(r[i], d[b], gm, n, k, TIDE.radius_m))
                 for b, gm, n, k in _terms()]
        cases.append((lambda x: lt.tide_acceleration(x, d_e[i], d_s[i], MU_EARTH_M3S2, MU_SUN_M3S2, TIDE),
                      lt.tide_gravity_gradient(r[i], d_e[i], d_s[i], MU_EARTH_M3S2, MU_SUN_M3S2, TIDE)))
        for f, g in cases:
            cs = _complex_step(f, r[i])
            worst_cs = max(worst_cs, np.linalg.norm(g - cs) / np.linalg.norm(cs))
            assert np.array_equal(g, g.T)
            worst_tr = max(worst_tr, abs(np.trace(g)) / np.linalg.norm(g))
    assert worst_cs <= 1e-13, worst_cs
    assert worst_tr <= 1e-12, worst_tr


def test_acceleration_and_gradient_come_from_one_evaluation(samples):
    r, d_e, d_s, _ = samples
    a, g = lt.tide_acceleration_and_gradient(r[0], d_e[0], d_s[0], MU_EARTH_M3S2, MU_SUN_M3S2, TIDE)
    np.testing.assert_array_equal(a, lt.tide_acceleration(r[0], d_e[0], d_s[0], MU_EARTH_M3S2, MU_SUN_M3S2, TIDE))
    np.testing.assert_array_equal(g, lt.tide_gravity_gradient(r[0], d_e[0], d_s[0], MU_EARTH_M3S2, MU_SUN_M3S2,
                                                               TIDE))


# ----------------------------------------------------------------------
# T_G5: variational equations against a fixed-step RK4 re-propagation
# ----------------------------------------------------------------------
def _production_rhs6(srp=None, tide=TIDE):
    rhs, _ = dyn._build_state_rhs(T6H, *BASE, J2_MOON_UNNORMALIZED, 0.0, "indirect", None, None, None, srp,
                                  lunar_tide=tide)
    return rhs


def _rk4(rhs, x0, t_out, dt=5.0):
    y = np.asarray(x0, dtype=float).copy()
    out, tt = [y.copy()], 0.0
    n_sub = int(round((t_out[1] - t_out[0]) / dt))
    for _ in range(1, t_out.size):
        for _ in range(n_sub):
            k1 = rhs(tt, y)
            k2 = rhs(tt + dt / 2, y + dt / 2 * k1)
            k3 = rhs(tt + dt / 2, y + dt / 2 * k2)
            k4 = rhs(tt + dt, y + dt * k3)
            y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            tt += dt
        out.append(y.copy())
    return np.array(out)


def make_rk4_columns():
    rhs = _production_rhs6()
    cols = []
    for j in range(6):
        h = 1.0 if j < 3 else 1e-3
        e = np.zeros(6)
        e[j] = h
        cols.append((_rk4(rhs, X0 + e, T6H) - _rk4(rhs, X0 - e, T6H)) / (2 * h))
    return np.stack(cols, axis=-1)          # (epochs, 6, 6)


@pytest.fixture(scope="module")
def rk4_columns():
    return make_rk4_columns()


def _stm_err(fd, phi):
    return max(float(np.max(np.abs(fd[:, :, j] - phi[:, :, j])) / np.max(np.abs(phi[:, :, j])))
               for j in range(6))


def _no_tide_gradient_replica():
    """Analysis-only negative control: the tide in the trajectory, none in the variational block."""
    def rhs(t, y):
        out = dyn.ode_fun_v3(t, y, *BASE, J2_MOON_UNNORMALIZED, 0.0, "indirect")
        out = np.array(out, dtype=float)
        out[3:6] += lt.tide_acceleration(y[:3], earth(t), sun(t), MU_EARTH_M3S2, MU_SUN_M3S2, TIDE)
        return out
    sol = solve_ivp(rhs, (T6H[0], T6H[-1]), _aug0(X0), method="DOP853", t_eval=T6H, rtol=1e-13, atol=1e-14)
    assert sol.success
    return sol.y.T


def test_tide_stm_matches_repropagation_and_omitting_its_gradient_does_not(rk4_columns):
    hist = dyn.propagate_augmented_state(T6H, _aug0(X0), *BASE, j2_moon=J2_MOON_UNNORMALIZED, lunar_tide=TIDE,
                                         **TIGHT)
    gated = _stm_err(rk4_columns, _phi(hist))
    control = _stm_err(rk4_columns, _phi(_no_tide_gradient_replica()))
    assert gated <= 1e-7, gated
    assert control >= 10 * gated, (control, gated)


def test_tide_k_sensitivity_matches_k_repropagation():
    k, dk = 0.01, 1e-3
    plus = _rk4(_production_rhs6(SRPOptions(k_srp_m2_per_kg=k + dk, shadow_model="NO_SHADOW")), X0, T6H)
    minus = _rk4(_production_rhs6(SRPOptions(k_srp_m2_per_kg=k - dk, shadow_model="NO_SHADOW")), X0, T6H)
    fd = (plus - minus) / (2 * dk)
    s_k = dyn.propagate_state_with_k_sensitivity(T6H, X0, *BASE, srp=NOSHADOW, j2_moon=J2_MOON_UNNORMALIZED,
                                                 lunar_tide=TIDE, **TIGHT)[:, 42:48]
    err = float(np.max(np.abs(fd - s_k)) / np.max(np.abs(s_k)))
    assert err <= 1e-6, err


@pytest.mark.parametrize("size", [6, 42, 48])
def test_wrapper_adds_exactly_the_tide_terms_and_nothing_else(size):
    rng = np.random.default_rng(7 + size)

    def base(t, y):
        return np.sin(np.arange(size) + t) * (1.0 + np.abs(y)) * 1e-3 + 1.45 * (np.arange(size) % 6 >= 3)
    wrapped = dyn._with_tide_acceleration(base, earth, sun, MU_EARTH_M3S2, MU_SUN_M3S2, TIDE, state_size=size)
    for _ in range(20):
        t = float(rng.uniform(0, DAY_S))
        y = rng.normal(size=size)
        y[:3] = _unit(rng, 1)[0] * rng.uniform(R_MOON_M + 20e3, R_MOON_M + 500e3)
        y[3:6] = rng.normal(size=3) * 1.6e3
        off, on = base(t, y), wrapped(t, y)
        a, g = lt.tide_acceleration_and_gradient(y[:3], earth(t), sun(t), MU_EARTH_M3S2, MU_SUN_M3S2, TIDE)
        diff = on - off
        assert not diff[0:3].any()
        np.testing.assert_allclose(diff[3:6], a, rtol=0, atol=4 * EPS * np.abs(off[3:6]).max())
        if size >= 42:
            phi = y[6:42].reshape((6, 6), order="F")
            dphi = diff[6:42].reshape((6, 6), order="F")
            assert not dphi[0:3, :].any()
            np.testing.assert_allclose(dphi[3:6, :], g @ phi[0:3, :], rtol=0,
                                       atol=4 * EPS * np.abs(off[6:42]).max())
        if size == 48:
            assert not diff[42:45].any()
            np.testing.assert_allclose(diff[45:48], g @ y[42:45], rtol=0, atol=4 * EPS * np.abs(off[42:48]).max())


def test_state_only_propagation_never_evaluates_the_tide_gradient(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("evaluated the tide gradient")
    monkeypatch.setattr(lt, "tide_acceleration_and_gradient", boom)
    monkeypatch.setattr(lt, "tide_gravity_gradient", boom)
    t = np.array([0.0, 60.0])
    dyn.propagate_state(t, X0, *BASE, lunar_tide=TIDE)
    with pytest.raises(AssertionError, match="tide gradient"):        # the patch is live
        dyn.propagate_augmented_state(t, _aug0(X0), *BASE, lunar_tide=TIDE, method="DOP853")


# ----------------------------------------------------------------------
# T_G6: 6-, 42- and 48-state paths agree, with and without harmonics
# ----------------------------------------------------------------------
def _zonal_model():
    cbar = np.zeros((4, 4))
    cbar[2, 0] = cbar_n0_from_unnormalized_jn(J2_MOON_UNNORMALIZED, 2)
    cbar[3, 0] = -3.2e-6
    return SphericalHarmonicGravityModel(mu_m3_s2=MU_MOON_M3S2, r_ref_m=1738.0e3, cbar=cbar,
                                         sbar=np.zeros((4, 4)), nmax=3, mmax=0, frame="MOON_PA_DE440")


def _rot_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


@pytest.mark.parametrize("harmonics", [False, True])
def test_six_42_and_48_state_paths_carry_the_same_tide(harmonics):
    kw = dict(srp=NOSHADOW, lunar_tide=TIDE, **TIGHT)
    if harmonics:
        kw.update(harmonic_model=_zonal_model(), harmonic_rotation=_rot_x(0.03))
        kw_var = dict(kw, harmonic_stm_opt_in=True)
    else:
        kw.update(j2_moon=J2_MOON_UNNORMALIZED)
        kw_var = kw
    p6 = dyn.propagate_state(T6H, X0, *BASE, **kw)
    p42 = dyn.propagate_augmented_state(T6H, _aug0(X0), *BASE, **kw_var)
    p48 = dyn.propagate_state_with_k_sensitivity(T6H, X0, *BASE, **kw_var)
    runs = [p42, p48]
    if not harmonics:
        runs.append(_reference(lambda: dyn.propagate_augmented_state(T6H, _aug0(X0), *BASE, **kw_var)))
    scale = np.max(np.abs(p48[:, 6:42]))
    for run in runs:
        assert np.max(np.abs(run[:, 0:3] - p6[:, 0:3])) <= 1e-5
        assert np.max(np.abs(run[:, 6:42] - p48[:, 6:42])) / scale <= 1e-8
    # the tide is really in all of them: metres in 6 h
    p6_free = dyn.propagate_state(T6H, X0, *BASE, **dict(kw, lunar_tide=None))
    assert np.max(np.abs(p6_free[:, 0:3] - p6[:, 0:3])) > 1.0


# ----------------------------------------------------------------------
# T_G7: event-segmented (D2) propagation with the tide
# ----------------------------------------------------------------------
T1D = np.arange(0.0, DAY_S + 1.0, 600.0)
PENUMBRA = SRPOptions(k_srp_m2_per_kg=0.01)
D2 = dict(srp=PENUMBRA, method="DOP853", shadow_events=True, rtol=1e-11, atol=1e-12, j2_moon=J2_MOON_UNNORMALIZED)


def test_shadow_events_with_the_tide_are_consistent_and_the_tide_reaches_the_pre_pass():
    for call in (lambda: dyn.propagate_state(T1D, X0_ECLIPSED, *BASE, lunar_tide=TIDE, **D2),
                 lambda: dyn.propagate_augmented_state(T1D, _aug0(X0_ECLIPSED), *BASE, lunar_tide=TIDE, **D2),
                 lambda: dyn.propagate_state_with_k_sensitivity(T1D, X0_ECLIPSED, *BASE, lunar_tide=TIDE, **D2)):
        assert np.all(np.isfinite(call()))
    pre = dict(srp=PENUMBRA, rtol=1e-11, atol=1e-12, j2_moon=J2_MOON_UNNORMALIZED)
    cuts_tide = np.array(dyn.shadow_contact_times(T1D, X0_ECLIPSED, *BASE, lunar_tide=TIDE, **pre))
    cuts_free = np.array(dyn.shadow_contact_times(T1D, X0_ECLIPSED, *BASE, **pre))
    assert cuts_tide.size == cuts_free.size > 20
    assert np.max(np.abs(cuts_tide - cuts_free)) > dyn._CONTACT_TIME_TOL_S
    rhs, sun_lookup = dyn._build_state_rhs(T1D, *BASE, J2_MOON_UNNORMALIZED, 0.0, "indirect", None, None, None,
                                           PENUMBRA, lunar_tide=TIDE)
    with pytest.raises(RuntimeError, match="Shadow contact inconsistent"):
        dyn._integrate_segmented(T1D, X0_ECLIPSED, rhs, 1e-11, 1e-12, "DOP853", list(cuts_free), "control",
                                 contact_check=dyn._shadow_contact_check(sun_lookup, PENUMBRA))


# ----------------------------------------------------------------------
# T_G8: fail closed
# ----------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    dict(static_field_convention=None),
    dict(static_field_convention="ZERO_TIDE"),
    dict(static_field_convention="MEAN_TIDE"),
    dict(static_field_convention="tide_free"),
    dict(static_field_convention="TIDE_FREE", love_numbers=((4, 0.01),)),
    dict(static_field_convention="TIDE_FREE", love_numbers=((2, 0.024), (2, 0.024))),
    dict(static_field_convention="TIDE_FREE", love_numbers=((2, 0.0),)),
    dict(static_field_convention="TIDE_FREE", love_numbers=((2, -0.02),)),
    dict(static_field_convention="TIDE_FREE", love_numbers=((2, float("nan")),)),
    dict(static_field_convention="TIDE_FREE", love_numbers=((2, float("inf")),)),
    dict(static_field_convention="TIDE_FREE", love_numbers=((True, 0.02),)),
    dict(static_field_convention="TIDE_FREE", love_numbers=()),
    dict(static_field_convention="TIDE_FREE", radius_m=0.0),
    dict(static_field_convention="TIDE_FREE", radius_m=-1738e3),
    dict(static_field_convention="TIDE_FREE", raising_bodies=("EARTH", "VENUS")),
    dict(static_field_convention="TIDE_FREE", raising_bodies=("EARTH", "EARTH")),
    dict(static_field_convention="TIDE_FREE", raising_bodies=()),
    dict(static_field_convention="TIDE_FREE", raising_bodies="EARTH"),
])
def test_invalid_tide_options_raise(kwargs):
    with pytest.raises(ValueError):
        LunarTideOptions(**kwargs)


def test_the_static_field_convention_has_no_default():
    with pytest.raises(TypeError):
        LunarTideOptions()
    with pytest.raises(TypeError):
        LunarTideOptions("TIDE_FREE")       # keyword-only


@pytest.mark.parametrize("bad", [True, "TIDE_FREE", {"static_field_convention": "TIDE_FREE"}])
def test_a_lunar_tide_that_is_not_options_raises_before_integrating(bad):
    t = np.array([0.0, 60.0])
    with pytest.raises(TypeError, match="LunarTideOptions"):
        dyn.propagate_state(t, X0, *BASE, lunar_tide=bad)
    with pytest.raises(TypeError, match="LunarTideOptions"):
        dyn.propagate_augmented_state(t, _aug0(X0), *BASE, lunar_tide=bad)


def test_a_raising_body_without_gm_raises_and_dropping_it_is_allowed():
    t = np.array([0.0, 60.0])
    with pytest.raises(ValueError, match="EARTH"):
        dyn.propagate_state(t, X0, MU_MOON_M3S2, 0.0, MU_SUN_M3S2, earth, sun, lunar_tide=TIDE)
    with pytest.raises(ValueError, match="EARTH"):
        dyn.propagate_state_with_k_sensitivity(t, X0, MU_MOON_M3S2, 0.0, MU_SUN_M3S2, earth, sun,
                                               srp=NOSHADOW, lunar_tide=TIDE)
    sun_only = LunarTideOptions(static_field_convention="TIDE_FREE", raising_bodies=("SUN",))
    assert np.all(np.isfinite(dyn.propagate_state(t, X0, MU_MOON_M3S2, 0.0, MU_SUN_M3S2, earth, sun,
                                                  lunar_tide=sun_only)))


def test_defaults_are_the_gl1800f_label_values():
    assert TIDE.love_numbers == ((2, 0.024223), (3, 0.0163))
    assert TIDE.radius_m == 1738.0e3
    assert TIDE.raising_bodies == ("EARTH", "SUN")
    assert "do not include the permanent tide" in TIDE.provenance
