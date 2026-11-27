"""F3 WP3.2 S4: the opt-in SRP light-time package in production (gates LT_G1(a), LT_G2-LT_G8).

Predeclaration: studies/phase17/configs/wp3_2_s4_srp_light_time_predeclaration.json (cc7a454, errata e7e261d
before any result).

Oracles never use the code under test:
- WP3.1's ray-crossing study term (``wp3_1_materiality_diagnostics.full_ray_crossing``) for LT_G2 and the finite
  differences of LT_G4;
- a Lorentz transformation of the radiation force, SPICE's apparent Sun (spkcvo 'CN+S') and the exact closest
  approach of the ray to the moving Moon (LT_G3);
- S3's fixed-step RK4 re-propagation of the PRODUCTION package-on RHS (LT_G5, the F1 pattern).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import lunar_od.dynamics as dyn
import lunar_od.earth_shadow as es
import lunar_od.srp_light_time as lt
from lunar_od.constants import (AU_M, J2_MOON_GRAIL_UNNORMALIZED, J2_MOON_UNNORMALIZED, R_MOON_M, R_SUN_M,
                                SPEED_OF_LIGHT_M_S)
from lunar_od.earth_shadow import EarthShadowOptions
from lunar_od.ephemeris import EphemerisCoverageError
from lunar_od.lunar_j2_model import use_lunar_j2_model
from lunar_od.lunar_tide import LunarTideOptions
from lunar_od.srp import SRPOptions, illumination_fraction, srp_acceleration_kernel, srp_acceleration_with_lunar_shadow
from lunar_od.srp_light_time import SRPLightTimeOptions, SSBVelocityTable
from tests import test_wp3_2_lunar_tide as G


@pytest.fixture(autouse=True, scope="module")
def _legacy_lunar_j2_model():
    """The synthetic gates run on the legacy mean-pole lunar J2 (predeclaration ``lunar_j2``); the real-E2 case
    passes the default grail_pa_de440 model explicitly."""
    with use_lunar_j2_model("iau_mean_pole_legacy"):
        yield


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "studies" / "phase17" / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "studies" / "phase17" / "scripts"))

import wp3_1_materiality_diagnostics as WD  # noqa: E402
import wp3_2_s3_geometry as GEO  # noqa: E402
import wp3_2_s4_geometry as GLT  # noqa: E402

C = SPEED_OF_LIGHT_M_S
ES = EarthShadowOptions()
LT = GLT.synthetic_options()
TIDE = LunarTideOptions(static_field_convention="TIDE_FREE")
EPS = np.finfo(float).eps
J2 = J2_MOON_UNNORMALIZED
K = 0.05
K_VAR = 0.2                                   # LT_G5 and LT_G6 (predeclaration)
NOS = SRPOptions(k_srp_m2_per_kg=K, shadow_model="NO_SHADOW")
PEN = SRPOptions(k_srp_m2_per_kg=K)
NOS_VAR = SRPOptions(k_srp_m2_per_kg=K_VAR, shadow_model="NO_SHADOW")
PEN_VAR = SRPOptions(k_srp_m2_per_kg=K_VAR)
PEN01 = SRPOptions(k_srp_m2_per_kg=0.01)
NOS01 = SRPOptions(k_srp_m2_per_kg=0.01, shadow_model="NO_SHADOW")
earth, sun = GEO.earth, GEO.sun_e
BASE, T6H, TERM, PLANE = GEO.BASE_E, GEO.T6H, GEO.TERM, GEO.PLANE
TIGHT = dict(method="DOP853", rtol=1e-13, atol=1e-14)


def _bits(x) -> bytes:
    return np.ascontiguousarray(np.asarray(x, dtype=float)).tobytes()


def _vel(t):
    return LT.ssb_velocities.velocities(t)


def a_lt_wp3_1(r, v, s, vs, vm, srp):
    """WP3.1's a_LT: the ray-crossing increment plus production's geometric SRP."""
    return WD.full_ray_crossing(r, v, s, vs, vm, srp) + srp_acceleration_with_lunar_shadow(r, s, srp)


# ----------------------------------------------------------------------
# LT_G1(a): off is the identity
# ----------------------------------------------------------------------
@pytest.mark.parametrize("size", [6, 42, 48])
def test_light_time_wrapper_returns_the_rhs_itself_when_off(size):
    def rhs(_t, y):
        return y
    for srp in (None, NOS, SRPOptions(enabled=False)):
        for earth_shadow in (None, ES):
            assert dyn._with_srp_light_time_acceleration(rhs, earth, sun, srp, None, earth_shadow,
                                                         state_size=size) is rhs


# ----------------------------------------------------------------------
# samples (seed 20261127; band relative to the local penumbra, erratum e7e261d)
# ----------------------------------------------------------------------
def make_samples(n_band=1000, n_shell=1000, seed=20261127):
    rng = np.random.default_rng(seed)
    n = n_band + n_shell
    s_hat = G._unit(rng, n)
    d_sun = rng.uniform(0.983 * AU_M, 1.017 * AU_M, size=n)
    r_sun = s_hat * d_sun[:, None]
    r_sc = np.empty((n, 3))
    x = rng.uniform(20e3, 3000e3, size=n_band)
    q = rng.uniform(-3.0, 3.0, size=n_band)
    rho = R_MOON_M + q * x * (R_SUN_M / d_sun[:n_band])
    p = G._unit(rng, n_band)
    p -= np.sum(p * s_hat[:n_band], axis=1)[:, None] * s_hat[:n_band]
    p /= np.linalg.norm(p, axis=1)[:, None]
    r_sc[:n_band] = -s_hat[:n_band] * x[:, None] + p * rho[:, None]
    r_sc[n_band:] = G._unit(rng, n_shell) * rng.uniform(R_MOON_M + 20e3, R_MOON_M + 3000e3, size=n_shell)[:, None]
    v_sc = G._unit(rng, n) * rng.uniform(0.5e3, 2.5e3, size=n)[:, None]
    v_moon = G._unit(rng, n) * rng.uniform(28.5e3, 31.5e3, size=n)[:, None]
    v_sun = G._unit(rng, n) * rng.uniform(0.0, 16.0, size=n)[:, None]
    return dict(r=r_sc, v=v_sc, s=r_sun, vs=v_sun, vm=v_moon, band=np.arange(n) < n_band)


@pytest.fixture(scope="module")
def samples():
    return make_samples()


# ----------------------------------------------------------------------
# LT_G2: production equals the WP3.1 ray-crossing term, bit for bit
# ----------------------------------------------------------------------
def wp3_1_parity(sm):
    mismatches, counts = [], dict(lit=0, penumbra=0, umbra=0)
    for i in range(sm["r"].shape[0]):
        r, v, s, vs, vm = sm["r"][i], sm["v"][i], sm["s"][i], sm["vs"][i], sm["vm"][i]
        g_prod, g_ref = lt.light_time_geometry(r, s, vs, vm), WD.ray_crossing_geometry(r, s, vs, vm)
        if _bits(g_prod[0]) != _bits(g_ref[0]) or _bits(g_prod[1]) != _bits(g_ref[1]):
            mismatches.append((i, "geometry"))
        for srp in (PEN01, NOS01):
            want = WD.full_ray_crossing(r, v, s, vs, vm, srp)
            if _bits(lt.srp_light_time_increment(r, v, s, vs, vm, srp)) != _bits(want):
                mismatches.append((i, srp.shadow_model, "increment"))
            if _bits(lt.srp_light_time_increment_and_jacobians(r, v, s, vs, vm, srp)[0]) != _bits(want):
                mismatches.append((i, srp.shadow_model, "jacobians"))
        nu = lt.light_time_illumination(r, s, vs, vm, PEN01)
        counts["lit" if nu == 1.0 else "umbra" if nu == 0.0 else "penumbra"] += 1
    return mismatches, counts


def test_production_equals_the_wp3_1_ray_crossing_term_bit_for_bit(samples):
    mismatches, counts = wp3_1_parity(samples)
    assert min(counts.values()) >= 300, counts                  # sample-design precondition
    assert not mismatches, mismatches[:10]


# ----------------------------------------------------------------------
# LT_G3(a): Lorentz oracle of the force law
# ----------------------------------------------------------------------
def lorentz_reference(r, v_sc, s, vs, vm, srp):
    """K P(d) nu gamma (1 - v.u/c)^2 [u' + (gamma - 1)(u'.vhat) vhat]: the rest-frame force of a body absorbing the
    flux and re-emitting it isotropically, transformed to the Sun frame (no first-order expansion)."""
    r_emit, r_occ = WD.ray_crossing_geometry(r, s, vs, vm)
    nu = WD.W.lunar_illumination(r, r_emit, r_occ, srp)
    rel = r - r_emit
    d = float(np.linalg.norm(rel))
    u = rel / d
    v = np.asarray(v_sc, float) + np.asarray(vm, float) - np.asarray(vs, float)
    speed = float(np.linalg.norm(v))
    vhat = v / speed if speed > 0.0 else np.zeros(3)
    beta = v / C
    gamma = 1.0 / math.sqrt(1.0 - float(beta @ beta))
    one = 1.0 - float(u @ beta)
    u_rest = (u - gamma * beta + (gamma - 1.0) * float(u @ vhat) * vhat) / (gamma * one)
    from lunar_od.srp import solar_pressure_at
    return srp.require_k() * solar_pressure_at(d) * nu * gamma * one * one * (
        u_rest + (gamma - 1.0) * float(u_rest @ vhat) * vhat)


def lorentz_values(sm, n_extra=200, seed=20261127):
    rng = np.random.default_rng(seed + 1)
    cases = [(sm["r"][i], sm["v"][i], sm["s"][i], sm["vs"][i], sm["vm"][i])
             for i in np.flatnonzero(~sm["band"])
             if lt.light_time_illumination(sm["r"][i], sm["s"][i], sm["vs"][i], sm["vm"][i], PEN01) == 1.0]
    for _ in range(n_extra):
        s = G._unit(rng, 1)[0] * rng.uniform(0.983 * AU_M, 1.017 * AU_M)
        r = G._unit(rng, 1)[0] * rng.uniform(R_MOON_M + 20e3, R_MOON_M + 3000e3)
        if float(r @ s) < 0.0:
            r = -r
        cases.append((r, np.zeros(3), s, np.zeros(3), G._unit(rng, 1)[0] * rng.uniform(0.0, 31.5e3)))
    worst = control = 0.0
    for r, v, s, vs, vm in cases:
        a_srp = srp_acceleration_with_lunar_shadow(r, s, PEN01)
        ref = lorentz_reference(r, v, s, vs, vm, PEN01)
        prod = lt.srp_light_time_increment(r, v, s, vs, vm, PEN01) + a_srp
        scale = float(np.linalg.norm(a_srp))
        worst = max(worst, float(np.linalg.norm(prod - ref)) / scale)
        control = max(control, float(np.linalg.norm(a_srp - ref)) / scale)
    r, s = cases[0][0], cases[0][2]
    zero = np.zeros(3)
    ref0 = lorentz_reference(r, zero, s, zero, zero, PEN01)
    a0 = PEN01.k_srp_m2_per_kg * srp_acceleration_kernel(r, s, PEN01)
    precondition = float(np.linalg.norm(ref0 - a0) / np.linalg.norm(a0))
    min_control = min(float(np.linalg.norm(srp_acceleration_with_lunar_shadow(r, s, PEN01)
                                           - lorentz_reference(r, v, s, vs, vm, PEN01)))
                      / float(np.linalg.norm(srp_acceleration_with_lunar_shadow(r, s, PEN01)))
                      for r, v, s, vs, vm in cases)
    return dict(n=len(cases), worst=worst, control_max=control, control_min=min_control, precondition=precondition)


def test_force_law_matches_the_lorentz_transformed_radiation_force(samples):
    v = lorentz_values(samples)
    assert v["n"] >= 600, v
    assert v["precondition"] <= 4 * EPS, v
    assert v["worst"] <= 1e-7, v
    assert v["control_min"] >= 100 * v["worst"], v


# ----------------------------------------------------------------------
# LT_G3(b): SPICE apparent Sun seen by the moving spacecraft
# ----------------------------------------------------------------------
def _angle(a, b):
    return math.atan2(float(np.linalg.norm(np.cross(a, b))), float(np.dot(a, b)))


def spice_direction_values(n=200, seed=20261127):
    import spiceypy as spice
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels()
    rng = np.random.default_rng(seed + 2)
    et_a, et_b = spice.str2et("2026-01-01 00:00:00 TDB"), spice.str2et("2027-01-01 00:00:00 TDB")
    worst = worst_corrected = precondition = 0.0
    control = math.inf
    for _ in range(n):
        et = float(rng.uniform(et_a, et_b))
        s = np.asarray(spice.spkpos("SUN", et, "J2000", "NONE", "MOON")[0], float) * 1e3
        vs = np.asarray(spice.spkezr("SUN", et, "J2000", "NONE", "SOLAR SYSTEM BARYCENTER")[0][3:6], float) * 1e3
        vm = np.asarray(spice.spkezr("MOON", et, "J2000", "NONE", "SOLAR SYSTEM BARYCENTER")[0][3:6], float) * 1e3
        r = G._unit(rng, 1)[0] * rng.uniform(R_MOON_M + 20e3, R_MOON_M + 3000e3)
        if float(r @ s) < 0.0:
            r = -r
        v = G._unit(rng, 1)[0] * rng.uniform(0.5e3, 2.5e3)
        obs = np.concatenate([r, v]) / 1e3
        app = np.asarray(spice.spkcvo("SUN", et, "J2000", "OBSERVER", "CN+S", obs, et, "MOON", "J2000")[0][:3],
                         float) * 1e3
        geo = np.asarray(spice.spkcvo("SUN", et, "J2000", "OBSERVER", "NONE", obs, et, "MOON", "J2000")[0][:3],
                         float) * 1e3
        precondition = max(precondition, float(np.linalg.norm(geo - (s - r))))
        a = lt.srp_light_time_increment(r, v, s, vs, vm, PEN01) + srp_acceleration_with_lunar_shadow(r, s, PEN01)
        ahat = a / np.linalg.norm(a)
        ref = -app / np.linalg.norm(app)
        worst = max(worst, _angle(ahat, ref))
        vs_perp = vs - float(vs @ ahat) * ahat
        corrected = ahat - vs_perp / C
        worst_corrected = max(worst_corrected, _angle(corrected, ref))
        control = min(control, _angle((r - s) / np.linalg.norm(r - s), ref))
    return dict(n=n, worst_rad=worst, corrected_rad=worst_corrected, control_min_rad=control,
                precondition_m=precondition)


def test_force_direction_matches_the_spice_apparent_sun():
    v = spice_direction_values()
    assert v["precondition_m"] <= 1.0, v
    assert v["worst_rad"] <= 1.5e-7, v
    assert v["control_min_rad"] >= 100 * v["worst_rad"], v


# ----------------------------------------------------------------------
# LT_G3(c): exact closest approach of the ray to the moving Moon
# ----------------------------------------------------------------------
def occulter_values(sm):
    worst = 0.0
    for i in np.flatnonzero(sm["band"]):
        r, s, vs, vm = sm["r"][i], sm["s"][i], sm["vs"][i], sm["vm"][i]
        r_emit, r_occ = lt.light_time_geometry(r, s, vs, vm)
        u = (r - r_emit) / np.linalg.norm(r - r_emit)
        w = C * u - vm
        exact = -vm * (float(r @ w) / float(w @ w))
        worst = max(worst, float(np.linalg.norm(r_occ - exact)))
    return dict(n=int(np.sum(sm["band"])), worst_m=worst)


def test_ray_crossing_occulter_matches_the_exact_closest_approach(samples):
    v = occulter_values(samples)
    assert v["worst_m"] <= 0.1, v


# ----------------------------------------------------------------------
# LT_G4: gradients against finite differences of the WP3.1 term
# ----------------------------------------------------------------------
def _stencil(x, h):
    pts = [x]
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        pts += [x + e, x - e, x + 2 * e, x - 2 * e]
    return pts


def _fd4(f, x, h):
    cols = []
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        cols.append((8.0 * (np.asarray(f(x + e)) - np.asarray(f(x - e)))
                     - (np.asarray(f(x + 2 * e)) - np.asarray(f(x - 2 * e)))) / (12.0 * h))
    return np.stack(cols, axis=-1)


def _lunar_contacts(r, s, vs, vm):
    from lunar_od.srp import apparent_radii_and_separation
    a, b, g, _, _ = apparent_radii_and_separation(r, s, np.zeros(3), R_MOON_M)
    return (*lt.light_time_contact_functions(r, s, vs, vm, R_MOON_M), g - a - b, g - abs(a - b))


def _stencil_clear(contacts, x, h, margin=1e-4):
    base = contacts(x)
    if min(abs(c) for c in base) < margin:
        return False
    for p in _stencil(x, h):
        if any(np.sign(c) != np.sign(c0) for c, c0 in zip(contacts(p), base)):
            return False
    return True


def gradient_values(sm):
    out = dict(n_pen=0, pos_conical=0.0, pos_no_shadow=0.0, n_full=0, inc_full_light=0.0, n_v=0, vel=0.0,
               vel_zero_in_umbra=True)
    for i in range(sm["r"].shape[0]):
        r, v, s, vs, vm = sm["r"][i], sm["v"][i], sm["s"][i], sm["vs"][i], sm["vm"][i]
        # (a) position gradient of a_LT
        for srp in (PEN01, NOS01):
            if srp is PEN01:
                nu = lt.light_time_illumination(r, s, vs, vm, srp)
                if not (sm["band"][i] and 0.0 < nu < 1.0):
                    continue
                if not _stencil_clear(lambda x: _lunar_contacts(x, s, vs, vm), r, 10.0):
                    continue
            _, g_r, _, _ = lt.srp_light_time_increment_and_jacobians(r, v, s, vs, vm, srp)
            from lunar_od.srp import srp_acceleration_position_gradient
            got = g_r + srp_acceleration_position_gradient(r, s, srp)
            fd = _fd4(lambda x: a_lt_wp3_1(x, v, s, vs, vm, srp), r, 10.0)
            err = float(np.linalg.norm(got - fd) / np.linalg.norm(fd))
            if srp is PEN01:
                out["n_pen"] += 1
                out["pos_conical"] = max(out["pos_conical"], err)
            else:
                out["pos_no_shadow"] = max(out["pos_no_shadow"], err)
        # (b) the increment's position gradient in full light
        if not sm["band"][i]:
            h = 1e6

            def sunward(x):
                r_emit, r_occ = lt.light_time_geometry(x, s, vs, vm)
                return float(x @ s) > 0.0 and float((x - r_occ) @ (r_emit - r_occ)) > 0.0
            if all(sunward(p) for p in _stencil(r, h)):
                _, g_r, _, _ = lt.srp_light_time_increment_and_jacobians(r, v, s, vs, vm, PEN01)
                fd = _fd4(lambda x: WD.full_ray_crossing(x, v, s, vs, vm, PEN01), r, h)
                out["n_full"] += 1
                out["inc_full_light"] = max(out["inc_full_light"], float(np.linalg.norm(g_r - fd) / np.linalg.norm(fd)))
        # (c) velocity gradient
        nu = lt.light_time_illumination(r, s, vs, vm, PEN01)
        _, _, g_v, _ = lt.srp_light_time_increment_and_jacobians(r, v, s, vs, vm, PEN01)
        if nu == 0.0:
            out["vel_zero_in_umbra"] &= not g_v.any()
            continue
        cols = []
        for k in range(3):
            e = np.zeros(3)
            e[k] = 100.0
            cols.append((a_lt_wp3_1(r, v + e, s, vs, vm, PEN01) - a_lt_wp3_1(r, v - e, s, vs, vm, PEN01)) / 200.0)
        fd_v = np.stack(cols, axis=-1)
        out["n_v"] += 1
        out["vel"] = max(out["vel"], float(np.linalg.norm(g_v - fd_v) / np.linalg.norm(fd_v)))
    return out


def test_gradients_match_finite_differences_of_the_wp3_1_term(samples):
    v = gradient_values(samples)
    assert v["n_pen"] >= 50 and v["n_full"] >= 50 and v["n_v"] >= 600, v
    assert v["pos_conical"] <= 1e-6 and v["pos_no_shadow"] <= 1e-6, v
    assert v["inc_full_light"] <= 1e-5, v
    assert v["vel"] <= 1e-6 and v["vel_zero_in_umbra"], v


def composed_gradient_values(n=2000, seed=20261125):
    """LT_G4(d): S3's E_G2 sampler (Earth penumbra and umbra), v as LT_G2; nu_E times the WP3.1 term."""
    from tests.test_wp3_2_earth_shadow import make_samples as s3_samples
    r_all, s_all, e_all = s3_samples(n, seed)
    rng = np.random.default_rng(20261127 + 3)
    v_all = G._unit(rng, n) * rng.uniform(0.5e3, 2.5e3, size=n)[:, None]
    vm_all = G._unit(rng, n) * rng.uniform(28.5e3, 31.5e3, size=n)[:, None]
    vs_all = G._unit(rng, n) * rng.uniform(0.0, 16.0, size=n)[:, None]
    worst_r = worst_v = 0.0
    count = 0
    for i in range(n):
        r, s, e, v, vs, vm = r_all[i], s_all[i], e_all[i], v_all[i], vs_all[i], vm_all[i]
        nu_e = es.earth_illumination(r, s, e, ES)
        if not (0.0 < nu_e < 1.0):
            continue

        def contacts(x):
            return (*es.earth_shadow_contact_functions(x, s, e, ES), *_lunar_contacts(x, s, vs, vm))
        if not _stencil_clear(contacts, r, 100.0):
            continue
        count += 1
        da, g_r, g_v, _ = lt.srp_light_time_increment_and_jacobians(r, v, s, vs, vm, PEN01)
        nu, grad_nu = es.earth_illumination_and_gradient(r, s, e, ES)
        got_r = nu * g_r + np.outer(da, grad_nu)
        fd = _fd4(lambda x: es.earth_illumination(x, s, e, ES) * WD.full_ray_crossing(x, v, s, vs, vm, PEN01),
                  r, 100.0)
        worst_r = max(worst_r, float(np.linalg.norm(got_r - fd) / np.linalg.norm(fd)))
        cols = []
        for k in range(3):
            d = np.zeros(3)
            d[k] = 100.0
            cols.append(nu_e * (WD.full_ray_crossing(r, v + d, s, vs, vm, PEN01)
                                - WD.full_ray_crossing(r, v - d, s, vs, vm, PEN01)) / 200.0)
        fd_v = np.stack(cols, axis=-1)
        if np.any(fd_v):
            worst_v = max(worst_v, float(np.linalg.norm(nu * g_v - fd_v) / np.linalg.norm(fd_v)))
    return dict(n=count, pos=worst_r, vel=worst_v)


def test_gradients_composed_with_the_earth_shadow_match_finite_differences():
    v = composed_gradient_values()
    assert v["n"] >= 50, v
    assert v["pos"] <= 1e-6 and v["vel"] <= 1e-6, v


# ----------------------------------------------------------------------
# LT_G5: variational equations against a fixed-step RK4 re-propagation
# ----------------------------------------------------------------------
H_POS, H_VEL = 3.0, 3e-3


def _production_rhs6(srp=NOS_VAR, srp_light_time=LT):
    rhs, _ = dyn._build_state_rhs(T6H, *BASE, J2, 0.0, "indirect", None, None, None, srp,
                                  dyn._legacy_j2_rotation, dyn.MOON_R_M, srp_light_time=srp_light_time)
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
    """Analysis-only 48-state replica; the controls drop the package's gradients or its df/dK part."""
    mus = BASE[:3]

    def rhs(t, y):
        x, phi, s_k = y[:6], y[6:42].reshape((6, 6), order="F"), y[42:48]
        r_e, r_s = earth(t), sun(t)
        v_sun, v_moon = _vel(t)
        x_dot = np.array(dyn.f3body_moon(x, *mus, r_e, r_s, j2_moon=J2, srp=NOS_VAR,
                                         moon_j2_rotation=dyn._legacy_j2_rotation(t), moon_j2_radius_m=dyn.MOON_R_M),
                         dtype=float)
        a = np.array(dyn.dynamics_jacobian_a_matrix(x, *mus, r_e, r_s, j2_moon=J2, srp=NOS_VAR,
                                                    moon_j2_rotation=dyn._legacy_j2_rotation(t),
                                                    moon_j2_radius_m=dyn.MOON_R_M), dtype=float)
        da, g_r, g_v, dk = lt.srp_light_time_increment_and_jacobians(x[:3], x[3:6], r_s, v_sun, v_moon, NOS_VAR)
        x_dot[3:6] += da
        if include_gradient:
            a[3:6, 0:3] += g_r
            a[3:6, 3:6] += g_v
        df_dk = np.zeros(6)
        df_dk[3:] = srp_acceleration_kernel(x[:3], r_s, NOS_VAR) + (dk if include_dfdk else 0.0)
        return np.concatenate([x_dot, (a @ phi).reshape(-1, order="F"), a @ s_k + df_dk])
    return rhs


def _replica(include_gradient, include_dfdk):
    y0 = np.concatenate([G._aug0(TERM), np.zeros(6)])
    sol = solve_ivp(_replica_rhs48(include_gradient, include_dfdk), (T6H[0], T6H[-1]), y0, method="DOP853",
                    t_eval=T6H, rtol=1e-13, atol=1e-14)
    assert sol.success
    return sol.y.T


def stm_values(rk4_columns):
    gated = G._stm_err(rk4_columns, G._phi(dyn.propagate_augmented_state(
        T6H, G._aug0(TERM), *BASE, j2_moon=J2, srp=NOS_VAR, srp_light_time=LT, **TIGHT)))
    control = G._stm_err(rk4_columns, G._phi(_replica(include_gradient=False, include_dfdk=True)))
    return gated, control


def test_light_time_stm_matches_repropagation(rk4_columns):
    gated, _control = stm_values(rk4_columns)            # the control is information only (predeclared)
    assert gated <= 1e-7, gated


def k_sensitivity_values():
    dk = 0.02
    plus = G._rk4(_production_rhs6(SRPOptions(k_srp_m2_per_kg=K_VAR + dk, shadow_model="NO_SHADOW")), TERM, T6H)
    minus = G._rk4(_production_rhs6(SRPOptions(k_srp_m2_per_kg=K_VAR - dk, shadow_model="NO_SHADOW")), TERM, T6H)
    fd = (plus - minus) / (2 * dk)
    s_k = dyn.propagate_state_with_k_sensitivity(T6H, TERM, *BASE, srp=NOS_VAR, j2_moon=J2, srp_light_time=LT,
                                                 **TIGHT)[:, 42:48]
    control = _replica(include_gradient=True, include_dfdk=False)[:, 42:48]
    scale = np.max(np.abs(s_k))
    return float(np.max(np.abs(fd - s_k)) / scale), float(np.max(np.abs(fd - control)) / scale)


def test_light_time_k_sensitivity_matches_k_repropagation():
    gated, control = k_sensitivity_values()
    assert gated <= 1e-6, gated
    assert control >= 10 * gated, (control, gated)


def _structure_states(rng):
    """Random states over the arc plus states in the light-time lunar penumbra and the Earth's penumbra."""
    states = []
    for _ in range(40):
        t = float(rng.uniform(0.0, T6H[-1]))
        states.append((t, G._unit(rng, 1)[0] * rng.uniform(R_MOON_M + 20e3, R_MOON_M + 500e3)))
    for t in np.linspace(0.3 * 3600.0, 1.6 * 3600.0, 6):
        s_hat = sun(t) / np.linalg.norm(sun(t))
        p = np.cross(s_hat, [0.0, 0.0, 1.0])
        p /= np.linalg.norm(p)
        v_sun, v_moon = _vel(t)
        for theta in np.radians(np.arange(65.0, 80.0, 1e-3)):
            r = (R_MOON_M + 100e3) * (-math.cos(theta) * s_hat + math.sin(theta) * p)
            if 0.0 < lt.light_time_illumination(r, sun(t), v_sun, v_moon, PEN) < 1.0:
                states.append((float(t), r))
                break
    return states


@pytest.mark.parametrize("earth_shadow", [None, ES], ids=["lt", "lt_es"])
@pytest.mark.parametrize("size", [6, 42, 48])
@pytest.mark.parametrize("srp", [NOS, PEN], ids=["no_shadow", "conical"])
def test_wrapper_adds_exactly_the_light_time_terms_and_nothing_else(size, srp, earth_shadow):
    rng = np.random.default_rng(41 + size)

    def base(t, y):
        return np.sin(np.arange(size) + t) * (1.0 + np.abs(y)) * 1e-3 + 1.45 * (np.arange(size) % 6 >= 3)
    wrapped = dyn._with_srp_light_time_acceleration(base, earth, sun, srp, LT, earth_shadow, state_size=size)
    n_pen = n_earth = 0
    for t, r in _structure_states(rng):
        y = rng.normal(size=size)
        y[:3] = r
        y[3:6] = rng.normal(size=3) * 1.6e3
        off, on = base(t, y), wrapped(t, y)
        v_sun, v_moon = _vel(t)
        da, g_r, g_v, dk = lt.srp_light_time_increment_and_jacobians(y[:3], y[3:6], sun(t), v_sun, v_moon, srp)
        if earth_shadow is not None:
            nu_e, grad_e = es.earth_illumination_and_gradient(y[:3], sun(t), earth(t), ES)
            g_r, g_v, dk, da = nu_e * g_r + np.outer(da, grad_e), nu_e * g_v, nu_e * dk, nu_e * da
            n_earth += 0.0 < nu_e < 1.0
        n_pen += 0.0 < lt.light_time_illumination(y[:3], sun(t), v_sun, v_moon, PEN) < 1.0
        diff = on - off
        assert not diff[0:3].any()
        np.testing.assert_allclose(diff[3:6], da, rtol=0, atol=4 * EPS * np.abs(off[3:6]).max())
        if size == 6:
            assert not diff[6:].any()
        if size >= 42:
            phi = y[6:42].reshape((6, 6), order="F")
            dphi = diff[6:42].reshape((6, 6), order="F")
            assert not dphi[0:3, :].any()
            np.testing.assert_allclose(dphi[3:6, :], g_r @ phi[0:3, :] + g_v @ phi[3:6, :], rtol=0,
                                       atol=4 * EPS * np.abs(off[6:42]).max())
        if size == 48:
            assert not diff[42:45].any()
            np.testing.assert_allclose(diff[45:48], g_r @ y[42:45] + g_v @ y[45:48] + dk, rtol=0,
                                       atol=4 * EPS * np.abs(off[42:48]).max())
    assert n_pen >= 5, n_pen
    if earth_shadow is not None:
        assert n_earth >= 5, n_earth


def test_state_only_propagation_never_evaluates_a_light_time_gradient(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("evaluated the light-time gradient")
    monkeypatch.setattr(lt, "srp_light_time_increment_and_jacobians", boom)
    monkeypatch.setattr(lt, "_illumination_partials", boom)
    t = np.array([0.0, 3600.0])
    dyn.propagate_state(t, PLANE, *BASE, srp=PEN, srp_light_time=LT)
    with pytest.raises(AssertionError, match="light-time gradient"):          # the patch is live
        dyn.propagate_augmented_state(t, G._aug0(PLANE), *BASE, srp=PEN, srp_light_time=LT, method="DOP853")


# ----------------------------------------------------------------------
# LT_G6: 6-, 42- and 48-state paths agree, relative to the package-off floor
# ----------------------------------------------------------------------
PATH_CASES = ("i_j2", "ii_j2_tide", "iii_harmonics", "iv_conical_events", "v_e2_grail_events",
              "vi_conical_earth_shadow_events")


def _path_setup(case):
    if case == "v_e2_grail_events":
        from lunar_od.spice_loader import load_spice_kernels
        load_spice_kernels()
        _et0, t, eph, mus, x0 = GEO.e2_case()
        base = (*mus, eph.earth_position, eph.sun_position)
        kw = dict(srp=PEN_VAR, shadow_events=True, j2_moon=J2_MOON_GRAIL_UNNORMALIZED,
                  lunar_j2_model="grail_pa_de440")
        return t, x0, base, kw, kw, True, GLT.e2_options()
    if case == "iv_conical_events":
        kw = dict(srp=PEN_VAR, shadow_events=True, j2_moon=J2)
        return T6H, PLANE, BASE, kw, kw, True, LT
    if case == "vi_conical_earth_shadow_events":
        kw = dict(srp=PEN_VAR, shadow_events=True, j2_moon=J2, earth_shadow=ES)
        return T6H, PLANE, BASE, kw, kw, True, LT
    kw = dict(srp=NOS_VAR)
    if case == "ii_j2_tide":
        kw["lunar_tide"] = TIDE
    if case == "iii_harmonics":
        kw.update(harmonic_model=G._zonal_model(), harmonic_rotation=G._rot_x(0.03))
        return T6H, TERM, BASE, kw, dict(kw, harmonic_stm_opt_in=True), False, LT
    kw["j2_moon"] = J2
    return T6H, TERM, BASE, kw, kw, True, LT


def _disagreement(t, x0, base, kw6, kw_var, reference):
    p6 = dyn.propagate_state(t, x0, *base, **kw6)
    runs = {"42": dyn.propagate_augmented_state(t, G._aug0(x0), *base, **kw_var),
            "48": dyn.propagate_state_with_k_sensitivity(t, x0, *base, **kw_var)}
    if reference:
        runs["42_reference"] = G._reference(lambda: dyn.propagate_augmented_state(t, G._aug0(x0), *base, **kw_var))
    scale = np.max(np.abs(runs["48"][:, 6:42]))
    pos = {name: float(np.max(np.abs(run[:, 0:3] - p6[:, 0:3]))) for name, run in runs.items()}
    phi = {name: float(np.max(np.abs(run[:, 6:42] - runs["48"][:, 6:42])) / scale) for name, run in runs.items()}
    return pos, phi, p6


def path_values(case):
    t, x0, base, kw6, kw_var, reference, options = _path_setup(case)
    pos_off, _phi_off, p6_off = _disagreement(t, x0, base, dict(kw6, **TIGHT), dict(kw_var, **TIGHT), reference)
    pos_on, phi_on, p6_on = _disagreement(t, x0, base, dict(kw6, srp_light_time=options, **TIGHT),
                                          dict(kw_var, srp_light_time=options, **TIGHT), reference)
    return dict(pos_on=pos_on, pos_off=pos_off, phi_on=phi_on,
                effect_m=float(np.max(np.abs(p6_on[:, 0:3] - p6_off[:, 0:3]))))


@pytest.mark.parametrize("case", PATH_CASES)
def test_six_42_and_48_state_paths_carry_the_same_light_time_term(case):
    v = path_values(case)
    assert max(v["pos_on"].values()) <= 2.0 * max(v["pos_off"].values()) + 1e-6, v
    assert max(v["phi_on"].values()) <= 1e-8, v
    assert v["effect_m"] >= 1e-4, v                    # the package is really in all of them


# ----------------------------------------------------------------------
# LT_G7: event-segmented (D2) propagation at the light-time contacts
# ----------------------------------------------------------------------
D2 = dict(method="DOP853", shadow_events=True, rtol=1e-11, atol=1e-12, j2_moon=J2)
PRE = dict(rtol=1e-11, atol=1e-12, j2_moon=J2)


def _lt_offsets(t, state):
    """Time offsets (s) of a state from the light-time lunar contacts (residual over rate along the trajectory)."""
    def f(tt, r):
        v_sun, v_moon = _vel(tt)
        return np.array(lt.light_time_contact_functions(r, sun(tt), v_sun, v_moon, R_MOON_M))
    h = 1e-2
    value = f(t, state[:3])
    rate = np.abs(f(t + h, state[:3] + h * state[3:6]) - f(t - h, state[:3] - h * state[3:6])) / (2 * h)
    return np.where(np.abs(value) <= 1e-9, 0.0, np.abs(value) / rate)


def shadow_event_values():
    x0 = PLANE
    kw = dict(srp=PEN, srp_light_time=LT)
    for call in (lambda: dyn.propagate_state(T6H, x0, *BASE, **kw, **D2),
                 lambda: dyn.propagate_augmented_state(T6H, G._aug0(x0), *BASE, **kw, **D2),
                 lambda: dyn.propagate_state_with_k_sensitivity(T6H, x0, *BASE, **kw, **D2)):
        assert np.all(np.isfinite(call()))
    cuts_lt = np.array(dyn.shadow_contact_times(T6H, x0, *BASE, **kw, **PRE))
    cuts_geo = np.array(dyn.shadow_contact_times(T6H, x0, *BASE, srp=PEN, **PRE))
    states = dyn.propagate_state(np.array([0.0, *cuts_lt]), x0, *BASE, **kw, **D2)
    offsets = [float(np.min(_lt_offsets(float(tc), st))) for tc, st in zip(cuts_lt, states[1:])]
    shifts = [float((cuts_lt - c)[np.argmin(np.abs(cuts_lt - c))]) for c in cuts_geo]
    nearest_geo = [float(np.min(np.abs(cuts_geo - c))) for c in cuts_lt]
    rhs, sun_lookup = dyn._build_state_rhs(T6H, *BASE, J2, 0.0, "indirect", None, None, None, PEN,
                                           dyn._legacy_j2_rotation, dyn.MOON_R_M, srp_light_time=LT)
    message = None
    try:
        dyn._integrate_segmented(T6H, x0, rhs, 1e-11, 1e-12, "DOP853", list(cuts_geo), "control",
                                 contact_check=dyn._shadow_contact_check(sun_lookup, PEN, None, light_time=LT))
    except RuntimeError as exc:
        message = str(exc)
    kw_es = dict(kw, earth_shadow=ES)
    cuts_es = np.array(dyn.shadow_contact_times(T6H, x0, *BASE, **kw_es, **PRE))
    for call in (lambda: dyn.propagate_state(T6H, x0, *BASE, **kw_es, **D2),
                 lambda: dyn.propagate_augmented_state(T6H, G._aug0(x0), *BASE, **kw_es, **D2),
                 lambda: dyn.propagate_state_with_k_sensitivity(T6H, x0, *BASE, **kw_es, **D2)):
        assert np.all(np.isfinite(call()))
    states_es = dyn.propagate_state(np.array([0.0, *cuts_es]), x0, *BASE, **kw_es, **D2)
    labels_es = []
    for tc, st in zip(cuts_es, states_es[1:]):
        earth_off = dyn._contact_time_offsets(float(tc), st, sun, NOS, (earth, ES))
        lunar_off = float(np.min(_lt_offsets(float(tc), st)))
        e_best = min(earth_off.values())
        labels_es.append(("earth", e_best) if e_best < lunar_off else ("lunar_lt", lunar_off))
    return dict(cuts_lt=cuts_lt, cuts_geo=cuts_geo, offsets=offsets, shifts=shifts, nearest_geo=nearest_geo,
                control_message=message, cuts_es=cuts_es, labels_es=labels_es)


def test_shadow_events_cut_at_the_light_time_contacts():
    v = shadow_event_values()
    tol = dyn._CONTACT_TIME_TOL_S
    assert len(v["cuts_lt"]) == len(v["cuts_geo"]) >= 2, v
    assert max(v["offsets"]) <= tol, v["offsets"]
    assert min(v["nearest_geo"]) > tol, v["nearest_geo"]
    assert all(0.02 <= abs(s) <= 0.5 for s in v["shifts"]), v["shifts"]
    assert v["control_message"] is not None and "Shadow contact inconsistent" in v["control_message"]
    n_earth = sum(lab == "earth" for lab, _ in v["labels_es"])
    assert n_earth >= 2 and any(lab == "lunar_lt" for lab, _ in v["labels_es"]), v["labels_es"]
    assert all(off <= tol for _, off in v["labels_es"]), v["labels_es"]


# ----------------------------------------------------------------------
# LT_G8: fail closed
# ----------------------------------------------------------------------
def _table(**over):
    kw = dict(t_s=np.array([0.0, 60.0, 120.0]), sun_vel_mps=np.zeros((3, 3)), moon_vel_mps=np.ones((3, 3)))
    kw.update(over)
    return SSBVelocityTable(**kw)


@pytest.mark.parametrize("over", [
    dict(t_s=np.array([0.0, 60.0, 60.0])), dict(t_s=np.array([0.0, np.nan, 120.0])),
    dict(t_s=np.array([0.0]), sun_vel_mps=np.zeros((1, 3)), moon_vel_mps=np.zeros((1, 3))),
    dict(sun_vel_mps=np.zeros((3, 2))), dict(moon_vel_mps=np.full((3, 3), np.inf)),
    dict(extrapolation_margin_s=-1.0), dict(extrapolation_margin_s=float("nan")),
])
def test_invalid_velocity_tables_raise(over):
    with pytest.raises(ValueError):
        _table(**over)


def test_options_are_keyword_only_typed_and_provenance_is_text():
    with pytest.raises(TypeError):
        SRPLightTimeOptions(_table())
    with pytest.raises(TypeError):
        SRPLightTimeOptions(ssb_velocities={"t_s": [0.0, 1.0]})
    with pytest.raises(ValueError):
        SRPLightTimeOptions(ssb_velocities=_table(), provenance=1)


def _entry_points(srp_light_time, srp):
    t = np.array([0.0, 60.0])
    kw = dict(srp_light_time=srp_light_time, srp=srp)
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
def test_light_time_without_active_srp_raises_everywhere(srp):
    for call in _entry_points(LT, srp):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("bad", [True, "CN+S", {"ssb_velocities": None}])
def test_a_light_time_that_is_not_options_raises_before_integrating(bad):
    for call in _entry_points(bad, NOS):
        with pytest.raises(TypeError, match="SRPLightTimeOptions"):
            call()


def test_propagation_beyond_the_velocity_table_raises():
    short = SRPLightTimeOptions(ssb_velocities=_table(moon_vel_mps=np.full((3, 3), 3e4)))
    with pytest.raises(EphemerisCoverageError):
        dyn.propagate_state(np.array([0.0, 600.0]), TERM, *BASE, srp=NOS, srp_light_time=short, method="DOP853")
