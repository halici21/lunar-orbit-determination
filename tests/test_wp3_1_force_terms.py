"""F3 WP3.1-R force terms against closed forms and the production force model (gate G1_UNIT a-d).

Predeclaration: studies/phase17/configs/wp3_1_materiality_predeclaration.json (amendment A1 for G1 b).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "studies" / "phase17" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import phase_c6_lunar_radiation as C6  # noqa: E402
import phase_c7_dynamics_candidates as C7  # noqa: E402
import wp3_1_force_terms as W  # noqa: E402
from lunar_od.constants import AU_M, J2_EARTH_UNNORMALIZED, R_EARTH_J2_REF_M, R_MOON_M  # noqa: E402
from lunar_od.dynamics import f3body_moon  # noqa: E402
from lunar_od.ephemeris_generation import gravitational_parameters  # noqa: E402
from lunar_od.srp import SRPOptions, solar_pressure_at, srp_acceleration_with_lunar_shadow  # noqa: E402

MU_M, MU_E, MU_S = gravitational_parameters("DE440")
SRP = SRPOptions(k_srp_m2_per_kg=W.K_SRP_M2_PER_KG)
ZERO = np.zeros(3)
Z_HAT = np.array([0.0, 0.0, 1.0])

# (a) the Earth 384400 km from the Moon along +y, the Sun 1 AU from the Earth along +x
R_E = np.array([0.0, 384_400e3, 0.0])
S_HAT = np.array([1.0, 0.0, 0.0])
R_S_FROM_MOON = R_E + AU_M * S_HAT
BEHIND_EARTH = R_E - 384_400e3 * S_HAT                    # anti-Sun axis, 384400 km behind the Earth
SUNWARD_OF_EARTH = R_E + 384_400e3 * S_HAT

# (b) the Sun 1 AU from the Moon along +x
R_SUN = np.array([AU_M, 0.0, 0.0])
R_SC_LIT = np.array([0.0, 1_837.4e3, 0.0])                # terminator, full light
R_SC_UMBRA = np.array([-1_837.4e3, 0.0, 0.0])
R_SC_PENUMBRA = np.array([-597.9e3, 1_737.4e3, 0.0])
V_SC = np.array([0.0, 0.0, 1_633.0])
V_MOON = np.array([1_000.0, 29_800.0, 300.0])
V_SUN = np.array([-8.0, 12.0, 0.5])


# ----------------------------------------------------------------------
# (a) EARTH_SHADOW
# ----------------------------------------------------------------------
def test_earth_shadow_umbra_sunward_side_and_far_off_axis():
    assert W.earth_illumination(BEHIND_EARTH, R_S_FROM_MOON, R_E) == 0.0
    assert W.earth_illumination(SUNWARD_OF_EARTH, R_S_FROM_MOON, R_E) == 1.0
    assert W.earth_illumination(BEHIND_EARTH + 50_000e3 * Z_HAT, R_S_FROM_MOON, R_E) == 1.0


def test_earth_shadow_is_continuous_and_monotone_across_the_penumbra():
    w = np.arange(0.0, 12_000e3 + 1.0, 1e3)               # 1 km samples on a line normal to the axis
    nu = np.array([W.earth_illumination(BEHIND_EARTH + x * Z_HAT, R_S_FROM_MOON, R_E) for x in w])
    steps = np.diff(nu)
    assert nu[0] == 0.0 and nu[-1] == 1.0
    assert np.all(steps >= 0.0)
    assert float(np.max(steps)) <= 0.01
    assert np.count_nonzero((nu > 0.0) & (nu < 1.0)) > 1000          # the ~3600 km penumbra is resolved


def test_earth_shadow_term_is_minus_srp_in_the_umbra_and_exactly_zero_in_light():
    a_srp = srp_acceleration_with_lunar_shadow(BEHIND_EARTH, R_S_FROM_MOON, SRP)
    assert np.linalg.norm(a_srp) > 0.0                    # the Moon does not shadow this point
    assert np.array_equal(W.earth_shadow_acceleration(BEHIND_EARTH, R_S_FROM_MOON, R_E, SRP), -a_srp)
    assert np.all(W.earth_shadow_acceleration(SUNWARD_OF_EARTH, R_S_FROM_MOON, R_E, SRP) == 0.0)


# ----------------------------------------------------------------------
# (b) SRP_LIGHT_TIME
# ----------------------------------------------------------------------
def test_light_time_term_vanishes_with_infinite_light_speed():
    assert 0.0 < W.lunar_illumination(R_SC_PENUMBRA, R_SUN, ZERO, SRP) < 1.0
    for r in (R_SC_LIT, R_SC_UMBRA, R_SC_PENUMBRA):
        a_srp = srp_acceleration_with_lunar_shadow(r, R_SUN, SRP)
        term = W.srp_light_time_term(r, V_SC, R_SUN, V_SUN, V_MOON, SRP, c=math.inf)
        assert np.linalg.norm(term) <= 1e-15 * np.linalg.norm(a_srp)


def test_aberration_part_with_velocity_normal_to_the_sun_line():
    # no Sun or Moon motion, so v = v_sc and nothing is displaced; V_SC is normal to u (u_z = 0)
    rel = R_SC_LIT - R_SUN
    d = float(np.linalg.norm(rel))
    assert abs(float(np.dot(V_SC, rel))) == 0.0
    term = W.srp_light_time_term(R_SC_LIT, V_SC, R_SUN, ZERO, ZERO, SRP)
    want = -(SRP.k_srp_m2_per_kg * solar_pressure_at(d)) * V_SC / W.SPEED_OF_LIGHT_M_S
    assert np.linalg.norm(term - want) <= 1e-12 * np.linalg.norm(want)


def test_velocity_along_the_sun_line_scales_srp_by_one_minus_two_v_over_c():
    # amendment A1: the Doppler factor of the flux and the aberration term each give -v/c along u
    rel = R_SC_LIT - R_SUN
    u = rel / np.linalg.norm(rel)
    for s in (1_633.0, -1_633.0, 30_000.0):
        a_lt = W.srp_light_time_acceleration(R_SC_LIT, s * u, R_SUN, ZERO, ZERO, SRP)
        a_srp = srp_acceleration_with_lunar_shadow(R_SC_LIT, R_SUN, SRP)
        factor = 1.0 - 2.0 * s / W.SPEED_OF_LIGHT_M_S
        assert np.linalg.norm(a_lt - factor * a_srp) <= 1e-12 * np.linalg.norm(a_srp)


def test_moon_displacement_moves_the_umbra_boundary_the_predicted_way():
    x = -600e3

    def nu_at(y, v_moon):
        r = np.array([x, y, 0.0])
        r_emit, r_occ = W.light_time_geometry(r, R_SUN, ZERO, v_moon)
        return W.lunar_illumination(r, r_emit, r_occ, SRP)

    def boundary(v_moon):
        lo, hi = 1_700e3, 1_760e3
        assert nu_at(lo, v_moon) == 0.0 and nu_at(hi, v_moon) > 0.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if nu_at(mid, v_moon) == 0.0 else (lo, mid)
        return lo, hi

    lo, hi = boundary(ZERO)
    v = np.array([0.0, 30_000.0, 0.0])
    # the Moon moving +y: the ray passed it -v delta back, so the shadow lags to -y
    assert nu_at(lo - 1.0, v) > 0.0
    assert nu_at(hi + 1.0, -v) == 0.0
    shift = boundary(v)[0] - lo
    v_delta = 30_000.0 * math.hypot(x, lo) / W.SPEED_OF_LIGHT_M_S
    assert abs(shift + v_delta) <= 1e-2 * v_delta


# ----------------------------------------------------------------------
# (c) EARTH_J2
# ----------------------------------------------------------------------
def _j2_closed_form(r, mu, radius, j2, pole):
    """(3/2) J2 mu R^2 / r^4 [(5 s^2 - 1) r_hat - 2 s pole], s = pole . r_hat (vector form, any pole)."""
    rn = float(np.linalg.norm(r))
    rh = r / rn
    s = float(np.dot(pole, rh))
    return 1.5 * j2 * mu * radius ** 2 / rn ** 4 * ((5.0 * s * s - 1.0) * rh - 2.0 * s * pole)


R_SC_J2 = np.array([1.20e6, -0.55e6, 1.17e6])
R_E_J2 = np.array([3.1e8, 2.0e8, 0.6e8])


def _tilted_rotation(angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])
    rz = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return rz @ rx


@pytest.mark.parametrize("rotation", [np.eye(3), _tilted_rotation(2.6e-3)], ids=["identity", "tilted"])
def test_earth_j2_parts_equal_an_independent_closed_form(rotation):
    a_sc, a_moon = W.earth_j2_parts(R_SC_J2, R_E_J2, MU_E, rotation)
    pole = rotation[2]                                    # the body z-axis in J2000
    for got, r in ((a_sc, R_SC_J2 - R_E_J2), (a_moon, -R_E_J2)):
        want = _j2_closed_form(r, MU_E, R_EARTH_J2_REF_M, J2_EARTH_UNNORMALIZED, pole)
        assert np.linalg.norm(got - want) <= 1e-12 * np.linalg.norm(want)
    assert np.array_equal(W.earth_j2_acceleration(R_SC_J2, R_E_J2, MU_E, rotation), a_sc - a_moon)


def test_earth_j2_term_with_identity_equals_the_production_difference():
    state = np.concatenate([R_SC_J2, [10.0, -1_500.0, 600.0]])
    r_s = np.array([1.2e11, -8.0e10, -3.5e10])
    f1 = f3body_moon(state, MU_M, MU_E, MU_S, R_E_J2, r_s, j2_earth=J2_EARTH_UNNORMALIZED)
    f0 = f3body_moon(state, MU_M, MU_E, MU_S, R_E_J2, r_s)
    term = W.earth_j2_acceleration(R_SC_J2, R_E_J2, MU_E, np.eye(3))
    assert float(np.max(np.abs((f1[3:] - f0[3:]) - term))) <= 1e-14


@pytest.fixture(scope="module")
def spice_kernels():
    spice = pytest.importorskip("spiceypy")
    from lunar_od.spice_loader import load_spice_kernels
    try:
        load_spice_kernels(None, clear=False)
    except FileNotFoundError:
        pytest.skip("SPICE kernels not available")
    return spice


@pytest.mark.parametrize("utc", ["2026-03-02T23:33:00", "2026-08-27T16:13:00", "2027-03-02T00:00:00"])
def test_itrf93_pole_is_the_iau2006_precessed_pole(spice_kernels, utc):
    et = float(spice_kernels.str2et(utc))
    pole = W.itrf93_rotation(et)[2]                       # the ITRF93 z-axis in J2000
    angle_arcsec = math.degrees(math.atan2(math.hypot(pole[0], pole[1]), pole[2])) * 3600.0
    t = et / (36525.0 * 86400.0)
    theta_a = 2004.191903 * t - 0.4294934 * t ** 2 - 0.04182264 * t ** 3
    assert abs(angle_arcsec - theta_a) <= 20.0


def test_planet_gm_values_come_from_the_de440_gm_kernel(spice_kernels):
    gm = W.planet_gm()
    assert abs(gm["VENUS"] / 3.24858592e14 - 1.0) < 1e-6
    assert abs(gm["JUPITER BARYCENTER"] / 1.26712764e17 - 1.0) < 1e-6


# ----------------------------------------------------------------------
# (d) TIDE, LRP, PN, PLANETS wrappers pass the declared parameters
# ----------------------------------------------------------------------
R_SC_D = np.array([1.20e6, -0.55e6, 1.17e6])
V_SC_D = np.array([900.0, 1_100.0, -700.0])
R_EARTH_D = np.array([3.1e8, 2.0e8, 0.6e8])
R_SUN_D = np.array([1.2e11, -8.0e10, -3.5e10])


def test_declared_parameters():
    assert W.TIDE_DEGREES == ((2, 0.024223), (3, 0.0163)) and W.TIDE_RADIUS_M == 1738.0e3
    assert W.K_LRP_M2_PER_KG == 0.010 == W.K_SRP_M2_PER_KG
    m = W.LRP_MODEL
    assert (m.albedo, m.emissivity, m.t_night_k, m.n_theta, m.n_phi, m.radius_m, m.thermal, m.reflected) == \
        (0.12, 0.95, 100.0, 60, 72, R_MOON_M, True, True)
    assert W.R_EARTH_SHADOW_M == 6_378_137.0
    assert W.TERMS == ("TIDE", "LRP", "EARTH_SHADOW", "PN", "PLANETS", "SRP_LIGHT_TIME", "EARTH_J2")


def test_tide_wrapper_sums_earth_and_sun_degrees_two_and_three():
    want = np.zeros(3)
    for d, mu in ((R_EARTH_D, MU_E), (R_SUN_D, MU_S)):
        for n, k in ((2, 0.024223), (3, 0.0163)):
            want = want + C7.tide_acceleration(R_SC_D[None], d[None], mu, n, k, 1738.0e3)[0]
    assert np.array_equal(W.tide_acceleration(R_SC_D, R_EARTH_D, R_SUN_D, MU_E, MU_S), want)


def test_lrp_pn_and_planet_wrappers():
    assert np.array_equal(W.lrp_acceleration(R_SC_D, R_SUN_D),
                          0.010 * C6.acceleration_kernel(R_SC_D[None], R_SUN_D[None], C6.LunarRadiation())[0])
    assert np.array_equal(W.pn_acceleration(R_SC_D, V_SC_D, MU_M), C7.pn_acceleration(R_SC_D[None], V_SC_D[None], MU_M)[0])
    pos = {"VENUS": np.array([4.0e10, -9.0e10, 1.0e10]), "JUPITER BARYCENTER": np.array([-6.0e11, 4.0e11, 1.5e11])}
    gm = {"VENUS": 3.2e14, "JUPITER BARYCENTER": 1.27e17}
    want = (C7.third_body_acceleration(R_SC_D[None], pos["VENUS"][None], gm["VENUS"])[0]
            + C7.third_body_acceleration(R_SC_D[None], pos["JUPITER BARYCENTER"][None], gm["JUPITER BARYCENTER"])[0])
    assert np.array_equal(W.planets_acceleration(R_SC_D, pos, gm), want)


def test_with_term_touches_only_the_acceleration_rows():
    def rhs(t, x):
        return np.array([x[3], x[4], x[5], -1.25, 2.5, -3.75])

    def term(t, x, ctx):
        return np.array([1e-9, -2e-9, 3e-9]) * (1.0 + t)

    x = np.arange(6.0)
    assert np.array_equal(W.with_term(rhs, term, None, 0.0)(5.0, x), rhs(5.0, x))
    out = W.with_term(rhs, term, None, 1.0)(5.0, x)
    assert np.array_equal(out[:3], x[3:]) and np.array_equal(out[3:], rhs(5.0, x)[3:] + term(5.0, x, None))
