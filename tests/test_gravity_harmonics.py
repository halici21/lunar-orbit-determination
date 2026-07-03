"""Phase 11B -- production low-degree harmonics engine verification.

Independent tests for ``lunar_od.gravity_harmonics`` (Pines, acceleration-only,
default-off).  The engine is NOT wired into the production propagation path, so
these tests import it directly and assert it reproduces the existing Moon-J2
math, respects its API contract, and leaves the J2 framework untouched.

Prototype scope: fixed / identity rotation only.  Real m>0 propagation needs the
epoch-dependent J2000->MOON_PA rotation (Phase 12/13).
"""
import math
import unittest

import numpy as np

from lunar_od.constants import J2_MOON_UNNORMALIZED, MU_MOON_M3S2, R_MOON_M
from lunar_od.force_models import body_j2_acceleration
from lunar_od.gravity_harmonics import (
    SphericalHarmonicGravityModel,
    cbar_n0_from_unnormalized_jn,
    spherical_harmonic_acceleration,
)

MU, R_REF, J2 = MU_MOON_M3S2, R_MOON_M, J2_MOON_UNNORMALIZED
R_LLO = R_REF + 100e3
IDENTITY = np.eye(3)


def _pos(lat_deg, lon_deg, radius_m):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    return radius_m * np.array(
        [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)]
    )


PROBE_POSITIONS = [
    _pos(0.0, 0.0, R_LLO), _pos(0.0, 90.0, R_LLO), _pos(35.0, 140.0, R_LLO),
    _pos(-60.0, 250.0, R_LLO), _pos(90.0, 0.0, R_LLO),
    _pos(20.0, 300.0, R_REF + 2000e3),
]


def _rot_axes(rx_deg, ry_deg, rz_deg):
    """Proper rotation matrix from intrinsic X, Y, Z angles (test fixture)."""
    ax, ay, az = map(math.radians, (rx_deg, ry_deg, rz_deg))
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _model(cbar, sbar, nmax, mmax=None, **kw):
    if mmax is None:
        mmax = nmax
    return SphericalHarmonicGravityModel(
        mu_m3_s2=MU, r_ref_m=R_REF, cbar=cbar, sbar=sbar, nmax=nmax, mmax=mmax, **kw
    )


def _c20_model():
    cbar = np.zeros((3, 3))
    cbar[2, 0] = cbar_n0_from_unnormalized_jn(J2, 2)   # Cbar20 = -J2/sqrt(5)
    return _model(cbar, np.zeros((3, 3)), nmax=2)


def _nominal_model(nmax=3):
    """Fixed synthetic model, nonzero in every (n, m) slot (not a real model)."""
    cbar = np.zeros((nmax + 1, nmax + 1))
    sbar = np.zeros((nmax + 1, nmax + 1))
    cbar[2, 0] = cbar_n0_from_unnormalized_jn(2.0346e-4, 2)
    cbar[2, 1], sbar[2, 1] = 1.2e-8, -3.0e-9
    cbar[2, 2], sbar[2, 2] = 3.47e-5, 1.0e-6
    if nmax >= 3:
        cbar[3, 0] = cbar_n0_from_unnormalized_jn(8.46e-6, 3)
        cbar[3, 1], sbar[3, 1] = 2.6e-5, 5.5e-6
        cbar[3, 2], sbar[3, 2] = 1.4e-5, 4.9e-6
        cbar[3, 3], sbar[3, 3] = 1.2e-5, -2.0e-6
    return _model(cbar, sbar, nmax=nmax)


class HarmonicsEngineTests(unittest.TestCase):
    # 1 -- zero coefficients -> zero acceleration -------------------------
    def test_zero_coefficients_zero_acceleration(self):
        model = _model(np.zeros((4, 4)), np.zeros((4, 4)), nmax=3)
        a = spherical_harmonic_acceleration(PROBE_POSITIONS[2], model)
        self.assertEqual(float(np.max(np.abs(a))), 0.0)

    # 2 -- C20-only == existing Moon J2 helper (< 1e-14 m/s^2) -------------
    def test_c20_only_matches_moon_j2_helper(self):
        model = _c20_model()
        worst = 0.0
        for r in PROBE_POSITIONS:
            a_eng = spherical_harmonic_acceleration(r, model, IDENTITY)
            a_ref = body_j2_acceleration(r, MU, R_REF, J2, IDENTITY)
            worst = max(worst, float(np.linalg.norm(a_eng - a_ref)))
        self.assertLess(worst, 1e-14, msg=f"C20/J2 bridge worst |da| = {worst:.3e}")

    # 3 -- wrong-sign C20 is caught ---------------------------------------
    def test_wrong_sign_c20_detected(self):
        ok = _c20_model()
        bad = _model(-ok.cbar.copy(), np.zeros((3, 3)), nmax=2)
        r = PROBE_POSITIONS[2]
        a_ok = spherical_harmonic_acceleration(r, ok, IDENTITY)
        a_bad = spherical_harmonic_acceleration(r, bad, IDENTITY)
        a_ref = body_j2_acceleration(r, MU, R_REF, J2, IDENTITY)
        np.testing.assert_allclose(a_bad, -a_ok, rtol=1e-15)
        rel = float(np.linalg.norm(a_bad - a_ref) / np.linalg.norm(a_ref))
        self.assertGreater(rel, 1.0)

    # 4 -- C22 longitude dependence ---------------------------------------
    def test_c22_longitude_dependence(self):
        cbar = np.zeros((3, 3)); cbar[2, 2] = 3.47e-5
        model = _model(cbar, np.zeros((3, 3)), nmax=2)
        radial = {}
        for lon in (0.0, 90.0, 180.0):
            r = _pos(0.0, lon, R_LLO)
            radial[lon] = float(spherical_harmonic_acceleration(r, model) @ r) / R_LLO
        self.assertNotAlmostEqual(radial[0.0], radial[90.0])
        self.assertTrue(np.isclose(radial[90.0], -radial[0.0], rtol=1e-10))
        self.assertTrue(np.isclose(radial[180.0], radial[0.0], rtol=1e-10))

    # 5 -- C20 zonal (axial) symmetry -------------------------------------
    def test_c20_zonal_axisymmetry(self):
        model = _c20_model()
        norms = [
            float(np.linalg.norm(spherical_harmonic_acceleration(_pos(0.0, lon, R_LLO), model)))
            for lon in (0.0, 45.0, 90.0, 210.0)
        ]
        for n in norms[1:]:
            self.assertTrue(np.isclose(n, norms[0], rtol=1e-12))

    # 6 -- C30/J3 north-south asymmetry -----------------------------------
    def test_c30_north_south_asymmetry(self):
        cbar = np.zeros((4, 4)); cbar[3, 0] = cbar_n0_from_unnormalized_jn(8.46e-6, 3)
        model = _model(cbar, np.zeros((4, 4)), nmax=3)
        a_n = spherical_harmonic_acceleration(_pos(45.0, 0.0, R_LLO), model)
        a_s = spherical_harmonic_acceleration(_pos(-45.0, 0.0, R_LLO), model)
        self.assertGreater(abs(a_n[2]), 0.0)
        self.assertTrue(np.isclose(a_n[2], a_s[2], rtol=1e-12))      # odd zonal
        self.assertTrue(np.isclose(a_n[0], -a_s[0], rtol=1e-12))
        # contrast: an even field (C20) mirrors az(N) = -az(S)
        b_n = spherical_harmonic_acceleration(_pos(45.0, 0.0, R_LLO), _c20_model())
        b_s = spherical_harmonic_acceleration(_pos(-45.0, 0.0, R_LLO), _c20_model())
        self.assertTrue(np.isclose(b_n[2], -b_s[2], rtol=1e-12))

    # 7 -- exact pole finite (Pines) --------------------------------------
    def test_exact_pole_finite(self):
        model = _nominal_model(3)
        for sign in (1.0, -1.0):
            a = spherical_harmonic_acceleration(_pos(90.0 * sign, 0.0, R_LLO), model)
            self.assertTrue(bool(np.all(np.isfinite(a))))
            self.assertGreater(float(np.linalg.norm(a)), 0.0)
            self.assertGreater(float(np.hypot(a[0], a[1])), 0.0)  # m=1 kept, not lost

    # 8 -- near-pole continuity -------------------------------------------
    def test_near_pole_continuity(self):
        model = _nominal_model(3)
        for sign in (1.0, -1.0):
            a_pole = spherical_harmonic_acceleration(_pos(90.0 * sign, 0.0, R_LLO), model)
            a_near = spherical_harmonic_acceleration(_pos(89.9999 * sign, 0.0, R_LLO), model)
            cont = float(np.linalg.norm(a_pole - a_near) / np.linalg.norm(a_near))
            self.assertLess(cont, 1e-4)

    # 9 -- unit sanity ----------------------------------------------------
    def test_unit_mistake_detectable(self):
        model = _nominal_model(3)
        r = _pos(20.0, 40.0, R_LLO)
        a_m = spherical_harmonic_acceleration(r, model)
        a_km = spherical_harmonic_acceleration(r / 1000.0, model)
        self.assertGreater(float(np.linalg.norm(a_km) / np.linalg.norm(a_m)), 1e6)

    # 10 -- invalid coefficient shape / params reject ---------------------
    def test_invalid_shapes_reject(self):
        with self.assertRaises(ValueError):   # non-square cbar
            _model(np.zeros((3, 4)), np.zeros((3, 4)), nmax=2)
        with self.assertRaises(ValueError):   # cbar/sbar shape mismatch
            _model(np.zeros((3, 3)), np.zeros((4, 4)), nmax=2)
        with self.assertRaises(ValueError):   # nmax exceeds arrays
            _model(np.zeros((3, 3)), np.zeros((3, 3)), nmax=5)
        with self.assertRaises(ValueError):   # nmax < 2
            _model(np.zeros((3, 3)), np.zeros((3, 3)), nmax=1)
        with self.assertRaises(ValueError):   # mmax > nmax
            _model(np.zeros((4, 4)), np.zeros((4, 4)), nmax=2, mmax=3)
        with self.assertRaises(ValueError):   # sbar[:, 0] must be zero
            s = np.zeros((3, 3)); s[2, 0] = 1e-6
            _model(np.zeros((3, 3)), s, nmax=2)
        with self.assertRaises(ValueError):   # non-positive mu / r_ref
            SphericalHarmonicGravityModel(
                mu_m3_s2=0.0, r_ref_m=R_REF,
                cbar=np.zeros((3, 3)), sbar=np.zeros((3, 3)), nmax=2, mmax=2)

    # 11 -- nmax / mmax truncation works ----------------------------------
    def test_nmax_mmax_truncation(self):
        cbar = np.zeros((5, 5)); sbar = np.zeros((5, 5))
        cbar[2, 0] = cbar_n0_from_unnormalized_jn(2.0346e-4, 2)
        cbar[2, 2], sbar[2, 2] = 3.47e-5, 1.0e-6
        cbar[3, 1], sbar[3, 1] = 2.6e-5, 5.5e-6
        cbar[4, 0], cbar[4, 3] = -1e-6, 2e-6
        r = _pos(35.0, 140.0, R_LLO)

        full4 = spherical_harmonic_acceleration(r, _model(cbar, sbar, nmax=4))
        trunc2 = spherical_harmonic_acceleration(r, _model(cbar, sbar, nmax=2))
        # nmax=2 model must equal a model whose n=3,4 rows are physically absent
        cbar2 = cbar.copy(); sbar2 = sbar.copy()
        cbar2[3:, :] = 0.0; sbar2[3:, :] = 0.0
        trunc2_ref = spherical_harmonic_acceleration(r, _model(cbar2, sbar2, nmax=2))
        self.assertTrue(np.allclose(trunc2, trunc2_ref, rtol=0, atol=0))
        self.assertGreater(float(np.linalg.norm(full4 - trunc2)), 0.0)

        # mmax=0 keeps only the zonal (m=0) part of an nmax=4 model
        zonal_only = spherical_harmonic_acceleration(r, _model(cbar, sbar, nmax=4, mmax=0))
        cbar_z = cbar.copy(); sbar_z = sbar.copy()
        cbar_z[:, 1:] = 0.0; sbar_z[:, 1:] = 0.0
        zonal_ref = spherical_harmonic_acceleration(r, _model(cbar_z, sbar_z, nmax=4))
        self.assertTrue(np.allclose(zonal_only, zonal_ref, rtol=0, atol=0))
        self.assertGreater(float(np.linalg.norm(full4 - zonal_only)), 0.0)

    # 12 -- n=0 / n=1 rows are ignored (not leaked) -----------------------
    def test_n0_n1_ignored(self):
        cbar = np.zeros((3, 3)); sbar = np.zeros((3, 3))
        cbar_leak = cbar.copy()
        cbar_leak[0, 0] = 1.0     # would-be point mass
        cbar_leak[1, 0] = 0.3     # would-be CoM offset
        cbar_leak[1, 1] = 0.5
        sbar_leak = sbar.copy(); sbar_leak[1, 1] = 0.7
        r = PROBE_POSITIONS[2]
        a_plain = spherical_harmonic_acceleration(r, _model(cbar, sbar, nmax=2))
        a_leak = spherical_harmonic_acceleration(r, _model(cbar_leak, sbar_leak, nmax=2))
        self.assertTrue(np.array_equal(a_plain, a_leak))
        self.assertEqual(float(np.max(np.abs(a_leak))), 0.0)

    # 13 -- inertial->body rotation works (frame covariance) --------------
    def test_inertial_to_body_rotation(self):
        model = _nominal_model(3)
        c = _rot_axes(17.0, -33.0, 128.0)
        self.assertTrue(np.allclose(c @ c.T, IDENTITY, atol=1e-12))
        # physics is frame-covariant: evaluating in body-fixed axes directly,
        # or in inertial axes with the rotation supplied, must agree.
        r_bf = _pos(28.0, 61.0, R_LLO)             # a body-fixed position
        a_bf = spherical_harmonic_acceleration(r_bf, model)        # identity
        r_in = c.T @ r_bf                          # same point, inertial axes
        a_in = spherical_harmonic_acceleration(r_in, model, c)
        self.assertTrue(np.allclose(a_in, c.T @ a_bf, rtol=0, atol=1e-18))
        # and the rotation is not a no-op for a non-trivial C
        self.assertGreater(float(np.linalg.norm(a_in - a_bf)), 1e-9)

    # 14 -- default identity rotation == explicit identity ----------------
    def test_default_identity_rotation(self):
        model = _nominal_model(3)
        for r in PROBE_POSITIONS:
            a_none = spherical_harmonic_acceleration(r, model)
            a_eye = spherical_harmonic_acceleration(r, model, IDENTITY)
            self.assertTrue(np.allclose(a_none, a_eye, rtol=0, atol=1e-18))

    # 15 -- metadata / frame preserved ------------------------------------
    def test_metadata_preserved(self):
        meta = {"source": "synthetic-11B", "checksum": "n/a", "note": "test model"}
        model = _model(np.zeros((3, 3)), np.zeros((3, 3)), nmax=2,
                       frame="MOON_ME", metadata=meta)
        self.assertEqual(model.frame, "MOON_ME")
        self.assertEqual(model.metadata, meta)
        self.assertEqual(model.mu_m3_s2, MU)
        self.assertEqual(model.r_ref_m, R_REF)


if __name__ == "__main__":
    unittest.main()
