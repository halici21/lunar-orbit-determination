"""Phase 11A -- spherical-harmonics prototype verification (Pines vs Cunningham).

Tests the two candidate formulations in ``examples/phase11a_harmonics_prototype.py``
against the existing Moon-J2 helper and against each other.  Prototype scope:
fixed frame / identity rotation only (real m>0 propagation needs the
epoch-dependent J2000->MOON_PA rotation, Phase 12/13); NO production module,
NO production code touched.
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from lunar_od.constants import J2_MOON_UNNORMALIZED, MU_MOON_M3S2, R_MOON_M
from lunar_od.force_models import body_j2_acceleration
from phase11a_harmonics_prototype import (
    cbar_n0_from_jn,
    cunningham_acceleration,
    nominal_coefficients,
    pines_acceleration,
    random_coefficients,
)

MU, R_REF, J2 = MU_MOON_M3S2, R_MOON_M, J2_MOON_UNNORMALIZED
R_LLO = R_REF + 100e3
IDENTITY = np.eye(3)


def _pos(lat_deg, lon_deg, radius_m):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    return radius_m * np.array(
        [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)]
    )


def _c20_only():
    cbar = np.zeros((3, 3))
    cbar[2, 0] = cbar_n0_from_jn(J2, 2)   # Cbar20 = -J2/sqrt(5)
    return cbar, np.zeros((3, 3))


PROBE_POSITIONS = [
    _pos(0.0, 0.0, R_LLO), _pos(0.0, 90.0, R_LLO), _pos(35.0, 140.0, R_LLO),
    _pos(-60.0, 250.0, R_LLO), _pos(90.0, 0.0, R_LLO),
    _pos(20.0, 300.0, R_REF + 2000e3),
]


class _FormulationChecks:
    """Shared battery; subclasses bind ``func`` to one formulation."""

    func = None  # staticmethod set by subclass

    # T1 ------------------------------------------------------------------
    def test_zero_coefficients_zero_acceleration(self):
        a = self.func(PROBE_POSITIONS[2], MU, R_REF, np.zeros((4, 4)), np.zeros((4, 4)))
        self.assertEqual(float(np.max(np.abs(a))), 0.0)

    def test_n0_n1_rows_ignored(self):
        # n=0 (point mass) and n=1 (CoM) rows must NOT leak into the output.
        cbar = np.zeros((4, 4))
        cbar[0, 0] = 1.0
        cbar[1, 0] = 0.3
        cbar[1, 1] = 0.5
        a = self.func(PROBE_POSITIONS[2], MU, R_REF, cbar, np.zeros((4, 4)))
        self.assertEqual(float(np.max(np.abs(a))), 0.0)

    # T2 ------------------------------------------------------------------
    def test_c20_only_matches_moon_j2_helper(self):
        cbar, sbar = _c20_only()
        for r in PROBE_POSITIONS:
            a_engine = self.func(r, MU, R_REF, cbar, sbar, nmax=2)
            a_helper = body_j2_acceleration(r, MU, R_REF, J2, IDENTITY)
            self.assertLess(float(np.linalg.norm(a_engine - a_helper)), 1e-14,
                            msg=f"C20/J2 equivalence broken at r={r}")

    # T3 ------------------------------------------------------------------
    def test_wrong_sign_c20_detected(self):
        cbar_ok, sbar = _c20_only()
        cbar_bad = -cbar_ok
        r = PROBE_POSITIONS[2]
        a_ok = self.func(r, MU, R_REF, cbar_ok, sbar, nmax=2)
        a_bad = self.func(r, MU, R_REF, cbar_bad, sbar, nmax=2)
        a_ref = body_j2_acceleration(r, MU, R_REF, J2, IDENTITY)
        # linear in the coefficients: wrong sign flips the whole perturbation
        np.testing.assert_allclose(a_bad, -a_ok, rtol=1e-15)
        rel = float(np.linalg.norm(a_bad - a_ref) / np.linalg.norm(a_ref))
        self.assertGreater(rel, 1.0)   # ~2x discrepancy vs the trusted helper

    # T5 ------------------------------------------------------------------
    def test_c22_longitude_dependence(self):
        cbar = np.zeros((3, 3))
        cbar[2, 2] = 3.47e-5
        sbar = np.zeros((3, 3))
        radial = {}
        for lon in (0.0, 90.0, 180.0):
            r = _pos(0.0, lon, R_LLO)
            a = self.func(r, MU, R_REF, cbar, sbar)
            radial[lon] = float(a @ r) / R_LLO
        # cos(2*lam): sign flip at 90 deg, same value again at 180 deg
        self.assertNotAlmostEqual(radial[0.0], radial[90.0])
        self.assertTrue(np.isclose(radial[90.0], -radial[0.0], rtol=1e-10))
        self.assertTrue(np.isclose(radial[180.0], radial[0.0], rtol=1e-10))

    def test_c20_zonal_axisymmetry(self):
        cbar, sbar = _c20_only()
        norms = [
            float(np.linalg.norm(self.func(_pos(0.0, lon, R_LLO), MU, R_REF, cbar, sbar)))
            for lon in (0.0, 45.0, 90.0, 210.0)
        ]
        for n in norms[1:]:
            self.assertTrue(np.isclose(n, norms[0], rtol=1e-12))

    # T6 ------------------------------------------------------------------
    def test_c30_north_south_asymmetry(self):
        cbar = np.zeros((4, 4))
        cbar[3, 0] = cbar_n0_from_jn(8.46e-6, 3)   # ~ Moon J3
        sbar = np.zeros((4, 4))
        a_n = self.func(_pos(45.0, 0.0, R_LLO), MU, R_REF, cbar, sbar)
        a_s = self.func(_pos(-45.0, 0.0, R_LLO), MU, R_REF, cbar, sbar)
        # odd zonal: a_z SAME at mirrored latitudes, a_x antisymmetric ...
        self.assertGreater(abs(a_n[2]), 0.0)
        self.assertTrue(np.isclose(a_n[2], a_s[2], rtol=1e-12))
        self.assertTrue(np.isclose(a_n[0], -a_s[0], rtol=1e-12))
        # ... which BREAKS mirror symmetry (an even field would have az(N)=-az(S),
        # as C20 does):
        cbar20, sbar20 = _c20_only()
        b_n = self.func(_pos(45.0, 0.0, R_LLO), MU, R_REF, cbar20, sbar20)
        b_s = self.func(_pos(-45.0, 0.0, R_LLO), MU, R_REF, cbar20, sbar20)
        self.assertTrue(np.isclose(b_n[2], -b_s[2], rtol=1e-12))

    # T8 ------------------------------------------------------------------
    def test_unit_mistake_detectable(self):
        cbar, sbar = nominal_coefficients(3)
        r = _pos(20.0, 40.0, R_LLO)
        a_m = self.func(r, MU, R_REF, cbar, sbar)
        a_km = self.func(r / 1000.0, MU, R_REF, cbar, sbar)
        ratio = float(np.linalg.norm(a_km) / np.linalg.norm(a_m))
        self.assertGreater(ratio, 1e6)


class PinesFormulationTests(_FormulationChecks, unittest.TestCase):
    func = staticmethod(pines_acceleration)

    # T4 (Pines must be clean at the exact poles) ---------------------------
    def test_exact_pole_finite_and_continuous(self):
        cbar, sbar = nominal_coefficients(3)
        for sign in (1.0, -1.0):
            a_pole = pines_acceleration(_pos(90.0 * sign, 0.0, R_LLO),
                                        MU, R_REF, cbar, sbar)
            a_near = pines_acceleration(_pos(89.9999 * sign, 0.0, R_LLO),
                                        MU, R_REF, cbar, sbar)
            self.assertTrue(bool(np.all(np.isfinite(a_pole))))
            self.assertGreater(float(np.linalg.norm(a_pole)), 0.0)
            cont = float(np.linalg.norm(a_pole - a_near) / np.linalg.norm(a_near))
            self.assertLess(cont, 1e-4)   # smooth field, no polar artefact


class CunninghamFormulationTests(_FormulationChecks, unittest.TestCase):
    func = staticmethod(cunningham_acceleration)


class CrossFormulationTests(unittest.TestCase):
    # T7 ------------------------------------------------------------------
    def test_consistency_nmax3_grid(self):
        cbar, sbar = nominal_coefficients(3)
        worst = 0.0
        for lat in (-80, -45, -20, 0, 20, 45, 80):
            for lon in range(0, 360, 45):
                for alt in (50e3, 500e3):
                    r = _pos(lat, lon, R_REF + alt)
                    ap = pines_acceleration(r, MU, R_REF, cbar, sbar)
                    ac = cunningham_acceleration(r, MU, R_REF, cbar, sbar)
                    worst = max(worst,
                                float(np.linalg.norm(ap - ac) / np.linalg.norm(ap)))
        self.assertLess(worst, 1e-12)

    def test_consistency_nmax8_random(self):
        cbar, sbar = random_coefficients(8)
        worst = 0.0
        for lat in (-70, -30, 0, 40, 75):
            for lon in (10, 100, 200, 305):
                r = _pos(lat, lon, R_LLO)
                ap = pines_acceleration(r, MU, R_REF, cbar, sbar)
                ac = cunningham_acceleration(r, MU, R_REF, cbar, sbar)
                worst = max(worst,
                            float(np.linalg.norm(ap - ac) / np.linalg.norm(ap)))
        self.assertLess(worst, 1e-11)

    def test_near_pole_agreement(self):
        cbar, sbar = nominal_coefficients(3)
        r = _pos(89.99, 0.0, R_LLO)
        ap = pines_acceleration(r, MU, R_REF, cbar, sbar)
        ac = cunningham_acceleration(r, MU, R_REF, cbar, sbar)
        rel = float(np.linalg.norm(ap - ac) / np.linalg.norm(ap))
        self.assertLess(rel, 1e-9)

    def test_exact_pole_cunningham_limitation_documented(self):
        """The classical formulation is finite (clamped) but WRONG at the exact
        pole: the m=1 horizontal acceleration is lost because the dU/dlam and
        dU/dphi channels are multiplied by x = y = 0.  Pines keeps it.  This
        test pins down the documented limitation rather than hiding it."""
        cbar, sbar = nominal_coefficients(3)   # includes C31/S31 (m=1 terms)
        r = _pos(90.0, 0.0, R_LLO)
        a_p = pines_acceleration(r, MU, R_REF, cbar, sbar)
        a_c = cunningham_acceleration(r, MU, R_REF, cbar, sbar)
        self.assertTrue(bool(np.all(np.isfinite(a_c))))            # clamp works
        hor_p = float(np.hypot(a_p[0], a_p[1]))
        hor_c = float(np.hypot(a_c[0], a_c[1]))
        self.assertGreater(hor_p, 1e-5)            # Pines keeps the real m=1 accel
        self.assertLess(hor_c, 1e-2 * hor_p)       # classical: essentially lost
        rel = float(np.linalg.norm(a_c - a_p) / np.linalg.norm(a_p))
        self.assertGreater(rel, 1e-3)              # documented degradation is real


if __name__ == "__main__":
    unittest.main()
