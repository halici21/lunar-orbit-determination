"""Phase 11C -- Python<->Numba parity gate for the Pines harmonics engine.

Locks the Numba twin in ``accelerated.pines_accel_bf_fast`` /
``_pines_accel_bf_numba`` against the pure-Python reference
``gravity_harmonics._pines_acceleration_bf`` and the analytic Moon-J2 helper.

Phase 11C acceptance REQUIRES Numba to be genuinely available and these tests to
actually run.  They are skipped only when Numba is absent; a skip does NOT
satisfy acceptance -- the implementer must stop and report "Numba unavailable,
parity not validated" in that case.

Scope: body-fixed core only; the inertial<->body rotation is applied on the
Python side here (identity/fixed frame).  Real m>0 propagation needs the
epoch-dependent J2000->MOON_PA rotation (Phase 12/13).
"""
import math
import unittest

import numpy as np

from lunar_od.accelerated import NUMBA_AVAILABLE, pines_accel_bf_fast
from lunar_od.constants import J2_MOON_UNNORMALIZED, MU_MOON_M3S2, R_MOON_M
from lunar_od.force_models import body_j2_acceleration
from lunar_od.gravity_harmonics import (
    SphericalHarmonicGravityModel,
    _pines_acceleration_bf,
    cbar_n0_from_unnormalized_jn,
)

MU, R_REF, J2 = MU_MOON_M3S2, R_MOON_M, J2_MOON_UNNORMALIZED
R_LLO = R_REF + 100e3
_ABS_FLOOR = 1e-18   # absolute fallback for near-zero accelerations


def _pos(lat_deg, lon_deg, radius_m):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    return radius_m * np.array(
        [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)]
    )


def _rel_or_abs(a_nb, a_py):
    """Relative error on the norm, with an absolute fallback near zero."""
    ref = float(np.linalg.norm(a_py))
    diff = float(np.linalg.norm(a_nb - a_py))
    if ref < _ABS_FLOOR:
        return diff          # absolute regime
    return diff / ref


def _ref_bf(r_bf, model):
    return _pines_acceleration_bf(
        r_bf, model.mu_m3_s2, model.r_ref_m, model.cbar, model.sbar,
        model.nmax, model.mmax,
    )


def _nb_bf(r_bf, model):
    return pines_accel_bf_fast(
        r_bf, model.mu_m3_s2, model.r_ref_m, model.cbar, model.sbar,
        model.nmax, model.mmax,
    )


def _model(cbar, sbar, nmax, mmax=None):
    if mmax is None:
        mmax = nmax
    return SphericalHarmonicGravityModel(
        mu_m3_s2=MU, r_ref_m=R_REF, cbar=cbar, sbar=sbar, nmax=nmax, mmax=mmax
    )


def _random_model(nmax, seed, scale=1e-6):
    rng = np.random.default_rng(seed)
    cbar = np.zeros((nmax + 1, nmax + 1))
    sbar = np.zeros((nmax + 1, nmax + 1))
    for n in range(2, nmax + 1):
        for m in range(0, n + 1):
            cbar[n, m] = scale * rng.standard_normal()
            if m > 0:
                sbar[n, m] = scale * rng.standard_normal()
    return _model(cbar, sbar, nmax)


def _nominal_model(nmax=3):
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
    return _model(cbar, sbar, nmax)


GRID = [
    _pos(0.0, 0.0, R_LLO), _pos(0.0, 90.0, R_LLO), _pos(35.0, 140.0, R_LLO),
    _pos(-60.0, 250.0, R_LLO), _pos(89.99, 12.0, R_LLO), _pos(90.0, 0.0, R_LLO),
    _pos(-90.0, 0.0, R_LLO), _pos(20.0, 300.0, R_REF + 50e3),
    _pos(-15.0, 200.0, R_REF + 2000e3),
]


@unittest.skipUnless(NUMBA_AVAILABLE, "Numba unavailable: Phase 11C parity NOT validated")
class PinesNumbaParityTests(unittest.TestCase):
    # 1 -- Numba C20/J2 bridge --------------------------------------------
    def test_numba_c20_matches_moon_j2(self):
        cbar = np.zeros((3, 3)); cbar[2, 0] = cbar_n0_from_unnormalized_jn(J2, 2)
        model = _model(cbar, np.zeros((3, 3)), nmax=2)
        worst = 0.0
        for r in GRID:
            a_nb = _nb_bf(r, model)                       # identity frame
            a_ref = body_j2_acceleration(r, MU, R_REF, J2, np.eye(3))
            worst = max(worst, float(np.linalg.norm(a_nb - a_ref)))
        self.assertLess(worst, 1e-14, msg=f"Numba C20/J2 bridge worst |da| = {worst:.3e}")

    # 2 -- Py vs Numba, random nmax=3 -------------------------------------
    def test_parity_random_nmax3(self):
        model = _random_model(3, seed=11031)
        worst = max(_rel_or_abs(_nb_bf(r, model), _ref_bf(r, model)) for r in GRID)
        self.assertLess(worst, 1e-13, msg=f"nmax=3 worst = {worst:.3e}")

    # 3 -- Py vs Numba, random nmax=8 -------------------------------------
    def test_parity_random_nmax8(self):
        model = _random_model(8, seed=11081)
        worst = max(_rel_or_abs(_nb_bf(r, model), _ref_bf(r, model)) for r in GRID)
        self.assertLess(worst, 1e-11, msg=f"nmax=8 worst = {worst:.3e}")

    # 4 -- Py vs Numba across many positions ------------------------------
    def test_parity_multiple_positions(self):
        model = _nominal_model(3)
        worst = 0.0
        for lat in (-90, -89.99, -45, -20, 0, 20, 45, 89.99, 90):
            for lon in (0, 45, 90, 140, 200, 305):
                for alt in (50e3, 500e3, 2000e3):
                    r = _pos(lat, lon, R_REF + alt)
                    worst = max(worst, _rel_or_abs(_nb_bf(r, model), _ref_bf(r, model)))
        self.assertLess(worst, 1e-12, msg=f"multi-position worst = {worst:.3e}")

    # 5 -- Truncation parity (nmax and mmax) ------------------------------
    def test_parity_truncation(self):
        cbar = np.zeros((5, 5)); sbar = np.zeros((5, 5))
        cbar[2, 0] = cbar_n0_from_unnormalized_jn(2.0346e-4, 2)
        cbar[2, 2], sbar[2, 2] = 3.47e-5, 1.0e-6
        cbar[3, 1], sbar[3, 1] = 2.6e-5, 5.5e-6
        cbar[4, 0], cbar[4, 3] = -1e-6, 2e-6
        r = _pos(35.0, 140.0, R_LLO)
        for nmax, mmax in ((2, 2), (3, 3), (4, 4), (4, 0), (4, 2)):
            model = _model(cbar, sbar, nmax, mmax)
            self.assertLess(_rel_or_abs(_nb_bf(r, model), _ref_bf(r, model)), 1e-12,
                            msg=f"truncation nmax={nmax} mmax={mmax}")

    # 6 -- Zero coefficients -> bit-exact zero ----------------------------
    def test_zero_coefficients_bit_zero(self):
        model = _model(np.zeros((4, 4)), np.zeros((4, 4)), nmax=3)
        a = _nb_bf(GRID[2], model)
        self.assertTrue(np.array_equal(a, np.zeros(3)))

    # 7 -- n=0 / n=1 rows ignored -----------------------------------------
    def test_n0_n1_ignored(self):
        cbar = np.zeros((3, 3)); sbar = np.zeros((3, 3))
        cbar[0, 0] = 1.0; cbar[1, 0] = 0.3; cbar[1, 1] = 0.5; sbar[1, 1] = 0.7
        model = _model(cbar, sbar, nmax=2)
        a = _nb_bf(GRID[2], model)
        self.assertTrue(np.array_equal(a, np.zeros(3)))

    # 8 -- C22 longitude behaviour preserved (Numba) ----------------------
    def test_c22_longitude_dependence(self):
        cbar = np.zeros((3, 3)); cbar[2, 2] = 3.47e-5
        model = _model(cbar, np.zeros((3, 3)), nmax=2)
        radial = {}
        for lon in (0.0, 90.0, 180.0):
            r = _pos(0.0, lon, R_LLO)
            radial[lon] = float(_nb_bf(r, model) @ r) / R_LLO
        self.assertNotAlmostEqual(radial[0.0], radial[90.0])
        self.assertTrue(np.isclose(radial[90.0], -radial[0.0], rtol=1e-9))
        self.assertTrue(np.isclose(radial[180.0], radial[0.0], rtol=1e-9))

    # 9 -- C30/J3 north-south asymmetry preserved (Numba) -----------------
    def test_c30_north_south_asymmetry(self):
        cbar = np.zeros((4, 4)); cbar[3, 0] = cbar_n0_from_unnormalized_jn(8.46e-6, 3)
        model = _model(cbar, np.zeros((4, 4)), nmax=3)
        a_n = _nb_bf(_pos(45.0, 0.0, R_LLO), model)
        a_s = _nb_bf(_pos(-45.0, 0.0, R_LLO), model)
        self.assertGreater(abs(a_n[2]), 0.0)
        self.assertTrue(np.isclose(a_n[2], a_s[2], rtol=1e-9))
        self.assertTrue(np.isclose(a_n[0], -a_s[0], rtol=1e-9))

    # 10 -- Exact pole finite in Numba ------------------------------------
    def test_exact_pole_finite(self):
        model = _nominal_model(3)
        for sign in (1.0, -1.0):
            a = _nb_bf(_pos(90.0 * sign, 0.0, R_LLO), model)
            self.assertTrue(bool(np.all(np.isfinite(a))))
            self.assertGreater(float(np.linalg.norm(a)), 0.0)
            self.assertGreater(float(np.hypot(a[0], a[1])), 0.0)  # m=1 kept


if __name__ == "__main__":
    unittest.main()
