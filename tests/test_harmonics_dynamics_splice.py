"""Phase 13B -- lunar harmonics 6-state dynamics-splice verification.

Locks the splice contract: (a) harmonics-off stays bit-identical to the
existing production dynamics, (b) a C20-only model bridges to the existing
Moon-J2 path, (c) the J2 double-count ban, the m>0 epoch-rotation rule, the
rotation-grid coverage guard and the STM refusal are enforced in real code at
the propagation entry points, and (d) the Python and Numba harmonics RHS paths
agree to machine precision.

All rotation grids here are SYNTHETIC (slow z-rotation at the lunar sidereal
rate) so these tests need no SPICE kernels; real MOON_PA sampling is covered by
tests/test_lunar_frames.py.  Uses frozen third-body geometry (no .mat/SPICE).
"""
import math
import unittest

import numpy as np

from lunar_od.accelerated import NUMBA_AVAILABLE, f3body_harmonics_rhs
from lunar_od.constants import (
    J2_EARTH_UNNORMALIZED,
    J2_MOON_UNNORMALIZED,
    MU_EARTH_M3S2,
    MU_MOON_M3S2,
    MU_SUN_M3S2,
    R_MOON_M,
)
from lunar_od.dynamics import (
    _MCI_TO_MOON_BF,
    f3body_moon,
    propagate_augmented_state,
    propagate_state,
)
from lunar_od.gravity_harmonics import (
    SphericalHarmonicGravityModel,
    cbar_n0_from_unnormalized_jn,
)

MU_M, MU_E, MU_S = MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2
J2, J2_E = J2_MOON_UNNORMALIZED, J2_EARTH_UNNORMALIZED

# Frozen third-body geometry (same vectors as the other dynamics tests).
R_ME = np.array([-83446893.0, 354010875.0, 178558253.0])
R_MS = np.array([1.40753701450e11, -4.21884124439e10, -1.82638284191e10])
GE = lambda t: R_ME    # noqa: E731
GS = lambda t: R_MS    # noqa: E731

LUNAR_RATE_RAD_S = 2.0 * math.pi / (27.321661 * 86400.0)


def _rotz(angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def _synthetic_rotation_pair(t_start, t_end, step_s=60.0):
    """Synthetic epoch-dependent grid: z-rotation at the lunar sidereal rate."""
    t_grid = np.arange(t_start, t_end + step_s / 2.0, step_s)
    grid = np.stack([_rotz(LUNAR_RATE_RAD_S * t) for t in t_grid])
    return t_grid, grid


def _polar_state(alt_m=100e3):
    r0 = R_MOON_M + alt_m
    vc = math.sqrt(MU_M / r0)
    return np.array([r0, 0.0, 0.0, 0.0, 0.0, vc])   # polar plane (x-z)


def _model(cbar, sbar, nmax, mmax):
    return SphericalHarmonicGravityModel(
        mu_m3_s2=MU_M, r_ref_m=R_MOON_M, cbar=cbar, sbar=sbar, nmax=nmax, mmax=mmax
    )


def _c20_model():
    cbar = np.zeros((3, 3))
    cbar[2, 0] = cbar_n0_from_unnormalized_jn(J2, 2)
    return _model(cbar, np.zeros((3, 3)), nmax=2, mmax=0)


def _c20_c22_model():
    cbar = np.zeros((3, 3))
    cbar[2, 0] = cbar_n0_from_unnormalized_jn(J2, 2)
    cbar[2, 2] = 3.47e-5
    sbar = np.zeros((3, 3))
    sbar[2, 2] = 1.0e-6
    return _model(cbar, sbar, nmax=2, mmax=2)


def _c22_only_model():
    cbar = np.zeros((3, 3))
    cbar[2, 2] = 3.47e-5
    return _model(cbar, np.zeros((3, 3)), nmax=2, mmax=2)


def _nominal3_model():
    cbar = np.zeros((4, 4))
    sbar = np.zeros((4, 4))
    cbar[2, 0] = cbar_n0_from_unnormalized_jn(J2, 2)
    cbar[2, 2], sbar[2, 2] = 3.47e-5, 1.0e-6
    cbar[3, 0] = cbar_n0_from_unnormalized_jn(8.46e-6, 3)
    cbar[3, 1], sbar[3, 1] = 2.6e-5, 5.5e-6
    cbar[3, 3], sbar[3, 3] = 1.2e-5, -2.0e-6
    return _model(cbar, sbar, nmax=3, mmax=3)


PROBE_STATES = [
    _polar_state(100e3),
    np.array([1.2e6, 1.1e6, 0.9e6, -0.9e3, 1.1e3, 0.4e3]),
    np.array([0.0, 0.0, R_MOON_M + 100e3, 1.6e3, 0.2e3, 0.0]),
]


class HarmonicsOffBitIdenticalTests(unittest.TestCase):
    # 1 -- RHS bit-identical with and without the new kwargs -----------------
    def test_rhs_bit_identical(self):
        for j2m, j2e in ((0.0, 0.0), (J2, 0.0), (J2, J2_E)):
            for s in PROBE_STATES:
                a_old = f3body_moon(s, MU_M, MU_E, MU_S, R_ME, R_MS,
                                    j2_moon=j2m, j2_earth=j2e)
                a_new = f3body_moon(s, MU_M, MU_E, MU_S, R_ME, R_MS,
                                    j2_moon=j2m, j2_earth=j2e,
                                    harmonic_model=None,
                                    c_inertial_to_bf_harmonic=None)
                np.testing.assert_array_equal(a_old, a_new)

    # 2 -- propagation smoke unchanged ----------------------------------------
    def test_propagation_bit_identical(self):
        teval = np.arange(0.0, 1200.0 + 1, 60.0)
        p_old = propagate_state(teval, _polar_state(), MU_M, MU_E, MU_S, GE, GS,
                                j2_moon=J2)
        p_new = propagate_state(teval, _polar_state(), MU_M, MU_E, MU_S, GE, GS,
                                j2_moon=J2, harmonic_model=None,
                                harmonic_rotation=None)
        np.testing.assert_array_equal(p_old, p_new)


class C20BridgeTests(unittest.TestCase):
    # 3 -- acceleration bridge to the existing Moon-J2 path -------------------
    def test_acceleration_bridge(self):
        model = _c20_model()
        worst = 0.0
        for s in PROBE_STATES:
            a_j2 = f3body_moon(s, MU_M, MU_E, MU_S, R_ME, R_MS, j2_moon=J2)
            a_h = f3body_moon(s, MU_M, MU_E, MU_S, R_ME, R_MS, j2_moon=0.0,
                              harmonic_model=model,
                              c_inertial_to_bf_harmonic=_MCI_TO_MOON_BF)
            worst = max(worst, float(np.linalg.norm(a_h - a_j2)))
        self.assertLess(worst, 1e-14, msg=f"C20 splice bridge worst |da| = {worst:.3e}")

    # 4 -- trajectory bridge over a 2 h arc ------------------------------------
    def test_trajectory_bridge(self):
        teval = np.arange(0.0, 7200.0 + 1, 60.0)
        traj_j2 = propagate_state(teval, _polar_state(), MU_M, MU_E, MU_S, GE, GS,
                                  j2_moon=J2)
        traj_h = propagate_state(teval, _polar_state(), MU_M, MU_E, MU_S, GE, GS,
                                 j2_moon=0.0, harmonic_model=_c20_model(),
                                 harmonic_rotation=_MCI_TO_MOON_BF)
        d_final = float(np.linalg.norm(traj_h[-1, :3] - traj_j2[-1, :3]))
        self.assertLess(d_final, 1e-3, msg=f"final pos diff = {d_final:.3e} m")


class CompositionGuardTests(unittest.TestCase):
    # 5 -- double-count: C20 model + j2_moon != 0 forbidden --------------------
    def test_double_count_valueerror(self):
        teval = np.arange(0.0, 300.0 + 1, 60.0)
        with self.assertRaises(ValueError) as ctx:
            propagate_state(teval, _polar_state(), MU_M, MU_E, MU_S, GE, GS,
                            j2_moon=J2, harmonic_model=_c20_model(),
                            harmonic_rotation=_MCI_TO_MOON_BF)
        self.assertIn("twice", str(ctx.exception))
        # rule precision: a model WITHOUT C20 may combine with j2_moon
        t_grid, grid = _synthetic_rotation_pair(-120.0, 300.0 + 120.0)
        out = propagate_state(teval, _polar_state(), MU_M, MU_E, MU_S, GE, GS,
                              j2_moon=J2, harmonic_model=_c22_only_model(),
                              harmonic_rotation=(t_grid, grid))
        self.assertTrue(bool(np.all(np.isfinite(out))))

    # 6 -- Earth J2 + lunar harmonics stays composable --------------------------
    def test_earth_j2_plus_harmonics_allowed(self):
        teval = np.arange(0.0, 600.0 + 1, 60.0)
        base = propagate_state(teval, _polar_state(), MU_M, MU_E, MU_S, GE, GS,
                               harmonic_model=_c20_model(),
                               harmonic_rotation=_MCI_TO_MOON_BF)
        both = propagate_state(teval, _polar_state(), MU_M, MU_E, MU_S, GE, GS,
                               j2_earth=J2_E, harmonic_model=_c20_model(),
                               harmonic_rotation=_MCI_TO_MOON_BF)
        self.assertTrue(bool(np.all(np.isfinite(both))))
        self.assertGreater(float(np.linalg.norm(both[-1, :3] - base[-1, :3])), 0.0)


class StmGuardTests(unittest.TestCase):
    def _aug0(self):
        return np.concatenate([_polar_state(), np.eye(6).reshape(-1, order="F")])

    # 7 -- STM + harmonics -> explicit error -----------------------------------
    def test_stm_with_harmonics_raises(self):
        teval = np.arange(0.0, 300.0 + 1, 60.0)
        with self.assertRaises(ValueError) as ctx:
            propagate_augmented_state(teval, self._aug0(), MU_M, MU_E, MU_S, GE, GS,
                                      harmonic_model=_c20_model())
        self.assertIn("gradient not implemented", str(ctx.exception))
        self.assertIn("6-state", str(ctx.exception))

    # 8 -- STM harmonics-off unchanged ------------------------------------------
    def test_stm_without_harmonics_unchanged(self):
        teval = np.arange(0.0, 600.0 + 1, 120.0)
        a = propagate_augmented_state(teval, self._aug0(), MU_M, MU_E, MU_S, GE, GS,
                                      j2_moon=J2)
        b = propagate_augmented_state(teval, self._aug0(), MU_M, MU_E, MU_S, GE, GS,
                                      j2_moon=J2, harmonic_model=None)
        np.testing.assert_array_equal(a, b)


class RotationGuardTests(unittest.TestCase):
    # 9 -- rotation guards: coverage / missing / m>0-with-constant --------------
    def test_grid_coverage_insufficient_raises(self):
        teval = np.arange(0.0, 1200.0 + 1, 60.0)
        t_grid, grid = _synthetic_rotation_pair(0.0, 600.0)     # too short
        with self.assertRaises(ValueError) as ctx:
            propagate_state(teval, _polar_state(), MU_M, MU_E, MU_S, GE, GS,
                            harmonic_model=_c20_c22_model(),
                            harmonic_rotation=(t_grid, grid))
        self.assertIn("cover", str(ctx.exception))

    def test_rotation_missing_raises(self):
        teval = np.arange(0.0, 300.0 + 1, 60.0)
        with self.assertRaises(ValueError):
            propagate_state(teval, _polar_state(), MU_M, MU_E, MU_S, GE, GS,
                            harmonic_model=_c20_model())

    def test_constant_rotation_with_tesseral_raises(self):
        teval = np.arange(0.0, 300.0 + 1, 60.0)
        with self.assertRaises(ValueError) as ctx:
            propagate_state(teval, _polar_state(), MU_M, MU_E, MU_S, GE, GS,
                            harmonic_model=_c20_c22_model(),
                            harmonic_rotation=_MCI_TO_MOON_BF)
        self.assertIn("m > 0", str(ctx.exception))

    def test_malformed_rotation_pair_raises(self):
        teval = np.arange(0.0, 300.0 + 1, 60.0)
        t_grid, grid = _synthetic_rotation_pair(-120.0, 420.0)
        with self.assertRaises(ValueError):        # length mismatch
            propagate_state(teval, _polar_state(), MU_M, MU_E, MU_S, GE, GS,
                            harmonic_model=_c20_model(),
                            harmonic_rotation=(t_grid[:-1], grid))
        with self.assertRaises(ValueError):        # non-monotonic grid
            propagate_state(teval, _polar_state(), MU_M, MU_E, MU_S, GE, GS,
                            harmonic_model=_c20_model(),
                            harmonic_rotation=(t_grid[::-1], grid))

    # 10 -- margined grid propagates cleanly --------------------------------------
    def test_grid_with_margin_works(self):
        teval = np.arange(0.0, 1800.0 + 1, 60.0)
        t_grid, grid = _synthetic_rotation_pair(-120.0, 1800.0 + 120.0)
        out = propagate_state(teval, _polar_state(), MU_M, MU_E, MU_S, GE, GS,
                              harmonic_model=_c20_c22_model(),
                              harmonic_rotation=(t_grid, grid))
        self.assertEqual(out.shape, (teval.size, 6))
        self.assertTrue(bool(np.all(np.isfinite(out))))


class ParityAndPhysicsTests(unittest.TestCase):
    # 11 -- Python vs Numba harmonics RHS parity -----------------------------------
    @unittest.skipUnless(NUMBA_AVAILABLE, "Numba unavailable: parity NOT validated")
    def test_python_numba_rhs_parity(self):
        model = _c20_c22_model()
        worst = 0.0
        for t_s in (0.0, 3600.0, 40000.0):
            c_t = _rotz(LUNAR_RATE_RAD_S * t_s)
            for s in PROBE_STATES:
                a_py = f3body_moon(s, MU_M, MU_E, MU_S, R_ME, R_MS,
                                   j2_earth=J2_E, harmonic_model=model,
                                   c_inertial_to_bf_harmonic=c_t)
                a_nb = f3body_harmonics_rhs(
                    s, MU_M, MU_E, MU_S, R_ME, R_MS,
                    0.0, 0.0, None, J2_E, 6378136.3, 1, None,
                    model.cbar, model.sbar, model.mu_m3_s2, model.r_ref_m,
                    model.nmax, model.mmax, c_t,
                )
                diff = float(np.linalg.norm(a_nb - a_py))
                ref = float(np.linalg.norm(a_py))
                worst = max(worst, diff / ref if ref > 1e-18 else diff)
        self.assertLess(worst, 1e-12, msg=f"Py<->Numba RHS parity worst = {worst:.3e}")

    # 12 -- nmax/mmax truncation reaches the dynamics path ---------------------------
    def test_truncation_reaches_dynamics(self):
        full = _c20_c22_model()
        trunc = SphericalHarmonicGravityModel(
            mu_m3_s2=MU_M, r_ref_m=R_MOON_M, cbar=full.cbar.copy(),
            sbar=full.sbar.copy(), nmax=2, mmax=0)     # same arrays, zonal cut
        c20 = _c20_model()
        s = PROBE_STATES[1]
        a_trunc = f3body_moon(s, MU_M, MU_E, MU_S, R_ME, R_MS,
                              harmonic_model=trunc,
                              c_inertial_to_bf_harmonic=_MCI_TO_MOON_BF)
        a_c20 = f3body_moon(s, MU_M, MU_E, MU_S, R_ME, R_MS,
                            harmonic_model=c20,
                            c_inertial_to_bf_harmonic=_MCI_TO_MOON_BF)
        np.testing.assert_array_equal(a_trunc, a_c20)   # mmax=0 == C20-only, bit-exact

    # 13 -- C22 produces a real, nonzero trajectory difference ------------------------
    def test_c22_differs_from_c20(self):
        teval = np.arange(0.0, 1800.0 + 1, 60.0)
        t_grid, grid = _synthetic_rotation_pair(-120.0, 1800.0 + 120.0)
        traj22 = propagate_state(teval, _polar_state(), MU_M, MU_E, MU_S, GE, GS,
                                 harmonic_model=_c20_c22_model(),
                                 harmonic_rotation=(t_grid, grid))
        traj20 = propagate_state(teval, _polar_state(), MU_M, MU_E, MU_S, GE, GS,
                                 harmonic_model=_c20_model(),
                                 harmonic_rotation=(t_grid, grid))
        d_final = float(np.linalg.norm(traj22[-1, :3] - traj20[-1, :3]))
        self.assertGreater(d_final, 0.1, msg=f"C22 effect = {d_final:.3e} m")
        self.assertTrue(bool(np.all(np.isfinite(traj22))))

    # 14 -- low-altitude polar smoke: finite, no NaN ------------------------------------
    def test_low_altitude_polar_finite(self):
        teval = np.arange(0.0, 1200.0 + 1, 60.0)
        t_grid, grid = _synthetic_rotation_pair(-120.0, 1200.0 + 120.0)
        out = propagate_state(teval, _polar_state(50e3), MU_M, MU_E, MU_S, GE, GS,
                              harmonic_model=_nominal3_model(),
                              harmonic_rotation=(t_grid, grid))
        self.assertTrue(bool(np.all(np.isfinite(out))))
        radii = np.linalg.norm(out[:, :3], axis=1)
        self.assertGreater(float(radii.min()), R_MOON_M)   # no surface crash


if __name__ == "__main__":
    unittest.main()
