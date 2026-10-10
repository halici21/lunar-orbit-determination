"""Q1-F09: direct exact-epoch MOON_PA_DE440 frame in the high-order RHS.

The high-order path previously took its lunar orientation from a 60 s sampled
grid via nearest-neighbour lookup, quantising the frame to +/- 30 s. Direct
``pxform`` costs ~5 us against ~15.5 ms for one GL1800F 100x100 acceleration,
so the approximation bought nothing. These tests pin the replacement:

    ET(t_s) = harmonic_epoch_et0 + t_s        exact, no rounding, no grid

``harmonic_epoch_et0`` is the ET at propagation-relative ``t_s = 0`` -- NOT at
``t_eval_s[0]``, which is negative when a campaign uses pre-roll. Test B guards
that specific trap.
"""

from __future__ import annotations

import unittest

import numpy as np
import spiceypy as spice

from lunar_od import dynamics as dyn
from lunar_od import lunar_frames as lf
from lunar_od.gravity_model_loader import load_lunar_gravity_model
from lunar_od.spice_loader import load_spice_kernels

ET0 = 820497669.18392
GL1800F_TAB = (
    "C:/Users/erayh/Documents/Python/Grad/python_port/data/gravity/"
    "gl1800f/jggrx_1800f_sha.tab"
)
MU_MOON = 4902.800066e9


def _earth(t):
    return np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))


def _sun(t):
    return np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))


class DirectFrameHelper(unittest.TestCase):
    """TEST A -- the helper must BE pxform, not approximate it."""

    @classmethod
    def setUpClass(cls):
        load_spice_kernels(None, clear=False)

    def test_helper_matches_pxform_exactly(self):
        for t_s in (-400.0, 0.0, 0.123, 17.4, 29.999, 30.001, 60.1, 1234.567):
            et = ET0 + t_s
            got = lf.moon_pa_de440_rotation_at_et(et)
            ref = np.asarray(
                spice.pxform("J2000", "MOON_PA_DE440", float(et)), dtype=float
            )
            self.assertEqual(got.shape, (3, 3))
            np.testing.assert_array_equal(got, ref)          # bitwise
            np.testing.assert_allclose(got.T @ got, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(float(np.linalg.det(got)), 1.0, places=12)


class _Spy:
    """Records every ET the RHS asks the frame helper for."""

    def __init__(self):
        self.ets = []

    def __call__(self, et):
        self.ets.append(float(et))
        return np.eye(3)


class DirectRhsEpochContract(unittest.TestCase):
    """TESTS B-F -- the RHS must query et0 + t_s, once, and ignore the grid."""

    @classmethod
    def setUpClass(cls):
        load_spice_kernels(None, clear=False)
        # A small tesseral model: m > 0 is what makes the frame matter at all.
        import dataclasses

        _m = load_lunar_gravity_model(GL1800F_TAB, nmax=4, mmax=4)
        # The augmented path requires the MODEL LABEL to name an explicit PA
        # realization (a pre-existing guard against the generic MOON_PA alias).
        # GL1800F ships the descriptive label "MOON_PA"; the real high-order
        # callers set the realization explicitly, so the test does the same.
        # Production is not weakened -- the guard still fires for generic labels.
        cls.model = dataclasses.replace(_m, frame="MOON_PA_DE440")
        cls.x0 = np.array([1.8374e6, 0.0, 2.0e5, 0.0, 1.6e3, 0.0])
        cls.aug0 = np.concatenate([cls.x0, np.eye(6).reshape(-1, order="F")])

    def _spy_on_helper(self):
        spy = _Spy()
        self._orig = dyn.moon_pa_de440_rotation_at_et
        dyn.moon_pa_de440_rotation_at_et = spy
        self.addCleanup(setattr, dyn, "moon_pa_de440_rotation_at_et", self._orig)
        return spy

    def test_b_negative_pre_roll_epoch(self):
        """t_s = -400 must query et0 - 400, NOT et0."""
        spy = self._spy_on_helper()
        dyn.propagate_state(
            np.array([-400.0, -399.0]), self.x0, MU_MOON, 0.0, 0.0,
            _earth, _sun, harmonic_model=self.model, harmonic_epoch_et0=ET0,
        )
        self.assertTrue(spy.ets, "the direct helper was never called")
        first = min(spy.ets)
        self.assertLess(first, ET0, "negative t_s must map BELOW et0")
        self.assertAlmostEqual(first, ET0 - 400.0, delta=1.0)

    def test_c_six_and_augmented_use_the_same_epoch(self):
        spy6 = self._spy_on_helper()
        dyn.propagate_state(
            np.array([0.0, 5.0]), self.x0, MU_MOON, 0.0, 0.0,
            _earth, _sun, harmonic_model=self.model, harmonic_epoch_et0=ET0,
        )
        six = spy6.ets[0]
        dyn.moon_pa_de440_rotation_at_et = self._orig
        spy42 = self._spy_on_helper()
        dyn.propagate_augmented_state(
            np.array([0.0, 5.0]), self.aug0, MU_MOON, 0.0, 0.0,
            _earth, _sun, harmonic_model=self.model, harmonic_epoch_et0=ET0,
            harmonic_stm_opt_in=True,
        )
        self.assertAlmostEqual(six, spy42.ets[0], places=9)
        self.assertAlmostEqual(six, ET0, delta=1e-6)

    def test_e_direct_mode_never_touches_the_grid(self):
        """The whole point: no nearest-neighbour lookup in direct mode."""
        def _boom(*a, **k):
            raise AssertionError("direct mode consulted the rotation grid")

        orig = dyn.nearest_rotation_at_time
        dyn.nearest_rotation_at_time = _boom
        self.addCleanup(setattr, dyn, "nearest_rotation_at_time", orig)
        dyn.propagate_state(
            np.array([0.0, 5.0]), self.x0, MU_MOON, 0.0, 0.0,
            _earth, _sun, harmonic_model=self.model, harmonic_epoch_et0=ET0,
        )
        dyn.propagate_augmented_state(
            np.array([0.0, 5.0]), self.aug0, MU_MOON, 0.0, 0.0,
            _earth, _sun, harmonic_model=self.model, harmonic_epoch_et0=ET0,
            harmonic_stm_opt_in=True,
        )

    def test_f_central_and_j2_never_reach_de440(self):
        spy = self._spy_on_helper()
        calls = []
        orig_pre = dyn.require_moon_pa_de440
        dyn.require_moon_pa_de440 = lambda et: calls.append(et)
        self.addCleanup(setattr, dyn, "require_moon_pa_de440", orig_pre)

        dyn.propagate_state(                       # central only
            np.array([0.0, 10.0]), self.x0, MU_MOON, 0.0, 0.0, _earth, _sun,
        )
        dyn.propagate_state(                       # J2 only
            np.array([0.0, 10.0]), self.x0, MU_MOON, 0.0, 0.0, _earth, _sun,
            j2_moon=2.0346e-4,
        )
        self.assertEqual(spy.ets, [], "central/J2 queried the DE440 frame")
        self.assertEqual(calls, [], "central/J2 ran the DE440 preflight")

    def test_g_preflight_fails_before_the_solver(self):
        from lunar_od.spice_loader import LunarFrameKernelError

        solver_calls = []
        orig_pre = dyn.require_moon_pa_de440

        def _fail(et):
            raise LunarFrameKernelError("simulated missing DE440")

        dyn.require_moon_pa_de440 = _fail
        self.addCleanup(setattr, dyn, "require_moon_pa_de440", orig_pre)
        orig_helper = dyn.moon_pa_de440_rotation_at_et
        dyn.moon_pa_de440_rotation_at_et = lambda et: solver_calls.append(et)
        self.addCleanup(setattr, dyn, "moon_pa_de440_rotation_at_et", orig_helper)

        with self.assertRaises(LunarFrameKernelError):
            dyn.propagate_state(
                np.array([0.0, 10.0]), self.x0, MU_MOON, 0.0, 0.0,
                _earth, _sun, harmonic_model=self.model, harmonic_epoch_et0=ET0,
            )
        self.assertEqual(solver_calls, [], "integration started despite preflight failure")

    def test_non_finite_epoch_rejected(self):
        with self.assertRaises(ValueError):
            dyn.propagate_state(
                np.array([0.0, 10.0]), self.x0, MU_MOON, 0.0, 0.0,
                _earth, _sun, harmonic_model=self.model,
                harmonic_epoch_et0=float("nan"),
            )


class GridNodeParity(unittest.TestCase):
    """TEST H -- at exact legacy grid nodes, old and new must agree."""

    @classmethod
    def setUpClass(cls):
        load_spice_kernels(None, clear=False)

    def test_direct_equals_grid_sample_at_nodes(self):
        t_grid = np.arange(0.0, 660.0 + 1e-9, 60.0)
        grid = lf.sample_moon_pa_rotations(
            ET0, t_grid, frame="MOON_PA_DE440", load_kernels=False
        )
        for t_s in (0.0, 60.0, 120.0, 600.0):
            k = int(np.argmin(np.abs(t_grid - t_s)))
            direct = lf.moon_pa_de440_rotation_at_et(ET0 + t_s)
            np.testing.assert_array_equal(grid[k], direct)   # same SPICE call


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
