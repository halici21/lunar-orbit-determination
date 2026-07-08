"""Phase 13A -- MOON_PA rotation provider verification.

SPICE-dependent tests sample real J2000->MOON_PA rotations (kernels
``moon_080317.tf`` + ``moon_pa_de421_1900-2050.bpc`` via the standard
resolver).  They are NOT skipped when kernels are missing: they fail with the
explicit "kernel unavailable" FileNotFoundError instead, because Phase 13A
acceptance requires the kernels to be genuinely present and exercised.

The nearest-neighbour lookup and matrix-validation tests are pure numpy (no
SPICE, no kernels) and always run.

No dynamics splice here: this file exercises the provider only.
"""
import inspect
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lunar_od.lunar_frames import (
    nearest_rotation_at_time,
    sample_moon_pa_rotations,
    validate_rotation_matrix,
)
from lunar_od.spice_loader import required_kernel_paths, resolve_kernel_dir

# Campaign epoch (phase6): first_jd = 2461467.480984 TDB (2027-03-02).
ET0 = (2461467.480984 - 2451545.0) * 86400.0
SIDEREAL_MONTH_S = 27.321661 * 86400.0


def _rotz(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotation_angle(c_a: np.ndarray, c_b: np.ndarray) -> float:
    """Angle (rad) of the relative rotation between two rotation matrices."""
    rel = c_b @ c_a.T
    cos_angle = (np.trace(rel) - 1.0) / 2.0
    return float(math.acos(min(1.0, max(-1.0, cos_angle))))


class MoonPaSamplingTests(unittest.TestCase):
    """SPICE-dependent: real MOON_PA rotations on a 2-day hourly grid."""

    @classmethod
    def setUpClass(cls):
        # Raises the explicit resolver FileNotFoundError when unavailable.
        cls.kernel_dir = resolve_kernel_dir()
        cls.kernel_paths = required_kernel_paths()
        cls.t_grid = np.arange(0.0, 2.0 * 86400.0 + 1.0, 3600.0)   # 49 epochs
        cls.rots = sample_moon_pa_rotations(ET0, cls.t_grid)

    # 1 -- kernel discovery ------------------------------------------------
    def test_kernel_discovery(self):
        names = [p.name for p in self.kernel_paths]
        self.assertIn("moon_pa_de421_1900-2050.bpc", names)
        self.assertIn("moon_080317.tf.txt", names)
        self.assertTrue(self.kernel_dir.is_dir())

    # 2 -- shape / dtype ---------------------------------------------------
    def test_shape_dtype(self):
        self.assertEqual(self.rots.shape, (self.t_grid.size, 3, 3))
        self.assertEqual(self.rots.dtype, np.float64)

    # 3 -- finite ----------------------------------------------------------
    def test_finite(self):
        self.assertTrue(bool(np.all(np.isfinite(self.rots))))

    # 4 -- orthonormality --------------------------------------------------
    def test_orthonormality(self):
        worst = max(
            float(np.max(np.abs(c.T @ c - np.eye(3)))) for c in self.rots
        )
        self.assertLess(worst, 1e-10, msg=f"max |C^T C - I| = {worst:.3e}")

    # 5 -- determinant +1 ----------------------------------------------------
    def test_determinant_plus_one(self):
        dets = np.array([np.linalg.det(c) for c in self.rots])
        self.assertLess(float(np.max(np.abs(dets - 1.0))), 1e-10,
                        msg=f"det range [{dets.min():.15f}, {dets.max():.15f}]")

    # 6 -- rotation changes over time ----------------------------------------
    def test_time_variation_over_one_day(self):
        c0 = self.rots[0]
        c1day = self.rots[24]                      # +24 h
        angle = _rotation_angle(c0, c1day)
        expected = 2.0 * math.pi * 86400.0 / SIDEREAL_MONTH_S   # ~0.23 rad (13.2 deg)
        self.assertGreater(angle, math.radians(5.0))
        self.assertLess(angle, math.radians(25.0))
        # loose physical sanity: within a factor ~2 of the sidereal rate
        self.assertLess(abs(angle - expected) / expected, 1.0)

    # 7 -- convention round-trip ---------------------------------------------
    def test_round_trip_convention(self):
        vectors = [
            np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]), np.array([1837.4e3, -523.1e3, 912.7e3]),
        ]
        for c in (self.rots[0], self.rots[17], self.rots[-1]):
            for r in vectors:
                back = c.T @ (c @ r)
                self.assertLess(float(np.linalg.norm(back - r)),
                                1e-12 * max(1.0, float(np.linalg.norm(r))))

    # 15 -- nearest 60 s grid vs direct SPICE (robust bound, not flaky) ------
    def test_nearest_60s_grid_vs_direct_spice(self):
        t_grid = np.arange(0.0, 7200.0 + 1.0, 60.0)          # 2 h, 60 s cadence
        rots = sample_moon_pa_rotations(ET0, t_grid)
        t_query = 90.0                                        # exact midpoint
        c_near = nearest_rotation_at_time(rots, t_grid, t_query)
        c_true = sample_moon_pa_rotations(ET0 + t_query, np.array([0.0]),
                                          load_kernels=False)[0]
        angle = _rotation_angle(c_near, c_true)
        # 30 s of lunar rotation ~ 8.0e-5 rad; bound is >10x above that.
        self.assertLess(angle, 1e-3,
                        msg=f"nearest-60s vs direct SPICE angle = {angle:.3e} rad")

    # 13 -- missing kernel raises explicit error ------------------------------
    def test_missing_kernel_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                sample_moon_pa_rotations(ET0, np.array([0.0]), kernel_dir=tmp)

    # input validation of the sampler (no SPICE call happens: grid checked first)
    def test_sampler_rejects_bad_grids(self):
        with self.assertRaises(ValueError):
            sample_moon_pa_rotations(ET0, np.zeros((2, 2)))          # 2-D
        with self.assertRaises(ValueError):
            sample_moon_pa_rotations(ET0, np.array([]))              # empty
        with self.assertRaises(ValueError):
            sample_moon_pa_rotations(ET0, np.array([0.0, 60.0, 30.0]))  # non-monotonic


class NearestLookupTests(unittest.TestCase):
    """Pure numpy: no SPICE, no kernels."""

    def setUp(self):
        self.t_grid = np.array([0.0, 60.0, 120.0, 180.0])
        self.grid = np.stack([_rotz(0.01 * i) for i in range(4)])

    # 8 -- exact grid point returns that matrix ------------------------------
    def test_exact_grid_point(self):
        got = nearest_rotation_at_time(self.grid, self.t_grid, 120.0)
        np.testing.assert_array_equal(got, self.grid[2])

    # 9 -- midpoint tie resolves to the lower index ---------------------------
    def test_midpoint_tie_lower_index(self):
        got = nearest_rotation_at_time(self.grid, self.t_grid, 30.0)
        np.testing.assert_array_equal(got, self.grid[0])     # not grid[1]
        got2 = nearest_rotation_at_time(self.grid, self.t_grid, 150.0)
        np.testing.assert_array_equal(got2, self.grid[2])

    # 10 -- out-of-range raises ----------------------------------------------
    def test_out_of_range_raises(self):
        for t_bad in (-0.001, 180.001, -1e6):
            with self.assertRaises(ValueError):
                nearest_rotation_at_time(self.grid, self.t_grid, t_bad)

    # 11 -- non-monotonic grid raises -----------------------------------------
    def test_non_monotonic_raises(self):
        t_bad = np.array([0.0, 120.0, 60.0, 180.0])
        with self.assertRaises(ValueError):
            nearest_rotation_at_time(self.grid, t_bad, 90.0)

    # 12 -- length mismatch raises ---------------------------------------------
    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            nearest_rotation_at_time(self.grid[:3], self.t_grid, 90.0)

    # malformed shapes
    def test_bad_shapes_raise(self):
        with self.assertRaises(ValueError):
            nearest_rotation_at_time(np.zeros((4, 2, 2)), self.t_grid, 90.0)
        with self.assertRaises(ValueError):
            nearest_rotation_at_time(self.grid, self.t_grid.reshape(2, 2), 90.0)

    # 14 -- hot-path lookup must not touch SPICE -------------------------------
    def test_no_spice_in_lookup(self):
        # Inspect actual code usage (imports/calls), not docstring prose.
        src = inspect.getsource(nearest_rotation_at_time).lower()
        self.assertNotIn("import spice", src)
        self.assertNotIn("spice.", src)
        self.assertNotIn("spiceypy", src)
        self.assertNotIn("furnsh", src)


class ValidateRotationMatrixTests(unittest.TestCase):
    # 16 -- accepts proper rotations, rejects broken matrices -------------------
    def test_accepts_proper_rotations(self):
        validate_rotation_matrix(np.eye(3))
        validate_rotation_matrix(_rotz(0.3))

    def test_rejects_scaled(self):
        with self.assertRaises(ValueError):
            validate_rotation_matrix(2.0 * np.eye(3))

    def test_rejects_reflection(self):
        with self.assertRaises(ValueError):        # orthonormal but det = -1
            validate_rotation_matrix(np.diag([1.0, 1.0, -1.0]))

    def test_rejects_nonfinite_and_bad_shape(self):
        bad = np.eye(3); bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_rotation_matrix(bad)
        with self.assertRaises(ValueError):
            validate_rotation_matrix(np.eye(2))


if __name__ == "__main__":
    unittest.main()
