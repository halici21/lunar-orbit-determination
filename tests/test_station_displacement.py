"""Phase C3-I: src/lunar_od/station_frames.py, src/lunar_od/solid_tide.py and the station-displacement wrapper.

Qualification against Orekit (G2, G3) and the C3 survey (G4) is studies/phase17/scripts/phase_c3i_qualification.py.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from lunar_od import station_frames as SF
from lunar_od import solid_tide as TD
from tests.test_estimator_lm_consistency import _SKIP_REASON, _SPICE_OK

GOLDSTONE = np.array([-2353621.0, -4641341.0, 3677052.0])


class HelmertTests(unittest.TestCase):
    def test_reference_epoch_parameters(self):
        t, d, r = SF.helmert_parameters(SF._ET_2015)
        np.testing.assert_allclose(t, [-0.0658, 0.0019, -0.0713], atol=1e-15)
        self.assertAlmostEqual(d, 4.47e-9, places=20)
        np.testing.assert_allclose(r / SF._MAS, [-3.36, -4.33, 0.75], atol=1e-12)

    def test_position_vector_convention(self):
        t, d, (r1, r2, r3) = SF.helmert_parameters(SF._ET_2015)
        x, y, z = GOLDSTONE
        expect = GOLDSTONE + t + d * GOLDSTONE + np.array([-r3 * y + r2 * z, r3 * x - r1 * z, -r2 * x + r1 * y])
        np.testing.assert_allclose(SF.itrf2020_to_itrf93(GOLDSTONE, SF._ET_2015), expect, atol=1e-9)

    def test_scale_switch(self):
        et = SF._ET_2015 + 12 * SF._YEAR_S
        diff = SF.itrf2020_to_itrf93(GOLDSTONE, et) - SF.itrf2020_to_itrf93(GOLDSTONE, et, include_scale=False)
        # a difference of two ~6e6 m vectors: the double floor is ~1e-9 m
        np.testing.assert_allclose(diff, (4.47 + 0.12 * 12) * 1e-9 * GOLDSTONE, atol=1e-8)

    def test_vectorised(self):
        both = SF.itrf2020_to_itrf93(np.vstack([GOLDSTONE, -GOLDSTONE]), SF._ET_2015)
        np.testing.assert_allclose(both[0], SF.itrf2020_to_itrf93(GOLDSTONE, SF._ET_2015), atol=1e-12)

    def test_predict_kernel_name(self):
        self.assertIsNotNone(SF._PREDICT.search("earth_2025_250826_2125_predict.bpc"))
        self.assertIsNone(SF._PREDICT.search("earth_latest_high_prec.bpc"))


class SolidTideTests(unittest.TestCase):
    def _tide(self, sun, moon):
        return TD.solid_tide_displacement_ecef(GOLDSTONE, sun, moon, 61471.0, 0.25, mass_ratio_sun=332946.0482,
                                               mass_ratio_moon=0.0123000371)

    def test_magnitude_is_decimetres(self):
        r = GOLDSTONE / np.linalg.norm(GOLDSTONE)
        d = self._tide(1.496e11 * r, 3.844e8 * r)            # both bodies at the zenith: maximal uplift
        self.assertGreater(float(d @ r), 0.25)
        self.assertLess(float(np.linalg.norm(d)), 0.6)

    def test_zenith_tide_is_mostly_radial(self):
        r = GOLDSTONE / np.linalg.norm(GOLDSTONE)
        d = self._tide(1.496e11 * r, 3.844e8 * r)
        self.assertLess(float(np.linalg.norm(d - (d @ r) * r)), 0.02)

    def test_doodson_arguments_are_angles(self):
        a = TD.doodson_arguments(0.27, 6.0)
        self.assertEqual(a.shape, (6,))
        self.assertTrue(np.all((a >= 0.0) & (a < 2 * math.pi)))


@unittest.skipUnless(_SPICE_OK, _SKIP_REASON)
class SpiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lunar_od.spice_loader import load_spice_kernels
        load_spice_kernels()

    def test_predicted_flag(self):
        import spiceypy as spice
        self.assertTrue(SF.is_predicted_earth_orientation(float(spice.str2et("2027-03-08T00:00:00"))))
        self.assertFalse(SF.is_predicted_earth_orientation(float(spice.str2et("2025-03-01T00:00:00"))))

    def test_tide_at_et_is_smooth_and_bounded(self):
        import spiceypy as spice
        et = float(spice.str2et("2027-03-08T00:00:00"))
        d = np.array([TD.solid_tide_at_et(GOLDSTONE, et + 600.0 * k) for k in range(144)])
        self.assertLess(float(np.max(np.linalg.norm(d, axis=1))), 0.6)
        self.assertGreater(float(np.ptp(d @ (GOLDSTONE / np.linalg.norm(GOLDSTONE)))), 0.1)
        self.assertLess(float(np.max(np.linalg.norm(np.diff(d, axis=0), axis=1))), 0.05)

    def test_wrapper_identity_and_velocity(self):
        import spiceypy as spice
        from lunar_od.relativistic_light_time import displaced_station_state
        et0 = float(spice.str2et("2027-03-08T00:00:00"))

        def fixed(t):
            x = np.asarray(spice.sxform("ITRF93", "J2000", et0 + t))
            return x @ np.concatenate([GOLDSTONE, np.zeros(3)])
        plain = displaced_station_state(fixed, et0, GOLDSTONE, solid_tide=False, coordinate_frame="ITRF93")
        np.testing.assert_array_equal(plain(100.0), fixed(100.0))
        moved = displaced_station_state(fixed, et0, GOLDSTONE, solid_tide=True, coordinate_frame="ITRF2020")
        h = 5.0
        fd = (moved(100.0 + h)[:3] - moved(100.0 - h)[:3] - (fixed(100.0 + h)[:3] - fixed(100.0 - h)[:3])) / (2 * h)
        np.testing.assert_allclose(moved(100.0)[3:] - fixed(100.0)[3:], fd, atol=1e-7)
        self.assertGreater(float(np.linalg.norm(moved(100.0)[:3] - fixed(100.0)[:3])), 0.1)
        with self.assertRaises(ValueError):
            displaced_station_state(fixed, et0, GOLDSTONE, coordinate_frame="ITRF2014")


if __name__ == "__main__":
    unittest.main()
