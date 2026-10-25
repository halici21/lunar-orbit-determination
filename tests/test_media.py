"""Phase C5-I: src/lunar_od/media.py and the media hooks of relativistic_light_time.

Qualification against Orekit (G2, G3) and the C5 survey (G4) is studies/phase17/scripts/phase_c5i_qualification.py.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from lunar_od import media as M
from tests.test_estimator_lm_consistency import _SKIP_REASON, _SPICE_OK


class FormulaTests(unittest.TestCase):
    def test_zhd_iers(self):
        # IERS 2010 eq. 9.11 at sea level, 45 deg, 1013.25 hPa
        self.assertAlmostEqual(M.zhd_iers(1013.25, math.radians(45.0), 0.0), 0.0022768 * 1013.25, places=12)
        self.assertGreater(M.zhd_iers(M.standard_pressure_hpa(1000.0), 0.6, 1000.0), 2.0)

    def test_mapping_functions_at_zenith_and_low_elevation(self):
        mh, mw = M.niell_mapping(math.pi / 2, math.radians(40.0), 0.0, 100.0)
        self.assertAlmostEqual(mh, 1.0, places=9)
        self.assertAlmostEqual(mw, 1.0, places=9)
        mh10, mw10 = M.niell_mapping(math.radians(10.0), math.radians(40.0), 0.0, 100.0)
        self.assertTrue(5.0 < mh10 < 6.0 and 5.0 < mw10 < 6.0)
        self.assertAlmostEqual(M.black_eisner_mapping(math.pi / 2), 1.001 / math.sqrt(1.002001), places=12)

    def test_niell_height_correction_and_hemispheres(self):
        el = math.radians(10.0)
        self.assertGreater(M.niell_mapping(el, 0.7, 2000.0, 50.0)[0], M.niell_mapping(el, 0.7, 0.0, 50.0)[0])
        # the southern hemisphere uses the season shifted by 183 days
        self.assertAlmostEqual(M.niell_mapping(el, -0.7, 0.0, 50.0 + 183.0)[0], M.niell_mapping(el, 0.7, 0.0, 50.0)[0],
                               places=12)

    def test_single_layer_and_iono(self):
        self.assertAlmostEqual(M.single_layer_mapping(math.pi / 2), 1.0, places=12)
        self.assertGreater(M.single_layer_mapping(math.radians(10.0)), 2.0)
        self.assertAlmostEqual(M.iono_delay_m(30.0, 8.4e9), 40.3e16 * 30.0 / 8.4e9 ** 2, places=15)
        self.assertEqual(M.default_vtec_tecu(2.0), 5.0)
        self.assertAlmostEqual(M.default_vtec_tecu(14.0), 30.0)

    def test_model_validation_and_metadata(self):
        with self.assertRaises(ValueError):
            M.MediaModel(mapping="GMF")
        with self.assertRaises(ValueError):
            M.MediaModel(vtec_tecu="IONEX")
        m = M.MediaModel(zwd_m=0.2, vtec_tecu=12.0)
        self.assertEqual(M.MediaModel.from_metadata(m.to_metadata()), m)


@unittest.skipUnless(_SPICE_OK, _SKIP_REASON)
class LegTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lunar_od.spice_loader import load_spice_kernels
        load_spice_kernels()

    def test_group_and_phase(self):
        import spiceypy as spice
        et = float(spice.str2et("2027-03-08T12:00:00"))
        site = M.MediaSite(math.radians(35.4), math.radians(-116.9), 1000.0)
        up = np.asarray(spice.pxform("ITRF93", "J2000", et)) @ site.up_ecef
        g, p, el = M.MediaModel(vtec_tecu=20.0).leg_delays(site, et, up * 4e8, uplink=False)
        self.assertAlmostEqual(el, math.pi / 2, places=6)
        tropo = 0.5 * (g + p)
        iono = 0.5 * (g - p)
        self.assertAlmostEqual(iono, M.iono_delay_m(20.0, 7.2e9 * 880.0 / 749.0), places=9)
        self.assertTrue(2.0 < tropo < 2.5)
        g0, p0, _ = M.MediaModel(troposphere=False, vtec_tecu=None).leg_delays(site, et, up, uplink=True)
        self.assertEqual((g0, p0), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
