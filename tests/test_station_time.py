"""Phase C2-I: src/lunar_od/station_time.py.

The series tests need no SPICE (the SOFA test-suite value of eraDtdb, the generated table's shape, the
analytic rate). The UTC tests read TAI - UTC from the loaded leapseconds kernel. The full qualification
against pyerfa (G2, G3) is studies/phase17/scripts/phase_c2i_station_time_gates.py.
"""
from __future__ import annotations

import unittest

import numpy as np

from lunar_od import _erfa_dtdb_table as T
from lunar_od import station_time as S
from tests.test_estimator_lm_consistency import _SKIP_REASON, _SPICE_OK


class SeriesTests(unittest.TestCase):
    def test_sofa_reference_value(self):
        # SOFA/ERFA t_erfa test of eraDtdb
        got = float(S.dtdb(2448939.5, 0.123, 0.76543, 5.0123, 5525.242, 3190.0))
        self.assertLessEqual(abs(got - (-0.1280368005936998991e-2)), 1e-15)

    def test_generated_table_shape(self):
        self.assertEqual(len(T.FAIRHD), 787)
        self.assertEqual(T.GROUPS, ((0, 473), (474, 678), (679, 763), (764, 783), (784, 786)))
        self.assertEqual(len(T.TOPOCENTRIC), 10)
        self.assertEqual(len(T.JPL), 4)
        self.assertEqual(len(T.SOURCE_SHA256), 64)

    def test_geocentric_has_no_site_dependence_on_ut(self):
        a = S.dtdb(2451545.0, 0.3, 0.1)
        b = S.dtdb(2451545.0, 0.3, 0.7)
        self.assertEqual(float(a), float(b))

    def test_rate_matches_a_central_difference(self):
        d1, d2 = 2451545.0 + 9928, np.linspace(0.0, 1.0, 7)
        # h = 1 s: the diurnal topocentric term's truncation (omega h)^2 / 6 is ~1e-9; eraDtdb's own epoch sum
        # (date1 - J2000) + date2 is quantised at ~1.6e-7 s, i.e. ~1e-17 s/s of difference noise
        h = 1.0 / 86400.0
        for site in ((0.0, 0.0, 0.0), (-2.03, 5204.0, 3666.0)):
            fd = (S.dtdb(d1, d2 + h, np.mod(d2 + 0.5 + h, 1.0), *site)
                  - S.dtdb(d1, d2 - h, np.mod(d2 + 0.5 - h, 1.0), *site)) / 2.0
            an = S.dtdb_rate(d1, d2, np.mod(d2 + 0.5, 1.0), *site)
            np.testing.assert_allclose(an, fd, rtol=1e-6, atol=5e-17)

    def test_site_from_itrf(self):
        s = S.Site.from_itrf_m([3000e3, 4000e3, 1000e3])
        self.assertAlmostEqual(s.u_km, 5000.0)
        self.assertAlmostEqual(s.v_km, 1000.0)
        self.assertAlmostEqual(s.elong_rad, np.arctan2(4.0, 3.0))

    def test_calendar_mjd(self):
        for (y, m, d), mjd in (((2000, 1, 1), 51544), ((1972, 6, 30), 41498), ((1972, 7, 1), 41499),
                               ((2026, 10, 22), 61335), ((1999, 3, 1), 51238), ((2024, 2, 29), 60369)):
            self.assertEqual(S._mjd(y, m, d), mjd)


@unittest.skipUnless(_SPICE_OK, _SKIP_REASON)
class UtcTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lunar_od.spice_loader import load_spice_kernels
        load_spice_kernels()

    def test_leap_second_boundary(self):
        midnight_2017 = (S._mjd(2017, 1, 1) - 51544) * 86400.0 - 43200.0
        self.assertEqual(float(S.tai_minus_utc(midnight_2017 - 1.0)), 36.0)
        self.assertEqual(float(S.tai_minus_utc(midnight_2017)), 37.0)

    def test_j2000_tag(self):
        # 2000-01-01T11:58:55.816 UTC = 12:00:00 TT (TAI - UTC = 32 s); TDB = TT + dtdb
        got = float(S.utc_to_tdb_seconds(["2000-01-01T11:58:55.816"], 0.0)[0])
        expect = float(S.dtdb(2451545.0, 0.0, 0.0))
        self.assertLessEqual(abs(got - expect), 1e-9)

    def test_matches_spice_chain_to_tens_of_microseconds(self):
        import spiceypy as spice
        tag = "2027-03-08T12:00:00"
        et0 = float(spice.str2et(tag))
        full = float(S.utc_to_tdb_seconds([tag], et0)[0])
        spice_chain = float(S.spice_utc_chain_tdb_seconds([tag], et0)[0])
        self.assertLessEqual(abs(spice_chain), 1e-6)           # SPICE's own str2et, relative to itself
        self.assertLess(abs(full - spice_chain), 3e-5)         # DELTET vs Fairhead-Bretagnon
        self.assertGreater(abs(full - spice_chain), 1e-6)

    def test_tuple_and_string_tags_agree(self):
        a = S.utc_to_tdb_seconds(["2027-03-08T01:02:03.25"], 8.578e8)
        b = S.utc_to_tdb_seconds([(2027, 3, 8, 3723.25)], 8.578e8)
        self.assertEqual(float(a[0]), float(b[0]))

    def test_pre_1972_is_rejected(self):
        with self.assertRaises(ValueError):
            S.utc_to_tdb_seconds(["1971-12-31T00:00:00"], 0.0)


if __name__ == "__main__":
    unittest.main()
