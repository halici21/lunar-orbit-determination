"""Phase C2-I additions to src/lunar_od/relativistic_light_time.py and src/lunar_od/consider_covariance.py.

Synthetic tests need no SPICE (model validation, metadata defaults, the consider covariance identities). The
rest reuse the short LLO two-way pass of tests/test_relativistic_light_time.py: the C2 station-time relation
changes range by millimetres to centimetres, needs the station site, and the clock-offset partials match a
finite difference of the offset. Population-level qualification is
studies/phase17/scripts/phase_c2i_qualification.py.
"""
from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from lunar_od import relativistic_light_time as R
from lunar_od import station_time as ST
from lunar_od.consider_covariance import consider_covariance
from tests.test_estimator_lm_consistency import _SKIP_REASON, _SPICE_OK, _truth_setup


class ModelTests(unittest.TestCase):
    def test_default_is_the_c1i_relation(self):
        self.assertEqual(R.LightTimeModel().station_time, "DELTET_GEOCENTRIC")
        self.assertEqual(R.DSN_LIGHT_TIME_MODEL.station_time, "DELTET_GEOCENTRIC")
        self.assertTrue(R.DSN_C2_LIGHT_TIME_MODEL.full_station_time)

    def test_full_station_time_needs_tt(self):
        with self.assertRaises(ValueError):
            R.LightTimeModel(frame="SSB", station_time="FULL_TOPOCENTRIC")
        with self.assertRaises(ValueError):
            R.LightTimeModel(station_time="UTC")

    def test_old_metadata_reads_as_c1i(self):
        meta = R._model_metadata(R.DSN_LIGHT_TIME_MODEL)
        del meta["station_time"]
        pg = SimpleNamespace(measurement_metadata={"light_time_model": meta})
        self.assertEqual(R.pass_light_time_model(pg), R.DSN_LIGHT_TIME_MODEL)
        pg2 = SimpleNamespace(measurement_metadata={"light_time_model": R._model_metadata(R.DSN_C2_LIGHT_TIME_MODEL)})
        self.assertEqual(R.pass_light_time_model(pg2), R.DSN_C2_LIGHT_TIME_MODEL)


class ConsiderTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.a_x = np.vstack([rng.normal(size=(40, 5)) * [1.0, 10.0, 1e3, 1e-2, 3.0], np.diag([1e-2] * 5)])
        self.a_c = np.vstack([rng.normal(size=(40, 2)) * [5.0, 0.1], np.zeros((5, 2))])
        self.p_cc = np.diag([0.3, 2.0]) ** 2

    def test_matches_dense_formulas(self):
        res = consider_covariance(self.a_x, self.a_c, self.p_cc)
        p = np.linalg.inv(self.a_x.T @ self.a_x)
        s = -p @ self.a_x.T @ self.a_c
        np.testing.assert_allclose(res.p_x, p, rtol=1e-9)
        np.testing.assert_allclose(res.sensitivity, s, rtol=1e-8)
        np.testing.assert_allclose(res.p_consider, p + s @ self.p_cc @ s.T, rtol=1e-8)

    def test_sensitivity_is_minus_the_bias_response(self):
        res = consider_covariance(self.a_x, self.a_c, self.p_cc)
        for j in range(2):
            dx = np.linalg.lstsq(self.a_x, self.a_c[:, j], rcond=None)[0]
            np.testing.assert_allclose(-res.sensitivity[:, j], dx, rtol=1e-8)

    def test_zero_consider_covariance_is_the_posterior(self):
        res = consider_covariance(self.a_x, self.a_c, np.zeros((2, 2)))
        self.assertTrue(np.array_equal(res.p_consider, res.p_x))

    def test_inflation_is_positive_semidefinite(self):
        res = consider_covariance(self.a_x, self.a_c, self.p_cc)
        ev = np.linalg.eigvalsh(res.p_consider - res.p_x)
        self.assertGreaterEqual(ev.min(), -1e-12 * ev.max())


@unittest.skipUnless(_SPICE_OK, _SKIP_REASON)
class StationTimeObservableTests(unittest.TestCase):
    """Short LLO pass, one DSN station (the LM-consistency fixture)."""

    @classmethod
    def setUpClass(cls):
        from lunar_od.config import Station
        from lunar_od.spice_loader import load_spice_kernels
        from lunar_od.two_way_range import _station_state_providers_for_pass, generate_two_way_range_measurements
        load_spice_kernels()
        cls.et0, cls.t, gep, gev, gsp, cls.x0, cls.aug = _truth_setup()
        station = Station(name="Goldstone DSN", lat_deg=35.30, lon_deg=-116.81, alt_m=969.67,
                          color_rgb=(0.85, 0.325, 0.098), sigma_range_m=5.0, sigma_angle_rad=math.radians(0.001))
        vis = np.ones((cls.t.size, 1), dtype=bool)
        cls.obs, cls.pg = generate_two_way_range_measurements(cls.t, cls.aug[:, :6], (station,), vis, gep, gev,
                                                              cls.et0, noise=False)
        prov = _station_state_providers_for_pass(cls.pg)[0]
        cls.station = staticmethod(R.exact_epoch_station_state(lambda t: np.asarray(prov.state(t), float), cls.et0,
                                                               cls.t, cls.pg.earth_pos_mci_m, cls.pg.earth_vel_mci_mps))
        cls.site = ST.Site.from_itrf_m(station.r_ecef_m)
        cls.eph = R.BarycentricEphemeris(float(cls.et0))
        cls.sc = R.SpacecraftHistory(cls.t, cls.aug[:, :6])

    def _range(self, t3, model, **kw):
        return R.two_way_range_m(R.solve_two_way(float(t3), self.station, self.sc, self.eph, model, **kw))

    def test_c2_relation_needs_the_site(self):
        with self.assertRaises(ValueError):
            self._range(self.obs[0, 0], R.DSN_C2_LIGHT_TIME_MODEL)

    def test_site_is_ignored_by_the_c1i_relation(self):
        for t3 in self.obs[:5, 0]:
            self.assertEqual(self._range(t3, R.DSN_LIGHT_TIME_MODEL),
                             self._range(t3, R.DSN_LIGHT_TIME_MODEL, site=self.site))

    def test_c2_relation_moves_range_by_millimetres_to_centimetres(self):
        d = [self._range(t3, R.DSN_C2_LIGHT_TIME_MODEL, site=self.site) - self._range(t3, R.DSN_LIGHT_TIME_MODEL)
             for t3 in self.obs[:, 0]]
        self.assertGreater(max(abs(x) for x in d), 1e-4)
        self.assertLess(max(abs(x) for x in d), 0.2)

    def test_range_clock_partial_matches_a_finite_difference(self):
        m, h = R.DSN_C2_LIGHT_TIME_MODEL, 1e-3
        for t3 in self.obs[::5, 0]:
            sol = R.solve_two_way(float(t3), self.station, self.sc, self.eph, m, site=self.site)
            fd = (self._range(t3, m, site=self.site, clock_offset_s=h)
                  - self._range(t3, m, site=self.site, clock_offset_s=-h)) / (2 * h)
            self.assertLessEqual(abs(R.two_way_range_clock_partial(sol) - fd), 1e-6 * abs(fd) + 1e-6)

    def test_doppler_clock_partial_matches_a_finite_difference(self):
        m, h, tc = R.DSN_C2_LIGHT_TIME_MODEL, 1e-2, 60.0
        for tm in self.obs[2:-2:5, 0]:
            _, _, _, an = R.counted_doppler(float(tm), tc, self.station, self.sc, self.eph, m, site=self.site,
                                            with_clock_partial=True)
            fd = (R.counted_doppler(float(tm), tc, self.station, self.sc, self.eph, m, site=self.site,
                                    clock_offset_s=h)[0]
                  - R.counted_doppler(float(tm), tc, self.station, self.sc, self.eph, m, site=self.site,
                                      clock_offset_s=-h)[0]) / (2 * h)
            self.assertLessEqual(abs(an - fd), 1e-5 * abs(fd) + 1e-7)


if __name__ == "__main__":
    unittest.main()
