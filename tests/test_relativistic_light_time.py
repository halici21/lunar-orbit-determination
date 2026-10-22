"""Phase C1-I: src/lunar_od/relativistic_light_time.py.

Synthetic tests need no SPICE (model validation, the fixed-point two-cycle guard, the Shapiro gradient).
The rest reuse the short central-body two-way pass of tests/test_estimator_lm_consistency.py: the MOON model
must reproduce production, and the SSB/Shapiro/TT partials must match re-propagated finite differences.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from lunar_od import relativistic_light_time as R
from tests.test_estimator_lm_consistency import _MU_MOON, _SKIP_REASON, _SPICE_OK, _truth_setup


class ModelValidationTests(unittest.TestCase):
    def test_defaults_are_the_historical_model(self):
        m = R.LightTimeModel()
        self.assertEqual((m.frame, m.shapiro_bodies, m.interval_time_scale), ("MOON", (), "TDB"))

    def test_dsn_model(self):
        m = R.DSN_LIGHT_TIME_MODEL
        self.assertEqual((m.frame, m.shapiro_bodies, m.interval_time_scale), ("SSB", ("SUN", "EARTH"), "TT"))

    def test_invalid_configurations_raise(self):
        for kw in (dict(frame="GCRS"), dict(frame="MOON", shapiro_bodies=("SUN",)),
                   dict(frame="SSB", shapiro_bodies=("PLUTO",)), dict(interval_time_scale="UTC")):
            with self.assertRaises(ValueError):
                R.LightTimeModel(**kw)


class TwoCycleGuardTests(unittest.TestCase):
    """Amendment A2 (b): a map with a position step settles into a two-cycle of amplitude step / c."""

    @staticmethod
    def _stepped_map(step_m):
        tau_star = 1.28

        def update(tau):
            return tau_star + (step_m / R.C_LIGHT_MPS if tau <= tau_star else 0.0)
        return update, tau_star

    def test_millimetre_step_is_accepted_and_flagged(self):
        update, tau_star = self._stepped_map(1e-3)
        tau, flagged = R._fixed_point(update, 1.0, R.LightTimeModel(), "test")
        self.assertTrue(flagged)
        self.assertLess(abs(tau - tau_star), 1e-11)

    def test_metre_step_raises(self):
        update, _ = self._stepped_map(1.0)
        with self.assertRaises(R.LightTimeConvergenceError):
            R._fixed_point(update, 1.0, R.LightTimeModel(), "test")

    def test_contraction_converges_unflagged(self):
        tau, flagged = R._fixed_point(lambda t: 1.0 + 1e-5 * t, 0.0, R.LightTimeModel(), "test")
        self.assertFalse(flagged)
        self.assertAlmostEqual(tau, 1.0 / (1.0 - 1e-5), delta=1e-14)


class _FakeEphemeris:
    def __init__(self):
        self.bodies = {"SUN": np.array([1.4e11, 3e10, 1e9]), "EARTH": np.array([-2e10, 1.45e11, 2e9])}

    def body_position(self, name, t_s):
        return self.bodies[name]

    def gm(self, name):
        return {"SUN": 1.32712440041939e20, "EARTH": 3.986004354360959e14}[name]


class ShapiroTests(unittest.TestCase):
    def test_gradient_matches_finite_differences(self):
        eph = _FakeEphemeris()
        m = R.LightTimeModel(frame="SSB", shapiro_bodies=("SUN", "EARTH"))
        r_tx = eph.bodies["EARTH"] + np.array([6.371e6, 1e5, -2e5])
        r_rx = r_tx + np.array([3.0e8, 2.2e8, 1.0e7])
        s0, g_tx, g_rx = R.shapiro_delay_s(m, eph, r_tx, r_rx, 0.0)
        self.assertGreater(s0, 1e-9)
        for which, grad in (("tx", g_tx), ("rx", g_rx)):
            fd = np.zeros(3)
            for j in range(3):
                e = np.zeros(3)
                e[j] = 100.0
                if which == "tx":
                    fd[j] = (R.shapiro_delay_s(m, eph, r_tx + e, r_rx, 0.0)[0]
                             - R.shapiro_delay_s(m, eph, r_tx - e, r_rx, 0.0)[0]) / 200.0
                else:
                    fd[j] = (R.shapiro_delay_s(m, eph, r_tx, r_rx + e, 0.0)[0]
                             - R.shapiro_delay_s(m, eph, r_tx, r_rx - e, 0.0)[0]) / 200.0
            np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-23)

    def test_moon_frame_has_no_shapiro(self):
        self.assertEqual(R.shapiro_delay_s(R.LightTimeModel(), None, np.zeros(3), np.ones(3), 0.0)[0], 0.0)


@unittest.skipUnless(_SPICE_OK, _SKIP_REASON)
class PassTests(unittest.TestCase):
    """Short LLO pass, one DSN station, 21 receive epochs (the LM-consistency fixture)."""

    @classmethod
    def setUpClass(cls):
        from lunar_od.config import Station
        from lunar_od.spice_loader import load_spice_kernels
        from lunar_od.two_way_range import _station_state_providers_for_pass, generate_two_way_range_measurements
        load_spice_kernels()
        cls.et0, cls.t, gep, gev, gsp, cls.x0, cls.aug = _truth_setup()
        cls.gep, cls.gsp = staticmethod(gep), staticmethod(gsp)
        station = Station(name="Goldstone DSN", lat_deg=35.30, lon_deg=-116.81, alt_m=969.67,
                          color_rgb=(0.85, 0.325, 0.098), sigma_range_m=5.0, sigma_angle_rad=math.radians(0.001))
        vis = np.ones((cls.t.size, 1), dtype=bool)
        cls.obs, cls.pg = generate_two_way_range_measurements(cls.t, cls.aug[:, :6], (station,), vis, gep, gev,
                                                              cls.et0, noise=False)
        prov = _station_state_providers_for_pass(cls.pg)[0]
        cls.station = staticmethod(lambda t: np.asarray(prov.state(t), float))
        cls.eph = R.BarycentricEphemeris(float(cls.et0))

    def test_moon_model_reproduces_production(self):
        sc = R.SpacecraftHistory(self.t, self.aug[:, :6])
        for t3, value in self.obs[:, :2]:
            got = R.two_way_range_m(R.solve_two_way(float(t3), self.station, sc, self.eph, R.LightTimeModel()))
            self.assertLessEqual(abs(got - value), 1e-6)

    def test_dsn_model_differs_by_metres(self):
        sc = R.SpacecraftHistory(self.t, self.aug[:, :6])
        t3 = float(self.obs[5, 0])
        moon = R.two_way_range_m(R.solve_two_way(t3, self.station, sc, self.eph, R.LightTimeModel()))
        dsn = R.two_way_range_m(R.solve_two_way(t3, self.station, sc, self.eph, R.DSN_LIGHT_TIME_MODEL))
        self.assertTrue(5.0 < dsn - moon < 15.0, dsn - moon)       # frame ~2 m + Sun Shapiro ~7.5 m

    def test_dsn_partials_match_finite_differences(self):
        from lunar_od.dynamics import propagate_augmented_state

        def history(x0):
            x_aug0 = np.concatenate([x0, np.eye(6).reshape(-1, order="F")])
            return propagate_augmented_state(self.t, x_aug0, _MU_MOON, 0.0, 0.0, self.gep, self.gsp,
                                             rtol=1e-12, atol=1e-13)
        m = R.DSN_LIGHT_TIME_MODEL
        sc_aug = R.SpacecraftHistory(self.t, self.aug)
        rows = (3, 10, 17)
        t_mid = float(self.t[10])

        def values(sc):
            out = [R.two_way_range_m(R.solve_two_way(float(self.obs[i, 0]), self.station, sc, self.eph, m))
                   for i in rows]
            out.append(R.counted_doppler(t_mid, 60.0, self.station, sc, self.eph, m)[0])
            out.append(R.one_way_range(float(self.obs[10, 0]), self.station, sc, self.eph, m)[0])
            return np.array(out)

        an = [R.two_way_range_partial(R.solve_two_way(float(self.obs[i, 0]), self.station, sc_aug, self.eph, m),
                                      sc_aug) for i in rows]
        an.append(R.counted_doppler(t_mid, 60.0, self.station, sc_aug, self.eph, m, with_partial=True)[1])
        an.append(R.one_way_range(float(self.obs[10, 0]), self.station, sc_aug, self.eph, m, with_partial=True)[1])
        an = np.array(an)
        fd = np.zeros_like(an)
        # velocity steps: the FD error falls as h grows (integration noise / h), 0.1 m/s is past that floor
        for j, h in enumerate((10.0, 10.0, 10.0, 0.1, 0.1, 0.1)):
            e = np.zeros(6)
            e[j] = h
            fd[:, j] = (values(R.SpacecraftHistory(self.t, history(self.x0 + e)[:, :6]))
                        - values(R.SpacecraftHistory(self.t, history(self.x0 - e)[:, :6]))) / (2 * h)
        rel = np.max(np.abs(fd - an), axis=1) / np.max(np.abs(an), axis=1)
        limits = np.array([1e-6, 1e-6, 1e-6, 1e-5, 1e-6])      # G5: range and one-way 1e-6, Doppler 1e-5
        self.assertTrue(np.all(rel < limits), rel)


@unittest.skipUnless(_SPICE_OK, _SKIP_REASON)
class EphemerisTests(unittest.TestCase):
    def setUp(self):
        from lunar_od.spice_loader import load_spice_kernels
        load_spice_kernels()

    def test_exact_epoch_removes_quantisation_steps(self):
        eph = R.BarycentricEphemeris(857806356.99998)
        ts = 97107.43527585268 + np.arange(-5, 6) * 1e-9
        steps = np.linalg.norm(np.diff([eph.moon_state(t)[:3] for t in ts], axis=0), axis=1)
        self.assertLess(float(steps.max()), 1e-4)      # 1e-9 s at 30 km/s is 3e-5 m; uncorrected steps are 3.5 mm

    def test_tdb_rate_matches_a_long_baseline_difference(self):
        import spiceypy as spice
        eph = R.BarycentricEphemeris(857806356.99998)
        h = 1e5

        def lag(et):
            return et - spice.unitim(et, "TDB", "TT")
        fd = (lag(eph.et0_s + h) - lag(eph.et0_s - h)) / (2 * h)
        self.assertLess(abs(eph.tdb_minus_tt_rate(0.0) - fd), 2e-12)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
