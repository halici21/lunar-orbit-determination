"""Phase 6A — force-model scenario-comparison verification (fast, frozen ephemeris).

Validates the scenario matrix, force-flag correctness, perturbation separation,
and a 1-day smoke run.  Uses the production propagator directly with a frozen
ephemeris (no .mat / SPICE dependency); the full .mat-driven campaign and the
PlanetEphemeris<->SPICE layer live in examples/phase6_*.py.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from lunar_od.dynamics import f3body_moon, propagate_state, MOON_J2
from lunar_od.orbit import coe2rv
from lunar_od.scenario_config import ScenarioConfig
from lunar_od.constants import (
    MU_MOON_M3S2 as MU_M, MU_EARTH_M3S2 as MU_E, MU_SUN_M3S2 as MU_S,
    J2_EARTH_UNNORMALIZED as J2_E, R_MOON_M,
)
from phase6_scenario_comparison import SCENARIOS, initial_state  # the actual scenario matrix

R_ME = np.array([-83446893.0, 354010875.0, 178558253.0])
R_MS = np.array([1.40753701450e11, -4.21884124439e10, -1.82638284191e10])
GE = lambda t: R_ME
GS = lambda t: R_MS


def _accel(state, mu_e, mu_s, j2_moon=0.0, j2_earth=0.0, mode="indirect"):
    return f3body_moon(state, MU_M, mu_e, mu_s, R_ME, R_MS,
                       j2_moon=j2_moon, j2_earth=j2_earth, earth_j2_mode=mode)[3:]


def _prop_1day(j2_moon=0.0, j2_earth=0.0, mu_e=MU_E, mu_s=MU_S):
    teval = np.arange(0.0, 86400.0 + 1, 1200.0)
    return propagate_state(teval, initial_state(), MU_M, mu_e, mu_s, GE, GS,
                           method="ADAMS", j2_moon=j2_moon, j2_earth=j2_earth,
                           earth_j2_mode="indirect")


class ScenarioComparisonTests(unittest.TestCase):
    def setUp(self):
        self.s0 = initial_state()

    # 1 -- Earth J2 off leaves the RHS unchanged (bit-identical)
    def test_earth_off_unchanged(self):
        a_no = _accel(self.s0, MU_E, MU_S, j2_moon=MOON_J2)
        a_off = _accel(self.s0, MU_E, MU_S, j2_moon=MOON_J2, j2_earth=0.0)
        self.assertEqual(np.linalg.norm(a_no - a_off), 0.0)

    # 2 -- Earth J2 flag actually reaches propagate (small nonzero change)
    def test_earth_j2_reaches_propagate(self):
        base = _prop_1day()
        earth = _prop_1day(j2_earth=J2_E)
        d = np.linalg.norm(earth[-1, :3] - base[-1, :3])
        self.assertGreater(d, 0.0)
        self.assertLess(d, 1.0)   # cm-scale over 1 day

    # 3 / 4 -- default mode is indirect, not direct
    def test_default_mode_indirect(self):
        self.assertEqual(ScenarioConfig("t", "range_rate", "ukf", "cold", "multi").earth_j2_mode, "indirect")

    # 5 -- Moon J2 float mechanism preserved
    def test_moon_j2_float_preserved(self):
        a0 = _accel(self.s0, MU_E, MU_S, j2_moon=0.0)
        aj = _accel(self.s0, MU_E, MU_S, j2_moon=MOON_J2)
        self.assertGreater(np.linalg.norm(aj - a0), 0.0)

    # 6 -- Keplerian (mu=0) is pure Moon two-body
    def test_keplerian_is_two_body(self):
        a = _accel(self.s0, 0.0, 0.0)
        r = self.s0[:3]
        a_kep = -MU_M * r / np.linalg.norm(r) ** 3
        np.testing.assert_allclose(a, a_kep, rtol=0, atol=1e-18)

    # 7 -- Third body adds Earth/Sun point-mass (differs from Keplerian)
    def test_third_body_differs_from_keplerian(self):
        self.assertGreater(np.linalg.norm(_accel(self.s0, MU_E, MU_S) - _accel(self.s0, 0.0, 0.0)), 0.0)

    # 8/9/10 -- scenario matrix flag correctness
    def test_scenario_matrix_flags(self):
        self.assertEqual(SCENARIOS["1_keplerian"]["mu_e"], 0.0)
        self.assertEqual(SCENARIOS["1_keplerian"]["mu_s"], 0.0)
        self.assertEqual(SCENARIOS["3_earth_j2"]["j2_moon"], 0.0)      # Moon J2 off in Earth-J2 scenario
        self.assertGreater(SCENARIOS["3_earth_j2"]["j2_earth"], 0.0)
        self.assertEqual(SCENARIOS["4_moon_j2"]["j2_earth"], 0.0)      # Earth J2 off in Moon-J2 scenario
        self.assertGreater(SCENARIOS["4_moon_j2"]["j2_moon"], 0.0)
        self.assertGreater(SCENARIOS["5_earth_moon_j2"]["j2_moon"], 0.0)
        self.assertGreater(SCENARIOS["5_earth_moon_j2"]["j2_earth"], 0.0)

    # 11 / 12 -- 1-day smoke: no NaN, Moon J2 >> Earth J2, both nonzero
    def test_one_day_smoke_separation(self):
        base = _prop_1day()
        earth = _prop_1day(j2_earth=J2_E)
        moon = _prop_1day(j2_moon=MOON_J2)
        both = _prop_1day(j2_moon=MOON_J2, j2_earth=J2_E)
        for tr in (base, earth, moon, both):
            self.assertTrue(np.all(np.isfinite(tr)))
        d_earth = np.linalg.norm(earth[-1, :3] - base[-1, :3])
        d_moon = np.linalg.norm(moon[-1, :3] - base[-1, :3])
        self.assertGreater(d_earth, 0.0)
        self.assertGreater(d_moon, 100.0 * d_earth)            # Moon J2 dominates
        # combined ~ moon (Earth negligible)
        self.assertLess(abs(np.linalg.norm(both[-1, :3] - base[-1, :3]) - d_moon) / d_moon, 1e-2)

    # 13 -- ephemeris TDB epoch mapping is consistent (arithmetic; SPICE layer in example)
    def test_tdb_epoch_mapping(self):
        first_jd = 2461467.480983796
        et0 = (first_jd - 2451545.0) * 86400.0
        self.assertAlmostEqual(et0, 857302356.99998, places=2)


if __name__ == "__main__":
    unittest.main()
