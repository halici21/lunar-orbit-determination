"""Focused tests from the Doppler / range-rate model review (sections 9A-9F).

These complement the existing two-way Doppler suite by making the textbook vs
operational distinction explicit:

* 9A/9B -- the geometric instantaneous mode is the textbook line-of-sight
  range-rate (receding = speed, transverse = 0).
* 9C/9D -- the two-way counted-Doppler mode is genuinely *averaged differenced
  range* over the count interval (it depends on Tc under non-zero jerk and
  reduces to the instantaneous rate as Tc -> 0), and is therefore physically
  distinct from the instantaneous mode.
"""

import unittest

import numpy as np

from lunar_od import (
    RangeRatePhysicsConfig,
    instantaneous_geometric_range_rate,
    two_way_counted_doppler_observable,
)

C_LIGHT = 299792458.0


class _Station:
    """Minimal station fixed at the ECEF origin (identity frame fixtures)."""
    r_ecef_m = np.zeros(3)
    lat_rad = 0.0
    lon_rad = 0.0


def _radial_cubic_states(t, x0, v0, a, j):
    """1-D radial motion x(t)=x0+v0 t+a t^2/2+j t^3/6 along +x (far from origin)."""
    t = np.asarray(t, float)
    x = x0 + v0 * t + 0.5 * a * t**2 + j * t**3 / 6.0
    vx = v0 + a * t + 0.5 * j * t**2
    states = np.zeros((t.size, 6))
    states[:, 0] = x
    states[:, 3] = vx
    return states


def _identity_frame(n):
    return np.repeat(np.eye(6)[None, :, :], n, axis=0)


class GeometricInstantaneousRangeRate(unittest.TestCase):
    def test_9a_receding_equals_speed(self):
        """Section 9A: radial recession -> rho_dot equals the closing speed."""
        r = np.array([1.0e7, 0.0, 0.0])
        v = np.array([125.0, 0.0, 0.0])           # straight along the line of sight
        self.assertAlmostEqual(instantaneous_geometric_range_rate(r, v), 125.0, places=9)
        # approaching (negative) is the mirror case
        self.assertAlmostEqual(instantaneous_geometric_range_rate(r, -v), -125.0, places=9)

    def test_9b_transverse_motion_is_zero(self):
        """Section 9B: velocity perpendicular to the line of sight -> rho_dot = 0."""
        r = np.array([1.0e7, 0.0, 0.0])
        v = np.array([0.0, 125.0, -80.0])         # purely transverse
        self.assertAlmostEqual(instantaneous_geometric_range_rate(r, v), 0.0, places=9)


class TwoWayCountedDopplerIsAveraged(unittest.TestCase):
    """Sections 9C/9D: confirm the two-way mode is Tc-averaged and distinct."""

    def _obs(self, t, states, tc):
        cfg = RangeRatePhysicsConfig(mode="two_way_counted_doppler", count_interval_s=tc)
        n = t.size
        zeros = np.zeros((n, 3))
        return two_way_counted_doppler_observable(
            0.0, _Station(), t, states, zeros, zeros, _identity_frame(n), cfg)

    def test_9c_observable_depends_on_count_interval_under_jerk(self):
        """A non-zero jerk makes the averaged observable depend on Tc; in the
        small-Tc limit it collapses onto the instantaneous range-rate. If the
        observable were instantaneous it would be Tc-independent."""
        t = np.arange(-200.0, 200.001, 1.0)
        jerk = 5.0e-3                              # m/s^3 (radial)
        states = _radial_cubic_states(t, x0=1.0e7, v0=100.0, a=0.0, j=jerk)
        inst = instantaneous_geometric_range_rate(states[t.size // 2, :3], states[t.size // 2, 3:])

        obs_short = self._obs(t, states, tc=2.0)
        obs_long = self._obs(t, states, tc=160.0)

        # small Tc -> instantaneous limit
        self.assertAlmostEqual(obs_short, inst, delta=1e-2)
        # large Tc -> averaged, measurably different (averaging over the jerk)
        self.assertGreater(abs(obs_long - obs_short), 0.1)
        # expected averaging signature: centred mean of a quadratic rho_dot is
        # inst + (jerk/6)*(Tc/2)^2  (one-way m/s-equivalent)
        expected_long = inst + (jerk / 6.0) * (160.0 / 2.0) ** 2
        self.assertAlmostEqual(obs_long, expected_long, delta=0.05)

    def test_9d_two_way_distinct_from_instantaneous_under_jerk(self):
        """Section 9D: with curvature/jerk the two-way counted Doppler differs
        from the instantaneous geometric range-rate -> the modes are physically
        distinct (not the same computation behind two labels)."""
        t = np.arange(-200.0, 200.001, 1.0)
        states = _radial_cubic_states(t, x0=1.0e7, v0=100.0, a=0.0, j=5.0e-3)
        inst = instantaneous_geometric_range_rate(states[t.size // 2, :3], states[t.size // 2, 3:])
        two_way = self._obs(t, states, tc=160.0)
        self.assertGreater(abs(two_way - inst), 0.1)


if __name__ == "__main__":
    unittest.main()
