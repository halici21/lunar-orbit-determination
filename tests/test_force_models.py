"""Phase 3 — generic Body-J2 helper verification.

Independent, behaviour-preserving tests for ``force_models.body_j2_*`` and the
standalone njit equivalents in ``accelerated``.  None of these touch or rely on
a change to the production propagation path; they only assert that the generic
helpers reproduce the existing Moon-J2 math and stay self-consistent.
"""
import unittest

import numpy as np

from lunar_od import (
    body_j2_acceleration,
    body_j2_gravity_gradient,
    dynamics_jacobian_a_matrix,
    f3body_moon,
    zonal_j2_acceleration,
    zonal_j2_gravity_gradient,
)
from lunar_od.dynamics import _MCI_TO_MOON_BF
from lunar_od.accelerated import body_j2_accel_fast, body_j2_gradient_fast
from lunar_od.constants import (
    J2_EARTH_UNNORMALIZED,
    J2_MOON_UNNORMALIZED,
    MU_EARTH_M3S2,
    MU_MOON_M3S2,
    R_EARTH_J2_REF_M,
    R_MOON_M,
)

# Representative third-body geometry (Earth/Sun relative to Moon), m.
R_ME = np.array([-83446893.0, 354010875.0, 178558253.0])
R_MS = np.array([1.40753701450e11, -4.21884124439e10, -1.82638284191e10])


def _moon_positions():
    """Equatorial / polar / oblique r vectors at 100 km and 2000 km altitude."""
    out = []
    for alt in (100e3, 2000e3):
        r0 = R_MOON_M + alt
        out += [
            np.array([r0, 0.0, 0.0]),
            np.array([0.0, 0.0, r0]),
            np.array([r0 * 0.6, r0 * 0.5, r0 * 0.62]),
        ]
    return out


class BodyJ2HelperTests(unittest.TestCase):
    # 1 -----------------------------------------------------------------
    def test_acceleration_matches_zonal_plus_rotation(self):
        for r in _moon_positions():
            got = body_j2_acceleration(
                r, MU_MOON_M3S2, R_MOON_M, J2_MOON_UNNORMALIZED, _MCI_TO_MOON_BF
            )
            ref = _MCI_TO_MOON_BF.T @ zonal_j2_acceleration(
                _MCI_TO_MOON_BF @ r, MU_MOON_M3S2, R_MOON_M, J2_MOON_UNNORMALIZED
            )
            np.testing.assert_array_equal(got, ref)  # bitwise identical

    # 2 -----------------------------------------------------------------
    def test_gradient_matches_zonal_plus_rotation(self):
        for r in _moon_positions():
            got = body_j2_gravity_gradient(
                r, MU_MOON_M3S2, R_MOON_M, J2_MOON_UNNORMALIZED, _MCI_TO_MOON_BF
            )
            g_bf = zonal_j2_gravity_gradient(
                _MCI_TO_MOON_BF @ r, MU_MOON_M3S2, R_MOON_M, J2_MOON_UNNORMALIZED
            )
            ref = _MCI_TO_MOON_BF.T @ g_bf @ _MCI_TO_MOON_BF
            np.testing.assert_array_equal(got, ref)  # bitwise identical

    # 3 -----------------------------------------------------------------
    def test_python_helper_matches_numba_helper(self):
        worst_a = worst_g = 0.0
        for r in _moon_positions():
            ap = body_j2_acceleration(
                r, MU_MOON_M3S2, R_MOON_M, J2_MOON_UNNORMALIZED, _MCI_TO_MOON_BF
            )
            an = body_j2_accel_fast(
                r, MU_MOON_M3S2, R_MOON_M, J2_MOON_UNNORMALIZED, _MCI_TO_MOON_BF
            )
            gp = body_j2_gravity_gradient(
                r, MU_MOON_M3S2, R_MOON_M, J2_MOON_UNNORMALIZED, _MCI_TO_MOON_BF
            )
            gn = body_j2_gradient_fast(
                r, MU_MOON_M3S2, R_MOON_M, J2_MOON_UNNORMALIZED, _MCI_TO_MOON_BF
            )
            worst_a = max(worst_a, np.linalg.norm(ap - an) / np.linalg.norm(ap))
            worst_g = max(worst_g, np.linalg.norm(gp - gn) / np.linalg.norm(gp))
        self.assertLess(worst_a, 1e-12)
        self.assertLess(worst_g, 1e-12)

    # 4 -----------------------------------------------------------------
    def test_j2_zero_is_noop(self):
        r = _moon_positions()[2]
        a = body_j2_acceleration(r, MU_MOON_M3S2, R_MOON_M, 0.0, _MCI_TO_MOON_BF)
        g = body_j2_gravity_gradient(r, MU_MOON_M3S2, R_MOON_M, 0.0, _MCI_TO_MOON_BF)
        an = body_j2_accel_fast(r, MU_MOON_M3S2, R_MOON_M, 0.0, _MCI_TO_MOON_BF)
        gn = body_j2_gradient_fast(r, MU_MOON_M3S2, R_MOON_M, 0.0, _MCI_TO_MOON_BF)
        np.testing.assert_array_equal(a, np.zeros(3))
        np.testing.assert_array_equal(g, np.zeros((3, 3)))
        np.testing.assert_array_equal(an, np.zeros(3))
        np.testing.assert_array_equal(gn, np.zeros((3, 3)))

    # 5 -----------------------------------------------------------------
    def test_finite_difference_gradient(self):
        h = 1.0
        worst = 0.0
        for r in _moon_positions():
            G = body_j2_gravity_gradient(
                r, MU_MOON_M3S2, R_MOON_M, J2_MOON_UNNORMALIZED, _MCI_TO_MOON_BF
            )
            G_fd = np.zeros((3, 3))
            for j in range(3):
                e = np.zeros(3); e[j] = h
                ap = body_j2_acceleration(r + e, MU_MOON_M3S2, R_MOON_M, J2_MOON_UNNORMALIZED, _MCI_TO_MOON_BF)
                am = body_j2_acceleration(r - e, MU_MOON_M3S2, R_MOON_M, J2_MOON_UNNORMALIZED, _MCI_TO_MOON_BF)
                G_fd[:, j] = (ap - am) / (2 * h)
            worst = max(worst, np.linalg.norm(G_fd - G) / np.linalg.norm(G))
        self.assertLess(worst, 1e-4)

    # 6 -----------------------------------------------------------------
    def test_matches_production_moon_j2_contribution(self):
        """helper == (production with J2) - (production without J2), bitwise.

        Composing the J2-off production acceleration/gradient with the generic
        helper must reproduce the full J2-on production result; this avoids
        catastrophic cancellation while still tying the helper to the live path.
        """
        j2 = J2_MOON_UNNORMALIZED
        worst_a = worst_g = 0.0
        for r in _moon_positions():
            state = np.concatenate([r, np.zeros(3)])
            a_on = f3body_moon(state, MU_MOON_M3S2, MU_EARTH_M3S2, 1.32712440018e20,
                               R_ME, R_MS, j2_moon=j2)[3:]
            a_off = f3body_moon(state, MU_MOON_M3S2, MU_EARTH_M3S2, 1.32712440018e20,
                                R_ME, R_MS, j2_moon=0.0)[3:]
            a_helper = body_j2_acceleration(r, MU_MOON_M3S2, R_MOON_M, j2, _MCI_TO_MOON_BF)
            worst_a = max(worst_a, np.linalg.norm(a_on - (a_off + a_helper)))

            A_on = dynamics_jacobian_a_matrix(state, MU_MOON_M3S2, MU_EARTH_M3S2, 1.32712440018e20,
                                              R_ME, R_MS, j2_moon=j2)
            A_off = dynamics_jacobian_a_matrix(state, MU_MOON_M3S2, MU_EARTH_M3S2, 1.32712440018e20,
                                               R_ME, R_MS, j2_moon=0.0)
            G_on = A_on[3:6, 0:3]; G_off = A_off[3:6, 0:3]
            G_helper = body_j2_gravity_gradient(r, MU_MOON_M3S2, R_MOON_M, j2, _MCI_TO_MOON_BF)
            worst_g = max(worst_g, np.linalg.norm(G_on - (G_off + G_helper)))
        self.assertEqual(worst_a, 0.0)   # bitwise identical composition
        self.assertEqual(worst_g, 0.0)

    # 7 -----------------------------------------------------------------
    def test_earth_parameters_generic_sanity(self):
        """Helper works generically with Earth parameters (NOT wired to any path)."""
        c_earth = np.eye(3)  # mean-pole ~ J2000 z-axis (Phase 5 assumption)
        r = np.array([7000e3, 1500e3, 3000e3])  # representative geocentric position
        a = body_j2_acceleration(r, MU_EARTH_M3S2, R_EARTH_J2_REF_M, J2_EARTH_UNNORMALIZED, c_earth)
        g = body_j2_gravity_gradient(r, MU_EARTH_M3S2, R_EARTH_J2_REF_M, J2_EARTH_UNNORMALIZED, c_earth)
        # identity rotation -> equals the body-fixed zonal kernels directly
        np.testing.assert_array_equal(
            a, zonal_j2_acceleration(r, MU_EARTH_M3S2, R_EARTH_J2_REF_M, J2_EARTH_UNNORMALIZED)
        )
        self.assertTrue(np.all(np.isfinite(a)) and np.linalg.norm(a) > 0.0)
        self.assertTrue(np.all(np.isfinite(g)))
        # J2 perturbation is small relative to Earth point-mass at this radius
        a_pm = MU_EARTH_M3S2 / np.dot(r, r)
        self.assertLess(np.linalg.norm(a) / a_pm, 1e-2)
        # njit Earth path agrees with Python helper
        an = body_j2_accel_fast(r, MU_EARTH_M3S2, R_EARTH_J2_REF_M, J2_EARTH_UNNORMALIZED, c_earth)
        self.assertLess(np.linalg.norm(a - an) / np.linalg.norm(a), 1e-12)


if __name__ == "__main__":
    unittest.main()
