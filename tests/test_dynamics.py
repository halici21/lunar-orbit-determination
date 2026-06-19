import json
import unittest
from pathlib import Path

import numpy as np

from lunar_od import (
    dynamics_jacobian_a_matrix,
    f3body_moon,
    ode_fun_v3,
    point_mass_acceleration,
    zonal_j2_acceleration,
    zonal_j2_gravity_gradient,
)


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


class DynamicsTests(unittest.TestCase):
    def test_point_mass_acceleration_points_inward(self):
        mu_moon = 4902.800066e9
        r_m = np.array([1737.4e3 + 100e3, 0.0, 0.0])
        accel = point_mass_acceleration(r_m, mu_moon)

        self.assertTrue(np.all(np.isfinite(accel)))
        self.assertLess(float(np.dot(accel, r_m)), 0.0)

    def test_zonal_j2_acceleration_has_equator_and_pole_anchors(self):
        mu_moon = 4902.800066e9
        radius_m = 1737.4e3
        j2 = 2.0e-4
        r_equator = np.array([radius_m + 100e3, 0.0, 0.0])
        r_pole = np.array([0.0, 0.0, radius_m + 100e3])

        a_equator = zonal_j2_acceleration(r_equator, mu_moon, radius_m, j2)
        a_pole = zonal_j2_acceleration(r_pole, mu_moon, radius_m, j2)

        expected_equator_x = -1.5 * j2 * mu_moon * radius_m**2 / np.linalg.norm(r_equator) ** 4
        expected_pole_z = 3.0 * j2 * mu_moon * radius_m**2 / np.linalg.norm(r_pole) ** 4
        np.testing.assert_allclose(a_equator, [expected_equator_x, 0.0, 0.0], rtol=1e-14, atol=0.0)
        np.testing.assert_allclose(a_pole, [0.0, 0.0, expected_pole_z], rtol=1e-14, atol=0.0)
        np.testing.assert_allclose(zonal_j2_acceleration(r_equator, mu_moon, radius_m, 0.0), np.zeros(3))

    def test_zonal_j2_gravity_gradient_matches_finite_difference(self):
        mu_moon = 4902.800066e9
        radius_m = 1737.4e3
        j2 = 2.0e-4
        r_m = np.array([radius_m + 200e3, 40e3, 75e3])

        gradient = zonal_j2_gravity_gradient(r_m, mu_moon, radius_m, j2)
        # finite-difference reference with 1 m step
        check = np.zeros((3, 3))
        for axis in range(3):
            delta = np.zeros(3)
            delta[axis] = 1.0
            check[:, axis] = (
                zonal_j2_acceleration(r_m + delta, mu_moon, radius_m, j2)
                - zonal_j2_acceleration(r_m - delta, mu_moon, radius_m, j2)
            ) / 2.0

        np.testing.assert_allclose(gradient, check, rtol=1e-6, atol=0.0)

    def test_f3body_and_stm_derivative_against_matlab_fixture(self):
        fixture_path = FIXTURES_DIR / "spice_snapshots.json"
        if not fixture_path.is_file():
            self.skipTest("MATLAB spice_snapshots.json fixture has not been exported yet.")

        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        constants = fixture["constants"]
        dynamics = fixture["dynamics"]
        initial = fixture["initial_state"]

        state_mci = np.asarray(initial["state_mci_j2000_m_mps"], dtype=float)
        r_earth_mci_m = np.asarray(dynamics["r_earth_mci_m"], dtype=float)
        r_sun_mci_m = np.asarray(dynamics["r_sun_mci_m"], dtype=float)

        mu_moon = float(initial["mu_moon_m3_s2"])
        mu_earth = float(np.asarray(constants["mu_earth_km3_s2"]).reshape(-1)[0] * 1e9)
        mu_sun = float(np.asarray(constants["mu_sun_km3_s2"]).reshape(-1)[0] * 1e9)

        state_dot = f3body_moon(state_mci, mu_moon, mu_earth, mu_sun, r_earth_mci_m, r_sun_mci_m)
        np.testing.assert_allclose(
            state_dot,
            dynamics["f3body_state_derivative_m_mps_mps2"],
            rtol=0.0,
            atol=1e-12,
        )

        phi0 = np.eye(6).reshape(-1, order="F")
        aug0 = np.concatenate([state_mci, phi0])
        aug_dot = ode_fun_v3(
            0.0,
            aug0,
            mu_moon,
            mu_earth,
            mu_sun,
            lambda _t: r_earth_mci_m,
            lambda _t: r_sun_mci_m,
        )
        np.testing.assert_allclose(
            aug_dot,
            dynamics["ode_v3_aug_derivative"],
            rtol=0.0,
            atol=1e-12,
        )

        a_matrix = dynamics_jacobian_a_matrix(state_mci, mu_moon, mu_earth, mu_sun, r_earth_mci_m, r_sun_mci_m)
        np.testing.assert_allclose(a_matrix, dynamics["a_matrix"], rtol=0.0, atol=1e-18)


if __name__ == "__main__":
    unittest.main()
