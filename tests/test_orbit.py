import json
import math
import unittest
from pathlib import Path

import numpy as np

from lunar_od import coe2rv, lunar_initial_state_mci, rot_x, rot_z


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


class OrbitTests(unittest.TestCase):
    def test_passive_rotation_conventions(self):
        np.testing.assert_allclose(rot_z(math.pi / 2.0) @ np.array([1.0, 0.0, 0.0]), [0.0, -1.0, 0.0], atol=1e-15)
        np.testing.assert_allclose(rot_x(math.pi / 2.0) @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, -1.0], atol=1e-15)

        for matrix in (rot_x(0.7), rot_z(-1.2)):
            np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-15)
            self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0, places=15)

    def test_coe2rv_circular_equatorial_anchor(self):
        mu = 398600.4418e9
        a = 7000e3
        r_m, v_mps = coe2rv(a, 0.0, 0.0, 0.0, 0.0, 0.0, mu)

        np.testing.assert_allclose(r_m, [a, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(v_mps, [0.0, math.sqrt(mu / a), 0.0], atol=1e-12)

    def test_coe2rv_rejects_open_orbits(self):
        with self.assertRaises(ValueError):
            coe2rv(7000e3, 1.0, 0.0, 0.0, 0.0, 0.0, 398600.4418e9)

    def test_initial_lunar_state_against_matlab_spice_fixture_when_available(self):
        fixture_path = FIXTURES_DIR / "spice_snapshots.json"
        if not fixture_path.is_file():
            self.skipTest("MATLAB spice_snapshots.json fixture has not been exported yet.")

        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        initial = fixture["initial_state"]
        orbit = initial["orbit"]

        r_mpa_m, v_mpa_mps, state_mci = lunar_initial_state_mci(
            moon_radius_m=initial["r_moon_mean_m"],
            altitude_m=orbit["altitude_m"],
            eccentricity=orbit["ecc"],
            inclination_rad=orbit["inc_rad"],
            raan_rad=orbit["raan_rad"],
            arg_periapsis_rad=orbit["argp_rad"],
            true_anomaly_rad=orbit["nu_rad"],
            mu_moon_m3_s2=initial["mu_moon_m3_s2"],
            moon_pa_to_j2000_sxform=fixture["transforms"]["moon_pa_to_j2000_sxform"],
        )

        np.testing.assert_allclose(r_mpa_m, initial["r_moon_pa_m"], rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(v_mpa_mps, initial["v_moon_pa_mps"], rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(state_mci, initial["state_mci_j2000_m_mps"], rtol=0.0, atol=1e-8)


if __name__ == "__main__":
    unittest.main()

