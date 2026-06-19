import json
import math
import unittest
from pathlib import Path

import numpy as np

from lunar_od import position_only_stations, range_rate_stations


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


class ConfigTests(unittest.TestCase):
    def test_station_counts_and_range_rate_membership(self):
        pos_stations = position_only_stations()
        rr_stations = range_rate_stations()

        self.assertEqual(len(pos_stations), 13)
        self.assertEqual(len(rr_stations), 15)
        self.assertIn("Daejeon KGS", [station.name for station in rr_stations])
        self.assertIn("Cebreros ESA", [station.name for station in rr_stations])
        self.assertNotIn("Daejeon KGS", [station.name for station in pos_stations])

    def test_station_noise_model(self):
        pos_stations = {station.name: station for station in position_only_stations()}
        rr_stations = {station.name: station for station in range_rate_stations()}

        self.assertEqual(pos_stations["ITU Ayazaga"].sigma_range_m, 94.0)
        self.assertAlmostEqual(pos_stations["ITU Ayazaga"].sigma_angle_rad, math.radians(0.005))
        self.assertEqual(pos_stations["Goldstone DSN"].sigma_range_m, 5.0)
        self.assertAlmostEqual(pos_stations["Goldstone DSN"].sigma_angle_rad, math.radians(0.001))

        self.assertEqual(rr_stations["ITU Ayazaga"].sigma_range_rate_mps, 1e-3)
        self.assertEqual(rr_stations["Goldstone DSN"].sigma_range_rate_mps, 1e-4)

    def test_against_station_config_fixture_when_available(self):
        fixture_path = FIXTURES_DIR / "station_config.json"
        if not fixture_path.is_file():
            self.skipTest("MATLAB station_config.json fixture has not been exported yet.")

        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        expected_pos = fixture["position_only"]["stations"]
        expected_rr = fixture["range_rate"]["stations"]

        self._assert_station_fixture_matches_python(expected_pos, position_only_stations())
        self._assert_station_fixture_matches_python(expected_rr, range_rate_stations())

    def _assert_station_fixture_matches_python(self, expected_stations, actual_stations):
        self.assertEqual(len(expected_stations), len(actual_stations))

        for expected, actual in zip(expected_stations, actual_stations):
            self.assertEqual(expected["name"], actual.name)
            self.assertAlmostEqual(expected["lat_deg"], actual.lat_deg, places=12)
            self.assertAlmostEqual(expected["lon_deg"], actual.lon_deg, places=12)
            self.assertAlmostEqual(expected["alt_m"], actual.alt_m, places=12)
            self.assertAlmostEqual(expected["lat_rad"], actual.lat_rad, places=15)
            self.assertAlmostEqual(expected["lon_rad"], actual.lon_rad, places=15)
            self.assertAlmostEqual(expected["sigma_range_m"], actual.sigma_range_m, places=15)
            self.assertAlmostEqual(expected["sigma_angle_rad"], actual.sigma_angle_rad, places=15)

            expected_rr = expected["sigma_range_rate_mps"]
            if expected_rr is None or (isinstance(expected_rr, float) and math.isnan(expected_rr)):
                self.assertIsNone(actual.sigma_range_rate_mps)
            else:
                self.assertAlmostEqual(expected_rr, actual.sigma_range_rate_mps, places=15)

            np.testing.assert_allclose(expected["r_ecef_m"], actual.r_ecef_m, rtol=0.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

