import math
import json
import unittest
from pathlib import Path

import numpy as np

from lunar_od import (
    ecef2razel_sez,
    ecef2sez_dcm,
    geodetic_to_ecef_wgs84,
    wrap_to_pi,
)

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


class GeometryTests(unittest.TestCase):
    def test_wgs84_equator_and_pole_anchors(self):
        x, y, z = geodetic_to_ecef_wgs84(0.0, 0.0, 0.0)
        self.assertAlmostEqual(float(x), 6378137.0, places=6)
        self.assertAlmostEqual(float(y), 0.0, places=9)
        self.assertAlmostEqual(float(z), 0.0, places=9)

        x, y, z = geodetic_to_ecef_wgs84(90.0, 0.0, 0.0)
        self.assertAlmostEqual(float(x), 0.0, places=6)
        self.assertAlmostEqual(float(y), 0.0, places=6)
        self.assertAlmostEqual(float(z), 6356752.314245179, places=6)

    def test_ecef2sez_dcm_is_orthogonal(self):
        c = ecef2sez_dcm(math.radians(39.0), math.radians(32.0))
        np.testing.assert_allclose(c @ c.T, np.eye(3), atol=1e-14)
        self.assertAlmostEqual(float(np.linalg.det(c)), 1.0, places=14)

    def test_razel_cardinal_vectors_at_equator(self):
        lat = 0.0
        lon = 0.0

        az, el, rng = ecef2razel_sez([1000.0, 0.0, 0.0], lat, lon)
        self.assertAlmostEqual(rng, 1000.0, places=12)
        self.assertAlmostEqual(math.degrees(az), 0.0, places=12)
        self.assertAlmostEqual(math.degrees(el), 90.0, places=12)

        az, el, rng = ecef2razel_sez([0.0, 1000.0, 0.0], lat, lon)
        self.assertAlmostEqual(rng, 1000.0, places=12)
        self.assertAlmostEqual(math.degrees(az), 90.0, places=12)
        self.assertAlmostEqual(math.degrees(el), 0.0, places=12)

        az, el, rng = ecef2razel_sez([0.0, 0.0, 1000.0], lat, lon)
        self.assertAlmostEqual(rng, 1000.0, places=12)
        self.assertAlmostEqual(math.degrees(az), 0.0, places=12)
        self.assertAlmostEqual(math.degrees(el), 0.0, places=12)

        az, el, rng = ecef2razel_sez([0.0, 0.0, -1000.0], lat, lon)
        self.assertAlmostEqual(rng, 1000.0, places=12)
        self.assertAlmostEqual(math.degrees(az), 180.0, places=12)
        self.assertAlmostEqual(math.degrees(el), 0.0, places=12)

    def test_wrap_to_pi_uses_short_angular_residual(self):
        wrapped = wrap_to_pi(np.array([2.0 * math.pi + 0.1, -2.0 * math.pi - 0.2]))
        np.testing.assert_allclose(wrapped, np.array([0.1, -0.2]), atol=1e-15)

    def test_against_foundation_fixture_when_available(self):
        fixture_path = FIXTURES_DIR / "foundation_geometry.json"
        if not fixture_path.is_file():
            self.skipTest("MATLAB foundation_geometry.json fixture has not been exported yet.")

        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

        for case in fixture["wgs84_cases"]:
            x_m, y_m, z_m = geodetic_to_ecef_wgs84(case["lat_deg"], case["lon_deg"], case["alt_m"])
            np.testing.assert_allclose([x_m, y_m, z_m], case["ecef_m"], rtol=0.0, atol=1e-6)

        for case in fixture["razel_cases"]:
            az_rad, el_rad, range_m = ecef2razel_sez(
                case["r_rel_ecef_m"],
                case["lat_rad"],
                case["lon_rad"],
            )
            self.assertAlmostEqual(range_m, case["range_m"], places=12)
            self.assertAlmostEqual(az_rad, case["azimuth_rad"], places=12)
            self.assertAlmostEqual(el_rad, case["elevation_rad"], places=12)

        for case in fixture["dcm_cases"]:
            actual = ecef2sez_dcm(case["lat_rad"], case["lon_rad"])
            np.testing.assert_allclose(actual, case["c_sez_ecef"], rtol=0.0, atol=1e-15)

        wrapped = wrap_to_pi(fixture["wrap_to_pi"]["input_rad"])
        np.testing.assert_allclose(wrapped, fixture["wrap_to_pi"]["output_rad"], rtol=0.0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
