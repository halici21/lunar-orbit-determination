import json
import unittest
from pathlib import Path

import numpy as np

from lunar_od import load_spice_kernels, sample_j2000_to_itrf93_transforms


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


class SpiceSnapshotTests(unittest.TestCase):
    def test_against_matlab_spice_snapshot_fixture_when_available(self):
        fixture_path = FIXTURES_DIR / "spice_snapshots.json"
        if not fixture_path.is_file():
            self.skipTest("MATLAB spice_snapshots.json fixture has not been exported yet.")

        import spiceypy as spice

        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        load_spice_kernels()

        try:
            et = spice.str2et(fixture["epoch_utc"])
            self.assertAlmostEqual(et, fixture["et"], places=9)

            self._assert_bodvrd_matches(spice, "MOON", "GM", fixture["constants"]["mu_moon_km3_s2"])
            self._assert_bodvrd_matches(spice, "EARTH", "GM", fixture["constants"]["mu_earth_km3_s2"])
            self._assert_bodvrd_matches(spice, "SUN", "GM", fixture["constants"]["mu_sun_km3_s2"])
            self._assert_bodvrd_matches(spice, "MOON", "RADII", fixture["constants"]["radii_moon_km"])

            np.testing.assert_allclose(
                spice.sxform("MOON_PA", "J2000", et),
                fixture["transforms"]["moon_pa_to_j2000_sxform"],
                rtol=0.0,
                atol=1e-14,
            )
            np.testing.assert_allclose(
                spice.sxform("J2000", "ITRF93", et),
                fixture["transforms"]["j2000_to_itrf93_sxform"],
                rtol=0.0,
                atol=1e-14,
            )
            np.testing.assert_allclose(
                sample_j2000_to_itrf93_transforms(et, [0.0])[0],
                fixture["transforms"]["j2000_to_itrf93_sxform"],
                rtol=0.0,
                atol=1e-14,
            )

            earth_state, earth_lt = spice.spkezr("EARTH", et, "J2000", "NONE", "MOON")
            sun_state, sun_lt = spice.spkezr("SUN", et, "J2000", "NONE", "MOON")

            np.testing.assert_allclose(
                earth_state,
                fixture["ephemerides"]["earth_wrt_moon_j2000_state_km_kms"],
                rtol=0.0,
                atol=1e-8,
            )
            self.assertAlmostEqual(earth_lt, fixture["ephemerides"]["earth_wrt_moon_light_time_s"], places=12)
            np.testing.assert_allclose(
                sun_state,
                fixture["ephemerides"]["sun_wrt_moon_j2000_state_km_kms"],
                rtol=0.0,
                atol=1e-5,
            )
            self.assertAlmostEqual(sun_lt, fixture["ephemerides"]["sun_wrt_moon_light_time_s"], places=9)
        finally:
            spice.kclear()

    def _assert_bodvrd_matches(self, spice, body, item, expected):
        expected = np.asarray(expected, dtype=float).reshape(-1)
        _, actual = spice.bodvrd(body, item, len(expected))
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
