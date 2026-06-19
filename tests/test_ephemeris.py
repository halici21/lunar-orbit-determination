import json
import unittest
from pathlib import Path

import numpy as np

from lunar_od import (
    MoonCenteredEphemeris,
    load_spice_kernels,
    perturb_moon_centered_ephemeris,
    sample_moon_centered_ephemeris,
)


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


class EphemerisTests(unittest.TestCase):
    def test_spice_sampling_and_pchip_interpolation_against_matlab_fixture(self):
        fixture_path = FIXTURES_DIR / "spice_snapshots.json"
        if not fixture_path.is_file():
            self.skipTest("MATLAB spice_snapshots.json fixture has not been exported yet.")

        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        grid = fixture["ephemeris_grid"]

        import spiceypy as spice

        load_spice_kernels()
        try:
            sampled = sample_moon_centered_ephemeris(fixture["et"], grid["t_ephem_s"])
        finally:
            spice.kclear()

        np.testing.assert_allclose(sampled.earth_pos_m, grid["earth_pos_m"], rtol=0.0, atol=1e-5)
        np.testing.assert_allclose(sampled.sun_pos_m, grid["sun_pos_m"], rtol=0.0, atol=1e-2)
        np.testing.assert_allclose(sampled.earth_vel_mps, grid["earth_vel_mps"], rtol=0.0, atol=1e-8)

        t_interp_s = np.asarray(grid["t_interp_s"], dtype=float)
        np.testing.assert_allclose(sampled.earth_position(t_interp_s), grid["earth_pos_interp_m"], rtol=0.0, atol=1e-5)
        np.testing.assert_allclose(sampled.sun_position(t_interp_s), grid["sun_pos_interp_m"], rtol=0.0, atol=1e-2)
        np.testing.assert_allclose(sampled.earth_velocity(t_interp_s), grid["earth_vel_interp_mps"], rtol=0.0, atol=1e-8)

    def test_interpolant_shape_validation(self):
        with self.assertRaises(ValueError):
            MoonCenteredEphemeris(
                t_ephem_s=[0.0, 1.0],
                earth_pos_m=[[0.0, 0.0, 0.0]],
                sun_pos_m=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                earth_vel_mps=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            )

    def test_ephemeris_perturbation_supports_model_mismatch_campaigns(self):
        nominal = MoonCenteredEphemeris(
            t_ephem_s=[0.0, 60.0],
            earth_pos_m=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            sun_pos_m=[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
            earth_vel_mps=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        )

        perturbed = perturb_moon_centered_ephemeris(
            nominal,
            earth_position_bias_m=[100.0, 0.0, -50.0],
            earth_velocity_bias_mps=[0.01, -0.02, 0.0],
            sun_position_bias_m=[1000.0, 2000.0, 3000.0],
        )

        np.testing.assert_allclose(
            perturbed.earth_pos_m - nominal.earth_pos_m,
            np.tile([100.0, 0.0, -50.0], (2, 1)),
        )
        np.testing.assert_allclose(
            perturbed.earth_vel_mps - nominal.earth_vel_mps,
            np.tile([0.01, -0.02, 0.0], (2, 1)),
        )
        np.testing.assert_allclose(
            perturbed.sun_pos_m - nominal.sun_pos_m,
            np.tile([1000.0, 2000.0, 3000.0], (2, 1)),
        )


if __name__ == "__main__":
    unittest.main()
