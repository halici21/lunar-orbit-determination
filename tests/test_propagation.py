import json
import unittest
from pathlib import Path

import numpy as np

from lunar_od import MoonCenteredEphemeris, propagate_augmented_state, propagate_state, propagate_truth_with_ephemeris


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


class PropagationTests(unittest.TestCase):
    def test_fixed_third_body_propagation_against_matlab_fixture(self):
        fixture_path = FIXTURES_DIR / "spice_snapshots.json"
        if not fixture_path.is_file():
            self.skipTest("MATLAB spice_snapshots.json fixture has not been exported yet.")

        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        constants = fixture["constants"]
        dynamics = fixture["dynamics"]
        initial = fixture["initial_state"]
        propagation = fixture["propagation"]
        expected_state_history = np.asarray(propagation["state_history_mci_m_mps"], dtype=float)
        expected_augmented_history = np.asarray(propagation["augmented_history_mci"], dtype=float)

        state0_mci = np.asarray(initial["state_mci_j2000_m_mps"], dtype=float)
        r_earth_mci_m = np.asarray(dynamics["r_earth_mci_m"], dtype=float)
        r_sun_mci_m = np.asarray(dynamics["r_sun_mci_m"], dtype=float)
        t_eval_s = np.asarray(propagation["t_eval_s"], dtype=float)

        mu_moon = float(initial["mu_moon_m3_s2"])
        mu_earth = float(np.asarray(constants["mu_earth_km3_s2"]).reshape(-1)[0] * 1e9)
        mu_sun = float(np.asarray(constants["mu_sun_km3_s2"]).reshape(-1)[0] * 1e9)

        state_history = propagate_state(
            t_eval_s,
            state0_mci,
            mu_moon,
            mu_earth,
            mu_sun,
            lambda _t: r_earth_mci_m,
            lambda _t: r_sun_mci_m,
            rtol=1e-12,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            state_history,
            expected_state_history,
            rtol=0.0,
            atol=3e-3,
        )

        state_aug0 = np.concatenate([state0_mci, np.eye(6).reshape(-1, order="F")])
        augmented_history = propagate_augmented_state(
            t_eval_s,
            state_aug0,
            mu_moon,
            mu_earth,
            mu_sun,
            lambda _t: r_earth_mci_m,
            lambda _t: r_sun_mci_m,
            rtol=1e-12,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            augmented_history[:, :6],
            expected_augmented_history[:, :6],
            rtol=0.0,
            atol=3e-3,
        )
        np.testing.assert_allclose(
            augmented_history[:, 6:],
            expected_augmented_history[:, 6:],
            rtol=0.0,
            atol=3e-8,
        )

    def test_truth_propagation_with_ephemeris_against_matlab_fixture(self):
        fixture_path = FIXTURES_DIR / "spice_snapshots.json"
        if not fixture_path.is_file():
            self.skipTest("MATLAB spice_snapshots.json fixture has not been exported yet.")

        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        constants = fixture["constants"]
        initial = fixture["initial_state"]
        truth = fixture["truth_propagation"]

        state0_mci = np.asarray(initial["state_mci_j2000_m_mps"], dtype=float)
        t_eval_s = np.asarray(truth["t_eval_s"], dtype=float)
        ephemeris = MoonCenteredEphemeris(
            t_ephem_s=truth["t_ephem_s"],
            earth_pos_m=truth["earth_pos_grid_m"],
            sun_pos_m=truth["sun_pos_grid_m"],
            earth_vel_mps=np.zeros_like(np.asarray(truth["earth_pos_grid_m"], dtype=float)),
        )

        mu_moon = float(initial["mu_moon_m3_s2"])
        mu_earth = float(np.asarray(constants["mu_earth_km3_s2"]).reshape(-1)[0] * 1e9)
        mu_sun = float(np.asarray(constants["mu_sun_km3_s2"]).reshape(-1)[0] * 1e9)

        state_history = propagate_truth_with_ephemeris(
            t_eval_s,
            state0_mci,
            mu_moon,
            mu_earth,
            mu_sun,
            ephemeris,
            rtol=1e-12,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            state_history,
            truth["state_history_mci_m_mps"],
            rtol=0.0,
            atol=4e-3,
        )


if __name__ == "__main__":
    unittest.main()
