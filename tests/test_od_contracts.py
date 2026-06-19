import unittest

import numpy as np

from lunar_od import (
    PassGeometry,
    RangeRatePhysicsConfig,
    Station,
    analyze_state_bias_correlation,
    compute_range_rate_residuals,
    compute_range_rate_residuals_analytic,
    measurement_covariance_matrix,
    measurement_sigma_vector,
    summarize_arc_observability_combinations,
    summarize_weighted_jacobian,
)


class ODContractTests(unittest.TestCase):
    def test_range_rate_weighting_order_is_range_rr_az_el_per_observation(self):
        stations = (
            _station("A", sigma_range_m=2.0, sigma_rr_mps=0.02, sigma_angle_rad=2e-5),
            _station("B", sigma_range_m=5.0, sigma_rr_mps=0.05, sigma_angle_rad=5e-5),
        )
        pass_geo = _pass_geo(np.array([0.0]), stations)
        obs_data = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0],
            ]
        )

        sigma = measurement_sigma_vector(obs_data, pass_geo, "range_rate")

        np.testing.assert_allclose(sigma, [2.0, 0.02, 2e-5, 2e-5, 5.0, 0.05, 5e-5, 5e-5])

    def test_measurement_covariance_r_is_diag_sigma_squared_in_residual_order(self):
        stations = (
            _station("A", sigma_range_m=2.0, sigma_rr_mps=0.02, sigma_angle_rad=2e-5),
            _station("B", sigma_range_m=5.0, sigma_rr_mps=0.05, sigma_angle_rad=5e-5),
        )
        pass_geo = _pass_geo(np.array([0.0]), stations)
        obs_data = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0],
            ]
        )

        r_cov = measurement_covariance_matrix(obs_data, pass_geo, "range_rate")

        expected_sigma = np.array([2.0, 0.02, 2e-5, 2e-5, 5.0, 0.05, 5e-5, 5e-5])
        np.testing.assert_allclose(r_cov, np.diag(expected_sigma**2), rtol=0.0, atol=0.0)

    def test_two_way_doppler_uses_same_mps_equivalent_r_order(self):
        station = _station("A", sigma_range_m=2.0, sigma_rr_mps=0.02, sigma_angle_rad=2e-5)
        pass_geo = _pass_geo(np.array([0.0]), (station,))
        object.__setattr__(
            pass_geo,
            "range_rate_physics",
            RangeRatePhysicsConfig(mode="two_way_counted_doppler", count_interval_s=60.0),
        )
        obs_data = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]])

        sigma = measurement_sigma_vector(obs_data, pass_geo, "range_rate")
        r_cov = measurement_covariance_matrix(obs_data, pass_geo, "range_rate")

        np.testing.assert_allclose(sigma, [2.0, 0.02, 2e-5, 2e-5])
        np.testing.assert_allclose(np.diag(r_cov), [4.0, 4e-4, 4e-10, 4e-10])

    def test_range_rate_residual_contract_is_observed_minus_computed(self):
        t_s = np.array([0.0])
        station = _station("A", sigma_range_m=2.0, sigma_rr_mps=0.02, sigma_angle_rad=2e-5)
        pass_geo = _pass_geo(t_s, (station,))
        state_history = np.array([[station.r_ecef_m[0] + 1000.0, station.r_ecef_m[1], station.r_ecef_m[2], 1.5, 0.0, 0.0]])
        obs_data = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]])
        _, h_meas = compute_range_rate_residuals(state_history, obs_data, pass_geo)

        offset = np.array([3.0, -0.02, 4e-5, -5e-5])
        obs_data[:, 1:5] = h_meas + offset.reshape(1, 4)
        residuals, _ = compute_range_rate_residuals(state_history, obs_data, pass_geo)

        np.testing.assert_allclose(residuals, offset, rtol=0.0, atol=1e-14)

    def test_range_rate_jacobian_is_computed_measurement_derivative(self):
        t_s = np.array([0.0])
        station = _station("A", sigma_range_m=2.0, sigma_rr_mps=0.02, sigma_angle_rad=2e-5)
        pass_geo = _pass_geo(t_s, (station,))
        state = np.array([station.r_ecef_m[0] + 1000.0, station.r_ecef_m[1] + 30.0, station.r_ecef_m[2] + 20.0, 1.5, 0.2, -0.1])
        state_history = state.reshape(1, 6)
        obs_data = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]])
        _, h0, h_tilde = compute_range_rate_residuals_analytic(state_history, obs_data, pass_geo)
        obs_data[:, 1:5] = h0
        residual0, _, h_tilde = compute_range_rate_residuals_analytic(state_history, obs_data, pass_geo)

        step = np.array([0.2, -0.1, 0.05, 1e-4, -2e-4, 3e-4])
        residual1, h1, _ = compute_range_rate_residuals_analytic((state + step).reshape(1, 6), obs_data, pass_geo)

        np.testing.assert_allclose(h1.reshape(-1) - h0.reshape(-1), h_tilde @ step, rtol=0.0, atol=2e-5)
        np.testing.assert_allclose(residual1 - residual0, -(h_tilde @ step), rtol=0.0, atol=2e-5)

    def test_state_bias_correlation_diagnostic_flags_coupling(self):
        covariance = np.eye(8)
        covariance[0, 6] = covariance[6, 0] = 0.5
        covariance[3, 7] = covariance[7, 3] = -0.25

        result = analyze_state_bias_correlation(covariance, num_state=6)

        self.assertEqual(result.num_bias, 2)
        self.assertAlmostEqual(result.correlation_matrix[0, 0], 0.5)
        self.assertAlmostEqual(result.correlation_matrix[3, 1], -0.25)
        self.assertAlmostEqual(result.max_abs_correlation, 0.5)

    def test_arc_combinations_can_improve_rank(self):
        arc1 = summarize_weighted_jacobian("position", 1, np.array([[1.0, 0.0], [0.0, 0.0]]), rank_tol=1e-12)
        arc2 = summarize_weighted_jacobian("position", 1, np.array([[0.0, 0.0], [0.0, 2.0]]), rank_tol=1e-12)

        combos = summarize_arc_observability_combinations(((1, arc1), (2, arc2)), rank_tol=1e-12)
        by_ids = {combo.arc_ids: combo.observability for combo in combos}

        self.assertEqual(by_ids[(1,)].rank, 1)
        self.assertEqual(by_ids[(2,)].rank, 1)
        self.assertEqual(by_ids[(1, 2)].rank, 2)
        self.assertEqual(by_ids[(1, 2)].rank_deficiency, 0)


def _station(name: str, *, sigma_range_m: float, sigma_rr_mps: float, sigma_angle_rad: float) -> Station:
    return Station(
        name=name,
        lat_deg=0.0,
        lon_deg=0.0,
        alt_m=0.0,
        color_rgb=(0.0, 0.0, 0.0),
        sigma_range_m=sigma_range_m,
        sigma_angle_rad=sigma_angle_rad,
        sigma_range_rate_mps=sigma_rr_mps,
    )


def _pass_geo(t_s, stations):
    return PassGeometry(
        t_s=np.asarray(t_s, dtype=float),
        earth_pos_mci_m=np.zeros((len(t_s), 3)),
        earth_vel_mci_mps=np.zeros((len(t_s), 3)),
        x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], len(t_s), axis=0),
        stations=tuple(stations),
        measurement_type="range_rate",
    )


if __name__ == "__main__":
    unittest.main()
