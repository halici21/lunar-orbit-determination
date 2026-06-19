import math
import unittest
from dataclasses import replace

import numpy as np

from lunar_od import (
    MeasurementNoiseConfig,
    PassGeometry,
    RangeRatePhysicsConfig,
    SquareRootUKFState,
    UKFAdaptiveConfig,
    UKFState,
    UnscentedTransformConfig,
    Station,
    assess_ukf_operational_stability,
    adapt_process_noise_scale,
    chi_square_nis_gate,
    compute_position_residuals_analytic,
    compute_range_rate_residuals_analytic,
    discretize_white_acceleration_process_noise,
    generate_measurement_noise,
    normalized_estimation_error_squared,
    normalized_innovation_squared,
    propagate_augmented_state,
    run_lunar_ukf,
    sigma_points,
    square_root_ukf_predict,
    square_root_ukf_update,
    ukf_predict,
    ukf_predict_update,
    ukf_update,
    unscented_mean_and_covariance,
)
from lunar_od.geometry import wrap_to_pi
from lunar_od.filters import _apply_state_constraints, _range_rate_measurement_from_state
from tests.slow import slow


class FilterTests(unittest.TestCase):
    def test_state_constraints_freeze_and_regularize_selected_biases(self):
        predicted = UKFState(
            x=np.array([0.0, 0.0, 2.0]),
            p=np.diag([1.0, 4.0, 9.0]),
        )
        updated = UKFState(
            x=np.array([1.0, 5.0, 8.0]),
            p=np.diag([0.5, 2.0, 3.0]),
        )

        constrained = _apply_state_constraints(
            updated,
            predicted,
            np.zeros(3),
            frozen_indices=(2,),
            regularization_std_by_state={1: 1.0},
            jitter=1e-12,
        )

        self.assertAlmostEqual(constrained.x[0], 1.0)
        self.assertLess(abs(constrained.x[1]), abs(updated.x[1]))
        self.assertEqual(constrained.x[2], predicted.x[2])
        self.assertEqual(constrained.p[2, 2], predicted.p[2, 2])

    def test_sigma_points_reconstruct_mean_and_covariance(self):
        mean = np.array([10.0, -2.0, 0.5])
        covariance = np.array(
            [
                [4.0, 0.2, -0.1],
                [0.2, 1.5, 0.3],
                [-0.1, 0.3, 0.8],
            ]
        )

        points, wm, wc = sigma_points(mean, covariance, UnscentedTransformConfig(alpha=0.4, beta=2.0))
        reconstructed_mean, reconstructed_covariance = unscented_mean_and_covariance(points, wm, wc)

        np.testing.assert_allclose(reconstructed_mean, mean, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(reconstructed_covariance, covariance, rtol=0.0, atol=5e-12)

    def test_predict_update_linear_constant_velocity_model(self):
        state = UKFState(
            x=np.array([0.0, 1.0]),
            p=np.diag([10.0, 1.0]),
        )
        dt = 2.0
        process_fn = lambda x: np.array([x[0] + dt * x[1], x[1]])
        measurement_fn = lambda x: np.array([x[0]])
        z = np.array([2.2])
        r = np.array([[0.25]])

        updated, diagnostics = ukf_predict_update(
            state,
            process_fn,
            z,
            measurement_fn,
            r,
            process_noise=np.diag([0.01, 0.001]),
            config=UnscentedTransformConfig(alpha=0.5),
        )

        self.assertLess(abs(updated.x[0] - z[0]), abs(2.0 - z[0]))
        self.assertLess(updated.p[0, 0], 10.0)
        self.assertTrue(np.all(np.linalg.eigvalsh(updated.p) > 0.0))
        self.assertEqual(diagnostics.innovation.shape, (1,))
        self.assertEqual(diagnostics.kalman_gain.shape, (2, 1))
        self.assertTrue(diagnostics.accepted)
        self.assertAlmostEqual(diagnostics.measurement_noise_scale, 1.0)
        self.assertGreaterEqual(diagnostics.normalized_innovation_squared, 0.0)

    def test_square_root_update_matches_standard_linear_ukf(self):
        covariance = np.diag([10.0, 1.0])
        standard = UKFState(x=np.array([0.0, 1.0]), p=covariance)
        square_root = SquareRootUKFState(x=standard.x.copy(), sqrt_p=np.linalg.cholesky(covariance))
        process_fn = lambda x: np.array([x[0] + 2.0 * x[1], x[1]])
        measurement_fn = lambda x: np.array([x[0]])
        config = UnscentedTransformConfig(alpha=0.5)

        standard_predicted, sigma_x, wm, wc = ukf_predict(
            standard,
            process_fn,
            process_noise=np.diag([0.01, 0.001]),
            config=config,
        )
        standard_updated, _ = ukf_update(
            standard_predicted,
            sigma_x,
            wm,
            wc,
            np.array([2.2]),
            measurement_fn,
            np.array([[0.25]]),
            config=config,
        )
        sqrt_predicted, sqrt_sigma_x, sqrt_wm, sqrt_wc = square_root_ukf_predict(
            square_root,
            process_fn,
            process_noise=np.diag([0.01, 0.001]),
            config=config,
        )
        sqrt_updated, _ = square_root_ukf_update(
            sqrt_predicted,
            sqrt_sigma_x,
            sqrt_wm,
            sqrt_wc,
            np.array([2.2]),
            measurement_fn,
            np.array([[0.25]]),
            config=config,
        )

        np.testing.assert_allclose(sqrt_updated.x, standard_updated.x, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(sqrt_updated.p, standard_updated.p, rtol=1e-9, atol=1e-10)
        self.assertTrue(np.all(np.diag(sqrt_updated.sqrt_p) > 0.0))

    def test_predict_adds_process_noise(self):
        state = UKFState(x=np.array([1.0, 2.0]), p=np.eye(2))
        q = np.diag([0.2, 0.3])
        predicted, _, _, _ = ukf_predict(
            state,
            lambda x: x,
            process_noise=q,
            config=UnscentedTransformConfig(alpha=0.5),
        )

        np.testing.assert_allclose(predicted.x, state.x, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(predicted.p, np.eye(2) + q, rtol=0.0, atol=5e-12)

    def test_predict_update_process_noise_affects_current_measurement_gain(self):
        state = UKFState(x=np.array([0.0]), p=np.array([[1.0]]))
        updated, diagnostics = ukf_predict_update(
            state,
            lambda x: x,
            np.array([1.0]),
            lambda x: x,
            np.array([[1.0]]),
            process_noise=np.array([[9.0]]),
            config=UnscentedTransformConfig(alpha=0.5),
        )

        expected_gain = 10.0 / 11.0
        expected_covariance = 10.0 / 11.0
        self.assertAlmostEqual(diagnostics.kalman_gain[0, 0], expected_gain, places=10)
        self.assertAlmostEqual(updated.x[0], expected_gain, places=10)
        self.assertAlmostEqual(updated.p[0, 0], expected_covariance, places=10)

    def test_white_acceleration_process_noise_discretization_scales_with_dt(self):
        acceleration_psd = np.diag([1.0, 2.0, 3.0])

        q = discretize_white_acceleration_process_noise(acceleration_psd, 2.0, state_size=8)

        np.testing.assert_allclose(q[:3, :3], (8.0 / 3.0) * acceleration_psd)
        np.testing.assert_allclose(q[:3, 3:6], 2.0 * acceleration_psd)
        np.testing.assert_allclose(q[3:6, :3], 2.0 * acceleration_psd)
        np.testing.assert_allclose(q[3:6, 3:6], 2.0 * acceleration_psd)
        np.testing.assert_allclose(q[6:, :], 0.0)
        np.testing.assert_allclose(q[:, 6:], 0.0)
        self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(q))), -1e-12)

    def test_white_acceleration_process_noise_rejects_invalid_psd(self):
        with self.assertRaises(ValueError):
            discretize_white_acceleration_process_noise(np.diag([1.0, -1.0, 1.0]), 1.0)

    def test_predict_supports_covariance_inflation(self):
        state = UKFState(x=np.array([1.0, 2.0]), p=np.eye(2))
        predicted, _, _, _ = ukf_predict(
            state,
            lambda x: x,
            covariance_inflation=1.5,
            config=UnscentedTransformConfig(alpha=0.5),
        )

        np.testing.assert_allclose(predicted.x, state.x, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(predicted.p, 1.5 * np.eye(2), rtol=0.0, atol=5e-12)

    def test_update_reports_nis_and_supports_adaptive_r_scaling(self):
        state = UKFState(x=np.array([0.0]), p=np.array([[1.0]]))
        predicted, sigma_x, wm, wc = ukf_predict(state, lambda x: x, config=UnscentedTransformConfig(alpha=0.5))
        updated, diagnostics = ukf_update(
            predicted,
            sigma_x,
            wm,
            wc,
            np.array([10.0]),
            lambda x: np.array([x[0]]),
            np.array([[1.0]]),
            adaptive_config=UKFAdaptiveConfig(adaptive_measurement_noise=True, max_measurement_noise_scale=10.0),
            config=UnscentedTransformConfig(alpha=0.5),
        )

        self.assertTrue(diagnostics.accepted)
        self.assertGreater(diagnostics.measurement_noise_scale, 1.0)
        self.assertLess(updated.x[0], 10.0)
        self.assertAlmostEqual(
            diagnostics.normalized_innovation_squared,
            normalized_innovation_squared(diagnostics.innovation, diagnostics.innovation_covariance),
        )

    def test_robust_student_t_update_downweights_outlier_component(self):
        state = UKFState(x=np.zeros(2), p=np.eye(2))
        config = UnscentedTransformConfig(alpha=0.5)
        predicted, sigma_x, wm, wc = ukf_predict(state, lambda x: x, config=config)

        plain, plain_diagnostics = ukf_update(
            predicted,
            sigma_x,
            wm,
            wc,
            np.array([20.0, 0.2]),
            lambda x: x,
            np.eye(2),
            config=config,
        )
        robust, robust_diagnostics = ukf_update(
            predicted,
            sigma_x,
            wm,
            wc,
            np.array([20.0, 0.2]),
            lambda x: x,
            np.eye(2),
            adaptive_config=UKFAdaptiveConfig(
                robust_measurement_update=True,
                robust_student_t_dof=4.0,
                robust_min_component_weight=0.05,
            ),
            config=config,
        )

        self.assertTrue(plain_diagnostics.accepted)
        self.assertTrue(robust_diagnostics.accepted)
        self.assertLess(robust_diagnostics.robust_component_weights[0], 0.1)
        self.assertGreater(robust_diagnostics.robust_component_weights[1], 0.95)
        self.assertGreater(robust_diagnostics.measurement_noise_scale, 1.0)
        self.assertLess(abs(robust.x[0]), abs(plain.x[0]))
        self.assertLess(abs(robust.x[1] - plain.x[1]), 0.02)

    def test_square_root_robust_update_matches_standard_linear_case(self):
        covariance = np.eye(2)
        config = UnscentedTransformConfig(alpha=0.5)
        adaptive = UKFAdaptiveConfig(
            robust_measurement_update=True,
            robust_loss="huber",
            robust_huber_threshold=2.0,
            robust_min_component_weight=0.1,
        )
        standard = UKFState(x=np.zeros(2), p=covariance)
        square_root = SquareRootUKFState(x=standard.x.copy(), sqrt_p=np.linalg.cholesky(covariance))

        standard_predicted, sigma_x, wm, wc = ukf_predict(standard, lambda x: x, config=config)
        sqrt_predicted, sqrt_sigma_x, sqrt_wm, sqrt_wc = square_root_ukf_predict(
            square_root,
            lambda x: x,
            config=config,
        )
        standard_updated, standard_diagnostics = ukf_update(
            standard_predicted,
            sigma_x,
            wm,
            wc,
            np.array([8.0, 0.5]),
            lambda x: x,
            np.eye(2),
            adaptive_config=adaptive,
            config=config,
        )
        sqrt_updated, sqrt_diagnostics = square_root_ukf_update(
            sqrt_predicted,
            sqrt_sigma_x,
            sqrt_wm,
            sqrt_wc,
            np.array([8.0, 0.5]),
            lambda x: x,
            np.eye(2),
            adaptive_config=adaptive,
            config=config,
        )

        np.testing.assert_allclose(sqrt_updated.x, standard_updated.x, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(sqrt_updated.p, standard_updated.p, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(
            sqrt_diagnostics.robust_component_weights,
            standard_diagnostics.robust_component_weights,
        )

    def test_update_rejects_measurement_when_nis_gate_is_exceeded(self):
        state = UKFState(x=np.array([0.0]), p=np.array([[1.0]]))
        predicted, sigma_x, wm, wc = ukf_predict(state, lambda x: x, config=UnscentedTransformConfig(alpha=0.5))
        updated, diagnostics = ukf_update(
            predicted,
            sigma_x,
            wm,
            wc,
            np.array([10.0]),
            lambda x: np.array([x[0]]),
            np.array([[1.0]]),
            adaptive_config=UKFAdaptiveConfig(nis_gate=1.0),
            config=UnscentedTransformConfig(alpha=0.5),
        )

        self.assertFalse(diagnostics.accepted)
        np.testing.assert_allclose(updated.x, predicted.x)
        np.testing.assert_allclose(updated.p, predicted.p)

    def test_chi_square_nis_gate_maps_three_sigma_to_measurement_dimension(self):
        self.assertAlmostEqual(chi_square_nis_gate(3), 14.156413609126675)
        self.assertAlmostEqual(chi_square_nis_gate(4), 16.251340813956187)
        with self.assertRaises(ValueError):
            chi_square_nis_gate(0)

    def test_update_component_gate_keeps_clean_measurement_components(self):
        state = UKFState(x=np.array([0.0, 0.0]), p=np.eye(2))
        predicted, sigma_x, wm, wc = ukf_predict(
            state,
            lambda x: x,
            config=UnscentedTransformConfig(alpha=0.5),
        )
        updated, diagnostics = ukf_update(
            predicted,
            sigma_x,
            wm,
            wc,
            np.array([20.0, 1.0]),
            lambda x: x,
            np.eye(2),
            adaptive_config=UKFAdaptiveConfig(component_nis_gate=9.0),
            config=UnscentedTransformConfig(alpha=0.5),
        )

        np.testing.assert_array_equal(diagnostics.accepted_components, [False, True])
        self.assertTrue(diagnostics.accepted)
        self.assertAlmostEqual(updated.x[0], 0.0, places=12)
        self.assertAlmostEqual(updated.x[1], 0.5, places=10)
        self.assertAlmostEqual(diagnostics.kalman_gain[0, 0], 0.0, places=12)

    def test_conditional_component_gate_respects_measurement_correlation(self):
        state = UKFState(x=np.zeros(2), p=np.eye(2))
        predicted, sigma_x, wm, wc = ukf_predict(
            state,
            lambda x: x,
            config=UnscentedTransformConfig(alpha=0.5),
        )
        correlated_noise = np.array([[1.0, 0.9], [0.9, 1.0]])

        _, marginal = ukf_update(
            predicted,
            sigma_x,
            wm,
            wc,
            np.array([4.0, 4.0]),
            lambda x: np.zeros(2),
            correlated_noise,
            adaptive_config=UKFAdaptiveConfig(component_nis_gate=9.0, component_gate_mode="marginal"),
            config=UnscentedTransformConfig(alpha=0.5),
        )
        _, conditional = ukf_update(
            predicted,
            sigma_x,
            wm,
            wc,
            np.array([4.0, 4.0]),
            lambda x: np.zeros(2),
            correlated_noise,
            adaptive_config=UKFAdaptiveConfig(component_nis_gate=9.0, component_gate_mode="conditional"),
            config=UnscentedTransformConfig(alpha=0.5),
        )

        np.testing.assert_array_equal(marginal.accepted_components, [False, False])
        np.testing.assert_array_equal(conditional.accepted_components, [True, True])

    def test_nees_utility_uses_estimation_error_and_covariance(self):
        nees = normalized_estimation_error_squared(
            np.array([1.0, 2.0]),
            np.array([0.0, 0.0]),
            np.diag([1.0, 4.0]),
        )

        self.assertAlmostEqual(nees, 2.0)

    def test_adaptive_process_noise_scale_tracks_normalized_nis(self):
        adaptive = UKFAdaptiveConfig(
            adaptive_process_noise=True,
            min_process_noise_scale=0.25,
            max_process_noise_scale=4.0,
            process_noise_adaptation_gain=0.5,
        )

        increased = adapt_process_noise_scale(1.0, nis=16.0, measurement_dim=4, adaptive=adaptive, enabled=True)
        decreased = adapt_process_noise_scale(1.0, nis=1.0, measurement_dim=4, adaptive=adaptive, enabled=True)
        disabled = adapt_process_noise_scale(1.0, nis=16.0, measurement_dim=4, adaptive=adaptive, enabled=False)
        clipped = adapt_process_noise_scale(4.0, nis=1600.0, measurement_dim=4, adaptive=adaptive, enabled=True)

        self.assertAlmostEqual(increased, 2.0)
        self.assertAlmostEqual(decreased, 0.5)
        self.assertAlmostEqual(disabled, 1.0)
        self.assertAlmostEqual(clipped, 4.0)

    def test_update_supports_wrapped_angle_residuals(self):
        state = UKFState(x=np.array([math.pi - 0.01]), p=np.array([[0.02]]))
        measurement = np.array([-math.pi + 0.02])
        measurement_residual = lambda value, reference: wrap_to_pi(np.asarray(value) - np.asarray(reference))

        updated, diagnostics = ukf_predict_update(
            state,
            lambda x: x,
            measurement,
            lambda x: np.array([x[0]]),
            np.array([[1e-3]]),
            config=UnscentedTransformConfig(alpha=0.5),
            measurement_residual_fn=measurement_residual,
        )

        self.assertAlmostEqual(float(diagnostics.innovation[0]), 0.03, delta=5e-3)
        self.assertLess(abs(float(wrap_to_pi(updated.x[0] - measurement[0]))), 0.05)

    def test_unscented_mean_respects_wrapped_angle_residuals(self):
        points = np.array([[math.pi - 0.01], [-math.pi + 0.02], [math.pi - 0.03]])
        weights = np.array([0.4, 0.3, 0.3])
        residual = lambda value, reference: wrap_to_pi(np.asarray(value) - np.asarray(reference))

        mean, covariance = unscented_mean_and_covariance(
            points,
            weights,
            weights,
            residual_fn=residual,
        )

        self.assertLess(abs(float(wrap_to_pi(mean[0] - math.pi))), 0.02)
        self.assertLess(covariance[0, 0], 1e-3)

    def test_lunar_ukf_position_noise_free_synthetic_arc_reduces_error(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_pass_s = np.arange(0.0, 361.0, 60.0)
        get_earth_pos = lambda t: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        get_sun_pos = lambda t: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))

        x_aug0 = np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")])
        x_aug_truth = propagate_augmented_state(
            t_pass_s,
            x_aug0,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            rtol=1e-12,
            atol=1e-13,
        )
        x_truth = x_aug_truth[:, :6]
        stations = (
            _synthetic_position_station(0.0, 0.0, 0.0),
            _synthetic_position_station(0.0, 90.0, 0.0),
            _synthetic_position_station(45.0, -30.0, 500.0),
        )
        pass_geo = PassGeometry(
            t_s=t_pass_s,
            earth_pos_mci_m=np.zeros((t_pass_s.size, 3)),
            earth_vel_mci_mps=np.zeros((t_pass_s.size, 3)),
            x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass_s.size, axis=0),
            stations=stations,
            measurement_type="position",
        )
        obs_data = _build_clean_position_observations(t_pass_s, x_truth, pass_geo)

        x_guess0 = x_true0 + np.array([40.0, -30.0, 20.0, 0.02, -0.015, 0.01])
        p0 = np.diag([80.0**2, 80.0**2, 80.0**2, 0.08**2, 0.08**2, 0.08**2])
        result = run_lunar_ukf(
            t_pass_s,
            obs_data,
            x_guess0,
            p0,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            process_noise=np.diag([0.01**2] * 3 + [1e-5**2] * 3),
            config=UnscentedTransformConfig(alpha=0.35),
            rtol=1e-12,
            atol=1e-13,
        )
        initial_position_error = float(np.linalg.norm(x_guess0[:3] - x_true0[:3]))
        final_position_error = float(np.linalg.norm(result.final_state[:3] - x_truth[-1, :3]))

        self.assertLess(final_position_error, initial_position_error)
        self.assertLess(final_position_error, 5.0)
        self.assertEqual(result.state_estimates.shape, (obs_data.shape[0], 6))
        self.assertEqual(result.covariances.shape, (obs_data.shape[0], 6, 6))
        self.assertEqual(result.innovation_covariances.shape, (obs_data.shape[0], 3, 3))
        self.assertEqual(result.normalized_innovation_squared.shape, (obs_data.shape[0],))
        self.assertEqual(result.process_noise_scales.shape, (obs_data.shape[0],))
        self.assertTrue(np.all(result.accepted_updates))
        self.assertTrue(np.all(np.linalg.eigvalsh(result.final_covariance) > 0.0))

    def test_lunar_ukf_stm_linearization_matches_full_sigma_propagation(self):
        """STM-linearized predict (use_stm_linearization=True) must track the
        exact full-sigma-point predict (False) within the small-spread bound.

        The default is False (exact, full-sigma); this test keeps the
        STM-linearization code path covered and pins its accuracy.
        """
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_pass_s = np.arange(0.0, 481.0, 60.0)
        get_earth_pos = lambda t: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        get_sun_pos = lambda t: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))

        x_aug0 = np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")])
        x_truth = propagate_augmented_state(
            t_pass_s, x_aug0, mu_moon, 0.0, 0.0, get_earth_pos, get_sun_pos, rtol=1e-12, atol=1e-13,
        )[:, :6]
        stations = (
            _synthetic_position_station(0.0, 0.0, 0.0),
            _synthetic_position_station(0.0, 90.0, 0.0),
            _synthetic_position_station(45.0, -30.0, 500.0),
        )
        pass_geo = PassGeometry(
            t_s=t_pass_s,
            earth_pos_mci_m=np.zeros((t_pass_s.size, 3)),
            earth_vel_mci_mps=np.zeros((t_pass_s.size, 3)),
            x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass_s.size, axis=0),
            stations=stations,
            measurement_type="position",
        )
        obs_data = _build_clean_position_observations(t_pass_s, x_truth, pass_geo)
        x_guess0 = x_true0 + np.array([40.0, -30.0, 20.0, 0.02, -0.015, 0.01])
        p0 = np.diag([80.0**2, 80.0**2, 80.0**2, 0.08**2, 0.08**2, 0.08**2])

        def _run(use_stm: bool):
            return run_lunar_ukf(
                t_pass_s, obs_data, x_guess0, p0, pass_geo, mu_moon, 0.0, 0.0,
                get_earth_pos, get_sun_pos,
                process_noise=np.diag([0.01**2] * 3 + [1e-5**2] * 3),
                config=UnscentedTransformConfig(alpha=0.35),
                rtol=1e-12, atol=1e-13, use_stm_linearization=use_stm,
            )

        full = _run(False)
        linearized = _run(True)
        # Small-spread bound from the plan: < 1e-3 m over a 60 s predict interval.
        np.testing.assert_allclose(linearized.final_state[:3], full.final_state[:3], atol=1e-3)
        np.testing.assert_allclose(linearized.final_state[3:], full.final_state[3:], atol=1e-6)

    def test_lunar_ukf_stm_linearization_holds_on_long_arc_wide_spread(self):
        """Stress variant: long arc, large predict step, and wide sigma spread
        (alpha=1.0) — the regime where STM linearization is most strained.

        Empirically the divergence is driven by the predict step size (dt),
        not by total arc length or alpha. This pins the worst-case error so a
        regression that degrades the STM path is caught even though the default
        is now the exact full-sigma path (use_stm_linearization=False).
        """
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        # 2 h arc, 120 s predict step (2x the nominal), wide spread.
        t_pass_s = np.arange(0.0, 7201.0, 120.0)
        get_earth_pos = lambda t: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        get_sun_pos = lambda t: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))

        x_aug0 = np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")])
        x_truth = propagate_augmented_state(
            t_pass_s, x_aug0, mu_moon, 0.0, 0.0, get_earth_pos, get_sun_pos, rtol=1e-12, atol=1e-13,
        )[:, :6]
        stations = (
            _synthetic_position_station(0.0, 0.0, 0.0),
            _synthetic_position_station(0.0, 90.0, 0.0),
            _synthetic_position_station(45.0, -30.0, 500.0),
        )
        pass_geo = PassGeometry(
            t_s=t_pass_s,
            earth_pos_mci_m=np.zeros((t_pass_s.size, 3)),
            earth_vel_mci_mps=np.zeros((t_pass_s.size, 3)),
            x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass_s.size, axis=0),
            stations=stations,
            measurement_type="position",
        )
        obs_data = _build_clean_position_observations(t_pass_s, x_truth, pass_geo)
        x_guess0 = x_true0 + np.array([40.0, -30.0, 20.0, 0.02, -0.015, 0.01])
        p0 = np.diag([80.0**2, 80.0**2, 80.0**2, 0.08**2, 0.08**2, 0.08**2])

        def _run(use_stm: bool):
            return run_lunar_ukf(
                t_pass_s, obs_data, x_guess0, p0, pass_geo, mu_moon, 0.0, 0.0,
                get_earth_pos, get_sun_pos,
                process_noise=np.diag([0.01**2] * 3 + [1e-5**2] * 3),
                config=UnscentedTransformConfig(alpha=1.0),
                rtol=1e-12, atol=1e-13, use_stm_linearization=use_stm,
            )

        full = _run(False)
        linearized = _run(True)
        # Measured worst-case here ~1.5e-4 m / ~1.7e-7 m/s; pin with ~3x margin.
        np.testing.assert_allclose(linearized.final_state[:3], full.final_state[:3], atol=5e-4)
        np.testing.assert_allclose(linearized.final_state[3:], full.final_state[3:], atol=5e-7)

    @slow
    def test_lunar_ukf_position_seeded_monte_carlo_is_statistically_consistent(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_pass_s = np.arange(0.0, 361.0, 60.0)
        get_earth_pos = lambda t: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        get_sun_pos = lambda t: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))

        x_aug0 = np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")])
        x_truth = propagate_augmented_state(
            t_pass_s,
            x_aug0,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            rtol=1e-12,
            atol=1e-13,
        )[:, :6]
        stations = (
            _synthetic_position_station(0.0, 0.0, 0.0),
            _synthetic_position_station(0.0, 90.0, 0.0),
            _synthetic_position_station(45.0, -30.0, 500.0),
        )
        pass_geo = PassGeometry(
            t_s=t_pass_s,
            earth_pos_mci_m=np.zeros((t_pass_s.size, 3)),
            earth_vel_mci_mps=np.zeros((t_pass_s.size, 3)),
            x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass_s.size, axis=0),
            stations=stations,
            measurement_type="position",
        )
        clean_obs = _build_clean_position_observations(t_pass_s, x_truth, pass_geo)
        p0 = np.diag([20.0**2] * 3 + [0.02**2] * 3)
        rng = np.random.default_rng(20260608)
        nis_values = []
        final_nees = []

        for _ in range(20):
            obs_data = clean_obs.copy()
            obs_data[:, 1] += rng.normal(0.0, 1.0, obs_data.shape[0])
            obs_data[:, 2] = wrap_to_pi(obs_data[:, 2] + rng.normal(0.0, 1e-5, obs_data.shape[0]))
            obs_data[:, 3] = wrap_to_pi(obs_data[:, 3] + rng.normal(0.0, 1e-5, obs_data.shape[0]))
            x_guess0 = x_true0 + rng.multivariate_normal(np.zeros(6), p0)

            result = run_lunar_ukf(
                t_pass_s,
                obs_data,
                x_guess0,
                p0,
                pass_geo,
                mu_moon,
                0.0,
                0.0,
                get_earth_pos,
                get_sun_pos,
                process_noise=None,
                config=UnscentedTransformConfig(alpha=0.35),
                rtol=1e-12,
                atol=1e-13,
            )
            nis_values.extend(result.normalized_innovation_squared)
            final_nees.append(
                normalized_estimation_error_squared(
                    result.final_state,
                    x_truth[-1],
                    result.final_covariance,
                )
            )

        self.assertAlmostEqual(float(np.mean(nis_values)), 3.0, delta=0.5)
        self.assertAlmostEqual(float(np.mean(final_nees)), 6.0, delta=2.0)

    def test_lunar_ukf_position_global_bias_recovery_synthetic_arc(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_pass_s = np.arange(0.0, 361.0, 60.0)
        get_earth_pos = lambda t: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        get_sun_pos = lambda t: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))

        x_aug0 = np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")])
        x_truth = propagate_augmented_state(
            t_pass_s,
            x_aug0,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            rtol=1e-12,
            atol=1e-13,
        )[:, :6]
        stations = (
            _synthetic_position_station(0.0, 0.0, 0.0),
            _synthetic_position_station(0.0, 90.0, 0.0),
            _synthetic_position_station(45.0, -30.0, 500.0),
        )
        pass_geo = PassGeometry(
            t_s=t_pass_s,
            earth_pos_mci_m=np.zeros((t_pass_s.size, 3)),
            earth_vel_mci_mps=np.zeros((t_pass_s.size, 3)),
            x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass_s.size, axis=0),
            stations=stations,
            measurement_type="position",
        )
        obs_data = _build_clean_position_observations(t_pass_s, x_truth, pass_geo)
        true_bias = np.array([20.0, 1.2e-5, -8.0e-6])
        obs_data[:, 1:4] += true_bias.reshape(1, 3)

        x_guess0 = np.concatenate([x_true0, np.zeros(3)])
        p0 = np.diag([5.0**2] * 3 + [0.01**2] * 3 + [100.0**2, 1e-4**2, 1e-4**2])
        result = run_lunar_ukf(
            t_pass_s,
            obs_data,
            x_guess0,
            p0,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            process_noise=np.diag([0.01**2] * 3 + [1e-5**2] * 3),
            config=UnscentedTransformConfig(alpha=0.35),
            rtol=1e-12,
            atol=1e-13,
        )

        bias_error = np.abs(result.final_state[6:9] - true_bias)
        self.assertEqual(result.state_estimates.shape, (obs_data.shape[0], 9))
        self.assertLess(bias_error[0], 3.0)
        self.assertLess(bias_error[1], 4e-6)
        self.assertLess(bias_error[2], 4e-6)
        self.assertTrue(np.all(np.linalg.eigvalsh(result.final_covariance) > 0.0))

    @slow
    def test_lunar_ukf_range_rate_global_bias_monte_carlo_is_consistent(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_pass_s = np.arange(0.0, 361.0, 60.0)
        get_earth_pos = lambda t: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        get_sun_pos = lambda t: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        x_truth = propagate_augmented_state(
            t_pass_s,
            np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")]),
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            rtol=1e-12,
            atol=1e-13,
        )[:, :6]
        stations = (
            _synthetic_range_rate_station(0.0, 0.0, 0.0),
            _synthetic_range_rate_station(0.0, 90.0, 0.0),
            _synthetic_range_rate_station(45.0, -30.0, 500.0),
        )
        pass_geo = PassGeometry(
            t_s=t_pass_s,
            earth_pos_mci_m=np.zeros((t_pass_s.size, 3)),
            earth_vel_mci_mps=np.zeros((t_pass_s.size, 3)),
            x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass_s.size, axis=0),
            stations=stations,
            measurement_type="range_rate",
        )
        clean_obs = _build_clean_range_rate_observations(t_pass_s, x_truth, pass_geo)
        dynamic_prior = np.diag([5.0**2] * 3 + [0.01**2] * 3)
        bias_prior = np.diag([30.0**2, 3e-4**2, 3e-5**2, 3e-5**2])
        p0 = np.zeros((10, 10), dtype=float)
        p0[:6, :6] = dynamic_prior
        p0[6:, 6:] = bias_prior
        rng = np.random.default_rng(20260608)
        nis_values = []
        final_nees = []

        for _ in range(12):
            true_bias = rng.multivariate_normal(np.zeros(4), bias_prior)
            obs_data = clean_obs.copy()
            obs_data[:, 1:5] += true_bias
            obs_data[:, 1] += rng.normal(0.0, 1.0, obs_data.shape[0])
            obs_data[:, 2] += rng.normal(0.0, 1e-4, obs_data.shape[0])
            obs_data[:, 3] = wrap_to_pi(obs_data[:, 3] + rng.normal(0.0, 1e-5, obs_data.shape[0]))
            obs_data[:, 4] = wrap_to_pi(obs_data[:, 4] + rng.normal(0.0, 1e-5, obs_data.shape[0]))
            x_guess0 = np.concatenate(
                [
                    x_true0 + rng.multivariate_normal(np.zeros(6), dynamic_prior),
                    np.zeros(4),
                ]
            )
            augmented_truth = np.concatenate([x_truth[-1], true_bias])

            result = run_lunar_ukf(
                t_pass_s,
                obs_data,
                x_guess0,
                p0,
                pass_geo,
                mu_moon,
                0.0,
                0.0,
                get_earth_pos,
                get_sun_pos,
                process_noise=None,
                config=UnscentedTransformConfig(alpha=0.35),
                rtol=1e-12,
                atol=1e-13,
            )
            nis_values.extend(result.normalized_innovation_squared)
            final_nees.append(
                normalized_estimation_error_squared(
                    result.final_state,
                    augmented_truth,
                    result.final_covariance,
                )
            )

        self.assertAlmostEqual(float(np.mean(nis_values)), 4.0, delta=0.7)
        self.assertAlmostEqual(float(np.mean(final_nees)), 10.0, delta=3.0)

    def test_lunar_ukf_geometric_range_rate_noise_free_synthetic_arc_reduces_error(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_pass_s = np.arange(0.0, 361.0, 60.0)
        get_earth_pos = lambda t: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        get_sun_pos = lambda t: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))

        x_aug0 = np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")])
        x_aug_truth = propagate_augmented_state(
            t_pass_s,
            x_aug0,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            rtol=1e-12,
            atol=1e-13,
        )
        x_truth = x_aug_truth[:, :6]
        stations = (
            _synthetic_range_rate_station(0.0, 0.0, 0.0),
            _synthetic_range_rate_station(0.0, 90.0, 0.0),
            _synthetic_range_rate_station(45.0, -30.0, 500.0),
        )
        pass_geo = PassGeometry(
            t_s=t_pass_s,
            earth_pos_mci_m=np.zeros((t_pass_s.size, 3)),
            earth_vel_mci_mps=np.zeros((t_pass_s.size, 3)),
            x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass_s.size, axis=0),
            stations=stations,
            measurement_type="range_rate",
        )
        obs_data = _build_clean_range_rate_observations(t_pass_s, x_truth, pass_geo)

        x_guess0 = x_true0 + np.array([40.0, -30.0, 20.0, 0.02, -0.015, 0.01])
        p0 = np.diag([80.0**2, 80.0**2, 80.0**2, 0.08**2, 0.08**2, 0.08**2])
        result = run_lunar_ukf(
            t_pass_s,
            obs_data,
            x_guess0,
            p0,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            process_noise=np.diag([0.01**2] * 3 + [1e-5**2] * 3),
            config=UnscentedTransformConfig(alpha=0.35),
            rtol=1e-12,
            atol=1e-13,
        )

        initial_position_error = float(np.linalg.norm(x_guess0[:3] - x_true0[:3]))
        initial_velocity_error = float(np.linalg.norm(x_guess0[3:] - x_true0[3:]))
        final_position_error = float(np.linalg.norm(result.final_state[:3] - x_truth[-1, :3]))
        final_velocity_error = float(np.linalg.norm(result.final_state[3:] - x_truth[-1, 3:]))

        self.assertLess(final_position_error, initial_position_error)
        self.assertLess(final_velocity_error, initial_velocity_error)
        self.assertLess(final_position_error, 5.0)
        self.assertLess(final_velocity_error, 0.01)
        self.assertEqual(result.innovations.shape, (obs_data.shape[0], 4))
        self.assertEqual(result.innovation_covariances.shape, (obs_data.shape[0], 4, 4))
        self.assertTrue(np.all(result.measurement_noise_scales >= 1.0))
        self.assertTrue(np.all(np.linalg.eigvalsh(result.final_covariance) > 0.0))

    @slow
    def test_lunar_ukf_range_rate_seeded_monte_carlo_is_statistically_consistent(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_pass_s = np.arange(0.0, 361.0, 60.0)
        get_earth_pos = lambda t: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        get_sun_pos = lambda t: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))

        x_aug0 = np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")])
        x_truth = propagate_augmented_state(
            t_pass_s,
            x_aug0,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            rtol=1e-12,
            atol=1e-13,
        )[:, :6]
        stations = (
            _synthetic_range_rate_station(0.0, 0.0, 0.0),
            _synthetic_range_rate_station(0.0, 90.0, 0.0),
            _synthetic_range_rate_station(45.0, -30.0, 500.0),
        )
        pass_geo = PassGeometry(
            t_s=t_pass_s,
            earth_pos_mci_m=np.zeros((t_pass_s.size, 3)),
            earth_vel_mci_mps=np.zeros((t_pass_s.size, 3)),
            x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass_s.size, axis=0),
            stations=stations,
            measurement_type="range_rate",
        )
        clean_obs = _build_clean_range_rate_observations(t_pass_s, x_truth, pass_geo)
        p0 = np.diag([20.0**2] * 3 + [0.02**2] * 3)
        rng = np.random.default_rng(20260608)
        nis_values = []
        final_nees = []

        for _ in range(20):
            obs_data = clean_obs.copy()
            obs_data[:, 1] += rng.normal(0.0, 1.0, obs_data.shape[0])
            obs_data[:, 2] += rng.normal(0.0, 1e-4, obs_data.shape[0])
            obs_data[:, 3] = wrap_to_pi(obs_data[:, 3] + rng.normal(0.0, 1e-5, obs_data.shape[0]))
            obs_data[:, 4] = wrap_to_pi(obs_data[:, 4] + rng.normal(0.0, 1e-5, obs_data.shape[0]))
            x_guess0 = x_true0 + rng.multivariate_normal(np.zeros(6), p0)

            result = run_lunar_ukf(
                t_pass_s,
                obs_data,
                x_guess0,
                p0,
                pass_geo,
                mu_moon,
                0.0,
                0.0,
                get_earth_pos,
                get_sun_pos,
                process_noise=None,
                config=UnscentedTransformConfig(alpha=0.35),
                rtol=1e-12,
                atol=1e-13,
            )
            nis_values.extend(result.normalized_innovation_squared)
            final_nees.append(
                normalized_estimation_error_squared(
                    result.final_state,
                    x_truth[-1],
                    result.final_covariance,
                )
            )

        self.assertAlmostEqual(float(np.mean(nis_values)), 4.0, delta=0.6)
        self.assertAlmostEqual(float(np.mean(final_nees)), 6.0, delta=2.0)

    def test_lunar_ukf_range_rate_nis_gate_rejects_large_outlier(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_pass_s = np.arange(0.0, 361.0, 60.0)
        get_earth_pos = lambda t: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        get_sun_pos = lambda t: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        x_truth = propagate_augmented_state(
            t_pass_s,
            np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")]),
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            rtol=1e-12,
            atol=1e-13,
        )[:, :6]
        stations = (
            _synthetic_range_rate_station(0.0, 0.0, 0.0),
            _synthetic_range_rate_station(0.0, 90.0, 0.0),
            _synthetic_range_rate_station(45.0, -30.0, 500.0),
        )
        pass_geo = PassGeometry(
            t_s=t_pass_s,
            earth_pos_mci_m=np.zeros((t_pass_s.size, 3)),
            earth_vel_mci_mps=np.zeros((t_pass_s.size, 3)),
            x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass_s.size, axis=0),
            stations=stations,
            measurement_type="range_rate",
        )
        obs_data = _build_clean_range_rate_observations(t_pass_s, x_truth, pass_geo)
        obs_data[10, 1] += 10_000.0
        x_guess0 = x_true0 + np.array([20.0, -15.0, 10.0, 0.01, -0.0075, 0.005])
        p0 = np.diag([20.0**2] * 3 + [0.02**2] * 3)

        ungated = run_lunar_ukf(
            t_pass_s,
            obs_data,
            x_guess0,
            p0,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            config=UnscentedTransformConfig(alpha=0.35),
            rtol=1e-12,
            atol=1e-13,
        )
        gated = run_lunar_ukf(
            t_pass_s,
            obs_data,
            x_guess0,
            p0,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            adaptive_config=UKFAdaptiveConfig(nis_gate=18.47),
            config=UnscentedTransformConfig(alpha=0.35),
            rtol=1e-12,
            atol=1e-13,
        )

        ungated_error = np.linalg.norm(ungated.final_state[:3] - x_truth[-1, :3])
        gated_error = np.linalg.norm(gated.final_state[:3] - x_truth[-1, :3])
        self.assertEqual(np.count_nonzero(~gated.accepted_updates), 1)
        self.assertLess(gated_error, 1.0)
        self.assertGreater(ungated_error, 100.0 * gated_error)

    @slow
    def test_lunar_ukf_adaptive_q_reduces_dynamics_mismatch_error(self):
        mu_moon, x_true0, t_pass_s, x_truth, pass_geo, clean_obs, get_earth_pos, get_sun_pos = (
            _synthetic_range_rate_case(duration_s=1200.0)
        )
        p0 = np.diag([20.0**2] * 3 + [0.02**2] * 3)
        rng = np.random.default_rng(20260609)
        fixed_errors = []
        adaptive_errors = []
        adaptive_scales = []

        for _ in range(4):
            obs_data = _add_range_rate_noise(clean_obs, rng)
            x_guess0 = x_true0 + rng.multivariate_normal(np.zeros(6), p0)
            fixed = run_lunar_ukf(
                t_pass_s,
                obs_data,
                x_guess0,
                p0,
                pass_geo,
                mu_moon * (1.0 - 1e-5),
                0.0,
                0.0,
                get_earth_pos,
                get_sun_pos,
                process_noise=1e-10,
                process_noise_model="continuous_white_acceleration",
                config=UnscentedTransformConfig(alpha=0.35),
                rtol=1e-12,
                atol=1e-13,
            )
            adaptive = run_lunar_ukf(
                t_pass_s,
                obs_data,
                x_guess0,
                p0,
                pass_geo,
                mu_moon * (1.0 - 1e-5),
                0.0,
                0.0,
                get_earth_pos,
                get_sun_pos,
                process_noise=1e-10,
                process_noise_model="continuous_white_acceleration",
                adaptive_config=UKFAdaptiveConfig(
                    adaptive_process_noise=True,
                    max_process_noise_scale=1e5,
                    process_noise_adaptation_gain=0.3,
                ),
                config=UnscentedTransformConfig(alpha=0.35),
                rtol=1e-12,
                atol=1e-13,
            )
            fixed_errors.append(np.linalg.norm(fixed.final_state[:3] - x_truth[-1, :3]))
            adaptive_errors.append(np.linalg.norm(adaptive.final_state[:3] - x_truth[-1, :3]))
            adaptive_scales.append(adaptive.process_noise_scales[-1])

        self.assertLess(float(np.mean(adaptive_errors)), 0.7 * float(np.mean(fixed_errors)))
        self.assertGreater(float(np.mean(adaptive_scales)), 2.0)

    @slow
    def test_lunar_ukf_adaptive_r_handles_underestimated_measurement_noise(self):
        mu_moon, x_true0, t_pass_s, x_truth, pass_geo, clean_obs, get_earth_pos, get_sun_pos = (
            _synthetic_range_rate_case(duration_s=360.0)
        )
        p0 = np.diag([20.0**2] * 3 + [0.02**2] * 3)
        rng = np.random.default_rng(20260610)
        trials = []
        for _ in range(8):
            obs_data = clean_obs.copy()
            obs_data[:, 1] += rng.normal(0.0, 5.0, obs_data.shape[0])
            obs_data[:, 2] += rng.normal(0.0, 5e-4, obs_data.shape[0])
            obs_data[:, 3] = wrap_to_pi(obs_data[:, 3] + rng.normal(0.0, 5e-5, obs_data.shape[0]))
            obs_data[:, 4] = wrap_to_pi(obs_data[:, 4] + rng.normal(0.0, 5e-5, obs_data.shape[0]))
            trials.append((obs_data, x_true0 + rng.multivariate_normal(np.zeros(6), p0)))

        metrics = {}
        for label, adaptive_config in (
            ("fixed", UKFAdaptiveConfig()),
            ("adaptive", UKFAdaptiveConfig(adaptive_measurement_noise=True, max_measurement_noise_scale=100.0)),
        ):
            nis_values = []
            position_errors = []
            noise_scales = []
            for obs_data, x_guess0 in trials:
                result = run_lunar_ukf(
                    t_pass_s,
                    obs_data,
                    x_guess0,
                    p0,
                    pass_geo,
                    mu_moon,
                    0.0,
                    0.0,
                    get_earth_pos,
                    get_sun_pos,
                    adaptive_config=adaptive_config,
                    config=UnscentedTransformConfig(alpha=0.35),
                    rtol=1e-12,
                    atol=1e-13,
                )
                nis_values.extend(result.normalized_innovation_squared)
                position_errors.append(np.linalg.norm(result.final_state[:3] - x_truth[-1, :3]))
                noise_scales.extend(result.measurement_noise_scales)
            metrics[label] = (
                float(np.mean(nis_values)),
                float(np.mean(position_errors)),
                float(np.mean(noise_scales)),
            )

        self.assertLess(metrics["adaptive"][0], 0.2 * metrics["fixed"][0])
        self.assertLess(metrics["adaptive"][1], 1.75 * metrics["fixed"][1])
        self.assertGreater(metrics["adaptive"][2], 2.0)

    def test_large_initial_error_is_detectable_from_inconsistent_nis(self):
        mu_moon, x_true0, t_pass_s, x_truth, pass_geo, obs_data, get_earth_pos, get_sun_pos = (
            _synthetic_range_rate_case(duration_s=360.0)
        )
        offset = np.array([50_000.0, -37_500.0, 25_000.0, 25.0, -18.75, 12.5])
        p0 = np.diag([100_000.0**2] * 3 + [100.0**2] * 3)

        result = run_lunar_ukf(
            t_pass_s,
            obs_data,
            x_true0 + offset,
            p0,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            config=UnscentedTransformConfig(alpha=0.35),
            rtol=1e-12,
            atol=1e-13,
        )

        final_error = np.linalg.norm(result.final_state[:3] - x_truth[-1, :3])
        self.assertLess(final_error, np.linalg.norm(offset[:3]))
        self.assertGreater(float(np.mean(result.normalized_innovation_squared)), 3.0 * 4.0)

    def test_lunar_ukf_range_rate_global_bias_recovery_synthetic_arc(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_pass_s = np.arange(0.0, 361.0, 60.0)
        get_earth_pos = lambda t: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        get_sun_pos = lambda t: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))

        x_aug0 = np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")])
        x_truth = propagate_augmented_state(
            t_pass_s,
            x_aug0,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            rtol=1e-12,
            atol=1e-13,
        )[:, :6]
        stations = (
            _synthetic_range_rate_station(0.0, 0.0, 0.0),
            _synthetic_range_rate_station(0.0, 90.0, 0.0),
            _synthetic_range_rate_station(45.0, -30.0, 500.0),
        )
        pass_geo = PassGeometry(
            t_s=t_pass_s,
            earth_pos_mci_m=np.zeros((t_pass_s.size, 3)),
            earth_vel_mci_mps=np.zeros((t_pass_s.size, 3)),
            x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass_s.size, axis=0),
            stations=stations,
            measurement_type="range_rate",
        )
        obs_data = _build_clean_range_rate_observations(t_pass_s, x_truth, pass_geo)
        true_bias = np.array([16.0, 2.0e-4, 1.0e-5, -7.0e-6])
        obs_data[:, 1:5] += true_bias.reshape(1, 4)

        x_guess0 = np.concatenate([x_true0, np.zeros(4)])
        p0 = np.diag([5.0**2] * 3 + [0.01**2] * 3 + [100.0**2, 1e-3**2, 1e-4**2, 1e-4**2])
        result = run_lunar_ukf(
            t_pass_s,
            obs_data,
            x_guess0,
            p0,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            process_noise=np.diag([0.01**2] * 3 + [1e-5**2] * 3),
            config=UnscentedTransformConfig(alpha=0.35),
            rtol=1e-12,
            atol=1e-13,
        )

        bias_error = np.abs(result.final_state[6:10] - true_bias)
        self.assertEqual(result.state_estimates.shape, (obs_data.shape[0], 10))
        self.assertLess(bias_error[0], 3.0)
        self.assertLess(bias_error[1], 8e-5)
        self.assertLess(bias_error[2], 4e-6)
        self.assertLess(bias_error[3], 4e-6)
        self.assertTrue(np.all(np.linalg.eigvalsh(result.final_covariance) > 0.0))

    def test_lunar_ukf_square_root_short_soak_reports_operational_stability(self):
        mu_moon, x_true0, t_pass_s, x_truth, pass_geo, clean_obs, get_earth_pos, get_sun_pos = (
            _synthetic_range_rate_case(duration_s=900.0)
        )
        nominal_r = np.diag([1.0, 1e-8, 1e-10, 1e-10])
        rng = np.random.default_rng(20260612)
        obs_data = clean_obs.copy()
        obs_data[:, 1:5] += generate_measurement_noise(
            obs_data.shape[0],
            nominal_r,
            rng=rng,
            config=MeasurementNoiseConfig(model="student_t", student_t_dof=5.0),
        )
        obs_data[::19, 2] += 8e-4
        obs_data[:, 3:5] = wrap_to_pi(obs_data[:, 3:5])

        initial_offset = np.array([20.0, -15.0, 10.0, 0.01, -0.008, 0.006])
        x_guess0 = x_true0 + initial_offset
        p0 = np.diag([50.0**2] * 3 + [0.03**2] * 3)
        result = run_lunar_ukf(
            t_pass_s,
            obs_data,
            x_guess0,
            p0,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            process_noise=1e-9,
            process_noise_model="continuous_white_acceleration",
            covariance_form="square_root",
            adaptive_config=UKFAdaptiveConfig(
                robust_measurement_update=True,
                robust_loss="student_t",
                robust_student_t_dof=5.0,
                component_nis_gate=36.0,
                component_gate_mode="conditional",
            ),
            config=UnscentedTransformConfig(alpha=0.35),
            rtol=1e-11,
            atol=1e-12,
        )

        stability = assess_ukf_operational_stability(
            result,
            max_covariance_condition_number=1e20,
            max_normalized_innovation_squared=1e8,
            min_accepted_update_fraction=0.7,
        )
        final_position_error = np.linalg.norm(result.final_state[:3] - x_truth[-1, :3])
        self.assertTrue(stability.stable, stability.failures)
        self.assertGreater(stability.min_covariance_eigenvalue, 0.0)
        self.assertGreater(stability.robust_reweighted_fraction, 0.0)
        self.assertEqual(result.robust_component_weights.shape, (obs_data.shape[0], 4))
        self.assertLess(final_position_error, np.linalg.norm(initial_offset[:3]))

    @slow
    def test_lunar_ukf_range_rate_station_full_bias_recovery_synthetic_arc(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_pass_s = np.arange(0.0, 481.0, 60.0)
        get_earth_pos = lambda t: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        get_sun_pos = lambda t: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))

        x_aug0 = np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")])
        x_truth = propagate_augmented_state(
            t_pass_s,
            x_aug0,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            rtol=1e-12,
            atol=1e-13,
        )[:, :6]
        stations = (
            _synthetic_range_rate_station(0.0, 0.0, 0.0),
            _synthetic_range_rate_station(0.0, 90.0, 0.0),
            _synthetic_range_rate_station(45.0, -30.0, 500.0),
        )
        pass_geo = PassGeometry(
            t_s=t_pass_s,
            earth_pos_mci_m=np.zeros((t_pass_s.size, 3)),
            earth_vel_mci_mps=np.zeros((t_pass_s.size, 3)),
            x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass_s.size, axis=0),
            stations=stations,
            measurement_type="range_rate",
        )
        obs_data = _build_clean_range_rate_observations(t_pass_s, x_truth, pass_geo)
        true_bias = np.array(
            [
                10.0,
                1.0e-4,
                8.0e-6,
                -6.0e-6,
                -12.0,
                -1.5e-4,
                -9.0e-6,
                7.0e-6,
                7.0,
                2.0e-4,
                6.0e-6,
                -5.0e-6,
            ]
        )
        _add_rr_station_full_biases(obs_data, true_bias)

        x_guess0 = np.concatenate([x_true0, np.zeros(true_bias.size)])
        bias_prior_diag = np.tile(np.array([100.0**2, 1e-3**2, 1e-4**2, 1e-4**2]), len(stations))
        p0 = np.diag([0.1**2] * 3 + [1e-4**2] * 3 + bias_prior_diag.tolist())
        result = run_lunar_ukf(
            t_pass_s,
            obs_data,
            x_guess0,
            p0,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            bias_mode="station_full",
            process_noise=None,
            config=UnscentedTransformConfig(alpha=0.35),
            rtol=1e-12,
            atol=1e-13,
        )
        square_root_result = run_lunar_ukf(
            t_pass_s,
            obs_data,
            x_guess0,
            p0,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            bias_mode="station_full",
            process_noise=None,
            covariance_form="square_root",
            config=UnscentedTransformConfig(alpha=0.35),
            rtol=1e-12,
            atol=1e-13,
        )

        bias_error = np.abs(result.final_state[6:] - true_bias)
        self.assertEqual(result.state_estimates.shape, (obs_data.shape[0], 18))
        self.assertLess(np.max(bias_error[0::4]), 4.0)
        self.assertLess(np.max(bias_error[1::4]), 9e-5)
        self.assertLess(np.max(bias_error[2::4]), 5e-6)
        self.assertLess(np.max(bias_error[3::4]), 5e-6)
        self.assertTrue(np.all(np.linalg.eigvalsh(result.final_covariance) > 0.0))
        np.testing.assert_allclose(square_root_result.final_state, result.final_state, rtol=1e-4, atol=1e-5)
        self.assertTrue(np.all(np.linalg.eigvalsh(square_root_result.final_covariance) > 0.0))
        self.assertGreater(result.performance.dynamic_propagation_cache_hits, 0)
        self.assertLess(
            result.performance.unique_dynamic_propagations,
            result.performance.process_function_evaluations,
        )
        self.assertEqual(
            result.performance.measurement_function_evaluations,
            obs_data.shape[0] * (2 * x_guess0.size + 1),
        )
        self.assertGreater(result.performance.measurement_model_cache_hits, 0)
        self.assertLess(
            result.performance.unique_measurement_model_evaluations,
            result.performance.measurement_function_evaluations,
        )

    @slow
    def test_square_root_station_bias_long_arc_with_mismatch_and_heavy_tails(self):
        mu_moon, x_true0, t_pass_s, x_truth, pass_geo, clean_obs, get_earth_pos, get_sun_pos = (
            _synthetic_range_rate_case(duration_s=1200.0)
        )
        true_bias = np.array(
            [
                8.0,
                8.0e-5,
                5.0e-6,
                -4.0e-6,
                -9.0,
                -1.0e-4,
                -6.0e-6,
                5.0e-6,
                6.0,
                1.2e-4,
                4.0e-6,
                -3.0e-6,
            ]
        )
        obs_data = clean_obs.copy()
        _add_rr_station_full_biases(obs_data, true_bias)
        nominal_r = np.diag([1.0, 1e-8, 1e-10, 1e-10])
        noise = generate_measurement_noise(
            obs_data.shape[0],
            nominal_r,
            rng=np.random.default_rng(20260611),
            config=MeasurementNoiseConfig(model="student_t", student_t_dof=5.0),
        )
        obs_data[:, 1:5] += noise
        obs_data[:, 3:5] = wrap_to_pi(obs_data[:, 3:5])

        initial_offset = np.array([25.0, -20.0, 15.0, 0.015, -0.01, 0.008])
        x_guess0 = np.concatenate([x_true0 + initial_offset, np.zeros(true_bias.size)])
        bias_prior_diag = np.tile(np.array([100.0**2, 1e-3**2, 1e-4**2, 1e-4**2]), 3)
        p0 = np.diag([40.0**2] * 3 + [0.03**2] * 3 + bias_prior_diag.tolist())

        result = run_lunar_ukf(
            t_pass_s,
            obs_data,
            x_guess0,
            p0,
            pass_geo,
            mu_moon * (1.0 - 2e-6),
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            bias_mode="station_full",
            process_noise=1e-8,
            process_noise_model="continuous_white_acceleration",
            covariance_form="square_root",
            adaptive_config=UKFAdaptiveConfig(
                component_nis_gate=16.0,
                component_gate_mode="conditional",
            ),
            config=UnscentedTransformConfig(alpha=0.35),
            rtol=1e-11,
            atol=1e-12,
        )

        final_position_error = np.linalg.norm(result.final_state[:3] - x_truth[-1, :3])
        self.assertLess(final_position_error, np.linalg.norm(initial_offset[:3]))
        self.assertGreater(float(np.mean(result.accepted_updates)), 0.8)
        self.assertTrue(np.all(np.isfinite(result.final_state)))
        self.assertGreater(np.min(np.linalg.eigvalsh(result.final_covariance)), 0.0)
        self.assertGreater(result.performance.elapsed_s, 0.0)
        self.assertGreater(result.performance.dynamic_propagation_cache_hits, 0)
        self.assertLess(
            result.performance.unique_dynamic_propagations,
            0.6 * result.performance.process_function_evaluations,
        )

    @slow
    def test_square_root_ukf_34_state_ill_conditioned_station_bias_stress(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_pass_s = np.arange(0.0, 601.0, 60.0)
        get_earth_pos = lambda t: np.tile(
            np.array([384400e3, 0.0, 0.0]),
            (np.size(np.asarray(t)), 1),
        )
        get_sun_pos = lambda t: np.tile(
            np.array([149.6e9, 0.0, 0.0]),
            (np.size(np.asarray(t)), 1),
        )
        x_truth = propagate_augmented_state(
            t_pass_s,
            np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")]),
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            rtol=1e-12,
            atol=1e-13,
        )[:, :6]
        stations = tuple(
            _synthetic_range_rate_station(lat, lon, 100.0 * idx)
            for idx, (lat, lon) in enumerate(
                [
                    (0.0, 0.0),
                    (0.0, 60.0),
                    (0.0, 120.0),
                    (30.0, -60.0),
                    (-30.0, 60.0),
                    (55.0, 150.0),
                    (-55.0, -150.0),
                ]
            )
        )
        pass_geo = PassGeometry(
            t_s=t_pass_s,
            earth_pos_mci_m=np.zeros((t_pass_s.size, 3)),
            earth_vel_mci_mps=np.zeros((t_pass_s.size, 3)),
            x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass_s.size, axis=0),
            stations=stations,
            measurement_type="range_rate",
        )
        obs_data = _build_clean_range_rate_observations(t_pass_s, x_truth, pass_geo)
        x_guess0 = np.concatenate(
            [
                x_true0 + np.array([10.0, -8.0, 6.0, 0.005, -0.004, 0.003]),
                np.zeros(4 * len(stations)),
            ]
        )
        bias_variances = np.tile(np.array([1e6, 1e-4, 1e-10, 1e-14]), len(stations))
        p0 = np.diag([25.0**2] * 3 + [0.02**2] * 3 + bias_variances.tolist())

        result = run_lunar_ukf(
            t_pass_s,
            obs_data,
            x_guess0,
            p0,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            bias_mode="station_full",
            covariance_form="square_root",
            config=UnscentedTransformConfig(alpha=0.25, jitter=1e-14),
            rtol=1e-11,
            atol=1e-12,
        )

        self.assertEqual(result.final_state.size, 34)
        self.assertTrue(np.all(np.isfinite(result.final_state)))
        self.assertGreater(np.min(np.linalg.eigvalsh(result.final_covariance)), 0.0)
        self.assertGreater(float(np.mean(result.accepted_updates)), 0.95)
        self.assertGreater(result.performance.dynamic_propagation_cache_hits, 0)
        self.assertGreater(result.performance.measurement_model_cache_hits, 0)

    @slow
    def test_lunar_ukf_two_way_range_rate_noise_free_synthetic_arc_reduces_error(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_pass_s = np.arange(0.0, 241.0, 60.0)
        get_earth_pos = lambda t: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        get_sun_pos = lambda t: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))

        x_aug0 = np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")])
        x_aug_truth = propagate_augmented_state(
            t_pass_s,
            x_aug0,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            rtol=1e-12,
            atol=1e-13,
        )
        x_truth = x_aug_truth[:, :6]
        stations = (
            _synthetic_range_rate_station(0.0, 0.0, 0.0),
            _synthetic_range_rate_station(0.0, 90.0, 0.0),
            _synthetic_range_rate_station(45.0, -30.0, 500.0),
        )
        pass_geo = PassGeometry(
            t_s=t_pass_s,
            earth_pos_mci_m=np.zeros((t_pass_s.size, 3)),
            earth_vel_mci_mps=np.zeros((t_pass_s.size, 3)),
            x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass_s.size, axis=0),
            stations=stations,
            measurement_type="range_rate",
            range_rate_physics=RangeRatePhysicsConfig(mode="two_way_counted_doppler", count_interval_s=20.0),
        )
        obs_data = _build_clean_two_way_range_rate_observations(
            t_pass_s,
            x_truth,
            pass_geo,
            mu_moon,
            get_earth_pos,
            get_sun_pos,
        )

        x_guess0 = x_true0 + np.array([35.0, -25.0, 15.0, 0.02, -0.015, 0.01])
        p0 = np.diag([80.0**2, 80.0**2, 80.0**2, 0.08**2, 0.08**2, 0.08**2])
        result = run_lunar_ukf(
            t_pass_s,
            obs_data,
            x_guess0,
            p0,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            process_noise=np.diag([0.01**2] * 3 + [1e-5**2] * 3),
            config=UnscentedTransformConfig(alpha=0.35),
            rtol=1e-12,
            atol=1e-13,
        )

        initial_position_error = float(np.linalg.norm(x_guess0[:3] - x_true0[:3]))
        initial_velocity_error = float(np.linalg.norm(x_guess0[3:] - x_true0[3:]))
        final_position_error = float(np.linalg.norm(result.final_state[:3] - x_truth[-1, :3]))
        final_velocity_error = float(np.linalg.norm(result.final_state[3:] - x_truth[-1, 3:]))

        self.assertLess(final_position_error, initial_position_error)
        self.assertLess(final_velocity_error, initial_velocity_error)
        self.assertLess(final_position_error, 10.0)
        self.assertLess(final_velocity_error, 0.02)
        self.assertEqual(result.innovations.shape, (obs_data.shape[0], 4))
        self.assertTrue(np.all(np.linalg.eigvalsh(result.final_covariance) > 0.0))

    def test_two_way_taylor3_local_state_model_matches_ode_observable(self):
        mu_moon, _, _, x_truth, pass_geo, obs_data, get_earth_pos, get_sun_pos = (
            _synthetic_range_rate_case(duration_s=240.0)
        )
        ode_geo = replace(
            pass_geo,
            range_rate_physics=RangeRatePhysicsConfig(
                mode="two_way_counted_doppler",
                count_interval_s=30.0,
                local_state_model="ode",
            ),
        )
        taylor_geo = replace(
            pass_geo,
            range_rate_physics=RangeRatePhysicsConfig(
                mode="two_way_counted_doppler",
                count_interval_s=30.0,
                local_state_model="taylor3",
            ),
        )
        max_error = 0.0
        for row in obs_data[:15]:
            time_idx = int(row[6]) - 1
            ode_value = _range_rate_measurement_from_state(
                x_truth[time_idx],
                row,
                ode_geo,
                mu_moon,
                0.0,
                0.0,
                get_earth_pos,
                get_sun_pos,
                1e-10,
                1e-11,
            )[1]
            taylor_value = _range_rate_measurement_from_state(
                x_truth[time_idx],
                row,
                taylor_geo,
                mu_moon,
                0.0,
                0.0,
                get_earth_pos,
                get_sun_pos,
                1e-10,
                1e-11,
            )[1]
            max_error = max(max_error, abs(taylor_value - ode_value))

        self.assertLess(max_error, 5e-6)

    def test_two_way_taylor3_error_remains_below_noise_for_supported_count_intervals(self):
        mu_moon, _, _, x_truth, pass_geo, obs_data, get_earth_pos, get_sun_pos = (
            _synthetic_range_rate_case(duration_s=240.0)
        )
        for count_interval_s in (10.0, 30.0, 60.0):
            with self.subTest(count_interval_s=count_interval_s):
                ode_geo = replace(
                    pass_geo,
                    range_rate_physics=RangeRatePhysicsConfig(
                        mode="two_way_counted_doppler",
                        count_interval_s=count_interval_s,
                        local_state_model="ode",
                    ),
                )
                taylor_geo = replace(
                    pass_geo,
                    range_rate_physics=RangeRatePhysicsConfig(
                        mode="two_way_counted_doppler",
                        count_interval_s=count_interval_s,
                        local_state_model="taylor3",
                    ),
                )
                max_error = 0.0
                for row in obs_data[:15]:
                    time_idx = int(row[6]) - 1
                    ode_value = _range_rate_measurement_from_state(
                        x_truth[time_idx],
                        row,
                        ode_geo,
                        mu_moon,
                        0.0,
                        0.0,
                        get_earth_pos,
                        get_sun_pos,
                        1e-10,
                        1e-11,
                    )[1]
                    taylor_value = _range_rate_measurement_from_state(
                        x_truth[time_idx],
                        row,
                        taylor_geo,
                        mu_moon,
                        0.0,
                        0.0,
                        get_earth_pos,
                        get_sun_pos,
                        1e-10,
                        1e-11,
                    )[1]
                    max_error = max(max_error, abs(taylor_value - ode_value))

                self.assertLess(max_error, 2e-5)

    @slow
    def test_two_way_long_arc_noise_clock_and_model_mismatch_with_station_biases(self):
        mu_moon, x_true0, t_pass_s, x_truth, geometric_geo, _, get_earth_pos, get_sun_pos = (
            _synthetic_range_rate_case(duration_s=600.0)
        )
        truth_physics = RangeRatePhysicsConfig(
            mode="two_way_counted_doppler",
            count_interval_s=30.0,
            station_clock_offset_s=2e-3,
            station_clock_drift=2e-6,
            transponder_delay_s=4e-6,
        )
        truth_geo = replace(geometric_geo, range_rate_physics=truth_physics)
        clean_obs = _build_clean_two_way_range_rate_observations(
            t_pass_s,
            x_truth,
            truth_geo,
            mu_moon,
            get_earth_pos,
            get_sun_pos,
        )
        noise = generate_measurement_noise(
            clean_obs.shape[0],
            np.diag([1.0, 1e-8, 1e-10, 1e-10]),
            rng=np.random.default_rng(20260612),
            config=MeasurementNoiseConfig(model="gaussian"),
        )
        obs_data = clean_obs.copy()
        obs_data[:, 1:5] += noise
        obs_data[:, 3:5] = wrap_to_pi(obs_data[:, 3:5])
        filter_geo = replace(
            truth_geo,
            range_rate_physics=RangeRatePhysicsConfig(
                mode="two_way_counted_doppler",
                count_interval_s=30.0,
                local_state_model="taylor3",
            ),
        )
        initial_offset = np.array([40.0, -30.0, 20.0, 0.02, -0.015, 0.01])
        x_guess0 = np.concatenate([x_true0 + initial_offset, np.zeros(12)])
        bias_prior_diag = np.tile(np.array([100.0**2, 1e-3**2, 1e-4**2, 1e-4**2]), 3)
        p0 = np.diag([80.0**2] * 3 + [0.05**2] * 3 + bias_prior_diag.tolist())

        result = run_lunar_ukf(
            t_pass_s,
            obs_data,
            x_guess0,
            p0,
            filter_geo,
            mu_moon * (1.0 - 1e-6),
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            bias_mode="station_full",
            covariance_form="square_root",
            process_noise=1e-9,
            process_noise_model="continuous_white_acceleration",
            adaptive_config=UKFAdaptiveConfig(
                component_nis_gate=25.0,
                component_gate_mode="conditional",
            ),
            config=UnscentedTransformConfig(alpha=0.3),
            rtol=1e-10,
            atol=1e-11,
        )

        final_position_error = np.linalg.norm(result.final_state[:3] - x_truth[-1, :3])
        self.assertLess(final_position_error, np.linalg.norm(initial_offset[:3]))
        self.assertGreater(float(np.mean(result.accepted_updates)), 0.8)
        self.assertGreater(result.performance.measurement_model_cache_hits, 0)
        self.assertLess(
            result.performance.unique_measurement_model_evaluations,
            0.5 * result.performance.measurement_function_evaluations,
        )

    @slow
    def test_lunar_ukf_two_way_range_rate_monte_carlo_is_consistent(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_pass_s = np.arange(0.0, 241.0, 60.0)
        get_earth_pos = lambda t: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        get_sun_pos = lambda t: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        x_truth = propagate_augmented_state(
            t_pass_s,
            np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")]),
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            rtol=1e-12,
            atol=1e-13,
        )[:, :6]
        stations = (
            _synthetic_range_rate_station(0.0, 0.0, 0.0),
            _synthetic_range_rate_station(0.0, 90.0, 0.0),
            _synthetic_range_rate_station(45.0, -30.0, 500.0),
        )
        pass_geo = PassGeometry(
            t_s=t_pass_s,
            earth_pos_mci_m=np.zeros((t_pass_s.size, 3)),
            earth_vel_mci_mps=np.zeros((t_pass_s.size, 3)),
            x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass_s.size, axis=0),
            stations=stations,
            measurement_type="range_rate",
            range_rate_physics=RangeRatePhysicsConfig(mode="two_way_counted_doppler", count_interval_s=20.0),
        )
        clean_obs = _build_clean_two_way_range_rate_observations(
            t_pass_s,
            x_truth,
            pass_geo,
            mu_moon,
            get_earth_pos,
            get_sun_pos,
        )
        p0 = np.diag([20.0**2] * 3 + [0.02**2] * 3)
        rng = np.random.default_rng(20260608)
        nis_values = []
        final_nees = []

        for _ in range(6):
            obs_data = clean_obs.copy()
            obs_data[:, 1] += rng.normal(0.0, 1.0, obs_data.shape[0])
            obs_data[:, 2] += rng.normal(0.0, 1e-4, obs_data.shape[0])
            obs_data[:, 3] = wrap_to_pi(obs_data[:, 3] + rng.normal(0.0, 1e-5, obs_data.shape[0]))
            obs_data[:, 4] = wrap_to_pi(obs_data[:, 4] + rng.normal(0.0, 1e-5, obs_data.shape[0]))
            x_guess0 = x_true0 + rng.multivariate_normal(np.zeros(6), p0)

            result = run_lunar_ukf(
                t_pass_s,
                obs_data,
                x_guess0,
                p0,
                pass_geo,
                mu_moon,
                0.0,
                0.0,
                get_earth_pos,
                get_sun_pos,
                process_noise=None,
                config=UnscentedTransformConfig(alpha=0.35),
                rtol=1e-12,
                atol=1e-13,
            )
            nis_values.extend(result.normalized_innovation_squared)
            final_nees.append(
                normalized_estimation_error_squared(
                    result.final_state,
                    x_truth[-1],
                    result.final_covariance,
                )
            )

        self.assertAlmostEqual(float(np.mean(nis_values)), 4.0, delta=0.8)
        self.assertAlmostEqual(float(np.mean(final_nees)), 6.0, delta=3.0)

    def test_stm_linearization_predict_matches_full_propagation(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        get_earth_pos = lambda t: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        get_sun_pos = lambda t: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        rtol, atol = 1e-11, 1e-12
        t0, t1 = 0.0, 240.0
        from lunar_od.dynamics import propagate_state

        p0 = np.diag([50.0**2, 50.0**2, 50.0**2, 0.05**2, 0.05**2, 0.05**2])
        sqrt_state = SquareRootUKFState(x=x_true0.copy(), sqrt_p=np.linalg.cholesky(p0))

        # Use alpha=0.1 so unscented weights are well-conditioned (wm[0] near 0).
        # With the default alpha=1e-3, weights reach ±1e6 and amplify ODE noise ~5 mm.
        cfg = UnscentedTransformConfig(alpha=0.1)

        def process_fn(x: np.ndarray) -> np.ndarray:
            return propagate_state(
                [t0, t1], x[:6], mu_moon, 0.0, 0.0, get_earth_pos, get_sun_pos,
                rtol=rtol, atol=atol,
            )[-1, :]

        def stm_fn(x6: np.ndarray) -> np.ndarray:
            x_aug0 = np.concatenate([x6, np.eye(6).reshape(-1, order="F")])
            return propagate_augmented_state(
                [t0, t1], x_aug0, mu_moon, 0.0, 0.0, get_earth_pos, get_sun_pos,
                rtol=rtol, atol=atol,
            )[-1, :]

        predicted_full, _, _, _ = square_root_ukf_predict(sqrt_state, process_fn, config=cfg)
        predicted_stm, _, _, _ = square_root_ukf_predict(sqrt_state, process_fn, config=cfg, stm_fn=stm_fn)

        pos_diff_m = float(np.linalg.norm(predicted_stm.x[:3] - predicted_full.x[:3]))
        self.assertLess(pos_diff_m, 1e-3)


def _synthetic_position_station(lat_deg: float, lon_deg: float, alt_m: float) -> Station:
    return Station(
        name=f"Synthetic {lat_deg:.1f} {lon_deg:.1f}",
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        alt_m=alt_m,
        color_rgb=(0.0, 0.0, 0.0),
        sigma_range_m=1.0,
        sigma_angle_rad=1e-5,
    )


def _synthetic_range_rate_station(lat_deg: float, lon_deg: float, alt_m: float) -> Station:
    return Station(
        name=f"Synthetic RR {lat_deg:.1f} {lon_deg:.1f}",
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        alt_m=alt_m,
        color_rgb=(0.0, 0.0, 0.0),
        sigma_range_m=1.0,
        sigma_angle_rad=1e-5,
        sigma_range_rate_mps=1e-4,
    )


def _build_clean_position_observations(t_pass_s, x_truth, pass_geo):
    rows = []
    for time_idx, t_s in enumerate(t_pass_s, start=1):
        for station_id in range(1, len(pass_geo.stations) + 1):
            rows.append([t_s, 0.0, 0.0, 0.0, station_id, time_idx])
    obs_data = np.asarray(rows, dtype=float)
    _, h_meas, _ = compute_position_residuals_analytic(x_truth, obs_data, pass_geo)
    obs_data[:, 1:4] = h_meas
    return obs_data


def _build_clean_range_rate_observations(t_pass_s, x_truth, pass_geo):
    rows = []
    for time_idx, t_s in enumerate(t_pass_s, start=1):
        for station_id in range(1, len(pass_geo.stations) + 1):
            rows.append([t_s, 0.0, 0.0, 0.0, 0.0, station_id, time_idx])
    obs_data = np.asarray(rows, dtype=float)
    _, h_meas, _ = compute_range_rate_residuals_analytic(x_truth, obs_data, pass_geo)
    obs_data[:, 1:5] = h_meas
    return obs_data


def _build_clean_two_way_range_rate_observations(t_pass_s, x_truth, pass_geo, mu_moon, get_earth_pos, get_sun_pos):
    rows = []
    for time_idx, t_s in enumerate(t_pass_s, start=1):
        for station_id in range(1, len(pass_geo.stations) + 1):
            rows.append([t_s, 0.0, 0.0, 0.0, 0.0, station_id, time_idx])
    obs_data = np.asarray(rows, dtype=float)
    h_meas = np.zeros((obs_data.shape[0], 4), dtype=float)
    for obs_idx, row in enumerate(obs_data):
        time_idx = int(row[6]) - 1
        h_meas[obs_idx, :] = _range_rate_measurement_from_state(
            x_truth[time_idx],
            row,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            1e-12,
            1e-13,
        )
    obs_data[:, 1:5] = h_meas
    return obs_data


def _add_rr_station_full_biases(obs_data, bias_vec):
    for obs_idx in range(obs_data.shape[0]):
        station_id = int(obs_data[obs_idx, 5]) - 1
        col0 = station_id * 4
        obs_data[obs_idx, 1:5] += bias_vec[col0 : col0 + 4]


def _synthetic_range_rate_case(duration_s):
    mu_moon = 4902.800066e9
    r0norm = 1737.4e3 + 100e3
    x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
    t_pass_s = np.arange(0.0, duration_s + 1.0, 60.0)
    get_earth_pos = lambda t: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
    get_sun_pos = lambda t: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
    x_truth = propagate_augmented_state(
        t_pass_s,
        np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")]),
        mu_moon,
        0.0,
        0.0,
        get_earth_pos,
        get_sun_pos,
        rtol=1e-12,
        atol=1e-13,
    )[:, :6]
    stations = (
        _synthetic_range_rate_station(0.0, 0.0, 0.0),
        _synthetic_range_rate_station(0.0, 90.0, 0.0),
        _synthetic_range_rate_station(45.0, -30.0, 500.0),
    )
    pass_geo = PassGeometry(
        t_s=t_pass_s,
        earth_pos_mci_m=np.zeros((t_pass_s.size, 3)),
        earth_vel_mci_mps=np.zeros((t_pass_s.size, 3)),
        x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass_s.size, axis=0),
        stations=stations,
        measurement_type="range_rate",
    )
    obs_data = _build_clean_range_rate_observations(t_pass_s, x_truth, pass_geo)
    return mu_moon, x_true0, t_pass_s, x_truth, pass_geo, obs_data, get_earth_pos, get_sun_pos


def _add_range_rate_noise(clean_obs, rng):
    obs_data = clean_obs.copy()
    obs_data[:, 1] += rng.normal(0.0, 1.0, obs_data.shape[0])
    obs_data[:, 2] += rng.normal(0.0, 1e-4, obs_data.shape[0])
    obs_data[:, 3] = wrap_to_pi(obs_data[:, 3] + rng.normal(0.0, 1e-5, obs_data.shape[0]))
    obs_data[:, 4] = wrap_to_pi(obs_data[:, 4] + rng.normal(0.0, 1e-5, obs_data.shape[0]))
    return obs_data


if __name__ == "__main__":
    unittest.main()
