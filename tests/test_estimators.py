import math
import unittest

import numpy as np
import lunar_od.estimators as estimator_helpers

from lunar_od import (
    MoonCenteredEphemeris,
    PassGeometry,
    RangeRatePhysicsConfig,
    Station,
    compute_position_residuals_analytic,
    compute_range_rate_residuals,
    compute_range_rate_residuals_analytic,
    estimate_position_bls_lm,
    estimate_range_rate_bls_lm,
    estimate_range_rate_srif,
    estimate_position_srif,
    load_spice_kernels,
    ode_fun_v3,
    propagate_augmented_state,
    range_rate_stations,
)
from pathlib import Path
import json


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


class EstimatorTests(unittest.TestCase):
    def test_position_srif_noise_free_recovery_synthetic_arc(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_pass_s = np.arange(0.0, 901.0, 60.0)

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
            _synthetic_station(0.0, 0.0, 0.0),
            _synthetic_station(0.0, 90.0, 0.0),
            _synthetic_station(45.0, -30.0, 500.0),
            _synthetic_station(-35.0, 150.0, 600.0),
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

        x_guess = x_true0 + np.array([100.0, -100.0, 50.0, 0.05, -0.03, 0.02])
        x_est, stop_reason, stats = estimate_position_srif(
            t_pass_s,
            obs_data,
            x_guess,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            max_iter=40,
            rtol=1e-12,
            atol=1e-13,
        )

        pos_err = float(np.linalg.norm(x_est[:3] - x_true0[:3]))
        vel_err = float(np.linalg.norm(x_est[3:] - x_true0[3:]))

        x_aug_est0 = np.concatenate([x_est, np.eye(6).reshape(-1, order="F")])
        x_aug_est = propagate_augmented_state(
            t_pass_s,
            x_aug_est0,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            rtol=1e-12,
            atol=1e-13,
        )
        residuals = compute_position_residuals_analytic(x_aug_est[:, :6], obs_data, pass_geo)[0]
        residual_rms = float(np.sqrt(np.mean(residuals**2)))

        self.assertIn(stop_reason, {"Converged", "J-Stab", "MaxIter"})
        self.assertLess(pos_err, 10.0)
        self.assertLess(vel_err, 0.05)
        self.assertLess(residual_rms, 1e-3)
        self.assertLessEqual(stats.iterations, 40)

    def test_position_srif_global_bias_recovery_synthetic_arc(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        true_bias = np.array([25.0, 2.0e-5, -1.5e-5])
        t_pass_s = np.arange(0.0, 1201.0, 60.0)

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
            _synthetic_station(0.0, 0.0, 0.0),
            _synthetic_station(0.0, 90.0, 0.0),
            _synthetic_station(45.0, -30.0, 500.0),
            _synthetic_station(-35.0, 150.0, 600.0),
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
        obs_data[:, 1:4] += true_bias.reshape(1, 3)

        x_guess = np.concatenate(
            [
                x_true0 + np.array([80.0, -70.0, 40.0, 0.04, -0.02, 0.015]),
                np.zeros(3),
            ]
        )
        x_est, stop_reason, stats = estimate_position_srif(
            t_pass_s,
            obs_data,
            x_guess,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            max_iter=50,
            rtol=1e-12,
            atol=1e-13,
        )

        pos_err = float(np.linalg.norm(x_est[:3] - x_true0[:3]))
        vel_err = float(np.linalg.norm(x_est[3:6] - x_true0[3:6]))
        bias_err = np.abs(x_est[6:9] - true_bias)

        x_aug_est0 = np.concatenate([x_est[:6], np.eye(6).reshape(-1, order="F")])
        x_aug_est = propagate_augmented_state(
            t_pass_s,
            x_aug_est0,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            rtol=1e-12,
            atol=1e-13,
        )
        residuals, h_est, _ = compute_position_residuals_analytic(x_aug_est[:, :6], obs_data, pass_geo)
        h_est_with_bias = h_est + x_est[6:9].reshape(1, 3)
        residuals_with_bias = obs_data[:, 1:4] - h_est_with_bias
        residuals_with_bias[:, 1] = np.arctan2(np.sin(residuals_with_bias[:, 1]), np.cos(residuals_with_bias[:, 1]))
        residuals_with_bias[:, 2] = np.arctan2(np.sin(residuals_with_bias[:, 2]), np.cos(residuals_with_bias[:, 2]))
        residual_rms = float(np.sqrt(np.mean(residuals_with_bias**2)))

        self.assertIn(stop_reason, {"Converged", "J-Stab", "MaxIter"})
        self.assertLess(pos_err, 20.0)
        self.assertLess(vel_err, 0.08)
        self.assertLess(bias_err[0], 2.0)
        self.assertLess(bias_err[1], 5e-6)
        self.assertLess(bias_err[2], 5e-6)
        self.assertLess(residual_rms, 1e-3)
        self.assertLessEqual(stats.iterations, 50)

    def test_position_srif_station_angle_bias_recovery_synthetic_arc(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_pass_s = np.arange(0.0, 1201.0, 60.0)

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
            _synthetic_station(0.0, 0.0, 0.0),
            _synthetic_station(0.0, 90.0, 0.0),
            _synthetic_station(45.0, -30.0, 500.0),
            _synthetic_station(-35.0, 150.0, 600.0),
        )
        true_bias = np.array(
            [
                1.0e-5,
                -1.5e-5,
                -0.8e-5,
                1.2e-5,
                1.7e-5,
                0.6e-5,
                -1.1e-5,
                -0.9e-5,
            ]
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
        _add_station_angle_biases(obs_data, true_bias)

        x_guess = np.concatenate(
            [
                x_true0 + np.array([60.0, -50.0, 30.0, 0.03, -0.02, 0.01]),
                np.zeros(true_bias.size),
            ]
        )
        x_est, stop_reason, stats = estimate_position_srif(
            t_pass_s,
            obs_data,
            x_guess,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            max_iter=50,
            rtol=1e-12,
            atol=1e-13,
            bias_mode="station_angles",
        )

        pos_err = float(np.linalg.norm(x_est[:3] - x_true0[:3]))
        vel_err = float(np.linalg.norm(x_est[3:6] - x_true0[3:6]))
        bias_err = np.max(np.abs(x_est[6:] - true_bias))
        residual_rms = _position_residual_rms_with_station_angle_bias(
            t_pass_s,
            obs_data,
            x_est,
            pass_geo,
            mu_moon,
            get_earth_pos,
            get_sun_pos,
        )

        self.assertIn(stop_reason, {"Converged", "J-Stab", "MaxIter"})
        self.assertLess(pos_err, 20.0)
        self.assertLess(vel_err, 0.08)
        self.assertLess(bias_err, 8e-6)
        self.assertLess(residual_rms, 1e-3)
        self.assertLessEqual(stats.iterations, 50)

    def test_position_srif_station_full_bias_recovery_synthetic_arc(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_pass_s = np.arange(0.0, 1201.0, 60.0)

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
            _synthetic_station(0.0, 0.0, 0.0),
            _synthetic_station(0.0, 90.0, 0.0),
            _synthetic_station(45.0, -30.0, 500.0),
            _synthetic_station(-35.0, 150.0, 600.0),
        )
        true_bias = np.array(
            [
                10.0,
                1.0e-5,
                -1.5e-5,
                -8.0,
                -0.8e-5,
                1.2e-5,
                14.0,
                1.7e-5,
                0.6e-5,
                -11.0,
                -1.1e-5,
                -0.9e-5,
            ]
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
        _add_station_full_biases(obs_data, true_bias)

        x_guess = np.concatenate(
            [
                x_true0 + np.array([60.0, -50.0, 30.0, 0.03, -0.02, 0.01]),
                np.zeros(true_bias.size),
            ]
        )
        x_est, stop_reason, stats = estimate_position_srif(
            t_pass_s,
            obs_data,
            x_guess,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            max_iter=50,
            rtol=1e-12,
            atol=1e-13,
            bias_mode="station_full",
        )

        pos_err = float(np.linalg.norm(x_est[:3] - x_true0[:3]))
        vel_err = float(np.linalg.norm(x_est[3:6] - x_true0[3:6]))
        bias_err = np.abs(x_est[6:] - true_bias)
        residual_rms = _position_residual_rms_with_station_full_bias(
            t_pass_s,
            obs_data,
            x_est,
            pass_geo,
            mu_moon,
            get_earth_pos,
            get_sun_pos,
        )

        self.assertIn(stop_reason, {"Converged", "J-Stab", "MaxIter"})
        self.assertLess(pos_err, 25.0)
        self.assertLess(vel_err, 0.1)
        self.assertLess(np.max(bias_err[0::3]), 4.0)
        self.assertLess(np.max(bias_err[1::3]), 8e-6)
        self.assertLess(np.max(bias_err[2::3]), 8e-6)
        self.assertLess(residual_rms, 1e-3)
        self.assertLessEqual(stats.iterations, 50)

    def test_range_rate_srif_noise_free_recovery_synthetic_arc(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_pass_s = np.arange(0.0, 901.0, 60.0)

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
            _synthetic_station(0.0, 0.0, 0.0, include_rr=True),
            _synthetic_station(0.0, 90.0, 0.0, include_rr=True),
            _synthetic_station(45.0, -30.0, 500.0, include_rr=True),
            _synthetic_station(-35.0, 150.0, 600.0, include_rr=True),
        )
        pass_geo = PassGeometry(
            t_s=t_pass_s,
            earth_pos_mci_m=np.zeros((t_pass_s.size, 3)),
            earth_vel_mci_mps=np.zeros((t_pass_s.size, 3)),
            x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass_s.size, axis=0),
            stations=stations,
            measurement_type="range_rate",
        )
        obs_data = _build_clean_rr_observations(t_pass_s, x_truth, pass_geo)

        x_guess = x_true0 + np.array([80.0, -70.0, 40.0, 0.04, -0.025, 0.015])
        x_est, stop_reason, stats = estimate_range_rate_srif(
            t_pass_s,
            obs_data,
            x_guess,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            max_iter=40,
            rtol=1e-12,
            atol=1e-13,
        )

        pos_err = float(np.linalg.norm(x_est[:3] - x_true0[:3]))
        vel_err = float(np.linalg.norm(x_est[3:] - x_true0[3:]))

        x_aug_est0 = np.concatenate([x_est, np.eye(6).reshape(-1, order="F")])
        x_aug_est = propagate_augmented_state(
            t_pass_s,
            x_aug_est0,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            rtol=1e-12,
            atol=1e-13,
        )
        residuals = compute_range_rate_residuals_analytic(x_aug_est[:, :6], obs_data, pass_geo)[0]
        residual_rms = float(np.sqrt(np.mean(residuals**2)))

        self.assertIn(stop_reason, {"Converged", "J-Stab", "MaxIter"})
        self.assertLess(pos_err, 10.0)
        self.assertLess(vel_err, 0.03)
        self.assertLess(residual_rms, 1e-3)
        self.assertLessEqual(stats.iterations, 40)

    def test_position_bls_lm_noise_free_recovery_synthetic_arc(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_pass_s = np.arange(0.0, 901.0, 60.0)

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
            _synthetic_station(0.0, 0.0, 0.0),
            _synthetic_station(0.0, 90.0, 0.0),
            _synthetic_station(45.0, -30.0, 500.0),
            _synthetic_station(-35.0, 150.0, 600.0),
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
        x_guess = x_true0 + np.array([120.0, -90.0, 55.0, 0.05, -0.025, 0.018])
        x_est, stop_reason, stats = estimate_position_bls_lm(
            t_pass_s,
            obs_data,
            x_guess,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            max_iter=50,
            rtol=1e-12,
            atol=1e-13,
        )

        self.assertIn(stop_reason, {"Converged", "J-Stab", "MaxIter"})
        self.assertLess(float(np.linalg.norm(x_est[:3] - x_true0[:3])), 15.0)
        self.assertLess(float(np.linalg.norm(x_est[3:] - x_true0[3:])), 0.06)
        self.assertLessEqual(stats.iterations, 50)

    def test_range_rate_bls_lm_two_way_uses_analytic_jacobian(self):
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
            _synthetic_station(0.0, 0.0, 0.0, include_rr=True),
            _synthetic_station(0.0, 90.0, 0.0, include_rr=True),
            _synthetic_station(45.0, -30.0, 500.0, include_rr=True),
            _synthetic_station(-35.0, 150.0, 600.0, include_rr=True),
        )
        pass_geo = PassGeometry(
            t_s=t_pass_s,
            earth_pos_mci_m=np.zeros((t_pass_s.size, 3)),
            earth_vel_mci_mps=np.zeros((t_pass_s.size, 3)),
            x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass_s.size, axis=0),
            stations=stations,
            measurement_type="range_rate",
            range_rate_physics=RangeRatePhysicsConfig(
                mode="two_way_counted_doppler",
                count_interval_s=20.0,
            ),
        )
        obs_data = _build_clean_rr_observations_from_model(t_pass_s, x_truth, pass_geo)

        x_guess = x_true0 + np.array([25.0, -20.0, 12.0, 0.01, -0.008, 0.004])
        initial_rms = _rr_residual_rms_from_model(
            t_pass_s,
            obs_data,
            x_guess,
            pass_geo,
            mu_moon,
            get_earth_pos,
            get_sun_pos,
        )
        x_est, stop_reason, stats = estimate_range_rate_bls_lm(
            t_pass_s,
            obs_data,
            x_guess,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            max_iter=8,
            rtol=1e-12,
            atol=1e-13,
        )
        final_rms = _rr_residual_rms_from_model(
            t_pass_s,
            obs_data,
            x_est,
            pass_geo,
            mu_moon,
            get_earth_pos,
            get_sun_pos,
        )

        self.assertIn(stop_reason, {"Converged", "J-Stab", "MaxIter"})
        self.assertTrue(np.isfinite(stats.condition_number))
        self.assertLess(final_rms, initial_rms)

    def test_range_rate_srif_two_way_uses_analytic_jacobian(self):
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
        stations = (
            _synthetic_station(0.0, 0.0, 0.0, include_rr=True),
            _synthetic_station(0.0, 90.0, 0.0, include_rr=True),
            _synthetic_station(45.0, -30.0, 500.0, include_rr=True),
            _synthetic_station(-35.0, 150.0, 600.0, include_rr=True),
        )
        pass_geo = PassGeometry(
            t_s=t_pass_s,
            earth_pos_mci_m=np.zeros((t_pass_s.size, 3)),
            earth_vel_mci_mps=np.zeros((t_pass_s.size, 3)),
            x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass_s.size, axis=0),
            stations=stations,
            measurement_type="range_rate",
            range_rate_physics=RangeRatePhysicsConfig(
                mode="two_way_counted_doppler",
                count_interval_s=20.0,
            ),
        )
        obs_data = _build_clean_rr_observations_from_model(t_pass_s, x_aug_truth[:, :6], pass_geo)

        x_guess = x_true0 + np.array([25.0, -20.0, 12.0, 0.01, -0.008, 0.004])
        initial_rms = _rr_residual_rms_from_model(
            t_pass_s,
            obs_data,
            x_guess,
            pass_geo,
            mu_moon,
            get_earth_pos,
            get_sun_pos,
        )
        x_est, stop_reason, stats = estimate_range_rate_srif(
            t_pass_s,
            obs_data,
            x_guess,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            max_iter=8,
            rtol=1e-12,
            atol=1e-13,
        )
        final_rms = _rr_residual_rms_from_model(
            t_pass_s,
            obs_data,
            x_est,
            pass_geo,
            mu_moon,
            get_earth_pos,
            get_sun_pos,
        )

        self.assertIn(stop_reason, {"Converged", "J-Stab", "MaxIter"})
        self.assertTrue(np.isfinite(stats.condition_number))
        self.assertLess(final_rms, initial_rms)

    def test_two_way_analytic_initial_jacobian_matches_numerical_reference(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_pass_s = np.arange(0.0, 241.0, 60.0)

        get_earth_pos = lambda t: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        get_sun_pos = lambda t: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        x_aug0 = np.concatenate([x0, np.eye(6).reshape(-1, order="F")])
        x_aug_hist = propagate_augmented_state(
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
        stations = (
            _synthetic_station(0.0, 0.0, 0.0, include_rr=True),
            _synthetic_station(0.0, 90.0, 0.0, include_rr=True),
        )
        pass_geo = PassGeometry(
            t_s=t_pass_s,
            earth_pos_mci_m=np.zeros((t_pass_s.size, 3)),
            earth_vel_mci_mps=np.zeros((t_pass_s.size, 3)),
            x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass_s.size, axis=0),
            stations=stations,
            measurement_type="range_rate",
            range_rate_physics=RangeRatePhysicsConfig(
                mode="two_way_counted_doppler",
                count_interval_s=20.0,
            ),
        )
        obs_data = _build_clean_rr_observations_from_model(t_pass_s, x_aug_hist[:, :6], pass_geo)

        _, h_analytic = estimator_helpers._range_rate_nominal_and_initial_jacobian(
            t_pass_s,
            obs_data,
            x0,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            x_aug_hist,
            1e-12,
            1e-13,
        )
        h_numerical = estimator_helpers._range_rate_numerical_initial_jacobian(
            t_pass_s,
            obs_data,
            x0,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            1e-12,
            1e-13,
        )

        np.testing.assert_allclose(h_analytic, h_numerical, rtol=5e-3, atol=3e-4)

    def test_range_rate_bls_lm_rejects_scalar_outlier(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_pass_s = np.arange(0.0, 901.0, 60.0)

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
            _synthetic_station(0.0, 0.0, 0.0, include_rr=True),
            _synthetic_station(0.0, 90.0, 0.0, include_rr=True),
            _synthetic_station(45.0, -30.0, 500.0, include_rr=True),
            _synthetic_station(-35.0, 150.0, 600.0, include_rr=True),
        )
        pass_geo = PassGeometry(
            t_s=t_pass_s,
            earth_pos_mci_m=np.zeros((t_pass_s.size, 3)),
            earth_vel_mci_mps=np.zeros((t_pass_s.size, 3)),
            x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass_s.size, axis=0),
            stations=stations,
            measurement_type="range_rate",
        )
        obs_data = _build_clean_rr_observations(t_pass_s, x_truth, pass_geo)
        obs_data[[5, 17, 41], 2] += np.array([0.20, -0.18, 0.16])
        x_guess = x_true0.copy()

        x_plain, _, stats_plain = estimate_range_rate_bls_lm(
            t_pass_s,
            obs_data,
            x_guess,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            max_iter=50,
            rtol=1e-12,
            atol=1e-13,
        )
        x_robust, _, stats_robust = estimate_range_rate_bls_lm(
            t_pass_s,
            obs_data,
            x_guess,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            max_iter=50,
            rtol=1e-12,
            atol=1e-13,
            robust_outlier_rejection=True,
        )

        err_plain = float(np.linalg.norm(x_plain[:3] - x_true0[:3]))
        err_robust = float(np.linalg.norm(x_robust[:3] - x_true0[:3]))
        self.assertGreater(stats_robust.rejected_components, 0)
        self.assertLess(stats_robust.active_weight_fraction, 1.0)
        self.assertLess(err_robust, err_plain)

    def test_range_rate_srif_global_bias_recovery_synthetic_arc(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        true_bias = np.array([18.0, 2.0e-4, 1.5e-5, -1.0e-5])
        t_pass_s = np.arange(0.0, 1201.0, 60.0)

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
            _synthetic_station(0.0, 0.0, 0.0, include_rr=True),
            _synthetic_station(0.0, 90.0, 0.0, include_rr=True),
            _synthetic_station(45.0, -30.0, 500.0, include_rr=True),
            _synthetic_station(-35.0, 150.0, 600.0, include_rr=True),
        )
        pass_geo = PassGeometry(
            t_s=t_pass_s,
            earth_pos_mci_m=np.zeros((t_pass_s.size, 3)),
            earth_vel_mci_mps=np.zeros((t_pass_s.size, 3)),
            x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass_s.size, axis=0),
            stations=stations,
            measurement_type="range_rate",
        )
        obs_data = _build_clean_rr_observations(t_pass_s, x_truth, pass_geo)
        obs_data[:, 1:5] += true_bias.reshape(1, 4)

        x_guess = np.concatenate(
            [
                x_true0 + np.array([70.0, -55.0, 35.0, 0.035, -0.02, 0.012]),
                np.zeros(4),
            ]
        )
        x_est, stop_reason, stats = estimate_range_rate_srif(
            t_pass_s,
            obs_data,
            x_guess,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            max_iter=50,
            rtol=1e-12,
            atol=1e-13,
        )

        pos_err = float(np.linalg.norm(x_est[:3] - x_true0[:3]))
        vel_err = float(np.linalg.norm(x_est[3:6] - x_true0[3:6]))
        bias_err = np.abs(x_est[6:10] - true_bias)
        residual_rms = _rr_residual_rms_with_global_bias(
            t_pass_s,
            obs_data,
            x_est,
            pass_geo,
            mu_moon,
            get_earth_pos,
            get_sun_pos,
        )

        self.assertIn(stop_reason, {"Converged", "J-Stab", "MaxIter"})
        self.assertLess(pos_err, 20.0)
        self.assertLess(vel_err, 0.08)
        self.assertLess(bias_err[0], 3.0)
        self.assertLess(bias_err[1], 8e-5)
        self.assertLess(bias_err[2], 6e-6)
        self.assertLess(bias_err[3], 6e-6)
        self.assertLess(residual_rms, 1e-3)
        self.assertLessEqual(stats.iterations, 50)

    def test_range_rate_srif_station_angle_bias_recovery_synthetic_arc(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_pass_s = np.arange(0.0, 1201.0, 60.0)

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
            _synthetic_station(0.0, 0.0, 0.0, include_rr=True),
            _synthetic_station(0.0, 90.0, 0.0, include_rr=True),
            _synthetic_station(45.0, -30.0, 500.0, include_rr=True),
            _synthetic_station(-35.0, 150.0, 600.0, include_rr=True),
        )
        true_bias = np.array(
            [
                1.0e-5,
                -1.4e-5,
                -0.7e-5,
                1.1e-5,
                1.5e-5,
                0.5e-5,
                -1.0e-5,
                -0.8e-5,
            ]
        )
        pass_geo = PassGeometry(
            t_s=t_pass_s,
            earth_pos_mci_m=np.zeros((t_pass_s.size, 3)),
            earth_vel_mci_mps=np.zeros((t_pass_s.size, 3)),
            x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass_s.size, axis=0),
            stations=stations,
            measurement_type="range_rate",
        )
        obs_data = _build_clean_rr_observations(t_pass_s, x_truth, pass_geo)
        _add_rr_station_angle_biases(obs_data, true_bias)

        x_guess = np.concatenate(
            [
                x_true0 + np.array([55.0, -45.0, 28.0, 0.025, -0.018, 0.01]),
                np.zeros(true_bias.size),
            ]
        )
        x_est, stop_reason, stats = estimate_range_rate_srif(
            t_pass_s,
            obs_data,
            x_guess,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            max_iter=50,
            rtol=1e-12,
            atol=1e-13,
            bias_mode="station_angles",
        )

        pos_err = float(np.linalg.norm(x_est[:3] - x_true0[:3]))
        vel_err = float(np.linalg.norm(x_est[3:6] - x_true0[3:6]))
        bias_err = np.max(np.abs(x_est[6:] - true_bias))
        residual_rms = _rr_residual_rms_with_station_angle_bias(
            t_pass_s,
            obs_data,
            x_est,
            pass_geo,
            mu_moon,
            get_earth_pos,
            get_sun_pos,
        )

        self.assertIn(stop_reason, {"Converged", "J-Stab", "MaxIter"})
        self.assertLess(pos_err, 25.0)
        self.assertLess(vel_err, 0.08)
        self.assertLess(bias_err, 8e-6)
        self.assertLess(residual_rms, 1e-3)
        self.assertLessEqual(stats.iterations, 50)

    def test_range_rate_srif_station_full_bias_recovery_synthetic_arc(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_pass_s = np.arange(0.0, 1201.0, 60.0)

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
            _synthetic_station(0.0, 0.0, 0.0, include_rr=True),
            _synthetic_station(0.0, 90.0, 0.0, include_rr=True),
            _synthetic_station(45.0, -30.0, 500.0, include_rr=True),
            _synthetic_station(-35.0, 150.0, 600.0, include_rr=True),
        )
        true_bias = np.array(
            [
                9.0,
                1.5e-4,
                1.0e-5,
                -1.4e-5,
                -7.0,
                -1.2e-4,
                -0.7e-5,
                1.1e-5,
                12.0,
                0.8e-4,
                1.5e-5,
                0.5e-5,
                -10.0,
                -0.6e-4,
                -1.0e-5,
                -0.8e-5,
            ]
        )
        pass_geo = PassGeometry(
            t_s=t_pass_s,
            earth_pos_mci_m=np.zeros((t_pass_s.size, 3)),
            earth_vel_mci_mps=np.zeros((t_pass_s.size, 3)),
            x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass_s.size, axis=0),
            stations=stations,
            measurement_type="range_rate",
        )
        obs_data = _build_clean_rr_observations(t_pass_s, x_truth, pass_geo)
        _add_rr_station_full_biases(obs_data, true_bias)

        x_guess = np.concatenate(
            [
                x_true0 + np.array([45.0, -35.0, 22.0, 0.02, -0.014, 0.008]),
                np.zeros(true_bias.size),
            ]
        )
        x_est, stop_reason, stats = estimate_range_rate_srif(
            t_pass_s,
            obs_data,
            x_guess,
            pass_geo,
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            max_iter=60,
            rtol=1e-12,
            atol=1e-13,
            bias_mode="station_full",
        )

        pos_err = float(np.linalg.norm(x_est[:3] - x_true0[:3]))
        vel_err = float(np.linalg.norm(x_est[3:6] - x_true0[3:6]))
        bias_err = np.abs(x_est[6:] - true_bias)
        residual_rms = _rr_residual_rms_with_station_full_bias(
            t_pass_s,
            obs_data,
            x_est,
            pass_geo,
            mu_moon,
            get_earth_pos,
            get_sun_pos,
        )

        self.assertIn(stop_reason, {"Converged", "J-Stab", "MaxIter"})
        self.assertLess(pos_err, 30.0)
        self.assertLess(vel_err, 0.1)
        self.assertLess(np.max(bias_err[0::4]), 4.0)
        self.assertLess(np.max(bias_err[1::4]), 8e-5)
        self.assertLess(np.max(bias_err[2::4]), 8e-6)
        self.assertLess(np.max(bias_err[3::4]), 8e-6)
        self.assertLess(residual_rms, 1e-3)
        self.assertLessEqual(stats.iterations, 60)

    def test_position_srif_reduces_cost_on_spice_fixture_arc(self):
        fixture_path = FIXTURES_DIR / "spice_snapshots.json"
        if not fixture_path.is_file():
            self.skipTest("MATLAB spice_snapshots.json fixture has not been exported yet.")

        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        truth = fixture["truth_propagation"]
        meas = fixture["position_measurements"]
        constants = fixture["constants"]
        initial = fixture["initial_state"]

        ephemeris = MoonCenteredEphemeris(
            t_ephem_s=truth["t_ephem_s"],
            earth_pos_m=truth["earth_pos_grid_m"],
            sun_pos_m=truth["sun_pos_grid_m"],
            earth_vel_mps=truth["earth_vel_grid_mps"],
        )
        stations_by_name = {station.name: station for station in range_rate_stations()}
        stations = [stations_by_name[name] for name in meas["station_names"]]

        import spiceypy as spice
        from lunar_od import generate_position_measurements

        load_spice_kernels()
        try:
            _, pass_geo, clean_obs = generate_position_measurements(
                meas["t_pass_s"],
                truth["state_history_mci_m_mps"],
                stations,
                meas["vis_mask_raw"],
                ephemeris.earth_position,
                ephemeris.earth_velocity,
                fixture["et"],
                noise=False,
                arc_id=7,
            )
        finally:
            spice.kclear()

        x_true0 = np.asarray(initial["state_mci_j2000_m_mps"], dtype=float)
        x_guess = x_true0 + np.array([50.0, -40.0, 25.0, 0.02, -0.015, 0.01])

        mu_moon = float(initial["mu_moon_m3_s2"])
        mu_earth = float(np.asarray(constants["mu_earth_km3_s2"]).reshape(-1)[0] * 1e9)
        mu_sun = float(np.asarray(constants["mu_sun_km3_s2"]).reshape(-1)[0] * 1e9)
        t_pass_s = np.asarray(meas["t_pass_s"], dtype=float)

        initial_rms = _position_residual_rms(
            t_pass_s,
            clean_obs,
            x_guess,
            pass_geo,
            mu_moon,
            mu_earth,
            mu_sun,
            ephemeris.earth_position,
            ephemeris.sun_position,
        )

        x_est, stop_reason, stats = estimate_position_srif(
            t_pass_s,
            clean_obs,
            x_guess,
            pass_geo,
            mu_moon,
            mu_earth,
            mu_sun,
            ephemeris.earth_position,
            ephemeris.sun_position,
            max_iter=20,
            rtol=1e-12,
            atol=1e-13,
        )

        final_rms = _position_residual_rms(
            t_pass_s,
            clean_obs,
            x_est,
            pass_geo,
            mu_moon,
            mu_earth,
            mu_sun,
            ephemeris.earth_position,
            ephemeris.sun_position,
        )

        self.assertIn(stop_reason, {"Converged", "J-Stab", "MaxIter"})
        self.assertLess(final_rms, initial_rms)
        self.assertLess(final_rms, 1e-3)
        self.assertLess(float(np.linalg.norm(x_est[:3] - x_true0[:3])), 20.0)
        self.assertLessEqual(stats.iterations, 20)

    def test_range_rate_srif_reduces_cost_on_spice_fixture_arc(self):
        fixture_path = FIXTURES_DIR / "spice_snapshots.json"
        if not fixture_path.is_file():
            self.skipTest("MATLAB spice_snapshots.json fixture has not been exported yet.")

        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        truth = fixture["truth_propagation"]
        meas = fixture["range_rate_measurements"]
        constants = fixture["constants"]
        initial = fixture["initial_state"]

        ephemeris = MoonCenteredEphemeris(
            t_ephem_s=truth["t_ephem_s"],
            earth_pos_m=truth["earth_pos_grid_m"],
            sun_pos_m=truth["sun_pos_grid_m"],
            earth_vel_mps=truth["earth_vel_grid_mps"],
        )
        stations_by_name = {station.name: station for station in range_rate_stations()}
        stations = [stations_by_name[name] for name in meas["station_names"]]

        import spiceypy as spice
        from lunar_od import generate_range_rate_measurements

        load_spice_kernels()
        try:
            obs_data, pass_geo = generate_range_rate_measurements(
                meas["t_pass_s"],
                truth["state_history_mci_m_mps"],
                stations,
                meas["vis_mask_raw"],
                ephemeris.earth_position,
                ephemeris.earth_velocity,
                fixture["et"],
                noise=False,
                arc_id=7,
            )
        finally:
            spice.kclear()

        x_true0 = np.asarray(initial["state_mci_j2000_m_mps"], dtype=float)
        x_guess = x_true0 + np.array([35.0, -30.0, 20.0, 0.01, -0.008, 0.005])

        mu_moon = float(initial["mu_moon_m3_s2"])
        mu_earth = float(np.asarray(constants["mu_earth_km3_s2"]).reshape(-1)[0] * 1e9)
        mu_sun = float(np.asarray(constants["mu_sun_km3_s2"]).reshape(-1)[0] * 1e9)
        t_pass_s = np.asarray(meas["t_pass_s"], dtype=float)

        initial_rms = _rr_residual_rms(
            t_pass_s,
            obs_data,
            x_guess,
            pass_geo,
            mu_moon,
            mu_earth,
            mu_sun,
            ephemeris.earth_position,
            ephemeris.sun_position,
        )

        x_est, stop_reason, stats = estimate_range_rate_srif(
            t_pass_s,
            obs_data,
            x_guess,
            pass_geo,
            mu_moon,
            mu_earth,
            mu_sun,
            ephemeris.earth_position,
            ephemeris.sun_position,
            max_iter=20,
            rtol=1e-12,
            atol=1e-13,
        )

        final_rms = _rr_residual_rms(
            t_pass_s,
            obs_data,
            x_est,
            pass_geo,
            mu_moon,
            mu_earth,
            mu_sun,
            ephemeris.earth_position,
            ephemeris.sun_position,
        )

        self.assertIn(stop_reason, {"Converged", "J-Stab", "MaxIter"})
        self.assertLess(final_rms, initial_rms)
        self.assertLess(final_rms, 1e-3)
        self.assertLess(float(np.linalg.norm(x_est[:3] - x_true0[:3])), 20.0)
        self.assertLess(float(np.linalg.norm(x_est[3:] - x_true0[3:])), 0.05)
        self.assertLessEqual(stats.iterations, 20)


def _synthetic_station(lat_deg: float, lon_deg: float, alt_m: float, include_rr: bool = False) -> Station:
    return Station(
        name=f"Synthetic {lat_deg:.1f} {lon_deg:.1f}",
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        alt_m=alt_m,
        color_rgb=(0.0, 0.0, 0.0),
        sigma_range_m=1.0,
        sigma_angle_rad=1e-5,
        sigma_range_rate_mps=1e-4 if include_rr else None,
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


def _build_clean_rr_observations(t_pass_s, x_truth, pass_geo):
    rows = []
    for time_idx, t_s in enumerate(t_pass_s, start=1):
        for station_id in range(1, len(pass_geo.stations) + 1):
            rows.append([t_s, 0.0, 0.0, 0.0, 0.0, station_id, time_idx])
    obs_data = np.asarray(rows, dtype=float)
    _, h_meas, _ = compute_range_rate_residuals_analytic(x_truth, obs_data, pass_geo)
    obs_data[:, 1:5] = h_meas
    return obs_data


def _build_clean_rr_observations_from_model(t_pass_s, x_truth, pass_geo):
    rows = []
    for time_idx, t_s in enumerate(t_pass_s, start=1):
        for station_id in range(1, len(pass_geo.stations) + 1):
            rows.append([t_s, 0.0, 0.0, 0.0, 0.0, station_id, time_idx])
    obs_data = np.asarray(rows, dtype=float)
    _, h_meas = compute_range_rate_residuals(x_truth, obs_data, pass_geo)
    obs_data[:, 1:5] = h_meas
    return obs_data


def _rr_residual_rms_from_model(
    t_pass_s,
    obs_data,
    x0,
    pass_geo,
    mu_moon,
    get_earth_pos,
    get_sun_pos,
):
    x_aug0 = np.concatenate([x0, np.eye(6).reshape(-1, order="F")])
    x_aug = propagate_augmented_state(
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
    residuals = compute_range_rate_residuals(x_aug[:, :6], obs_data, pass_geo)[0]
    return float(np.sqrt(np.mean(residuals**2)))


def _rr_residual_rms_with_global_bias(
    t_pass_s,
    obs_data,
    x_est,
    pass_geo,
    mu_moon,
    get_earth_pos,
    get_sun_pos,
):
    x_aug0 = np.concatenate([x_est[:6], np.eye(6).reshape(-1, order="F")])
    x_aug = propagate_augmented_state(
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
    _, h_est, _ = compute_range_rate_residuals_analytic(x_aug[:, :6], obs_data, pass_geo)
    h_aug = h_est + x_est[6:10].reshape(1, 4)
    diff = obs_data[:, 1:5] - h_aug
    diff[:, 2] = np.arctan2(np.sin(diff[:, 2]), np.cos(diff[:, 2]))
    diff[:, 3] = np.arctan2(np.sin(diff[:, 3]), np.cos(diff[:, 3]))
    return float(np.sqrt(np.mean(diff**2)))


def _rr_residual_rms(
    t_pass_s,
    obs_data,
    x0,
    pass_geo,
    mu_moon,
    mu_earth,
    mu_sun,
    get_earth_pos,
    get_sun_pos,
):
    x_aug0 = np.concatenate([x0, np.eye(6).reshape(-1, order="F")])
    x_aug = propagate_augmented_state(
        t_pass_s,
        x_aug0,
        mu_moon,
        mu_earth,
        mu_sun,
        get_earth_pos,
        get_sun_pos,
        rtol=1e-12,
        atol=1e-13,
    )
    residuals = compute_range_rate_residuals_analytic(x_aug[:, :6], obs_data, pass_geo)[0]
    return float(np.sqrt(np.mean(residuals**2)))


def _rr_residual_rms_with_station_angle_bias(
    t_pass_s,
    obs_data,
    x_est,
    pass_geo,
    mu_moon,
    get_earth_pos,
    get_sun_pos,
):
    x_aug0 = np.concatenate([x_est[:6], np.eye(6).reshape(-1, order="F")])
    x_aug = propagate_augmented_state(
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
    _, h_est, _ = compute_range_rate_residuals_analytic(x_aug[:, :6], obs_data, pass_geo)
    h_aug = h_est.copy()
    _add_rr_station_angle_biases_to_h(h_aug, obs_data, x_est[6:])
    diff = obs_data[:, 1:5] - h_aug
    diff[:, 2] = np.arctan2(np.sin(diff[:, 2]), np.cos(diff[:, 2]))
    diff[:, 3] = np.arctan2(np.sin(diff[:, 3]), np.cos(diff[:, 3]))
    return float(np.sqrt(np.mean(diff**2)))


def _rr_residual_rms_with_station_full_bias(
    t_pass_s,
    obs_data,
    x_est,
    pass_geo,
    mu_moon,
    get_earth_pos,
    get_sun_pos,
):
    x_aug0 = np.concatenate([x_est[:6], np.eye(6).reshape(-1, order="F")])
    x_aug = propagate_augmented_state(
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
    _, h_est, _ = compute_range_rate_residuals_analytic(x_aug[:, :6], obs_data, pass_geo)
    h_aug = h_est.copy()
    _add_rr_station_full_biases_to_h(h_aug, obs_data, x_est[6:])
    diff = obs_data[:, 1:5] - h_aug
    diff[:, 2] = np.arctan2(np.sin(diff[:, 2]), np.cos(diff[:, 2]))
    diff[:, 3] = np.arctan2(np.sin(diff[:, 3]), np.cos(diff[:, 3]))
    return float(np.sqrt(np.mean(diff**2)))


def _add_station_angle_biases(obs_data, bias_vec):
    for obs_idx in range(obs_data.shape[0]):
        station_id = int(obs_data[obs_idx, 4]) - 1
        col0 = station_id * 2
        obs_data[obs_idx, 2:4] += bias_vec[col0 : col0 + 2]


def _add_station_full_biases(obs_data, bias_vec):
    for obs_idx in range(obs_data.shape[0]):
        station_id = int(obs_data[obs_idx, 4]) - 1
        col0 = station_id * 3
        obs_data[obs_idx, 1:4] += bias_vec[col0 : col0 + 3]


def _add_rr_station_angle_biases(obs_data, bias_vec):
    for obs_idx in range(obs_data.shape[0]):
        station_id = int(obs_data[obs_idx, 5]) - 1
        col0 = station_id * 2
        obs_data[obs_idx, 3:5] += bias_vec[col0 : col0 + 2]


def _add_rr_station_full_biases(obs_data, bias_vec):
    for obs_idx in range(obs_data.shape[0]):
        station_id = int(obs_data[obs_idx, 5]) - 1
        col0 = station_id * 4
        obs_data[obs_idx, 1:5] += bias_vec[col0 : col0 + 4]


def _position_residual_rms(
    t_pass_s,
    obs_data,
    x0,
    pass_geo,
    mu_moon,
    mu_earth,
    mu_sun,
    get_earth_pos,
    get_sun_pos,
):
    x_aug0 = np.concatenate([x0, np.eye(6).reshape(-1, order="F")])
    x_aug = propagate_augmented_state(
        t_pass_s,
        x_aug0,
        mu_moon,
        mu_earth,
        mu_sun,
        get_earth_pos,
        get_sun_pos,
        rtol=1e-12,
        atol=1e-13,
    )
    residuals = compute_position_residuals_analytic(x_aug[:, :6], obs_data, pass_geo)[0]
    return float(np.sqrt(np.mean(residuals**2)))


def _position_residual_rms_with_station_angle_bias(
    t_pass_s,
    obs_data,
    x_est,
    pass_geo,
    mu_moon,
    get_earth_pos,
    get_sun_pos,
):
    x_aug0 = np.concatenate([x_est[:6], np.eye(6).reshape(-1, order="F")])
    x_aug = propagate_augmented_state(
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
    _, h_est, _ = compute_position_residuals_analytic(x_aug[:, :6], obs_data, pass_geo)
    h_aug = h_est.copy()
    _add_station_angle_biases_to_h(h_aug, obs_data, x_est[6:])
    diff = obs_data[:, 1:4] - h_aug
    diff[:, 1] = np.arctan2(np.sin(diff[:, 1]), np.cos(diff[:, 1]))
    diff[:, 2] = np.arctan2(np.sin(diff[:, 2]), np.cos(diff[:, 2]))
    return float(np.sqrt(np.mean(diff**2)))


def _position_residual_rms_with_station_full_bias(
    t_pass_s,
    obs_data,
    x_est,
    pass_geo,
    mu_moon,
    get_earth_pos,
    get_sun_pos,
):
    x_aug0 = np.concatenate([x_est[:6], np.eye(6).reshape(-1, order="F")])
    x_aug = propagate_augmented_state(
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
    _, h_est, _ = compute_position_residuals_analytic(x_aug[:, :6], obs_data, pass_geo)
    h_aug = h_est.copy()
    _add_station_full_biases_to_h(h_aug, obs_data, x_est[6:])
    diff = obs_data[:, 1:4] - h_aug
    diff[:, 1] = np.arctan2(np.sin(diff[:, 1]), np.cos(diff[:, 1]))
    diff[:, 2] = np.arctan2(np.sin(diff[:, 2]), np.cos(diff[:, 2]))
    return float(np.sqrt(np.mean(diff**2)))


def _add_station_angle_biases_to_h(h_meas, obs_data, bias_vec):
    for obs_idx in range(obs_data.shape[0]):
        station_id = int(obs_data[obs_idx, 4]) - 1
        col0 = station_id * 2
        h_meas[obs_idx, 1:3] += bias_vec[col0 : col0 + 2]


def _add_station_full_biases_to_h(h_meas, obs_data, bias_vec):
    for obs_idx in range(obs_data.shape[0]):
        station_id = int(obs_data[obs_idx, 4]) - 1
        col0 = station_id * 3
        h_meas[obs_idx, :] += bias_vec[col0 : col0 + 3]


def _add_rr_station_angle_biases_to_h(h_meas, obs_data, bias_vec):
    for obs_idx in range(obs_data.shape[0]):
        station_id = int(obs_data[obs_idx, 5]) - 1
        col0 = station_id * 2
        h_meas[obs_idx, 2:4] += bias_vec[col0 : col0 + 2]


def _add_rr_station_full_biases_to_h(h_meas, obs_data, bias_vec):
    for obs_idx in range(obs_data.shape[0]):
        station_id = int(obs_data[obs_idx, 5]) - 1
        col0 = station_id * 4
        h_meas[obs_idx, :] += bias_vec[col0 : col0 + 4]


if __name__ == "__main__":
    unittest.main()
