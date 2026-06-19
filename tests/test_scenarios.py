import math
import unittest
from unittest import mock

import numpy as np

from lunar_od import (
    PassGeometry,
    PreparedArc,
    RangeRatePhysicsConfig,
    Station,
    UKFAdaptiveConfig,
    UnscentedTransformConfig,
    build_measurement_arcs,
    compute_position_residuals_analytic,
    compute_range_rate_residuals,
    compute_range_rate_residuals_analytic,
    make_cold_start_bank,
    propagate_augmented_state,
    run_batch_arc_sequence,
    run_srif_arc_sequence,
)
from tests.slow import slow


def _get_earth_pos_parallel(t):
    return np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))


def _get_sun_pos_parallel(t):
    return np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))


class ScenarioTests(unittest.TestCase):
    def test_make_cold_start_bank_is_reproducible(self):
        bank_a = make_cold_start_bank(3, 100.0, 0.1, seed=42)
        bank_b = make_cold_start_bank(3, 100.0, 0.1, seed=42)

        self.assertEqual(len(bank_a), 3)
        self.assertTrue(all(item.shape == (6,) for item in bank_a))
        for left, right in zip(bank_a, bank_b):
            np.testing.assert_allclose(left, right)

    def test_build_measurement_arcs_carries_two_way_range_rate_physics(self):
        t_sim_s = np.arange(0.0, 181.0, 60.0)
        state_history = np.column_stack(
            [
                np.full(t_sim_s.size, 2.0e6),
                np.zeros(t_sim_s.size),
                np.zeros(t_sim_s.size),
                np.full(t_sim_s.size, 100.0),
                np.zeros(t_sim_s.size),
                np.zeros(t_sim_s.size),
            ]
        )
        stations = (_synthetic_rr_station(0.0, 0.0, 0.0),)
        with mock.patch("spiceypy.sxform", return_value=np.eye(6)):
            arcs = build_measurement_arcs(
                "range_rate",
                t_sim_s,
                state_history,
                np.array([0]),
                np.array([t_sim_s.size - 1]),
                np.ones((t_sim_s.size, len(stations)), dtype=bool),
                stations,
                lambda t: np.zeros((np.size(np.asarray(t)), 3)),
                lambda t: np.zeros((np.size(np.asarray(t)), 3)),
                0.0,
                noise=False,
                range_rate_physics="two_way_counted_doppler",
                count_interval_s=20.0,
            )

        self.assertEqual(len(arcs), 1)
        self.assertEqual(arcs[0].pass_geo.range_rate_physics.mode, "two_way_counted_doppler")
        self.assertEqual(arcs[0].pass_geo.range_rate_physics.count_interval_s, 20.0)
        self.assertGreater(arcs[0].obs_data.shape[0], 0)

    def test_ukf_scenario_reports_operational_stability_for_short_arc(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_all_s = np.arange(0.0, 241.0, 60.0)
        get_earth_pos = lambda t: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        get_sun_pos = lambda t: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        truth = propagate_augmented_state(
            t_all_s,
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
            _synthetic_station(0.0, 0.0, 0.0),
            _synthetic_station(0.0, 90.0, 0.0),
            _synthetic_station(45.0, -30.0, 500.0),
        )
        arc = _build_position_arc(1, 0, 4, t_all_s, truth, stations)

        result = run_batch_arc_sequence(
            (arc,),
            "position",
            "cold",
            "ukf",
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            cold_start_bank=(np.array([10.0, -8.0, 5.0, 0.005, -0.004, 0.003]),),
            ukf_transform_config=UnscentedTransformConfig(alpha=0.35),
            rtol=1e-12,
            atol=1e-13,
        )

        arc_result = result.arc_results[0]
        self.assertTrue(arc_result.ukf_stability_passed)
        self.assertGreater(arc_result.ukf_min_covariance_eigenvalue, 0.0)
        self.assertTrue(np.isfinite(arc_result.ukf_max_covariance_condition_number))
        self.assertEqual(arc_result.ukf_robust_reweighted_fraction, 0.0)

    def test_cold_start_parallel_matches_serial(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_all_s = np.arange(0.0, 1441.0, 60.0)
        x_truth = propagate_augmented_state(
            t_all_s,
            np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")]),
            mu_moon,
            0.0,
            0.0,
            _get_earth_pos_parallel,
            _get_sun_pos_parallel,
            rtol=1e-12,
            atol=1e-13,
        )[:, :6]
        stations = (
            _synthetic_station(0.0, 0.0, 0.0),
            _synthetic_station(0.0, 90.0, 0.0),
            _synthetic_station(45.0, -30.0, 500.0),
        )
        arcs = (
            _build_position_arc(1, 0, 8, t_all_s, x_truth, stations),
            _build_position_arc(2, 4, 12, t_all_s, x_truth, stations),
            _build_position_arc(3, 8, 16, t_all_s, x_truth, stations),
            _build_position_arc(4, 12, 20, t_all_s, x_truth, stations),
        )
        cold_bank = (
            np.array([80.0, -70.0, 40.0, 0.04, -0.025, 0.015]),
            np.array([60.0, -50.0, 30.0, 0.03, -0.020, 0.012]),
            np.array([-50.0, 45.0, -25.0, -0.025, 0.018, -0.011]),
            np.array([70.0, -55.0, 35.0, 0.035, -0.022, 0.013]),
        )
        common_positional = (
            "position", "cold", "srif",
            mu_moon, 0.0, 0.0,
            _get_earth_pos_parallel, _get_sun_pos_parallel,
        )
        common_kwargs = dict(
            cold_start_bank=cold_bank,
            label="parallel_test",
            max_iter=30,
            rtol=1e-12,
            atol=1e-13,
        )
        serial = run_batch_arc_sequence(arcs, *common_positional, **common_kwargs, parallel=False)
        parallel_result = run_batch_arc_sequence(arcs, *common_positional, **common_kwargs, parallel=True)
        self.assertEqual(len(serial.arc_results), 4)
        self.assertEqual(len(parallel_result.arc_results), 4)
        for s_arc, p_arc in zip(serial.arc_results, parallel_result.arc_results):
            np.testing.assert_allclose(s_arc.estimated_state, p_arc.estimated_state, atol=1e-10)

    @slow
    def test_noisy_multi_station_batch_matrix_covers_measurement_and_start_modes(self):
        mu_moon = 4902.800066e9
        t_all_s, x_truth, get_earth_pos, get_sun_pos = _matrix_truth_history(mu_moon)
        position_stations = (
            _synthetic_station(0.0, 0.0, 0.0),
            _synthetic_station(0.0, 90.0, 0.0),
            _synthetic_station(45.0, -30.0, 500.0),
            _synthetic_station(-35.0, 150.0, 600.0),
        )
        rr_stations = (
            _synthetic_rr_station(0.0, 0.0, 0.0),
            _synthetic_rr_station(0.0, 90.0, 0.0),
            _synthetic_rr_station(45.0, -30.0, 500.0),
            _synthetic_rr_station(-35.0, 150.0, 600.0),
        )
        arc_builders = {
            "position": lambda arc_id, s, e: _with_measurement_noise(
                _build_position_arc(arc_id, s, e, t_all_s, x_truth, position_stations),
                seed=100 + arc_id,
            ),
            "geometric_rr": lambda arc_id, s, e: _with_measurement_noise(
                _build_range_rate_arc(arc_id, s, e, t_all_s, x_truth, rr_stations),
                seed=200 + arc_id,
            ),
            "two_way_rr": lambda arc_id, s, e: _with_measurement_noise(
                _build_two_way_range_rate_arc(arc_id, s, e, t_all_s, x_truth, rr_stations),
                seed=300 + arc_id,
            ),
        }
        cold_bank = (
            np.array([60.0, -50.0, 35.0, 0.025, -0.020, 0.012]),
            np.array([80.0, -65.0, 45.0, 0.030, -0.025, 0.016]),
        )

        cases = []
        for measurement_case in ("position", "geometric_rr", "two_way_rr"):
            measurement_type = "position" if measurement_case == "position" else "range_rate"
            for estimator_type in ("bls_lm", "srif"):
                start_modes = ("cold", "hot", "formal") if estimator_type == "bls_lm" else (
                    "cold",
                    "hot",
                    "formal",
                    "sqrt_formal",
                )
                for start_mode in start_modes:
                    cases.append((measurement_case, measurement_type, estimator_type, start_mode))

        for measurement_case, measurement_type, estimator_type, start_mode in cases:
            with self.subTest(measurement_case=measurement_case, estimator_type=estimator_type, start_mode=start_mode):
                arcs = (
                    arc_builders[measurement_case](1, 0, 8),
                    arc_builders[measurement_case](2, 10, 18),
                )
                result = run_batch_arc_sequence(
                    arcs,
                    measurement_type,
                    start_mode,
                    estimator_type,
                    mu_moon,
                    0.0,
                    0.0,
                    get_earth_pos,
                    get_sun_pos,
                    cold_start_bank=cold_bank,
                    label=f"{measurement_case}_{estimator_type}_{start_mode}",
                    max_iter=10,
                    rtol=1e-12,
                    atol=1e-13,
                )

                self.assertEqual(len(result.arc_results), 2)
                self.assertEqual(result.estimator_type, estimator_type)
                if measurement_case == "two_way_rr":
                    self.assertEqual(result.range_rate_physics, "two_way_counted_doppler")
                for arc_result in result.arc_results:
                    self.assertIn(arc_result.stop_reason, {"Converged", "J-Stab", "MaxIter"})
                    self.assertTrue(np.isfinite(arc_result.final_position_error_m))
                    self.assertTrue(np.isfinite(arc_result.final_velocity_error_mps))

    @slow
    def test_hot_start_uses_previous_arc_solution(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_all_s = np.arange(0.0, 2101.0, 60.0)

        get_earth_pos = lambda t: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        get_sun_pos = lambda t: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))

        x_aug0 = np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")])
        x_aug_truth = propagate_augmented_state(
            t_all_s,
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
        arcs = (
            _build_position_arc(1, 0, 12, t_all_s, x_truth, stations),
            _build_position_arc(2, 20, 32, t_all_s, x_truth, stations),
        )
        cold_bank = (
            np.array([80.0, -70.0, 40.0, 0.04, -0.025, 0.015]),
            np.array([500.0, -400.0, 250.0, 0.20, -0.15, 0.10]),
        )

        cold = run_srif_arc_sequence(
            arcs,
            "position",
            "cold",
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            cold_start_bank=cold_bank,
            label="cold",
            max_iter=50,
            rtol=1e-12,
            atol=1e-13,
        )
        hot = run_srif_arc_sequence(
            arcs,
            "position",
            "hot",
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            cold_start_bank=cold_bank,
            label="hot",
            max_iter=50,
            rtol=1e-12,
            atol=1e-13,
        )

        self.assertEqual(len(cold.arc_results), 2)
        self.assertEqual(len(hot.arc_results), 2)
        self.assertLess(hot.arc_results[1].initial_position_error_m, cold.arc_results[1].initial_position_error_m)
        self.assertLess(cold.arc_results[1].final_position_error_m, 30.0)
        self.assertLess(hot.arc_results[1].final_position_error_m, 30.0)
        self.assertEqual(hot.measurement_type, "position")
        self.assertEqual(hot.start_mode, "hot")

    @slow
    def test_ukf_arc_sequence_supports_cold_hot_and_formal_handoff(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_all_s = np.arange(0.0, 2101.0, 60.0)

        get_earth_pos = lambda t: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        get_sun_pos = lambda t: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))

        x_aug0 = np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")])
        x_truth = propagate_augmented_state(
            t_all_s,
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
            _synthetic_station(0.0, 0.0, 0.0),
            _synthetic_station(0.0, 90.0, 0.0),
            _synthetic_station(45.0, -30.0, 500.0),
        )
        arcs = (
            _build_position_arc(1, 0, 8, t_all_s, x_truth, stations),
            _build_position_arc(2, 12, 20, t_all_s, x_truth, stations),
        )
        cold_bank = (
            np.array([40.0, -30.0, 20.0, 0.02, -0.015, 0.01]),
            np.array([200.0, -160.0, 90.0, 0.08, -0.06, 0.03]),
        )

        cold = run_batch_arc_sequence(
            arcs,
            "position",
            "cold",
            "ukf",
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            cold_start_bank=cold_bank,
            label="ukf cold",
            rtol=1e-12,
            atol=1e-13,
        )
        hot = run_batch_arc_sequence(
            arcs,
            "position",
            "hot",
            "ukf",
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            cold_start_bank=cold_bank,
            label="ukf hot",
            rtol=1e-12,
            atol=1e-13,
        )
        formal = run_batch_arc_sequence(
            arcs,
            "position",
            "formal",
            "ukf",
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            cold_start_bank=cold_bank,
            label="ukf formal",
            rtol=1e-12,
            atol=1e-13,
        )
        gated = run_batch_arc_sequence(
            arcs[:1],
            "position",
            "cold",
            "ukf",
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            cold_start_bank=cold_bank[:1],
            ukf_transform_config=UnscentedTransformConfig(alpha=0.2),
            ukf_adaptive_config=UKFAdaptiveConfig(nis_gate=1e-20),
            label="ukf gated",
            rtol=1e-12,
            atol=1e-13,
        )

        self.assertEqual(cold.estimator_type, "ukf")
        self.assertEqual(len(cold.arc_results), 2)
        self.assertLess(cold.arc_results[0].final_position_error_m, cold.arc_results[0].initial_position_error_m)
        self.assertLess(hot.arc_results[1].initial_position_error_m, cold.arc_results[1].initial_position_error_m)
        self.assertIsNotNone(formal.arc_results[1].prior_covariance)
        self.assertIsNotNone(formal.arc_results[1].posterior_covariance)
        self.assertEqual(formal.arc_results[1].posterior_covariance.shape, (6, 6))
        self.assertTrue(cold.arc_results[0].ukf_stability_passed)
        self.assertGreater(cold.arc_results[0].ukf_min_covariance_eigenvalue, 0.0)
        self.assertTrue(np.isfinite(cold.arc_results[0].ukf_max_covariance_condition_number))
        self.assertEqual(cold.arc_results[0].ukf_robust_reweighted_fraction, 0.0)
        self.assertEqual(gated.arc_results[0].stop_reason, "Gated")
        self.assertEqual(gated.arc_results[0].ukf_accepted_update_fraction, 0.0)
        self.assertFalse(gated.arc_results[0].ukf_stability_passed)

    @slow
    def test_ukf_auto_bias_constraints_freeze_unobserved_station_states(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_all_s = np.arange(0.0, 721.0, 60.0)
        get_earth_pos = lambda t: np.tile(
            np.array([384400e3, 0.0, 0.0]),
            (np.size(np.asarray(t)), 1),
        )
        get_sun_pos = lambda t: np.tile(
            np.array([149.6e9, 0.0, 0.0]),
            (np.size(np.asarray(t)), 1),
        )
        truth = propagate_augmented_state(
            t_all_s,
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
            _synthetic_rr_station(0.0, 0.0, 0.0),
            _synthetic_rr_station(0.0, 90.0, 0.0),
            _synthetic_rr_station(45.0, -30.0, 500.0),
        )
        arc = _build_range_rate_arc(1, 0, 12, t_all_s, truth, stations)
        arc = PreparedArc(
            arc_id=arc.arc_id,
            start_idx=arc.start_idx,
            end_idx=arc.end_idx,
            t_pass_s=arc.t_pass_s,
            truth_state_history_mci=arc.truth_state_history_mci,
            obs_data=arc.obs_data[arc.obs_data[:, 5] == 1],
            pass_geo=arc.pass_geo,
        )

        result = run_batch_arc_sequence(
            (arc,),
            "range_rate",
            "cold",
            "ukf",
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            cold_start_bank=(np.zeros(6),),
            bias_mode="station_full",
            initial_bias=np.zeros(12),
            ukf_covariance_form="square_root",
            ukf_auto_bias_constraints=True,
            rtol=1e-11,
            atol=1e-12,
        )

        frozen = result.arc_results[0].ukf_frozen_state_indices
        self.assertTrue(set(range(10, 18)).issubset(frozen))

    @slow
    def test_formal_start_propagates_covariance_handoff(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_all_s = np.arange(0.0, 2101.0, 60.0)

        get_earth_pos = lambda t: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        get_sun_pos = lambda t: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))

        x_aug0 = np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")])
        x_aug_truth = propagate_augmented_state(
            t_all_s,
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
        arcs = (
            _build_position_arc(1, 0, 12, t_all_s, x_truth, stations),
            _build_position_arc(2, 20, 32, t_all_s, x_truth, stations),
        )
        cold_bank = (
            np.array([80.0, -70.0, 40.0, 0.04, -0.025, 0.015]),
            np.array([500.0, -400.0, 250.0, 0.20, -0.15, 0.10]),
        )

        formal = run_srif_arc_sequence(
            arcs,
            "position",
            "formal",
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            cold_start_bank=cold_bank,
            label="formal",
            max_iter=40,
            rtol=1e-12,
            atol=1e-13,
        )

        self.assertEqual(formal.start_mode, "formal")
        self.assertEqual(len(formal.arc_results), 2)
        self.assertIsNone(formal.arc_results[0].prior_covariance)
        self.assertIsNotNone(formal.arc_results[0].posterior_covariance)
        self.assertIsNotNone(formal.arc_results[1].prior_covariance)
        self.assertIsNotNone(formal.arc_results[1].posterior_covariance)
        self.assertLess(formal.arc_results[1].initial_position_error_m, np.linalg.norm(cold_bank[1][:3]))

        p_prior = formal.arc_results[1].prior_covariance
        p_post = formal.arc_results[1].posterior_covariance
        np.testing.assert_allclose(p_prior, p_prior.T, atol=1e-8)
        np.testing.assert_allclose(p_post, p_post.T, atol=1e-8)
        self.assertTrue(np.all(np.linalg.eigvalsh(p_prior) > 0.0))
        self.assertTrue(np.all(np.linalg.eigvalsh(p_post) > 0.0))

    @slow
    def test_sqrt_formal_start_carries_square_root_information(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_all_s = np.arange(0.0, 2101.0, 60.0)

        get_earth_pos = lambda t: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        get_sun_pos = lambda t: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))

        x_aug0 = np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")])
        x_aug_truth = propagate_augmented_state(
            t_all_s,
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
        arcs = (
            _build_position_arc(1, 0, 12, t_all_s, x_truth, stations),
            _build_position_arc(2, 20, 32, t_all_s, x_truth, stations),
        )
        cold_bank = (
            np.array([80.0, -70.0, 40.0, 0.04, -0.025, 0.015]),
            np.array([500.0, -400.0, 250.0, 0.20, -0.15, 0.10]),
        )

        formal = run_srif_arc_sequence(
            arcs,
            "position",
            "formal",
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            cold_start_bank=cold_bank,
            label="formal",
            max_iter=40,
            rtol=1e-12,
            atol=1e-13,
        )
        sqrt_formal = run_srif_arc_sequence(
            arcs,
            "position",
            "sqrt_formal",
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            cold_start_bank=cold_bank,
            label="sqrt formal",
            max_iter=40,
            rtol=1e-12,
            atol=1e-13,
        )

        self.assertEqual(sqrt_formal.start_mode, "sqrt_formal")
        self.assertIsNone(sqrt_formal.arc_results[0].prior_sqrt_information)
        self.assertIsNotNone(sqrt_formal.arc_results[0].posterior_sqrt_information)
        self.assertIsNotNone(sqrt_formal.arc_results[1].prior_sqrt_information)
        self.assertIsNotNone(sqrt_formal.arc_results[1].posterior_sqrt_information)
        self.assertLess(sqrt_formal.arc_results[1].initial_position_error_m, np.linalg.norm(cold_bank[1][:3]))

        r_post = sqrt_formal.arc_results[0].posterior_sqrt_information
        info_post = sqrt_formal.arc_results[0].stats.posterior_information
        rel_err = np.linalg.norm(r_post.T @ r_post - info_post) / np.linalg.norm(info_post)
        self.assertLess(rel_err, 1e-10)

        np.testing.assert_allclose(
            sqrt_formal.arc_results[1].prior_covariance,
            formal.arc_results[1].prior_covariance,
            rtol=1e-7,
            atol=1e-7,
        )

    @slow
    def test_sqrt_formal_process_noise_matches_covariance_handoff(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_all_s = np.arange(0.0, 2101.0, 60.0)

        get_earth_pos = lambda t: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        get_sun_pos = lambda t: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))

        x_aug0 = np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")])
        x_aug_truth = propagate_augmented_state(
            t_all_s,
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
        arcs = (
            _build_position_arc(1, 0, 12, t_all_s, x_truth, stations),
            _build_position_arc(2, 20, 32, t_all_s, x_truth, stations),
        )
        cold_bank = (
            np.array([80.0, -70.0, 40.0, 0.04, -0.025, 0.015]),
            np.array([500.0, -400.0, 250.0, 0.20, -0.15, 0.10]),
        )
        q_state = np.diag([2.0**2] * 3 + [2.0e-4**2] * 3)

        formal = run_srif_arc_sequence(
            arcs,
            "position",
            "formal",
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            cold_start_bank=cold_bank,
            process_noise_covariance=q_state,
            label="formal q",
            max_iter=40,
            rtol=1e-12,
            atol=1e-13,
        )
        sqrt_formal = run_srif_arc_sequence(
            arcs,
            "position",
            "sqrt_formal",
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            cold_start_bank=cold_bank,
            process_noise_covariance=q_state,
            label="sqrt formal q",
            max_iter=40,
            rtol=1e-12,
            atol=1e-13,
        )

        self.assertIsNotNone(sqrt_formal.arc_results[1].prior_sqrt_information)
        np.testing.assert_allclose(
            sqrt_formal.arc_results[1].prior_covariance,
            formal.arc_results[1].prior_covariance,
            rtol=5e-7,
            atol=5e-7,
        )

    @slow
    def test_formal_start_preserves_augmented_bias_covariance(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_all_s = np.arange(0.0, 2101.0, 60.0)
        true_bias = np.array([10.0, 1.0e-5, -1.0e-5])

        get_earth_pos = lambda t: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        get_sun_pos = lambda t: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))

        x_aug0 = np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")])
        x_aug_truth = propagate_augmented_state(
            t_all_s,
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
        arcs = [
            _build_position_arc(1, 0, 12, t_all_s, x_truth, stations),
            _build_position_arc(2, 20, 32, t_all_s, x_truth, stations),
        ]
        for arc in arcs:
            arc.obs_data[:, 1:4] += true_bias.reshape(1, 3)

        formal = run_srif_arc_sequence(
            tuple(arcs),
            "position",
            "formal",
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            cold_start_bank=(
                np.array([50.0, -45.0, 30.0, 0.03, -0.02, 0.01]),
                np.array([300.0, -250.0, 120.0, 0.10, -0.08, 0.04]),
            ),
            initial_bias=np.zeros(3),
            label="formal augmented",
            max_iter=45,
            rtol=1e-12,
            atol=1e-13,
        )

        self.assertEqual(formal.arc_results[0].posterior_covariance.shape, (9, 9))
        self.assertEqual(formal.arc_results[1].prior_covariance.shape, (9, 9))
        self.assertEqual(formal.arc_results[1].posterior_covariance.shape, (9, 9))
        self.assertGreater(np.linalg.norm(formal.arc_results[1].prior_covariance[:6, 6:]), 0.0)
        self.assertGreater(np.linalg.norm(formal.arc_results[1].posterior_covariance[:6, 6:]), 0.0)
        self.assertLess(np.linalg.norm(formal.arc_results[-1].estimated_bias - true_bias), 5.0)

    @slow
    def test_formal_range_rate_global_bias_recovery_regression(self):
        mu_moon = 4902.800066e9
        r0norm = 1737.4e3 + 100e3
        x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
        t_all_s = np.arange(0.0, 3601.0, 60.0)
        true_bias = np.array([12.0, 6.0e-5, np.deg2rad(0.0008), np.deg2rad(-0.0006)])

        get_earth_pos = lambda t: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
        get_sun_pos = lambda t: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))

        x_aug0 = np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")])
        x_aug_truth = propagate_augmented_state(
            t_all_s,
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
            _synthetic_rr_station(0.0, 0.0, 0.0),
            _synthetic_rr_station(0.0, 90.0, 0.0),
            _synthetic_rr_station(45.0, -30.0, 500.0),
            _synthetic_rr_station(-35.0, 150.0, 600.0),
        )
        arcs = [
            _build_range_rate_arc(1, 0, 12, t_all_s, x_truth, stations),
            _build_range_rate_arc(2, 20, 32, t_all_s, x_truth, stations),
            _build_range_rate_arc(3, 40, 52, t_all_s, x_truth, stations),
        ]
        for arc in arcs:
            arc.obs_data[:, 1:5] += true_bias.reshape(1, 4)

        formal = run_srif_arc_sequence(
            tuple(arcs),
            "range_rate",
            "formal",
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            cold_start_bank=(
                np.array([60.0, -50.0, 30.0, 0.03, -0.02, 0.01]),
                np.array([300.0, -240.0, 150.0, 0.10, -0.08, 0.04]),
                np.array([-260.0, 220.0, -140.0, -0.09, 0.07, -0.035]),
            ),
            initial_bias=np.zeros(4),
            label="formal rr bias regression",
            max_iter=45,
            rtol=1e-12,
            atol=1e-13,
        )

        final = formal.arc_results[-1]
        bias_err = final.estimated_bias - true_bias

        self.assertEqual(len(formal.arc_results), 3)
        self.assertEqual(formal.arc_results[0].posterior_covariance.shape, (10, 10))
        self.assertEqual(formal.arc_results[1].prior_covariance.shape, (10, 10))
        self.assertEqual(final.posterior_covariance.shape, (10, 10))
        self.assertGreater(np.linalg.norm(final.posterior_covariance[:6, 6:]), 0.0)
        self.assertLess(final.final_position_error_m, 35.0)
        self.assertLess(abs(bias_err[0]), 2.0)
        self.assertLess(abs(bias_err[1]), 2.0e-5)
        self.assertLess(abs(bias_err[2]), 4.0e-6)
        self.assertLess(abs(bias_err[3]), 4.0e-6)


def _synthetic_station(lat_deg: float, lon_deg: float, alt_m: float) -> Station:
    return Station(
        name=f"Synthetic {lat_deg:.1f} {lon_deg:.1f}",
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        alt_m=alt_m,
        color_rgb=(0.0, 0.0, 0.0),
        sigma_range_m=1.0,
        sigma_angle_rad=1e-5,
    )


def _synthetic_rr_station(lat_deg: float, lon_deg: float, alt_m: float) -> Station:
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


def _matrix_truth_history(mu_moon):
    r0norm = 1737.4e3 + 120e3
    x_true0 = np.array([r0norm, 42e3, -28e3, -18.0, math.sqrt(mu_moon / r0norm), 5.0])
    t_all_s = np.arange(0.0, 2281.0, 120.0)
    get_earth_pos = lambda t: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))
    get_sun_pos = lambda t: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))

    x_aug0 = np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")])
    x_aug_truth = propagate_augmented_state(
        t_all_s,
        x_aug0,
        mu_moon,
        0.0,
        0.0,
        get_earth_pos,
        get_sun_pos,
        rtol=1e-12,
        atol=1e-13,
    )
    return t_all_s, x_aug_truth[:, :6], get_earth_pos, get_sun_pos


def _with_measurement_noise(arc: PreparedArc, *, seed: int) -> PreparedArc:
    rng = np.random.default_rng(seed)
    obs_data = arc.obs_data.copy()
    if arc.pass_geo.measurement_type == "position":
        for row_idx in range(obs_data.shape[0]):
            station = arc.pass_geo.stations[int(obs_data[row_idx, 4]) - 1]
            obs_data[row_idx, 1] += 0.05 * station.sigma_range_m * rng.standard_normal()
            obs_data[row_idx, 2] += 0.05 * station.sigma_angle_rad * rng.standard_normal()
            obs_data[row_idx, 3] += 0.05 * station.sigma_angle_rad * rng.standard_normal()
    else:
        for row_idx in range(obs_data.shape[0]):
            station = arc.pass_geo.stations[int(obs_data[row_idx, 5]) - 1]
            obs_data[row_idx, 1] += 0.05 * station.sigma_range_m * rng.standard_normal()
            obs_data[row_idx, 2] += 0.05 * station.sigma_range_rate_mps * rng.standard_normal()
            obs_data[row_idx, 3] += 0.05 * station.sigma_angle_rad * rng.standard_normal()
            obs_data[row_idx, 4] += 0.05 * station.sigma_angle_rad * rng.standard_normal()
    return PreparedArc(
        arc_id=arc.arc_id,
        start_idx=arc.start_idx,
        end_idx=arc.end_idx,
        t_pass_s=arc.t_pass_s,
        truth_state_history_mci=arc.truth_state_history_mci,
        obs_data=obs_data,
        pass_geo=arc.pass_geo,
    )


def _build_position_arc(arc_id, start_idx, end_idx, t_all_s, x_truth, stations):
    t_pass_s = np.asarray(t_all_s[start_idx : end_idx + 1], dtype=float)
    x_pass = np.asarray(x_truth[start_idx : end_idx + 1, :], dtype=float)
    pass_geo = PassGeometry(
        t_s=t_pass_s,
        earth_pos_mci_m=np.zeros((t_pass_s.size, 3)),
        earth_vel_mci_mps=np.zeros((t_pass_s.size, 3)),
        x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass_s.size, axis=0),
        stations=stations,
        measurement_type="position",
    )
    rows = []
    for time_idx, t_s in enumerate(t_pass_s, start=1):
        for station_id in range(1, len(stations) + 1):
            rows.append([t_s, 0.0, 0.0, 0.0, station_id, time_idx, arc_id])
    obs_data = np.asarray(rows, dtype=float)
    _, h_meas, _ = compute_position_residuals_analytic(x_pass, obs_data, pass_geo)
    obs_data[:, 1:4] = h_meas
    return PreparedArc(
        arc_id=arc_id,
        start_idx=start_idx,
        end_idx=end_idx,
        t_pass_s=t_pass_s,
        truth_state_history_mci=x_pass,
        obs_data=obs_data,
        pass_geo=pass_geo,
    )


def _build_range_rate_arc(arc_id, start_idx, end_idx, t_all_s, x_truth, stations):
    t_pass_s = np.asarray(t_all_s[start_idx : end_idx + 1], dtype=float)
    x_pass = np.asarray(x_truth[start_idx : end_idx + 1, :], dtype=float)
    pass_geo = PassGeometry(
        t_s=t_pass_s,
        earth_pos_mci_m=np.zeros((t_pass_s.size, 3)),
        earth_vel_mci_mps=np.zeros((t_pass_s.size, 3)),
        x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass_s.size, axis=0),
        stations=stations,
        measurement_type="range_rate",
    )
    rows = []
    for time_idx, t_s in enumerate(t_pass_s, start=1):
        for station_id in range(1, len(stations) + 1):
            rows.append([t_s, 0.0, 0.0, 0.0, 0.0, station_id, time_idx, arc_id])
    obs_data = np.asarray(rows, dtype=float)
    _, h_meas, _ = compute_range_rate_residuals_analytic(x_pass, obs_data, pass_geo)
    obs_data[:, 1:5] = h_meas
    return PreparedArc(
        arc_id=arc_id,
        start_idx=start_idx,
        end_idx=end_idx,
        t_pass_s=t_pass_s,
        truth_state_history_mci=x_pass,
        obs_data=obs_data,
        pass_geo=pass_geo,
    )


def _build_two_way_range_rate_arc(arc_id, start_idx, end_idx, t_all_s, x_truth, stations):
    t_pass_s = np.asarray(t_all_s[start_idx : end_idx + 1], dtype=float)
    x_pass = np.asarray(x_truth[start_idx : end_idx + 1, :], dtype=float)
    pass_geo = PassGeometry(
        t_s=t_pass_s,
        earth_pos_mci_m=np.zeros((t_pass_s.size, 3)),
        earth_vel_mci_mps=np.zeros((t_pass_s.size, 3)),
        x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass_s.size, axis=0),
        stations=stations,
        measurement_type="range_rate",
        range_rate_physics=RangeRatePhysicsConfig(mode="two_way_counted_doppler", count_interval_s=20.0),
    )
    rows = []
    for time_idx, t_s in enumerate(t_pass_s, start=1):
        for station_id in range(1, len(stations) + 1):
            rows.append([t_s, 0.0, 0.0, 0.0, 0.0, station_id, time_idx, arc_id])
    obs_data = np.asarray(rows, dtype=float)
    _, h_meas = compute_range_rate_residuals(x_pass, obs_data, pass_geo)
    obs_data[:, 1:5] = h_meas
    return PreparedArc(
        arc_id=arc_id,
        start_idx=start_idx,
        end_idx=end_idx,
        t_pass_s=t_pass_s,
        truth_state_history_mci=x_pass,
        obs_data=obs_data,
        pass_geo=pass_geo,
    )


if __name__ == "__main__":
    unittest.main()
