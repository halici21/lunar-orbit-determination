import math
import unittest

import numpy as np

from lunar_od import (
    PassGeometry,
    PreparedArc,
    RangeRatePhysicsConfig,
    Station,
    BiasObservabilityPolicy,
    analyze_arc_observability,
    analyze_augmented_arc_observability,
    build_initial_state_jacobian,
    compute_position_residuals_analytic,
    compute_range_rate_residuals,
    compute_range_rate_residuals_analytic,
    decide_bias_state_handling,
    measurement_sigma_vector,
    propagate_augmented_state,
)


class ObservabilityTests(unittest.TestCase):
    def test_position_arc_observability_builds_full_rank_fisher_information(self):
        mu_moon = 4902.800066e9
        t_all_s, x_truth, get_earth_pos, get_sun_pos = _truth_history(mu_moon)
        stations = (
            _synthetic_station(0.0, 0.0, 0.0),
            _synthetic_station(0.0, 90.0, 0.0),
            _synthetic_station(45.0, -30.0, 500.0),
            _synthetic_station(-35.0, 150.0, 600.0),
        )
        arc = _build_position_arc(1, 0, 12, t_all_s, x_truth, stations)

        result = analyze_arc_observability(
            arc,
            "position",
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            rtol=1e-12,
            atol=1e-13,
        )

        h_initial = build_initial_state_jacobian(
            "position",
            arc.t_pass_s,
            arc.obs_data,
            arc.pass_geo,
            arc.truth_state_history_mci[0, :6],
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            rtol=1e-12,
            atol=1e-13,
        )
        sigma = measurement_sigma_vector(arc.obs_data, arc.pass_geo, "position")

        self.assertEqual(result.weighted_jacobian.shape, (arc.obs_data.shape[0] * 3, 6))
        self.assertEqual(result.rank, 6)
        self.assertEqual(result.rank_deficiency, 0)
        self.assertTrue(np.isfinite(result.condition_number))
        np.testing.assert_allclose(result.weighted_jacobian, h_initial / sigma[:, None], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            result.fisher_information,
            result.weighted_jacobian.T @ result.weighted_jacobian,
            rtol=1e-12,
            atol=1e-8,
        )
        self.assertGreater(result.information_eigenvalues[0], 0.0)

    def test_single_epoch_position_observability_is_rank_deficient(self):
        mu_moon = 4902.800066e9
        t_all_s, x_truth, get_earth_pos, get_sun_pos = _truth_history(mu_moon)
        stations = (_synthetic_station(0.0, 0.0, 0.0),)
        arc = _build_position_arc(1, 0, 1, t_all_s, x_truth, stations)
        arc = PreparedArc(
            arc_id=arc.arc_id,
            start_idx=arc.start_idx,
            end_idx=arc.end_idx,
            t_pass_s=arc.t_pass_s,
            truth_state_history_mci=arc.truth_state_history_mci,
            obs_data=arc.obs_data[:1, :],
            pass_geo=arc.pass_geo,
        )

        result = analyze_arc_observability(
            arc,
            "position",
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            rtol=1e-12,
            atol=1e-13,
        )

        self.assertEqual(result.weighted_jacobian.shape, (3, 6))
        self.assertEqual(result.rank, 3)
        self.assertEqual(result.rank_deficiency, 3)
        self.assertTrue(math.isinf(result.condition_number))
        self.assertLessEqual(result.information_eigenvalues[0], 1e-12)

    def test_range_rate_arc_observability_uses_four_component_weights(self):
        mu_moon = 4902.800066e9
        t_all_s, x_truth, get_earth_pos, get_sun_pos = _truth_history(mu_moon)
        stations = (
            _synthetic_rr_station(0.0, 0.0, 0.0),
            _synthetic_rr_station(0.0, 90.0, 0.0),
            _synthetic_rr_station(45.0, -30.0, 500.0),
            _synthetic_rr_station(-35.0, 150.0, 600.0),
        )
        arc = _build_range_rate_arc(1, 0, 12, t_all_s, x_truth, stations)

        result = analyze_arc_observability(
            arc,
            "range_rate",
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            rtol=1e-12,
            atol=1e-13,
        )
        sigma = measurement_sigma_vector(arc.obs_data[:1, :], arc.pass_geo, "range_rate")

        self.assertEqual(result.weighted_jacobian.shape, (arc.obs_data.shape[0] * 4, 6))
        self.assertEqual(result.rank, 6)
        self.assertTrue(np.isfinite(result.condition_number))
        np.testing.assert_allclose(sigma, np.array([1.0, 1e-4, 1e-5, 1e-5]), rtol=0.0, atol=0.0)

    def test_two_way_range_rate_observability_uses_analytic_light_time_partials(self):
        mu_moon = 4902.800066e9
        t_all_s, x_truth, get_earth_pos, get_sun_pos = _truth_history(mu_moon)
        stations = (
            _synthetic_rr_station(0.0, 0.0, 0.0),
            _synthetic_rr_station(0.0, 90.0, 0.0),
            _synthetic_rr_station(45.0, -30.0, 500.0),
            _synthetic_rr_station(-35.0, 150.0, 600.0),
        )
        arc = _build_two_way_range_rate_arc(1, 0, 12, t_all_s, x_truth, stations)

        h_initial = build_initial_state_jacobian(
            "range_rate",
            arc.t_pass_s,
            arc.obs_data,
            arc.pass_geo,
            arc.truth_state_history_mci[0, :6],
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            rtol=1e-12,
            atol=1e-13,
        )
        result = analyze_arc_observability(
            arc,
            "range_rate",
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            rtol=1e-12,
            atol=1e-13,
        )

        self.assertEqual(h_initial.shape, (arc.obs_data.shape[0] * 4, 6))
        self.assertTrue(np.all(np.isfinite(h_initial)))
        self.assertEqual(result.rank, 6)
        self.assertTrue(np.isfinite(result.condition_number))

    def test_global_range_rate_bias_augmented_observability_is_full_rank(self):
        mu_moon = 4902.800066e9
        t_all_s, x_truth, get_earth_pos, get_sun_pos = _truth_history(mu_moon)
        stations = (
            _synthetic_rr_station(0.0, 0.0, 0.0),
            _synthetic_rr_station(0.0, 90.0, 0.0),
            _synthetic_rr_station(45.0, -30.0, 500.0),
        )
        arc = _build_range_rate_arc(1, 0, 12, t_all_s, x_truth, stations)

        result = analyze_augmented_arc_observability(
            arc,
            "range_rate",
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            bias_mode="global_full",
            rtol=1e-12,
            atol=1e-13,
        )

        self.assertEqual(result.num_parameters, 10)
        self.assertEqual(result.rank, 10)
        self.assertEqual(result.rank_deficiency, 0)

    def test_unobserved_station_biases_are_rank_deficient(self):
        mu_moon = 4902.800066e9
        t_all_s, x_truth, get_earth_pos, get_sun_pos = _truth_history(mu_moon)
        stations = (
            _synthetic_rr_station(0.0, 0.0, 0.0),
            _synthetic_rr_station(0.0, 90.0, 0.0),
            _synthetic_rr_station(45.0, -30.0, 500.0),
        )
        arc = _build_range_rate_arc(1, 0, 12, t_all_s, x_truth, stations)
        observed_first_station = PreparedArc(
            arc_id=arc.arc_id,
            start_idx=arc.start_idx,
            end_idx=arc.end_idx,
            t_pass_s=arc.t_pass_s,
            truth_state_history_mci=arc.truth_state_history_mci,
            obs_data=arc.obs_data[arc.obs_data[:, 5] == 1],
            pass_geo=arc.pass_geo,
        )

        result = analyze_augmented_arc_observability(
            observed_first_station,
            "range_rate",
            mu_moon,
            0.0,
            0.0,
            get_earth_pos,
            get_sun_pos,
            bias_mode="station_full",
            rtol=1e-12,
            atol=1e-13,
        )

        self.assertEqual(result.num_parameters, 18)
        self.assertGreaterEqual(result.rank_deficiency, 8)
        self.assertTrue(math.isinf(result.condition_number))
        decision = decide_bias_state_handling(result)
        self.assertTrue(set(range(10, 18)).issubset(decision.frozen_state_indices))
        self.assertTrue(set(decision.active_state_indices).issubset(range(6, 10)))

    def test_weak_bias_information_is_regularized_before_it_is_frozen(self):
        weighted = np.zeros((6, 6))
        weighted[:3, :3] = np.eye(3)
        weighted[3, 3] = 1.0
        weighted[4, 4] = 1e-3
        from lunar_od import summarize_weighted_jacobian

        result = summarize_weighted_jacobian("position", 1, weighted)
        decision = decide_bias_state_handling(
            result,
            dynamic_state_size=3,
            policy=BiasObservabilityPolicy(
                freeze_relative_information=1e-12,
                regularize_relative_information=1e-5,
                regularization_std=2.0,
            ),
        )

        self.assertEqual(decision.active_state_indices, (3,))
        self.assertEqual(decision.regularized_state_indices, (4,))
        self.assertEqual(decision.frozen_state_indices, (5,))
        self.assertEqual(decision.regularization_std_by_state[4], 2.0)


def _truth_history(mu_moon):
    r0norm = 1737.4e3 + 100e3
    x_true0 = np.array([r0norm, 30e3, -20e3, -15.0, math.sqrt(mu_moon / r0norm), 4.0])
    t_all_s = np.arange(0.0, 901.0, 60.0)
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
