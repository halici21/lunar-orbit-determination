import dataclasses
import json
import unittest
from pathlib import Path

import numpy as np

from lunar_od import (
    ANGLE_JACOBIAN_MIN_HORIZONTAL_UNIT_NORM,
    C_LIGHT_MPS,
    MeasurementJacobianError,
    MoonCenteredEphemeris,
    PassGeometry,
    RangeRatePhysicsConfig,
    compute_position_residuals,
    compute_position_residuals_analytic,
    compute_range_rate_residuals,
    compute_range_rate_residuals_analytic,
    ecef2sez_dcm,
    generate_position_measurements,
    generate_range_rate_measurements,
    measurement_model_metadata,
    one_way_light_time_initial_state_sensitivity,
    one_way_light_time_position_initial_state_jacobian,
    one_way_light_time_position_local_state_jacobian,
    one_way_light_time_range_initial_state_jacobian,
    one_way_light_time_range_sensitivity,
    instantaneous_geometric_range_rate,
    load_spice_kernels,
    range_rate_stations,
    round_trip_light_time_initial_state_jacobian,
    solve_one_way_light_time,
    solve_two_way_light_time,
    two_way_counted_doppler_initial_state_jacobian,
    two_way_counted_doppler_observable,
)


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


class MeasurementTests(unittest.TestCase):
    def test_two_way_counted_doppler_zero_for_static_geometry(self):
        t_grid, states, earth_pos, earth_vel, xforms, station = _linear_two_way_fixture(speed_mps=0.0)
        config = RangeRatePhysicsConfig(mode="two_way_counted_doppler", count_interval_s=60.0)

        rr_eq = two_way_counted_doppler_observable(
            0.0,
            station,
            t_grid,
            states,
            earth_pos,
            earth_vel,
            xforms,
            config,
        )

        self.assertAlmostEqual(rr_eq, 0.0, places=8)

    def test_two_way_counted_doppler_tracks_receding_range_rate(self):
        speed_mps = 125.0
        t_grid, states, earth_pos, earth_vel, xforms, station = _linear_two_way_fixture(speed_mps=speed_mps)
        config = RangeRatePhysicsConfig(mode="two_way_counted_doppler", count_interval_s=60.0)

        rr_eq = two_way_counted_doppler_observable(
            0.0,
            station,
            t_grid,
            states,
            earth_pos,
            earth_vel,
            xforms,
            config,
        )
        geometric_rr = instantaneous_geometric_range_rate(states[t_grid.size // 2, :3], states[t_grid.size // 2, 3:])

        self.assertGreater(rr_eq, 0.0)
        self.assertAlmostEqual(rr_eq, geometric_rr, delta=1e-3)

    def test_two_way_counted_doppler_is_stable_across_constant_acceleration_grid_density(self):
        coarse_t = np.arange(-120.0, 121.0, 60.0)
        dense_t = np.arange(-120.0, 121.0, 1.0)
        coarse_states = _constant_acceleration_states(coarse_t)
        dense_states = _constant_acceleration_states(dense_t)
        coarse_earth = np.zeros((coarse_t.size, 3))
        dense_earth = np.zeros((dense_t.size, 3))
        coarse_xforms = np.repeat(np.eye(6)[None, :, :], coarse_t.size, axis=0)
        dense_xforms = np.repeat(np.eye(6)[None, :, :], dense_t.size, axis=0)
        config = RangeRatePhysicsConfig(mode="two_way_counted_doppler", count_interval_s=60.0)

        coarse_rr = two_way_counted_doppler_observable(
            0.0,
            _DummyStation(),
            coarse_t,
            coarse_states,
            coarse_earth,
            coarse_earth,
            coarse_xforms,
            config,
        )
        dense_rr = two_way_counted_doppler_observable(
            0.0,
            _DummyStation(),
            dense_t,
            dense_states,
            dense_earth,
            dense_earth,
            dense_xforms,
            config,
        )

        self.assertAlmostEqual(coarse_rr, dense_rr, delta=1e-6)

    def test_two_way_hz_observable_responds_to_turnaround_ratio_and_clock_drift(self):
        t_grid, states, earth_pos, earth_vel, xforms, station = _linear_two_way_fixture(speed_mps=125.0)
        base = RangeRatePhysicsConfig(
            mode="two_way_counted_doppler",
            count_interval_s=60.0,
            output_unit="hz",
            turnaround_ratio=1.0,
        )
        shifted = RangeRatePhysicsConfig(
            mode="two_way_counted_doppler",
            count_interval_s=60.0,
            output_unit="hz",
            turnaround_ratio=1.2,
            station_clock_drift=2e-4,
        )

        base_hz = two_way_counted_doppler_observable(
            0.0, station, t_grid, states, earth_pos, earth_vel, xforms, base
        )
        shifted_hz = two_way_counted_doppler_observable(
            0.0, station, t_grid, states, earth_pos, earth_vel, xforms, shifted
        )

        self.assertAlmostEqual(shifted_hz / base_hz, 1.2 * (1.0 + 2e-4), delta=2e-5)

    def test_two_way_light_time_includes_transponder_delay(self):
        t_grid, states, earth_pos, earth_vel, xforms, station = _linear_two_way_fixture(speed_mps=0.0)
        nominal = solve_two_way_light_time(
            0.0, station, t_grid, states, earth_pos, earth_vel, xforms
        )
        delayed = solve_two_way_light_time(
            0.0,
            station,
            t_grid,
            states,
            earth_pos,
            earth_vel,
            xforms,
            RangeRatePhysicsConfig(transponder_delay_s=2.5e-6),
        )

        self.assertAlmostEqual(
            delayed.round_trip_light_time_s - nominal.round_trip_light_time_s,
            2.5e-6,
            delta=1e-11,
        )

    def test_taylor3_two_way_model_rejects_long_count_intervals(self):
        RangeRatePhysicsConfig(
            mode="two_way_counted_doppler",
            count_interval_s=60.0,
            local_state_model="taylor3",
        )
        with self.assertRaises(ValueError):
            RangeRatePhysicsConfig(
                mode="two_way_counted_doppler",
                count_interval_s=120.0,
                local_state_model="taylor3",
            )

    def test_two_way_counted_doppler_residual_closure(self):
        t_grid, states, earth_pos, earth_vel, xforms, station = _linear_two_way_fixture(speed_mps=150.0)
        pass_geo = PassGeometry(
            t_s=t_grid,
            earth_pos_mci_m=earth_pos,
            earth_vel_mci_mps=earth_vel,
            x_j2000_to_itrf93=xforms,
            stations=(station,),
            measurement_type="range_rate",
            range_rate_physics=RangeRatePhysicsConfig(mode="two_way_counted_doppler", count_interval_s=60.0),
        )
        obs_data = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, float(t_grid.size // 2 + 1)]], dtype=float)
        _, h_meas = compute_range_rate_residuals(states, obs_data, pass_geo)
        obs_data[0, 1:5] = h_meas[0, :]

        residuals, h_closed = compute_range_rate_residuals(states, obs_data, pass_geo)

        np.testing.assert_allclose(h_closed, h_meas, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(residuals, np.zeros(4), rtol=0.0, atol=1e-10)
        with self.assertRaises(NotImplementedError):
            compute_range_rate_residuals_analytic(states, obs_data, pass_geo)

    def test_range_rate_apparent_companion_geometry_residual_closure(self):
        t_grid, states, earth_pos, earth_vel, xforms, station = _linear_two_way_fixture(speed_mps=150.0)
        pass_geo = PassGeometry(
            t_s=t_grid,
            earth_pos_mci_m=earth_pos,
            earth_vel_mci_mps=earth_vel,
            x_j2000_to_itrf93=xforms,
            stations=(station,),
            measurement_type="range_rate",
            range_rate_physics=RangeRatePhysicsConfig(mode="geometric_instantaneous"),
            measurement_model_profile="one_way_light_time",
            companion_geometry="apparent_one_way",
            jacobian_model="analytic_first_order_light_time",
        )
        time_index_1based = float(t_grid.size // 2 + 1)
        obs_data = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, time_index_1based]], dtype=float)
        _, h_apparent = compute_range_rate_residuals(states, obs_data, pass_geo)
        obs_data[0, 1:5] = h_apparent[0, :]

        residuals, h_closed = compute_range_rate_residuals(states, obs_data, pass_geo)
        instant_geo = dataclasses.replace(
            pass_geo,
            measurement_model_profile="geometric_instantaneous",
            companion_geometry="instantaneous",
            jacobian_model="analytic_exact_geometric",
        )
        _, h_instant = compute_range_rate_residuals(states, obs_data, instant_geo)

        np.testing.assert_allclose(h_closed, h_apparent, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(residuals, np.zeros(4), rtol=0.0, atol=1e-10)
        self.assertGreater(abs(float(h_apparent[0, 0] - h_instant[0, 0])), 10.0)
        meta = measurement_model_metadata(pass_geo)
        self.assertEqual(meta["companion_geometry"], "apparent_one_way")
        self.assertEqual(meta["measurement_model_profile"], "one_way_light_time")

    def test_two_way_counted_doppler_rr_bias_is_mps_equivalent_residual(self):
        t_grid, states, earth_pos, earth_vel, xforms, station = _linear_two_way_fixture(speed_mps=150.0)
        pass_geo = PassGeometry(
            t_s=t_grid,
            earth_pos_mci_m=earth_pos,
            earth_vel_mci_mps=earth_vel,
            x_j2000_to_itrf93=xforms,
            stations=(station,),
            measurement_type="range_rate",
            range_rate_physics=RangeRatePhysicsConfig(mode="two_way_counted_doppler", count_interval_s=60.0),
        )
        obs_data = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, float(t_grid.size // 2 + 1)]], dtype=float)
        bias_rr_mps = 0.0125
        _, h_meas = compute_range_rate_residuals(states, obs_data, pass_geo)
        obs_data[0, 1:5] = h_meas[0, :]
        obs_data[0, 2] += bias_rr_mps

        residuals, _ = compute_range_rate_residuals(states, obs_data, pass_geo)
        residual_block = residuals.reshape(-1, 4)[0]

        np.testing.assert_allclose(residual_block, [0.0, bias_rr_mps, 0.0, 0.0], rtol=0.0, atol=1e-12)

    def test_one_way_light_time_for_static_target(self):
        target_range_m = 2.5 * C_LIGHT_MPS
        solution = solve_one_way_light_time(
            100.0,
            np.zeros(3),
            lambda _t: np.array([target_range_m, 0.0, 0.0]),
        )

        self.assertTrue(solution.converged)
        self.assertAlmostEqual(solution.light_time_s, 2.5)
        self.assertAlmostEqual(solution.range_m, target_range_m)
        self.assertAlmostEqual(solution.transmit_time_s, 97.5)

    def test_one_way_light_time_for_linearly_moving_target(self):
        r0_m = 1000.0
        speed_mps = 10.0
        receive_time_s = 20.0

        solution = solve_one_way_light_time(
            receive_time_s,
            np.zeros(3),
            lambda t_s: np.array([r0_m + speed_mps * t_s, 0.0, 0.0]),
            tolerance_s=1e-15,
        )

        expected_light_time_s = (r0_m + speed_mps * receive_time_s) / (C_LIGHT_MPS + speed_mps)
        self.assertTrue(solution.converged)
        self.assertAlmostEqual(solution.light_time_s, expected_light_time_s, places=15)
        self.assertAlmostEqual(solution.transmit_time_s, receive_time_s - expected_light_time_s, places=15)

    def test_position_measurement_generation_and_clean_residual_closure(self):
        fixture_path = FIXTURES_DIR / "spice_snapshots.json"
        if not fixture_path.is_file():
            self.skipTest("MATLAB spice_snapshots.json fixture has not been exported yet.")

        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        truth = fixture["truth_propagation"]
        meas = fixture["position_measurements"]

        ephemeris = MoonCenteredEphemeris(
            t_ephem_s=truth["t_ephem_s"],
            earth_pos_m=truth["earth_pos_grid_m"],
            sun_pos_m=truth["sun_pos_grid_m"],
            earth_vel_mps=truth["earth_vel_grid_mps"],
        )
        stations_by_name = {station.name: station for station in range_rate_stations()}
        stations = [stations_by_name[name] for name in meas["station_names"]]

        import spiceypy as spice

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

        expected_clean_obs = np.asarray(meas["clean_obs_data"], dtype=float)
        np.testing.assert_allclose(clean_obs, expected_clean_obs, rtol=0.0, atol=2e-6)

        residuals, h_meas = compute_position_residuals(
            truth["state_history_mci_m_mps"],
            clean_obs,
            pass_geo,
        )
        np.testing.assert_allclose(h_meas, meas["h_meas_clean"], rtol=0.0, atol=2e-6)
        np.testing.assert_allclose(residuals, meas["residuals_clean"], rtol=0.0, atol=2e-9)
        self.assertLess(float(np.linalg.norm(residuals)), 1e-8)

        residuals_an, h_an, h_tilde = compute_position_residuals_analytic(
            truth["state_history_mci_m_mps"],
            clean_obs,
            pass_geo,
        )
        np.testing.assert_allclose(h_an, meas["h_meas_analytic"], rtol=0.0, atol=2e-6)
        np.testing.assert_allclose(residuals_an, meas["residuals_analytic"], rtol=0.0, atol=2e-9)
        np.testing.assert_allclose(h_tilde, meas["h_tilde_analytic"], rtol=0.0, atol=1e-12)

    def _position_light_time_fixture(self):
        fixture_path = FIXTURES_DIR / "spice_snapshots.json"
        if not fixture_path.is_file():
            self.skipTest("spice_snapshots.json fixture has not been exported yet.")
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        truth = fixture["truth_propagation"]
        meas = fixture["position_measurements"]
        ephemeris = MoonCenteredEphemeris(
            t_ephem_s=truth["t_ephem_s"],
            earth_pos_m=truth["earth_pos_grid_m"],
            sun_pos_m=truth["sun_pos_grid_m"],
            earth_vel_mps=truth["earth_vel_grid_mps"],
        )
        stations_by_name = {station.name: station for station in range_rate_stations()}
        stations = [stations_by_name[name] for name in meas["station_names"]]
        return {
            "state_history": np.asarray(truth["state_history_mci_m_mps"], dtype=float),
            "t_pass": np.asarray(meas["t_pass_s"], dtype=float),
            "vis_mask": meas["vis_mask_raw"],
            "stations": stations,
            "earth_pos": ephemeris.earth_position,
            "earth_vel": ephemeris.earth_velocity,
            "et0": fixture["et"],
        }

    def _generate_position_clean(self, fx, *, apply_light_time, apply_stellar_aberration=False,
                                 stellar_aberration_model=None, measurement_model_profile=None):
        import spiceypy as spice

        try:
            load_spice_kernels()
        except FileNotFoundError:
            self.skipTest("SPICE kernels not available.")
        extra = {}
        if stellar_aberration_model is not None:
            extra["stellar_aberration_model"] = stellar_aberration_model
        if measurement_model_profile is not None:
            extra["measurement_model_profile"] = measurement_model_profile
        try:
            _, pass_geo, clean_obs = generate_position_measurements(
                fx["t_pass"], fx["state_history"], fx["stations"], fx["vis_mask"],
                fx["earth_pos"], fx["earth_vel"], fx["et0"],
                noise=False, apply_light_time=apply_light_time,
                apply_stellar_aberration=apply_stellar_aberration,
                **extra,
            )
        finally:
            spice.kclear()
        return pass_geo, clean_obs

    def test_position_light_time_residual_closure(self):
        fx = self._position_light_time_fixture()
        pass_geo, clean_obs = self._generate_position_clean(fx, apply_light_time=True)
        self.assertTrue(pass_geo.apply_light_time)

        residuals, h_meas = compute_position_residuals(fx["state_history"], clean_obs, pass_geo)
        self.assertLess(float(np.max(np.abs(clean_obs[:, 1] - h_meas[:, 0]))), 1e-6)      # range [m]
        self.assertLess(float(np.max(np.abs(clean_obs[:, 2:4] - h_meas[:, 1:3]))), 1e-9)  # az/el [rad]
        self.assertLess(float(np.linalg.norm(residuals)), 1e-6)

        residuals_an, h_an, h_tilde = compute_position_residuals_analytic(
            fx["state_history"], clean_obs, pass_geo
        )
        np.testing.assert_allclose(h_an, h_meas, rtol=0.0, atol=1e-9)
        self.assertLess(float(np.linalg.norm(residuals_an)), 1e-6)
        self.assertEqual(h_tilde.shape, (3 * clean_obs.shape[0], 6))

    def test_position_profile_drives_light_time_metadata_and_residuals(self):
        fx = self._position_light_time_fixture()
        pass_geo, clean_obs = self._generate_position_clean(
            fx,
            apply_light_time=False,
            measurement_model_profile="one_way_light_time",
        )

        self.assertEqual(pass_geo.measurement_model_profile, "one_way_light_time")
        self.assertTrue(pass_geo.apply_light_time)
        self.assertFalse(pass_geo.apply_stellar_aberration)
        self.assertEqual(pass_geo.jacobian_model, "analytic_first_order_light_time")
        residuals, _h = compute_position_residuals(fx["state_history"], clean_obs, pass_geo)
        self.assertLess(float(np.linalg.norm(residuals)), 1e-6)
        meta = measurement_model_metadata(pass_geo)
        self.assertEqual(meta["measurement_model_profile"], "one_way_light_time")
        self.assertTrue(meta["apply_light_time"])

    def test_one_way_light_time_static_range_sensitivity(self):
        t_grid, states, earth_pos, _earth_vel, xforms, station = _linear_two_way_fixture(
            speed_mps=0.0
        )
        k = t_grid.size // 2

        _solution, sensitivity = one_way_light_time_range_sensitivity(
            0.0,
            station,
            t_grid,
            states,
            earth_pos[k],
            xforms[k],
        )

        np.testing.assert_allclose(
            sensitivity.d_range_d_state,
            [1.0, 0.0, 0.0, -100.0e6 / C_LIGHT_MPS, 0.0, 0.0],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            sensitivity.d_light_time_d_state,
            sensitivity.d_range_d_state / C_LIGHT_MPS,
            rtol=0.0,
            atol=1e-20,
        )
        self.assertAlmostEqual(sensitivity.condition_metric, 1.0, places=15)

    def test_one_way_light_time_range_sensitivity_matches_central_difference(self):
        t_grid, states, earth_pos, _earth_vel, xforms, station = _linear_two_way_fixture(
            speed_mps=150.0
        )
        receive_time_s = 0.0
        k = t_grid.size // 2
        _solution, sensitivity = one_way_light_time_range_sensitivity(
            receive_time_s,
            station,
            t_grid,
            states,
            earth_pos[k],
            xforms[k],
        )

        def perturb_history(delta):
            perturbed = states.copy()
            dt = (t_grid - receive_time_s)[:, None]
            perturbed[:, :3] += delta[None, :3] + dt * delta[None, 3:]
            perturbed[:, 3:] += delta[None, 3:]
            return perturbed

        steps = np.array([0.1, 0.1, 0.1, 1e-2, 1e-2, 1e-2])
        finite_difference = np.zeros(6)
        for col, step in enumerate(steps):
            delta = np.zeros(6)
            delta[col] = step
            plus, _ = one_way_light_time_range_sensitivity(
                receive_time_s,
                station,
                t_grid,
                perturb_history(delta),
                earth_pos[k],
                xforms[k],
            )
            minus, _ = one_way_light_time_range_sensitivity(
                receive_time_s,
                station,
                t_grid,
                perturb_history(-delta),
                earth_pos[k],
                xforms[k],
            )
            finite_difference[col] = (plus.range_m - minus.range_m) / (2.0 * step)

        np.testing.assert_allclose(
            sensitivity.d_range_d_state,
            finite_difference,
            rtol=0.0,
            atol=2e-5,
        )
        self.assertGreater(abs(float(sensitivity.d_range_d_state[3])), 0.1)

    def test_one_way_light_time_initial_jacobian_uses_transmit_epoch_stm(self):
        t_grid, states, earth_pos, _earth_vel, xforms, station = _linear_two_way_fixture(
            speed_mps=150.0
        )
        receive_time_s = 0.0
        k = t_grid.size // 2
        t0 = float(t_grid[0])
        phi_history = np.zeros((t_grid.size, 6, 6), dtype=float)
        for idx, epoch_s in enumerate(t_grid):
            phi_history[idx] = np.eye(6)
            phi_history[idx, :3, 3:] = (float(epoch_s) - t0) * np.eye(3)

        _solution, _sensitivity, h_initial = one_way_light_time_range_initial_state_jacobian(
            receive_time_s,
            station,
            t_grid,
            states,
            phi_history,
            earth_pos[k],
            xforms[k],
        )

        def perturb_from_initial(delta):
            perturbed = states.copy()
            dt = (t_grid - t0)[:, None]
            perturbed[:, :3] += delta[None, :3] + dt * delta[None, 3:]
            perturbed[:, 3:] += delta[None, 3:]
            return perturbed

        steps = np.array([0.1, 0.1, 0.1, 1e-2, 1e-2, 1e-2])
        finite_difference = np.zeros(6)
        for col, step in enumerate(steps):
            delta = np.zeros(6)
            delta[col] = step
            plus, _ = one_way_light_time_range_sensitivity(
                receive_time_s,
                station,
                t_grid,
                perturb_from_initial(delta),
                earth_pos[k],
                xforms[k],
            )
            minus, _ = one_way_light_time_range_sensitivity(
                receive_time_s,
                station,
                t_grid,
                perturb_from_initial(-delta),
                earth_pos[k],
                xforms[k],
            )
            finite_difference[col] = (plus.range_m - minus.range_m) / (2.0 * step)

        np.testing.assert_allclose(h_initial, finite_difference, rtol=0.0, atol=2e-5)
        self.assertGreater(abs(float(h_initial[3])), 100.0)

    def test_initial_los_sensitivity_reuses_m21_range_row(self):
        t_grid, states, earth_pos, _earth_vel, xforms, station = _linear_two_way_fixture(
            speed_mps=150.0
        )
        k = t_grid.size // 2
        t0 = float(t_grid[0])
        phi_history = np.repeat(np.eye(6)[None, :, :], t_grid.size, axis=0)
        phi_history[:, :3, 3:] = (
            (t_grid - t0)[:, None, None] * np.eye(3)[None, :, :]
        )

        solution, _local, sensitivity = one_way_light_time_initial_state_sensitivity(
            0.0, station, t_grid, states, phi_history, earth_pos[k], xforms[k]
        )
        old_solution, _old_local, old_range_row = (
            one_way_light_time_range_initial_state_jacobian(
                0.0, station, t_grid, states, phi_history, earth_pos[k], xforms[k]
            )
        )

        self.assertEqual(solution.transmit_time_s, old_solution.transmit_time_s)
        np.testing.assert_array_equal(sensitivity.d_range_dx0, old_range_row)
        self.assertLess(
            float(np.max(np.abs(sensitivity.unit_line_of_sight @ sensitivity.j_unit_los_dx0))),
            1e-20,
        )

    def test_initial_unit_los_sensitivity_matches_step_sweep_finite_difference(self):
        t_grid = np.arange(-240.0, 241.0, 20.0)
        t0 = float(t_grid[0])
        velocity = np.array([150.0, -20.0, 5.0])
        states = np.zeros((t_grid.size, 6), dtype=float)
        states[:, :3] = np.array([100.0e6, 40.0e6, 20.0e6]) + t_grid[:, None] * velocity
        states[:, 3:] = velocity
        earth_pos = np.zeros((t_grid.size, 3), dtype=float)
        xforms = np.repeat(np.eye(6)[None, :, :], t_grid.size, axis=0)
        station = _DummyStation()
        k = t_grid.size // 2
        phi_history = np.repeat(np.eye(6)[None, :, :], t_grid.size, axis=0)
        phi_history[:, :3, 3:] = (
            (t_grid - t0)[:, None, None] * np.eye(3)[None, :, :]
        )

        _solution, _local, sensitivity = one_way_light_time_initial_state_sensitivity(
            0.0, station, t_grid, states, phi_history, earth_pos[k], xforms[k]
        )

        def perturb_from_initial(delta):
            perturbed = states.copy()
            dt = (t_grid - t0)[:, None]
            perturbed[:, :3] += delta[None, :3] + dt * delta[None, 3:]
            perturbed[:, 3:] += delta[None, 3:]
            return perturbed

        sweep = (
            np.array([1.0, 1.0, 1.0, 1e-2, 1e-2, 1e-2]),
            np.array([0.1, 0.1, 0.1, 1e-3, 1e-3, 1e-3]),
            np.array([0.01, 0.01, 0.01, 1e-4, 1e-4, 1e-4]),
        )
        relative_errors = []
        for steps in sweep:
            finite_difference = np.zeros((3, 6))
            for col, step in enumerate(steps):
                delta = np.zeros(6)
                delta[col] = step
                _sp, _lp, plus = one_way_light_time_initial_state_sensitivity(
                    0.0,
                    station,
                    t_grid,
                    perturb_from_initial(delta),
                    phi_history,
                    earth_pos[k],
                    xforms[k],
                )
                _sm, _lm, minus = one_way_light_time_initial_state_sensitivity(
                    0.0,
                    station,
                    t_grid,
                    perturb_from_initial(-delta),
                    phi_history,
                    earth_pos[k],
                    xforms[k],
                )
                finite_difference[:, col] = (
                    plus.unit_line_of_sight - minus.unit_line_of_sight
                ) / (2.0 * step)
            relative_errors.append(
                np.linalg.norm(finite_difference - sensitivity.j_unit_los_dx0)
                / max(np.linalg.norm(sensitivity.j_unit_los_dx0), 1e-30)
            )

        self.assertLess(min(relative_errors), 2e-5)
        self.assertLess(relative_errors[1], 1e-4)

    def test_three_row_initial_helper_preserves_m21_range_row(self):
        t_grid, states, earth_pos, xforms, station, phi_history = (
            _static_one_way_position_fixture(az_deg=35.0, el_deg=30.0)
        )
        k = t_grid.size // 2
        position_solution, _position_sensitivity, block = (
            one_way_light_time_position_initial_state_jacobian(
                0.0, station, t_grid, states, phi_history, earth_pos[k], xforms[k]
            )
        )
        range_solution, _range_sensitivity, range_row = (
            one_way_light_time_range_initial_state_jacobian(
                0.0, station, t_grid, states, phi_history, earth_pos[k], xforms[k]
            )
        )

        self.assertEqual(position_solution.transmit_time_s, range_solution.transmit_time_s)
        np.testing.assert_array_equal(block[0], range_row)

    def test_three_row_initial_helper_matches_wrap_aware_finite_difference(self):
        from lunar_od.measurements import _apparent_position_observable
        from lunar_od.geometry import wrap_to_pi

        t_grid = np.arange(-240.0, 241.0, 20.0)
        t0 = float(t_grid[0])
        velocity = np.array([150.0, -20.0, 5.0])
        states = np.zeros((t_grid.size, 6), dtype=float)
        states[:, :3] = np.array([100.0e6, 40.0e6, 20.0e6]) + t_grid[:, None] * velocity
        states[:, 3:] = velocity
        earth_pos = np.zeros((t_grid.size, 3), dtype=float)
        xforms = np.repeat(np.eye(6)[None, :, :], t_grid.size, axis=0)
        station = _DummyStation()
        k = t_grid.size // 2
        phi_history = np.repeat(np.eye(6)[None, :, :], t_grid.size, axis=0)
        phi_history[:, :3, 3:] = (
            (t_grid - t0)[:, None, None] * np.eye(3)[None, :, :]
        )
        _solution, _sensitivity, block = one_way_light_time_position_initial_state_jacobian(
            0.0, station, t_grid, states, phi_history, earth_pos[k], xforms[k]
        )

        def perturb_from_initial(delta):
            perturbed = states.copy()
            dt = (t_grid - t0)[:, None]
            perturbed[:, :3] += delta[None, :3] + dt * delta[None, 3:]
            perturbed[:, 3:] += delta[None, 3:]
            return perturbed

        steps = np.array([0.1, 0.1, 0.1, 1e-2, 1e-2, 1e-2])
        finite_difference = np.zeros((3, 6))
        for col, step in enumerate(steps):
            delta = np.zeros(6)
            delta[col] = step
            plus, *_ = _apparent_position_observable(
                0.0,
                station,
                t_grid,
                perturb_from_initial(delta),
                earth_pos[k],
                xforms[k],
            )
            minus, *_ = _apparent_position_observable(
                0.0,
                station,
                t_grid,
                perturb_from_initial(-delta),
                earth_pos[k],
                xforms[k],
            )
            difference = plus - minus
            difference[1] = wrap_to_pi(difference[1])
            difference[2] = wrap_to_pi(difference[2])
            finite_difference[:, col] = difference / (2.0 * step)

        np.testing.assert_allclose(block[0], finite_difference[0], rtol=0.0, atol=2e-5)
        angle_relative_error = np.linalg.norm(block[1:] - finite_difference[1:]) / max(
            np.linalg.norm(finite_difference[1:]), 1e-30
        )
        self.assertLess(angle_relative_error, 2e-5)

    def test_cn_plus_s_initial_helper_matches_full_chain_fd_and_preserves_range(self):
        from lunar_od.measurements import (
            _apparent_position_observable,
            _stellar_aberration_local_jacobian,
        )
        from lunar_od.geometry import wrap_to_pi

        t_grid = np.arange(-240.0, 241.0, 20.0)
        t0 = float(t_grid[0])
        velocity = np.array([150.0, -20.0, 5.0])
        states = np.zeros((t_grid.size, 6), dtype=float)
        states[:, :3] = np.array([100.0e6, 40.0e6, 20.0e6]) + t_grid[:, None] * velocity
        states[:, 3:] = velocity
        earth_pos = np.zeros((t_grid.size, 3), dtype=float)
        xforms = np.repeat(np.eye(6)[None, :, :], t_grid.size, axis=0)
        station = _DummyStation()
        k = t_grid.size // 2
        phi_history = np.repeat(np.eye(6)[None, :, :], t_grid.size, axis=0)
        phi_history[:, :3, 3:] = (
            (t_grid - t0)[:, None, None] * np.eye(3)[None, :, :]
        )
        observer_reference_velocity = np.array([29780.0, 4200.0, -1100.0])

        _cn_solution, _cn_sensitivity, cn_block = (
            one_way_light_time_position_initial_state_jacobian(
                0.0, station, t_grid, states, phi_history, earth_pos[k], xforms[k]
            )
        )
        _app_solution, app_sensitivity, app_block = (
            one_way_light_time_position_initial_state_jacobian(
                0.0,
                station,
                t_grid,
                states,
                phi_history,
                earth_pos[k],
                xforms[k],
                apply_stellar=True,
                observer_reference_velocity_j2000_mps=observer_reference_velocity,
            )
        )
        np.testing.assert_array_equal(app_block[0], cn_block[0])

        local_aberration = _stellar_aberration_local_jacobian(
            app_sensitivity.unit_line_of_sight, observer_reference_velocity
        )
        hybrid_apparent_los = (
            local_aberration.local_jacobian @ app_sensitivity.j_unit_los_dx0
        )

        def perturb_from_initial(delta):
            perturbed = states.copy()
            dt = (t_grid - t0)[:, None]
            perturbed[:, :3] += delta[None, :3] + dt * delta[None, 3:]
            perturbed[:, 3:] += delta[None, 3:]
            return perturbed

        c_sez_mci = ecef2sez_dcm(station.lat_rad, station.lon_rad)

        def apparent_measurement_and_unit_los(perturbed):
            value, *_ = _apparent_position_observable(
                0.0,
                station,
                t_grid,
                perturbed,
                earth_pos[k],
                xforms[k],
                observer_earth_vel_rx=observer_reference_velocity,
                apply_stellar=True,
            )
            azimuth, elevation = value[1], value[2]
            unit_sez = np.array(
                [
                    -np.cos(elevation) * np.cos(azimuth),
                    np.cos(elevation) * np.sin(azimuth),
                    np.sin(elevation),
                ]
            )
            return value, c_sez_mci.T @ unit_sez

        steps = np.array([0.1, 0.1, 0.1, 1e-3, 1e-3, 1e-3])
        measurement_fd = np.zeros((3, 6))
        apparent_los_fd = np.zeros((3, 6))
        for col, step in enumerate(steps):
            delta = np.zeros(6)
            delta[col] = step
            plus_value, plus_los = apparent_measurement_and_unit_los(
                perturb_from_initial(delta)
            )
            minus_value, minus_los = apparent_measurement_and_unit_los(
                perturb_from_initial(-delta)
            )
            difference = plus_value - minus_value
            difference[1] = wrap_to_pi(difference[1])
            difference[2] = wrap_to_pi(difference[2])
            measurement_fd[:, col] = difference / (2.0 * step)
            apparent_los_fd[:, col] = (plus_los - minus_los) / (2.0 * step)

        los_relative_error = np.linalg.norm(
            hybrid_apparent_los - apparent_los_fd
        ) / max(np.linalg.norm(apparent_los_fd), 1e-30)
        azimuth_relative_error = np.linalg.norm(
            app_block[1] - measurement_fd[1]
        ) / max(np.linalg.norm(measurement_fd[1]), 1e-30)
        elevation_relative_error = np.linalg.norm(
            app_block[2] - measurement_fd[2]
        ) / max(np.linalg.norm(measurement_fd[2]), 1e-30)

        self.assertLess(los_relative_error, 2e-5)
        self.assertLess(azimuth_relative_error, 2e-5)
        self.assertLess(elevation_relative_error, 2e-5)
        np.testing.assert_allclose(app_block[0], measurement_fd[0], rtol=0.0, atol=2e-5)
        self.assertLess(
            float(np.max(np.abs(local_aberration.apparent_unit_los @ hybrid_apparent_los))),
            2e-16,
        )

    def test_implicit_angle_chain_rule_matches_wrap_aware_fd_across_sez_geometry(self):
        from lunar_od.measurements import _apparent_position_observable
        from lunar_od.geometry import wrap_to_pi

        azimuths_deg = (45.0, 135.0, 225.0, 315.0)
        elevations_deg = (5.0, 15.0, 30.0, 60.0, 80.0)
        worst_relative_error = 0.0
        for azimuth_deg in azimuths_deg:
            for elevation_deg in elevations_deg:
                t_grid, states, earth_pos, xforms, station, _phi_history = (
                    _static_one_way_position_fixture(azimuth_deg, elevation_deg)
                )
                k = t_grid.size // 2
                _solution, _sensitivity, block = (
                    one_way_light_time_position_local_state_jacobian(
                        0.0, station, t_grid, states, earth_pos[k], xforms[k]
                    )
                )

                finite_difference = np.zeros((3, 3))
                steps = (1.0, 0.1, 0.01)
                angle_relative_errors = []
                for step in steps:
                    fd_step = np.zeros((3, 3))
                    for col in range(3):
                        delta = np.zeros(3)
                        delta[col] = step
                        plus_states = states.copy()
                        minus_states = states.copy()
                        plus_states[:, :3] += delta
                        minus_states[:, :3] -= delta
                        plus, *_ = _apparent_position_observable(
                            0.0, station, t_grid, plus_states, earth_pos[k], xforms[k]
                        )
                        minus, *_ = _apparent_position_observable(
                            0.0, station, t_grid, minus_states, earth_pos[k], xforms[k]
                        )
                        difference = plus - minus
                        difference[1] = wrap_to_pi(difference[1])
                        difference[2] = wrap_to_pi(difference[2])
                        fd_step[:, col] = difference / (2.0 * step)
                    angle_relative_errors.append(
                        np.linalg.norm(fd_step[1:] - block[1:, :3])
                        / max(np.linalg.norm(block[1:, :3]), 1e-30)
                    )
                    if step == 0.1:
                        finite_difference = fd_step
                relative_error = np.linalg.norm(
                    finite_difference[1:] - block[1:, :3]
                ) / max(
                    np.linalg.norm(block[1:, :3]), 1e-30
                )
                worst_relative_error = max(worst_relative_error, relative_error)
                self.assertLess(min(angle_relative_errors), 2e-5)

                # Angle rows are radians per state unit.  For a static geometry,
                # elevation sensitivity has norm 1/range [rad/m].
                range_m = float(np.linalg.norm(states[k, :3]))
                self.assertAlmostEqual(
                    float(np.linalg.norm(block[2, :3]) * range_m), 1.0, delta=2e-10
                )
        self.assertLess(worst_relative_error, 2e-5)

    def test_implicit_angle_jacobian_near_zenith_policy_is_explicit(self):
        threshold = ANGLE_JACOBIAN_MIN_HORIZONTAL_UNIT_NORM
        inside_el_deg = np.degrees(np.arccos(0.5 * threshold))
        outside_el_deg = np.degrees(np.arccos(2.0 * threshold))

        outside = _static_one_way_position_fixture(az_deg=30.0, el_deg=outside_el_deg)
        t_grid, states, earth_pos, xforms, station, _phi = outside
        k = t_grid.size // 2
        _solution, _sensitivity, block = one_way_light_time_position_local_state_jacobian(
            0.0, station, t_grid, states, earth_pos[k], xforms[k]
        )
        self.assertTrue(np.all(np.isfinite(block)))

        inside = _static_one_way_position_fixture(az_deg=30.0, el_deg=inside_el_deg)
        t_grid, states, earth_pos, xforms, station, _phi = inside
        k = t_grid.size // 2
        with self.assertRaisesRegex(MeasurementJacobianError, "near zenith"):
            one_way_light_time_position_local_state_jacobian(
                0.0, station, t_grid, states, earth_pos[k], xforms[k]
            )

    def test_cn_plus_s_zenith_policy_uses_apparent_los(self):
        t_grid, states, earth_pos, xforms, station, phi_history = (
            _static_one_way_position_fixture(az_deg=30.0, el_deg=90.0)
        )
        k = t_grid.size // 2
        with self.assertRaisesRegex(MeasurementJacobianError, "near zenith"):
            one_way_light_time_position_initial_state_jacobian(
                0.0, station, t_grid, states, phi_history, earth_pos[k], xforms[k]
            )

        c_sez_mci = ecef2sez_dcm(station.lat_rad, station.lon_rad)
        observer_reference_velocity = c_sez_mci.T @ np.array([0.0, 3.0e4, 0.0])
        _solution, _sensitivity, apparent_block = (
            one_way_light_time_position_initial_state_jacobian(
                0.0,
                station,
                t_grid,
                states,
                phi_history,
                earth_pos[k],
                xforms[k],
                apply_stellar=True,
                observer_reference_velocity_j2000_mps=observer_reference_velocity,
            )
        )
        self.assertTrue(np.all(np.isfinite(apparent_block)))

    def test_position_azimuth_residual_wraps_across_zero_boundary(self):
        t_grid, states, earth_pos, xforms, station, _phi = _static_one_way_position_fixture(
            az_deg=359.999, el_deg=30.0
        )
        pass_geo = PassGeometry(
            t_s=t_grid,
            earth_pos_mci_m=earth_pos,
            earth_vel_mci_mps=np.zeros_like(earth_pos),
            x_j2000_to_itrf93=xforms,
            stations=(station,),
            measurement_type="position",
        )
        k = t_grid.size // 2
        obs_data = np.array([[0.0, 0.0, 0.0, 0.0, 1.0, float(k + 1)]])
        _residual, predicted = compute_position_residuals(states, obs_data, pass_geo)
        obs_data[0, 1:4] = predicted[0]
        obs_data[0, 2] = np.deg2rad(0.001)

        residual, _predicted = compute_position_residuals(states, obs_data, pass_geo)

        self.assertAlmostEqual(float(residual[1]), float(np.deg2rad(0.002)), delta=1e-14)

    def test_position_light_time_creates_physical_correction(self):
        fx = self._position_light_time_fixture()
        _, clean_inst = self._generate_position_clean(fx, apply_light_time=False)
        pass_geo_lt, clean_lt = self._generate_position_clean(fx, apply_light_time=True)
        # P0B-2B (FA-03B): the light-time run may drop leading candidates
        # whose transmit epoch precedes the frozen fixture history; those
        # drops must be structured metadata, and the physical-correction
        # comparison runs on the common (station, time-index) rows.
        metadata = pass_geo_lt.measurement_metadata
        dropped = metadata["history_domain_dropped_measurements"]
        self.assertEqual(clean_inst.shape[0], clean_lt.shape[0] + dropped)
        for record in metadata["history_domain_drop_records"]:
            self.assertEqual(record["history_name"], "spacecraft_state")
            self.assertGreater(record["required_pre_roll_s"], 0.0)
        keys_inst = {tuple(row[4:6]): i for i, row in enumerate(clean_inst)}
        matched = [
            (keys_inst[tuple(row[4:6])], j) for j, row in enumerate(clean_lt)
        ]
        self.assertEqual(len(matched), clean_lt.shape[0])
        range_diff = np.abs(
            np.array([clean_lt[j, 1] - clean_inst[i, 1] for i, j in matched])
        )
        self.assertGreater(float(np.max(range_diff)), 100.0)

    def test_position_light_time_model_mismatch_is_biased(self):
        import dataclasses

        fx = self._position_light_time_fixture()
        pass_geo_lt, clean_lt = self._generate_position_clean(fx, apply_light_time=True)
        pass_geo_inst = dataclasses.replace(
            pass_geo_lt,
            apply_light_time=False,
            measurement_model_profile="geometric_instantaneous",
            jacobian_model="analytic_exact_geometric",
        )
        _residuals, h_inst = compute_position_residuals(
            fx["state_history"], clean_lt, pass_geo_inst
        )
        range_residual = clean_lt[:, 1] - h_inst[:, 0]
        self.assertGreater(float(np.max(np.abs(range_residual))), 100.0)

    def test_position_light_time_analytic_jacobian_matches_finite_difference(self):
        from lunar_od.measurements import _apparent_position_observable
        from lunar_od.geometry import wrap_to_pi

        fx = self._position_light_time_fixture()
        pass_geo, clean = self._generate_position_clean(fx, apply_light_time=True)
        state = fx["state_history"]
        tp = fx["t_pass"]
        _, _, h_tilde = compute_position_residuals_analytic(state, clean, pass_geo)

        def lt_h(perturbed_state, i):
            k = int(clean[i, 5]) - 1
            sid = int(clean[i, 4]) - 1
            z, _t, _lt, _it = _apparent_position_observable(
                float(clean[i, 0]), pass_geo.stations[sid], tp, perturbed_state,
                pass_geo.earth_pos_mci_m[k], pass_geo.x_j2000_to_itrf93[k],
            )
            return z

        def rigid_local_perturb(k, delta):
            sp = state.copy()
            dr = delta[:3]
            dv = delta[3:6]
            dt = (tp - tp[k])[:, None]
            sp[:, :3] = sp[:, :3] + dr[None, :] + dt * dv[None, :]
            sp[:, 3:6] = sp[:, 3:6] + dv[None, :]
            return sp

        eps = np.array([10.0, 10.0, 10.0, 0.01, 0.01, 0.01])
        light_time = clean[:, 1] / 299792458.0
        for i in range(clean.shape[0]):
            k = int(clean[i, 5]) - 1
            jac = np.zeros((3, 6))
            for m in range(6):
                d = np.zeros(6)
                d[m] = eps[m]
                dz = lt_h(rigid_local_perturb(k, d), i) - lt_h(rigid_local_perturb(k, -d), i)
                dz[1] = wrap_to_pi(dz[1])
                dz[2] = wrap_to_pi(dz[2])
                jac[:, m] = dz / (2.0 * eps[m])
            block = h_tilde[3 * i:3 * i + 3, :]
            # Captured position block matches the finite difference: the neglected
            # d(tau)/dx coupling is only ~parts-per-million.
            rel = np.linalg.norm(jac[:, :3] - block[:, :3]) / max(
                np.linalg.norm(block[:, :3]), 1e-30
            )
            self.assertLess(rel, 1e-3)
            # The analytic velocity block is zero by construction (first-stage approx)...
            self.assertTrue(np.allclose(block[:, 3:6], 0.0))
            # ...and the fully-neglected d(range)/d(velocity) term equals the light time.
            self.assertAlmostEqual(float(np.linalg.norm(jac[0, 3:6])), float(light_time[i]), delta=1e-2)

    def test_position_implicit_light_time_range_row_matches_finite_difference(self):
        from lunar_od.measurements import _apparent_position_observable

        fx = self._position_light_time_fixture()
        pass_geo, clean = self._generate_position_clean(fx, apply_light_time=True)
        pass_geo = dataclasses.replace(pass_geo, jacobian_model="implicit_light_time")
        state = fx["state_history"]
        tp = fx["t_pass"]
        _, _, h_tilde = compute_position_residuals_analytic(state, clean, pass_geo)

        obs_idx = clean.shape[0] // 2
        k = int(clean[obs_idx, 5]) - 1
        station = pass_geo.stations[int(clean[obs_idx, 4]) - 1]

        def range_value(perturbed_state):
            value, _tt, _lt, _iterations = _apparent_position_observable(
                float(clean[obs_idx, 0]),
                station,
                tp,
                perturbed_state,
                pass_geo.earth_pos_mci_m[k],
                pass_geo.x_j2000_to_itrf93[k],
            )
            return float(value[0])

        def perturb_history(delta):
            perturbed = state.copy()
            dt = (tp - tp[k])[:, None]
            perturbed[:, :3] += delta[None, :3] + dt * delta[None, 3:]
            perturbed[:, 3:] += delta[None, 3:]
            return perturbed

        steps = np.array([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3])
        finite_difference = np.zeros(6)
        for col, step in enumerate(steps):
            delta = np.zeros(6)
            delta[col] = step
            finite_difference[col] = (
                range_value(perturb_history(delta)) - range_value(perturb_history(-delta))
            ) / (2.0 * step)

        np.testing.assert_allclose(
            h_tilde[3 * obs_idx, :], finite_difference, rtol=0.0, atol=5e-4
        )
        metadata = measurement_model_metadata(pass_geo)
        self.assertEqual(metadata["light_time_sensitivity"], "enabled")
        self.assertEqual(metadata["range_jacobian_model"], "implicit_light_time")
        self.assertEqual(metadata["initial_state_sensitivity_epoch"], "transmit")
        self.assertEqual(metadata["angle_jacobian_model"], "implicit_light_time_chain_rule")
        self.assertTrue(metadata["angle_jacobian_matches_full_residual_physics"])

    def test_implicit_cn_plus_s_metadata_reports_hybrid_apparent_chain(self):
        fx = self._position_light_time_fixture()
        pass_geo, _clean = self._generate_position_clean(
            fx,
            apply_light_time=True,
            apply_stellar_aberration=True,
            stellar_aberration_model="local_mci",
        )
        pass_geo = dataclasses.replace(pass_geo, jacobian_model="implicit_light_time")

        metadata = measurement_model_metadata(pass_geo)

        self.assertEqual(metadata["range_jacobian_model"], "implicit_light_time")
        self.assertEqual(metadata["line_of_sight_jacobian_model"], "implicit_light_time_chain_rule")
        self.assertEqual(metadata["angle_jacobian_model"], "hybrid_apparent_chain_rule")
        self.assertEqual(
            metadata["aberration_jacobian_model"], "local_central_finite_difference"
        )
        self.assertTrue(metadata["angle_jacobian_matches_full_residual_physics"])
        self.assertEqual(metadata["observer_velocity_epoch"], "receive")
        self.assertEqual(metadata["observer_velocity_frame"], "J2000")
        self.assertEqual(metadata["observer_velocity_reference_center"], "MOON")
        self.assertEqual(metadata["aberration_local_jacobian_input"], "unit_cn_los")
        self.assertEqual(metadata["aberration_local_jacobian_space"], "tangent")
        self.assertEqual(metadata["aberration_local_jacobian_step"], 1.0e-5)

    def test_stellar_aberration_perpendicular_shift_matches_v_over_c(self):
        """Test 3 (unit): for observer velocity perpendicular to the line of
        sight, the apparent direction shifts by phi = arcsin(v/c) toward v, and
        the vector magnitude (range) is preserved. No SPICE kernels required."""
        from lunar_od.measurements import apply_stellar_aberration

        c = 299792458.0
        r = np.array([1.0e8, 0.0, 0.0])        # line of sight along +x
        v = np.array([0.0, 3.0e4, 0.0])        # 30 km/s perpendicular, +y
        r_app = apply_stellar_aberration(r, v, light_speed_mps=c)

        # pure rotation: magnitude (range) preserved
        np.testing.assert_allclose(np.linalg.norm(r_app), np.linalg.norm(r), rtol=1e-12)
        # rotation angle equals arcsin(v/c) for the perpendicular case
        cos_ang = float(np.dot(r, r_app) / (np.linalg.norm(r) * np.linalg.norm(r_app)))
        ang = float(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
        expected = float(np.arcsin(np.linalg.norm(v) / c))
        np.testing.assert_allclose(ang, expected, rtol=1e-6)  # arccos small-angle floor
        # phi ~ v/c at this speed, and rotation is toward +v
        self.assertAlmostEqual(ang, float(np.linalg.norm(v) / c), delta=1e-9)
        self.assertGreater(float(r_app[1]), 0.0)

    def test_position_stellar_aberration_off_matches_cn(self):
        """Test 1: apply_stellar_aberration=False reproduces the CN result exactly."""
        fx = self._position_light_time_fixture()
        _, clean_cn = self._generate_position_clean(fx, apply_light_time=True)
        _, clean_off = self._generate_position_clean(
            fx, apply_light_time=True, apply_stellar_aberration=False
        )
        np.testing.assert_array_equal(clean_cn, clean_off)

    def test_position_stellar_aberration_changes_angles_not_range(self):
        """Test 2: stellar aberration leaves range essentially unchanged but
        shifts az/el by a small (sub-arcminute) non-zero amount."""
        from lunar_od.geometry import wrap_to_pi

        fx = self._position_light_time_fixture()
        _, clean_cn = self._generate_position_clean(fx, apply_light_time=True)
        pass_geo, clean_sab = self._generate_position_clean(
            fx, apply_light_time=True, apply_stellar_aberration=True
        )
        self.assertTrue(pass_geo.apply_stellar_aberration)
        self.assertEqual(clean_cn.shape, clean_sab.shape)

        # range (col 1) unchanged to sub-millimetre (rotation preserves norm)
        range_diff = np.abs(clean_sab[:, 1] - clean_cn[:, 1])
        self.assertLess(float(np.max(range_diff)), 1e-3)

        # az/el (cols 2, 3) shifted by a small but non-zero amount, well under the
        # 30 km/s (~1e-4 rad) scale since the observer MCI speed is only ~1-2 km/s
        az_diff = np.abs(wrap_to_pi(clean_sab[:, 2] - clean_cn[:, 2]))
        el_diff = np.abs(clean_sab[:, 3] - clean_cn[:, 3])
        ang_shift = np.maximum(az_diff, el_diff)
        self.assertGreater(float(np.max(ang_shift)), 1e-7)
        self.assertLess(float(np.max(ang_shift)), 1e-4)

    def test_position_stellar_aberration_residual_closure(self):
        """Test 4: noiseless measurements generated with stellar aberration
        produce near-zero residuals when predicted with the same model."""
        fx = self._position_light_time_fixture()
        pass_geo, clean = self._generate_position_clean(
            fx, apply_light_time=True, apply_stellar_aberration=True
        )
        self.assertTrue(pass_geo.apply_stellar_aberration)

        residuals, h_meas = compute_position_residuals(fx["state_history"], clean, pass_geo)
        self.assertLess(float(np.max(np.abs(clean[:, 1] - h_meas[:, 0]))), 1e-6)      # range [m]
        self.assertLess(float(np.max(np.abs(clean[:, 2:4] - h_meas[:, 1:3]))), 1e-9)  # az/el [rad]
        self.assertLess(float(np.linalg.norm(residuals)), 1e-6)

        residuals_an, h_an, h_tilde = compute_position_residuals_analytic(
            fx["state_history"], clean, pass_geo
        )
        np.testing.assert_allclose(h_an, h_meas, rtol=0.0, atol=1e-9)
        self.assertLess(float(np.linalg.norm(residuals_an)), 1e-6)
        self.assertEqual(h_tilde.shape, (3 * clean.shape[0], 6))

    def test_position_stellar_local_mci_matches_default(self):
        """Model parity: stellar_aberration_model='local_mci' reproduces the
        default (backward-compatible) local-MCI behaviour exactly."""
        fx = self._position_light_time_fixture()
        _, clean_default = self._generate_position_clean(
            fx, apply_light_time=True, apply_stellar_aberration=True
        )
        _, clean_local = self._generate_position_clean(
            fx, apply_light_time=True, apply_stellar_aberration=True,
            stellar_aberration_model="local_mci",
        )
        np.testing.assert_array_equal(clean_default, clean_local)

    def test_position_stellar_ssb_larger_and_range_preserved(self):
        """SPICE-like +S: the SSB observer velocity yields an angular correction
        of order |v_earth_ssb|/c (~1e-4 rad) -- much larger than local-MCI -- while
        range is preserved (the correction is a pure rotation)."""
        from lunar_od.geometry import wrap_to_pi

        fx = self._position_light_time_fixture()
        _, clean_cn = self._generate_position_clean(fx, apply_light_time=True)
        _, clean_loc = self._generate_position_clean(
            fx, apply_light_time=True, apply_stellar_aberration=True,
            stellar_aberration_model="local_mci",
        )
        pass_geo, clean_ssb = self._generate_position_clean(
            fx, apply_light_time=True, apply_stellar_aberration=True,
            stellar_aberration_model="spice_ssb",
        )
        self.assertEqual(pass_geo.stellar_aberration_model, "spice_ssb")
        self.assertIsNotNone(pass_geo.earth_vel_ssb_j2000_mps)

        def ang_shift(a, b):
            az = np.abs(wrap_to_pi(a[:, 2] - b[:, 2]))
            el = np.abs(a[:, 3] - b[:, 3])
            return np.maximum(az, el)

        loc_max = float(np.max(ang_shift(clean_loc, clean_cn)))
        ssb_max = float(np.max(ang_shift(clean_ssb, clean_cn)))

        # SSB correction is substantially larger (|v_ssb| ~ 30 km/s vs ~1 km/s MCI)
        self.assertGreater(ssb_max, 5.0 * loc_max)
        # ...and of order |v_earth_ssb|/c ~ 1e-4 rad, below the sin_phi <= v/c ceiling
        self.assertGreater(ssb_max, 1e-5)
        self.assertLess(ssb_max, 1.2e-4)

        # range preserved by the pure rotation (both models)
        self.assertLess(float(np.max(np.abs(clean_ssb[:, 1] - clean_cn[:, 1]))), 1e-3)

    def test_position_stellar_ssb_residual_closure(self):
        """SSB closure: noiseless spice_ssb measurements produce near-zero
        residuals when predicted with the same model (no estimator bias)."""
        fx = self._position_light_time_fixture()
        pass_geo, clean = self._generate_position_clean(
            fx, apply_light_time=True, apply_stellar_aberration=True,
            stellar_aberration_model="spice_ssb",
        )
        self.assertTrue(pass_geo.apply_stellar_aberration)
        self.assertEqual(pass_geo.stellar_aberration_model, "spice_ssb")

        residuals, h_meas = compute_position_residuals(fx["state_history"], clean, pass_geo)
        self.assertLess(float(np.max(np.abs(clean[:, 1] - h_meas[:, 0]))), 1e-6)      # range [m]
        self.assertLess(float(np.max(np.abs(clean[:, 2:4] - h_meas[:, 1:3]))), 1e-9)  # az/el [rad]
        self.assertLess(float(np.linalg.norm(residuals)), 1e-6)

        residuals_an, h_an, h_tilde = compute_position_residuals_analytic(
            fx["state_history"], clean, pass_geo
        )
        np.testing.assert_allclose(h_an, h_meas, rtol=0.0, atol=1e-9)
        self.assertLess(float(np.linalg.norm(residuals_an)), 1e-6)
        self.assertEqual(h_tilde.shape, (3 * clean.shape[0], 6))

    def test_range_rate_measurement_generation_and_clean_residual_closure(self):
        fixture_path = FIXTURES_DIR / "spice_snapshots.json"
        if not fixture_path.is_file():
            self.skipTest("MATLAB spice_snapshots.json fixture has not been exported yet.")

        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        truth = fixture["truth_propagation"]
        meas = fixture["range_rate_measurements"]

        ephemeris = MoonCenteredEphemeris(
            t_ephem_s=truth["t_ephem_s"],
            earth_pos_m=truth["earth_pos_grid_m"],
            sun_pos_m=truth["sun_pos_grid_m"],
            earth_vel_mps=truth["earth_vel_grid_mps"],
        )
        stations_by_name = {station.name: station for station in range_rate_stations()}
        stations = [stations_by_name[name] for name in meas["station_names"]]

        import spiceypy as spice

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

        expected_obs = np.asarray(meas["obs_data"], dtype=float)
        np.testing.assert_allclose(obs_data, expected_obs, rtol=0.0, atol=2e-6)

        residuals, h_meas = compute_range_rate_residuals(
            truth["state_history_mci_m_mps"],
            obs_data,
            pass_geo,
        )
        np.testing.assert_allclose(h_meas, meas["h_meas_clean"], rtol=0.0, atol=2e-6)
        np.testing.assert_allclose(residuals, meas["residuals_clean"], rtol=0.0, atol=2e-9)
        self.assertLess(float(np.linalg.norm(residuals)), 1e-8)

        residuals_an, h_an, h_tilde = compute_range_rate_residuals_analytic(
            truth["state_history_mci_m_mps"],
            obs_data,
            pass_geo,
        )
        np.testing.assert_allclose(h_an, meas["h_meas_analytic"], rtol=0.0, atol=2e-6)
        np.testing.assert_allclose(residuals_an, meas["residuals_analytic"], rtol=0.0, atol=2e-9)
        np.testing.assert_allclose(h_tilde, meas["h_tilde_analytic"], rtol=0.0, atol=1e-12)

        self._assert_rr_h_tilde_finite_difference(
            np.asarray(truth["state_history_mci_m_mps"], dtype=float),
            obs_data,
            pass_geo,
            h_tilde,
        )

    def _assert_rr_h_tilde_finite_difference(self, state_history, obs_data, pass_geo, h_tilde):
        obs0 = 0
        time_idx = int(obs_data[obs0, 6]) - 1
        row0 = obs0 * 4
        steps = np.array([1e-2, 1e-2, 1e-2, 1e-6, 1e-6, 1e-6])
        h_fd = np.zeros((4, 6))

        for col in range(6):
            x_plus = state_history.copy()
            x_minus = state_history.copy()
            x_plus[time_idx, col] += steps[col]
            x_minus[time_idx, col] -= steps[col]
            _, hp = compute_range_rate_residuals(x_plus, obs_data[[obs0], :], pass_geo)
            _, hm = compute_range_rate_residuals(x_minus, obs_data[[obs0], :], pass_geo)
            dh = (hp[0, :] - hm[0, :]) / (2.0 * steps[col])
            dh[2] = np.arctan2(np.sin(hp[0, 2] - hm[0, 2]), np.cos(hp[0, 2] - hm[0, 2])) / (2.0 * steps[col])
            dh[3] = np.arctan2(np.sin(hp[0, 3] - hm[0, 3]), np.cos(hp[0, 3] - hm[0, 3])) / (2.0 * steps[col])
            h_fd[:, col] = dh

        np.testing.assert_allclose(h_tilde[row0 : row0 + 4, :], h_fd, rtol=0.0, atol=2e-5)


class _DummyStation:
    name = "Synthetic Station"
    lat_rad = 0.0
    lon_rad = 0.0
    sigma_range_m = 1.0
    sigma_angle_rad = 1e-6
    sigma_range_rate_mps = 1e-4
    bias = ()

    @property
    def r_ecef_m(self):
        return np.zeros(3)


def _linear_two_way_fixture(speed_mps: float):
    t_grid = np.arange(-240.0, 241.0, 20.0)
    states = np.zeros((t_grid.size, 6), dtype=float)
    states[:, 0] = 100.0e6 + speed_mps * t_grid
    states[:, 3] = speed_mps
    earth_pos = np.zeros((t_grid.size, 3), dtype=float)
    earth_vel = np.zeros((t_grid.size, 3), dtype=float)
    xforms = np.repeat(np.eye(6)[None, :, :], t_grid.size, axis=0)
    return t_grid, states, earth_pos, earth_vel, xforms, _DummyStation()


def _static_one_way_position_fixture(
    az_deg: float,
    el_deg: float,
    range_m: float = 100.0e6,
):
    t_grid = np.arange(-20.0, 21.0, 10.0)
    station = _DummyStation()
    az_rad = np.deg2rad(az_deg)
    el_rad = np.deg2rad(el_deg)
    unit_sez = np.array(
        [
            -np.cos(el_rad) * np.cos(az_rad),
            np.cos(el_rad) * np.sin(az_rad),
            np.sin(el_rad),
        ]
    )
    c_sez_ecef = ecef2sez_dcm(station.lat_rad, station.lon_rad)
    position_mci = c_sez_ecef.T @ (range_m * unit_sez)
    states = np.zeros((t_grid.size, 6), dtype=float)
    states[:, :3] = position_mci
    earth_pos = np.zeros((t_grid.size, 3), dtype=float)
    xforms = np.repeat(np.eye(6)[None, :, :], t_grid.size, axis=0)
    phi_history = np.repeat(np.eye(6)[None, :, :], t_grid.size, axis=0)
    return t_grid, states, earth_pos, xforms, station, phi_history


def _constant_acceleration_states(t_grid):
    t = np.asarray(t_grid, dtype=float)
    states = np.zeros((t.size, 6), dtype=float)
    initial_range_m = 100.0e6
    speed_mps = 125.0
    acceleration_mps2 = 0.25
    states[:, 0] = initial_range_m + speed_mps * t + 0.5 * acceleration_mps2 * t**2
    states[:, 3] = speed_mps + acceleration_mps2 * t
    return states


class _SpyRng:
    """Counts standard_normal draws; returns a deterministic stream."""

    def __init__(self) -> None:
        self.calls = 0
        self._inner = np.random.default_rng(0)

    def standard_normal(self):
        self.calls += 1
        return self._inner.standard_normal()


class HistoryDomainCountedTests(unittest.TestCase):
    """P0B-2C1 tests T014-T017 for legacy counted event histories."""

    @staticmethod
    def _static_counted_fixture(
        t_start: float,
        t_end: float,
        light_time_s: float,
    ):
        t_grid = np.array([t_start, t_end], dtype=float)
        states = np.zeros((2, 6), dtype=float)
        states[:, 0] = light_time_s * C_LIGHT_MPS
        earth = np.zeros((2, 3), dtype=float)
        xforms = np.repeat(np.eye(6)[None, :, :], 2, axis=0)
        return t_grid, states, earth, xforms

    @staticmethod
    def _config():
        return RangeRatePhysicsConfig(
            mode="two_way_counted_doppler",
            count_interval_s=0.5,
            light_time_tolerance_s=1.0e-13,
        )

    def test_counted_intermediate_downlink_probe_rejected_immediately(self):
        """T014: the first unsupported spacecraft t2 probe never reaches
        the low-level state interpolator and no solution object is returned."""
        from unittest import mock

        import lunar_od.radiometrics as radiometrics_module
        from lunar_od.history_domain import HistoryDomainError

        t_grid, states, earth, xforms = self._static_counted_fixture(
            5.0, 10.0, 0.5
        )
        probes: list[float] = []
        raw_interp = radiometrics_module._interp_state

        def spy_interp(grid, history, epoch_s):
            probes.append(float(epoch_s))
            return raw_interp(grid, history, epoch_s)

        with mock.patch.object(
            radiometrics_module, "_interp_state", side_effect=spy_interp
        ):
            with self.assertRaises(HistoryDomainError) as ctx:
                solve_two_way_light_time(
                    5.3,
                    _DummyStation(),
                    t_grid,
                    states,
                    earth,
                    earth,
                    xforms,
                    self._config(),
                    endpoint_label="T014 count-start endpoint",
                )

        error = ctx.exception
        self.assertEqual(probes, [5.3])
        self.assertEqual(error.history_name, "spacecraft_state")
        self.assertEqual(error.endpoint_label, "T014 count-start endpoint")
        self.assertEqual(error.event_label, "downlink")
        self.assertAlmostEqual(error.requested_epoch_s, 4.8, delta=1.0e-14)
        self.assertAlmostEqual(error.required_pre_roll_s, 0.2, delta=1.0e-14)

    def test_counted_intermediate_uplink_probe_rejected_immediately(self):
        """T015: an unsupported station t1 probe is rejected before any
        Earth ephemeris or transform interpolation at that epoch."""
        from unittest import mock

        import lunar_od.radiometrics as radiometrics_module
        from lunar_od.history_domain import HistoryDomainError

        t_grid, states, earth, xforms = self._static_counted_fixture(
            5.0, 10.0, 0.4
        )
        probes: list[float] = []
        raw_interp = radiometrics_module._interp_vector

        def spy_interp(grid, history, epoch_s):
            probes.append(float(epoch_s))
            return raw_interp(grid, history, epoch_s)

        with mock.patch.object(
            radiometrics_module, "_interp_vector", side_effect=spy_interp
        ):
            with self.assertRaises(HistoryDomainError) as ctx:
                solve_two_way_light_time(
                    5.7,
                    _DummyStation(),
                    t_grid,
                    states,
                    earth,
                    earth,
                    xforms,
                    self._config(),
                    endpoint_label="T015 count-end endpoint",
                )

        error = ctx.exception
        # The receive-epoch station query legitimately evaluates all three
        # station histories (Earth position, Earth velocity, and sxform).
        # The important contract is that no uplink-epoch interpolation at
        # 4.9 s occurs before the domain guard raises.
        self.assertEqual(probes, [5.7, 5.7, 5.7])
        self.assertEqual(error.history_name, "earth_position_mci")
        self.assertEqual(error.endpoint_label, "T015 count-end endpoint")
        self.assertEqual(error.event_label, "uplink")
        self.assertAlmostEqual(error.requested_epoch_s, 4.9, delta=1.0e-14)
        self.assertAlmostEqual(error.required_pre_roll_s, 0.1, delta=1.0e-14)

    def test_counted_start_end_and_leg_diagnostics(self):
        """T016: endpoint and uplink/downlink labels survive every failure."""
        from lunar_od.history_domain import HistoryDomainError

        observed = set()
        for endpoint_label in ("count-start endpoint", "count-end endpoint"):
            for event_label, receive_time_s, light_time_s in (
                ("downlink", 5.3, 0.5),
                ("uplink", 5.7, 0.4),
            ):
                t_grid, states, earth, xforms = self._static_counted_fixture(
                    5.0, 10.0, light_time_s
                )
                with self.assertRaises(HistoryDomainError) as ctx:
                    solve_two_way_light_time(
                        receive_time_s,
                        _DummyStation(),
                        t_grid,
                        states,
                        earth,
                        earth,
                        xforms,
                        self._config(),
                        endpoint_label=endpoint_label,
                    )
                observed.add((ctx.exception.endpoint_label, ctx.exception.event_label))
                self.assertEqual(ctx.exception.event_label, event_label)

        self.assertEqual(
            observed,
            {
                ("count-start endpoint", "downlink"),
                ("count-start endpoint", "uplink"),
                ("count-end endpoint", "downlink"),
                ("count-end endpoint", "uplink"),
            },
        )

    def test_counted_five_guarded_histories_report_exact_names(self):
        """T017: nominal/Jacobian wiring covers all five named histories."""
        from unittest import mock

        import lunar_od.radiometrics as radiometrics_module
        from lunar_od.history_domain import HistoryDomainError

        t_grid, states, earth, earth_vel, xforms, station = _linear_two_way_fixture(0.0)
        phi_flat = np.eye(6).reshape(-1, order="F")
        augmented = np.column_stack(
            [states, np.repeat(phi_flat[None, :], t_grid.size, axis=0)]
        )
        original_normalize = radiometrics_module.normalize_supported_epoch
        expected_names = (
            "spacecraft_state",
            "spacecraft_stm",
            "earth_position_mci",
            "earth_velocity_mci",
            "j2000_to_itrf93_state_transform",
        )

        for target_name in expected_names:
            def selective_normalize(
                requested_epoch_s,
                support_start_s,
                support_end_s,
                *,
                history_name,
                model_context,
                consumer,
                endpoint_label=None,
                event_label=None,
                observation_index=None,
            ):
                if history_name == target_name:
                    raise HistoryDomainError.from_violation(
                        history_name=history_name,
                        requested_epoch_s=float(support_start_s) - 1.0,
                        support_start_s=support_start_s,
                        support_end_s=support_end_s,
                        model_context=model_context,
                        consumer=consumer,
                        endpoint_label=endpoint_label,
                        event_label=event_label,
                        observation_index=observation_index,
                    )
                return original_normalize(
                    requested_epoch_s,
                    support_start_s,
                    support_end_s,
                    history_name=history_name,
                    model_context=model_context,
                    consumer=consumer,
                    endpoint_label=endpoint_label,
                    event_label=event_label,
                    observation_index=observation_index,
                )

            with self.subTest(history_name=target_name):
                with mock.patch.object(
                    radiometrics_module,
                    "normalize_supported_epoch",
                    side_effect=selective_normalize,
                ):
                    with self.assertRaises(HistoryDomainError) as ctx:
                        round_trip_light_time_initial_state_jacobian(
                            0.0,
                            station,
                            t_grid,
                            augmented,
                            earth,
                            earth_vel,
                            xforms,
                            self._config(),
                            endpoint_label="T017 Jacobian endpoint",
                        )
                self.assertEqual(ctx.exception.history_name, target_name)
                self.assertEqual(
                    ctx.exception.endpoint_label, "T017 Jacobian endpoint"
                )


class HistoryDomainOneWayTests(unittest.TestCase):
    """P0B-2B tests T007-T012 plus the range-rate generation portions of
    T018/T019/T028 (drops driven through the one-way apparent companion)."""

    @staticmethod
    def _static_history(t_start: float, t_end: float, light_time_s: float = 0.25):
        t_grid = np.array([t_start, t_end], dtype=float)
        states = np.zeros((2, 6), dtype=float)
        states[:, 0] = light_time_s * C_LIGHT_MPS
        return t_grid, states

    @staticmethod
    def _station():
        class _Station:
            name = "history-domain station"
            lat_rad = 0.0
            lon_rad = 0.0
            sigma_range_m = 5.0
            sigma_angle_rad = 1.0e-5
            sigma_range_rate_mps = 1.0e-4
            bias = ()
            r_ecef_m = np.zeros(3)

        return _Station()

    def test_one_way_intermediate_probe_rejected_immediately(self):
        """T007: the FIRST unsupported probe raises; no later evaluation, no
        solution object, and the guard never feeds the extrapolator."""
        from lunar_od.history_domain import (
            HistoryDomainError,
            guarded_history_callback,
        )
        from lunar_od.measurements import solve_one_way_light_time as raw_solver

        probes: list[float] = []

        def spy_target(t_s: float) -> np.ndarray:
            probes.append(float(t_s))
            # A hypothetical converged root would be inside support (static
            # geometry, light time 0.5 s from receive 5.3 -> t_t = 4.8), but
            # the guard must reject the first outside trial regardless.
            return np.array([0.5 * C_LIGHT_MPS, 0.0, 0.0])

        guarded = guarded_history_callback(
            spy_target,
            5.0,
            10.0,
            history_name="spacecraft_state",
            model_context="one_way_light_time",
            consumer="T007",
        )
        with self.assertRaises(HistoryDomainError) as ctx:
            raw_solver(5.3, np.zeros(3), guarded)
        # First probe is the receive epoch 5.3 (inside, evaluated); the next
        # probe is the transmit trial 4.8 (outside) and must never reach the
        # spy. No solution/convergence object exists after the raise.
        self.assertEqual(probes, [5.3])
        self.assertEqual(ctx.exception.requested_epoch_s, 5.3 - 0.5)
        self.assertEqual(ctx.exception.required_pre_roll_s, 5.0 - 4.8)

    def test_one_way_raw_solver_generic_callback_compatibility(self):
        """T008: the raw solver keeps working with analytic callbacks that
        have no repository history at all."""
        solution = solve_one_way_light_time(
            0.0,
            np.zeros(3),
            lambda t: np.array([0.25 * C_LIGHT_MPS, 0.0, 0.0]),
        )
        self.assertTrue(solution.converged)
        self.assertAlmostEqual(solution.light_time_s, 0.25, delta=1e-12)
        self.assertAlmostEqual(solution.transmit_time_s, -0.25, delta=1e-12)

    def test_one_way_state_and_stm_share_normalized_epoch(self):
        """T009: at a 2-ULP-below-start transmit epoch the state and the STM
        position block use the exact same snapped endpoint sample."""
        ulp_one = np.nextafter(1.0, np.inf) - 1.0
        t_grid, states = self._static_history(0.0, 10.0)
        # Distinguishable STM node values: start node scaled, end node identity.
        phi_start = np.eye(6)
        phi_start[:3, :3] *= 3.0
        phi_history = np.stack([phi_start, np.eye(6)])
        receive_time_s = 0.25 - 2.0 * ulp_one  # transmit lands 2 ULP below 0
        solution, _sens, initial = one_way_light_time_initial_state_sensitivity(
            receive_time_s,
            self._station(),
            t_grid,
            states,
            phi_history,
            np.zeros(3),
            np.eye(6),
        )
        self.assertLess(solution.transmit_time_s, t_grid[0])  # event unclipped
        np.testing.assert_array_equal(initial.phi_r_tx, phi_start[:3, :])
        np.testing.assert_array_equal(
            initial.spacecraft_position_tx_m, states[0, :3]
        )

    def _generate_position(self, t_grid, states, *, noise, rng, profile="one_way_light_time"):
        from unittest import mock

        vis = np.ones((t_grid.size, 1), dtype=bool)
        sample = lambda tt: np.zeros((np.size(np.atleast_1d(tt)), 3))
        with mock.patch("spiceypy.sxform", return_value=np.eye(6)):
            return generate_position_measurements(
                t_grid,
                states,
                (self._station(),),
                vis,
                sample,
                sample,
                0.0,
                noise=noise,
                rng=rng,
                measurement_model_profile=profile,
            )

    def test_position_generation_drops_candidate_and_preserves_rng(self):
        """T010: a dropped first candidate consumes its RNG slot so later
        noisy rows are bit-identical to a pre-rolled reference run."""
        light_time_s = 0.25
        t_enforced = np.linspace(0.0, 4.0, 5)
        states_enforced = np.zeros((5, 6))
        states_enforced[:, 0] = light_time_s * C_LIGHT_MPS
        # Pre-rolled P0B-1-equivalent reference: same receive tags, but the
        # history extends early enough that every candidate is supported.
        t_reference = np.concatenate([[-1.0], t_enforced])
        states_reference = np.zeros((6, 6))
        states_reference[:, 0] = light_time_s * C_LIGHT_MPS
        vis_reference = np.ones((6, 1), dtype=bool)
        vis_reference[0, 0] = False  # same visible candidates as enforced run

        from unittest import mock

        sample = lambda tt: np.zeros((np.size(np.atleast_1d(tt)), 3))
        obs_enforced, pass_geo, clean_enforced = self._generate_position(
            t_enforced, states_enforced, noise=True, rng=np.random.default_rng(42)
        )
        with mock.patch("spiceypy.sxform", return_value=np.eye(6)):
            obs_reference, _, clean_reference = generate_position_measurements(
                t_reference,
                states_reference,
                (self._station(),),
                vis_reference,
                sample,
                sample,
                0.0,
                noise=True,
                rng=np.random.default_rng(42),
                measurement_model_profile="one_way_light_time",
            )

        metadata = pass_geo.measurement_metadata
        self.assertEqual(metadata["history_domain_dropped_measurements"], 1)
        self.assertEqual(len(metadata["history_domain_drop_records"]), 1)
        record = metadata["history_domain_drop_records"][0]
        self.assertEqual(record["station_index"], 0)
        self.assertEqual(record["time_index"], 0)
        self.assertEqual(record["candidate_ordinal"], 0)
        self.assertFalse(metadata["history_domain_all_candidates_dropped"])
        # Candidate 0 (t=0, transmit -0.25 s) dropped; rows 1.. survive and
        # must match the reference run rows at the same receive tags bitwise.
        # (The reference grid has one extra leading node, so its 1-based
        # time-index column is shifted by exactly +1; every physical and
        # noise column must be bit-identical.)
        self.assertEqual(obs_enforced.shape[0], 4)
        np.testing.assert_array_equal(obs_enforced[:, :5], obs_reference[1:, :5])
        np.testing.assert_array_equal(clean_enforced[:, :5], clean_reference[1:, :5])
        np.testing.assert_array_equal(obs_enforced[:, 5], obs_reference[1:, 5] - 1.0)
        self.assertEqual(obs_enforced.shape[0], clean_enforced.shape[0])

    def test_position_generation_noise_false_consumes_no_rng(self):
        """T011: with noise=False the generator makes zero RNG calls even
        when a candidate is dropped."""
        t_grid = np.linspace(0.0, 4.0, 5)
        states = np.zeros((5, 6))
        states[:, 0] = 0.25 * C_LIGHT_MPS
        spy = _SpyRng()
        obs, pass_geo, _clean = self._generate_position(
            t_grid, states, noise=False, rng=spy
        )
        self.assertEqual(spy.calls, 0)
        self.assertEqual(
            pass_geo.measurement_metadata["history_domain_dropped_measurements"], 1
        )
        self.assertEqual(obs.shape[0], 4)

    def test_one_way_drop_metadata_reports_geometry_dependent_preroll(self):
        """T012: the maximum required pre-roll follows the solved/probed
        light-time geometry; it is not a fixed duration."""
        required = {}
        for light_time_s in (0.25, 0.75):
            t_grid = np.linspace(0.0, 4.0, 5)
            states = np.zeros((5, 6))
            states[:, 0] = light_time_s * C_LIGHT_MPS
            _obs, pass_geo, _clean = self._generate_position(
                t_grid, states, noise=False, rng=None
            )
            required[light_time_s] = pass_geo.measurement_metadata[
                "history_domain_required_pre_roll_s"
            ]
        self.assertGreater(required[0.75], required[0.25])
        self.assertAlmostEqual(required[0.25], 0.25, delta=1e-2)
        self.assertAlmostEqual(required[0.75], 0.75, delta=1e-2)

    def _generate_range_rate(self, t_grid, states, *, noise, rng):
        from unittest import mock

        vis = np.ones((t_grid.size, 1), dtype=bool)
        sample = lambda tt: np.zeros((np.size(np.atleast_1d(tt)), 3))
        with mock.patch("spiceypy.sxform", return_value=np.eye(6)):
            return generate_range_rate_measurements(
                t_grid,
                states,
                (self._station(),),
                vis,
                sample,
                sample,
                0.0,
                noise=noise,
                rng=rng,
                companion_geometry="apparent_one_way",
            )

    def _generate_counted_range_rate(
        self,
        t_grid,
        states,
        visibility,
        *,
        noise,
        rng,
    ):
        from unittest import mock

        sample = lambda tt: np.zeros((np.size(np.atleast_1d(tt)), 3))
        config = RangeRatePhysicsConfig(
            mode="two_way_counted_doppler",
            count_interval_s=0.5,
            light_time_tolerance_s=1.0e-13,
        )
        with mock.patch("spiceypy.sxform", return_value=np.eye(6)):
            return generate_range_rate_measurements(
                t_grid,
                states,
                (self._station(),),
                visibility,
                sample,
                sample,
                0.0,
                noise=noise,
                rng=rng,
                range_rate_physics=config,
                companion_geometry="instantaneous",
            )

    def test_range_rate_generation_drops_candidate_and_preserves_rng(self):
        """T018: apparent and counted drops preserve four-draw RNG slots."""
        from unittest import mock

        light_time_s = 0.25
        t_enforced = np.linspace(0.0, 4.0, 5)
        states_enforced = np.zeros((5, 6))
        states_enforced[:, 0] = light_time_s * C_LIGHT_MPS
        t_reference = np.concatenate([[-1.0], t_enforced])
        states_reference = np.zeros((6, 6))
        states_reference[:, 0] = light_time_s * C_LIGHT_MPS
        vis_reference = np.ones((6, 1), dtype=bool)
        vis_reference[0, 0] = False
        sample = lambda tt: np.zeros((np.size(np.atleast_1d(tt)), 3))

        obs_enforced, pass_geo = self._generate_range_rate(
            t_enforced, states_enforced, noise=True, rng=np.random.default_rng(7)
        )
        with mock.patch("spiceypy.sxform", return_value=np.eye(6)):
            obs_reference, _ = generate_range_rate_measurements(
                t_reference,
                states_reference,
                (self._station(),),
                vis_reference,
                sample,
                sample,
                0.0,
                noise=True,
                rng=np.random.default_rng(7),
                companion_geometry="apparent_one_way",
            )
        metadata = pass_geo.measurement_metadata
        self.assertEqual(metadata["history_domain_dropped_measurements"], 1)
        self.assertEqual(obs_enforced.shape[0], 4)
        # Physical/noise columns bit-identical; the reference time-index
        # column is shifted by the extra pre-rolled leading node.
        np.testing.assert_array_equal(obs_enforced[:, :6], obs_reference[1:, :6])
        np.testing.assert_array_equal(obs_enforced[:, 6], obs_reference[1:, 6] - 1.0)

        # Counted C1 path: the first midpoint's count-start/downlink query is
        # unsupported on [0, 5], while the same physical candidate is valid
        # against a pre-rolled [-1, 5] history. Later noisy rows must remain
        # bit-identical despite the dropped candidate consuming its RNG slot.
        counted_t_enforced = np.arange(0.0, 6.0)
        counted_states_enforced = np.zeros((6, 6))
        counted_states_enforced[:, 0] = light_time_s * C_LIGHT_MPS
        counted_vis_enforced = np.ones((6, 1), dtype=bool)
        counted_vis_enforced[-1, 0] = False

        counted_t_reference = np.arange(-1.0, 6.0)
        counted_states_reference = np.zeros((7, 6))
        counted_states_reference[:, 0] = light_time_s * C_LIGHT_MPS
        counted_vis_reference = np.ones((7, 1), dtype=bool)
        counted_vis_reference[0, 0] = False
        counted_vis_reference[-1, 0] = False

        counted_enforced, counted_pass_geo = self._generate_counted_range_rate(
            counted_t_enforced,
            counted_states_enforced,
            counted_vis_enforced,
            noise=True,
            rng=np.random.default_rng(11),
        )
        counted_reference, _ = self._generate_counted_range_rate(
            counted_t_reference,
            counted_states_reference,
            counted_vis_reference,
            noise=True,
            rng=np.random.default_rng(11),
        )

        counted_metadata = counted_pass_geo.measurement_metadata
        self.assertEqual(counted_metadata["history_domain_dropped_measurements"], 1)
        self.assertEqual(counted_enforced.shape[0], 4)
        counted_record = counted_metadata["history_domain_drop_records"][0]
        self.assertEqual(counted_record["endpoint_label"], "count-start endpoint")
        self.assertEqual(counted_record["event_label"], "downlink")
        np.testing.assert_array_equal(
            counted_enforced[:, :6], counted_reference[1:, :6]
        )
        np.testing.assert_array_equal(
            counted_enforced[:, 6], counted_reference[1:, 6] - 1.0
        )

    def test_range_rate_generation_noise_false_consumes_no_rng(self):
        """T019: apparent and counted drops consume no RNG when noise=False."""
        t_grid = np.linspace(0.0, 4.0, 5)
        states = np.zeros((5, 6))
        states[:, 0] = 0.25 * C_LIGHT_MPS
        spy = _SpyRng()
        obs, pass_geo = self._generate_range_rate(t_grid, states, noise=False, rng=spy)
        self.assertEqual(spy.calls, 0)
        self.assertEqual(
            pass_geo.measurement_metadata["history_domain_dropped_measurements"], 1
        )
        self.assertEqual(obs.shape[0], 4)

        counted_t = np.arange(0.0, 6.0)
        counted_states = np.zeros((6, 6))
        counted_states[:, 0] = 0.25 * C_LIGHT_MPS
        counted_vis = np.ones((6, 1), dtype=bool)
        counted_vis[-1, 0] = False
        counted_spy = _SpyRng()
        counted_obs, counted_pass_geo = self._generate_counted_range_rate(
            counted_t,
            counted_states,
            counted_vis,
            noise=False,
            rng=counted_spy,
        )
        self.assertEqual(counted_spy.calls, 0)
        self.assertEqual(
            counted_pass_geo.measurement_metadata[
                "history_domain_dropped_measurements"
            ],
            1,
        )
        self.assertEqual(counted_obs.shape[0], 4)

    def test_metadata_structured_drop_record_order_is_deterministic(self):
        """T028 (P0B-2B portion): drop records follow candidate order and are
        identical across identical runs (multiple stations, multiple drops)."""
        from unittest import mock

        t_grid = np.linspace(0.0, 4.0, 5)
        states = np.zeros((5, 6))
        states[:, 0] = 0.25 * C_LIGHT_MPS
        vis = np.ones((5, 2), dtype=bool)
        sample = lambda tt: np.zeros((np.size(np.atleast_1d(tt)), 3))
        stations = (self._station(), self._station())

        def run():
            with mock.patch("spiceypy.sxform", return_value=np.eye(6)):
                _obs, pass_geo, _clean = generate_position_measurements(
                    t_grid,
                    states,
                    stations,
                    vis,
                    sample,
                    sample,
                    0.0,
                    noise=False,
                    measurement_model_profile="one_way_light_time",
                )
            return pass_geo.measurement_metadata["history_domain_drop_records"]

        first = run()
        second = run()
        self.assertEqual(first, second)
        # Both stations drop at time index 0, in candidate order.
        self.assertEqual(
            [(r["station_index"], r["time_index"], r["candidate_ordinal"]) for r in first],
            [(0, 0, 0), (1, 0, 1)],
        )


if __name__ == "__main__":
    unittest.main()
