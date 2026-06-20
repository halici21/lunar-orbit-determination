import json
import unittest
from pathlib import Path

import numpy as np

from lunar_od import (
    C_LIGHT_MPS,
    MoonCenteredEphemeris,
    PassGeometry,
    RangeRatePhysicsConfig,
    compute_position_residuals,
    compute_position_residuals_analytic,
    compute_range_rate_residuals,
    compute_range_rate_residuals_analytic,
    generate_position_measurements,
    generate_range_rate_measurements,
    instantaneous_geometric_range_rate,
    load_spice_kernels,
    range_rate_stations,
    solve_one_way_light_time,
    solve_two_way_light_time,
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

    def _generate_position_clean(self, fx, *, apply_light_time, apply_stellar_aberration=False):
        import spiceypy as spice

        try:
            load_spice_kernels()
        except FileNotFoundError:
            self.skipTest("SPICE kernels not available.")
        try:
            _, pass_geo, clean_obs = generate_position_measurements(
                fx["t_pass"], fx["state_history"], fx["stations"], fx["vis_mask"],
                fx["earth_pos"], fx["earth_vel"], fx["et0"],
                noise=False, apply_light_time=apply_light_time,
                apply_stellar_aberration=apply_stellar_aberration,
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

    def test_position_light_time_creates_physical_correction(self):
        fx = self._position_light_time_fixture()
        _, clean_inst = self._generate_position_clean(fx, apply_light_time=False)
        _, clean_lt = self._generate_position_clean(fx, apply_light_time=True)
        self.assertEqual(clean_inst.shape, clean_lt.shape)
        range_diff = np.abs(clean_lt[:, 1] - clean_inst[:, 1])
        self.assertGreater(float(np.max(range_diff)), 100.0)

    def test_position_light_time_model_mismatch_is_biased(self):
        import dataclasses

        fx = self._position_light_time_fixture()
        pass_geo_lt, clean_lt = self._generate_position_clean(fx, apply_light_time=True)
        pass_geo_inst = dataclasses.replace(pass_geo_lt, apply_light_time=False)
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


def _constant_acceleration_states(t_grid):
    t = np.asarray(t_grid, dtype=float)
    states = np.zeros((t.size, 6), dtype=float)
    initial_range_m = 100.0e6
    speed_mps = 125.0
    acceleration_mps2 = 0.25
    states[:, 0] = initial_range_m + speed_mps * t + 0.5 * acceleration_mps2 * t**2
    states[:, 3] = speed_mps + acceleration_mps2 * t
    return states


if __name__ == "__main__":
    unittest.main()
