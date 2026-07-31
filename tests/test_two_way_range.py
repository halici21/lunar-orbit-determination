"""Focused physics and sensitivity tests for the M3 two-way range observable.

All fixtures here are SPICE-free: station states are injected through
synthetic ``TwoWayStationStateProvider`` objects and spacecraft histories are
exact linear (force-free) trajectories whose cubic-Hermite interpolation and
analytic STM ``[[I, (t-t0) I], [0, I]]`` are exact, so event-solution and
Jacobian comparisons are limited only by the solver tolerances and
finite-difference noise.
"""

from __future__ import annotations

import types
import unittest

import numpy as np

from lunar_od.radiometrics import (
    C_LIGHT_MPS,
    RangeRatePhysicsConfig,
    solve_two_way_light_time,
)
from lunar_od.two_way_range import (
    TwoWayEventConvergenceError,
    TwoWayEventHistoryError,
    TwoWayRangeConfig,
    TwoWayRangeEventSolution,
    TwoWayStationStateProvider,
    solve_two_way_range_events,
    two_way_range_event_sensitivity,
    two_way_range_from_solution,
)

C = C_LIGHT_MPS


def _constant_station_provider(position_m, velocity_mps=(0.0, 0.0, 0.0)):
    state = np.concatenate(
        [np.asarray(position_m, dtype=float), np.asarray(velocity_mps, dtype=float)]
    )

    return TwoWayStationStateProvider(
        state_fn=lambda t: state + np.concatenate([state[3:] * t, np.zeros(3)]),
        station_state_method="synthetic_constant_velocity",
        earth_ephemeris_method="synthetic",
    )


def _nan_station_provider():
    return TwoWayStationStateProvider(
        state_fn=lambda t: np.full(6, np.nan),
        station_state_method="synthetic_nan",
        earth_ephemeris_method="synthetic",
    )


def _linear_state_history(t_grid, r0, v0, t0):
    t_grid = np.asarray(t_grid, dtype=float)
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    states = np.zeros((t_grid.size, 6), dtype=float)
    dt = (t_grid - t0).reshape(-1, 1)
    states[:, :3] = r0 + dt * v0
    states[:, 3:6] = v0
    return states


def _linear_augmented_history(t_grid, r0, v0, t0):
    t_grid = np.asarray(t_grid, dtype=float)
    states = _linear_state_history(t_grid, r0, v0, t0)
    aug = np.zeros((t_grid.size, 42), dtype=float)
    aug[:, :6] = states
    for k, t in enumerate(t_grid):
        phi = np.eye(6)
        phi[:3, 3:] = (float(t) - t0) * np.eye(3)
        aug[k, 6:] = phi.reshape(-1, order="F")
    return aug


def _nominal_fixture(
    *,
    v0=(0.0, 0.0, 0.0),
    station_pos=(0.0, 0.0, 0.0),
    station_vel=(0.0, 0.0, 0.0),
    delay_s=0.0,
    r0=(3.844e8, 0.0, 0.0),
    t_start=-10.0,
    t_end=5.0,
    n_grid=61,
):
    t_grid = np.linspace(t_start, t_end, n_grid)
    t0 = float(t_grid[0])
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    states = _linear_state_history(t_grid, r0, v0, t0)
    aug = _linear_augmented_history(t_grid, r0, v0, t0)
    provider = _constant_station_provider(station_pos, station_vel)
    cfg = TwoWayRangeConfig(transponder_delay_s=delay_s)
    return t_grid, t0, r0, v0, states, aug, provider, cfg


class TwoWayRangeConfigTests(unittest.TestCase):
    def test_negative_transponder_delay_rejected(self):
        with self.assertRaises(ValueError):
            TwoWayRangeConfig(transponder_delay_s=-1e-6)

    def test_nonfinite_transponder_delay_rejected(self):
        with self.assertRaises(ValueError):
            TwoWayRangeConfig(transponder_delay_s=float("nan"))

    def test_unknown_convention_rejected(self):
        with self.assertRaises(ValueError):
            TwoWayRangeConfig(convention="two_way_range")

    def test_default_convention_is_delay_calibrated(self):
        self.assertEqual(TwoWayRangeConfig().convention, "delay_calibrated_half_round_trip")


class TwoWayRangeStaticGeometryTests(unittest.TestCase):
    def test_static_zero_delay_half_round_trip(self):
        t_grid, _, r0, _, states, _, provider, cfg = _nominal_fixture()
        rho = float(np.linalg.norm(r0))
        solution = solve_two_way_range_events(0.0, provider, t_grid, states, cfg)

        self.assertTrue(solution.converged)
        self.assertAlmostEqual(solution.round_trip_light_time_s, 2.0 * rho / C, delta=1e-12)
        self.assertAlmostEqual(solution.raw_half_round_trip_range_m, rho, delta=1e-6)
        self.assertAlmostEqual(
            solution.delay_calibrated_half_round_trip_range_m, rho, delta=1e-6
        )
        self.assertAlmostEqual(solution.uplink_range_m, rho, delta=1e-6)
        self.assertAlmostEqual(solution.downlink_range_m, rho, delta=1e-6)
        self.assertEqual(solution.t2u_s, solution.t2d_s)
        # Factor-of-two mutation guard: the full round-trip distance is 2*rho,
        # so reporting it (or half of one leg) would fail the equality above
        # by rho or rho/2 respectively.
        self.assertGreater(abs(2.0 * rho - solution.raw_half_round_trip_range_m), 1e7)

    def test_static_nonzero_delay_raw_and_calibrated_conventions(self):
        delay = 0.5
        t_grid, _, r0, _, states, _, provider, cfg = _nominal_fixture(delay_s=delay)
        rho = float(np.linalg.norm(r0))
        solution = solve_two_way_range_events(0.0, provider, t_grid, states, cfg)

        self.assertAlmostEqual(
            solution.round_trip_light_time_s, 2.0 * rho / C + delay, delta=1e-12
        )
        self.assertAlmostEqual(
            solution.raw_half_round_trip_range_m, rho + 0.5 * C * delay, delta=1e-6
        )
        self.assertAlmostEqual(
            solution.delay_calibrated_half_round_trip_range_m, rho, delta=1e-6
        )
        # Delay-subtracted-twice mutation guard: raw minus calibrated must be
        # exactly one half-delay range equivalent.
        self.assertAlmostEqual(
            solution.raw_half_round_trip_range_m
            - solution.delay_calibrated_half_round_trip_range_m,
            0.5 * C * delay,
            delta=1e-9,
        )
        self.assertAlmostEqual(solution.t2d_s - solution.t2u_s, delay, delta=1e-12)

    def test_reported_convention_selection(self):
        delay = 0.25
        t_grid, _, r0, _, states, _, provider, _ = _nominal_fixture(delay_s=delay)
        raw_cfg = TwoWayRangeConfig(transponder_delay_s=delay, convention="raw_half_round_trip")
        cal_cfg = TwoWayRangeConfig(
            transponder_delay_s=delay, convention="delay_calibrated_half_round_trip"
        )
        solution = solve_two_way_range_events(0.0, provider, t_grid, states, raw_cfg)
        self.assertEqual(
            two_way_range_from_solution(solution, raw_cfg),
            solution.raw_half_round_trip_range_m,
        )
        self.assertEqual(
            two_way_range_from_solution(solution, cal_cfg),
            solution.delay_calibrated_half_round_trip_range_m,
        )


class TwoWayRangeConstantVelocityTests(unittest.TestCase):
    def _analytic_events(self, x0_m, v_mps, t0, t3, delay_s):
        """1-D closed form for spacecraft x(t) = x0 + v (t - t0), station at 0."""
        t2d = (C * t3 - x0_m + v_mps * t0) / (C + v_mps)
        t2u = t2d - delay_s
        p_t2u = x0_m + v_mps * (t2u - t0)
        t1 = t2u - p_t2u / C
        return t1, t2u, t2d

    def _run_case(self, v_mps, delay_s):
        t_grid, t0, r0, _, states, _, provider, cfg = _nominal_fixture(
            v0=(v_mps, 0.0, 0.0), delay_s=delay_s
        )
        t3 = 0.0
        solution = solve_two_way_range_events(t3, provider, t_grid, states, cfg)
        t1_ref, t2u_ref, t2d_ref = self._analytic_events(float(r0[0]), v_mps, t0, t3, delay_s)

        self.assertAlmostEqual(solution.t2d_s, t2d_ref, delta=5e-12)
        self.assertAlmostEqual(solution.t2u_s, t2u_ref, delta=5e-12)
        self.assertAlmostEqual(solution.t1_s, t1_ref, delta=5e-12)
        self.assertAlmostEqual(
            solution.raw_half_round_trip_range_m, 0.5 * C * (t3 - t1_ref), delta=2e-3
        )
        self.assertLessEqual(solution.uplink_equation_residual_s, cfg.equation_tolerance_s)
        self.assertLessEqual(solution.downlink_equation_residual_s, cfg.equation_tolerance_s)
        return solution

    def _range_at_receive(self, v_mps):
        # Initial state is defined at t0 = grid start (-10 s); the receive
        # tag is t3 = 0, so the instantaneous range there is |r0 + v (t3-t0)|.
        return 3.844e8 + v_mps * 10.0

    def test_receding_spacecraft(self):
        solution = self._run_case(+1600.0, 0.0)
        rho_t3 = self._range_at_receive(+1600.0)
        # Receding: both event legs sample earlier, closer positions, so the
        # round-trip light time is below the instantaneous 2 rho(t3) / c.
        self.assertLess(solution.round_trip_light_time_s, 2.0 * rho_t3 / C)

    def test_approaching_spacecraft(self):
        solution = self._run_case(-1600.0, 0.0)
        rho_t3 = self._range_at_receive(-1600.0)
        self.assertGreater(solution.round_trip_light_time_s, 2.0 * rho_t3 / C)

    def test_nonzero_delay_uses_distinct_uplink_receive_state(self):
        delay = 0.5
        v = 1600.0
        solution = self._run_case(v, delay)
        t_grid, t0, r0, _, states, _, provider, cfg = _nominal_fixture(
            v0=(v, 0.0, 0.0), delay_s=delay
        )
        # Uplink leg must use the spacecraft position at t2u, not t2d.
        p_t2u = float(r0[0]) + v * (solution.t2u_s - t0)
        p_t2d = float(r0[0]) + v * (solution.t2d_s - t0)
        station_tx = provider.state(solution.t1_s)
        self.assertAlmostEqual(
            solution.uplink_range_m, abs(p_t2u - station_tx[0]), delta=1e-6
        )
        self.assertGreater(abs(p_t2d - p_t2u), 700.0)  # v * delay = 800 m
        self.assertNotAlmostEqual(
            solution.uplink_range_m, abs(p_t2d - station_tx[0]), delta=100.0
        )


class TwoWayRangeMovingStationTests(unittest.TestCase):
    def test_station_velocity_changes_observable(self):
        t_grid, _, _, _, states, _, static_provider, cfg = _nominal_fixture()
        moving_provider = _constant_station_provider(
            (0.0, 0.0, 0.0), (400.0, 0.0, 0.0)
        )
        static_solution = solve_two_way_range_events(0.0, static_provider, t_grid, states, cfg)
        moving_solution = solve_two_way_range_events(0.0, moving_provider, t_grid, states, cfg)

        # The uplink station position at t1 differs by ~|v_st| * tau_rt from
        # the receive position, so the observable must shift measurably.
        delta = abs(
            moving_solution.raw_half_round_trip_range_m
            - static_solution.raw_half_round_trip_range_m
        )
        self.assertGreater(delta, 100.0)
        self.assertLessEqual(
            moving_solution.uplink_equation_residual_s, cfg.equation_tolerance_s
        )


class TwoWayRangeEventOrderingAndDomainTests(unittest.TestCase):
    def test_event_ordering_zero_and_positive_delay(self):
        for delay in (0.0, 1e-6, 0.5):
            t_grid, _, _, _, states, _, provider, _ = _nominal_fixture(delay_s=delay)
            cfg = TwoWayRangeConfig(transponder_delay_s=delay)
            solution = solve_two_way_range_events(0.0, provider, t_grid, states, cfg)
            self.assertLess(solution.t1_s, solution.t2u_s)
            self.assertLessEqual(solution.t2u_s, solution.t2d_s)
            self.assertLess(solution.t2d_s, solution.t3_s)
            self.assertAlmostEqual(
                solution.t2d_s - solution.t2u_s, delay, delta=1e-12
            )

    def test_history_error_when_receive_tag_too_early(self):
        # First receive tag at the grid start: t2d ~ t3 - 1.28 s falls before
        # the history and must raise a controlled error, not extrapolate.
        t_grid, _, _, _, states, _, provider, cfg = _nominal_fixture()
        with self.assertRaises(TwoWayEventHistoryError) as ctx:
            solve_two_way_range_events(float(t_grid[0]), provider, t_grid, states, cfg)
        message = str(ctx.exception)
        self.assertIn("pre-roll", message)
        self.assertIn("t2d", message)

    def test_history_error_reports_required_preroll_for_uplink_receive(self):
        # Position the receive tag so that t2d stays inside the history but
        # the delayed t2u = t2d - delay falls before the grid start.
        delay = 0.8
        t_grid, _, _, _, states, _, provider, _ = _nominal_fixture(delay_s=delay)
        cfg = TwoWayRangeConfig(transponder_delay_s=delay)
        t3 = float(t_grid[0]) + 1.30 + delay / 2.0
        with self.assertRaises(TwoWayEventHistoryError):
            solve_two_way_range_events(t3, provider, t_grid, states, cfg)

    def test_nonfinite_receive_time_rejected(self):
        t_grid, _, _, _, states, _, provider, cfg = _nominal_fixture()
        with self.assertRaises(TwoWayEventHistoryError):
            solve_two_way_range_events(float("nan"), provider, t_grid, states, cfg)

    def test_nonfinite_station_state_rejected(self):
        t_grid, _, _, _, states, _, _, cfg = _nominal_fixture()
        with self.assertRaises(TwoWayEventHistoryError):
            solve_two_way_range_events(0.0, _nan_station_provider(), t_grid, states, cfg)

    def test_insufficient_iterations_raise_convergence_error(self):
        t_grid, _, _, _, states, _, provider, _ = _nominal_fixture(v0=(1600.0, 0.0, 0.0))
        cfg = TwoWayRangeConfig(max_iter=1)
        with self.assertRaises(TwoWayEventConvergenceError) as ctx:
            solve_two_way_range_events(0.0, provider, t_grid, states, cfg)
        self.assertIn("equation", str(ctx.exception).lower())


class TwoWayRangeLegacySolverConsistencyTests(unittest.TestCase):
    def test_zero_delay_round_trip_matches_legacy_counted_doppler_solver(self):
        """Diagnostic-only: the legacy solver is unchanged; with constant
        identity transforms and zero Earth ephemeris both solvers see the same
        geometry, so their zero-delay round-trip light times must agree."""
        t_grid, t0, r0, _, states, _, provider, cfg = _nominal_fixture(
            v0=(1600.0, 0.0, 0.0)
        )
        station = types.SimpleNamespace(r_ecef_m=np.zeros(3))
        n = t_grid.size
        xforms = np.tile(np.eye(6), (n, 1, 1))
        earth_pos = np.zeros((n, 3))
        earth_vel = np.zeros((n, 3))
        legacy = solve_two_way_light_time(
            0.0,
            station,
            t_grid,
            states,
            earth_pos,
            earth_vel,
            xforms,
            RangeRatePhysicsConfig(light_time_tolerance_s=1e-12, light_time_max_iter=25),
        )
        new = solve_two_way_range_events(0.0, provider, t_grid, states, cfg)
        self.assertTrue(legacy.converged)
        self.assertAlmostEqual(
            new.round_trip_light_time_s, legacy.round_trip_light_time_s, delta=1e-11
        )
        self.assertAlmostEqual(new.t1_s, legacy.transmit_time_s, delta=1e-11)
        self.assertAlmostEqual(new.t2d_s, legacy.transponder_time_s, delta=1e-11)


def _solve_for_initial_state(x0, t_grid, t0, provider, cfg, t3):
    states = _linear_state_history(t_grid, x0[:3], x0[3:], t0)
    return solve_two_way_range_events(t3, provider, t_grid, states, cfg)


class TwoWayRangeSensitivityTests(unittest.TestCase):
    def _fixture(self, *, v0=(1600.0, -300.0, 250.0), delay_s=0.0, station_vel=(0.0, 0.0, 0.0)):
        t_grid, t0, r0, v0_arr, states, aug, provider, cfg = _nominal_fixture(
            v0=v0, delay_s=delay_s, station_vel=station_vel, r0=(3.844e8, 1.2e7, -6.0e6)
        )
        t3 = 0.0
        solution = solve_two_way_range_events(t3, provider, t_grid, states, cfg)
        sensitivity = two_way_range_event_sensitivity(solution, provider, t_grid, aug, cfg)
        x0 = np.concatenate([r0, v0_arr])
        return t_grid, t0, x0, provider, cfg, t3, solution, sensitivity

    def _fd_events_and_range(self, x0, col, step, t_grid, t0, provider, cfg, t3):
        perturb = np.zeros(6)
        perturb[col] = step
        sol_p = _solve_for_initial_state(x0 + perturb, t_grid, t0, provider, cfg, t3)
        sol_m = _solve_for_initial_state(x0 - perturb, t_grid, t0, provider, cfg, t3)
        inv = 1.0 / (2.0 * step)
        events = np.array(
            [
                (sol_p.t1_s - sol_m.t1_s) * inv,
                (sol_p.t2u_s - sol_m.t2u_s) * inv,
                (sol_p.t2d_s - sol_m.t2d_s) * inv,
            ]
        )
        range_fd_raw = (
            sol_p.raw_half_round_trip_range_m - sol_m.raw_half_round_trip_range_m
        ) * inv
        range_fd_cal = (
            sol_p.delay_calibrated_half_round_trip_range_m
            - sol_m.delay_calibrated_half_round_trip_range_m
        ) * inv
        return events, range_fd_raw, range_fd_cal

    def test_event_time_sensitivities_match_finite_difference(self):
        t_grid, t0, x0, provider, cfg, t3, _, sens = self._fixture(delay_s=0.25)
        steps = [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]
        for col, step in enumerate(steps):
            events_fd, _, _ = self._fd_events_and_range(
                x0, col, step, t_grid, t0, provider, cfg, t3
            )
            analytic = sens.event_time_sensitivity[:, col]
            for row in range(3):
                scale = max(abs(analytic[row]), 1e-12)
                self.assertLess(
                    abs(analytic[row] - events_fd[row]) / scale,
                    2e-5,
                    msg=f"event row {row}, state column {col}",
                )

    def test_range_jacobian_matches_finite_difference_with_step_sweep(self):
        t_grid, t0, x0, provider, cfg, t3, _, sens = self._fixture(delay_s=0.25)
        h = sens.two_way_range_jacobian_dx0
        pos_steps = (0.1, 1.0, 10.0)
        vel_steps = (0.01, 0.1, 1.0)
        worst_rel = 0.0
        for col in range(6):
            steps = pos_steps if col < 3 else vel_steps
            best = np.inf
            for step in steps:
                _, fd_raw, _ = self._fd_events_and_range(
                    x0, col, step, t_grid, t0, provider, cfg, t3
                )
                scale = max(abs(h[col]), 1e-9)
                best = min(best, abs(h[col] - fd_raw) / scale)
            worst_rel = max(worst_rel, best)
            self.assertLess(best, 1e-6, msg=f"state column {col}")
        # Position columns are O(1) (unit-LOS projections); velocity columns
        # carry seconds units and must be near u_hat * (t2 - t0).
        self.assertLess(worst_rel, 1e-6)

    def test_raw_and_calibrated_jacobians_are_identical_for_fixed_delay(self):
        t_grid, t0, x0, provider, cfg, t3, _, sens = self._fixture(delay_s=0.5)
        for col, step in ((0, 1.0), (3, 0.1)):
            _, fd_raw, fd_cal = self._fd_events_and_range(
                x0, col, step, t_grid, t0, provider, cfg, t3
            )
            self.assertAlmostEqual(fd_raw, fd_cal, delta=5e-8 * max(1.0, abs(fd_raw)))
            scale = max(abs(sens.two_way_range_jacobian_dx0[col]), 1e-9)
            self.assertLess(abs(sens.two_way_range_jacobian_dx0[col] - fd_cal) / scale, 1e-6)

    def test_static_geometry_analytic_derivative(self):
        """Static spacecraft, static station, zero delay: dR/dr0 = u_hat and
        dR/dv0 = u_hat * (t2 - t0)."""
        t_grid, t0, r0, _, states, aug, provider, cfg = _nominal_fixture(
            r0=(3.0e8, 2.0e8, -1.0e8)
        )
        solution = solve_two_way_range_events(0.0, provider, t_grid, states, cfg)
        sens = two_way_range_event_sensitivity(solution, provider, t_grid, aug, cfg)
        u_hat = np.asarray(r0, dtype=float) / np.linalg.norm(r0)
        np.testing.assert_allclose(sens.two_way_range_jacobian_dx0[:3], u_hat, rtol=0, atol=1e-9)
        dt2 = solution.t2u_s - t0
        np.testing.assert_allclose(
            sens.two_way_range_jacobian_dx0[3:], u_hat * dt2, rtol=1e-9, atol=1e-12
        )

    def test_scaled_event_matrix_is_well_conditioned_and_reported(self):
        _, _, _, _, _, _, _, sens = self._fixture()
        self.assertTrue(np.all(np.isfinite(sens.scaled_event_matrix)))
        self.assertTrue(np.isfinite(sens.event_matrix_condition_number))
        # Uniform-second scaling keeps the nominal event matrix O(1).
        self.assertLess(sens.event_matrix_condition_number, 10.0)
        self.assertFalse(sens.transponder_delay_is_solve_for)

    def test_no_double_stm_mutation_is_detected(self):
        """Mapping the initial-state row through the receive-node STM again
        must produce a measurably different row (guards double-STM misuse)."""
        t_grid, t0, x0, provider, cfg, t3, solution, sens = self._fixture()
        phi_end = np.eye(6)
        phi_end[:3, 3:] = (t3 - t0) * np.eye(3)
        mutated = sens.two_way_range_jacobian_dx0 @ phi_end
        self.assertGreater(
            np.linalg.norm(mutated - sens.two_way_range_jacobian_dx0),
            1.0,
        )

    def test_sensitivity_uses_separate_uplink_and_downlink_states(self):
        """With a long transponder delay the uplink/downlink unit LOS differ."""
        _, _, _, _, _, _, _, sens = self._fixture(delay_s=0.5, v0=(1600.0, -300.0, 250.0))
        self.assertGreater(
            float(np.linalg.norm(sens.uplink_unit_los - sens.downlink_unit_los)), 0.0
        )

    def test_station_velocity_enters_event_matrix(self):
        """Zeroing the station velocity (a pxform-style position-only station
        state) must shift the event matrix away from the FD-validated one."""
        station_vel = (40000.0, 0.0, 0.0)  # exaggerated to expose the v_st/c term
        t_grid, t0, x0, provider, cfg, t3, solution, sens = self._fixture(
            station_vel=station_vel
        )
        zero_vel_provider = TwoWayStationStateProvider(
            state_fn=lambda t: np.concatenate(
                [provider.state(t)[:3], np.zeros(3)]
            ),
            station_state_method="mutated_position_only",
            earth_ephemeris_method="synthetic",
        )
        aug = _linear_augmented_history(t_grid, x0[:3], x0[3:], t0)
        mutated = two_way_range_event_sensitivity(
            solution, zero_vel_provider, t_grid, aug, cfg
        )
        # G_y[0, 0] = -1 + u_u . v_st / c must lose its velocity term.
        self.assertGreater(
            abs(mutated.scaled_event_matrix[0, 0] - sens.scaled_event_matrix[0, 0]),
            1e-5,
        )
        rel = np.abs(
            mutated.two_way_range_jacobian_dx0 - sens.two_way_range_jacobian_dx0
        ) / np.maximum(np.abs(sens.two_way_range_jacobian_dx0), 1e-9)
        self.assertGreater(float(np.max(rel)), 5e-5)


class TwoWayRangeIndependentRootReferenceTests(unittest.TestCase):
    def test_production_solver_matches_independent_scipy_root(self):
        """Structurally independent reference: solve the simultaneous 3x3
        event system G(y) = 0 with scipy.optimize.root instead of the nested
        production fixed-point iteration."""
        from scipy.optimize import root

        delay = 0.25
        t_grid, t0, r0, v0, states, _, provider, _ = _nominal_fixture(
            v0=(1600.0, -300.0, 250.0), delay_s=delay, r0=(3.844e8, 1.2e7, -6.0e6)
        )
        cfg = TwoWayRangeConfig(transponder_delay_s=delay)
        t3 = 0.0
        solution = solve_two_way_range_events(t3, provider, t_grid, states, cfg)

        from lunar_od.radiometrics import interp_state_history

        def event_equations(y):
            t1, t2u, t2d = y
            sc_t2u = interp_state_history(t_grid, states, t2u)
            sc_t2d = interp_state_history(t_grid, states, t2d)
            st_t1 = provider.state(t1)
            st_t3 = provider.state(t3)
            g_u = t2u - t1 - np.linalg.norm(sc_t2u[:3] - st_t1[:3]) / C
            g_d = t3 - t2d - np.linalg.norm(sc_t2d[:3] - st_t3[:3]) / C
            g_tr = t2d - t2u - delay
            return [g_u, g_d, g_tr]

        guess = np.array([t3 - 3.0, t3 - 1.5 - delay, t3 - 1.5])
        result = root(event_equations, guess, method="hybr", tol=1e-13)
        self.assertTrue(result.success)
        t1_ref, t2u_ref, t2d_ref = result.x

        self.assertAlmostEqual(solution.t1_s, t1_ref, delta=1e-10)
        self.assertAlmostEqual(solution.t2u_s, t2u_ref, delta=1e-10)
        self.assertAlmostEqual(solution.t2d_s, t2d_ref, delta=1e-10)
        self.assertAlmostEqual(
            solution.raw_half_round_trip_range_m,
            0.5 * C * (t3 - t1_ref),
            delta=0.05,
        )


if __name__ == "__main__":
    unittest.main()
