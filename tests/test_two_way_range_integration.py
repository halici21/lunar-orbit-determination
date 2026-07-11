"""Kernel-gated integration tests for the M3 two-way range observable.

Covers the production Option-A frame policy (exact event-epoch ``sxform``),
measurement generation and residual closure, the BLS-LM/SRIF integration and
residual-consistency check (residual RMS collapses at the estimate; the
truth-state error is reported alongside and is NOT claimed to recover), the shared
(N, 6) initial-state row across estimators/observability, UKF rejection,
scenario-config validation, legacy counted-Doppler zero-delay consistency,
and the exact-versus-interpolated frame / Earth-ephemeris diagnostics
(printed under ``pytest -s`` for the M3 report).

Skips cleanly when spiceypy or the kernel set is unavailable.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from lunar_od.config import Station
from lunar_od.radiometrics import (
    C_LIGHT_MPS,
    RangeRatePhysicsConfig,
    _interp_matrix,
    _interp_state,
    solve_two_way_light_time,
)
from lunar_od.two_way_range import (
    TwoWayRangeConfig,
    TwoWayStationStateProvider,
    compute_two_way_range_residuals,
    generate_two_way_range_measurements,
    make_exact_sxform_station_state_provider,
    solve_two_way_range_events,
    two_way_range_event_sensitivity,
    two_way_range_nominal_and_initial_jacobian,
)


def _spice_status() -> tuple[bool, str]:
    try:
        import spiceypy  # noqa: F401
    except Exception as exc:
        return False, f"spiceypy unavailable: {exc}"
    try:
        from lunar_od.spice_loader import required_kernel_paths

        required_kernel_paths()
    except FileNotFoundError as exc:
        return False, f"SPICE kernels unavailable: {exc}"
    return True, ""


_SPICE_OK, _SKIP_REASON = _spice_status()
_ET_UTC = "2027-03-02 00:00:00"

_MU_MOON = 4902.800066e9

_STATION = Station(
    name="Goldstone DSN",
    lat_deg=35.30,
    lon_deg=-116.81,
    alt_m=969.67,
    color_rgb=(0.85, 0.325, 0.098),
    sigma_range_m=5.0,
    sigma_angle_rad=math.radians(0.001),
)


def setUpModule():  # noqa: N802
    if not _SPICE_OK:
        raise unittest.SkipTest(_SKIP_REASON)
    from lunar_od.spice_loader import load_spice_kernels

    load_spice_kernels()


def _et0() -> float:
    import spiceypy as spice

    return float(spice.str2et(_ET_UTC))


def _earth_state_mci(et: float) -> np.ndarray:
    import spiceypy as spice

    state, _ = spice.spkezr("EARTH", float(et), "J2000", "NONE", "MOON")
    return np.asarray(state, dtype=float) * 1e3  # km, km/s -> m, m/s


def _truth_setup(step_s: float = 60.0, duration_s: float = 1200.0):
    """Propagated LLO truth over a short pass with real Earth ephemeris."""
    from lunar_od.dynamics import propagate_augmented_state

    et0 = _et0()
    t_pass = np.arange(0.0, duration_s + 0.5 * step_s, step_s)
    earth_states = np.array([_earth_state_mci(et0 + t) for t in t_pass])
    earth_pos = earth_states[:, :3]
    earth_vel = earth_states[:, 3:]

    def get_earth_pos(t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        return np.array(
            [_interp_state(t_pass, earth_states, float(ti))[:3] for ti in t_arr]
        )

    def get_earth_vel(t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        return np.array(
            [_interp_state(t_pass, earth_states, float(ti))[3:] for ti in t_arr]
        )

    def get_sun_pos(t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        return np.tile(np.array([149.6e9, 0.0, 0.0]), (t_arr.size, 1))

    r0 = 1737.4e3 + 100e3
    x_true0 = np.array(
        [r0, 30e3, -20e3, -15.0, math.sqrt(_MU_MOON / r0), 4.0], dtype=float
    )
    x_aug0 = np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")])
    x_aug_truth = propagate_augmented_state(
        t_pass, x_aug0, _MU_MOON, 0.0, 0.0, get_earth_pos, get_sun_pos,
        rtol=1e-12, atol=1e-13,
    )
    return (
        et0,
        t_pass,
        earth_pos,
        earth_vel,
        get_earth_pos,
        get_earth_vel,
        get_sun_pos,
        x_true0,
        x_aug_truth,
    )


@unittest.skipUnless(_SPICE_OK, _SKIP_REASON)
class TwoWayRangeGenerationTests(unittest.TestCase):
    def test_generation_drops_pregrid_events_and_records_metadata(self):
        et0, t_pass, earth_pos, earth_vel, gep, gev, _, _, x_aug = _truth_setup()
        vis = np.ones((t_pass.size, 1), dtype=bool)
        obs, pass_geo = generate_two_way_range_measurements(
            t_pass, x_aug[:, :6], (_STATION,), vis, gep, gev, et0, noise=False
        )
        # The first receive tag (t3 = grid start) cannot host t2d inside the
        # history and must be dropped, not extrapolated.
        self.assertEqual(obs.shape[0], t_pass.size - 1)
        meta = pass_geo.measurement_metadata
        self.assertEqual(meta["dropped_measurements"], 1)
        self.assertIn("pre-roll", meta["dropped_measurement_reason"])
        self.assertEqual(meta["measurement_type"], "two_way_range")
        self.assertEqual(meta["station_state_evaluation"], "exact_event_epoch_sxform")
        self.assertEqual(meta["two_way_range_convention"], "delay_calibrated_half_round_trip")
        self.assertEqual(meta["stm_application_count"], "one")
        self.assertFalse(meta["ukf_supported"])
        self.assertEqual(pass_geo.et0_s, et0)

    def test_noise_free_residual_closure_at_truth(self):
        et0, t_pass, _, _, gep, gev, _, _, x_aug = _truth_setup()
        vis = np.ones((t_pass.size, 1), dtype=bool)
        obs, pass_geo = generate_two_way_range_measurements(
            t_pass, x_aug[:, :6], (_STATION,), vis, gep, gev, et0, noise=False
        )
        residuals, h_meas = compute_two_way_range_residuals(x_aug[:, :6], obs, pass_geo)
        self.assertTrue(np.all(np.isfinite(h_meas)))
        np.testing.assert_array_equal(residuals, np.zeros(obs.shape[0]))

    def test_two_way_range_magnitude_is_station_spacecraft_distance(self):
        """Factor-1/2 guard with real geometry: the reported range must be
        Earth-Moon scale (one-way), not double it."""
        et0, t_pass, _, _, gep, gev, _, _, x_aug = _truth_setup()
        vis = np.ones((t_pass.size, 1), dtype=bool)
        obs, _ = generate_two_way_range_measurements(
            t_pass, x_aug[:, :6], (_STATION,), vis, gep, gev, et0, noise=False
        )
        ranges = obs[:, 1]
        self.assertTrue(np.all(ranges > 3.0e8))
        self.assertTrue(np.all(ranges < 4.5e8))


@unittest.skipUnless(_SPICE_OK, _SKIP_REASON)
class TwoWayRangeEstimatorIntegrationTests(unittest.TestCase):
    def _observations(self):
        (et0, t_pass, _, _, gep, gev, gsp, x_true0, x_aug) = _truth_setup()
        vis = np.ones((t_pass.size, 1), dtype=bool)
        obs, pass_geo = generate_two_way_range_measurements(
            t_pass, x_aug[:, :6], (_STATION,), vis, gep, gev, et0, noise=False
        )
        return et0, t_pass, gep, gev, gsp, x_true0, x_aug, obs, pass_geo

    def _residual_rms(self, t_pass, obs, x0, pass_geo, gep, gsp):
        from lunar_od.dynamics import propagate_state

        x_hist = propagate_state(
            t_pass, x0, _MU_MOON, 0.0, 0.0, gep, gsp, rtol=1e-12, atol=1e-13
        )
        residuals, _ = compute_two_way_range_residuals(x_hist, obs, pass_geo)
        return float(np.sqrt(np.mean(residuals**2)))

    def test_bls_lm_and_srif_integration_and_residual_consistency_with_shared_rows(self):
        """Integration and residual-consistency test (NOT a truth-recovery
        claim): a single short two-way-range pass is weakly observable
        transverse to the line of sight, so the residual RMS collapses while
        the truth-state error may remain near its initial level; both are
        printed for the M3 report."""
        from lunar_od.estimators import (
            estimate_two_way_range_bls_lm,
            estimate_two_way_range_srif,
        )
        from lunar_od.observability import build_initial_state_jacobian

        et0, t_pass, gep, gev, gsp, x_true0, x_aug, obs, pass_geo = self._observations()
        x_guess = x_true0 + np.array([25.0, -20.0, 12.0, 0.01, -0.008, 0.004])
        initial_rms = self._residual_rms(t_pass, obs, x_guess, pass_geo, gep, gsp)

        results = {}
        for name, estimator in (
            ("bls_lm", estimate_two_way_range_bls_lm),
            ("srif", estimate_two_way_range_srif),
        ):
            x_est, stop_reason, stats = estimator(
                t_pass,
                obs,
                x_guess,
                pass_geo,
                _MU_MOON,
                0.0,
                0.0,
                gep,
                gsp,
                max_iter=8,
                rtol=1e-12,
                atol=1e-13,
                return_posterior=True,
            )
            final_rms = self._residual_rms(t_pass, obs, x_est, pass_geo, gep, gsp)
            self.assertIn(stop_reason, {"Converged", "J-Stab", "MaxIter"}, msg=name)
            self.assertTrue(np.isfinite(stats.condition_number), msg=name)
            self.assertLess(final_rms, 0.05 * initial_rms, msg=name)
            self.assertLess(
                np.linalg.norm(x_est[:3] - x_true0[:3]),
                np.linalg.norm(x_guess[:3] - x_true0[:3]),
                msg=name,
            )
            self.assertIsNotNone(stats.posterior_information, msg=name)
            results[name] = x_est
            print(
                f"[two-way {name}] stop={stop_reason} iters={stats.iterations} "
                f"final cost={stats.final_cost:.3e} "
                f"residual rms {initial_rms:.3f} -> {final_rms:.3e} m | "
                f"truth pos err {np.linalg.norm(x_guess[:3] - x_true0[:3]):.3e}"
                f" -> {np.linalg.norm(x_est[:3] - x_true0[:3]):.3e} m | "
                f"truth vel err {np.linalg.norm(x_guess[3:] - x_true0[3:]):.3e}"
                f" -> {np.linalg.norm(x_est[3:] - x_true0[3:]):.3e} m/s"
            )

        # Shared-row contract: the estimator-side helper and the
        # observability path must produce the identical (N, 6) matrix.
        from lunar_od.dynamics import propagate_augmented_state

        x_aug_guess = propagate_augmented_state(
            t_pass,
            np.concatenate([x_guess, np.eye(6).reshape(-1, order="F")]),
            _MU_MOON, 0.0, 0.0, gep, gsp, rtol=1e-12, atol=1e-13,
        )
        _, h_estimator = two_way_range_nominal_and_initial_jacobian(
            obs, pass_geo, x_aug_guess
        )
        h_observability = build_initial_state_jacobian(
            "two_way_range", t_pass, obs, pass_geo, x_guess,
            _MU_MOON, 0.0, 0.0, gep, gsp, rtol=1e-12, atol=1e-13,
        )
        self.assertEqual(h_estimator.shape, (obs.shape[0], 6))
        np.testing.assert_allclose(h_observability, h_estimator, rtol=0.0, atol=1e-10)

    def test_ukf_rejects_two_way_range(self):
        from lunar_od.filters import run_lunar_ukf
        from lunar_od.scenarios import run_batch_arc_sequence

        _, t_pass, _, _, _, x_true0, x_aug, obs, pass_geo = self._observations()
        with self.assertRaises(ValueError) as ctx:
            run_lunar_ukf(
                t_pass, obs, x_true0, np.eye(6), pass_geo,
                _MU_MOON, 0.0, 0.0,
                lambda t: np.zeros((np.size(np.atleast_1d(t)), 3)),
                lambda t: np.zeros((np.size(np.atleast_1d(t)), 3)),
            )
        self.assertIn("two_way_range", str(ctx.exception))
        with self.assertRaises(ValueError):
            run_batch_arc_sequence(
                (), "two_way_range", "cold", "ukf",
                _MU_MOON, 0.0, 0.0,
                lambda t: np.zeros((np.size(np.atleast_1d(t)), 3)),
                lambda t: np.zeros((np.size(np.atleast_1d(t)), 3)),
            )


@unittest.skipUnless(_SPICE_OK, _SKIP_REASON)
class TwoWayRangeFramePolicyTests(unittest.TestCase):
    """Directive 18: exact event-epoch sxform versus interpolated grids, and
    the separately reported Earth-center ephemeris interpolation error."""

    def _production_case(self, delay_s: float = 0.0):
        et0, t_pass, earth_pos, earth_vel, gep, gev, _, _, x_aug = _truth_setup()
        cfg = TwoWayRangeConfig(transponder_delay_s=delay_s)
        provider = make_exact_sxform_station_state_provider(
            _STATION, et0, t_pass, earth_pos, earth_vel
        )
        t3 = float(t_pass[t_pass.size // 2])
        solution = solve_two_way_range_events(t3, provider, t_pass, x_aug[:, :6], cfg)
        sens = two_way_range_event_sensitivity(solution, provider, t_pass, x_aug, cfg)
        return et0, t_pass, earth_pos, earth_vel, x_aug, cfg, provider, t3, solution, sens

    def _interpolated_provider(self, et0, t_pass, earth_pos, earth_vel, grid_s):
        import spiceypy as spice

        earth_states = np.hstack([earth_pos, earth_vel])
        grid = np.arange(t_pass[0] - 2.0 * grid_s, t_pass[-1] + 2.5 * grid_s, grid_s)
        xforms = np.array(
            [spice.sxform("J2000", "ITRF93", et0 + float(t)) for t in grid]
        )
        station_ecef_state = np.concatenate([_STATION.r_ecef_m, np.zeros(3)])

        def state_fn(t):
            earth_state = _interp_state(t_pass, earth_states, t)
            xform = _interp_matrix(grid, xforms, float(t))
            return earth_state + np.linalg.solve(xform, station_ecef_state)

        return TwoWayStationStateProvider(
            state_fn=state_fn,
            station_state_method=f"linear_interpolated_sxform_{grid_s:g}s",
            earth_ephemeris_method="cubic_hermite_grid_interpolation",
        )

    def test_exact_vs_interpolated_transform_grids(self):
        (et0, t_pass, earth_pos, earth_vel, x_aug, cfg, provider, t3,
         solution, sens) = self._production_case()
        print("[frame policy] production station_state_method =",
              solution.station_state_method)
        self.assertEqual(solution.station_state_method, "exact_event_epoch_sxform")
        for grid_s in (10.0, 30.0, 60.0):
            interp_provider = self._interpolated_provider(
                et0, t_pass, earth_pos, earth_vel, grid_s
            )
            interp_solution = solve_two_way_range_events(
                t3, interp_provider, t_pass, x_aug[:, :6], cfg
            )
            interp_sens = two_way_range_event_sensitivity(
                interp_solution, interp_provider, t_pass, x_aug, cfg
            )
            st_exact = provider.state(solution.t1_s)
            st_interp = interp_provider.state(solution.t1_s)
            d_pos = float(np.linalg.norm(st_exact[:3] - st_interp[:3]))
            d_vel = float(np.linalg.norm(st_exact[3:] - st_interp[3:]))
            d_t1 = abs(interp_solution.t1_s - solution.t1_s)
            d_range = abs(
                interp_solution.delay_calibrated_half_round_trip_range_m
                - solution.delay_calibrated_half_round_trip_range_m
            )
            d_jac = float(
                np.max(
                    np.abs(
                        interp_sens.two_way_range_jacobian_dx0
                        - sens.two_way_range_jacobian_dx0
                    )
                )
            )
            sigma_fraction = d_range / _STATION.sigma_range_m
            print(
                f"[frame policy] grid {grid_s:5.1f} s: station pos {d_pos:.3e} m, "
                f"vel {d_vel:.3e} m/s, t1 {d_t1:.3e} s, range {d_range:.3e} m "
                f"({sigma_fraction:.2%} of sigma), jacobian {d_jac:.3e}"
            )
            self.assertTrue(np.isfinite(d_range))
            # Diagnostic bound: interpolation error stays metre-level.
            self.assertLess(d_range, 5.0)

    def test_earth_ephemeris_interpolation_error_is_separately_bounded(self):
        """The Option-A policy removes transform interpolation; the Earth
        translation term keeps its cubic-Hermite grid interpolation.  Measure
        it directly against SPICE at off-grid epochs, alongside the linear
        interpolation the legacy paths use."""
        et0 = _et0()
        step = 60.0
        t_grid = np.arange(0.0, 1201.0, step)
        earth_states = np.array([_earth_state_mci(et0 + t) for t in t_grid])
        worst_hermite_pos = 0.0
        worst_hermite_vel = 0.0
        worst_linear_pos = 0.0
        for t in t_grid[:-1] + 0.5 * step:
            truth = _earth_state_mci(et0 + float(t))
            hermite = _interp_state(t_grid, earth_states, float(t))
            k = int(np.searchsorted(t_grid, t)) - 1
            frac = (t - t_grid[k]) / step
            linear = earth_states[k] + frac * (earth_states[k + 1] - earth_states[k])
            worst_hermite_pos = max(
                worst_hermite_pos, float(np.linalg.norm(hermite[:3] - truth[:3]))
            )
            worst_hermite_vel = max(
                worst_hermite_vel, float(np.linalg.norm(hermite[3:] - truth[3:]))
            )
            worst_linear_pos = max(
                worst_linear_pos, float(np.linalg.norm(linear[:3] - truth[:3]))
            )
        print(
            f"[earth ephemeris] 60 s grid midpoints: cubic-Hermite pos "
            f"{worst_hermite_pos:.3e} m, vel {worst_hermite_vel:.3e} m/s; "
            f"linear pos {worst_linear_pos:.3e} m"
        )
        self.assertLess(worst_hermite_pos, 1e-3)
        self.assertLess(worst_hermite_vel, 1e-6)

    def test_wrong_uplink_epoch_mutation_is_detected(self):
        (et0, t_pass, earth_pos, earth_vel, x_aug, cfg, provider, t3,
         solution, _) = self._production_case()
        frozen_rx_state = provider.state(t3)
        frozen_provider = TwoWayStationStateProvider(
            state_fn=lambda t: frozen_rx_state,
            station_state_method="mutated_receive_epoch_everywhere",
            earth_ephemeris_method="cubic_hermite_grid_interpolation",
        )
        mutated = solve_two_way_range_events(
            t3, frozen_provider, t_pass, x_aug[:, :6], cfg
        )
        d_range = abs(
            mutated.delay_calibrated_half_round_trip_range_m
            - solution.delay_calibrated_half_round_trip_range_m
        )
        print(f"[epoch mutation] frozen-t3 station state shifts range {d_range:.3f} m")
        self.assertGreater(d_range, 50.0)

    def test_reversed_transform_mutation_is_detected(self):
        import spiceypy as spice

        (et0, t_pass, earth_pos, earth_vel, x_aug, cfg, provider, t3,
         solution, _) = self._production_case()
        earth_states = np.hstack([earth_pos, earth_vel])
        station_ecef_state = np.concatenate([_STATION.r_ecef_m, np.zeros(3)])

        def reversed_fn(t):
            earth_state = _interp_state(t_pass, earth_states, t)
            xform = np.asarray(
                spice.sxform("ITRF93", "J2000", et0 + float(t)), dtype=float
            )
            return earth_state + np.linalg.solve(xform, station_ecef_state)

        reversed_provider = TwoWayStationStateProvider(
            state_fn=reversed_fn,
            station_state_method="mutated_reversed_transform",
            earth_ephemeris_method="cubic_hermite_grid_interpolation",
        )
        mutated = solve_two_way_range_events(
            t3, reversed_provider, t_pass, x_aug[:, :6], cfg
        )
        d_range = abs(
            mutated.delay_calibrated_half_round_trip_range_m
            - solution.delay_calibrated_half_round_trip_range_m
        )
        print(f"[transform mutation] reversed sxform shifts range {d_range:.1f} m")
        self.assertGreater(d_range, 1000.0)


@unittest.skipUnless(_SPICE_OK, _SKIP_REASON)
class TwoWayRangeLegacyConsistencyTests(unittest.TestCase):
    def test_zero_delay_round_trip_matches_legacy_solver_with_real_frames(self):
        """Zero-delay diagnostic against the untouched counted-Doppler solver.
        The legacy path linearly interpolates the pass-grid transforms, so
        agreement is bounded by that measured interpolation error, not by the
        solver tolerances."""
        et0, t_pass, earth_pos, earth_vel, _, _, _, _, x_aug = _truth_setup(step_s=10.0)
        import spiceypy as spice

        xforms = np.array(
            [spice.sxform("J2000", "ITRF93", et0 + float(t)) for t in t_pass]
        )
        t3 = float(t_pass[t_pass.size // 2])
        legacy = solve_two_way_light_time(
            t3, _STATION, t_pass, x_aug[:, :6], earth_pos, earth_vel, xforms,
            RangeRatePhysicsConfig(light_time_tolerance_s=1e-12, light_time_max_iter=25),
        )
        provider = make_exact_sxform_station_state_provider(
            _STATION, et0, t_pass, earth_pos, earth_vel
        )
        new = solve_two_way_range_events(
            t3, provider, t_pass, x_aug[:, :6], TwoWayRangeConfig()
        )
        d_rtlt = abs(new.round_trip_light_time_s - legacy.round_trip_light_time_s)
        print(
            f"[legacy consistency] zero-delay RTLT diff {d_rtlt:.3e} s "
            f"({C_LIGHT_MPS * d_rtlt:.3e} m equivalent) on a 10 s grid"
        )
        self.assertTrue(legacy.converged)
        self.assertLess(d_rtlt, 5e-8)

    def test_counted_doppler_endpoint_consistency_at_zero_delay(self):
        """Directive 28.4: the counted-Doppler m/s equivalent equals the
        endpoint difference of half-round-trip ranges divided by the count
        interval.  The new helper must reproduce that endpoint quantity within
        the measured transform-interpolation bound (the legacy solver linearly
        interpolates the pass-grid transforms; the new solver uses exact
        event-epoch sxform).  Counted-Doppler production code is unchanged."""
        from lunar_od.radiometrics import two_way_counted_doppler_observable

        et0, t_pass, earth_pos, earth_vel, _, _, _, _, x_aug = _truth_setup(step_s=10.0)
        import spiceypy as spice

        xforms = np.array(
            [spice.sxform("J2000", "ITRF93", et0 + float(t)) for t in t_pass]
        )
        count_interval_s = 60.0
        t_mid = float(t_pass[t_pass.size // 2])
        rr_cfg = RangeRatePhysicsConfig(
            mode="two_way_counted_doppler",
            count_interval_s=count_interval_s,
            light_time_tolerance_s=1e-12,
            light_time_max_iter=25,
        )
        doppler_mps = two_way_counted_doppler_observable(
            t_mid, _STATION, t_pass, x_aug[:, :6], earth_pos, earth_vel, xforms, rr_cfg
        )

        provider = make_exact_sxform_station_state_provider(
            _STATION, et0, t_pass, earth_pos, earth_vel
        )
        cfg = TwoWayRangeConfig()
        r_start = solve_two_way_range_events(
            t_mid - 0.5 * count_interval_s, provider, t_pass, x_aug[:, :6], cfg
        ).raw_half_round_trip_range_m
        r_end = solve_two_way_range_events(
            t_mid + 0.5 * count_interval_s, provider, t_pass, x_aug[:, :6], cfg
        ).raw_half_round_trip_range_m
        endpoint_rate = (r_end - r_start) / count_interval_s
        diff = abs(endpoint_rate - doppler_mps)
        print(
            f"[doppler consistency] counted {doppler_mps:.9f} m/s vs endpoint "
            f"{endpoint_rate:.9f} m/s, diff {diff:.3e} m/s"
        )
        self.assertLess(diff, 1e-3)


class TwoWayRangeScenarioConfigTests(unittest.TestCase):
    """SPICE-free configuration validation for the new measurement type."""

    def _payload(self, **overrides):
        payload = {
            "name": "two-way-demo",
            "measurement_type": "two_way_range",
            "estimator_type": "bls_lm",
            "start_mode": "cold",
            "network": "multi",
        }
        payload.update(overrides)
        return payload

    def test_valid_two_way_range_config_accepted(self):
        from lunar_od.scenario_config import (
            scenario_config_from_mapping,
            scenario_two_way_range_config,
        )

        config = scenario_config_from_mapping(
            self._payload(
                transponder_delay_s=2.5e-6,
                two_way_range_convention="raw_half_round_trip",
            )
        )
        self.assertEqual(config.measurement_type, "two_way_range")
        two_way = scenario_two_way_range_config(config)
        self.assertEqual(two_way.transponder_delay_s, 2.5e-6)
        self.assertEqual(two_way.convention, "raw_half_round_trip")

    def test_two_way_range_with_ukf_rejected(self):
        from lunar_od.scenario_config import scenario_config_from_mapping

        with self.assertRaises(ValueError) as ctx:
            scenario_config_from_mapping(self._payload(estimator_type="ukf"))
        self.assertIn("two_way_range", str(ctx.exception))

    def test_two_way_range_with_bias_mode_rejected(self):
        from lunar_od.scenario_config import scenario_config_from_mapping

        with self.assertRaises(ValueError):
            scenario_config_from_mapping(
                self._payload(estimator_type="srif", bias_mode="global")
            )

    def test_convention_requires_two_way_measurement_type(self):
        from lunar_od.scenario_config import scenario_config_from_mapping

        with self.assertRaises(ValueError):
            scenario_config_from_mapping(
                self._payload(
                    measurement_type="range_rate",
                    two_way_range_convention="raw_half_round_trip",
                )
            )

    def test_negative_delay_rejected(self):
        from lunar_od.scenario_config import scenario_config_from_mapping

        with self.assertRaises(ValueError):
            scenario_config_from_mapping(self._payload(transponder_delay_s=-1.0))


if __name__ == "__main__":
    unittest.main()
