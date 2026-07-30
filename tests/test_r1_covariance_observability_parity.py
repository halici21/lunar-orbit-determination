"""R1 posterior-covariance, observability and numerical-safety qualification."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import lunar_od.dynamics as dynamics
import lunar_od.estimators as estimators
import lunar_od.filters as filters
import lunar_od.observability as observability
from lunar_od import (
    ArcResult,
    EstimatorStats,
    ObservabilityNumericalError,
    ScenarioResult,
    analyze_arc_observability,
    analyze_augmented_arc_observability,
    analyze_initial_state_observability,
    build_initial_state_jacobian,
    estimate_position_bls_lm,
    estimate_position_srif,
    estimate_range_rate_bls_lm,
    estimate_range_rate_srif,
    run_batch_arc_sequence,
    run_lunar_ukf,
    summarize_weighted_jacobian,
)
from lunar_od.constants import J2_MOON_UNNORMALIZED, R_MOON_M
from lunar_od.dynamics import zonal_j2_acceleration, zonal_j2_gravity_gradient
from lunar_od.diagnostics import analyze_convergence
from lunar_od.force_contract import (
    ConsumerReadiness,
    ConsumerRole,
    consumer_capabilities_for,
)
from lunar_od.force_models import body_j2_acceleration, body_j2_gravity_gradient
from lunar_od.scenario_config import (
    force_model_contract_from_scenario_config,
    scenario_config_from_mapping,
)
from tests.test_observability import (
    _build_position_arc,
    _build_range_rate_arc,
    _synthetic_rr_station,
    _synthetic_station,
    _truth_history,
)
from tests.test_r0a_force_config_parity import (
    MU_MOON,
    _position_case,
    _prepared_position_arc,
    _synthetic_ephemeris,
    _ukf_inputs,
)


J2 = float(J2_MOON_UNNORMALIZED)
ZERO_J2_FINGERPRINT = "sha256:9b93897a545d0d2f1cb2b5329ce6be79051fef97e1529bff3bea565c4c31418d"
R1_J2_FINGERPRINT = "sha256:11d33466c53e4c4a48cb8f72984ad1740e81de26ef342b4ecd106a30a7ddb52d"


def _position_fixture(duration_s: float = 300.0):
    x0, times, truth, pass_geo, observations, get_earth, get_sun = _position_case(
        duration_s=duration_s,
        step_s=60.0,
    )
    initial = x0 + np.array([20.0, -10.0, 5.0, 0.01, -0.005, 0.002])
    return times, truth, pass_geo, observations, get_earth, get_sun, initial


def _range_rate_fixture():
    times, truth, get_earth, get_sun = _truth_history(MU_MOON)
    stations = tuple(
        _synthetic_rr_station(*definition)
        for definition in (
            (0.0, 0.0, 0.0),
            (0.0, 90.0, 0.0),
            (45.0, -30.0, 500.0),
            (-35.0, 150.0, 600.0),
        )
    )
    arc = _build_range_rate_arc(1, 1, 6, times, truth, stations)
    initial = arc.truth_state_history_mci[0, :6] + np.array(
        [20.0, -10.0, 5.0, 0.01, -0.005, 0.002]
    )
    return arc, get_earth, get_sun, initial


def _recording_augmented_wrapper(real_function, sink):
    def _wrapped(*args, **kwargs):
        caller = inspect.currentframe().f_back.f_code.co_name
        sink.append((caller, kwargs.get("j2_moon")))
        return real_function(*args, **kwargs)

    return _wrapped


def _recording_j2_wrapper(real_function, sink):
    def _wrapped(*args, **kwargs):
        sink.append(kwargs.get("j2_moon"))
        return real_function(*args, **kwargs)

    return _wrapped


def _run_position_estimator(function, j2_moon: float):
    times, _, pass_geo, observations, get_earth, get_sun, initial = _position_fixture()
    return function(
        times,
        observations,
        initial,
        pass_geo,
        MU_MOON,
        0.0,
        0.0,
        get_earth,
        get_sun,
        max_iter=1,
        rtol=1.0e-11,
        atol=1.0e-12,
        j2_moon=j2_moon,
        return_posterior=True,
    )


def _run_range_rate_estimator(function, j2_moon: float):
    arc, get_earth, get_sun, initial = _range_rate_fixture()
    return function(
        arc.t_pass_s,
        arc.obs_data,
        initial,
        arc.pass_geo,
        MU_MOON,
        0.0,
        0.0,
        get_earth,
        get_sun,
        max_iter=1,
        rtol=1.0e-11,
        atol=1.0e-12,
        j2_moon=j2_moon,
        return_posterior=True,
    )


def _relative_sigma_difference(first: np.ndarray, second: np.ndarray) -> float:
    sigma_first = np.sqrt(np.diag(first))
    sigma_second = np.sqrt(np.diag(second))
    return float(
        np.max(
            np.abs(sigma_second - sigma_first)
            / np.maximum(sigma_first, np.finfo(float).tiny)
        )
    )


def _observability_arc():
    times, truth, get_earth, get_sun = _truth_history(MU_MOON)
    stations = tuple(
        _synthetic_station(*definition)
        for definition in (
            (0.0, 0.0, 0.0),
            (0.0, 90.0, 0.0),
            (45.0, -30.0, 500.0),
            (-35.0, 150.0, 600.0),
        )
    )
    return _build_position_arc(1, 1, 10, times, truth, stations), get_earth, get_sun


def _scenario_arc_result(condition_number: float) -> ArcResult:
    stats = EstimatorStats(
        iterations=1,
        final_cost=1.0,
        position_step_norm_m=0.0,
        velocity_step_norm_mps=0.0,
        condition_number=condition_number,
        rank=6,
    )
    return ArcResult(
        arc_id=1,
        start_idx=0,
        end_idx=1,
        num_observations=1,
        initial_position_error_m=1.0,
        initial_velocity_error_mps=0.1,
        final_position_error_m=1.0,
        final_velocity_error_mps=0.1,
        stop_reason="Converged",
        stats=stats,
        estimated_state=np.zeros(6),
        estimated_bias=np.zeros(0),
    )


class R1PosteriorAndObservabilityParityTests(unittest.TestCase):
    def test_position_bls_and_srif_posterior_calls_bind_exact_j2(self):
        for function, required_callers in (
            (estimate_position_bls_lm, {"_position_posterior_information"}),
            (
                estimate_position_srif,
                {"_position_posterior_information", "_position_initial_jacobian"},
            ),
        ):
            sink = []
            with mock.patch.object(
                estimators,
                "propagate_augmented_state",
                _recording_augmented_wrapper(estimators.propagate_augmented_state, sink),
            ):
                _run_position_estimator(function, J2)
            self.assertGreater(len(sink), 0, function.__name__)
            self.assertTrue(all(value == J2 for _, value in sink), sink)
            self.assertTrue(required_callers.issubset({caller for caller, _ in sink}), sink)

    def test_range_rate_bls_and_srif_posterior_calls_bind_exact_j2(self):
        for function, required_callers in (
            (estimate_range_rate_bls_lm, {"_range_rate_posterior_information"}),
            (
                estimate_range_rate_srif,
                {"_range_rate_posterior_information", "_range_rate_initial_jacobian"},
            ),
        ):
            sink = []
            with mock.patch.object(
                estimators,
                "propagate_augmented_state",
                _recording_augmented_wrapper(estimators.propagate_augmented_state, sink),
            ):
                _run_range_rate_estimator(function, J2)
            self.assertGreater(len(sink), 0, function.__name__)
            self.assertTrue(all(value == J2 for _, value in sink), sink)
            self.assertTrue(required_callers.issubset({caller for caller, _ in sink}), sink)

        arc, get_earth, get_sun, initial = _range_rate_fixture()
        fallback_sink = []
        with mock.patch.object(
            estimators,
            "propagate_augmented_state",
            _recording_augmented_wrapper(
                estimators.propagate_augmented_state, fallback_sink
            ),
        ):
            estimators._range_rate_nominal_and_initial_jacobian(
                arc.t_pass_s,
                arc.obs_data,
                initial,
                arc.pass_geo,
                MU_MOON,
                0.0,
                0.0,
                get_earth,
                get_sun,
                None,
                1.0e-11,
                1.0e-12,
                j2_moon=J2,
            )
        self.assertGreater(len(fallback_sink), 0)
        self.assertTrue(all(value == J2 for _, value in fallback_sink), fallback_sink)

    def test_j2_gradient_fd_sweep_has_accuracy_floor_and_two_flanks(self):
        state = np.array([1.81e6, 2.7e5, -1.4e5])
        rotation = np.array(
            [[0.9, -0.3, 0.316227766], [0.35, 0.93675, -0.01054], [-0.293, 0.120, 0.9485]]
        )
        rotation, _ = np.linalg.qr(rotation)
        epsilons = np.logspace(-4.0, 4.0, 17)
        cases = (
            (
                zonal_j2_gravity_gradient(state, MU_MOON, R_MOON_M, J2),
                lambda value: zonal_j2_acceleration(value, MU_MOON, R_MOON_M, J2),
            ),
            (
                body_j2_gravity_gradient(state, MU_MOON, R_MOON_M, J2, np.eye(3)),
                lambda value: body_j2_acceleration(
                    value, MU_MOON, R_MOON_M, J2, np.eye(3)
                ),
            ),
            (
                body_j2_gravity_gradient(state, MU_MOON, R_MOON_M, J2, rotation),
                lambda value: body_j2_acceleration(
                    value, MU_MOON, R_MOON_M, J2, rotation
                ),
            ),
        )
        for analytic, acceleration in cases:
            errors = []
            for epsilon in epsilons:
                finite_difference = np.empty((3, 3))
                for column in range(3):
                    plus = state.copy()
                    minus = state.copy()
                    plus[column] += epsilon
                    minus[column] -= epsilon
                    finite_difference[:, column] = (
                        acceleration(plus) - acceleration(minus)
                    ) / (2.0 * epsilon)
                errors.append(
                    np.linalg.norm(analytic - finite_difference) / np.linalg.norm(analytic)
                )
            minimum = int(np.argmin(errors))
            self.assertLessEqual(errors[minimum], 1.0e-8)
            self.assertGreater(errors[0], errors[minimum])
            self.assertGreater(errors[-1], errors[minimum])

    def test_stm_finite_difference_matches_j2_off_and_on(self):
        times = np.arange(0.0, 601.0, 60.0)
        perturbations = np.array([10.0, 10.0, 10.0, 0.1, 0.1, 0.1])
        _, _, _, _, get_earth, get_sun, initial = _position_fixture()
        for j2_moon in (0.0, J2):
            augmented = dynamics.propagate_augmented_state(
                times,
                np.concatenate([initial, np.eye(6).reshape(-1, order="F")]),
                MU_MOON,
                0.0,
                0.0,
                get_earth,
                get_sun,
                method="DOP853",
                rtol=1.0e-12,
                atol=1.0e-13,
                j2_moon=j2_moon,
            )
            phi = np.stack([row[6:].reshape(6, 6, order="F") for row in augmented])
            for column, epsilon in enumerate(perturbations):
                plus = initial.copy()
                minus = initial.copy()
                plus[column] += epsilon
                minus[column] -= epsilon
                propagated_plus = dynamics.propagate_state(
                    times,
                    plus,
                    MU_MOON,
                    0.0,
                    0.0,
                    get_earth,
                    get_sun,
                    method="DOP853",
                    rtol=1.0e-12,
                    atol=1.0e-13,
                    j2_moon=j2_moon,
                )
                propagated_minus = dynamics.propagate_state(
                    times,
                    minus,
                    MU_MOON,
                    0.0,
                    0.0,
                    get_earth,
                    get_sun,
                    method="DOP853",
                    rtol=1.0e-12,
                    atol=1.0e-13,
                    j2_moon=j2_moon,
                )
                finite_difference = (propagated_plus - propagated_minus) / (2.0 * epsilon)
                relative = np.linalg.norm(
                    phi[:, :, column] - finite_difference, axis=1
                ) / np.maximum(np.linalg.norm(phi[:, :, column], axis=1), 1.0e-30)
                self.assertLessEqual(float(np.max(relative)), 1.0e-6)

    def test_fast_and_pure_python_augmented_dynamics_match_with_j2(self):
        self.assertTrue(dynamics._FAST_DYNAMICS, "R1 qualification requires Numba fast dynamics")
        times, _, _, _, get_earth, get_sun, initial = _position_fixture()
        augmented0 = np.concatenate([initial, np.eye(6).reshape(-1, order="F")])
        fast = dynamics.propagate_augmented_state(
            times,
            augmented0,
            MU_MOON,
            0.0,
            0.0,
            get_earth,
            get_sun,
            method="DOP853",
            rtol=1.0e-11,
            atol=1.0e-12,
            j2_moon=J2,
        )
        with mock.patch.object(dynamics, "_FAST_DYNAMICS", False):
            pure = dynamics.propagate_augmented_state(
                times,
                augmented0,
                MU_MOON,
                0.0,
                0.0,
                get_earth,
                get_sun,
                method="DOP853",
                rtol=1.0e-11,
                atol=1.0e-12,
                j2_moon=J2,
            )
        np.testing.assert_allclose(fast, pure, rtol=2.0e-12, atol=2.0e-9)

    def test_bls_posterior_changes_with_j2_for_both_measurement_types(self):
        position_off = _run_position_estimator(estimate_position_bls_lm, 0.0)[2]
        position_on = _run_position_estimator(estimate_position_bls_lm, J2)[2]
        range_rate_off = _run_range_rate_estimator(estimate_range_rate_bls_lm, 0.0)[2]
        range_rate_on = _run_range_rate_estimator(estimate_range_rate_bls_lm, J2)[2]
        self.assertGreater(
            _relative_sigma_difference(
                position_off.posterior_covariance, position_on.posterior_covariance
            ),
            1.0e-9,
        )
        self.assertGreater(
            _relative_sigma_difference(
                range_rate_off.posterior_covariance, range_rate_on.posterior_covariance
            ),
            1.0e-9,
        )

    def test_srif_posterior_changes_and_sqrt_information_reconstructs(self):
        for runner in (_run_position_estimator, _run_range_rate_estimator):
            off = runner(
                estimate_position_srif if runner is _run_position_estimator else estimate_range_rate_srif,
                0.0,
            )[2]
            on = runner(
                estimate_position_srif if runner is _run_position_estimator else estimate_range_rate_srif,
                J2,
            )[2]
            self.assertGreater(
                _relative_sigma_difference(off.posterior_covariance, on.posterior_covariance),
                1.0e-9,
            )
            reconstructed = on.posterior_sqrt_information.T @ on.posterior_sqrt_information
            relative = np.linalg.norm(reconstructed - on.posterior_information) / np.linalg.norm(
                on.posterior_information
            )
            self.assertLessEqual(float(relative), 1.0e-10)

    def _run_ukf_with_state_spy(self, covariance_form: str):
        x0, times, truth, pass_geo, observations, get_earth, get_sun = _position_case(
            duration_s=180.0, step_s=60.0
        )
        initial, covariance = _ukf_inputs(x0, truth)
        sink = []
        with mock.patch.object(
            filters,
            "propagate_state",
            _recording_j2_wrapper(filters.propagate_state, sink),
        ):
            result = run_lunar_ukf(
                times,
                observations,
                initial,
                covariance,
                pass_geo,
                MU_MOON,
                0.0,
                0.0,
                get_earth,
                get_sun,
                covariance_form=covariance_form,
                rtol=1.0e-10,
                atol=1.0e-11,
                j2_moon=J2,
            )
        self.assertGreater(len(sink), 0)
        self.assertTrue(all(value == J2 for value in sink), sink)
        self.assertTrue(np.all(np.isfinite(result.final_covariance)))
        return result

    def test_standard_ukf_covariance_uses_nonzero_j2(self):
        self._run_ukf_with_state_spy("standard")
        capabilities = consumer_capabilities_for(
            lunar_j2_on=True, earth_j2_on=False, harmonics_on=False
        )
        self.assertEqual(capabilities[ConsumerRole.UKF_STANDARD], ConsumerReadiness.VERIFIED)

    def test_square_root_ukf_covariance_uses_nonzero_j2(self):
        self._run_ukf_with_state_spy("square_root")
        capabilities = consumer_capabilities_for(
            lunar_j2_on=True, earth_j2_on=False, harmonics_on=False
        )
        self.assertEqual(
            capabilities[ConsumerRole.UKF_SQUARE_ROOT], ConsumerReadiness.VERIFIED
        )

    def test_optional_ukf_stm_path_uses_nonzero_j2(self):
        x0, times, truth, pass_geo, observations, get_earth, get_sun = _position_case(
            duration_s=180.0, step_s=60.0
        )
        initial, covariance = _ukf_inputs(x0, truth)
        sink = []
        with mock.patch.object(
            filters,
            "propagate_augmented_state",
            _recording_j2_wrapper(filters.propagate_augmented_state, sink),
        ):
            run_lunar_ukf(
                times,
                observations,
                initial,
                covariance,
                pass_geo,
                MU_MOON,
                0.0,
                0.0,
                get_earth,
                get_sun,
                covariance_form="square_root",
                use_stm_linearization=True,
                rtol=1.0e-10,
                atol=1.0e-11,
                j2_moon=J2,
            )
        self.assertGreater(len(sink), 0)
        self.assertTrue(all(value == J2 for value in sink), sink)

    def test_fast_sigma_factory_and_runtime_use_nonzero_j2(self):
        ephemeris = _synthetic_ephemeris(120.0)
        sink = []
        real_fast = dynamics._rk4_6state_fast

        def _runtime_spy(*args, **kwargs):
            sink.append(kwargs.get("j2_moon"))
            return real_fast(*args, **kwargs)

        with mock.patch.object(dynamics, "_rk4_6state_fast", _runtime_spy):
            fast = dynamics.make_fast_sigma_propagator(
                ephemeris,
                MU_MOON,
                0.0,
                0.0,
                rk4_dt_s=10.0,
                j2_moon=J2,
            )
            self.assertIsNotNone(fast)
            fast_state = fast(0.0, 60.0, np.array([1.8374e6, 0.0, 0.0, 0.0, 1633.0, 0.0]))
        self.assertGreater(len(sink), 0)
        self.assertTrue(all(value == J2 for value in sink), sink)
        reference = dynamics.propagate_state(
            [0.0, 60.0],
            np.array([1.8374e6, 0.0, 0.0, 0.0, 1633.0, 0.0]),
            MU_MOON,
            0.0,
            0.0,
            ephemeris.earth_position,
            ephemeris.sun_position,
            method="DOP853",
            rtol=1.0e-11,
            atol=1.0e-12,
            j2_moon=J2,
        )[-1]
        self.assertLess(float(np.linalg.norm(fast_state[:3] - reference[:3])), 1.0e-3)

    def test_all_observability_public_apis_forward_exact_j2(self):
        arc, get_earth, get_sun = _observability_arc()
        common = (
            "position",
            arc.t_pass_s,
            arc.obs_data,
            arc.pass_geo,
            arc.truth_state_history_mci[0, :6],
            MU_MOON,
            0.0,
            0.0,
            get_earth,
            get_sun,
        )
        calls = (
            lambda: build_initial_state_jacobian(*common, j2_moon=J2),
            lambda: analyze_initial_state_observability(*common, j2_moon=J2),
            lambda: analyze_arc_observability(
                arc, "position", MU_MOON, 0.0, 0.0, get_earth, get_sun, j2_moon=J2
            ),
            lambda: analyze_augmented_arc_observability(
                arc,
                "position",
                MU_MOON,
                0.0,
                0.0,
                get_earth,
                get_sun,
                bias_mode="global_full",
                j2_moon=J2,
            ),
        )
        for call in calls:
            sink = []
            with mock.patch.object(
                observability,
                "propagate_augmented_state",
                _recording_j2_wrapper(observability.propagate_augmented_state, sink),
            ):
                call()
            self.assertGreater(len(sink), 0)
            self.assertTrue(all(value == J2 for value in sink), sink)

    def test_observability_spectrum_changes_and_is_deterministic(self):
        arc, get_earth, get_sun = _observability_arc()

        def _run(j2_moon):
            return analyze_arc_observability(
                arc,
                "position",
                MU_MOON,
                0.0,
                0.0,
                get_earth,
                get_sun,
                rtol=1.0e-12,
                atol=1.0e-13,
                j2_moon=j2_moon,
            )

        off = _run(0.0)
        on = _run(J2)
        repeated = _run(J2)
        relative = np.max(
            np.abs(on.singular_values - off.singular_values)
            / np.maximum(off.singular_values, np.finfo(float).tiny)
        )
        self.assertGreater(float(relative), 1.0e-9)
        self.assertTrue(np.all(np.isfinite(on.singular_values)))
        np.testing.assert_array_equal(on.singular_values, repeated.singular_values)
        np.testing.assert_allclose(
            np.sort(on.information_eigenvalues)[::-1],
            on.singular_values**2,
            rtol=1.0e-12,
            atol=1.0e-8,
        )
        tolerance = max(on.weighted_jacobian.shape) * np.finfo(float).eps * on.singular_values[0]
        self.assertEqual(on.rank, int(np.sum(on.singular_values > tolerance)))

    def test_scenario_auto_bias_observability_forwards_j2(self):
        arc, get_earth, get_sun = _prepared_position_arc()
        sink = []
        with mock.patch.object(
            observability,
            "propagate_augmented_state",
            _recording_j2_wrapper(observability.propagate_augmented_state, sink),
        ):
            result = run_batch_arc_sequence(
                (arc,),
                "position",
                "cold",
                "ukf",
                MU_MOON,
                0.0,
                0.0,
                get_earth,
                get_sun,
                cold_start_bank=(np.zeros(6),),
                bias_mode="global_full",
                initial_bias=np.zeros(3),
                ukf_auto_bias_constraints=True,
                j2_moon=J2,
                rtol=1.0e-10,
                atol=1.0e-11,
            )
        self.assertGreater(len(sink), 0)
        self.assertTrue(all(value == J2 for value in sink), sink)
        self.assertTrue(result.observability_finite)
        self.assertEqual(result.posterior_covariance_stm_mode, "ukf_sigma_point")

    def test_nonfinite_inputs_fail_before_decomposition(self):
        for bad_value in (np.nan, np.inf, -np.inf):
            bad = np.eye(6)
            bad[0, 0] = bad_value
            with self.assertRaisesRegex(ObservabilityNumericalError, "weighted_jacobian"):
                summarize_weighted_jacobian("position", 1, bad)
            with self.assertRaisesRegex(ObservabilityNumericalError, "fisher_information"):
                summarize_weighted_jacobian(
                    "position", 1, np.eye(6), fisher_information=bad
                )
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                estimators._safe_covariance_from_information(bad)
            with self.assertRaisesRegex(ObservabilityNumericalError, "rank_tol"):
                summarize_weighted_jacobian("position", 1, np.eye(6), rank_tol=bad_value)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            summarize_weighted_jacobian("position", 1, np.eye(6), rank_tol=-1.0)

    def test_degenerate_observability_is_diagnostic_not_an_exception(self):
        rank_deficient = summarize_weighted_jacobian(
            "position", 1, np.diag([1.0, 1.0e-20, 0.0])
        )
        self.assertLess(rank_deficient.rank, rank_deficient.num_parameters)
        self.assertTrue(np.isinf(rank_deficient.condition_number))
        large_finite = summarize_weighted_jacobian(
            "position", 1, np.diag([1.0, 1.0e-13])
        )
        self.assertTrue(np.isfinite(large_finite.condition_number))
        self.assertGreater(large_finite.condition_number, 1.0e12)

    def test_nonfinite_condition_numbers_fail_and_availability_is_distinct(self):
        for value in (np.nan, np.inf, -np.inf):
            result = _scenario_arc_result(value)
            self.assertFalse(result.condition_acceptable)
            self.assertFalse(result.operational_success)
            self.assertEqual(result.condition_number_available, not np.isnan(value))
            diagnostics = analyze_convergence(
                "Converged", stats=result.stats, expected_rank=6
            )
            self.assertTrue(diagnostics.singular_or_ill_conditioned)

    def test_zero_j2_contract_is_byte_frozen_and_j2_readiness_changes_identity(self):
        def _contract(j2_moon):
            config = scenario_config_from_mapping(
                {
                    "name": "r1_contract",
                    "measurement_type": "range_rate",
                    "estimator_type": "bls_lm",
                    "start_mode": "cold",
                    "network": "multi",
                    "j2_moon": j2_moon,
                }
            )
            return force_model_contract_from_scenario_config(config)

        zero = _contract(0.0)
        enabled = _contract(J2)
        self.assertEqual(zero.force_model_fingerprint(), ZERO_J2_FINGERPRINT)
        self.assertEqual(
            "sha256:" + hashlib.sha256(zero.canonical_json_bytes()).hexdigest(),
            ZERO_J2_FINGERPRINT,
        )
        self.assertEqual(enabled.force_model_fingerprint(), R1_J2_FINGERPRINT)
        self.assertNotEqual(enabled.force_model_fingerprint(), zero.force_model_fingerprint())
        payload = enabled.to_canonical_payload()
        self.assertEqual(payload["consumer_capabilities"]["posterior_covariance"], "verified")
        self.assertEqual(payload["consumer_capabilities"]["observability"], "verified")

    def test_earth_j2_and_harmonics_remain_fail_closed(self):
        earth_capabilities = consumer_capabilities_for(
            lunar_j2_on=False, earth_j2_on=True, harmonics_on=False
        )
        for role, readiness in earth_capabilities.items():
            self.assertEqual(readiness, ConsumerReadiness.UNSUPPORTED, role)
        harmonic_capabilities = consumer_capabilities_for(
            lunar_j2_on=False, earth_j2_on=False, harmonics_on=True
        )
        self.assertEqual(
            harmonic_capabilities[ConsumerRole.TRUTH_STATE],
            ConsumerReadiness.EXPERIMENTAL_DIRECT_TRAJECTORY_ONLY,
        )
        for role in (
            ConsumerRole.ESTIMATOR_STATE,
            ConsumerRole.ESTIMATOR_STM,
            ConsumerRole.POSTERIOR_COVARIANCE,
            ConsumerRole.OBSERVABILITY,
        ):
            self.assertEqual(harmonic_capabilities[role], ConsumerReadiness.UNSUPPORTED)
        _, _, _, _, get_earth, get_sun, initial = _position_fixture()
        with self.assertRaisesRegex(ValueError, "harmonics gradient not implemented"):
            dynamics.propagate_augmented_state(
                [0.0, 60.0],
                np.concatenate([initial, np.eye(6).reshape(-1, order="F")]),
                MU_MOON,
                0.0,
                0.0,
                get_earth,
                get_sun,
                harmonic_model=object(),
            )

    def test_scenario_result_r1_defaults_are_backward_compatible(self):
        scenario = ScenarioResult(
            label="default",
            measurement_type="position",
            start_mode="cold",
            arc_results=(),
        )
        self.assertEqual(scenario.posterior_covariance_stm_mode, "")
        self.assertEqual(scenario.posterior_covariance_rank, 0)
        self.assertFalse(scenario.posterior_covariance_finite)
        self.assertFalse(scenario.observability_finite)
        self.assertEqual(scenario.derivative_validation_profile, "")

    def test_validation_reproducer_emits_all_six_png_csv_pairs(self):
        script_path = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "r1_covariance_observability_validation.py"
        )
        spec = importlib.util.spec_from_file_location("r1_validation_under_test", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        basenames = (
            "r1_j2_dynamics_jacobian_fd_sweep",
            "r1_stm_fd_validation",
            "r1_estimation_error_vs_3sigma",
            "r1_posterior_covariance_j2_comparison",
            "r1_observability_singular_values",
            "r1_observability_rank_condition",
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            summary = module.generate(output)
            self.assertEqual(tuple(summary["outputs"]), basenames)
            for basename in basenames:
                png = output / f"{basename}.png"
                raw_csv = output / f"{basename}.csv"
                self.assertTrue(png.is_file(), png)
                self.assertTrue(raw_csv.is_file(), raw_csv)
                self.assertGreater(png.stat().st_size, 10_000)
                header = raw_csv.read_text(encoding="utf-8").splitlines()[0]
                self.assertIn("commit_sha", header)
                self.assertIn("branch", header)


if __name__ == "__main__":
    unittest.main()
