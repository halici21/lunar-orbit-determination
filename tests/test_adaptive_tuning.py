import unittest

import numpy as np

from lunar_od import (
    AdaptiveTuningEvaluation,
    AdaptiveTuningObjective,
    LunarUKFResult,
    MonteCarloCase,
    UKFPerformanceDiagnostics,
    adaptive_evaluations_from_campaign,
    adaptive_evaluation_from_ukf_result,
    adaptive_evaluations_from_ukf_results,
    adaptive_tuning_score,
    analyze_adaptive_tuning_tradeoffs,
    run_monte_carlo_campaign,
    select_adaptive_tuning,
)


class AdaptiveTuningTests(unittest.TestCase):
    def test_selection_penalizes_good_nis_with_bad_state_accuracy(self):
        evaluations = (
            AdaptiveTuningEvaluation(
                "adaptive_r_nis_only",
                mean_nis=4.0,
                measurement_dimension=4,
                mean_nees=30.0,
                state_dimension=6,
                rms_position_error_m=900.0,
            ),
            AdaptiveTuningEvaluation(
                "balanced",
                mean_nis=4.8,
                measurement_dimension=4,
                mean_nees=7.0,
                state_dimension=6,
                rms_position_error_m=80.0,
            ),
        )

        selection = select_adaptive_tuning(evaluations)

        self.assertEqual(selection.best_label, "balanced")
        self.assertLess(selection.scores["balanced"], selection.scores["adaptive_r_nis_only"])

    def test_tradeoff_analysis_flags_nis_fix_that_worsens_state_accuracy(self):
        evaluations = (
            AdaptiveTuningEvaluation(
                "fixed",
                mean_nis=12.0,
                measurement_dimension=4,
                mean_nees=7.0,
                state_dimension=6,
                rms_position_error_m=100.0,
            ),
            AdaptiveTuningEvaluation(
                "adaptive_r_aggressive",
                mean_nis=4.2,
                measurement_dimension=4,
                mean_nees=18.0,
                state_dimension=6,
                rms_position_error_m=180.0,
            ),
        )

        (tradeoff,) = analyze_adaptive_tuning_tradeoffs(evaluations, reference_label="fixed")

        self.assertEqual(tradeoff.candidate_label, "adaptive_r_aggressive")
        self.assertTrue(tradeoff.nis_consistency_improved)
        self.assertTrue(tradeoff.position_accuracy_degraded)
        self.assertTrue(tradeoff.nees_consistency_degraded)
        self.assertTrue(tradeoff.flagged)
        self.assertGreater(tradeoff.nis_consistency_improvement_fraction, 0.0)
        self.assertAlmostEqual(tradeoff.position_error_increase_fraction, 0.8)

    def test_tradeoff_analysis_accepts_balanced_adaptive_candidate(self):
        evaluations = (
            AdaptiveTuningEvaluation(
                "fixed",
                mean_nis=12.0,
                measurement_dimension=4,
                mean_nees=7.0,
                state_dimension=6,
                rms_position_error_m=100.0,
            ),
            AdaptiveTuningEvaluation(
                "adaptive_balanced",
                mean_nis=5.0,
                measurement_dimension=4,
                mean_nees=7.05,
                state_dimension=6,
                rms_position_error_m=105.0,
            ),
        )

        (tradeoff,) = analyze_adaptive_tuning_tradeoffs(evaluations, reference_label="fixed")

        self.assertTrue(tradeoff.nis_consistency_improved)
        self.assertFalse(tradeoff.position_accuracy_degraded)
        self.assertFalse(tradeoff.nees_consistency_degraded)
        self.assertFalse(tradeoff.flagged)
        with self.assertRaises(ValueError):
            analyze_adaptive_tuning_tradeoffs(evaluations, reference_label="missing")

    def test_ukf_results_feed_tradeoff_analysis_directly(self):
        truth = np.zeros((3, 6))
        fixed = _ukf_result_for_tuning(mean_nis=12.0, position_error_m=10.0)
        adaptive = _ukf_result_for_tuning(
            mean_nis=4.2,
            position_error_m=18.0,
            measurement_noise_scale=8.0,
            robust_weight=0.75,
        )

        evaluations = adaptive_evaluations_from_ukf_results(
            {"fixed": fixed, "adaptive_r": adaptive},
            truth,
            state_dimension=6,
        )
        (tradeoff,) = analyze_adaptive_tuning_tradeoffs(evaluations, reference_label="fixed")

        self.assertEqual(evaluations[0].measurement_dimension, 4)
        self.assertAlmostEqual(evaluations[0].rms_position_error_m, 10.0)
        self.assertGreater(evaluations[1].mean_nees, evaluations[0].mean_nees)
        self.assertEqual(evaluations[1].metadata["mean_measurement_noise_scale"], "8")
        self.assertGreater(float(evaluations[1].metadata["robust_reweighted_fraction"]), 0.0)
        self.assertTrue(tradeoff.flagged)

    def test_single_ukf_result_evaluation_accepts_final_truth_vector(self):
        result = _ukf_result_for_tuning(mean_nis=4.0, position_error_m=3.0)

        evaluation = adaptive_evaluation_from_ukf_result(
            "single",
            result,
            np.zeros(6),
            measurement_dimension=4,
            state_dimension=6,
        )

        self.assertEqual(evaluation.label, "single")
        self.assertAlmostEqual(evaluation.mean_nis, 4.0)
        self.assertAlmostEqual(evaluation.rms_position_error_m, 3.0)
        self.assertEqual(evaluation.metadata["num_updates"], "3")

    def test_objective_weights_can_disable_nees_when_truth_is_unavailable(self):
        evaluation = AdaptiveTuningEvaluation(
            "flight_like",
            mean_nis=3.5,
            measurement_dimension=4,
            rms_position_error_m=20.0,
        )
        objective = AdaptiveTuningObjective(nees_weight=0.0, position_error_scale_m=50.0)

        self.assertGreaterEqual(adaptive_tuning_score(evaluation, objective), 0.0)

    def test_invalid_or_duplicate_evaluations_are_rejected(self):
        with self.assertRaises(ValueError):
            AdaptiveTuningEvaluation(
                "bad",
                mean_nis=4.0,
                measurement_dimension=4,
                mean_nees=6.0,
                rms_position_error_m=1.0,
            )
        duplicate = AdaptiveTuningEvaluation(
            "same",
            mean_nis=4.0,
            measurement_dimension=4,
            rms_position_error_m=1.0,
        )
        with self.assertRaises(ValueError):
            select_adaptive_tuning((duplicate, duplicate))

    def test_campaign_metrics_feed_balanced_adaptive_selection(self):
        cases = (MonteCarloCase("nis_only"), MonteCarloCase("balanced"))

        def trial(case, spec, rng):
            if case.label == "nis_only":
                return {
                    "mean_nis": 4.0,
                    "final_nees": 25.0,
                    "final_position_error_m": 500.0,
                }
            return {
                "mean_nis": 4.5,
                "final_nees": 6.5,
                "final_position_error_m": 40.0,
            }

        campaign = run_monte_carlo_campaign(cases, 4, trial, base_seed=9, max_workers=2)
        evaluations = adaptive_evaluations_from_campaign(
            campaign,
            measurement_dimension=4,
            state_dimension=6,
        )
        selection = select_adaptive_tuning(evaluations)

        self.assertEqual(selection.best_label, "balanced")
        self.assertAlmostEqual(evaluations[0].rms_position_error_m, 500.0)
        self.assertEqual(evaluations[1].metadata["num_trials"], "4")


def _ukf_result_for_tuning(
    *,
    mean_nis: float,
    position_error_m: float,
    measurement_noise_scale: float = 1.0,
    robust_weight: float = 1.0,
) -> LunarUKFResult:
    num_updates = 3
    state_size = 6
    measurement_dim = 4
    states = np.zeros((num_updates, state_size), dtype=float)
    states[:, 0] = position_error_m
    covariances = np.repeat(np.eye(state_size)[None, :, :], num_updates, axis=0)
    innovations = np.zeros((num_updates, measurement_dim), dtype=float)
    innovation_covariances = np.repeat(np.eye(measurement_dim)[None, :, :], num_updates, axis=0)
    robust_weights = np.full((num_updates, measurement_dim), robust_weight, dtype=float)
    return LunarUKFResult(
        t_update_s=np.arange(num_updates, dtype=float),
        obs_indices=np.arange(num_updates, dtype=int),
        state_estimates=states,
        covariances=covariances,
        innovations=innovations,
        innovation_covariances=innovation_covariances,
        predicted_measurements=np.zeros((num_updates, measurement_dim), dtype=float),
        normalized_innovation_squared=np.full(num_updates, mean_nis, dtype=float),
        measurement_noise_scales=np.full(num_updates, measurement_noise_scale, dtype=float),
        robust_component_weights=robust_weights,
        process_noise_scales=np.ones(num_updates, dtype=float),
        accepted_updates=np.ones(num_updates, dtype=bool),
        accepted_components=np.ones((num_updates, measurement_dim), dtype=bool),
        final_state=states[-1].copy(),
        final_covariance=covariances[-1].copy(),
        performance=UKFPerformanceDiagnostics(
            elapsed_s=0.1,
            process_function_evaluations=0,
            unique_dynamic_propagations=0,
            dynamic_propagation_cache_hits=0,
            measurement_function_evaluations=0,
            unique_measurement_model_evaluations=0,
            measurement_model_cache_hits=0,
        ),
    )


if __name__ == "__main__":
    unittest.main()
