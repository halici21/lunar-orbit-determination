import math
import unittest

import numpy as np

from lunar_od import (
    analyze_boundary_jumps,
    analyze_convergence,
    analyze_estimation_error_consistency,
    analyze_innovation_consistency,
    analyze_innovation_whiteness,
    analyze_residuals,
)


class DiagnosticsTests(unittest.TestCase):
    def test_residual_diagnostics_with_sigma_weights(self):
        residuals = np.array([2.0, -2.0, 4.0, -4.0])
        sigma = np.full(4, 2.0)

        result = analyze_residuals(residuals, sigma=sigma, num_solve_for=1)

        self.assertEqual(result.count, 4)
        self.assertEqual(result.dof, 3)
        self.assertAlmostEqual(result.rms, math.sqrt(10.0))
        self.assertAlmostEqual(result.whitened_rms, math.sqrt(2.5))
        self.assertAlmostEqual(result.chi_square, 10.0)
        self.assertAlmostEqual(result.reduced_chi_square, 10.0 / 3.0)
        self.assertAlmostEqual(result.mahalanobis_norm, math.sqrt(10.0))
        self.assertAlmostEqual(result.lag1_autocorrelation, -0.7)

    def test_residual_diagnostics_with_covariance_matches_sigma_case(self):
        residuals = np.array([2.0, -2.0, 4.0, -4.0])
        covariance = np.diag([4.0, 4.0, 4.0, 4.0])

        result = analyze_residuals(residuals, covariance=covariance)

        self.assertAlmostEqual(result.chi_square, 10.0)
        self.assertAlmostEqual(result.whitened_rms, math.sqrt(2.5))

    def test_residual_diagnostics_rejects_invalid_weight_inputs(self):
        residuals = np.array([1.0, 2.0])

        with self.assertRaises(ValueError):
            analyze_residuals(residuals, sigma=[1.0])
        with self.assertRaises(ValueError):
            analyze_residuals(residuals, sigma=[1.0, 0.0])
        with self.assertRaises(ValueError):
            analyze_residuals(residuals, covariance=np.eye(3))
        with self.assertRaises(ValueError):
            analyze_residuals(residuals, sigma=[1.0, 1.0], covariance=np.eye(2))

    def test_empty_residuals_return_nan_metrics(self):
        result = analyze_residuals([])

        self.assertEqual(result.count, 0)
        self.assertTrue(math.isnan(result.rms))
        self.assertTrue(math.isnan(result.chi_square))

    def test_boundary_jump_diagnostics(self):
        previous = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        next_ = np.array([[3.0, 4.0, 0.0], [2.0, 3.0, 1.0]])

        result = analyze_boundary_jumps(previous, next_)

        np.testing.assert_allclose(result.jump_norms, [5.0, math.sqrt(5.0)])
        self.assertEqual(result.count, 2)
        self.assertAlmostEqual(result.rms_jump, math.sqrt(15.0))
        self.assertAlmostEqual(result.median_jump, 0.5 * (5.0 + math.sqrt(5.0)))
        self.assertAlmostEqual(result.max_jump, 5.0)

    def test_convergence_diagnostics_classify_stop_reasons_and_quality_flags(self):
        cost_stable = analyze_convergence(
            "J-Stab",
            rank=6,
            expected_rank=6,
            condition_number=1e5,
            rejected_components=2,
            final_cost=1.0,
        )
        self.assertEqual(cost_stable.category, "converged_cost_stability")
        self.assertTrue(cost_stable.converged)
        self.assertTrue(cost_stable.converged_by_cost_stability)
        self.assertTrue(cost_stable.outlier_rejected)
        self.assertTrue(cost_stable.finite_final_cost)

        rank_bad = analyze_convergence("Converged", rank=5, expected_rank=6, condition_number=1e5, final_cost=1.0)
        self.assertEqual(rank_bad.category, "converged_step")
        self.assertFalse(rank_bad.converged)
        self.assertTrue(rank_bad.rank_deficient)
        self.assertTrue(rank_bad.singular_or_ill_conditioned)

        max_iter = analyze_convergence("MaxIter", rank=6, expected_rank=6, condition_number=1e5, final_cost=np.inf)
        self.assertEqual(max_iter.category, "max_iter")
        self.assertTrue(max_iter.max_iter_reached)
        self.assertFalse(max_iter.converged)
        self.assertFalse(max_iter.finite_final_cost)

    def test_innovation_consistency_computes_nis(self):
        innovation = np.array([2.0, -1.0])
        covariance = np.diag([4.0, 0.25])

        result = analyze_innovation_consistency(innovation, covariance)

        np.testing.assert_allclose(result.whitened, [1.0, -2.0])
        self.assertEqual(result.dimension, 2)
        self.assertAlmostEqual(result.statistic, 5.0)
        self.assertAlmostEqual(result.normalized_statistic, 2.5)
        self.assertAlmostEqual(result.sigma_norm, math.sqrt(5.0))

    def test_estimation_error_consistency_computes_nees_with_full_covariance(self):
        error = np.array([2.0, 2.0])
        covariance = np.array([[5.0, 3.0], [3.0, 5.0]])

        result = analyze_estimation_error_consistency(error, covariance)

        expected = float(error.T @ np.linalg.solve(covariance, error))
        self.assertEqual(result.dimension, 2)
        self.assertAlmostEqual(result.statistic, expected)
        self.assertAlmostEqual(result.normalized_statistic, expected / 2.0)
        self.assertAlmostEqual(result.sigma_norm, math.sqrt(expected))

    def test_innovation_whiteness_reports_component_lag1_correlation(self):
        innovations = np.column_stack(
            [
                np.arange(1.0, 7.0),
                np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),
            ]
        )
        covariances = np.repeat(np.eye(2)[None, :, :], innovations.shape[0], axis=0)

        result = analyze_innovation_whiteness(innovations, covariances)

        np.testing.assert_allclose(result.whitened_innovations, innovations)
        self.assertEqual(result.count, 6)
        self.assertEqual(result.dimension, 2)
        self.assertGreater(result.component_lag1_autocorrelation[0], 0.4)
        self.assertLess(result.component_lag1_autocorrelation[1], -0.8)
        self.assertGreater(result.max_abs_lag1_autocorrelation, 0.8)

    def test_consistency_diagnostics_reject_invalid_inputs(self):
        with self.assertRaises(ValueError):
            analyze_innovation_consistency([], np.eye(0))
        with self.assertRaises(ValueError):
            analyze_innovation_consistency([1.0, 2.0], np.eye(3))
        with self.assertRaises(ValueError):
            analyze_estimation_error_consistency([1.0, 2.0], np.array([[1.0, 2.0], [2.0, 1.0]]))


if __name__ == "__main__":
    unittest.main()
