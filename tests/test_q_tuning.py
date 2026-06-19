import tempfile
import unittest
from pathlib import Path

import numpy as np

from lunar_od import (
    ProcessNoiseCase,
    default_process_noise_cases,
    run_process_noise_sweep,
    select_best_process_noise,
    state_process_noise_covariance,
    write_process_noise_sweep_csv,
)


class QTuningTests(unittest.TestCase):
    def test_default_process_noise_grid_is_canonical(self):
        cases = default_process_noise_cases()

        self.assertEqual([case.label for case in cases], ["Q0", "Qsmall", "Qmedium", "Qlarge"])
        self.assertIsNone(cases[0].covariance)
        np.testing.assert_allclose(np.diag(cases[1].covariance), [0.01] * 3 + [1e-10] * 3)
        np.testing.assert_allclose(np.diag(cases[3].covariance), [100.0] * 3 + [1e-6] * 3)
        self.assertEqual([case.label for case in default_process_noise_cases(include_zero=False)], ["Qsmall", "Qmedium", "Qlarge"])

    def test_default_process_noise_covariances_are_state_diagonal_psd(self):
        for case in default_process_noise_cases(include_zero=False):
            with self.subTest(case=case.label):
                self.assertEqual(case.covariance.shape, (6, 6))
                np.testing.assert_allclose(case.covariance, case.covariance.T, rtol=0.0, atol=0.0)
                np.testing.assert_allclose(case.covariance, np.diag(np.diag(case.covariance)), rtol=0.0, atol=0.0)
                self.assertTrue(np.all(np.linalg.eigvalsh(case.covariance) >= 0.0))
                np.testing.assert_allclose(np.diag(case.covariance)[:3], [case.sigma_pos_m**2] * 3)
                np.testing.assert_allclose(np.diag(case.covariance)[3:], [case.sigma_vel_mps**2] * 3)

    def test_state_process_noise_covariance_validates_positive_sigmas(self):
        covariance = state_process_noise_covariance(2.0, 0.5)

        np.testing.assert_allclose(np.diag(covariance), [4.0] * 3 + [0.25] * 3)
        with self.assertRaises(ValueError):
            state_process_noise_covariance(0.0, 0.5)

    def test_process_noise_sweep_and_best_selection(self):
        cases = default_process_noise_cases()

        def evaluator(case):
            score_by_label = {"Q0": 4.0, "Qsmall": 2.0, "Qmedium": 1.0, "Qlarge": 3.0}
            return {"final_position_error_m": score_by_label[case.label], "metadata": {"family": "demo"}}

        evaluations = run_process_noise_sweep(cases, evaluator)
        best = select_best_process_noise(evaluations, "final_position_error_m")

        self.assertEqual([evaluation.label for evaluation in evaluations], ["Q0", "Qsmall", "Qmedium", "Qlarge"])
        self.assertEqual(best.best_label, "Qmedium")
        self.assertEqual(best.best_metric_value, 1.0)
        self.assertEqual(evaluations[0].metadata["family"], "demo")

    def test_process_noise_sweep_rejects_duplicate_labels(self):
        cases = (ProcessNoiseCase("Qdup", None), ProcessNoiseCase("Qdup", None))

        with self.assertRaises(ValueError):
            run_process_noise_sweep(cases, lambda case: {"metric": 1.0})

    def test_process_noise_sweep_csv_is_created(self):
        evaluations = run_process_noise_sweep(
            default_process_noise_cases(include_zero=False),
            lambda case: {"score": float(case.sigma_pos_m), "metadata": {"case": case.label}},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_process_noise_sweep_csv(evaluations, Path(tmpdir) / "q_sweep.csv")
            text = path.read_text(encoding="utf-8")

        self.assertIn("metric_score", text)
        self.assertIn("metadata_case", text)


if __name__ == "__main__":
    unittest.main()
