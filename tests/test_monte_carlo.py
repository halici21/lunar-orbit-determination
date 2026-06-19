import tempfile
import unittest
from pathlib import Path

import numpy as np

from lunar_od import (
    MonteCarloCase,
    MonteCarloTrialResult,
    make_trial_specs,
    run_monte_carlo_trials,
    run_monte_carlo_campaign,
    summarize_monte_carlo_trials,
    write_monte_carlo_summary_csv,
    write_monte_carlo_trials_csv,
)


class MonteCarloTests(unittest.TestCase):
    def test_trial_specs_are_one_based_and_seeded(self):
        specs = make_trial_specs(4, base_seed=100)

        self.assertEqual([spec.trial_id for spec in specs], [1, 2, 3, 4])
        self.assertEqual([spec.seed for spec in specs], [100, 101, 102, 103])
        with self.assertRaises(ValueError):
            make_trial_specs(-1)

    def test_runner_is_reproducible_for_same_base_seed(self):
        def trial(spec, rng):
            return {
                "position_error_m": float(rng.normal(loc=spec.trial_id, scale=0.1)),
                "success": spec.trial_id != 3,
                "metadata": {"case": "demo"},
            }

        first = run_monte_carlo_trials(5, trial, base_seed=42)
        second = run_monte_carlo_trials(5, trial, base_seed=42)

        self.assertEqual([result.seed for result in first], [42, 43, 44, 45, 46])
        self.assertEqual([result.metrics for result in first], [result.metrics for result in second])
        self.assertEqual([result.success for result in first], [True, True, False, True, True])
        self.assertEqual(first[0].metadata["case"], "demo")

    def test_summary_aggregates_finite_metric_values(self):
        results = (
            MonteCarloTrialResult(1, 10, True, {"error_m": 1.0}),
            MonteCarloTrialResult(2, 11, False, {"error_m": 2.0}),
            MonteCarloTrialResult(3, 12, True, {"error_m": 10.0, "nan_metric": np.nan}),
        )

        summary = summarize_monte_carlo_trials(results)
        by_name = {metric.metric_name: metric for metric in summary.metric_summaries}

        self.assertEqual(summary.num_trials, 3)
        self.assertEqual(summary.success_count, 2)
        self.assertAlmostEqual(summary.success_fraction, 2.0 / 3.0)
        self.assertEqual(by_name["error_m"].count, 3)
        self.assertEqual(by_name["error_m"].median, 2.0)
        self.assertAlmostEqual(by_name["error_m"].p95, 9.2)
        self.assertEqual(by_name["nan_metric"].count, 0)

    def test_runner_can_capture_failed_trials(self):
        def trial(spec, rng):
            if spec.trial_id == 2:
                raise RuntimeError("boom")
            return {"value": float(spec.seed)}

        results = run_monte_carlo_trials(3, trial, base_seed=7, continue_on_error=True)

        self.assertEqual([result.success for result in results], [True, False, True])
        self.assertIn("RuntimeError", results[1].metadata["error"])

    def test_trial_and_summary_csv_outputs_are_created(self):
        results = (
            MonteCarloTrialResult(1, 20, True, {"error_m": 1.5}, {"case": "a"}),
            MonteCarloTrialResult(2, 21, True, {"error_m": 2.5}, {"case": "b"}),
        )
        summary = summarize_monte_carlo_trials(results)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            trials_csv = write_monte_carlo_trials_csv(results, tmp_path / "trials.csv")
            summary_csv = write_monte_carlo_summary_csv(summary, tmp_path / "summary.csv")

            self.assertTrue(trials_csv.is_file())
            self.assertTrue(summary_csv.is_file())
            self.assertIn("metric_error_m", trials_csv.read_text(encoding="utf-8"))
            self.assertIn("success_fraction", summary_csv.read_text(encoding="utf-8"))

    def test_parallel_campaign_is_reproducible_and_case_seeded(self):
        cases = (
            MonteCarloCase("small_error", {"sigma": 1.0}),
            MonteCarloCase("large_error", {"sigma": 10.0}),
        )

        def trial(case, spec, rng):
            return {
                "error": abs(float(rng.normal(scale=case.parameters["sigma"]))),
                "trial_copy": float(spec.trial_id),
            }

        serial = run_monte_carlo_campaign(cases, 6, trial, base_seed=100, max_workers=1)
        parallel = run_monte_carlo_campaign(cases, 6, trial, base_seed=100, max_workers=3)

        self.assertEqual(
            [[result.metrics for result in case.trials] for case in serial],
            [[result.metrics for result in case.trials] for case in parallel],
        )
        self.assertEqual([trial.seed for trial in serial[0].trials], list(range(100, 106)))
        self.assertEqual([trial.seed for trial in serial[1].trials], list(range(106, 112)))
        self.assertTrue(all(trial.metadata["case"] == "large_error" for trial in serial[1].trials))
        self.assertEqual(serial[0].summary.num_trials, 6)

    def test_campaign_captures_errors_and_reports_progress(self):
        progress = []

        def trial(case, spec, rng):
            if case.label == "bad":
                raise RuntimeError("campaign failure")
            return {"value": float(rng.normal())}

        results = run_monte_carlo_campaign(
            (MonteCarloCase("good"), MonteCarloCase("bad")),
            2,
            trial,
            base_seed=7,
            max_workers=2,
            continue_on_error=True,
            progress_fn=lambda completed, total, label, trial_id: progress.append(
                (completed, total, label, trial_id)
            ),
        )

        self.assertEqual(len(progress), 4)
        self.assertEqual(results[0].summary.success_fraction, 1.0)
        self.assertEqual(results[1].summary.success_fraction, 0.0)
        self.assertIn("RuntimeError", results[1].trials[0].metadata["error"])


if __name__ == "__main__":
    unittest.main()
