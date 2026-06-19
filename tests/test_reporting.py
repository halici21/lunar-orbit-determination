import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lunar_od import ArcResult, EstimatorStats, ScenarioResult
from lunar_od import plot_scenario_comparison, plot_visibility_analysis
from lunar_od import write_scenario_summary_csv, write_visibility_summary_csv


class ReportingTests(unittest.TestCase):
    def test_plot_and_csv_outputs_are_created(self):
        scenario = ScenarioResult(
            label="demo",
            measurement_type="position",
            start_mode="cold",
            arc_results=(
                _arc_result(1, 100.0, 1.0),
                _arc_result(2, 200.0, 2.0),
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_path = write_scenario_summary_csv([scenario], tmp_path / "summary.csv")
            png_path = plot_scenario_comparison([scenario], tmp_path / "plot.png")

            self.assertTrue(csv_path.is_file())
            self.assertTrue(png_path.is_file())
            self.assertGreater(csv_path.stat().st_size, 100)
            self.assertGreater(png_path.stat().st_size, 1000)

    def test_scenario_csv_includes_state_bias_correlation_metrics(self):
        posterior_covariance = np.eye(8)
        posterior_covariance[0, 6] = posterior_covariance[6, 0] = 0.5
        posterior_covariance[3, 7] = posterior_covariance[7, 3] = -0.25
        scenario = ScenarioResult(
            label="bias_demo",
            measurement_type="range_rate",
            range_rate_physics="two_way_counted_doppler",
            count_interval_s=20.0,
            start_mode="formal",
            arc_results=(
                _arc_result(1, 100.0, 1.0, posterior_covariance=posterior_covariance),
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = write_scenario_summary_csv([scenario], Path(tmp_dir) / "summary.csv")
            with csv_path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))

        self.assertEqual(row["range_rate_physics"], "two_way_counted_doppler")
        self.assertAlmostEqual(float(row["count_interval_s"]), 20.0)
        self.assertEqual(int(row["posterior_num_bias_states"]), 2)
        self.assertAlmostEqual(float(row["posterior_max_state_bias_corr"]), 0.5)
        self.assertAlmostEqual(float(row["posterior_state_rss_sigma"]), math.sqrt(6.0))
        self.assertAlmostEqual(float(row["posterior_bias_rss_sigma"]), math.sqrt(2.0))
        self.assertEqual(int(row["prior_num_bias_states"]), 0)
        self.assertTrue(math.isnan(float(row["prior_max_state_bias_corr"])))

    def test_scenario_csv_includes_convergence_reason_columns(self):
        scenario = ScenarioResult(
            label="conv_demo",
            measurement_type="range_rate",
            start_mode="hot",
            arc_results=(
                _arc_result(
                    1,
                    100.0,
                    1.0,
                    stop_reason="MaxIter",
                    rank=5,
                    estimated_bias=np.zeros(2),
                    rejected_components=3,
                    final_cost=np.inf,
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = write_scenario_summary_csv([scenario], Path(tmp_dir) / "summary.csv")
            with csv_path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))

        self.assertEqual(row["stop_reason"], "MaxIter")
        self.assertEqual(row["convergence_category"], "max_iter")
        self.assertEqual(row["converged"], "False")
        self.assertEqual(row["max_iter_reached"], "True")
        self.assertEqual(row["rank_deficient"], "True")
        self.assertEqual(row["outlier_rejected"], "True")
        self.assertEqual(row["finite_final_cost"], "False")
        self.assertEqual(row["algorithmic_success"], "False")
        self.assertEqual(row["operational_success"], "False")
        self.assertEqual(row["operational_category"], "operational_failure")

    def test_scenario_csv_separates_algorithmic_and_operational_success(self):
        scenario = ScenarioResult(
            label="operational_demo",
            measurement_type="range_rate",
            start_mode="hot",
            arc_results=(
                _arc_result(
                    1,
                    250.0,
                    8.0,
                    stop_reason="MaxIter",
                    rank=6,
                    condition_number=1e5,
                    final_cost=50.0,
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = write_scenario_summary_csv([scenario], Path(tmp_dir) / "summary.csv")
            with csv_path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))

        self.assertEqual(scenario.algorithmic_success_fraction, 0.0)
        self.assertEqual(scenario.operational_success_fraction, 1.0)
        self.assertEqual(scenario.success_fraction, 1.0)
        self.assertEqual(row["algorithmic_success"], "False")
        self.assertEqual(row["operational_success"], "True")
        self.assertEqual(row["operational_category"], "max_iter_acceptable")
        self.assertEqual(row["final_error_acceptable"], "True")
        self.assertEqual(row["condition_acceptable"], "True")

    def test_scenario_csv_accepts_accuracy_stable_singular_exact_fit(self):
        scenario = ScenarioResult(
            label="exact_fit_demo",
            measurement_type="position",
            start_mode="cold",
            arc_results=(
                _arc_result(
                    1,
                    150.0,
                    1e-6,
                    stop_reason="Singular",
                    rank=6,
                    condition_number=500.0,
                    final_cost=1e-12,
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = write_scenario_summary_csv([scenario], Path(tmp_dir) / "summary.csv")
            with csv_path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))

        self.assertEqual(scenario.algorithmic_success_fraction, 0.0)
        self.assertEqual(scenario.operational_success_fraction, 1.0)
        self.assertEqual(row["algorithmic_success"], "False")
        self.assertEqual(row["operational_success"], "True")
        self.assertEqual(row["operational_category"], "accuracy_stable")

    def test_scenario_csv_includes_ukf_diagnostics_columns(self):
        scenario = ScenarioResult(
            label="ukf_demo",
            measurement_type="position",
            estimator_type="ukf",
            start_mode="hot",
            arc_results=(
                _arc_result(
                    1,
                    100.0,
                    2.0,
                    ukf_mean_nis=2.5,
                    ukf_max_nis=8.0,
                    ukf_accepted_update_fraction=0.95,
                    ukf_final_process_noise_scale=1.2,
                    ukf_innovation_mean_abs_lag1=0.12,
                    ukf_innovation_max_abs_lag1=0.3,
                    ukf_normalized_mean_nis=0.83,
                    ukf_nis_upper_consistent=True,
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = write_scenario_summary_csv([scenario], Path(tmp_dir) / "summary.csv")
            with csv_path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))

        self.assertEqual(row["estimator_type"], "ukf")
        self.assertAlmostEqual(float(row["ukf_mean_nis"]), 2.5)
        self.assertAlmostEqual(float(row["ukf_max_nis"]), 8.0)
        self.assertAlmostEqual(float(row["ukf_accepted_update_fraction"]), 0.95)
        self.assertAlmostEqual(float(row["ukf_final_process_noise_scale"]), 1.2)
        self.assertAlmostEqual(float(row["ukf_innovation_mean_abs_lag1"]), 0.12)
        self.assertAlmostEqual(float(row["ukf_innovation_max_abs_lag1"]), 0.3)
        self.assertAlmostEqual(float(row["ukf_normalized_mean_nis"]), 0.83)
        self.assertEqual(row["ukf_nis_upper_consistent"], "True")

    def test_visibility_plot_and_csv_outputs_are_created(self):
        t_s = np.arange(0.0, 601.0, 60.0)
        station_names = ("A", "B")
        vis_mask = np.zeros((t_s.size, 2), dtype=bool)
        vis_mask[1:5, 0] = True
        vis_mask[6:9, 1] = True
        net_filled = np.any(vis_mask, axis=1)
        net_filled[5] = True
        seg_starts = np.array([1])
        seg_ends = np.array([8])

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_path = write_visibility_summary_csv(
                t_s,
                station_names,
                vis_mask,
                net_filled,
                seg_starts,
                seg_ends,
                tmp_path / "visibility.csv",
            )
            png_path = plot_visibility_analysis(
                t_s,
                station_names,
                vis_mask,
                net_filled,
                seg_starts,
                seg_ends,
                tmp_path / "visibility.png",
            )

            self.assertTrue(csv_path.is_file())
            self.assertTrue(png_path.is_file())
            self.assertGreater(csv_path.stat().st_size, 100)
            self.assertGreater(png_path.stat().st_size, 1000)


def _arc_result(
    arc_id: int,
    initial_pos_err: float,
    final_pos_err: float,
    *,
    posterior_covariance=None,
    stop_reason="J-Stab",
    rank=6,
    condition_number=float("nan"),
    rejected_components=0,
    final_cost=1e-8,
    estimated_bias=None,
    ukf_mean_nis=float("nan"),
    ukf_max_nis=float("nan"),
    ukf_accepted_update_fraction=float("nan"),
    ukf_final_process_noise_scale=float("nan"),
    ukf_innovation_mean_abs_lag1=float("nan"),
    ukf_innovation_max_abs_lag1=float("nan"),
    ukf_normalized_mean_nis=float("nan"),
    ukf_nis_upper_consistent=None,
) -> ArcResult:
    return ArcResult(
        arc_id=arc_id,
        start_idx=0,
        end_idx=10,
        num_observations=20,
        initial_position_error_m=initial_pos_err,
        initial_velocity_error_mps=0.1,
        final_position_error_m=final_pos_err,
        final_velocity_error_mps=0.001,
        stop_reason=stop_reason,
        stats=EstimatorStats(
            iterations=4,
            final_cost=final_cost,
            position_step_norm_m=0.0,
            velocity_step_norm_mps=0.0,
            condition_number=condition_number,
            rank=rank,
            rejected_components=rejected_components,
        ),
        estimated_state=np.zeros(6),
        estimated_bias=np.zeros(0) if estimated_bias is None else estimated_bias,
        ukf_mean_nis=ukf_mean_nis,
        ukf_max_nis=ukf_max_nis,
        ukf_accepted_update_fraction=ukf_accepted_update_fraction,
        ukf_final_process_noise_scale=ukf_final_process_noise_scale,
        ukf_innovation_mean_abs_lag1=ukf_innovation_mean_abs_lag1,
        ukf_innovation_max_abs_lag1=ukf_innovation_max_abs_lag1,
        ukf_normalized_mean_nis=ukf_normalized_mean_nis,
        ukf_nis_upper_consistent=ukf_nis_upper_consistent,
        posterior_covariance=posterior_covariance,
    )

if __name__ == "__main__":
    unittest.main()
