import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from lunar_od import (
    load_scenario_config_json,
    scenario_config_from_mapping,
    scenario_config_schema,
    scenario_config_summary,
    scenario_range_rate_physics_config,
    scenario_ukf_configs,
    write_normalized_scenario_config,
)


class ScenarioConfigTests(unittest.TestCase):
    def test_minimal_config_is_normalized_with_defaults(self):
        config = scenario_config_from_mapping(
            {
                "name": "demo",
                "measurement_type": "range_rate",
                "estimator_type": "srif",
                "start_mode": "hot",
                "network": "multi",
            }
        )

        self.assertEqual(config.name, "demo")
        self.assertEqual(config.duration_h, 4.0)
        self.assertFalse(config.noise)
        self.assertEqual(config.output_dir, "python_port/results")
        self.assertEqual(config.range_rate_physics, "geometric_instantaneous")
        self.assertEqual(config.count_interval_s, 60.0)
        self.assertEqual(config.ukf_alpha, 0.35)
        self.assertEqual(config.ukf_covariance_form, "square_root")
        self.assertIsNone(config.ukf_nis_gate)
        self.assertIn("multi range_rate srif/hot", scenario_config_summary(config))

    def test_schema_lists_required_fields_and_enums(self):
        schema = scenario_config_schema()

        self.assertIn("measurement_type", schema["required"])
        self.assertEqual(schema["properties"]["measurement_type"]["enum"], ["position", "range_rate"])
        self.assertIn("ukf", schema["properties"]["estimator_type"]["enum"])
        self.assertEqual(
            schema["properties"]["range_rate_physics"]["enum"],
            ["geometric_instantaneous", "two_way_counted_doppler"],
        )
        self.assertIn("single", schema["properties"]["network"]["enum"])
        self.assertEqual(schema["properties"]["ukf_alpha"]["default"], 0.35)
        self.assertEqual(schema["properties"]["ukf_covariance_form"]["default"], "square_root")
        self.assertEqual(schema["properties"]["ukf_nis_gate"]["default"], None)

    def test_ukf_runtime_configs_are_normalized(self):
        config = scenario_config_from_mapping(
            {
                "name": "ukf_tuned",
                "measurement_type": "range_rate",
                "estimator_type": "ukf",
                "start_mode": "hot",
                "network": "multi",
                "ukf_alpha": 0.2,
                "ukf_beta": 2.5,
                "ukf_kappa": 1.0,
                "ukf_covariance_inflation": 1.01,
                "ukf_adaptive_measurement_noise": True,
                "ukf_max_measurement_noise_scale": 25.0,
                "ukf_nis_gate": 16.0,
                "ukf_component_nis_gate": 9.0,
                "ukf_component_gate_mode": "conditional",
                "ukf_robust_measurement_update": True,
                "ukf_robust_loss": "huber",
                "ukf_robust_student_t_dof": 7.0,
                "ukf_robust_huber_threshold": 2.5,
                "ukf_robust_min_component_weight": 0.2,
                "two_way_local_state_model": "taylor3",
                "station_clock_offset_s": 2e-6,
                "station_clock_drift": 1e-10,
                "transponder_delay_s": 3e-6,
                "ukf_auto_bias_constraints": True,
                "ukf_bias_freeze_relative_information": 1e-13,
                "ukf_bias_regularize_relative_information": 1e-6,
                "ukf_bias_regularization_std": 2.5,
                "ukf_process_noise_model": "continuous_white_acceleration",
                "ukf_acceleration_psd_m2_s3": 1e-10,
                "ukf_adaptive_process_noise": True,
                "ukf_initial_process_noise_scale": 2.0,
                "ukf_min_process_noise_scale": 0.5,
                "ukf_max_process_noise_scale": 50.0,
                "ukf_process_noise_adaptation_gain": 0.3,
                "ukf_covariance_form": "square_root",
            }
        )

        transform, adaptive = scenario_ukf_configs(config)

        self.assertEqual(transform.alpha, 0.2)
        self.assertEqual(transform.beta, 2.5)
        self.assertEqual(transform.kappa, 1.0)
        self.assertEqual(adaptive.covariance_inflation, 1.01)
        self.assertTrue(adaptive.adaptive_measurement_noise)
        self.assertEqual(adaptive.max_measurement_noise_scale, 25.0)
        self.assertEqual(adaptive.nis_gate, 16.0)
        self.assertEqual(adaptive.component_nis_gate, 9.0)
        self.assertEqual(adaptive.component_gate_mode, "conditional")
        self.assertTrue(adaptive.robust_measurement_update)
        self.assertEqual(adaptive.robust_loss, "huber")
        self.assertEqual(adaptive.robust_student_t_dof, 7.0)
        self.assertEqual(adaptive.robust_huber_threshold, 2.5)
        self.assertEqual(adaptive.robust_min_component_weight, 0.2)
        physics = scenario_range_rate_physics_config(config)
        self.assertEqual(physics.local_state_model, "taylor3")
        self.assertEqual(physics.station_clock_offset_s, 2e-6)
        self.assertEqual(physics.station_clock_drift, 1e-10)
        self.assertEqual(physics.transponder_delay_s, 3e-6)
        self.assertTrue(config.ukf_auto_bias_constraints)
        self.assertEqual(config.ukf_bias_freeze_relative_information, 1e-13)
        self.assertEqual(config.ukf_bias_regularize_relative_information, 1e-6)
        self.assertEqual(config.ukf_bias_regularization_std, 2.5)
        self.assertTrue(adaptive.adaptive_process_noise)
        self.assertEqual(adaptive.initial_process_noise_scale, 2.0)
        self.assertEqual(adaptive.max_process_noise_scale, 50.0)
        self.assertEqual(config.ukf_covariance_form, "square_root")

    def test_invalid_ukf_tuning_is_rejected(self):
        base = {
            "name": "bad_ukf",
            "measurement_type": "range_rate",
            "estimator_type": "ukf",
            "start_mode": "hot",
            "network": "multi",
        }

        with self.assertRaises(ValueError):
            scenario_config_from_mapping({**base, "ukf_alpha": 0.0})
        with self.assertRaises(ValueError):
            scenario_config_from_mapping({**base, "ukf_covariance_inflation": 0.99})
        with self.assertRaises(ValueError):
            scenario_config_from_mapping({**base, "ukf_nis_gate": -1.0})
        with self.assertRaises(ValueError):
            scenario_config_from_mapping({**base, "ukf_adaptive_measurement_noise": "false"})
        with self.assertRaises(ValueError):
            scenario_config_from_mapping({**base, "ukf_adaptive_process_noise": True})
        with self.assertRaises(ValueError):
            scenario_config_from_mapping({**base, "ukf_robust_loss": "cauchy"})
        with self.assertRaises(ValueError):
            scenario_config_from_mapping({**base, "ukf_robust_student_t_dof": 2.0})
        with self.assertRaises(ValueError):
            scenario_config_from_mapping({**base, "ukf_robust_min_component_weight": 0.0})
        with self.assertRaises(ValueError):
            scenario_config_from_mapping(
                {
                    **base,
                    "ukf_bias_freeze_relative_information": 1e-3,
                    "ukf_bias_regularize_relative_information": 1e-6,
                }
            )

    def test_invalid_config_cross_field_rules_are_rejected(self):
        base = {
            "name": "bad",
            "measurement_type": "position",
            "estimator_type": "bls_lm",
            "start_mode": "sqrt_formal",
            "network": "single",
        }

        with self.assertRaises(ValueError):
            scenario_config_from_mapping(base)

        base["start_mode"] = "hot"
        base["bias_mode"] = "global"
        with self.assertRaises(ValueError):
            scenario_config_from_mapping(base)

        base["measurement_type"] = "range_rate"
        base["estimator_type"] = "ukf"
        config = scenario_config_from_mapping(base)
        self.assertEqual(config.estimator_type, "ukf")

        base["measurement_type"] = "position"
        config = scenario_config_from_mapping(base)
        self.assertEqual(config.bias_mode, "global")

        base["bias_mode"] = None
        base["measurement_type"] = "position"
        base["estimator_type"] = "bls_lm"
        base["range_rate_physics"] = "two_way_counted_doppler"
        with self.assertRaises(ValueError):
            scenario_config_from_mapping(base)

    def test_aberration_corrections_default_off_and_roundtrip(self):
        base = {
            "name": "aberr",
            "measurement_type": "position",
            "estimator_type": "bls_lm",
            "start_mode": "cold",
            "network": "multi",
        }
        # defaults: corrections off, frame local_mci (backward compatible)
        config = scenario_config_from_mapping(base)
        self.assertFalse(config.apply_light_time)
        self.assertFalse(config.apply_stellar_aberration)
        self.assertEqual(config.stellar_aberration_model, "local_mci")

        # full SPICE-like CN+S round-trips through the mapping
        config = scenario_config_from_mapping(
            {
                **base,
                "apply_light_time": True,
                "apply_stellar_aberration": True,
                "stellar_aberration_model": "spice_ssb",
            }
        )
        self.assertTrue(config.apply_light_time)
        self.assertTrue(config.apply_stellar_aberration)
        self.assertEqual(config.stellar_aberration_model, "spice_ssb")

        # schema advertises the new keys
        props = scenario_config_schema()["properties"]
        self.assertIn("apply_light_time", props)
        self.assertEqual(props["stellar_aberration_model"]["enum"], ["local_mci", "spice_ssb"])

    def test_aberration_corrections_cross_field_rules(self):
        base = {
            "name": "aberr_bad",
            "measurement_type": "position",
            "estimator_type": "bls_lm",
            "start_mode": "cold",
            "network": "multi",
        }
        # stellar aberration requires light time
        with self.assertRaises(ValueError):
            scenario_config_from_mapping({**base, "apply_stellar_aberration": True})

        # corrections apply only to position measurements
        with self.assertRaises(ValueError):
            scenario_config_from_mapping(
                {**base, "measurement_type": "range_rate", "apply_light_time": True}
            )

        # valid: light time alone on position measurements
        config = scenario_config_from_mapping({**base, "apply_light_time": True})
        self.assertTrue(config.apply_light_time)
        self.assertFalse(config.apply_stellar_aberration)

    def test_json_load_and_normalized_write_roundtrip(self):
        payload = {
            "name": "json_demo",
            "measurement_type": "position",
            "estimator_type": "srif",
            "start_mode": "cold",
            "network": "single",
            "duration_h": 2.5,
            "range_rate_physics": "geometric_instantaneous",
            "count_interval_s": 30.0,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "scenario.json"
            output_path = tmp_path / "normalized.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")

            config = load_scenario_config_json(input_path)
            write_normalized_scenario_config(config, output_path)
            normalized = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(normalized["name"], "json_demo")
        self.assertEqual(normalized["sample_step_s"], 240.0)
        self.assertEqual(normalized["count_interval_s"], 30.0)

    def test_cli_prints_schema_and_validates_config(self):
        script_path = Path(__file__).resolve().parents[1] / "examples" / "scenario_config_cli.py"
        payload = {
            "name": "cli_demo",
            "measurement_type": "range_rate",
            "estimator_type": "srif",
            "start_mode": "hot",
            "network": "multi",
            "noise": True,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "scenario.json"
            normalized_path = tmp_path / "normalized.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")

            schema_run = subprocess.run(
                [sys.executable, str(script_path), "--schema"],
                text=True,
                capture_output=True,
                check=False,
            )
            config_run = subprocess.run(
                [sys.executable, str(script_path), str(input_path), "--write-normalized", str(normalized_path)],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(schema_run.returncode, 0, schema_run.stderr)
            self.assertEqual(config_run.returncode, 0, config_run.stderr)
            self.assertIn("measurement_type", schema_run.stdout)
            self.assertIn("cli_demo", config_run.stdout)
            self.assertTrue(normalized_path.is_file())

    def test_run_scenario_config_cli_dry_run_accepts_ukf(self):
        script_path = Path(__file__).resolve().parents[1] / "examples" / "run_scenario_config.py"
        payload = {
            "name": "ukf_cli_demo",
            "measurement_type": "range_rate",
            "estimator_type": "ukf",
            "start_mode": "hot",
            "network": "multi",
            "duration_h": 0.25,
            "sample_step_s": 300.0,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "scenario.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            run = subprocess.run(
                [sys.executable, str(script_path), str(input_path), "--dry-run"],
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("ukf_cli_demo", run.stdout)
        self.assertIn("Would write", run.stdout)

    def test_spice_mismatch_campaign_dry_run_accepts_ukf_config(self):
        script_path = Path(__file__).resolve().parents[1] / "examples" / "ukf_spice_mismatch_campaign.py"
        payload = {
            "name": "ukf_spice_demo",
            "measurement_type": "range_rate",
            "estimator_type": "ukf",
            "start_mode": "hot",
            "network": "multi",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "scenario.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            run = subprocess.run(
                [sys.executable, str(script_path), str(input_path), "--dry-run"],
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("Cases:", run.stdout)
        self.assertIn("spice_mismatch", run.stdout)

    def test_ukf_stress_monte_carlo_campaign_dry_run_accepts_ukf_config(self):
        script_path = Path(__file__).resolve().parents[1] / "examples" / "ukf_stress_monte_carlo_campaign.py"
        payload = {
            "name": "ukf_stress_demo",
            "measurement_type": "range_rate",
            "estimator_type": "ukf",
            "start_mode": "hot",
            "network": "multi",
            "range_rate_physics": "two_way_counted_doppler",
            "bias_mode": "station_full",
            "ukf_covariance_form": "square_root",
            "ukf_auto_bias_constraints": True,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "scenario.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            run = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    str(input_path),
                    "--trials",
                    "3",
                    "--earth-position-bias-m",
                    "0,100",
                    "--cold-start-scale",
                    "1",
                    "--dry-run",
                ],
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("Trials per case: 3", run.stdout)
        self.assertIn("earth_dx_100m_cold_1x", run.stdout)


if __name__ == "__main__":
    unittest.main()
