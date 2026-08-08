import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

import lunar_od.scenario_config as scenario_config_module

from lunar_od import (
    EXACT_EVENT_EPOCH_STATION_METHOD,
    LEGACY_INTERPOLATED_STATION_METHOD,
    RangeRatePhysicsConfig,
    make_exact_counted_doppler_station_state_provider,
    two_way_counted_doppler_observable,
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
        self.assertEqual(config.measurement_model_profile, "geometric_instantaneous")
        self.assertEqual(config.companion_geometry, "instantaneous")
        self.assertEqual(config.jacobian_model, "analytic_exact_geometric")
        self.assertEqual(config.ukf_alpha, 0.35)
        self.assertEqual(config.ukf_covariance_form, "square_root")
        self.assertIsNone(config.ukf_nis_gate)
        self.assertIn("multi range_rate srif/hot", scenario_config_summary(config))

    def test_schema_lists_required_fields_and_enums(self):
        schema = scenario_config_schema()

        self.assertIn("measurement_type", schema["required"])
        self.assertEqual(
            schema["properties"]["measurement_type"]["enum"],
            ["position", "range_rate", "two_way_range"],
        )
        self.assertIn("ukf", schema["properties"]["estimator_type"]["enum"])
        self.assertEqual(
            schema["properties"]["range_rate_physics"]["enum"],
            ["geometric_instantaneous", "two_way_counted_doppler"],
        )
        self.assertEqual(
            schema["properties"]["measurement_model_profile"]["enum"],
            [
                "geometric_instantaneous",
                "one_way_light_time",
                "one_way_light_time_aberrated_local_mci",
                "one_way_light_time_aberrated_spice_ssb",
            ],
        )
        self.assertEqual(
            schema["properties"]["companion_geometry"]["enum"],
            ["instantaneous", "apparent_one_way"],
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

    def test_measurement_model_profile_and_companion_geometry_rules(self):
        position_base = {
            "name": "profile_demo",
            "measurement_type": "position",
            "estimator_type": "srif",
            "start_mode": "cold",
            "network": "multi",
        }
        config = scenario_config_from_mapping(
            {
                **position_base,
                "measurement_model_profile": "one_way_light_time_aberrated_local_mci",
                "jacobian_model": "analytic_first_order_light_time",
            }
        )
        self.assertEqual(config.measurement_model_profile, "one_way_light_time_aberrated_local_mci")
        self.assertEqual(config.jacobian_model, "analytic_first_order_light_time")
        self.assertIn("profile=one_way_light_time_aberrated_local_mci", scenario_config_summary(config))

        implicit = scenario_config_from_mapping(
            {
                **position_base,
                "measurement_model_profile": "one_way_light_time",
                "jacobian_model": "implicit_light_time",
            }
        )
        self.assertEqual(implicit.jacobian_model, "implicit_light_time")
        with self.assertRaises(ValueError):
            scenario_config_from_mapping(
                {**position_base, "jacobian_model": "implicit_light_time"}
            )

        range_rate_base = {
            **position_base,
            "measurement_type": "range_rate",
            "estimator_type": "ukf",
            "start_mode": "hot",
        }
        config = scenario_config_from_mapping(
            {**range_rate_base, "companion_geometry": "apparent_one_way"}
        )
        self.assertEqual(config.companion_geometry, "apparent_one_way")
        self.assertIn("companion=apparent_one_way", scenario_config_summary(config))

        with self.assertRaises(ValueError):
            scenario_config_from_mapping(
                {**range_rate_base, "measurement_model_profile": "one_way_light_time"}
            )
        with self.assertRaises(ValueError):
            scenario_config_from_mapping(
                {**position_base, "companion_geometry": "apparent_one_way"}
            )

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


# ---------------------------------------------------------------------------
# R3 exact event-epoch station transform — configuration and disclosure gates
#
# R3-P15 configuration, schema, default and reporting provenance
# R3-P23 the default change is deliberate, detectable and documented
# ---------------------------------------------------------------------------


def _r3_scenario_payload(**overrides):
    payload = {
        "name": "r3-scenario",
        "measurement_type": "range_rate",
        "estimator_type": "bls_lm",
        "start_mode": "cold",
        "network": "multi",
    }
    payload.update(overrides)
    return payload


class R3ScenarioConfigurationContract(unittest.TestCase):
    """R3-P15: every supported route executes and records the method."""

    def test_p15_omitted_field_defaults_to_the_exact_method(self):
        config = scenario_config_from_mapping(_r3_scenario_payload())
        self.assertEqual(config.station_state_method, EXACT_EVENT_EPOCH_STATION_METHOD)
        physics = scenario_range_rate_physics_config(config)
        self.assertEqual(
            physics.station_state_method, EXACT_EVENT_EPOCH_STATION_METHOD
        )
        self.assertTrue(physics.exact_event_epoch_enabled)
        self.assertFalse(physics.legacy_compatibility_mode)

    def test_p15_explicit_exact_is_accepted_and_reported(self):
        config = scenario_config_from_mapping(
            _r3_scenario_payload(
                range_rate_physics="two_way_counted_doppler",
                station_state_method=EXACT_EVENT_EPOCH_STATION_METHOD,
            )
        )
        physics = scenario_range_rate_physics_config(config)
        self.assertTrue(physics.exact_event_epoch_enabled)
        self.assertEqual(
            physics.station_velocity_model, "exact_event_epoch_sxform"
        )

    def test_p15_explicit_legacy_is_accepted_and_warns_once(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = scenario_config_from_mapping(
                _r3_scenario_payload(
                    range_rate_physics="two_way_counted_doppler",
                    station_state_method=LEGACY_INTERPOLATED_STATION_METHOD,
                )
            )
        deprecations = [
            item for item in caught if issubclass(item.category, DeprecationWarning)
        ]
        self.assertEqual(len(deprecations), 1)
        self.assertIn(
            "EXACT_STATION_TRANSFORM_UPGRADE_REQUIRED", str(deprecations[0].message)
        )
        physics = scenario_range_rate_physics_config(config)
        self.assertTrue(physics.legacy_compatibility_mode)
        self.assertEqual(physics.station_velocity_model, "interpolated_sxform_grid")

    def test_p15_json_schema_accepts_both_values_and_rejects_a_third(self):
        schema = scenario_config_schema()
        enum = schema["properties"]["station_state_method"]["enum"]
        self.assertEqual(
            sorted(enum),
            sorted(
                [
                    EXACT_EVENT_EPOCH_STATION_METHOD,
                    LEGACY_INTERPOLATED_STATION_METHOD,
                ]
            ),
        )
        self.assertEqual(
            schema["properties"]["station_state_method"]["default"],
            EXACT_EVENT_EPOCH_STATION_METHOD,
        )
        with self.assertRaises(ValueError):
            scenario_config_from_mapping(
                _r3_scenario_payload(station_state_method="something_else")
            )

    def test_p15_legacy_requires_the_counted_doppler_physics(self):
        """F12 cross-field rule: legacy is meaningless without counted Doppler."""
        with self.assertRaises(ValueError) as ctx:
            scenario_config_from_mapping(
                _r3_scenario_payload(
                    station_state_method=LEGACY_INTERPOLATED_STATION_METHOD
                )
            )
        self.assertIn("only", str(ctx.exception).lower())

    def test_p15_no_cli_or_desktop_source_change_was_required(self):
        """Both entry points route through scenario_range_rate_physics_config."""
        root = Path(scenario_config_module.__file__).resolve().parents[1]
        for relative in (
            "examples/run_scenario_config.py",
            "desktop_app/controllers/analysis_controller.py",
        ):
            source = (root / relative).read_text(encoding="utf-8")
            self.assertNotIn("station_state_method", source, relative)


class R3DefaultChangeDisclosure(unittest.TestCase):
    """R3-P23: a pre-R3 scenario file changes behaviour, visibly."""

    def test_p23_pre_r3_scenario_without_the_field_selects_exact(self):
        """The gate asserts the change is disclosed, not that it is prevented."""
        pre_r3 = _r3_scenario_payload(
            range_rate_physics="two_way_counted_doppler", count_interval_s=60.0
        )
        self.assertNotIn("station_state_method", pre_r3)
        config = scenario_config_from_mapping(pre_r3)
        self.assertEqual(config.station_state_method, EXACT_EVENT_EPOCH_STATION_METHOD)

    def test_p23_the_two_methods_produce_a_different_observable(self):
        """Old files change behaviour: that is the accepted R2 consequence."""
        t_grid = np.linspace(-400.0, 400.0, 81)
        earth_pos = np.zeros((t_grid.size, 3))
        earth_vel = np.zeros((t_grid.size, 3))
        earth_pos[:, 0] = 3.8e8
        omega = 7.292115e-5

        def sxform_fn(_source, _target, et):
            theta = omega * float(et)
            c, s = np.cos(theta), np.sin(theta)
            rot = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
            rot_dot = omega * np.array(
                [[-s, c, 0.0], [-c, -s, 0.0], [0.0, 0.0, 0.0]]
            )
            xform = np.zeros((6, 6))
            xform[:3, :3] = rot
            xform[3:, :3] = rot_dot
            xform[3:, 3:] = rot
            return xform

        xforms = np.array([sxform_fn("J2000", "ITRF93", float(t)) for t in t_grid])
        states = np.zeros((t_grid.size, 6))
        states[:, 0] = 2.0e6 + 90.0 * t_grid
        states[:, 1] = 5.0e5 + 1400.0 * t_grid
        states[:, 3] = 90.0
        states[:, 4] = 1400.0

        class _Site:
            name = "R3 disclosure site"
            r_ecef_m = np.array([6378137.0, 0.0, 0.0])

        common = dict(
            mode="two_way_counted_doppler",
            count_interval_s=60.0,
            light_time_tolerance_s=1e-12,
            light_time_equation_tolerance_s=1e-11,
            light_time_max_iter=30,
        )
        exact_value = two_way_counted_doppler_observable(
            0.0, _Site(), t_grid, states, earth_pos, earth_vel, xforms,
            RangeRatePhysicsConfig(
                **common, station_state_method=EXACT_EVENT_EPOCH_STATION_METHOD
            ),
            station_state_provider=make_exact_counted_doppler_station_state_provider(
                _Site(), 0.0, t_grid, earth_pos, earth_vel, sxform_fn=sxform_fn
            ),
        )
        legacy_value = two_way_counted_doppler_observable(
            0.0, _Site(), t_grid, states, earth_pos, earth_vel, xforms,
            RangeRatePhysicsConfig(
                **common, station_state_method=LEGACY_INTERPOLATED_STATION_METHOD
            ),
        )
        self.assertNotEqual(exact_value, legacy_value)
        self.assertTrue(np.isfinite(exact_value))

    def test_p23_documentation_states_that_old_files_change_behaviour(self):
        root = Path(scenario_config_module.__file__).resolve().parents[1]
        doc = (root / "docs" / "two_way_counted_doppler.md").read_text(encoding="utf-8")
        lowered = doc.lower()
        self.assertIn("station_state_method", lowered)
        self.assertIn("exact_event_epoch_sxform", lowered)
        self.assertIn("legacy_interpolated_transform_grid", lowered)
        self.assertIn("automatically", lowered)


class R4CountedDopplerModelSelection(unittest.TestCase):
    """R4-P16: the counted-Doppler event model is explicit opt-in and fails closed."""

    BASE = {
        "name": "r4",
        "measurement_type": "range_rate",
        "estimator_type": "bls_lm",
        "start_mode": "cold",
        "network": "multi",
    }
    COUNTED = "two_way_counted_doppler"

    def _config(self, **overrides):
        payload = dict(self.BASE)
        payload.update(overrides)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return scenario_config_from_mapping(payload)

    def test_omitted_field_keeps_the_accepted_r3_model(self):
        """A pre-R4 scenario file must behave exactly as before (owner Q4)."""
        self.assertEqual(
            self._config().counted_doppler_model, "single_bounce_exact_station"
        )

    def test_explicit_four_event_is_accepted_with_counted_doppler(self):
        config = self._config(
            range_rate_physics=self.COUNTED, counted_doppler_model="four_event_delay"
        )
        self.assertEqual(config.counted_doppler_model, "four_event_delay")

    def test_four_event_requires_counted_doppler_physics(self):
        with self.assertRaises(ValueError) as ctx:
            self._config(counted_doppler_model="four_event_delay")
        self.assertIn("two_way_counted_doppler", str(ctx.exception))

    def test_four_event_rejects_the_legacy_station_grid(self):
        with self.assertRaises(ValueError) as ctx:
            self._config(
                range_rate_physics=self.COUNTED,
                counted_doppler_model="four_event_delay",
                station_state_method="legacy_interpolated_transform_grid",
            )
        self.assertIn("exact_event_epoch_sxform", str(ctx.exception))

    def test_four_event_is_rejected_for_the_ukf(self):
        """Four-event SR-UKF support is deferred (R4-FUTURE-UKF-FOUR-EVENT)."""
        with self.assertRaises(ValueError) as ctx:
            self._config(
                range_rate_physics=self.COUNTED,
                counted_doppler_model="four_event_delay",
                estimator_type="ukf",
            )
        self.assertIn("ukf", str(ctx.exception).lower())

    def test_r3_model_still_rejects_a_nonzero_delay(self):
        """The P0A gate is narrowed by model selection, never deleted."""
        with self.assertRaises(ValueError) as ctx:
            self._config(range_rate_physics=self.COUNTED, transponder_delay_s=1e-3)
        self.assertIn("single-bounce counted-Doppler model", str(ctx.exception))

    def test_four_event_permits_a_nonzero_delay(self):
        config = self._config(
            range_rate_physics=self.COUNTED,
            counted_doppler_model="four_event_delay",
            transponder_delay_s=1e-3,
        )
        self.assertEqual(config.transponder_delay_s, 1e-3)

    def test_unknown_model_value_is_rejected(self):
        with self.assertRaises(ValueError):
            self._config(
                range_rate_physics=self.COUNTED, counted_doppler_model="four_event"
            )

    def test_schema_publishes_the_enum_and_the_r3_default(self):
        schema = scenario_config_schema()["properties"]["counted_doppler_model"]
        self.assertEqual(
            schema["enum"], ["single_bounce_exact_station", "four_event_delay"]
        )
        self.assertEqual(schema["default"], "single_bounce_exact_station")

    def test_no_delay_drift_field_is_introduced(self):
        """Owner decision Q1: drifting delay is out of R4 scope."""
        config = self._config(range_rate_physics=self.COUNTED)
        self.assertFalse(hasattr(config, "transponder_delay_rate_s_per_s"))
        self.assertFalse(hasattr(config, "transponder_delay_reference_epoch_s"))
        self.assertNotIn(
            "transponder_delay_rate_s_per_s", scenario_config_schema()["properties"]
        )

    def test_model_selection_reaches_the_range_rate_physics_config(self):
        physics = scenario_range_rate_physics_config(
            self._config(
                range_rate_physics=self.COUNTED,
                counted_doppler_model="four_event_delay",
            )
        )
        self.assertEqual(physics.counted_doppler_model, "four_event_delay")
        self.assertTrue(physics.four_event_enabled)
        self.assertEqual(
            physics.counted_doppler_model_version,
            "r4.counted-doppler.four-event-delay.v1",
        )
