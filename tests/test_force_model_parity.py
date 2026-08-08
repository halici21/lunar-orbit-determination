"""R0B-2 parity enforcement, mismatch policy, and reporting tests (P1-P12).

Deterministic and fast: no SPICE, no real propagation, no RNG.
"""

from __future__ import annotations
from lunar_od import reporting as reporting_module

import csv
import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from lunar_od.force_contract import (
    FORCE_CONTRACT_SCHEMA_VERSION,
    ConsumerReadiness,
    ConsumerRole,
    ForceContractError,
    ForceModelMismatchPolicy,
    ForceModelParityError,
    evaluate_force_model_parity,
    force_model_manifest,
    manifest_canonical_bytes,
    manifest_sha256,
    scenario_result_force_fields,
)
from lunar_od.reporting import write_force_model_manifest, write_scenario_summary_csv
from lunar_od.scenario_config import (
    force_execution_spec_from_scenario_config,
    force_model_contract_from_scenario_config,
    scenario_config_from_mapping,
    scenario_force_model_preflight,
)


def _spec_for(config, **kwargs):
    return force_execution_spec_from_scenario_config(config, **kwargs)
from lunar_od.scenarios import ScenarioResult

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GRAVITY_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "gravity" / "synthetic_norm_sha.tab"
)
J2_MOON = 2.0346e-4

_RUNNER_PATH = _REPO_ROOT / "examples" / "run_scenario_config.py"
_spec = importlib.util.spec_from_file_location("r0b_runner_under_test", _RUNNER_PATH)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

_BASE_PAYLOAD = {
    "name": "r0b_parity",
    "measurement_type": "range_rate",
    "estimator_type": "bls_lm",
    "start_mode": "cold",
    "network": "multi",
}


def _config(**overrides):
    return scenario_config_from_mapping({**_BASE_PAYLOAD, **overrides})


def _contract(config=None, **kwargs):
    return force_model_contract_from_scenario_config(config or _config(), **kwargs)


def _harmonics_config(base=None):
    return dataclasses.replace(
        base or _config(),
        enable_lunar_harmonics=True,
        lunar_gravity_model_path=str(_GRAVITY_FIXTURE),
        lunar_gravity_nmax=4,
        lunar_gravity_mmax=4,
    )


class P1MatchedParityTests(unittest.TestCase):
    def test_default_scenario_is_matched(self):
        decision = scenario_force_model_preflight(_config())
        self.assertTrue(decision.match)
        self.assertEqual(decision.truth_fingerprint, decision.estimator_fingerprint)
        self.assertFalse(decision.explicit_mismatch)

    def test_nonzero_lunar_j2_scenario_is_matched(self):
        decision = scenario_force_model_preflight(_config(j2_moon=J2_MOON))
        self.assertTrue(decision.match)
        self.assertEqual(decision.schema_version, FORCE_CONTRACT_SCHEMA_VERSION)


class P2AccidentalMismatchTests(unittest.TestCase):
    def test_accidental_mismatch_is_rejected_before_any_execution(self):
        config = _config()
        truth_spec = _spec_for(_config(j2_moon=J2_MOON))
        with self.assertRaises(ForceModelParityError) as caught:
            scenario_force_model_preflight(config, truth_spec=truth_spec)
        self.assertIn("does not match", str(caught.exception))

    def test_runner_rejects_mismatch_with_zero_execution_calls(self):
        config = _config()
        spies = {
            name: mock.MagicMock()
            for name in (
                "spice_load", "truth", "state", "stm", "ukf", "fast_sigma",
                "bls", "srif", "batch",
            )
        }
        fixture_read = mock.MagicMock(side_effect=AssertionError("fixture was read"))
        with mock.patch.object(runner, "load_spice_kernels", spies["spice_load"]), \
                mock.patch.object(runner, "propagate_truth_with_ephemeris", spies["truth"]), \
                mock.patch("lunar_od.dynamics.propagate_state", spies["state"]), \
                mock.patch("lunar_od.dynamics.propagate_augmented_state", spies["stm"]), \
                mock.patch("lunar_od.scenarios.run_lunar_ukf", spies["ukf"]), \
                mock.patch("lunar_od.scenarios.make_fast_sigma_propagator", spies["fast_sigma"]), \
                mock.patch("lunar_od.scenarios.estimate_position_bls_lm", spies["bls"]), \
                mock.patch("lunar_od.scenarios.estimate_position_srif", spies["srif"]), \
                mock.patch("lunar_od.run_batch_arc_sequence", spies["batch"]), \
                mock.patch.object(Path, "read_text", fixture_read):
            # truth J2 nonzero vs estimator (config) J2 zero, no opt-in ->
            # rejected in Stage A, before the fixture is ever read.
            with self.assertRaises(ForceModelParityError):
                runner.run_configured_scenario(config, truth_j2_moon=J2_MOON)
        for name, spy in spies.items():
            self.assertEqual(spy.call_count, 0, f"{name} ran before the parity gate")
        self.assertEqual(fixture_read.call_count, 0)


class P3ExplicitMismatchTests(unittest.TestCase):
    def test_explicit_mismatch_runs_and_is_reported(self):
        config = _config(
            allow_explicit_force_model_mismatch=True,
            force_model_mismatch_reason="R0B truth-J2 vs point-mass estimator campaign",
        )
        truth_spec = _spec_for(_config(j2_moon=J2_MOON))
        decision = scenario_force_model_preflight(config, truth_spec=truth_spec)
        self.assertFalse(decision.match)
        self.assertTrue(decision.explicit_mismatch)
        self.assertNotEqual(decision.truth_fingerprint, decision.estimator_fingerprint)
        fields = scenario_result_force_fields(decision)
        self.assertEqual(
            fields["force_model_mismatch_reason"],
            "R0B truth-J2 vs point-mass estimator campaign",
        )
        self.assertFalse(fields["force_model_match"])
        self.assertTrue(fields["explicit_force_model_mismatch"])
        self.assertEqual(
            decision.manifest["mismatch_policy"]["reason"],
            "R0B truth-J2 vs point-mass estimator campaign",
        )


class P4MissingReasonTests(unittest.TestCase):
    def test_optin_without_reason_is_rejected(self):
        for reason in (None, "", "   "):
            with self.subTest(reason=reason):
                with self.assertRaises(ForceModelParityError):
                    ForceModelMismatchPolicy(enabled=True, reason=reason)

    def test_config_optin_without_reason_fails_preflight_with_zero_calls(self):
        config = _config(allow_explicit_force_model_mismatch=True)
        spies = {
            name: mock.MagicMock()
            for name in ("truth", "state", "stm", "ukf", "fast_sigma", "bls", "srif")
        }
        with mock.patch("lunar_od.dynamics.propagate_state", spies["state"]), \
                mock.patch("lunar_od.dynamics.propagate_augmented_state", spies["stm"]), \
                mock.patch("lunar_od.scenarios.run_lunar_ukf", spies["ukf"]), \
                mock.patch("lunar_od.scenarios.make_fast_sigma_propagator", spies["fast_sigma"]), \
                mock.patch("lunar_od.scenarios.estimate_position_bls_lm", spies["bls"]), \
                mock.patch("lunar_od.scenarios.estimate_position_srif", spies["srif"]):
            with self.assertRaises(ForceModelParityError):
                scenario_force_model_preflight(config)
        for name, spy in spies.items():
            self.assertEqual(spy.call_count, 0, name)


class P5EarthJ2Tests(unittest.TestCase):
    def test_loader_still_rejects_earth_j2(self):
        with self.assertRaisesRegex(
            ValueError, "enable_earth_j2=True is not supported on the official OD path"
        ):
            _config(enable_earth_j2=True)

    def test_mismatch_optin_cannot_bypass_earth_j2_capability(self):
        # Direct capability-gate check: an Earth-J2 estimator contract is
        # unsupported and an explicit-mismatch opt-in cannot bypass it. (The
        # official runner rejects Earth J2 even earlier via the R0A gate; this
        # asserts the parity-layer capability gate itself.)
        earth_config = dataclasses.replace(_config(), enable_earth_j2=True)
        earth_contract = force_model_contract_from_scenario_config(earth_config)
        policy = ForceModelMismatchPolicy(enabled=True, reason="attempted bypass")
        with self.assertRaises(ForceModelParityError) as caught:
            evaluate_force_model_parity(
                _contract(), earth_contract, policy, context="bypass attempt"
            )
        self.assertIn("cannot be bypassed", str(caught.exception))

    def test_official_runner_earth_j2_optin_still_rejected(self):
        config = dataclasses.replace(
            _config(),
            enable_earth_j2=True,
            allow_explicit_force_model_mismatch=True,
            force_model_mismatch_reason="attempted bypass",
        )
        with self.assertRaisesRegex(ValueError, "not supported on the official OD path"):
            scenario_force_model_preflight(config)


class P6HarmonicsStatusTests(unittest.TestCase):
    def test_official_harmonics_is_fail_closed_even_with_optin(self):
        # R0B-F1/V06: official harmonics execution is not implemented, so the
        # preflight rejects it (opt-in or not) rather than reporting physics it
        # never ran.
        config = _harmonics_config(
            _config(
                allow_explicit_force_model_mismatch=True,
                force_model_mismatch_reason="harmonics estimator attempt",
            )
        )
        with self.assertRaises(ForceContractError) as caught:
            scenario_force_model_preflight(config)
        self.assertIn("EXPERIMENTAL", str(caught.exception))
        self.assertIn("not implemented", str(caught.exception))

    def test_harmonics_config_without_optin_also_rejected(self):
        with self.assertRaises(ForceContractError):
            scenario_force_model_preflight(_harmonics_config())

    def test_harmonics_truth_capability_is_experimental_direct_trajectory_only(self):
        contract = _contract(_harmonics_config())
        caps = contract.consumer_capabilities
        self.assertEqual(
            caps[ConsumerRole.TRUTH_STATE],
            ConsumerReadiness.EXPERIMENTAL_DIRECT_TRAJECTORY_ONLY,
        )
        for role in (
            ConsumerRole.ESTIMATOR_STATE,
            ConsumerRole.ESTIMATOR_STM,
            ConsumerRole.UKF_STANDARD,
            ConsumerRole.UKF_SQUARE_ROOT,
            ConsumerRole.UKF_FAST_SIGMA,
            ConsumerRole.POSTERIOR_COVARIANCE,
            ConsumerRole.OBSERVABILITY,
        ):
            self.assertEqual(caps[role], ConsumerReadiness.UNSUPPORTED, role)


class P7VerifiedR1Tests(unittest.TestCase):
    def test_nonzero_lunar_j2_verifies_posterior_and_observability(self):
        decision = scenario_force_model_preflight(_config(j2_moon=J2_MOON))
        fields = scenario_result_force_fields(decision)
        self.assertEqual(fields["posterior_force_role_status"], "verified")
        self.assertEqual(fields["observability_force_role_status"], "verified")


class P8SummaryCsvTests(unittest.TestCase):
    EXPECTED_R0B_COLUMNS = [
        "force_contract_schema_version",
        "truth_force_fingerprint",
        "estimator_force_fingerprint",
        "force_model_match",
        "explicit_force_model_mismatch",
        "force_model_mismatch_reason",
        "force_contract_manifest_sha256",
        "posterior_force_role_status",
        "observability_force_role_status",
    ]
    EXPECTED_R1_COLUMNS = [
        "posterior_covariance_stm_mode",
        "posterior_covariance_rank",
        "posterior_covariance_condition_number",
        "posterior_covariance_min_eigenvalue",
        "posterior_covariance_finite",
        "observability_rank",
        "observability_condition_number",
        "observability_singular_value_min",
        "observability_singular_value_max",
        "observability_finite",
        "condition_number_available",
        "derivative_validation_profile",
    ]

    def _scenario(self):
        from tests.test_reporting import _arc_result

        decision = scenario_force_model_preflight(_config(j2_moon=J2_MOON))
        scenario = ScenarioResult(
            label="csv",
            measurement_type="range_rate",
            start_mode="cold",
            arc_results=(_arc_result(1, 1000.0, 25.0),),
            estimator_type="bls_lm",
            **scenario_result_force_fields(decision),
        )
        return scenario, decision

    def _write(self, scenario):
        with tempfile.TemporaryDirectory() as tmp:
            out = write_scenario_summary_csv([scenario], Path(tmp) / "summary.csv")
            with out.open(newline="", encoding="utf-8") as handle:
                return list(csv.reader(handle))

    def test_new_columns_are_appended_after_history_domain(self):
        scenario, _ = self._scenario()
        rows = self._write(scenario)
        header = rows[0]
        # Owner Addendum 03 Decision 4: R1 columns keep their identity and
        # order; the R3 measurement-provenance columns are appended after them.
        # R4 (Owner Addendum 03) appends its own provenance block after the
        # complete R3 block, so the tail is
        # R1_COLUMNS + R3_MEASUREMENT_PROVENANCE_COLUMNS
        #            + R4_MEASUREMENT_PROVENANCE_COLUMNS.
        r3_columns = list(reporting_module.R3_MEASUREMENT_PROVENANCE_COLUMNS)
        r4_columns = list(reporting_module.R4_MEASUREMENT_PROVENANCE_COLUMNS)
        expected_tail = [*self.EXPECTED_R1_COLUMNS, *r3_columns, *r4_columns]
        self.assertEqual(header[-len(expected_tail):], expected_tail)
        self.assertEqual(
            header[-len(expected_tail) : -(len(r3_columns) + len(r4_columns))],
            self.EXPECTED_R1_COLUMNS,
        )
        self.assertEqual(
            header[-(len(r3_columns) + len(r4_columns)) : -len(r4_columns)],
            r3_columns,
        )
        self.assertEqual(header[-len(r4_columns):], r4_columns)
        self.assertEqual(len(header), len(set(header)))
        history_idx = header.index("history_domain_all_measurement_arcs_empty")
        r0b_slice = header[
            history_idx + 1 : history_idx + 1 + len(self.EXPECTED_R0B_COLUMNS)
        ]
        self.assertEqual(r0b_slice, self.EXPECTED_R0B_COLUMNS)
        self.assertEqual(
            header[history_idx + 1 + len(self.EXPECTED_R0B_COLUMNS)],
            self.EXPECTED_R1_COLUMNS[0],
        )
        # pre-existing columns keep their identity and order
        self.assertEqual(header[0], "scenario")
        self.assertLess(header.index("measurement_type"), history_idx)

    def test_csv_carries_full_fingerprints_and_no_json_cell(self):
        scenario, decision = self._scenario()
        rows = self._write(scenario)
        header, values = rows[0], rows[1]
        record = dict(zip(header, values))
        self.assertEqual(record["truth_force_fingerprint"], decision.truth_fingerprint)
        self.assertTrue(record["truth_force_fingerprint"].startswith("sha256:"))
        self.assertEqual(len(record["truth_force_fingerprint"].split(":")[1]), 64)
        for cell in values:
            self.assertFalse(cell.strip().startswith("{"), "CSV must not embed JSON")


class P9ManifestTests(unittest.TestCase):
    def test_manifest_shape_and_hash(self):
        decision = scenario_force_model_preflight(_config(j2_moon=J2_MOON))
        manifest = decision.manifest
        self.assertEqual(manifest["schema_version"], FORCE_CONTRACT_SCHEMA_VERSION)
        for side in ("truth", "estimator"):
            self.assertIn("fingerprint", manifest[side])
            self.assertIn("canonical_payload", manifest[side])
            self.assertIn("consumer_capabilities", manifest[side])
        self.assertIn("match", manifest)
        self.assertIn("mismatch_policy", manifest)
        self.assertEqual(decision.manifest_sha256, manifest_sha256(manifest))

    def test_manifest_bytes_are_deterministic_and_path_free(self):
        decision = scenario_force_model_preflight(_config(j2_moon=J2_MOON))
        first = manifest_canonical_bytes(decision.manifest)
        second = manifest_canonical_bytes(decision.manifest)
        self.assertEqual(first, second)
        text = first.decode("utf-8")
        self.assertNotIn(str(_REPO_ROOT), text)
        self.assertNotIn(":\\", text)

    def test_written_manifest_matches_reported_sha(self):
        decision = scenario_force_model_preflight(_config(j2_moon=J2_MOON))
        with tempfile.TemporaryDirectory() as tmp:
            path = write_force_model_manifest(decision.manifest, Path(tmp) / "force.json")
            written = path.read_bytes()
        self.assertEqual(written, manifest_canonical_bytes(decision.manifest))
        self.assertEqual(
            "sha256:" + __import__("hashlib").sha256(written).hexdigest(),
            decision.manifest_sha256,
        )
        # the CSV-reported hash is the same value
        fields = scenario_result_force_fields(decision)
        self.assertEqual(fields["force_contract_manifest_sha256"], decision.manifest_sha256)

    def test_same_run_in_different_cwd_gives_identical_manifest(self):
        script = (
            "import sys, os; sys.path.insert(0, r'%s')\n"
            "from lunar_od.scenario_config import scenario_config_from_mapping, scenario_force_model_preflight\n"
            "from lunar_od.force_contract import manifest_canonical_bytes\n"
            "c = scenario_config_from_mapping(%r)\n"
            "d = scenario_force_model_preflight(c)\n"
            "sys.stdout.write(manifest_canonical_bytes(d.manifest).hex())\n"
            % (str(_REPO_ROOT), {**_BASE_PAYLOAD, "j2_moon": J2_MOON})
        )
        outputs = []
        for cwd in (str(_REPO_ROOT), tempfile.gettempdir()):
            out = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, check=True, cwd=cwd,
            )
            outputs.append(out.stdout.strip())
        self.assertEqual(outputs[0], outputs[1])


class P10DesktopCliParityTests(unittest.TestCase):
    def test_desktop_and_json_runner_use_the_same_shared_preflight(self):
        import controllers.analysis_controller as ac

        source = Path(ac.__file__).read_text(encoding="utf-8")
        runner_source = _RUNNER_PATH.read_text(encoding="utf-8")
        for text in (source, runner_source):
            self.assertIn("scenario_force_model_preflight", text)

    def test_same_config_yields_same_fingerprints_on_both_paths(self):
        config = _config(j2_moon=J2_MOON)
        json_decision = scenario_force_model_preflight(
            config, context="run_configured_scenario force-model preflight"
        )
        desktop_decision = scenario_force_model_preflight(
            config, context="desktop analysis variant 'V0' force-model preflight"
        )
        self.assertEqual(
            json_decision.truth_fingerprint, desktop_decision.truth_fingerprint
        )
        self.assertEqual(
            json_decision.estimator_fingerprint, desktop_decision.estimator_fingerprint
        )
        self.assertEqual(json_decision.manifest_sha256, desktop_decision.manifest_sha256)


class P11ZeroJ2RegressionTests(unittest.TestCase):
    def test_zero_j2_default_preflight_is_matched_and_verified(self):
        decision = scenario_force_model_preflight(_config())
        fields = scenario_result_force_fields(decision)
        self.assertTrue(fields["force_model_match"])
        self.assertFalse(fields["explicit_force_model_mismatch"])
        self.assertEqual(fields["posterior_force_role_status"], "verified")
        self.assertEqual(fields["observability_force_role_status"], "verified")

    def test_default_scenario_result_force_fields_are_inert(self):
        scenario = ScenarioResult(
            label="legacy",
            measurement_type="range_rate",
            start_mode="cold",
            arc_results=(),
            estimator_type="bls_lm",
        )
        self.assertEqual(scenario.truth_force_fingerprint, "")
        self.assertTrue(scenario.force_model_match)
        self.assertFalse(scenario.explicit_force_model_mismatch)


class P12ImportDirectionTests(unittest.TestCase):
    def test_no_import_cycle_introduced(self):
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                "import lunar_od, lunar_od.force_contract, lunar_od.scenario_config, "
                "lunar_od.reporting, lunar_od.scenarios; print('ok')",
            ],
            capture_output=True, text=True, check=True, cwd=str(_REPO_ROOT),
        )
        self.assertEqual(out.stdout.strip(), "ok")

    def test_force_contract_still_free_of_project_imports(self):
        source = (_REPO_ROOT / "lunar_od" / "force_contract.py").read_text(encoding="utf-8")
        for forbidden in (
            "from .scenario_config", "from .scenarios", "from .filters",
            "from .estimators", "from .reporting", "from .dynamics",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
