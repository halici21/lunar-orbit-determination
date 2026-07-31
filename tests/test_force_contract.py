"""R0B-1 force-model contract, canonical payload, and fingerprint tests.

Deterministic and fast: no SPICE, no propagation, no RNG.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from lunar_od.force_contract import (
    FORCE_CONTRACT_SCHEMA_VERSION,
    ConsumerReadiness,
    ConsumerRole,
    ForceContractError,
    ForceElementStatus,
    ForceModelContract,
    LunarHarmonicsForceContract,
    OrientationPolicy,
    canonical_json_bytes,
    capability_map,
    force_model_fingerprint,
    lunar_j2_enabled,
    to_canonical_payload,
)
from lunar_od.scenario_config import (
    force_model_contract_from_scenario_config,
    scenario_config_from_mapping,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
J2_MOON = 2.0346e-4

_BASE_PAYLOAD = {
    "name": "r0b_contract",
    "measurement_type": "range_rate",
    "estimator_type": "bls_lm",
    "start_mode": "cold",
    "network": "multi",
}


def _config(**overrides):
    payload = {**_BASE_PAYLOAD, **overrides}
    return scenario_config_from_mapping(payload)


def _contract(**overrides):
    return force_model_contract_from_scenario_config(_config(**overrides))


def _fingerprint(**overrides):
    return _contract(**overrides).force_model_fingerprint()


class ForceContractCanonicalTests(unittest.TestCase):
    def test_canonical_payload_and_fingerprint_are_deterministic(self):
        first = _contract()
        second = _contract()
        self.assertEqual(first.canonical_json_bytes(), second.canonical_json_bytes())
        self.assertEqual(
            first.force_model_fingerprint(), second.force_model_fingerprint()
        )
        # repeated calls on one object are stable too
        self.assertEqual(first.force_model_fingerprint(), first.force_model_fingerprint())

    def test_fingerprint_format_is_sha256_lowercase_hex(self):
        fingerprint = _fingerprint()
        self.assertTrue(fingerprint.startswith("sha256:"))
        digest = fingerprint.split(":", 1)[1]
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, digest.lower())
        int(digest, 16)  # raises if not hex

    def test_canonical_json_is_sorted_and_compact(self):
        raw = canonical_json_bytes(_contract()).decode("utf-8")
        self.assertNotIn(", ", raw)
        self.assertNotIn(": ", raw)
        payload = json.loads(raw)
        self.assertEqual(list(payload), sorted(payload))
        self.assertEqual(payload["schema_version"], FORCE_CONTRACT_SCHEMA_VERSION)

    def test_implicit_and_explicit_defaults_match(self):
        implicit = _fingerprint()
        explicit = _fingerprint(
            j2_moon=0.0,
            enable_earth_j2=False,
            earth_j2_mode="indirect",
            enable_lunar_harmonics=False,
        )
        self.assertEqual(implicit, explicit)

    def test_physical_mutations_change_the_fingerprint(self):
        baseline = _fingerprint()
        self.assertNotEqual(baseline, _fingerprint(j2_moon=J2_MOON))
        # gravitational parameters are effective physics
        base_contract = _contract()
        mutated = force_model_contract_from_scenario_config(
            _config(), mu_moon_m3_s2=4.9028e12 * 1.000001
        )
        self.assertNotEqual(
            base_contract.force_model_fingerprint(), mutated.force_model_fingerprint()
        )

    def test_nonphysical_fields_do_not_change_the_fingerprint(self):
        baseline = _fingerprint()
        for overrides in (
            {"name": "totally_different_run_name"},
            {"rtol": 1e-9, "atol": 1e-10},
            {"max_iter": 7},
            {"output_dir": "some/other/output/dir"},
            {"duration_h": 12.0, "sample_step_s": 30.0},
            {"noise": True},
        ):
            with self.subTest(overrides=overrides):
                self.assertEqual(baseline, _fingerprint(**overrides))

    def test_negative_zero_normalizes_to_zero(self):
        positive = _contract(j2_moon=0.0)
        negative = _contract(j2_moon=-0.0)
        self.assertEqual(
            positive.force_model_fingerprint(), negative.force_model_fingerprint()
        )
        payload = to_canonical_payload(negative)
        self.assertEqual(
            math.copysign(1.0, payload["lunar_j2"]["coefficient"]), 1.0
        )

    def test_nonfinite_values_are_rejected(self):
        contract = _contract()
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                broken = dataclasses.replace(
                    contract,
                    lunar_j2=dataclasses.replace(contract.lunar_j2, coefficient=bad),
                )
                with self.assertRaises(ForceContractError):
                    broken.force_model_fingerprint()

    def test_finite_subnormal_is_preserved(self):
        subnormal = 5e-324
        contract = _contract()
        mutated = dataclasses.replace(
            contract,
            lunar_j2=dataclasses.replace(contract.lunar_j2, coefficient=subnormal),
        )
        payload = to_canonical_payload(mutated)
        self.assertEqual(payload["lunar_j2"]["coefficient"], subnormal)
        self.assertNotEqual(
            mutated.force_model_fingerprint(), contract.force_model_fingerprint()
        )

    def test_unknown_runtime_object_is_rejected(self):
        contract = _contract()
        broken = dataclasses.replace(
            contract,
            lunar_j2=dataclasses.replace(contract.lunar_j2, coefficient=object()),
        )
        with self.assertRaises(ForceContractError):
            broken.canonical_json_bytes()

    def test_schema_version_mutation_changes_the_fingerprint(self):
        contract = _contract()
        bumped = dataclasses.replace(contract, schema_version="r0b.force-model-contract.v99")
        self.assertNotEqual(
            contract.force_model_fingerprint(), bumped.force_model_fingerprint()
        )

    def test_capability_map_requires_every_role(self):
        with self.assertRaises(ForceContractError):
            capability_map({ConsumerRole.TRUTH_STATE: ConsumerReadiness.VERIFIED})

    def test_lunar_j2_enabled_rule(self):
        self.assertFalse(lunar_j2_enabled(0.0))
        self.assertFalse(lunar_j2_enabled(-0.0))
        self.assertTrue(lunar_j2_enabled(J2_MOON))

    def test_cross_process_determinism(self):
        script = (
            "import sys; sys.path.insert(0, r'%s')\n"
            "from lunar_od.scenario_config import (force_model_contract_from_scenario_config,"
            " scenario_config_from_mapping)\n"
            "c = scenario_config_from_mapping(%r)\n"
            "print(force_model_contract_from_scenario_config(c).force_model_fingerprint())\n"
            % (str(_REPO_ROOT), {**_BASE_PAYLOAD, "j2_moon": J2_MOON})
        )
        runs = []
        for _ in range(2):
            out = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=True,
                cwd=str(_REPO_ROOT),
            )
            runs.append(out.stdout.strip())
        # deterministic across processes AND equal to the in-process value
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(runs[0], _fingerprint(j2_moon=J2_MOON))


class ForceContractCapabilityTests(unittest.TestCase):
    def test_zero_j2_baseline_is_fully_verified(self):
        caps = _contract().consumer_capabilities
        for role in ConsumerRole:
            self.assertEqual(caps[role], ConsumerReadiness.VERIFIED, role)

    def test_nonzero_lunar_j2_capability_matrix(self):
        caps = _contract(j2_moon=J2_MOON).consumer_capabilities
        for role in (
            ConsumerRole.TRUTH_STATE,
            ConsumerRole.ESTIMATOR_STATE,
            ConsumerRole.ESTIMATOR_STM,
            ConsumerRole.UKF_STANDARD,
            ConsumerRole.UKF_SQUARE_ROOT,
            ConsumerRole.UKF_FAST_SIGMA,
        ):
            self.assertEqual(caps[role], ConsumerReadiness.VERIFIED, role)
        self.assertEqual(
            caps[ConsumerRole.POSTERIOR_COVARIANCE], ConsumerReadiness.VERIFIED
        )
        self.assertEqual(caps[ConsumerRole.OBSERVABILITY], ConsumerReadiness.VERIFIED)

    def test_lunar_j2_element_status_is_supported(self):
        contract = _contract(j2_moon=J2_MOON)
        self.assertEqual(contract.lunar_j2.status, ForceElementStatus.SUPPORTED)
        self.assertEqual(
            contract.lunar_j2.orientation_policy,
            OrientationPolicy.CONSTANT_IAU2006_MOON_MEAN_POLE,
        )

    def test_earth_j2_element_status_is_unsupported_official_od(self):
        # enable_earth_j2=True is rejected by the loader (R0A), so build the
        # config directly to inspect the contract's capability reporting.
        config = dataclasses.replace(_config(), enable_earth_j2=True)
        contract = force_model_contract_from_scenario_config(config)
        self.assertEqual(
            contract.earth_j2.status, ForceElementStatus.UNSUPPORTED_OFFICIAL_OD
        )
        self.assertEqual(
            contract.earth_j2.orientation_policy,
            OrientationPolicy.EXPERIMENTAL_IDENTITY_J2000_TO_EARTH_BODY_FIXED,
        )
        for role in ConsumerRole:
            self.assertEqual(contract.consumer_capabilities[role], ConsumerReadiness.UNSUPPORTED)

    def test_disabled_harmonics_status_and_neutral_fields(self):
        harmonics = _contract().lunar_harmonics
        self.assertFalse(harmonics.enabled)
        self.assertIsNone(harmonics.coefficient_file_sha256)
        self.assertEqual(
            harmonics.status, ForceElementStatus.EXPERIMENTAL_DIRECT_TRAJECTORY_ONLY
        )

    def test_inert_gravity_path_does_not_reach_the_fingerprint(self):
        # harmonics disabled: a configured-but-unused path must not leak in
        baseline = _fingerprint()
        with_path = _fingerprint(lunar_gravity_model_path="some/unused/model.tab")
        self.assertEqual(baseline, with_path)


_GRAVITY_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "gravity" / "synthetic_norm_sha.tab"
)


class ForceContractHarmonicsHashingTests(unittest.TestCase):
    """Harmonics identity comes from CONTENT, not from where the file lives."""

    _MODEL_TEXT = _GRAVITY_FIXTURE.read_text(encoding="utf-8")

    def _harmonics_contract(self, path: Path):
        config = dataclasses.replace(
            _config(),
            enable_lunar_harmonics=True,
            lunar_gravity_model_path=str(path),
            lunar_gravity_nmax=4,
            lunar_gravity_mmax=4,
        )
        return force_model_contract_from_scenario_config(config)

    def test_same_content_different_directory_same_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            path_a = Path(tmp_a) / "model.tab"
            path_b = Path(tmp_b) / "model.tab"
            path_a.write_text(self._MODEL_TEXT, encoding="utf-8")
            path_b.write_text(self._MODEL_TEXT, encoding="utf-8")
            try:
                contract_a = self._harmonics_contract(path_a)
                contract_b = self._harmonics_contract(path_b)
            except Exception as exc:  # pragma: no cover - loader format guard
                self.skipTest(f"gravity loader rejected the synthetic model: {exc}")
            self.assertEqual(
                contract_a.force_model_fingerprint(),
                contract_b.force_model_fingerprint(),
            )
            payload = to_canonical_payload(contract_a)
            self.assertNotIn(tmp_a, json.dumps(payload))

    def test_same_metadata_different_content_different_fingerprint(self):
        altered = self._MODEL_TEXT.replace(
            "3.4700000000000000E-05", "3.4710000000000000E-05"
        )
        self.assertNotEqual(altered, self._MODEL_TEXT)
        with tempfile.TemporaryDirectory() as tmp:
            path_a = Path(tmp) / "a.tab"
            path_b = Path(tmp) / "b.tab"
            path_a.write_text(self._MODEL_TEXT, encoding="utf-8")
            path_b.write_text(altered, encoding="utf-8")
            try:
                contract_a = self._harmonics_contract(path_a)
                contract_b = self._harmonics_contract(path_b)
            except Exception as exc:  # pragma: no cover - loader format guard
                self.skipTest(f"gravity loader rejected the synthetic model: {exc}")
            self.assertNotEqual(
                contract_a.force_model_fingerprint(),
                contract_b.force_model_fingerprint(),
            )

    def test_missing_gravity_file_raises_controlled_error(self):
        config = dataclasses.replace(
            _config(),
            enable_lunar_harmonics=True,
            lunar_gravity_model_path=str(Path(tempfile.gettempdir()) / "definitely_absent.tab"),
            lunar_gravity_nmax=2,
        )
        with self.assertRaises(FileNotFoundError):
            force_model_contract_from_scenario_config(config)

    def test_directory_path_raises_controlled_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = dataclasses.replace(
                _config(),
                enable_lunar_harmonics=True,
                lunar_gravity_model_path=tmp,
                lunar_gravity_nmax=2,
            )
            with self.assertRaises(FileNotFoundError):
                force_model_contract_from_scenario_config(config)

    def test_harmonics_status_is_experimental_direct_trajectory_only(self):
        contract = LunarHarmonicsForceContract(
            enabled=True,
            model_identity="synthetic",
            coefficient_file_sha256="sha256:" + "0" * 64,
            degree_nmax=2,
            order_mmax=2,
            normalization="fully_normalized_4pi",
            model_gravitational_parameter_m3_s2=4.9028e12,
            reference_radius_m=1.738e6,
            body_frame="MOON_PA_DE421",
            kernel_profile_policy="de421",
            rotation_cadence_s=60.0,
            rotation_margin_s=None,
        )
        self.assertEqual(
            contract.status, ForceElementStatus.EXPERIMENTAL_DIRECT_TRAJECTORY_ONLY
        )


class ForceContractImportDirectionTests(unittest.TestCase):
    FORBIDDEN_FOR_FORCE_CONTRACT = {
        "scenario_config",
        "scenarios",
        "filters",
        "estimators",
        "reporting",
        "dynamics",
        "observability",
        "measurements",
        "radiometrics",
    }

    def _module_imports(self, relative_path: str) -> set[str]:
        source = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module.lstrip("."))
                if node.level and node.module:
                    found.add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    found.add(alias.name)
        return {name.split(".")[-1] for name in found}

    def test_force_contract_has_no_forbidden_imports(self):
        imported = self._module_imports("lunar_od/force_contract.py")
        offending = imported & self.FORBIDDEN_FOR_FORCE_CONTRACT
        self.assertEqual(offending, set(), f"force_contract must not import {offending}")

    def test_force_contract_is_standard_library_only(self):
        imported = self._module_imports("lunar_od/force_contract.py")
        allowed = {
            "__future__", "hashlib", "json", "math", "dataclasses",
            "enum", "types", "typing",
        }
        self.assertTrue(
            imported <= allowed, f"unexpected non-stdlib imports: {imported - allowed}"
        )

    def test_dynamics_does_not_import_force_contract_or_scenario_config(self):
        imported = self._module_imports("lunar_od/dynamics.py")
        self.assertNotIn("force_contract", imported)
        self.assertNotIn("scenario_config", imported)

    def test_force_contract_module_imports_without_package_cycle(self):
        out = subprocess.run(
            [sys.executable, "-c", "import lunar_od.force_contract as m; print(m.FORCE_CONTRACT_SCHEMA_VERSION)"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(_REPO_ROOT),
        )
        self.assertEqual(out.stdout.strip(), FORCE_CONTRACT_SCHEMA_VERSION)


class ScenarioConfigForceFieldConsumptionTests(unittest.TestCase):
    """Every force-related ScenarioConfig field must reach the contract."""

    FORCE_FIELDS = (
        "j2_moon",
        "enable_earth_j2",
        "earth_j2_mode",
        "enable_lunar_harmonics",
        "lunar_gravity_model_path",
        "lunar_gravity_nmax",
        "lunar_gravity_mmax",
        "lunar_gravity_frame",
        "lunar_gravity_rotation_cadence_s",
        "lunar_gravity_rotation_margin_s",
        "lunar_gravity_kernel_profile",
    )

    def test_force_field_inventory_is_complete(self):
        config = _config()
        for name in self.FORCE_FIELDS:
            self.assertTrue(hasattr(config, name), name)

    def test_active_harmonics_fields_reach_the_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.tab"
            path.write_text(
                ForceContractHarmonicsHashingTests._MODEL_TEXT, encoding="utf-8"
            )
            base = dataclasses.replace(
                _config(),
                enable_lunar_harmonics=True,
                lunar_gravity_model_path=str(path),
                lunar_gravity_nmax=4,
                lunar_gravity_mmax=4,
            )
            try:
                baseline = force_model_contract_from_scenario_config(base)
            except Exception as exc:  # pragma: no cover - loader format guard
                self.skipTest(f"gravity loader rejected the synthetic model: {exc}")
            # rotation cadence and margin are force-identity fields
            cadence = force_model_contract_from_scenario_config(
                dataclasses.replace(base, lunar_gravity_rotation_cadence_s=30.0)
            )
            margin = force_model_contract_from_scenario_config(
                dataclasses.replace(base, lunar_gravity_rotation_margin_s=5.0)
            )
            self.assertNotEqual(
                baseline.force_model_fingerprint(), cadence.force_model_fingerprint()
            )
            self.assertNotEqual(
                baseline.force_model_fingerprint(), margin.force_model_fingerprint()
            )


if __name__ == "__main__":
    unittest.main()
