"""R0B-F1 executed-physics binding & production provenance tests (F1-F9).

These drive the REAL scenario runner, CLI main, and desktop worker (not just
helpers) so provenance defects that only appear on the production paths are
caught. Deterministic; SPICE/propagation are spied or stubbed, never really run.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import contextlib

import numpy as np

from lunar_od.force_contract import (
    ConsumerReadiness,
    ConsumerRole,
    ForceContractError,
    bind_spec_to_runtime,
    force_model_fingerprint,
)
from lunar_od.scenario_config import (
    force_execution_spec_from_scenario_config,
    scenario_config_from_mapping,
)
from lunar_od.scenarios import ScenarioResult


def _empty_scenario_result():
    return ScenarioResult(
        label="r0b_f1",
        measurement_type="range_rate",
        start_mode="cold",
        arc_results=(),
        estimator_type="bls_lm",
    )


def _scenario_result_with_arc():
    """A ScenarioResult with one arc, so the summary CSV emits a data row."""
    from tests.test_reporting import _arc_result

    return ScenarioResult(
        label="r0b_f1",
        measurement_type="range_rate",
        start_mode="cold",
        arc_results=(_arc_result(1, 1000.0, 25.0),),
        estimator_type="bls_lm",
    )

_REPO_ROOT = Path(__file__).resolve().parents[1]
J2_MOON = 2.0346e-4

_RUNNER_PATH = _REPO_ROOT / "examples" / "run_scenario_config.py"
_spec = importlib.util.spec_from_file_location("r0b_f1_runner", _RUNNER_PATH)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

_FIXTURE = {
    "initial_state": {
        "state_mci_j2000_m_mps": [1.8374e6, 0.0, 0.0, 0.0, 1633.0, 0.0],
        "mu_moon_m3_s2": 4.9028000661e12,
        "r_moon_mean_m": 1.7374e6,
    },
    # Chosen so `value * 1e9` differs from MU_EARTH_M3S2's literal by ~1 ULP,
    # which is exactly the V01 hazard the effective binding must absorb.
    "constants": {
        "mu_earth_km3_s2": [398600.4354360959],
        "mu_sun_km3_s2": [132712440041.9393],
    },
    "epoch_utc": "2026-01-01T00:00:00",
}

_BASE_PAYLOAD = {
    "name": "r0b_f1",
    "measurement_type": "range_rate",
    "estimator_type": "bls_lm",
    "start_mode": "cold",
    "network": "multi",
}


def _config(**overrides):
    return scenario_config_from_mapping({**_BASE_PAYLOAD, **overrides})


def _fake_ephemeris():
    return types.SimpleNamespace(
        earth_position=lambda t: np.zeros((np.size(np.asarray(t)), 3)),
        earth_velocity=lambda t: np.zeros((np.size(np.asarray(t)), 3)),
        sun_position=lambda t: np.zeros((np.size(np.asarray(t)), 3)),
    )


class _RunnerHarness:
    """Patch the real run_configured_scenario down to sp-able boundaries.

    truth_spy / batch_spy capture the exact GM and j2 that reach the propagator
    and the estimator dispatch. Fixture I/O is redirected to an in-memory dict
    and a temp output dir; SPICE is stubbed.
    """

    def __init__(self, stack, tmpdir):
        self.truth_spy = mock.MagicMock(return_value=np.zeros((6, 6)))
        self.batch_spy = mock.MagicMock(return_value=_scenario_result_with_arc())
        self.spice_load_spy = mock.MagicMock()
        # Downstream propagation/estimator entries. Once run_batch is mocked
        # these never run, but patching them makes the F3 zero-call assertion
        # explicit for every production entry.
        self.state_spy = mock.MagicMock()
        self.stm_spy = mock.MagicMock()
        self.ukf_spy = mock.MagicMock()
        self.fast_sigma_spy = mock.MagicMock(return_value=None)
        self.bls_spy = mock.MagicMock()
        self.srif_spy = mock.MagicMock()
        self.fixture_reads = 0
        eph = _fake_ephemeris()
        spice_stub = types.SimpleNamespace(str2et=lambda _s: 0.0, kclear=lambda: None)
        fake_station = types.SimpleNamespace(name="S1")
        net = types.SimpleNamespace(station_names=("S1",))
        real_read_text = Path.read_text

        def _read_text(self_path, *a, **k):
            if self_path.name == "spice_snapshots.json":
                self.fixture_reads += 1
                return json.dumps(_FIXTURE)
            return real_read_text(self_path, *a, **k)

        stack.enter_context(mock.patch.dict(sys.modules, {"spiceypy": spice_stub}))
        stack.enter_context(mock.patch.object(Path, "is_file", lambda self: True))
        stack.enter_context(mock.patch.object(Path, "read_text", _read_text))
        stack.enter_context(mock.patch.object(runner, "load_spice_kernels", self.spice_load_spy))
        stack.enter_context(mock.patch("lunar_od.dynamics.propagate_state", self.state_spy))
        stack.enter_context(mock.patch("lunar_od.dynamics.propagate_augmented_state", self.stm_spy))
        stack.enter_context(mock.patch("lunar_od.scenarios.run_lunar_ukf", self.ukf_spy))
        stack.enter_context(mock.patch("lunar_od.scenarios.make_fast_sigma_propagator", self.fast_sigma_spy))
        stack.enter_context(mock.patch("lunar_od.scenarios.estimate_position_bls_lm", self.bls_spy))
        stack.enter_context(mock.patch("lunar_od.scenarios.estimate_position_srif", self.srif_spy))
        stack.enter_context(mock.patch.object(runner, "sample_moon_centered_ephemeris", return_value=eph))
        stack.enter_context(mock.patch.object(runner, "perturb_moon_centered_ephemeris", return_value=eph))
        stack.enter_context(
            mock.patch.object(
                runner, "sample_j2000_to_itrf93_transforms",
                side_effect=lambda _et, t: np.repeat(np.eye(6)[None, :, :], np.size(np.asarray(t)), axis=0),
            )
        )
        stack.enter_context(mock.patch.object(runner, "propagate_truth_with_ephemeris", self.truth_spy))
        stack.enter_context(mock.patch.object(runner, "range_rate_stations", return_value=[fake_station]))
        stack.enter_context(mock.patch.object(runner, "thesis_network_by_name", return_value=net))
        # analyze_visibility_gap_with_transforms and run_batch_arc_sequence are
        # imported from lunar_od INSIDE the runner function, so patch them there.
        stack.enter_context(
            mock.patch(
                "lunar_od.analyze_visibility_gap_with_transforms",
                side_effect=lambda *a, **k: (np.array([0]), np.array([5]), np.ones(6, dtype=bool), None),
            )
        )
        stack.enter_context(mock.patch("lunar_od.run_batch_arc_sequence", self.batch_spy))
        stack.enter_context(mock.patch.object(runner, "build_measurement_arcs", return_value=(object(),)))
        stack.enter_context(mock.patch.object(runner, "_with_estimator_ephemeris", side_effect=lambda arc, e: arc))
        stack.enter_context(mock.patch.object(runner, "make_cold_start_bank", return_value=(np.zeros(6),)))
        stack.enter_context(mock.patch.object(runner, "scenario_ukf_configs", return_value=(None, None)))
        stack.enter_context(mock.patch.object(runner, "scenario_range_rate_physics_config", return_value=None))
        stack.enter_context(mock.patch.object(runner, "_initial_bias", return_value=np.zeros(0)))


class F1EffectiveGmBindingTests(unittest.TestCase):
    def test_effective_gm_reaches_truth_and_estimator_bit_exact(self):
        config = _config(j2_moon=J2_MOON)
        import contextlib
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            h = _RunnerHarness(stack, tmp)
            runner.run_configured_scenario(config, manifest_dir=tmp)
            fixture_earth = float(np.asarray(_FIXTURE["constants"]["mu_earth_km3_s2"]).reshape(-1)[0] * 1e9)
            fixture_moon = float(_FIXTURE["initial_state"]["mu_moon_m3_s2"])
            fixture_sun = float(np.asarray(_FIXTURE["constants"]["mu_sun_km3_s2"]).reshape(-1)[0] * 1e9)
            # truth propagator positional args: (t, x0, mu_moon, mu_earth, mu_sun, eph, ...)
            targs = h.truth_spy.call_args.args
            self.assertEqual(targs[2], fixture_moon)
            self.assertEqual(targs[3], fixture_earth)
            self.assertEqual(targs[4], fixture_sun)
            # estimator/batch positional args: (arcs, mtype, start, est, mu_moon, mu_earth, mu_sun, ...)
            bargs = h.batch_spy.call_args.args
            self.assertEqual(bargs[4], fixture_moon)
            self.assertEqual(bargs[5], fixture_earth)
            self.assertEqual(bargs[6], fixture_sun)
            # And the fingerprinted contract Earth GM equals that exact runtime value.
            spec = force_execution_spec_from_scenario_config(
                config, mu_moon_m3_s2=fixture_moon, mu_earth_m3_s2=fixture_earth, mu_sun_m3_s2=fixture_sun
            )
            self.assertEqual(spec.mu_earth_m3_s2, fixture_earth)
            bind_spec_to_runtime(
                spec,
                {"mu_moon_m3_s2": fixture_moon, "mu_earth_m3_s2": fixture_earth,
                 "mu_sun_m3_s2": fixture_sun, "j2_moon": J2_MOON},
                context="F1",
            )


class F2ExecutableLunarJ2MismatchTests(unittest.TestCase):
    def test_declared_j2_mismatch_runs_truth_and_estimator_differently(self):
        config = _config(
            j2_moon=J2_MOON,
            allow_explicit_force_model_mismatch=True,
            force_model_mismatch_reason="truth J2 vs point-mass estimator",
        )
        import contextlib
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            h = _RunnerHarness(stack, tmp)
            result = runner.run_configured_scenario(
                config, truth_j2_moon=J2_MOON, estimator_j2_moon=0.0, manifest_dir=tmp
            )
        # truth propagator got nonzero J2; estimator/batch got zero J2. The
        # estimator J2 (0.0) is what the batch forwards to BLS/STM, standard UKF,
        # square-root UKF, and fast-sigma — all consumer boundaries share the
        # single estimator-spec value (see F5/T3-T5 for per-path J2 execution).
        self.assertEqual(h.truth_spy.call_args.kwargs["j2_moon"], J2_MOON)
        self.assertEqual(h.batch_spy.call_args.kwargs["j2_moon"], 0.0)
        # fingerprints differ; match false; reported consistently
        self.assertNotEqual(result.truth_force_fingerprint, result.estimator_force_fingerprint)
        self.assertFalse(result.force_model_match)
        self.assertTrue(result.explicit_force_model_mismatch)
        self.assertEqual(result.force_model_mismatch_reason, "truth J2 vs point-mass estimator")

    def test_estimator_j2_reaches_all_consumer_boundaries(self):
        # Direct check that one estimator spec's J2 is what BLS/STM, standard
        # UKF, square-root UKF, and fast-sigma all consume: run_lunar_ukf and
        # make_fast_sigma_propagator receive it, and the batch dispatch forwards
        # the same value the estimators use. Uses the shared reproducer helper.
        from lunar_od.scenarios import run_batch_arc_sequence
        from tests.test_r0a_force_config_parity import (
            _prepared_position_arc,
            _synthetic_ephemeris,
            MU_MOON,
        )

        arc, get_earth, get_sun = _prepared_position_arc()
        captured = {}
        real_ukf = __import__("lunar_od.filters", fromlist=["run_lunar_ukf"]).run_lunar_ukf

        def _ukf_spy(*args, **kwargs):
            captured["ukf_j2"] = kwargs.get("j2_moon")
            return real_ukf(*args, **kwargs)

        fast_spy = mock.MagicMock(return_value=None)
        with mock.patch("lunar_od.scenarios.run_lunar_ukf", side_effect=_ukf_spy), \
                mock.patch("lunar_od.scenarios.make_fast_sigma_propagator", fast_spy):
            run_batch_arc_sequence(
                (arc,), "position", "cold", "ukf", MU_MOON, 0.0, 0.0,
                get_earth, get_sun, cold_start_bank=(np.zeros(6),),
                ephemeris=_synthetic_ephemeris(700.0), j2_moon=J2_MOON,
            )
        self.assertEqual(captured["ukf_j2"], J2_MOON)  # standard + SR UKF path
        self.assertEqual(fast_spy.call_args.kwargs["j2_moon"], J2_MOON)  # fast sigma

    def test_undeclared_j2_mismatch_is_rejected(self):
        config = _config(j2_moon=J2_MOON)  # no opt-in
        import contextlib
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            _RunnerHarness(stack, tmp)
            from lunar_od.force_contract import ForceModelParityError
            with self.assertRaises(ForceModelParityError):
                runner.run_configured_scenario(
                    config, truth_j2_moon=J2_MOON, estimator_j2_moon=0.0, manifest_dir=tmp
                )


class F3HarmonicsCampaignRejectionTests(unittest.TestCase):
    def test_official_harmonics_rejected_before_any_execution(self):
        fixture_gravity = _REPO_ROOT / "tests" / "fixtures" / "gravity" / "synthetic_norm_sha.tab"
        config = dataclasses.replace(
            _config(),
            enable_lunar_harmonics=True,
            lunar_gravity_model_path=str(fixture_gravity),
            lunar_gravity_nmax=4,
            lunar_gravity_mmax=4,
        )
        import contextlib
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            h = _RunnerHarness(stack, tmp)
            with self.assertRaises(ForceContractError) as caught:
                runner.run_configured_scenario(config, manifest_dir=tmp)
        # The message conveys all four required meanings (not just "experimental").
        msg = str(caught.exception)
        self.assertIn("EXPERIMENTAL", msg)  # experimental direct-trajectory capability
        self.assertIn("direct", msg.lower())
        self.assertIn("not wired into the official", msg)  # not bound to official execution
        self.assertIn("not implemented", msg)  # official mismatch campaign not implemented
        self.assertIn("as if it were executed", msg)  # no as-if-executed provenance
        # EVERY production entry counter is zero: rejection precedes all of them.
        self.assertEqual(h.fixture_reads, 0, "fixture read")
        self.assertEqual(h.spice_load_spy.call_count, 0, "SPICE load")
        self.assertEqual(h.truth_spy.call_count, 0, "truth")
        self.assertEqual(h.state_spy.call_count, 0, "state")
        self.assertEqual(h.stm_spy.call_count, 0, "STM (optional UKF STM incl.)")
        self.assertEqual(h.ukf_spy.call_count, 0, "UKF (std/SR)")
        self.assertEqual(h.fast_sigma_spy.call_count, 0, "fast sigma factory/runtime")
        self.assertEqual(h.bls_spy.call_count, 0, "BLS")
        self.assertEqual(h.srif_spy.call_count, 0, "SRIF")
        self.assertEqual(h.batch_spy.call_count, 0, "batch")

    def test_harmonics_capability_metadata(self):
        # Capability is reported independently of the execution gate: harmonics
        # truth_state is experimental_direct_trajectory_only; all official
        # estimator roles are unsupported.
        from lunar_od.force_contract import consumer_capabilities_for

        caps = consumer_capabilities_for(lunar_j2_on=False, earth_j2_on=False, harmonics_on=True)
        self.assertEqual(
            caps[ConsumerRole.TRUTH_STATE],
            ConsumerReadiness.EXPERIMENTAL_DIRECT_TRAJECTORY_ONLY,
        )
        for role in ConsumerRole:
            if role is ConsumerRole.TRUTH_STATE:
                continue
            self.assertEqual(caps[role], ConsumerReadiness.UNSUPPORTED, role)


class F4RealCliManifestTests(unittest.TestCase):
    """Drive the REAL CLI main() and prove one output bundle owns everything."""

    def _write_config_json(self, config_output_dir):
        cfg = {
            "name": "r0b_f2_cli",
            "measurement_type": "range_rate",
            "estimator_type": "bls_lm",
            "start_mode": "cold",
            "network": "multi",
            "j2_moon": J2_MOON,
            "duration_h": 1.0,
            "sample_step_s": 600.0,
            "output_dir": str(config_output_dir),
        }
        cfg_path = Path(config_output_dir) / "scenario.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        return cfg_path

    def _read_csv_row(self, csv_path):
        with Path(csv_path).open(newline="", encoding="utf-8") as fh:
            return next(csv.DictReader(fh))

    def test_cli_output_dir_override_owns_csv_png_and_force_manifest(self):
        import contextlib

        with tempfile.TemporaryDirectory() as config_dir, \
                tempfile.TemporaryDirectory() as override_dir, \
                contextlib.ExitStack() as stack:
            _RunnerHarness(stack, override_dir)
            # PNG plotting is unrelated to bundle ownership; stub it so it writes
            # to whatever path main() computes and return that path.
            def _fake_plot(scenarios, output_path, *, title=""):
                Path(output_path).write_bytes(b"\x89PNG\r\n")
                return Path(output_path)
            stack.enter_context(mock.patch.object(runner, "plot_scenario_comparison", side_effect=_fake_plot))

            cfg_path = self._write_config_json(config_dir)
            cwd_before = os.getcwd()
            os.chdir(override_dir)  # so a stray CWD manifest would be detectable
            try:
                rc = runner.main([str(cfg_path), "--output-dir", override_dir])
            finally:
                os.chdir(cwd_before)

            self.assertEqual(rc, 0)  # argument parsing + run really happened
            name = "r0b_f2_cli"
            csv_path = Path(override_dir) / f"{name}_summary.csv"
            png_path = Path(override_dir) / f"{name}_comparison.png"
            manifest_path = Path(override_dir) / f"{name}_force_model_manifest.json"
            # (3-5) CSV, PNG, manifest all under the override bundle
            self.assertTrue(csv_path.is_file(), "CSV not in override dir")
            self.assertTrue(png_path.is_file(), "PNG not in override dir")
            self.assertTrue(manifest_path.is_file(), "manifest not in override dir")
            # (6) not in config.output_dir; (7) not in CWD (== override here, so
            # check the config dir which is the only other candidate)
            self.assertFalse(
                (Path(config_dir) / f"{name}_force_model_manifest.json").exists(),
                "manifest leaked into config.output_dir",
            )
            # (8-9) valid JSON, no absolute path
            raw = manifest_path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
            self.assertIn("schema_version", payload)
            self.assertNotIn(override_dir, raw.decode("utf-8"))
            self.assertNotIn(config_dir, raw.decode("utf-8"))
            self.assertNotIn(str(_REPO_ROOT), raw.decode("utf-8"))
            # (10) file-byte SHA == CSV SHA == result/provenance SHA
            file_sha = "sha256:" + hashlib.sha256(raw).hexdigest()
            row = self._read_csv_row(csv_path)
            self.assertEqual(file_sha, row["force_contract_manifest_sha256"])
            # (11) truth/estimator fingerprints equal across manifest and CSV
            self.assertEqual(payload["truth"]["fingerprint"], row["truth_force_fingerprint"])
            self.assertEqual(payload["estimator"]["fingerprint"], row["estimator_force_fingerprint"])
            # (12) atomic write leaves a complete file and no leftover temp
            self.assertFalse(manifest_path.with_name(manifest_path.name + ".tmp").exists())

    def test_cli_native_output_dir_owns_the_whole_bundle(self):
        import contextlib

        with tempfile.TemporaryDirectory() as config_dir, contextlib.ExitStack() as stack:
            _RunnerHarness(stack, config_dir)
            def _fake_plot(scenarios, output_path, *, title=""):
                Path(output_path).write_bytes(b"\x89PNG\r\n")
                return Path(output_path)
            stack.enter_context(mock.patch.object(runner, "plot_scenario_comparison", side_effect=_fake_plot))

            cfg_path = self._write_config_json(config_dir)
            rc = runner.main([str(cfg_path)])  # no --output-dir override

            self.assertEqual(rc, 0)
            name = "r0b_f2_cli"
            for suffix in ("_summary.csv", "_comparison.png", "_force_model_manifest.json"):
                self.assertTrue(
                    (Path(config_dir) / f"{name}{suffix}").is_file(),
                    f"{suffix} not under config.output_dir",
                )
            manifest_path = Path(config_dir) / f"{name}_force_model_manifest.json"
            raw = manifest_path.read_bytes()
            file_sha = "sha256:" + hashlib.sha256(raw).hexdigest()
            row = self._read_csv_row(Path(config_dir) / f"{name}_summary.csv")
            self.assertEqual(file_sha, row["force_contract_manifest_sha256"])


class F6NestedImmutabilityTests(unittest.TestCase):
    def test_nested_manifest_mutation_cannot_stale_the_sha(self):
        from lunar_od.scenario_config import scenario_force_model_preflight

        decision = scenario_force_model_preflight(_config(j2_moon=J2_MOON))
        stored = decision.manifest_sha256
        payload = decision.manifest
        payload["truth"]["fingerprint"] = "sha256:" + "0" * 64
        payload["match"] = False
        # stored SHA is unchanged and a fresh access is intact
        self.assertEqual(decision.manifest_sha256, stored)
        self.assertNotEqual(decision.manifest["truth"]["fingerprint"], "sha256:" + "0" * 64)
        recomputed = "sha256:" + hashlib.sha256(decision.manifest_bytes).hexdigest()
        self.assertEqual(recomputed, stored)


class F7InvalidMappingKeyTests(unittest.TestCase):
    def test_non_string_keys_raise_controlled_error(self):
        from lunar_od.force_contract import canonical_json_bytes, ForceContractError as FCE

        class _Fake:
            consumer_capabilities = None

        for bad_key in (object(), 42, 3.14, (1, 2)):
            with self.subTest(bad_key=type(bad_key).__name__):
                # exercise the normalizer directly through the manifest path
                from lunar_od.force_contract import manifest_canonical_bytes
                with self.assertRaises(FCE):
                    manifest_canonical_bytes({bad_key: "x"})

    def test_invalid_key_error_is_deterministic_across_processes(self):
        script = (
            "import sys; sys.path.insert(0, r'%s')\n"
            "from lunar_od.force_contract import manifest_canonical_bytes, ForceContractError\n"
            "try:\n"
            "    manifest_canonical_bytes({object(): 'x'})\n"
            "    print('NO_ERROR')\n"
            "except ForceContractError as e:\n"
            "    print('ForceContractError')\n" % str(_REPO_ROOT)
        )
        outs = []
        for _ in range(2):
            out = __import__("subprocess").run(
                [sys.executable, "-c", script], capture_output=True, text=True, check=True, cwd=str(_REPO_ROOT)
            )
            outs.append(out.stdout.strip())
        self.assertEqual(outs, ["ForceContractError", "ForceContractError"])


class F8CapabilityMatrixTests(unittest.TestCase):
    def test_harmonics_truth_cell_is_experimental_direct_trajectory_only(self):
        from lunar_od.force_contract import consumer_capabilities_for

        caps = consumer_capabilities_for(lunar_j2_on=False, earth_j2_on=False, harmonics_on=True)
        self.assertEqual(
            caps[ConsumerRole.TRUTH_STATE],
            ConsumerReadiness.EXPERIMENTAL_DIRECT_TRAJECTORY_ONLY,
        )

    def test_lunar_j2_posterior_observability_verified_r1(self):
        from lunar_od.force_contract import consumer_capabilities_for

        caps = consumer_capabilities_for(lunar_j2_on=True, earth_j2_on=False, harmonics_on=False)
        self.assertEqual(caps[ConsumerRole.POSTERIOR_COVARIANCE], ConsumerReadiness.VERIFIED)
        self.assertEqual(caps[ConsumerRole.OBSERVABILITY], ConsumerReadiness.VERIFIED)

    def test_earth_j2_all_unsupported(self):
        from lunar_od.force_contract import consumer_capabilities_for

        caps = consumer_capabilities_for(lunar_j2_on=False, earth_j2_on=True, harmonics_on=False)
        for role in ConsumerRole:
            self.assertEqual(caps[role], ConsumerReadiness.UNSUPPORTED, role)


class F9ExistingGatesTests(unittest.TestCase):
    def test_whitespace_only_reason_zero_call(self):
        from lunar_od.scenario_config import scenario_force_model_preflight
        from lunar_od.force_contract import ForceModelParityError

        config = _config(
            j2_moon=J2_MOON,
            allow_explicit_force_model_mismatch=True,
            force_model_mismatch_reason="   ",
        )
        import contextlib
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            h = _RunnerHarness(stack, tmp)
            with self.assertRaises(ForceModelParityError):
                runner.run_configured_scenario(config, truth_j2_moon=J2_MOON, estimator_j2_moon=0.0, manifest_dir=tmp)
        self.assertEqual(h.truth_spy.call_count, 0)
        self.assertEqual(h.batch_spy.call_count, 0)

    def test_zero_j2_default_runs_and_reports_verified(self):
        config = _config()  # zero J2
        import contextlib
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            h = _RunnerHarness(stack, tmp)
            result = runner.run_configured_scenario(config, manifest_dir=tmp)
        self.assertEqual(h.truth_spy.call_args.kwargs["j2_moon"], 0.0)
        self.assertEqual(h.batch_spy.call_args.kwargs["j2_moon"], 0.0)
        self.assertTrue(result.force_model_match)
        self.assertEqual(result.posterior_force_role_status, "verified")


# ---------------------------------------------------------------------------
# R0B-F3 (A) — executable mismatch consumer matrix
#
# A single declared campaign (truth J2 nonzero, estimator J2 = 0) is locked at
# EVERY production consumer boundary. The campaign-level evidence runs the real
# run_configured_scenario; the estimator-internal boundaries drive the real
# production estimator orchestrators (run_batch_arc_sequence / run_lunar_ukf)
# with the estimator-side J2 and spy the actual propagation primitives. Every
# spy wraps the REAL function (records j2, then delegates) so a consumer that
# never ran cannot pass by an empty call list.
# ---------------------------------------------------------------------------

def _j2_recording_wrapper(real_fn, sink, key):
    def _wrapped(*args, **kwargs):
        sink.setdefault(key, []).append(kwargs.get("j2_moon"))
        return real_fn(*args, **kwargs)
    return _wrapped


class AExecutableMismatchConsumerMatrixTests(unittest.TestCase):
    ESTIMATOR_J2 = 0.0

    def test_campaign_dispatch_provenance_and_contract_runtime_equality(self):
        """Campaign level: truth J2 nonzero, estimator dispatch J2 = 0, and the
        reported contracts equal the runtime values; fingerprints/match/reason
        and the manifest SHA are consistent across result, CSV, and file."""
        config = _config(
            j2_moon=J2_MOON,
            allow_explicit_force_model_mismatch=True,
            force_model_mismatch_reason="R0B-F3 truth-J2 vs point-mass estimator",
            duration_h=1.0,
            sample_step_s=600.0,
        )
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            h = _RunnerHarness(stack, tmp)
            def _fake_plot(scenarios, output_path, *, title=""):
                Path(output_path).write_bytes(b"\x89PNG\r\n")
                return Path(output_path)
            stack.enter_context(mock.patch.object(runner, "plot_scenario_comparison", side_effect=_fake_plot))
            result = runner.run_configured_scenario(
                config, truth_j2_moon=J2_MOON, estimator_j2_moon=self.ESTIMATOR_J2, manifest_dir=tmp
            )
            manifest_path = Path(tmp) / f"{config.name}_force_model_manifest.json"
            raw = manifest_path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
        # boundary 1: truth propagation ran with nonzero J2
        self.assertEqual(h.truth_spy.call_count, 1)
        self.assertEqual(h.truth_spy.call_args.kwargs["j2_moon"], J2_MOON)
        # boundary 2: estimator batch dispatch ran with J2 = 0
        self.assertEqual(h.batch_spy.call_count, 1)
        self.assertEqual(h.batch_spy.call_args.kwargs["j2_moon"], self.ESTIMATOR_J2)
        # fingerprints / match / reason
        self.assertNotEqual(result.truth_force_fingerprint, result.estimator_force_fingerprint)
        self.assertFalse(result.force_model_match)
        self.assertEqual(result.force_model_mismatch_reason, config.force_model_mismatch_reason)
        # reason identical across result, manifest
        self.assertEqual(payload["mismatch_policy"]["reason"], config.force_model_mismatch_reason)
        # reported contract J2 == runtime J2 (both sides)
        self.assertEqual(payload["truth"]["canonical_payload"]["lunar_j2"]["coefficient"], J2_MOON)
        self.assertEqual(
            payload["estimator"]["canonical_payload"]["lunar_j2"]["coefficient"], self.ESTIMATOR_J2
        )
        # 3-way SHA: file bytes == result provenance (CSV SHA is checked in the CLI tests)
        file_sha = "sha256:" + hashlib.sha256(raw).hexdigest()
        self.assertEqual(file_sha, result.force_contract_manifest_sha256)

    def _drive_batch(self, estimator_type, covariance_form, sink, *, with_ephemeris=False):
        """Run the REAL run_batch_arc_sequence for one estimator config."""
        from lunar_od.scenarios import run_batch_arc_sequence
        from tests.test_r0a_force_config_parity import (
            _prepared_position_arc,
            _synthetic_ephemeris,
        )
        import lunar_od.estimators as est
        import lunar_od.filters as filt
        import lunar_od.scenarios as scen

        arc, get_earth, get_sun = _prepared_position_arc()
        MU_MOON = 4.9028000661e12
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                est, "propagate_state", _j2_recording_wrapper(est.propagate_state, sink, "bls_state")))
            stack.enter_context(mock.patch.object(
                est, "propagate_augmented_state",
                _j2_recording_wrapper(est.propagate_augmented_state, sink, "bls_stm")))
            stack.enter_context(mock.patch.object(
                filt, "propagate_state", _j2_recording_wrapper(filt.propagate_state, sink, "ukf_state")))
            stack.enter_context(mock.patch.object(
                filt, "propagate_augmented_state",
                _j2_recording_wrapper(filt.propagate_augmented_state, sink, "ukf_stm")))
            kwargs = dict(
                cold_start_bank=(np.zeros(6),),
                ukf_covariance_form=covariance_form,
                j2_moon=self.ESTIMATOR_J2,
            )
            if with_ephemeris:
                real_factory = scen.make_fast_sigma_propagator

                def _factory_spy(*a, **k):
                    sink.setdefault("fast_factory", []).append(k.get("j2_moon"))
                    closure = real_factory(*a, **k)
                    if closure is None:
                        return None
                    def _rt(t0, t1, x6):
                        sink.setdefault("fast_runtime", []).append(k.get("j2_moon"))
                        return closure(t0, t1, x6)
                    return _rt
                stack.enter_context(mock.patch.object(scen, "make_fast_sigma_propagator", _factory_spy))
                kwargs["ephemeris"] = _synthetic_ephemeris(700.0)
            run_batch_arc_sequence(
                (arc,), "position", "cold", estimator_type, MU_MOON, 0.0, 0.0,
                get_earth, get_sun, **kwargs,
            )

    def test_bls_augmented_state_and_stm_consume_estimator_j2(self):
        # BLS propagates state AND STM together via the augmented propagation
        # (augmented state = 6 dynamic + 36 STM), so one propagate_augmented_state
        # call carries both boundaries; it must consume the estimator J2.
        sink: dict = {}
        self._drive_batch("bls_lm", "square_root", sink)
        self.assertGreater(
            len(sink.get("bls_stm", [])), 0, "BLS augmented (state+STM) propagation did not run"
        )
        self.assertTrue(all(v == self.ESTIMATOR_J2 for v in sink["bls_stm"]))

    def test_standard_ukf_process_consumes_estimator_j2(self):
        sink: dict = {}
        self._drive_batch("ukf", "standard", sink)
        self.assertGreater(len(sink.get("ukf_state", [])), 0, "standard UKF process propagation did not run")
        self.assertTrue(all(v == self.ESTIMATOR_J2 for v in sink["ukf_state"]))

    def test_square_root_ukf_process_consumes_estimator_j2(self):
        sink: dict = {}
        self._drive_batch("ukf", "square_root", sink)
        self.assertGreater(len(sink.get("ukf_state", [])), 0, "square-root UKF process propagation did not run")
        self.assertTrue(all(v == self.ESTIMATOR_J2 for v in sink["ukf_state"]))

    def test_optional_ukf_stm_consumes_estimator_j2(self):
        """Optional UKF STM path (use_stm_linearization=True) runs augmented
        propagation with the estimator J2."""
        from lunar_od.filters import run_lunar_ukf
        import lunar_od.filters as filt
        from tests.test_r0a_force_config_parity import _position_case, _ukf_inputs, MU_MOON

        x0, t_pass, truth, pass_geo, obs, get_earth, get_sun = _position_case()
        x_start, p0 = _ukf_inputs(x0, truth)
        sink: dict = {}
        with mock.patch.object(
            filt, "propagate_augmented_state",
            _j2_recording_wrapper(filt.propagate_augmented_state, sink, "ukf_stm")
        ):
            run_lunar_ukf(
                t_pass, obs, x_start, p0, pass_geo, MU_MOON, 0.0, 0.0, get_earth, get_sun,
                covariance_form="square_root", rtol=1e-10, atol=1e-11,
                use_stm_linearization=True, j2_moon=self.ESTIMATOR_J2,
            )
        self.assertGreater(len(sink.get("ukf_stm", [])), 0, "optional UKF STM did not run")
        self.assertTrue(all(v == self.ESTIMATOR_J2 for v in sink["ukf_stm"]))

    def test_fast_sigma_factory_and_runtime_consume_estimator_j2(self):
        sink: dict = {}
        self._drive_batch("ukf", "square_root", sink, with_ephemeris=True)
        self.assertGreater(len(sink.get("fast_factory", [])), 0, "fast-sigma factory did not run")
        self.assertTrue(all(v == self.ESTIMATOR_J2 for v in sink["fast_factory"]))
        if not sink.get("fast_runtime"):
            self.skipTest("numba unavailable; fast-sigma runtime closure not built")
        self.assertTrue(all(v == self.ESTIMATOR_J2 for v in sink["fast_runtime"]))


if __name__ == "__main__":
    unittest.main()
