from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
import types
from pathlib import Path
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from app import _NAV, _TITLES
from controllers.run_monitor_controller import RunMonitorController
from controllers.system_pages_controller import (
    DynamicsController,
    EstimatorsController,
    MeasurementsController,
    SettingsController,
)


_APP = None


def _app():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def test_desktop_navigation_registry_contains_core_pages():
    _app()
    expected = {
        "dashboard",
        "results",
        "run_monitor",
        "comparison",
        "scenario",
        "dynamics",
        "measurements",
        "estimators",
        "stations",
        "visibility",
        "analysis",
        "ground_track",
        "settings",
    }
    nav_ids = {page_id for page_id, _label, _hint in _NAV}
    assert expected.issubset(nav_ids)
    assert expected.issubset(set(_TITLES))


def test_new_system_page_shells_load_offscreen():
    _app()
    for cls in (DynamicsController, MeasurementsController, EstimatorsController, SettingsController):
        widget = cls()
        assert widget.mainLayout.count() >= 3


def test_run_monitor_preset_keeps_extra_args_in_command_preview():
    _app()
    monitor = RunMonitorController()
    script = Path("python_port/examples/scenario_config_cli.py").resolve()
    monitor.preset_script(str(script), extra_args=["C:/tmp/scenario file.json"])

    text = monitor.commandPreviewLabel.text()
    assert "scenario_config_cli.py" in text
    assert "scenario file.json" in text
    assert "--help" not in text


# ---------------------------------------------------------------------------
# R0A-F1 — desktop analysis worker preflight + QSettings repair
#
# These drive the REAL _AnalysisWorker._do_run (never injecting the QSettings
# symbol) with only the heavy/external dependencies patched, so we can assert
# on call order and forwarded force parameters. scenario_config_from_mapping is
# left REAL so the shared Earth-J2 validator actually runs (F3).
# ---------------------------------------------------------------------------

from lunar_od.constants import J2_MOON_UNNORMALIZED  # noqa: E402
import lunar_od.scenario_config as _scenario_config_mod  # noqa: E402


def _analysis_controller():
    import controllers.analysis_controller as ac

    return ac


def _fake_ephemeris():
    return types.SimpleNamespace(
        earth_position=lambda t: np.zeros((np.size(np.asarray(t)), 3)),
        earth_velocity=lambda t: np.zeros((np.size(np.asarray(t)), 3)),
        sun_position=lambda t: np.zeros((np.size(np.asarray(t)), 3)),
    )


def _spec(estimator="bls_lm", variant_overrides=None):
    ac = _analysis_controller()
    if variant_overrides is None:
        variant_overrides = [{}]
    variants = [
        ac.VariantSpec(f"V{i}", dict(ov)) for i, ov in enumerate(variant_overrides)
    ]
    return ac.ComparisonSpec(
        title="R0A-F1", desc="preflight test", estimator=estimator, variants=variants
    )


_DESKTOP_OUTPUT_DIR = tempfile.mkdtemp(prefix="r0b_desktop_out_")


def _base_params(**overrides):
    base = dict(
        duration_h=1.0,
        sample_step_s=600.0,
        max_iter=5,
        noise=False,
        network="multi",
        start_mode="cold",
        measurement_type="range_rate",
        range_rate_physics="geometric_instantaneous",
        bias_mode=None,
        station_names=("S1",),
        output_dir=_DESKTOP_OUTPUT_DIR,
        j2_moon=0.0,
    )
    base.update(overrides)
    return base


def _empty_scenario_result():
    from lunar_od.scenarios import ScenarioResult

    return ScenarioResult(
        label="desktop",
        measurement_type="range_rate",
        start_mode="cold",
        arc_results=(),
        estimator_type="bls_lm",
    )


@contextlib.contextmanager
def _patched_worker_env(order_log=None):
    """Patch heavy/external deps of _AnalysisWorker._do_run; yield the spy dict.

    QSettings is deliberately NOT patched — the real module import must resolve
    it, which is exactly what Blocker 1 fixes.
    """
    fake_eph = _fake_ephemeris()
    fake_station = types.SimpleNamespace(name="S1")
    fake_result = _empty_scenario_result()

    def _logged(name, retval):
        def _fn(*args, **kwargs):
            if order_log is not None:
                order_log.append(name)
            return retval
        return mock.MagicMock(side_effect=_fn)

    truth_spy = _logged("truth", np.zeros((6, 6)))
    batch_spy = _logged("batch", fake_result)
    spice_stub = types.SimpleNamespace(str2et=lambda _s: 0.0, kclear=lambda: None)

    spies = {
        "truth": truth_spy,
        "batch": batch_spy,
        "state": mock.MagicMock(name="propagate_state"),
        "stm": mock.MagicMock(name="propagate_augmented_state"),
        "ukf": mock.MagicMock(name="run_lunar_ukf"),
        "fast_sigma": mock.MagicMock(name="make_fast_sigma_propagator"),
        "bls": mock.MagicMock(name="estimate_position_bls_lm"),
        "srif": mock.MagicMock(name="estimate_position_srif"),
    }

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.dict("sys.modules", {"spiceypy": spice_stub}))
        stack.enter_context(mock.patch("lunar_od.load_spice_kernels"))
        stack.enter_context(
            mock.patch("lunar_od.sample_moon_centered_ephemeris", return_value=fake_eph)
        )
        stack.enter_context(
            mock.patch(
                "lunar_od.sample_j2000_to_itrf93_transforms",
                side_effect=lambda _et, t: np.repeat(
                    np.eye(6)[None, :, :], np.size(np.asarray(t)), axis=0
                ),
            )
        )
        stack.enter_context(
            mock.patch("lunar_od.propagate_truth_with_ephemeris", truth_spy)
        )
        stack.enter_context(
            mock.patch("lunar_od.range_rate_stations", return_value=[fake_station])
        )
        stack.enter_context(
            mock.patch(
                "lunar_od.analyze_visibility_gap_with_transforms",
                side_effect=lambda *a, **k: (
                    np.array([0]), np.array([5]), np.ones(6, dtype=bool), None,
                ),
            )
        )
        stack.enter_context(
            mock.patch("lunar_od.build_measurement_arcs", return_value=(object(),))
        )
        stack.enter_context(
            mock.patch("lunar_od.make_cold_start_bank", return_value=(np.zeros(6),))
        )
        stack.enter_context(mock.patch("lunar_od.run_batch_arc_sequence", batch_spy))
        # Propagation/estimator entries that must never fire on the Earth-J2 path.
        stack.enter_context(mock.patch("lunar_od.dynamics.propagate_state", spies["state"]))
        stack.enter_context(
            mock.patch("lunar_od.dynamics.propagate_augmented_state", spies["stm"])
        )
        stack.enter_context(mock.patch("lunar_od.scenarios.run_lunar_ukf", spies["ukf"]))
        stack.enter_context(
            mock.patch(
                "lunar_od.scenarios.make_fast_sigma_propagator", spies["fast_sigma"]
            )
        )
        stack.enter_context(
            mock.patch("lunar_od.scenarios.estimate_position_bls_lm", spies["bls"])
        )
        stack.enter_context(
            mock.patch("lunar_od.scenarios.estimate_position_srif", spies["srif"])
        )
        yield spies


def test_f1_real_worker_qsettings_resolves_and_forwards_lunar_j2():
    """F1: real _do_run reaches config+dispatch (no NameError) and forwards J2."""
    _app()
    ac = _analysis_controller()
    worker = ac._AnalysisWorker(
        _spec("bls_lm"), _base_params(j2_moon=J2_MOON_UNNORMALIZED)
    )
    with _patched_worker_env() as spies:
        worker._do_run()  # must NOT raise NameError: name 'QSettings' is not defined
    assert spies["truth"].call_count == 1
    assert spies["truth"].call_args.kwargs["j2_moon"] == J2_MOON_UNNORMALIZED
    assert spies["batch"].call_count == 1
    assert spies["batch"].call_args.kwargs["j2_moon"] == J2_MOON_UNNORMALIZED


def test_f2_desktop_earth_j2_zero_propagation():
    """F2: Earth J2 on a desktop variant → controlled error, no propagation."""
    _app()
    ac = _analysis_controller()
    worker = ac._AnalysisWorker(
        _spec("bls_lm", [{"enable_earth_j2": True}]), _base_params()
    )
    with _patched_worker_env() as spies:
        with mock.patch(
            "lunar_od.scenario_config.validate_official_earth_j2_support",
            wraps=_scenario_config_mod.validate_official_earth_j2_support,
        ):
            try:
                worker._do_run()
                raised = None
            except ValueError as exc:
                raised = exc
    assert raised is not None
    assert "enable_earth_j2=True is not supported on the official OD path" in str(raised)
    for name, spy in spies.items():
        assert spy.call_count == 0, f"{name} was called before Earth-J2 rejection"


def test_f3_desktop_preflight_uses_shared_validator():
    """F3: desktop preflight routes through the shared scenario_config validator."""
    _app()
    ac = _analysis_controller()
    worker = ac._AnalysisWorker(_spec("bls_lm", [{}, {}]), _base_params())
    with _patched_worker_env():
        with mock.patch(
            "lunar_od.scenario_config.validate_official_earth_j2_support",
            wraps=_scenario_config_mod.validate_official_earth_j2_support,
        ) as shared_gate:
            worker._do_run()
    # Called once per variant during preflight, with the (False) enable flag.
    assert shared_gate.call_count >= 2
    for call in shared_gate.call_args_list:
        assert call.args[0] is False


def test_f4_real_worker_lunar_j2_parity_truth_and_dispatch():
    """F4: nonzero lunar J2 reaches BOTH truth and batch dispatch, same value."""
    _app()
    ac = _analysis_controller()
    worker = ac._AnalysisWorker(
        _spec("ukf"), _base_params(j2_moon=J2_MOON_UNNORMALIZED)
    )
    with _patched_worker_env() as spies:
        worker._do_run()
    assert spies["truth"].call_args.kwargs["j2_moon"] == J2_MOON_UNNORMALIZED
    assert spies["batch"].call_args.kwargs["j2_moon"] == J2_MOON_UNNORMALIZED


def test_f5_common_truth_propagated_once_after_preflight():
    """F5: multiple valid variants share one truth propagation, after preflight."""
    _app()
    ac = _analysis_controller()
    worker = ac._AnalysisWorker(
        _spec("bls_lm", [{"max_iter": 5}, {"max_iter": 7}]), _base_params()
    )
    order: list[str] = []
    with _patched_worker_env(order_log=order):
        with mock.patch(
            "lunar_od.scenario_config.validate_official_earth_j2_support",
            side_effect=lambda *a, **k: order.append("validate"),
        ):
            worker._do_run()
    assert order.count("truth") == 1  # common truth, not per-variant
    assert order.count("batch") == 2  # one dispatch per valid variant
    # Config validation (Stage A) precedes the single truth propagation, and
    # truth precedes the first estimator dispatch. (The per-variant Stage B
    # force binding validates again after truth, so the exact validate count is
    # an implementation detail; the ordering invariant is what matters.)
    truth_pos = order.index("truth")
    assert "validate" in order[:truth_pos]
    assert truth_pos < order.index("batch")


def test_f6_zero_j2_default_path_unchanged():
    """F6: default (zero-J2) desktop path forwards j2_moon=0.0 and completes."""
    _app()
    ac = _analysis_controller()
    worker = ac._AnalysisWorker(_spec("bls_lm"), _base_params())
    with _patched_worker_env() as spies:
        worker._do_run()
    assert spies["truth"].call_args.kwargs["j2_moon"] == 0.0
    assert spies["batch"].call_args.kwargs["j2_moon"] == 0.0
    # default path never hits the fail-closed propagation spies directly
    assert spies["ukf"].call_count == 0


# ---------------------------------------------------------------------------
# R0B-F1 (V04) — real desktop worker force-model provenance
# ---------------------------------------------------------------------------

from lunar_od.constants import J2_MOON_UNNORMALIZED as _J2_MOON  # noqa: E402
from lunar_od.scenario_config import (  # noqa: E402
    force_execution_spec_from_scenario_config,
    scenario_config_from_mapping,
)


def _run_worker_capture_results(spec, base):
    """Run the real _do_run and capture the emitted (label, ScenarioResult)."""
    ac = _analysis_controller()
    worker = ac._AnalysisWorker(spec, base)
    captured: list = []
    worker.variant_done.connect(lambda label, result: captured.append((label, result)))
    with _patched_worker_env() as spies:
        worker._do_run()
    return captured, spies


def _fixture_earth_gm():
    fixture = json.loads(
        (Path(str(_analysis_controller().FIXTURE))).read_text(encoding="utf-8")
    )
    return float(np.asarray(fixture["constants"]["mu_earth_km3_s2"]).reshape(-1)[0] * 1e9)


import json  # noqa: E402


def test_f5_real_desktop_result_provenance_and_manifest_default():
    """F5: default-J2 real worker fills provenance fields and persists a manifest."""
    _app()
    captured, _ = _run_worker_capture_results(_spec("bls_lm"), _base_params())
    assert captured, "worker emitted no result"
    label, result = captured[0]
    assert result.truth_force_fingerprint.startswith("sha256:")
    assert result.estimator_force_fingerprint == result.truth_force_fingerprint
    assert result.force_model_match is True
    assert result.force_contract_manifest_sha256.startswith("sha256:")
    assert result.force_contract_schema_version
    # manifest persisted in the desktop output bundle, hash consistent
    manifest_path = Path(_DESKTOP_OUTPUT_DIR) / f"analysis_R0A-F1_{label}_force_model_manifest.json"
    assert manifest_path.is_file(), "desktop manifest not persisted"
    file_sha = "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert file_sha == result.force_contract_manifest_sha256


def test_f5_real_desktop_nonzero_j2_matches_contract_fingerprint():
    """F5: nonzero-J2 provenance fingerprint equals the spec built from runtime GM."""
    _app()
    captured, spies = _run_worker_capture_results(
        _spec("bls_lm"), _base_params(j2_moon=_J2_MOON)
    )
    label, result = captured[0]
    # runtime truth/batch J2 is the nonzero value
    assert spies["truth"].call_args.kwargs["j2_moon"] == _J2_MOON
    assert spies["batch"].call_args.kwargs["j2_moon"] == _J2_MOON
    # provenance fingerprint equals the spec built from the effective (fixture) GM
    config = scenario_config_from_mapping(
        {**{k: v for k, v in _base_params(j2_moon=_J2_MOON).items()
            if k in {"measurement_type", "start_mode", "network", "j2_moon"}},
         "name": "x", "estimator_type": "bls_lm"}
    )
    spec = force_execution_spec_from_scenario_config(config, mu_earth_m3_s2=_fixture_earth_gm())
    assert result.truth_force_fingerprint == spec.fingerprint()
    assert result.posterior_force_role_status == "verified"
    assert result.observability_force_role_status == "verified"
