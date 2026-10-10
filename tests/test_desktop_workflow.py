"""Workflow contracts, recorded epochs, and real QProcess lifecycle tests."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

from models.scenario_model import ScenarioModel
from services.scenario_execution import execution_payload, prepare_desktop_job, planned_segments
from services.scenario_artifacts import write_console_bundle, publish_completion
from services.tracking_results import load_tracking_results
from controllers.run_monitor_controller import RunMonitorController
from controllers.tracking_console_controller import TrackingConsoleController
from workers.process_worker import ProcessWorker


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def wait(app, condition, timeout=5):
    end = time.monotonic() + timeout
    while not condition() and time.monotonic() < end:
        app.processEvents()
        time.sleep(.005)
    assert condition()


def bundle(tmp_path, estimator="bls_lm"):
    model = ScenarioModel(name="fixture", estimator_type=estimator, output_dir=str(tmp_path))
    job, folder, run_id = prepare_desktop_job(model, tmp_path)
    payload = json.loads(job.read_text())
    (folder / "effective_config.json").write_text(json.dumps(payload["config"]))
    (folder / "fixture_force_model_manifest.json").write_text('{"fixture":true}')
    solved = SimpleNamespace(arc_id=7, estimated_state=np.array([1800000., 0., 0., 0., 1600., 0.]),
        final_position_error_m=25., final_velocity_error_mps=.2, ukf_normalized_mean_nis=1.3 if estimator == "ukf" else float("nan"),
        operational_success=False, stop_reason="MaxIter", observed_station_ids=(0, 2))
    result = SimpleNamespace(estimator_type=estimator, arc_results=(solved,))
    arc = SimpleNamespace(arc_id=7, t_pass_s=np.array([10., 50., 100.]), obs_data=np.array([[10.], [90.]]))
    truth = np.array([[1800025., 0., 0., 0., 1599.8, 0.], [1800000., 10000., 0., 0., 1600., 0.]])
    path = write_console_bundle(result, [arc], [10., 100.], truth, "2027-03-02 23:32:37",
        config=SimpleNamespace(name="fixture", output_dir=str(folder)), job_path=job,
        options=payload["options"], project_root=tmp_path)
    return job, folder, run_id, path


def test_form_translation_and_saved_inputs(tmp_path):
    model = ScenarioModel(name="exact", duration_days=.5, random_seed=0, noise_enabled=True,
        state_step_s=123, bls_damping_init=.04, ukf_alpha=.6, max_arcs=2, output_dir="chosen")
    config, options = execution_payload(model)
    assert config["duration_h"] == 12
    assert config["sample_step_s"] == 123
    assert config["bls_lambda0"] == .04
    assert config["ukf_alpha"] == .6
    assert config["noise"] is True
    assert options["measurement_seed"] == options["cold_start_seed"] == 0
    path, folder, run_id = prepare_desktop_job(model, tmp_path)
    saved = json.loads(path.read_text())
    assert saved["form"]["random_seed"] == 0
    assert saved["config"]["output_dir"] == str(folder)
    assert folder.parent == tmp_path / "chosen" / "desktop_runs"
    assert saved["run_id"] == run_id
    second = prepare_desktop_job(model, tmp_path)[1]
    assert second != folder


@pytest.mark.parametrize("overrides", [
    {"name":"../escape"}, {"network_name":"my-network"}, {"random_seed":-1},
    {"state_step_s":float("nan")}, {"max_arcs":0}, {"visibility_enabled":False},
    {"arc_duration_h":7, "arc_stride_h":6}, {"elevation_mask_deg":90},
])
def test_unexecutable_form_fails_explicitly(overrides):
    with pytest.raises(ValueError):
        execution_payload(ScenarioModel(**overrides))


def test_planned_windows_keep_each_boundary_once():
    starts, ends = planned_segments([0, 1800, 3600, 5400, 7200], 1, 1)
    assert starts == [0, 3]
    assert ends == [2, 4]
    assert planned_segments([0, 1000], 1, 2) == ([], [])


@pytest.mark.parametrize("estimator,epoch", [("bls_lm", 10.), ("srif", 10.), ("ukf", 90.)])
def test_actual_solver_state_epoch_and_failure_are_preserved(tmp_path, estimator, epoch):
    _, _, _, path = bundle(tmp_path, estimator)
    result = load_tracking_results(path)
    row = result.updates[0]
    assert row["t_s"] == epoch
    assert row["end_s"] == 100
    assert row["operational_success"] is False
    assert row["accepted"] is None
    assert row["station_id"] is None
    assert row["x_m"] == 1800000
    assert row["position_error_m"] == 25
    assert result.states[0]["x_m"] == 1800025
    assert result.metadata["state_role"] == "truth"
    assert result.metadata["record_kind"] == "arc_solution"
    assert len(result.sources) == 6


def test_console_does_not_label_batch_solutions_as_ukf_updates(app, tmp_path):
    _, _, _, path = bundle(tmp_path)
    console = TrackingConsoleController()
    try:
        console._accept_result(load_tracking_results(path))
        assert "Yay başlangıç epoku" in console.selection_detail.text()
        assert "Operasyonel başarı: Hayır" in console.selection_detail.text()
        assert console.table_model.COLUMNS[-1][0] == "operational_success"
        assert not console.accepted_check.isEnabled()
        assert console.result.metadata["estimator_type"] == "bls_lm"
    finally:
        console.close()


def test_failed_or_foreign_completion_cannot_open_previous_results(app, tmp_path, capsys):
    _, folder, run_id, path = bundle(tmp_path)
    publish_completion(folder, run_id, path, [path])
    message = capsys.readouterr().out.strip()
    monitor = RunMonitorController()
    monitor._job_context = {"folder":str(folder), "run_id":run_id}
    try:
        monitor._capture_result(message.replace(run_id, "different"))
        assert monitor._pending_result is None
        monitor._capture_result("LUNAR_OD_RESULT []")
        assert monitor._pending_result is None
        monitor._capture_result(message)
        assert monitor._pending_result == path
        monitor._on_finished(1)
        assert monitor._result_path is None
        assert not monitor.openConsoleBtn.isEnabled()
        monitor.autoConsoleCheck.setChecked(False)
        monitor._on_finished(0)
        assert monitor._result_path == path
        assert monitor.openConsoleBtn.isEnabled()
    finally:
        monitor.close()


def test_qprocess_keeps_ui_alive_and_flushes_last_line(app):
    worker = ProcessWorker()
    lines, finished, ticks = [], [], []
    worker.output_line.connect(lines.append)
    worker.finished.connect(finished.append)
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(10)
    worker.run(sys.executable, ["-u", "-c", "import time; time.sleep(.15); print('last', end='', flush=True)"])
    assert worker.is_running  # QProcess.Starting is also busy
    wait(app, lambda: bool(finished))
    timer.stop()
    assert finished == [0]
    assert lines == ["last"]
    assert len(ticks) >= 5


def test_qprocess_failed_start_restores_monitor(app, tmp_path):
    monitor = RunMonitorController()
    try:
        monitor.startBtn.setEnabled(False)
        monitor._worker.run(str(tmp_path / "missing.exe"), [])
        wait(app, lambda: monitor._last_exit_code is not None)
        assert monitor._last_exit_code == -1
        assert monitor.startBtn.isEnabled()
        assert not monitor.openConsoleBtn.isEnabled()
    finally:
        monitor.close()


def test_preset_cannot_replace_running_job(app, tmp_path):
    monitor = RunMonitorController()
    monitor.preset_script("old.py", extra_args=["old.json"])
    previous = monitor._override_command
    try:
        monitor._worker.run(sys.executable, ["-c", "import time; time.sleep(3)"])
        monitor.preset_script("new.py", extra_args=["new.json"])
        assert monitor._override_command == previous
        monitor._cancel()
        wait(app, lambda: monitor._last_exit_code is not None)
        assert not monitor.openConsoleBtn.isEnabled()
    finally:
        monitor.shutdown()
        monitor.close()


def test_builder_roundtrip_preserves_hidden_execution_settings(app):
    from controllers.scenario_builder_controller import ScenarioBuilderController
    builder = ScenarioBuilderController()
    model = ScenarioModel(ephemeris_step_s=321, min_visibility_samples=8, max_arcs=3,
        ukf_alpha=.72, bls_damping_init=.032)
    builder._model_to_widgets(model)
    recovered = builder._widgets_to_model()
    assert recovered.ephemeris_step_s == 321
    assert recovered.min_visibility_samples == 8
    assert recovered.max_arcs == 3
    assert recovered.ukf_alpha == .72
    assert recovered.bls_damping_init == .032
    builder.close()


def test_unsupported_imported_network_is_not_silently_replaced(app):
    from controllers.scenario_builder_controller import ScenarioBuilderController
    builder = ScenarioBuilderController()
    builder._model_to_widgets(ScenarioModel(network_name="unsupported_station_network"))
    builder.on_shown()
    recovered = builder._widgets_to_model()
    assert recovered.network_name == "unsupported_station_network"
    with pytest.raises(ValueError, match="single/multi"):
        execution_payload(recovered)
    builder.close()
