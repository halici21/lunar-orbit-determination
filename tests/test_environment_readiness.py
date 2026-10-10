"""Preflight truthfulness, path propagation, and non-blocking desktop contracts."""
import json
import os
from pathlib import Path
import sys
import time

import pytest
from PyQt5.QtCore import QSettings, QTimer
from PyQt5.QtWidgets import QApplication

from lunar_od.spice_loader import REQUIRED_KERNELS
from services import environment_readiness as readiness
from controllers.environment_controller import EnvironmentController
from workers.process_worker import ProcessWorker


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def wait(app, condition, timeout=30):
    end = time.monotonic() + timeout
    while not condition() and time.monotonic() < end:
        app.processEvents()
        time.sleep(.005)
    assert condition()


@pytest.fixture
def file_only(monkeypatch):
    monkeypatch.setattr(readiness, "_runtime_checks", lambda: [])


def make_kernels(folder):
    folder.mkdir()
    for name in REQUIRED_KERNELS:
        (folder / name).write_bytes(b"invalid-but-nonempty")
    return folder


def test_explicit_missing_directory_cannot_fall_back(tmp_path, file_only, monkeypatch):
    fallback = make_kernels(tmp_path / "fallback")
    monkeypatch.setenv("LUNAR_OD_KERNEL_DIR", str(fallback))
    report = readiness.inspect_environment(kernel_dir=tmp_path / "chosen-missing")
    assert report["ready"] is False
    assert report["kernel_dir"] is None
    assert report["searched_paths"] == [str(tmp_path / "chosen-missing")]
    assert len(report["kernels"]) == 10


def test_empty_kernel_and_directory_named_as_kernel_block_spice(tmp_path, file_only, monkeypatch):
    folder = make_kernels(tmp_path / "kernels")
    (folder / REQUIRED_KERNELS[1]).write_bytes(b"")
    (folder / REQUIRED_KERNELS[2]).unlink()
    (folder / REQUIRED_KERNELS[2]).mkdir()
    calls = []
    monkeypatch.setattr(readiness, "_check_spice", lambda *args: calls.append(args))
    report = readiness.inspect_environment(kernel_dir=folder)
    failures = {r["name"] for r in report["kernels"] if r["status"] == "error"}
    assert failures == {"moon_de440_220930.txt", "moon_pa_de440_200625.bpc"}
    assert report["ready"] is False
    assert calls == []


def test_import_failure_retains_complete_kernel_inventory(tmp_path, monkeypatch):
    folder = make_kernels(tmp_path / "kernels")
    monkeypatch.setattr(readiness, "_runtime_checks", lambda: [
        {"name": "spiceypy", "status": "error", "detail": "broken import"}])
    report = readiness.inspect_environment(kernel_dir=folder)
    assert report["ready"] is False
    assert all(r["status"] == "ok" for r in report["kernels"])
    assert report["checks"][-1]["status"] == "pending"


def test_spice_failure_is_not_mistaken_for_file_readiness(tmp_path, file_only, monkeypatch):
    folder = make_kernels(tmp_path / "kernels")
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    def broken(*args):
        raise RuntimeError("SPICE(BADARCHTYPE)")
    monkeypatch.setattr(readiness, "_check_spice", broken)
    report = readiness.inspect_environment(kernel_dir=folder, output_dir=tmp_path / "new-results")
    assert report["ready"] is False
    assert "BADARCHTYPE" in report["checks"][-1]["detail"]
    assert not (tmp_path / "new-results").exists()
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}


def test_grid_endpoint_and_spice_pool_cleanup_match_runner(tmp_path, monkeypatch):
    import spiceypy as spice
    calls, cleared = [], []
    monkeypatch.setattr(spice, "str2et", lambda epoch: 1000.)
    monkeypatch.setattr(spice, "spkezr", lambda body, et, *args: calls.append((body, et)))
    monkeypatch.setattr(spice, "sxform", lambda *args: None)
    monkeypatch.setattr(spice, "pxform", lambda *args: None)
    monkeypatch.setattr(spice, "kclear", lambda: cleared.append(True))
    report = readiness._check_spice({"load_spice_kernels": lambda path: None}, tmp_path, 1.01, 600., 3600.)
    assert {et for _, et in calls} == {1000., 4600., 8200.}
    assert report["status"] == "ok"
    assert "bütün epokların" in report["detail"]
    assert cleared == [True]
    def broken(*args):
        raise RuntimeError("cannot load")
    with pytest.raises(RuntimeError):
        readiness._check_spice({"load_spice_kernels": broken}, tmp_path, 1., 600., 3600.)
    assert cleared == [True, True]


def test_selected_kernel_is_persistent_and_auto_restores_inherited(tmp_path, monkeypatch, app):
    monkeypatch.setenv("LUNAR_OD_KERNEL_DIR", "inherited-kernels")
    settings = QSettings(str(tmp_path / "preferences.ini"), QSettings.IniFormat)
    monkeypatch.setattr(EnvironmentController, "refresh", lambda *args, **kwargs: True)
    controller = EnvironmentController(settings)
    selected = tmp_path / "user & selected kernels"
    assert controller.select_directory(selected)
    assert os.environ["LUNAR_OD_KERNEL_DIR"] == str(selected.resolve())
    reopened = QSettings(str(tmp_path / "preferences.ini"), QSettings.IniFormat)
    assert reopened.value(readiness.KERNEL_KEY) == str(selected.resolve())
    assert controller.select_directory("")
    assert os.environ["LUNAR_OD_KERNEL_DIR"] == "inherited-kernels"
    controller.shutdown()


def test_reused_process_receives_new_kernel_preference_and_drops_old_overrides(tmp_path, monkeypatch, app):
    worker = ProcessWorker()
    lines, done = [], []
    worker.output_line.connect(lines.append)
    worker.finished.connect(done.append)
    code = "import os,json; print(json.dumps([os.getenv('LUNAR_OD_KERNEL_DIR'),os.getenv('LUNAR_OD_TEST_OLD')]))"
    monkeypatch.setenv("LUNAR_OD_KERNEL_DIR", "first")
    monkeypatch.delenv("LUNAR_OD_TEST_OLD", raising=False)
    worker.run(sys.executable, ["-u", "-c", code], env={"LUNAR_OD_TEST_OLD": "one"})
    wait(app, lambda: len(done) == 1)
    monkeypatch.setenv("LUNAR_OD_KERNEL_DIR", "second")
    worker.run(sys.executable, ["-u", "-c", code])
    wait(app, lambda: len(done) == 2)
    assert done == [0, 0]
    assert [json.loads(line) for line in lines] == [["first", "one"], ["second", None]]


def test_real_probe_rejects_corrupt_kernels_without_blocking_event_loop(tmp_path, app):
    folder = make_kernels(tmp_path / "corrupt & kernels")
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    worker = ProcessWorker()
    lines, done, ticks = [], [], []
    worker.output_line.connect(lines.append)
    worker.finished.connect(done.append)
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(True))
    timer.start(10)
    try:
        worker.run(sys.executable, ["-u", str(readiness.ROOT / "desktop_app/environment_probe.py"),
                   "--kernel-dir", str(folder)], env={"PYTHONIOENCODING": "utf-8"})
        wait(app, lambda: bool(done))
        reports = [json.loads(line.removeprefix("LUNAR_OD_ENVIRONMENT ")) for line in lines
                   if line.startswith("LUNAR_OD_ENVIRONMENT ")]
        assert done == [0]
        assert len(reports) == 1 and reports[0]["ready"] is False
        assert reports[0]["checks"][-1]["status"] == "error"
        assert all(r["status"] == "ok" for r in reports[0]["kernels"])
        assert len(ticks) >= 3
        assert before == {p.name: p.read_bytes() for p in folder.iterdir()}
    finally:
        timer.stop()
        worker.cancel()
        worker._proc.waitForFinished(2000)


@pytest.mark.parametrize("ready", [False, True])
def test_scenario_launch_is_deferred_until_preflight_and_no_failed_job_is_created(tmp_path, monkeypatch, app, ready):
    import app as desktop
    from services import scenario_execution
    settings = QSettings(str(tmp_path / "window.ini"), QSettings.IniFormat)
    monkeypatch.setattr(desktop, "QSettings", lambda *args: settings)
    callbacks = []
    def capture(self, **kwargs):
        if kwargs.get("callback"):
            callbacks.append(kwargs["callback"])
        return True
    monkeypatch.setattr(EnvironmentController, "refresh", capture)
    window = desktop.MainWindow()
    launched = []
    monkeypatch.setattr(window, "run_desktop_job", lambda *args: launched.append(args))
    jobs = []
    def prepare(model, root, **kwargs):
        jobs.append(model)
        return tmp_path / "job.json", tmp_path, "id"
    monkeypatch.setattr(scenario_execution, "prepare_desktop_job", prepare)
    try:
        window.navigate("scenario")
        page = window._pages["scenario"]
        page._run_scenario()
        assert len(callbacks) == 1
        assert not page.runScenarioBtn.isEnabled()
        assert jobs == launched == []
        callbacks[0](ready)
        assert page.runScenarioBtn.isEnabled()
        assert len(jobs) == len(launched) == int(ready)
        if not ready:
            assert window._active_page == "settings"
            assert hasattr(window._pages["settings"], "environment_panel")
        assert not (tmp_path / "job.json").exists()
    finally:
        window.close()


def test_closing_window_cancels_delayed_startup(app, tmp_path):
    controller = EnvironmentController(QSettings(str(tmp_path / "closed.ini"), QSettings.IniFormat))
    controller.shutdown()
    assert controller.refresh() is False
    assert not controller.busy


def test_timeout_never_launches_pending_scenario(tmp_path, app):
    controller = EnvironmentController(QSettings(str(tmp_path / "timeout.ini"), QSettings.IniFormat))
    results = []
    controller._callback = results.append
    controller._timeout()
    controller._on_finished(-1)
    assert results == [False]
    assert controller.report is None
    assert "60 saniyede" in controller.error
    controller.shutdown()


def test_inconsistent_report_cannot_trigger_solver(tmp_path, app):
    controller = EnvironmentController(QSettings(str(tmp_path / "bad-report.ini"), QSettings.IniFormat))
    results = []
    controller._callback = results.append
    controller._on_line("LUNAR_OD_ENVIRONMENT " + json.dumps({
        "schema": "lunar-od-environment-v1", "ready": True, "checked_at": "2026-10-10T00:00:00+00:00",
        "checks": [{"name": "spiceypy", "status": "error", "detail": "cannot import"}], "kernels": []}))
    controller._on_finished(0)
    assert results == [False]
    assert controller.report is None
    assert "uyuşmuyor" in controller.error
    controller.shutdown()


def test_panel_shows_actionable_failures_before_healthy_packages(tmp_path, app):
    from widgets.environment_panel import EnvironmentPanel
    controller = EnvironmentController(QSettings(str(tmp_path / "panel.ini"), QSettings.IniFormat))
    controller.report = {"ready": False, "checked_at": "2026-10-10T00:00:00+00:00", "checks": [
        {"name": "numpy", "status": "ok", "detail": "2.4.3"},
        {"name": "Kernel klasörü", "status": "error", "detail": "missing"}],
        "kernels": [{"name": "moon_pa_de440_200625.bpc", "status": "error", "detail": "missing"}]}
    panel = EnvironmentPanel(controller)
    try:
        assert panel.table.item(0, 0).text() == "Kernel klasörü"
        assert panel.table.item(1, 0).text() == "moon_pa_de440_200625.bpc"
        assert panel.table.item(2, 0).text() == "numpy"
        assert "eksikler" in panel.summary.text()
        assert "2026-10-10T" not in panel.summary.text()
    finally:
        panel.close()
        controller.shutdown()
