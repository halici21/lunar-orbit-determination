"""Exercise the actual tutorial menu, child solver, reader and cancellation."""
import time
from pathlib import Path
import pytest
from PyQt5.QtWidgets import QApplication
from services import project_paths
from services.tracking_results import load_tracking_results
from services.run_comparison import load_run_comparison


def wait(qt, predicate, timeout=60):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        qt.processEvents()
        time.sleep(.005)
    qt.processEvents()
    assert predicate(), "Tutorial Qt workflow timed out"


def _isolate_session(monkeypatch, desktop, tmp_path):
    """Never let a test window save its session into the user's real settings."""
    from PyQt5.QtCore import QSettings
    settings = QSettings(str(tmp_path / "tutorial-window.ini"), QSettings.IniFormat)
    monkeypatch.setattr(desktop, "QSettings", lambda *args, **kwargs: settings)


def test_menu_runs_real_solver_and_produces_two_readable_seeded_runs(monkeypatch, tmp_path):
    import app as desktop
    _isolate_session(monkeypatch, desktop, tmp_path)
    monkeypatch.setattr(project_paths, "RESULTS_DIR", tmp_path)
    qt = QApplication.instance() or QApplication([])
    window = desktop.MainWindow()
    try:
        action = window.menuBar().actions()[-1].menu().actions()[0]
        action.trigger()
        monitor = window._pages["run_monitor"]
        wait(qt, lambda: monitor._result_path is not None and not monitor._worker.is_running)
        path = monitor._result_path
        assert path.is_relative_to(tmp_path)
        result = load_tracking_results(path)
        assert len(result.updates) == 124 and result.metadata["seed"] == 41000
        assert result.metadata["frame"].startswith("Synthetic")
        assert "Level 0" in result.metadata["producer"]
        assert result.updates[-1]["position_error_m"] < 1000.
        other = path.with_name("tutorial_seed_41001_ukf_updates.csv")
        comparison = load_run_comparison(path, other)
        assert len(comparison.pairs) == 124
        assert comparison.right.metadata["seed"] == 41001
        assert comparison.differences
    finally:
        window.close()
        qt.processEvents()


def test_cancelled_tutorial_cannot_publish_success(monkeypatch, tmp_path):
    import app as desktop
    _isolate_session(monkeypatch, desktop, tmp_path)
    monkeypatch.setattr(project_paths, "RESULTS_DIR", tmp_path)
    qt = QApplication.instance() or QApplication([])
    window = desktop.MainWindow()
    try:
        window.start_tutorial()
        monitor = window._pages["run_monitor"]
        folder = Path(monitor._job_context["folder"])
        monitor._cancel()
        wait(qt, lambda: not monitor._worker.is_running)
        assert monitor._result_path is None
        assert not (folder / "completed.json").exists()
    finally:
        window.close()
        qt.processEvents()
