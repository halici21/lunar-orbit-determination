from __future__ import annotations

import os
from pathlib import Path

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
