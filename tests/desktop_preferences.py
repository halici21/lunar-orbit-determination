"""Opt-in pytest plugin: isolate desktop settings from user preferences."""
import pytest
from PyQt5.QtCore import QSettings


@pytest.fixture(autouse=True)
def isolated_window_settings(monkeypatch, tmp_path):
    import app as desktop
    settings = QSettings(str(tmp_path / "isolated-window.ini"), QSettings.IniFormat)
    monkeypatch.setattr(desktop, "QSettings", lambda *args, **kwargs: settings)
