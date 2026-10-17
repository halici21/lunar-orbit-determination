"""Independent visibility requests must stay distinguishable from their last result."""
import csv
from unittest.mock import patch

import numpy as np
import pytest
from PyQt5.QtCore import QObject, Qt, QSettings, pyqtSignal
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from controllers import visibility_controller as visibility


class PendingWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, **parameters):
        super().__init__()
        self.parameters = parameters
        self.running = False

    def start(self):
        self.running = True

    def isRunning(self):
        return self.running

    def complete(self, data):
        self.running = False
        self.finished.emit(data)

    def fail(self):
        self.running = False
        self.failed.emit("Kernel unavailable\nDetailed test failure")


@pytest.fixture
def page(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "visibility.ini"), QSettings.IniFormat)
    monkeypatch.setattr("PyQt5.QtCore.QSettings", lambda *a, **k: settings)
    monkeypatch.setattr(visibility, "_VisWorker", PendingWorker)
    widget = visibility.VisibilityController()
    widget._deselect_all()
    widget._checkboxes["Madrid DSN"].setChecked(True)
    widget.durationSpin.setValue(1)
    widget.elevSpin.setValue(10)
    widget.gapSpin.setValue(30)
    widget.show(); app.processEvents()
    yield widget, app, settings
    widget.close(); app.processEvents()


def payload(name="Madrid DSN", duration=1):
    # Controlled UI data; this is not a newly qualified visibility solution.
    return {"t_eval_s": np.array([0., 600., 1200.]),
            "raw_mask": np.array([[True, False, True]]),
            "station_names": [name], "duration_days": duration,
            "epoch_utc": "2000-01-01T00:00:00"}


def complete(page, data=None):
    page._compute()
    page._worker.complete(data if data is not None else payload())


def csv_rows(page, target):
    with patch.object(visibility.QFileDialog, "getSaveFileName", return_value=(str(target), "CSV")):
        page._export_csv()
    with target.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def test_visibility_busy_request_locks_inputs_exports_and_settings_resync(page):
    widget, app, settings = page
    widget._compute()
    worker = widget._worker
    assert worker.parameters == {"station_names": ["Madrid DSN"], "duration_days": 1.,
                                 "elev_deg": 10., "max_gap_min": 30.}
    for control in (widget.paramsGroup, widget.stationScrollArea, widget.selectAllBtn,
                    widget.deselectAllBtn, widget.computeBtn, widget.exportPngBtn, widget.exportCsvBtn):
        assert not control.isEnabled()
    settings.setValue("dynamics/duration_days", 3.)
    widget.on_shown()
    assert widget.durationSpin.value() == 1.
    assert "Hesap sürüyor" in widget._result_context.text()
    widget._compute()
    assert widget._worker is worker
    worker.complete(payload())
    assert widget.paramsGroup.isEnabled() and widget.computeBtn.isEnabled()


@pytest.mark.parametrize("field", ["durationSpin", "elevSpin", "gapSpin"])
def test_visibility_edited_inputs_keep_original_result_context_and_raw_csv(page, tmp_path, field):
    widget, app, _ = page
    data = payload(); complete(widget, data)
    old_context = widget._result_context.text()
    assert "2000-01-01T00:00:00" in old_context
    assert "10°" in old_context and "30 dk" in old_context
    control = getattr(widget, field); original = control.value()
    control.setValue(original+1)
    assert "Ayarlar değişti" in widget._result_context.text()
    assert old_context in widget._result_context.text()
    assert widget._last_data is data
    assert csv_rows(widget, tmp_path / "previous.csv") == [
        ["time_s", "time_days", "Madrid DSN", "network_visible"],
        ["0.000", "0.00000000", "1", "1"],
        ["600.000", "0.00694444", "0", "0"],
        ["1200.000", "0.01388889", "1", "1"],
    ]
    control.setValue(original)
    assert widget._result_context.text() == old_context


def test_visibility_station_change_and_rerun_replace_data_plot_and_caption_together(page, tmp_path):
    widget, app, _ = page
    original = payload(); complete(widget, original)
    widget._checkboxes["Madrid DSN"].setChecked(False)
    widget._checkboxes["Goldstone DSN"].setChecked(True)
    assert "Ayarlar değişti" in widget._result_context.text()
    assert "Madrid DSN" in widget._result_context.toolTip()
    widget._compute()
    assert not widget.exportPngBtn.isEnabled() and not widget.exportCsvBtn.isEnabled()
    newer = payload("Goldstone DSN"); newer["raw_mask"] = np.array([[False, True, False]])
    widget._worker.complete(newer)
    assert widget._last_data is newer
    assert [t.get_text() for t in widget._fig.axes[0].get_yticklabels()] == ["Goldstone DSN"]
    assert "Ayarlar değişti" not in widget._result_context.text()
    assert "Goldstone DSN" in widget._result_context.toolTip()
    assert csv_rows(widget, tmp_path / "new.csv")[0][2] == "Goldstone DSN"
    assert np.array_equal(original["raw_mask"], [[True, False, True]])


@pytest.mark.parametrize("previous", [False, True])
def test_failed_visibility_request_preserves_only_a_known_previous_result(page, previous):
    widget, app, _ = page
    if previous:
        complete(widget)
    old_data, old_figure = widget._last_data, widget._fig
    widget.elevSpin.setValue(20); widget._compute()
    with patch.object(visibility, "_show_error_dialog") as dialog:
        widget._worker.fail()
    dialog.assert_called_once()
    assert widget._last_data is old_data and widget._fig is old_figure
    assert widget.exportCsvBtn.isEnabled() == previous
    assert widget.exportPngBtn.isEnabled() == previous
    assert widget.computeBtn.isEnabled() and widget.paramsGroup.isEnabled()
    assert "Error:" in widget.statusLabel.text()
    assert ("önceki" if previous else "Sonuç üretilmedi") in widget._result_context.text()


@pytest.mark.parametrize("previous", [False, True])
def test_plot_failure_never_pairs_new_visibility_csv_with_an_old_png(page, previous):
    widget, app, _ = page
    if previous:
        complete(widget)
    old_data, old_figure = widget._last_data, widget._fig
    widget.durationSpin.setValue(2); widget._compute()
    with patch.object(visibility, "_draw_gantt", side_effect=RuntimeError("controlled plot failure")):
        widget._worker.complete(payload(duration=2))
    assert widget._last_data is old_data and widget._fig is old_figure
    assert widget.exportCsvBtn.isEnabled() == previous
    assert widget.exportPngBtn.isEnabled() == previous
    assert "Plot error:" in widget.statusLabel.text()
    assert ("önceki" if previous else "Sonuç üretilmedi") in widget._result_context.text()


def test_visibility_empty_selection_scope_and_keyboard_parameter_labels(page):
    widget, app, _ = page
    assert "örnek başlangıç" in widget._scope_note.text()
    assert "10 dk" in widget._scope_note.text() and "Görev görünümü" in widget._scope_note.text()
    widget._deselect_all()
    assert not widget.computeBtn.isEnabled() and "istasyon seçin" in widget.statusLabel.text()
    widget._checkboxes["Madrid DSN"].setChecked(True)
    QTest.keyClick(widget, Qt.Key_E, Qt.AltModifier)
    assert widget.elevSpin.hasFocus() or widget.elevSpin.lineEdit().hasFocus()
    assert widget.elevLabel.buddy() is widget.elevSpin
    assert "ham" in widget.gapSpin.toolTip()
