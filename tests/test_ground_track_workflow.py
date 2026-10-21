"""Ground-track request settings and displayed model must remain distinguishable."""
from unittest.mock import patch
import csv
import numpy as np
import pytest
from PyQt5.QtCore import QObject, QSettings, pyqtSignal
from PyQt5.QtWidgets import QApplication, QPushButton
from controllers import ground_track_controller as gt


class PendingWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, **parameters):
        super().__init__(); self.parameters = parameters; self.running = False

    def start(self): self.running = True
    def isRunning(self): return self.running
    def complete(self, data):
        self.running = False; self.finished.emit(data)
    def fail(self):
        self.running = False; self.failed.emit("Controlled worker failure")


@pytest.fixture
def page(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "ground-track.ini"), QSettings.IniFormat)
    monkeypatch.setattr("PyQt5.QtCore.QSettings", lambda *a, **kw: settings)
    monkeypatch.setattr(gt, "_TrackWorker", PendingWorker)
    widget = gt.GroundTrackController(); widget.resize(1250,850); widget.show(); app.processEvents()
    widget._deselect_all(); widget._checkboxes["Madrid DSN"].setChecked(True)
    widget.durationSpin.setValue(1); widget.elevSpin.setValue(10); widget.gapSpin.setValue(30)
    yield widget, app, settings
    widget.shutdown(); widget.close(); app.processEvents()


def payload(name="Madrid DSN", duration=1.):
    # Controlled UI samples, not a newly qualified propagation.
    return dict(lons=np.array([10.,11.,12.,13.]), lats=np.array([-30.,-10.,10.,30.]),
                t_s=np.array([0.,240.,480.,720.]), vis_mask=np.array([[True,False,True,False]]),
                station_names=[name], duration_days=duration, epoch_utc="2000-01-01T00:00:00")


def complete(widget, data=None):
    widget._compute(); widget._worker.complete(payload() if data is None else data)


def export_rows(widget, target):
    with patch.object(gt.QFileDialog,"getSaveFileName",return_value=(str(target),"CSV")):
        widget._export_csv()
    with target.open(encoding="utf-8",newline="") as stream: return list(csv.reader(stream))


def generate_button(widget):
    button = widget._orbit_designer.findChild(QPushButton, "generateAnalyticalTrackButton")
    assert button is widget._orbit_designer._generate_btn
    return button


def test_pending_ground_track_locks_request_controls_and_settings_resync(page):
    widget, app, settings = page
    button = generate_button(widget)
    widget._compute(); worker = widget._worker
    assert worker.parameters == dict(station_names=["Madrid DSN"],duration_days=1.,elev_deg=10.,max_gap_min=30.)
    for control in (widget.paramsGroup,widget.stationScrollArea,widget.selectAllBtn,
                    widget.deselectAllBtn,widget.computeBtn,widget.animGroup,
                    widget.exportPngBtn,widget.exportCsvBtn,button):
        assert not control.isEnabled()
    settings.setValue("dynamics/duration_days",3.); widget.on_shown()
    assert widget.durationSpin.value()==1.
    assert "Hesap sürüyor" in widget._result_context.text()
    assert "SPICE" in button.text() and "tamamlan" in button.toolTip()
    widget._compute(); assert widget._worker is worker
    worker.complete(payload())
    assert widget.paramsGroup.isEnabled() and widget.computeBtn.isEnabled() and button.isEnabled()
    assert button.text() == "Analitik yüzey izi üret"


def test_pending_spice_cannot_be_replaced_by_analytical_generation(page):
    widget, app, _ = page
    complete(widget); old_data, old_figure = widget._data,widget._fig
    widget._compute(); worker = widget._worker
    designer = widget._orbit_designer; designer._dur_spin.setValue(.1)
    with patch.object(gt,"_keplerian_ground_track",wraps=gt._keplerian_ground_track) as propagate:
        designer._generate_track()
    propagate.assert_not_called()
    widget._show_keplerian_track(payload(),dict(alt_km=100.,e=0.,inc_deg=90.))
    assert widget._data is old_data and widget._fig is old_figure
    assert "önceki" in widget._result_context.text()
    newer=payload(duration=2.); worker.complete(newer)
    assert widget._data is newer and "SPICE" in widget._model_note.text()
    designer._generate_track()
    assert widget._data is not newer and "Analitik" in widget._model_note.text()


@pytest.mark.parametrize("field",["durationSpin","elevSpin","gapSpin"])
def test_spice_input_edits_keep_captured_result_settings_and_csv(page,tmp_path,field):
    widget, app, _ = page
    complete(widget); old_data,old_figure=widget._data,widget._fig
    original=widget._result_context.text()
    assert "10°" in original and "30 dk" in original and "1 gün" in original
    before=export_rows(widget,tmp_path/"before.csv")
    control=getattr(widget,field); value=control.value(); control.setValue(value+1)
    assert "Ayarlar değişti" in widget._result_context.text() and original in widget._result_context.text()
    assert widget._data is old_data and widget._fig is old_figure
    assert "Madrid DSN" in widget._result_context.toolTip()
    assert export_rows(widget,tmp_path/"after.csv")==before
    control.setValue(value); assert widget._result_context.text()==original


def test_station_edit_and_rerun_replace_context_only_when_new_result_succeeds(page):
    widget, app, _ = page
    complete(widget)
    widget._checkboxes["Madrid DSN"].setChecked(False)
    widget._checkboxes["Goldstone DSN"].setChecked(True)
    assert "Ayarlar değişti" in widget._result_context.text()
    assert "Madrid DSN" in widget._result_context.toolTip()
    widget._compute(); assert "önceki" in widget._result_context.text()
    widget._worker.complete(payload("Goldstone DSN"))
    assert "Ayarlar değişti" not in widget._result_context.text()
    assert "Goldstone DSN" in widget._result_context.toolTip()
    assert "Madrid DSN" not in widget._result_context.toolTip()


@pytest.mark.parametrize("previous",[False,True])
@pytest.mark.parametrize("failure",["worker","plot"])
def test_pending_failure_releases_controls_and_preserves_known_metadata(page,previous,failure):
    widget, app, _ = page
    button=generate_button(widget)
    if previous: complete(widget)
    old_data,old_figure=widget._data,widget._fig
    widget.elevSpin.setValue(20); widget._compute()
    with patch.object(gt,"_show_error_dialog"):
        if failure=="worker": widget._worker.fail()
        else:
            with patch.object(gt,"_build_static_figure",side_effect=RuntimeError("Controlled plot failure")):
                widget._worker.complete(payload())
    assert widget._data is old_data and widget._fig is old_figure
    assert not widget._busy and widget._pending_inputs is None
    assert widget.paramsGroup.isEnabled() and button.isEnabled() and widget.computeBtn.isEnabled()
    assert widget.exportCsvBtn.isEnabled()==previous
    assert ("önceki" if previous else "Sonuç üretilmedi") in widget._result_context.text()
    if previous:
        assert "10°" in widget._result_context.text() and "20°" not in widget._result_context.text()
    complete(widget)
    assert "20°" in widget._result_context.text()


def test_analytical_result_has_its_own_captured_elements_and_ignores_spice_input_edits(page):
    widget, app, _ = page
    designer=widget._orbit_designer; designer._dur_spin.setValue(.1); designer._generate_track()
    context=widget._result_context.text(); details=widget._result_context.toolTip()
    assert "Analitik" in context and "0.1 gün" in context
    assert "a − R_Ay (km)" in details and "Eğim (°)" in details
    assert "Başlangıç gerçek anomalisi (°)" in details and "nu0_deg" not in details
    widget.elevSpin.setValue(20); widget.gapSpin.setValue(40)
    assert widget._result_context.text()==context
    assert widget._result_context.toolTip()==details


def test_result_without_request_metadata_does_not_invent_thresholds(page):
    widget, app, _ = page
    widget.elevSpin.setValue(25); widget._on_data(payload())
    assert "ayarları kaydedilmemiş" in widget._result_context.text()
    assert "25°" not in widget._result_context.text()


def test_empty_station_selection_disables_spice_compute_but_keeps_designer(page):
    widget, app, _ = page
    button=generate_button(widget); widget._deselect_all()
    assert not widget.computeBtn.isEnabled() and "istasyon seçin" in widget.statusLabel.text()
    assert button.isEnabled()
    widget._checkboxes["Madrid DSN"].setChecked(True)
    assert widget.computeBtn.isEnabled()
