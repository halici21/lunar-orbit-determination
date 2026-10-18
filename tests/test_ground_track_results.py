"""A failed ground-track render must never replace only the CSV side of a result."""
import csv
from unittest.mock import patch

import numpy as np
import pytest
from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from controllers import ground_track_controller as gt


@pytest.fixture
def page(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "ground-track.ini"), QSettings.IniFormat)
    monkeypatch.setattr("PyQt5.QtCore.QSettings", lambda *a, **kw: settings)
    widget = gt.GroundTrackController()
    widget.resize(1250, 850); widget.show(); app.processEvents()
    yield widget, app
    widget.shutdown(); widget.close(); app.processEvents()


def payload(new=False):
    # Controlled UI samples, not a qualified scientific result.
    return dict(lons=np.array([10., 11., 12., 13.]) + (50 if new else 0),
                lats=np.array([-30., -10., 10., 30.]), t_s=np.array([0., 240., 480., 720.]),
                vis_mask=np.asarray([[not new, False, True, new]]),
                station_names=["Goldstone DSN" if new else "Madrid DSN"],
                duration_days=2. if new else 1., epoch_utc="2000-01-01T00:00:00")


def deliver(widget, data, model):
    if model == "spice":
        widget._on_data(data)
    else:
        data["vis_mask"] = np.zeros((0, len(data["t_s"])), dtype=bool)
        data["station_names"] = []
        data["epoch_utc"] = "Keplerian (analytical)"
        widget._show_keplerian_track(data, dict(alt_km=100., e=0., inc_deg=90.))


def export_rows(widget, target):
    with patch.object(gt.QFileDialog, "getSaveFileName", return_value=(str(target), "CSV")):
        widget._export_csv()
    with target.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


@pytest.mark.parametrize("model", ["spice", "analytical"])
@pytest.mark.parametrize("previous", [False, True])
@pytest.mark.parametrize("stage", ["figure", "draw"])
def test_ground_track_render_failure_keeps_matching_data_canvas_and_exports(page, tmp_path, model, previous, stage):
    widget, app = page
    if previous:
        deliver(widget, payload(), "spice")
        widget._anim_step()
    old = (widget._data, widget._fig, widget._canvas, widget._anim_dot,
           widget._anim_tail, widget._anim_label, widget._anim_idx, widget._model_note.text())
    old_rows = export_rows(widget, tmp_path / "before.csv") if previous else None
    if previous:
        widget._play_pause()
        assert widget._anim_timer.isActive()
    target = (gt, "_build_static_figure") if stage == "figure" else (FigureCanvasQTAgg, "draw")
    with patch.object(*target, side_effect=RuntimeError("controlled render failure")), \
         patch.object(gt, "_show_error_dialog") as dialog:
        deliver(widget, payload(new=True), model)
    dialog.assert_called_once()
    assert (widget._data, widget._fig, widget._canvas, widget._anim_dot,
            widget._anim_tail, widget._anim_label, widget._anim_idx, widget._model_note.text()) == old
    assert not widget._anim_timer.isActive()
    assert widget.computeBtn.isEnabled() and not widget.progressBar.isVisible()
    assert widget.exportCsvBtn.isEnabled() == previous
    assert widget.exportPngBtn.isEnabled() == previous
    assert widget.animGroup.isEnabled() == previous
    assert ("önceki" if previous else "Sonuç üretilmedi") in widget.statusLabel.text()
    assert "controlled render failure" in dialog.call_args.args[2]
    if previous:
        assert export_rows(widget, tmp_path / "after.csv") == old_rows
        widget._canvas.draw()
        widget._play_pause(); assert widget._anim_timer.isActive()
        widget._pause_playback()


@pytest.mark.parametrize("model", ["spice", "analytical"])
def test_ground_track_success_replaces_complete_result_and_resets_playback(page, tmp_path, model):
    widget, app = page
    deliver(widget, payload(), "spice"); widget._anim_step()
    old_canvas, old_figure, old_dot = widget._canvas, widget._fig, widget._anim_dot
    newer = payload(new=True); deliver(widget, newer, model)
    assert widget._data is newer and widget._fig is not old_figure
    assert widget._canvas is not old_canvas and widget._canvas.figure is widget._fig
    assert widget._anim_dot is not old_dot and widget._anim_dot.axes is widget._fig.axes[0]
    assert widget._anim_idx == 0 and not widget._anim_timer.isActive()
    assert widget.exportCsvBtn.isEnabled() and widget.exportPngBtn.isEnabled()
    rows = export_rows(widget, tmp_path / "new.csv")
    assert float(rows[1][2]) == 60.
    assert ("SPICE" if model == "spice" else "Analitik") in widget._model_note.text()
    widget._anim_step()
    assert widget._anim_dot.get_xdata()[0] == newer["lons"][widget._anim_idx]


@pytest.mark.parametrize("previous", [False, True])
def test_ground_track_worker_failure_explains_retained_result_and_restores_playback(page, previous):
    widget, app = page
    if previous:
        deliver(widget, payload(), "spice"); widget._anim_step()
    old_data, old_figure, old_index = widget._data, widget._fig, widget._anim_idx
    with patch.object(gt, "_show_error_dialog"):
        widget._on_fail("controlled worker failure")
    assert widget._data is old_data and widget._fig is old_figure
    assert widget._anim_idx == old_index and not widget._anim_timer.isActive()
    assert widget.animGroup.isEnabled() == previous
    assert widget.exportCsvBtn.isEnabled() == previous and widget.exportPngBtn.isEnabled() == previous
    assert ("önceki" if previous else "Sonuç üretilmedi") in widget.statusLabel.text()
