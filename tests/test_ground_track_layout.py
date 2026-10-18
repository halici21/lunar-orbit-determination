"""Both ground-track tabs must fit a narrow viewport without losing controls/data."""
import pytest
from PyQt5.QtCore import QSettings, QPoint, QRect, Qt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from app import MainWindow
from controllers import ground_track_controller as gt
from tests.test_ground_track_workflow import PendingWorker, payload

@pytest.fixture
def window(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / 'ground-track-layout.ini'), QSettings.IniFormat)
    monkeypatch.setattr('app.QSettings', lambda *a, **kw: settings)
    monkeypatch.setattr('PyQt5.QtCore.QSettings', lambda *a, **kw: settings)
    monkeypatch.setattr(gt, '_TrackWorker', PendingWorker)
    view = MainWindow()
    view.environment.shutdown()
    view.resize(1480, 1000)
    view.show()
    view.navigate('ground_track')
    app.processEvents()
    page = view._pages['ground_track']
    page._deselect_all()
    page._checkboxes['Madrid DSN'].setChecked(True)
    page.durationSpin.setValue(1)
    page._compute()
    data = payload()
    page._worker.complete(data)
    page._anim_step()
    app.processEvents()
    yield (view, page, app)
    view.close()
    app.processEvents()

def settle(app):
    for _ in range(6):
        app.processEvents()

def reveal(view, control, app):
    view.pageScroll.ensureWidgetVisible(control, 0, 0)
    settle(app)
    viewport = view.pageScroll.viewport()
    rect = QRect(control.mapTo(viewport, QPoint(0, 0)), control.size())
    # Expose the full widget border after Qt scrolls its focus rectangle.
    for delta, bar in ((rect.bottom() - viewport.rect().bottom(), view.pageScroll.verticalScrollBar()), (rect.right() - viewport.rect().right(), view.pageScroll.horizontalScrollBar())):
        if delta > 0:
            bar.setValue(bar.value() + delta)
    if rect.top() < 0:
        bar = view.pageScroll.verticalScrollBar()
        bar.setValue(bar.value() + rect.top())
    settle(app)
    rect = QRect(control.mapTo(viewport, QPoint(0, 0)), control.size())
    assert viewport.rect().contains(rect), (control.objectName(), rect, viewport.rect())
    assert not control.visibleRegion().isEmpty()

@pytest.mark.parametrize('size', [(910, 465), (760, 420)])
@pytest.mark.parametrize('tab', [0, 1], ids=['track', 'designer'])
def test_ground_track_tabs_fit_small_window_keep_results_and_restore_wide(window, size, tab):
    view, page, app = window
    designer = page._orbit_designer
    old = (page._data, page._fig, page._canvas, page._anim_idx, page._result_context.text())
    preview = designer._fig
    page._tab_widget.setCurrentIndex(tab)
    settle(app)
    view.resize(*size)
    settle(app)
    assert view.pageScroll.horizontalScrollBar().maximum() == 0
    viewport = view.pageScroll.viewport()
    if tab == 0:
        assert page.controlPanel.geometry().bottom() < page.chartPanel.geometry().top()
        for control in (page.selectAllBtn, page.deselectAllBtn, page.durationSpin, page.elevSpin, page.gapSpin, page.computeBtn, page.playBtn, page.resetBtn, page.exportPngBtn, page.exportCsvBtn, page.statusLabel):
            reveal(view, control, app)
        page._log('Additional log shown during narrow layout review')
        settle(app)
        assert page.controlPanel.geometry().bottom() < page.chartPanel.geometry().top()
        assert page.controlPanel.rect().contains(page._log_edit.geometry())
        reveal(view, page._result_context, app)
        assert page._result_context.height() >= page._result_context.heightForWidth(page._result_context.width())
        canvas = page._canvas
        assert canvas.height() >= 360
        rect = QRect(canvas.mapTo(viewport, QPoint(0, 0)), canvas.size())
        assert rect.left() >= 0 and rect.right() < viewport.width()
        view.pageScroll.verticalScrollBar().setValue(view.pageScroll.verticalScrollBar().maximum())
        settle(app)
        assert canvas.mapTo(viewport, QPoint(0, canvas.height())).y() <= viewport.height()
    else:
        for control in (*designer._spins.values(), designer._dur_spin, designer._generate_btn):
            reveal(view, control, app)
        holder = designer._preview_holder
        assert designer._generate_btn.mapTo(designer, QPoint()).y() < holder.geometry().top()
        assert holder.height() >= 420
        assert designer._control_scroll.verticalScrollBar().maximum() == 0
        assert designer._fig is preview and designer._preview_timer.isActive()
        control = designer._spins['inc_deg']
        reveal(view, control, app)
        control.setFocus()
        control.selectAll()
        QTest.keyClicks(control, '45')
        QTest.keyClick(control, Qt.Key_Return)
        assert control.value() == 45.0
        assert page._data is old[0]
        view.pageScroll.verticalScrollBar().setValue(view.pageScroll.verticalScrollBar().maximum())
        settle(app)
        assert designer._canvas.mapTo(viewport, QPoint(0, designer._canvas.height())).y() <= viewport.height()
    assert (page._data, page._fig, page._canvas, page._anim_idx, page._result_context.text()) == old
    view.resize(1480, 1000)
    settle(app)
    assert view.pageScroll.horizontalScrollBar().maximum() == 0
    assert page.controlPanel.geometry().right() < page.chartPanel.geometry().left()
    assert designer._preview_holder.geometry().right() < designer._generate_btn.mapTo(designer, QPoint()).x()
    assert designer._fig is preview and page._data is old[0]
    if tab == 1:
        designer._dur_spin.setValue(0.1)
        QTest.mouseClick(designer._generate_btn, Qt.LeftButton)
        settle(app)
        assert page._tab_widget.currentIndex() == 0 and 'Analitik' in page._model_note.text()
        assert page._data is not old[0] and page._data['vis_mask'].shape[0] == 0
        assert '45' in page._result_context.toolTip()
        assert not designer._preview_timer.isActive()
    view.navigate('dynamics')
    settle(app)
    assert view.pageStack.minimumWidth() == 960
