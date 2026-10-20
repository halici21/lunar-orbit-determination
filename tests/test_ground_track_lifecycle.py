"""Hidden ground-track views must not redraw or advance the user's playback."""
from unittest.mock import patch
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtTest import QSignalSpy, QTest
from app import MainWindow
from controllers import ground_track_controller as gt


@pytest.fixture
def window(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    settings=QSettings(str(tmp_path/'lifecycle.ini'),QSettings.IniFormat)
    monkeypatch.setattr('app.QSettings',lambda *a,**kw:settings)
    monkeypatch.setattr('PyQt5.QtCore.QSettings',lambda *a,**kw:settings)
    w=MainWindow();w.environment.shutdown();w.resize(1480,1000)
    w.show();w.navigate('ground_track');app.processEvents()
    yield w,w._pages['ground_track'],app
    page=w._pages['ground_track']
    page._anim_timer.stop();page._orbit_designer._preview_timer.stop()
    w.close();app.processEvents()


def test_initially_hidden_designer_has_no_timer_and_visible_preview_still_rotates(window):
    w,page,app=window; designer=page._orbit_designer
    spy=QSignalSpy(designer._preview_timer.timeout)
    phase=designer._preview_phase_deg
    QTest.qWait(220)
    assert not designer.isVisible() and not designer._preview_timer.isActive()
    assert len(spy)==0 and designer._preview_phase_deg==phase
    page._tab_widget.setCurrentIndex(1);app.processEvents()
    assert designer._preview_timer.isActive()
    QTest.qWait(220)
    assert len(spy)>0 and designer._preview_phase_deg!=phase
    page._tab_widget.setCurrentIndex(0);app.processEvents()
    count=len(spy);phase=designer._preview_phase_deg
    QTest.qWait(180)
    assert not designer._preview_timer.isActive()
    assert len(spy)==count and designer._preview_phase_deg==phase


@pytest.mark.parametrize('destination',['mission','designer'])
def test_leaving_playback_view_pauses_without_losing_result_or_export(window,tmp_path,destination):
    w,page,app=window
    page._orbit_designer._dur_spin.setValue(.1)
    page._orbit_designer._generate_track();app.processEvents()
    data,figure=page._data,page._fig
    arrays={k:v.tobytes() for k,v in data.items() if isinstance(v,np.ndarray)}
    target=tmp_path/'before.csv'
    with patch.object(gt.QFileDialog,'getSaveFileName',return_value=(str(target),'CSV')):page._export_csv()
    original=target.read_bytes()
    spy=QSignalSpy(page._anim_timer.timeout)
    QTest.mouseClick(page.playBtn,Qt.LeftButton);QTest.qWait(180)
    assert page._anim_timer.isActive() and len(spy)>0
    if destination=='mission':w.navigate('mission')
    else:page._tab_widget.setCurrentIndex(1)
    app.processEvents();index=page._anim_idx;count=len(spy)
    QTest.qWait(220)
    assert not page._anim_timer.isActive()
    assert page._anim_idx==index and len(spy)==count
    assert page.playBtn.text() == 'Oynat'
    page._play_pause();assert not page._anim_timer.isActive()
    if destination=='mission':w.navigate('ground_track')
    else:page._tab_widget.setCurrentIndex(0)
    app.processEvents()
    assert not page._anim_timer.isActive() and page._anim_idx==index
    assert page._data is data and page._fig is figure
    assert arrays=={k:data[k].tobytes() for k in arrays}
    with patch.object(gt.QFileDialog,'getSaveFileName',return_value=(str(target),'CSV')):page._export_csv()
    assert target.read_bytes()==original
    QTest.mouseClick(page.playBtn,Qt.LeftButton);QTest.qWait(180)
    assert page._anim_timer.isActive() and len(spy)>count
    page._reset_anim();assert page._anim_idx==0 and not page._anim_timer.isActive()


def test_window_close_stops_timers_on_retained_page_objects(window):
    w,page,app=window
    page._orbit_designer._dur_spin.setValue(.1)
    page._orbit_designer._generate_track();app.processEvents()
    QTest.mouseClick(page.playBtn,Qt.LeftButton)
    spy=QSignalSpy(page._anim_timer.timeout)
    assert page._anim_timer.isActive()
    w.close();app.processEvents();count=len(spy)
    QTest.qWait(180)
    assert not page._anim_timer.isActive() and not page._orbit_designer._preview_timer.isActive()
    assert len(spy)==count
