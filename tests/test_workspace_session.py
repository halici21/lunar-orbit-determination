import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import json
from unittest.mock import patch
import pytest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QSettings
from app import MainWindow
from services.workspace_session import KEY, read_session
from tests.test_recorded_run_workspace import write_run
from tests.test_tracking_console import wait_until


@pytest.fixture
def environment(tmp_path):
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path/"settings.ini"), QSettings.IniFormat)
    windows = []
    def create():
        with patch("app.QSettings",return_value=settings):
            window = MainWindow()
        windows.append(window)
        return window
    yield app,settings,create
    for window in windows:
        window.close()
    app.processEvents()


def test_console_restart_restores_filter_selection_and_panels(environment,tmp_path):
    app,settings,create = environment
    path = write_run(tmp_path/"a")
    window = create(); page = window._pages["console"]
    page.open_result(path); wait_until(app,lambda:not page._busy)
    page.arc_combo.setCurrentIndex(page.arc_combo.findData(2)); page.apply_filters()
    page.select_index(3); page._toggle_scale()
    page.view_tabs.setCurrentIndex(1); page.details_toggle.setChecked(True)
    page.filters_toggle.setChecked(True)
    window.close()
    restored = create()._pages["console"]
    wait_until(app,lambda:not restored._busy)
    assert restored.arc_combo.currentData() == 2
    assert restored.visible_indices == (2,3)
    assert restored.selected_index == 3 and restored.charts[0].linear
    assert restored.view_tabs.currentIndex() == 1
    assert restored.details_toggle.isChecked() and restored.filters_toggle.isChecked()


def test_changed_source_does_not_reuse_old_filter(environment,tmp_path):
    app,settings,create = environment
    path = write_run(tmp_path/"a")
    window=create(); page=window._pages["console"]
    page.open_result(path); wait_until(app,lambda:not page._busy)
    page.arc_combo.setCurrentIndex(page.arc_combo.findData(2)); page.apply_filters()
    window.close()
    text = path.read_text()
    path.write_text(text.replace("3.0","30.0"))
    restored=create()._pages["console"]
    wait_until(app,lambda:not restored._busy)
    assert restored.arc_combo.currentData() is None
    assert "değişmiş" in restored.recovery_message.text()
    assert len(restored.visible_indices)==4


def test_missing_source_is_recoverable_on_restart(environment,tmp_path):
    app,settings,create = environment
    path=write_run(tmp_path/"a")
    window=create(); page=window._pages["console"]
    page.open_result(path); wait_until(app,lambda:not page._busy)
    window.close(); path.unlink()
    restored=create()._pages["console"]
    wait_until(app,lambda:not restored._busy)
    assert restored.result is None and not restored.retry_button.isHidden()
    assert "Sonuç açılamadı" in restored.recovery_message.text()


def test_comparison_restart_is_lazy_and_preserves_view(environment,tmp_path):
    app,settings,create = environment
    a=write_run(tmp_path/"a"); b=write_run(tmp_path/"b",seed=43)
    window=create(); window.navigate("comparison"); page=window._pages["comparison"]
    page.set_files(a,b); wait_until(app,lambda:not page._jobs)
    page.linear_check.setChecked(True); page.paired_check.setChecked(True)
    page.tabs.setCurrentIndex(3); page.pair_table.setCurrentIndex(page.pair_model.index(2,0))
    before=page.capture_view()
    window.close()
    restored_window=create()
    assert restored_window._active_page=="comparison"
    assert "console" not in restored_window._pages
    restored=restored_window._pages["comparison"]
    wait_until(app,lambda:not restored._jobs)
    assert restored.capture_view()==before


@pytest.mark.parametrize("raw",["broken","[]",'{"schema":2}','{"schema":1,"active":"run_monitor"}'])
def test_invalid_or_unsafe_active_page_never_starts_job(environment,raw):
    app,settings,create=environment
    settings.setValue(KEY,raw)
    window=create()
    assert window._active_page=="console"
    assert "run_monitor" not in window._pages



def test_factory_error_page_stays_retryable_with_saved_session(environment,tmp_path):
    import importlib
    from PyQt5.QtWidgets import QPushButton
    app,settings,create = environment
    source=write_run(tmp_path/"a")
    settings.setValue(KEY,json.dumps({"schema":1,"active":"console","console":{"path":str(source)}}))
    original=importlib.import_module
    def importing(name,*args,**kwargs):
        if name=="controllers.tracking_console_controller":
            raise ImportError("simulated unavailable page")
        return original(name,*args,**kwargs)
    with patch("app.importlib.import_module",side_effect=importing):
        window=create()
    error=window._pages["console"]
    assert not hasattr(error,"result")
    assert "console" not in window._restored_pages
    from services.workspace_session import save_session
    save_session(window)
    button=next(b for b in error.findChildren(QPushButton) if b.text()=="Yeniden dene")
    button.click()
    restored=window._pages["console"]
    wait_until(app,lambda:not restored._busy)
    assert restored.result is not None
