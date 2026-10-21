import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
from unittest.mock import patch
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QSettings, QPoint, QRect
from controllers.tracking_console_controller import TrackingConsoleController
from controllers.recorded_comparison_controller import RecordedComparisonController
from services.run_details import load_run_with_details
from tests.test_tracking_console import make_run, wait_until
from tests.test_recorded_run_workspace import write_run


def test_failed_load_preserves_identity_and_retry_recovers(tmp_path):
    app = QApplication.instance() or QApplication([])
    page = TrackingConsoleController(); page.show()
    first = make_run(tmp_path/"first")
    missing = tmp_path/"restored"/"test_ukf_updates.csv"
    try:
        page.open_result(first); wait_until(app,lambda: not page._busy)
        original = page.result
        page.select_index(1)
        page.open_result(missing); wait_until(app,lambda: not page._busy)
        assert page.result is original and page.selected_index == 1
        assert Path(page.run_combo.currentData()) == first.resolve()
        assert page.retry_button.isVisible() and page.recovery_message.isVisible()
        make_run(missing.parent)
        page.retry_button.click(); wait_until(app,lambda: not page._busy)
        assert Path(page.result.sources[0]["path"]) == missing.resolve()
        assert not page.retry_button.isVisible() and not page.recovery_message.isVisible()
    finally:
        page.shutdown(); page.close()


def test_cancel_ignores_late_load_and_can_retry(tmp_path):
    app = QApplication.instance() or QApplication([])
    page = TrackingConsoleController(); page.show()
    source = make_run(tmp_path)
    try:
        page.open_result(source)
        page.cancel_load()
        wait_until(app,lambda: not page._workers)
        assert page.result is None
        assert "iptal" in page.recovery_message.text()
        assert "Önceki sonuç" not in page.recovery_message.text()
        page.retry_button.click(); wait_until(app,lambda: not page._busy)
        assert page.result is not None
    finally:
        page.shutdown(); page.close()


def test_partial_run_and_failed_comparison_have_next_steps(tmp_path):
    app = QApplication.instance() or QApplication([])
    a = write_run(tmp_path/"a",completed=False)
    b = write_run(tmp_path/"b")
    console = TrackingConsoleController()
    comparison = RecordedComparisonController()
    try:
        console._accept_result(*load_run_with_details(a))
        assert "yarım kalmış olabilir" in console.result_overview.text()
        assert "Çalıştır ekranını kontrol edin" in console.result_overview.text()
        comparison.set_files(a,tmp_path/"missing.csv")
        wait_until(app,lambda: not comparison._jobs)
        assert comparison.comparison is None
        assert "korunuyor" not in comparison.status_label.text()
        comparison.set_files(a,b)
        wait_until(app,lambda: not comparison._jobs)
        previous = comparison.comparison
        assert previous is not None
        comparison.set_files(a,tmp_path/"missing.csv")
        wait_until(app,lambda: not comparison._jobs)
        assert comparison.comparison is previous
        assert "Son karşılaştırma korunuyor" in comparison.status_label.text()
    finally:
        console.shutdown(); console.close()
        comparison.shutdown(); comparison.close()


def test_small_window_reaches_primary_actions_by_scroll(tmp_path):
    from app import MainWindow
    app = QApplication.instance() or QApplication([])
    with patch("app.QSettings",return_value=QSettings(str(tmp_path/"ui.ini"),QSettings.IniFormat)):
        window = MainWindow()
    try:
        window.resize(910,465); window.show(); app.processEvents()
        assert window.width() == 910 and window.height() == 465
        page = window._pages["console"]
        for button in (page.open_button,page.details_toggle):
            window.pageScroll.ensureWidgetVisible(button,0,0); app.processEvents()
            rect = QRect(button.mapTo(window.pageScroll.viewport(),QPoint(0,0)),button.size())
            assert window.pageScroll.viewport().rect().contains(rect)
    finally:
        window.close()
