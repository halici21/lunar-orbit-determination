"""Workflow navigation, verified inspector data, and cached plot parity."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QApplication, QPushButton
from app import MainWindow, _PRIMARY_NAV, _ADVANCED
from services.run_details import load_run_with_details, inspect_run
from services.tracking_results import ResultLoadError
from widgets.tracking_plots import TrackingChart
from tests.test_run_workspace_resume import native_run, rewrite_job
from tests.test_tracking_console import make_run, wait_until
from services.tracking_results import load_tracking_results


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, tmp_path):
    with patch("app.QSettings", return_value=QSettings(str(tmp_path / "isolated.ini"), QSettings.IniFormat)):
        widget = MainWindow()
    widget.resize(1400, 950)
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    app.processEvents()


def test_navigation_keeps_lazy_pages_and_selected_group(window, app):
    assert set(window._pages) == {"mission"}
    assert [v[0] for v in _PRIMARY_NAV] == ["mission", "scenario", "run_monitor", "console", "archive", "comparison", "stations", "visibility", "ground_track"]
    assert not window.advanced_panel.isVisible()
    window.advanced_toggle.click()
    assert window.advanced_panel.isVisible()
    assert set(window._pages) == {"mission"}  # expanding menu does not load tools
    window.navigate("settings")
    assert window._nav_btns["settings"].isChecked()
    assert not window._nav_btns["console"].isChecked()
    window.navigate("scenario")
    assert window._nav_btns["scenario"].isChecked()
    assert sum(b.isChecked() for b in window._nav_btns.values()) == 1
    window.advanced_toggle.setChecked(False)
    # Route a lightweight page through an advanced key, avoiding scientific modules.
    window._pages["dynamics"] = window._pages["scenario"]
    window.navigate("dynamics")
    assert window.advanced_panel.isVisible()
    assert window._nav_btns["dynamics"].isChecked()
    assert sum(b.isChecked() for b in window._nav_btns.values()) == 1


def test_error_page_retry_replaces_failed_widget(window, app):
    window._page_factories["retry_fixture"] = ("does_not_exist", "Missing")
    window.navigate("retry_fixture")
    failed = window._pages["retry_fixture"]
    window._page_factories["retry_fixture"] = ("controllers.tracking_console_controller", "TrackingConsoleController")
    next(b for b in failed.findChildren(QPushButton) if b.text() == "Yeniden dene").click()
    assert window._pages["retry_fixture"] is not failed
    assert hasattr(window._pages["retry_fixture"], "resultLoaded")


def test_details_show_saved_settings_lineage_and_completion(tmp_path):
    job, folder, run_id, path = native_run(tmp_path)
    rewrite_job(job, lambda j: j.update(derived_from={"schema": "lunar-od-derived-run-v1",
        "run_id": "parent-uuid", "form_changes": [{"field": "random_seed", "before": 42, "after": 43}]}))
    result, data = load_run_with_details(path)
    assert data["job"]["run_id"] == run_id
    assert data["job"]["derived_from"]["run_id"] == "parent-uuid"
    assert data["config"]["estimator_type"] == result.metadata["estimator_type"]
    assert data["status"] == "Tamamlanma kaydı yok"
    completed = folder / "completed.json"
    raw = {"schema": "lunar-od-completed-job-v1", "run_id": run_id, "result_path": str(path)}
    completed.write_text(json.dumps(raw))
    _, complete = load_run_with_details(path)
    assert complete["status"] == "İşlem tamamlandı"
    raw["run_id"] = "unrelated-run"
    completed.write_text(json.dumps(raw))
    _, wrong = load_run_with_details(path)
    assert wrong["status"] == "Tamamlanma kaydı geçersiz"


def test_detail_snapshot_rejects_changed_job_and_legacy_is_explicit(tmp_path):
    job, _, _, path = native_run(tmp_path)
    result = load_tracking_results(path)
    job.write_text(job.read_text() + " ")
    with pytest.raises(ResultLoadError, match="değişti"):
        inspect_run(result)
    legacy, details = load_run_with_details(make_run(tmp_path / "legacy"))
    assert details["status"] == "Eski / kayıtsız kaynak"
    assert details["config"] == {}
    assert legacy.metadata["frame"] == "Bilinmiyor"


def tree_text(tree):
    def rows(item):
        yield item.text(0) + " " + item.text(1)
        for index in range(item.childCount()):
            yield from rows(item.child(index))
    return "\n".join(rows(tree.invisibleRootItem()))


def test_console_details_async_and_failed_replacement_preserves_view(window, app, tmp_path):
    _, _, run_id, path = native_run(tmp_path)
    window.open_in_result_console(path)
    console = window._pages["console"]
    wait_until(app, lambda: console.result is not None)
    assert run_id in tree_text(console.run_details.tree)
    assert "Etkin ayarlar" in tree_text(console.run_details.tree)
    before = console.result
    summary = tree_text(console.run_details.tree)
    window.open_in_result_console(tmp_path / "missing.csv")
    wait_until(app, lambda: "açılamadı" in console.status_label.text())
    assert console.result is before
    assert console.path_label.text() == str(path)
    assert tree_text(console.run_details.tree) == summary


def pixels(widget):
    image = widget.grab().toImage()
    ptr = image.bits()
    ptr.setsize(image.byteCount())
    return bytes(ptr)


@pytest.mark.parametrize("linear", [False, True])
def test_cached_chart_matches_fresh_render_after_selection_filter_and_resize(app, tmp_path, linear):
    result = load_tracking_results(make_run(tmp_path))
    chart = TrackingChart("position_error_m", "Konum", "m")
    chart.resize(600, 220)
    chart.linear = linear
    all_rows = tuple(range(3))
    chart.set_data(result, all_rows, (0, 50), 0)
    pixels(chart)
    with patch.object(chart, "_draw_base", wraps=chart._draw_base) as draw:
        chart.set_data(result, all_rows, (0, 50), 1)
        cached = pixels(chart)
        assert draw.call_count == 0
        chart._base_key = None
        assert pixels(chart) == cached  # pixel equality including selected overlay
        assert draw.call_count == 1
        chart.set_data(result, (0, 2), (0, 45), 2)
        chart.resize(510, 260)
        changed = pixels(chart)
        assert draw.call_count == 2
        chart._base_key = None
        assert pixels(chart) == changed
    chart.close()
