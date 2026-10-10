"""Regression checks for optional panels and preserved scenario actions."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from dataclasses import asdict
from PyQt5.QtWidgets import QApplication
from controllers.scenario_builder_controller import ScenarioBuilderController
from controllers.tracking_console_controller import TrackingConsoleController
from services.archived_scenario import load_archived_scenario
from tests.test_run_workspace_resume import native_run
from tests.test_tracking_console import make_run, wait_until


def test_collapsed_scenario_preserves_actions_and_draft_recovery(tmp_path):
    app = QApplication.instance() or QApplication([])
    page = ScenarioBuilderController()
    page.resize(1100, 800)
    page.show()
    app.processEvents()
    try:
        before = asdict(page._widgets_to_model())
        assert not page.previewPanel.isVisible()
        assert not page.advanced_form.isVisible()
        assert all(w.isVisible() for w in (page.runScenarioBtn, page.exportJsonBtn,
            page.importJsonBtn, page.openAnalysisBtn))
        page.advanced_form_toggle.click()
        page.preview_toggle.click()
        assert page.measGroup.isVisible() and page.arcGroup.isVisible()
        assert page.jsonPreview.isVisible()
        assert asdict(page._widgets_to_model()) == before
        *_, source = native_run(tmp_path)
        page.preview_toggle.click()
        page.load_archived(load_archived_scenario(source))
        app.processEvents()
        assert page.origin_label.isVisible() and page.restore_draft_button.isVisible()
        page.restore_draft_button.click()
        assert asdict(page._widgets_to_model()) == before
    finally:
        page.close()


def test_result_tabs_and_optional_panels_keep_selection_and_filtered_rows(tmp_path):
    app = QApplication.instance() or QApplication([])
    page = TrackingConsoleController()
    page.resize(1200, 800)
    page.show()
    try:
        page.open_result(make_run(tmp_path))
        wait_until(app, lambda: page.result is not None)
        original = page.result
        page.select_index(1)
        page.details_toggle.click()
        page.filters_toggle.click()
        assert page.inspector_tabs.isVisible() and page.filter_panel.isVisible()
        page.view_tabs.setCurrentIndex(1)
        assert page.table.isVisible()
        assert page.selected_index == 1
        page.accepted_check.setChecked(True)
        page.apply_filters()
        rows = page.visible_indices
        page.filters_toggle.click()
        page.details_toggle.click()
        page.view_tabs.setCurrentIndex(0)
        assert not page.filter_panel.isVisible() and not page.inspector_tabs.isVisible()
        assert page.visible_indices == rows and page.result is original
        assert str(len(rows)) in page.filters_toggle.text()
    finally:
        page.shutdown()
        page.close()
