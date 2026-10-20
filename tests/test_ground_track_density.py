"""Narrow track controls should not spend scrolling space on an elastic gap."""
from unittest.mock import patch

import pytest
from controllers import ground_track_controller as gt
from tests.test_ground_track_layout import window, settle, reveal


@pytest.mark.parametrize("size", [(910, 465), (760, 420)])
@pytest.mark.parametrize("state", ["ready", "pending", "failed"])
def test_narrow_controls_stay_compact_without_clipping_results(window, size, state):
    view, page, app = window
    if state != "ready":
        page._compute()
        if state == "failed":
            with patch.object(gt, "_show_error_dialog"):
                page._worker.fail()
    original = (page._data, page._fig, page._canvas, page._anim_idx,
                page._input_snapshot(), page._result_context.text())
    view.resize(*size)
    settle(app)
    assert view.pageScroll.horizontalScrollBar().maximum() == 0
    gap = page._log_edit.y() - page.statusLabel.geometry().bottom() - 1
    assert gap <= 48, f"Unnecessary {gap}px status-to-log gap"
    controls = (page.durationSpin, page.elevSpin, page.gapSpin, page.computeBtn,
                page.playBtn, page.resetBtn, page.exportCsvBtn, page.exportPngBtn,
                page.statusLabel, page._log_edit)
    for control in controls:
        reveal(view, control, app)
    assert page.controlPanel.geometry().bottom() < page.chartPanel.geometry().top()
    assert page._canvas.height() >= 360
    for caption in (page._scope_note, page._model_note, page._result_context):
        assert caption.height() >= caption.heightForWidth(caption.width())
    # A wrapped request/error explanation must grow the form instead of clipping it.
    status = page.statusLabel.text()
    page._set_status("A long request explanation with named units and recovery steps. " * 5)
    settle(app)
    assert page.statusLabel.height() >= page.statusLabel.heightForWidth(page.statusLabel.width())
    assert page.controlPanel.rect().contains(page._log_edit.geometry())
    assert page._log_edit.y() - page.statusLabel.geometry().bottom() - 1 <= 48
    page._set_status(status)
    settle(app)
    heights = []
    for _ in range(3):
        view.resize(1480, 1000)
        settle(app)
        assert page.controlPanel.geometry().right() < page.chartPanel.geometry().left()
        assert page.controlPanel.height() == page.chartPanel.height()
        view.resize(*size)
        settle(app)
        heights.append(page.height())
        assert page._log_edit.y() - page.statusLabel.geometry().bottom() - 1 <= 48
    assert len(set(heights)) == 1
    assert (page._data, page._fig, page._canvas, page._anim_idx,
            page._input_snapshot(), page._result_context.text()) == original
    assert page.computeBtn.isEnabled() == (state != "pending")
    assert page.exportCsvBtn.isEnabled() == (state != "pending")
