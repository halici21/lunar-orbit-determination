"""Designer labels identify their inputs while Generate retains model/guard behavior."""
from unittest.mock import patch

import pytest
from PyQt5.QtCore import QPoint, QRect, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QLabel, QScrollArea

from controllers import ground_track_controller as gt
from tests.test_ground_track_layout import window, settle, reveal
from tests.test_ground_track_workflow import export_rows


@pytest.mark.parametrize("size", [(1480, 1000), (910, 465), (760, 420)])
def test_designer_tab_and_backtab_expose_complete_controls_without_replacing_result(window, tmp_path, size):
    view, page, app = window
    view.resize(*size)
    view.activateWindow()
    designer = page._orbit_designer
    page._tab_widget.setCurrentIndex(1)
    settle(app)
    original = (page._data, page._fig, page._canvas, page._anim_idx)
    before = export_rows(page, tmp_path / "before-keyboard.csv")
    ordered = [control for key in designer._spins
               for control in (designer._sliders[key], designer._spins[key])]
    ordered += [designer._dur_spin, designer._generate_btn]
    # Start before the first field; all subsequent movement uses actual Qt keys.
    designer._sliders["alt_km"].setFocus()
    settle(app)
    for expected, key in ([(control, Qt.Key_Tab) for control in ordered[1:]]
                          + [(control, Qt.Key_Backtab) for control in ordered[-2::-1]]):
        QTest.keyClick(app.focusWidget(), key)
        settle(app)
        assert expected.hasFocus(), (expected.accessibleName(), app.focusWidget())
        numeric = next((designer._spins[k] for k, slider in designer._sliders.items()
                        if slider is expected), expected)
        label = next((label for label in designer.findChildren(QLabel)
                      if label.buddy() is numeric), None)
        ancestor = expected.parentWidget()
        while ancestor is not None:
            if isinstance(ancestor, QScrollArea):
                viewport = ancestor.viewport()
                rect = QRect(expected.mapTo(viewport, QPoint()), expected.size())
                assert viewport.rect().contains(rect), (expected.accessibleName(), rect, viewport.rect())
                if label is not None:
                    label_rect = QRect(label.mapTo(viewport, QPoint()), label.size())
                    assert viewport.rect().contains(label_rect), (label.text(), label_rect, viewport.rect())
            ancestor = ancestor.parentWidget()
        assert not expected.visibleRegion().isEmpty()
    # Keyboard edits change only the preview until the explicit Generate action.
    QTest.keyClick(designer._sliders["alt_km"], Qt.Key_Right)
    assert designer._spins["alt_km"].value() > 200
    assert (page._data, page._fig, page._canvas, page._anim_idx) == original
    assert export_rows(page, tmp_path / "after-keyboard.csv") == before
    assert view.pageScroll.horizontalScrollBar().maximum() == 0


@pytest.mark.parametrize("size", [(1480, 1000), (760, 420)])
def test_designer_identifies_axis_offset_and_preserves_explicit_generation_and_guard(window, tmp_path, size):
    view, page, app = window
    view.resize(*size)
    designer = page._orbit_designer
    page._tab_widget.setCurrentIndex(1)
    settle(app)
    old = (page._data, page._fig, page._canvas, page._anim_idx)
    before = export_rows(page, tmp_path / "before.csv")
    for key, spin in designer._spins.items():
        label = next((label for label in designer.findChildren(QLabel) if label.buddy() is spin), None)
        assert label is not None, key
        assert spin.accessibleName() and designer._sliders[key].accessibleName()
        assert spin.toolTip() and designer._sliders[key].toolTip()
    axis = designer._spins["alt_km"]
    assert "a − R_Ay" in axis.toolTip() and "periapsis" in axis.toolTip()
    assert "a − R_Ay" in axis.accessibleName() and "km" in axis.accessibleName()
    assert designer._dur_spin.suffix() == " gün"

    # Precise numeric editing updates the preview, not the old exported result.
    eccentricity = designer._spins["e"]
    reveal(view, eccentricity, app)
    eccentricity.setFocus()
    eccentricity.selectAll()
    QTest.keyClicks(eccentricity, "0.05")
    QTest.keyClick(eccentricity, Qt.Key_Return)
    assert eccentricity.value() == .05
    designer._dur_spin.setValue(.1)
    assert (page._data, page._fig, page._canvas, page._anim_idx) == old
    assert export_rows(page, tmp_path / "still-old.csv") == before
    reveal(view, designer._generate_btn, app)
    with patch.object(gt, "_keplerian_ground_track", wraps=gt._keplerian_ground_track) as propagate:
        QTest.mouseClick(designer._generate_btn, Qt.LeftButton)
    settle(app)
    propagate.assert_called_once_with(duration_days=.1, alt_km=200., e=.05, inc_deg=90., raan_deg=0., argp_deg=0., nu0_deg=0.)
    assert page._data is not old[0] and page._result_model == "analytical"
    assert page._data["vis_mask"].shape[0] == 0
    assert "UTC epoku yok" in page._model_note.text() and "hesaplanmadı" in page._model_note.text()
    assert "a − R_Ay" in page._result_context.toolTip()
    generated = (page._data, page._fig, page._canvas)
    after = export_rows(page, tmp_path / "generated.csv")

    # The original near-surface rejection must still retain the displayed pair.
    page._tab_widget.setCurrentIndex(1)
    settle(app)
    eccentricity.setValue(.5)
    reveal(view, designer._generate_btn, app)
    with patch.object(gt.QMessageBox, "warning") as warning, patch.object(gt, "_keplerian_ground_track") as propagate:
        QTest.mouseClick(designer._generate_btn, Qt.LeftButton)
    warning.assert_called_once()
    assert "Periapsis Ay yüzeyinin altında" in warning.call_args.args[2]
    propagate.assert_not_called()
    assert (page._data, page._fig, page._canvas) == generated
    assert export_rows(page, tmp_path / "after-rejection.csv") == after
