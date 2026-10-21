"""Designer labels identify their inputs while Generate retains model/guard behavior."""
from unittest.mock import patch

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QLabel

from controllers import ground_track_controller as gt
from tests.test_ground_track_layout import window, settle, reveal
from tests.test_ground_track_workflow import export_rows


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
    assert "Periapsis is below" in warning.call_args.args[2]
    propagate.assert_not_called()
    assert (page._data, page._fig, page._canvas) == generated
    assert export_rows(page, tmp_path / "after-rejection.csv") == after
