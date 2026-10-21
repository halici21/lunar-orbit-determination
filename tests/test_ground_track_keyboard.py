"""Native label shortcuts reach the intended field without changing saved results."""
from unittest.mock import patch

import pytest
from PyQt5.QtCore import QPoint, QRect, Qt
from PyQt5.QtTest import QTest

from controllers import ground_track_controller as gt
from tests.test_ground_track_layout import window, settle
from tests.test_ground_track_workflow import export_rows


@pytest.mark.parametrize("size", [(1480, 1000), (760, 420)])
@pytest.mark.parametrize("field,key,name", [
    ("durationSpin", Qt.Key_S, "Süre (gün)"),
    ("elevSpin", Qt.Key_Y, "Yükseliş eşiği (°)"),
    ("gapSpin", Qt.Key_B, "Boşluk eşiği (dk)"),
    ("speedSpin", Qt.Key_H, "Hız ayarı"),
])
def test_label_shortcut_reveals_field_respects_busy_and_hidden_page(window, tmp_path, size, field, key, name):
    view, page, app = window
    view.resize(*size)
    view.activateWindow()
    settle(app)
    control = getattr(page, field)
    original = (page._data, page._fig, page._canvas, page._anim_idx)
    before = export_rows(page, tmp_path / "before.csv")
    view.pageScroll.verticalScrollBar().setValue(0)
    page.selectAllBtn.setFocus()
    QTest.keyClick(view, key, Qt.AltModifier)
    settle(app)
    assert control.hasFocus(), (field, app.focusWidget())
    assert control.accessibleName() == name
    viewport = view.pageScroll.viewport()
    rect = QRect(control.mapTo(viewport, QPoint()), control.size())
    assert viewport.rect().contains(rect), (field, rect, viewport.rect())
    value = control.value()
    QTest.keyClick(control, Qt.Key_Up)
    assert control.value() == value + control.singleStep()
    assert (page._data, page._fig, page._canvas, page._anim_idx) == original
    assert export_rows(page, tmp_path / "after.csv") == before
    if field == "durationSpin":
        QTest.keyClick(control, Qt.Key_Tab)
        settle(app)
        assert page.elevSpin.hasFocus()
        assert viewport.rect().contains(QRect(page.elevSpin.mapTo(viewport, QPoint()), page.elevSpin.size()))
        QTest.keyClick(page.elevSpin, Qt.Key_Backtab)
        settle(app)
        assert control.hasFocus()
        assert viewport.rect().contains(QRect(control.mapTo(viewport, QPoint()), control.size()))

    # Native label shortcuts must honor disabled controls during a SPICE request.
    page._compute()
    assert page._busy and not control.isEnabled()
    values = [c.value() for c in (page.durationSpin, page.elevSpin, page.gapSpin, page.speedSpin)]
    QTest.keyClick(view, key, Qt.AltModifier)
    QTest.keyClick(view, Qt.Key_Up)
    settle(app)
    assert not control.hasFocus()
    assert values == [c.value() for c in (page.durationSpin, page.elevSpin, page.gapSpin, page.speedSpin)]
    with patch.object(gt, "_show_error_dialog"):
        page._worker.fail()
    assert control.isEnabled() and page._data is original[0]

    # A hidden tool must not intercept a key from the current Mission page.
    view.navigate("mission")
    view._nav_btns["mission"].setFocus()
    focus = app.focusWidget()
    QTest.keyClick(view, key, Qt.AltModifier)
    settle(app)
    assert app.focusWidget() is focus and not control.hasFocus()
    assert values == [c.value() for c in (page.durationSpin, page.elevSpin, page.gapSpin, page.speedSpin)]
