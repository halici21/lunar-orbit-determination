"""Görünürlük page: contact plan, lunar ground track and orbit designer as one page.

The contact plan (VisibilityController) and the ground-track tool
(GroundTrackController, which owns the track and designer tabs) stay separate
widgets; one tab bar drives both, and the shared request inputs (stations,
duration, elevation and gap thresholds) follow the user between tabs.
"""
from __future__ import annotations

from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTabBar, QStackedWidget

from controllers.visibility_controller import VisibilityController
from controllers.ground_track_controller import GroundTrackController

TABS = ("Temas planı", "Ay yüzey izi", "Yörünge tasarımı")


class _CurrentStack(QStackedWidget):
    """Size to the visible tool so a tall hidden tool never stretches the page."""

    def sizeHint(self):
        page = self.currentWidget()
        return page.sizeHint() if page is not None else QSize()

    def minimumSizeHint(self):
        page = self.currentWidget()
        return page.minimumSizeHint() if page is not None else QSize()


class VisibilityWorkspaceController(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.contact = VisibilityController(self)
        self.track = GroundTrackController(self)
        # The ground-track tool keeps its two internal tabs; the page tab bar selects them.
        self.track._tab_widget.tabBar().hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.tabs = QTabBar()
        self.tabs.setObjectName("pageTabs")
        self.tabs.setExpanding(False)
        for title in TABS:
            self.tabs.addTab(title)
        tab_strip = QWidget()
        tab_strip.setObjectName("pageTabStrip")  # full-width hairline under the tabs
        tab_row = QHBoxLayout(tab_strip)
        tab_row.setContentsMargins(24, 6, 24, 0)  # align with the page content edge
        tab_row.addWidget(self.tabs)
        tab_row.addStretch(1)
        layout.addWidget(tab_strip)
        self.stack = _CurrentStack()
        self.stack.addWidget(self.contact)
        self.stack.addWidget(self.track)
        layout.addWidget(self.stack, 1)
        self._shown_tool = self.contact
        self.tabs.currentChanged.connect(self._show_tab)
        self.track._tab_widget.currentChanged.connect(self._follow_track_tab)

    def show_tab(self, index: int) -> None:
        self.tabs.setCurrentIndex(index)

    def _show_tab(self, index: int) -> None:
        tool = self.contact if index == 0 else self.track
        if tool is not self._shown_tool:
            _copy_inputs(self._shown_tool, tool)
            self._shown_tool = tool
        self.stack.setCurrentWidget(tool)
        if index > 0 and self.track._tab_widget.currentIndex() != index - 1:
            self.track._tab_widget.setCurrentIndex(index - 1)
        self.stack.updateGeometry()

    def _follow_track_tab(self, index: int) -> None:
        # The designer switches to the track tab itself after generating a track.
        if self.stack.currentWidget() is self.track and self.tabs.currentIndex() != index + 1:
            self.tabs.setCurrentIndex(index + 1)

    def on_shown(self) -> None:
        tool = self.stack.currentWidget()
        if hasattr(tool, "on_shown"):
            tool.on_shown()

    def shutdown(self) -> None:
        for tool in (self.contact, self.track):
            if hasattr(tool, "shutdown"):
                tool.shutdown()


def _copy_inputs(source, target) -> None:
    """Carry the station choice and request thresholds to the other tool."""
    if getattr(target, "_busy", False) or getattr(source, "_busy", False):
        return
    for name in ("durationSpin", "elevSpin", "gapSpin"):
        src, dst = getattr(source, name, None), getattr(target, name, None)
        if src is not None and dst is not None and dst.value() != src.value():
            dst.setValue(src.value())
    for station, box in getattr(target, "_checkboxes", {}).items():
        chosen = getattr(source, "_checkboxes", {}).get(station)
        if chosen is not None and box.isChecked() != chosen.isChecked():
            box.setChecked(chosen.isChecked())
