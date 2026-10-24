"""Settings page: environment readiness, result folder and colour palette.

Scenario parameters live on the Scenario page only; this page holds no model
settings, so nothing here can silently diverge from what a run uses.
"""
from __future__ import annotations

from PyQt5 import uic
from PyQt5.QtCore import QSettings, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QPushButton, QLineEdit, QScrollArea,
)

from services.project_paths import DESKTOP_APP_DIR, RESULTS_DIR
from styles.theme import C, FONT

_SETTINGS_ORG = "LunarOD"
_SETTINGS_APP = "DesktopApp"


def _make_scroll_page(ui_file: str, parent=None) -> tuple[QWidget, QVBoxLayout]:
    """Load .ui, wrap content in a scroll area, return (top_widget, inner_layout)."""
    w = QWidget(parent)
    uic.loadUi(str(DESKTOP_APP_DIR / "ui" / "pages" / ui_file), w)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.NoFrame)
    inner = QWidget()
    lay = QVBoxLayout(inner)
    lay.setContentsMargins(24, 18, 24, 24)
    lay.setSpacing(16)
    scroll.setWidget(inner)
    w.mainLayout.setContentsMargins(0, 0, 0, 0)
    w.mainLayout.addWidget(scroll)
    if parent is not None:
        fill_lay = QVBoxLayout(parent)
        fill_lay.setContentsMargins(0, 0, 0, 0)
        fill_lay.addWidget(w)
    return w, lay


def _muted(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(f"color:{C.TEXT_MUTED}; font-size:{FONT.SIZE_SM}px;")
    return lbl


class SettingsController(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._w, lay = _make_scroll_page("settings_page.ui", self)
        self.mainLayout = lay
        self._settings = getattr(parent, "_settings", None) or QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        self._build(lay)

    def on_shown(self) -> None:
        pass

    def _build(self, lay: QVBoxLayout) -> None:
        manager = getattr(self.window(), "environment", None)
        if manager is not None:
            from widgets.environment_panel import EnvironmentPanel
            readiness_box = QGroupBox("Hesaplama ortamı")
            readiness_layout = QVBoxLayout(readiness_box)
            self.environment_panel = EnvironmentPanel(manager)
            readiness_layout.addWidget(self.environment_panel)
            lay.addWidget(readiness_box)

        folder_box = QGroupBox("Sonuç klasörü")
        folder_form = QFormLayout(folder_box)
        self._results_edit = QLineEdit(str(RESULTS_DIR))
        self._results_edit.setReadOnly(True)
        open_results = QPushButton("Klasörü aç")
        open_results.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(RESULTS_DIR))))
        row = QHBoxLayout()
        row.addWidget(self._results_edit)
        row.addWidget(open_results)
        folder_form.addRow("Konum:", row)
        lay.addWidget(folder_box)

        from styles.appearance import THEMES, THEME_KEY
        appearance_box = QGroupBox("Görünüm")
        appearance_form = QFormLayout(appearance_box)
        self._theme_combo = QComboBox()
        for key, label in THEMES.items():
            self._theme_combo.addItem(label, key)
        stored = str(self._settings.value(THEME_KEY, "modern"))
        self._theme_combo.setCurrentIndex(max(0, self._theme_combo.findData(stored)))
        self._theme_combo.currentIndexChanged.connect(self._save_theme)
        appearance_form.addRow("Renk paleti:", self._theme_combo)
        self._saved = _muted("Uygulamayı yeniden açınca uygulanır.")
        appearance_form.addRow(self._saved)
        lay.addWidget(appearance_box)
        lay.addStretch()

    def _save_theme(self) -> None:
        from styles.appearance import THEME_KEY
        self._settings.setValue(THEME_KEY, self._theme_combo.currentData())
        self._saved.setText("Kaydedildi · uygulamayı yeniden açınca uygulanır.")
