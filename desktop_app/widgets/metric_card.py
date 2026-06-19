"""Rounded metric card widget — title / value / subtitle with a colour accent."""
from __future__ import annotations
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel, QSizePolicy
from PyQt5.QtCore import Qt

from styles.theme import ACCENT, C, FONT


class MetricCard(QFrame):
    """Dark card with a top accent stripe and three text rows."""

    def __init__(
        self,
        title: str,
        value: str,
        subtitle: str = "",
        color: str = "blue",
        tooltip: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._accent = ACCENT.get(color, color)
        self._build(title, value, subtitle)
        if tooltip:
            self.setToolTip(tooltip)

    # ------------------------------------------------------------------
    def _build(self, title: str, value: str, subtitle: str) -> None:
        self.setObjectName("metricCard")
        self.setMinimumSize(150, 104)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._apply_style()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(3)

        self._title_lbl = QLabel(title.upper())
        self._title_lbl.setStyleSheet(
            f"color:{C.TEXT_MUTED}; font-size:{FONT.SIZE_SM}px; font-weight:bold;"
            " letter-spacing:0.5px; background:transparent;"
        )
        lay.addWidget(self._title_lbl)

        self._value_lbl = QLabel(value)
        self._value_lbl.setStyleSheet(
            f"color:{self._accent}; font-size:{FONT.SIZE_H1}px; font-weight:bold;"
            " background:transparent;"
        )
        lay.addWidget(self._value_lbl)

        self._sub_lbl = QLabel(subtitle)
        self._sub_lbl.setStyleSheet(
            f"color:{C.TEXT_MUTED}; font-size:{FONT.SIZE_MD}px; background:transparent;"
        )
        self._sub_lbl.setWordWrap(True)
        lay.addWidget(self._sub_lbl)

        lay.addStretch()

    def _apply_style(self) -> None:
        self.setStyleSheet(f"""
            QFrame#metricCard {{
                background-color: {C.BG_CARD};
                border: 1px solid {C.BORDER_CARD};
                border-top: 3px solid {self._accent};
                border-radius: 8px;
            }}
        """)

    # ------------------------------------------------------------------
    def set_value(self, value: str) -> None:
        self._value_lbl.setText(value)

    def set_subtitle(self, subtitle: str) -> None:
        self._sub_lbl.setText(subtitle)

    def set_color(self, color: str) -> None:
        self._accent = ACCENT.get(color, color)
        self._apply_style()
        self._value_lbl.setStyleSheet(
            f"color:{self._accent}; font-size:{FONT.SIZE_H1}px; font-weight:bold;"
            " background:transparent;"
        )
