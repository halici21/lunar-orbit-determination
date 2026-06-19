#!/usr/bin/env python3
"""Lunar OD PyQt5 Desktop App — entry point.

Run from the repository root or from python_port/:

    python python_port/desktop_app/main.py

Or add python_port to PYTHONPATH and run from anywhere.
"""
from __future__ import annotations
import sys
import os
from pathlib import Path

# ------------------------------------------------------------------
# Ensure python_port is on sys.path so both the backend (lunar_od)
# and the desktop_app package can be imported.
DESKTOP_DIR = Path(__file__).resolve().parent
PYTHON_PORT_DIR = DESKTOP_DIR.parent
for d in [str(PYTHON_PORT_DIR), str(DESKTOP_DIR)]:
    if d not in sys.path:
        sys.path.insert(0, d)
# ------------------------------------------------------------------

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QFont

from app import MainWindow


def _load_stylesheet(app: QApplication) -> None:
    qss_path = DESKTOP_DIR / "styles" / "dark_theme.qss"
    if qss_path.exists():
        with open(qss_path, "r", encoding="utf-8") as fh:
            app.setStyleSheet(fh.read())


def main() -> None:
    # High-DPI support
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Lunar OD")
    app.setOrganizationName("LunarOD")
    app.setApplicationVersion("1.0.0")

    # Global font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Dark stylesheet
    _load_stylesheet(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
