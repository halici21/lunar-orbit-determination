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



def _load_stylesheet(app: QApplication) -> None:
    from styles.workspace_light import workspace_stylesheet
    app.setStyle("Fusion")
    app.setStyleSheet(workspace_stylesheet())


def _warm_plotting() -> None:
    try:
        import matplotlib.backends.backend_qt5agg  # noqa: F401
        import matplotlib.figure  # noqa: F401
    except Exception:
        pass  # pages report plotting problems themselves when opened


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

    from styles.appearance import load_saved_theme
    load_saved_theme()
    _load_stylesheet(app)
    from app import MainWindow

    window = MainWindow()
    window.show()
    # Import the plotting stack while the user reads the first page, so the
    # first map/plot page does not pay ~0.4 s for it on click.
    from PyQt5.QtCore import QTimer
    QTimer.singleShot(1500, _warm_plotting)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
