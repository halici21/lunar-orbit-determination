"""Launch the native result console without constructing scientific work pages."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, help="Result folder or *_ukf_updates.csv")
    args = parser.parse_args()
    desktop = Path(__file__).resolve().parent
    for path in (desktop, desktop.parent):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from PyQt5.QtCore import Qt, QCoreApplication, QTimer
    from PyQt5.QtGui import QFont
    from PyQt5.QtWidgets import QApplication
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    from styles.appearance import load_saved_theme
    load_saved_theme()
    from main import _load_stylesheet
    _load_stylesheet(app)
    from controllers.tracking_console_controller import TrackingConsoleController
    app.setApplicationName("Lunar OD Result Console")
    app.setFont(QFont("Segoe UI", 10))
    window = TrackingConsoleController()
    window.setWindowTitle("Lunar OD · Sonuç konsolu")
    window.resize(1500, 950)
    window.show()
    app.aboutToQuit.connect(window.shutdown)
    if args.path:
        QTimer.singleShot(0, lambda: window.open_folder(args.path) if args.path.is_dir() else window.open_result(args.path))
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
