"""Dashboard page controller."""
from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QTreeWidgetItem, QPushButton, QLabel,
    QHBoxLayout, QVBoxLayout, QSizePolicy, QHeaderView,
)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5 import uic

from services.project_paths import DESKTOP_APP_DIR, RESULTS_DIR, EXAMPLES_DIR
from services.result_indexer import get_recent_files, count_results
from widgets.metric_card import MetricCard

UI_PATH = DESKTOP_APP_DIR / "ui" / "pages" / "dashboard_page.ui"

# Quick-action buttons: (label, tooltip, script_relative_to_examples, color)
_QUICK_ACTIONS = [
    ("Baseline BLS/UKF", "Run 1-day BLS vs SR-UKF comparison",
     "baseline_bls_ukf_comparison.py", "blue"),
    ("Sequential Tracking", "Run 3-day sequential tracking comparison",
     "sequential_tracking_comparison.py", "cyan"),
    ("28-day Visibility Gantt", "Generate 28-day DSN+ITU visibility chart",
     "visibility_28day_dsn_itu_gantt.py", "purple"),
    ("Two-Way Doppler", "Run two-way Doppler BLS/UKF comparison",
     "two_way_doppler_bls_ukf_comparison.py", "yellow"),
    ("Open Results Folder", "Open results directory in file explorer",
     None, "green"),
]


def _read_last_median_error(results_dir) -> tuple:
    """Scan the most recent CSV for a position error column and return median."""
    import re
    _ERR_COLS = ["final_position_error_m", "median_final_position_error_m",
                 "pos_error_m", "position_error_m", "p50_position_error_m"]
    try:
        csvs = sorted(results_dir.rglob("*.csv"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        for csv_path in csvs[:5]:
            try:
                import csv as _csv
                with open(csv_path, newline="", encoding="utf-8") as f:
                    reader = _csv.DictReader(f)
                    headers = [h.lower().strip() for h in (reader.fieldnames or [])]
                    col = next((c for c in _ERR_COLS if c in headers), None)
                    if col is None:
                        continue
                    vals = []
                    for row in reader:
                        try:
                            vals.append(float(row.get(col, "")))
                        except (ValueError, TypeError):
                            pass
                import numpy as _np
                if vals:
                    return float(_np.median(vals)), csv_path.name
            except Exception:
                continue
    except Exception:
        pass
    return None, ""


class DashboardController(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cards: list[MetricCard] = []
        self._zoom = 1.0
        self._current_pixmap: QPixmap | None = None
        uic.loadUi(str(UI_PATH), self)
        self._setup()

    def _setup(self) -> None:
        self.refreshBtn.clicked.connect(self.refresh)
        header = self.recentTree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.recentTree.itemDoubleClicked.connect(self._on_recent_double_click)
        self.plotPreview.setMaximumHeight(320)
        self._build_quick_actions()
        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(30_000)
        self._auto_timer.timeout.connect(self.refresh)
        self._auto_timer.start()
        self.refresh()

    def on_shown(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        self._refresh_cards()
        self._refresh_recent_tree()
        self._refresh_latest_plot()

    # ------------------------------------------------------------------
    def _refresh_cards(self) -> None:
        stats = count_results(RESULTS_DIR)
        recent = get_recent_files(RESULTS_DIR, limit=1)
        last_run = "—"
        if recent:
            mtime = datetime.fromtimestamp(recent[0].stat().st_mtime)
            last_run = mtime.strftime("%Y-%m-%d")

        median_err, err_source = _read_last_median_error(RESULTS_DIR)
        try:
            from lunar_od import ACCELERATION_BACKEND
            backend = ACCELERATION_BACKEND
        except Exception:
            backend = "unknown"

        cards_data = [
            ("Result Files", str(stats["total"]), f"{stats['csv']} CSV · {stats['png']} PNG",
             "blue", "Total CSV and PNG files in results/"),
            ("Folders", str(stats["folders"]), "result sub-directories",
             "cyan", "Sub-directories under results/"),
            ("Last Run", last_run, "most recent output file", "green",
             "Date of the most recently modified result file"),
            ("Examples", str(len(list(EXAMPLES_DIR.glob("*.py")))) if EXAMPLES_DIR.exists() else "?",
             "runnable scripts", "purple", "Python scripts under examples/"),
            ("Median Pos Error",
             f"{median_err:.0f} m" if median_err is not None else "—",
             err_source or "last run result",
             "green" if (median_err is not None and median_err < 500) else "yellow",
             "Median final position error from the most recent result CSV"),
            ("Compute Backend",
             backend.upper(),
             "JIT-compiled kernels" if backend == "numba" else "pure NumPy fallback",
             "cyan" if backend == "numba" else "yellow",
             "Whether Numba JIT acceleration is active for visibility/observable kernels"),
        ]

        if len(self._cards) == len(cards_data):
            # Update in place — no widget churn every 30 s
            for i, (_, value, subtitle, color, tip) in enumerate(cards_data):
                self._cards[i].set_value(value)
                self._cards[i].set_subtitle(subtitle)
                self._cards[i].set_color(color)
                self._cards[i].setToolTip(tip)
        else:
            layout = self.cardsFrame.layout()
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self._cards.clear()
            for title, value, subtitle, color, tip in cards_data:
                card = MetricCard(title, value, subtitle, color, tooltip=tip)
                layout.addWidget(card)
                self._cards.append(card)
            layout.addStretch()

    def _refresh_recent_tree(self) -> None:
        self.recentTree.clear()
        files = get_recent_files(RESULTS_DIR, limit=60)
        for p in files:
            try:
                size_kb = p.stat().st_size / 1024
                mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            except OSError:
                size_kb = 0
                mtime = ""
            item = QTreeWidgetItem([
                p.name,
                f"{size_kb:.1f} KB",
                mtime,
            ])
            item.setData(0, Qt.UserRole, str(p))
            # Colour-code CSV vs PNG
            if p.suffix.lower() == ".csv":
                item.setForeground(0, Qt.GlobalColor.cyan)
            else:
                item.setForeground(0, Qt.GlobalColor.yellow)
            self.recentTree.addTopLevelItem(item)

    def _refresh_latest_plot(self) -> None:
        # Find the most recent comparison PNG
        best: Path | None = None
        for pattern in ("*comparison*.png", "*baseline*.png", "*sequential*.png", "*.png"):
            candidates = sorted(RESULTS_DIR.rglob(pattern),
                                key=lambda p: p.stat().st_mtime if p.exists() else 0,
                                reverse=True)
            if candidates:
                best = candidates[0]
                break

        if best is None or not best.exists():
            self.plotPreview.setText("No plots found in results/")
            return

        pix = QPixmap(str(best))
        if pix.isNull():
            self.plotPreview.setText(f"Could not load:\n{best.name}")
            return
        self._current_pixmap = pix
        self._update_plot_label()
        self.plotPreview.setToolTip(str(best))

    def _update_plot_label(self) -> None:
        if self._current_pixmap is None:
            return
        size = self.plotPreview.size()
        if size.width() < 10 or size.height() < 10:
            return
        scaled = self._current_pixmap.scaled(
            size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.plotPreview.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_plot_label()

    # ------------------------------------------------------------------
    def _build_quick_actions(self) -> None:
        layout = self.actionsFrame.layout()
        for label, tip, script, color in _QUICK_ACTIONS:
            btn = QPushButton(label)
            btn.setToolTip(tip)
            btn.setObjectName("primaryButton" if script else "successButton")
            btn.setMinimumHeight(36)
            script_path = EXAMPLES_DIR / script if script else None
            btn.clicked.connect(
                lambda checked, sp=script_path, lbl=label: self._on_quick_action(sp, lbl)
            )
            layout.addWidget(btn)

    def _on_quick_action(self, script_path: Path | None, label: str) -> None:
        if script_path is None:
            # Open results folder
            import subprocess
            subprocess.Popen(f'explorer "{RESULTS_DIR}"', shell=True)
            return
        # Navigate to Run Monitor and pre-select this script
        main_win = self.window()
        if hasattr(main_win, "navigate_and_run"):
            main_win.navigate_and_run(str(script_path), auto_start=True)
        elif hasattr(main_win, "navigate"):
            main_win.navigate("run_monitor")

    def _on_recent_double_click(self, item: QTreeWidgetItem, col: int) -> None:
        path_str = item.data(0, Qt.UserRole)
        if not path_str:
            return
        path = Path(path_str)
        # Navigate to results browser and show this file
        main_win = self.window()
        if hasattr(main_win, "open_in_results_browser"):
            main_win.open_in_results_browser(path)
        elif hasattr(main_win, "navigate"):
            main_win.navigate("results")
