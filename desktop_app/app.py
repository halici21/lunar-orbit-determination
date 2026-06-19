"""Main window — loads main_window.ui, wires sidebar nav and page stack."""
from __future__ import annotations
import sys
from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QFrame,
    QSizePolicy, QProgressBar,
)
from PyQt5.QtCore import Qt, QSettings
from PyQt5 import uic

from styles.theme import C, FONT

DESKTOP_DIR = Path(__file__).resolve().parent
UI_PATH = DESKTOP_DIR / "ui" / "main_window.ui"

# (page_id, sidebar_label, hint_text)
_NAV = [
    ("dashboard",   "Dashboard",        "Mission overview"),
    ("results",     "Results Browser",  "CSV tables & PNG plots"),
    ("run_monitor", "Run Monitor",       "Launch scripts live"),
    ("comparison",  "Comparison",        "Estimator comparison"),
    ("scenario",    "Scenario Builder",  "Configure runs"),
    ("dynamics",    "Dynamics",          "Force model & propagation"),
    ("measurements","Measurements",      "Noise, bias & Doppler"),
    ("estimators",  "Estimators",        "BLS / SRIF / UKF tuning"),
    ("stations",    "Stations",          "Network & precision"),
    ("visibility",  "Visibility Gantt",  "Station contact windows"),
    ("analysis",     "Analysis",          "BLS / UKF comparison"),
    ("ground_track", "Ground Track",      "Selenographic ground track"),
    ("settings",     "Settings",          "Paths, theme & runtime"),
]

_TITLES: dict[str, tuple[str, str]] = {
    "dashboard":   ("Mission Dashboard",   "Lunar OD research overview"),
    "results":     ("Results Browser",     "Browse CSV tables and PNG plots"),
    "run_monitor": ("Run Monitor",         "Launch example scripts with live output"),
    "comparison":  ("Comparison",          "Compare BLS-LM vs SR-UKF, start modes, physics"),
    "scenario":    ("Scenario Builder",    "Configure, export, and run scenarios"),
    "dynamics":    ("Dynamics",            "Force model, SPICE kernels, propagation grid, and numerical tolerances"),
    "measurements":("Measurements",        "Synthetic observables, noise, biases, and Doppler configuration"),
    "estimators":  ("Estimators",          "Centralized BLS-LM, SRIF, and SR-UKF tuning controls"),
    "stations":    ("Ground Stations",     "Station locations and measurement precision"),
    "visibility":  ("Visibility Gantt",    "Station contact windows — selectable stations, Gantt chart export"),
    "analysis":     ("Estimator Analysis",  "BLS-LM vs UKF comparison — observable type, start mode, α/β/κ sensitivity"),
    "ground_track": ("Ground Track",        "Selenographic ground track — station coverage, animation"),
    "settings":     ("Settings",            "Application paths, theme options, and reproducibility defaults"),
}


# ---------------------------------------------------------------------------
class _NavButton(QPushButton):
    """Sidebar navigation button — single line, compact height, tooltip for hint."""

    # Template for the button rule — placeholders filled per-check in _refresh_style.
    # Braces for CSS selectors are doubled so str.format() treats them as literals.
    _TPL = (
        "QPushButton {{"
        " background:transparent; border:none;"
        " border-left:2px solid {border};"
        " border-radius:0; padding:8px 14px 8px 16px;"
        " text-align:left; color:{fg}; font-size:" + str(FONT.SIZE_SM) + "px;"
        " font-weight:{fw}; }}"
    )
    # Hover rule is fully resolved — no .format() placeholders needed.
    _HOVER = (
        f"QPushButton:hover {{ background:{C.BG_ACTIVE};"
        f" border-left-color:#2a4a6a; color:{C.TEXT_SECONDARY}; }}"
    )

    def __init__(self, label: str, hint: str, parent=None) -> None:
        super().__init__(parent)
        self._label = label
        self.setCheckable(True)
        self.setFlat(True)
        self.setAutoExclusive(True)
        self.setMinimumHeight(38)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setToolTip(hint)
        self._refresh_style()

    def _refresh_style(self) -> None:
        checked = self.isChecked()
        sheet = self._TPL.format(
            border=C.NAV_ACTIVE if checked else "transparent",
            fg=C.NAV_ACTIVE_FG if checked else C.TEXT_MUTED,
            fw="600" if checked else "normal",
        ) + self._HOVER
        self.setStyleSheet(sheet)
        self.setText(self._label)

    def nextCheckState(self) -> None:
        super().nextCheckState()
        self._refresh_style()

    def setChecked(self, checked: bool) -> None:
        super().setChecked(checked)
        self._refresh_style()


# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        uic.loadUi(str(UI_PATH), self)

        self._settings = QSettings("LunarOD", "DesktopApp")
        self._nav_btns: dict[str, _NavButton] = {}
        self._pages: dict[str, QWidget] = {}

        self._apply_styles()
        self._build_nav()
        self._build_status_bar()
        self._load_pages()
        self._restore_geometry()
        self.navigate("dashboard")

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
    def _apply_styles(self) -> None:
        self.sidebarFrame.setObjectName("sidebarFrame")

        self.brandFrame.setStyleSheet(
            f"background-color:{C.BG_SIDEBAR}; border-bottom:1px solid {C.BORDER_MAIN};"
        )
        self.brandTitleLabel.setStyleSheet(
            f"color:{C.NAV_ACTIVE_FG}; font-size:20px; font-weight:bold; background:transparent;"
        )
        self.brandSubLabel.setStyleSheet(
            f"color:{C.TEXT_MUTED}; font-size:10px; background:transparent;"
        )

        self.navScrollArea.setStyleSheet(
            f"QScrollArea {{ background-color:{C.BG_SIDEBAR}; border:none; }}"
        )
        self.navWidget.setStyleSheet(f"background-color:{C.BG_SIDEBAR};")

        self.footerFrame.setStyleSheet(
            f"background-color:{C.BG_SIDEBAR}; border-top:1px solid {C.BORDER_MAIN};"
        )
        self.footerNativeLabel.setStyleSheet(
            f"color:{C.GREEN}; font-size:11px; font-weight:bold; background:transparent;"
        )
        self.footerVerLabel.setStyleSheet(
            f"color:{C.TEXT_MUTED}; font-size:10px; background:transparent;"
        )

        self.contentFrame.setObjectName("contentFrame")
        self.pageHeaderFrame.setObjectName("pageHeaderFrame")
        self.pageHeaderFrame.setStyleSheet(
            f"QFrame#pageHeaderFrame {{ background-color:{C.BG_SIDEBAR}; "
            f"border-bottom:1px solid {C.BORDER_MAIN}; }}"
        )
        self.pageTitleLabel.setStyleSheet(
            f"color:{C.TEXT_PRIMARY}; font-size:17px; font-weight:600; background:transparent;"
        )
        self.pageSubLabel.setStyleSheet(
            f"color:{C.TEXT_MUTED}; font-size:{FONT.SIZE_SM}px; background:transparent;"
        )

    def _build_nav(self) -> None:
        layout = self.navLayout
        # Group nav items with thin dividers
        groups = [
            [("dashboard",    "Dashboard",       "Mission overview"),
             ("results",      "Results Browser", "CSV tables & PNG plots")],
            [("run_monitor",  "Run Monitor",      "Launch scripts live"),
             ("scenario",     "Scenario Builder", "Configure runs"),
             ("comparison",   "Comparison",       "Estimator comparison")],
            [("dynamics",     "Dynamics",         "Force model & propagation"),
             ("measurements", "Measurements",     "Noise, bias & Doppler"),
             ("estimators",   "Estimators",       "BLS / SRIF / UKF tuning"),
             ("stations",     "Stations",         "Network & precision")],
            [("visibility",   "Visibility",       "Station contact windows"),
             ("analysis",     "Analysis",         "BLS / UKF comparison"),
             ("ground_track", "Ground Track",     "Selenographic ground track")],
            [("settings",     "Settings",         "Paths, theme & runtime")],
        ]
        first = True
        for group in groups:
            if not first:
                div = QFrame(self)
                div.setFrameShape(QFrame.HLine)
                div.setFixedHeight(1)
                div.setStyleSheet(f"background:{C.BORDER_MAIN}; border:none; margin:3px 12px;")
                layout.addWidget(div)
            first = False
            for page_id, label, hint in group:
                btn = _NavButton(label, hint)
                btn.clicked.connect(lambda _checked, pid=page_id: self.navigate(pid))
                layout.addWidget(btn)
                self._nav_btns[page_id] = btn
        layout.addStretch()

    def _build_status_bar(self) -> None:
        sb = self.statusBar
        sb.setFixedHeight(26)
        self._status_lbl = QLabel("Ready")
        self._status_lbl.setStyleSheet(f"color:{C.TEXT_MUTED}; padding:0 8px;")
        sb.addWidget(self._status_lbl)
        self._progress = QProgressBar()
        self._progress.setFixedSize(180, 12)
        self._progress.setTextVisible(False)
        self._progress.setVisible(False)
        sb.addPermanentWidget(self._progress)

    # ------------------------------------------------------------------
    # Page loading
    # ------------------------------------------------------------------
    def _load_pages(self) -> None:
        sys.path.insert(0, str(DESKTOP_DIR))
        from controllers.dashboard_controller import DashboardController
        from controllers.results_browser_controller import ResultsBrowserController
        from controllers.run_monitor_controller import RunMonitorController
        from controllers.comparison_controller import ComparisonController
        from controllers.scenario_builder_controller import ScenarioBuilderController
        from controllers.stations_controller import StationsController
        from controllers.visibility_controller import VisibilityController
        from controllers.analysis_controller import AnalysisController
        from controllers.ground_track_controller import GroundTrackController
        from controllers.system_pages_controller import (
            DynamicsController,
            MeasurementsController,
            EstimatorsController,
            SettingsController,
        )

        page_map: dict[str, type] = {
            "dashboard":    DashboardController,
            "results":      ResultsBrowserController,
            "run_monitor":  RunMonitorController,
            "comparison":   ComparisonController,
            "scenario":     ScenarioBuilderController,
            "dynamics":     DynamicsController,
            "measurements": MeasurementsController,
            "estimators":   EstimatorsController,
            "stations":     StationsController,
            "visibility":   VisibilityController,
            "analysis":     AnalysisController,
            "ground_track": GroundTrackController,
            "settings":     SettingsController,
        }
        for page_id, cls in page_map.items():
            try:
                widget = cls(self)
            except Exception as exc:
                widget = self._error_page(page_id, exc)
            self._pages[page_id] = widget
            self.pageStack.addWidget(widget)

    def _error_page(self, page_id: str, exc: Exception) -> QWidget:
        from PyQt5.QtWidgets import QVBoxLayout
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addStretch()
        msg = QLabel(f"Error loading '{page_id}':\n\n{exc}")
        msg.setAlignment(Qt.AlignCenter)
        msg.setStyleSheet(f"color:{C.RED}; font-size:13px;")
        msg.setWordWrap(True)
        lay.addWidget(msg)
        lay.addStretch()
        return w

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def navigate(self, page_id: str) -> None:
        if page_id not in self._pages:
            return
        page = self._pages[page_id]
        self.pageStack.setCurrentWidget(page)
        self._fade_in(page)

        title, sub = _TITLES.get(page_id, (page_id, ""))
        self.pageTitleLabel.setText(title)
        self.pageSubLabel.setText(sub)

        for pid, btn in self._nav_btns.items():
            btn.setChecked(pid == page_id)

        if hasattr(page, "on_shown"):
            page.on_shown()

    def _fade_in(self, page: QWidget) -> None:
        pass  # page switch via QStackedWidget is instant; no animation needed

    def navigate_and_run(
        self,
        script_path: str,
        extra_args: list[str] | None = None,
        auto_start: bool = False,
    ) -> None:
        self.navigate("run_monitor")
        page = self._pages.get("run_monitor")
        if page and hasattr(page, "preset_script"):
            page.preset_script(script_path, extra_args=extra_args, auto_start=auto_start)

    def open_in_results_browser(self, path: "Path") -> None:
        self.navigate("results")
        page = self._pages.get("results")
        if page and hasattr(page, "show_file"):
            page.show_file(path)

    # ------------------------------------------------------------------
    # Status bar helpers
    # ------------------------------------------------------------------
    def set_status(self, message: str, color: str = C.TEXT_MUTED) -> None:
        self._status_lbl.setText(message)
        self._status_lbl.setStyleSheet(f"color:{color}; padding:0 8px;")

    def show_progress(self, visible: bool = True) -> None:
        self._progress.setVisible(visible)
        if visible:
            self._progress.setRange(0, 0)
        else:
            self._progress.setRange(0, 1)
            self._progress.setValue(1)

    # ------------------------------------------------------------------
    def _restore_geometry(self) -> None:
        geo = self._settings.value("mainWindow/geometry")
        if geo:
            self.restoreGeometry(geo)

    def closeEvent(self, event) -> None:
        self._settings.setValue("mainWindow/geometry", self.saveGeometry())
        super().closeEvent(event)
