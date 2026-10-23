"""Main window — loads main_window.ui, wires sidebar nav and page stack."""
from __future__ import annotations
import sys
import importlib
from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QFrame,
    QSizePolicy, QProgressBar, QVBoxLayout, QButtonGroup, QShortcut, QScrollArea, QStackedWidget,
)
from PyQt5.QtCore import Qt, QSettings, QSize, QTimer
from PyQt5 import uic
from PyQt5.QtGui import QKeySequence

from styles.theme import C, FONT

DESKTOP_DIR = Path(__file__).resolve().parent
UI_PATH = DESKTOP_DIR / "ui" / "main_window.ui"

# (page_id, sidebar_label, hint_text) — the workflow order, one page per task.
_PRIMARY_NAV = [
    ("scenario", "Senaryo", "Koşu ayarlarını hazırla ve başlat"),
    ("run_monitor", "Çalıştır", "Çalışan koşuyu ve günlüğünü izle"),
    ("console", "Sonuçlar", "Açık koşunun yörüngesi, grafikleri ve kayıtları"),
    ("archive", "Arşiv", "Kaydedilmiş koşuları bul, aç ve karşılaştır"),
    ("stations", "İstasyonlar", "Yer istasyonu kataloğu ve ağlar"),
    ("visibility", "Görünürlük", "Temas planı, Ay yüzey izi ve yörünge tasarımı"),
]
_SETTINGS_NAV = ("settings", "Ayarlar", "Python ortamı, SPICE dosyaları ve görünüm")

_NAV = _PRIMARY_NAV + [_SETTINGS_NAV]

# Pages without a sidebar entry live inside a sidebar page; they highlight it.
_NAV_PARENT = {"comparison": "archive"}

_TITLES: dict[str, tuple[str, str]] = {
    "scenario": ("Senaryo", "Ölçüm türünü, istasyon ağını, süreyi ve kestiriciyi seçip koşuyu başlatın."),
    "run_monitor": ("Çalıştır", "Çalışan koşunun durumu ve günlüğü."),
    "console": ("Sonuçlar", "Açık koşunun yörüngesi, hata grafikleri ve kayıtları."),

    "archive": ("Arşiv", "Kaydedilmiş koşular. İki koşu seçerek karşılaştırabilirsiniz."),
    "comparison": ("Arşiv · Karşılaştırma", "İki koşunun ortak epoklardaki farkları ve ayar farkları."),
    "stations": ("İstasyonlar", "Yer istasyonu kataloğu. Kaydettiğiniz ağlar Senaryo sayfasında seçilebilir."),
    "visibility": ("Görünürlük", "Örnek yörünge için istasyon temas planı, Ay yüzey izi ve yörünge tasarımı."),

    "settings": ("Ayarlar", "Python ortamı, SPICE dosyaları, sonuç klasörü ve renk paleti."),
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
        f"QPushButton:hover {{ background:{C.BG_HOVER}; border-left-color:{C.BORDER_MID}; color:{C.TEXT_PRIMARY}; }}"
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
        self.toggled.connect(self._refresh_style)
        self._refresh_style()

    def _refresh_style(self) -> None:
        checked = self.isChecked()
        sheet = self._TPL.format(
            border=C.NAV_ACTIVE if checked else "transparent",
            fg=C.TEXT_PRIMARY if checked else C.TEXT_MUTED,
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
class _VisiblePageStack(QStackedWidget):
    """A tall hidden page must not stretch every other page in the viewport."""
    def sizeHint(self):
        page = self.currentWidget()
        return page.sizeHint() if page else QSize(960, 720)

    def minimumSizeHint(self):
        page = self.currentWidget()
        return page.minimumSizeHint().expandedTo(QSize(0, 720)) if page else QSize(960, 720)

    def heightForWidth(self, width):
        page = self.currentWidget()
        if page and page.hasHeightForWidth():
            return page.heightForWidth(width)
        return self.minimumSizeHint().height()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        uic.loadUi(str(UI_PATH), self)
        # Keep controls reachable on small/high-DPI desktops without compressing
        # scientific plots or removing advanced controls.
        self.setMinimumSize(760, 420)
        self.pageScroll = QScrollArea()
        self.pageScroll.setWidgetResizable(True)
        self.pageScroll.setFrameShape(QFrame.NoFrame)
        self.pageScroll.setAccessibleName("Çalışma alanı; taşan içerik kaydırılabilir")
        original_stack = self.pageStack
        parent_layout = original_stack.parentWidget().layout()
        parent_layout.replaceWidget(original_stack, self.pageScroll)
        self.pageStack = _VisiblePageStack()
        self.pageStack.setObjectName("pageStack")
        original_stack.deleteLater()
        self.pageStack.setMinimumWidth(960)
        self.pageScroll.setWidget(self.pageStack)

        self._settings = QSettings("LunarOD", "DesktopApp")
        from controllers.environment_controller import EnvironmentController
        self.environment = EnvironmentController(self._settings, self)
        from services.workspace_session import read_session
        self._workspace_session = read_session(self._settings)
        self._restored_pages = set()
        self._active_page = "scenario"
        self._latest_result = None
        self._nav_btns: dict[str, _NavButton] = {}
        self._pages: dict[str, QWidget] = {}

        self.brandSubLabel.setText("Yörünge belirleme çalışma alanı")
        self.footerNativeLabel.setText("Yerel çalışma alanı")
        self._apply_styles()
        self._build_nav()
        self._build_status_bar()
        self._load_pages()
        tutorial_menu = self.menuBar().addMenu("Örnek")
        tutorial_action = tutorial_menu.addAction("Verisiz sentetik eğitimi başlat")
        tutorial_action.triggered.connect(self.start_tutorial)
        self._restore_geometry()
        active = self._workspace_session.get("active", "scenario")
        self.navigate(active if active in ("console", "comparison", "archive") else "scenario")
        self._startup_timer = QTimer(self)
        self._startup_timer.setSingleShot(True)
        self._startup_timer.timeout.connect(self.environment.refresh)
        self._startup_timer.start(0)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
    def _apply_styles(self) -> None:
        from styles.workspace_light import workspace_stylesheet
        self.setStyleSheet(workspace_stylesheet())
        self.setWindowTitle("Lunar OD · Çalışma alanı")
        self.sidebarFrame.setFixedWidth(192)
        self.sidebarFrame.setStyleSheet(
            f"QFrame#sidebarFrame {{background:{C.BG_SIDEBAR}; border-right:1px solid {C.BORDER_MAIN};}}")
        for widget in (self.brandFrame, self.navWidget, self.footerFrame):
            widget.setStyleSheet(f"background:{C.BG_SIDEBAR};")
        self.navScrollArea.setStyleSheet(f"QScrollArea {{background:{C.BG_SIDEBAR}; border:0;}}")
        self.brandTitleLabel.setStyleSheet(f"color:{C.TEXT_PRIMARY}; font-size:20px; font-weight:500;")
        self.brandSubLabel.setText("Yörünge belirleme")
        self.brandSubLabel.setStyleSheet(f"color:{C.TEXT_MUTED}; font-size:11px;")
        self.footerNativeLabel.setStyleSheet(f"color:{C.TEXT_MUTED}; font-size:11px;")
        self.footerVerLabel.setText("Araştırma çalışma alanı")
        self.footerVerLabel.setStyleSheet(f"color:{C.TEXT_MUTED}; font-size:11px;")
        self.pageHeaderFrame.setStyleSheet(
            f"QFrame#pageHeaderFrame {{background:{C.BG_DEEP}; border-bottom:1px solid {C.BORDER_MAIN};}}")
        self.pageTitleLabel.setStyleSheet(f"color:{C.TEXT_PRIMARY}; font-size:19px; font-weight:500;")
        self.pageHeaderFrame.setMaximumHeight(16777215)
        self.pageHeaderFrame.setMinimumHeight(72)
        self.pageHeaderFrame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.pageSubLabel.setTextFormat(Qt.PlainText)
        self.pageSubLabel.setWordWrap(True)
        self.pageSubLabel.setStyleSheet(f"color:{C.TEXT_MUTED}; font-size:12px;")
        self.pageSubLabel.show()

    def _build_nav(self) -> None:
        layout = self.navLayout
        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        def add_button(item, target):
            page_id, label, hint = item
            btn = _NavButton(label, hint)
            btn.setAutoExclusive(False)
            self._nav_group.addButton(btn)
            btn.clicked.connect(lambda _checked, pid=page_id: self.navigate(pid))
            target.addWidget(btn)
            self._nav_btns[page_id] = btn
        label = QLabel("ÇALIŞMA ALANI")
        label.setStyleSheet(f"color:{C.TEXT_MUTED}; padding:14px 16px 6px; font-size:11px;")
        layout.addWidget(label)
        for number, item in enumerate(_PRIMARY_NAV, 1):
            add_button(item, layout)
            shortcut = QShortcut(QKeySequence(f"Alt+{number}"), self)
            shortcut.activated.connect(lambda pid=item[0]: self.navigate(pid))
        layout.addStretch()
        add_button(_SETTINGS_NAV, layout)

    def _build_status_bar(self) -> None:
        sb = self.statusBar
        sb.setFixedHeight(26)
        self._status_lbl = QLabel("Hazır")
        self._status_lbl.setStyleSheet(f"color:{C.TEXT_MUTED}; padding:0 8px;")
        sb.addWidget(self._status_lbl)
        self.environment_button = QPushButton("Ortam · kontrol bekleniyor")
        self.environment_button.setFlat(True)
        self.environment_button.setFixedHeight(22)
        self.environment_button.setStyleSheet("QPushButton {padding:0 8px; min-height:18px; font-size:11px; border:none;}")
        self.environment_button.setToolTip("Ortam ve SPICE kernel raporunu aç")
        self.environment_button.clicked.connect(lambda: self.navigate("settings"))
        sb.addPermanentWidget(self.environment_button)
        self.environment.changed.connect(self._update_environment_status)
        self._progress = QProgressBar()
        self._progress.setFixedSize(180, 12)
        self._progress.setTextVisible(False)
        self._progress.setVisible(False)
        sb.addPermanentWidget(self._progress)

    def _update_environment_status(self):
        env = self.environment
        if env.busy:
            text = "Ortam · kontrol ediliyor"
        elif env.error:
            text = "Ortam · kontrol hatası"
        elif env.report:
            warnings = any(c["status"] == "warning" for c in env.report["checks"])
            text = ("Ortam · profil farkı" if warnings else "Ortam · kontrol geçti") if env.report["ready"] else "Ortam · eksikler var"
        else:
            text = "Ortam · kontrol bekleniyor"
        self.environment_button.setText(text)

    # ------------------------------------------------------------------
    # Page loading
    # ------------------------------------------------------------------
    def _load_pages(self) -> None:
        """Register pages; expensive modules are imported on first navigation."""
        sys.path.insert(0, str(DESKTOP_DIR))
        self._page_factories = {
            "console": ("controllers.tracking_console_controller", "TrackingConsoleController"),
            "archive": ("controllers.run_archive_controller", "RunArchiveController"),
            "run_monitor": ("controllers.run_monitor_controller", "RunMonitorController"),
            "comparison": ("controllers.recorded_comparison_controller", "RecordedComparisonController"),
            "scenario": ("controllers.scenario_builder_controller", "ScenarioBuilderController"),
            "stations": ("controllers.stations_controller", "StationsController"),
            "visibility": ("controllers.visibility_workspace_controller", "VisibilityWorkspaceController"),
            "settings": ("controllers.system_pages_controller", "SettingsController"),
        }

    def _ensure_page(self, page_id: str) -> QWidget | None:
        if page_id in self._pages:
            return self._pages[page_id]
        factory = self._page_factories.get(page_id)
        if factory is None:
            return None
        try:
            module_name, class_name = factory
            cls = getattr(importlib.import_module(module_name), class_name)
            widget = cls(self)
        except Exception as exc:
            widget = self._error_page(page_id, exc)
        if page_id == "console" and hasattr(widget, "resultLoaded"):
            widget.resultLoaded.connect(self._on_result_loaded)
            widget.loadFailed.connect(self._on_overview_load_failed)
            try:
                from controllers.mission_controller import MissionController
                widget.set_overview(MissionController(widget))
            except Exception as exc:
                widget.set_overview(self._error_page("console", exc))
        widget._workspace_size_policy = widget.sizePolicy()
        self._pages[page_id] = widget
        self.pageStack.addWidget(widget)
        return widget

    def _error_page(self, page_id: str, exc: Exception) -> QWidget:
        from PyQt5.QtWidgets import QVBoxLayout
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addStretch()
        msg = QLabel(f"Sayfa açılamadı: {page_id}\n\n{exc}\n\nYeniden deneyin veya başka bir sayfaya geçin.")
        msg.setTextFormat(Qt.PlainText)
        msg.setAlignment(Qt.AlignCenter)
        msg.setStyleSheet(f"color:{C.RED}; font-size:13px;")
        msg.setWordWrap(True)
        lay.addWidget(msg)
        retry = QPushButton("Yeniden dene")
        def retry_page():
            if self._pages.get(page_id) is w:
                self._pages.pop(page_id)
                self.pageStack.removeWidget(w)
                w.deleteLater()
            self.navigate(page_id)
        retry.clicked.connect(retry_page)
        lay.addWidget(retry, alignment=Qt.AlignCenter)
        lay.addStretch()
        return w

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def navigate(self, page_id: str) -> None:
        if page_id == "mission":  # The overview is the first tab of Sonuçlar.
            self.navigate("console")
            console = self._pages.get("console")
            if hasattr(console, "show_overview"):
                console.show_overview()
            return
        if page_id == "ground_track":  # Ay yüzey izi is the second tab of Görünürlük.
            self.navigate("visibility")
            workspace = self._pages.get("visibility")
            if hasattr(workspace, "show_tab"):
                workspace.show_tab(1)
            return
        page = self._ensure_page(page_id)
        if page is None:
            return
        self._active_page = page_id
        for other in self._pages.values():
            other.setSizePolicy(other._workspace_size_policy if other is page else
                                QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored))
        self.pageStack.setCurrentWidget(page)
        # Mission plots, catalogue and visibility panels can stack; other forms retain
        # their existing scrollable minimum width.
        self.pageStack.setMinimumWidth(540 if page_id in {"console", "stations", "visibility"} else 960)
        self.pageStack.updateGeometry()
        self.pageScroll.verticalScrollBar().setValue(0)
        self.pageScroll.horizontalScrollBar().setValue(0)
        self._fade_in(page)

        title, sub = _TITLES.get(page_id, (page_id, ""))
        self.pageTitleLabel.setText(title)
        self.pageSubLabel.setText(sub)

        highlighted = _NAV_PARENT.get(page_id, page_id)
        for pid, btn in self._nav_btns.items():
            btn.setChecked(pid == highlighted)

        if hasattr(page, "on_shown"):
            page.on_shown()
        from services.workspace_session import restore_page
        restore_page(self, page_id, page)

    def _fade_in(self, page: QWidget) -> None:
        pass  # page switch via QStackedWidget is instant; no animation needed

    def open_in_result_console(self, path: "Path") -> None:
        self.navigate("console")
        page = self._pages.get("console")
        if page and hasattr(page, "open_result"):
            page.open_folder(path) if path.is_dir() else page.open_result(path)

    def open_result_details(self) -> None:
        """Expose the shared loaded result's inspector without reopening its files."""
        self.navigate("console")
        console = self._pages.get("console")
        if hasattr(console, "details_toggle"):
            console.details_toggle.setChecked(True)
            console.inspector_tabs.setCurrentIndex(0)

    def overview(self):
        """The Sonuçlar overview tab (orbit, station map, Gantt), once created."""
        return getattr(self._pages.get("console"), "overview", None)

    def _on_result_loaded(self, result):
        console = self._pages.get("console")
        overview = self.overview()
        if hasattr(overview, "set_result"):
            overview.set_result(result, getattr(console, "loaded_details", {}))

    def _on_overview_load_failed(self, message):
        overview = self.overview()
        if hasattr(overview, "run_label"):
            previous = f"{overview.result.name} korunuyor. " if overview.result else ""
            overview.run_label.setText(previous + "Koşu açılamadı: " + message + " · Arşivden tekrar seçin.")

    def open_in_mission(self, path):
        if path.is_dir():
            self.open_in_result_console(path)  # Multiple-run folders need the visible picker.
            return
        self.navigate("mission")
        console = self._pages.get("console")
        overview = self.overview()
        if hasattr(overview, "run_label"):
            overview.run_label.setText("Koşu yükleniyor… Önceki görüntü yeni kaynak doğrulanana kadar korunur.")
        if hasattr(console, "open_result"):
            console.open_result(path)

    def start_tutorial(self) -> None:
        """Start a data-free educational run through the normal process monitor."""
        import uuid
        from services.project_paths import RESULTS_DIR
        page = self._ensure_page("run_monitor")
        if page._worker.is_running:
            self.navigate("run_monitor")
            return
        run_id = uuid.uuid4().hex
        folder = RESULTS_DIR / "tutorial" / run_id[:12]
        folder.mkdir(parents=True, exist_ok=False)
        self.navigate("run_monitor")
        page.preset_script(str(DESKTOP_DIR.parent / "examples/desktop_tutorial.py"),
            extra_args=["--output", str(folder), "--run-id", run_id], auto_start=True,
            job_context={"folder": str(folder), "run_id": run_id})

    def run_desktop_job(self, job_path: "Path", folder: "Path", run_id: str) -> None:
        self.navigate("run_monitor")
        page = self._pages.get("run_monitor")
        if page and hasattr(page, "preset_script"):
            page.preset_script(str(DESKTOP_DIR / "scenario_job.py"), extra_args=[str(job_path)], auto_start=True,
                job_context={"folder": str(folder), "run_id": run_id})

    def compare_runs(self, left: "Path", right: "Path") -> None:
        self.navigate("comparison")
        page = self._pages.get("comparison")
        if page and hasattr(page, "set_files"):
            page.set_files(left, right)

    def edit_archived_scenario(self, archived) -> None:
        page = self._ensure_page("scenario")
        if page and hasattr(page, "load_archived"):
            page.load_archived(archived)
            self.navigate("scenario")

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
        from services.workspace_session import save_session
        self._startup_timer.stop()
        self.environment.shutdown()
        save_session(self)
        for page in self._pages.values():
            if hasattr(page, "shutdown"):
                page.shutdown()
        self._settings.setValue("mainWindow/geometry", self.saveGeometry())
        super().closeEvent(event)
