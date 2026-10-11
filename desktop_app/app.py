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

# (page_id, sidebar_label, hint_text)
_PRIMARY_NAV = [
    ("mission", "Görev görünümü", "Yörünge, Dünya istasyonları ve ölçüm zamanlarını birlikte izle"),
    ("scenario", "Senaryo", "Ayarları hazırla veya arşivden bir koşuyu temel al"),
    ("run_monitor", "Çalıştır", "Çalışan işi ve kayıtlarını izle"),
    ("console", "Sonuçlar", "Grafikler, kayıtlar ve koşu ayrıntıları"),
    ("archive", "Arşiv", "Kaydedilmiş koşuları bul ve yeniden kullan"),
    ("comparison", "Karşılaştırma", "İki koşunun kayıtlarını ve ayarlarını karşılaştır"),
    ("stations", "İstasyonlar", "Dünya haritası, koordinatlar ve ağ kataloğu"),
    ("visibility", "Temas planı", "SPICE ile istasyon görünürlüğü ve Gantt hesabı"),
    ("ground_track", "Yüzey izi", "Ay yüzeyindeki iz ve yörünge tasarım önizlemesi"),
]
_ADVANCED = [
    ("dashboard", "Genel bakış", "Araştırma özeti"),
    ("results", "Dosya tarayıcı", "Diğer CSV ve görsel dosyalar"),
    ("dynamics", "Dinamik", "Kuvvet modeli ve yayılım"),
    ("measurements", "Ölçümler", "Gürültü, bias ve Doppler"),
    ("estimators", "Kestiriciler", "BLS / SRIF / UKF ayarları"),
    ("analysis", "Analiz", "Kestirici incelemesi"),
]

_NAV = _PRIMARY_NAV + _ADVANCED + [("settings", "Ayarlar", "Uygulama ayarları")]

_TITLES: dict[str, tuple[str, str]] = {
    "mission": ("Görev görünümü", "Kayıtlı yörüngeyi oynatın; istasyonları ve ölçüm aralıklarını aynı zamanda inceleyin."),
    "console": ("Sonuçlar", "Kayıtlı grafikler, koşu ayarları ve kaynak ilişkileri"),
    "archive": ("Koşu arşivi", "Gerçek koşular, ayarlar ve tamamlanma kayıtları"),
    "dashboard":   ("Mission Dashboard",   "Lunar OD research overview"),
    "results":     ("Results Browser",     "Browse CSV tables and PNG plots"),
    "run_monitor": ("Çalıştır", "İşlem durumu, canlı günlük ve çıktı dosyaları"),
    "comparison":  ("Koşu karşılaştırması", "Kaydedilmiş değerler, tam epok eşleşmeleri ve ayar farkları"),
    "summary_comparison": ("Diğer CSV özetleri", "Eski özet CSV araçları"),
    "scenario": ("Senaryo", "Ölçüm türü, istasyon ağı, süre ve kestiriciyi seçin; etkin ayarları kaydedip yeni bir sayısal koşu başlatın."),
    "dynamics":    ("Dinamik model", "Kuvvet modeli, SPICE dosyaları ve yayılım ayarları. Çalıştırılacak etkin yapılandırmayı Senaryo sayfasında doğrulayın."),
    "measurements":("Ölçüm modeli", "Menzil, menzil hızı ve konum ölçümleri; gürültü ve bias ayarlarını inceleyin."),
    "estimators":  ("Kestiriciler", "BLS ölçüm grubunu birlikte çözer; UKF sırayla günceller; SRIF bilgi matrisiyle çalışır. Ayarların etkin karşılığını Senaryo sayfasında kontrol edin."),
    "stations":    ("İstasyonlar", "Dünya üzerindeki konumları ve ağ kataloğunu inceleyin. Özel istasyonlar katalogda saklanır; yeni koşunun ağı Senaryo sayfasında seçilir."),
    "visibility":  ("Temas planı", "Seçilen istasyonlar için bağımsız görünürlük hesabı yapın. Gantt geometrik görünürlük örneklerini gösterir; kayıtlı koşunun ölçüm takvimi değildir."),
    "analysis":     ("Estimator Analysis",  "BLS-LM vs UKF comparison — observable type, start mode, α/β/κ sensitivity"),
    "ground_track": ("Ay yüzey izi", "Uzay aracının Ay üzerindeki izdüşümünü inceleyin. Yörünge tasarım önizlemesi Kepler modelidir; kayıtlı sonuçlardan ayrıdır."),
    "settings":     ("Ayarlar", "Python ortamını ve SPICE dosyalarını kontrol edin. Hazırlık denetimi hesaplamayı başlatmadan eksikleri gösterir."),
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
        return page.minimumSizeHint() if page else QSize(960, 720)

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
        self.pageStack.setMinimumSize(960, 720)
        self.pageScroll.setWidget(self.pageStack)

        self._settings = QSettings("LunarOD", "DesktopApp")
        from controllers.environment_controller import EnvironmentController
        self.environment = EnvironmentController(self._settings, self)
        from services.workspace_session import read_session
        self._workspace_session = read_session(self._settings)
        self._restored_pages = set()
        self._active_page = "mission"
        self._latest_result = None
        self._nav_btns: dict[str, _NavButton] = {}
        self._pages: dict[str, QWidget] = {}

        self.brandSubLabel.setText("Yörünge belirleme çalışma alanı")
        self.footerNativeLabel.setText("Yerel çalışma alanı")
        self._apply_styles()
        self._build_nav()
        self._build_status_bar()
        self._load_pages()
        self._restore_geometry()
        active = self._workspace_session.get("active", "mission")
        self.navigate(active if active in ("mission", "console", "comparison", "archive") else "mission")
        if self._active_page == "mission" and isinstance(self._workspace_session.get("console"), dict):
            from services.workspace_session import restore_page
            restore_page(self, "console", self._ensure_page("console"))
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
        self.advanced_toggle = QPushButton("Gelişmiş araçlar  ›")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setStyleSheet(f"text-align:left; padding:12px 16px; color:{C.TEXT_MUTED}; border:0; background:transparent;")
        layout.addWidget(self.advanced_toggle)
        self.advanced_panel = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_panel)
        advanced_layout.setContentsMargins(10, 0, 0, 0)
        advanced_layout.setSpacing(0)
        for item in _ADVANCED:
            add_button(item, advanced_layout)
        layout.addWidget(self.advanced_panel)
        self.advanced_panel.hide()
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        layout.addStretch()
        add_button(("settings", "Ayarlar", "Dosya yolları ve uygulama ayarları"), layout)

    def _toggle_advanced(self, expanded):
        self.advanced_panel.setVisible(expanded)
        self.advanced_toggle.setText("Gelişmiş araçlar  ⌄" if expanded else "Gelişmiş araçlar  ›")

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
            "mission": ("controllers.mission_controller", "MissionController"),
            "console": ("controllers.tracking_console_controller", "TrackingConsoleController"),
            "archive": ("controllers.run_archive_controller", "RunArchiveController"),
            "dashboard": ("controllers.dashboard_controller", "DashboardController"),
            "results": ("controllers.results_browser_controller", "ResultsBrowserController"),
            "run_monitor": ("controllers.run_monitor_controller", "RunMonitorController"),
            "comparison": ("controllers.recorded_comparison_controller", "RecordedComparisonController"),
            "summary_comparison": ("controllers.comparison_controller", "ComparisonController"),
            "scenario": ("controllers.scenario_builder_controller", "ScenarioBuilderController"),
            "stations": ("controllers.stations_controller", "StationsController"),
            "visibility": ("controllers.visibility_controller", "VisibilityController"),
            "analysis": ("controllers.analysis_controller", "AnalysisController"),
            "ground_track": ("controllers.ground_track_controller", "GroundTrackController"),
            **{page: ("controllers.system_pages_controller", cls) for page, cls in (
                ("dynamics", "DynamicsController"), ("measurements", "MeasurementsController"),
                ("estimators", "EstimatorsController"), ("settings", "SettingsController"))},
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
            widget.loadFailed.connect(self._on_mission_load_failed)
        if page_id == "mission" and self._latest_result and hasattr(widget, "set_result"):
            widget.set_result(*self._latest_result)
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
        page = self._ensure_page(page_id)
        if page is None:
            return
        if page_id in {item[0] for item in _ADVANCED}:
            self.advanced_toggle.setChecked(True)
        self._active_page = page_id
        for other in self._pages.values():
            other.setSizePolicy(other._workspace_size_policy if other is page else
                                QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored))
        self.pageStack.setCurrentWidget(page)
        self.pageStack.updateGeometry()
        self.pageScroll.verticalScrollBar().setValue(0)
        self.pageScroll.horizontalScrollBar().setValue(0)
        self._fade_in(page)

        title, sub = _TITLES.get(page_id, (page_id, ""))
        self.pageTitleLabel.setText(title)
        self.pageSubLabel.setText(sub)

        for pid, btn in self._nav_btns.items():
            btn.setChecked(pid == page_id)

        if hasattr(page, "on_shown"):
            page.on_shown()
        from services.workspace_session import restore_page
        restore_page(self, page_id, page)

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

    def open_in_result_console(self, path: "Path") -> None:
        self.navigate("console")
        page = self._pages.get("console")
        if page and hasattr(page, "open_result"):
            page.open_folder(path) if path.is_dir() else page.open_result(path)

    def _on_result_loaded(self, result):
        console = self._pages.get("console")
        self._latest_result = (result, getattr(console, "loaded_details", {}))
        mission = self._pages.get("mission")
        if mission and hasattr(mission, "set_result"):
            mission.set_result(*self._latest_result)

    def _on_mission_load_failed(self, message):
        mission = self._pages.get("mission")
        if mission and hasattr(mission, "run_label"):
            previous = f"{mission.result.name} korunuyor. " if mission.result else ""
            mission.run_label.setText(previous + "Koşu açılamadı: " + message + " · Arşivden tekrar seçin.")

    def open_in_mission(self, path):
        if path.is_dir():
            self.open_in_result_console(path)  # Multiple-run folders need the visible picker.
            return
        console = self._ensure_page("console")
        self.navigate("mission")
        mission = self._pages.get("mission")
        if hasattr(mission, "run_label"):
            mission.run_label.setText("Koşu yükleniyor… Önceki görüntü yeni kaynak doğrulanana kadar korunur.")
        if hasattr(console, "open_result"):
            console.open_folder(path) if path.is_dir() else console.open_result(path)

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
