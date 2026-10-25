"""Native real-run archive with filtered, stable source identities."""
from __future__ import annotations
from functools import partial

import json
from pathlib import Path

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel, QTimer
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QTableView, QSplitter, QPlainTextEdit, QFileDialog, QHeaderView, QTabWidget, QMenu)

from services.project_paths import RESULTS_DIR
from services.archive_sources import SOURCES_KEY, extra_sources, report_roots, scan_archive_sources
from services.archived_scenario import load_archived_scenario
from styles.artifact_workspace import ARTIFACT_STYLE
from workers.artifact_task import ArtifactJobs
from widgets.run_details_panel import RunDetailsPanel

STATUS = {"completed": "İşlem tamamlandı", "incomplete": "Tamamlanma kaydı yok", "legacy": "Eski / kayıtsız", "invalid": "Dizin hatası"}
KIND = {"arc_solution": "Yay çözümü", "ukf_update": "UKF güncellemeleri"}


class ArchiveModel(QAbstractTableModel):
    COLUMNS = (("created_at", "Zaman / UTC"), ("name", "Koşu"), ("run_id", "Kimlik"), ("estimator", "Çözücü"),
        ("records", "N"), ("status", "İşlem durumu"), ("kind", "Veri"),
        ("measurement", "Ölçüm"), ("network", "Ağ"), ("seed", "Seed"), ("source", "Kaynak"))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.entries = ()

    def replace(self, entries):
        self.beginResetModel()
        self.entries = entries
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.entries)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.COLUMNS)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        entry = self.entries[index.row()]
        value = getattr(entry, self.COLUMNS[index.column()][0])
        if role == Qt.DisplayRole:
            if index.column() == 0:
                return str(value).replace("T", " ")[:19]
            if index.column() == 2:
                return value[:12]
            if index.column() == 6:
                return KIND[value]
            if index.column() == 5:
                return STATUS[value]
            return str(value) if value is not None else "—"
        if role == Qt.UserRole:
            return value if value is not None else ""
        if role == Qt.ToolTipRole:
            return str(entry.path) + "\n" + "\n".join(entry.notes)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.COLUMNS[section][1]


class ArchiveFilter(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.query, self.estimator, self.status, self.kind = "", "", "", ""
        self.setSortRole(Qt.UserRole)

    def filterAcceptsRow(self, row, parent):
        entry = self.sourceModel().entries[row]
        text = f"{entry.name} {entry.run_id} {entry.path} {entry.measurement} {entry.network} {entry.seed} {entry.derived_from.get('run_id', '')}".lower()
        return (self.query in text and (not self.estimator or entry.estimator == self.estimator)
            and (not self.status or entry.status == self.status) and (not self.kind or entry.kind == self.kind))


class RunArchiveController(ArtifactJobs, QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_jobs()
        self.folder = RESULTS_DIR
        self._started = False
        self._count_cache = {}
        self._task = "scan"
        self.setObjectName("artifactWorkspace")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(ARTIFACT_STYLE)
        settings = getattr(parent, "_settings", None)
        self._settings = settings
        self.extra_sources = extra_sources(settings)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 14, 24, 18)
        self.sections = QTabWidget()
        outer.addWidget(self.sections)
        self.runs_page = QWidget()
        layout = QVBoxLayout(self.runs_page)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(14)
        header = QHBoxLayout()
        self.path_label = QLabel(str(self.folder))
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.path_label.setWordWrap(True)
        self.path_label.setProperty("role", "caption")
        header.addWidget(self.path_label, 1)
        self.sources_button = QPushButton("Kaynaklar")
        self.sources_button.setToolTip("Arşive eklenen diğer sonuç klasörleri (salt okunur)")
        self.sources_menu = QMenu(self)
        self.sources_menu.aboutToShow.connect(self._fill_sources_menu)
        self.sources_button.setMenu(self.sources_menu)
        header.addWidget(self.sources_button)
        choose = QPushButton("Klasör seç…")
        choose.setToolTip("Başka bir sonuç klasöründeki koşuları listele")
        choose.clicked.connect(self._choose_folder)
        header.addWidget(choose)
        self.refresh_button = QPushButton("Yenile")
        self.refresh_button.setToolTip("Tüm CSV dosyalarını yeniden tara; saklanan satır sayılarını kullanma.")
        self.refresh_button.clicked.connect(lambda: self.refresh())
        header.addWidget(self.refresh_button)
        self.cancel_button = QPushButton("İşlemi iptal et")
        self.cancel_button.clicked.connect(self.cancel_scan)
        self.cancel_button.setEnabled(False)
        header.addWidget(self.cancel_button)
        layout.addLayout(header)
        filters = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Koşu, kimlik, ağ, seed veya dosya yolu ara…")
        self.search.setAccessibleName("Arşivde ara")
        filters.addWidget(self.search, 1)
        self.estimator_combo = QComboBox()
        self.estimator_combo.addItem("Tüm çözücüler", "")
        self.status_combo = QComboBox()
        self.status_combo.setToolTip("İşlem durumu yalnız sürecin bitip bitmediğini söyler; sayısal çözüm başarısını belirtmez.")
        self.status_combo.addItem("Tüm işlem durumları", "")
        for key, label in STATUS.items():
            self.status_combo.addItem(label, key)
        self.kind_combo = QComboBox()
        self.kind_combo.addItem("Tüm veri türleri", "")
        for key, label in KIND.items():
            self.kind_combo.addItem(label, key)
        for combo in (self.estimator_combo, self.status_combo, self.kind_combo):
            combo.currentIndexChanged.connect(self.apply_filter)
            filters.addWidget(combo)
        layout.addLayout(filters)
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(120)
        self.search.textChanged.connect(lambda: self._search_timer.start())
        self._search_timer.timeout.connect(self.apply_filter)
        splitter = QSplitter(Qt.Horizontal)
        self.table = QTableView()
        self.model = ArchiveModel(self)
        self.proxy = ArchiveFilter(self)
        self.proxy.setSourceModel(self.model)
        self.table.setModel(self.proxy)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QTableView.ExtendedSelection)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(0, Qt.DescendingOrder)
        self.table.verticalHeader().hide()
        self.table.setShowGrid(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        for column, width in enumerate((150, 190, 105, 70, 40, 175, 110, 90, 60, 50)):
            self.table.setColumnWidth(column, width)
        self.table.selectionModel().selectionChanged.connect(self._selection_changed)
        self.table.doubleClicked.connect(lambda _: self.open_mission())
        splitter.addWidget(self.table)
        detail_panel = QWidget()
        detail_layout = QVBoxLayout(detail_panel)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        self.lineage_label = QLabel()
        self.lineage_label.setWordWrap(True)
        self.lineage_label.setTextFormat(Qt.PlainText)
        detail_layout.addWidget(self.lineage_label)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumWidth(290)
        self.details.setPlaceholderText("Bir koşu seçin. Kaynak, koşu kimliği ve ayarlar burada görünür.")
        self.detail_tabs = QTabWidget()
        self.summary = RunDetailsPanel()
        self.detail_tabs.addTab(self.summary, "Özet")
        self.detail_tabs.addTab(self.details, "Teknik")
        detail_layout.addWidget(self.detail_tabs, 1)
        splitter.addWidget(detail_panel)
        splitter.setSizes([1000, 320])
        layout.addWidget(splitter, 1)
        actions = QHBoxLayout()
        # Overview, charts and records are tabs of one Sonuçlar page: one open action.
        self.open_button = QPushButton("Sonuçlarda aç")
        self.open_button.setProperty("primary", True)
        self.open_button.clicked.connect(self.open_mission)
        actions.addWidget(self.open_button)
        self.clone_button = QPushButton("Senaryoya kopyala")
        self.clone_button.setToolTip("Kayıtlı formu doğrula ve Senaryo sayfasında yeni bir taslak olarak aç")
        self.clone_button.clicked.connect(self.clone_selected)
        actions.addWidget(self.clone_button)
        self.compare_button = QPushButton("Karşılaştır")
        self.compare_button.setToolTip("İki koşu seçin (Ctrl veya Shift ile)")
        self.compare_button.clicked.connect(self.compare_selected)
        actions.addWidget(self.compare_button)
        actions.addStretch()
        layout.addLayout(actions)
        self.status_label = QLabel("UKF güncelleme ve native yay çözümü paketleri listelenir.")
        layout.addWidget(self.status_label)
        self._selection_changed()
        self._update_path_label()
        self.sections.addTab(self.runs_page, "Koşular")
        from widgets.archive_reports import ArchiveReports
        from widgets.archive_test_runs import ArchiveTestRuns
        self.reports = ArchiveReports(lambda: report_roots(self.extra_sources))
        self.test_runs = ArchiveTestRuns()
        self.sections.addTab(self.reports, "Raporlar ve grafikler")
        self.sections.addTab(self.test_runs, "Test koşuları")
        self.sections.currentChanged.connect(self._section_changed)

    def _section_changed(self, index):
        page = self.sections.widget(index)
        if hasattr(page, "on_shown"):
            page.on_shown()

    def _update_path_label(self):
        extra = [str(p) for p in self.extra_sources if p != self.folder]
        self.path_label.setText(str(self.folder) + (f"  ·  +{len(extra)} kaynak" if extra else ""))
        self.path_label.setToolTip("\n".join([str(self.folder), *extra]))

    def _fill_sources_menu(self):
        self.sources_menu.clear()
        header = self.sources_menu.addAction("Bu çalışma alanı: " + str(RESULTS_DIR))
        header.setEnabled(False)
        for path in self.extra_sources:
            action = self.sources_menu.addAction(f"Kaldır: {path}" + ("" if path.exists() else "  (bulunamadı)"))
            action.triggered.connect(lambda _=False, p=path: self.set_extra_sources([s for s in self.extra_sources if s != p]))
        self.sources_menu.addSeparator()
        self.sources_menu.addAction("Klasör ekle…", self._add_source)

    def _add_source(self):
        folder = QFileDialog.getExistingDirectory(self, "Arşive kaynak klasör ekle", str(self.folder))
        if folder and Path(folder) not in self.extra_sources:
            self.set_extra_sources([*self.extra_sources, Path(folder)])

    def set_extra_sources(self, folders):
        self.extra_sources = tuple(Path(f) for f in folders)
        if self._settings is not None:
            self._settings.setValue(SOURCES_KEY, [str(p) for p in self.extra_sources])
        self._update_path_label()
        self.refresh()
        if getattr(self.reports, "_loaded", False):
            self.reports.refresh()

    def on_shown(self):
        if not self._jobs:
            self.refresh(force=False)
        self._section_changed(self.sections.currentIndex())

    def _choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Gerçek koşu arşivi", str(self.folder))
        if folder:
            self.set_folder(Path(folder))

    def set_folder(self, folder):
        self.folder = Path(folder).resolve()
        self._update_path_label()
        self.refresh()

    def refresh(self, *, force=True):
        self._task = "scan"
        self._started = True
        self._restore_paths = {entry.path for entry in self.selected_entries()}
        self.cancel_button.setEnabled(True)
        self.status_label.setText("Kaynak klasörleri arka planda taranıyor…")
        # Each worker owns its cache snapshot; canceled generations cannot
        # overwrite the active generation. Explicit refresh rescans all CSVs.
        self._count_cache = {} if force else dict(self._count_cache)
        folders = [self.folder, *(p for p in self.extra_sources if p != self.folder)]
        self.start_job(partial(scan_archive_sources, count_cache=self._count_cache,
            missing_ok=self.folder == RESULTS_DIR), (folders,))

    def cancel_scan(self):
        self.cancel_jobs()
        self.cancel_button.setEnabled(False)
        self.status_label.setText("İşlem iptal edildi; önceki görünüm korunuyor.")

    def _job_ready(self, generation, entries):
        if self._closing or generation != self._generation:
            return
        if self._task == "clone":
            self.cancel_button.setEnabled(False)
            if hasattr(self.window(), "edit_archived_scenario"):
                try:
                    self.window().edit_archived_scenario(entries)
                except (ValueError, TypeError) as exc:
                    self.status_label.setText("Form geri yüklenemedi: " + str(exc))
            return
        self.model.replace(entries)
        self.estimator_combo.blockSignals(True)
        current = self.estimator_combo.currentData()
        self.estimator_combo.clear()
        self.estimator_combo.addItem("Tüm çözücüler", "")
        for value in sorted({e.estimator for e in entries}):
            self.estimator_combo.addItem(value.upper(), value)
        index = self.estimator_combo.findData(current)
        self.estimator_combo.setCurrentIndex(max(0, index))
        self.estimator_combo.blockSignals(False)
        self.apply_filter()
        for row in range(self.proxy.rowCount()):
            source = self.proxy.mapToSource(self.proxy.index(row, 0)).row()
            if entries[source].path in self._restore_paths:
                self.table.selectionModel().select(self.proxy.index(row, 0), self.table.selectionModel().Select | self.table.selectionModel().Rows)
        self.cancel_button.setEnabled(False)
        self._selection_changed()

    def _job_failed(self, generation, message):
        if not self._closing and generation == self._generation:
            self.cancel_button.setEnabled(False)
            self.status_label.setText(("Koşu ayarları geri yüklenemedi: " if self._task == "clone" else "Arşiv okunamadı: ") + message)

    def apply_filter(self, *args):
        self.proxy.query = self.search.text().strip().lower()
        self.proxy.estimator = self.estimator_combo.currentData() or ""
        self.proxy.status = self.status_combo.currentData() or ""
        self.proxy.kind = self.kind_combo.currentData() or ""
        self.proxy.invalidateFilter()
        self._selection_changed()

    def selected_entries(self):
        return tuple(self.model.entries[self.proxy.mapToSource(i).row()] for i in self.table.selectionModel().selectedRows())

    def _selection_changed(self, *args):
        entries = self.selected_entries()
        self.open_button.setEnabled(len(entries) == 1 and entries[0].status != "invalid")

        native = len(entries) == 1 and entries[0].job_schema == "lunar-od-desktop-job-v1"
        self.clone_button.setEnabled(native and entries[0].status != "invalid" and bool(entries[0].metadata.get("job_file")))
        self.clone_button.setToolTip(
            "Kayıtlı formu doğrula ve Senaryo sayfasında yeni bir taslak olarak aç"
            if native or len(entries) != 1 else
            "Bu koşu Senaryo formundan başlatılmadı (ör. örnek eğitim); ayarları forma kopyalanamaz.")
        self.compare_button.setEnabled(len(entries) == 2 and all(e.status != "invalid" for e in entries))
        if entries:
            entry = entries[0]
            self.summary.show_details(entry.name, entry.metadata, {"config": entry.config, "options": entry.options,
                "job": {"run_id": entry.run_id, "created_at": entry.created_at, "derived_from": entry.derived_from},
                "status": STATUS[entry.status], "notes": entry.notes}, records=entry.records)
            origin = entry.derived_from
            changes = origin.get("form_changes", [])
            names = {"name": "koşu adı", "random_seed": "seed", "bls_max_iter": "iterasyon sınırı"}
            changed = ", ".join(names.get(d["field"], d["field"]) for d in changes
                if isinstance(d, dict) and isinstance(d.get("field"), str)) if isinstance(changes, list) else "Bilinmiyor"
            self.lineage_label.setText(f"Kaynak koşu: {origin.get('run_id', 'Bilinmiyor')}\nDeğişen alanlar: {changed or 'Yok'}" if origin else "")
            self.lineage_label.setVisible(bool(origin))
            self.details.setPlainText(json.dumps({"name": entry.name, "run_id": entry.run_id,
                "source": str(entry.path), "created_at": entry.created_at, "status": STATUS[entry.status],
                "records": entry.records, "notes": entry.notes, "metadata": entry.metadata,
                "config": entry.config, "options": entry.options, "derived_from": origin or None}, ensure_ascii=False, indent=2))
        else:
            self.lineage_label.clear()
            self.lineage_label.hide()
            self.details.clear()
            self.summary.clear()
        if not self.model.entries:
            self.status_label.setText("Henüz kayıtlı koşu yok. Senaryo sayfasından bir koşu başlatın ya da başka bir klasör seçin.")
        else:
            hint = " · karşılaştırmak için Ctrl ile ikinci bir koşu seçin" if len(entries) == 1 else ""
            self.status_label.setText(f"{self.proxy.rowCount()} / {len(self.model.entries)} koşu · {len(entries)} seçili{hint}")

    def open_mission(self):
        entries = self.selected_entries()
        if len(entries) == 1 and entries[0].status != "invalid" and hasattr(self.window(), "open_in_mission"):
            self.window().open_in_mission(entries[0].path)

    def open_selected(self):
        entries = self.selected_entries()
        if len(entries) == 1 and entries[0].status != "invalid" and hasattr(self.window(), "open_in_result_console"):
            self.window().open_in_result_console(entries[0].path)

    def compare_selected(self):
        entries = self.selected_entries()
        if len(entries) == 2 and all(e.status != "invalid" for e in entries) and hasattr(self.window(), "compare_runs"):
            self.window().compare_runs(entries[0].path, entries[1].path)

    def clone_selected(self):
        entries = self.selected_entries()
        if len(entries) == 1 and entries[0].status != "invalid" and entries[0].metadata.get("job_file"):
            self._task = "clone"
            self.cancel_button.setEnabled(True)
            self.status_label.setText("Kayıtlı form, kaynak hash'leri ve etkin ayarlar doğrulanıyor…")
            self.start_job(load_archived_scenario, (entries[0].path,))
