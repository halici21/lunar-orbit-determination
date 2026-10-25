"""Arşiv › Raporlar ve grafikler: read-only list of figures, tables and records."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel, QTimer, QUrl
from PyQt5.QtGui import QDesktopServices, QPixmap
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QComboBox,
    QTableView, QSplitter, QStackedWidget, QPlainTextEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QScrollArea)

from services.archive_sources import REPORT_KINDS, scan_reports
from workers.artifact_task import ArtifactJobs

_PREVIEW_ROWS = 200
_PREVIEW_BYTES = 200_000


def _size(n):
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.0f} kB" if n >= 1e3 else f"{n} B"


class ReportModel(QAbstractTableModel):
    COLUMNS = (("modified", "Tarih / UTC"), ("name", "Dosya"), ("kind", "Tür"), ("folder", "Klasör"),
               ("source", "Kaynak"), ("size", "Boyut"))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.files = ()

    def replace(self, files):
        self.beginResetModel()
        self.files = files
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.files)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.COLUMNS)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        item = self.files[index.row()]
        key = self.COLUMNS[index.column()][0]
        value = item.path.name if key == "name" else getattr(item, key)
        if role == Qt.DisplayRole:
            if key == "modified":
                return value.replace("T", " ")[:16]
            if key == "size":
                return _size(value)
            return value or "—"
        if role == Qt.UserRole:
            return value
        if role == Qt.ToolTipRole:
            return str(item.path)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.COLUMNS[section][1]


class ReportFilter(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.query, self.kind = "", ""
        self.setSortRole(Qt.UserRole)

    def filterAcceptsRow(self, row, parent):
        item = self.sourceModel().files[row]
        text = f"{item.path.name} {item.folder} {item.source}".lower()
        return self.query in text and (not self.kind or item.kind == self.kind)


class ArchiveReports(ArtifactJobs, QWidget):
    """Figures, tables and records from the archive sources, with a preview."""

    def __init__(self, roots_provider, parent=None):
        super().__init__(parent)
        self.init_jobs()
        self._roots = roots_provider
        self._loaded = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(12)
        bar = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Dosya, klasör veya kaynak ara…")
        self.search.setAccessibleName("Raporlarda ara")
        bar.addWidget(self.search, 1)
        self.kind_combo = QComboBox()
        self.kind_combo.addItem("Tüm türler", "")
        for kind in dict.fromkeys(REPORT_KINDS.values()):
            self.kind_combo.addItem(kind, kind)
        bar.addWidget(self.kind_combo)
        self.refresh_button = QPushButton("Yenile")
        self.refresh_button.clicked.connect(self.refresh)
        bar.addWidget(self.refresh_button)
        layout.addLayout(bar)
        splitter = QSplitter(Qt.Horizontal)
        self.model = ReportModel(self)
        self.proxy = ReportFilter(self)
        self.proxy.setSourceModel(self.model)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QTableView.SingleSelection)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(0, Qt.DescendingOrder)
        self.table.verticalHeader().hide()
        self.table.setShowGrid(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        for column, width in enumerate((130, 300, 70, 220, 150, 70)):
            self.table.setColumnWidth(column, width)
        self.table.selectionModel().currentRowChanged.connect(self._show_preview)
        splitter.addWidget(self.table)
        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(12, 0, 0, 0)
        self.preview_title = QLabel("Bir dosya seçin.")
        self.preview_title.setWordWrap(True)
        self.preview_title.setObjectName("detailHeading")
        preview_layout.addWidget(self.preview_title)
        self.preview_stack = QStackedWidget()
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignCenter)
        image_scroll = QScrollArea()
        image_scroll.setWidgetResizable(True)
        image_scroll.setWidget(self.image)
        self.grid = QTableWidget()
        self.grid.setEditTriggers(QTableWidget.NoEditTriggers)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setProperty("role", "log")
        for widget in (image_scroll, self.grid, self.text):
            self.preview_stack.addWidget(widget)
        preview_layout.addWidget(self.preview_stack, 1)
        actions = QHBoxLayout()
        self.reveal_button = QPushButton("Klasörde göster")
        self.reveal_button.clicked.connect(self._reveal)
        self.open_button = QPushButton("Varsayılan uygulamayla aç")
        self.open_button.clicked.connect(self._open_external)
        for button in (self.reveal_button, self.open_button):
            button.setEnabled(False)
            actions.addWidget(button)
        actions.addStretch(1)
        preview_layout.addLayout(actions)
        splitter.addWidget(preview)
        splitter.setSizes([700, 460])
        layout.addWidget(splitter, 1)
        self.status_label = QLabel("")
        self.status_label.setProperty("role", "caption")
        layout.addWidget(self.status_label)
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(120)
        self.search.textChanged.connect(lambda: self._search_timer.start())
        self._search_timer.timeout.connect(self.apply_filter)
        self.kind_combo.currentIndexChanged.connect(self.apply_filter)

    def on_shown(self):
        if not self._loaded and not self._jobs:
            self.refresh()

    def refresh(self):
        roots = self._roots()
        self.status_label.setText("Rapor ve grafikler taranıyor…")
        self.start_job(scan_reports, (roots,))

    def _job_ready(self, generation, files):
        if self._closing or generation != self._generation:
            return
        self._loaded = True
        self.model.replace(files)
        self.apply_filter()

    def _job_failed(self, generation, message):
        if not self._closing and generation == self._generation:
            self.status_label.setText("Raporlar okunamadı: " + message)

    def apply_filter(self, *args):
        self.proxy.query = self.search.text().strip().lower()
        self.proxy.kind = self.kind_combo.currentData() or ""
        self.proxy.invalidateFilter()
        roots = len({f.source for f in self.model.files})
        self.status_label.setText(f"{self.proxy.rowCount()} / {len(self.model.files)} dosya · {roots} kaynak · salt okunur")

    def selected_file(self):
        index = self.table.selectionModel().currentIndex()
        if not index.isValid():
            return None
        return self.model.files[self.proxy.mapToSource(index).row()]

    def _show_preview(self, current, _previous=None):
        item = self.selected_file()
        for button in (self.reveal_button, self.open_button):
            button.setEnabled(item is not None)
        if item is None:
            return
        self.preview_title.setText(f"{item.path.name}\n{item.folder or item.source} · {_size(item.size)}")
        suffix = item.path.suffix.lower()
        try:
            if item.kind == "Grafik" and suffix != ".svg":
                pixmap = QPixmap(str(item.path))
                width = max(200, self.preview_stack.width() - 24)
                self.image.setPixmap(pixmap.scaledToWidth(min(width, pixmap.width()), Qt.SmoothTransformation)
                                     if not pixmap.isNull() else QPixmap())
                self.image.setText("" if not pixmap.isNull() else "Görüntü açılamadı.")
                self.preview_stack.setCurrentIndex(0)
            elif suffix == ".csv":
                self._preview_csv(item.path)
                self.preview_stack.setCurrentIndex(1)
            else:
                raw = item.path.read_bytes()[:_PREVIEW_BYTES].decode("utf-8", errors="replace")
                if suffix == ".json":
                    try:
                        raw = json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
                    except ValueError:
                        pass  # truncated or not JSON: show the text as is
                self.text.setPlainText(raw)
                self.preview_stack.setCurrentIndex(2)
        except OSError as exc:
            self.text.setPlainText(f"Dosya okunamadı: {exc}")
            self.preview_stack.setCurrentIndex(2)

    def _preview_csv(self, path):
        with open(path, newline="", encoding="utf-8", errors="replace") as handle:
            rows = []
            for row in csv.reader(handle):
                rows.append(row)
                if len(rows) > _PREVIEW_ROWS:
                    break
        header, body = (rows[0], rows[1:]) if rows else ([], [])
        self.grid.clear()
        self.grid.setColumnCount(len(header))
        self.grid.setRowCount(len(body))
        self.grid.setHorizontalHeaderLabels(header)
        for r, row in enumerate(body):
            for c, value in enumerate(row[:len(header)]):
                self.grid.setItem(r, c, QTableWidgetItem(value))
        self.grid.resizeColumnsToContents()

    def _reveal(self):
        item = self.selected_file()
        if item is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(item.path.parent)))

    def _open_external(self):
        item = self.selected_file()
        if item is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(item.path)))
