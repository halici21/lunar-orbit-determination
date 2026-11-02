"""Arşiv › Test koşuları: recorded pytest sessions and imported CI runs."""
from __future__ import annotations

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QUrl
from PyQt5.QtGui import QColor, QDesktopServices
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableView, QSplitter,
    QPlainTextEdit, QHeaderView)

from services.archive_sources import TEST_RUNS_DIR, load_test_runs
from styles.theme import C
from workers.artifact_task import ArtifactJobs


def _outcome(run):
    if run.get("source") == "ci":
        return {"success": "Geçti", "failure": "Başarısız", "cancelled": "İptal"}.get(run.get("conclusion"), run.get("conclusion") or "Sürüyor")
    if run.get("failed") or run.get("errors"):
        return "Başarısız"
    return "Geçti" if run.get("exit_status") == 0 else "Kesildi"


def _scope(args):
    """Which tests a session ran: its path / node-id arguments, else the whole suite."""
    targets = [a.replace("\\", "/").split("/")[-1] for a in args
               if a.endswith(".py") or "::" in a or a.startswith(("tests", "./tests"))]
    if not targets:
        return "tüm paket"
    return ", ".join(targets[:3]) + (f" +{len(targets) - 3}" if len(targets) > 3 else "")


class TestRunModel(QAbstractTableModel):
    COLUMNS = ("Bitiş / UTC", "Kaynak", "Sonuç", "Geçti", "Başarısız", "Atlandı", "Süre", "Dal", "Commit", "Kapsam")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.runs = ()

    def replace(self, runs):
        self.beginResetModel()
        self.runs = runs
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.runs)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.COLUMNS)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        run = self.runs[index.row()]
        failed = (run.get("failed") or 0) + (run.get("errors") or 0)
        values = (str(run.get("finished_at", ""))[:16].replace("T", " "),
                  "CI · " + str(run.get("workflow", "")) if run.get("source") == "ci" else "Yerel pytest",
                  _outcome(run), run.get("passed"), failed if run.get("source") != "ci" else None,
                  run.get("skipped"), f"{run['duration_s']:.0f} s" if run.get("duration_s") is not None else None,
                  run.get("branch"), str(run.get("commit") or "")[:8] + (" *" if run.get("dirty") else ""),
                  _scope(run.get("args", ())))
        if role == Qt.DisplayRole:
            value = values[index.column()]
            return "—" if value in (None, "") else str(value)
        if role == Qt.ForegroundRole and index.column() == 2:
            outcome = values[2]
            return QColor(C.GREEN if outcome == "Geçti" else C.RED if outcome == "Başarısız" else C.TEXT_MUTED)
        if role == Qt.ToolTipRole and index.column() == 8 and run.get("dirty"):
            return "Commit edilmemiş değişikliklerle çalıştırıldı"
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.COLUMNS[section]


class ArchiveTestRuns(ArtifactJobs, QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_jobs()
        self._loaded = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(12)
        bar = QHBoxLayout()
        self.summary = QLabel("")
        self.summary.setObjectName("detailHeading")
        bar.addWidget(self.summary, 1)
        folder = QPushButton("Kayıt klasörü")
        folder.setToolTip(str(TEST_RUNS_DIR))
        folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(TEST_RUNS_DIR))))
        bar.addWidget(folder)
        refresh = QPushButton("Yenile")
        refresh.clicked.connect(self.refresh)
        bar.addWidget(refresh)
        layout.addLayout(bar)
        hint = QLabel("Her pytest oturumu bitince buraya bir kayıt düşer (LUNAR_OD_RECORD_TESTS=0 kapatır). "
                      "CI koşuları tools/import_ci_test_runs.py ile eklenir.")
        hint.setProperty("role", "caption")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        splitter = QSplitter(Qt.Vertical)
        self.model = TestRunModel(self)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QTableView.SingleSelection)
        self.table.verticalHeader().hide()
        self.table.setShowGrid(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        for column, width in enumerate((130, 150, 90, 60, 80, 70, 70, 190, 90)):
            self.table.setColumnWidth(column, width)
        self.table.selectionModel().currentRowChanged.connect(self._show_detail)
        splitter.addWidget(self.table)
        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setProperty("role", "log")
        self.detail.setPlaceholderText("Bir koşu seçin: komut, ortam ve başarısız testler burada görünür.")
        splitter.addWidget(self.detail)
        splitter.setSizes([420, 160])
        layout.addWidget(splitter, 1)

    def on_shown(self):
        if not self._loaded and not self._jobs:
            self.refresh()

    def refresh(self):
        self.summary.setText("Test kayıtları okunuyor…")
        self.start_job(load_test_runs, (TEST_RUNS_DIR,))

    def _job_ready(self, generation, runs):
        if self._closing or generation != self._generation:
            return
        self._loaded = True
        self.model.replace(runs)
        local = [r for r in runs if r.get("source") != "ci"]
        if not runs:
            self.summary.setText("Henüz test kaydı yok")
        elif local:
            last = local[0]
            self.summary.setText(f"Son yerel koşu: {_outcome(last)} · {last.get('passed', 0)} geçti, "
                                 f"{(last.get('failed') or 0) + (last.get('errors') or 0)} başarısız, "
                                 f"{last.get('skipped', 0)} atlandı · {len(runs)} kayıt")
        else:
            self.summary.setText(f"{len(runs)} CI kaydı")

    def _job_failed(self, generation, message):
        if not self._closing and generation == self._generation:
            self.summary.setText("Test kayıtları okunamadı: " + message)

    def _show_detail(self, current, _previous=None):
        if not current.isValid():
            return
        run = self.model.runs[current.row()]
        lines = [f"Komut: pytest {' '.join(run.get('args', ()))}" if run.get("source") != "ci"
                 else f"CI: {run.get('workflow', '')} · {run.get('event', '')} · {run.get('url', '')}",
                 f"Commit: {run.get('commit', '')}  dal: {run.get('branch', '')}" + ("  (commit edilmemiş değişiklik var)" if run.get("dirty") else ""),
                 f"Başlangıç: {run.get('started_at', '')}  bitiş: {run.get('finished_at', '')}"]
        if run.get("python"):
            lines.append(f"Python {run['python']} · toplanan test {run.get('collected', '—')}")
        failures = run.get("failures") or []
        lines.append("")
        lines.append("Başarısız testler:" if failures else "Başarısız test yok.")
        lines.extend("  " + f for f in failures)
        self.detail.setPlainText("\n".join(lines))
