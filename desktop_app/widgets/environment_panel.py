"""Compact setup report with explicit states and actionable kernel selection."""
from __future__ import annotations
from datetime import datetime
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                            QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
                            QHeaderView, QAbstractItemView)


class EnvironmentPanel(QWidget):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.summary = QLabel("Ortam henüz kontrol edilmedi")
        self.summary.setTextFormat(Qt.PlainText)
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setAccessibleName("Hesaplamada kullanılacak SPICE kernel klasörü")
        layout.addWidget(self.path_edit)
        row = QHBoxLayout()
        self.choose_button = QPushButton("Kernel klasörü seç")
        self.auto_button = QPushButton("Otomatik bul")
        self.refresh_button = QPushButton("Yeniden kontrol et")
        for button in (self.choose_button, self.auto_button, self.refresh_button):
            row.addWidget(button)
        row.addStretch()
        layout.addLayout(row)
        self.choose_button.clicked.connect(self._choose)
        self.auto_button.clicked.connect(lambda: controller.select_directory(""))
        self.refresh_button.clicked.connect(lambda: controller.refresh())
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Bileşen / dosya", "Durum", "Ayrıntı"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setWordWrap(True)
        self.table.setMinimumHeight(330)
        layout.addWidget(self.table)
        self.note = QLabel("Kernel dosyalarını windows-quickstart kılavuzundaki tam adlarla yerleştirin. "
                           "DE440 çifti birlikte gerekir. Seçim sonraki hesaplamalarda kullanılır; "
                           "başlatılmış alt süreç değişmez. Paket sürümü uyarıları referans profilden farklılığı gösterir. "
                           "Arşiv ve sonuçlar ortam eksikken de açılabilir.")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        controller.changed.connect(self.render)
        self.render()

    def _choose(self):
        path = QFileDialog.getExistingDirectory(self, "SPICE kernel klasörünü seç", self.path_edit.text())
        if path:
            self.controller.select_directory(path)

    def render(self):
        controller = self.controller
        report = controller.report
        self.path_edit.setText(controller.selected_path or (report or {}).get("kernel_dir") or "Otomatik arama")
        for button in (self.choose_button, self.auto_button, self.refresh_button):
            button.setEnabled(not controller.busy)
        if controller.busy:
            self.summary.setText("Ortam kontrol ediliyor · diğer ekranları kullanabilirsiniz")
        elif controller.error:
            self.summary.setText(controller.error)
        elif report:
            self.summary.setText(("Ortam kontrolü geçti" if report["ready"] else "Hesaplama öncesi eksikler var")
                                 + " · " + datetime.fromisoformat(report["checked_at"]).astimezone().strftime("%d.%m.%Y %H:%M:%S"))
        else:
            self.summary.setText("Ortam henüz kontrol edilmedi")
        rows = (report["checks"] + report["kernels"]) if report else []
        # Actionable failures must be visible before healthy package versions.
        priority = {"error": 0, "warning": 1, "pending": 2, "ok": 3}
        rows = sorted(rows, key=lambda record: priority[record["status"]])
        self.table.setRowCount(len(rows))
        labels = {"ok": "Geçti", "warning": "Profil farkı", "error": "Eksik / hata", "pending": "Çalıştırılmadı"}
        for index, record in enumerate(rows):
            for column, text in enumerate((record["name"], labels[record["status"]], record["detail"])):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.table.setItem(index, column, item)
        self.table.resizeRowsToContents()
