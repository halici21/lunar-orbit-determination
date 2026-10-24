"""Native result console: asynchronous artifact I/O and linked source views."""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel,
    QPushButton, QComboBox, QDoubleSpinBox, QCheckBox, QSlider, QTableView,
    QFileDialog, QPlainTextEdit, QTabWidget, QHeaderView, QScrollArea, QBoxLayout)

from services.tracking_results import (discover_tracking_runs, load_tracking_results,
    result_selection, LoadCancelled)
from widgets.tracking_plots import TrackingChart, StoredOrbit, elapsed
from widgets.run_details_panel import RunDetailsPanel
from services.run_details import inspect_run
from services.result_summary import result_summary
from styles.artifact_workspace import ARTIFACT_STYLE


CONSOLE_STYLE = ARTIFACT_STYLE


class _ResultWorker(QThread):
    ready = pyqtSignal(int, str, object)
    failed = pyqtSignal(int, str)

    def __init__(self, generation, kind, path, parent):
        super().__init__(parent)
        self.generation, self.kind, self.path = generation, kind, path

    def run(self):
        try:
            fn = discover_tracking_runs if self.kind == "discover" else load_tracking_results
            result = fn(self.path, self.isInterruptionRequested)
            if self.kind != "discover":
                result = (result, inspect_run(result, self.isInterruptionRequested))
            if not self.isInterruptionRequested():
                self.ready.emit(self.generation, self.kind, result)
        except LoadCancelled:
            pass
        except Exception as exc:
            self.failed.emit(self.generation, str(exc))


class TrackingTableModel(QAbstractTableModel):
    COLUMNS = (("update_id", "#"), ("t_s", "T+ / s"), ("arc_id", "Yay"),
        ("station_id", "İstasyon"), ("position_error_m", "Konum / m"),
        ("velocity_error_mps", "Hız / m/s"), ("normalized_nis", "Norm. NIS"), ("accepted", "Kabul"))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result, self.indices = None, ()

    def set_rows(self, result, indices):
        self.beginResetModel()
        self.result, self.indices = result, indices
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.indices)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.COLUMNS)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not self.result:
            return None
        value = self.result.updates[self.indices[index.row()]][self.COLUMNS[index.column()][0]]
        if role == Qt.DisplayRole:
            if value is None:
                return "—"
            if isinstance(value, bool):
                return "Evet" if value else "Hayır" if self.COLUMNS[index.column()][0] == "operational_success" else "Red"
            if isinstance(value, int):
                return str(value)
            if isinstance(value, str):
                return value or "—"
            return f"{value:.6g}"
        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        if role == Qt.UserRole:
            return value
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.COLUMNS[section][1]
        return None


class TrackingConsoleController(QWidget):
    """First production workflow: folder -> run -> linked analysis."""
    resultLoaded = pyqtSignal(object)
    loadFailed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("trackingConsole")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(CONSOLE_STYLE)
        self.result = None
        self.selected_index = None
        self.visible_indices = ()
        self.runs = ()
        self._workers = set()
        self._generation = 0
        self._busy = False
        self._last_request = None
        self._closing = False
        self._selection_lock = False
        self._row_positions = {}
        self.view_ms = 0
        self.overview = None
        self._build_ui()

    def resizeEvent(self, event):
        # Three side-by-side charts would force the whole tab stack (and the
        # overview tab with it) wider than a laptop window; stack them instead.
        direction = QBoxLayout.TopToBottom if self.width() < 980 else QBoxLayout.LeftToRight
        if self._chart_layout.direction() != direction:
            self._chart_layout.setDirection(direction)
        super().resizeEvent(event)

    def set_overview(self, widget):
        """Insert the orbit/station/Gantt overview as the first results tab."""
        self.overview = widget
        self.view_tabs.insertTab(0, widget, "Genel görünüm")
        self.view_tabs.setCurrentIndex(0)
        self._sync_chart_tools()

    def show_overview(self):
        if self.overview is not None:
            self.view_tabs.setCurrentWidget(self.overview)

    def _sync_chart_tools(self, *_):
        on_overview = self.overview is not None and self.view_tabs.currentWidget() is self.overview
        self.chart_tools.setVisible(not on_overview)
        if on_overview:
            self.filters_toggle.setChecked(False)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 16, 24, 16)
        root.setSpacing(12)
        header = QHBoxLayout()
        self.run_combo = QComboBox()
        self.run_combo.setMinimumWidth(220)
        self.run_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.run_combo.currentIndexChanged.connect(self._choose_run)
        header.addWidget(QLabel("Koşu"))
        header.addWidget(self.run_combo, 1)
        self.cancel_button = QPushButton("Yüklemeyi iptal et")
        self.cancel_button.clicked.connect(self.cancel_load)
        header.addWidget(self.cancel_button)
        if hasattr(self.window(), "navigate"):
            self.archive_button = QPushButton("Arşivden aç")
            self.archive_button.setProperty("primary", True)
            self.archive_button.clicked.connect(lambda: self.window().navigate("archive"))
            header.addWidget(self.archive_button)
        self.open_button = QPushButton("Klasörden aç…")
        self.open_button.clicked.connect(self._choose_folder)
        header.addWidget(self.open_button)
        self.details_toggle = QPushButton("Ayrıntılar")
        self.details_toggle.setCheckable(True)
        self.details_toggle.setToolTip("Koşu bilgisi, seçili kayıt ve teknik kaynaklar")
        header.addWidget(self.details_toggle)
        root.addLayout(header)
        self.path_label = QLabel("Arşivden bir koşu açın veya bir sonuç klasörü seçin.")
        self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.path_label.setProperty("role", "caption")
        root.addWidget(self.path_label)
        feedback = QHBoxLayout()
        self.recovery_message = QLabel()
        self.recovery_message.setTextFormat(Qt.PlainText)
        self.recovery_message.setWordWrap(True)
        self.retry_button = QPushButton("Yeniden dene")
        self.retry_button.clicked.connect(self._retry_load)
        feedback.addWidget(self.recovery_message, 1)
        feedback.addWidget(self.retry_button)
        root.addLayout(feedback)
        self.recovery_message.hide()
        self.retry_button.hide()
        # Chart/record tools: hidden on the overview tab, where they do nothing.
        self.chart_tools = QWidget()
        bar = QHBoxLayout(self.chart_tools)
        bar.setContentsMargins(0, 0, 0, 0)
        self.scale_button = QPushButton("Ölçek: otomatik")
        self.scale_button.clicked.connect(self._toggle_scale)
        bar.addWidget(self.scale_button)
        self.reset_button = QPushButton("Tüm zaman")
        self.reset_button.clicked.connect(self.reset_time)
        bar.addWidget(self.reset_button)
        self.filters_toggle = QPushButton("Filtreler")
        self.filters_toggle.setCheckable(True)
        bar.addWidget(self.filters_toggle)
        bar.addStretch()
        self.export_button = QPushButton("Görünümü CSV aktar")
        self.export_button.clicked.connect(self._export)
        bar.addWidget(self.export_button)
        root.addWidget(self.chart_tools)
        self.filter_panel = QWidget()
        filter_container = QVBoxLayout(self.filter_panel)
        filter_container.setContentsMargins(0, 4, 0, 8)
        filters = QHBoxLayout()
        self.station_combo, self.arc_combo = QComboBox(), QComboBox()
        for label, combo in (("İstasyon", self.station_combo), ("Yay", self.arc_combo)):
            filters.addWidget(QLabel(label))
            filters.addWidget(combo)
            combo.currentIndexChanged.connect(self._queue_filter)
        self.start_hours, self.end_hours = QDoubleSpinBox(), QDoubleSpinBox()
        for spin in (self.start_hours, self.end_hours):
            spin.setRange(0, 1e9)
            spin.setDecimals(6)
            spin.setSuffix(" sa")
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self._queue_filter)
        filters.addWidget(QLabel("T+"))
        filters.addWidget(self.start_hours)
        filters.addWidget(QLabel("–"))
        filters.addWidget(self.end_hours)
        self.accepted_check = QCheckBox("Yalnız kabul")
        self.accepted_check.stateChanged.connect(self._queue_filter)
        filters.addWidget(self.accepted_check)
        filters.addStretch()
        filter_container.addLayout(filters)
        root.addWidget(self.filter_panel)
        self.filter_panel.hide()
        self.filters_toggle.toggled.connect(self.filter_panel.setVisible)
        self.result_overview = QLabel()
        self.result_overview.setTextFormat(Qt.PlainText)
        self.result_overview.setWordWrap(True)
        self.result_overview.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.result_overview.setAccessibleName("Koşu sonucu ve tanılama özeti")
        self.result_overview.setProperty("role", "summary")
        self.result_overview.hide()
        root.addWidget(self.result_overview)
        self.splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.orbit = StoredOrbit()
        self.view_tabs = QTabWidget()
        graph_page = QWidget()
        graph_layout = QVBoxLayout(graph_page)
        graph_layout.setContentsMargins(0, 12, 0, 0)
        graph_layout.addWidget(self.orbit, 2)
        charts = QWidget()
        chart_layout = QHBoxLayout(charts)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        self._chart_layout = chart_layout
        self.charts = [TrackingChart("position_error_m", "Konum farkı", "m", "arc_end_position_error_m"),
            TrackingChart("velocity_error_mps", "Hız farkı", "m/s", "arc_end_velocity_error_mps"),
            TrackingChart("normalized_nis", "Normalize NIS", "boyutsuz")]
        for chart in self.charts:
            chart.selected.connect(self.select_index)
            chart.zoomRequested.connect(self.set_time_range)
            chart.stepRequested.connect(self.step_selection)
            chart_layout.addWidget(chart)
        graph_layout.addWidget(charts, 1)
        timeline = QHBoxLayout()
        for label, delta in (("‹", -1), ("›", 1)):
            btn = QPushButton(label)
            btn.setAccessibleName("Önceki kayıt" if delta < 0 else "Sonraki kayıt")
            btn.clicked.connect(lambda checked=False, d=delta: self.step_selection(d))
            timeline.addWidget(btn)
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.valueChanged.connect(self._slider_selection)
        timeline.addWidget(self.time_slider, 1)
        self.time_label = QLabel("Seçim yok")
        timeline.addWidget(self.time_label)
        graph_layout.addLayout(timeline)
        self.table = QTableView()
        self.table_model = TrackingTableModel(self)
        self.table.setModel(self.table_model)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QTableView.SingleSelection)
        self.table.setShowGrid(False)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.selectionModel().currentRowChanged.connect(self._table_selection)
        self.view_tabs.addTab(graph_page, "Grafikler")
        self.view_tabs.addTab(self.table, "Kayıtlar")
        self.view_tabs.currentChanged.connect(self._sync_chart_tools)
        left_layout.addWidget(self.view_tabs)
        plot_scroll = QScrollArea()
        plot_scroll.setWidgetResizable(True)
        plot_scroll.setMinimumWidth(360)
        plot_scroll.setWidget(left)
        self.splitter.addWidget(plot_scroll)
        right = QTabWidget()
        self.inspector_tabs = right
        self.run_details = RunDetailsPanel()
        right.addTab(self.run_details, "Koşu")
        self.selected_page = QWidget()
        inspector = QVBoxLayout(self.selected_page)
        self.selection_value = QLabel("—")
        self.selection_value.setObjectName("selectionValue")
        self.selection_detail = QLabel("Bir arşiv kaydı seçin.")
        self.selection_detail.setWordWrap(True)
        self.selection_detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        inspector.addWidget(QLabel("Seçili kaydın konum hatası"))
        inspector.addWidget(self.selection_value)
        inspector.addWidget(self.selection_detail)
        inspector.addStretch()
        right.addTab(self.selected_page, "Seçim")
        self.source_text = QPlainTextEdit()
        self.source_text.setReadOnly(True)
        right.addTab(self.source_text, "Teknik")
        right.setMinimumWidth(270)
        right.setMaximumWidth(480)
        self.splitter.addWidget(right)
        self.splitter.setSizes([1100, 300])
        self.details_toggle.toggled.connect(right.setVisible)
        right.hide()
        root.addWidget(self.splitter, 1)
        self.status_label = QLabel("Hazır · sonuç dosyaları salt okunur")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)
        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(100)
        self._filter_timer.timeout.connect(self.apply_filters)
        self._set_busy(False)

    def _choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Sonuç klasörü seç")
        if folder:
            self.open_folder(Path(folder))

    def open_folder(self, folder):
        if self.result is None:
            self.path_label.setText(str(Path(folder).resolve()))
        self._start_worker("discover", Path(folder))

    def open_result(self, path):
        if self.result is None:
            self.path_label.setText(str(Path(path).resolve()))
        self._start_worker("load", Path(path))

    def _retry_load(self):
        if self._last_request:
            self._start_worker(*self._last_request)

    def _restore_loaded_choice(self):
        if not self.result:
            return
        path = self.result.sources[0]["path"]
        self.run_combo.blockSignals(True)
        index = self.run_combo.findData(path)
        if index < 0:
            self.run_combo.addItem(self.result.name, path)
            index = self.run_combo.count()-1
        self.run_combo.setCurrentIndex(index)
        self.run_combo.blockSignals(False)

    def _show_recovery(self, text):
        self.recovery_message.setText(text)
        self.recovery_message.show()
        self.retry_button.setVisible(self._last_request is not None)
        self._restore_loaded_choice()

    def _start_worker(self, kind, path):
        self._last_request = (kind, path)
        self.recovery_message.hide()
        self.retry_button.hide()
        for worker in self._workers:
            worker.requestInterruption()
        self._generation += 1
        worker = _ResultWorker(self._generation, kind, path, self)
        self._workers.add(worker)
        worker.ready.connect(self._receive)
        worker.failed.connect(self._failed)
        worker.finished.connect(lambda w=worker: self._retire(w))
        self._set_busy(True)
        self.status_label.setText(("Sonuç klasörü taranıyor: " if kind == "discover" else "Sonuç dosyaları arka planda okunuyor: ") + path.name)
        worker.start()

    def _retire(self, worker):
        self._workers.discard(worker)
        worker.deleteLater()

    def _set_busy(self, busy):
        self._busy = busy
        self.cancel_button.setEnabled(busy)
        self.cancel_button.setVisible(busy)
        self.run_combo.setEnabled(not busy)
        for control in (self.export_button, self.scale_button, self.reset_button,
                        self.station_combo, self.arc_combo, self.start_hours, self.end_hours, self.accepted_check):
            control.setEnabled(self.result is not None and not busy)
        self.time_slider.setEnabled(bool(self.visible_indices) and not busy)
        if self.result and self.result.metadata.get("record_kind") == "arc_solution":
            self.accepted_check.setEnabled(False)
            self.station_combo.setEnabled(False)

    def cancel_load(self):
        self._generation += 1
        for worker in self._workers:
            worker.requestInterruption()
        self._set_busy(False)
        message = "Yükleme iptal edildi. " + ("Ekrandaki önceki sonuç korunuyor. " if self.result else "")
        message += "Yeniden dene ile aynı kaynağı yükleyin veya başka bir sonuç klasörü seçin."
        self.status_label.setText(message)
        self._show_recovery(message)

    def _receive(self, generation, kind, payload):
        if self._closing or generation != self._generation:
            return
        if kind == "discover":
            self.runs = payload
            self.run_combo.blockSignals(True)
            self.run_combo.clear()
            for run in payload:
                self.run_combo.addItem(run.name, str(run.updates_path))
            self.run_combo.blockSignals(False)
            if not payload:
                self._set_busy(False)
                self.status_label.setText("Bu klasörde UKF güncellemesi veya yay çözümü bulunamadı. Başka bir sonuç klasörü seçin.")
            elif len(payload) == 1:
                self.open_result(payload[0].updates_path)
            else:
                self._set_busy(False)
                self.run_combo.setCurrentIndex(-1)
                self.status_label.setText(f"{len(payload)} koşu bulundu. İnceleyeceğiniz koşuyu seçin.")
            return
        result, details = payload
        self._accept_result(result, details)

    def _choose_run(self, index):
        if index >= 0 and self.run_combo.itemData(index):
            self.open_result(Path(self.run_combo.itemData(index)))

    def _failed(self, generation, message):
        if generation == self._generation and not self._closing:
            self._set_busy(False)
            text = "Sonuç açılamadı: " + message + "\n"
            text += ("Ekrandaki önceki sonuç korunuyor. " if self.result else "")
            text += "Dosyaları geri getirdikten sonra Yeniden dene'ye basın veya başka bir sonuç klasörü seçin."
            self.status_label.setText(text)
            self._show_recovery(text)
            self.loadFailed.emit(message)

    def _accept_result(self, result, details=None):
        self.loaded_details = details or {}
        self.recovery_message.hide()
        self.retry_button.hide()
        self.result = result
        details = details or {"notes": result.notes, "status": "Ayrıntılar bu yükleme yolunda okunmadı"}
        self.run_details.show_details(result.name, result.metadata, details, result.sources, len(result.updates))
        self.result_overview.setText("\n".join(result_summary(result, details)))
        self.result_overview.show()
        source_path = result.sources[0]["path"]
        self.path_label.setText(source_path)
        self.run_combo.blockSignals(True)
        run_index = self.run_combo.findData(source_path)
        if run_index < 0:
            self.run_combo.addItem(result.name, source_path)
            run_index = self.run_combo.count() - 1
        self.run_combo.setCurrentIndex(run_index)
        self.run_combo.blockSignals(False)
        self.selected_index = 0
        self.orbit.set_result(result)
        self.orbit.setMinimumHeight(230 if result.states else 135)
        self.orbit.setMaximumHeight(16777215 if result.states else 135)
        for combo, key in ((self.station_combo, "station_id"), (self.arc_combo, "arc_id")):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("Tümü", None)
            for value in sorted({row[key] for row in result.updates if row[key] is not None}):
                combo.addItem(str(value), value)
            combo.blockSignals(False)
        start, end = self._data_time_range()
        for spin, t in ((self.start_hours, start), (self.end_hours, end)):
            spin.blockSignals(True)
            spin.setValue(t / 3600)
            spin.blockSignals(False)
        self.accepted_check.blockSignals(True)
        self.accepted_check.setChecked(False)
        self.accepted_check.blockSignals(False)
        self.source_text.setPlainText(json.dumps({"name": result.name, "metadata": result.metadata,
            "sources": result.sources, "notes": result.notes}, ensure_ascii=False, indent=2))
        self._set_busy(False)
        batch = result.metadata.get("record_kind") == "arc_solution"
        self.station_combo.setEnabled(not batch)
        self.accepted_check.setEnabled(not batch)
        self.table_model.COLUMNS = (("update_id", "Kayıt"), ("t_s", "T+ / s"), ("arc_id", "Yay"),
            ("observed_station_ids", "İstasyonlar / 0 bazlı"), ("position_error_m", "Konum / m"),
            ("velocity_error_mps", "Hız / m/s"), ("normalized_nis", "Yay ort. NIS"), ("operational_success", "Op. başarı")) if batch else TrackingTableModel.COLUMNS
        self.charts[2].title = "Yay ort. normalize NIS" if batch else "Normalize NIS"
        self.apply_filters()
        self.resultLoaded.emit(result)

    def _queue_filter(self, *args):
        if self.result:
            self._filter_timer.start()

    def apply_filters(self):
        if not self.result:
            return
        started = time.perf_counter()
        self.visible_indices = result_selection(self.result.updates,
            station=self.station_combo.currentData(), arc=self.arc_combo.currentData(),
            start=self.start_hours.value() * 3600 - .002, end=self.end_hours.value() * 3600 + .002,
            accepted_only=self.accepted_check.isChecked())
        if self.selected_index not in self.visible_indices:
            self.selected_index = self.visible_indices[0] if self.visible_indices else None
        self.filters_toggle.setText(
            f"Filtreler · {len(self.visible_indices)}/{len(self.result.updates)}"
            if len(self.visible_indices) != len(self.result.updates) else "Filtreler")
        self.table_model.set_rows(self.result, self.visible_indices)
        self._row_positions = {index: row for row, index in enumerate(self.visible_indices)}
        self.time_slider.blockSignals(True)
        self.time_slider.setRange(0, max(0, len(self.visible_indices) - 1))
        self.time_slider.blockSignals(False)
        self.time_slider.setEnabled(bool(self.visible_indices) and not self._busy)
        self.select_index(self.selected_index)
        self.view_ms = (time.perf_counter() - started) * 1000
        states = f"{len(self.result.states)} durum" if self.result.states else "Durum vektörleri yok"
        kind = "yay çözümü" if self.result.metadata.get("record_kind") == "arc_solution" else "güncelleme"
        bls = f" · {len(self.result.arcs)} BLS yay sonu" if kind == "güncelleme" else ""
        self.status_label.setText(f"{len(self.visible_indices)} / {len(self.result.updates)} {kind}{bls} · {states}")

    def select_index(self, index):
        if index is not None and index not in self._row_positions:
            return
        self.selected_index = index
        row = self.result.updates[index] if index is not None and self.result else None
        self.selection_value.setText(f"{row['position_error_m']:.2f} m" if row else "—")
        self.time_label.setText(elapsed(row["t_s"]) if row else "Seçim yok")
        if row:
            nis = "Bilinmiyor" if row["normalized_nis"] is None else f"{row['normalized_nis']:.3g}"
            accept = "Bilinmiyor" if row["accepted"] is None else "Kabul" if row["accepted"] else "Red"
            arc = next((a for a in self.result.arcs if a["arc_id"] == row["arc_id"]), None)
            detail = f"{elapsed(row['t_s'])}\n\nHız farkı: {row['velocity_error_mps']:.5f} m/s\nNormalize NIS: {nis}\nGüncelleme: {row['update_id']}\nYay: {row['arc_id']}\nİstasyon kimliği: {row['station_id']}\nDurum: {accept}"
            if self.result.metadata.get("record_kind") == "arc_solution":
                success = "Bilinmiyor" if row["operational_success"] is None else "Evet" if row["operational_success"] else "Hayır"
                estimator = self.result.metadata.get("estimator_type", "Bilinmiyor")
                epoch_scope = "Son gözlem epoku" if estimator == "ukf" else "Yay başlangıç epoku"
                detail = f"{elapsed(row['t_s'])} · {epoch_scope}\n\nÇözücü: {estimator}\nHız farkı: {row['velocity_error_mps']:.5f} m/s\nYay ort. normalize NIS: {nis}\nYay: {row['arc_id']}\nİstasyonlar (0 bazlı): {row['observed_station_ids']}\nOperasyonel başarı: {success}\nDuruş: {row['stop_reason']}\n\nYörünge: referans (truth) geçmişi. İşaret: kayıtlı tahmin. Bu veri tek tek UKF güncellemelerini içermez."
            if arc:
                success = "Bilinmiyor" if arc["operational_success"] is None else "Evet" if arc["operational_success"] else "Hayır"
                detail += f"\n\nBLS yay sonu: {arc['arc_end_position_error_m']:.2f} m\nBLS epoku: {elapsed(arc['end_s'])}\nOperasyonel başarı: {success}\nDuruş: {arc['stop_reason']}\n\nBLS yay sonuna, UKF seçili epoka aittir."
            self.selection_detail.setText(detail)
        else:
            self.selection_detail.setText("Bu filtre ve zaman aralığında kayıt yok. Filtreleri değiştirin veya Tüm zaman düğmesini kullanın.")
        self.orbit.selected_t = row["t_s"] if row else None
        self.orbit.update()
        lo, hi = self.start_hours.value() * 3600, self.end_hours.value() * 3600
        for chart in self.charts:
            chart.set_data(self.result, self.visible_indices, (lo, max(lo + 1, hi)), index)
        if row:
            position = self._row_positions[index]
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(position)
            self.time_slider.blockSignals(False)
            self._selection_lock = True
            self.table.selectRow(position)
            self._selection_lock = False

    def _table_selection(self, current, previous):
        if not self._selection_lock and current.isValid():
            self.select_index(self.visible_indices[current.row()])

    def _slider_selection(self, position):
        if 0 <= position < len(self.visible_indices):
            self.select_index(self.visible_indices[position])

    def step_selection(self, delta):
        if self.visible_indices:
            position = self._row_positions.get(self.selected_index, 0)
            self.select_index(self.visible_indices[max(0, min(len(self.visible_indices) - 1, position + delta))])

    def set_time_range(self, lo, hi):
        if not self.result or not all(isinstance(n, (int, float)) and math.isfinite(n) for n in (lo, hi)):
            return
        start, end = self._data_time_range()
        lo, hi = max(start, lo), min(end, hi)
        if hi - lo < 1:
            return
        for spin, value in ((self.start_hours, lo), (self.end_hours, hi)):
            spin.blockSignals(True)
            spin.setValue(value / 3600)
            spin.blockSignals(False)
        self.apply_filters()

    def reset_time(self):
        if self.result:
            self.set_time_range(*self._data_time_range())

    def _data_time_range(self):
        if self.result.metadata.get("record_kind") == "arc_solution":
            return min(r["start_s"] for r in self.result.updates), max(r["end_s"] for r in self.result.updates)
        return self.result.updates[0]["t_s"], self.result.updates[-1]["t_s"]

    def _toggle_scale(self):
        linear = not self.charts[0].linear
        self.scale_button.setText("Ölçek: doğrusal" if linear else "Ölçek: otomatik")
        for chart in self.charts:
            chart.linear = linear
            chart.update()

    def _export(self):
        if not self.result:
            return
        name, _ = QFileDialog.getSaveFileName(self, "Görünür kayıtları dışa aktar", self.result.name + "_filtered.csv", "CSV (*.csv)")
        if name:
            try:
                self.export_visible(Path(name))
                self.status_label.setText("Filtrelenmiş CSV ve kaynak JSON kaydedildi.")
            except OSError as exc:
                self.status_label.setText("Dışa aktarma başarısız: " + str(exc))

    def export_visible(self, path):
        path = Path(path).resolve()
        sidecar = path.with_suffix(".source.json")
        if any(path == Path(s["path"]).resolve() or sidecar == Path(s["path"]).resolve() for s in self.result.sources):
            raise OSError("Kaynak arşiv dosyasının üzerine dışa aktarılamaz.")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.result.updates[0]))
            writer.writeheader()
            writer.writerows(self.result.updates[i] for i in self.visible_indices)
        info = {"schema": "lunar-od-view-export-v1", "source_metadata": self.result.metadata,
            "sources": self.result.sources, "count": len(self.visible_indices),
            "filters": {"station_id": self.station_combo.currentData(), "arc_id": self.arc_combo.currentData(),
                "start_s": self.start_hours.value() * 3600, "end_s": self.end_hours.value() * 3600,
                "accepted_only": self.accepted_check.isChecked()}, "notes": self.result.notes}
        sidecar.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    def shutdown(self):
        self._closing = True
        self._filter_timer.stop()
        for worker in list(self._workers):
            worker.requestInterruption()
        for worker in list(self._workers):
            worker.wait(3000)

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)
