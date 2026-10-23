"""Compare validated stored runs without running an estimator or propagation."""
from __future__ import annotations
from services.result_summary import comparison_summary

import json
from functools import partial
from pathlib import Path

from PyQt5.QtCore import Qt, QModelIndex, QAbstractTableModel, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QLineEdit, QComboBox, QCheckBox, QTableView, QTabWidget,
    QPlainTextEdit, QFileDialog, QHeaderView, QSplitter, QFrame, QScrollArea)

from services.run_comparison import load_run_comparison, export_comparison
from services.comparison_session import load_comparison_session, TABS, view_tab
from services.project_paths import RESULTS_DIR
from styles.artifact_workspace import ARTIFACT_STYLE
from widgets.comparison_chart import ComparisonChart
from workers.artifact_task import ArtifactJobs


def display(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple, bool)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


class RowsModel(QAbstractTableModel):
    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.columns, self.rows = columns, ()

    def replace(self, rows):
        self.beginResetModel()
        self.rows = rows
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.columns)

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role in (Qt.DisplayRole, Qt.ToolTipRole):
            return display(self.rows[index.row()].get(self.columns[index.column()][0]))

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.columns[section][1]


class PairModel(QAbstractTableModel):
    COLUMNS = ("T+ / s", "Yay", "İstasyonlar", "A kayıt", "B kayıt", "A konum / m", "B konum / m", "Δ B−A / m",
        "A hız / m/s", "B hız / m/s", "Δ B−A / m/s")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.comparison = None

    def replace(self, comparison):
        self.beginResetModel()
        self.comparison = comparison
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() or not self.comparison else len(self.comparison.pairs)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.COLUMNS)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.ToolTipRole):
            return None
        i, j = self.comparison.pairs[index.row()]
        a, b = self.comparison.left.updates[i], self.comparison.right.updates[j]
        values = (a["t_s"], a["arc_id"], a.get("observed_station_ids", a["station_id"]), a["update_id"], b["update_id"],
            a["position_error_m"], b["position_error_m"], b["position_error_m"] - a["position_error_m"],
            a["velocity_error_mps"], b["velocity_error_mps"], b["velocity_error_mps"] - a["velocity_error_mps"])
        return display(values[index.column()])

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.COLUMNS[section]


def _table(model):
    view = QTableView()
    view.setModel(model)
    view.setSelectionBehavior(QTableView.SelectRows)
    view.setSelectionMode(QTableView.SingleSelection)
    view.setShowGrid(False)
    view.verticalHeader().hide()
    view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
    view.horizontalHeader().setStretchLastSection(True)
    return view


class RecordedComparisonController(ArtifactJobs, QWidget):
    comparisonLoaded = pyqtSignal(object)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_jobs()
        self.comparison = None
        self._task = "load"
        self._pair_lookup = ({}, {})
        self.setObjectName("artifactWorkspace")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(ARTIFACT_STYLE)
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 18)
        root.setSpacing(11)
        head = QHBoxLayout()
        self.back_button = QPushButton("← Arşive dön")
        self.back_button.setToolTip("Koşu listesine dönün; başka iki koşu seçerek yeniden karşılaştırın.")
        self.back_button.clicked.connect(lambda: self.window().navigate("archive") if hasattr(self.window(), "navigate") else None)
        head.addWidget(self.back_button)
        head.addStretch()
        self.open_report_button = QPushButton("Kayıtlı karşılaştırmayı aç…")
        self.open_report_button.clicked.connect(self._choose_report)
        head.addWidget(self.open_report_button)
        self.export_button = QPushButton("Raporu aktar…")
        self.export_button.setToolTip("Karşılaştırma raporunu (JSON) ve eşleşen kayıtları (CSV) kaydet")
        self.export_button.clicked.connect(self._export)
        head.addWidget(self.export_button)
        root.addLayout(head)
        selectors = QGridLayout()
        self.paths = []
        for side in (0, 1):
            selectors.addWidget(QLabel("AB"[side]), side, 0)
            edit = QLineEdit()
            edit.setPlaceholderText("Gerçek *_arc_solutions.csv veya *_ukf_updates.csv kaynağı")
            edit.setAccessibleName(f"{'AB'[side]} sonuç dosyası")
            edit.textChanged.connect(self._inputs_changed)
            selectors.addWidget(edit, side, 1)
            self.paths.append(edit)
            button = QPushButton("Dosya seç")
            button.clicked.connect(lambda checked=False, s=side: self._choose_file(s))
            selectors.addWidget(button, side, 2)
        root.addLayout(selectors)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("İncelenen değişken"))
        self.variable = QComboBox()
        for key, label in (("none", "Genel / değişken belirtilmedi"), ("estimator_type", "Çözücü"),
            ("start_mode", "Başlangıç modu"), ("noise", "Gürültü"), ("network", "İstasyon ağı"),
            ("measurement_type", "Ölçüm türü"), ("max_iter", "İterasyon sınırı"),
            ("sample_step_s", "Örnekleme adımı"), ("ukf_alpha", "UKF α"), ("rtol", "İntegrasyon rtol")):
            self.variable.addItem(label, key)
        self.variable.currentIndexChanged.connect(self._inputs_changed)
        controls.addWidget(self.variable)
        self.load_button = QPushButton("Karşılaştır")
        self.load_button.setProperty("primary", True)
        self.load_button.clicked.connect(self.load)
        controls.addWidget(self.load_button)
        swap = QPushButton("A ↔ B")
        swap.clicked.connect(self.swap)
        controls.addWidget(swap)
        self.cancel_button = QPushButton("İptal et")
        self.cancel_button.clicked.connect(self.cancel_load)
        controls.addWidget(self.cancel_button)
        controls.addStretch()
        root.addLayout(controls)
        self.summary_labels = []
        summaries = QHBoxLayout()
        for side in (0, 1):
            frame = QFrame()
            frame.setObjectName("runSummary")
            frame.setStyleSheet("")
            box = QVBoxLayout(frame)
            label = QLabel(f"{'AB'[side]}  /  Koşu seçilmedi")
            label.setWordWrap(True)
            label.setMinimumHeight(label.fontMetrics().lineSpacing() * 3 + 10)
            label.setTextFormat(Qt.PlainText)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            box.addWidget(label)
            self.summary_labels.append(label)
            open_btn = QPushButton(f"{'AB'[side]} koşusunu Sonuçlarda aç")
            open_btn.clicked.connect(lambda checked=False, s=side: self.open_source(s))
            box.addWidget(open_btn, 0, Qt.AlignLeft)
            summaries.addWidget(frame, 1)
        root.addLayout(summaries)
        self.compatibility_label = QLabel("Ortak epoklar doğrulanınca eşleşme farkları hesaplanır. Eksik bilgiler bilinmiyor olarak kalır.")
        self.compatibility_label.setWordWrap(True)
        root.addWidget(self.compatibility_label)
        self.caution_label = QLabel("Betimleyici inceleme · ölçüm gerçekleşimi ve başlangıç kovaryansı eşitliği henüz kanıtlanmaz · algoritma sıralaması yapılmaz")
        self.caution_label.setWordWrap(True)
        root.addWidget(self.caution_label)
        self.effect_summary = QLabel()
        self.effect_summary.setTextFormat(Qt.PlainText)
        self.effect_summary.setWordWrap(True)
        self.effect_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.effect_summary.hide()
        root.addWidget(self.effect_summary)
        self.tabs = QTabWidget()
        graphs = QWidget()
        graphs_layout = QVBoxLayout(graphs)
        graphs_layout.setContentsMargins(12, 8, 12, 8)
        graph_controls = QHBoxLayout()
        self.paired_check = QCheckBox("Grafikte yalnız eşleşen kayıtlar")
        self.paired_check.toggled.connect(self._refresh_charts)
        graph_controls.addWidget(self.paired_check)
        self.linear_check = QCheckBox("Doğrusal ölçek")
        self.linear_check.toggled.connect(self._scale_changed)
        graph_controls.addWidget(self.linear_check)
        reset = QPushButton("Tüm zaman")
        reset.clicked.connect(self.reset_range)
        graph_controls.addWidget(reset)
        graph_controls.addStretch()
        graphs_layout.addLayout(graph_controls)
        self.charts = []
        for key, label, unit in (("position_error_m", "Konum hatası", "m"), ("velocity_error_mps", "Hız hatası", "m/s"),
            ("normalized_nis", "Normalize NIS", "boyutsuz")):
            chart = ComparisonChart(key, label, unit)
            chart.selected.connect(self._chart_selected)
            chart.zoomRequested.connect(self.set_range)
            self.charts.append(chart)
            graphs_layout.addWidget(chart, 1)
        graph_scroll = QScrollArea()
        graph_scroll.setWidgetResizable(True)
        graph_scroll.setFrameShape(QFrame.NoFrame)
        graph_scroll.setWidget(graphs)
        self.tabs.addTab(graph_scroll, "Grafikler")
        self.pair_model = PairModel(self)
        self.pair_table = _table(self.pair_model)
        self.pair_table.selectionModel().currentRowChanged.connect(self._pair_selected)
        self.tabs.addTab(self.pair_table, "Eşleşen kayıtlar")
        self.stats_model = RowsModel((("pool", "Kayıt havuzu"), ("metric", "Alan / birim"), ("side", "Koşu"),
            ("N", "N"), ("median", "Medyan"), ("p95", "p95"), ("max", "Maksimum"), ("rms", "RMS")), self)
        self.tabs.addTab(_table(self.stats_model), "İstatistikler")
        self.difference_model = RowsModel((("field", "Ayar / kaynak alanı"), ("A", "A"), ("B", "B"), ("role", "Değişken")), self)
        diff_table = _table(self.difference_model)
        diff_table.setColumnWidth(0, 370)
        diff_table.setColumnWidth(1, 290)
        diff_table.setColumnWidth(2, 290)
        # Setting differences and the files they come from are read together.
        provenance = QSplitter(Qt.Horizontal)
        self.provenance = []
        for side in (0, 1):
            editor = QPlainTextEdit()
            editor.setReadOnly(True)
            editor.setPlaceholderText(f"{'AB'[side]} / dosya yolları, SHA-256 ve kaydedilmiş ayarlar")
            provenance.addWidget(editor)
            self.provenance.append(editor)
        settings_and_sources = QSplitter(Qt.Vertical)
        settings_and_sources.addWidget(diff_table)
        settings_and_sources.addWidget(provenance)
        settings_and_sources.setSizes([320, 220])
        self.tabs.addTab(settings_and_sources, "Ayarlar ve kaynaklar")
        self.notes = QPlainTextEdit()
        self.notes.setReadOnly(True)
        self.tabs.addTab(self.notes, "Yorum sınırları")
        root.addWidget(self.tabs, 1)
        self.selection_label = QLabel("Eşleşme seçilmedi · Δ = B−A, kaydedilmiş hata normlarının farkı")
        self.selection_label.setWordWrap(True)
        root.addWidget(self.selection_label)
        self.status_label = QLabel("Dosya yükleme, doğrulama ve rapor aktarımı arka planda çalışır.")
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(Qt.PlainText)
        root.insertWidget(1, self.status_label)
        self._busy(False)

    def on_shown(self):
        pass

    def _inputs_changed(self, *args):
        self.load_button.setEnabled(all(p.text().strip() for p in self.paths))
        if self.comparison:
            self.status_label.setText("Seçim değişti. Ekranda son yüklenen A/B kaynakları duruyor; güncellemek için Karşılaştır'a basın.")

    def _choose_file(self, side):
        path, _ = QFileDialog.getOpenFileName(self, f"{'AB'[side]} sonuç kaynağı", str(RESULTS_DIR), "Kaydedilmiş sonuçlar (*.csv)")
        if path:
            self.paths[side].setText(path)

    def set_files(self, left, right, auto_load=True):
        self.paths[0].setText(str(left))
        self.paths[1].setText(str(right))
        if auto_load:
            self.load()

    def swap(self):
        a, b = (p.text() for p in self.paths)
        self.set_files(b, a, auto_load=False)
        if a and b:
            self.load()

    def load(self):
        if not all(p.text().strip() for p in self.paths):
            return
        self._task = "load"
        self._busy(True)
        self.status_label.setText("A ve B yükleniyor; kaynaklar ve tam epok eşleşmeleri doğrulanıyor…")
        self.start_job(load_run_comparison, (*[Path(p.text().strip()) for p in self.paths], self.variable.currentData()))

    def _busy(self, busy):
        self.cancel_button.setEnabled(busy)
        self.export_button.setEnabled(self.comparison is not None and not busy)
        self.load_button.setEnabled(all(p.text().strip() for p in self.paths))

    def cancel_load(self):
        self.cancel_jobs()
        self._busy(False)
        self.status_label.setText("İşlem iptal edildi. " + ("Son yüklenen karşılaştırma korunuyor. " if self.comparison else "") + "Yüklemeyi yinelemek için Karşılaştır veya Kayıtlı karşılaştırma aç düğmesini kullanın.")

    def _job_ready(self, generation, result):
        if self._closing or generation != self._generation:
            return
        if self._task == "export":
            self.status_label.setText("Rapor kaydedildi: " + str(result[0]) + (" · " + str(result[1]) if result[1] else " · eşleşme CSV'si yok (N=0)"))
        elif self._task == "session":
            comparison, view, path = result
            self.set_files(comparison.left.sources[0]["path"], comparison.right.sources[0]["path"], auto_load=False)
            index = self.variable.findData(comparison.variable)
            if index < 0:
                self.variable.addItem(comparison.variable, comparison.variable)
                index = self.variable.count() - 1
            self.variable.setCurrentIndex(index)
            self.accept_comparison(comparison)
            self.restore_view(view)
            self.status_label.setText(f"Karşılaştırma geri açıldı; kaynak hash'leri doğrulandı ve değerler yeniden hesaplandı: {path.name}")
        else:
            self.accept_comparison(result)
            self.status_label.setText("Kaynaklar salt okunur yüklendi. Ayarlar ve kaynaklar ile Yorum sınırları sekmelerini inceleyin.")
        self._busy(False)

    def _job_failed(self, generation, message):
        if not self._closing and generation == self._generation:
            self._busy(False)
            self.status_label.setText("İşlem başarısız: " + message + "\n" + ("Son karşılaştırma korunuyor. " if self.comparison else "") + "Kaynak dosyaları ve hedef klasörü kontrol edip ilgili yükleme/aktarma düğmesiyle yeniden deneyin.")

    def accept_comparison(self, result):
        self.comparison = result
        self.effect_summary.setText(comparison_summary(result))
        self.effect_summary.show()
        self._pair_lookup = tuple({p[s]: n for n, p in enumerate(result.pairs)} for s in (0, 1))
        self.pair_model.replace(result)
        self.difference_model.replace(result.differences)
        stats = tuple({"pool": "Tüm kaynak" if pool == "all" else "Tam epok eşleşmesi", "metric": metric, "side": side, **values}
            for pool, fields in result.metrics.items() for metric, sides in fields.items() for side, values in sides.items())
        self.stats_model.replace(stats)
        for side, source in enumerate((result.left, result.right)):
            median = result.metrics["all"]["position_error_m"]["AB"[side]]["median"]
            batch = source.metadata.get("record_kind") == "arc_solution"
            successes = sum(r.get("operational_success") is True for r in source.updates)
            known = sum(r.get("operational_success") is not None for r in source.updates)
            unknown = len(source.updates) - known
            success_label = (f"Operasyonel başarı {successes}/{known} yay" if known else "Operasyonel başarı bilinmiyor") + (f" · {unknown} yay bilinmiyor" if unknown else "")
            estimator = source.metadata.get("estimator_type") or ("Bilinmiyor" if batch else "UKF")
            self.summary_labels[side].setText(f"{'AB'[side]}  /  {source.name}\n{estimator.upper()} · N={len(source.updates)} · Seed {display(source.metadata.get('seed'))}"
                f"\nKonum medyanı {display(median)} m · " + (success_label if batch else "UKF güncelleme kayıtları"))
            self.summary_labels[side].setToolTip(source.sources[0]["path"])
            self.provenance[side].setPlainText(json.dumps({"name": source.name, "metadata": source.metadata, "sources": source.sources,
                "snapshot": result.snapshots[side]}, ensure_ascii=False, indent=2))
        n = len(result.pairs)
        self.compatibility_label.setText(("Ortak zaman/çerçeve doğrulandı" if result.time_compatible else "Ortak zaman/çerçeve doğrulanamadı; fark hesabı kapalı")
            + f" · Tam eşleşme N={n} · Eşleşmeyen A={len(result.left.updates) - n}, B={len(result.right.updates) - n} · Ayar farkı {len(result.differences)}")
        self.notes.setPlainText("\n\n".join(result.notes))
        self.caution_label.setText("Betimleyici inceleme · algoritma sıralaması yapılmaz · " +
            (f"Eşleşen N={n}: küçük örneklem; p95 genellemesi yapılmaz. " if n < 10 else "") + "Ayrıntılar: Yorum sınırları.")
        if not any(row.get("normalized_nis") is not None for source in (result.left, result.right) for row in source.updates):
            self.caution_label.setText(self.caution_label.text() + " NIS kaynaklarda kayıtlı değil.")
        self.tabs.setTabText(1, f"Eşleşen kayıtlar ({n})")
        self.tabs.setTabText(3, f"Ayarlar ve kaynaklar ({len(result.differences)} fark)")
        self.paired_check.setEnabled(result.time_compatible and bool(result.pairs))
        if not self.paired_check.isEnabled():
            self.paired_check.setChecked(False)
        self._refresh_charts()
        if result.pairs:
            self.pair_table.setCurrentIndex(self.pair_model.index(0, 0))
        else:
            self.selection_label.setText("Ortak kayıt yok · kayıtlar kendi eksenlerinde incelenebilir · başka epoka taşıma yapılmaz")

        self.comparisonLoaded.emit(result)

    def _refresh_charts(self, *args):
        for chart in self.charts:
            chart.selected_pair = None
            chart.set_comparison(self.comparison, self.paired_check.isChecked())
            chart.setVisible(chart.key != "normalized_nis" or any(chart._series))
        if self.pair_table.currentIndex().isValid():
            self._pair_selected(self.pair_table.currentIndex())

    def _scale_changed(self, linear):
        for chart in self.charts:
            chart.linear = linear
            chart.update()

    def reset_range(self):
        for chart in self.charts:
            chart.time_range = None
            chart._cache.clear()
            chart.update()

    def set_range(self, lo, hi):
        if not self.comparison or not self.comparison.time_compatible:
            return
        a, b = self.charts[0].domain()
        span = min(b - a, max((b - a) / 100000, hi - lo))
        lo = max(a, min(b - span, lo))
        for chart in self.charts:
            if chart.shared_axis():
                chart.set_range(lo, lo + span)

    def _chart_selected(self, side, index):
        n = self._pair_lookup[side].get(index)
        if n is None:
            source = (self.comparison.left, self.comparison.right)[side]
            row = source.updates[index]
            self.selection_label.setText(f"{'AB'[side]} kayıt {row['update_id']} · T+{row['t_s']:.6g} s · bu kaydın tam eşleşmesi yok")
            for chart in self.charts:
                chart.selected_pair = None
                chart.update()
        else:
            self.pair_table.setCurrentIndex(self.pair_model.index(n, 0))
            self._pair_selected(self.pair_model.index(n, 0))

    def _pair_selected(self, index, previous=QModelIndex()):
        if not self.comparison or not index.isValid():
            return
        n = index.row()
        i, j = self.comparison.pairs[n]
        a, b = self.comparison.left.updates[i], self.comparison.right.updates[j]
        self.selection_label.setText(f"T+{a['t_s']:.6g} s · Yay {a['arc_id']} · A kayıt {a['update_id']} / B kayıt {b['update_id']}"
            f" · Δ konum {b['position_error_m'] - a['position_error_m']:.6g} m · Δ hız {b['velocity_error_mps'] - a['velocity_error_mps']:.6g} m/s")
        for chart in self.charts:
            chart.selected_pair = n
            chart.update()

    def open_source(self, side):
        if self.comparison and hasattr(self.window(), "open_in_result_console"):
            source = (self.comparison.left, self.comparison.right)[side]
            self.window().open_in_result_console(Path(source.sources[0]["path"]))

    def _export(self):
        if not self.comparison:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Karşılaştırma raporu", "run_comparison.json", "JSON raporu (*.json)")
        if path:
            self.export_to(path)

    def export_to(self, path):
        if self.comparison:
            self._task = "export"
            self._busy(True)
            self.status_label.setText("Kaynak bilgileri ve eşleşme kayıtları aktarılıyor…")
            self.start_job(partial(export_comparison, view=self.capture_view()), (self.comparison, Path(path)))

    def _choose_report(self):
        path, _ = QFileDialog.getOpenFileName(self, "Kaydedilmiş karşılaştırma", str(RESULTS_DIR), "Karşılaştırma JSON (*.json)")
        if path:
            self.open_report(path)

    def open_report(self, path):
        self._task = "session"
        self._busy(True)
        self.status_label.setText("Kaydedilmiş karşılaştırma açılıyor; kaynak hash'leri doğrulanıyor…")
        self.start_job(load_comparison_session, (Path(path),))

    def capture_view(self):
        selected = self.charts[0].selected_pair
        records = None
        if self.comparison and selected is not None:
            i, j = self.comparison.pairs[selected]
            records = [self.comparison.left.updates[i]["update_id"], self.comparison.right.updates[j]["update_id"]]
        return {"schema": "lunar-od-comparison-view-v1", "linear": self.linear_check.isChecked(),
            "paired_only": self.paired_check.isChecked(), "tab": TABS[self.tabs.currentIndex()],
            "time_range": self.charts[0].time_range, "selected_records": records}

    def restore_view(self, view):
        self.linear_check.setChecked(view.get("linear", False))
        self.paired_check.setChecked(view.get("paired_only", False) and self.paired_check.isEnabled())
        self.reset_range()
        if view.get("time_range"):
            self.set_range(*view["time_range"])
        records = view.get("selected_records")
        if records is not None:
            for n, (i, j) in enumerate(self.comparison.pairs):
                if [self.comparison.left.updates[i]["update_id"], self.comparison.right.updates[j]["update_id"]] == records:
                    self.pair_table.setCurrentIndex(self.pair_model.index(n, 0))
                    self._pair_selected(self.pair_model.index(n, 0))
                    break
        else:
            self.pair_table.clearSelection()
            self.pair_table.setCurrentIndex(QModelIndex())
            for chart in self.charts:
                chart.selected_pair = None
                chart.update()
            self.selection_label.setText("Eşleşme seçilmedi · Δ = B−A, kaydedilmiş hata normlarının farkı")
        self.tabs.setCurrentIndex(TABS.index(view_tab(view)))
