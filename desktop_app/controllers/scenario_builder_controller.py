"""Senaryo page controller — configure a run, save/open its settings, start it."""
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import replace, asdict
from decimal import Decimal

from PyQt5.QtWidgets import (
    QWidget, QFileDialog, QMessageBox, QApplication, QLabel, QPushButton, QDoubleSpinBox,
    QCheckBox, QComboBox, QGroupBox, QFormLayout, QVBoxLayout, QHBoxLayout, QBoxLayout, QSizePolicy,
)
from PyQt5.QtCore import Qt
from PyQt5 import uic

from services.project_paths import DESKTOP_APP_DIR, PYTHON_PORT
from models.scenario_model import ScenarioModel, PRESETS
from styles.artifact_workspace import ARTIFACT_STYLE
from styles.theme import C
from styles.workspace_light import set_role

UI_PATH = DESKTOP_APP_DIR / "ui" / "pages" / "scenario_builder_page.ui"

# Readable labels for the stored identifiers; the identifier stays in itemData.
_CHOICES = {
    "arcModeCombo": (("prescribed", "Sabit aralıklı"), ("visibility", "Görünürlüğe göre")),
    "measTypeCombo": (("range_rate", "Menzil, açılar + menzil hızı"), ("position", "Menzil ve açılar")),
    "rrPhysicsCombo": (("geometric_instantaneous", "Geometrik (anlık)"),
                       ("two_way_counted_doppler", "İki yönlü sayılmış Doppler")),
    # The desktop executor supports exactly these two networks (scenario_execution.py).
    "networkCombo": (("multi", "Çoklu · Goldstone, Madrid, Canberra, İTÜ Ayazağa"),
                     ("single", "Tek · Canberra DSN")),
    "estimatorCombo": (("bls_lm", "BLS-LM (toplu)"), ("srif", "SRIF (karekök bilgi)"), ("ukf", "SR-UKF (sıralı)")),
    "startModeCombo": (("cold", "Soğuk · her yay bağımsız"), ("hot", "Sıcak · önceki yayın durumu"),
                       ("formal", "Formal · kovaryans aktarımı"), ("sqrt_formal", "Karekök formal")),
}
_PRESET_LABELS = {
    "Gaussian Baseline": ("Temel karşılaştırma · 1 gün",
                          "1 günlük BLS-LM / SR-UKF karşılaştırması: sabit yaylar, soğuk başlangıç, menzil hızı, çoklu ağ."),
    "28-Day Stability": ("Uzun süre kararlılığı · 28 gün",
                         "Uzun süreli sayısal kararlılık ve kestirici karşılaştırması: 112 sabit yay, formal aktarım."),
    "Sequential Tracking": ("Sıralı izleme · 3 gün",
                            "3 günlük parçalı SPICE görünürlüğüyle SR-UKF sıralı izleme."),
    "Two-Way Doppler": ("İki yönlü Doppler · 1 gün",
                        "Basitleştirilmiş iki yönlü sayılmış Doppler denemesi."),
    "BLS Ablation": ("BLS ölçüm karşılaştırması · 3 gün",
                     "BLS-LM ile yalnız konum ölçümleri ve menzil hızı eklenmiş ölçümlerin karşılaştırması."),
    "28-Day Network Visibility": ("Ağ görünürlüğü · 28 gün",
                                  "28 günlük İTÜ + DSN istasyon görünürlüğü."),
}


def _value(combo):
    """Stored identifier of the selected item (labels are for display only)."""
    data = combo.currentData()
    return data if data is not None else combo.currentText()


class ScenarioBuilderController(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        uic.loadUi(str(UI_PATH), self)
        self.setObjectName("artifactWorkspace")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(ARTIFACT_STYLE)
        self._current_model = ScenarioModel()
        self._updating_from_model = False
        self._archive_origin = None
        self._draft_before_archive = None
        self._setup()
        self._simplify_layout()

    def _setup(self) -> None:
        self.origin_label = QLabel()
        self.origin_label.setWordWrap(True)
        self.origin_label.setTextFormat(Qt.PlainText)
        self.origin_label.setStyleSheet(f"color:{C.TEXT_PRIMARY}; padding:8px; background:{C.BG_PANEL};")
        self.restore_draft_button = QPushButton("Önceki taslağa dön")
        self.restore_draft_button.clicked.connect(self.restore_previous_draft)
        self.previewLayout.insertWidget(0, self.origin_label)
        self.previewLayout.insertWidget(1, self.restore_draft_button)
        self._show_origin()

        self.presetCombo.addItem("Özel ayarlar", None)
        for name in PRESETS:
            self.presetCombo.addItem(_PRESET_LABELS.get(name, (name,))[0], name)
        self.presetCombo.currentIndexChanged.connect(
            lambda _index: self._on_preset_selected(self.presetCombo.currentData()))
        for name, choices in _CHOICES.items():
            combo = getattr(self, name)
            combo.clear()
            for value, label in choices:
                combo.addItem(label, value)

        for widget in [self.nameEdit, self.descEdit, self.outputDirEdit]:
            widget.textChanged.connect(self._update_json)
        for spinbox in [
            self.durationSpin, self.stepSpin, self.arcDurSpin,
            self.arcStrideSpin, self.elevMaskSpin, self.gapThreshSpin,
        ]:
            spinbox.valueChanged.connect(self._update_json)
        self.seedSpin.valueChanged.connect(self._update_json)
        for name in _CHOICES:
            getattr(self, name).currentIndexChanged.connect(self._update_json)
        self.arcModeCombo.currentIndexChanged.connect(self._on_arc_mode_changed)
        self.noiseCheck.stateChanged.connect(self._update_json)

        self._arc_count_label = QLabel("")
        self._arc_count_label.setWordWrap(True)
        set_role(self._arc_count_label, "caption")
        self.arcGroup.layout().addRow("Tahmin", self._arc_count_label)

        self.copyJsonBtn.clicked.connect(self._copy_json)
        self.exportJsonBtn.clicked.connect(self._export_json)
        self.importJsonBtn.clicked.connect(self._import_json)
        self.runScenarioBtn.setProperty("primary", True)
        self.runScenarioBtn.clicked.connect(self._run_scenario)

        # Light time and aberration apply to position (range/az/el) measurements only.
        self.lightTimeCheck = QCheckBox("Tek yönlü ışık zamanı (CN)")
        self.lightTimeCheck.setToolTip("Menzil/azimut/yükseliş için yakınsamış Newton tek yönlü ışık zamanı düzeltmesi.\n"
                                       "Yalnız 'Menzil ve açılar' ölçümünde.")
        self.stellarCheck = QCheckBox("Yıldız sapınçlığı (+S)")
        self.stellarCheck.setToolTip("Işık zamanı çözümünün üstüne yıldız sapınçlığı uygular; ışık zamanı gerekir.")
        self.stellarModelCombo = QComboBox()
        self.stellarModelCombo.addItem("Ay merkezli hız (yerel MCI)", "local_mci")
        self.stellarModelCombo.addItem("Güneş sistemi ağırlık merkezi (SPICE SSB)", "spice_ssb")
        self.stellarModelCombo.setToolTip("+S için gözlemci hızının çerçevesi:\n"
                                          "  SPICE SSB — SPICE CN+S ile aynı (SSB hızı, ~1e-4 rad)\n"
                                          "  yerel MCI — Ay merkezli hız (daha ucuz, ~20 kat küçük)")
        meas_layout = self.measGroup.layout()
        meas_layout.addRow("Işık zamanı", self.lightTimeCheck)
        meas_layout.addRow("Sapınçlık", self.stellarCheck)
        meas_layout.addRow("Sapınçlık çerçevesi", self.stellarModelCombo)
        for _w in (self.lightTimeCheck, self.stellarCheck):
            _w.stateChanged.connect(self._update_json)
            _w.stateChanged.connect(self._on_aberration_changed)
        self.stellarModelCombo.currentIndexChanged.connect(self._update_json)
        self.measTypeCombo.currentIndexChanged.connect(self._on_aberration_changed)

        self._model_to_widgets(self._current_model)
        self._update_json()
        self._on_arc_mode_changed()

    def _simplify_layout(self):
        self.layout().setDirection(QBoxLayout.TopToBottom)
        self.layout().setContentsMargins(24, 12, 24, 16)
        self.layout().setSpacing(12)
        self.formScroll.setMaximumWidth(16777215)
        # A readable measure: fields stop growing past ~880 px on wide windows.
        self.formContents.setMaximumWidth(880)
        form_layout = self.estimatorGroup.parentWidget().layout()
        form_layout.setContentsMargins(0, 0, 20, 0)
        form_layout.setSpacing(10)
        set_role(self.presetDescLabel, "caption")

        # What is measured and by which stations is a primary choice, not an advanced one.
        self.observationGroup = QGroupBox("Ölçüm ve istasyonlar")
        observation_form = QFormLayout(self.observationGroup)
        meas_form = self.measGroup.layout()
        for label, field in ((self.measTypeLabel, self.measTypeCombo),
                             (self.networkLabel, self.networkCombo),
                             (self.noiseLabel, self.noiseCheck)):
            meas_form.takeRow(field)
            observation_form.addRow(label, field)
        form_layout.insertWidget(form_layout.indexOf(self.estimatorGroup), self.observationGroup)
        for group in (self.presetsGroup, self.identityGroup, self.timeGroup, self.observationGroup, self.estimatorGroup):
            group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        self.advanced_form_toggle = QPushButton("Gelişmiş ayarlar: yaylar ve ölçüm modeli  ›")
        self.advanced_form_toggle.setCheckable(True)
        set_role(self.advanced_form_toggle, "disclosure")
        self.advanced_form_toggle.setMinimumHeight(34)
        self.advanced_form_toggle.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.advanced_form = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_form)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        for group in (self.arcGroup, self.measGroup):
            form_layout.removeWidget(group)
            advanced_layout.addWidget(group)
        form_layout.insertWidget(form_layout.count() - 1, self.advanced_form_toggle)
        form_layout.insertWidget(form_layout.count() - 1, self.advanced_form)
        self.advanced_form.hide()
        self.advanced_form_toggle.toggled.connect(self._toggle_advanced)

        self.preview_toggle = QPushButton("JSON önizlemesi")
        self.preview_toggle.setCheckable(True)
        self.preview_toggle.setChecked(False)
        # One action row under the form: start (primary), then save/open settings and the JSON preview.
        self.previewLayout.removeItem(self.actionButtonsLayout)
        for button in (self.runScenarioBtn, self.exportJsonBtn, self.importJsonBtn):
            self.actionButtonsLayout.removeWidget(button)
        actions = QWidget()
        actions.setObjectName("actionBar")
        row = QHBoxLayout(actions)
        row.setContentsMargins(0, 12, 0, 0)
        row.setSpacing(8)
        for button in (self.runScenarioBtn, self.exportJsonBtn, self.importJsonBtn):
            row.addWidget(button)
        row.addStretch(1)
        row.addWidget(self.preview_toggle)
        self.layout().addWidget(actions)
        # Provenance and draft recovery remain visible with the JSON panel closed.
        self.previewLayout.removeWidget(self.origin_label)
        self.previewLayout.removeWidget(self.restore_draft_button)
        self.layout().insertWidget(0, self.restore_draft_button)
        self.layout().insertWidget(0, self.origin_label)
        self.layout().setStretchFactor(self.splitter, 1)
        self.previewPanel.hide()
        self.preview_toggle.toggled.connect(self.previewPanel.setVisible)
        for button in (self.exportJsonBtn, self.importJsonBtn, self.runScenarioBtn, self.preview_toggle):
            button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self._align_form_labels()
        self.exportJsonBtn.setToolTip("Bu formun ayarlarını JSON dosyasına kaydet")
        self.importJsonBtn.setToolTip("Daha önce kaydedilmiş ayarları forma yükle")
        self.splitter.setSizes([650, 400])

    def _align_form_labels(self):
        """One label column for every group, so all fields start on the same edge."""
        forms = [layout for layout in self.formContents.findChildren(QFormLayout)]
        labels = [form.itemAt(row, QFormLayout.LabelRole).widget()
                  for form in forms for row in range(form.rowCount())
                  if form.itemAt(row, QFormLayout.LabelRole) is not None
                  and form.itemAt(row, QFormLayout.LabelRole).widget() is not None]
        width = max((label.sizeHint().width() for label in labels), default=0) + 12
        for label in labels:
            label.setMinimumWidth(width)

    def _toggle_advanced(self, expanded):
        self.advanced_form.setVisible(expanded)
        self.advanced_form_toggle.setText("Gelişmiş ayarlar: yaylar ve ölçüm modeli  " + ("⌄" if expanded else "›"))

    def on_shown(self) -> None:
        pass

    # ------------------------------------------------------------------
    def _on_arc_mode_changed(self, _index=None) -> None:
        is_prescribed = _value(self.arcModeCombo) == "prescribed"
        for w in (self.arcDurLabel, self.arcDurSpin, self.arcStrideLabel, self.arcStrideSpin):
            w.setVisible(is_prescribed)
        # The station elevation mask applies to planned windows as well.
        for w in (self.elevMaskLabel, self.elevMaskSpin):
            w.setVisible(True)
        for w in (self.gapThreshLabel, self.gapThreshSpin):
            w.setVisible(not is_prescribed)
        self._update_arc_count()

    def _on_aberration_changed(self, _v=None) -> None:
        """Enable aberration controls only for position measurements; stellar
        aberration requires light time, and the frame combo requires stellar."""
        is_position = _value(self.measTypeCombo) == "position"
        self.lightTimeCheck.setEnabled(is_position)
        lt_on = is_position and self.lightTimeCheck.isChecked()
        self.stellarCheck.setEnabled(lt_on)
        self.stellarModelCombo.setEnabled(lt_on and self.stellarCheck.isChecked())

    def _update_arc_count(self) -> None:
        if _value(self.arcModeCombo) == "prescribed":
            total_h = self.durationSpin.value() * 24.0
            stride_h = self.arcStrideSpin.value()
            count = int(total_h / stride_h) if stride_h > 0 else 0
            self._arc_count_label.setText(
                f"~{count} yay ({self.durationSpin.value():.1f} gün ÷ {stride_h:.1f} sa adım). "
                "Yalnız gerçekten görünen istasyonlardan ölçüm alınır.")
        else:
            gap_min = self.gapThreshSpin.value() / 60.0
            self._arc_count_label.setText(
                f"Görünürlük pencerelerine göre (yükseliş ≥ {self.elevMaskSpin.value():.0f}°, "
                f"boşluk ≤ {gap_min:.0f} dk)")

    def _on_preset_selected(self, name) -> None:
        if name in PRESETS:
            self._archive_origin = None
            self._show_origin()
            self._current_model = ScenarioModel(**vars(PRESETS[name]))
            self.presetDescLabel.setText(_PRESET_LABELS.get(name, (name, self._current_model.description))[1])
            self._model_to_widgets(self._current_model)
            self._update_json()
        else:
            self.presetDescLabel.setText("")

    def _model_to_widgets(self, model: ScenarioModel) -> None:
        from services.archived_scenario import form_from_mapping
        form_from_mapping(asdict(model))
        numbers = ((self.durationSpin, model.duration_days), (self.stepSpin, model.state_step_s),
            (self.seedSpin, model.random_seed), (self.arcDurSpin, model.arc_duration_h),
            (self.arcStrideSpin, model.arc_stride_h), (self.elevMaskSpin, model.elevation_mask_deg),
            (self.gapThreshSpin, model.gap_threshold_s))
        # Check representability before touching the current draft. Never clip an
        # archived value to a widget's old range or quietly round its precision.
        for widget, value in numbers:
            if max(0, -Decimal(str(value)).as_tuple().exponent) > 323:
                raise ValueError("Kayıtlı sayı bu formun gösterim hassasiyetini aşıyor.")
        self._current_model = model
        self._updating_from_model = True
        try:
            self.nameEdit.setText(model.name)
            self.descEdit.setText(model.description)
            self.outputDirEdit.setText(model.output_dir)
            for widget, value in numbers:
                _set_number(widget, value)
            _set_combo(self.arcModeCombo, model.arc_mode)
            _set_combo(self.measTypeCombo, model.measurement_type)
            _set_combo(self.rrPhysicsCombo, model.range_rate_physics)
            _set_combo(self.networkCombo, model.network_name)
            self.noiseCheck.setChecked(model.noise_enabled)
            _set_combo(self.estimatorCombo, model.estimator_type)
            _set_combo(self.startModeCombo, model.start_mode)
            self.lightTimeCheck.setChecked(bool(getattr(model, "apply_light_time", False)))
            self.stellarCheck.setChecked(bool(getattr(model, "apply_stellar_aberration", False)))
            _set_combo(self.stellarModelCombo, getattr(model, "stellar_aberration_model", "local_mci"))
        finally:
            self._updating_from_model = False
        self._on_arc_mode_changed()
        self._on_aberration_changed()

    def _widgets_to_model(self) -> ScenarioModel:
        is_position = _value(self.measTypeCombo) == "position"
        light_time = is_position and self.lightTimeCheck.isChecked()
        stellar = light_time and self.stellarCheck.isChecked()
        return replace(self._current_model,
            name=self.nameEdit.text().strip() or "unnamed",
            description=self.descEdit.text().strip(),
            output_dir=self.outputDirEdit.text().strip() or "results",
            duration_days=self.durationSpin.value(),
            state_step_s=self.stepSpin.value(),
            random_seed=int(self.seedSpin.value()),
            arc_mode=_value(self.arcModeCombo),
            arc_duration_h=self.arcDurSpin.value(),
            arc_stride_h=self.arcStrideSpin.value(),
            elevation_mask_deg=self.elevMaskSpin.value(),
            gap_threshold_s=self.gapThreshSpin.value(),
            measurement_type=_value(self.measTypeCombo),
            range_rate_physics=_value(self.rrPhysicsCombo),
            network_name=_value(self.networkCombo),
            noise_enabled=self.noiseCheck.isChecked(),
            estimator_type=_value(self.estimatorCombo),
            start_mode=_value(self.startModeCombo),
            apply_light_time=light_time,
            apply_stellar_aberration=stellar,
            stellar_aberration_model=_value(self.stellarModelCombo),
        )

    def _update_json(self, *_args) -> None:
        if self._updating_from_model:
            return
        model = self._widgets_to_model()
        self._current_model = model
        self.jsonPreview.setPlainText(model.to_json())
        self._update_arc_count()

    # ------------------------------------------------------------------
    def _copy_json(self) -> None:
        QApplication.clipboard().setText(self.jsonPreview.toPlainText())

    def _export_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Senaryo ayarlarını kaydet",
            str(Path.home() / f"{self._current_model.name}.json"),
            "JSON dosyaları (*.json)",
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._current_model.to_json())

    def _import_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Senaryo ayarlarını aç",
            str(Path.home()),
            "JSON dosyaları (*.json)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError("Senaryo JSON nesne olmalı.")
            unknown = set(payload) - set(ScenarioModel.__dataclass_fields__)
            if unknown:
                raise ValueError("Bu form JSON'unda desteklenmeyen alanlar var: " + ", ".join(sorted(unknown)))
            model = ScenarioModel.from_json(text)
            self._model_to_widgets(model)
            self._archive_origin = None
            self._show_origin()
            self._update_json()
            self.presetCombo.setCurrentIndex(0)
        except Exception as exc:
            QMessageBox.warning(self, "Ayarlar açılamadı", str(exc))

    def _run_scenario(self) -> None:
        """Validate the environment before persisting a new job or starting the solver."""
        from services.scenario_execution import execution_payload, prepare_desktop_job
        model = self._widgets_to_model()
        main_win = self.window()
        if not hasattr(main_win, "run_desktop_job"):
            return
        monitor = main_win._pages.get("run_monitor")
        if monitor and monitor._worker.is_running:
            main_win.navigate("run_monitor")
            main_win.set_status("Önce çalışan koşunun tamamlanmasını bekleyin veya iptal edin.")
            return
        try:
            execution_payload(model)
        except ValueError as exc:
            QMessageBox.warning(self, "Koşu oluşturulamadı", str(exc))
            return
        origin = self._archive_origin
        def launch(ready):
            self.runScenarioBtn.setEnabled(True)
            if not ready:
                main_win.navigate("settings")
                main_win.set_status("Koşu başlatılmadı; ortam raporundaki eksikleri giderin.")
                return
            monitor = main_win._pages.get("run_monitor")
            if monitor and monitor._worker.is_running:
                main_win.set_status("Başka bir koşu çalışıyor; önce tamamlanmasını bekleyin.")
                return
            try:
                path, folder, run_id = prepare_desktop_job(model, PYTHON_PORT, derived_from=origin)
            except (ValueError, OSError) as exc:
                QMessageBox.warning(self, "Koşu oluşturulamadı", str(exc))
                return
            main_win.run_desktop_job(path, folder, run_id)
        manager = getattr(main_win, "environment", None)
        if manager is not None:
            if manager.busy:
                main_win.set_status("Ortam kontrolü sürüyor; tamamlanınca koşuyu başlatın.")
                return
            self.runScenarioBtn.setEnabled(False)
            manager.refresh(model=model, callback=launch)
            main_win.set_status("Koşu öncesi ortam kontrol ediliyor…")
        else:
            launch(True)

    def load_archived(self, archived):
        previous = (replace(self._widgets_to_model()), self._archive_origin)
        self._model_to_widgets(replace(archived.model))
        self._draft_before_archive = previous
        self._archive_origin = archived.origin
        self.presetCombo.setCurrentIndex(0)
        self._update_json()
        self._show_origin()

    def _show_origin(self):
        self.origin_label.setVisible(self._archive_origin is not None)
        self.restore_draft_button.setVisible(self._draft_before_archive is not None)
        if self._archive_origin:
            self.origin_label.setText(f"Kaynak koşu: {self._archive_origin['run_id']}\n"
                "Ayarlar yeni taslağa alındı. Çalıştırınca yeni koşu kimliği ve değişiklik kaydı oluşturulur.")
            self.origin_label.setToolTip(self._archive_origin["result_path"])

    def restore_previous_draft(self):
        if self._draft_before_archive:
            model, origin = self._draft_before_archive
            self._model_to_widgets(model)
            self._archive_origin = origin
            self._draft_before_archive = None
            self.presetCombo.setCurrentIndex(0)
            self._update_json()
            self._show_origin()


# -------------------------------------------------------------------------
def _set_combo(combo, value: str) -> None:
    idx = combo.findData(value)
    if idx < 0:
        idx = combo.findText(value)
    if idx < 0:
        # An archived value outside the offered choices is shown, never replaced.
        combo.addItem(value, value)
        idx = combo.count() - 1
    combo.setCurrentIndex(idx)


def _set_number(widget, value):
    if isinstance(widget, QDoubleSpinBox):
        widget.setDecimals(max(widget.decimals(), max(0, -Decimal(str(value)).as_tuple().exponent)))
    widget.setRange(min(widget.minimum(), value), max(widget.maximum(), value))
    widget.setValue(value)
