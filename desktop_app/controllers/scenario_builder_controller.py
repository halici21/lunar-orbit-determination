"""Scenario Builder page controller — configure and export scenario JSON."""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
from dataclasses import replace, asdict
from decimal import Decimal

from PyQt5.QtWidgets import (
    QWidget, QFileDialog, QMessageBox, QApplication, QLabel, QPushButton, QDoubleSpinBox,
)
from PyQt5.QtCore import Qt
from PyQt5 import uic

from services.project_paths import DESKTOP_APP_DIR, PYTHON_PORT, RESULTS_DIR
from models.scenario_model import ScenarioModel, PRESETS
from styles.artifact_workspace import ARTIFACT_STYLE
from styles.theme import C

UI_PATH = DESKTOP_APP_DIR / "ui" / "pages" / "scenario_builder_page.ui"


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
        # Populate preset combo
        self.presetCombo.addItem("— custom —")
        for name in PRESETS:
            self.presetCombo.addItem(name)
        self.presetCombo.currentTextChanged.connect(self._on_preset_selected)

        # Connect all form fields to _update_json
        for widget in [
            self.nameEdit, self.descEdit, self.outputDirEdit,
        ]:
            widget.textChanged.connect(self._update_json)

        for spinbox in [
            self.durationSpin, self.stepSpin, self.arcDurSpin,
            self.arcStrideSpin, self.elevMaskSpin, self.gapThreshSpin,
        ]:
            spinbox.valueChanged.connect(self._update_json)

        self.seedSpin.valueChanged.connect(self._update_json)

        for combo in [
            self.arcModeCombo, self.measTypeCombo, self.rrPhysicsCombo,
            self.networkCombo, self.estimatorCombo, self.startModeCombo,
        ]:
            combo.currentTextChanged.connect(self._update_json)

        self.arcModeCombo.currentTextChanged.connect(self._on_arc_mode_changed)
        self.noiseCheck.stateChanged.connect(self._update_json)

        # Arc count estimate label — added dynamically to arcGroup's form layout
        self._arc_count_label = QLabel("")
        self._arc_count_label.setStyleSheet(f"QLabel {{ color:{C.TEXT_MUTED}; font-size:12px; }}")
        self.arcGroup.layout().addRow("Estimate:", self._arc_count_label)

        # Action buttons
        self.copyJsonBtn.clicked.connect(self._copy_json)
        self.exportJsonBtn.clicked.connect(self._export_json)
        self.importJsonBtn.clicked.connect(self._import_json)
        self.runScenarioBtn.setProperty("primary", True)
        self.runScenarioBtn.clicked.connect(self._run_scenario)
        self.runScenarioBtn.setToolTip("Formu yeni koşu kimliğiyle kaydet ve Run Monitor üzerinden çalıştır")
        if not hasattr(self, "openAnalysisBtn"):
            self.openAnalysisBtn = QPushButton("→ Open in Analysis")
            self.openAnalysisBtn.setToolTip(
                "Load current scenario duration and parameters into the Analysis page"
            )
            self.runScenarioBtn.parentWidget().layout().addWidget(self.openAnalysisBtn)
        self.openAnalysisBtn.clicked.connect(self._open_in_analysis)

        # --- Aberration corrections (position measurements only) -------------
        # Added programmatically (like the buttons above) to avoid editing the
        # .ui; wired into the same JSON/model round-trip as the other fields.
        from PyQt5.QtWidgets import QCheckBox, QComboBox
        self.lightTimeCheck = QCheckBox("One-way light time (CN)")
        self.lightTimeCheck.setToolTip(
            "Converged Newtonian one-way light-time correction for range/az/el.\n"
            "Position measurements only."
        )
        self.stellarCheck = QCheckBox("Stellar aberration (+S)")
        self.stellarCheck.setToolTip(
            "Apply stellar aberration on top of the light-time solution.\n"
            "Requires light time."
        )
        self.stellarModelCombo = QComboBox()
        self.stellarModelCombo.addItems(["local_mci", "spice_ssb"])
        self.stellarModelCombo.setToolTip(
            "Observer-velocity frame for +S:\n"
            "  spice_ssb  - SPICE-like CN+S (SSB velocity, ~1e-4 rad)\n"
            "  local_mci  - Moon-centred velocity (cheaper, ~20x smaller)"
        )
        meas_layout = self.measGroup.layout()
        meas_layout.addRow("Light time:", self.lightTimeCheck)
        meas_layout.addRow("Aberration:", self.stellarCheck)
        meas_layout.addRow("Aberration frame:", self.stellarModelCombo)
        for _w in (self.lightTimeCheck, self.stellarCheck):
            _w.stateChanged.connect(self._update_json)
            _w.stateChanged.connect(self._on_aberration_changed)
        self.stellarModelCombo.currentTextChanged.connect(self._update_json)
        self.measTypeCombo.currentTextChanged.connect(self._on_aberration_changed)

        # Initial render
        self._model_to_widgets(self._current_model)
        self._update_json()
        self._on_arc_mode_changed()

    def _simplify_layout(self):
        from PyQt5.QtWidgets import QVBoxLayout, QBoxLayout, QSizePolicy
        self.layout().setDirection(QBoxLayout.TopToBottom)
        self.layout().setContentsMargins(24, 12, 24, 16)
        self.layout().setSpacing(12)
        self.formScroll.setMaximumWidth(16777215)
        form_layout = self.estimatorGroup.parentWidget().layout()
        form_layout.setContentsMargins(0, 0, 20, 0)
        form_layout.setSpacing(10)
        for group in (self.presetsGroup, self.identityGroup, self.timeGroup, self.estimatorGroup):
            group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.presetDescLabel.setStyleSheet(f"color:{C.TEXT_MUTED}; font-size:12px;")
        self.advanced_form_toggle = QPushButton("Gelişmiş ayarlar · yaylar ve ölçümler")
        self.advanced_form_toggle.setCheckable(True)
        self.advanced_form = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_form)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        for group in (self.arcGroup, self.measGroup):
            form_layout.removeWidget(group)
            advanced_layout.addWidget(group)
        form_layout.insertWidget(form_layout.count() - 1, self.advanced_form_toggle)
        form_layout.insertWidget(form_layout.count() - 1, self.advanced_form)
        self.advanced_form.hide()
        self.advanced_form_toggle.toggled.connect(self.advanced_form.setVisible)
        self.presetsGroup.setTitle("Başlangıç şablonu")
        self.identityGroup.setTitle("Senaryo")
        self.timeGroup.setTitle("Zaman aralığı")
        self.estimatorGroup.setTitle("Kestirici")
        self.runScenarioBtn.setText("Koşuyu başlat")
        self.exportJsonBtn.setText("Ayarları kaydet")
        self.importJsonBtn.setText("Ayarları aç")
        self.openAnalysisBtn.setText("Analizde aç")
        self.jsonLabel.setText("Etkin ayarlar · JSON")
        self.copyJsonBtn.setText("Kopyala")
        self.preview_toggle = QPushButton("JSON önizlemesi")
        self.preview_toggle.setCheckable(True)
        self.preview_toggle.setChecked(False)
        # Keep all action buttons outside the optional technical preview.
        self.previewLayout.removeItem(self.actionButtonsLayout)
        actions = QWidget()
        actions.setLayout(self.actionButtonsLayout)
        self.previewLayout.removeWidget(self.openAnalysisBtn)
        self.actionButtonsLayout.insertWidget(2, self.openAnalysisBtn)
        self.layout().addWidget(actions)
        # Provenance and draft recovery remain visible with the JSON panel closed.
        self.previewLayout.removeWidget(self.origin_label)
        self.previewLayout.removeWidget(self.restore_draft_button)
        self.layout().insertWidget(0, self.restore_draft_button)
        self.layout().insertWidget(0, self.origin_label)
        toolbar = QWidget()
        toolbar_layout = QVBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.addWidget(self.preview_toggle, 0, Qt.AlignRight)
        self.layout().insertWidget(0, toolbar)
        self.layout().setStretchFactor(self.splitter, 1)
        self.previewPanel.hide()
        self.preview_toggle.toggled.connect(self.previewPanel.setVisible)
        for button in (self.exportJsonBtn, self.importJsonBtn, self.openAnalysisBtn, self.runScenarioBtn):
            button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.splitter.setSizes([650, 400])
        for name, label in (
            ("nameLabel", "Ad"), ("descLabel", "Açıklama"), ("outputDirLabel", "Çıktı klasörü"),
            ("durationLabel", "Süre (gün)"), ("stepLabel", "Örnekleme (s)"),
            ("seedLabel", "Rastgelelik tohumu"), ("estimatorLabel", "Yöntem"),
            ("startModeLabel", "Başlangıç")):
            getattr(self, name).setText(label)

    def on_shown(self) -> None:
        self._reload_network_presets()

    def _reload_network_presets(self) -> None:
        from PyQt5.QtCore import QSettings
        s = QSettings("LunarOD", "DesktopApp")
        custom = s.value("stations/custom_networks", {})
        if not isinstance(custom, dict):
            custom = {}
        current = self.networkCombo.currentText()
        # Remove old custom entries (keep "multi" and "single")
        while self.networkCombo.count() > 2:
            self.networkCombo.removeItem(2)
        for name in sorted(custom.keys()):
            self.networkCombo.addItem(name)
        # Restore selection
        idx = self.networkCombo.findText(current)
        if idx < 0 and current:
            self.networkCombo.addItem(current)
            idx = self.networkCombo.count() - 1
        self.networkCombo.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    def _on_arc_mode_changed(self, _text: str = "") -> None:
        is_prescribed = self.arcModeCombo.currentText() == "prescribed"
        # prescribed-only fields
        for w in (self.arcDurLabel, self.arcDurSpin,
                  self.arcStrideLabel, self.arcStrideSpin):
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
        is_position = self.measTypeCombo.currentText() == "position"
        self.lightTimeCheck.setEnabled(is_position)
        lt_on = is_position and self.lightTimeCheck.isChecked()
        self.stellarCheck.setEnabled(lt_on)
        self.stellarModelCombo.setEnabled(lt_on and self.stellarCheck.isChecked())

    def _update_arc_count(self) -> None:
        mode = self.arcModeCombo.currentText()
        if mode == "prescribed":
            total_h = self.durationSpin.value() * 24.0
            stride_h = self.arcStrideSpin.value()
            count = int(total_h / stride_h) if stride_h > 0 else 0
            self._arc_count_label.setText(
                f"~{count} arcs  "
                f"({self.durationSpin.value():.1f} days ÷ {stride_h:.1f}h stride)\n"
                "Planlı pencereler: yalnızca gerçekten görünen istasyonlardan ölçüm alınır."
            )
        else:
            gap_min = self.gapThreshSpin.value() / 60.0
            self._arc_count_label.setText(
                f"by visibility windows  "
                f"(elv ≥ {self.elevMaskSpin.value():.0f}°, "
                f"gap ≤ {gap_min:.0f} min)"
            )

    def _on_preset_selected(self, name: str) -> None:
        if name in PRESETS:
            self._archive_origin = None
            self._show_origin()
            self._current_model = ScenarioModel(**vars(PRESETS[name]))
            self.presetDescLabel.setText(self._current_model.description)
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
        is_position = self.measTypeCombo.currentText() == "position"
        light_time = is_position and self.lightTimeCheck.isChecked()
        stellar = light_time and self.stellarCheck.isChecked()
        return replace(self._current_model,
            name=self.nameEdit.text().strip() or "unnamed",
            description=self.descEdit.text().strip(),
            output_dir=self.outputDirEdit.text().strip() or "results",
            duration_days=self.durationSpin.value(),
            state_step_s=self.stepSpin.value(),
            random_seed=int(self.seedSpin.value()),
            arc_mode=self.arcModeCombo.currentText(),
            arc_duration_h=self.arcDurSpin.value(),
            arc_stride_h=self.arcStrideSpin.value(),
            elevation_mask_deg=self.elevMaskSpin.value(),
            gap_threshold_s=self.gapThreshSpin.value(),
            measurement_type=self.measTypeCombo.currentText(),
            range_rate_physics=self.rrPhysicsCombo.currentText(),
            network_name=self.networkCombo.currentText(),
            noise_enabled=self.noiseCheck.isChecked(),
            estimator_type=self.estimatorCombo.currentText(),
            start_mode=self.startModeCombo.currentText(),
            apply_light_time=light_time,
            apply_stellar_aberration=stellar,
            stellar_aberration_model=self.stellarModelCombo.currentText(),
        )

    def _update_json(self) -> None:
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
            self, "Export Scenario JSON",
            str(Path.home() / f"{self._current_model.name}.json"),
            "JSON files (*.json)",
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._current_model.to_json())

    def _import_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Scenario JSON",
            str(Path.home()),
            "JSON files (*.json)",
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
            # Reset preset combo to custom
            self.presetCombo.setCurrentIndex(0)
        except Exception as exc:
            QMessageBox.warning(self, "Import failed", str(exc))

    def _open_in_analysis(self) -> None:
        """Navigate to Analysis page with current scenario duration/params pre-loaded."""
        model = self._widgets_to_model()
        main_win = self.window()
        if hasattr(main_win, "navigate"):
            main_win.navigate("analysis")
            analysis = main_win._pages.get("analysis")
            if analysis and hasattr(analysis, "apply_scenario_params"):
                analysis.apply_scenario_params(model)

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
    idx = combo.findText(value)
    if idx < 0:
        combo.addItem(value)
        idx = combo.count() - 1
    combo.setCurrentIndex(idx)


def _set_number(widget, value):
    if isinstance(widget, QDoubleSpinBox):
        widget.setDecimals(max(widget.decimals(), max(0, -Decimal(str(value)).as_tuple().exponent)))
    widget.setRange(min(widget.minimum(), value), max(widget.maximum(), value))
    widget.setValue(value)
