"""Scenario Builder page controller — configure and export scenario JSON."""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QFileDialog, QMessageBox, QApplication, QLabel, QPushButton,
)
from PyQt5.QtCore import Qt
from PyQt5 import uic

from services.project_paths import DESKTOP_APP_DIR, PYTHON_PORT, RESULTS_DIR
from models.scenario_model import ScenarioModel, PRESETS

UI_PATH = DESKTOP_APP_DIR / "ui" / "pages" / "scenario_builder_page.ui"


class ScenarioBuilderController(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        uic.loadUi(str(UI_PATH), self)
        self._current_model = ScenarioModel()
        self._updating_from_model = False
        self._setup()

    def _setup(self) -> None:
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
        self._arc_count_label.setStyleSheet("QLabel { color: #00d4ff; font-size: 11px; }")
        self.arcGroup.layout().addRow("Estimate:", self._arc_count_label)

        # Action buttons
        self.copyJsonBtn.clicked.connect(self._copy_json)
        self.exportJsonBtn.clicked.connect(self._export_json)
        self.importJsonBtn.clicked.connect(self._import_json)
        self.runScenarioBtn.clicked.connect(self._run_scenario)
        if not hasattr(self, "openAnalysisBtn"):
            self.openAnalysisBtn = QPushButton("→ Open in Analysis")
            self.openAnalysisBtn.setToolTip(
                "Load current scenario duration and parameters into the Analysis page"
            )
            self.runScenarioBtn.parentWidget().layout().addWidget(self.openAnalysisBtn)
        self.openAnalysisBtn.clicked.connect(self._open_in_analysis)

        # Initial render
        self._model_to_widgets(self._current_model)
        self._update_json()
        self._on_arc_mode_changed()

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
        if idx >= 0:
            self.networkCombo.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    def _on_arc_mode_changed(self, _text: str = "") -> None:
        is_prescribed = self.arcModeCombo.currentText() == "prescribed"
        # prescribed-only fields
        for w in (self.arcDurLabel, self.arcDurSpin,
                  self.arcStrideLabel, self.arcStrideSpin):
            w.setVisible(is_prescribed)
        # visibility-only fields
        for w in (self.elevMaskLabel, self.elevMaskSpin,
                  self.gapThreshLabel, self.gapThreshSpin):
            w.setVisible(not is_prescribed)
        self._update_arc_count()

    def _update_arc_count(self) -> None:
        mode = self.arcModeCombo.currentText()
        if mode == "prescribed":
            total_h = self.durationSpin.value() * 24.0
            stride_h = self.arcStrideSpin.value()
            count = int(total_h / stride_h) if stride_h > 0 else 0
            self._arc_count_label.setText(
                f"~{count} arcs  "
                f"({self.durationSpin.value():.1f} days ÷ {stride_h:.1f}h stride)"
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
            self._current_model = ScenarioModel(**vars(PRESETS[name]))
            self.presetDescLabel.setText(self._current_model.description)
            self._model_to_widgets(self._current_model)
            self._update_json()
        else:
            self.presetDescLabel.setText("")

    def _model_to_widgets(self, model: ScenarioModel) -> None:
        self._updating_from_model = True
        try:
            self.nameEdit.setText(model.name)
            self.descEdit.setText(model.description)
            self.outputDirEdit.setText(model.output_dir)
            self.durationSpin.setValue(model.duration_days)
            self.stepSpin.setValue(model.state_step_s)
            self.seedSpin.setValue(model.random_seed)
            _set_combo(self.arcModeCombo, model.arc_mode)
            self.arcDurSpin.setValue(model.arc_duration_h)
            self.arcStrideSpin.setValue(model.arc_stride_h)
            self.elevMaskSpin.setValue(model.elevation_mask_deg)
            self.gapThreshSpin.setValue(model.gap_threshold_s)
            _set_combo(self.measTypeCombo, model.measurement_type)
            _set_combo(self.rrPhysicsCombo, model.range_rate_physics)
            _set_combo(self.networkCombo, model.network_name)
            self.noiseCheck.setChecked(model.noise_enabled)
            _set_combo(self.estimatorCombo, model.estimator_type)
            _set_combo(self.startModeCombo, model.start_mode)
        finally:
            self._updating_from_model = False
        self._on_arc_mode_changed()

    def _widgets_to_model(self) -> ScenarioModel:
        return ScenarioModel(
            name=self.nameEdit.text().strip() or "unnamed",
            description=self.descEdit.text().strip(),
            output_dir=self.outputDirEdit.text().strip() or "results",
            duration_days=self.durationSpin.value(),
            state_step_s=self.stepSpin.value(),
            random_seed=self.seedSpin.value(),
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
            model = ScenarioModel.from_json(text)
            self._model_to_widgets(model)
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
        """Export a temp JSON and launch scenario_config_cli.py with it."""
        import tempfile
        import os
        model = self._widgets_to_model()
        tmp_file = Path(tempfile.mktemp(suffix=".json"))
        tmp_file.write_text(model.to_json(), encoding="utf-8")

        script = PYTHON_PORT / "examples" / "scenario_config_cli.py"
        if not script.exists():
            QMessageBox.warning(
                self, "Script not found",
                f"Could not find:\n{script}\n\nPlease run from the Run Monitor page.",
            )
            return

        # Navigate to run monitor and launch
        main_win = self.window()
        if hasattr(main_win, "navigate_and_run"):
            main_win.navigate_and_run(str(script), extra_args=[str(tmp_file)], auto_start=True)
        elif hasattr(main_win, "navigate"):
            main_win.navigate("run_monitor")


# -------------------------------------------------------------------------
def _set_combo(combo, value: str) -> None:
    idx = combo.findText(value)
    if idx >= 0:
        combo.setCurrentIndex(idx)
