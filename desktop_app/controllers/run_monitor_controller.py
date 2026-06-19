"""Run Monitor page controller — launches scripts, streams live output."""
from __future__ import annotations
import sys
import subprocess
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QMessageBox, QGroupBox, QFormLayout,
    QDoubleSpinBox, QComboBox, QCheckBox, QLineEdit, QLabel,
)
from PyQt5.QtCore import Qt
from PyQt5 import uic

from services.project_paths import DESKTOP_APP_DIR, EXAMPLES_DIR, PYTHON_PORT
from workers.process_worker import ProcessWorker
from widgets.log_console import LogConsole
from styles.theme import C, FONT

UI_PATH = DESKTOP_APP_DIR / "ui" / "pages" / "run_monitor_page.ui"

# Scripts offered in the combo (label, rel_path_or_None where None = section header)
_SCRIPTS: list[tuple[str, str | None]] = [
    # Core comparisons
    ("── Core Comparisons ──", None),
    ("Baseline BLS/UKF Comparison",    "examples/baseline_bls_ukf_comparison.py"),
    ("BLS/UKF Matrix Campaign",        "examples/baseline_bls_ukf_matrix_campaign.py"),
    ("Sequential Tracking Comparison", "examples/sequential_tracking_comparison.py"),
    ("Two-Way Doppler BLS/UKF",        "examples/two_way_doppler_bls_ukf_comparison.py"),
    ("Real Visibility BLS/UKF Matrix", "examples/real_visibility_bls_ukf_matrix.py"),
    # Thesis / ablation
    ("── Thesis & Ablation ──", None),
    ("BLS 7-Day Ablation",             "examples/bls_7day_ablation_appendix.py"),
    ("Thesis Factorial Report",        "examples/thesis_factorial_report.py"),
    ("Synthetic Hot Start",            "examples/synthetic_hot_start_report.py"),
    # Formal handoff
    ("── Formal Handoff ──", None),
    ("Formal Handoff",                 "examples/formal_handoff_report.py"),
    ("Formal + Process Noise",         "examples/formal_handoff_process_noise_report.py"),
    ("Formal Bias Handoff",            "examples/formal_bias_handoff_report.py"),
    # Campaigns
    ("── Campaigns ──", None),
    ("28-Day ITU Campaign",            "examples/campaign_28day_itu_report.py"),
    ("28-Day ITU Hot Start",           "examples/campaign_28day_itu_all_arc_hot_report.py"),
    ("4-Day Visibility+RR Campaign",   "examples/campaign_4day_visibility_rr_report.py"),
    ("Campaign Diagnostic Plots",      "examples/campaign_diagnostic_plots.py"),
    # Visibility
    ("── Visibility ──", None),
    ("28-Day Visibility Gantt",        "examples/visibility_28day_dsn_itu_gantt.py"),
    ("Long Visibility Report",         "examples/long_visibility_report.py"),
    ("Long Visibility OD",             "examples/long_visibility_od_report.py"),
    ("Long Vis RR+Noise+Bias",         "examples/long_visibility_rr_noise_bias_report.py"),
    ("Long Vis RR OD",                 "examples/long_visibility_rr_od_report.py"),
    ("Visibility from Fixture",        "examples/visibility_report_from_fixture.py"),
    # UKF
    ("── UKF ──", None),
    ("UKF SPICE Mismatch Campaign",    "examples/ukf_spice_mismatch_campaign.py"),
    ("UKF Stress Monte Carlo",         "examples/ukf_stress_monte_carlo_campaign.py"),
    ("UKF Overnight Validation",       "examples/overnight_ukf_validation.py"),
    # Physics comparisons
    ("── Physics Comparisons ──", None),
    ("Compare RR Physics",             "examples/compare_range_rate_physics.py"),
    ("Compare Visibility Models",      "examples/compare_visibility_models.py"),
    ("Quick Two-Way SPICE",            "examples/quick_two_way_spice_campaign.py"),
    # CLI / utilities
    ("── Utilities ──", None),
    ("Run Scenario JSON",              "examples/run_scenario_config.py"),
    ("Run All Experiments (list)",     "examples/run_all_experiments.py --list"),
    ("Scenario Config CLI",            "examples/scenario_config_cli.py --help"),
]


class RunMonitorController(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        uic.loadUi(str(UI_PATH), self)
        self._worker = ProcessWorker(self)
        self._last_exit_code: int | None = None
        self._override_command: tuple[Path, list[str]] | None = None
        self._setting_preset = False
        self._setup()

    def _setup(self) -> None:
        # Insert LogConsole into placeholder
        self._log = LogConsole(self)
        placeholder = self.consolePlaceholder
        parent_layout = placeholder.parent().layout()
        idx = parent_layout.indexOf(placeholder)
        placeholder.hide()
        parent_layout.insertWidget(idx, self._log, stretch=1)

        # Populate script combo
        self._script_map: dict[int, str] = {}  # combo_idx → rel_path
        for label, rel in _SCRIPTS:
            idx = self.scriptCombo.count()
            if rel is None:
                self.scriptCombo.addItem(label)
                item = self.scriptCombo.model().item(idx)
                if item:
                    from PyQt5.QtGui import QColor as _QColor
                    item.setEnabled(False)
                    item.setForeground(_QColor(C.TEXT_MUTED))
            else:
                self.scriptCombo.addItem(label)
                self._script_map[idx] = rel
        self.scriptCombo.currentIndexChanged.connect(self._on_script_changed)
        self._update_command_preview()

        # Inject scenario-params group between scriptGroup and consoleGroup
        self._build_scenario_params_group()

        # Buttons
        self.startBtn.clicked.connect(self._start)
        self.cancelBtn.clicked.connect(self._cancel)
        self.clearLogBtn.clicked.connect(self._log.clear_log)
        self.openOutputBtn.clicked.connect(self._open_output)

        # Worker signals
        self._worker.output_line.connect(self._log.append_stdout)
        self._worker.error_line.connect(self._log.append_stderr)
        self._worker.started.connect(self._on_started)
        self._worker.finished.connect(self._on_finished)
        self._worker.elapsed_tick.connect(self._on_tick)

    def on_shown(self) -> None:
        pass  # nothing to refresh on show

    # ------------------------------------------------------------------
    def _build_scenario_params_group(self) -> None:
        """Insert a 'Scenario Parameters' group that generates --arg flags."""
        box = QGroupBox("Scenario Parameters")
        box.setCheckable(True)
        box.setChecked(False)
        box.setStyleSheet(
            f"QGroupBox {{ color:{C.TEXT_SECONDARY}; font-size:{FONT.SIZE_SM}px;"
            f" border:1px solid {C.BORDER_MAIN}; border-radius:4px; margin-top:6px; }}"
            f"QGroupBox::title {{ subcontrol-origin:margin; left:8px; padding:0 3px; }}"
        )

        form = QFormLayout(box)
        form.setContentsMargins(12, 6, 12, 10)
        form.setSpacing(6)
        form.setLabelAlignment(Qt.AlignRight)

        def _spin(lo, hi, val, suffix="", decimals=1):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(decimals)
            s.setValue(val)
            if suffix:
                s.setSuffix(f" {suffix}")
            s.setStyleSheet(
                f"QDoubleSpinBox {{ background:{C.BG_DEEP}; color:{C.TEXT_PRIMARY};"
                f" border:1px solid {C.BORDER_MAIN}; border-radius:3px; padding:2px 4px;"
                f" font-size:{FONT.SIZE_SM}px; }}"
            )
            s.valueChanged.connect(self._update_command_preview)
            return s

        def _combo(opts, cur=""):
            c = QComboBox()
            for o in opts:
                c.addItem(o)
            if cur:
                c.setCurrentText(cur)
            c.setStyleSheet(
                f"QComboBox {{ background:{C.BG_DEEP}; color:{C.TEXT_PRIMARY};"
                f" border:1px solid {C.BORDER_MAIN}; border-radius:3px; padding:2px 4px;"
                f" font-size:{FONT.SIZE_SM}px; }}"
                f"QComboBox::drop-down {{ border:none; }}"
            )
            c.currentTextChanged.connect(self._update_command_preview)
            return c

        self._sp_duration = _spin(0.5, 28.0, 3.0, "days")
        form.addRow("Duration:", self._sp_duration)

        self._sp_network = _combo(["multi", "single"], "multi")
        form.addRow("Network:", self._sp_network)

        self._sp_meas = _combo(["range_rate", "position"], "range_rate")
        form.addRow("Measurement type:", self._sp_meas)

        self._sp_noise = QCheckBox("Enabled")
        self._sp_noise.setChecked(True)
        self._sp_noise.setStyleSheet(
            f"QCheckBox {{ color:{C.TEXT_SECONDARY}; font-size:{FONT.SIZE_SM}px; }}"
        )
        self._sp_noise.stateChanged.connect(self._update_command_preview)
        form.addRow("Noise:", self._sp_noise)

        self._sp_estimator = _combo(["bls_lm", "ukf", "srif"], "bls_lm")
        form.addRow("Estimator:", self._sp_estimator)

        hint = QLabel(
            "Parameters are passed as --duration, --network, etc. "
            "Only scenario_config_cli.py accepts these flags."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{C.TEXT_MUTED}; font-size:{FONT.SIZE_XS}px;")
        form.addRow("", hint)

        box.toggled.connect(self._update_command_preview)

        # Insert above consoleGroup
        parent_layout = self.mainLayout
        console_idx = parent_layout.indexOf(self.consoleGroup)
        parent_layout.insertWidget(console_idx, box)
        self._sp_box = box

    def _scenario_extra_args(self) -> list[str]:
        if not self._sp_box.isChecked():
            return []
        args = [
            "--duration", str(self._sp_duration.value()),
            "--network", self._sp_network.currentText(),
            "--measurement_type", self._sp_meas.currentText(),
            "--estimator", self._sp_estimator.currentText(),
        ]
        if self._sp_noise.isChecked():
            args.append("--noise")
        return args

    def _on_script_changed(self) -> None:
        if not self._setting_preset:
            self._override_command = None
        idx = self.scriptCombo.currentIndex()
        if idx not in self._script_map and idx >= 0:
            # User selected a separator/header — advance to next valid item
            for i in range(idx + 1, self.scriptCombo.count()):
                if i in self._script_map:
                    self.scriptCombo.setCurrentIndex(i)
                    return
        self._update_command_preview()

    def _update_command_preview(self) -> None:
        sp_args = self._scenario_extra_args() if hasattr(self, "_sp_box") else []

        if self._override_command is not None:
            script, extra_args = self._override_command
            all_args = list(extra_args) + sp_args
            quoted_args = " ".join(_quote_arg(arg) for arg in all_args)
            cmd = f"python {script}"
            if quoted_args:
                cmd += f" {quoted_args}"
            self.commandPreviewLabel.setText(cmd)
            return

        idx = self.scriptCombo.currentIndex()
        rel = self._script_map.get(idx) if hasattr(self, "_script_map") else None
        if rel is None:
            # find next valid
            for i in range(idx + 1, self.scriptCombo.count()):
                if hasattr(self, "_script_map") and i in self._script_map:
                    rel = self._script_map[i]
                    break
        if rel is not None:
            cmd = f"python {rel}"
            if sp_args:
                cmd += " " + " ".join(_quote_arg(a) for a in sp_args)
            self.commandPreviewLabel.setText(cmd)

    def _get_current_args(self) -> tuple[str, list[str]]:
        sp_args = self._scenario_extra_args() if hasattr(self, "_sp_box") else []

        if self._override_command is not None:
            script, extra_args = self._override_command
            return sys.executable, [str(script)] + list(extra_args) + sp_args

        idx = self.scriptCombo.currentIndex()
        rel_path = self._script_map.get(idx) if hasattr(self, "_script_map") else None
        if rel_path is None:
            # Landed on a separator — find nearest valid script
            for i in range(idx + 1, self.scriptCombo.count()):
                if hasattr(self, "_script_map") and i in self._script_map:
                    rel_path = self._script_map[i]
                    break
            if rel_path is None:
                return sys.executable, []
        parts = rel_path.split()
        script = PYTHON_PORT / parts[0]
        extra_args = parts[1:]
        return sys.executable, [str(script)] + extra_args + sp_args

    def _start(self) -> None:
        if self._worker.is_running:
            return
        self._log.clear_log()
        program, args = self._get_current_args()
        self._log.append_info(f"Starting: {program} {' '.join(args)}")
        self._worker.run(program, args, working_dir=str(PYTHON_PORT.parent))

    def _cancel(self) -> None:
        if not self._worker.is_running:
            return
        reply = QMessageBox.question(
            self, "Cancel run?",
            "Kill the running process?\nPartial output files may remain.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._log.append_warning("User cancelled — killing process.")
            self._worker.cancel()

    def _on_started(self) -> None:
        self.startBtn.setEnabled(False)
        self.cancelBtn.setEnabled(True)
        self.openOutputBtn.setEnabled(False)
        self.progressBar.setVisible(True)
        self._set_status("Running…", C.YELLOW)

    def _on_finished(self, exit_code: int) -> None:
        self._last_exit_code = exit_code
        self.startBtn.setEnabled(True)
        self.cancelBtn.setEnabled(False)
        self.openOutputBtn.setEnabled(True)
        self.progressBar.setVisible(False)
        if exit_code == 0:
            self._log.append_success(f"Process exited successfully (code 0)")
            self._set_status("Completed successfully", C.GREEN)
        else:
            self._log.append_stderr(f"Process exited with code {exit_code}")
            self._set_status(f"Failed (exit code {exit_code})", C.RED)

    def _on_tick(self, elapsed: float) -> None:
        mins, secs = divmod(int(elapsed), 60)
        self.elapsedLabel.setText(f"Elapsed: {mins:02d}:{secs:02d}")

    def _set_status(self, text: str, color: str) -> None:
        self.statusLabel.setText(text)
        self.statusLabel.setStyleSheet(f"QLabel {{ color: {color}; }}")
        self.statusDot.setStyleSheet(f"QLabel {{ color: {color}; font-size: 16px; }}")

    def _open_output(self) -> None:
        try:
            import subprocess as sp
            from services.project_paths import RESULTS_DIR
            sp.Popen(f'explorer "{RESULTS_DIR}"', shell=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def preset_script(
        self,
        script_path: str,
        extra_args: list[str] | None = None,
        auto_start: bool = False,
    ) -> None:
        """Pre-select a script by path and optional arguments.

        The override path is important for Scenario Builder: the combo entry for
        scenario_config_cli intentionally shows ``--help`` for manual use, while
        scenario runs must pass the exported JSON path instead.
        """
        target = Path(script_path)
        if not target.is_absolute():
            target = PYTHON_PORT / target

        self._override_command = (target, list(extra_args or []))
        self._setting_preset = True
        try:
            for combo_idx, rel in self._script_map.items():
                if Path(rel.split()[0]).name == target.name:
                    self.scriptCombo.setCurrentIndex(combo_idx)
                    break
        finally:
            self._setting_preset = False
        self._update_command_preview()

        if auto_start:
            self._start()


def _quote_arg(arg: str) -> str:
    text = str(arg)
    if not text or any(ch.isspace() for ch in text):
        return f'"{text}"'
    return text
