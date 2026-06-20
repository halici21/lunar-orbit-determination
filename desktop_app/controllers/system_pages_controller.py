"""Configuration pages: Dynamics, Measurements, Estimators, Settings.

Each page provides real form controls that persist values via QSettings so other
pages (Scenario Builder, Analysis) can read the user's preferred defaults.
"""
from __future__ import annotations

from pathlib import Path

from PyQt5 import uic
from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QPushButton, QLineEdit,
    QScrollArea, QFileDialog, QFrame, QSizePolicy, QMessageBox,
)
from PyQt5.QtCore import Qt, QSettings

from services.project_paths import DESKTOP_APP_DIR, PYTHON_PORT, RESULTS_DIR
from styles.theme import C, FONT

_SETTINGS_ORG = "LunarOD"
_SETTINGS_APP = "DesktopApp"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_scroll_page(ui_file: str, parent=None) -> tuple[QWidget, QVBoxLayout]:
    """Load .ui, wrap content in a scroll area, return (top_widget, inner_layout).

    If parent is provided, a fill layout is added to parent so the returned
    widget stretches to fill the parent (required when parent is a QWidget
    added to a QStackedWidget without its own layout).
    """
    w = QWidget(parent)
    uic.loadUi(str(DESKTOP_APP_DIR / "ui" / "pages" / ui_file), w)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.NoFrame)
    scroll.setStyleSheet(
        f"QScrollArea {{ background:{C.BG_PANEL}; border:none; }}"
        f"QScrollBar:vertical {{ background:{C.BG_DEEP}; width:8px; border-radius:4px; }}"
        f"QScrollBar::handle:vertical {{ background:{C.BORDER_MID}; border-radius:4px; }}"
    )

    inner = QWidget()
    inner.setStyleSheet(f"background:{C.BG_PANEL};")
    lay = QVBoxLayout(inner)
    lay.setContentsMargins(24, 18, 24, 24)
    lay.setSpacing(16)
    scroll.setWidget(inner)

    outer_lay = w.mainLayout
    outer_lay.setContentsMargins(0, 0, 0, 0)
    outer_lay.addWidget(scroll)

    if parent is not None:
        fill_lay = QVBoxLayout(parent)
        fill_lay.setContentsMargins(0, 0, 0, 0)
        fill_lay.addWidget(w)

    return w, lay


def _group(title: str) -> tuple[QGroupBox, QFormLayout]:
    box = QGroupBox(title)
    box.setStyleSheet(
        f"QGroupBox {{ color:{C.TEXT_SECONDARY}; font-size:{FONT.SIZE_MD}px; font-weight:bold;"
        f" border:1px solid {C.BORDER_MAIN}; border-radius:6px; margin-top:8px;"
        f" padding-top:4px; }}"
        f"QGroupBox::title {{ subcontrol-origin:margin; left:12px; padding:0 4px;"
        f" color:{C.CYAN}; }}"
    )
    form = QFormLayout(box)
    form.setContentsMargins(14, 12, 14, 12)
    form.setSpacing(8)
    form.setLabelAlignment(Qt.AlignRight)
    return box, form


def _spin(lo: float, hi: float, val: float, suffix: str = "", decimals: int = 2) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setValue(val)
    if suffix:
        s.setSuffix(f" {suffix}")
    s.setStyleSheet(
        f"QDoubleSpinBox {{ background:{C.BG_DEEP}; color:{C.TEXT_PRIMARY};"
        f" border:1px solid {C.BORDER_MID}; border-radius:3px; padding:2px 6px;"
        f" font-size:{FONT.SIZE_SM}px; }}"
    )
    return s


def _ispin(lo: int, hi: int, val: int, suffix: str = "") -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    if suffix:
        s.setSuffix(f" {suffix}")
    s.setStyleSheet(
        f"QSpinBox {{ background:{C.BG_DEEP}; color:{C.TEXT_PRIMARY};"
        f" border:1px solid {C.BORDER_MID}; border-radius:3px; padding:2px 6px;"
        f" font-size:{FONT.SIZE_SM}px; }}"
    )
    return s


def _combo(options: list[str], current: str = "") -> QComboBox:
    c = QComboBox()
    for opt in options:
        c.addItem(opt)
    if current in options:
        c.setCurrentText(current)
    c.setStyleSheet(
        f"QComboBox {{ background:{C.BG_DEEP}; color:{C.TEXT_PRIMARY};"
        f" border:1px solid {C.BORDER_MID}; border-radius:3px; padding:2px 6px;"
        f" font-size:{FONT.SIZE_SM}px; }}"
        f"QComboBox::drop-down {{ border:none; }}"
    )
    return c


def _check(checked: bool = True) -> QCheckBox:
    cb = QCheckBox()
    cb.setChecked(checked)
    cb.setStyleSheet(f"QCheckBox {{ color:{C.TEXT_SECONDARY}; }}")
    return cb


def _label(text: str, muted: bool = False) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(
        f"color:{C.TEXT_MUTED if muted else C.TEXT_SECONDARY};"
        f" font-size:{FONT.SIZE_SM}px;"
    )
    return lbl


def _save_btn() -> QPushButton:
    btn = QPushButton("Save")
    btn.setFixedWidth(90)
    btn.setStyleSheet(
        f"QPushButton {{ background:{C.BLUE}; color:#fff; border-radius:4px;"
        f" font-size:{FONT.SIZE_SM}px; font-weight:bold; padding:5px 12px; }}"
        f"QPushButton:hover {{ background:{C.CYAN}; }}"
    )
    return btn


def _saved_lbl() -> QLabel:
    lbl = QLabel("")
    lbl.setStyleSheet(f"color:{C.GREEN}; font-size:{FONT.SIZE_SM}px;")
    return lbl


def _divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet(f"color:{C.BORDER_MAIN}; background:{C.BORDER_MAIN};")
    line.setFixedHeight(1)
    return line


# ---------------------------------------------------------------------------
# Dynamics page
# ---------------------------------------------------------------------------
class DynamicsController(QWidget):
    _SECTION = "dynamics"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._w, lay = _make_scroll_page("dynamics_page.ui", self)
        self.mainLayout = lay
        self._settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        self._build(lay)

    # public API so parent can access
    @property
    def widget(self) -> QWidget:
        return self._w

    def on_shown(self) -> None:
        pass

    def _build(self, lay: QVBoxLayout) -> None:
        lay.addWidget(_label(
            "Controls for orbit propagation, SPICE ephemeris, force models, "
            "and numerical solver tolerances.",
            muted=True,
        ))

        # --- Propagation ---
        box, form = _group("Propagation")
        self._dur = _spin(1.0, 28.0, self._get("duration_days", 3.0), "days", 1)
        form.addRow("Duration:", self._dur)
        self._step = _spin(10.0, 3600.0, self._get("step_s", 600.0), "s", 0)
        form.addRow("State step:", self._step)
        self._ephem_step = _spin(60.0, 7200.0, self._get("ephem_step_s", 3600.0), "s", 0)
        form.addRow("Ephemeris step:", self._ephem_step)
        lay.addWidget(box)

        # --- Solver tolerances ---
        box2, form2 = _group("Numerical Tolerances (DOP853)")
        self._rtol = _combo(["1e-6", "1e-7", "1e-8", "1e-9", "1e-10", "1e-11"],
                            self._gets("rtol", "1e-9"))
        form2.addRow("rtol:", self._rtol)
        self._atol = _combo(["1e-8", "1e-9", "1e-10", "1e-11", "1e-12"],
                            self._gets("atol", "1e-10"))
        form2.addRow("atol:", self._atol)
        lay.addWidget(box2)

        # --- Force model ---
        box3, form3 = _group("Force Model")
        self._j2 = _check(self._getb("use_j2", True))
        form3.addRow("Moon J₂ (lunisolar):", self._j2)
        self._third_body = _check(self._getb("third_body", True))
        form3.addRow("Earth/Sun third-body:", self._third_body)
        self._j2_mismatch = _check(self._getb("j2_mismatch", False))
        form3.addRow("J₂ mismatch (truth≠est):", self._j2_mismatch)
        lay.addWidget(box3)

        # --- SPICE kernel status ---
        box4, _ = _group("SPICE Kernel Status")
        try:
            from lunar_od.spice_loader import resolve_kernel_dir, REQUIRED_KERNELS
            kernel_dir = resolve_kernel_dir()
            kernels = sorted(kernel_dir.glob("*"))
            dir_lbl = QLabel(f"  {kernel_dir}")
            dir_lbl.setStyleSheet(f"color:{C.TEXT_MUTED}; font-size:{FONT.SIZE_SM - 1}px;")
            box4.layout().addRow("Directory:", dir_lbl)
            found, missing = [], []
            for name in REQUIRED_KERNELS:
                (found if (kernel_dir / name).exists() else missing).append(name)
            for name in found:
                lbl = QLabel(f"  ✓  {name}")
                lbl.setStyleSheet(f"color:{C.GREEN}; font-size:{FONT.SIZE_SM}px;")
                box4.layout().addRow("", lbl)
            for name in missing:
                lbl = QLabel(f"  ✗  {name}")
                lbl.setStyleSheet(f"color:{C.RED}; font-size:{FONT.SIZE_SM}px;")
                box4.layout().addRow("", lbl)
            extra = [k for k in kernels if k.name not in REQUIRED_KERNELS]
            if extra:
                more = QLabel(f"  …and {len(extra)} additional file(s)")
                more.setStyleSheet(f"color:{C.TEXT_MUTED}; font-size:{FONT.SIZE_SM}px;")
                box4.layout().addRow("", more)
        except FileNotFoundError as exc:
            warn = QLabel(f"  {exc}")
            warn.setWordWrap(True)
            warn.setStyleSheet(f"color:{C.RED}; font-size:{FONT.SIZE_SM}px;")
            box4.layout().addRow("", warn)
        lay.addWidget(box4)

        # Save row
        lay.addWidget(_divider())
        row = QHBoxLayout()
        btn = _save_btn()
        self._saved = _saved_lbl()
        btn.clicked.connect(self._save)
        row.addWidget(btn)
        row.addWidget(self._saved)
        row.addStretch()
        lay.addLayout(row)
        lay.addStretch()

    def _save(self) -> None:
        self._settings.setValue(f"{self._SECTION}/duration_days", self._dur.value())
        self._settings.setValue(f"{self._SECTION}/step_s", self._step.value())
        self._settings.setValue(f"{self._SECTION}/ephem_step_s", self._ephem_step.value())
        self._settings.setValue(f"{self._SECTION}/rtol", self._rtol.currentText())
        self._settings.setValue(f"{self._SECTION}/atol", self._atol.currentText())
        self._settings.setValue(f"{self._SECTION}/use_j2", self._j2.isChecked())
        self._settings.setValue(f"{self._SECTION}/third_body", self._third_body.isChecked())
        self._settings.setValue(f"{self._SECTION}/j2_mismatch", self._j2_mismatch.isChecked())
        self._saved.setText("Saved")

    def _get(self, key: str, default: float) -> float:
        return float(self._settings.value(f"{self._SECTION}/{key}", default))

    def _gets(self, key: str, default: str) -> str:
        return str(self._settings.value(f"{self._SECTION}/{key}", default))

    def _getb(self, key: str, default: bool) -> bool:
        v = self._settings.value(f"{self._SECTION}/{key}", default)
        return str(v).lower() not in ("false", "0", "")


# ---------------------------------------------------------------------------
# Measurements page
# ---------------------------------------------------------------------------
class MeasurementsController(QWidget):
    _SECTION = "measurements"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._w, lay = _make_scroll_page("measurements_page.ui", self)
        self.mainLayout = lay
        self._settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        self._build(lay)

    def on_shown(self) -> None:
        pass

    def _build(self, lay: QVBoxLayout) -> None:
        lay.addWidget(_label(
            "Configure observable type, noise assumptions, station biases, "
            "and Doppler count interval.",
            muted=True,
        ))

        # --- Observable ---
        box, form = _group("Observable Type")
        self._meas_type = _combo(
            ["range_rate", "position"],
            self._gets("measurement_type", "range_rate"),
        )
        form.addRow("Measurement type:", self._meas_type)
        self._rr_physics = _combo(
            ["geometric_instantaneous", "two_way_counted_doppler"],
            self._gets("range_rate_physics", "geometric_instantaneous"),
        )
        form.addRow("Range-rate model:", self._rr_physics)
        self._count_interval = _spin(0.1, 60.0, self._get("count_interval_s", 1.0), "s", 1)
        form.addRow("Doppler count interval:", self._count_interval)
        lay.addWidget(box)

        # --- Light time & stellar aberration (position observables only) ---
        box_ab, form_ab = _group("Light-Time & Aberration (position observables)")
        self._light_time = _check(self._getb("apply_light_time", False))
        form_ab.addRow("One-way light time (CN):", self._light_time)
        self._stellar = _check(self._getb("apply_stellar_aberration", False))
        form_ab.addRow("Stellar aberration (+S):", self._stellar)
        self._stellar_model = _combo(
            ["local_mci", "spice_ssb"],
            self._gets("stellar_aberration_model", "local_mci"),
        )
        form_ab.addRow("Aberration frame:", self._stellar_model)
        lay.addWidget(box_ab)

        # --- Noise ---
        box2, form2 = _group("Measurement Noise")
        self._noise_enabled = _check(self._getb("noise_enabled", True))
        form2.addRow("Enabled:", self._noise_enabled)
        self._range_sigma = _spin(0.1, 1000.0, self._get("range_sigma_m", 5.0), "m", 1)
        form2.addRow("Range σ:", self._range_sigma)
        self._rr_sigma = _spin(0.001, 1.0, self._get("rr_sigma_m_s", 0.001), "m/s", 4)
        form2.addRow("Range-rate σ:", self._rr_sigma)
        lay.addWidget(box2)

        # --- Bias ---
        box3, form3 = _group("Station Bias Model")
        self._bias_mode = _combo(
            ["none", "range_bias", "range_rate_bias", "common"],
            self._gets("bias_mode", "none"),
        )
        form3.addRow("Bias mode:", self._bias_mode)
        self._bias_sigma = _spin(0.0, 100.0, self._get("bias_sigma_m", 0.0), "m", 1)
        form3.addRow("Initial bias σ:", self._bias_sigma)
        lay.addWidget(box3)

        # Save
        lay.addWidget(_divider())
        row = QHBoxLayout()
        btn = _save_btn()
        self._saved = _saved_lbl()
        btn.clicked.connect(self._save)
        row.addWidget(btn)
        row.addWidget(self._saved)
        row.addStretch()
        lay.addLayout(row)
        lay.addStretch()

    def _save(self) -> None:
        self._settings.setValue(f"{self._SECTION}/measurement_type", self._meas_type.currentText())
        self._settings.setValue(f"{self._SECTION}/range_rate_physics", self._rr_physics.currentText())
        self._settings.setValue(f"{self._SECTION}/count_interval_s", self._count_interval.value())
        self._settings.setValue(f"{self._SECTION}/noise_enabled", self._noise_enabled.isChecked())
        self._settings.setValue(f"{self._SECTION}/range_sigma_m", self._range_sigma.value())
        self._settings.setValue(f"{self._SECTION}/rr_sigma_m_s", self._rr_sigma.value())
        self._settings.setValue(f"{self._SECTION}/bias_mode", self._bias_mode.currentText())
        self._settings.setValue(f"{self._SECTION}/bias_sigma_m", self._bias_sigma.value())
        self._settings.setValue(f"{self._SECTION}/apply_light_time", self._light_time.isChecked())
        self._settings.setValue(f"{self._SECTION}/apply_stellar_aberration", self._stellar.isChecked())
        self._settings.setValue(f"{self._SECTION}/stellar_aberration_model", self._stellar_model.currentText())
        self._saved.setText("Saved")

    def _get(self, key: str, default: float) -> float:
        return float(self._settings.value(f"{self._SECTION}/{key}", default))

    def _gets(self, key: str, default: str) -> str:
        return str(self._settings.value(f"{self._SECTION}/{key}", default))

    def _getb(self, key: str, default: bool) -> bool:
        v = self._settings.value(f"{self._SECTION}/{key}", default)
        return str(v).lower() not in ("false", "0", "")


# ---------------------------------------------------------------------------
# Estimators page
# ---------------------------------------------------------------------------
class EstimatorsController(QWidget):
    _SECTION = "estimators"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._w, lay = _make_scroll_page("estimators_page.ui", self)
        self.mainLayout = lay
        self._settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        self._build(lay)

    def on_shown(self) -> None:
        pass

    def _build(self, lay: QVBoxLayout) -> None:
        lay.addWidget(_label(
            "Centralised tuning for BLS-LM, SRIF, and SR-UKF. "
            "Values saved here become defaults in Scenario Builder and Analysis.",
            muted=True,
        ))

        # --- Common ---
        box0, form0 = _group("Common")
        self._start_mode = _combo(
            ["cold", "hot", "formal"],
            self._gets("start_mode", "cold"),
        )
        form0.addRow("Start mode:", self._start_mode)
        lay.addWidget(box0)

        # --- BLS-LM ---
        box, form = _group("BLS-LM (Batch Least Squares)")
        self._bls_max_iter = _ispin(1, 200, int(self._get("bls_max_iter", 40)), "iter")
        form.addRow("Max iterations:", self._bls_max_iter)
        self._bls_damping = _spin(1e-8, 1e-2, self._get("bls_damping", 1e-4), "", 8)
        form.addRow("LM damping λ₀:", self._bls_damping)
        self._bls_tol = _combo(
            ["1e-6", "1e-7", "1e-8", "1e-9", "1e-10"],
            self._gets("bls_tol", "1e-8"),
        )
        form.addRow("Cost tolerance:", self._bls_tol)
        self._robust = _check(self._getb("robust_rejection", False))
        form.addRow("Robust outlier rejection:", self._robust)
        lay.addWidget(box)

        # --- UKF ---
        box2, form2 = _group("SR-UKF (Square-Root Unscented Kalman Filter)")
        self._alpha = _spin(1e-4, 1.0, self._get("ukf_alpha", 0.35), "", 4)
        form2.addRow("α (spread):", self._alpha)
        self._beta = _spin(0.0, 10.0, self._get("ukf_beta", 2.0), "", 1)
        form2.addRow("β (kurtosis weight):", self._beta)
        self._kappa = _spin(-5.0, 5.0, self._get("ukf_kappa", 0.0), "", 1)
        form2.addRow("κ (scaling):", self._kappa)
        self._pn_model = _combo(
            ["discrete", "continuous_white_acceleration"],
            self._gets("ukf_process_noise_model", "discrete"),
        )
        form2.addRow("Process noise model:", self._pn_model)
        self._pn_psd = _spin(0.0, 1e-6, self._get("ukf_psd", 1e-12), "m²/s³", 14)
        form2.addRow("Acceleration PSD:", self._pn_psd)
        self._adapt_q = _check(self._getb("ukf_adaptive_q", False))
        form2.addRow("Adaptive process noise:", self._adapt_q)
        self._adapt_r = _check(self._getb("ukf_adaptive_r", False))
        form2.addRow("Adaptive meas noise:", self._adapt_r)
        self._gate_sigma = _spin(1.0, 20.0, self._get("ukf_gate_sigma", 5.0), "σ", 1)
        form2.addRow("Innovation gate:", self._gate_sigma)
        lay.addWidget(box2)

        # Save
        lay.addWidget(_divider())
        row = QHBoxLayout()
        btn = _save_btn()
        self._saved = _saved_lbl()
        btn.clicked.connect(self._save)
        row.addWidget(btn)
        row.addWidget(self._saved)
        row.addStretch()
        lay.addLayout(row)
        lay.addStretch()

    def _save(self) -> None:
        s = self._settings
        p = self._SECTION
        s.setValue(f"{p}/start_mode", self._start_mode.currentText())
        s.setValue(f"{p}/bls_max_iter", self._bls_max_iter.value())
        s.setValue(f"{p}/bls_damping", self._bls_damping.value())
        s.setValue(f"{p}/bls_tol", self._bls_tol.currentText())
        s.setValue(f"{p}/robust_rejection", self._robust.isChecked())
        s.setValue(f"{p}/ukf_alpha", self._alpha.value())
        s.setValue(f"{p}/ukf_beta", self._beta.value())
        s.setValue(f"{p}/ukf_kappa", self._kappa.value())
        s.setValue(f"{p}/ukf_process_noise_model", self._pn_model.currentText())
        s.setValue(f"{p}/ukf_psd", self._pn_psd.value())
        s.setValue(f"{p}/ukf_adaptive_q", self._adapt_q.isChecked())
        s.setValue(f"{p}/ukf_adaptive_r", self._adapt_r.isChecked())
        s.setValue(f"{p}/ukf_gate_sigma", self._gate_sigma.value())
        self._saved.setText("Saved")

    def _get(self, key: str, default: float) -> float:
        return float(self._settings.value(f"{self._SECTION}/{key}", default))

    def _gets(self, key: str, default: str) -> str:
        return str(self._settings.value(f"{self._SECTION}/{key}", default))

    def _getb(self, key: str, default: bool) -> bool:
        v = self._settings.value(f"{self._SECTION}/{key}", default)
        return str(v).lower() not in ("false", "0", "")


# ---------------------------------------------------------------------------
# Settings page
# ---------------------------------------------------------------------------
class SettingsController(QWidget):
    _SECTION = "app_settings"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._w, lay = _make_scroll_page("settings_page.ui", self)
        self.mainLayout = lay
        self._settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        self._build(lay)

    def on_shown(self) -> None:
        pass

    def _build(self, lay: QVBoxLayout) -> None:
        lay.addWidget(_label(
            "Application paths, theme options, runtime profiles, and reproducibility controls.",
            muted=True,
        ))

        # --- Paths ---
        box, form = _group("Paths")

        self._results_edit = QLineEdit(str(RESULTS_DIR))
        self._results_edit.setReadOnly(True)
        self._results_edit.setStyleSheet(
            f"QLineEdit {{ background:{C.BG_DEEP}; color:{C.TEXT_SECONDARY};"
            f" border:1px solid {C.BORDER_MID}; border-radius:3px; padding:2px 6px;"
            f" font-size:{FONT.SIZE_SM}px; }}"
        )
        browse_results = QPushButton("Browse…")
        browse_results.setFixedWidth(80)
        browse_results.setStyleSheet(
            f"QPushButton {{ background:{C.BG_CARD}; color:{C.TEXT_SECONDARY};"
            f" border:1px solid {C.BORDER_MID}; border-radius:3px; font-size:{FONT.SIZE_SM}px; }}"
        )
        results_row = QHBoxLayout()
        results_row.addWidget(self._results_edit)
        results_row.addWidget(browse_results)
        form.addRow("Results directory:", results_row)
        browse_results.clicked.connect(self._browse_results)

        try:
            from lunar_od.spice_loader import resolve_kernel_dir
            _kdir = str(resolve_kernel_dir())
        except Exception:
            _kdir = str(PYTHON_PORT / "kernels")
        self._kernels_edit = QLineEdit(_kdir)
        self._kernels_edit.setReadOnly(True)
        self._kernels_edit.setStyleSheet(self._results_edit.styleSheet())
        browse_kernels = QPushButton("Browse…")
        browse_kernels.setFixedWidth(80)
        browse_kernels.setStyleSheet(browse_results.styleSheet())
        kernels_row = QHBoxLayout()
        kernels_row.addWidget(self._kernels_edit)
        kernels_row.addWidget(browse_kernels)
        form.addRow("SPICE kernels:", kernels_row)
        browse_kernels.clicked.connect(self._browse_kernels)

        lay.addWidget(box)

        # --- Runtime ---
        box2, form2 = _group("Runtime")
        self._seed = _ispin(0, 99999, int(self._get("random_seed", 42)))
        form2.addRow("Random seed:", self._seed)
        self._n_workers = _ispin(1, 16, int(self._get("n_workers", 1)))
        form2.addRow("Worker threads:", self._n_workers)
        lay.addWidget(box2)

        # --- Environment info ---
        box3, _ = _group("Environment")
        import sys
        info = [
            ("Python", sys.version.split()[0]),
            ("PyQt5", self._pkg_version("PyQt5")),
            ("numpy", self._pkg_version("numpy")),
            ("matplotlib", self._pkg_version("matplotlib")),
            ("spiceypy", self._pkg_version("spiceypy")),
        ]
        for lib, ver in info:
            lbl = QLabel(f"{ver}")
            lbl.setStyleSheet(
                f"color:{C.GREEN if ver != 'N/A' else C.RED}; font-size:{FONT.SIZE_SM}px;"
                f" font-family:{FONT.MONO};"
            )
            box3.layout().addRow(f"{lib}:", lbl)
        lay.addWidget(box3)

        # Save
        lay.addWidget(_divider())
        row = QHBoxLayout()
        btn = _save_btn()
        self._saved = _saved_lbl()
        btn.clicked.connect(self._save)
        row.addWidget(btn)
        row.addWidget(self._saved)
        row.addStretch()
        lay.addLayout(row)
        lay.addStretch()

    def _save(self) -> None:
        self._settings.setValue(f"{self._SECTION}/random_seed", self._seed.value())
        self._settings.setValue(f"{self._SECTION}/n_workers", self._n_workers.value())
        self._saved.setText("Saved")

    def _browse_results(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select results directory",
                                                self._results_edit.text())
        if path:
            self._results_edit.setText(path)

    def _browse_kernels(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select SPICE kernels directory",
                                                self._kernels_edit.text())
        if path:
            self._kernels_edit.setText(path)

    @staticmethod
    def _pkg_version(name: str) -> str:
        try:
            import importlib.metadata
            return importlib.metadata.version(name)
        except Exception:
            return "N/A"

    def _get(self, key: str, default: float) -> float:
        return float(self._settings.value(f"{self._SECTION}/{key}", default))
