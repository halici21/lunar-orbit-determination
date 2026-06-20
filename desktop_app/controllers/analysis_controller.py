"""Estimator Analysis page — BLS-LM and UKF comparison panels with live in-app plotting."""
from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTabWidget, QLabel, QPushButton, QPlainTextEdit,
    QFormLayout, QGroupBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QSizePolicy, QProgressBar,
    QScrollArea, QMessageBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5 import uic

from services.project_paths import DESKTOP_APP_DIR, PYTHON_PORT
from styles.theme import C, FONT, MPL_DARK, BAR_COLORS

UI_PATH = DESKTOP_APP_DIR / "ui" / "pages" / "analysis_page.ui"
FIXTURE  = PYTHON_PORT / "fixtures" / "spice_snapshots.json"

ANALYSIS_STATIONS: tuple[str, ...] = (
    "ITU Ayazaga",
    "Goldstone DSN",
    "Madrid DSN",
    "Canberra DSN",
    "Daejeon KGS",
    "Dongara KGS",
    "Chuuk KGS",
    "Svalbard KGS",
    "Malargue ESA",
    "Cebreros ESA",
    "New Norcia ESA",
    "Evpatoria RUS",
    "Ussuriisk RUS",
    "Bear Lakes RUS",
    "Byalalu ISRO",
)

DEFAULT_ANALYSIS_STATIONS: tuple[str, ...] = (
    "Goldstone DSN",
    "Madrid DSN",
    "Canberra DSN",
    "ITU Ayazaga",
)

STATION_PRESETS: tuple[tuple[str, tuple[str, ...] | None], ...] = (
    ("Thesis multi (DSN + ITU)", DEFAULT_ANALYSIS_STATIONS),
    ("DSN triplet", ("Goldstone DSN", "Madrid DSN", "Canberra DSN")),
    ("ITU only", ("ITU Ayazaga",)),
    ("Canberra only", ("Canberra DSN",)),
    ("All available", ANALYSIS_STATIONS),
    ("Custom", None),
)

# ---------------------------------------------------------------------------
# Comparison specifications
# ---------------------------------------------------------------------------

@dataclass
class VariantSpec:
    label: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonSpec:
    title: str
    desc: str
    estimator: str          # "bls_lm" or "ukf"
    variants: list[VariantSpec]


# Common base parameters for all analyses
_BASE = dict(
    duration_h=72.0,
    sample_step_s=600.0,
    max_iter=40,
    noise=True,
    network="multi",
    start_mode="cold",
    measurement_type="range_rate",
    range_rate_physics="geometric_instantaneous",
    bias_mode=None,
    output_dir="python_port/results",
)

# UKF is an onboard sequential filter — state persists between passes; use formal
# handoff (propagate both state and covariance) as the natural operating mode.
# Only the "Start Mode" comparison tab overrides this via variant overrides.
_UKF_BASE = {**_BASE, "start_mode": "formal"}


def _base_from_settings(estimator: str = "bls_lm") -> dict:
    """Build base params dict, preferring saved QSettings values over hardcoded defaults."""
    from PyQt5.QtCore import QSettings
    s = QSettings("LunarOD", "DesktopApp")

    def _f(key, default):
        return float(s.value(key, default))

    def _s(key, default):
        return str(s.value(key, default))

    def _b(key, default):
        return str(s.value(key, default)).lower() not in ("false", "0", "")

    dur_days = _f("dynamics/duration_days", 3.0)
    _bias_raw = _s("measurements/bias_mode", "none")
    base = dict(
        duration_h=dur_days * 24.0,
        sample_step_s=_f("dynamics/step_s", 600.0),
        max_iter=int(_f("estimators/bls_max_iter", 40)),
        tol_cost_stability=float(_s("estimators/bls_tol", "1e-8")),
        bls_lambda0=_f("estimators/bls_damping", 1e-4),
        rtol=float(_s("dynamics/rtol", "1e-11")),
        atol=float(_s("dynamics/atol", "1e-12")),
        j2_moon=2.0346e-4 if _b("dynamics/use_j2", False) else 0.0,
        noise=_b("measurements/noise_enabled", True),
        network="multi",
        start_mode=_s("estimators/start_mode", "cold"),
        measurement_type=_s("measurements/measurement_type", "range_rate"),
        range_rate_physics=_s("measurements/range_rate_physics", "geometric_instantaneous"),
        count_interval_s=_f("measurements/count_interval_s", 1.0),
        bias_mode=None if _bias_raw == "none" else _bias_raw,
        apply_light_time=_b("measurements/apply_light_time", False),
        apply_stellar_aberration=_b("measurements/apply_stellar_aberration", False),
        stellar_aberration_model=_s("measurements/stellar_aberration_model", "local_mci"),
        output_dir="python_port/results",
    )
    if estimator == "ukf":
        base["start_mode"] = "formal"  # onboard sequential filter
    return base


def _ukf_overrides_from_settings() -> dict:
    """Extra UKF tuning params from QSettings (fall back to spec overrides)."""
    from PyQt5.QtCore import QSettings
    s = QSettings("LunarOD", "DesktopApp")

    def _f(key, default):
        return float(s.value(key, default))

    def _s(key, default):
        return str(s.value(key, default))

    def _b(key, default):
        return str(s.value(key, default)).lower() not in ("false", "0", "")

    return {
        "ukf_alpha": _f("estimators/ukf_alpha", 0.35),
        "ukf_beta":  _f("estimators/ukf_beta",  2.0),
        "ukf_kappa": _f("estimators/ukf_kappa", 0.0),
        "ukf_process_noise_model": _s("estimators/ukf_process_noise_model", "discrete"),
        "ukf_acceleration_psd_m2_s3": _f("estimators/ukf_psd", 1e-12),
        "ukf_adaptive_process_noise": _b("estimators/ukf_adaptive_q", False),
        "ukf_adaptive_measurement_noise": _b("estimators/ukf_adaptive_r", False),
        "ukf_innovation_gate_sigma": _f("estimators/ukf_gate_sigma", 5.0),
    }


BLS_COMPARISONS: list[ComparisonSpec] = [
    ComparisonSpec(
        title="Observable Type",
        desc="Range-only (position) vs Range+Rate — effect of observable set on position accuracy",
        estimator="bls_lm",
        variants=[
            VariantSpec("Range-only",  {"measurement_type": "position"}),
            VariantSpec("Range+Rate",  {"measurement_type": "range_rate"}),
        ],
    ),
    ComparisonSpec(
        title="Start Mode",
        desc="Effect of cold, hot, and formal initialization on per-arc position error",
        estimator="bls_lm",
        variants=[
            VariantSpec("Cold",   {"start_mode": "cold"}),
            VariantSpec("Hot",    {"start_mode": "hot"}),
            VariantSpec("Formal", {"start_mode": "formal"}),
        ],
    ),
    ComparisonSpec(
        title="LM Iterations",
        desc="Effect of Levenberg-Marquardt maximum iteration count on convergence quality",
        estimator="bls_lm",
        variants=[
            VariantSpec("5 iter",  {"max_iter": 5}),
            VariantSpec("10 iter", {"max_iter": 10}),
            VariantSpec("20 iter", {"max_iter": 20}),
            VariantSpec("40 iter", {"max_iter": 40}),
        ],
    ),
    ComparisonSpec(
        title="Measurement Noise",
        desc="BLS performance with and without measurement noise",
        estimator="bls_lm",
        variants=[
            VariantSpec("Noiseless", {"noise": False}),
            VariantSpec("Noisy",     {"noise": True}),
        ],
    ),
]


UKF_COMPARISONS: list[ComparisonSpec] = [
    ComparisonSpec(
        title="Observable Type",
        desc="Range-only (position) vs Range+Rate — UKF version",
        estimator="ukf",
        variants=[
            VariantSpec("Range-only", {"measurement_type": "position"}),
            VariantSpec("Range+Rate", {"measurement_type": "range_rate"}),
        ],
    ),
    ComparisonSpec(
        title="Start Mode",
        desc="Effect of cold, hot, and formal initialization on UKF position error",
        estimator="ukf",
        variants=[
            VariantSpec("Cold",   {"start_mode": "cold"}),
            VariantSpec("Hot",    {"start_mode": "hot"}),
            VariantSpec("Formal", {"start_mode": "formal"}),
        ],
    ),
    ComparisonSpec(
        title="Adaptive Q & R",
        desc="Effect of adaptive process (Q) and measurement (R) noise on filter stability",
        estimator="ukf",
        variants=[
            VariantSpec(
                "Fixed Q+R",
                {"ukf_adaptive_process_noise": False,
                 "ukf_adaptive_measurement_noise": False},
            ),
            VariantSpec(
                "Adaptive Q",
                {"ukf_adaptive_process_noise": True,
                 "ukf_adaptive_measurement_noise": False,
                 "ukf_process_noise_model": "continuous_white_acceleration",
                 "ukf_acceleration_psd_m2_s3": 1e-12},
            ),
            VariantSpec(
                "Adaptive R",
                {"ukf_adaptive_process_noise": False,
                 "ukf_adaptive_measurement_noise": True},
            ),
            VariantSpec(
                "Adaptive Q+R",
                {"ukf_adaptive_process_noise": True,
                 "ukf_adaptive_measurement_noise": True,
                 "ukf_process_noise_model": "continuous_white_acceleration",
                 "ukf_acceleration_psd_m2_s3": 1e-12},
            ),
        ],
    ),
    ComparisonSpec(
        title="α Sensitivity",
        desc="Effect of sigma-point spread parameter α on position error (β=2, κ=0 fixed)",
        estimator="ukf",
        variants=[
            VariantSpec("α=0.001", {"ukf_alpha": 0.001}),
            VariantSpec("α=0.01",  {"ukf_alpha": 0.01}),
            VariantSpec("α=0.1",   {"ukf_alpha": 0.1}),
            VariantSpec("α=0.35",  {"ukf_alpha": 0.35}),
            VariantSpec("α=1.0",   {"ukf_alpha": 1.0}),
        ],
    ),
    ComparisonSpec(
        title="β Sensitivity",
        desc="Effect of prior kurtosis weight β on position error (α=0.35, κ=0 fixed)",
        estimator="ukf",
        variants=[
            VariantSpec("β=0",  {"ukf_beta": 0.0}),
            VariantSpec("β=1",  {"ukf_beta": 1.0}),
            VariantSpec("β=2",  {"ukf_beta": 2.0}),
            VariantSpec("β=4",  {"ukf_beta": 4.0}),
        ],
    ),
    ComparisonSpec(
        title="κ Sensitivity",
        desc="Effect of scaling parameter κ on position error (α=0.35, β=2 fixed)",
        estimator="ukf",
        variants=[
            VariantSpec("κ=−2", {"ukf_kappa": -2.0}),
            VariantSpec("κ=−1", {"ukf_kappa": -1.0}),
            VariantSpec("κ=0",  {"ukf_kappa":  0.0}),
            VariantSpec("κ=1",  {"ukf_kappa":  1.0}),
            VariantSpec("κ=2",  {"ukf_kappa":  2.0}),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _AnalysisWorker(QThread):
    log_line      = pyqtSignal(str)
    variant_done  = pyqtSignal(str, object)  # (label, ScenarioResult)
    all_done      = pyqtSignal()
    cancelled     = pyqtSignal()
    failed        = pyqtSignal(str)

    def __init__(self, spec: ComparisonSpec, base_params: dict) -> None:
        super().__init__()
        self._spec  = spec
        self._base  = base_params
        self._stop  = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            self._do_run()
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")

    def _do_run(self) -> None:
        import spiceypy as spice
        from lunar_od import (
            VisibilityConfig,
            analyze_visibility_gap_with_transforms,
            build_measurement_arcs,
            load_spice_kernels,
            make_cold_start_bank,
            propagate_truth_with_ephemeris,
            range_rate_stations,
            run_batch_arc_sequence,
            sample_j2000_to_itrf93_transforms,
            sample_moon_centered_ephemeris,
            scenario_config_from_mapping,
            scenario_range_rate_physics_config,
            scenario_ukf_configs,
            thesis_network_by_name,
            thesis_seed_for,
        )
        from lunar_od.thesis_matrix import (
            THESIS_COLD_START_SIGMA_POS_M,
            THESIS_COLD_START_SIGMA_VEL_MPS,
            THESIS_EPHEMERIS_STEP_S,
            THESIS_MAX_GAP_S,
            THESIS_MIN_ELEVATION_DEG,
        )

        self.log_line.emit("Loading fixture…")
        fixture   = json.loads(FIXTURE.read_text(encoding="utf-8"))
        initial   = fixture["initial_state"]
        constants = fixture["constants"]
        epoch_utc = fixture["epoch_utc"]

        x0      = np.asarray(initial["state_mci_j2000_m_mps"], dtype=float)
        mu_moon = float(initial["mu_moon_m3_s2"])
        mu_earth = float(np.asarray(constants["mu_earth_km3_s2"]).reshape(-1)[0] * 1e9)
        mu_sun   = float(np.asarray(constants["mu_sun_km3_s2"]).reshape(-1)[0] * 1e9)
        r_moon   = float(initial["r_moon_mean_m"])

        dur_h   = float(self._base.get("duration_h", 72.0))
        t_eval  = np.arange(0.0, dur_h * 3600.0 + 600.0, 600.0)
        t_ephem = np.arange(0.0, dur_h * 3600.0 + THESIS_EPHEMERIS_STEP_S, THESIS_EPHEMERIS_STEP_S)

        self.log_line.emit("Loading SPICE kernels…")
        load_spice_kernels()
        try:
            et0 = float(spice.str2et(epoch_utc))

            self.log_line.emit("Propagating reference trajectory…")
            _rtol = float(self._base.get("rtol", 1e-11))
            _atol = float(self._base.get("atol", 1e-12))
            _j2   = float(self._base.get("j2_moon", 0.0))
            ephemeris = sample_moon_centered_ephemeris(et0, t_ephem)
            xforms    = sample_j2000_to_itrf93_transforms(et0, t_eval)
            truth     = propagate_truth_with_ephemeris(
                t_eval, x0, mu_moon, mu_earth, mu_sun, ephemeris,
                rtol=_rtol, atol=_atol, j2_moon=_j2,
            )

            all_stns = {s.name: s for s in range_rate_stations()}
            selected_station_names = tuple(self._base.get("station_names") or ())
            if not selected_station_names:
                selected_station_names = thesis_network_by_name(
                    str(self._base.get("network", "multi"))
                ).station_names

            missing = [name for name in selected_station_names if name not in all_stns]
            if missing:
                raise ValueError(
                    "Selected stations not found in catalogue: "
                    + ", ".join(missing)
                )

            selected_stations = [all_stns[name] for name in selected_station_names]
            self.log_line.emit(
                "Stations: " + ", ".join(selected_station_names)
            )

            vis_config = VisibilityConfig(
                r_moon_mean_m=r_moon,
                earth_rotation_rad_s=7.292115e-5,
                epoch_utc=epoch_utc,
                min_elevation_deg=THESIS_MIN_ELEVATION_DEG,
            )

            # Arc cache: keyed by (measurement_type, network, noise) so we only rebuild
            # arcs when measurement-level parameters actually change between variants.
            arc_cache: dict[tuple, Any] = {}

            n_total = len(self._spec.variants)
            for vi, variant in enumerate(self._spec.variants):
                if self._stop:
                    self.log_line.emit("Stopped.")
                    break

                self.log_line.emit(f"[{vi + 1}/{n_total}] Variant: {variant.label}…")

                params = {**self._base, **variant.overrides}
                params["name"]           = f"analysis_{self._spec.title}_{variant.label}"
                params["estimator_type"] = self._spec.estimator
                # light-time / stellar aberration apply only to position observables,
                # and stellar requires light time — sanitize so variant overrides that
                # switch the observable type cannot trip the config cross-field rules.
                if params.get("measurement_type") != "position":
                    params["apply_light_time"] = False
                if not params.get("apply_light_time", False):
                    params["apply_stellar_aberration"] = False

                try:
                    config = scenario_config_from_mapping(params)
                except Exception as exc:
                    self.log_line.emit(f"  Config error — {exc}")
                    continue

                stations = selected_stations

                arc_key = (
                    config.measurement_type,
                    selected_station_names,
                    config.noise,
                    config.range_rate_physics,
                    config.apply_light_time,
                    config.apply_stellar_aberration,
                    config.stellar_aberration_model,
                )
                if arc_key not in arc_cache:
                    self.log_line.emit(f"  Building visibility arcs ({config.measurement_type})…")
                    seg_starts, seg_ends, vis_mask_raw, _ = analyze_visibility_gap_with_transforms(
                        t_eval, truth, stations,
                        ephemeris.earth_position, xforms,
                        THESIS_MAX_GAP_S, vis_config,
                    )
                    arcs = build_measurement_arcs(
                        config.measurement_type, t_eval, truth,
                        seg_starts, seg_ends, vis_mask_raw, stations,
                        ephemeris.earth_position, ephemeris.earth_velocity, et0,
                        noise=config.noise, rng=None, min_samples=4,
                        range_rate_physics=scenario_range_rate_physics_config(config),
                        apply_light_time=config.apply_light_time,
                        apply_stellar_aberration=config.apply_stellar_aberration,
                        stellar_aberration_model=config.stellar_aberration_model,
                    )
                    arc_cache[arc_key] = arcs
                else:
                    self.log_line.emit("  (arcs from cache)")
                    arcs = arc_cache[arc_key]

                if not arcs:
                    self.log_line.emit("  Warning: no visible arcs — skipping")
                    continue

                _seed = int(float(
                    QSettings("LunarOD", "DesktopApp").value("app_settings/random_seed", 42)
                ))
                cold_bank = make_cold_start_bank(
                    len(arcs),
                    THESIS_COLD_START_SIGMA_POS_M, THESIS_COLD_START_SIGMA_VEL_MPS,
                    seed=_seed,
                )
                ukf_tf, ukf_adaptive = scenario_ukf_configs(config)

                result = run_batch_arc_sequence(
                    arcs,
                    config.measurement_type,
                    config.start_mode,
                    config.estimator_type,
                    mu_moon, mu_earth, mu_sun,
                    ephemeris.earth_position,
                    ephemeris.sun_position,
                    cold_start_bank=cold_bank,
                    label=variant.label,
                    max_iter=config.max_iter,
                    tol_cost_stability=config.tol_cost_stability,
                    bls_lambda0=config.bls_lambda0,
                    bias_mode=config.bias_mode,
                    rtol=config.rtol,
                    atol=config.atol,
                    ukf_transform_config=ukf_tf,
                    ukf_adaptive_config=ukf_adaptive,
                    ukf_covariance_form=config.ukf_covariance_form,
                    ukf_process_noise_model=config.ukf_process_noise_model,
                    ukf_auto_bias_constraints=config.ukf_auto_bias_constraints,
                    ukf_bias_freeze_relative_information=config.ukf_bias_freeze_relative_information,
                    ukf_bias_regularize_relative_information=config.ukf_bias_regularize_relative_information,
                    ukf_bias_regularization_std=config.ukf_bias_regularization_std,
                    j2_moon=config.j2_moon,
                )

                n_arcs = len(result.arc_results)
                med = float(np.median(result.final_position_errors_m)) if n_arcs else 0.0
                self.log_line.emit(f"  → {n_arcs} arcs, median error: {med:.1f} m")
                self.variant_done.emit(variant.label, result)

        finally:
            spice.kclear()

        if self._stop:
            self.cancelled.emit()
        else:
            self.all_done.emit()


# ---------------------------------------------------------------------------
# Comparison panel — shared layout for every sub-tab
# ---------------------------------------------------------------------------

class _ComparisonPanel(QWidget):
    def __init__(self, spec: ComparisonSpec, parent=None) -> None:
        super().__init__(parent)
        self._spec    = spec
        self._worker: _AnalysisWorker | None = None
        self._results: list[tuple[str, Any]] = []
        self._canvas  = None
        self._fig     = None
        self._station_checks: dict[str, QCheckBox] = {}
        self._applying_station_preset = False
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Left control panel ──────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(315)
        left.setStyleSheet(
            f"background:{C.BG_SIDEBAR}; border-right:1px solid {C.BORDER_MAIN};"
        )
        ll = QVBoxLayout(left)
        ll.setContentsMargins(12, 14, 12, 14)
        ll.setSpacing(10)

        # Description
        desc = QLabel(self._spec.desc)
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color:{C.TEXT_MUTED}; font-size:{FONT.SIZE_SM}px;")
        ll.addWidget(desc)

        # Variants list
        var_box = QGroupBox("Variants")
        var_box.setStyleSheet(self._group_style())
        var_layout = QVBoxLayout(var_box)
        var_layout.setContentsMargins(8, 8, 8, 8)
        var_layout.setSpacing(3)
        for i, v in enumerate(self._spec.variants):
            color = BAR_COLORS[i % len(BAR_COLORS)]
            vl = QLabel(f"● {v.label}")
            vl.setStyleSheet(
                f"color:{color}; font-size:{FONT.SIZE_SM}px; font-family:{FONT.MONO};"
            )
            var_layout.addWidget(vl)
        ll.addWidget(var_box)

        # Params form
        params_box = QGroupBox("Common Parameters")
        params_box.setStyleSheet(self._group_style())
        pfl = QFormLayout(params_box)
        pfl.setContentsMargins(8, 8, 8, 10)
        pfl.setSpacing(6)
        pfl.setLabelAlignment(Qt.AlignRight)

        self._dur_spin = QDoubleSpinBox()
        self._dur_spin.setRange(0.5, 7.0)
        self._dur_spin.setSingleStep(0.5)
        self._dur_spin.setValue(3.0)
        self._dur_spin.setSuffix(" days")
        self._dur_spin.setStyleSheet(self._spin_style())
        pfl.addRow("Duration:", self._dur_spin)

        self._net_combo = QComboBox()
        for label, station_names in STATION_PRESETS:
            self._net_combo.addItem(label, station_names)
        self._net_combo.setStyleSheet(self._combo_style())
        pfl.addRow("Preset:", self._net_combo)

        self._noise_cb = QCheckBox("Enabled")
        self._noise_cb.setChecked(True)
        self._noise_cb.setStyleSheet(
            f"QCheckBox {{ color:{C.TEXT_SECONDARY}; font-size:{FONT.SIZE_SM}px; }}"
        )
        pfl.addRow("Noise:", self._noise_cb)

        ll.addWidget(params_box)

        station_box = QGroupBox("Station Selection")
        station_box.setStyleSheet(self._group_style())
        station_outer = QVBoxLayout(station_box)
        station_outer.setContentsMargins(8, 8, 8, 8)
        station_outer.setSpacing(6)

        self._station_count_lbl = QLabel("")
        self._station_count_lbl.setWordWrap(True)
        self._station_count_lbl.setStyleSheet(
            f"color:{C.TEXT_MUTED}; font-size:{FONT.SIZE_XS}px;"
        )
        station_outer.addWidget(self._station_count_lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(150)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setStyleSheet(
            f"QScrollArea {{ background:transparent; border:none; }}"
            f"QScrollBar:vertical {{ background:{C.BG_DEEP}; width:8px; }}"
            f"QScrollBar::handle:vertical {{ background:{C.BORDER_MID}; border-radius:4px; }}"
        )
        station_widget = QWidget()
        station_layout = QVBoxLayout(station_widget)
        station_layout.setContentsMargins(0, 0, 0, 0)
        station_layout.setSpacing(2)

        for name in ANALYSIS_STATIONS:
            cb = QCheckBox(name)
            cb.setStyleSheet(
                f"QCheckBox {{ color:{C.TEXT_SECONDARY}; font-size:{FONT.SIZE_XS}px; }}"
                f"QCheckBox::indicator {{ width:12px; height:12px; }}"
            )
            cb.stateChanged.connect(self._on_station_manual_change)
            self._station_checks[name] = cb
            station_layout.addWidget(cb)

        station_layout.addStretch()
        scroll.setWidget(station_widget)
        station_outer.addWidget(scroll)
        ll.addWidget(station_box)

        self._net_combo.currentIndexChanged.connect(self._apply_station_preset)
        self._apply_station_preset()

        # Run button
        self._run_btn = QPushButton("Run Analysis")
        self._run_btn.setMinimumHeight(36)
        self._run_btn.setStyleSheet(
            f"QPushButton {{ background:{C.BLUE}; color:#fff; border-radius:4px;"
            f" font-size:{FONT.SIZE_MD}px; font-weight:bold; }}"
            f"QPushButton:hover {{ background:{C.CYAN}; }}"
            f"QPushButton:disabled {{ background:{C.BORDER_MID}; color:{C.TEXT_MUTED}; }}"
        )
        self._run_btn.clicked.connect(self._start)
        ll.addWidget(self._run_btn)

        self._cancel_btn = QPushButton("Stop")
        self._cancel_btn.setMinimumHeight(32)
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setStyleSheet(
            f"QPushButton {{ background:{C.BG_CARD}; color:{C.TEXT_SECONDARY};"
            f" border:1px solid {C.BORDER_MID}; border-radius:4px;"
            f" font-size:{FONT.SIZE_SM}px; }}"
            f"QPushButton:hover {{ background:{C.BG_HOVER}; color:{C.YELLOW}; }}"
            f"QPushButton:disabled {{ color:{C.TEXT_MUTED}; border-color:{C.BORDER_MAIN}; }}"
        )
        self._cancel_btn.clicked.connect(self._cancel)
        ll.addWidget(self._cancel_btn)

        self._export_btn = QPushButton("Export CSV")
        self._export_btn.setEnabled(False)
        self._export_btn.setStyleSheet(
            f"QPushButton {{ background:{C.BG_HOVER}; color:{C.TEXT_PRIMARY};"
            f" border:1px solid {C.BORDER_MAIN}; border-radius:4px;"
            f" padding:5px 14px; font-size:{FONT.SIZE_SM}px; }}"
            f"QPushButton:hover {{ background:{C.BG_ACTIVE}; }}"
            f"QPushButton:disabled {{ color:{C.TEXT_MUTED}; }}"
        )
        self._export_btn.clicked.connect(self._export_csv)
        ll.addWidget(self._export_btn)

        self._status_lbl = QLabel("Ready")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setAlignment(Qt.AlignCenter)
        self._status_lbl.setStyleSheet(f"color:{C.TEXT_MUTED}; font-size:{FONT.SIZE_SM}px;")
        ll.addWidget(self._status_lbl)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setMaximumHeight(5)
        self._progress.setTextVisible(False)
        self._progress.setVisible(False)
        self._progress.setStyleSheet(
            f"QProgressBar {{ background:{C.BG_CARD}; border:none; border-radius:2px; }}"
            f"QProgressBar::chunk {{ background:{C.BLUE}; border-radius:2px; }}"
        )
        ll.addWidget(self._progress)
        ll.addStretch()

        outer.addWidget(left)

        # ── Right area: log (top) + canvas (bottom) ─────────────────────
        right = QSplitter(Qt.Vertical)
        right.setChildrenCollapsible(False)
        right.setStyleSheet("QSplitter::handle { background: transparent; height: 3px; }")

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(110)
        self._log.setMinimumHeight(70)
        self._log.setStyleSheet(
            f"QPlainTextEdit {{ background:{C.BG_DEEP}; color:{C.TEXT_SECONDARY};"
            f" font-family:{FONT.MONO}; font-size:{FONT.SIZE_SM}px; border:none; padding:4px; }}"
        )
        right.addWidget(self._log)

        self._canvas_holder = QWidget()
        self._canvas_holder.setMinimumHeight(200)
        ch_lay = QVBoxLayout(self._canvas_holder)
        ch_lay.setContentsMargins(0, 0, 0, 0)
        ch_lay.setSpacing(0)
        self._placeholder = QLabel(
            "Chart will appear here.\nPress 'Run Analysis' to start."
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(
            f"color:{C.TEXT_MUTED}; font-size:{FONT.SIZE_LG}px;"
        )
        ch_lay.addWidget(self._placeholder)
        right.addWidget(self._canvas_holder)

        right.setStretchFactor(0, 0)
        right.setStretchFactor(1, 1)
        outer.addWidget(right, 1)

    # ------------------------------------------------------------------
    def _start(self) -> None:
        if self._worker and self._worker.isRunning():
            return

        station_names = self._selected_station_names()
        if not station_names:
            self._set_status("Select at least one station.", C.RED)
            return

        self._results.clear()
        self._log.clear()
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progress.setVisible(True)
        self._set_status(
            f"0/{len(self._spec.variants)} variants running…", C.YELLOW
        )

        # Reset canvas to placeholder
        ch_lay = self._canvas_holder.layout()
        while ch_lay.count():
            item = ch_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        ph = QLabel("Computing variants…")
        ph.setAlignment(Qt.AlignCenter)
        ph.setStyleSheet(f"color:{C.TEXT_MUTED}; font-size:{FONT.SIZE_LG}px;")
        ch_lay.addWidget(ph)
        self._canvas = None
        self._fig    = None

        base_template = _base_from_settings(self._spec.estimator)
        base = {
            **base_template,
            "duration_h":     self._dur_spin.value() * 24.0,
            "network":        "single" if len(station_names) == 1 else "multi",
            "station_names":  station_names,
            "noise":          self._noise_cb.isChecked(),
            "estimator_type": self._spec.estimator,
        }
        if self._spec.estimator == "ukf":
            ukf_defaults = _ukf_overrides_from_settings()
            # variant overrides win; settings fill unset keys
            for k, v in ukf_defaults.items():
                if k not in base:
                    base[k] = v

        self._worker = _AnalysisWorker(self._spec, base)
        self._worker.log_line.connect(self._on_log)
        self._worker.variant_done.connect(self._on_variant_done)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _cancel(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._cancel_btn.setEnabled(False)
            self._set_status("Stop requested — will halt after current variant.", C.YELLOW)
            self._on_log("User requested stop. Will halt safely after current computation block.")

    def _selected_station_names(self) -> tuple[str, ...]:
        return tuple(
            name for name in ANALYSIS_STATIONS
            if self._station_checks.get(name) is not None
            and self._station_checks[name].isChecked()
        )

    def _apply_station_preset(self) -> None:
        station_names = self._net_combo.currentData()
        if station_names is None:
            self._update_station_count()
            return

        selected = set(station_names)
        self._applying_station_preset = True
        try:
            for name, cb in self._station_checks.items():
                cb.setChecked(name in selected)
        finally:
            self._applying_station_preset = False
        self._update_station_count()

    def _on_station_manual_change(self) -> None:
        if self._applying_station_preset:
            return

        selected = set(self._selected_station_names())
        matched_index = None
        custom_index = None
        for i, (_, preset_names) in enumerate(STATION_PRESETS):
            if preset_names is None:
                custom_index = i
            elif selected == set(preset_names):
                matched_index = i
                break

        target = matched_index if matched_index is not None else custom_index
        if target is not None and self._net_combo.currentIndex() != target:
            self._applying_station_preset = True
            try:
                self._net_combo.setCurrentIndex(target)
            finally:
                self._applying_station_preset = False
        self._update_station_count()

    def _update_station_count(self) -> None:
        selected = self._selected_station_names()
        if not selected:
            self._station_count_lbl.setText("No station selected.")
            self._station_count_lbl.setStyleSheet(
                f"color:{C.RED}; font-size:{FONT.SIZE_XS}px;"
            )
            return

        preview = ", ".join(selected[:3])
        if len(selected) > 3:
            preview += f" +{len(selected) - 3}"
        self._station_count_lbl.setText(f"{len(selected)} station(s): {preview}")
        self._station_count_lbl.setStyleSheet(
            f"color:{C.TEXT_MUTED}; font-size:{FONT.SIZE_XS}px;"
        )

    def _on_log(self, line: str) -> None:
        self._log.appendPlainText(line)
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum()
        )

    def _on_variant_done(self, label: str, result: Any) -> None:
        self._results.append((label, result))
        n_done  = len(self._results)
        n_total = len(self._spec.variants)
        self._set_status(f"{n_done}/{n_total} variants complete", C.BLUE)
        self._redraw_plot()

    def _on_all_done(self) -> None:
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress.setVisible(False)
        self._set_status(f"Done — {len(self._results)} variants", C.GREEN)
        self._export_btn.setEnabled(bool(self._results))

    def _on_cancelled(self) -> None:
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress.setVisible(False)
        self._set_status(f"Stopped — {len(self._results)} variant(s) completed", C.YELLOW)

    def _on_fail(self, msg: str) -> None:
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress.setVisible(False)
        short = msg.splitlines()[0][:120] if msg else "Unknown error"
        self._set_status(f"Error: {short}", C.RED)
        self._on_log(f"ERROR:\n{msg}")
        _show_error_dialog(self._run_btn.window(), "Analysis failed", msg)

    # ------------------------------------------------------------------
    def _redraw_plot(self) -> None:
        try:
            import matplotlib
            matplotlib.use("Qt5Agg")
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import (
                FigureCanvasQTAgg, NavigationToolbar2QT,
            )
        except Exception:
            return

        first_time = self._canvas is None
        if first_time:
            with plt.rc_context(MPL_DARK):
                fig = Figure(figsize=(9.5, 5.2), tight_layout={"pad": 0.9, "w_pad": 1.5})
            canvas = FigureCanvasQTAgg(fig)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            toolbar = NavigationToolbar2QT(canvas, self._canvas_holder)
            toolbar.setStyleSheet(
                f"QToolBar {{ background:{C.BG_PANEL}; border:none; }}"
                f"QToolButton {{ color:{C.TEXT_SECONDARY}; background:transparent; }}"
            )
            ch_lay = self._canvas_holder.layout()
            while ch_lay.count():
                item = ch_lay.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            ch_lay.addWidget(toolbar)
            ch_lay.addWidget(canvas)
            self._canvas = canvas
            self._fig = fig
        else:
            fig = self._fig
            canvas = self._canvas
            fig.clf()

        with plt.rc_context(MPL_DARK):
            ax_lines = fig.add_subplot(121)
            ax_bars  = fig.add_subplot(122)

        ax_lines.set_facecolor(C.BG_PLOT)
        ax_bars.set_facecolor(C.BG_PLOT)

        bar_labels: list[str] = []
        bar_medians: list[float] = []

        for i, (label, result) in enumerate(self._results):
            color = BAR_COLORS[i % len(BAR_COLORS)]
            errs  = result.final_position_errors_m
            if len(errs) == 0:
                continue
            arc_ids = np.array([a.arc_id for a in result.arc_results], dtype=float)
            ax_lines.semilogy(
                arc_ids, np.maximum(errs, 0.01),
                marker="o", linewidth=1.8, markersize=3.5,
                color=color, label=label, alpha=0.9,
            )
            bar_labels.append(label)
            bar_medians.append(float(np.median(errs)))

        # Line chart styling
        ax_lines.set_title(
            "Final Position Error per Arc",
            color=C.TEXT_SECONDARY, fontsize=FONT.SIZE_SM,
        )
        ax_lines.set_xlabel("Arc", color=C.TEXT_MUTED, fontsize=FONT.SIZE_SM - 1)
        ax_lines.set_ylabel("Error (m)", color=C.TEXT_MUTED, fontsize=FONT.SIZE_SM - 1)
        ax_lines.grid(True, which="both", alpha=0.12)
        ax_lines.tick_params(colors=C.TEXT_TICK, labelsize=FONT.SIZE_XS)
        for sp in ax_lines.spines.values():
            sp.set_edgecolor(C.BORDER_MAIN)
        if bar_labels:
            leg = ax_lines.legend(
                fontsize=FONT.SIZE_XS,
                facecolor=C.BG_PANEL, edgecolor=C.BORDER_MID, framealpha=0.8,
            )
            for t in leg.get_texts():
                t.set_color(C.TEXT_SECONDARY)

        # Bar chart (medians)
        if bar_labels:
            x = np.arange(len(bar_labels))
            clrs = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(bar_labels))]
            bars = ax_bars.bar(x, bar_medians, color=clrs, width=0.6, alpha=0.88)
            ax_bars.set_xticks(x)
            ax_bars.set_xticklabels(
                bar_labels, rotation=28, ha="right",
                fontsize=FONT.SIZE_XS, color=C.TEXT_SECONDARY,
            )
            ax_bars.set_yscale("log")
            # Value labels on bars
            for bar, val in zip(bars, bar_medians):
                ax_bars.text(
                    bar.get_x() + bar.get_width() / 2,
                    val * 1.15,
                    f"{val:.0f}",
                    ha="center", va="bottom",
                    fontsize=FONT.SIZE_XS - 1, color=C.TEXT_MUTED,
                    fontfamily=FONT.MONO,
                )

        ax_bars.set_title(
            "Median Final Position Error",
            color=C.TEXT_SECONDARY, fontsize=FONT.SIZE_SM,
        )
        ax_bars.set_ylabel("Error (m)", color=C.TEXT_MUTED, fontsize=FONT.SIZE_SM - 1)
        ax_bars.grid(True, which="both", alpha=0.12, axis="y")
        ax_bars.tick_params(colors=C.TEXT_TICK, labelsize=FONT.SIZE_XS)
        for sp in ax_bars.spines.values():
            sp.set_edgecolor(C.BORDER_MAIN)

        canvas.draw_idle()

    # ------------------------------------------------------------------
    def _set_status(self, text: str, color: str = C.TEXT_MUTED) -> None:
        self._status_lbl.setText(text)
        self._status_lbl.setStyleSheet(f"color:{color}; font-size:{FONT.SIZE_SM}px;")

    def _export_csv(self) -> None:
        if not self._results:
            return
        from PyQt5.QtWidgets import QFileDialog
        import csv
        path, _ = QFileDialog.getSaveFileName(
            self, "Export results CSV",
            str(Path.home() / f"{self._spec.title.lower().replace(' ', '_')}_results.csv"),
            "CSV files (*.csv)",
        )
        if not path:
            return
        rows = []
        for label, result in self._results:
            try:
                arc_errors = result.final_position_errors_m
                for arc_idx, err in enumerate(arc_errors):
                    rows.append({
                        "variant": label,
                        "arc_id": arc_idx,
                        "final_position_error_m": float(err),
                    })
            except Exception:
                rows.append({"variant": label, "arc_id": -1, "final_position_error_m": float("nan")})
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["variant", "arc_id", "final_position_error_m"])
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _group_style() -> str:
        return (
            f"QGroupBox {{ color:{C.TEXT_SECONDARY}; font-size:{FONT.SIZE_SM}px;"
            f" border:1px solid {C.BORDER_MAIN}; border-radius:4px; margin-top:6px; }}"
            f"QGroupBox::title {{ subcontrol-origin:margin; left:8px; padding:0 3px; }}"
        )

    @staticmethod
    def _spin_style() -> str:
        return (
            f"QDoubleSpinBox {{ background:{C.BG_DEEP}; color:{C.TEXT_PRIMARY};"
            f" border:1px solid {C.BORDER_MID}; border-radius:3px; padding:2px 4px;"
            f" font-size:{FONT.SIZE_SM}px; }}"
        )

    @staticmethod
    def _combo_style() -> str:
        return (
            f"QComboBox {{ background:{C.BG_DEEP}; color:{C.TEXT_PRIMARY};"
            f" border:1px solid {C.BORDER_MID}; border-radius:3px; padding:2px 6px;"
            f" font-size:{FONT.SIZE_SM}px; }}"
            f"QComboBox::drop-down {{ border:none; }}"
        )


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------

_TAB_STYLE = f"""
    QTabWidget::pane {{
        background: {C.BG_PANEL};
        border: none;
        border-top: 2px solid {C.BORDER_MAIN};
    }}
    QTabBar::tab {{
        background: {C.BG_SIDEBAR};
        color: {C.TEXT_MUTED};
        padding: 9px 28px;
        border: none;
        border-right: 1px solid {C.BORDER_MAIN};
        font-size: {FONT.SIZE_MD}px;
        font-weight: bold;
        min-width: 80px;
    }}
    QTabBar::tab:selected {{
        background: {C.BG_PANEL};
        color: {C.NAV_ACTIVE_FG};
        border-bottom: 2px solid {C.BLUE};
    }}
    QTabBar::tab:hover:!selected {{
        background: {C.BG_HOVER};
        color: {C.TEXT_SECONDARY};
    }}
"""

_SUB_TAB_STYLE = f"""
    QTabWidget::pane {{
        background: {C.BG_PANEL};
        border: none;
    }}
    QTabBar::tab {{
        background: {C.BG_CARD};
        color: {C.TEXT_MUTED};
        padding: 6px 16px;
        border: none;
        border-right: 1px solid {C.BORDER_FAINT};
        font-size: {FONT.SIZE_SM}px;
    }}
    QTabBar::tab:selected {{
        background: {C.BG_PANEL};
        color: {C.TEXT_PRIMARY};
        border-bottom: 2px solid {C.CYAN};
    }}
    QTabBar::tab:hover:!selected {{
        background: {C.BG_HOVER};
        color: {C.TEXT_SECONDARY};
    }}
"""


class AnalysisController(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        uic.loadUi(str(UI_PATH), self)
        self._build_tabs()

    def on_shown(self) -> None:
        pass

    def apply_scenario_params(self, model) -> None:
        """Pre-fill common params from a ScenarioModel (called from Scenario Builder)."""
        try:
            # Walk all sub-tabs looking for _ComparisonPanel instances
            top_tabs = self.mainLayout.itemAt(0).widget() if self.mainLayout.count() > 0 else None
            if top_tabs is None:
                return
            for ti in range(top_tabs.count()):
                est_tab = top_tabs.widget(ti)
                sub = est_tab.layout().itemAt(0).widget() if est_tab.layout().count() > 0 else None
                if sub is None:
                    continue
                for si in range(sub.count()):
                    panel = sub.widget(si)
                    if isinstance(panel, _ComparisonPanel):
                        try:
                            panel._dur_spin.setValue(model.duration_days)
                        except Exception:
                            pass
                        try:
                            if hasattr(panel, "_meas_combo"):
                                idx = panel._meas_combo.findText(model.measurement_type)
                                if idx >= 0:
                                    panel._meas_combo.setCurrentIndex(idx)
                        except Exception:
                            pass
        except Exception:
            pass

    def _build_tabs(self) -> None:
        top_tabs = QTabWidget()
        top_tabs.setStyleSheet(_TAB_STYLE)

        top_tabs.addTab(self._build_estimator_tab(BLS_COMPARISONS), "  BLS-LM  ")
        top_tabs.addTab(self._build_estimator_tab(UKF_COMPARISONS), "  UKF (SR)  ")

        self.mainLayout.addWidget(top_tabs)

    def _build_estimator_tab(self, comparisons: list[ComparisonSpec]) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        sub = QTabWidget()
        sub.setStyleSheet(_SUB_TAB_STYLE)
        for spec in comparisons:
            sub.addTab(_ComparisonPanel(spec), spec.title)

        lay.addWidget(sub)
        return w


def _show_error_dialog(parent, title: str, msg: str) -> None:
    """Show a QMessageBox with the full traceback in the details pane."""
    box = QMessageBox(parent)
    box.setWindowTitle(title)
    box.setIcon(QMessageBox.Critical)
    short = msg.splitlines()[0][:200] if msg else "An error occurred."
    box.setText(short)
    if "\n" in msg:
        box.setDetailedText(msg)
    box.exec_()
