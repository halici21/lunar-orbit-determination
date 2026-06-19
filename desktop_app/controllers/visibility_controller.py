"""Visibility Gantt chart page — interactive station selection + live computation."""
from __future__ import annotations
import json
import csv
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QCheckBox, QSizePolicy,
    QPushButton, QHBoxLayout, QFileDialog, QMessageBox, QPlainTextEdit,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5 import uic

from services.project_paths import DESKTOP_APP_DIR, PYTHON_PORT
from styles.theme import C, FONT, MPL_DARK, NET_COLORS

UI_PATH   = DESKTOP_APP_DIR / "ui" / "pages" / "visibility_page.ui"
FIXTURE   = PYTHON_PORT / "fixtures" / "spice_snapshots.json"

# Sample step for orbit propagation (seconds) — finer = slower but more accurate windows
_SAMPLE_STEP_S  = 600.0   # 10-min
_EPHEM_STEP_S   = 3600.0  # 1-h ephemeris grid


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------
class _VisWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)   # dict with result data
    failed   = pyqtSignal(str)

    def __init__(
        self,
        station_names: list[str],
        duration_days: float,
        elev_deg: float,
        max_gap_min: float,
    ) -> None:
        super().__init__()
        self.station_names = station_names
        self.duration_days = duration_days
        self.elev_deg      = elev_deg
        self.max_gap_s     = max_gap_min * 60.0

    def run(self) -> None:
        try:
            import spiceypy as spice
            from lunar_od import (
                load_spice_kernels,
                range_rate_stations,
                VisibilityConfig,
                analyze_visibility_gap_with_transforms,
                propagate_truth_with_ephemeris,
                sample_j2000_to_itrf93_transforms,
                sample_moon_centered_ephemeris,
            )

            self.progress.emit("Loading fixture…")
            fixture    = json.loads(FIXTURE.read_text(encoding="utf-8"))
            initial    = fixture["initial_state"]
            constants  = fixture["constants"]
            epoch_utc  = fixture["epoch_utc"]

            x0       = np.asarray(initial["state_mci_j2000_m_mps"], dtype=float)
            mu_moon  = float(initial["mu_moon_m3_s2"])
            mu_earth = float(np.asarray(constants["mu_earth_km3_s2"]).reshape(-1)[0] * 1e9)
            mu_sun   = float(np.asarray(constants["mu_sun_km3_s2"]).reshape(-1)[0] * 1e9)
            r_moon   = float(initial["r_moon_mean_m"])

            t_eval  = np.arange(0.0, self.duration_days * 86400.0 + _SAMPLE_STEP_S, _SAMPLE_STEP_S)
            t_ephem = np.arange(0.0, self.duration_days * 86400.0 + _EPHEM_STEP_S,  _EPHEM_STEP_S)

            self.progress.emit("Loading SPICE kernels…")
            load_spice_kernels()
            try:
                self.progress.emit("Propagating orbit…")
                et0     = float(spice.str2et(epoch_utc))
                ephem   = sample_moon_centered_ephemeris(et0, t_ephem)
                xforms  = sample_j2000_to_itrf93_transforms(et0, t_eval)
                states  = propagate_truth_with_ephemeris(
                    t_eval, x0, mu_moon, mu_earth, mu_sun, ephem,
                    rtol=1e-9, atol=1e-10,
                )

                self.progress.emit("Computing visibility…")
                by_name  = {s.name: s for s in range_rate_stations()}
                stations = [by_name[n] for n in self.station_names if n in by_name]
                if not stations:
                    self.failed.emit("No stations selected.")
                    return

                config = VisibilityConfig(
                    r_moon_mean_m=r_moon,
                    earth_rotation_rad_s=7.292115e-5,
                    epoch_utc=epoch_utc,
                    min_elevation_deg=self.elev_deg,
                )
                _seg_s, _seg_e, raw_mask, _filled = analyze_visibility_gap_with_transforms(
                    t_eval, states, stations,
                    ephem.earth_position, xforms,
                    self.max_gap_s, config,
                )
            finally:
                spice.kclear()

            raw_mask = np.asarray(raw_mask, dtype=bool)
            # ensure shape (N_stations, N_steps)
            if raw_mask.shape[0] == t_eval.size and raw_mask.shape[1] == len(stations):
                raw_mask = raw_mask.T

            self.finished.emit({
                "t_eval_s":      t_eval,
                "raw_mask":      raw_mask,
                "station_names": [s.name for s in stations],
                "duration_days": self.duration_days,
                "epoch_utc":     epoch_utc,
            })

        except Exception as exc:
            import traceback
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Gantt chart renderer
# ---------------------------------------------------------------------------
def _draw_gantt(data: dict) -> "Figure":
    import matplotlib
    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    t_s      = data["t_eval_s"]
    mask     = data["raw_mask"]          # (N_stations, N_steps)
    names    = data["station_names"]
    dur_days = data["duration_days"]
    step_s   = float(t_s[1] - t_s[0]) if len(t_s) > 1 else _SAMPLE_STEP_S
    n_st     = len(names)

    def _network(name: str) -> str:
        for word in name.split():
            if word in NET_COLORS:
                return word
        return "OTHER"

    row_h   = 0.7
    fig_h   = max(3.5, 0.9 + n_st * (row_h + 0.18))

    with plt.rc_context(MPL_DARK):
        fig = Figure(figsize=(max(10, dur_days * 0.6), fig_h),
                     tight_layout={"pad": 0.8, "h_pad": 0.4})
        ax  = fig.add_subplot(111)

    ax.set_facecolor(C.BG_PANEL)

    # --- Day & 6-h grid lines ---
    for d in np.arange(0, dur_days + 0.01, 0.25):   # every 6 h
        ax.axvline(d, color=C.BORDER_MAIN, linewidth=0.4, zorder=1)
    for d in range(int(dur_days) + 1):
        ax.axvline(d, color=C.BORDER_MID, linewidth=0.9, zorder=2)

    # --- Bars per station ---
    legend_patches: dict[str, Patch] = {}
    for row_idx, (name, s_mask) in enumerate(zip(names, mask)):
        net   = _network(name)
        color = NET_COLORS.get(net, C.TEXT_MUTED)
        y_bot = row_idx - row_h / 2

        # Find windows
        padded = np.r_[False, s_mask, False].astype(int)
        edges  = np.diff(padded)
        starts = np.flatnonzero(edges == 1)
        stops  = np.flatnonzero(edges == -1)
        for st, sp in zip(starts, stops):
            x_start  = st * step_s / 86400.0
            x_width  = (sp - st) * step_s / 86400.0
            ax.broken_barh(
                [(x_start, x_width)], (y_bot, row_h),
                facecolors=color, edgecolors="none", alpha=0.88, zorder=3,
            )

        if net not in legend_patches:
            legend_patches[net] = Patch(facecolor=color, label=net)

        # Coverage % annotation
        pct = 100.0 * s_mask.sum() / max(len(s_mask), 1)
        ax.text(
            dur_days + 0.05, row_idx,
            f"{pct:.0f}%",
            va="center", ha="left",
            fontsize=FONT.SIZE_XS, color=color, fontfamily=FONT.MONO,
        )

    # Axes
    ax.set_xlim(0, dur_days * 1.08)
    ax.set_ylim(-0.7, n_st - 0.3)
    ax.set_yticks(range(n_st))
    ax.set_yticklabels(names, fontsize=FONT.SIZE_SM - 1)
    ax.set_xlabel("Day (since epoch)", fontsize=FONT.SIZE_SM)

    # Day ticks
    day_ticks = list(range(int(dur_days) + 1))
    ax.set_xticks(day_ticks)
    ax.set_xticklabels([f"D{d}" for d in day_ticks], fontsize=FONT.SIZE_XS)

    # Tick/spine colours
    ax.tick_params(colors=C.TEXT_TICK)
    for spine in ax.spines.values():
        spine.set_edgecolor(C.BORDER_MAIN)

    # Legend
    if legend_patches:
        leg = ax.legend(
            handles=list(legend_patches.values()),
            loc="upper right", fontsize=FONT.SIZE_XS,
            framealpha=0.7, facecolor=C.BG_DEEP,
            edgecolor=C.BORDER_MID,
        )
        for t in leg.get_texts():
            t.set_color(C.TEXT_SECONDARY)

    return fig


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
def _net_label(net: str, color: str) -> QLabel:
    lbl = QLabel(f"  {net}")
    lbl.setStyleSheet(
        f"QLabel {{ color:{color}; font-size:{FONT.SIZE_SM}px; font-weight:bold;"
        f" border-bottom:1px solid {color}; margin-top:6px; }}"
    )
    return lbl


class VisibilityController(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        uic.loadUi(str(UI_PATH), self)
        self._checkboxes: dict[str, QCheckBox] = {}
        self._worker: _VisWorker | None = None
        self._canvas = None
        self._fig = None
        self._last_data: dict | None = None
        self._setup()

    def _setup(self) -> None:
        self._build_station_checkboxes()
        self.selectAllBtn.clicked.connect(self._select_all)
        self.deselectAllBtn.clicked.connect(self._deselect_all)
        self.computeBtn.clicked.connect(self._compute)
        self._build_export_buttons()
        self._build_log_panel()
        self._set_status("Select stations and click 'Compute Gantt'.", C.TEXT_MUTED)
        self._init_canvas_placeholder()

    def _build_export_buttons(self) -> None:
        row = QHBoxLayout()
        row.setSpacing(6)
        self.exportPngBtn = QPushButton("Export PNG")
        self.exportCsvBtn = QPushButton("Export CSV")
        self.exportPngBtn.setEnabled(False)
        self.exportCsvBtn.setEnabled(False)
        self.exportPngBtn.clicked.connect(self._export_png)
        self.exportCsvBtn.clicked.connect(self._export_csv)
        row.addWidget(self.exportPngBtn)
        row.addWidget(self.exportCsvBtn)
        self.controlLayout.insertLayout(self.controlLayout.indexOf(self.statusLabel), row)

    def _build_log_panel(self) -> None:
        self._log_edit = QPlainTextEdit()
        self._log_edit.setReadOnly(True)
        self._log_edit.setFixedHeight(82)
        self._log_edit.setPlaceholderText("Computation log…")
        self._log_edit.setStyleSheet(
            f"QPlainTextEdit {{ background:{C.BG_DEEP}; color:{C.TEXT_MUTED};"
            f" font-family:Consolas,monospace; font-size:10px;"
            f" border:1px solid {C.BORDER_MAIN}; border-radius:3px; }}"
        )
        self._log_edit.setVisible(False)
        self.controlLayout.addWidget(self._log_edit)

    def _log(self, msg: str) -> None:
        self._log_edit.setVisible(True)
        self._log_edit.appendPlainText(msg)
        self._log_edit.ensureCursorVisible()

    def on_shown(self) -> None:
        self._sync_duration_from_settings()

    def _sync_duration_from_settings(self) -> None:
        from PyQt5.QtCore import QSettings
        s = QSettings("LunarOD", "DesktopApp")
        try:
            dur_days = float(s.value("dynamics/duration_days", self.durationSpin.value()))
            self.durationSpin.setValue(dur_days)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _build_station_checkboxes(self) -> None:
        try:
            from lunar_od import range_rate_stations
            stations = range_rate_stations()
        except Exception:
            return

        layout = self.stationCheckLayout
        # Group by network
        groups: dict[str, list] = {}
        for s in stations:
            net = self._network(s.name)
            groups.setdefault(net, []).append(s)

        for net, slist in sorted(groups.items()):
            color = NET_COLORS.get(net, C.TEXT_MUTED)
            layout.addWidget(_net_label(net, color))
            for s in slist:
                cb = QCheckBox(s.name)
                cb.setChecked(True)
                cb.setStyleSheet(
                    f"QCheckBox {{ color:{C.TEXT_SECONDARY}; font-size:{FONT.SIZE_MD}px; }}"
                    f"QCheckBox::indicator:checked {{ background-color:{color}; border:1px solid {color}; }}"
                )
                layout.addWidget(cb)
                self._checkboxes[s.name] = cb

        layout.addStretch()

    @staticmethod
    def _network(name: str) -> str:
        for word in name.split():
            if word in NET_COLORS:
                return word
        return "OTHER"

    # ------------------------------------------------------------------
    def _select_all(self) -> None:
        for cb in self._checkboxes.values():
            cb.setChecked(True)

    def _deselect_all(self) -> None:
        for cb in self._checkboxes.values():
            cb.setChecked(False)

    # ------------------------------------------------------------------
    def _compute(self) -> None:
        if self._worker and self._worker.isRunning():
            return

        selected = [n for n, cb in self._checkboxes.items() if cb.isChecked()]
        if not selected:
            self._set_status("Select at least one station.", C.YELLOW)
            return

        self.computeBtn.setEnabled(False)
        self.progressBar.setVisible(True)
        self._log_edit.clear()
        self._log(f"Starting: {len(selected)} station(s), {self.durationSpin.value():.0f} days")
        self._set_status(f"{len(selected)} station(s), {self.durationSpin.value():.0f} days…", C.BLUE)

        self._worker = _VisWorker(
            station_names=selected,
            duration_days=self.durationSpin.value(),
            elev_deg=self.elevSpin.value(),
            max_gap_min=self.gapSpin.value(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_result)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_progress(self, msg: str) -> None:
        self._set_status(msg, C.YELLOW)
        self._log(msg)

    def _on_result(self, data: dict) -> None:
        self.computeBtn.setEnabled(True)
        self.progressBar.setVisible(False)
        self._last_data = data
        self.exportCsvBtn.setEnabled(True)
        n = len(data["station_names"])
        done_msg = f"Done — {n} station(s), {data['duration_days']:.0f} days"
        self._set_status(done_msg, C.GREEN)
        self._log(done_msg)
        self._show_gantt(data)

    def _on_fail(self, msg: str) -> None:
        self.computeBtn.setEnabled(True)
        self.progressBar.setVisible(False)
        short = msg.splitlines()[0][:120] if msg else "Unknown error"
        self._set_status(f"Error: {short}", C.RED)
        self._log(f"FAILED: {short}")
        _show_error_dialog(self, "Visibility computation failed", msg)

    # ------------------------------------------------------------------
    def _init_canvas_placeholder(self) -> None:
        lay = QVBoxLayout(self.canvasHolder)
        lay.setContentsMargins(0, 0, 0, 0)
        placeholder = QLabel("Gantt chart will appear here.\nSelect stations on the left and compute.")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet(f"color:{C.TEXT_MUTED}; font-size:{FONT.SIZE_LG}px;")
        lay.addWidget(placeholder)
        self._placeholder_label = placeholder

    def _show_gantt(self, data: dict) -> None:
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
            fig    = _draw_gantt(data)
            canvas = FigureCanvasQTAgg(fig)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            toolbar = NavigationToolbar2QT(canvas, self.canvasHolder)
            toolbar.setStyleSheet(
                f"QToolBar {{ background:{C.BG_PANEL}; border:none; }}"
                f"QToolButton {{ color:{C.TEXT_SECONDARY}; background:transparent; }}"
            )
        except Exception as exc:
            self._set_status(f"Plot error: {exc}", C.RED)
            return

        # Replace existing canvas
        lay = self.canvasHolder.layout()
        while lay.count():
            item = lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        lay.addWidget(toolbar)
        lay.addWidget(canvas)
        self._canvas = canvas
        self._fig = fig
        self.exportPngBtn.setEnabled(True)

    def _export_png(self) -> None:
        if self._fig is None:
            QMessageBox.information(self, "Export unavailable", "Compute the Gantt chart first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export visibility PNG",
            str(Path.home() / "visibility_gantt.png"),
            "PNG images (*.png);;PDF files (*.pdf)",
        )
        if path:
            self._fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=self._fig.get_facecolor())

    def _export_csv(self) -> None:
        if self._last_data is None:
            QMessageBox.information(self, "Export unavailable", "Compute visibility first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export visibility CSV",
            str(Path.home() / "visibility_matrix.csv"),
            "CSV files (*.csv)",
        )
        if not path:
            return

        data = self._last_data
        t_s = np.asarray(data["t_eval_s"], dtype=float)
        mask = np.asarray(data["raw_mask"], dtype=bool)
        names = list(data["station_names"])
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["time_s", "time_days", *names, "network_visible"])
            for idx, t in enumerate(t_s):
                row_mask = mask[:, idx] if mask.size else np.zeros(len(names), dtype=bool)
                writer.writerow([
                    f"{t:.3f}",
                    f"{t / 86400.0:.8f}",
                    *[int(v) for v in row_mask],
                    int(np.any(row_mask)),
                ])

    # ------------------------------------------------------------------
    def _set_status(self, text: str, color: str = C.TEXT_MUTED) -> None:
        self.statusLabel.setText(text)
        self.statusLabel.setStyleSheet(
            f"QLabel {{ color:{color}; font-size:{FONT.SIZE_MD}px; }}"
        )


def _show_error_dialog(parent, title: str, msg: str) -> None:
    """Show a QMessageBox with the full traceback in the details pane."""
    from PyQt5.QtWidgets import QMessageBox
    box = QMessageBox(parent)
    box.setWindowTitle(title)
    box.setIcon(QMessageBox.Critical)
    short = msg.splitlines()[0][:200] if msg else "An error occurred."
    box.setText(short)
    if "\n" in msg:
        box.setDetailedText(msg)
    box.exec_()
