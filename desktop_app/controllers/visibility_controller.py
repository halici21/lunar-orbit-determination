"""Visibility Gantt chart page — interactive station selection + live computation."""
from __future__ import annotations
import json
import csv
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QCheckBox, QSizePolicy,
    QPushButton, QHBoxLayout, QFileDialog, QMessageBox, QPlainTextEdit, QFormLayout,
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

            self.progress.emit("Örnek başlangıç durumu okunuyor…")
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

            self.progress.emit("SPICE dosyaları yükleniyor…")
            load_spice_kernels()
            try:
                self.progress.emit("Yörünge yayılıyor…")
                et0     = float(spice.str2et(epoch_utc))
                ephem   = sample_moon_centered_ephemeris(et0, t_ephem)
                xforms  = sample_j2000_to_itrf93_transforms(et0, t_eval)
                states  = propagate_truth_with_ephemeris(
                    t_eval, x0, mu_moon, mu_earth, mu_sun, ephem,
                    rtol=1e-9, atol=1e-10,
                )

                self.progress.emit("Görünürlük hesaplanıyor…")
                by_name  = {s.name: s for s in range_rate_stations()}
                stations = [by_name[n] for n in self.station_names if n in by_name]
                if not stations:
                    self.failed.emit("İstasyon seçilmedi.")
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
    ax.set_xlabel("Gün (başlangıç epokundan)", fontsize=FONT.SIZE_SM)

    # Day ticks
    day_ticks = list(range(int(dur_days) + 1))
    ax.set_xticks(day_ticks)
    ax.set_xticklabels([f"G{d}" for d in day_ticks], fontsize=FONT.SIZE_XS)

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
        self._busy = False
        self._pending_inputs = None
        self._result_inputs = None
        self._setup()

    def _setup(self) -> None:
        self._build_station_checkboxes()
        self.selectAllBtn.clicked.connect(self._select_all)
        self.deselectAllBtn.clicked.connect(self._deselect_all)
        self.computeBtn.clicked.connect(self._compute)
        self._build_export_buttons()
        self._build_log_panel()
        self._build_result_context()
        self._init_canvas_placeholder()
        self.stationHeaderLabel.setText("İstasyonlar")
        self.selectAllBtn.setText("Tümünü seç")
        self.deselectAllBtn.setText("Seçimi kaldır")
        self.paramsGroup.setTitle("Hesap ayarları")
        self.computeBtn.setText("Görünürlüğü hesapla")
        self.paramsFormLayout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        for label, control, text in (
            (self.durationLabel, self.durationSpin, "&Süre (gün)"),
            (self.elevLabel, self.elevSpin, "&En az yükseliş (°)"),
            (self.gapLabel, self.gapSpin, "&Boşluk eşiği (dk)"),
        ):
            label.setText(text)
            label.setBuddy(control)
            control.valueChanged.connect(self._inputs_changed)
        self.gapSpin.setToolTip("Analize iletilen boşluk eşiği. Grafik ve CSV ham görünürlük maskesini gösterir.")
        for checkbox in self._checkboxes.values():
            checkbox.toggled.connect(self._inputs_changed)
        self._inputs_changed()

    def resizeEvent(self, event) -> None:
        self._adapt_layout()
        super().resizeEvent(event)

    def _adapt_layout(self) -> None:
        if not hasattr(self, "splitter"):
            return
        narrow = self.width() < 900
        orientation = Qt.Vertical if narrow else Qt.Horizontal
        if self.splitter.orientation() != orientation:
            self.controlPanel.setMaximumWidth(16777215 if narrow else 320)
            self.stationScrollArea.setMaximumHeight(160 if narrow else 16777215)
            self.splitter.setOrientation(orientation)
            self.splitter.setSizes([self.controlPanel.minimumSizeHint().height(), 520]
                                   if narrow else [320, max(400, self.width() - 320)])
        # Keep the complete form and a readable plot in the outer vertical scroll.
        # Re-evaluate when the computation log or result context becomes visible.
        height = (self.controlLayout.minimumSize().height()
                  + max(520, self.chartLayout.minimumSize().height())
                  + self.splitter.handleWidth()) if narrow else 0
        self.setMinimumHeight(height)

    def minimumSizeHint(self):
        return super().minimumSizeHint().expandedTo(self.minimumSize())

    def _build_result_context(self) -> None:
        self._scope_note = QLabel(
            "Sabit örnek başlangıç yörüngesi · SPICE ile ayrı görünürlük hesabı · "
            f"{_SAMPLE_STEP_S / 60:g} dk örnekleme. "
            "Kayıtlı koşunun ölçüm takvimi: Sonuçlar › Genel görünüm."
        )
        self._result_context = QLabel()
        self._result_context.setAccessibleName("Görünen görünürlük sonucunun kaynağı ve hesap ayarları")
        self._result_context.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        for index, label in enumerate((self._scope_note, self._result_context)):
            label.setWordWrap(True)
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
            label.setStyleSheet(f"color:{C.TEXT_SECONDARY}; font-size:{FONT.SIZE_SM}px; padding:8px 12px;")
            self.chartLayout.insertWidget(index, label)

    def _input_snapshot(self):
        return (tuple(n for n, cb in self._checkboxes.items() if cb.isChecked()),
                self.durationSpin.value(), self.elevSpin.value(), self.gapSpin.value())

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        for widget in (self.paramsGroup, self.stationScrollArea, self.selectAllBtn, self.deselectAllBtn):
            widget.setEnabled(not busy)
        self.computeBtn.setEnabled(not busy and bool(self._input_snapshot()[0]))
        self.progressBar.setVisible(busy)
        self.exportCsvBtn.setEnabled(not busy and self._last_data is not None)
        self.exportPngBtn.setEnabled(not busy and self._last_data is not None and self._fig is not None)

    def _inputs_changed(self) -> None:
        if self._busy:
            return
        selected = bool(self._input_snapshot()[0])
        self.computeBtn.setEnabled(selected)
        if not selected:
            self._set_status("En az bir istasyon seçin.", C.YELLOW)
        elif self._last_data is None:
            self._set_status("İstasyonları ve ayarları seçip hesaplayın.", C.TEXT_MUTED)
        elif self._input_snapshot() != self._result_inputs:
            self._set_status("Ayarlar değişti; yeniden hesaplayın.", C.YELLOW)
        else:
            self._set_status("Son sonuç hazır.", C.GREEN)
        self._refresh_result_context()

    def _refresh_result_context(self, message: str = "") -> None:
        if self._busy:
            message = ("Yeni hesap sürüyor; görünen grafik önceki sonuç." if self._last_data is not None
                       else "Hesap sürüyor; sonuç bekleniyor.")
        elif not message and self._last_data is not None and self._input_snapshot() != self._result_inputs:
            message = "Ayarlar değişti; grafik ve dışa aktarım önceki sonuç. Yeniden hesaplayın."
        body = ""
        if self._last_data is not None:
            data = self._last_data
            body = (f"Son sonuç · T0 {data['epoch_utc']} UTC · {data['duration_days']:g} gün · "
                    f"{len(data['station_names'])} istasyon")
            if self._result_inputs is not None:
                body += f" · en az yükseliş {self._result_inputs[2]:g}° · boşluk eşiği {self._result_inputs[3]:g} dk"
            else:
                body += " · hesap ayarları kaydedilmemiş"
            details = f"Kaynak başlangıç: {FIXTURE.name}\nİstasyonlar: " + ", ".join(data['station_names'])
            self._result_context.setToolTip(details)
            self._result_context.setAccessibleDescription(details + "\n" + body)
        self._result_context.setText("\n".join(part for part in (message, body) if part))
        self._result_context.setVisible(bool(message or body))
        self._adapt_layout()

    def _build_export_buttons(self) -> None:
        row = QHBoxLayout()
        row.setSpacing(6)
        self.exportPngBtn = QPushButton("PNG kaydet")
        self.exportCsvBtn = QPushButton("CSV kaydet")
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
        self._log_edit.setPlaceholderText("Hesap günlüğü…")
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
        self._adapt_layout()

    def on_shown(self) -> None:
        pass  # inputs belong to this page; nothing external overrides them on show

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
        if self._busy or (self._worker and self._worker.isRunning()):
            return

        selected = [n for n, cb in self._checkboxes.items() if cb.isChecked()]
        if not selected:
            self._set_status("En az bir istasyon seçin.", C.YELLOW)
            return

        self._pending_inputs = self._input_snapshot()
        self._set_busy(True)
        self._refresh_result_context()
        self._log_edit.clear()
        self._log(f"Başlıyor: {len(selected)} istasyon, {self.durationSpin.value():.0f} gün")
        self._set_status(f"{len(selected)} istasyon, {self.durationSpin.value():.0f} gün…", C.BLUE)

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
        if not self._show_gantt(data):
            self._pending_inputs = None
            self._set_busy(False)
            self._refresh_result_context("Grafik oluşturulamadı; görünen grafik ve dışa aktarım önceki sonuç."
                                         if self._last_data is not None else "Sonuç üretilmedi; yeniden hesaplayın.")
            return
        self._last_data = data
        self._result_inputs = self._pending_inputs
        self._pending_inputs = None
        self._set_busy(False)
        n = len(data["station_names"])
        done_msg = f"Tamamlandı — {n} istasyon, {data['duration_days']:.0f} gün"
        self._set_status(done_msg, C.GREEN)
        self._log(done_msg)
        self._refresh_result_context()

    def _on_fail(self, msg: str) -> None:
        self._pending_inputs = None
        self._set_busy(False)
        self._refresh_result_context("Yeni hesap başarısız; görünen grafik ve dışa aktarım önceki sonuç."
                                     if self._last_data is not None else "Sonuç üretilmedi; ayarları ve SPICE dosyalarını kontrol edin.")
        short = msg.splitlines()[0][:120] if msg else "Bilinmeyen hata"
        self._set_status(f"Hata: {short}", C.RED)
        self._log(f"BAŞARISIZ: {short}")
        _show_error_dialog(self, "Görünürlük hesaplanamadı", msg)

    # ------------------------------------------------------------------
    def _init_canvas_placeholder(self) -> None:
        lay = QVBoxLayout(self.canvasHolder)
        lay.setContentsMargins(0, 0, 0, 0)
        placeholder = QLabel("İstasyonları seçin; görünürlüğü hesaplayın.\nSPICE dosyalarını Ayarlar → Hesaplama ortamı bölümünden kontrol edebilirsiniz.")
        placeholder.setWordWrap(True)
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet(f"color:{C.TEXT_MUTED}; font-size:{FONT.SIZE_LG}px;")
        lay.addWidget(placeholder)
        self._placeholder_label = placeholder

    def _show_gantt(self, data: dict) -> bool:
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
            self._set_status(f"Grafik çizilemedi: {exc}", C.RED)
            return False

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
        return True

    def _export_png(self) -> None:
        if self._fig is None:
            QMessageBox.information(self, "Kaydedilecek sonuç yok", "Önce görünürlüğü hesaplayın.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Temas planını görüntü olarak kaydet",
            str(Path.home() / "visibility_gantt.png"),
            "PNG görüntüleri (*.png);;PDF dosyaları (*.pdf)",
        )
        if path:
            self._fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=self._fig.get_facecolor())

    def _export_csv(self) -> None:
        if self._last_data is None:
            QMessageBox.information(self, "Kaydedilecek sonuç yok", "Önce görünürlüğü hesaplayın.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Görünürlük tablosunu CSV olarak kaydet",
            str(Path.home() / "visibility_matrix.csv"),
            "CSV dosyaları (*.csv)",
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
    short = msg.splitlines()[0][:200] if msg else "Bir hata oluştu."
    box.setText(short)
    if "\n" in msg:
        box.setDetailedText(msg)
    box.exec_()
