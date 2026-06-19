"""Comparison page controller — grouped bar charts and summary tables from CSVs."""
from __future__ import annotations
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidgetItem, QLabel,
    QFileDialog, QMessageBox, QPushButton,
)
from PyQt5.QtCore import Qt
from PyQt5 import uic

from services.project_paths import DESKTOP_APP_DIR, RESULTS_DIR, EXAMPLES_DIR
from services.result_indexer import index_result_folders
from services.csv_loader import load_csv
from styles.theme import MPL_DARK, BAR_COLORS, C, FONT

UI_PATH = DESKTOP_APP_DIR / "ui" / "pages" / "comparison_page.ui"

try:
    import matplotlib
    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Domain-specific column search order for position error (in priority order)
_POS_ERROR_CANDIDATES = [
    "final_position_error_m",
    "median_final_position_error_m",
    "p50_position_error_m",
    "position_error_m",
    "pos_error_m",
    "bls_median_pos_error_m",
    "ukf_median_pos_error_m",
    "median_error",
    "pos_err",
]

# Domain-specific column search order for runtime
_RUNTIME_CANDIDATES = [
    "ukf_elapsed_s", "elapsed_s", "runtime_s", "cpu_time_s",
    "elapsed", "runtime", "cpu_time",
]

# Keys for grouping by comparison type
_GROUP_COL_MAP = {
    "BLS-LM vs SR-UKF":               ["estimator_type", "estimator", "filter"],
    "Cold vs Hot vs Formal":           ["start_mode", "initialization", "start"],
    "Position-only vs Range-rate":     ["measurement_type", "observable_set", "observables"],
    "Geometric RR vs Two-way Doppler": ["range_rate_physics", "doppler_model", "rr_physics"],
    "Single vs Multi-station":         ["network", "network_name", "station_set", "scenario"],
}

_FOLDER_HINTS = {
    "BLS-LM vs SR-UKF":            ["baseline_bls_ukf", "sequential_tracking"],
    "Cold vs Hot vs Formal":        ["bls_3day_ablation", "bls_7day_ablation"],
    "Position-only vs Range-rate":  ["bls_3day_ablation", "bls_7day_ablation"],
    "Geometric RR vs Two-way Doppler": ["two_way_doppler_bls_ukf"],
    "Single vs Multi-station":      ["real_visibility_bls_ukf", "baseline_bls_ukf"],
}

# Metric display name for the y-axis label
_Y_AXIS_LABEL = {
    "final_position_error_m": "Final position error (m)",
    "median_final_position_error_m": "Median final position error (m)",
    "p50_position_error_m": "Median position error (m)",
    "position_error_m": "Position error (m)",
    "pos_error_m": "Position error (m)",
    "bls_median_pos_error_m": "BLS median position error (m)",
    "ukf_median_pos_error_m": "UKF median position error (m)",
    "median_error": "Median error (m)",
    "pos_err": "Position error (m)",
}

_MPL_STYLE = MPL_DARK
_ACCENT_COLORS = BAR_COLORS

# Maps each comparison type to the example script that generates the required data
_RUN_SCRIPTS = {
    "BLS-LM vs SR-UKF":               "baseline_bls_ukf_comparison.py",
    "Cold vs Hot vs Formal":           "bls_7day_ablation_appendix.py",
    "Position-only vs Range-rate":     "bls_7day_ablation_appendix.py",
    "Geometric RR vs Two-way Doppler": "two_way_doppler_bls_ukf_comparison.py",
    "Single vs Multi-station":         "real_visibility_bls_ukf_matrix.py",
}

_BTN_STYLE = (
    f"QPushButton {{ background:#1a2a3a; color:{C.TEXT_PRIMARY};"
    f" border:1px solid #2a4a6a; border-radius:4px;"
    f" padding:4px 12px; font-size:12px; }}"
    f"QPushButton:hover {{ background:#243550; }}"
    f"QPushButton:disabled {{ color:#4a5a6a; border-color:#1a2030; }}"
)


class ComparisonController(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        uic.loadUi(str(UI_PATH), self)
        self._canvas: Optional["FigureCanvas"] = None
        self._figure: Optional["Figure"] = None
        self._current_df = None
        self._current_folder: Optional[Path] = None
        self._setup()

    def _setup(self) -> None:
        self._populate_folders()
        self.loadBtn.clicked.connect(self._load_data)
        self.exportPlotBtn.clicked.connect(self._export_plot)
        self.compareTypeCombo.currentIndexChanged.connect(self._on_type_changed)
        self.resultFolderCombo.currentIndexChanged.connect(self._update_png_combo)
        self.pngSelectCombo.currentIndexChanged.connect(self._load_raw_png)
        self._add_run_button()
        self._setup_canvas()

    def _add_run_button(self) -> None:
        self._run_btn = QPushButton("▶ Run Script")
        self._run_btn.setToolTip("Run the example script that generates data for this comparison type")
        self._run_btn.setStyleSheet(_BTN_STYLE)
        self._run_btn.clicked.connect(self._run_script)
        self.controlsLayout.addWidget(self._run_btn)
        self._update_run_btn_tooltip()

    def _update_run_btn_tooltip(self) -> None:
        compare_type = self.compareTypeCombo.currentText()
        script = _RUN_SCRIPTS.get(compare_type, "")
        if script:
            path = EXAMPLES_DIR / script
            exists = path.exists()
            tip = f"Run: {script}" + ("" if exists else "  (script not found)")
            self._run_btn.setEnabled(exists)
        else:
            tip = "No script mapped for this comparison type"
            self._run_btn.setEnabled(False)
        self._run_btn.setToolTip(tip)

    def on_shown(self) -> None:
        if self.resultFolderCombo.count() == 0:
            self._populate_folders()

    # ------------------------------------------------------------------
    def _populate_folders(self) -> None:
        self.resultFolderCombo.blockSignals(True)
        self.resultFolderCombo.clear()
        folders = index_result_folders(RESULTS_DIR)
        for f in folders:
            self.resultFolderCombo.addItem(f["name"], userData=f["path"])
        self.resultFolderCombo.blockSignals(False)
        self._select_suggested_folder()

    def _setup_canvas(self) -> None:
        if not HAS_MPL:
            placeholder = QLabel("Install matplotlib to enable comparison plots.")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("color: #8090aa; font-size: 13px;")
            self.plotCanvasHolder.layout() or QVBoxLayout(self.plotCanvasHolder)
            self.plotCanvasHolder.layout().addWidget(placeholder)
            return

        self._figure = Figure(figsize=(8, 5))
        self._figure.set_tight_layout(True)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setStyleSheet("background-color: #080c18;")

        lay = QVBoxLayout(self.plotCanvasHolder)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._canvas)
        self._draw_empty()

    def _draw_empty(self) -> None:
        if not HAS_MPL or self._figure is None:
            return
        with matplotlib.rc_context(_MPL_STYLE):
            self._figure.clear()
            ax = self._figure.add_subplot(111)
            ax.text(0.5, 0.5, "Load data to generate comparison plot",
                    ha="center", va="center", fontsize=14, color="#8090aa",
                    transform=ax.transAxes)
            ax.set_axis_off()
            self._canvas.draw()

    # ------------------------------------------------------------------
    def _load_data(self) -> None:
        try:
            self._load_data_inner()
        except Exception as exc:
            import traceback
            self._draw_message(f"Error loading data:\n{exc}")

    def _load_data_inner(self) -> None:
        idx = self.resultFolderCombo.currentIndex()
        if idx < 0:
            self._draw_message("No result folder selected.")
            return
        folder: Path = self.resultFolderCombo.itemData(idx)
        if folder is None or not folder.exists():
            self._draw_message(f"Folder not found:\n{folder}")
            return
        self._current_folder = folder
        self._update_png_combo()

        if not HAS_PANDAS:
            self._draw_no_pandas()
            return

        csv_file = self._pick_csv_file(folder, self.compareTypeCombo.currentText())
        if csv_file is None:
            self._draw_message(
                f"No CSV files found in:\n{folder.name}\n\n"
                "Run the comparison script first (▶ Run Script)."
            )
            return

        df = load_csv(csv_file)
        if df is None or df.empty:
            self._draw_message(
                f"Could not parse: {csv_file.name}\n"
                "File may be empty, corrupt, or in an unexpected format."
            )
            return
        self._current_df = df

        compare_type = self.compareTypeCombo.currentText()
        plot_type = self.plotTypeCombo.currentText()
        self._render_comparison(df, compare_type, plot_type, folder)
        self._populate_summary_table(df)

    def _run_script(self) -> None:
        compare_type = self.compareTypeCombo.currentText()
        script_name = _RUN_SCRIPTS.get(compare_type)
        if not script_name:
            QMessageBox.information(self, "No script", f"No script mapped for '{compare_type}'.")
            return
        script_path = EXAMPLES_DIR / script_name
        if not script_path.exists():
            QMessageBox.warning(self, "Script not found",
                                f"Script not found:\n{script_path}")
            return
        main_win = self.window()
        if hasattr(main_win, "navigate_and_run"):
            main_win.navigate_and_run(str(script_path), auto_start=True)
        elif hasattr(main_win, "navigate"):
            main_win.navigate("run_monitor")

    def _on_type_changed(self) -> None:
        self._select_suggested_folder()
        self._update_run_btn_tooltip()
        if self._current_df is not None and self._current_folder is not None:
            self._load_data()

    def _select_suggested_folder(self) -> None:
        compare_type = self.compareTypeCombo.currentText()
        hints = _FOLDER_HINTS.get(compare_type, [])
        if not hints:
            return
        for i in range(self.resultFolderCombo.count()):
            folder_name = self.resultFolderCombo.itemText(i).lower()
            folder_path = str(self.resultFolderCombo.itemData(i) or "").lower()
            if any(hint.lower() in folder_name or hint.lower() in folder_path for hint in hints):
                self.resultFolderCombo.setCurrentIndex(i)
                return

    def _pick_csv_file(self, folder: Path, compare_type: str) -> Path | None:
        csv_files = sorted(folder.glob("*.csv"))
        if not csv_files:
            return None

        hints = [h.lower() for h in _FOLDER_HINTS.get(compare_type, [])]

        def score(path: Path) -> tuple[int, str]:
            name = path.name.lower()
            s = 100
            if any(h in name for h in hints):
                s -= 40
            if "od_summary" in name:
                s -= 35
            if "summary" in name:
                s -= 20
            if "aggregate" in name:
                s -= 15
            if "error" in name:
                s -= 10
            if compare_type == "Geometric RR vs Two-way Doppler" and any(k in name for k in ("doppler", "two_way", "twoway", "range_rate")):
                s -= 35
            return s, name

        return sorted(csv_files, key=score)[0]

    # ------------------------------------------------------------------
    def _render_comparison(self, df, compare_type: str, plot_type: str, folder: Path) -> None:
        if not HAS_MPL or self._figure is None:
            return
        with matplotlib.rc_context(_MPL_STYLE):
            self._figure.clear()
            ax = self._figure.add_subplot(111)

            if self._render_domain_comparison(ax, df, compare_type, plot_type, folder):
                self._canvas.draw()
                return

            # Find numeric columns to plot
            num_cols = list(df.select_dtypes(include="number").columns)
            if not num_cols:
                ax.text(0.5, 0.5, "No numeric columns found",
                        ha="center", va="center", color="#8090aa",
                        transform=ax.transAxes)
                ax.set_axis_off()
                self._canvas.draw()
                return

            # Pick best columns: prefer error/runtime columns
            def score(c):
                lc = c.lower()
                for kw in ("error", "pos", "vel", "runtime", "nis", "condition"):
                    if kw in lc:
                        return 0
                return 1

            selected = sorted(num_cols, key=score)[:6]

            if plot_type == "Bar chart":
                self._draw_bar(ax, df, selected, compare_type)
            elif plot_type == "Box plot":
                self._draw_box(ax, df, selected, compare_type)
            elif plot_type == "Time history":
                self._draw_time(ax, df, selected, compare_type)
            else:  # Error table → just show bar chart of first few
                self._draw_bar(ax, df, selected, compare_type)

            ax.set_title(f"{compare_type}  ·  {folder.name}", fontsize=12, pad=10)
            ax.grid(True, axis="y", alpha=0.3)
            self._canvas.draw()

    def _render_domain_comparison(self, ax, df, compare_type: str, plot_type: str, folder: Path) -> bool:
        # Handle metric/value pivot tables (aggregate summary CSVs)
        metric_value = {"metric", "value"}.issubset(set(df.columns))
        if metric_value:
            metric_rows = df.copy()
            metric_rows["metric"] = metric_rows["metric"].astype(str)
            priority = ["position", "error", "velocity", "runtime", "success", "nis", "condition"]

            def _mrow_score(m: str) -> int:
                ml = m.lower()
                for i, kw in enumerate(priority):
                    if kw in ml:
                        return i
                return 99

            chosen = metric_rows[
                metric_rows["metric"].str.lower().str.contains(
                    "|".join(priority), regex=True,
                )
            ].copy()
            chosen["_score"] = chosen["metric"].map(_mrow_score)
            chosen = chosen.sort_values("_score").head(10)
            if chosen.empty:
                return False
            values = [float(v) if _is_number(v) else np.nan for v in chosen["value"]]
            labels = [str(m)[:32] for m in chosen["metric"]]
            x = np.arange(len(labels))
            ax.bar(x, values, color=[_ACCENT_COLORS[i % len(_ACCENT_COLORS)] for i in range(len(labels))], alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
            ax.set_ylabel("Metric value")
            ax.set_title(f"{compare_type}  ·  {folder.name}", fontsize=12, pad=10)
            ax.grid(True, axis="y", alpha=0.3)
            return True

        # Find the best position-error column from priority list
        y_col = _first_existing(df, _POS_ERROR_CANDIDATES)
        if y_col is None:
            return False

        # Find the grouping column from priority list for this comparison type
        group_col_names = _GROUP_COL_MAP.get(compare_type, [])
        group_col = _first_existing(df, group_col_names)
        if group_col is None:
            group_col = _first_existing(df, ["scenario", "label", "case", "name"])
        if group_col is None:
            return False

        extra_cols = [c for c in ["arc_id"] if c in df.columns]
        plot_df = df[[group_col, y_col] + extra_cols].copy()
        plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
        plot_df[group_col] = plot_df[group_col].astype(str)
        plot_df = plot_df.dropna(subset=[y_col])
        if plot_df.empty:
            return False

        groups = [
            (name, grp[y_col].to_numpy(dtype=float))
            for name, grp in plot_df.groupby(group_col, sort=False)
        ]
        groups = [(name, vals[np.isfinite(vals)]) for name, vals in groups]
        groups = [(name, vals) for name, vals in groups if vals.size]
        if not groups:
            return False

        y_label = _Y_AXIS_LABEL.get(y_col, f"{y_col} (m)")

        if plot_type == "Box plot":
            data = [vals for _, vals in groups]
            bp = ax.boxplot(data, patch_artist=True, notch=False,
                            medianprops={"color": "#00d4ff", "linewidth": 2})
            for patch, color in zip(bp["boxes"], _ACCENT_COLORS):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
            ax.set_xticklabels([name[:22] for name, _ in groups], rotation=28, ha="right", fontsize=9)
            ax.set_yscale("log")
            ax.set_ylabel(y_label)
            ax.set_title(f"{compare_type}  ·  error distribution", fontsize=12, pad=10)
            ax.grid(True, which="both", axis="y", alpha=0.25)
            return True

        if plot_type == "Time history" and "arc_id" in plot_df.columns:
            for i, (name, grp) in enumerate(plot_df.groupby(group_col, sort=False)):
                grp = grp.sort_values("arc_id")
                ax.semilogy(
                    grp["arc_id"],
                    np.maximum(pd.to_numeric(grp[y_col], errors="coerce"), 1e-6),
                    marker="o", linewidth=1.5, markersize=3,
                    color=_ACCENT_COLORS[i % len(_ACCENT_COLORS)],
                    label=name[:28],
                )
            ax.set_xlabel("Arc")
            ax.set_ylabel(y_label)
            ax.set_title(f"{compare_type}  ·  arc history", fontsize=12, pad=10)
            ax.legend(fontsize=8)
            ax.grid(True, which="both", alpha=0.25)
            return True

        # Default: median bar chart with p95 whisker
        labels = [name for name, _ in groups]
        med = np.array([np.median(vals) for _, vals in groups], dtype=float)
        p95 = np.array([np.percentile(vals, 95) for _, vals in groups], dtype=float)
        p5  = np.array([np.percentile(vals, 5)  for _, vals in groups], dtype=float)
        upper = np.maximum(p95 - med, 0.0)
        lower = np.maximum(med - p5, 0.0)
        x = np.arange(len(labels))
        ax.bar(
            x, med,
            yerr=[lower, upper],
            capsize=5,
            color=[_ACCENT_COLORS[i % len(_ACCENT_COLORS)] for i in range(len(labels))],
            alpha=0.86,
            error_kw={"ecolor": "#8090aa", "linewidth": 1.5, "capthick": 1.5},
        )
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([label[:22] for label in labels], rotation=28, ha="right", fontsize=9)
        ax.set_ylabel(f"Median  ·  {y_label}  (p5–p95 bars)")
        ax.set_title(f"{compare_type}  ·  {folder.name}", fontsize=12, pad=10)
        ax.grid(True, which="both", axis="y", alpha=0.25)
        for xi, (val, n_obs) in enumerate(zip(med, [vals.size for _, vals in groups])):
            ax.text(xi, val * 1.15, f"{val:.0f}", ha="center", va="bottom",
                    fontsize=8, color="#c8d4e8")
            ax.text(xi, val * 0.6, f"n={n_obs}", ha="center", va="top",
                    fontsize=7, color="#6070a0")
        return True

    def _draw_bar(self, ax, df, cols: list[str], title: str) -> None:
        means = [df[c].mean() for c in cols]
        stds = [df[c].std() for c in cols]
        x = range(len(cols))
        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color=_ACCENT_COLORS[:len(cols)],
                      error_kw={"ecolor": "#8090aa", "linewidth": 1.5})
        ax.set_xticks(list(x))
        ax.set_xticklabels([c[:20] for c in cols], rotation=30, ha="right", fontsize=10)
        ax.set_ylabel("Mean value", fontsize=11)

    def _draw_box(self, ax, df, cols: list[str], title: str) -> None:
        data = [df[c].dropna().values for c in cols if len(df[c].dropna()) > 0]
        if not data:
            return
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops={"color": "#00d4ff", "linewidth": 2})
        for patch, color in zip(bp["boxes"], _ACCENT_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticklabels([c[:20] for c in cols[:len(data)]], rotation=30, ha="right", fontsize=10)
        ax.set_ylabel("Value", fontsize=11)

    def _draw_time(self, ax, df, cols: list[str], title: str) -> None:
        # Look for an index-like column
        time_col = None
        for c in df.columns:
            if any(kw in c.lower() for kw in ("time", "epoch", "arc", "step", "idx")):
                time_col = c
                break
        x = df[time_col] if time_col else range(len(df))
        for i, c in enumerate(cols[:4]):
            ax.plot(x, df[c], color=_ACCENT_COLORS[i], linewidth=1.5,
                    label=c[:24], alpha=0.9)
        ax.set_xlabel(time_col or "Index", fontsize=11)
        ax.set_ylabel("Value", fontsize=11)
        ax.legend(fontsize=9)

    def _draw_message(self, msg: str) -> None:
        if not HAS_MPL or self._figure is None:
            return
        with matplotlib.rc_context(_MPL_STYLE):
            self._figure.clear()
            ax = self._figure.add_subplot(111)
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    color="#8090aa", fontsize=13, transform=ax.transAxes)
            ax.set_axis_off()
            self._canvas.draw()

    def _draw_no_pandas(self) -> None:
        self._draw_message("Install pandas to load CSV data.")

    # ------------------------------------------------------------------
    def _populate_summary_table(self, df) -> None:
        self.summaryTable.clear()
        self.summaryTable.setRowCount(min(len(df), 500))
        self.summaryTable.setColumnCount(len(df.columns))
        self.summaryTable.setHorizontalHeaderLabels(list(df.columns))
        for r, row in enumerate(df.head(500).itertuples(index=False)):
            for c, val in enumerate(row):
                item = QTableWidgetItem(str(val) if val is not None else "")
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.summaryTable.setItem(r, c, item)
        self.summaryTable.resizeColumnsToContents()

    # ------------------------------------------------------------------
    def _update_png_combo(self) -> None:
        idx = self.resultFolderCombo.currentIndex()
        if idx < 0:
            return
        folder: Path = self.resultFolderCombo.itemData(idx)
        if folder is None:
            return
        self.pngSelectCombo.clear()
        for p in sorted(folder.glob("*.png")):
            self.pngSelectCombo.addItem(p.name, userData=p)

    def _load_raw_png(self) -> None:
        idx = self.pngSelectCombo.currentIndex()
        if idx < 0:
            return
        path: Path = self.pngSelectCombo.itemData(idx)
        if path is None or not path.exists():
            return
        from PyQt5.QtGui import QPixmap
        pix = QPixmap(str(path))
        if pix.isNull():
            return
        scroll = self.rawPngScroll
        lbl = self.rawPngLabel
        avail = scroll.viewport().size()
        scaled = pix.scaled(avail.width() - 20, avail.height() - 20,
                             Qt.KeepAspectRatio, Qt.SmoothTransformation)
        lbl.setPixmap(scaled)

    # ------------------------------------------------------------------
    def _export_plot(self) -> None:
        if not HAS_MPL or self._figure is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export plot", str(Path.home() / "comparison.png"),
            "PNG images (*.png);;PDF files (*.pdf)",
        )
        if path:
            self._figure.savefig(path, dpi=150, bbox_inches="tight",
                                 facecolor=self._figure.get_facecolor())


def _first_existing(df, names: list[str]) -> str | None:
    lowered = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _group_column(df, compare_type: str) -> str | None:
    candidates = _GROUP_COL_MAP.get(compare_type)
    if candidates:
        return _first_existing(df, candidates)
    return None


def _is_number(value) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False
