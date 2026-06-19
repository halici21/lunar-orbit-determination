"""Stations page — world map, precision table, station editing and network presets."""
from __future__ import annotations
import copy
import csv as _csv
import json
from pathlib import Path
from typing import Any

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QLineEdit, QLabel, QDialog, QFormLayout,
    QDoubleSpinBox, QDialogButtonBox, QMessageBox, QComboBox, QInputDialog,
    QFileDialog,
)
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QColor
from PyQt5 import uic

from services.project_paths import DESKTOP_APP_DIR, PYTHON_PORT
from services.world_map import draw_land
from styles.theme import C, FONT, MPL_DARK, NET_COLORS

UI_PATH = DESKTOP_APP_DIR / "ui" / "pages" / "stations_page.ui"
STATION_CFG = PYTHON_PORT / "fixtures" / "station_config.json"
_SETTINGS = QSettings("LunarOD", "DesktopApp")

_NET_COLORS = NET_COLORS
_DEFAULT_COLOR = C.TEXT_MUTED
_MPL_STYLE = MPL_DARK

_COL_HEADERS = [
    "Station", "Network", "Lat (°)", "Lon (°)",
    "σ_range (m)", "σ_angle (mrad)", "σ_rr (mm/s)",
]
_BTN_STYLE = (
    f"QPushButton {{ background:{C.BG_HOVER}; color:{C.TEXT_PRIMARY};"
    f" border:1px solid {C.BORDER_MAIN}; border-radius:4px;"
    f" padding:4px 12px; font-size:{FONT.SIZE_SM}px; }}"
    f"QPushButton:hover {{ background:{C.BG_ACTIVE}; }}"
    f"QPushButton:disabled {{ color:{C.TEXT_MUTED}; border-color:{C.BORDER_FAINT}; }}"
)
_SEARCH_STYLE = (
    f"QLineEdit {{ background:{C.BG_DEEP}; color:{C.TEXT_PRIMARY};"
    f" border:1px solid {C.BORDER_MAIN}; border-radius:4px;"
    f" padding:3px 8px; font-size:{FONT.SIZE_SM}px; }}"
)


def _network(name: str) -> str:
    for word in name.strip().split():
        if word in _NET_COLORS:
            return word
    return "OTHER"


def _load_stations() -> list[dict]:
    """Return deduplicated station list from station_config.json."""
    try:
        data = json.loads(STATION_CFG.read_text(encoding="utf-8"))
    except Exception:
        return []
    seen: dict[str, dict] = {}
    for section in data.values():
        if not isinstance(section, dict):
            continue
        for s in section.get("stations", []):
            name = s.get("name", "")
            if name and name not in seen:
                seen[name] = s
    result = []
    for name, s in seen.items():
        result.append({
            "name":             name,
            "network":          _network(name),
            "lat_deg":          float(s.get("lat_deg", 0.0)),
            "lon_deg":          float(s.get("lon_deg", 0.0)),
            "sigma_range_m":    s.get("sigma_range_m"),
            "sigma_angle_rad":  s.get("sigma_angle_rad"),
            "sigma_rr_mps":     s.get("sigma_range_rate_mps"),
        })
    result.sort(key=lambda x: (x["network"], x["name"]))
    return result


def _load_custom_stations() -> list[dict]:
    raw = _SETTINGS.value("stations/custom_list", [])
    return list(raw) if isinstance(raw, list) else []


def _save_custom_stations(stations: list[dict]) -> None:
    _SETTINGS.setValue("stations/custom_list", stations)


def _load_custom_networks() -> dict[str, list[str]]:
    raw = _SETTINGS.value("stations/custom_networks", {})
    return dict(raw) if isinstance(raw, dict) else {}


def _save_custom_networks(nets: dict[str, list[str]]) -> None:
    _SETTINGS.setValue("stations/custom_networks", nets)


def _build_map(stations: list[dict]):
    import matplotlib
    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    with plt.rc_context(_MPL_STYLE):
        fig = Figure(figsize=(11, 4.5), tight_layout={"pad": 0.8})
        ax = fig.add_subplot(111)

    draw_land(ax)

    for lat in range(-90, 91, 30):
        ax.axhline(lat, color=C.BORDER_FAINT, linewidth=0.4, zorder=2)
    for lon in range(-180, 181, 30):
        ax.axvline(lon, color=C.BORDER_FAINT, linewidth=0.4, zorder=2)

    ax.axhline(0,    color=C.BORDER_EQUATOR, linewidth=1.0, zorder=3)
    for lat in [23.5, -23.5]:
        ax.axhline(lat, color=C.BORDER_TROPIC, linewidth=0.7, linestyle="--", zorder=3)
    for lat in [66.5, -66.5]:
        ax.axhline(lat, color=C.BORDER_POLAR, linewidth=0.6, linestyle=":", zorder=3)

    for s in stations:
        color = _NET_COLORS.get(s["network"], _DEFAULT_COLOR)
        marker = "*" if s.get("_custom") else "o"
        ax.scatter(s["lon_deg"], s["lat_deg"], c=color, s=120 if marker == "o" else 200,
                   zorder=6, edgecolors=C.TEXT_SECONDARY, linewidth=0.7, marker=marker,
                   picker=5, label=s["name"])
        short = s["name"].split()[0]
        ax.annotate(short, (s["lon_deg"], s["lat_deg"]),
                    xytext=(6, 5), textcoords="offset points",
                    fontsize=FONT.SIZE_XS - 0.5, color=color, zorder=7,
                    fontfamily=FONT.MONO)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-90, 91, 30))
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Longitude (°)", fontsize=FONT.SIZE_SM - 1)
    ax.set_ylabel("Latitude (°)",  fontsize=FONT.SIZE_SM - 1)

    for spine in ax.spines.values():
        spine.set_edgecolor(C.BORDER_MAIN)

    visible_networks = {s["network"] for s in stations}
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=c, markersize=8, label=net, linestyle="None")
        for net, c in _NET_COLORS.items() if net in visible_networks
    ]
    if any(s.get("_custom") for s in stations):
        handles.append(
            Line2D([0], [0], marker="*", color="w",
                   markerfacecolor=C.YELLOW, markersize=10,
                   label="Custom", linestyle="None")
        )
    leg = ax.legend(handles=handles, loc="lower left",
                    framealpha=0.6, facecolor=C.BG_PANEL,
                    edgecolor=C.BORDER_MID, fontsize=FONT.SIZE_XS)
    for text in leg.get_texts():
        text.set_color(C.TEXT_SECONDARY)

    return fig


# ---------------------------------------------------------------------------
class _StationDialog(QDialog):
    """Add / edit a station record."""

    def __init__(self, station: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Station" if station else "Add Station")
        self.setMinimumWidth(340)
        self._station = copy.deepcopy(station) if station else {
            "name": "", "network": "OTHER",
            "lat_deg": 0.0, "lon_deg": 0.0,
            "sigma_range_m": 5.0, "sigma_angle_rad": None, "sigma_rr_mps": 0.001,
            "_custom": True,
        }
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setSpacing(8)
        layout.addLayout(form)

        def _spin(lo, hi, val, decimals=4, suffix=""):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(decimals)
            s.setValue(float(val) if val is not None else 0.0)
            if suffix:
                s.setSuffix(f" {suffix}")
            s.setStyleSheet(
                f"QDoubleSpinBox {{ background:{C.BG_DEEP}; color:{C.TEXT_PRIMARY};"
                f" border:1px solid {C.BORDER_MAIN}; border-radius:3px; padding:2px; }}"
            )
            return s

        self._name_edit = QLineEdit(self._station.get("name", ""))
        self._name_edit.setStyleSheet(_SEARCH_STYLE)
        form.addRow("Name:", self._name_edit)

        self._net_combo = QComboBox()
        for net in list(_NET_COLORS.keys()) + ["OTHER"]:
            self._net_combo.addItem(net)
        cur_net = self._station.get("network", "OTHER")
        idx = self._net_combo.findText(cur_net)
        if idx >= 0:
            self._net_combo.setCurrentIndex(idx)
        form.addRow("Network:", self._net_combo)

        self._lat = _spin(-90, 90, self._station.get("lat_deg", 0.0), 4, "°")
        form.addRow("Latitude:", self._lat)

        self._lon = _spin(-180, 180, self._station.get("lon_deg", 0.0), 4, "°")
        form.addRow("Longitude:", self._lon)

        sr = self._station.get("sigma_range_m") or 5.0
        self._sr = _spin(0.01, 9999, sr, 2, "m")
        form.addRow("σ range:", self._sr)

        sa_rad = self._station.get("sigma_angle_rad")
        sa_mrad = (sa_rad * 1e3) if sa_rad is not None else 0.0
        self._sa = _spin(0.0, 100, sa_mrad, 4, "mrad")
        form.addRow("σ angle:", self._sa)

        rr_mps = self._station.get("sigma_rr_mps") or 0.001
        rr_mm = rr_mps * 1e3
        self._rr = _spin(0.0, 9999, rr_mm, 4, "mm/s")
        form.addRow("σ range-rate:", self._rr)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _accept(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation", "Station name cannot be empty.")
            return
        self._station["name"] = name
        self._station["network"] = self._net_combo.currentText()
        self._station["lat_deg"] = self._lat.value()
        self._station["lon_deg"] = self._lon.value()
        self._station["sigma_range_m"] = self._sr.value()
        sa_mrad = self._sa.value()
        self._station["sigma_angle_rad"] = (sa_mrad / 1e3) if sa_mrad > 0 else None
        rr_mm = self._rr.value()
        self._station["sigma_rr_mps"] = (rr_mm / 1e3) if rr_mm > 0 else None
        self._station["_custom"] = True
        self.accept()

    def result_station(self) -> dict:
        return self._station


# ---------------------------------------------------------------------------
class StationsController(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        uic.loadUi(str(UI_PATH), self)
        self._canvas = None
        self._all_stations: list[dict] = []
        self._filtered_stations: list[dict] = []
        self._setup()

    def _setup(self) -> None:
        self._all_stations = _load_stations() + [
            dict(s, _custom=True) for s in _load_custom_stations()
        ]

        # Build map
        self._build_map_widget(self._all_stations)

        # Toolbar: search + action buttons
        toolbar = QWidget(self)
        tb_lay = QHBoxLayout(toolbar)
        tb_lay.setContentsMargins(0, 4, 0, 4)
        tb_lay.setSpacing(8)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search stations…")
        self._search.setStyleSheet(_SEARCH_STYLE)
        self._search.setMaximumWidth(220)
        self._search.textChanged.connect(self._apply_filter)
        tb_lay.addWidget(self._search)
        tb_lay.addStretch()

        lbl = QLabel("Selection:")
        lbl.setStyleSheet(f"color:{C.TEXT_MUTED}; font-size:{FONT.SIZE_SM}px;")
        tb_lay.addWidget(lbl)

        self._btn_save_net = QPushButton("Save as Network Preset")
        self._btn_save_net.setStyleSheet(_BTN_STYLE)
        self._btn_save_net.setEnabled(False)
        self._btn_save_net.clicked.connect(self._save_network_preset)
        tb_lay.addWidget(self._btn_save_net)

        self._btn_export = QPushButton("Export CSV")
        self._btn_export.setStyleSheet(_BTN_STYLE)
        self._btn_export.setToolTip("Export displayed stations to CSV")
        self._btn_export.clicked.connect(self._export_csv)
        tb_lay.addWidget(self._btn_export)

        self._btn_add = QPushButton("Add Station")
        self._btn_add.setStyleSheet(_BTN_STYLE)
        self._btn_add.clicked.connect(self._add_station)
        tb_lay.addWidget(self._btn_add)

        self._btn_edit = QPushButton("Edit")
        self._btn_edit.setStyleSheet(_BTN_STYLE)
        self._btn_edit.setEnabled(False)
        self._btn_edit.clicked.connect(self._edit_station)
        tb_lay.addWidget(self._btn_edit)

        self._btn_delete = QPushButton("Delete")
        self._btn_delete.setStyleSheet(
            _BTN_STYLE.replace(f"background:{C.BG_HOVER}", f"background:{C.BG_HOVER}")
            .replace(f"border:1px solid {C.BORDER_MAIN}", f"border:1px solid {C.RED}")
        )
        self._btn_delete.setEnabled(False)
        self._btn_delete.clicked.connect(self._delete_station)
        tb_lay.addWidget(self._btn_delete)

        # Insert toolbar above table
        self.mainLayout.insertWidget(1, toolbar)

        # Table
        self._populate_table(self._all_stations)
        self.stationsTable.selectionModel().selectionChanged.connect(self._on_selection)

    def on_shown(self) -> None:
        pass

    # ------------------------------------------------------------------
    def _build_map_widget(self, stations: list[dict]) -> None:
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            fig = _build_map(stations)
            canvas = FigureCanvasQTAgg(fig)
            canvas.mpl_connect("pick_event", self._on_map_pick)
            self._canvas = canvas
        except Exception as exc:
            canvas = QLabel(f"Map failed to load: {exc}")
            canvas.setAlignment(Qt.AlignCenter)
            canvas.setStyleSheet(f"color:{C.RED};")

        holder = self.mapCanvasHolder
        lay = QVBoxLayout(holder)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(canvas)

    def _rebuild_map(self) -> None:
        if self._canvas is None:
            return
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            fig = _build_map(self._all_stations)
            canvas = FigureCanvasQTAgg(fig)
            canvas.mpl_connect("pick_event", self._on_map_pick)
            holder = self.mapCanvasHolder
            lay = holder.layout()
            while lay.count():
                item = lay.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            lay.addWidget(canvas)
            self._canvas = canvas
        except Exception:
            pass

    def _on_map_pick(self, event) -> None:
        try:
            from matplotlib.collections import PathCollection
        except ImportError:
            return
        if not isinstance(event.artist, PathCollection):
            return
        station_name = event.artist.get_label()
        if not station_name or station_name.startswith("_"):
            return
        # Clear search filter and reset to all stations so the row is visible
        self._search.blockSignals(True)
        self._search.clear()
        self._search.blockSignals(False)
        self._populate_table(self._all_stations)
        for row, s in enumerate(self._filtered_stations):
            if s["name"] == station_name:
                self.stationsTable.selectRow(row)
                self.stationsTable.scrollToItem(self.stationsTable.item(row, 0))
                break

    # ------------------------------------------------------------------
    def _populate_table(self, stations: list[dict]) -> None:
        self._filtered_stations = stations
        tbl = self.stationsTable
        tbl.setColumnCount(len(_COL_HEADERS))
        tbl.setHorizontalHeaderLabels(_COL_HEADERS)
        tbl.setRowCount(len(stations))
        tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, len(_COL_HEADERS)):
            tbl.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)

        for row, s in enumerate(stations):
            net = s["network"]
            net_color = QColor(_NET_COLORS.get(net, _DEFAULT_COLOR))
            is_custom = s.get("_custom", False)

            def cell(text: str, color: QColor | None = None,
                     bold: bool = False) -> QTableWidgetItem:
                item = QTableWidgetItem(str(text))
                item.setTextAlignment(Qt.AlignCenter)
                if color:
                    item.setForeground(color)
                if bold:
                    from PyQt5.QtGui import QFont as _QFont
                    f = item.font()
                    f.setBold(True)
                    item.setFont(f)
                return item

            name_item = cell(s["name"], bold=is_custom)
            if is_custom:
                name_item.setForeground(QColor(C.YELLOW))
                name_item.setToolTip("Custom station (saved in settings)")
            tbl.setItem(row, 0, name_item)
            tbl.setItem(row, 1, cell(net, net_color))
            tbl.setItem(row, 2, cell(f"{s['lat_deg']:.2f}"))
            tbl.setItem(row, 3, cell(f"{s['lon_deg']:.2f}"))

            sr = s["sigma_range_m"]
            sr_color = QColor(C.YELLOW) if sr and sr > 10 else QColor(C.GREEN)
            tbl.setItem(row, 4, cell(f"{sr:.1f}" if sr is not None else "—", sr_color))

            sa = s["sigma_angle_rad"]
            sa_mrad = sa * 1e3 if sa is not None else None
            sa_color = QColor(C.YELLOW) if sa_mrad and sa_mrad > 0.05 else QColor(C.GREEN)
            tbl.setItem(row, 5, cell(f"{sa_mrad:.3f}" if sa_mrad is not None else "—", sa_color))

            rr = s["sigma_rr_mps"]
            rr_mm = rr * 1e3 if rr is not None else None
            tbl.setItem(row, 6, cell(f"{rr_mm:.3f}" if rr_mm is not None else "—"))

        self._on_selection()

    def _apply_filter(self, text: str) -> None:
        q = text.lower().strip()
        if not q:
            self._populate_table(self._all_stations)
        else:
            filtered = [
                s for s in self._all_stations
                if q in s["name"].lower() or q in s["network"].lower()
            ]
            self._populate_table(filtered)

    # ------------------------------------------------------------------
    def _on_selection(self) -> None:
        rows = self._selected_rows()
        self._btn_save_net.setEnabled(len(rows) >= 1)
        single_selected = len(rows) == 1
        self._btn_edit.setEnabled(single_selected)
        if single_selected:
            station = self._filtered_stations[rows[0]]
            self._btn_delete.setEnabled(station.get("_custom", False))
        else:
            self._btn_delete.setEnabled(False)

    def _selected_rows(self) -> list[int]:
        return sorted({idx.row() for idx in self.stationsTable.selectedIndexes()})

    # ------------------------------------------------------------------
    def _add_station(self) -> None:
        dlg = _StationDialog(parent=self)
        if dlg.exec_() == QDialog.Accepted:
            new_s = dlg.result_station()
            custom = _load_custom_stations()
            custom.append(new_s)
            _save_custom_stations(custom)
            self._all_stations.append(dict(new_s, _custom=True))
            self._apply_filter(self._search.text())
            self._rebuild_map()

    def _edit_station(self) -> None:
        rows = self._selected_rows()
        if len(rows) != 1:
            return
        station = self._filtered_stations[rows[0]]
        dlg = _StationDialog(station=station, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            updated = dlg.result_station()
            # Update in _all_stations
            for i, s in enumerate(self._all_stations):
                if s["name"] == station["name"]:
                    self._all_stations[i] = updated
                    break
            # Persist if custom
            if updated.get("_custom"):
                custom = _load_custom_stations()
                for i, s in enumerate(custom):
                    if s["name"] == station["name"]:
                        custom[i] = updated
                        break
                _save_custom_stations(custom)
            self._apply_filter(self._search.text())
            self._rebuild_map()

    def _delete_station(self) -> None:
        rows = self._selected_rows()
        if len(rows) != 1:
            return
        station = self._filtered_stations[rows[0]]
        if not station.get("_custom"):
            QMessageBox.information(self, "Cannot delete",
                "Only custom stations can be deleted.\n"
                "Built-in stations come from the fixture file.")
            return
        reply = QMessageBox.question(
            self, "Delete station?",
            f"Delete custom station '{station['name']}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._all_stations = [s for s in self._all_stations if s["name"] != station["name"]]
        custom = [s for s in _load_custom_stations() if s["name"] != station["name"]]
        _save_custom_stations(custom)
        self._apply_filter(self._search.text())
        self._rebuild_map()

    def _export_csv(self) -> None:
        stations = self._filtered_stations if self._filtered_stations else self._all_stations
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Stations CSV",
            str(Path.home() / "stations.csv"),
            "CSV files (*.csv)",
        )
        if not path:
            return
        fieldnames = ["name", "network", "lat_deg", "lon_deg",
                      "sigma_range_m", "sigma_angle_rad", "sigma_rr_mps"]
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = _csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for s in stations:
                    writer.writerow({k: s.get(k, "") for k in fieldnames})
            QMessageBox.information(
                self, "Export complete",
                f"Exported {len(stations)} station(s) to:\n{path}",
            )
        except Exception as exc:
            QMessageBox.warning(self, "Export failed", str(exc))

    def _save_network_preset(self) -> None:
        rows = self._selected_rows()
        if not rows:
            return
        selected_names = [self._filtered_stations[r]["name"] for r in rows]
        name, ok = QInputDialog.getText(
            self, "Save Network Preset",
            f"Preset name for {len(selected_names)} station(s):",
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        nets = _load_custom_networks()
        nets[name] = selected_names
        _save_custom_networks(nets)
        QMessageBox.information(
            self, "Saved",
            f"Network preset '{name}' saved with {len(selected_names)} station(s).\n\n"
            f"Stations: {', '.join(selected_names)}",
        )
