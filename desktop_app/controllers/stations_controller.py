"""Stations page — world map, precision table and custom-station editing."""
from __future__ import annotations
import copy
import csv as _csv
from html import escape
import json
from pathlib import Path
from typing import Any

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QLineEdit, QLabel, QDialog, QFormLayout,
    QDoubleSpinBox, QDialogButtonBox, QMessageBox, QComboBox,
    QFileDialog, QShortcut, QSizePolicy, QBoxLayout,
)
from PyQt5.QtCore import Qt, QSettings, QSignalBlocker
from PyQt5.QtGui import QColor, QKeySequence
from PyQt5 import uic

from services.project_paths import DESKTOP_APP_DIR, PYTHON_PORT
from services.world_map import draw_land
from styles.theme import C, FONT, MPL_DARK, NET_COLORS
from styles.workspace_light import set_role

UI_PATH = DESKTOP_APP_DIR / "ui" / "pages" / "stations_page.ui"
STATION_CFG = PYTHON_PORT / "fixtures" / "station_config.json"
_SETTINGS = QSettings("LunarOD", "DesktopApp")

_NET_COLORS = NET_COLORS
_DEFAULT_COLOR = C.TEXT_MUTED
_MPL_STYLE = MPL_DARK

_COL_HEADERS = [
    "İstasyon", "Ağ", "Enlem (°)", "Boylam (°)",
    "σ menzil (m)", "σ açı (mrad)", "σ menzil hızı (mm/s)",
]



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



def _build_map(stations: list[dict]):
    import matplotlib
    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    from widgets.catalogue_map_labels import CatalogueMapLabels

    with plt.rc_context(_MPL_STYLE):
        fig = Figure(figsize=(11, 4.5), tight_layout={"pad": 0.8})
        ax = fig.add_subplot(111)

    draw_land(ax, land_color=C.BG_PANEL, edge_color=C.BORDER_MID, ocean_color=C.BG_DEEP)

    for lat in range(-90, 91, 30):
        ax.axhline(lat, color=C.BORDER_FAINT, linewidth=0.4, zorder=2)
    for lon in range(-180, 181, 30):
        ax.axvline(lon, color=C.BORDER_FAINT, linewidth=0.4, zorder=2)

    ax.axhline(0,    color=C.BORDER_EQUATOR, linewidth=1.0, zorder=3)
    for lat in [23.5, -23.5]:
        ax.axhline(lat, color=C.BORDER_TROPIC, linewidth=0.7, linestyle="--", zorder=3)
    for lat in [66.5, -66.5]:
        ax.axhline(lat, color=C.BORDER_POLAR, linewidth=0.6, linestyle=":", zorder=3)

    labels = []
    for s in stations:
        color = _NET_COLORS.get(s["network"], _DEFAULT_COLOR)
        marker = "*" if s.get("_custom") else "o"
        ax.scatter(s["lon_deg"], s["lat_deg"], c=color, s=120 if marker == "o" else 200,
                   zorder=6, edgecolors=C.TEXT_SECONDARY, linewidth=0.7, marker=marker,
                   picker=5, label=s["name"])
        name = s["name"].removesuffix(" " + s["network"])
        label = ax.annotate(name, (s["lon_deg"], s["lat_deg"]),
                            xytext=(10, 10), textcoords="offset points",
                            fontsize=FONT.SIZE_XS - 0.5, color=color, zorder=7,
                            fontfamily="sans-serif", picker=Text.contains, gid=s["name"], parse_math=False,
                            arrowprops={"arrowstyle": "-", "color": C.TEXT_MUTED,
                                        "linewidth": .5, "shrinkA": 2, "shrinkB": 7})
        label.set_in_layout(False)
        labels.append(label)
    CatalogueMapLabels(ax, labels)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-90, 91, 30))
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Boylam (°)", fontsize=FONT.SIZE_SM - 1)
    ax.set_ylabel("Enlem (°)",  fontsize=FONT.SIZE_SM - 1)

    for spine in ax.spines.values():
        spine.set_edgecolor(C.BORDER_MAIN)

    return fig


class _StationNumericItem(QTableWidgetItem):
    """Keep display precision separate from numeric ordering and absent values."""

    def __init__(self, value: float | None, decimals: int, unit: str,
                 color: QColor | None = None) -> None:
        super().__init__(f"{value:.{decimals}f}" if value is not None else "—")
        self.setData(Qt.UserRole, value)
        self.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.setToolTip(f"{value} {unit}" if value is not None else "Tanımlı değil (—)")
        if color is not None:
            self.setForeground(color)

    def __lt__(self, other: QTableWidgetItem) -> bool:
        value, other_value = self.data(Qt.UserRole), other.data(Qt.UserRole)
        # Missing sorts after every number ascending; descending reverses that order.
        return ((value is None, value if value is not None else 0.0)
                < (other_value is None, other_value if other_value is not None else 0.0))


# ---------------------------------------------------------------------------
class _StationDialog(QDialog):
    """Add / edit a station record."""

    def __init__(self, station: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("İstasyonu düzenle" if station else "İstasyon ekle")
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
            return s

        self._name_edit = QLineEdit(self._station.get("name", ""))

        form.addRow("Ad:", self._name_edit)

        self._net_combo = QComboBox()
        for net in list(_NET_COLORS.keys()) + ["OTHER"]:
            self._net_combo.addItem(net)
        cur_net = self._station.get("network", "OTHER")
        idx = self._net_combo.findText(cur_net)
        if idx >= 0:
            self._net_combo.setCurrentIndex(idx)
        form.addRow("Ağ:", self._net_combo)

        self._lat = _spin(-90, 90, self._station.get("lat_deg", 0.0), 4, "°")
        form.addRow("Enlem:", self._lat)

        self._lon = _spin(-180, 180, self._station.get("lon_deg", 0.0), 4, "°")
        form.addRow("Boylam:", self._lon)

        sr = self._station.get("sigma_range_m") or 5.0
        self._sr = _spin(0.01, 9999, sr, 2, "m")
        form.addRow("σ menzil:", self._sr)

        sa_rad = self._station.get("sigma_angle_rad")
        sa_mrad = (sa_rad * 1e3) if sa_rad is not None else 0.0
        self._sa = _spin(0.0, 100, sa_mrad, 4, "mrad")
        form.addRow("σ açı:", self._sa)

        rr_mps = self._station.get("sigma_rr_mps") or 0.001
        rr_mm = rr_mps * 1e3
        self._rr = _spin(0.0, 9999, rr_mm, 4, "mm/s")
        form.addRow("σ menzil hızı:", self._rr)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("Kaydet")
        btns.button(QDialogButtonBox.Cancel).setText("Vazgeç")
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _accept(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Eksik bilgi", "İstasyon adı boş olamaz.")
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
        # Keep the key outside the geographic area so it cannot cover stations.
        self._map_legend = QLabel()
        self._map_legend.setTextFormat(Qt.RichText)
        self._map_legend.setWordWrap(True)
        self._map_legend.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self._map_legend.setStyleSheet(f"color:{C.TEXT_SECONDARY}; font-size:{FONT.SIZE_SM}px;")
        self._map_legend.setAccessibleName("Harita ağ renkleri ve istasyon işaretleri")
        self.mainLayout.insertWidget(1, self._map_legend)
        self._update_map_legend()

        # Toolbar: search + action buttons
        toolbar = QWidget(self)
        # The outer row must be allowed to shrink before its groups can stack.
        toolbar.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        self._toolbar = toolbar
        tb_lay = QBoxLayout(QBoxLayout.LeftToRight, toolbar)
        self._toolbar_layout = tb_lay
        tb_lay.setContentsMargins(0, 4, 0, 4)
        tb_lay.setSpacing(8)
        self._catalogue_actions = QWidget(toolbar)
        self._selection_actions = QWidget(toolbar)
        catalogue_lay = QHBoxLayout(self._catalogue_actions)
        selection_lay = QHBoxLayout(self._selection_actions)
        for layout in (catalogue_lay, selection_lay):
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(8)
        tb_lay.addWidget(self._catalogue_actions, 1)
        tb_lay.addWidget(self._selection_actions, 1)

        self._search = QLineEdit()
        self._search.setPlaceholderText("İstasyon adı / ağ kodu…")
        self._search.setClearButtonEnabled(True)
        self._search.setAccessibleName("İstasyon adı veya ağ koduyla tabloyu filtrele")
        self._search.setToolTip("Ad veya ağ kodu (DSN, ESA…) ile tablodaki istasyonları filtreleyin. Ctrl+F: aramaya geç. Harita tüm kataloğu gösterir.")

        self._search.setMaximumWidth(220)
        self._search.textChanged.connect(self._apply_filter)
        self._find_shortcut = QShortcut(QKeySequence.Find, self)
        self._find_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._find_shortcut.activated.connect(self._focus_search)
        catalogue_lay.addWidget(self._search)
        catalogue_lay.addStretch()

        lbl = QLabel("Seçili istasyon:")
        set_role(lbl, "caption")
        selection_lay.addWidget(lbl)
        selection_lay.addStretch()

        self._btn_export = QPushButton("CSV aktar…")

        self._btn_export.setToolTip("Tabloda görünen istasyonları CSV dosyasına kaydet")
        self._btn_export.clicked.connect(self._export_csv)
        catalogue_lay.addWidget(self._btn_export)

        self._btn_add = QPushButton("İstasyon ekle")

        self._btn_add.clicked.connect(self._add_station)
        catalogue_lay.addWidget(self._btn_add)

        self._btn_edit = QPushButton("Düzenle")

        self._btn_edit.setEnabled(False)
        self._btn_edit.clicked.connect(self._edit_station)
        selection_lay.addWidget(self._btn_edit)

        self._btn_delete = QPushButton("Sil")
        set_role(self._btn_delete, "danger")
        self._btn_delete.setEnabled(False)
        self._btn_delete.clicked.connect(self._delete_station)
        selection_lay.addWidget(self._btn_delete)

        # Insert toolbar above table
        self.mainLayout.insertWidget(2, toolbar)
        self._filter_status = QLabel()
        self._filter_status.setWordWrap(True)
        self._filter_status.setMaximumWidth(520)
        self._filter_status.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        set_role(self._filter_status, "caption")
        self._filter_status.setAccessibleName("İstasyon tablosunun filtre ve seçim durumu")
        self.mainLayout.insertWidget(3, self._filter_status)

        # Table
        self._populate_table(self._all_stations)
        self.stationsTable.selectionModel().selectionChanged.connect(self._on_selection)

    def on_shown(self) -> None:
        pass

    def _update_map_legend(self) -> None:
        visible = {s["network"] for s in self._all_stations}
        networks = [net for net in _NET_COLORS if net in visible]
        networks.extend(sorted(visible - set(networks)))
        items = [
            f'<span style="color:{_NET_COLORS.get(net, _DEFAULT_COLOR)}">'
            f'●&nbsp;{escape(net)}</span>' for net in networks
        ]
        description = "Ağlar: " + ", ".join(networks)
        if any(s.get("_custom") for s in self._all_stations):
            items.append("★&nbsp;Özel&nbsp;istasyon")
            description += ". Yıldız: kullanıcı tarafından eklenen istasyon; rengi ait olduğu ağı gösterir."
        help_text = "Nokta / ad: tabloda seç. Alan yetmezse bazı adlar gizlenir; tüm istasyonlar tabloda kalır."
        description += ". " + help_text
        self._map_legend.setText("Ağlar: " + " &nbsp; ".join(items) + " · Nokta / ad: seç" if items else "")
        self._map_legend.setVisible(bool(items))
        self._map_legend.setAccessibleDescription(description)
        self._map_legend.setToolTip(description + " Harita, arama filtresinden bağımsız tüm kataloğu gösterir.")

    def resizeEvent(self, event) -> None:
        if hasattr(self, "_toolbar_layout"):
            margins = self.mainLayout.contentsMargins()
            available = self.width() - margins.left() - margins.right()
            row_width = (self._catalogue_actions.minimumSizeHint().width()
                         + self._selection_actions.minimumSizeHint().width()
                         + self._toolbar_layout.spacing())
            direction = (QBoxLayout.TopToBottom if available < row_width
                         else QBoxLayout.LeftToRight)
            if self._toolbar_layout.direction() != direction:
                self._toolbar_layout.setDirection(direction)
            self.mapCanvasHolder.setFixedHeight(max(240, min(440, available // 2 + 40)))
        super().resizeEvent(event)

    # ------------------------------------------------------------------
    def _build_map_widget(self, stations: list[dict]) -> None:
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            fig = _build_map(stations)
            canvas = FigureCanvasQTAgg(fig)
            canvas.mpl_connect("pick_event", self._on_map_pick)
            self._canvas = canvas
        except Exception as exc:
            canvas = QLabel(f"Harita yüklenemedi: {exc}")
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
            self._update_map_legend()
        except Exception:
            pass

    def _on_map_pick(self, event) -> None:
        try:
            from matplotlib.collections import PathCollection
            from matplotlib.text import Annotation
        except ImportError:
            return
        if isinstance(event.artist, Annotation):
            station_name = event.artist.get_gid()
        elif isinstance(event.artist, PathCollection):
            station_name = event.artist.get_label()
        else:
            return
        if not station_name or station_name.startswith("_"):
            return
        # Clear search filter and reset to all stations so the row is visible
        self._search.blockSignals(True)
        self._search.clear()
        self._search.blockSignals(False)
        self._populate_table(self._all_stations)
        for row in range(self.stationsTable.rowCount()):
            if self._station_at_row(row)["name"] == station_name:
                self.stationsTable.selectRow(row)
                self.stationsTable.scrollToItem(self.stationsTable.item(row, 0))
                break

    # ------------------------------------------------------------------
    def _populate_table(self, stations: list[dict]) -> None:
        tbl = self.stationsTable
        # Sorting during cell insertion can move a name before its values arrive.
        selection_blocker = QSignalBlocker(tbl.selectionModel())
        sorting = tbl.isSortingEnabled()
        tbl.setSortingEnabled(False)
        tbl.clearSelection()
        self._filtered_stations = stations
        tbl.setColumnCount(len(_COL_HEADERS))
        tbl.setHorizontalHeaderLabels(_COL_HEADERS)
        tbl.setRowCount(len(stations))
        tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, len(_COL_HEADERS)):
            tbl.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
            if col >= 2:
                tbl.horizontalHeaderItem(col).setToolTip(
                    "Başlığa tıklayarak kayıtlı değerin tam duyarlığıyla sayısal sıralayın. "
                    "—: tanımlı değil; artan sıralamada sonda, azalan sıralamada başta. "
                    "Hücrenin üzerine gelerek yuvarlanmamış değeri ve birimini görün."
                )

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
            name_item.setData(Qt.UserRole, row)
            if is_custom:
                name_item.setForeground(QColor(C.YELLOW))
                name_item.setToolTip("Özel istasyon (ayarlarda saklanır)")
            tbl.setItem(row, 0, name_item)
            tbl.setItem(row, 1, cell(net, net_color))
            tbl.setItem(row, 2, _StationNumericItem(s["lat_deg"], 2, "°"))
            tbl.setItem(row, 3, _StationNumericItem(s["lon_deg"], 2, "°"))

            sr = s["sigma_range_m"]
            sr_color = QColor(C.YELLOW) if sr and sr > 10 else QColor(C.GREEN)
            tbl.setItem(row, 4, _StationNumericItem(sr, 1, "m", sr_color))

            sa = s["sigma_angle_rad"]
            sa_mrad = sa * 1e3 if sa is not None else None
            sa_color = QColor(C.YELLOW) if sa_mrad and sa_mrad > 0.05 else QColor(C.GREEN)
            tbl.setItem(row, 5, _StationNumericItem(sa_mrad, 3, "mrad", sa_color))

            rr = s["sigma_rr_mps"]
            rr_mm = rr * 1e3 if rr is not None else None
            tbl.setItem(row, 6, _StationNumericItem(rr_mm, 3, "mm/s"))

        tbl.setSortingEnabled(sorting)
        del selection_blocker
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

    def _focus_search(self) -> None:
        self._search.setFocus(Qt.ShortcutFocusReason)
        self._search.selectAll()

    # ------------------------------------------------------------------
    def _on_selection(self) -> None:
        rows = self._selected_rows()
        count = len(self._filtered_stations)
        self._btn_export.setEnabled(count > 0)
        if not self._all_stations:
            message = "İstasyon kataloğu boş. İstasyon ekleyebilir veya katalog kaynağını kontrol edebilirsiniz."
        elif not count:
            message = "Eşleşen istasyon yok. Aramayı temizleyin veya ad / ağ koduyla (DSN, ESA…) arayın."
        else:
            message = f"{count}/{len(self._all_stations)} istasyon · {len(rows)} seçili. CSV dışa aktarımı tablodaki satırları içerir; harita tüm kataloğu gösterir."
        self._filter_status.setText(message)

        single_selected = len(rows) == 1
        self._btn_edit.setEnabled(single_selected)
        if single_selected:
            station = self._station_at_row(rows[0])
            self._btn_delete.setEnabled(station.get("_custom", False))
        else:
            self._btn_delete.setEnabled(False)

    def _selected_rows(self) -> list[int]:
        return sorted({idx.row() for idx in self.stationsTable.selectedIndexes()})

    def _station_at_row(self, row: int) -> dict:
        """Resolve a visual row after sorting through its stable source index."""
        index = self.stationsTable.item(row, 0).data(Qt.UserRole)
        return self._filtered_stations[index]

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
        station = self._station_at_row(rows[0])
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
        station = self._station_at_row(rows[0])
        if not station.get("_custom"):
            QMessageBox.information(self, "Silinemez",
                "Yalnız eklediğiniz özel istasyonlar silinebilir.\n"
                "Yerleşik istasyonlar katalog dosyasından gelir.")
            return
        reply = QMessageBox.question(
            self, "İstasyon silinsin mi?",
            f"'{station['name']}' özel istasyonu silinsin mi?",
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
        stations = self._filtered_stations
        if not stations:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "İstasyonları CSV olarak kaydet",
            str(Path.home() / "stations.csv"),
            "CSV dosyaları (*.csv)",
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
                self, "Kaydedildi",
                f"{len(stations)} istasyon kaydedildi:\n{path}",
            )
        except Exception as exc:
            QMessageBox.warning(self, "Kaydedilemedi", str(exc))
