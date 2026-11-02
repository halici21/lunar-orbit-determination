"""Cached native Earth map and clickable mission Gantt, sharing recorded time."""
from __future__ import annotations
import json
import math
from pathlib import Path

from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap, QFont
from PyQt5.QtWidgets import QWidget, QToolTip
from styles.theme import C, NET_COLORS


def network(name):
    return next((word for word in name.split() if word in NET_COLORS), "OTHER")


class StationWorldMap(QWidget):
    stationSelected = pyqtSignal(str)

    def __init__(self, stations=(), parent=None):
        super().__init__(parent)
        self.stations, self.active, self.selected = tuple(stations), set(), None
        self._base_key, self._base = None, None
        self._markers = []
        self._labels = []
        self.setMinimumSize(320, 230)
        self.setMouseTracking(True)
        self.setAccessibleName("Dünya haritasında yer istasyonları; konumu seçmek için tıklayın")
        path = Path(__file__).resolve().parents[1] / "data/ne_110m_land.geojson"
        self.rings = []
        if path.is_file():
            for feature in json.loads(path.read_bytes())["features"]:
                geometry = feature["geometry"]
                polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
                self.rings.extend(polygon[0] for polygon in polygons if polygon)

    def set_stations(self, stations):
        self.stations = tuple(stations)
        self.active.clear()
        self.selected = None
        self.update()

    def _map_rect(self):
        """Fit the 360° by 180° equirectangular map without stretching it."""
        width = min(self.width()-40, 2*(self.height()-52))
        height = width/2
        return QRectF((self.width()-width)/2, 20+(self.height()-52-height)/2, width, height)

    def _point(self, lon, lat):
        rect = self._map_rect()
        return QPointF(rect.left()+(lon+180)/360*rect.width(), rect.top()+(90-lat)/180*rect.height())

    def _place_labels(self, metrics):
        bounds = self._map_rect().adjusted(3, 3, -3, -3)
        occupied = []
        markers = [QRectF(p.x()-10, p.y()-10, 20, 20) for p, _ in self._markers]
        placements = []
        # Give the selected station first choice; nearby optional names may hide.
        for point, station in sorted(self._markers, key=lambda item: item[1]["name"] != self.selected):
            selected = station["name"] == self.selected
            if not selected and station["name"] not in {"Goldstone DSN", "Madrid DSN", "Canberra DSN", "ITU Ayazaga"}:
                continue
            text = station["name"] if selected else station["name"].split()[0]
            text = metrics.elidedText(text, Qt.ElideRight, int(bounds.width()-4))
            width, height = metrics.horizontalAdvance(text)+4, metrics.height()+4
            candidates = []
            for dx, dy in ((12, -height-12), (-width-12, -height-12), (12, 12), (-width-12, 12)):
                rect = QRectF(point.x()+dx, point.y()+dy, width, height)
                rect.moveLeft(max(bounds.left(), min(rect.left(), bounds.right()-width)))
                rect.moveTop(max(bounds.top(), min(rect.top(), bounds.bottom()-height)))
                collisions = sum(rect.intersects(other) for other in occupied+markers)
                candidates.append((collisions, rect))
            collisions, rect = min(candidates, key=lambda candidate: candidate[0])
            if collisions and not selected:
                continue
            occupied.append(rect.adjusted(-3, -3, 3, 3))
            placements.append((point, station, text, rect))
        return placements

    def _background(self):
        key = (self.width(), self.height(), self.devicePixelRatioF())
        if key == self._base_key:
            return
        ratio = self.devicePixelRatioF()
        self._base = QPixmap(round(self.width()*ratio), round(self.height()*ratio))
        self._base.setDevicePixelRatio(ratio)
        self._base.fill(QColor(C.BG_DEEP))
        p = QPainter(self._base)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(C.BORDER_MID), .7))
        p.setBrush(QColor(C.LAND))
        for ring in self.rings:
            path = QPainterPath()
            for index, (lon, lat, *_) in enumerate(ring):
                point = self._point(lon, lat)
                path.moveTo(point) if index == 0 else path.lineTo(point)
            path.closeSubpath()
            p.drawPath(path)
        p.setPen(QPen(QColor(C.BORDER_FAINT), .5))
        for lon in (-120, -60, 0, 60, 120):
            p.drawLine(self._point(lon, -90), self._point(lon, 90))
        for lat in (-60, -30, 0, 30, 60):
            p.drawLine(self._point(-180, lat), self._point(180, lat))
        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QColor(C.TEXT_MUTED))
        p.drawText(20, self.height()-7, "Boylam / enlem · Natural Earth 110m")
        p.end()
        self._base_key = key

    def paintEvent(self, event):
        self._background()
        p = QPainter(self)
        p.drawPixmap(0, 0, self._base)
        p.setRenderHint(QPainter.Antialiasing)
        self._markers = []
        for station in self.stations:
            point = self._point(station["lon_deg"], station["lat_deg"])
            self._markers.append((point, station))
            color = NET_COLORS.get(network(station["name"]), C.TEXT_PRIMARY)
            active = station["name"] in self.active
            selected = station["name"] == self.selected
            p.setPen(QPen(QColor(color), 1.4))
            p.setBrush(QColor(color if active else C.BG_DEEP))
            p.drawEllipse(point, 5, 5)
            if active or selected:
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(point, 9, 9)
        p.setFont(QFont("Segoe UI", 8))
        self._labels = self._place_labels(p.fontMetrics())
        for point, station, text, rect in self._labels:
            anchor = QPointF(max(rect.left(), min(point.x(), rect.right())),
                             max(rect.top(), min(point.y(), rect.bottom())))
            p.setPen(QPen(QColor(C.BORDER_MID), .8))
            length = math.hypot(anchor.x()-point.x(), anchor.y()-point.y())
            if length > 10:
                p.drawLine(point+(anchor-point)*(10/length), anchor)
            p.fillRect(rect, QColor(C.BG_DEEP))
            p.setPen(QColor(C.TEXT_PRIMARY))
            p.drawText(rect.adjusted(2, 2, -2, -2), Qt.AlignLeft | Qt.AlignVCenter, text)

    def _hit(self, point):
        for _, station, _, rect in self._labels:
            if rect.contains(QPointF(point)):
                return station
        nearby = [(abs(p.x()-point.x())**2+abs(p.y()-point.y())**2, station) for p, station in self._markers]
        hit = min(nearby, key=lambda pair: pair[0], default=(9999, None))
        if hit[0] < 144:
            # Integer pointer coordinates cannot resolve subpixel station pairs
            # such as Madrid/Cebreros. Retain an explicit existing selection.
            selected = next((pair for pair in nearby if pair[1]["name"] == self.selected), None)
            if selected and selected[0] < 144 and selected[0] <= hit[0]+1:
                return selected[1]
        return hit[1] if hit[0] < 144 else None

    def mouseMoveEvent(self, event):
        station = self._hit(event.pos())
        if station:
            self.setCursor(Qt.PointingHandCursor)
            QToolTip.showText(event.globalPos(), f"{station['name']}\n{station['lat_deg']:.3f}° enlem · {station['lon_deg']:.3f}° boylam\nİstasyonu seçmek için tıklayın.", self)
        else:
            self.unsetCursor()
            QToolTip.hideText()

    def mousePressEvent(self, event):
        station = self._hit(event.pos())
        if station:
            self.selected = station["name"]
            self.stationSelected.emit(self.selected)
            self.update()


class MissionGantt(QWidget):
    timeSelected = pyqtSignal(float)
    stationSelected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.geometry, self.result = None, None
        self.cursor = None
        self.selected_station = None
        self.bounds = (0., 1.)
        self._base_key, self._base = None, None
        self._by_station = {}
        self.setMinimumHeight(180)
        self.setMouseTracking(True)
        self.setAccessibleName("İstasyon ölçüm aralıkları Gantt; satır adı istasyonu, çizelge zamanı seçer")
        self.setAccessibleDescription(self.empty_message())

    def empty_message(self):
        if self.result is None:
            return "Ölçüm çizelgesini görmek için bir koşu açın.\nArşivden bir koşu açın veya Senaryo sayfasından yeni bir koşu başlatın."
        if not self.geometry:
            return "Bu koşuda istasyon ölçüm zamanları kaydedilmemiş.\nGörünürlük hesabını açabilir veya yeni bir senaryo çalıştırabilirsiniz.\nEski yay başlangıç/sonları temas süresi olarak gösterilmez."
        return ""

    def set_result(self, result):
        self.result = result
        self.cursor, self.selected_station = None, None
        self._base_key = None
        self._by_station = {}
        self.geometry = result.metadata.get("mission_geometry") if result else None
        self.setAccessibleDescription(self.empty_message())
        for window in (self.geometry or {}).get("windows", []):
            self._by_station.setdefault(window["station_id"], []).append(window)
        states = result.states if result else ()
        if states:
            self.bounds = (states[0]["t_s"], max(states[0]["t_s"]+1, states[-1]["t_s"]))
        else:
            rows = result.updates if result else ()
            self.bounds = (0., max((r.get("end_s", r["t_s"]) for r in rows), default=1.) or 1.)
        self.setMinimumHeight(max(180, 54+27*len((self.geometry or {}).get("stations", []))))
        self.update()

    def _draw_base(self, p):
        p.fillRect(self.rect(), QColor(C.BG_DEEP))
        p.setRenderHint(QPainter.Antialiasing)
        p.setFont(QFont("Segoe UI", 9))
        p.setPen(QColor(C.TEXT_MUTED))
        if not self.geometry:
            p.drawText(QRectF(12, 18, self.width()-24, self.height()-36), Qt.AlignCenter | Qt.TextWordWrap,
                       self.empty_message())
            return
        left, right = 151., self.width()-24.
        lo, hi = self.bounds
        def x(t): return left+(t-lo)/(hi-lo)*(right-left)
        for tick in range(5):
            t = lo+(hi-lo)*tick/4
            p.setPen(QPen(QColor(C.BORDER_FAINT), .8))
            p.drawLine(QPointF(x(t), 12), QPointF(x(t), self.height()-30))
            p.setPen(QColor(C.TEXT_MUTED))
            p.drawText(QRectF(x(t)-30, self.height()-25, 60, 20), Qt.AlignCenter, f"{t/3600:.2f} sa")
        stations = self.geometry["stations"]
        for index, station in enumerate(stations):
            y = 22+index*27
            p.setPen(QColor(C.TEXT_PRIMARY))
            label = p.fontMetrics().elidedText(station["name"], Qt.ElideRight, int(left-15))
            p.drawText(QRectF(8, y-9, left-15, 20), Qt.AlignRight | Qt.AlignVCenter, label)
            color = NET_COLORS.get(network(station["name"]), C.TEXT_PRIMARY)
            if self.selected_station and self.selected_station != station["name"]:
                color = C.BORDER_MID
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(color))
            for window in self._by_station.get(station["station_id"], ()):
                p.drawRoundedRect(QRectF(x(window["start_s"]), y-5, max(2., x(window["end_s"])-x(window["start_s"])), 10), 2, 2)

    def paintEvent(self, event):
        key = (self.width(), self.height(), self.devicePixelRatioF(), self.selected_station)
        if key != self._base_key:
            ratio = self.devicePixelRatioF()
            self._base = QPixmap(round(self.width()*ratio), round(self.height()*ratio))
            self._base.setDevicePixelRatio(ratio)
            self._base.fill(QColor(C.BG_DEEP))
            base_painter = QPainter(self._base)
            self._draw_base(base_painter)
            base_painter.end()
            self._base_key = key
        p = QPainter(self)
        p.drawPixmap(0, 0, self._base)
        if self.cursor is not None and self.geometry:
            lo, hi = self.bounds
            x = 151+(self.cursor-lo)/(hi-lo)*(self.width()-175)
            p.setPen(QPen(QColor(C.NAV_ACTIVE), 1.3))
            p.drawLine(QPointF(x, 10), QPointF(x, self.height()-30))

    def mouseMoveEvent(self, event):
        station = self._label_station(event.pos())
        if station:
            self.setCursor(Qt.PointingHandCursor)
            QToolTip.showText(event.globalPos(), f"{station['name']}\nHaritada ve çizelgede seçmek için tıklayın.", self)
            return
        self.unsetCursor()
        if self.geometry and 151 <= event.x() <= self.width()-24:
            row = round((event.y()-22)/27)
            stations = self.geometry["stations"]
            if 0 <= row < len(stations) and abs(event.y()-(22+row*27)) <= 9:
                station = stations[row]
                lo, hi = self.bounds
                t_s = lo+(event.x()-151)/(self.width()-175)*(hi-lo)
                window = next((w for w in self._by_station.get(station["station_id"], ())
                               if w["start_s"] <= t_s <= w["end_s"]), None)
                if window:
                    QToolTip.showText(event.globalPos(), f"{station['name']} · yay {window['arc_id']}\n"
                                      f"T+ {window['start_s']/3600:.3f}–{window['end_s']/3600:.3f} sa · {window['samples']} ölçüm epoku\n"
                                      "Zamanı seçmek için tıklayın.", self)
                    return
        QToolTip.hideText()

    def _label_station(self, point):
        if not self.geometry or not 8 <= point.x() < 151:
            return None
        row = round((point.y()-22)/27)
        stations = self.geometry["stations"]
        if 0 <= row < len(stations) and abs(point.y()-(22+row*27)) <= 10:
            return stations[row]
        return None

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        station = self._label_station(event.pos())
        if station:
            self.stationSelected.emit(station["name"])
        elif self.geometry and 151 <= event.x() <= self.width()-24:
            lo, hi = self.bounds
            self.timeSelected.emit(lo+(event.x()-151)/(self.width()-175)*(hi-lo))
