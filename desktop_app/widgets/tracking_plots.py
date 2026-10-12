"""Lightweight native Qt plots for stored results, without matplotlib imports."""
from __future__ import annotations

import bisect
import math

from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPen, QFont, QFontMetrics, QPixmap
from PyQt5.QtWidgets import QWidget


from styles.theme import C

BG, TEXT, MUTED, ACCENT = C.BG_PLOT, C.TEXT_PRIMARY, C.TEXT_MUTED, C.NAV_ACTIVE


def elapsed(t):
    if t is None:
        return "Seçim yok"
    sign = "−" if t < 0 else ""
    s = abs(round(t))
    return f"T{sign}+{s // 3600:02}:{s % 3600 // 60:02}:{s % 60:02}"


class TrackingChart(QWidget):
    selected = pyqtSignal(int)  # index into immutable source rows
    zoomRequested = pyqtSignal(float, float)
    stepRequested = pyqtSignal(int)

    def __init__(self, key, title, unit, bls_key=None, parent=None):
        super().__init__(parent)
        self.key, self.title, self.unit, self.bls_key = key, title, unit, bls_key
        self.result = None
        self.indices = ()
        self.time_range = (0, 1)
        self.selected_index = None
        self.linear = False
        self._times = []
        self._base_key = None
        self._base_image = None
        self._base_inputs = None
        self._plot_mapping = None
        self.setMinimumSize(220, 165)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName(f"{title}, {unit}. Sol/sağ ok: kayıt seç. Tekerlek: ortak zaman aralığı.")
        self.setToolTip("Tıklayın: kaydı seç · Tekerlek: ortak zamanı yakınlaştır · Ok tuşları: önceki/sonraki kayıt")

    def set_data(self, result, indices, time_range, selected):
        if result is not self.result or indices is not self.indices:
            self._times = [result.updates[i]["t_s"] for i in indices] if result else []
        self.result, self.indices, self.time_range, self.selected_index = result, indices, time_range, selected
        self.update()

    def _time_at(self, x):
        lo, hi = self.time_range
        return lo + max(0, min(1, (x - 58) / max(1, self.width() - 76))) * (hi - lo)

    def mousePressEvent(self, event):
        if not self._times:
            return
        t = self._time_at(event.x())
        i = bisect.bisect_left(self._times, t)
        candidates = [n for n in (i - 1, i) if 0 <= n < len(self._times)]
        best = min(candidates, key=lambda n: abs(self._times[n] - t))
        self.selected.emit(self.indices[best])
        self.setFocus()

    def wheelEvent(self, event):
        if not self.result:
            return
        t = self._time_at(event.position().x() if hasattr(event, "position") else event.pos().x())
        factor = math.exp(-event.angleDelta().y() * .002)
        lo, hi = self.time_range
        self.zoomRequested.emit(t - (t - lo) * factor, t + (hi - t) * factor)
        event.accept()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Left, Qt.Key_Right):
            self.stepRequested.emit(-1 if event.key() == Qt.Key_Left else 1)
        elif event.key() in (Qt.Key_Plus, Qt.Key_Equal, Qt.Key_Minus):
            lo, hi = self.time_range
            mid, factor = (lo + hi) / 2, 1.4 if event.key() == Qt.Key_Minus else .7
            self.zoomRequested.emit(mid - (hi - lo) * factor / 2, mid + (hi - lo) * factor / 2)
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event):
        # Immutable result/selection tuples let pointer-only selection reuse the
        # background. Invalidate on data, filters, scale, size or display DPI.
        key = (id(self.result), id(self.indices), self.time_range, self.linear,
            self.width(), self.height(), self.devicePixelRatioF(), self.title, self.key, self.bls_key)
        if key != self._base_key:
            ratio = self.devicePixelRatioF()
            self._base_image = QPixmap(round(self.width() * ratio), round(self.height() * ratio))
            self._base_image.setDevicePixelRatio(ratio)
            self._plot_mapping = None
            base = QPainter(self._base_image)
            self._draw_base(base)
            base.end()
            self._base_key = key
            self._base_inputs = (self.result, self.indices)
        p = QPainter(self)
        p.drawPixmap(0, 0, self._base_image)
        p.setRenderHint(QPainter.Antialiasing)
        if self._plot_mapping and self.selected_index is not None:
            lo, hi, L, R, T, B, x, y = self._plot_mapping
            row = self.result.updates[self.selected_index]
            if row[self.key] is not None and lo <= row["t_s"] <= hi:
                p.save()
                p.setClipRect(QRectF(L - 4, T - 4, R - L + 8, B - T + 8))
                p.setPen(QPen(QColor(ACCENT), 1, Qt.DashLine))
                p.drawLine(QPointF(x(row["t_s"]), T), QPointF(x(row["t_s"]), B))
                p.setBrush(QColor(ACCENT))
                p.drawEllipse(QPointF(x(row["t_s"]), y(row[self.key])), 3, 3)
                p.restore()
        if self.hasFocus():
            p.setPen(QPen(QColor(ACCENT), 1))
            p.drawRect(self.rect().adjusted(1, 1, -2, -2))

    def _draw_base(self, p):
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(BG))
        p.setFont(QFont("Segoe UI", 9))
        p.setPen(QColor(TEXT))
        mode = "log" if not self.linear and self.key != "normalized_nis" else "doğrusal"
        p.drawText(10, 19, f"{self.title} / {self.unit}  ·  {mode}")
        p.setPen(QColor(MUTED))
        batch = self.result and self.result.metadata.get("record_kind") == "arc_solution"
        legend = f"○ {self.result.metadata.get('estimator_type', 'Çözücü').upper()} yay çözümü" if batch else "— UKF" + ("    ○ BLS yay sonu" if self.bls_key else "")
        p.drawText(58, 39, legend)
        if not self.result or not self.indices:
            p.drawText(QRectF(self.rect()), Qt.AlignCenter, "Bu görünümde kayıt yok")
            return
        rows = [(i, self.result.updates[i]) for i in self.indices if self.result.updates[i][self.key] is not None]
        lo, hi = self.time_range
        visible_arcs = {r["arc_id"] for _, r in rows}
        arcs = [a for a in self.result.arcs if self.bls_key and a["arc_id"] in visible_arcs and lo <= a["end_s"] <= hi]
        values = [r[self.key] for _, r in rows] + [a[self.bls_key] for a in arcs]
        if not values:
            p.drawText(QRectF(self.rect()), Qt.AlignCenter, "Bu tanılama alanı kaynakta yok")
            return
        log = not self.linear and self.key != "normalized_nis" and all(v > 0 for v in values)
        # Zero is a valid result, so switch to linear instead of inventing a floor.
        if mode == "log" and not log:
            p.fillRect(QRectF(0, 0, self.width(), 25), QColor(BG))
            p.setPen(QColor(TEXT))
            p.drawText(10, 19, f"{self.title} / {self.unit}  ·  doğrusal (sıfır kayıt)")
        if log:
            low = math.floor(math.log10(min(v for v in values if v > 0)))
            high = max(low + 1, math.ceil(math.log10(max(values))))
            transform = lambda v: math.log10(max(v, 10 ** low))
            ticks = [10 ** low, 10 ** ((low + high) / 2), 10 ** high]
        else:
            low, high = 0, max(1e-12, max(values) * 1.1)
            transform = lambda v: v
            ticks = [0, high / 2, high]
        L, R, T, B = 58, self.width() - 18, 52, self.height() - 30
        x = lambda t: L + (t - lo) / max(1e-9, hi - lo) * (R - L)
        y = lambda v: B - (transform(v) - low) / (high - low) * (B - T)
        for tick in ticks:
            p.setPen(QPen(QColor(C.BORDER_MAIN), 1))
            p.drawLine(QPointF(L, y(tick)), QPointF(R, y(tick)))
            p.setPen(QColor(MUTED))
            p.drawText(QRectF(0, y(tick) - 9, L - 7, 18), Qt.AlignRight | Qt.AlignVCenter, f"{tick:.2g}")
        p.setPen(QColor(MUTED))
        p.drawText(L, self.height() - 8, f"{lo / 3600:.2f} sa")
        p.drawText(QRectF(R - 85, self.height() - 25, 85, 22), Qt.AlignRight | Qt.AlignVCenter, f"{hi / 3600:.2f} sa")
        p.save()
        p.setClipRect(QRectF(L - 4, T - 4, R - L + 8, B - T + 8))
        # Preserve each pixel bucket's extrema and arc boundaries. Dense data
        # cannot erase a narrow spike simply because the screen is smaller.
        plotted = []
        bucket = []
        previous = None
        for i, row in rows:
            key = (row["arc_id"], int(x(row["t_s"])))
            if previous is not None and key != previous:
                plotted.extend(self._bucket_points(bucket))
                bucket = []
            bucket.append((i, row))
            previous = key
        plotted.extend(self._bucket_points(bucket))
        p.setPen(QPen(QColor(TEXT), 1.3))
        if not batch:
            # Independent arcs are independent painter paths. A single compound
            # path makes Qt's antialiased stroker expensive on dense zigzags.
            path, last_arc = QPainterPath(), None
            for _, row in plotted:
                point = QPointF(x(row["t_s"]), y(row[self.key]))
                if row["arc_id"] != last_arc:
                    if not path.isEmpty():
                        p.drawPath(path)
                    path = QPainterPath()
                    path.moveTo(point)
                else:
                    path.lineTo(point)
                last_arc = row["arc_id"]
            p.drawPath(path)
        # Batch solutions are discrete estimates, never a continuous series.
        if batch or len(rows) < 400:
            p.setBrush(QColor(BG if batch else TEXT))
            radius = 3.3 if batch else 1.3
            for _, row in (plotted if batch else rows):
                p.drawEllipse(QPointF(x(row["t_s"]), y(row[self.key])), radius, radius)
        p.setPen(QPen(QColor(ACCENT), 1.2))
        p.setBrush(QColor(BG))
        for a in arcs:
            p.drawEllipse(QPointF(x(a["end_s"]), y(a[self.bls_key])), 3.3, 3.3)
        p.restore()
        self._plot_mapping = (lo, hi, L, R, T, B, x, y)

    def _bucket_points(self, bucket):
        if not bucket:
            return []
        keep = {bucket[0][0], bucket[-1][0], min(bucket, key=lambda pair: pair[1][self.key])[0], max(bucket, key=lambda pair: pair[1][self.key])[0]}
        return [pair for pair in bucket if pair[0] in keep]


class StoredOrbit(QWidget):
    """Display recorded Cartesian positions. Never synthesize a missing orbit."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.selected_t = None
        self.yaw, self.pitch, self.zoom = .65, .55, 1.0
        self._drag = None
        self._points = []
        self._times = []
        self._extent = 1
        self._base_key = None
        self._base_image = None
        self._project = None
        self._plot_rect = QRectF()
        self._epoch_rect = QRectF()
        self.setMinimumSize(350, 230)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName("Kaydedilmiş Ay merkezli durum vektörlerinin yörüngesi")
        self.setAccessibleDescription(self.empty_message())

    def empty_message(self):
        """Distinguish an unopened workspace from a loaded run without states."""
        if self.result is None:
            return "Yörüngeyi görmek için kaydedilmiş bir koşu açın.\nAy merkezli x/y/z durum geçmişi olan sonuçlar burada gösterilir."
        if not self.result.states:
            return "Bu sonuçta durum vektörü bulunmuyor.\nKonum/hız hata grafikleri kaynak CSV'den gösterilir.\nYörünge için Ay merkezli x/y/z durum geçmişi gerekli."
        return ""

    def set_result(self, result):
        self._base_key = None
        self.result = result
        self.setAccessibleDescription(self.empty_message())
        self._points = [(r["x_m"] / 1000, r["y_m"] / 1000, r["z_m"] / 1000) for r in result.states] if result else []
        self._times = [r["t_s"] for r in result.states] if result else []
        self._extent = max(1737.4, max((math.sqrt(sum(v * v for v in p)) for p in self._points), default=1737.4))
        if result and result.metadata.get("record_kind") == "arc_solution":
            self._extent = max(self._extent, max((math.sqrt(sum(r[k] ** 2 for k in ("x_m", "y_m", "z_m"))) / 1000 for r in result.updates), default=1737.4))
        self.update()

    def reset(self):
        self.yaw, self.pitch, self.zoom = .65, .55, 1.0
        self.update()

    def mousePressEvent(self, e):
        self._drag = (e.x(), e.y(), self.yaw, self.pitch)
        self.setFocus()

    def mouseMoveEvent(self, e):
        if self._drag and e.buttons() & Qt.LeftButton:
            x, y, yaw, pitch = self._drag
            self.yaw, self.pitch = yaw + (e.x() - x) * .008, pitch + (e.y() - y) * .008
            self.update()

    def mouseReleaseEvent(self, e):
        self._drag = None

    def wheelEvent(self, e):
        self.zoom = max(.3, min(4, self.zoom * math.exp(e.angleDelta().y() * .001)))
        self.update()
        e.accept()

    def paintEvent(self, event):
        key = (self.width(), self.height(), self.devicePixelRatioF(),
               self.yaw, self.pitch, self.zoom)
        if key != self._base_key:
            ratio = self.devicePixelRatioF()
            self._base_image = QPixmap(round(self.width()*ratio), round(self.height()*ratio))
            self._base_image.setDevicePixelRatio(ratio)
            base = QPainter(self._base_image)
            self._project = None
            self._draw_base(base)
            base.end()
            self._base_key = key
        p = QPainter(self)
        p.drawPixmap(0, 0, self._base_image)
        p.setRenderHint(QPainter.Antialiasing)
        if not self._project:
            return
        t_label = ""
        if self.selected_t is not None:
            i = bisect.bisect_left(self._times, self.selected_t)
            choices = [j for j in (i - 1, i) if 0 <= j < len(self._times)]
            j = min(choices, key=lambda n: abs(self._times[n] - self.selected_t))
            p.save()
            p.setClipRect(self._plot_rect)
            p.setBrush(QColor(ACCENT))
            p.setPen(Qt.NoPen)
            p.drawEllipse(self._project(self._points[j]), 4, 4)
            p.restore()
            t_label = f"Durum epoku: {elapsed(self._times[j])}"
            if self._times[j] != self.selected_t:
                t_label += " (en yakın kayıt)"
        p.setPen(QColor(MUTED))
        p.setFont(QFont("Segoe UI", 9))
        p.drawText(self._epoch_rect, Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, t_label)

    def _label_layout(self):
        """Reserve separate wrapped text bands and a clipped drawing viewport."""
        width = self.width() - 36
        title_font, detail_font = QFont("Segoe UI", 10), QFont("Segoe UI", 9)
        truth = self.result and self.result.metadata.get("state_role") == "truth"
        title = "Referans yörünge (truth) · ○ kayıtlı yay tahmini" if truth else "Kaydedilmiş yörünge"
        help_text = "Sürükle: döndür · Tekerlek: yakınlaştır" if self._points else ""
        frame = f"Çerçeve: {self.result.metadata['frame']} · görünüm: km" if self._points else ""
        # Reserve the longest cursor message so playback never moves the plot.
        metrics = QFontMetrics(detail_font)
        longest_time = max((elapsed(t) for t in (self._times[0], self._times[-1])),
                           key=metrics.horizontalAdvance) if self._times else ""
        epoch = f"Durum epoku: {longest_time} (en yakın kayıt)" if self._points else ""
        flags = Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap
        def height(text, font):
            return QFontMetrics(font).boundingRect(0, 0, width, 1000, flags, text).height() if text else 0
        title_rect = QRectF(18, 12, width, height(title, title_font))
        help_rect = QRectF(18, title_rect.bottom()+5, width, height(help_text, detail_font))
        epoch_rect = QRectF(18, self.height()-12-height(epoch, detail_font), width, height(epoch, detail_font))
        frame_rect = QRectF(18, epoch_rect.top()-4-height(frame, detail_font), width, height(frame, detail_font))
        plot_rect = QRectF(18, help_rect.bottom()+10, width, max(1, frame_rect.top()-help_rect.bottom()-18))
        return ((title, title_font, title_rect), (help_text, detail_font, help_rect),
                (frame, detail_font, frame_rect)), epoch_rect, plot_rect

    def _draw_base(self, p):
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(BG))
        labels, self._epoch_rect, self._plot_rect = self._label_layout()
        for index, (text, font, rect) in enumerate(labels):
            p.setFont(font)
            p.setPen(QColor(TEXT if index == 0 else MUTED))
            p.drawText(rect, Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, text)
        truth = self.result and self.result.metadata.get("state_role") == "truth"
        if not self._points:
            p.setPen(QColor(MUTED))
            p.drawText(self._plot_rect, Qt.AlignCenter | Qt.TextWordWrap,
                       self.empty_message())
            return
        p.save()
        p.setClipRect(self._plot_rect)
        scale = min(self._plot_rect.width(), self._plot_rect.height()) * .40 / self._extent * self.zoom
        cx, cy = self._plot_rect.center().x(), self._plot_rect.center().y()
        cos_yaw, sin_yaw = math.cos(self.yaw), math.sin(self.yaw)
        cos_pitch, sin_pitch = math.cos(self.pitch), math.sin(self.pitch)
        def project(point):
            x, y, z = point
            xx, yy = x * cos_yaw - y * sin_yaw, x * sin_yaw + y * cos_yaw
            return QPointF(cx + xx * scale, cy - (yy * cos_pitch - z * sin_pitch) * scale)
        self._project = project
        p.setPen(QPen(QColor(C.BORDER_MID), 1))
        p.setBrush(QColor(C.BG_PANEL))
        radius = 1737.4 * scale
        p.drawEllipse(QPointF(cx, cy), radius, radius)
        p.setPen(QColor(MUTED))
        p.drawText(round(cx - 10), round(cy), "Ay")
        # Geometric reference grid on the Moon; it is not terrain or ephemeris.
        p.setPen(QPen(QColor(C.BORDER_FAINT), .8))
        for latitude in (-60, -30, 0, 30, 60):
            phi = math.radians(latitude)
            ring = QPainterPath()
            for n in range(73):
                angle = n*math.pi/36
                point = project((1737.4*math.cos(phi)*math.cos(angle),
                                 1737.4*math.cos(phi)*math.sin(angle), 1737.4*math.sin(phi)))
                ring.moveTo(point) if n == 0 else ring.lineTo(point)
            p.drawPath(ring)
        p.setPen(QPen(QColor(C.BORDER_MID), .8, Qt.DashLine))
        for axis, endpoint in zip(("X", "Y", "Z"), ((self._extent*1.07, 0, 0), (0, self._extent*1.07, 0), (0, 0, self._extent*1.07))):
            point = project(endpoint)
            p.drawLine(QPointF(cx, cy), point)
            p.drawText(point+QPointF(4, -4), axis)
        path = QPainterPath()
        # Raw recorded samples only. The display connects samples; it performs
        # no propagation, frame transform, or ephemeris interpolation.
        stride = max(1, len(self._points) // 6000)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(TEXT), 1.3))
        for segment, i in enumerate(range(0, len(self._points), stride)):
            point = project(self._points[i])
            if i == 0:
                path.moveTo(point)
            else:
                path.lineTo(point)
            # Bound Qt's stroker workload without dropping a displayed segment.
            # Adjacent paths share the boundary vertex.
            if segment and segment % 256 == 0:
                p.drawPath(path)
                path = QPainterPath()
                path.moveTo(point)
        if len(self._points) > 1:
            path.lineTo(project(self._points[-1]))
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(TEXT), 1.3))
        p.drawPath(path)
        if truth:
            p.setPen(QPen(QColor(ACCENT), 1.2))
            p.setBrush(QColor(BG))
            for row in self.result.updates if self.result.metadata.get("record_kind") == "arc_solution" else ():
                p.drawEllipse(project(tuple(row[k] / 1000 for k in ("x_m", "y_m", "z_m"))), 3.3, 3.3)
        p.restore()
