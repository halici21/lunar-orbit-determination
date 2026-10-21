"""Recorded A/B series; cached pixel extrema, explicit gaps and independent axes."""
from __future__ import annotations

import bisect
import math
import sys

from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPen, QFont
from PyQt5.QtWidgets import QWidget

from widgets.tracking_plots import BG, TEXT, MUTED, ACCENT
from styles.theme import C


class ComparisonChart(QWidget):
    selected = pyqtSignal(int, int)  # side, original record index
    zoomRequested = pyqtSignal(float, float)

    def __init__(self, key, title, unit, parent=None):
        super().__init__(parent)
        self.key, self.title, self.unit = key, title, unit
        self.comparison = None
        self.linear, self.paired_only = False, False
        self.time_range = None
        self.selected_pair = None
        self._series, self._times, self._cache = ((), ()), ((), ()), {}
        self.setMinimumSize(320, 175)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName(f"A ve B: {title}, {unit}")
        self.setToolTip("Tıkla: gerçek kaydı seç · ortak eksende tekerlek: zaman aralığı · +/−: yakınlaştır")

    def set_comparison(self, comparison, paired_only=False):
        self.comparison, self.paired_only = comparison, paired_only
        selected = tuple({pair[side] for pair in comparison.pairs} for side in (0, 1)) if comparison else (set(), set())
        self._series = tuple(tuple((i, r) for i, r in enumerate(result.updates)
            if r.get(self.key) is not None and (not paired_only or i in selected[side]))
            for side, result in enumerate((comparison.left, comparison.right))) if comparison else ((), ())
        self._times = tuple(tuple(r["t_s"] for _, r in rows) for rows in self._series)
        self.time_range, self._cache = None, {}
        self.update()

    def shared_axis(self):
        return bool(self.comparison and self.comparison.time_compatible and (self.key != "normalized_nis"
            or (self.comparison.left.metadata.get("nis_scope") and self.comparison.left.metadata.get("nis_scope") == self.comparison.right.metadata.get("nis_scope"))))

    def domain(self, side=None):
        times = tuple(t for rows in self._times for t in rows) if side is None else self._times[side]
        if not times:
            return 0., 1.
        lo, hi = min(times), max(times)
        # Pad a single exact epoch only for display; recorded time is unchanged.
        pad = max(1., abs(lo) * .001) if hi == lo else 0.
        return lo - pad, hi + min(pad, sys.float_info.max - hi)

    def set_range(self, lo, hi):
        self.time_range = (lo, hi)
        self._cache.clear()
        self.update()

    def _panels(self):
        area = QRectF(0, 25, self.width(), self.height() - 25)
        if self.shared_axis():
            return [(area, (0, 1), self.time_range or self.domain())]
        half = area.width() / 2
        return [(QRectF(n * half, area.top(), half, area.height()), (n,), self.domain(n)) for n in (0, 1)]

    def _plot_rect(self, panel):
        return panel.adjusted(62, 23, -17, -29)

    def _location(self, x):
        for panel, sides, domain in self._panels():
            rect = self._plot_rect(panel)
            if panel.left() <= x <= panel.right():
                t = domain[0] + max(0., min(1., (x - rect.left()) / max(1., rect.width()))) * (domain[1] - domain[0])
                return sides, t, domain
        return (), 0., (0., 1.)

    def mousePressEvent(self, event):
        sides, t, _ = self._location(event.x())
        candidates = []
        for side in sides:
            n = bisect.bisect_left(self._times[side], t)
            candidates.extend((abs(self._times[side][i] - t), side, self._series[side][i][0])
                for i in (n - 1, n) if 0 <= i < len(self._times[side]))
        if candidates:
            _, side, index = min(candidates)
            self.selected.emit(side, index)
        self.setFocus()

    def wheelEvent(self, event):
        if self.shared_axis():
            _, t, (lo, hi) = self._location(event.pos().x())
            factor = math.exp(-event.angleDelta().y() * .002)
            self.zoomRequested.emit(t - (t - lo) * factor, t + (hi - t) * factor)
        event.accept()

    def keyPressEvent(self, event):
        if self.shared_axis() and event.key() in (Qt.Key_Plus, Qt.Key_Equal, Qt.Key_Minus):
            lo, hi = self.time_range or self.domain()
            mid, span = (hi + lo) / 2, (hi - lo) * (1.4 if event.key() == Qt.Key_Minus else .7)
            self.zoomRequested.emit(mid - span / 2, mid + span / 2)
        else:
            super().keyPressEvent(event)

    def _points(self, side, lo, hi, width):
        cache_key = side, lo, hi, width
        if cache_key in self._cache:
            return self._cache[cache_key]
        times = self._times[side]
        first, last = bisect.bisect_left(times, lo), bisect.bisect_right(times, hi)
        rows, plotted, bucket, prior = self._series[side][first:last], [], [], None
        def flush():
            if bucket:
                keep = {bucket[0][0], bucket[-1][0], min(bucket, key=lambda p: p[1][self.key])[0], max(bucket, key=lambda p: p[1][self.key])[0]}
                plotted.extend(p for p in bucket if p[0] in keep)
        for item in rows:
            i, r = item
            group = (r["arc_id"], r["station_id"], int((r["t_s"] - lo) / (hi - lo) * width))
            if prior is not None and group != prior:
                flush()
                bucket = []
            bucket.append(item)
            prior = group
        flush()
        self._cache[cache_key] = tuple(plotted)
        return self._cache[cache_key]

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(BG))
        p.setFont(QFont("Segoe UI", 9))
        p.setPen(QColor(TEXT))
        p.drawText(12, 19, f"{self.title} / {self.unit}" + (" · yalnız eşleşen kayıtlar" if self.paired_only else " · tüm kayıtlar"))
        if not self.comparison:
            p.drawText(QRectF(self.rect()), Qt.AlignCenter, "Karşılaştırılacak iki koşu seçin")
            return
        for panel, sides, (lo, hi) in self._panels():
            rect = self._plot_rect(panel)
            series = {side: self._points(side, lo, hi, int(max(1, rect.width()))) for side in sides}
            values = [r[self.key] for rows in series.values() for _, r in rows]
            p.setPen(QColor(MUTED))
            scope = (self.comparison.left, self.comparison.right)[sides[0]].metadata.get("nis_scope", "kapsam bilinmiyor") if self.key == "normalized_nis" else ""
            batch_legend = self.comparison.left.metadata.get("record_kind") == "arc_solution"
            legend = "A ○   B □   ortak epok" if batch_legend else "A —   B - -   ortak epok"
            p.drawText(int(panel.left() + 62), int(panel.top() + 15), (legend if len(sides) == 2 else f"{'AB'[sides[0]]} · bağımsız zaman ekseni") + (f" · {scope}" if scope else ""))
            if not values:
                p.drawText(rect, Qt.AlignCenter, "Kaynakta bu alana ait kayıt yok")
                continue
            log = not self.linear and self.key != "normalized_nis" and all(v > 0 for v in values)
            if log:
                low, high = math.log10(min(values)), math.log10(max(values))
                transform = math.log10
                ticks = sorted({min(values), max(values)})
                if high == low:
                    low, high = low - .5, high + .5
                else:
                    ticks.insert(1, 10. ** ((low + high) / 2))
            else:
                maximum = max(values)
                low, high = 0., max(1e-12, maximum * 1.05 if maximum < sys.float_info.max / 1.05 else maximum)
                transform = lambda v: v
                ticks = [0., high / 2, high]
            x = lambda t: rect.left() + (t - lo) / (hi - lo) * rect.width()
            y = lambda v: rect.bottom() - (transform(v) - low) / (high - low) * rect.height()
            for tick in ticks:
                p.setPen(QPen(QColor(C.BORDER_MAIN), 1))
                p.drawLine(QPointF(rect.left(), y(tick)), QPointF(rect.right(), y(tick)))
                p.setPen(QColor(MUTED))
                p.drawText(QRectF(panel.left(), y(tick) - 9, 54, 18), Qt.AlignRight | Qt.AlignVCenter, f"{tick:.2g}")
            factor, time_unit = (1, "s") if hi - lo < 300 else (3600, "sa")
            timestamp = lambda t: f"T{'−' if t < 0 else '+'}{abs(t) / factor:.3g} {time_unit}"
            p.drawText(int(rect.left()), int(panel.bottom() - 7), timestamp(lo))
            p.drawText(QRectF(rect.right() - 150, panel.bottom() - 24, 150, 20), Qt.AlignRight | Qt.AlignVCenter,
                f"{timestamp(hi)} · {'log' if log else 'doğrusal'}")
            p.save()
            p.setClipRect(rect.adjusted(-4, -4, 4, 4))
            for side, rows in series.items():
                color = QColor(TEXT if side == 0 else ACCENT)
                p.setPen(QPen(color, 1.4, Qt.SolidLine if side == 0 else Qt.DashLine))
                result = (self.comparison.left, self.comparison.right)[side]
                batch = result.metadata.get("record_kind") == "arc_solution"
                if not batch:
                    path, prior = QPainterPath(), None
                    for i, r in rows:
                        point = QPointF(x(r["t_s"]), y(r[self.key]))
                        group = r["arc_id"], r["station_id"]
                        if group != prior:
                            path.moveTo(point)
                        else:
                            path.lineTo(point)
                        prior = group
                    p.drawPath(path)
                p.setBrush(QColor(BG) if batch else color)
                if batch or len(rows) < 600:
                    for i, r in rows:
                        point = QPointF(x(r["t_s"]), y(r[self.key]))
                        if side == 0:
                            p.drawEllipse(point, 3.5 if batch else 1.7, 3.5 if batch else 1.7)
                        else:
                            p.drawRect(QRectF(point.x() - 2.5, point.y() - 2.5, 5, 5))
                if self.selected_pair is not None:
                    r = result.updates[self.comparison.pairs[self.selected_pair][side]]
                    if r.get(self.key) is not None and lo <= r["t_s"] <= hi:
                        p.setPen(QPen(color, 1, Qt.DotLine))
                        p.drawLine(QPointF(x(r["t_s"]), rect.top()), QPointF(x(r["t_s"]), rect.bottom()))
            p.restore()
