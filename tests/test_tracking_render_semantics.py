"""Scientific plot semantics: gaps, discrete solutions and narrow extrema."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from types import SimpleNamespace
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QColor
from widgets.tracking_plots import TrackingChart, TEXT


def render(rows, batch=False):
    chart = TrackingChart("position_error_m", "Konum", "m")
    chart.resize(600, 220)
    chart.linear = True
    result = SimpleNamespace(updates=rows, arcs=[], metadata={"record_kind": "arc_solution" if batch else "update"})
    chart.set_data(result, tuple(range(len(rows))), (0, 100), None)
    image = chart.grab().toImage()
    return chart, image


def ink_near(chart, image, t, value, radius=2):
    *_, x, y = chart._plot_mapping
    cx, cy = round(x(t)), round(y(value))
    target = QColor(TEXT)
    return any(sum(abs(image.pixelColor(px, py).getRgb()[i] - target.getRgb()[i]) for i in range(3)) < 120
        for px in range(cx-radius, cx+radius+1)
        for py in range(cy-radius, cy+radius+1))


def test_arc_gap_and_batch_estimates_are_not_connected():
    app = QApplication.instance() or QApplication([])
    rows = [{"arc_id": 1, "t_s": 20, "position_error_m": 20},
            {"arc_id": 1, "t_s": 80, "position_error_m": 80}]
    continuous, image = render(rows)
    assert ink_near(continuous, image, 50, 50)
    discrete, image = render(rows, batch=True)
    assert not ink_near(discrete, image, 50, 50)
    assert ink_near(discrete, image, 20, 20, radius=4)
    separated, image = render([rows[0], dict(rows[1], arc_id=2)])
    assert not ink_near(separated, image, 50, 50)
    for chart in (continuous, discrete, separated):
        chart.close()


def test_dense_pixel_bucket_preserves_narrow_spike():
    app = QApplication.instance() or QApplication([])
    rows = [{"arc_id": 1, "t_s": 40 + i/10000, "position_error_m": 90 if i == 500 else 20} for i in range(1000)]
    chart, image = render(rows)
    assert ink_near(chart, image, rows[500]["t_s"], 90, radius=3)
    assert len(chart.result.updates) == 1000
    assert chart.result.updates[500] is rows[500]
    chart.close()
