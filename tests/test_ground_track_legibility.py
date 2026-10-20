"""Readability oracles for both plot models at actual narrow canvas sizes."""
import numpy as np
import pytest
from matplotlib.colors import to_rgb
from matplotlib.transforms import Bbox
from PyQt5.QtWidgets import QApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from controllers import ground_track_controller as gt
from styles import appearance, theme
from tests.test_ground_track_models import samples


def contrast(first, second):
    def luminance(color):
        values = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
                  for v in to_rgb(color)]
        return sum(v * weight for v, weight in zip(values, (0.2126, 0.7152, 0.0722)))
    bright, dark = sorted((luminance(first), luminance(second)), reverse=True)
    return (bright + 0.05) / (dark + 0.05)


@pytest.fixture
def palette_state(monkeypatch):
    """Restore actual tokens, including the pre-startup default palette."""
    for name, value in vars(theme.C).copy().items():
        if not name.startswith("_"):
            monkeypatch.setattr(theme.C, name, value)
    if not hasattr(theme.C, "ON_ACCENT"):
        monkeypatch.setattr(theme.C, "ON_ACCENT", None, raising=False)
    for mapping in (theme.ACCENT, theme.NET_COLORS, theme.MPL_DARK):
        for key, value in mapping.copy().items():
            monkeypatch.setitem(mapping, key, value)
    monkeypatch.setattr(theme, "BAR_COLORS", list(theme.BAR_COLORS))
    monkeypatch.setattr(appearance, "CURRENT_THEME", appearance.CURRENT_THEME)


@pytest.mark.parametrize("palette", ["modern", "orchid"])
@pytest.mark.parametrize("width", [548, 698])
@pytest.mark.parametrize("sampled", [False, True], ids=["analytical", "sampled"])
def test_ground_track_text_fits_canvas_and_keeps_data(palette, width, sampled, palette_state):
    app = QApplication.instance() or QApplication([])
    appearance.apply_palette(palette)
    canvas = None
    try:
        data = samples([True, False, True, False] if sampled else None)
        original = {k: v.tobytes() for k, v in data.items() if isinstance(v, np.ndarray)}
        figure = gt._build_static_figure(data)
        canvas = FigureCanvasQTAgg(figure)
        canvas.resize(width, 446)
        canvas.show()
        app.processEvents()
        canvas.draw()
        ax = figure.axes[0]
        reference = next(t for t in ax.texts if "Sub-Earth" in t.get_text())
        # A deliberate readability target, checked on both base and highlighted fill.
        background = np.asarray(ax.get_facecolor()[:3])
        near = ax.patches[0]
        highlighted = (np.asarray(near.get_facecolor()[:3]) * near.get_alpha()
                       + background * (1 - near.get_alpha()))
        assert min(contrast(reference.get_color(), fill)
                   for fill in (background, highlighted)) >= 4.5
        footer = next(t for t in figure.texts if "Moon rotates" in t.get_text())
        assert contrast(footer.get_color(), figure.get_facecolor()) >= 4.5
        assert " ".join(footer.get_text().split()) == (
            "Moon rotates ~27× slower than Earth — passes drift only ~1°/orbit, "
            "so selenographic tracks stay near-vertical.")
        renderer = canvas.get_renderer()
        bounds = Bbox.from_bounds(0, 0, canvas.width(), canvas.height())
        for text in (ax.title, ax.xaxis.label, ax.yaxis.label, footer):
            box = text.get_window_extent(renderer)
            assert bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1)
        footer_box = footer.get_window_extent(renderer)
        assert not footer_box.overlaps(ax.get_legend().get_window_extent(renderer))
        assert not footer_box.overlaps(ax.xaxis.label.get_window_extent(renderer))
        title = ax.get_title()
        assert "1 day ·" in title and "1 days" not in title
        assert ("Visible samples: 50%" if sampled else "Visibility not computed") in title
        assert "(°)" in ax.get_xlabel() and "(°)" in ax.get_ylabel()
        assert original == {k: data[k].tobytes() for k in original}
        assert ax.get_xlim() == (-180, 180) and ax.get_ylim() == (-90, 90)
        # Resizing the same figure must retain its data and the readable title.
        before = [line.get_xydata().copy() for line in ax.lines]
        canvas.resize(1100, 650)
        app.processEvents()
        canvas.draw()
        assert all(np.array_equal(a, b.get_xydata(), equal_nan=True)
                   for a, b in zip(before, ax.lines))
        assert ax.get_title() == title
    finally:
        if canvas is not None:
            canvas.close()
