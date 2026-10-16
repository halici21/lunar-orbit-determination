"""Renderer-aware catalogue label placement without changing geographic markers."""
from matplotlib.artist import Artist
from matplotlib.text import Text
from matplotlib.transforms import Bbox


_STEPS = (0, 10, -10, 20, -20, 32, -32, 46, -46, 64, -64)
_OFFSETS = sorted(((dx, dy) for dx in _STEPS for dy in _STEPS if dx or dy),
                  key=lambda offset: offset[0]**2 + offset[1]**2)


class CatalogueMapLabels(Artist):
    """Lay out names before annotation drawing, including the first frame/resize."""

    def __init__(self, axes, labels):
        super().__init__()
        self.labels = labels
        self.set_zorder(6.5)  # Above markers; before annotation text at zorder 7.
        self.set_in_layout(False)
        axes.add_artist(self)

    def draw(self, renderer):
        bounds = self.axes.bbox.padded(-renderer.points_to_pixels(2))
        radius = renderer.points_to_pixels(8)
        anchors = [self.axes.transData.transform(label.xy) for label in self.labels]
        markers = [Bbox.from_bounds(x-radius, y-radius, 2*radius, 2*radius) for x, y in anchors]
        occupied = []
        # Edge stations have fewer valid placements; reserve their names first.
        entries = sorted(zip(self.labels, anchors), key=lambda item: min(
            item[1][0]-bounds.x0, bounds.x1-item[1][0],
            item[1][1]-bounds.y0, bounds.y1-item[1][1]))
        for label, (x, y) in entries:
            label.set_visible(True)
            label.update_positions(renderer)
            # Measure once; Annotation's full extent also includes its leader.
            extent = Text.get_window_extent(label, renderer)
            width, height = extent.width, extent.height
            placed = False
            for dx, dy in _OFFSETS:
                px, py = x+renderer.points_to_pixels(dx), y+renderer.points_to_pixels(dy)
                left = px if dx > 0 else px-width if dx < 0 else px-width/2
                bottom = py if dy > 0 else py-height if dy < 0 else py-height/2
                box = Bbox.from_bounds(left, bottom, width, height).padded(renderer.points_to_pixels(1.5))
                if (bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1)
                        and not any(box.overlaps(other) for other in occupied+markers)):
                    label.set_position((dx, dy))
                    label.set_horizontalalignment("left" if dx > 0 else "right" if dx < 0 else "center")
                    label.set_verticalalignment("bottom" if dy > 0 else "top" if dy < 0 else "center")
                    occupied.append(box)
                    placed = True
                    break
            # Dense/oversized custom names remain in the table; never move a marker.
            label.set_visible(placed)
        self.stale = False
