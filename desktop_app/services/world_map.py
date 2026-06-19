"""World map background renderer using Natural Earth 110m land polygons."""
from __future__ import annotations
import json
from pathlib import Path

_GEOJSON = Path(__file__).resolve().parents[1] / "data" / "ne_110m_land.geojson"

# Land fill and border colour (dark, muted — lets station markers stand out)
_LAND_FACE  = "#0e1e30"
_LAND_EDGE  = "#1e3a52"
_OCEAN_FACE = "#07111e"


def draw_land(ax, *, land_color: str = _LAND_FACE,
              edge_color: str = _LAND_EDGE,
              ocean_color: str = _OCEAN_FACE) -> None:
    """Draw Natural Earth 110m land polygons onto *ax* (equirectangular).

    Falls back silently if the GeoJSON file is missing.
    """
    if not _GEOJSON.exists():
        ax.set_facecolor(ocean_color)
        return

    try:
        import numpy as np
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path as MPath

        geo = json.loads(_GEOJSON.read_bytes())
        ax.set_facecolor(ocean_color)

        for feature in geo["features"]:
            geom = feature["geometry"]
            gtype = geom["type"]
            coords_list = (
                geom["coordinates"]
                if gtype == "MultiPolygon"
                else [geom["coordinates"]]
            )
            for polygon in coords_list:
                for ring in polygon:
                    pts = np.asarray(ring, dtype=float)
                    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 3:
                        continue
                    # GeoJSON rings are closed (last pt == first pt); n points, n codes
                    n = len(pts)
                    codes = (
                        [MPath.MOVETO]
                        + [MPath.LINETO] * (n - 2)
                        + [MPath.CLOSEPOLY]
                    )
                    patch = PathPatch(
                        MPath(pts[:, :2], codes),
                        facecolor=land_color,
                        edgecolor=edge_color,
                        linewidth=0.4,
                        zorder=1,
                    )
                    ax.add_patch(patch)
    except Exception:
        ax.set_facecolor(ocean_color)
