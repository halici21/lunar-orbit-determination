"""Central theme definitions — colours, fonts, matplotlib style.

Import this module anywhere you need palette values:
    from styles.theme import C, FONT, MPL_DARK, ACCENT, NET_COLORS
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
class C:
    """All palette colours in one namespace."""

    # Background layers (darkest → lightest)
    BG_DEEP    = "#080c18"   # main window / figure background
    BG_SIDEBAR = "#080e1e"   # sidebar + footer
    BG_PLOT    = "#0a1428"   # matplotlib axes background
    BG_PANEL   = "#0f1932"   # panels, legend fill
    BG_CARD    = "#101d37"   # metric card fill
    BG_HOVER   = "#0d1a31"   # nav button hover
    BG_ACTIVE  = "#162038"   # nav button active/pressed state

    # Border / grid layers
    BORDER_FAINT  = "#111e30"  # very faint grid lines
    BORDER_POLAR  = "#172840"  # polar circle hint
    BORDER_MAIN   = "#1a2842"  # default borders + grid
    BORDER_CARD   = "#1d3150"  # metric card border
    BORDER_TROPIC = "#1e3050"  # tropic dashed hint
    BORDER_MID    = "#2a3d5a"  # legend edge, secondary borders
    BORDER_EQUATOR = "#2a4060" # equator line

    # Nav accent (active sidebar left border)
    NAV_ACTIVE = "#4facfe"
    NAV_ACTIVE_FG = "#00d4ff"

    # Text
    TEXT_MUTED    = "#8090aa"  # subtitles, labels, tick labels
    TEXT_TICK     = "#4a5a7a"  # axis tick marks
    TEXT_SECONDARY = "#c8d8e8" # secondary text, legend items
    TEXT_PRIMARY  = "#e8edf5"  # headings, important text

    # Semantic accent colours
    BLUE   = "#4facfe"   # info / DSN
    CYAN   = "#00d4ff"   # highlight / ITU
    GREEN  = "#43e97b"   # success / good precision
    YELLOW = "#f6d365"   # warning / coarse precision
    RED    = "#f5576c"   # error / fail
    ORANGE = "#f97316"   # secondary warning
    PURPLE = "#a78bfa"   # RUS network / misc


# ---------------------------------------------------------------------------
# Named accent palette  (used by MetricCard and anywhere a "colour name" is needed)
# ---------------------------------------------------------------------------
ACCENT: dict[str, str] = {
    "blue":   C.BLUE,
    "cyan":   C.CYAN,
    "green":  C.GREEN,
    "yellow": C.YELLOW,
    "red":    C.RED,
    "orange": C.ORANGE,
    "purple": C.PURPLE,
}

# ---------------------------------------------------------------------------
# Ground-station network colours
# ---------------------------------------------------------------------------
NET_COLORS: dict[str, str] = {
    "DSN":  C.BLUE,
    "ITU":  C.RED,
    "KGS":  C.GREEN,
    "ESA":  C.YELLOW,
    "RUS":  C.PURPLE,
    "ISRO": C.CYAN,
}

# ---------------------------------------------------------------------------
# Font settings
# ---------------------------------------------------------------------------
class FONT:
    MONO     = "Consolas"     # log console, code previews, axis ticks
    UI       = ""             # empty = system default
    SIZE_XS  = 8
    SIZE_SM  = 10
    SIZE_MD  = 11
    SIZE_LG  = 13
    SIZE_XL  = 19
    SIZE_H1  = 26             # metric card value


# ---------------------------------------------------------------------------
# Matplotlib rc_context dict for all dark figures
# ---------------------------------------------------------------------------
MPL_DARK: dict = {
    "figure.facecolor":  C.BG_DEEP,
    "axes.facecolor":    C.BG_PLOT,
    "axes.edgecolor":    C.BORDER_MAIN,
    "axes.labelcolor":   C.TEXT_MUTED,
    "axes.titlecolor":   C.TEXT_SECONDARY,
    "xtick.color":       C.TEXT_TICK,
    "ytick.color":       C.TEXT_TICK,
    "text.color":        C.TEXT_SECONDARY,
    "legend.facecolor":  C.BG_PANEL,
    "legend.edgecolor":  C.BORDER_MID,
    "legend.labelcolor": C.TEXT_SECONDARY,
    "grid.color":        C.BORDER_MAIN,
    "grid.linewidth":    0.5,
    "lines.linewidth":   1.6,
    "patch.edgecolor":   C.BORDER_MAIN,
}

# Convenience: bar chart series colours in order
BAR_COLORS = [C.BLUE, C.GREEN, C.YELLOW, C.RED, C.PURPLE, C.ORANGE, C.CYAN]
