"""Colour tokens, fonts and matplotlib style for native pages.

Import this module anywhere you need palette values:
    from styles.theme import C, FONT, MPL_DARK, NET_COLORS

The values below are the Modern light palette. ``styles.appearance`` swaps
in the palette chosen in Ayarlar before any page is built; widget looks are
defined once in ``styles.workspace_light``.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
class C:
    """All palette colours in one namespace."""

    # Backgrounds
    BG_DEEP    = "#FFFFFF"   # page and figure background
    BG_SIDEBAR = "#F4F3EE"   # sidebar + footer
    BG_PLOT    = "#FFFFFF"   # matplotlib axes background
    BG_PANEL   = "#F4F3EE"   # panels, headers, legend fill
    BG_CARD    = "#FFFFFF"
    BG_HOVER   = "#EDEBE5"   # hover
    BG_ACTIVE  = "#F3E4E1"   # selection / pressed
    BG_FIELD   = "#FFFFFF"   # inputs, logs and other editable fields
    LAND       = "#F4F3EE"   # land masses on station maps
    DARK       = False       # True for the night palette (plots pick light-on-dark marks)

    # Borders and grid lines
    BORDER_FAINT  = "#ECEAE4"
    BORDER_POLAR  = "#DAD8D0"
    BORDER_MAIN   = "#DAD8D0"  # default borders + grid
    BORDER_CARD   = "#DAD8D0"
    BORDER_TROPIC = "#DAD8D0"
    BORDER_MID    = "#B9B6AD"  # legend edge, secondary borders
    BORDER_EQUATOR = "#B9B6AD"

    # Accent (primary buttons, active sidebar item, focus)
    NAV_ACTIVE = "#B83A2E"
    NAV_ACTIVE_FG = "#B83A2E"
    ON_ACCENT = "#FFFFFF"

    # Text
    TEXT_MUTED    = "#55524C"  # subtitles, captions, tick labels
    TEXT_TICK     = "#55524C"
    TEXT_SECONDARY = "#55524C"
    TEXT_PRIMARY  = "#151515"

    # Semantic colours (also the station network colours)
    BLUE   = "#2C5A7A"   # DSN
    CYAN   = "#046262"   # ISRO
    GREEN  = "#38623D"   # success / KGS
    YELLOW = "#805C10"   # warning / ESA
    RED    = "#A32622"   # error / ITU
    ORANGE = "#92531D"
    PURPLE = "#784C98"   # RUS

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
    SIZE_XS  = 11
    SIZE_SM  = 12
    SIZE_MD  = 13
    SIZE_LG  = 13
    SIZE_XL  = 19
    SIZE_H1  = 26             # metric card value


# ---------------------------------------------------------------------------
# Matplotlib rc_context dict for all figures (name kept for existing callers)
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

# Apply the default palette through the same path a saved choice uses.
from styles.appearance import apply_palette
apply_palette()
