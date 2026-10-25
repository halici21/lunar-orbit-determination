"""Sunumatik palettes (Gece, Modern, Orchid), selected at application startup.

Gece follows the sunumatik.vercel.app catalogue: near-black ground, two surface
steps, hairline borders and a single gold accent. Modern and Orchid are the
light deck palettes.
"""
THEMES = {"modern": "Modern · beyaz / kırmızı", "gece": "Gece · grafit / altın", "orchid": "Orchid · gül / bordo"}
THEME_KEY = "app_settings/appearance"
# Light by default: the owner found the near-black ground heavy for long sessions.
DEFAULT_THEME = "modern"
CURRENT_THEME = DEFAULT_THEME
# Gece is a softened graphite (not near-black) so long sessions stay easy on the eye.
GECE = dict(bg="#12151B", paper="#181C24", field="#181C24", ink="#E6E9EF", secondary="#C0C6D1",
    muted="#8F97A6", accent="#C9A35C", accent_2="#DCBB7C", on_accent="#14171D", line="#262C38",
    faint="#1D222B", strong_line="#323A48", hover="#1D222B", selection="#2B2820", land="#252C3A")
MODERN = dict(bg="#FFFFFF", paper="#F6F5F1", ink="#1C1B19", secondary="#3F3D39", muted="#6A665F",
    accent="#B83A2E", on_accent="#FFFFFF", line="#E4E2DB", faint="#EFEDE7",
    strong_line="#CFCCC3", hover="#F1EFEA", selection="#F6E7E4", land="#ECEAE3")
ORCHID = dict(bg="#F8F3F2", paper="#EFE9E9", ink="#5A2132", muted="#76545F",
    accent="#A84D6A", on_accent="#FFF8F5", line="#D8C7CC", faint="#E8DCDF",
    strong_line="#B99AA4", hover="#E8DCDF", selection="#E8DCDF")
PALETTES = {"gece": GECE, "modern": MODERN, "orchid": ORCHID}
# Semantic colours double as station-network colours, so each palette keeps them
# readable on its own ground: deep tones on paper, light tones on the night ground.
SEMANTIC = {
    "gece": dict(BLUE="#7FA7D9", CYAN="#5EC4C4", GREEN="#6FBF8E", YELLOW="#DCBB7C",
                 RED="#E07A6F", ORANGE="#E0A064", PURPLE="#B49CE0"),
    "light": dict(BLUE="#2C5A7A", CYAN="#046262", GREEN="#38623D", YELLOW="#805C10",
                  RED="#A32622", ORANGE="#92531D", PURPLE="#784C98"),
}


def apply_palette(name=DEFAULT_THEME):
    """Call before constructing pages; plot constants stay fixed for the session."""
    global CURRENT_THEME
    from styles import theme
    CURRENT_THEME = name if name in THEMES else DEFAULT_THEME
    p = PALETTES[CURRENT_THEME]
    mapping = {
        "BG_DEEP": "bg", "BG_SIDEBAR": "paper", "BG_PLOT": "bg", "BG_PANEL": "paper",
        "BG_CARD": "paper", "BG_FIELD": "field", "BG_HOVER": "hover", "BG_ACTIVE": "selection",
        "BORDER_FAINT": "faint", "BORDER_POLAR": "line", "BORDER_MAIN": "line",
        "BORDER_CARD": "line", "BORDER_TROPIC": "line", "BORDER_MID": "strong_line",
        "BORDER_EQUATOR": "strong_line", "NAV_ACTIVE": "accent", "NAV_ACTIVE_FG": "accent_2",
        "ON_ACCENT": "on_accent", "TEXT_MUTED": "muted", "TEXT_TICK": "muted", "LAND": "land",
        "TEXT_SECONDARY": "secondary", "TEXT_PRIMARY": "ink",
    }
    fallback = {"field": "bg", "secondary": "muted", "accent_2": "accent", "land": "paper"}
    for key, token in mapping.items():
        setattr(theme.C, key, p.get(token, p.get(fallback.get(token, token))))
    theme.C.DARK = CURRENT_THEME == "gece"
    for key, value in SEMANTIC["gece" if theme.C.DARK else "light"].items():
        setattr(theme.C, key, value)
    c = theme.C
    theme.NET_COLORS.update(DSN=c.BLUE, ITU=c.RED, KGS=c.GREEN, ESA=c.YELLOW, RUS=c.PURPLE, ISRO=c.CYAN)
    theme.BAR_COLORS[:] = [c.BLUE, c.GREEN, c.YELLOW, c.RED, c.PURPLE, c.ORANGE, c.CYAN]
    theme.MPL_DARK.update({"figure.facecolor": c.BG_DEEP, "axes.facecolor": c.BG_PLOT,
        "axes.edgecolor": c.BORDER_MAIN, "axes.labelcolor": c.TEXT_MUTED,
        "axes.titlecolor": c.TEXT_SECONDARY, "xtick.color": c.TEXT_TICK,
        "ytick.color": c.TEXT_TICK, "text.color": c.TEXT_SECONDARY,
        "legend.facecolor": c.BG_PANEL, "legend.edgecolor": c.BORDER_MID,
        "legend.labelcolor": c.TEXT_SECONDARY, "grid.color": c.BORDER_MAIN,
        "patch.edgecolor": c.BORDER_MAIN})
    return CURRENT_THEME


def load_saved_theme(settings=None):
    if settings is None:
        from PyQt5.QtCore import QSettings
        settings = QSettings("LunarOD", "DesktopApp")
    return apply_palette(str(settings.value(THEME_KEY, DEFAULT_THEME)))
