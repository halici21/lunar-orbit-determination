"""Sunumatik Modern and Orchid palettes, selected at application startup."""
THEMES = {"modern": "Modern · beyaz / kırmızı", "orchid": "Orchid · gül / bordo"}
THEME_KEY = "app_settings/appearance"
CURRENT_THEME = "modern"
MODERN = dict(bg="#FFFFFF", paper="#F4F3EE", ink="#151515", muted="#55524C",
    accent="#B83A2E", on_accent="#FFFFFF", line="#DAD8D0", faint="#ECEAE4",
    strong_line="#B9B6AD", hover="#EDEBE5", selection="#F3E4E1")
ORCHID = dict(bg="#F8F3F2", paper="#EFE9E9", ink="#5A2132", muted="#76545F",
    accent="#A84D6A", on_accent="#FFF8F5", line="#D8C7CC", faint="#E8DCDF",
    strong_line="#B99AA4", hover="#E8DCDF", selection="#E8DCDF")


def apply_palette(name="modern"):
    """Call before constructing pages; plot constants stay fixed for the session."""
    global CURRENT_THEME
    from styles import theme
    CURRENT_THEME = name if name in THEMES else "modern"
    p = ORCHID if CURRENT_THEME == "orchid" else MODERN
    mapping = {
        "BG_DEEP": "bg", "BG_SIDEBAR": "paper", "BG_PLOT": "bg", "BG_PANEL": "paper",
        "BG_CARD": "bg", "BG_HOVER": "hover", "BG_ACTIVE": "selection",
        "BORDER_FAINT": "faint", "BORDER_POLAR": "line", "BORDER_MAIN": "line",
        "BORDER_CARD": "line", "BORDER_TROPIC": "line", "BORDER_MID": "strong_line",
        "BORDER_EQUATOR": "strong_line", "NAV_ACTIVE": "accent", "NAV_ACTIVE_FG": "accent",
        "ON_ACCENT": "on_accent", "TEXT_MUTED": "muted", "TEXT_TICK": "muted",
        "TEXT_SECONDARY": "muted", "TEXT_PRIMARY": "ink",
    }
    for key, token in mapping.items():
        setattr(theme.C, key, p[token])
    for key, value in dict(BLUE="#2C5A7A", CYAN="#046262", GREEN="#38623D",
        YELLOW="#805C10", RED="#A32622", ORANGE="#92531D", PURPLE="#784C98").items():
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
    return apply_palette(str(settings.value(THEME_KEY, "modern")))
