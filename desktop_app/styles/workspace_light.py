"""Shared controls for the shell and every workspace page.

The look follows sunumatik.vercel.app: a quiet ground, hairline borders, one
accent used only for the primary action, the active item and focus, and small
upper-case section labels instead of boxed group frames.

Pages give a widget a look through a ``role`` property (``set_role``) rather
than their own stylesheet, so every page draws captions, logs and danger
buttons the same way and the palette lives in one place.
"""
import tempfile
from pathlib import Path

from styles.theme import C

UI_FONT = "'Segoe UI Variable Text', 'Segoe UI'"
DISPLAY_FONT = "'Segoe UI Variable Display', 'Segoe UI'"
MONO_FONT = "'Cascadia Mono', Consolas"


def tr_upper(text):
    """Turkish-aware upper case: i → İ and ı → I (str.upper maps i → I)."""
    return text.replace("i", "İ").replace("ı", "I").upper()


def polish_page(root):
    """Give every group box on a page the small upper-case section label."""
    from PyQt5.QtWidgets import QGroupBox
    for box in root.findChildren(QGroupBox):
        if box.title() and not box.property("sectionLabel"):
            box.setProperty("sectionLabel", True)
            box.setTitle(tr_upper(box.title()))


def set_role(widget, role):
    """Apply one of the shared roles below (caption, eyebrow, placeholder, log, disclosure, danger)."""
    widget.setProperty("role", role)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    return widget


_ICONS = {
    "chevron-down": '<path d="M3 5.5 7 9.5 11 5.5" fill="none" stroke="{c}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>',
    "chevron-up": '<path d="M3 8.5 7 4.5 11 8.5" fill="none" stroke="{c}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>',
    "check": '<path d="M3 7.2 5.8 10 11 4.2" fill="none" stroke="{c}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>',
}


def _icon(name, color):
    """Write a palette-coloured 14 px SVG once and return a QSS url() path."""
    folder = Path(tempfile.gettempdir()) / "lunar_od_ui_icons"
    folder.mkdir(exist_ok=True)
    path = folder / f"{name}-{color.lstrip('#').lower()}.svg"
    if not path.exists():
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 14 14">'
                        + _ICONS[name].format(c=color) + "</svg>", encoding="utf-8")
    return path.as_posix()


def workspace_stylesheet():
    down, up = _icon("chevron-down", C.TEXT_MUTED), _icon("chevron-up", C.TEXT_MUTED)
    check = _icon("check", C.ON_ACCENT)
    return f"""
QWidget {{ background:{C.BG_DEEP}; color:{C.TEXT_PRIMARY}; font-family:{UI_FONT}; font-size:13px; }}
QWidget#actionBar {{ border-top:1px solid {C.BORDER_MAIN}; }}
QLabel {{ background:transparent; }}
QLabel#workspaceTitle, QLabel#consoleTitle {{ font-family:{DISPLAY_FONT}; font-size:20px; font-weight:600; }}
QLabel#detailHeading {{ font-size:14px; font-weight:600; }}
QLabel#selectionValue {{ font-size:26px; font-family:{MONO_FONT}; }}

QPushButton {{ background:transparent; color:{C.TEXT_PRIMARY}; border:1px solid {C.BORDER_MID}; border-radius:8px; padding:6px 14px; min-height:20px; font-weight:normal; }}
QPushButton:hover {{ background:{C.BG_HOVER}; border-color:{C.TEXT_MUTED}; }}
QPushButton:pressed {{ background:{C.BG_ACTIVE}; }}
QPushButton:checked {{ background:{C.BG_ACTIVE}; color:{C.NAV_ACTIVE_FG}; border-color:{C.NAV_ACTIVE}; }}
QPushButton:focus {{ border-color:{C.NAV_ACTIVE}; }}
QPushButton:disabled {{ color:{C.TEXT_MUTED}; background:transparent; border-color:{C.BORDER_MAIN}; }}
QPushButton[primary="true"] {{ color:{C.ON_ACCENT}; background:{C.NAV_ACTIVE}; border-color:{C.NAV_ACTIVE}; font-weight:600; }}
QPushButton[primary="true"]:hover {{ background:{C.NAV_ACTIVE_FG}; border-color:{C.NAV_ACTIVE_FG}; }}
QPushButton[primary="true"]:disabled {{ color:{C.TEXT_MUTED}; background:{C.BG_PANEL}; border-color:{C.BORDER_MAIN}; }}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ background:{C.BG_FIELD}; color:{C.TEXT_PRIMARY}; border:1px solid {C.BORDER_MID}; border-radius:8px; padding:5px 10px; min-height:22px; selection-background-color:{C.BG_ACTIVE}; selection-color:{C.TEXT_PRIMARY}; }}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color:{C.TEXT_MUTED}; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color:{C.NAV_ACTIVE}; }}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{ color:{C.TEXT_MUTED}; border-color:{C.BORDER_MAIN}; }}
QLineEdit[readOnly="true"] {{ background:transparent; border-color:{C.BORDER_MAIN}; color:{C.TEXT_SECONDARY}; }}
QComboBox {{ padding-right:28px; }}
QComboBox::drop-down {{ border:0; width:26px; subcontrol-origin:padding; subcontrol-position:center right; }}
QComboBox::down-arrow {{ image:url("{down}"); width:14px; height:14px; }}
QComboBox QAbstractItemView {{ background:{C.BG_PANEL}; color:{C.TEXT_PRIMARY}; border:1px solid {C.BORDER_MID}; padding:4px; outline:0; selection-background-color:{C.BG_ACTIVE}; selection-color:{C.TEXT_PRIMARY}; }}
QSpinBox, QDoubleSpinBox {{ padding-right:24px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-origin:border; subcontrol-position:top right; width:22px; border:0; background:transparent; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-origin:border; subcontrol-position:bottom right; width:22px; border:0; background:transparent; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image:url("{up}"); width:11px; height:11px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image:url("{down}"); width:11px; height:11px; }}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background:{C.BG_HOVER}; border-radius:6px; }}

QCheckBox {{ spacing:8px; background:transparent; }}
QCheckBox::indicator {{ width:16px; height:16px; border:1px solid {C.BORDER_MID}; border-radius:4px; background:{C.BG_FIELD}; }}
QCheckBox::indicator:hover {{ border-color:{C.TEXT_MUTED}; }}
QCheckBox::indicator:checked {{ background:{C.NAV_ACTIVE}; border-color:{C.NAV_ACTIVE}; image:url("{check}"); }}
QCheckBox::indicator:disabled {{ border-color:{C.BORDER_MAIN}; background:transparent; }}

QGroupBox {{ border:0; margin-top:26px; padding-top:10px; font-size:11px; font-weight:600; color:{C.TEXT_MUTED}; }}
QGroupBox::title {{ subcontrol-origin:margin; left:0; padding:0 10px 0 0; background:{C.BG_DEEP}; color:{C.TEXT_MUTED}; }}

QTableView, QTreeWidget, QTreeView, QListView {{ background:{C.BG_DEEP}; alternate-background-color:{C.BG_PANEL}; color:{C.TEXT_PRIMARY}; border:1px solid {C.BORDER_MAIN}; border-radius:10px; gridline-color:{C.BORDER_FAINT}; selection-background-color:{C.BG_ACTIVE}; selection-color:{C.TEXT_PRIMARY}; outline:0; }}

QTableView::item:selected, QTreeWidget::item:selected {{ background:{C.BG_ACTIVE}; color:{C.TEXT_PRIMARY}; }}
QTableView::item:hover, QTreeWidget::item:hover {{ background:{C.BG_HOVER}; }}
QHeaderView {{ background:transparent; border:0; }}
QHeaderView::section {{ background:{C.BG_PANEL}; color:{C.TEXT_MUTED}; border:0; border-bottom:1px solid {C.BORDER_MAIN}; padding:9px 8px; font-size:12px; font-weight:normal; }}
QTableCornerButton::section {{ background:{C.BG_PANEL}; border:0; }}
QTreeWidget::item {{ padding:6px 2px; }}

QPlainTextEdit, QTextEdit {{ background:{C.BG_FIELD}; color:{C.TEXT_PRIMARY}; border:1px solid {C.BORDER_MAIN}; border-radius:10px; padding:10px; selection-background-color:{C.BG_ACTIVE}; selection-color:{C.TEXT_PRIMARY}; }}

QTabWidget::pane {{ border:0; border-top:1px solid {C.BORDER_MAIN}; }}
QTabBar {{ background:transparent; }}
QTabBar::tab {{ background:transparent; color:{C.TEXT_MUTED}; padding:10px 4px; margin-right:20px; border:0; border-bottom:2px solid transparent; }}
QTabBar::tab:selected {{ color:{C.TEXT_PRIMARY}; border-bottom-color:{C.NAV_ACTIVE}; }}
QTabBar::tab:hover:!selected {{ color:{C.TEXT_SECONDARY}; }}
QWidget#pageTabStrip {{ border-bottom:1px solid {C.BORDER_MAIN}; }}

QSplitter::handle {{ background:{C.BORDER_MAIN}; width:1px; height:1px; }}
QScrollArea {{ border:0; background:transparent; }}
QScrollBar:vertical {{ background:transparent; width:10px; margin:2px; }}
QScrollBar::handle:vertical {{ background:{C.BORDER_MID}; min-height:28px; border-radius:3px; }}
QScrollBar::handle:vertical:hover {{ background:{C.TEXT_MUTED}; }}
QScrollBar:horizontal {{ background:transparent; height:10px; margin:2px; }}
QScrollBar::handle:horizontal {{ background:{C.BORDER_MID}; min-width:28px; border-radius:3px; }}
QScrollBar::handle:horizontal:hover {{ background:{C.TEXT_MUTED}; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{ width:0; height:0; background:transparent; }}

QProgressBar {{ border:0; border-radius:3px; background:{C.BORDER_MAIN}; color:{C.TEXT_PRIMARY}; text-align:center; max-height:6px; }}
QProgressBar::chunk {{ background:{C.NAV_ACTIVE}; border-radius:3px; }}
QSlider::groove:horizontal {{ background:{C.BORDER_MID}; height:3px; border-radius:1px; }}
QSlider::sub-page:horizontal {{ background:{C.NAV_ACTIVE}; height:3px; border-radius:1px; }}
QSlider::handle:horizontal {{ background:{C.NAV_ACTIVE}; width:14px; height:14px; margin:-6px 0; border-radius:7px; }}
QSlider::handle:horizontal:focus {{ background:{C.BG_DEEP}; border:2px solid {C.NAV_ACTIVE}; width:10px; height:10px; margin:-6px 0; }}
QToolTip {{ background:{C.BG_PANEL}; color:{C.TEXT_PRIMARY}; border:1px solid {C.BORDER_MID}; border-radius:6px; padding:6px 8px; }}
QMenuBar {{ background:{C.BG_DEEP}; color:{C.TEXT_SECONDARY}; border-bottom:1px solid {C.BORDER_MAIN}; }}
QMenuBar::item {{ padding:5px 10px; background:transparent; }}
QMenuBar::item:selected {{ background:{C.BG_HOVER}; color:{C.TEXT_PRIMARY}; }}
QMenu {{ background:{C.BG_PANEL}; border:1px solid {C.BORDER_MID}; padding:4px; }}
QMenu::item {{ padding:6px 18px; border-radius:6px; }}
QMenu::item:selected {{ background:{C.BG_ACTIVE}; }}
QStatusBar {{ background:{C.BG_DEEP}; color:{C.TEXT_MUTED}; border-top:1px solid {C.BORDER_MAIN}; }}
QToolBar {{ background:transparent; border:none; }}
QToolButton {{ color:{C.TEXT_SECONDARY}; background:transparent; border-radius:6px; padding:3px; }}
QToolButton:hover {{ background:{C.BG_HOVER}; }}
QDialog, QMessageBox {{ background:{C.BG_PANEL}; }}

QLabel[role="caption"] {{ color:{C.TEXT_MUTED}; font-size:12px; }}
QLabel[role="eyebrow"] {{ color:{C.NAV_ACTIVE}; font-size:11px; font-weight:600; letter-spacing:1px; }}
QLabel[role="placeholder"] {{ color:{C.TEXT_MUTED}; font-size:13px; }}
QLabel[role="summary"] {{ background:{C.BG_PANEL}; border:1px solid {C.BORDER_MAIN}; border-radius:10px; padding:12px 16px; color:{C.TEXT_SECONDARY}; }}
QPlainTextEdit[role="log"] {{ color:{C.TEXT_SECONDARY}; font-family:{MONO_FONT}; font-size:11px; border-radius:8px; padding:6px; }}
QPushButton[role="disclosure"] {{ background:transparent; border:0; text-align:left; padding:6px 0; color:{C.TEXT_MUTED}; }}
QPushButton[role="disclosure"]:hover {{ color:{C.TEXT_PRIMARY}; background:transparent; }}
QPushButton[role="disclosure"]:focus {{ border:0; color:{C.NAV_ACTIVE_FG}; }}
QPushButton[role="danger"] {{ color:{C.RED}; border-color:{C.RED}; }}
QPushButton[role="danger"]:disabled {{ color:{C.TEXT_MUTED}; border-color:{C.BORDER_MAIN}; }}
"""
