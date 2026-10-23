"""Shared light controls for the shell and native result workspaces.

Pages give a widget a look through a ``role`` property (``set_role``) rather
than their own stylesheet, so every page draws captions, logs and danger
buttons the same way and the palette lives in one place.
"""
from styles.theme import C


def set_role(widget, role):
    """Apply one of the shared roles below (caption, placeholder, log, disclosure, danger)."""
    widget.setProperty("role", role)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    return widget


def workspace_stylesheet():
    return f"""
QWidget {{ background:{C.BG_DEEP}; color:{C.TEXT_PRIMARY}; font-family:'Segoe UI'; font-size:13px; }}
QLabel {{ background:transparent; }}
QLabel#workspaceTitle, QLabel#consoleTitle {{ font-size:20px; font-weight:500; }}
QLabel#detailHeading {{ font-size:14px; font-weight:600; }}
QLabel#selectionValue {{ font-size:26px; font-family:Consolas; }}
QPushButton {{ background:{C.BG_DEEP}; color:{C.TEXT_PRIMARY}; border:1px solid {C.BORDER_MAIN}; border-radius:4px; padding:6px 11px; min-height:20px; }}
QPushButton:hover {{ background:{C.BG_HOVER}; border-color:{C.BORDER_MID}; }}
QPushButton:checked {{ background:{C.BG_ACTIVE}; border-color:{C.NAV_ACTIVE}; }}
QPushButton:focus {{ border:1px solid {C.NAV_ACTIVE}; }}
QPushButton:disabled {{ color:{C.TEXT_MUTED}; background:{C.BG_PANEL}; border-color:{C.BORDER_FAINT}; }}
QPushButton[primary="true"] {{ color:{C.ON_ACCENT}; background:{C.NAV_ACTIVE}; border-color:{C.NAV_ACTIVE}; }}
QPushButton[primary="true"]:disabled {{ color:{C.TEXT_MUTED}; background:{C.BG_PANEL}; border-color:{C.BORDER_MAIN}; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ background:{C.BG_DEEP}; color:{C.TEXT_PRIMARY}; border:1px solid {C.BORDER_MAIN}; border-radius:3px; padding:5px 8px; min-height:22px; selection-background-color:{C.BG_ACTIVE}; selection-color:{C.TEXT_PRIMARY}; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color:{C.NAV_ACTIVE}; }}
QComboBox QAbstractItemView {{ background:{C.BG_DEEP}; color:{C.TEXT_PRIMARY}; selection-background-color:{C.BG_ACTIVE}; selection-color:{C.TEXT_PRIMARY}; }}
QGroupBox {{ border:0; border-top:1px solid {C.BORDER_MAIN}; margin-top:20px; padding-top:16px; font-weight:500; }}
QGroupBox::title {{ subcontrol-origin:margin; left:0; padding:0 8px 0 0; background:{C.BG_DEEP}; }}
QTableView, QTreeWidget {{ background:{C.BG_DEEP}; alternate-background-color:{C.BG_PANEL}; color:{C.TEXT_PRIMARY}; border:0; gridline-color:{C.BORDER_FAINT}; selection-background-color:{C.BG_ACTIVE}; selection-color:{C.TEXT_PRIMARY}; }}
QTableView::item:selected, QTreeWidget::item:selected {{ background:{C.BG_ACTIVE}; color:{C.TEXT_PRIMARY}; }}
QTableView::item:hover, QTreeWidget::item:hover {{ background:{C.BG_HOVER}; }}
QHeaderView::section {{ background:{C.BG_PANEL}; color:{C.TEXT_MUTED}; border:0; border-bottom:1px solid {C.BORDER_MAIN}; padding:9px 8px; }}
QTreeWidget::item {{ padding:6px 2px; }}
QPlainTextEdit, QTextEdit {{ background:{C.BG_DEEP}; color:{C.TEXT_PRIMARY}; border:1px solid {C.BORDER_MAIN}; padding:10px; selection-background-color:{C.BG_ACTIVE}; selection-color:{C.TEXT_PRIMARY}; }}
QTabWidget::pane {{ border:0; border-top:1px solid {C.BORDER_MAIN}; }}
QTabBar::tab {{ background:transparent; color:{C.TEXT_MUTED}; padding:10px 14px; border:0; border-bottom:2px solid transparent; }}
QTabBar::tab:selected {{ color:{C.TEXT_PRIMARY}; border-bottom-color:{C.NAV_ACTIVE}; }}
QTabBar::tab:hover {{ background:{C.BG_HOVER}; }}
QSplitter::handle {{ background:{C.BORDER_FAINT}; width:1px; height:1px; }}
QScrollArea {{ border:0; }}
QScrollBar:vertical {{ background:{C.BG_PANEL}; width:10px; }}
QScrollBar::handle:vertical {{ background:{C.BORDER_MID}; min-height:24px; border-radius:4px; }}
QScrollBar:horizontal {{ background:{C.BG_PANEL}; height:10px; }}
QScrollBar::handle:horizontal {{ background:{C.BORDER_MID}; min-width:24px; border-radius:4px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width:0; height:0; }}
QProgressBar {{ border:0; background:{C.BG_PANEL}; color:{C.TEXT_PRIMARY}; text-align:center; }}
QProgressBar::chunk {{ background:{C.NAV_ACTIVE}; }}
QSlider::groove:horizontal {{ background:{C.BORDER_MAIN}; height:3px; }}
QSlider::handle:horizontal {{ background:{C.NAV_ACTIVE}; width:12px; margin:-5px 0; border-radius:6px; }}
QToolTip {{ background:{C.BG_PANEL}; color:{C.TEXT_PRIMARY}; border:1px solid {C.BORDER_MID}; padding:6px; }}
QStatusBar {{ background:{C.BG_PANEL}; color:{C.TEXT_MUTED}; }}
QToolBar {{ background:{C.BG_PANEL}; border:none; }}
QToolButton {{ color:{C.TEXT_SECONDARY}; background:transparent; }}
QLabel[role="caption"] {{ color:{C.TEXT_MUTED}; font-size:12px; }}
QLabel[role="placeholder"] {{ color:{C.TEXT_MUTED}; font-size:13px; }}
QPlainTextEdit[role="log"] {{ color:{C.TEXT_MUTED}; font-family:Consolas,monospace; font-size:10px; border-radius:3px; padding:4px; }}
QPushButton[role="disclosure"] {{ background:transparent; border:0; text-align:left; padding:6px 0; color:{C.TEXT_MUTED}; }}
QPushButton[role="disclosure"]:hover {{ color:{C.TEXT_PRIMARY}; }}
QPushButton[role="disclosure"]:focus {{ border:0; color:{C.NAV_ACTIVE}; }}
QPushButton[role="danger"] {{ color:{C.RED}; border-color:{C.RED}; }}
QPushButton[role="danger"]:disabled {{ color:{C.TEXT_MUTED}; border-color:{C.BORDER_FAINT}; }}
"""
