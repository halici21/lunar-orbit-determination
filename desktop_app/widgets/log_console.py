"""Dark monospace log console with colour-coded output types."""
from __future__ import annotations
from PyQt5.QtWidgets import QPlainTextEdit, QWidget
from PyQt5.QtGui import QTextCharFormat, QColor, QFont, QTextCursor

from styles.theme import C, FONT


class LogConsole(QPlainTextEdit):
    """Read-only console widget.  Append lines via append_stdout / append_stderr etc."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(8000)
        font = QFont(FONT.MONO, 11)
        font.setStyleHint(QFont.Monospace)
        self.setFont(font)
        self.setStyleSheet(f"""
            QPlainTextEdit {{
                background-color: {C.BG_DEEP};
                color: {C.TEXT_SECONDARY};
                border: 1px solid {C.BORDER_MAIN};
                border-radius: 6px;
                padding: 8px;
                selection-background-color: #1d4770;
            }}
        """)

    # ------------------------------------------------------------------
    def append_stdout(self, line: str) -> None:
        self._insert(line, C.TEXT_SECONDARY)

    def append_stderr(self, line: str) -> None:
        self._insert(f"[ERR] {line}", C.RED)

    def append_info(self, line: str) -> None:
        self._insert(f"[INF] {line}", C.BLUE)

    def append_success(self, line: str) -> None:
        self._insert(f"[OK]  {line}", C.GREEN)

    def append_warning(self, line: str) -> None:
        self._insert(f"[WRN] {line}", C.YELLOW)

    def clear_log(self) -> None:
        self.clear()

    # ------------------------------------------------------------------
    def _insert(self, text: str, hex_color: str) -> None:
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(hex_color))
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text + "\n", fmt)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
