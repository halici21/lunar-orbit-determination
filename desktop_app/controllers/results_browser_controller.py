"""Results browser page controller — file tree, CSV table, PNG viewer."""
from __future__ import annotations
import os
import subprocess
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QTreeWidgetItem, QTableWidgetItem, QFileDialog, QMessageBox,
    QAbstractItemView, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QLineEdit,
)
from PyQt5.QtCore import Qt, QSortFilterProxyModel, QTimer
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5 import uic

from services.project_paths import DESKTOP_APP_DIR, RESULTS_DIR
from services.result_indexer import index_result_folders
from services.csv_loader import load_csv_page, csv_summary

UI_PATH = DESKTOP_APP_DIR / "ui" / "pages" / "results_browser_page.ui"

_STACK_EMPTY = 0
_STACK_CSV = 1
_STACK_IMAGE = 2
_CSV_PAGE_SIZE = 1000


class ResultsBrowserController(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        uic.loadUi(str(UI_PATH), self)
        self._zoom_factor = 1.0
        self._current_path: Path | None = None
        self._current_pixmap: QPixmap | None = None
        self._csv_page_index = 0
        self._csv_total_rows = 0
        self._csv_total_pages = 0
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(300)
        self._setup()

    def _setup(self) -> None:
        self.fileTree.setColumnWidth(0, 240)
        self.fileTree.itemClicked.connect(self._on_item_clicked)
        self.searchBox.textChanged.connect(lambda _: self._search_timer.start())
        self._search_timer.timeout.connect(self._filter_tree)
        self.openExternalBtn.clicked.connect(self._open_external)
        self.exportBtn.clicked.connect(self._export_file)
        self.zoomInBtn.clicked.connect(self._zoom_in)
        self.zoomOutBtn.clicked.connect(self._zoom_out)
        self.zoomFitBtn.clicked.connect(self._zoom_fit)
        self._build_csv_pager()
        self._build_csv_filter()
        self.contentStack.setCurrentIndex(_STACK_EMPTY)
        self._populate_tree()

    def _build_csv_pager(self) -> None:
        pager = QHBoxLayout()
        pager.setContentsMargins(0, 0, 0, 0)
        pager.setSpacing(8)

        self.csvPrevBtn = QPushButton("Prev")
        self.csvNextBtn = QPushButton("Next")
        self.csvPageLabel = QLabel("")
        self.csvPageLabel.setStyleSheet("QLabel { color: #8090aa; font-size: 11px; }")
        self.csvPrevBtn.clicked.connect(self._prev_csv_page)
        self.csvNextBtn.clicked.connect(self._next_csv_page)

        pager.addWidget(self.csvPrevBtn)
        pager.addWidget(self.csvNextBtn)
        pager.addWidget(self.csvPageLabel)
        pager.addStretch()
        self.csvLayout.insertLayout(1, pager)

    def _build_csv_filter(self) -> None:
        """Build a filter bar: column selector + operator + value + Apply/Clear."""
        from styles.theme import C, FONT
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 4)
        bar.setSpacing(6)

        filter_lbl = QLabel("Filter:")
        filter_lbl.setStyleSheet(f"color:{C.TEXT_MUTED}; font-size:{FONT.SIZE_SM}px;")
        bar.addWidget(filter_lbl)

        self._filter_col = QComboBox()
        self._filter_col.setFixedWidth(180)
        self._filter_col.setStyleSheet(
            f"QComboBox {{ background:#0a1428; color:#c8d8e8;"
            f" border:1px solid #1a2842; border-radius:3px; padding:2px 4px;"
            f" font-size:{FONT.SIZE_SM}px; }}"
            f"QComboBox::drop-down {{ border:none; }}"
        )
        bar.addWidget(self._filter_col)

        self._filter_op = QComboBox()
        self._filter_op.addItems(["contains", "=", "!=", ">", ">=", "<", "<="])
        self._filter_op.setFixedWidth(80)
        self._filter_op.setStyleSheet(self._filter_col.styleSheet())
        bar.addWidget(self._filter_op)

        self._filter_val = QLineEdit()
        self._filter_val.setPlaceholderText("value…")
        self._filter_val.setFixedWidth(120)
        self._filter_val.setStyleSheet(
            f"QLineEdit {{ background:#0a1428; color:#c8d8e8;"
            f" border:1px solid #1a2842; border-radius:3px; padding:2px 6px;"
            f" font-size:{FONT.SIZE_SM}px; }}"
        )
        self._filter_val.returnPressed.connect(self._apply_csv_filter)
        bar.addWidget(self._filter_val)

        apply_btn = QPushButton("Apply")
        apply_btn.setFixedWidth(60)
        apply_btn.clicked.connect(self._apply_csv_filter)
        apply_btn.setStyleSheet(
            f"QPushButton {{ background:#4facfe; color:#fff;"
            f" border-radius:3px; font-size:{FONT.SIZE_SM}px; padding:3px 8px; }}"
            f"QPushButton:hover {{ background:#00d4ff; }}"
        )
        bar.addWidget(apply_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(50)
        clear_btn.clicked.connect(self._clear_csv_filter)
        clear_btn.setStyleSheet(
            f"QPushButton {{ background:#0d1a31; color:#8090aa;"
            f" border:1px solid #1a2842; border-radius:3px;"
            f" font-size:{FONT.SIZE_SM}px; padding:3px 6px; }}"
            f"QPushButton:hover {{ background:#162038; }}"
        )
        bar.addWidget(clear_btn)

        self._filter_match_lbl = QLabel("")
        self._filter_match_lbl.setStyleSheet(f"color:#8090aa; font-size:{FONT.SIZE_SM}px;")
        bar.addWidget(self._filter_match_lbl)
        bar.addStretch()

        # Insert above csvTable in csvLayout (index 0 = info label, 1 = pager, 2+ = table)
        self.csvLayout.insertLayout(1, bar)
        self._csv_filter_active = False

    def _apply_csv_filter(self) -> None:
        col_name = self._filter_col.currentText()
        op = self._filter_op.currentText()
        val_str = self._filter_val.text().strip()
        if not col_name or not val_str:
            return

        tbl = self.csvTable
        # Find column index
        col_idx = -1
        for c in range(tbl.columnCount()):
            if tbl.horizontalHeaderItem(c) and tbl.horizontalHeaderItem(c).text() == col_name:
                col_idx = c
                break
        if col_idx < 0:
            return

        match_count = 0
        for row in range(tbl.rowCount()):
            item = tbl.item(row, col_idx)
            cell_str = item.text() if item else ""
            show = self._filter_match(cell_str, op, val_str)
            tbl.setRowHidden(row, not show)
            if show:
                match_count += 1

        self._csv_filter_active = True
        self._filter_match_lbl.setText(f"{match_count} rows match")

    def _clear_csv_filter(self) -> None:
        tbl = self.csvTable
        for row in range(tbl.rowCount()):
            tbl.setRowHidden(row, False)
        self._filter_match_lbl.setText("")
        self._filter_val.clear()
        self._csv_filter_active = False

    @staticmethod
    def _filter_match(cell: str, op: str, val: str) -> bool:
        if op == "contains":
            return val.lower() in cell.lower()
        try:
            cell_f = float(cell)
            val_f = float(val)
            if op == "=":   return cell_f == val_f
            if op == "!=":  return cell_f != val_f
            if op == ">":   return cell_f > val_f
            if op == ">=":  return cell_f >= val_f
            if op == "<":   return cell_f < val_f
            if op == "<=":  return cell_f <= val_f
        except (ValueError, TypeError):
            if op in ("=", "!="):
                match = cell.strip().lower() == val.strip().lower()
                return match if op == "=" else not match
        return True

    def on_shown(self) -> None:
        # Refresh tree if it's empty (first visit)
        if self.fileTree.topLevelItemCount() == 0:
            self._populate_tree()

    # ------------------------------------------------------------------
    def _populate_tree(self) -> None:
        self.fileTree.clear()
        folders = index_result_folders(RESULTS_DIR)
        total_files = 0
        for folder in folders:
            folder_item = QTreeWidgetItem([folder["name"]])
            folder_item.setData(0, Qt.UserRole, ("folder", folder["path"]))
            folder_item.setForeground(0, Qt.GlobalColor.cyan)
            folder_item.setExpanded(False)

            for csv_path in folder["csv_files"]:
                child = QTreeWidgetItem([csv_path.name])
                child.setData(0, Qt.UserRole, ("csv", csv_path))
                child.setForeground(0, Qt.GlobalColor.white)
                folder_item.addChild(child)
                total_files += 1

            for png_path in folder["png_files"]:
                child = QTreeWidgetItem([png_path.name])
                child.setData(0, Qt.UserRole, ("image", png_path))
                child.setForeground(0, Qt.GlobalColor.yellow)
                folder_item.addChild(child)
                total_files += 1

            self.fileTree.addTopLevelItem(folder_item)

        self.fileCountLabel.setText(f"{total_files} files in {len(folders)} folders")

    def _filter_tree(self) -> None:
        query = self.searchBox.text().strip().lower()
        root = self.fileTree.invisibleRootItem()
        for i in range(root.childCount()):
            folder_item = root.child(i)
            folder_visible = False
            for j in range(folder_item.childCount()):
                child = folder_item.child(j)
                match = not query or query in child.text(0).lower()
                child.setHidden(not match)
                if match:
                    folder_visible = True
            folder_item.setHidden(not folder_visible and bool(query))
            if folder_visible:
                folder_item.setExpanded(True)

    # ------------------------------------------------------------------
    def _on_item_clicked(self, item: QTreeWidgetItem, col: int) -> None:
        data = item.data(0, Qt.UserRole)
        if data is None:
            return
        kind, path = data
        if kind == "folder":
            return  # expand/collapse handled by Qt
        self._load_file(kind, path)

    def _load_file(self, kind: str, path: Path) -> None:
        self._current_path = path
        self.filePathLabel.setText(str(path))
        self.openExternalBtn.setEnabled(True)
        self.exportBtn.setEnabled(True)

        if kind == "csv":
            self._load_csv(path)
        elif kind == "image":
            self._load_image(path)

    def _load_csv(self, path: Path) -> None:
        self.contentStack.setCurrentIndex(_STACK_CSV)
        info = csv_summary(path)
        if info.get("error"):
            self.csvInfoLabel.setText(f"Failed to load CSV: {info['error']}")
            self._clear_csv_table()
            return

        self._csv_page_index = 0
        self._csv_total_rows = int(info["rows"])
        self._csv_total_pages = max(1, int(np.ceil(self._csv_total_rows / _CSV_PAGE_SIZE))) if self._csv_total_rows else 1
        self.csvInfoLabel.setText(
            f"{info['rows']} rows × {info['cols']} columns  ·  {info['size_kb']:.1f} KB  ·  {path.name}"
        )
        self._render_csv_page()

    def _render_csv_page(self) -> None:
        if self._current_path is None:
            return

        df = load_csv_page(self._current_path, self._csv_page_index, _CSV_PAGE_SIZE)
        if df is None:
            self.csvInfoLabel.setText("Failed to load CSV page (pandas missing or parse error)")
            self._clear_csv_table()
            return

        self.csvTable.setSortingEnabled(False)
        self.csvTable.clear()
        self.csvTable.setRowCount(len(df))
        self.csvTable.setColumnCount(len(df.columns))
        self.csvTable.setHorizontalHeaderLabels(list(df.columns))
        # Update filter column combo
        self._filter_col.clear()
        self._filter_col.addItems(list(df.columns))
        self._clear_csv_filter()

        for row_idx, row in enumerate(df.itertuples(index=False)):
            for col_idx, val in enumerate(row):
                cell = QTableWidgetItem(str(val) if val is not None else "")
                cell.setFlags(cell.flags() & ~Qt.ItemIsEditable)
                self.csvTable.setItem(row_idx, col_idx, cell)

        self.csvTable.setSortingEnabled(True)
        self.csvTable.resizeColumnsToContents()
        self._update_csv_pager()

    def _clear_csv_table(self) -> None:
        self.csvTable.clear()
        self.csvTable.setRowCount(0)
        self.csvTable.setColumnCount(0)
        self.csvPrevBtn.setEnabled(False)
        self.csvNextBtn.setEnabled(False)
        self.csvPageLabel.setText("")

    def _update_csv_pager(self) -> None:
        first = self._csv_page_index * _CSV_PAGE_SIZE + 1 if self._csv_total_rows else 0
        last = min((self._csv_page_index + 1) * _CSV_PAGE_SIZE, self._csv_total_rows)
        self.csvPageLabel.setText(
            f"Rows {first}-{last} of {self._csv_total_rows}  ·  "
            f"Page {self._csv_page_index + 1}/{self._csv_total_pages}"
        )
        self.csvPrevBtn.setEnabled(self._csv_page_index > 0)
        self.csvNextBtn.setEnabled(self._csv_page_index + 1 < self._csv_total_pages)

    def _prev_csv_page(self) -> None:
        if self._csv_page_index <= 0:
            return
        self._csv_page_index -= 1
        self._render_csv_page()

    def _next_csv_page(self) -> None:
        if self._csv_page_index + 1 >= self._csv_total_pages:
            return
        self._csv_page_index += 1
        self._render_csv_page()

    def _load_image(self, path: Path) -> None:
        self.contentStack.setCurrentIndex(_STACK_IMAGE)
        pix = QPixmap(str(path))
        if pix.isNull():
            self.imageLabel.setText(f"Could not load:\n{path.name}")
            return
        self._current_pixmap = pix
        w, h = pix.width(), pix.height()
        self.imageSizeLabel.setText(f"{path.name}  ·  {w}×{h} px")
        self._zoom_factor = 1.0
        self._render_image()

    def _render_image(self) -> None:
        if self._current_pixmap is None:
            return
        pix = self._current_pixmap
        new_w = int(pix.width() * self._zoom_factor)
        new_h = int(pix.height() * self._zoom_factor)
        scaled = pix.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.imageLabel.setPixmap(scaled)

    def _zoom_in(self) -> None:
        self._zoom_factor = min(self._zoom_factor * 1.25, 8.0)
        self._render_image()

    def _zoom_out(self) -> None:
        self._zoom_factor = max(self._zoom_factor / 1.25, 0.1)
        self._render_image()

    def _zoom_fit(self) -> None:
        if self._current_pixmap is None:
            return
        scroll = self.imageScrollArea
        avail_w = scroll.viewport().width() - 20
        avail_h = scroll.viewport().height() - 20
        ratio_w = avail_w / max(1, self._current_pixmap.width())
        ratio_h = avail_h / max(1, self._current_pixmap.height())
        self._zoom_factor = min(ratio_w, ratio_h, 1.0)
        self._render_image()

    # ------------------------------------------------------------------
    def _open_external(self) -> None:
        if self._current_path is None:
            return
        folder = self._current_path.parent
        try:
            subprocess.Popen(f'explorer "{folder}"', shell=True)
        except Exception:
            pass

    def _export_file(self) -> None:
        if self._current_path is None:
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Export file",
            str(Path.home() / self._current_path.name),
            f"{self._current_path.suffix.upper()[1:]} files (*{self._current_path.suffix})",
        )
        if dest:
            import shutil
            shutil.copy2(self._current_path, dest)

    # ------------------------------------------------------------------
    def show_file(self, path: Path) -> None:
        """External API: navigate to and display a specific file."""
        if not path.exists():
            return
        kind = "csv" if path.suffix.lower() == ".csv" else "image"
        self._load_file(kind, path)
        # Expand the parent folder in tree
        self._expand_to(path)

    def _expand_to(self, path: Path) -> None:
        root = self.fileTree.invisibleRootItem()
        for i in range(root.childCount()):
            folder_item = root.child(i)
            for j in range(folder_item.childCount()):
                child = folder_item.child(j)
                data = child.data(0, Qt.UserRole)
                if data and data[1] == path:
                    folder_item.setExpanded(True)
                    self.fileTree.setCurrentItem(child)
                    return
