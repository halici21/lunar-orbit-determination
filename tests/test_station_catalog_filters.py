"""Station catalogue search/export scope and keyboard access, without computation."""
import copy
import csv
from unittest.mock import patch

import pytest
from PyQt5.QtCore import Qt, QSettings, QPoint, QRect, QItemSelectionModel
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from controllers import stations_controller as stations


@pytest.fixture
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def page(tmp_path, qt_app):
    settings = QSettings(str(tmp_path / "stations.ini"), QSettings.IniFormat)
    with patch.object(stations, "_SETTINGS", settings), patch.object(stations.StationsController, "_build_map_widget"):
        widget = stations.StationsController()
        widget.show(); qt_app.processEvents()
        yield widget
        widget.close(); qt_app.processEvents()


def test_empty_station_search_never_exports_hidden_catalogue(page):
    before = copy.deepcopy(page._all_stations)
    page._search.setText("no-station-matches-this-query")
    assert page.stationsTable.rowCount() == 0
    assert "Eşleşen istasyon yok" in page._filter_status.text()
    assert not page._btn_export.isEnabled()
    with patch.object(stations.QFileDialog, "getSaveFileName") as choose:
        page._export_csv()
    choose.assert_not_called()
    assert page._all_stations == before
    page._search.clear()
    assert page.stationsTable.rowCount() == len(before) and page._btn_export.isEnabled()


def test_filtered_station_csv_has_exact_visible_names_and_original_units(page, tmp_path):
    page._search.setText(" DSN ")
    assert page.stationsTable.rowCount() == 3
    page.stationsTable.selectRow(0)
    assert "3/15 istasyon · 1 seçili" in page._filter_status.text()
    assert page._btn_edit.isEnabled()
    assert not page._btn_delete.isEnabled()
    target = tmp_path / "dsn.csv"
    with patch.object(stations.QFileDialog, "getSaveFileName", return_value=(str(target), "CSV")), \
         patch.object(stations.QMessageBox, "information"):
        page._export_csv()
    with target.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {r["name"] for r in rows} == {"Canberra DSN", "Madrid DSN", "Goldstone DSN"}
    assert len(rows) == 3  # Export visible rows, not merely the selected row.
    for row in rows:
        source = next(s for s in page._all_stations if s["name"] == row["name"])
        for key in ("lat_deg", "lon_deg", "sigma_range_m", "sigma_angle_rad", "sigma_rr_mps"):
            assert row[key] == ("" if source[key] is None else str(source[key]))


def test_find_shortcut_selects_query_and_clear_restores_full_catalogue(page):
    page._search.setText("DSN")
    page.stationsTable.setFocus()
    QTest.keyClick(page.stationsTable, Qt.Key_F, Qt.ControlModifier)
    assert page._search.hasFocus() and page._search.selectedText() == "DSN"
    assert page._search.isClearButtonEnabled()
    assert "ağ koduyla" in page._search.accessibleName()
    QTest.keyClick(page._search, Qt.Key_Backspace)
    assert page.stationsTable.rowCount() == 15
    assert "15/15 istasyon" in page._filter_status.text()


def test_empty_catalogue_explains_source_or_addition_instead_of_search(page):
    page._all_stations = []
    page._apply_filter("")
    assert "İstasyon kataloğu boş" in page._filter_status.text()
    assert "Aramayı temizleyin" not in page._filter_status.text()
    assert page._btn_add.isEnabled() and not page._btn_export.isEnabled()


@pytest.mark.parametrize("size", [(760, 420), (910, 465)])
def test_narrow_station_window_keeps_actions_and_selection_without_page_horizontal_scroll(qt_app, tmp_path, size):
    from app import MainWindow

    settings = QSettings(str(tmp_path / "stations-window.ini"), QSettings.IniFormat)
    with patch.object(stations, "_SETTINGS", settings), patch.object(stations.StationsController, "_build_map_widget"), \
            patch("app.QSettings", return_value=settings):
        window = MainWindow()
        try:
            window.resize(1480, 1000); window.show(); window.environment.shutdown()
            window.navigate("stations"); qt_app.processEvents()
            page = window._pages["stations"]
            page._search.setText("DSN"); page.stationsTable.selectRow(0)
            before = copy.deepcopy(page._all_stations)
            table = page.stationsTable
            visible = [[table.item(r, c).text() for c in range(7)] for r in range(3)]
            selected = table.item(table.currentRow(), 0).text()
            for width, height in (size, (1480, 1000), size):
                window.resize(width, height); qt_app.processEvents()
                assert window.pageScroll.horizontalScrollBar().maximum() == 0
                for control in (page._search, page._btn_export, page._btn_add,
                                page._btn_edit, page._btn_delete):
                    rect = QRect(control.mapTo(page._toolbar, QPoint(0, 0)), control.size())
                    assert page._toolbar.rect().contains(rect), (control.text(), rect)
                assert page._search.text() == "DSN"
                assert table.item(table.currentRow(), 0).text() == selected
                assert [[table.item(r, c).text() for c in range(7)] for r in range(3)] == visible
                assert all(not table.isColumnHidden(c) for c in range(7))
                assert page._all_stations == before
                assert page._btn_edit.isEnabled()
                assert not page._btn_delete.isEnabled()
            # Other scientific forms retain their existing scrolling policy.
            window.navigate("scenario"); qt_app.processEvents()
            assert window.pageScroll.horizontalScrollBar().maximum() > 0
            window.navigate("stations"); qt_app.processEvents()
            assert window.pageScroll.horizontalScrollBar().maximum() == 0
        finally:
            window.close(); qt_app.processEvents()


@pytest.mark.parametrize("size", [(520, 300), (1120, 440)])
def test_catalogue_map_preserves_degree_scale_and_actual_coordinates(qt_app, size):
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.collections import PathCollection

    catalogue = stations._load_stations()
    canvas = FigureCanvasQTAgg(stations._build_map(catalogue))
    try:
        canvas.resize(*size); canvas.show(); qt_app.processEvents(); canvas.draw()
        axes = canvas.figure.axes[0]
        assert axes.get_legend() is None  # The key must not cover any geography.
        origin, east, north = axes.transData.transform([(0, 0), (10, 0), (0, 10)])
        assert east[0] - origin[0] == pytest.approx(north[1] - origin[1], abs=1e-8)
        assert axes.get_xlim() == (-180, 180) and axes.get_ylim() == (-90, 90)
        markers = {artist.get_label(): artist for artist in axes.collections if isinstance(artist, PathCollection)}
        assert set(markers) == {station["name"] for station in catalogue}
        for station in catalogue:
            assert tuple(markers[station["name"]].get_offsets()[0]) == (station["lon_deg"], station["lat_deg"])
    finally:
        canvas.close(); qt_app.processEvents()


def test_map_key_describes_full_catalogue_and_custom_shape_without_hidden_networks(page):
    from PyQt5.QtGui import QTextDocument

    def text():
        document = QTextDocument()
        document.setHtml(page._map_legend.text())
        return document.toPlainText().replace("\u00a0", " ")

    assert all(net in text() for net in ("DSN", "ESA", "ITU", "KGS", "RUS", "ISRO"))
    before = page._map_legend.text()
    page._search.setText("DSN")
    assert page._map_legend.text() == before  # The map still shows all stations.
    page._all_stations = [dict(page._all_stations[0], network="OTHER<tag>", _custom=True)]
    page._update_map_legend()
    assert "OTHER<tag>" in text() and "★ Özel istasyon" in text()
    assert "DSN" not in text() and "OTHER&lt;tag&gt;" in page._map_legend.text()
    assert "rengi ait olduğu ağı gösterir" in page._map_legend.accessibleDescription()
    page._all_stations = []
    page._update_map_legend()
    assert page._map_legend.isHidden() and not page._map_legend.text()


@pytest.mark.parametrize("column", [0, 2])
def test_sorted_filter_rebuild_keeps_coordinates_with_their_station(page, column):
    table = page.stationsTable
    expected = {s["name"]: (s["network"], f"{s['lat_deg']:.2f}", f"{s['lon_deg']:.2f}")
                for s in page._all_stations}
    table.sortItems(column, Qt.DescendingOrder)
    for query in ("DSN", "", "ESA", ""):
        page._search.setText(query)
        actual_names = set()
        for row in range(table.rowCount()):
            name = table.item(row, 0).text()
            actual_names.add(name)
            assert tuple(table.item(row, c).text() for c in (1, 2, 3)) == expected[name]
        assert len(actual_names) == table.rowCount()
        assert actual_names == {s["name"] for s in page._filtered_stations}
        assert table.isSortingEnabled() and not page._selected_rows()


def test_sorted_row_actions_use_visible_station_and_preserve_other_catalogue_entries(page):
    from PyQt5.QtWidgets import QDialog, QMessageBox

    original = copy.deepcopy(page._all_stations)
    custom = dict(original[0], name="ZZZ Custom", network="OTHER", _custom=True)
    page._all_stations.append(custom)
    page._apply_filter("")
    table = page.stationsTable
    table.sortItems(0, Qt.DescendingOrder)
    table.selectRow(0)
    assert table.item(0, 0).text() == "ZZZ Custom" and page._btn_delete.isEnabled()
    with patch.object(stations, "_StationDialog") as dialog:
        dialog.return_value.exec_.return_value = QDialog.Rejected
        page._edit_station()
        assert dialog.call_args.kwargs["station"] == custom
    table.selectRow(1)
    assert table.item(1, 0).text() == "Ussuriisk RUS" and not page._btn_delete.isEnabled()
    table.clearSelection()
    for row in (0, 1):
        table.selectionModel().select(table.model().index(row, 0),
                                      QItemSelectionModel.Select | QItemSelectionModel.Rows)
    # A sorted, multi-row selection still maps to the right catalogue records.
    assert [page._station_at_row(r)["name"] for r in page._selected_rows()] == ["ZZZ Custom", "Ussuriisk RUS"]
    assert not hasattr(page, "_btn_save_net")  # network presets fed no run; removed
    table.selectRow(0)
    with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes), \
         patch.object(QMessageBox, "information"):
        page._delete_station()
    assert page._all_stations == original
    assert "ZZZ Custom" not in {table.item(r, 0).text() for r in range(table.rowCount())}


def test_map_pick_after_sorting_selects_named_station_instead_of_source_row_number(page):
    from types import SimpleNamespace
    from matplotlib.collections import PathCollection

    page._search.setText("DSN")
    page.stationsTable.sortItems(0, Qt.DescendingOrder)
    page._on_map_pick(SimpleNamespace(artist=PathCollection([], label="Madrid DSN")))
    assert page._search.text() == "" and page.stationsTable.rowCount() == 15
    selected = page.stationsTable.currentRow()
    assert page.stationsTable.item(selected, 0).text() == "Madrid DSN"
    assert not page._btn_delete.isEnabled() and page._btn_edit.isEnabled()


@pytest.mark.parametrize("column,field,values,ascending", [
    (2, "lat_deg", [-20, -2, 2, 10, 0, 60, -70], [6, 0, 1, 4, 2, 3, 5]),
    (3, "lon_deg", [-100, -20, -2, 2, 10, 0, 60], [0, 1, 2, 5, 3, 4, 6]),
    (4, "sigma_range_m", [None, 10, 2, 0, 1.04, 1.03, 0.1], [3, 6, 5, 4, 2, 1, 0]),
    (5, "sigma_angle_rad", [None, .01, .002, 0, .0010004, .0010003, .0001], [3, 6, 5, 4, 2, 1, 0]),
    (6, "sigma_rr_mps", [None, .01, .002, 0, .0010004, .0010003, .0001], [3, 6, 5, 4, 2, 1, 0]),
])
def test_station_numeric_sort_orders_signed_values_and_precision_without_changing_records(
        page, tmp_path, column, field, values, ascending):
    """Hand-ordered records distinguish numbers from formatted strings and missing data."""
    template = page._all_stations[0]
    catalogue = [dict(template, name=f"Row {i}", **{field: value}) for i, value in enumerate(values)]
    page._all_stations = catalogue
    before = copy.deepcopy(catalogue)
    page._apply_filter("")
    table = page.stationsTable
    for direction, indices in ((Qt.AscendingOrder, ascending),
                               (Qt.DescendingOrder, list(reversed(ascending)))):
        table.sortItems(column, direction)
        assert [table.item(r, 0).text() for r in range(len(values))] == [f"Row {i}" for i in indices]
        for row, index in enumerate(indices):
            assert page._station_at_row(row) == catalogue[index]
        table.selectRow(2)
        with patch.object(stations, "_StationDialog") as dialog:
            dialog.return_value.exec_.return_value = stations.QDialog.Rejected
            page._edit_station()
        assert dialog.call_args.kwargs["station"] == catalogue[indices[2]]
        page._search.setText("Row 1")
        assert table.rowCount() == 1 and table.item(0, 0).text() == "Row 1"
        page._search.clear()
        assert [table.item(r, 0).text() for r in range(len(values))] == [f"Row {i}" for i in indices]
    if column >= 4:
        missing_row = next(r for r in range(len(values)) if table.item(r, 0).text() == "Row 0")
        assert table.item(missing_row, column).text() == "—"
        assert "Tanımlı değil" in table.item(missing_row, column).toolTip()
        rounded_rows = [next(r for r in range(len(values)) if table.item(r, 0).text() == f"Row {i}")
                        for i in (4, 5)]
        high, low = [table.item(r, column) for r in rounded_rows]
        assert high.text() == low.text() and high.toolTip() != low.toolTip()
    assert "sayısal" in table.horizontalHeaderItem(column).toolTip()
    target = tmp_path / "numeric-sort.csv"
    with patch.object(stations.QFileDialog, "getSaveFileName", return_value=(str(target), "CSV")), \
         patch.object(stations.QMessageBox, "information"):
        page._export_csv()
    with target.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [r["name"] for r in rows] == [r["name"] for r in catalogue]
    assert [r[field] for r in rows] == ["" if value is None else str(value) for value in values]
    assert page._all_stations == before


@pytest.mark.parametrize("size", [(520, 300), (1120, 440)])
def test_catalogue_station_labels_fit_map_without_covering_each_other_or_moving_markers(qt_app, size):
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.text import Annotation, Text

    catalogue = stations._load_stations()
    canvas = FigureCanvasQTAgg(stations._build_map(catalogue))
    try:
        canvas.resize(*size); canvas.show(); qt_app.processEvents(); canvas.draw()
        axes = canvas.figure.axes[0]
        renderer = canvas.get_renderer()
        labels = [a for a in axes.texts if isinstance(a, Annotation) and a.get_visible()]
        boxes = [Text.get_window_extent(a, renderer) for a in labels]
        assert len(labels) == len(catalogue)
        assert all(axes.bbox.contains(b.x0, b.y0) and axes.bbox.contains(b.x1, b.y1) for b in boxes)
        assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i+1:])
        expected = {s["name"]: (s["lon_deg"], s["lat_deg"]) for s in catalogue}
        assert {a.get_gid(): a.xy for a in labels} == expected
        # Labels cannot conceal even a neighboring station's exact marker center.
        centers = axes.transData.transform(list(expected.values()))
        assert not any(box.contains(x, y) for box in boxes for x, y in centers)
        # A leader line must not steal a marker click; only the name is pickable.
        from matplotlib.backend_bases import MouseEvent
        for label, box in zip(labels, boxes):
            picker = label.get_picker()
            x, y = axes.transData.transform(label.xy)
            assert not picker(label, MouseEvent("button_press_event", canvas, x, y))[0]
            assert picker(label, MouseEvent("button_press_event", canvas,
                                           (box.x0+box.x1)/2, (box.y0+box.y1)/2))[0]
        assert any(a.get_text() == "New Norcia" for a in labels)
        # Layout must be deterministic across repeated draws and a resize round trip.
        positions = {a.get_gid(): a.get_position() for a in labels}
        canvas.draw()
        assert {a.get_gid(): a.get_position() for a in labels} == positions
        canvas.resize(760, 360); qt_app.processEvents(); canvas.draw()
        canvas.resize(*size); qt_app.processEvents(); canvas.draw()
        assert {a.get_gid(): a.get_position() for a in labels} == positions
    finally:
        canvas.close(); qt_app.processEvents()


def test_station_label_pick_uses_full_identity_through_a_sorted_filter(page):
    from types import SimpleNamespace
    from matplotlib.text import Annotation

    page._search.setText("DSN")
    page.stationsTable.sortItems(2, Qt.DescendingOrder)
    label = Annotation("Cebreros", (0, 0), gid="Cebreros ESA")
    page._on_map_pick(SimpleNamespace(artist=label))
    assert page._search.text() == ""
    selected = page.stationsTable.currentRow()
    assert page.stationsTable.item(selected, 0).text() == "Cebreros ESA"
    assert page._station_at_row(selected)["name"] == "Cebreros ESA"
    assert page._btn_edit.isEnabled() and not page._btn_delete.isEnabled()


def test_dense_and_edge_custom_labels_keep_every_exact_marker_without_text_overflow(qt_app):
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.collections import PathCollection
    from matplotlib.text import Annotation, Text

    template = stations._load_stations()[0]
    catalogue = [dict(template, name=f"Cluster station {i}", lat_deg=40, lon_deg=-4, _custom=True)
                 for i in range(24)]
    catalogue += [dict(template, name="Özel $x$ <station>", lat_deg=89.8, lon_deg=179.8, _custom=True),
                  dict(template, name="Very long station name " * 30, lat_deg=-89.8, lon_deg=-179.8)]
    before = copy.deepcopy(catalogue)
    canvas = FigureCanvasQTAgg(stations._build_map(catalogue))
    try:
        canvas.resize(520, 300); canvas.show(); qt_app.processEvents(); canvas.draw()
        axes = canvas.figure.axes[0]
        annotations = [a for a in axes.texts if isinstance(a, Annotation)]
        visible = [a for a in annotations if a.get_visible()]
        assert 0 < len(visible) < len(catalogue)  # Only optional names may hide.
        boxes = [Text.get_window_extent(a, canvas.get_renderer()) for a in visible]
        assert all(axes.bbox.contains(b.x0, b.y0) and axes.bbox.contains(b.x1, b.y1) for b in boxes)
        assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i+1:])
        markers = {a.get_label(): tuple(a.get_offsets()[0]) for a in axes.collections
                   if isinstance(a, PathCollection)}
        assert markers == {s["name"]: (s["lon_deg"], s["lat_deg"]) for s in catalogue}
        assert len(annotations) == len(catalogue) and all(a.get_picker() for a in annotations)
        assert all(not a.get_parse_math() for a in annotations)
        assert catalogue == before
    finally:
        canvas.close(); qt_app.processEvents()
