"""Station catalogue search/export scope and keyboard access, without computation."""
import copy
import csv
from unittest.mock import patch

import pytest
from PyQt5.QtCore import Qt, QSettings, QPoint, QRect
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
    assert page._btn_save_net.isEnabled() and page._btn_edit.isEnabled()
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
    with patch.object(stations, "_SETTINGS", settings), patch.object(stations.StationsController, "_build_map_widget"):
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
                                page._btn_save_net, page._btn_edit, page._btn_delete):
                    rect = QRect(control.mapTo(page._toolbar, QPoint(0, 0)), control.size())
                    assert page._toolbar.rect().contains(rect), (control.text(), rect)
                assert page._search.text() == "DSN"
                assert table.item(table.currentRow(), 0).text() == selected
                assert [[table.item(r, c).text() for c in range(7)] for r in range(3)] == visible
                assert all(not table.isColumnHidden(c) for c in range(7))
                assert page._all_stations == before
                assert page._btn_save_net.isEnabled() and page._btn_edit.isEnabled()
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
