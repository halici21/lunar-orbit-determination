"""Station catalogue search/export scope and keyboard access, without computation."""
import copy
import csv
from unittest.mock import patch

import pytest
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from controllers import stations_controller as stations


@pytest.fixture
def page(tmp_path):
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "stations.ini"), QSettings.IniFormat)
    with patch.object(stations, "_SETTINGS", settings), patch.object(stations.StationsController, "_build_map_widget"):
        widget = stations.StationsController()
        widget.show(); app.processEvents()
        yield widget
        widget.close(); app.processEvents()


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
