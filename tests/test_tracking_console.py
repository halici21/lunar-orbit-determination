"""Artifact integrity and native desktop workflow tests; no physics execution."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from services.tracking_results import (LoadCancelled, ResultLoadError,
    discover_tracking_runs, load_tracking_results, result_selection)
from controllers.tracking_console_controller import TrackingConsoleController


def make_run(folder, prefix="test", rows=None):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{prefix}_ukf_updates.csv"
    rows = rows or [
        [1, 2, 10, 1, 200, .08, 3, 1, True],
        [2, 2, 20, 2, 100, .04, 6, 2, False],
        [3, 3, 40, 1, 50, .02, 1.5, .5, True],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["update_id", "arc_id", "t_s", "station_id", "position_error_m", "velocity_error_mps", "nis", "normalized_nis", "accepted"])
        writer.writerows(rows)
    return path


def wait_until(app, predicate, timeout=5):
    end = time.monotonic() + timeout
    while not predicate() and time.monotonic() < end:
        app.processEvents()
        time.sleep(.005)
    app.processEvents()
    assert predicate(), "Qt workflow timed out"


@pytest.fixture
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


@pytest.fixture
def console(app):
    widget = TrackingConsoleController()
    widget.resize(1450, 920)
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    app.processEvents()


def test_discovery_recurses_without_guessing_other_csv(tmp_path):
    a = make_run(tmp_path / "nested", "one")
    (tmp_path / "summary.csv").write_text("a,b\n1,2")
    runs = discover_tracking_runs(tmp_path)
    assert [r.updates_path for r in runs] == [a.resolve()]


def test_loader_preserves_values_and_unknown_provenance(tmp_path):
    path = make_run(tmp_path)
    result = load_tracking_results(path)
    assert result.updates[0]["position_error_m"] == 200
    assert result.updates[1]["accepted"] is False
    assert result.updates[2]["t_s"] == 40
    assert result.sources[0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result.metadata["frame"] == "Bilinmiyor"
    assert result.states == ()
    assert any("Durum vektörleri yok" in note for note in result.notes)


@pytest.mark.parametrize("change", ["NaN", "Infinity", "-3", ""])
def test_loader_rejects_nonfinite_or_negative_error(tmp_path, change):
    path = make_run(tmp_path)
    path.write_text(path.read_text().replace("200,", change + ",", 1))
    with pytest.raises(ResultLoadError, match="position_error_m"):
        load_tracking_results(path)


def test_loader_rejects_duplicate_id_and_bad_boolean(tmp_path):
    path = make_run(tmp_path)
    original = path.read_text()
    path.write_text(original.replace("2,2,20", "1,2,20"))
    with pytest.raises(ResultLoadError, match="yinelenen update_id"):
        load_tracking_results(path)
    path.write_text(original.replace("False", "maybe"))
    with pytest.raises(ResultLoadError, match="boolean"):
        load_tracking_results(path)


def test_bls_pairs_by_exact_prefix_and_preserves_failure(tmp_path):
    path = make_run(tmp_path)
    (tmp_path / "unrelated_bls_arc_markers.csv").write_text("bad")
    assert load_tracking_results(path).arcs == ()
    (tmp_path / "test_bls_arc_markers.csv").write_text(
        "arc_id,start_s,end_s,arc_end_position_error_m,arc_end_velocity_error_mps,operational_success,stop_reason\n2,10,20,30,.01,False,J-Stab\n")
    result = load_tracking_results(path)
    assert result.arcs[0]["operational_success"] is False
    assert result.arcs[0]["end_s"] == 20
    assert len(result.sources) == 2


def write_manifest(folder, **overrides):
    info = {"schema": "lunar-od-result-v1", "units": {"time": "s", "position": "m", "velocity": "m/s"},
        "origin": "Moon", "frame": "MCI/J2000", "epoch": "2025-01-01T00:00:00", "time_scale": "TDB", "seed": 42,
        "state_file": "test_states.csv"}
    info.update(overrides)
    (folder / "test_result_manifest.json").write_text(json.dumps(info))
    (folder / "test_states.csv").write_text("t_s,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps\n10,1800000,0,0,0,1600,0\n20,1790000,16000,0,-20,1600,0\n")


def test_state_contract_preserves_units_frame_and_source(tmp_path):
    path = make_run(tmp_path)
    write_manifest(tmp_path)
    result = load_tracking_results(path)
    assert result.states[0]["x_m"] == 1800000
    assert result.states[1]["vy_mps"] == 1600
    assert result.metadata["frame"] == "MCI/J2000"
    assert result.metadata["seed"] == 42
    assert len(result.sources) == 3


@pytest.mark.parametrize("override", [
    {"units": {"time": "s", "position": "km", "velocity": "km/s"}},
    {"origin": "Earth"}, {"frame": ""}, {"state_file": "../test_states.csv"},
])
def test_state_contract_rejects_ambiguous_geometry(tmp_path, override):
    path = make_run(tmp_path)
    write_manifest(tmp_path, **override)
    with pytest.raises(ResultLoadError):
        load_tracking_results(path)


def test_cancel_stops_discovery_and_loading(tmp_path):
    path = make_run(tmp_path)
    with pytest.raises(LoadCancelled):
        load_tracking_results(path, lambda: True)
    with pytest.raises(LoadCancelled):
        discover_tracking_runs(tmp_path, lambda: True)


def test_filters_preserve_source_indices_and_empty_range(tmp_path):
    result = load_tracking_results(make_run(tmp_path))
    assert result_selection(result.updates, station=1) == (0, 2)
    assert result_selection(result.updates, arc=2, accepted_only=True) == (0,)
    assert result_selection(result.updates, start=30, end=25) == ()


def test_async_folder_load_and_table_chart_slider_linkage(app, console, tmp_path):
    make_run(tmp_path)
    console.open_folder(tmp_path)
    wait_until(app, lambda: console.result is not None)
    assert len(console.result.updates) == 3
    assert console.table_model.rowCount() == 3
    console.time_slider.setValue(1)
    assert console.selected_index == 1
    assert console.time_label.text() == "T+00:00:20"
    assert all(chart.selected_index == 1 for chart in console.charts)
    console.table.setCurrentIndex(console.table_model.index(2, 0))
    assert console.selected_index == 2
    QTest.keyClick(console.charts[0], Qt.Key_Left)
    assert console.selected_index == 1
    console.set_time_range(11, 21)
    assert console.visible_indices == (1,)
    console.start_hours.setValue(1)
    console.end_hours.setValue(0)
    console.apply_filters()
    assert console.selected_index is None
    assert not console.time_slider.isEnabled()


def test_source_export_contains_only_visible_original_rows(console, tmp_path):
    path = make_run(tmp_path)
    original = path.read_bytes()
    console._accept_result(load_tracking_results(path))
    console.station_combo.setCurrentIndex(console.station_combo.findData(1))
    console.apply_filters()
    dest = tmp_path / "filtered.csv"
    console.export_visible(dest)
    with dest.open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r["update_id"]) for r in rows] == [1, 3]
    info = json.loads(dest.with_suffix(".source.json").read_text())
    assert info["sources"][0]["sha256"] == hashlib.sha256(original).hexdigest()
    assert info["filters"]["station_id"] == 1
    with pytest.raises(OSError, match="Kaynak"):
        console.export_visible(path)
    assert path.read_bytes() == original


def test_load_error_preserves_previous_result(app, console, tmp_path):
    path = make_run(tmp_path)
    console._accept_result(load_tracking_results(path))
    previous = console.result
    errors = []
    console.loadFailed.connect(errors.append)
    bad = tmp_path / "bad_ukf_updates.csv"
    bad.write_text("x,y\n1,2")
    console.open_result(bad)
    wait_until(app, lambda: bool(errors))
    assert console.result is previous
    assert console.export_button.isEnabled()


def test_worker_does_not_block_event_loop_and_cancel_discards_result(app, console, tmp_path):
    path = make_run(tmp_path)
    original = load_tracking_results
    def slow_load(path, cancelled):
        for _ in range(30):
            if cancelled():
                raise LoadCancelled()
            time.sleep(.01)
        return original(path, cancelled)
    ticks = []
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start()
    with patch("controllers.tracking_console_controller.load_tracking_results", slow_load):
        console.open_result(path)
        wait_until(app, lambda: len(ticks) >= 5)
        console.cancel_load()
        wait_until(app, lambda: not console._workers)
    timer.stop()
    assert console.result is None
    assert len(ticks) >= 5


def test_main_window_creates_only_requested_pages(app):
    from app import MainWindow
    window = MainWindow()
    try:
        assert set(window._pages) == {"mission"}
        window.navigate("console")
        instance = window._pages["console"]
        window.navigate("console")
        assert window._pages["console"] is instance
        window.navigate("settings")
        assert set(window._pages) == {"mission", "console", "settings"}
    finally:
        window.close()


def test_main_window_routes_real_artifact_to_console(app, tmp_path):
    from app import MainWindow
    window = MainWindow()
    path = make_run(tmp_path)
    try:
        window.open_in_result_console(path)
        console = window._pages["console"]
        wait_until(app, lambda: console.result is not None)
        assert console.result.sources[0]["path"] == str(path.resolve())
        assert console.table_model.rowCount() == 3
    finally:
        window.close()
