"""Recorded geometry provenance, analytical orbit oracle, and linked native views."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import copy
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PyQt5.QtCore import QSettings, Qt, QPoint, QRect
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from app import MainWindow, _PRIMARY_NAV
from controllers.mission_controller import MissionController
from services.mission_data import recorded_geometry, validate_geometry, orbit_summary, station_catalog
from services.scenario_artifacts import write_console_bundle
from services.run_details import load_run_with_details, inspect_run
from services.tracking_results import load_tracking_results, ResultLoadError
from tests.test_run_workspace_resume import native_run
from tests.test_tracking_console import wait_until


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def observation_arc(kind="range_rate", *, arc_id=7):
    stations = [SimpleNamespace(name="Goldstone DSN", lat_deg=35.3, lon_deg=-116.81, alt_m=969.67),
                SimpleNamespace(name="Canberra DSN", lat_deg=-35.4, lon_deg=148.98, alt_m=680.)]
    column = {"position": 4, "range_rate": 5, "two_way_range": 2}[kind]
    data = np.zeros((5, column+1))
    data[:, 0] = [10., 20., 40., 50., 20.]
    data[:, column] = [1, 1, 1, 1, 2]
    return SimpleNamespace(arc_id=arc_id, pass_geo=SimpleNamespace(stations=stations, measurement_type=kind),
                           obs_data=data, t_pass_s=np.array([10., 20., 40., 50.]))


def geometry_run(tmp_path):
    job, folder, _, path = native_run(tmp_path)
    saved = json.loads(job.read_text())
    solved = SimpleNamespace(arc_id=7, estimated_state=np.array([1800000., 0., 0., 0., 1600., 0.]),
        final_position_error_m=25., final_velocity_error_mps=.2, ukf_normalized_mean_nis=None,
        operational_success=False, stop_reason="MaxIter", observed_station_ids=(0, 1))
    write_console_bundle(SimpleNamespace(estimator_type="bls_lm", arc_results=[solved]),
        [observation_arc()], [10., 100.], np.array([[1800000., 0., 0., 0., 1600., 0.], [0., 1800000., 0., -1600., 0., 0.]]),
        "2027-03-02 23:32:37", config=SimpleNamespace(name="fixture", output_dir=str(folder), sample_step_s=10.),
        job_path=job, options=saved["options"], project_root=tmp_path)
    return path


@pytest.mark.parametrize("kind", ["position", "range_rate", "two_way_range"])
def test_actual_measurement_column_and_gap_are_preserved(kind):
    arc = observation_arc(kind)
    data = validate_geometry(recorded_geometry([arc], SimpleNamespace(sample_step_s=10)))
    assert data["windows"] == [
        {"station_id": 0, "arc_id": 7, "start_s": 10., "end_s": 20., "samples": 2},
        {"station_id": 0, "arc_id": 7, "start_s": 40., "end_s": 50., "samples": 2},
        {"station_id": 1, "arc_id": 7, "start_s": 20., "end_s": 20., "samples": 1}]
    assert data["stations"][1]["name"] == "Canberra DSN"  # Local network ID, not full catalogue index.


def test_arc_boundaries_are_not_joined_and_station_changes_fail():
    first, second = observation_arc(), observation_arc(arc_id=8)
    data = recorded_geometry([first, second], SimpleNamespace(sample_step_s=10))
    assert len(data["windows"]) == 6 and {w["arc_id"] for w in data["windows"]} == {7, 8}
    second.pass_geo.stations.reverse()
    with pytest.raises(ValueError, match="kimliği"):
        recorded_geometry([first, second], SimpleNamespace(sample_step_s=10))
    first.obs_data[0, 5] = 3
    with pytest.raises(ValueError):
        recorded_geometry([first], SimpleNamespace(sample_step_s=10))


@pytest.mark.parametrize("change", [
    lambda d: d.update(scope="visibility"), lambda d: d.update(sample_step_s=float("nan")),
    lambda d: d["stations"][0].update(lat_deg=91), lambda d: d["stations"][1].update(station_id=0),
    lambda d: d["windows"][0].update(station_id=99), lambda d: d["windows"][0].update(end_s=0),
    lambda d: d["windows"][0].update(samples=True),
])
def test_bad_geometry_is_rejected(change):
    data = recorded_geometry([observation_arc()], SimpleNamespace(sample_step_s=10))
    change(data)
    with pytest.raises(ValueError):
        validate_geometry(data)


def test_native_bundle_roundtrip_tracks_geometry_source_and_rejects_tampering(tmp_path):
    path = geometry_run(tmp_path)
    result, details = load_run_with_details(path)
    assert len(result.sources) == 7
    assert result.metadata["mission_geometry"]["scope"] == "recorded_measurement_samples"
    assert details["config"] and details["job"]
    sidecar = path.parent / result.metadata["mission_file"]
    sidecar.write_text(sidecar.read_text()+" ")
    with pytest.raises(ResultLoadError, match="SHA-256"):
        load_tracking_results(path)
    with pytest.raises(ResultLoadError, match="değişti"):
        inspect_run(result)


def test_geometry_cannot_escape_source_folder_or_lack_hash(tmp_path):
    path = geometry_run(tmp_path)
    manifest = path.with_name("fixture_result_manifest.json")
    data = json.loads(manifest.read_text())
    sidecar = path.parent / data["mission_file"]
    outside = tmp_path / "external.json"
    outside.write_bytes(sidecar.read_bytes())
    data["mission_file"] = str(outside)
    manifest.write_text(json.dumps(data))
    with pytest.raises(ResultLoadError, match="aynı klasörde"):
        load_tracking_results(path)
    data["mission_file"] = sidecar.name
    data.pop("mission_sha256")
    manifest.write_text(json.dumps(data))
    with pytest.raises(ResultLoadError, match="SHA-256"):
        load_tracking_results(path)


def test_polar_circular_orbit_matches_analytical_oracle_and_requires_recorded_gm():
    radius, mu = 1837400., 4.9028e12
    state = dict(x_m=radius, y_m=0., z_m=0., vx_mps=0., vy_mps=0., vz_mps=math.sqrt(mu/radius))
    values = orbit_summary(state, mu=mu)
    assert values["altitude_km"] == pytest.approx(100., abs=1e-10)
    assert values["inclination_deg"] == pytest.approx(90., abs=1e-12)
    assert values["a_km"] == pytest.approx(radius/1000, rel=1e-12)
    assert values["e"] == pytest.approx(0., abs=1e-12)
    assert values["period_h"] == pytest.approx(2*math.pi*math.sqrt(radius**3/mu)/3600, rel=1e-12)
    assert orbit_summary(state)["period_h"] is None
    state["vz_mps"] = math.sqrt(3*mu/radius)
    assert orbit_summary(state, mu=mu)["a_km"] is None  # Unbound, no invented period.


@pytest.fixture
def mission(app, tmp_path):
    path = geometry_run(tmp_path)
    result, details = load_run_with_details(path)
    initial = result.states[0]
    result = replace(result, states=tuple(dict(initial, t_s=t) for t in (10., 20., 30., 40., 50., 100.)))
    page = MissionController()
    page.resize(1240, 920); page.show()
    page.set_result(result, details)
    app.processEvents()
    yield page, path
    page.shutdown(); page.close(); app.processEvents()


def test_map_click_and_time_selection_link_all_views(mission, app):
    page, _ = mission
    point = page.world_map._point(-116.81, 35.3).toPoint()
    QTest.mouseClick(page.world_map, Qt.LeftButton, pos=point)
    assert page.gantt.selected_station == "Goldstone DSN"
    assert page.station_picker.currentData() == "Goldstone DSN"
    assert "35.300" in page.station_info.text()
    page.select_time(20.)
    assert page.orbit.selected_t == page.gantt.cursor == 20.
    assert page.world_map.active == {"Goldstone DSN", "Canberra DSN"}
    page.select_time(30.)
    assert not page.world_map.active  # An actual missing-observation gap remains empty.
    page.gantt.timeSelected.emit(39.)
    assert page.orbit.selected_t == page.gantt.cursor == 40.
    assert page.slider.value() == 3


@pytest.mark.parametrize("successes,expected", [
    ([False, False], "0/2 yay kaydı başarılı"),
    ([True, None], "1/1 yay kaydı başarılı · 1 kayıtta başarı alanı yok"),
    ([None, None], "operasyonel başarı bilinmiyor"),
])
def test_mission_completion_does_not_infer_success_from_truth(mission, successes, expected):
    page, _ = mission
    row = page.result.updates[0]
    result = replace(page.result, updates=tuple(dict(row, operational_success=v) for v in successes))
    page.set_result(result, {"status": "İşlem tamamlandı"})
    assert "İşlem tamamlandı" in page.run_status.text() and expected in page.run_status.text()
    assert "Yakınsama: kaynakta ayrı bir doğrulama alanı yok" in page.run_status.text()
    assert "başarı kanıtı değildir" in page.run_status.toolTip()
    assert page.details_button.isEnabled()
    before = page.run_status.text()
    page.select_time(40.)
    assert page.run_status.text() == before  # Playback never changes completion/acceptance.


def test_mission_details_opens_same_result_and_preserves_status_after_failed_replacement(app, tmp_path):
    with patch("app.QSettings", return_value=QSettings(str(tmp_path / "details.ini"), QSettings.IniFormat)):
        window = MainWindow()
    try:
        window.show(); app.processEvents()
        page = window._pages["mission"]
        assert not page.details_button.isEnabled()
        path = geometry_run(tmp_path)
        window.open_in_mission(path)
        wait_until(app, lambda: page.result is not None)
        status = page.run_status.text()
        page.play_button.setChecked(True)
        page.details_button.click()
        assert window._active_page == "console"
        assert window._pages["console"].result is page.result
        assert window._pages["console"].details_toggle.isChecked()
        assert not window._pages["console"].inspector_tabs.isHidden()
        assert window._pages["console"].inspector_tabs.currentIndex() == 0
        assert not page._timer.isActive()
        window.open_in_mission(tmp_path / "missing.csv")
        wait_until(app, lambda: not window._pages["console"]._busy)
        assert page.run_status.text() == status and "Koşu açılamadı" in page.run_label.text()
    finally:
        window.close()


def test_keyboard_station_choice_and_gantt_label_keep_time_and_link_selection(mission, app):
    page, _ = mission
    page.select_time(20.)
    page.station_picker.setFocus()
    QTest.keyClick(page.station_picker, Qt.Key_Down)
    assert page.world_map.selected == page.gantt.selected_station == "Goldstone DSN"
    assert "ölçüm kaydı var" in page.station_info.text()
    QTest.mouseClick(page.gantt, Qt.LeftButton, pos=QPoint(70, 49))
    assert page.station_picker.currentData() == "Canberra DSN"
    assert page.world_map.selected == page.gantt.selected_station == "Canberra DSN"
    assert page.orbit.selected_t == page.gantt.cursor == 20.  # A row label never seeks time.
    page.select_time(30.)
    assert "ölçüm kaydı yok" in page.station_info.text()  # Real gap, regardless of selection.
    page.station_picker.setCurrentIndex(0)
    assert page.world_map.selected is None and page.gantt.selected_station is None
    assert "hiçbir istasyonun" in page.station_info.text()
    page.select_time(20.)
    assert "Goldstone DSN" in page.station_info.text() and "Canberra DSN" in page.station_info.text()


def test_reload_clears_station_selection_and_missing_geometry_never_implies_contact(mission):
    page, _ = mission
    page._select_station("Canberra DSN")
    result = replace(page.result, metadata={k: v for k, v in page.result.metadata.items() if k != "mission_geometry"})
    page.set_result(result)
    assert page.station_picker.currentData() is None
    assert page.world_map.selected is None and page.gantt.selected_station is None
    assert not page.world_map.active
    page._select_station("Madrid DSN")
    assert "ölçüm zamanları kaydedilmemiş" in page.station_info.text()
    assert "kaydı yok" not in page.station_info.text()  # Missing data is not evidence of no measurement.


def test_narrow_layout_stacks_full_width_plots_and_restores_selection(mission, app):
    page, _ = mission
    page._select_station("Canberra DSN")
    page.resize(740, 1300); app.processEvents()
    assert page.plots.orientation() == Qt.Vertical
    assert page.world_map.width() > 600 and page.orbit.width() > 600
    assert not page.plots.childrenCollapsible()
    page.resize(1240, 920); app.processEvents()
    assert page.plots.orientation() == Qt.Horizontal
    assert page.station_picker.currentData() == "Canberra DSN"
    assert page.world_map.selected == page.gantt.selected_station == "Canberra DSN"


def test_playback_uses_elapsed_time_pauses_at_end_and_on_hide(mission, app):
    page, _ = mission
    with patch("controllers.mission_controller.time.monotonic", return_value=100.):
        page.play_button.setChecked(True)
    with patch("controllers.mission_controller.time.monotonic", return_value=100.005):
        page._tick()
    assert page._position == 1 and page.orbit.selected_t == 20.  # T=28 selects last stored sample.
    page.slider.setValue(3)
    assert not page._timer.isActive()
    with patch("controllers.mission_controller.time.monotonic", return_value=101.):
        page.play_button.setChecked(True)
    with patch("controllers.mission_controller.time.monotonic", return_value=102.):
        page._tick()
    assert page._position == 5 and not page.play_button.isChecked() and not page._timer.isActive()
    page.play_button.setChecked(True)
    assert page._position == 0  # Restart from end.
    page.hide(); app.processEvents()
    assert not page._timer.isActive() and not page.play_button.isChecked()


def test_dense_contacts_reuse_gantt_and_map_background_and_keep_sources(mission, app):
    page, path = mission
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.parent.iterdir() if p.is_file()}
    data = copy.deepcopy(page.result.metadata["mission_geometry"])
    data["windows"] = [{"station_id": i % 2, "arc_id": 7, "start_s": 10+i*.003,
                        "end_s": 10+i*.003+.001, "samples": 1} for i in range(20000)]
    page.set_result(replace(page.result, metadata={**page.result.metadata, "mission_geometry": data}))
    page.gantt.grab(); page.world_map.grab(); page.orbit.grab()
    caches = (page.gantt._base, page.world_map._base, page.orbit._base_image)
    page.select_time(50.)
    page.gantt.grab(); page.world_map.grab(); page.orbit.grab()
    assert all(a is b for a, b in zip(caches, (page.gantt._base, page.world_map._base, page.orbit._base_image)))
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}


def test_legacy_without_geometry_or_states_never_fakes_contacts(app, tmp_path):
    _, _, _, path = native_run(tmp_path)
    result = load_tracking_results(path)
    page = MissionController()
    try:
        page.set_result(replace(result, states=()))
        assert page.gantt.geometry is None and page.world_map.active == set()
        assert not page.play_button.isEnabled() and page.orbit.selected_t is None
        assert "eşlemesi kaydedilmemiş" in page.map_caption.text()
        assert "durum geçmişi yok" in page.orbit_info.text()
    finally:
        page.close()


def test_unopened_and_missing_data_messages_remain_distinct_and_accessible(app, tmp_path):
    from widgets.tracking_plots import StoredOrbit
    from widgets.mission_plots import MissionGantt
    orbit, gantt = StoredOrbit(), MissionGantt()
    try:
        assert "koşu açın" in orbit.accessibleDescription()
        assert "koşu açın" in gantt.accessibleDescription()
        assert "bulunmuyor" not in orbit.accessibleDescription()
        assert "kaydedilmemiş" not in gantt.accessibleDescription()
        _, _, _, path = native_run(tmp_path)
        result = load_tracking_results(path)
        orbit.set_result(replace(result, states=()))
        gantt.set_result(result)
        assert "durum vektörü bulunmuyor" in orbit.accessibleDescription()
        assert "ölçüm zamanları kaydedilmemiş" in gantt.accessibleDescription()
        assert "temas süresi olarak gösterilmez" in gantt.accessibleDescription()
        loaded, _ = load_run_with_details(geometry_run(tmp_path / "complete"))
        orbit.set_result(loaded); gantt.set_result(loaded)
        assert orbit.accessibleDescription() == gantt.accessibleDescription() == ""
        orbit.set_result(None); gantt.set_result(None)
        assert "koşu açın" in orbit.accessibleDescription() and "koşu açın" in gantt.accessibleDescription()
        orbit.grab(); gantt.grab()  # Reset also invalidates previously rendered data.
        assert orbit._points == [] and gantt.geometry is None
    finally:
        orbit.close(); gantt.close()


def test_mission_startup_routing_help_and_restart_revalidate_source(app, tmp_path):
    settings = QSettings(str(tmp_path / "ui.ini"), QSettings.IniFormat)
    path = geometry_run(tmp_path)
    with patch("app.QSettings", return_value=settings):
        window = MainWindow()
    try:
        assert set(window._pages) == {"mission"}
        assert {"stations", "visibility", "ground_track"} <= {p[0] for p in _PRIMARY_NAV}
        assert not window.pageSubLabel.isHidden()
        window.open_in_mission(path)
        page = window._pages["mission"]
        wait_until(app, lambda: page.result is not None)
        assert window._active_page == "mission" and page.gantt.geometry is not None
        assert page.result is window._pages["console"].result
        page.help_button.click()
        assert not page.help_text.isHidden() and "Ay ekvatoru değildir" in page.help_text.text()
        window.open_in_mission(tmp_path / "missing.csv")
        wait_until(app, lambda: not window._pages["console"]._busy)
        assert page.result is not None and "Koşu açılamadı" in page.run_label.text()
    finally:
        window.close()
    with patch("app.QSettings", return_value=settings):
        restored = MainWindow()
    try:
        page = restored._pages["mission"]
        wait_until(app, lambda: page.result is not None)
        assert page.result.sources[0]["path"] == str(path.resolve())
        assert not page._timer.isActive()  # Never autoplay on restore.
    finally:
        restored.close()


def test_catalogue_contains_actual_backend_coordinates():
    from lunar_od.config import range_rate_stations
    catalogue = {s["name"]: s for s in station_catalog()}
    for station in range_rate_stations():
        assert catalogue[station.name]["lat_deg"] == station.lat_deg
        assert catalogue[station.name]["lon_deg"] == station.lon_deg


def test_small_mission_window_keeps_playback_and_help_reachable(app, tmp_path):
    with patch("app.QSettings", return_value=QSettings(str(tmp_path / "compact.ini"), QSettings.IniFormat)):
        window = MainWindow()
    try:
        window.resize(760, 420); window.show(); app.processEvents()
        page = window._pages["mission"]
        assert page.plots.orientation() == Qt.Vertical
        assert window.pageScroll.horizontalScrollBar().maximum() == 0
        for button in (page.play_button, page.help_button, page.archive_button, page.station_picker, page.details_button):
            window.pageScroll.ensureWidgetVisible(button, 0, 0); app.processEvents()
            bounds = QRect(button.mapTo(window.pageScroll.viewport(), QPoint(0, 0)), button.size())
            assert window.pageScroll.viewport().rect().contains(bounds)
    finally:
        window.close()


def test_close_before_event_loop_cancels_startup_check(app, tmp_path):
    from controllers.environment_controller import EnvironmentController
    calls = []
    with patch("app.QSettings", return_value=QSettings(str(tmp_path / "closed.ini"), QSettings.IniFormat)), \
         patch.object(EnvironmentController, "refresh", lambda self: calls.append(True)):
        window = MainWindow()
        assert window._startup_timer.isActive()
        window.close()
        app.processEvents()
        assert calls == [] and not window._startup_timer.isActive()


def test_geometry_stays_hashed_without_expanding_comparison_cells_or_reports(tmp_path):
    from services.run_comparison import load_run_comparison, export_comparison
    from services.comparison_session import load_comparison_session
    paths = [geometry_run(tmp_path / name) for name in ("left", "right")]
    for path in paths:
        manifest = path.with_name("fixture_result_manifest.json")
        info = json.loads(manifest.read_text())
        geometry = path.parent / info["mission_file"]
        data = json.loads(geometry.read_text())
        data["windows"] = [dict(data["windows"][0], arc_id=7+i) for i in range(2000)]
        geometry.write_text(json.dumps(data))
        info["mission_sha256"] = hashlib.sha256(geometry.read_bytes()).hexdigest()
        manifest.write_text(json.dumps(info))
    comparison = load_run_comparison(*paths)
    report, _ = export_comparison(comparison, tmp_path / "comparison.json")
    data = json.loads(report.read_text())
    assert report.stat().st_size < 100000
    assert all(not d["field"].startswith("metadata.mission_geometry") for d in comparison.differences)
    assert "mission_geometry" not in data["A"]["metadata"]
    assert any(s["path"].endswith("_mission_geometry.json") for s in data["A"]["sources"])
    restored, _, _ = load_comparison_session(report)
    assert len(restored.left.metadata["mission_geometry"]["windows"]) == 2000
    assert restored.metrics == comparison.metrics and restored.pairs == comparison.pairs
