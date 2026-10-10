"""Archive identities and descriptive comparison contracts, no physics campaign."""
from __future__ import annotations

import csv
import json
import math
import hashlib
import os
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from PyQt5.QtCore import Qt, QSettings, QTimer, QItemSelectionModel
from PyQt5.QtWidgets import QApplication

from services.tracking_results import load_tracking_results, LoadCancelled, ResultLoadError
from services.run_archive import scan_run_archive
from services.run_comparison import load_run_comparison, compare_results, describe, export_comparison, _snapshot
from controllers.run_archive_controller import RunArchiveController
from controllers.recorded_comparison_controller import RecordedComparisonController


def write_run(folder, *, seed=42, completed=True, rows=None, kind="ukf_update", **metadata):
    folder.mkdir(parents=True, exist_ok=True)
    suffix = "_arc_solutions.csv" if kind == "arc_solution" else "_ukf_updates.csv"
    path = folder / ("sample" + suffix)
    fields = ["update_id", "arc_id", "t_s", "station_id", "position_error_m", "velocity_error_mps", "normalized_nis"]
    rows = rows if rows is not None else [[1, 1, 10, 1, 1., .1, 1.], [2, 1, 20, 1, 2., .2, 2.],
        [3, 2, 40, 1, 3., .3, 3.], [4, 2, 50, 1, 4., .4, 4.]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)
    manifest = {"schema": "lunar-od-result-v1", "units": {"time": "s", "position": "m", "velocity": "m/s"},
        "frame": "J2000", "origin": "Moon", "epoch": "2027-03-02 23:32:37", "time_scale": "TDB elapsed seconds",
        "record_kind": kind, "seed": seed, "station_id_convention": "backend_0_based", "nis_scope": "update",
        "job_file": "job.json", "config_file": "config.json", "estimator_type": "ukf"}
    manifest.update(metadata)
    (folder / "sample_result_manifest.json").write_text(json.dumps(manifest))
    (folder / "config.json").write_text(json.dumps({"name": "sample", "estimator_type": "ukf", "noise": True,
        "network": "multi", "measurement_type": "position", "max_iter": 3, "output_dir": str(folder)}))
    (folder / "job.json").write_text(json.dumps({"run_id": folder.name, "created_at": "2026-10-08T10:00:00+00:00",
        "options": {"measurement_seed": seed, "cold_start_seed": seed}}))
    if completed:
        (folder / "completed.json").write_text(json.dumps({"schema": "lunar-od-completed-job-v1",
            "run_id": folder.name, "result_path": str(path.resolve())}))
    return path


def wait(app, condition, timeout=5):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    app.processEvents()
    assert condition(), "Qt archive/comparison timed out"


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_archive_distinguishes_completed_partial_legacy_and_broken(tmp_path):
    completed = write_run(tmp_path / "complete")
    partial = write_run(tmp_path / "partial", completed=False)
    legacy = write_run(tmp_path / "legacy")
    legacy.with_name("sample_result_manifest.json").unlink()
    broken = write_run(tmp_path / "broken")
    broken.write_text("bad,headers\n1,2")
    rows = {entry.path: entry for entry in scan_run_archive(tmp_path)}
    assert rows[completed].status == "completed"
    assert rows[partial].status == "incomplete"
    assert any("Tamamlanma kaydı yok" in n for n in rows[partial].notes)
    assert rows[legacy].status == "legacy"
    assert rows[broken].status == "invalid"
    assert rows[completed].records == 4
    assert rows[completed].run_id != rows[partial].run_id
    assert rows[completed].name == rows[partial].name  # identity is path/UUID, not human label


def test_archive_batch_incomplete_is_explicit(tmp_path):
    write_run(tmp_path, kind="arc_solution", completed=False)
    entry, = scan_run_archive(tmp_path)
    assert entry.status == "incomplete"


def test_archive_mismatched_completion_is_not_accepted(tmp_path):
    write_run(tmp_path)
    (tmp_path / "completed.json").write_text('{"schema":"lunar-od-completed-job-v1","run_id":"other","result_path":"sample_ukf_updates.csv"}')
    assert scan_run_archive(tmp_path)[0].status == "invalid"


def test_archive_accepts_moved_folder_with_explicit_note(tmp_path):
    write_run(tmp_path)
    record = json.loads((tmp_path / "completed.json").read_text())
    record["result_path"] = str(tmp_path.parent / "old" / "sample_ukf_updates.csv")
    (tmp_path / "completed.json").write_text(json.dumps(record))
    entry, = scan_run_archive(tmp_path)
    assert entry.status == "completed"
    assert any("taşınmış" in n for n in entry.notes)


def test_archive_contains_numeric_bad_rows_but_loader_rejects_them(tmp_path):
    path = write_run(tmp_path, rows=[[1, 1, 10, 1, "NaN", .1, 1]])
    assert scan_run_archive(tmp_path)[0].status == "completed"
    with pytest.raises(ResultLoadError, match="position_error_m"):
        load_tracking_results(path)


def test_archive_cancellation_and_missing_folder(tmp_path):
    write_run(tmp_path)
    with pytest.raises(LoadCancelled):
        scan_run_archive(tmp_path, cancel=lambda: True)
    with pytest.raises(ResultLoadError):
        scan_run_archive(tmp_path / "not-there")


def test_statistics_match_independent_known_values_and_do_not_overflow():
    actual = describe([1., 2., 3., 4.])
    assert actual == {"N": 4, "median": 2.5, "p95": pytest.approx(3.85), "max": 4., "rms": pytest.approx(math.sqrt(7.5))}
    assert describe([])["rms"] is None
    assert describe([0, 0])["rms"] == 0
    assert math.isfinite(describe([1e308, 1e308])["rms"])


def test_exact_pairs_do_not_interpolate_and_stats_show_distinct_pools(tmp_path):
    a = write_run(tmp_path / "A")
    b = write_run(tmp_path / "B", rows=[[8, 1, 10, 1, 10., 1., 1.], [9, 1, 21, 1, 200., 2., 1.],
        [10, 2, 40, 2, 300., 3., 1.], [11, 2, 50, 1, 40., 4., 1.]])
    result = load_run_comparison(a, b)
    assert result.pairs == ((0, 0), (3, 3))
    assert result.metrics["all"]["position_error_m"]["B"]["N"] == 4
    assert result.metrics["paired"]["position_error_m"]["B"]["N"] == 2
    assert result.metrics["paired"]["position_error_m"]["B"]["median"] == 25
    assert any("küçük örneklem" in note for note in result.notes)


@pytest.mark.parametrize("change", [{"epoch": "different"}, {"frame": "OTHER"}, {"origin": "Earth"},
    {"time_scale": "UTC"}, {"epoch": ""}, {"station_id_convention": "1_based"}])
def test_incompatible_reference_disables_deltas(tmp_path, change):
    a, b = write_run(tmp_path / "A"), write_run(tmp_path / "B", **change)
    result = load_run_comparison(a, b)
    assert not result.time_compatible
    assert result.pairs == ()
    assert result.metrics["paired"]["position_error_m"]["A"]["median"] is None
    assert result.metrics["all"]["position_error_m"]["A"]["N"] == 4


def test_unmanifested_legacy_stays_visible_without_paired_deltas(tmp_path):
    a, b = write_run(tmp_path / "A"), write_run(tmp_path / "B")
    (b.parent / "sample_result_manifest.json").unlink()
    result = load_run_comparison(a, b)
    assert result.pairs == ()
    assert any("bilinmiyor" in n for n in result.notes)


def test_same_time_different_record_kinds_cannot_pair(tmp_path):
    left = load_tracking_results(write_run(tmp_path))
    right = replace(left, metadata={**left.metadata, "record_kind": "arc_solution"})
    assert not compare_results(left, right).time_compatible


def test_duplicate_epoch_key_is_excluded_not_arbitrarily_selected(tmp_path):
    a = write_run(tmp_path / "A", rows=[[1, 1, 10, 1, 1, .1, 1], [2, 1, 10, 1, 2, .2, 2]])
    b = write_run(tmp_path / "B")
    result = load_run_comparison(a, b)
    assert result.ambiguous == (2, 0)
    assert result.pairs == ()


def test_controlled_variable_does_not_hide_other_settings_or_seed(tmp_path):
    a, b = write_run(tmp_path / "A", seed=0), write_run(tmp_path / "B", seed=43)
    config = json.loads((b.parent / "config.json").read_text())
    config.update(max_iter=10, noise=False)
    (b.parent / "config.json").write_text(json.dumps(config))
    diffs = {d["field"]: d for d in load_run_comparison(a, b, "max_iter").differences}
    assert diffs["config.max_iter"]["role"] == "Seçilen değişken"
    assert diffs["config.noise"]["B"] is False
    assert diffs["options.measurement_seed"]["A"] == 0
    assert diffs["metadata.seed"]["B"] == 43
    assert "config.output_dir" not in diffs


def test_snapshot_rejects_source_changed_after_load(tmp_path):
    result = load_tracking_results(write_run(tmp_path))
    (tmp_path / "config.json").write_text('{"max_iter":999}')
    with pytest.raises(ValueError, match="değişti"):
        _snapshot(result)


def test_csv_change_between_parse_and_hash_is_rejected(tmp_path):
    import services.tracking_results as tr
    path = write_run(tmp_path)
    original = tr._source
    def mutate(source, cancel):
        if source == path:
            path.write_text(path.read_text().replace("1.0,", "12345.0,", 1))
        return original(source, cancel)
    with patch.object(tr, "_source", mutate), pytest.raises(ResultLoadError, match="değişti"):
        load_tracking_results(path)


def test_persisted_job_hash_mismatch_is_rejected(tmp_path):
    path = write_run(tmp_path)
    manifest = path.with_name("sample_result_manifest.json")
    metadata = json.loads(manifest.read_text())
    metadata["job_sha256"] = hashlib.sha256((tmp_path / "job.json").read_bytes()).hexdigest()
    manifest.write_text(json.dumps(metadata))
    (tmp_path / "job.json").write_text('{"options":{"measurement_seed":999}}')
    with pytest.raises(ResultLoadError, match="SHA-256"):
        load_tracking_results(path)


def test_export_roundtrip_and_source_overwrite_protection(tmp_path):
    a, b = write_run(tmp_path / "A"), write_run(tmp_path / "B")
    result = load_run_comparison(a, b)
    report, paired = export_comparison(result, tmp_path / "report.json")
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["paired_N"] == 4
    assert saved["A"]["sources"][0]["sha256"] == result.left.sources[0]["sha256"]
    with paired.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert float(rows[0]["delta_position_error_m_B_minus_A"]) == 0
    with pytest.raises(OSError, match="Kaynak"):
        export_comparison(result, a.parent / "config.json")


def test_export_zero_pairs_and_cancel_do_not_leave_false_csv(tmp_path):
    a, b = write_run(tmp_path / "A"), write_run(tmp_path / "B", epoch="different")
    result = load_run_comparison(a, b)
    report, paired = export_comparison(result, tmp_path / "report.json")
    assert paired is None
    assert not (tmp_path / "report_paired.csv").exists()
    (tmp_path / "report_paired.csv").write_text("older")
    with pytest.raises(OSError, match="eski"):
        export_comparison(result, report)
    before = report.read_bytes()
    with pytest.raises(LoadCancelled):
        export_comparison(result, tmp_path / "other.json", cancel=lambda: True)
    assert report.read_bytes() == before
    assert not (tmp_path / "other.json").exists()


def test_same_source_and_cancel_are_rejected(tmp_path):
    a, b = write_run(tmp_path / "A"), write_run(tmp_path / "B")
    with pytest.raises(ValueError, match="farklı"):
        load_run_comparison(a, a)
    with pytest.raises(LoadCancelled):
        load_run_comparison(a, b, cancel=lambda: True)


def test_archive_filters_and_sort_keep_actual_source_identity(app, tmp_path):
    a, b = write_run(tmp_path / "A", seed=0), write_run(tmp_path / "B", seed=99)
    widget = RunArchiveController()
    try:
        widget.set_folder(tmp_path)
        wait(app, lambda: widget.model.rowCount() == 2)
        widget.search.setText("99")
        widget.apply_filter()
        assert widget.proxy.rowCount() == 1
        widget.table.selectRow(0)
        assert widget.selected_entries()[0].path == b
        assert widget.open_button.isEnabled()
        widget.search.clear()
        widget.apply_filter()
        widget.table.sortByColumn(9, Qt.DescendingOrder)
        widget.table.selectRow(0)
        assert widget.selected_entries()[0].path == b
        generation = widget._generation
        widget._job_ready(generation - 1, ())
        assert widget.model.rowCount() == 2
    finally:
        widget.close()


def test_comparison_async_load_pair_links_errors_and_export(app, tmp_path):
    a, b = write_run(tmp_path / "A"), write_run(tmp_path / "B")
    widget = RecordedComparisonController()
    try:
        widget.resize(1400, 1050)
        widget.show()
        widget.set_files(a, b)
        wait(app, lambda: widget.comparison is not None)
        saved = widget.comparison
        assert widget.pair_model.rowCount() == 4
        widget._chart_selected(1, 2)
        assert all(c.selected_pair == 2 for c in widget.charts)
        assert "A kayıt 3 / B kayıt 3" in widget.selection_label.text()
        widget.set_range(10, 20)
        assert widget.charts[0].time_range == (10, 20)
        widget.paired_check.setChecked(True)
        app.processEvents()
        widget._job_ready(widget._generation - 1, None)
        assert widget.comparison is saved
        widget.set_files(a, tmp_path / "bad.csv")
        wait(app, lambda: "başarısız" in widget.status_label.text())
        assert widget.comparison is saved
        assert widget.export_button.isEnabled()
        widget.export_to(tmp_path / "ui_report.json")
        wait(app, lambda: "Rapor kaydedildi" in widget.status_label.text())
        assert json.loads((tmp_path / "ui_report.json").read_text(encoding="utf-8"))["paired_N"] == 4
        widget.cancel_load()
        assert widget.comparison is saved
    finally:
        widget.close()


def test_unknown_reference_separates_chart_axes_and_nis_scope(app, tmp_path):
    a, b = write_run(tmp_path / "A"), write_run(tmp_path / "B", nis_scope="arc_mean")
    widget = RecordedComparisonController()
    try:
        widget.accept_comparison(load_run_comparison(a, b))
        assert widget.charts[0].shared_axis()
        assert not widget.charts[2].shared_axis()
        (b.parent / "sample_result_manifest.json").unlink()
        widget.accept_comparison(load_run_comparison(a, b))
        assert all(not c.shared_axis() for c in widget.charts)
        assert not widget.paired_check.isEnabled()
        assert widget.pair_model.rowCount() == 0
    finally:
        widget.close()


@pytest.mark.parametrize("value", [5e-324, 1.79e308])
def test_chart_accepts_extreme_finite_values_without_paint_exception(app, tmp_path, value):
    from widgets.comparison_chart import ComparisonChart
    path = write_run(tmp_path, rows=[[1, 1, 10, 1, value, .1, value]])
    result = load_tracking_results(path)
    chart = ComparisonChart("position_error_m", "Error", "m")
    try:
        chart.set_comparison(compare_results(result, result))
        chart.resize(600, 200)
        assert not chart.grab().isNull()
        chart.linear = True
        assert not chart.grab().isNull()
    finally:
        chart.close()


def test_unknown_batch_estimator_and_operational_status_stay_unknown(app, tmp_path):
    result = load_tracking_results(write_run(tmp_path))
    result = replace(result, metadata={**result.metadata, "record_kind": "arc_solution", "estimator_type": None})
    widget = RecordedComparisonController()
    try:
        widget.accept_comparison(compare_results(result, result))
        assert "BİLİNMİYOR" in widget.summary_labels[0].text() or "BILINMIYOR" in widget.summary_labels[0].text()
        assert "Operasyonel başarı bilinmiyor" in widget.summary_labels[0].text()
    finally:
        widget.close()


def test_large_source_loading_keeps_qt_ticking(app, tmp_path):
    rows = [[i + 1, 1, i, 1, 1 + i % 100, .1, 1.] for i in range(20000)]
    a, b = write_run(tmp_path / "A", rows=rows), write_run(tmp_path / "B", rows=rows)
    widget = RecordedComparisonController()
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(2)
    try:
        widget.set_files(a, b)
        wait(app, lambda: widget.comparison is not None, timeout=15)
        assert widget.pair_model.rowCount() == 20000
        assert len(ticks) >= 5
    finally:
        timer.stop()
        widget.close()


def test_native_navigation_archive_to_comparison_to_console(app, tmp_path):
    from app import MainWindow
    import controllers.run_archive_controller as archive_module
    a, b = write_run(tmp_path / "A"), write_run(tmp_path / "B")
    isolated = QSettings(str(tmp_path / "test-settings.ini"), QSettings.IniFormat)
    with patch("app.QSettings", return_value=isolated), patch.object(archive_module, "RESULTS_DIR", tmp_path):
        window = MainWindow()
        assert set(window._pages) == {"console"}
        window.navigate("archive")
    try:
        archive = window._pages["archive"]
        wait(app, lambda: archive.model.rowCount() == 2)
        for row in range(2):
            archive.table.selectionModel().select(archive.proxy.index(row, 0), QItemSelectionModel.Select | QItemSelectionModel.Rows)
        assert archive.compare_button.isEnabled()
        archive.compare_selected()
        comparison = window._pages["comparison"]
        wait(app, lambda: comparison.comparison is not None)
        assert window.pageStack.currentWidget() is comparison
        assert "border-left:2px solid transparent" in window._nav_btns["archive"].styleSheet()
        assert {Path(comparison.comparison.left.sources[0]["path"]), Path(comparison.comparison.right.sources[0]["path"])} == {a, b}
        comparison.open_source(0)
        console = window._pages["console"]
        wait(app, lambda: console.result is not None)
        assert console.result.sources[0]["path"] == comparison.comparison.left.sources[0]["path"]
        assert window.pageStack.currentWidget() is console
    finally:
        window.close()


def test_new_workspace_missing_archive_is_empty_without_creating_folders(tmp_path):
    folder = tmp_path / "first-run-results"
    assert scan_run_archive(folder, missing_ok=True) == ()
    assert not folder.exists()
    with pytest.raises(ResultLoadError):
        scan_run_archive(folder)
    with pytest.raises(LoadCancelled):
        scan_run_archive(folder, cancel=lambda: True, missing_ok=True)


def test_default_empty_archive_guides_user_and_custom_missing_folder_stays_an_error(
    app, tmp_path, monkeypatch,
):
    import controllers.run_archive_controller as module

    default = tmp_path / "not-created-yet"
    monkeypatch.setattr(module, "RESULTS_DIR", default)
    page = RunArchiveController()
    try:
        page.on_shown()
        wait(app, lambda: not page._jobs)
        assert page.model.rowCount() == 0
        assert "Henüz kayıtlı koşu yok" in page.status_label.text()
        assert not page.open_button.isEnabled()
        assert not page.compare_button.isEnabled()
        assert not default.exists()

        source = write_run(tmp_path / "previous-runs")
        page.set_folder(source.parent)
        wait(app, lambda: not page._jobs)
        assert page.model.rowCount() == 1

        page.set_folder(tmp_path / "missing-custom-folder")
        wait(app, lambda: not page._jobs)
        assert "Arşiv okunamadı" in page.status_label.text()
        assert page.model.entries[0].path == source
    finally:
        page.shutdown()
        page.close()
