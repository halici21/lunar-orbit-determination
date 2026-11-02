"""Archive sources: several run folders, report files and recorded test runs."""
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QApplication

from services import archive_sources
from services.archive_sources import extra_sources, load_test_runs, scan_archive_sources, scan_reports
from tests.test_recorded_run_workspace import write_run


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_runs_from_several_sources_are_listed_with_their_source(tmp_path):
    main, legacy = tmp_path / "workspace", tmp_path / "old" / "results"
    write_run(main / "a")
    write_run(legacy / "b", seed=7)
    entries = scan_archive_sources([main, legacy])
    assert len(entries) == 2
    sources = sorted(e.source for e in entries)
    assert sources == sorted([f"{tmp_path.name}/workspace", "old/results"])


def test_missing_extra_source_is_skipped_not_an_error(tmp_path):
    write_run(tmp_path / "workspace" / "a")
    entries = scan_archive_sources([tmp_path / "workspace", tmp_path / "gone"])
    assert len(entries) == 1


def test_extra_sources_round_trip_through_settings(tmp_path):
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    assert extra_sources(settings) == ()
    settings.setValue(archive_sources.SOURCES_KEY, [str(tmp_path / "x"), str(tmp_path / "y")])
    assert extra_sources(settings) == (tmp_path / "x", tmp_path / "y")
    settings.setValue(archive_sources.SOURCES_KEY, str(tmp_path / "only"))  # QSettings may hand back one string
    assert extra_sources(settings) == (tmp_path / "only",)


def test_reports_list_figures_tables_and_records_but_not_test_runs(tmp_path):
    (tmp_path / "campaign").mkdir()
    (tmp_path / "campaign" / "summary.csv").write_text("a,b\n1,2\n")
    (tmp_path / "campaign" / "plot.png").write_bytes(b"\x89PNG")
    (tmp_path / "notes.md").write_text("# not")
    (tmp_path / "data.bin").write_bytes(b"\0")
    (tmp_path / "test_runs").mkdir()
    (tmp_path / "test_runs" / "local.json").write_text("{}")
    files = scan_reports([tmp_path])
    assert {(f.path.name, f.kind, f.folder) for f in files} == {
        ("summary.csv", "Tablo", "campaign"), ("plot.png", "Grafik", "campaign"), ("notes.md", "Metin", "")}


def test_recorded_test_runs_load_newest_first_and_ignore_foreign_json(tmp_path):
    for name, finished, schema in (("a", "2026-10-24T10:00:00", "lunar-od-test-run-v1"),
                                   ("b", "2026-10-25T10:00:00", "lunar-od-test-run-v1"),
                                   ("c", "2026-10-26T10:00:00", "other")):
        (tmp_path / f"{name}.json").write_text(json.dumps({"schema": schema, "finished_at": finished}))
    (tmp_path / "broken.json").write_text("{")
    runs = load_test_runs(tmp_path)
    assert [r["finished_at"][:10] for r in runs] == ["2026-10-25", "2026-10-24"]
    assert load_test_runs(tmp_path / "missing") == ()


def test_reports_tab_previews_csv_and_json(app, tmp_path):
    from widgets.archive_reports import ArchiveReports
    (tmp_path / "t.csv").write_text("x,y\n1,2\n3,4\n")
    (tmp_path / "r.json").write_text('{"k": [1, 2]}')
    page = ArchiveReports(lambda: [tmp_path])
    try:
        page.model.replace(scan_reports([tmp_path]))
        page.apply_filter()
        rows = {page.model.files[page.proxy.mapToSource(page.proxy.index(r, 0)).row()].path.name: r
                for r in range(page.proxy.rowCount())}
        page.table.setCurrentIndex(page.proxy.index(rows["t.csv"], 0))
        assert page.preview_stack.currentIndex() == 1 and page.grid.rowCount() == 2
        page.table.setCurrentIndex(page.proxy.index(rows["r.json"], 0))
        assert page.preview_stack.currentIndex() == 2 and '"k"' in page.text.toPlainText()
        page.search.setText("t.csv"); page.apply_filter()
        assert page.proxy.rowCount() == 1
    finally:
        page.close()


def test_test_runs_tab_summarises_the_latest_local_run(app, tmp_path):
    from widgets.archive_test_runs import ArchiveTestRuns
    page = ArchiveTestRuns()
    try:
        runs = ({"schema": "lunar-od-test-run-v1", "source": "local", "finished_at": "2026-10-25T10:00:00",
                 "passed": 10, "failed": 1, "errors": 0, "skipped": 2, "exit_status": 1,
                 "args": ["tests/test_x.py"], "failures": ["tests/test_x.py::test_a"]},)
        page._job_ready(page._generation, runs)
        assert "Başarısız" in page.summary.text() and "10 geçti" in page.summary.text()
        page.table.setCurrentIndex(page.model.index(0, 0))
        assert "tests/test_x.py::test_a" in page.detail.toPlainText()
    finally:
        page.close()
