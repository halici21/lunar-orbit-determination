"""Reopen verified comparison workspaces and derive new jobs from native runs."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QApplication

from models.scenario_model import ScenarioModel
from services.archived_scenario import load_archived_scenario, form_from_mapping
from services.scenario_execution import prepare_desktop_job
from services.comparison_session import load_comparison_session
from services.run_comparison import load_run_comparison, export_comparison
from services.tracking_results import ResultLoadError, LoadCancelled
from controllers.scenario_builder_controller import ScenarioBuilderController
from controllers.recorded_comparison_controller import RecordedComparisonController
from tests.test_recorded_run_workspace import write_run, wait
from tests.test_desktop_workflow import bundle


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def native_run(tmp_path):
    from lunar_od import scenario_config_from_mapping
    job, folder, run_id, result = bundle(tmp_path)
    saved = json.loads(job.read_text())
    (folder / "effective_config.json").write_text(json.dumps(scenario_config_from_mapping(saved["config"]).to_dict()))
    return job, folder, run_id, result


def rewrite_job(job, change):
    saved = json.loads(job.read_text())
    change(saved)
    job.write_text(json.dumps(saved))
    manifest = next(job.parent.glob("*_result_manifest.json"))
    data = json.loads(manifest.read_text())
    data["job_sha256"] = hashlib.sha256(job.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(data))


def test_archived_form_and_derived_job_preserve_origin_and_record_changes(tmp_path):
    job, folder, run_id, result = native_run(tmp_path)
    original_bytes = job.read_bytes()
    archived = load_archived_scenario(result)
    form = json.loads(original_bytes)["form"]
    recovered = asdict(archived.model)
    assert recovered == {**form, "name": form["name"] + "_copy"}
    assert archived.origin["run_id"] == run_id
    derived = replace(archived.model, bls_max_iter=17)
    new_job, new_folder, new_id = prepare_desktop_job(derived, tmp_path, derived_from=archived.origin)
    saved = json.loads(new_job.read_text())
    assert new_id != run_id and new_folder != folder
    assert job.read_bytes() == original_bytes
    assert saved["derived_from"]["result_path"] == str(result)
    changes = {d["field"]: d for d in saved["derived_from"]["form_changes"]}
    assert set(changes) == {"name", "bls_max_iter"}
    assert changes["bls_max_iter"] == {"field": "bls_max_iter", "before": form["bls_max_iter"], "after": 17}
    assert saved["config"]["max_iter"] == 17
    assert saved["options"]["measurement_seed"] == form["random_seed"]


@pytest.mark.parametrize("change", [
    lambda f: f.pop("ephemeris_step_s"), lambda f: f.update(unknown_setting=1),
    lambda f: f.update(noise_enabled="false"), lambda f: f.update(random_seed=True),
    lambda f: f.update(state_step_s=float("nan")), lambda f: f.update(bls_max_iter="4"),
])
def test_unrepresentable_form_is_rejected_instead_of_guessing(change):
    form = asdict(ScenarioModel())
    change(form)
    with pytest.raises(ResultLoadError):
        form_from_mapping(form)


def test_archived_inconsistent_form_and_job_are_rejected(tmp_path):
    job, _, _, result = native_run(tmp_path)
    rewrite_job(job, lambda saved: saved["form"].update(random_seed=99))
    with pytest.raises(ResultLoadError, match="uyuşmuyor"):
        load_archived_scenario(result)


def test_archived_effective_settings_that_form_cannot_recreate_are_rejected(tmp_path):
    _, folder, _, result = native_run(tmp_path)
    config = folder / "effective_config.json"
    data = json.loads(config.read_text())
    data["j2_moon"] = .0001
    config.write_text(json.dumps(data))
    with pytest.raises(ResultLoadError, match="j2_moon"):
        load_archived_scenario(result)


def test_legacy_cannot_become_invented_scenario_and_loading_can_cancel(tmp_path):
    path = write_run(tmp_path)
    path.with_name("sample_result_manifest.json").unlink()
    with pytest.raises(ResultLoadError, match="manifesti yok"):
        load_archived_scenario(path)
    with pytest.raises(LoadCancelled):
        load_archived_scenario(path, cancel=lambda: True)


def test_form_roundtrip_preserves_uint32_precision_and_extended_ranges(app):
    builder = ScenarioBuilderController()
    try:
        model = ScenarioModel(random_seed=2**32 - 1, duration_days=.0012345678912345,
            state_step_s=.125, arc_duration_h=.1234567890123, arc_stride_h=.25,
            elevation_mask_deg=75.123456, gap_threshold_s=7200.125, ukf_alpha=.87654)
        builder._model_to_widgets(model)
        builder._update_json()
        assert asdict(builder._widgets_to_model()) == asdict(model)
        assert json.loads(builder.jsonPreview.toPlainText())["random_seed"] == 2**32 - 1
    finally:
        builder.close()


def test_clone_can_restore_previous_draft_and_preset_clears_lineage(app, tmp_path):
    *_, path = native_run(tmp_path)
    archived = load_archived_scenario(path)
    builder = ScenarioBuilderController()
    try:
        original = ScenarioModel(name="draft_to_keep", random_seed=123)
        builder._model_to_widgets(original)
        builder.load_archived(archived)
        assert builder._archive_origin["run_id"] == archived.origin["run_id"]
        assert builder._widgets_to_model().name.endswith("_copy")
        builder.restore_previous_draft()
        assert asdict(builder._widgets_to_model()) == asdict(original)
        assert builder._archive_origin is None
        builder.load_archived(archived)
        builder._on_preset_selected("Gaussian Baseline")
        assert builder._archive_origin is None
    finally:
        builder.close()


def comparison_report(folder, view=None):
    a, b = write_run(folder / "A"), write_run(folder / "B")
    result = load_run_comparison(a, b, "max_iter")
    path, _ = export_comparison(result, folder / "comparison.json", view=view)
    return path, result


def test_saved_metrics_are_recomputed_from_verified_sources(tmp_path):
    path, original = comparison_report(tmp_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    report["metrics"]["all"]["position_error_m"]["A"]["median"] = 999999
    path.write_text(json.dumps(report), encoding="utf-8")
    result, view, _ = load_comparison_session(path)
    assert result.metrics == original.metrics
    assert result.pairs == original.pairs
    assert result.variable == "max_iter"
    assert view == {}


def test_report_and_sources_move_together_without_losing_identity(tmp_path):
    folder = tmp_path / "original"
    path, original = comparison_report(folder)
    moved = tmp_path / "moved"
    assert folder.resolve().parent == moved.resolve().parent == tmp_path.resolve()
    folder.rename(moved)
    result, _, _ = load_comparison_session(moved / path.name)
    assert result.left.sources[0]["sha256"] == original.left.sources[0]["sha256"]
    assert Path(result.left.sources[0]["path"]).parent == moved / "A"
    assert result.metrics == original.metrics


def test_changed_or_added_sidecars_cannot_silently_reopen(tmp_path):
    path, original = comparison_report(tmp_path)
    config = tmp_path / "A" / "config.json"
    config.write_text('{"name":"changed"}')
    with pytest.raises(ResultLoadError, match="SHA-256"):
        load_comparison_session(path)


def test_missing_sources_and_cancel_return_explicit_errors(tmp_path):
    path, original = comparison_report(tmp_path)
    Path(original.left.sources[0]["path"]).unlink()
    with pytest.raises(ResultLoadError, match="kaynak bulunamadı"):
        load_comparison_session(path)
    with pytest.raises(LoadCancelled):
        load_comparison_session(path, cancel=lambda: True)


@pytest.mark.parametrize("view", [
    {"schema": "wrong"}, {"schema": "lunar-od-comparison-view-v1", "linear": "yes"},
    {"schema": "lunar-od-comparison-view-v1", "time_range": [10, 0]},
    {"schema": "lunar-od-comparison-view-v1", "time_range": [0, float("inf")]},
    {"schema": "lunar-od-comparison-view-v1", "selected_records": [True, 1]},
    {"schema": "lunar-od-comparison-view-v1", "tab": "something"},
])
def test_bad_view_does_not_replace_existing_export(tmp_path, view):
    path, comparison = comparison_report(tmp_path)
    before = path.read_bytes()
    with pytest.raises(ResultLoadError):
        export_comparison(comparison, path, view=view)
    assert path.read_bytes() == before


def test_comparison_ui_restores_view_and_keeps_it_after_bad_reopen(app, tmp_path):
    path, comparison = comparison_report(tmp_path)
    widget = RecordedComparisonController()
    try:
        widget.accept_comparison(comparison)
        widget.linear_check.setChecked(True)
        widget.paired_check.setChecked(True)
        widget._chart_selected(0, 2)
        widget.set_range(11, 49)
        widget.tabs.setCurrentIndex(3)
        view = widget.capture_view()
        widget.export_to(path)
        wait(app, lambda: "Rapor kaydedildi" in widget.status_label.text())
        widget.linear_check.setChecked(False)
        widget.paired_check.setChecked(False)
        widget.tabs.setCurrentIndex(0)
        widget.open_report(path)
        wait(app, lambda: "geri açıldı" in widget.status_label.text())
        assert widget.capture_view() == view
        saved = widget.comparison
        source = Path(comparison.left.sources[0]["path"])
        source.write_text(source.read_text().replace("1.0,", "99.0,", 1))
        widget.open_report(path)
        wait(app, lambda: "başarısız" in widget.status_label.text())
        assert widget.comparison is saved
        assert widget.capture_view() == view
    finally:
        widget.close()


def test_native_archive_clone_creates_draft_without_starting_solver(app, tmp_path):
    from app import MainWindow
    import controllers.run_archive_controller as module
    *_, result = native_run(tmp_path)
    settings = QSettings(str(tmp_path / "isolated.ini"), QSettings.IniFormat)
    with patch("app.QSettings", return_value=settings), patch.object(module, "RESULTS_DIR", tmp_path):
        window = MainWindow()
        window.navigate("archive")
    try:
        archive = window._pages["archive"]
        wait(app, lambda: archive.model.rowCount() == 1)
        archive.table.selectRow(0)
        assert archive.clone_button.isEnabled()
        archive.clone_selected()
        wait(app, lambda: "scenario" in window._pages and window.pageStack.currentWidget() is window._pages["scenario"])
        builder = window._pages["scenario"]
        assert builder._archive_origin["result_path"] == str(result)
        assert builder._widgets_to_model().name == "fixture_copy"
        assert "run_monitor" not in window._pages
    finally:
        window.close()


def test_old_absolute_report_and_unknown_selection(tmp_path):
    path, _ = comparison_report(tmp_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    for side in ("A", "B"):
        for source in report[side]["sources"]:
            source.pop("relative_path", None)
    path.write_text(json.dumps(report), encoding="utf-8")
    result, _, _ = load_comparison_session(path)
    assert len(result.pairs) == 4
    report["view"] = {"schema": "lunar-od-comparison-view-v1", "selected_records": [9999, 9999]}
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ResultLoadError, match="seçim"):
        load_comparison_session(path)


def test_archive_exposes_parent_and_changed_fields(app, tmp_path):
    from services.run_archive import scan_run_archive
    from controllers.run_archive_controller import RunArchiveController
    job, _, _, result = native_run(tmp_path)
    rewrite_job(job, lambda saved: saved.update(derived_from={
        "schema": "lunar-od-derived-run-v1", "run_id": "parent-run",
        "form_changes": [{"field": "random_seed", "before": 42, "after": 43}]}))
    entries = scan_run_archive(tmp_path)
    assert entries[0].derived_from["run_id"] == "parent-run"
    widget = RunArchiveController()
    try:
        widget.set_folder(tmp_path)
        wait(app, lambda: widget.model.rowCount() == 1)
        widget.search.setText("parent-run")
        widget.apply_filter()
        assert widget.proxy.rowCount() == 1
        widget.table.selectRow(0)
        assert "parent-run" in widget.lineage_label.text()
        assert "seed" in widget.lineage_label.text()
        assert widget.selected_entries()[0].path == result
    finally:
        widget.close()
