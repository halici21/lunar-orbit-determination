import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from types import SimpleNamespace
from dataclasses import replace
from PyQt5.QtWidgets import QApplication
from services.result_summary import result_summary, comparison_summary
from services.run_comparison import load_run_comparison
from services.run_details import load_run_with_details
from controllers.tracking_console_controller import TrackingConsoleController
from controllers.recorded_comparison_controller import RecordedComparisonController
from tests.test_recorded_run_workspace import write_run


def test_success_does_not_imply_convergence_and_missing_is_not_zero():
    rows = [dict(position_error_m=0., velocity_error_mps=0., operational_success=True, stop_reason="MaxIter", normalized_nis=None),
            dict(position_error_m=2., velocity_error_mps=.2, operational_success=None, normalized_nis=None)]
    source = SimpleNamespace(updates=rows, arcs=[], states=[], metadata={"record_kind":"arc_solution"})
    headline, errors, diagnostics = result_summary(source, {"status":"İşlem tamamlandı"})
    assert "1/1" in headline and "1 kayıtta başarı alanı yok" in headline
    assert "Konum hata medyanı 1 m" in errors
    assert "MaxIter" in diagnostics and "ayrı bir doğrulama alanı yok" in diagnostics
    assert "Normalize NIS: kaynakta yok" in diagnostics
    assert "Referans yörünge: yok" in diagnostics
    assert "Çalışma süresi: doğrulanmış kayıt yok" in diagnostics
    rows[0]["normalized_nis"] = 0
    assert "1/2 kayıtta mevcut" in result_summary(source, {})[2]


def test_comparison_delta_direction_zero_baseline_and_incompatible_metadata(tmp_path):
    rows_a = [[1,1,10,1,0,0,1],[2,1,20,1,0,0,1]]
    rows_b = [[1,1,10,1,2,.2,1],[2,1,20,1,4,.4,1]]
    a = write_run(tmp_path/"a", rows=rows_a)
    b = write_run(tmp_path/"b", rows=rows_b, seed=43)
    comparison = load_run_comparison(a,b)
    text = comparison_summary(comparison)
    assert "measurement_seed: 42 → 43" in text
    assert "Konum +3 m" in text and "A=0; yüzde tanımsız" in text
    assert "Çalışma süresi farkı: doğrulanmış süre kaydı yok" in text
    disabled = comparison_summary(replace(comparison,time_compatible=False,pairs=()))
    assert "karşılaştırılamıyor" in disabled and "Konum +3" not in disabled
    b = write_run(tmp_path/"b", rows=rows_b, nis_scope="different")
    assert "kapsam eksik/farklı" in comparison_summary(load_run_comparison(a,b))


def test_summary_is_visible_without_details_and_comparison_survives_export(tmp_path):
    app = QApplication.instance() or QApplication([])
    a = write_run(tmp_path/"a")
    b = write_run(tmp_path/"b", seed=43)
    result, details = load_run_with_details(a)
    console = TrackingConsoleController()
    console.resize(1300,900); console.show()
    console._accept_result(result, details)
    app.processEvents()
    assert console.result_overview.isVisible()
    assert not console.inspector_tabs.isVisible()
    assert "İşlem tamamlandı" in console.result_overview.text()
    assert "başarı bilinmiyor" in console.result_overview.text()
    comparison = load_run_comparison(a,b)
    page = RecordedComparisonController()
    page.resize(1300,900); page.show()
    page.accept_comparison(comparison)
    assert page.effect_summary.isVisible()
    assert "42 → 43" in page.effect_summary.text()
    from services.run_comparison import export_comparison
    from services.comparison_session import load_comparison_session
    report = tmp_path/"comparison.json"
    export_comparison(comparison, report)
    restored = load_comparison_session(report)
    # Loader returns comparison plus view state; use the same summary path.
    assert comparison_summary(restored[0]) == comparison_summary(comparison)
    console.shutdown(); console.close()
    page.shutdown(); page.close()
