"""Persist view preferences only; reopen and validate sources through normal workers."""
import json
import math
from pathlib import Path
from services.comparison_session import validate_view

KEY = "workspace/session-v1"


def identities(result):
    return sorted((str(Path(s["path"]).resolve()), s["sha256"]) for s in result.sources)


def same_sources(result, saved):
    return [list(pair) for pair in identities(result)] == saved


def read_session(settings):
    try:
        raw = settings.value(KEY, "")
        if not isinstance(raw, str) or len(raw) > 512000:
            return {}
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get("schema") != 1:
            return {}
        return data
    except (ValueError, TypeError):
        return {}


def capture_console(page):
    row = page.result.updates[page.selected_index] if page.selected_index is not None else None
    return {"path": page.result.sources[0]["path"], "sources": identities(page.result),
        "view": {"station": page.station_combo.currentData(), "arc": page.arc_combo.currentData(),
            "hours": [page.start_hours.value(), page.end_hours.value()],
            "accepted": page.accepted_check.isChecked(), "linear": page.charts[0].linear,
            "selected": row["update_id"] if row else None, "tab": page.view_tabs.currentIndex(),
            "details": page.details_toggle.isChecked(), "filters": page.filters_toggle.isChecked()}}


def restore_console_view(page, view):
    if not isinstance(view, dict):
        return
    hours = view.get("hours")
    if not isinstance(hours, list) or len(hours) != 2 or not all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in hours) or hours[0] > hours[1]:
        return
    for combo, key in ((page.station_combo,"station"), (page.arc_combo,"arc")):
        index = combo.findData(view.get(key))
        combo.blockSignals(True)
        combo.setCurrentIndex(max(0,index))
        combo.blockSignals(False)
    for spin, value in zip((page.start_hours,page.end_hours),hours):
        spin.blockSignals(True); spin.setValue(value); spin.blockSignals(False)
    page.accepted_check.blockSignals(True)
    page.accepted_check.setChecked(view.get("accepted") is True and page.accepted_check.isEnabled())
    page.accepted_check.blockSignals(False)
    if page.charts[0].linear != (view.get("linear") is True):
        page._toggle_scale()
    page.apply_filters()
    selected = view.get("selected")
    if selected is None:
        page.select_index(None)
    else:
        for index in page.visible_indices:
            if page.result.updates[index]["update_id"] == selected:
                page.select_index(index); break
    page.view_tabs.setCurrentIndex(1 if view.get("tab") == 1 else 0)
    page.details_toggle.setChecked(view.get("details") is True)
    page.filters_toggle.setChecked(view.get("filters") is True)


def restore_page(window, name, page):
    if name in window._restored_pages:
        return
    required = {"console": "open_result", "comparison": "set_files", "archive": "set_folder"}
    if name in required and not hasattr(page, required[name]):
        return  # A factory error page must remain retryable.
    window._restored_pages.add(name)
    saved = window._workspace_session.get(name)
    if not isinstance(saved, dict):
        return
    if name == "console" and isinstance(saved.get("path"), str):
        def loaded(result):
            page.resultLoaded.disconnect(loaded)
            if str(Path(result.sources[0]["path"]).resolve()) != str(Path(saved["path"]).resolve()):
                return
            if same_sources(result, saved.get("sources")):
                restore_console_view(page, saved.get("view"))
            else:
                page.recovery_message.setText("Kaynak dosyaları değişmiş. Sonuç yeniden okundu; eski filtre ve seçim uygulanmadı.")
                page.recovery_message.show()
        page.resultLoaded.connect(loaded)
        page.open_result(Path(saved["path"]))
    elif name == "comparison":
        paths = saved.get("paths")
        if not isinstance(paths,list) or len(paths)!=2 or not all(isinstance(p,str) for p in paths):
            return
        try:
            view = validate_view(saved.get("view"))
        except ValueError:
            return
        def loaded(comparison):
            page.comparisonLoaded.disconnect(loaded)
            actual = [comparison.left.sources[0]["path"], comparison.right.sources[0]["path"]]
            if [str(Path(p).resolve()) for p in paths] != [str(Path(p).resolve()) for p in actual]:
                return
            if all(same_sources(result, saved.get(side)) for result,side in ((comparison.left,"A"),(comparison.right,"B"))):
                page.restore_view(view)
            else:
                page.effect_summary.setText(page.effect_summary.text()+"\nKaynaklar değişmiş; eski görünüm uygulanmadı.")
        page.comparisonLoaded.connect(loaded)
        variable = page.variable.findData(saved.get("variable","none"))
        page.variable.setCurrentIndex(max(0,variable))
        page.set_files(*paths)
    elif name == "archive" and isinstance(saved.get("folder"),str):
        page.set_folder(Path(saved["folder"]))


def save_session(window):
    data = dict(window._workspace_session)
    data["schema"] = 1
    data["active"] = window._active_page if window._active_page in ("console","comparison","archive") else "console"
    for name,page in window._pages.items():
        if name == "console" and getattr(page, "result", None) is not None:
            data[name] = capture_console(page)
        elif name == "comparison" and getattr(page, "comparison", None) is not None:
            comparison = page.comparison
            data[name] = {"paths":[comparison.left.sources[0]["path"],comparison.right.sources[0]["path"]],
                "A":identities(comparison.left), "B":identities(comparison.right),
                "variable":comparison.variable, "view":page.capture_view()}
        elif name == "archive" and hasattr(page,"folder"):
            data[name] = {"folder":str(page.folder)}
    window._settings.setValue(KEY,json.dumps(data,ensure_ascii=False))
