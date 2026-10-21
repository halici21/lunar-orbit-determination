"""Reopen reports by checking original artifact hashes and recomputing values."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from services.tracking_results import load_tracking_results, LoadCancelled, ResultLoadError

TABS = ("charts", "pairs", "statistics", "differences", "sources", "notes")


def validate_view(view):
    if view is None:
        return {}
    if not isinstance(view, dict) or view.get("schema") != "lunar-od-comparison-view-v1":
        raise ResultLoadError("Desteklenmeyen karşılaştırma görünümü.")
    if set(view) - {"schema", "linear", "paired_only", "tab", "time_range", "selected_records"}:
        raise ResultLoadError("Görünümde desteklenmeyen alan var.")
    for key in ("linear", "paired_only"):
        if type(view.get(key, False)) is not bool:
            raise ResultLoadError(f"Görünümde {key} boolean olmalı.")
    if view.get("tab", "charts") not in TABS:
        raise ResultLoadError("Görünüm sekmesi geçersiz.")
    interval = view.get("time_range")
    if interval is not None and (not isinstance(interval, (list, tuple)) or len(interval) != 2
            or not all(type(v) in (int, float) and math.isfinite(v) for v in interval) or interval[0] >= interval[1]):
        raise ResultLoadError("Görünüm zaman aralığı geçersiz.")
    selected = view.get("selected_records")
    if selected is not None and (not isinstance(selected, (list, tuple)) or len(selected) != 2
            or not all(type(v) is int and v >= 0 for v in selected)):
        raise ResultLoadError("Görünüm kayıt kimlikleri geçersiz.")
    return dict(view)


def _open_source(side, folder, label, cancel):
    if not isinstance(side, dict) or not isinstance(side.get("sources"), list) or not 1 <= len(side["sources"]) <= 16:
        raise ResultLoadError(f"{label}: kaynak listesi eksik/geçersiz.")
    expected = {}
    for source in side["sources"]:
        if (not isinstance(source, dict) or not isinstance(source.get("path"), str) or not source["path"]
                or not isinstance(source.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"])
                or type(source.get("bytes")) is not int or source["bytes"] < 0):
            raise ResultLoadError(f"{label}: kaynak yolu/hash/boyut kaydı geçersiz.")
        name = Path(source["path"]).name
        if name in expected:
            raise ResultLoadError(f"{label}: yinelenen kaynak adı.")
        expected[name] = (source["sha256"], source["bytes"])
    primary = side["sources"][0]
    candidates = []
    relative = primary.get("relative_path")
    if relative is not None:
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ResultLoadError(f"{label}: göreli kaynak yolu geçersiz.")
        candidates.append((folder / relative).resolve())
    candidates.append(Path(primary["path"]).resolve())
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise ResultLoadError(f"{label}: kaynak bulunamadı: {primary['path']}. Rapor ve koşu klasörlerinin göreli düzenini koruyun.")
    result = load_tracking_results(path, cancel)
    actual = {Path(s["path"]).name: (s["sha256"], s["bytes"]) for s in result.sources}
    if actual != expected:
        raise ResultLoadError(f"{label}: kaynak dosyaları kaydedilmiş SHA-256/boyut bilgisiyle uyuşmuyor. Eski karşılaştırma korunuyor.")
    return result


def load_comparison_session(path, cancel=lambda: False):
    from services.run_comparison import compare_results, _snapshot
    path = Path(path).resolve()
    if cancel():
        raise LoadCancelled()
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ResultLoadError("Karşılaştırma raporu 16 MB sınırını aşıyor.")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or data.get("schema") != "lunar-od-recorded-comparison-v1":
        raise ResultLoadError("Desteklenmeyen karşılaştırma raporu.")
    variable = data.get("variable", "none")
    if not isinstance(variable, str) or len(variable) > 100:
        raise ResultLoadError("İncelenen değişken geçersiz.")
    view = validate_view(data.get("view"))
    left = _open_source(data.get("A"), path.parent, "A", cancel)
    right = _open_source(data.get("B"), path.parent, "B", cancel)
    if Path(left.sources[0]["path"]) == Path(right.sources[0]["path"]):
        raise ResultLoadError("Rapor iki farklı kaynak içermeli.")
    if cancel():
        raise LoadCancelled()
    comparison = compare_results(left, right, (_snapshot(left), _snapshot(right)), variable, cancel)
    selected = view.get("selected_records")
    if selected is not None and not any([left.updates[i]["update_id"], right.updates[j]["update_id"]] == list(selected)
            for i, j in comparison.pairs):
        raise ResultLoadError("Kaydedilmiş seçim doğrulanan eşleşmelerde bulunamadı.")
    return comparison, view, path
