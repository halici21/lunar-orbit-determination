"""Descriptive comparison of recorded values at exact, unambiguous epochs."""
from __future__ import annotations

import csv
import json
import math
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from services.tracking_results import load_tracking_results, LoadCancelled
from services.run_archive import local_reference


@dataclass(frozen=True)
class RunComparison:
    left: object
    right: object
    snapshots: tuple[dict, dict]
    pairs: tuple[tuple[int, int], ...]
    time_compatible: bool
    notes: tuple[str, ...]
    differences: tuple[dict, ...]
    metrics: dict
    ambiguous: tuple[int, int]
    variable: str


def _snapshot(result):
    path = Path(result.sources[0]["path"])
    snapshot = {"config": {}, "options": {}, "force": {}}
    manifests = [s for s in result.sources if s["path"].endswith("_result_manifest.json")]
    if not manifests:
        return snapshot
    def verified_json(source_path):
        source_path = Path(source_path).resolve()
        source = next((s for s in result.sources if Path(s["path"]).resolve() == source_path), None)
        data = source_path.read_bytes()
        if source is None or hashlib.sha256(data).hexdigest() != source["sha256"]:
            raise ValueError(f"{source_path.name}: kaynak yüklemeden sonra değişti; yeniden karşılaştırın.")
        value = json.loads(data.decode("utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("Yapılandırma JSON nesnesi olmalı.")
        return value
    manifest = verified_json(manifests[0]["path"])
    for name, key in (("config_file", "config"), ("job_file", "options"), ("force_model_manifest_file", "force")):
        if manifest.get(name):
            value = verified_json(local_reference(path.parent, manifest[name]))
            snapshot[key] = value.get("options", {}) if key == "options" else value
            if not isinstance(snapshot[key], dict):
                raise ValueError(f"{name} sözleşmesi geçersiz.")
    return snapshot


def _flatten(value, prefix=""):
    rows = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict) and item:
            rows.update(_flatten(item, name))
        else:
            rows[name] = item
    return rows


def _differences(a, b, variable):
    left, right = _flatten(a), _flatten(b)
    ignored = {"config.name", "config.output_dir"}
    missing = "(alan yok)"
    return tuple({"field": key, "A": left.get(key, missing), "B": right.get(key, missing),
                  "role": "Seçilen değişken" if key == f"config.{variable}" else "Diğer fark"}
        for key in sorted(left.keys() | right.keys()) if key not in ignored and left.get(key, missing) != right.get(key, missing))


def quantile(values, fraction):
    """Linear order-statistic quantile; this does not interpolate physical states."""
    ordered = sorted(values)
    if not ordered:
        return None
    index = (len(ordered) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    weight = index - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * weight


def describe(values):
    values = tuple(values)
    if not values:
        return {"N": 0, "median": None, "p95": None, "max": None, "rms": None}
    maximum = max(values)
    rms = maximum * math.sqrt(sum((v / maximum)**2 for v in values) / len(values)) if maximum else 0.
    return {"N": len(values), "median": quantile(values, .5), "p95": quantile(values, .95), "max": maximum, "rms": rms}


def _keys(result):
    keys = {}
    batch = result.metadata.get("record_kind") == "arc_solution"
    for index, row in enumerate(result.updates):
        station = row.get("observed_station_ids", "") if batch else row["station_id"]
        key = (row["t_s"], row["arc_id"], station)
        keys.setdefault(key, []).append(index)
    return keys


def _report_metadata(result):
    # Parsed geometry is a drawing payload, like the separately held states.
    # Its original file/hash stays in sources; duplicating all windows in a
    # settings cell or report can exceed the workspace's 16 MB reopen limit.
    return {key: value for key, value in result.metadata.items() if key != "mission_geometry"}


def compare_results(left, right, snapshots=({}, {}), variable="none", cancel=lambda: False):
    notes, compatible = [], True
    for key in ("epoch", "time_scale", "frame", "origin", "record_kind"):
        a, b = left.metadata.get(key), right.metadata.get(key)
        if a in (None, "", "Bilinmiyor") or b in (None, "", "Bilinmiyor"):
            notes.append(f"{key} bilinmiyor; ortak zaman/çerçeve doğrulanamadı.")
            compatible = False
        elif a != b:
            notes.append(f"{key} farklı; kayıt farkları hesaplanmaz.")
            compatible = False
    a_convention, b_convention = left.metadata.get("station_id_convention"), right.metadata.get("station_id_convention")
    if a_convention != b_convention:
        notes.append("İstasyon kimliği sözleşmeleri farklı/biri eksik; kayıt farkları hesaplanmaz.")
        compatible = False
    elif not a_convention:
        notes.append("İstasyon kimliği sözleşmesi kayıtlı değil; aynı sayısal kimliklerin aynı istasyona ait olduğu bağımsız doğrulanmadı.")
    lk, rk = _keys(left), _keys(right)
    pairs = []
    if compatible:
        for n, key in enumerate(sorted(lk.keys() & rk.keys())):
            if n % 512 == 0 and cancel():
                raise LoadCancelled()
            if len(lk[key]) == len(rk[key]) == 1:
                pairs.append((lk[key][0], rk[key][0]))
    ambiguous = tuple(sum(len(v) for v in keys.values() if len(v) > 1) for keys in (lk, rk))
    if any(ambiguous):
        notes.append(f"Aynı epok/yay/istasyon anahtarında belirsiz kayıt: A={ambiguous[0]}, B={ambiguous[1]}; eşleşmeye alınmadı.")
    if compatible and not pairs:
        notes.append("Aynı epok, yay ve istasyon anahtarında ortak kayıt yok. Durumlar başka epoka taşınmadı.")
    compared_snapshots = tuple({**s, "metadata": _report_metadata(r)} for s, r in zip(snapshots, (left, right)))
    changes = _differences(compared_snapshots[0], compared_snapshots[1], variable)
    if changes:
        notes.append(f"{len(changes)} ayar/kuvvet sözleşmesi farkı var; seçilen değişken ayrıca işaretlendi.")
    if not all(s.get("config") for s in snapshots):
        notes.append("En az bir koşunun etkili yapılandırması eksik.")
    if not left.metadata.get("nis_scope") or left.metadata.get("nis_scope") != right.metadata.get("nis_scope"):
        notes.append("NIS kapsamları farklı veya bilinmiyor; NIS grafikleri ayrı gösterilir.")
    notes.extend(f"{side}: {note}" for side, result in (("A", left), ("B", right)) for note in result.notes)
    notes.append("Betimleyici karşılaştırma: ölçüm gerçekleşimi ve başlangıç kovaryansı artefaktlarıyla tam adil karşılaştırma doğrulanmadı. Algoritma üstünlüğü veya istatistiksel anlamlılık sonucu verilmez.")
    if len(pairs) < 10:
        notes.append(f"Eşleşen N={len(pairs)}; küçük örneklem, p95 güvenilir bir kuyruk genellemesi değildir.")
    metrics = {"all": {}, "paired": {}}
    for key in ("position_error_m", "velocity_error_mps"):
        metrics["all"][key] = {"A": describe(r[key] for r in left.updates), "B": describe(r[key] for r in right.updates)}
        metrics["paired"][key] = {"A": describe(left.updates[i][key] for i, _ in pairs),
                                  "B": describe(right.updates[j][key] for _, j in pairs)}
    return RunComparison(left, right, snapshots, tuple(pairs), compatible, tuple(notes), changes, metrics, ambiguous, variable)


def load_run_comparison(left_path, right_path, variable="none", cancel=lambda: False):
    if Path(left_path).resolve() == Path(right_path).resolve():
        raise ValueError("İki farklı kaynak dosyası seçin.")
    left = load_tracking_results(left_path, cancel)
    right = load_tracking_results(right_path, cancel)
    if cancel():
        raise LoadCancelled()
    return compare_results(left, right, (_snapshot(left), _snapshot(right)), variable, cancel)


def paired_rows(comparison):
    for i, j in comparison.pairs:
        a, b = comparison.left.updates[i], comparison.right.updates[j]
        yield {"t_s": a["t_s"], "arc_id": a["arc_id"], "station_id": a["station_id"],
            "observed_station_ids": a.get("observed_station_ids", ""), "A_record_id": a["update_id"], "B_record_id": b["update_id"],
            "A_position_error_m": a["position_error_m"], "B_position_error_m": b["position_error_m"],
            "delta_position_error_m_B_minus_A": b["position_error_m"] - a["position_error_m"],
            "A_velocity_error_mps": a["velocity_error_mps"], "B_velocity_error_mps": b["velocity_error_mps"],
            "delta_velocity_error_mps_B_minus_A": b["velocity_error_mps"] - a["velocity_error_mps"]}


def export_comparison(comparison, path, cancel=lambda: False, *, view=None):
    from services.comparison_session import validate_view
    view = validate_view(view)
    path = Path(path).resolve()
    csv_path = path.with_name(path.stem + "_paired.csv")
    sources = (*comparison.left.sources, *comparison.right.sources)
    if any(Path(s["path"]).resolve() in (path, csv_path) for s in sources):
        raise OSError("Kaynak dosyaların üzerine rapor yazılamaz.")
    report = {"schema": "lunar-od-recorded-comparison-v1", "variable": comparison.variable,
        "exact_epoch_only": True, "time_compatible": comparison.time_compatible,
        "paired_csv": csv_path.name if comparison.pairs else None, "delta_definition": "B_minus_A_recorded_error_norms",
        "units": {"time": "s", "position": "m", "velocity": "m/s"},
        "paired_N": len(comparison.pairs), "ambiguous": comparison.ambiguous,
        "notes": comparison.notes, "metrics": comparison.metrics, "differences": comparison.differences,
        "A": {"name": comparison.left.name, "metadata": _report_metadata(comparison.left), "sources": comparison.left.sources, "snapshot": comparison.snapshots[0]},
        "B": {"name": comparison.right.name, "metadata": _report_metadata(comparison.right), "sources": comparison.right.sources, "snapshot": comparison.snapshots[1]}}
    if view:
        report["view"] = view
    # Keep the absolute provenance path and add a relocatable reference. Source
    # identities remain content hashes; moving a whole directory tree is safe.
    for label in ("A", "B"):
        items = []
        for source in report[label]["sources"]:
            item = dict(source)
            try:
                item["relative_path"] = os.path.relpath(item["path"], path.parent)
            except ValueError:  # Different Windows volumes: retain absolute path.
                pass
            items.append(item)
        report[label]["sources"] = items
    rows, staging = paired_rows(comparison), []
    first = next(rows, None)
    if not first and csv_path.exists():
        raise OSError("Bu adda eski eşleşme CSV'si var; rapor için yeni bir ad seçin.")
    try:
        if cancel():
            raise LoadCancelled()
        if first:
            with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
                staging.append(Path(handle.name))
                writer = csv.DictWriter(handle, fieldnames=list(first))
                writer.writeheader()
                writer.writerow(first)
                for index, row in enumerate(rows):
                    if index % 512 == 0 and cancel():
                        raise LoadCancelled()
                    writer.writerow(row)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            staging.append(Path(handle.name))
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        if cancel():
            raise LoadCancelled()
        if first:
            os.replace(staging[0], csv_path)
        os.replace(staging[-1], path)
    finally:
        for temporary in staging:
            temporary.unlink(missing_ok=True)
    return path, csv_path if first else None
