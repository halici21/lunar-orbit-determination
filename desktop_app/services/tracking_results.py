"""Read-only contract for sequential tracking artifacts; no scientific imports."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


class ResultLoadError(ValueError):
    """An artifact does not satisfy the display contract."""


class LoadCancelled(Exception):
    pass


@dataclass(frozen=True)
class TrackingRun:
    name: str
    updates_path: Path


@dataclass(frozen=True)
class TrackingResults:
    updates: tuple[dict, ...]
    arcs: tuple[dict, ...]
    states: tuple[dict, ...]
    sources: tuple[dict, ...]
    metadata: dict
    notes: tuple[str, ...]
    load_ms: float
    # Preserve arc gaps in plots; rows are sorted only after validating identity.
    name: str


def _cancelled(cancel: Callable[[], bool]) -> None:
    if cancel():
        raise LoadCancelled()


def discover_tracking_runs(folder: Path, cancel=lambda: False) -> tuple[TrackingRun, ...]:
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ResultLoadError(f"Sonuç klasörü bulunamadı: {folder}")
    runs = []
    # Explicit suffix prevents pairing an unrelated summary with an update series.
    for path in folder.rglob("*.csv"):
        _cancelled(cancel)
        suffix = next((s for s in ("_ukf_updates.csv", "_arc_solutions.csv") if path.name.endswith(s)), None)
        if suffix and path.is_file():
            runs.append(TrackingRun(str(path.relative_to(folder))[:-len(suffix)], path.resolve()))
    return tuple(sorted(runs, key=lambda r: r.name))


def _number(row, key, path, line, *, nonnegative=False):
    try:
        value = float(row[key])
    except (ValueError, TypeError, KeyError):
        raise ResultLoadError(f"{path.name}, satır {line}: geçersiz {key}.") from None
    if not math.isfinite(value) or nonnegative and value < 0:
        raise ResultLoadError(f"{path.name}, satır {line}: sonlu/geçerli {key} gerekli.")
    return value


def _integer(row, key, path, line):
    n = _number(row, key, path, line, nonnegative=True)
    if not n.is_integer():
        raise ResultLoadError(f"{path.name}, satır {line}: {key} tam sayı olmalı.")
    return int(n)


def _boolean(row, key, path, line):
    text = str(row.get(key, "")).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ResultLoadError(f"{path.name}, satır {line}: geçersiz {key} boolean değeri.")


def _read_csv(path, required, cancel):
    before = _fingerprint(path)
    if before[0] > 128 * 1024 * 1024:
        raise ResultLoadError(f"{path.name}: 128 MB görüntüleme sınırını aşıyor.")
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            missing = set(required) - set(headers)
            if missing:
                raise ResultLoadError(f"{path.name}: eksik sütunlar: {', '.join(sorted(missing))}")
            if len(headers) != len(set(headers)):
                raise ResultLoadError(f"{path.name}: yinelenen sütun başlığı.")
            for line, row in enumerate(reader, 2):
                if line % 1024 == 0:
                    _cancelled(cancel)
                if line > 250001:
                    raise ResultLoadError(f"{path.name}: 250.000 kayıt görüntüleme sınırını aşıyor.")
                if None in row or any(v is None for v in row.values()):
                    raise ResultLoadError(f"{path.name}, satır {line}: sütun sayısı eşleşmiyor.")
                yield line, row
        _unchanged(path, before)
    except (UnicodeError, csv.Error, OSError) as exc:
        raise ResultLoadError(f"{path.name} okunamadı: {exc}") from exc


def _fingerprint(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _unchanged(path, before):
    if _fingerprint(path) != before:
        raise ResultLoadError(f"{path.name}: okuma sırasında kaynak değişti; yeniden yükleyin.")


def _source(path, cancel):
    before = _fingerprint(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            _cancelled(cancel)
            digest.update(block)
    _unchanged(path, before)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": before[0]}


def load_tracking_results(updates_path: Path, cancel=lambda: False) -> TrackingResults:
    started = time.perf_counter()
    path = Path(updates_path).resolve()
    suffix = next((s for s in ("_ukf_updates.csv", "_arc_solutions.csv") if path.name.endswith(s)), None)
    if not path.is_file() or suffix is None:
        raise ResultLoadError("Bir *_ukf_updates.csv veya *_arc_solutions.csv sonuç dosyası seçin.")
    _cancelled(cancel)
    fingerprints = {path: _fingerprint(path)}
    prefix = path.name.removesuffix(suffix)
    batch = suffix == "_arc_solutions.csv"
    rows, seen = [], set()
    required = ("update_id", "arc_id", "t_s", "station_id", "position_error_m", "velocity_error_mps")
    for line, row in _read_csv(path, required, cancel):
        item = {k: _integer(row, k, path, line) for k in ("update_id", "arc_id")}
        item["station_id"] = None if batch and not row["station_id"].strip() else _integer(row, "station_id", path, line)
        for k in ("t_s", "position_error_m", "velocity_error_mps"):
            item[k] = _number(row, k, path, line, nonnegative=True)
        if item["update_id"] in seen:
            raise ResultLoadError(f"{path.name}: yinelenen update_id {item['update_id']}.")
        seen.add(item["update_id"])
        # Missing diagnostic fields remain unknown, never become zero/accepted.
        for k in ("nis", "normalized_nis", "covariance_condition_number", "accepted_component_fraction"):
            item[k] = _number(row, k, path, line, nonnegative=True) if row.get(k, "").strip() else None
        item["accepted"] = _boolean(row, "accepted", path, line) if row.get("accepted", "").strip() else None
        if batch:
            item["operational_success"] = _boolean(row, "operational_success", path, line) if row.get("operational_success", "").strip() else None
            item["stop_reason"] = row.get("stop_reason", "Bilinmiyor")
            item["observed_station_ids"] = row.get("observed_station_ids", "")
            for key in ("start_s", "end_s", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"):
                item[key] = _number(row, key, path, line)
            if not 0 <= item["start_s"] <= item["t_s"] <= item["end_s"]:
                raise ResultLoadError("Yay çözüm epoku yay aralığı dışında.")
        rows.append(item)
    if not rows:
        raise ResultLoadError(f"{path.name}: görüntülenecek kayıt yok.")
    rows.sort(key=lambda r: (r["t_s"], r["update_id"]))
    notes, sources = [], [_source(path, cancel)]
    arcs = []
    arc_path = path.with_name(prefix + "_bls_arc_markers.csv")
    if arc_path.is_file():
        fingerprints[arc_path] = _fingerprint(arc_path)
        seen_arcs = set()
        for line, row in _read_csv(arc_path, ("arc_id", "start_s", "end_s", "arc_end_position_error_m", "arc_end_velocity_error_mps"), cancel):
            item = {"arc_id": _integer(row, "arc_id", arc_path, line)}
            for k in ("start_s", "end_s", "arc_end_position_error_m", "arc_end_velocity_error_mps"):
                item[k] = _number(row, k, arc_path, line, nonnegative=True)
            if item["end_s"] < item["start_s"] or item["arc_id"] in seen_arcs:
                raise ResultLoadError(f"{arc_path.name}, satır {line}: geçersiz/yinelenen yay.")
            seen_arcs.add(item["arc_id"])
            item["operational_success"] = _boolean(row, "operational_success", arc_path, line) if row.get("operational_success", "").strip() else None
            item["stop_reason"] = row.get("stop_reason", "Bilinmiyor")
            arcs.append(item)
        arcs.sort(key=lambda a: a["end_s"])
        sources.append(_source(arc_path, cancel))
    elif not batch:
        notes.append("Bu koşuya ait BLS yay sonu dosyası yok.")
    metadata = {"frame": "Bilinmiyor", "epoch": "Bilinmiyor", "time_scale": "Bilinmiyor", "origin": "Bilinmiyor", "seed": None, "producer": None, "commit": None}
    manifest = path.with_name(prefix + "_result_manifest.json")
    states = []
    metadata["record_kind"] = "arc_solution" if batch else "ukf_update"
    if batch and not (path.parent / "completed.json").is_file():
        notes.append("Tamamlanma kaydı yok; bu dosya kısmi çıktı olabilir.")
    if manifest.is_file():
        fingerprints[manifest] = _fingerprint(manifest)
        if manifest.stat().st_size > 1024 * 1024:
            raise ResultLoadError("Sonuç manifesti 1 MB sınırını aşıyor.")
        try:
            info = json.loads(manifest.read_text(encoding="utf-8-sig"))
        except (ValueError, OSError) as exc:
            raise ResultLoadError(f"Sonuç manifesti okunamadı: {exc}") from exc
        if not isinstance(info, dict) or info.get("schema") != "lunar-od-result-v1":
            raise ResultLoadError("Desteklenmeyen sonuç manifesti şeması.")
        if info.get("units") != {"time": "s", "position": "m", "velocity": "m/s"}:
            raise ResultLoadError("Manifest birimleri time=s, position=m, velocity=m/s olmalı.")
        for key in metadata:
            value = info.get(key)
            if value is not None:
                if key in {"frame", "epoch", "time_scale", "origin", "producer", "commit"} and not isinstance(value, str):
                    raise ResultLoadError(f"Manifestte {key} metin olmalı.")
                metadata[key] = value if key == "seed" else value or "Bilinmiyor"
        if metadata["record_kind"] != ("arc_solution" if batch else "ukf_update"):
            raise ResultLoadError("Manifest kayıt türü dosya adıyla uyuşmuyor.")
        for key in ("estimator_type", "state_role", "nis_scope", "schedule_policy", "station_id_convention", "cold_start_seed", "job_sha256"):
            if key in info:
                if key != "cold_start_seed" and not isinstance(info[key], str):
                    raise ResultLoadError(f"Manifestte {key} metin olmalı.")
                metadata[key] = info[key]
        if batch and metadata.get("estimator_type") not in {"bls_lm", "srif", "ukf"}:
            raise ResultLoadError("Yay çözümü için desteklenen estimator_type gerekli.")
        metadata["working_tree_dirty"] = info.get("working_tree_dirty")
        metadata["code_hashes"] = info.get("code_hashes", {})
        for key in ("config_file", "job_file", "force_model_manifest_file"):
            if info.get(key):
                if not isinstance(info[key], str):
                    raise ResultLoadError(f"Manifestte {key} dosya adı olmalı.")
                side_path = (path.parent / info[key]).resolve()
                if side_path.parent != path.parent or not side_path.is_file():
                    raise ResultLoadError(f"Manifestte {key} aynı klasörde bulunmalı.")
                side_source = _source(side_path, cancel)
                if key == "job_file" and info.get("job_sha256") and side_source["sha256"] != info["job_sha256"]:
                    raise ResultLoadError("job.json SHA-256 manifest kaydıyla uyuşmuyor.")
                sources.append(side_source)
        if info.get("mission_file"):
            name = info["mission_file"]
            if not isinstance(name, str):
                raise ResultLoadError("mission_file bir dosya adı olmalı.")
            mission_path = (path.parent / name).resolve()
            if mission_path.parent != path.parent or not mission_path.is_file():
                raise ResultLoadError("Görev geometrisi manifest ile aynı klasörde bulunmalı.")
            fingerprints[mission_path] = _fingerprint(mission_path)
            if mission_path.stat().st_size > 16 * 1024 * 1024:
                raise ResultLoadError("Görev geometrisi 16 MB sınırını aşıyor.")
            mission_source = _source(mission_path, cancel)
            if mission_source["sha256"] != info.get("mission_sha256"):
                raise ResultLoadError("Görev geometrisi SHA-256 manifest kaydıyla uyuşmuyor.")
            try:
                from services.mission_data import validate_geometry
                metadata["mission_geometry"] = validate_geometry(json.loads(mission_path.read_text(encoding="utf-8-sig")))
            except (ValueError, TypeError, OSError) as exc:
                raise ResultLoadError(f"Görev geometrisi geçersiz: {exc}") from exc
            metadata["mission_file"] = mission_path.name
            sources.append(mission_source)
        sources.append(_source(manifest, cancel))
        state_file = info.get("state_file")
        if state_file:
            if not isinstance(state_file, str):
                raise ResultLoadError("state_file bir dosya adı olmalı.")
            state_path = (path.parent / state_file).resolve()
            if state_path.parent != path.parent or not state_path.is_file():
                raise ResultLoadError("Durum dosyası manifest ile aynı klasörde bulunmalı.")
            fingerprints[state_path] = _fingerprint(state_path)
            if metadata["origin"] != "Moon" or metadata["frame"] == "Bilinmiyor":
                raise ResultLoadError("Yörünge için origin=Moon ve açık referans çerçevesi gerekli.")
            coords = ("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps")
            seen_times = set()
            for line, row in _read_csv(state_path, ("t_s", *coords), cancel):
                item = {k: _number(row, k, state_path, line) for k in ("t_s", *coords)}
                if item["t_s"] in seen_times:
                    raise ResultLoadError("Durum dosyasında yinelenen t_s.")
                seen_times.add(item["t_s"])
                states.append(item)
            states.sort(key=lambda r: r["t_s"])
            if not states:
                raise ResultLoadError("Durum dosyası boş.")
            sources.append(_source(state_path, cancel))
    else:
        notes.append("Sonuç manifesti yok; epok, çerçeve ve üretici bilgisi bilinmiyor.")
    if not states:
        notes.append("Durum vektörleri yok; bu koşu için yörünge çizilmez.")
    if "_5deg_backup" in path.parts:
        notes.append("Eski arşiv: güncel fizik/test geçerliliği doğrulanmadı.")
    _cancelled(cancel)
    for source, fingerprint in fingerprints.items():
        _unchanged(source, fingerprint)
    return TrackingResults(tuple(rows), tuple(arcs), tuple(states), tuple(sources), metadata, tuple(notes), (time.perf_counter() - started) * 1000, prefix)


def result_selection(rows, *, station=None, arc=None, start=-math.inf, end=math.inf, accepted_only=False):
    """Indices into the immutable source, so filtering cannot relabel update IDs."""
    return tuple(i for i, row in enumerate(rows) if (station is None or row["station_id"] == station)
                 and (arc is None or row["arc_id"] == arc) and start <= row["t_s"] <= end
                 and (not accepted_only or row["accepted"] is True))
