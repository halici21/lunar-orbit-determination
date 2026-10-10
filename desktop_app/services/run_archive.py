"""Read-only artifact index. CSV payloads and trajectories load only on demand."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from services.tracking_results import ResultLoadError, LoadCancelled, discover_tracking_runs


@dataclass(frozen=True)
class ArchiveEntry:
    path: Path
    name: str
    run_id: str
    created_at: str
    estimator: str
    measurement: str
    network: str
    seed: object
    kind: str
    status: str
    records: int
    bytes: int
    metadata: dict
    config: dict
    options: dict
    notes: tuple[str, ...]
    derived_from: dict = field(default_factory=dict)


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 1024 * 1024:
        raise ResultLoadError(f"{path.name}: 1 MB üstünde JSON.")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ResultLoadError(f"{path.name}: JSON nesnesi gerekli.")
    return payload


def local_reference(folder, name):
    if not isinstance(name, str) or not name:
        raise ResultLoadError("Geçerli kaynak dosya adı gerekli.")
    path = (folder / name).resolve()
    if path.parent != folder.resolve() or not path.is_file():
        raise ResultLoadError(f"Aynı klasörde kaynak bulunamadı: {name}")
    return path


def _count_csv(path, cancel):
    if path.stat().st_size > 128 * 1024 * 1024:
        raise ResultLoadError("CSV 128 MB sınırını aşıyor.")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        required = {"update_id", "arc_id", "t_s", "station_id", "position_error_m", "velocity_error_mps"}
        if not required <= set(header) or len(header) != len(set(header)):
            raise ResultLoadError("CSV sütun sözleşmesi eksik/yinelenen.")
        count = 0
        for row in reader:
            count += 1
            if count % 512 == 0 and cancel():
                raise LoadCancelled()
            if len(row) != len(header) or count > 250000:
                raise ResultLoadError("CSV satır biçimi veya kayıt sınırı geçersiz.")
        if not count:
            raise ResultLoadError("CSV boş.")
        return count


def _csv_signature(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def scan_run_archive(folder, cancel=lambda: False, *, count_cache=None, missing_ok=False):
    """Index artifacts; optional per-worker cache stores structural row counts.

    Stat signatures are a navigation optimization, not an integrity guarantee.
    Explicit refresh uses a fresh cache; opening a result validates its payload.
    missing_ok is for a new workspace before its first result folder is created.
    """
    folder = Path(folder)
    if missing_ok and not folder.exists():
        if cancel():
            raise LoadCancelled()
        return ()
    entries = []
    for run in discover_tracking_runs(folder, cancel):
        if cancel():
            raise LoadCancelled()
        path = run.updates_path
        batch = path.name.endswith("_arc_solutions.csv")
        stem = path.name.removesuffix("_arc_solutions.csv" if batch else "_ukf_updates.csv")
        metadata, config, options, notes, derived = {}, {}, {}, [], {}
        run_id, status, count = "—", "incomplete" if batch else "legacy", 0
        size, created = 0, "Bilinmiyor"
        try:
            stat = path.stat()
            size = stat.st_size
            created = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
            signature = _csv_signature(stat)
            cached = count_cache.get(path) if count_cache is not None else None
            if cached is not None and cached[0] == signature:
                count = cached[1]
            else:
                count = _count_csv(path, cancel)
                # Never retain a count for a file modified during the scan.
                if count_cache is not None and _csv_signature(path.stat()) == signature:
                    count_cache[path] = (signature, count)
                    while len(count_cache) > 512:
                        count_cache.pop(next(iter(count_cache)))
            # Manifests, job records and completion state are always read afresh.
            manifest = path.with_name(stem + "_result_manifest.json")
            if manifest.is_file():
                metadata = read_json(manifest)
                if metadata.get("schema") != "lunar-od-result-v1":
                    raise ResultLoadError("Manifest şeması desteklenmiyor.")
                if metadata.get("config_file"):
                    config = read_json(local_reference(path.parent, metadata["config_file"]))
                if metadata.get("job_file"):
                    job = read_json(local_reference(path.parent, metadata["job_file"]))
                    run_id = job.get("run_id", "—")
                    options = job.get("options", {})
                    lineage = job.get("derived_from")
                    if lineage is not None:
                        if isinstance(lineage, dict) and lineage.get("schema") == "lunar-od-derived-run-v1":
                            derived = lineage
                        else:
                            notes.append("Kaynak koşu ilişkisinin şeması desteklenmiyor.")
                    created = job.get("created_at", created)
                    if not isinstance(options, dict) or not isinstance(created, str) or not isinstance(run_id, str):
                        raise ResultLoadError("Koşu kaydı alanları geçersiz.")
                    completion_path = path.parent / "completed.json"
                    if completion_path.is_file():
                        completed = read_json(completion_path)
                        if (completed.get("schema") != "lunar-od-completed-job-v1"
                                or completed.get("run_id") != run_id
                                or Path(completed.get("result_path", "")).name != path.name):
                            raise ResultLoadError("Tamamlanma kaydı bu koşuyla uyuşmuyor.")
                        status = "completed"
                        if Path(completed["result_path"]).resolve() != path:
                            notes.append("Koşu klasörü taşınmış; tamamlanma kaydı eski yolu içeriyor.")
                    else:
                        status = "incomplete"
                        notes.append("Tamamlanma kaydı yok; koşu sürüyor veya kesilmiş olabilir.")
            else:
                notes.append("Manifest yok; epok, çerçeve ve üretici bilinmiyor.")
            if "_5deg_backup" in path.parts:
                notes.append("Eski arşiv; güncel bilimsel geçerlilik doğrulanmadı.")
        except (OSError, ValueError, TypeError, csv.Error) as exc:
            status = "invalid"
            notes.append(str(exc))
        # Indexing checks structure; scientific/numeric payload validation happens
        # in load_tracking_results when opening or comparing selected files.
        notes.append("Dizin kaydıdır; sayısal içerik açılışta doğrulanır.")
        entries.append(ArchiveEntry(path, stem, run_id, created,
            str(metadata.get("estimator_type") or config.get("estimator_type") or ("ukf" if not batch else "Bilinmiyor")),
            str(config.get("measurement_type", "Bilinmiyor")), str(config.get("network", "Bilinmiyor")),
            metadata.get("seed"), "arc_solution" if batch else "ukf_update", status, count, size,
            metadata, config, options, tuple(notes), derived))
    return tuple(sorted(entries, key=lambda e: (e.created_at, str(e.path)), reverse=True))
