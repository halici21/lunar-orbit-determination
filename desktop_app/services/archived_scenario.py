"""Recover a verified native run's form without guessing unsupported settings."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import get_args, get_type_hints

from models.scenario_model import ScenarioModel
from services.run_archive import local_reference
from services.scenario_execution import execution_payload
from services.tracking_results import load_tracking_results, ResultLoadError, LoadCancelled


@dataclass(frozen=True)
class ArchivedScenario:
    model: ScenarioModel
    origin: dict
    notes: tuple[str, ...]


def form_from_mapping(form):
    """Native job forms are complete, typed snapshots, not partial presets."""
    expected = {field.name for field in fields(ScenarioModel)}
    if not isinstance(form, dict) or set(form) != expected:
        raise ResultLoadError("Kayıtlı formun alanları bu sürümle uyuşmuyor; eksik/fazla ayarlar tahmin edilmedi.")
    for name, hint in get_type_hints(ScenarioModel).items():
        value = form[name]
        types = get_args(hint) or (hint,)
        valid = any((value is None if typ is type(None) else
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) if typ is float else
            type(value) is typ) for typ in types)
        if not valid:
            raise ResultLoadError(f"Kayıtlı formda {name} türü/değeri geçersiz.")
    return ScenarioModel(**form)


def load_archived_scenario(result_path, cancel=lambda: False):
    result = load_tracking_results(result_path, cancel)
    def read_verified(path):
        path = Path(path).resolve()
        source = next((s for s in result.sources if Path(s["path"]).resolve() == path), None)
        if source is None:
            raise ResultLoadError(f"Doğrulanmış kaynak yok: {path.name}")
        data = path.read_bytes()
        if len(data) > 1024 * 1024 or hashlib.sha256(data).hexdigest() != source["sha256"]:
            raise ResultLoadError(f"{path.name}: boyut sınırı aşıldı veya kaynak okuma sırasında değişti.")
        value = json.loads(data.decode("utf-8-sig"))
        if not isinstance(value, dict):
            raise ResultLoadError(f"{path.name}: JSON nesnesi gerekli.")
        return value
    manifest_source = next((s for s in result.sources if s["path"].endswith("_result_manifest.json")), None)
    if not manifest_source:
        raise ResultLoadError("Koşunun manifesti yok; kayıtlı form geri yüklenemiyor.")
    manifest = read_verified(manifest_source["path"])
    folder = Path(result.sources[0]["path"]).parent
    job_path = local_reference(folder, manifest.get("job_file"))
    job = read_verified(job_path)
    if job.get("schema") != "lunar-od-desktop-job-v1" or not isinstance(job.get("run_id"), str) or not job["run_id"]:
        raise ResultLoadError("Desteklenen native koşu kaydı/kimliği gerekli.")
    model = form_from_mapping(job.get("form"))
    config, options = execution_payload(model)
    saved_config = job.get("config")
    if not isinstance(saved_config, dict) or not isinstance(saved_config.get("output_dir"), str):
        raise ResultLoadError("Kaydedilmiş koşu yapılandırması eksik.")
    if {k: v for k, v in saved_config.items() if k != "output_dir"} != config or job.get("options") != options:
        raise ResultLoadError("Kayıtlı form, yapılandırma ve zamanlama ayarları uyuşmuyor.")
    effective = read_verified(local_reference(folder, manifest.get("config_file")))
    if cancel():
        raise LoadCancelled()
    # Validate current defaults against the recorded effective configuration on
    # this worker. Unsupported old defaults must not be silently replaced.
    from lunar_od import scenario_config_from_mapping
    current = scenario_config_from_mapping(saved_config).to_dict()
    differences = sorted(k for k in effective.keys() | current.keys()
        if k not in {"name", "output_dir"} and effective.get(k, "<missing>") != current.get(k, "<missing>"))
    if differences:
        raise ResultLoadError("Etkin ayarlar bu form/yürütücü ile aynen kurulamadı: " + ", ".join(differences[:12]))
    if cancel():
        raise LoadCancelled()
    origin = {"schema": "lunar-od-derived-run-v1", "run_id": job["run_id"],
        "result_path": str(Path(result_path).resolve()), "sources": list(result.sources),
        "original_form": job["form"], "effective_config": effective}
    return ArchivedScenario(replace(model, name=model.name[:94] + "_copy"), origin,
        ("Kayıtlı form doğrulandı. Yeni koşu güncel kod ve ortamla, yeni kimlik altında çalışır.",))
