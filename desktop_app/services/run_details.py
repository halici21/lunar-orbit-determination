"""Verified display snapshots for the result inspector; no solver calculations."""
import hashlib
import json
from pathlib import Path

from services.tracking_results import load_tracking_results, LoadCancelled, ResultLoadError
from services.run_archive import read_json


def inspect_run(result, cancel=lambda: False):
    data = {"config": {}, "options": {}, "force": {}, "job": {}, "notes": list(result.notes)}
    manifest = None
    documents = {}
    for source in result.sources:
        if cancel():
            raise LoadCancelled()
        path = Path(source["path"])
        if path.suffix.lower() != ".json":
            continue
        raw = path.read_bytes()
        if len(raw) > 1024 * 1024 or hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ResultLoadError(f"{path.name}: ayrıntı kaynağı değişti veya boyut sınırı aşıldı.")
        value = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(value, dict):
            raise ResultLoadError(f"{path.name}: JSON nesnesi gerekli.")
        documents[path.name] = value
        if path.name.endswith("_result_manifest.json"):
            manifest = value
    if manifest:
        for key, field in (("config_file", "config"), ("job_file", "job"), ("force_model_manifest_file", "force")):
            name = manifest.get(key)
            if name:
                value = documents.get(Path(name).name)
                if value is None:
                    raise ResultLoadError(f"{name}: doğrulanmış ayrıntı kaynağı yok.")
                data[field] = value
        data["options"] = data["job"].get("options", {})
    data["units"] = manifest["units"] if manifest else {"time": "s", "position": "m", "velocity": "m/s"}
    data["status"] = "Tamamlanma kaydı yok"
    completed = Path(result.sources[0]["path"]).parent / "completed.json"
    if data["job"] and completed.is_file():
        try:
            record = read_json(completed)
            if (record.get("schema") != "lunar-od-completed-job-v1"
                    or record.get("run_id") != data["job"].get("run_id")
                    or Path(record.get("result_path", "")).name != Path(result.sources[0]["path"]).name):
                raise ValueError("Koşu kimliği veya sonuç dosyası uyuşmuyor.")
            data["status"] = "İşlem tamamlandı"
        except (OSError, ValueError, TypeError) as exc:
            data["status"] = "Tamamlanma kaydı geçersiz"
            data["notes"].append(str(exc))
    elif not data["job"]:
        data["status"] = "Eski / kayıtsız kaynak"
    data["notes"].append("İşlem durumu sayısal başarı veya bilimsel doğruluk anlamına gelmez.")
    return data


def load_run_with_details(path, cancel=lambda: False):
    result = load_tracking_results(path, cancel)
    return result, inspect_run(result, cancel)
