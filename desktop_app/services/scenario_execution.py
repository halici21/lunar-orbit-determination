"""Persist reproducible desktop jobs and translate the form's field names."""
from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


def execution_payload(model):
    """Return canonical backend config + explicit desktop scheduling options."""
    if not re.fullmatch(r"[\w.-]{1,100}", model.name) or model.name in {".", ".."}:
        raise ValueError("Koşu adı yalnızca harf, rakam, nokta, alt çizgi ve tire içerebilir.")
    if model.network_name not in {"single", "multi"}:
        raise ValueError("Bu yürütücü single/multi ağlarını destekliyor. Özel istasyon ağı henüz bağlanmadı.")
    if model.arc_mode not in {"prescribed", "visibility"}:
        raise ValueError("Yay modu prescribed veya visibility olmalı.")
    if model.visibility_enabled is not True:
        raise ValueError("Bu yürütücü gerçek istasyon görünürlüğünü kullanır; görünürlüğü kapatma desteklenmiyor.")
    positive = ("duration_days", "state_step_s", "ephemeris_step_s", "arc_duration_h", "arc_stride_h")
    for key in positive:
        value = getattr(model, key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} sonlu ve pozitif olmalı.")
    if isinstance(model.gap_threshold_s, bool) or not isinstance(model.gap_threshold_s, (int, float)) or not math.isfinite(model.gap_threshold_s) or model.gap_threshold_s < 0:
        raise ValueError("Görünürlük boşluğu sonlu ve negatif olmayan bir süre olmalı.")
    if model.arc_mode == "prescribed" and model.arc_duration_h > model.arc_stride_h:
        raise ValueError("Planlı yay süresi yay adımından uzun olamaz.")
    if model.duration_days * 86400 / model.state_step_s > 249999:
        raise ValueError("Durum geçmişi 250.000 kayıt görüntüleme sınırını aşıyor; örnek adımını artırın.")
    if model.arc_mode == "prescribed" and model.duration_days * 24 / model.arc_stride_h > 250000:
        raise ValueError("Planlı pencere sayısı 250.000 sınırını aşıyor; yay adımını artırın.")
    if not isinstance(model.random_seed, int) or isinstance(model.random_seed, bool) or not 0 <= model.random_seed < 2**32:
        raise ValueError("Seed 0 ile 2^32−1 arasında tam sayı olmalı.")
    if not isinstance(model.min_visibility_samples, int) or model.min_visibility_samples < 2:
        raise ValueError("En az iki yay örneği gerekli.")
    if model.max_arcs is not None and (isinstance(model.max_arcs, bool) or not isinstance(model.max_arcs, int) or model.max_arcs < 1):
        raise ValueError("max_arcs pozitif tam sayı olmalı.")
    if not isinstance(model.elevation_mask_deg, (int, float)) or not math.isfinite(model.elevation_mask_deg) or not 0 <= model.elevation_mask_deg < 90:
        raise ValueError("Yükselim maskesi 0–90 derece arasında olmalı.")
    config = {
        "name": model.name, "measurement_type": model.measurement_type,
        "estimator_type": model.estimator_type, "start_mode": model.start_mode,
        "network": model.network_name, "duration_h": model.duration_days * 24,
        "sample_step_s": model.state_step_s, "noise": model.noise_enabled,
        "bias_mode": None if model.bias_mode == "none" else model.bias_mode,
        "max_iter": model.bls_max_iter, "bls_lambda0": model.bls_damping_init,
        "rtol": model.bls_rtol, "atol": model.bls_atol,
        "range_rate_physics": model.range_rate_physics,
        "count_interval_s": model.count_interval_s,
        "uplink_frequency_hz": model.uplink_frequency_hz,
        "turnaround_ratio": model.turnaround_ratio,
        "apply_light_time": model.apply_light_time,
        "apply_stellar_aberration": model.apply_stellar_aberration,
        "stellar_aberration_model": model.stellar_aberration_model,
        "ukf_alpha": model.ukf_alpha, "ukf_beta": model.ukf_beta,
        "ukf_kappa": model.ukf_kappa, "ukf_covariance_inflation": model.ukf_covariance_inflation,
        "ukf_nis_gate": model.ukf_nis_gate,
    }
    options = {key: getattr(model, key) for key in (
        "arc_mode", "arc_duration_h", "arc_stride_h", "elevation_mask_deg",
        "gap_threshold_s", "ephemeris_step_s", "min_visibility_samples", "max_arcs")}
    options["measurement_seed"] = model.random_seed
    options["cold_start_seed"] = model.random_seed
    options["schedule_policy"] = "planned_windows_with_real_visibility" if model.arc_mode == "prescribed" else "real_visibility_windows"
    return config, options


def prepare_desktop_job(model, project_root: Path, *, derived_from=None):
    config, options = execution_payload(model)
    origin = None
    if derived_from is not None:
        origin = json.loads(json.dumps(derived_from, ensure_ascii=False, allow_nan=False))
        if not isinstance(origin, dict) or origin.get("schema") != "lunar-od-derived-run-v1" or not isinstance(origin.get("original_form"), dict):
            raise ValueError("Kaynak koşu bilgisi geçersiz.")
        original, updated = origin["original_form"], asdict(model)
        origin["form_changes"] = [{"field": k, "before": original.get(k), "after": updated.get(k)}
            for k in sorted(original.keys() | updated.keys()) if original.get(k) != updated.get(k)]
    output = Path(model.output_dir).expanduser()
    if not output.is_absolute():
        output = Path(project_root) / output
    run_id = uuid.uuid4().hex
    folder = output.resolve() / "desktop_runs" / f"{model.name}_{run_id[:12]}"
    folder.mkdir(parents=True, exist_ok=False)
    config["output_dir"] = str(folder)
    payload = {"schema": "lunar-od-desktop-job-v1", "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(), "config": config,
        "options": options, "form": asdict(model)}
    if origin is not None:
        payload["derived_from"] = origin
    path = folder / "job.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, folder, run_id


def planned_segments(times, duration_h, stride_h):
    """Inclusive, non-overlapping sample windows; actual station mask stays intact."""
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (duration_h, stride_h)):
        raise ValueError("Planlı yay süreleri sonlu ve pozitif olmalı.")
    if duration_h > stride_h:
        raise ValueError("Planlı yay süresi yay adımından uzun olamaz.")
    span, stride = duration_h * 3600, stride_h * 3600
    starts, ends = [], []
    import bisect
    origin = 0.0
    if len(times) == 0:
        return starts, ends
    if times[-1] / stride > 250000:
        raise ValueError("Planlı pencere sayısı 250.000 sınırını aşıyor.")
    while origin + span <= times[-1] + 1e-9:
        lo = bisect.bisect_left(times, origin)
        hi = bisect.bisect_right(times, origin + span) - 1
        if hi >= lo:
            # Adjacent closed windows share an endpoint: keep it only once.
            if ends and lo <= ends[-1]:
                lo = ends[-1] + 1
            if hi >= lo:
                starts.append(lo)
                ends.append(hi)
        origin += stride
    return starts, ends
