"""Serialize existing solver output; never propagate or estimate new states."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path


def _json(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def _csv(path, rows, columns):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _optional(value):
    return float(value) if value is not None and math.isfinite(float(value)) else ""


def write_console_bundle(result, arcs, times, truth, epoch, *, config, job_path, options, project_root):
    folder = Path(config.output_dir).resolve()
    stem = config.name
    by_id = {arc.arc_id: arc for arc in arcs}
    rows = []
    for number, solved in enumerate(result.arc_results, 1):
        arc = by_id[solved.arc_id]
        # Batch state/error is at arc start. UKF final state/error is at last
        # observation, which can precede the sampled visibility arc endpoint.
        t_s = float(max(arc.obs_data[:, 0])) if result.estimator_type == "ukf" else float(arc.t_pass_s[0])
        state = [float(v) for v in solved.estimated_state[:6]]
        pos, vel = float(solved.final_position_error_m), float(solved.final_velocity_error_mps)
        if len(state) != 6 or not all(math.isfinite(v) for v in (*state, pos, vel)) or min(pos, vel) < 0:
            raise ValueError("Çözücü sonlu bir durum/hata döndürmedi; sonuç konsolu paketi oluşturulmadı.")
        rows.append({"update_id": number, "arc_id": solved.arc_id, "t_s": t_s,
            "station_id": "", "position_error_m": pos, "velocity_error_mps": vel,
            "normalized_nis": _optional(solved.ukf_normalized_mean_nis), "accepted": "",
            "operational_success": solved.operational_success, "stop_reason": solved.stop_reason,
            "observed_station_ids": ",".join(map(str, solved.observed_station_ids)),
            "start_s": float(arc.t_pass_s[0]), "end_s": float(arc.t_pass_s[-1]),
            **dict(zip(("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"), state))})
    if not rows:
        raise ValueError("Görüntülenecek yay çözümü yok.")
    csv_path = folder / f"{stem}_arc_solutions.csv"
    _csv(csv_path, rows, list(rows[0]))
    state_path = folder / f"{stem}_truth_states.csv"
    state_columns = ("t_s", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps")
    def state_rows():
        for t_s, state in zip(times, truth):
            values = [float(t_s), *(float(v) for v in state[:6])]
            if len(values) != 7 or not all(math.isfinite(v) for v in values):
                raise ValueError("Referans durum geçmişi geçersiz.")
            yield dict(zip(state_columns, values))
    _csv(state_path, state_rows(), state_columns)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, stderr=subprocess.DEVNULL, timeout=3).decode().strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    try:
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=project_root, stderr=subprocess.DEVNULL, timeout=3).strip())
    except (OSError, subprocess.SubprocessError):
        dirty = None
    code_hashes = {}
    for relative in ("examples/run_scenario_config.py", "desktop_app/scenario_job.py", "desktop_app/services/scenario_execution.py",
                     "desktop_app/services/scenario_artifacts.py", "lunar_od/scenarios.py", "lunar_od/estimators.py", "lunar_od/filters.py"):
        source = Path(project_root) / relative
        if source.is_file():
            code_hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    _json(folder / f"{stem}_result_manifest.json", {
        "schema": "lunar-od-result-v1", "units": {"time": "s", "position": "m", "velocity": "m/s"},
        "origin": "Moon", "frame": "J2000", "epoch": epoch, "time_scale": "ET elapsed seconds",
        "seed": options["measurement_seed"], "cold_start_seed": options["cold_start_seed"],
        "producer": "desktop_app/scenario_job.py → examples/run_scenario_config.py",
        "commit": commit, "record_kind": "arc_solution", "estimator_type": result.estimator_type,
        "working_tree_dirty": dirty, "code_hashes": code_hashes,
        "code_hash_scope": "source files at artifact serialization; not a full environment snapshot",
        "state_role": "truth", "state_file": state_path.name,
        "config_file": "effective_config.json", "job_file": Path(job_path).name,
        "job_sha256": hashlib.sha256(Path(job_path).read_bytes()).hexdigest(),
        "schedule_policy": options["schedule_policy"],
        "station_id_convention": "backend zero-based observed_station_ids",
        "force_model_manifest_file": f"{stem}_force_model_manifest.json",
        "nis_scope": "arc_mean", "state_points": len(times),
    })
    return csv_path


def publish_completion(folder, run_id, result_path, artifacts):
    payload = {"schema": "lunar-od-completed-job-v1", "run_id": run_id,
        "result_path": str(result_path.resolve()), "artifacts": [str(Path(p).resolve()) for p in artifacts]}
    _json(Path(folder) / "completed.json", payload)
    # Explicit protocol, emitted only after all artifacts have been written.
    print("LUNAR_OD_RESULT " + json.dumps(payload, ensure_ascii=False), flush=True)
