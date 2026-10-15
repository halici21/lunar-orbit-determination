"""Produce two real synthetic SR-UKF runs for the desktop, without external data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.benchmark_estimators import PLAN, run_trial
from examples.quickstart_od import CONFIG, sha256, source_identity, write_csv
from desktop_app.services.scenario_artifacts import publish_completion


def run(output, run_id):
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    base = json.loads(CONFIG.read_text(encoding="utf-8"))
    paths = []
    print("Sentetik eğitim: Ay noktasal kütle, sabit istasyonlar, görünürlük filtresi yok; doğruluk düzeyi 0.", flush=True)
    for seed in plan["seeds"][:2]:
        rows = run_trial(seed, 1.0, plan, base, output)
        metric = next(r for r in rows if r["estimator"] == "SR-UKF")
        if metric["failure"]:
            raise RuntimeError(metric["failure"])
        stem = f"tutorial_seed_{seed}"
        with np.load(output / f"seed-{seed}-noise-1.npz") as data:
            times, states, truth = data["ukf_times_s"], data["ukf_states"], data["truth_m_mps"]
            reference = truth[np.searchsorted(data["times_s"], times)]
            error = states[:, :6] - reference
            csv_path = output / f"{stem}_ukf_updates.csv"
            write_csv(csv_path, ["update_id", "arc_id", "t_s", "station_id", "position_error_m",
                "velocity_error_mps", "nis", "normalized_nis", "accepted", "covariance_condition_number"],
                ([i + 1, 1, t, int(obs[4]), np.linalg.norm(e[:3]), np.linalg.norm(e[3:]), nis,
                  nis / 3., bool(accepted), np.linalg.cond(cov)]
                 for i, (t, obs, e, nis, accepted, cov) in enumerate(zip(times, data["observations"], error,
                     data["ukf_nis"], data["ukf_accepted"], data["ukf_covariances"]))))
            write_csv(output / f"{stem}_truth.csv", ["t_s", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"],
                      ([t, *state] for t, state in zip(data["times_s"], truth)))
        config = {**base, "seed": seed, "estimator_type": "ukf", "explicit_prior_std_m_mps": plan["initial_std_m_mps"],
                  "process_noise": None, "ukf": plan["ukf"], "truth_level": 0,
                  "scope": "synthetic identity-frame tutorial; all observations retained"}
        config_name, job_name = f"{stem}_config.json", f"{stem}_job.json"
        (output / config_name).write_text(json.dumps(config, indent=2), encoding="utf-8")
        job = {"schema": "lunar-od-tutorial-job-v1", "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
               "options": {"measurement_seed": seed, "cold_start_seed": seed}}
        (output / job_name).write_text(json.dumps(job, indent=2), encoding="utf-8")
        source = source_identity()
        source["source_sha256"].update({name: sha256(ROOT / name) for name in
            ("examples/desktop_tutorial.py", "examples/benchmark_estimators.py")})
        manifest = {"schema": "lunar-od-result-v1", "units": {"time": "s", "position": "m", "velocity": "m/s"},
            "origin": "Moon", "frame": "Synthetic inertial / identity Earth-fixed transform",
            "epoch": "Synthetic t=0", "time_scale": "elapsed seconds", "seed": seed, "cold_start_seed": seed,
            "producer": "examples/desktop_tutorial.py (Level 0 synthetic tutorial)", "commit": source["git_commit"],
            "working_tree_dirty": source["git_dirty"], "code_hashes": source["source_sha256"],
            "record_kind": "ukf_update", "estimator_type": "ukf", "nis_scope": "update / 3 dof",
            "schedule_policy": "all synthetic scheduled observations; no visibility filtering",
            "station_id_convention": "one-based synthetic station IDs", "state_role": "truth",
            "state_file": f"{stem}_truth.csv", "config_file": config_name, "job_file": job_name,
            "job_sha256": sha256(output / job_name)}
        (output / f"{stem}_result_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        paths.append(csv_path)
        print(f"Seed {seed}: {len(times)} gerçek SR-UKF güncellemesi; son konum hatası {metric['position_error_m']:.3f} m.", flush=True)
    (output / "tutorial_runs.json").write_text(json.dumps([p.name for p in paths], indent=2), encoding="utf-8")
    publish_completion(output, run_id, paths[0], paths)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run(args.output.resolve(), args.run_id)
