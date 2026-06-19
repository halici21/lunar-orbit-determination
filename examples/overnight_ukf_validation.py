"""Run long UKF validation campaigns and write an execution manifest.

This orchestrates the expensive checks that are intentionally outside the
normal unit-test loop:

- full slow regression suite
- larger UKF stress Monte Carlo over seeds, station network, cold-start scale,
  and SPICE Earth-position mismatch
- SPICE mismatch sweep
- long two-way counted Doppler SPICE campaign
- 28-day long-duration all-arc campaign
- real tracking CSV ingestion discovery, when local tracking files exist

Run from the project root:

    python python_port/examples/overnight_ukf_validation.py
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_PORT = PROJECT_ROOT / "python_port"
RESULTS_DIR = PYTHON_PORT / "results" / "overnight_ukf_validation"


@dataclass(frozen=True)
class StepResult:
    name: str
    command: str
    returncode: int
    elapsed_s: float
    stdout_log: str
    stderr_log: str


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    scenario_path = _write_ukf_scenario_config()
    tracking_manifest = _write_tracking_ingestion_manifest()

    trials = int(os.environ.get("LUNAR_OD_OVERNIGHT_MC_TRIALS", "12"))
    max_workers = int(os.environ.get("LUNAR_OD_OVERNIGHT_MAX_WORKERS", "1"))
    skip_28day = _env_flag("LUNAR_OD_OVERNIGHT_SKIP_28DAY")
    skip_slow_tests = _env_flag("LUNAR_OD_OVERNIGHT_SKIP_SLOW_TESTS")

    steps: list[tuple[str, list[str], dict[str, str] | None]] = []
    if not skip_slow_tests:
        steps.append(
            (
                "full_slow_regression_suite",
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    str(PYTHON_PORT / "tests"),
                    "-t",
                    str(PYTHON_PORT),
                ],
                {"LUNAR_OD_RUN_SLOW_TESTS": "1"},
            )
        )
    steps.extend(
        [
            (
                "long_two_way_spice_campaign",
                [
                    sys.executable,
                    str(PYTHON_PORT / "examples" / "quick_two_way_spice_campaign.py"),
                    "--duration-h",
                    os.environ.get("LUNAR_OD_OVERNIGHT_TWOWAY_DURATION_H", "72"),
                    "--sample-step-s",
                    "600",
                    "--ephemeris-step-s",
                    "3600",
                    "--count-interval-s",
                    "30",
                    "--max-arcs",
                    "8",
                    "--max-iter",
                    "12",
                    "--noise",
                    "--seed",
                    "20260609",
                ],
                None,
            ),
            (
                "ukf_spice_ephemeris_mismatch_sweep",
                [
                    sys.executable,
                    str(PYTHON_PORT / "examples" / "ukf_spice_mismatch_campaign.py"),
                    str(scenario_path),
                    "--earth-position-bias-m",
                    "0,100,1000,5000,10000",
                ],
                None,
            ),
            (
                "ukf_stress_monte_carlo",
                [
                    sys.executable,
                    str(PYTHON_PORT / "examples" / "ukf_stress_monte_carlo_campaign.py"),
                    str(scenario_path),
                    "--trials",
                    str(trials),
                    "--max-workers",
                    str(max_workers),
                    "--base-seed",
                    "20260609",
                    "--earth-position-bias-m",
                    "0,1000,5000",
                    "--cold-start-scale",
                    "0.5,1,2,4",
                    "--continue-on-error",
                    "--output-dir",
                    str(RESULTS_DIR),
                ],
                None,
            ),
        ]
    )
    if not skip_28day:
        steps.append(
            (
                "long_28day_all_arc_hot_campaign",
                [
                    sys.executable,
                    str(PYTHON_PORT / "examples" / "campaign_28day_itu_all_arc_hot_report.py"),
                    "--max-iter",
                    os.environ.get("LUNAR_OD_OVERNIGHT_28DAY_MAX_ITER", "20"),
                ],
                None,
            )
        )

    step_results = []
    for name, command, extra_env in steps:
        step_results.append(_run_step(name, command, extra_env))

    manifest = {
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_config": str(scenario_path),
        "tracking_ingestion_manifest": str(tracking_manifest),
        "results_dir": str(RESULTS_DIR),
        "monte_carlo_trials_per_case": trials,
        "monte_carlo_max_workers": max_workers,
        "steps": [asdict(result) for result in step_results],
        "generated_files": _generated_files(),
    }
    manifest_path = RESULTS_DIR / "overnight_ukf_validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_manifest_csv(step_results, RESULTS_DIR / "overnight_ukf_validation_steps.csv")

    failed = [result for result in step_results if result.returncode != 0]
    print(f"Wrote {manifest_path}")
    print(f"Wrote {RESULTS_DIR / 'overnight_ukf_validation_steps.csv'}")
    if failed:
        print(f"Completed with {len(failed)} failing step(s); see logs in {RESULTS_DIR}.")
        return 1
    print("All overnight validation steps completed successfully.")
    return 0


def _write_ukf_scenario_config() -> Path:
    config = {
        "name": "overnight_ukf_multi_rr",
        "measurement_type": "range_rate",
        "estimator_type": "ukf",
        "start_mode": "hot",
        "network": "multi",
        "duration_h": float(os.environ.get("LUNAR_OD_OVERNIGHT_UKF_DURATION_H", "12")),
        "sample_step_s": float(os.environ.get("LUNAR_OD_OVERNIGHT_UKF_SAMPLE_STEP_S", "1800")),
        "max_iter": 12,
        "rtol": 1e-8,
        "atol": 1e-10,
        "noise": True,
        "bias_mode": "station_full",
        "range_rate_physics": os.environ.get(
            "LUNAR_OD_OVERNIGHT_UKF_RANGE_RATE_PHYSICS",
            "geometric_instantaneous",
        ),
        "count_interval_s": 30.0,
        "two_way_local_state_model": "taylor3",
        "ukf_alpha": 0.35,
        "ukf_beta": 2.0,
        "ukf_kappa": 0.0,
        "ukf_covariance_inflation": 1.001,
        "ukf_process_noise_model": "continuous_white_acceleration",
        "ukf_acceleration_psd_m2_s3": 1e-9,
        "ukf_adaptive_process_noise": True,
        "ukf_initial_process_noise_scale": 1.0,
        "ukf_min_process_noise_scale": 0.25,
        "ukf_max_process_noise_scale": 25.0,
        "ukf_process_noise_adaptation_gain": 0.2,
        "ukf_adaptive_measurement_noise": True,
        "ukf_max_measurement_noise_scale": 50.0,
        "ukf_component_nis_gate": 36.0,
        "ukf_component_gate_mode": "conditional",
        "ukf_robust_measurement_update": True,
        "ukf_robust_loss": "student_t",
        "ukf_robust_student_t_dof": 5.0,
        "ukf_robust_huber_threshold": 3.0,
        "ukf_robust_min_component_weight": 0.05,
        "ukf_covariance_form": "square_root",
        "ukf_auto_bias_constraints": True,
        "ukf_bias_freeze_relative_information": 1e-12,
        "ukf_bias_regularize_relative_information": 1e-5,
        "ukf_bias_regularization_std": 1.0,
        "output_dir": str(RESULTS_DIR),
    }
    path = RESULTS_DIR / "overnight_ukf_multi_rr.scenario.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def _write_tracking_ingestion_manifest() -> Path:
    candidates = []
    search_roots = [
        PYTHON_PORT / "data",
        PYTHON_PORT / "tracking",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "tracking",
    ]
    for root in search_roots:
        if root.exists():
            candidates.extend(sorted(root.rglob("*.csv")))

    output_path = RESULTS_DIR / "tracking_ingestion_manifest.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "status", "details"])
        if not candidates:
            writer.writerow(["", "not_found", "No local real tracking CSV files found in data/tracking roots."])
        else:
            for path in candidates:
                writer.writerow([str(path), "candidate_found", "Manual measurement_type/station mapping still required."])
    return output_path


def _run_step(name: str, command: list[str], extra_env: dict[str, str] | None) -> StepResult:
    print(f"[overnight] START {name}: {' '.join(command)}", flush=True)
    stdout_log = RESULTS_DIR / f"{name}.stdout.log"
    stderr_log = RESULTS_DIR / f"{name}.stderr.log"
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    start = time.perf_counter()
    with stdout_log.open("w", encoding="utf-8") as stdout, stderr_log.open("w", encoding="utf-8") as stderr:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    elapsed = time.perf_counter() - start
    print(f"[overnight] END {name}: returncode={process.returncode} elapsed_s={elapsed:.1f}", flush=True)
    return StepResult(
        name=name,
        command=" ".join(command),
        returncode=int(process.returncode),
        elapsed_s=float(elapsed),
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
    )


def _write_manifest_csv(step_results: list[StepResult], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "returncode", "elapsed_s", "stdout_log", "stderr_log", "command"])
        for result in step_results:
            writer.writerow(
                [
                    result.name,
                    result.returncode,
                    result.elapsed_s,
                    result.stdout_log,
                    result.stderr_log,
                    result.command,
                ]
            )


def _generated_files() -> list[str]:
    return [str(path) for path in sorted(RESULTS_DIR.rglob("*")) if path.is_file()]


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
