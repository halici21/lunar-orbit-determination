"""Run Python Lunar OD example reports from one manifest.

Run from the project root:

    python python_port/examples/run_all_experiments.py --quick
    python python_port/examples/run_all_experiments.py --full
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    script: str
    description: str
    quick: bool
    requires_spice: bool
    long_running: bool
    expected_outputs: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentRun:
    experiment_id: str
    script: str
    mode: str
    status: str
    returncode: int | None
    duration_s: float
    started_utc: str
    ended_utc: str
    requires_spice: bool
    long_running: bool
    expected_outputs: tuple[str, ...]
    stdout_tail: str = ""
    stderr_tail: str = ""


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        "synthetic_hot_start",
        "synthetic_hot_start_report.py",
        "Short synthetic cold/hot-start SRIF comparison.",
        quick=True,
        requires_spice=False,
        long_running=False,
        expected_outputs=(
            "python_port/results/synthetic_hot_start_comparison.png",
            "python_port/results/synthetic_hot_start_summary.csv",
        ),
    ),
    Experiment(
        "visibility_fixture",
        "visibility_report_from_fixture.py",
        "Fixture-based GST visibility report without live SPICE calls.",
        quick=True,
        requires_spice=False,
        long_running=False,
        expected_outputs=(
            "python_port/results/visibility_single_analysis.png",
            "python_port/results/visibility_single_summary.csv",
            "python_port/results/visibility_multi_analysis.png",
            "python_port/results/visibility_multi_summary.csv",
        ),
    ),
    Experiment(
        "formal_handoff",
        "formal_handoff_report.py",
        "SPICE visibility formal/sqrt-formal handoff comparison.",
        quick=False,
        requires_spice=True,
        long_running=False,
        expected_outputs=(
            "python_port/results/formal_handoff_comparison.png",
            "python_port/results/formal_handoff_covariance.png",
            "python_port/results/formal_handoff_summary.csv",
        ),
    ),
    Experiment(
        "quick_two_way_spice",
        "quick_two_way_spice_campaign.py",
        "Short SPICE-backed two-way counted Doppler SRIF campaign smoke test.",
        quick=False,
        requires_spice=True,
        long_running=False,
        expected_outputs=(
            "python_port/results/quick_two_way_spice_summary.csv",
            "python_port/results/quick_two_way_spice_od_summary.csv",
            "python_port/results/quick_two_way_spice_od_comparison.png",
            "python_port/results/quick_two_way_spice_visibility_summary.csv",
        ),
    ),
    Experiment(
        "formal_handoff_process_noise",
        "formal_handoff_process_noise_report.py",
        "Formal and square-root formal handoff process-noise sweep.",
        quick=False,
        requires_spice=True,
        long_running=False,
        expected_outputs=(
            "python_port/results/formal_handoff_process_noise_comparison.png",
            "python_port/results/formal_handoff_process_noise_covariance.png",
            "python_port/results/formal_handoff_process_noise_summary.csv",
        ),
    ),
    Experiment(
        "formal_bias_handoff",
        "formal_bias_handoff_report.py",
        "Augmented global range-rate bias formal handoff report.",
        quick=False,
        requires_spice=True,
        long_running=False,
        expected_outputs=(
            "python_port/results/formal_bias_handoff_comparison.png",
            "python_port/results/formal_bias_handoff_bias_recovery.png",
            "python_port/results/formal_bias_handoff_summary.csv",
            "python_port/results/formal_bias_handoff_bias_recovery.csv",
        ),
    ),
    Experiment(
        "thesis_factorial",
        "thesis_factorial_report.py",
        "Compact thesis factorial estimator/start/measurement/network matrix.",
        quick=False,
        requires_spice=True,
        long_running=False,
        expected_outputs=(
            "python_port/results/thesis_factorial_summary.png",
            "python_port/results/thesis_factorial_aggregate.csv",
            "python_port/results/thesis_factorial_detail.csv",
        ),
    ),
    Experiment(
        "compare_visibility_models",
        "compare_visibility_models.py",
        "GST vs SPICE J2000->ITRF93 visibility comparison.",
        quick=False,
        requires_spice=True,
        long_running=True,
        expected_outputs=(
            "python_port/results/visibility_gst_vs_spice_comparison.png",
            "python_port/results/visibility_gst_vs_spice_summary.csv",
        ),
    ),
    Experiment(
        "long_visibility",
        "long_visibility_report.py",
        "Long SPICE visibility reports for single and network cases.",
        quick=False,
        requires_spice=True,
        long_running=True,
        expected_outputs=(
            "python_port/results/long_single_canberra_visibility_analysis.png",
            "python_port/results/long_single_canberra_visibility_summary.csv",
            "python_port/results/long_multi_dsn_itu_visibility_analysis.png",
            "python_port/results/long_multi_dsn_itu_visibility_summary.csv",
            "python_port/results/long_multi_extended_visibility_analysis.png",
            "python_port/results/long_multi_extended_visibility_summary.csv",
        ),
    ),
    Experiment(
        "long_visibility_od",
        "long_visibility_od_report.py",
        "Long SPICE visibility position-only SRIF OD report.",
        quick=False,
        requires_spice=True,
        long_running=True,
        expected_outputs=(
            "python_port/results/long_visibility_position_od_comparison.png",
            "python_port/results/long_visibility_position_od_summary.csv",
        ),
    ),
    Experiment(
        "long_visibility_rr_od",
        "long_visibility_rr_od_report.py",
        "Long SPICE visibility range-rate SRIF OD report.",
        quick=False,
        requires_spice=True,
        long_running=True,
        expected_outputs=(
            "python_port/results/long_visibility_rr_od_comparison.png",
            "python_port/results/long_visibility_rr_od_summary.csv",
        ),
    ),
    Experiment(
        "long_visibility_rr_noise_bias",
        "long_visibility_rr_noise_bias_report.py",
        "Long noisy/station-biased range-rate SRIF OD report.",
        quick=False,
        requires_spice=True,
        long_running=True,
        expected_outputs=(
            "python_port/results/long_visibility_rr_noise_bias_comparison.png",
            "python_port/results/long_visibility_rr_noise_bias_summary.csv",
            "python_port/results/long_visibility_rr_noise_bias_recovery.png",
            "python_port/results/long_visibility_rr_noise_bias_recovery.csv",
        ),
    ),
    Experiment(
        "campaign_4day_visibility_rr",
        "campaign_4day_visibility_rr_report.py",
        "Four-day SPICE visibility and representative clean RR SRIF campaign report.",
        quick=False,
        requires_spice=True,
        long_running=True,
        expected_outputs=(
            "python_port/results/campaign_4day_summary.csv",
            "python_port/results/campaign_4day_rr_od_summary.csv",
            "python_port/results/campaign_4day_rr_od_comparison.png",
            "python_port/results/campaign_4day_single_canberra_visibility_analysis.png",
            "python_port/results/campaign_4day_single_canberra_visibility_summary.csv",
            "python_port/results/campaign_4day_multi_dsn_itu_visibility_analysis.png",
            "python_port/results/campaign_4day_multi_dsn_itu_visibility_summary.csv",
            "python_port/results/campaign_4day_multi_extended_visibility_analysis.png",
            "python_port/results/campaign_4day_multi_extended_visibility_summary.csv",
        ),
    ),
    Experiment(
        "campaign_28day_itu",
        "campaign_28day_itu_report.py",
        "Twenty-eight-day ITU-only SPICE visibility and representative clean RR SRIF report.",
        quick=False,
        requires_spice=True,
        long_running=True,
        expected_outputs=(
            "python_port/results/campaign_28day_itu_summary.csv",
            "python_port/results/campaign_28day_itu_selected_arcs.csv",
            "python_port/results/campaign_28day_itu_rr_od_summary.csv",
            "python_port/results/campaign_28day_itu_rr_od_comparison.png",
            "python_port/results/campaign_28day_itu_visibility_analysis.png",
            "python_port/results/campaign_28day_itu_visibility_summary.csv",
        ),
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    experiments = select_experiments(args.mode, only=_split_csv_arg(args.only), skip=_split_csv_arg(args.skip))

    if args.list:
        print_manifest(experiments)
        return 0

    runs = run_experiments(
        experiments,
        mode=args.mode,
        dry_run=args.dry_run,
        stop_on_fail=args.stop_on_fail,
        timeout_s=args.timeout_s,
        python_executable=args.python_executable,
        project_root=_project_root(),
    )
    csv_path, json_path = write_run_summaries(runs, args.summary_dir)
    _print_run_summary(runs, csv_path, json_path)
    return 1 if any(run.status == "failed" for run in runs) else 0


def select_experiments(
    mode: str,
    *,
    only: Sequence[str] = (),
    skip: Sequence[str] = (),
    manifest: Sequence[Experiment] = EXPERIMENTS,
) -> tuple[Experiment, ...]:
    """Select experiments for quick/full orchestration."""
    if mode not in {"quick", "full"}:
        raise ValueError("mode must be 'quick' or 'full'.")

    selected = tuple(manifest if mode == "full" else [experiment for experiment in manifest if experiment.quick])
    if only:
        selected = tuple(experiment for experiment in manifest if _matches_any(experiment, only))
    if skip:
        selected = tuple(experiment for experiment in selected if not _matches_any(experiment, skip))
    return selected


def run_experiments(
    experiments: Sequence[Experiment],
    *,
    mode: str,
    dry_run: bool,
    stop_on_fail: bool,
    timeout_s: float | None,
    python_executable: str,
    project_root: Path,
) -> tuple[ExperimentRun, ...]:
    """Run selected experiments as subprocesses and collect status rows."""
    runs: list[ExperimentRun] = []
    for index, experiment in enumerate(experiments, start=1):
        print(f"[{index}/{len(experiments)}] {experiment.experiment_id}: {experiment.script}")
        if dry_run:
            runs.append(_planned_run(experiment, mode))
            continue

        run = _run_one_experiment(
            experiment,
            mode=mode,
            timeout_s=timeout_s,
            python_executable=python_executable,
            project_root=project_root,
        )
        runs.append(run)
        if stop_on_fail and run.status == "failed":
            break
    return tuple(runs)


def write_run_summaries(runs: Sequence[ExperimentRun], summary_dir) -> tuple[Path, Path]:
    """Write CSV and JSON orchestration summaries."""
    summary_dir = Path(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "run_all_experiments_summary.csv"
    json_path = summary_dir / "run_all_experiments_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "experiment_id",
                "script",
                "mode",
                "status",
                "returncode",
                "duration_s",
                "started_utc",
                "ended_utc",
                "requires_spice",
                "long_running",
                "expected_outputs",
            ]
        )
        for run in runs:
            writer.writerow(
                [
                    run.experiment_id,
                    run.script,
                    run.mode,
                    run.status,
                    "" if run.returncode is None else run.returncode,
                    f"{run.duration_s:.6f}",
                    run.started_utc,
                    run.ended_utc,
                    run.requires_spice,
                    run.long_running,
                    ";".join(run.expected_outputs),
                ]
            )

    payload = [asdict(run) for run in runs]
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return csv_path, json_path


def print_manifest(experiments: Sequence[Experiment]) -> None:
    """Print a compact manifest table."""
    if not experiments:
        print("No experiments selected.")
        return
    for experiment in experiments:
        flags = []
        if experiment.quick:
            flags.append("quick")
        if experiment.requires_spice:
            flags.append("spice")
        if experiment.long_running:
            flags.append("long")
        print(f"{experiment.experiment_id:34s} {experiment.script:44s} [{', '.join(flags) or 'standard'}]")
        print(f"  {experiment.description}")


def _run_one_experiment(
    experiment: Experiment,
    *,
    mode: str,
    timeout_s: float | None,
    python_executable: str,
    project_root: Path,
) -> ExperimentRun:
    started = _utc_now()
    t0 = time.perf_counter()
    script_path = project_root / "python_port" / "examples" / experiment.script
    try:
        completed = subprocess.run(
            [python_executable, str(script_path)],
            cwd=project_root,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        status = "passed" if completed.returncode == 0 else "failed"
        returncode = int(completed.returncode)
        stdout_tail = _tail(completed.stdout)
        stderr_tail = _tail(completed.stderr)
    except subprocess.TimeoutExpired as exc:
        status = "failed"
        returncode = None
        stdout_tail = _tail(exc.stdout or "")
        stderr_tail = _tail(exc.stderr or f"Timed out after {timeout_s} seconds.")

    ended = _utc_now()
    return ExperimentRun(
        experiment_id=experiment.experiment_id,
        script=experiment.script,
        mode=mode,
        status=status,
        returncode=returncode,
        duration_s=time.perf_counter() - t0,
        started_utc=started,
        ended_utc=ended,
        requires_spice=experiment.requires_spice,
        long_running=experiment.long_running,
        expected_outputs=experiment.expected_outputs,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )


def _planned_run(experiment: Experiment, mode: str) -> ExperimentRun:
    now = _utc_now()
    return ExperimentRun(
        experiment_id=experiment.experiment_id,
        script=experiment.script,
        mode=mode,
        status="planned",
        returncode=None,
        duration_s=0.0,
        started_utc=now,
        ended_utc=now,
        requires_spice=experiment.requires_spice,
        long_running=experiment.long_running,
        expected_outputs=experiment.expected_outputs,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Lunar OD Python example reports from one manifest.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--quick", action="store_const", const="quick", dest="mode", help="Run fast non-long reports.")
    mode_group.add_argument("--full", action="store_const", const="full", dest="mode", help="Run the full report manifest.")
    parser.set_defaults(mode="quick")
    parser.add_argument("--list", action="store_true", help="Print selected experiments and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Write planned summary rows without running scripts.")
    parser.add_argument("--only", default="", help="Comma-separated experiment ids or script stems to run.")
    parser.add_argument("--skip", default="", help="Comma-separated experiment ids or script stems to skip.")
    parser.add_argument("--summary-dir", default=str(Path("python_port") / "results"), help="Directory for CSV/JSON run summaries.")
    parser.add_argument("--timeout-s", type=float, default=None, help="Optional timeout in seconds per experiment.")
    parser.add_argument("--stop-on-fail", action="store_true", help="Stop after the first failed experiment.")
    parser.add_argument("--python-executable", default=sys.executable, help="Python executable used for child scripts.")
    return parser.parse_args(argv)


def _split_csv_arg(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _matches_any(experiment: Experiment, selectors: Sequence[str]) -> bool:
    names = {experiment.experiment_id, Path(experiment.script).stem, experiment.script}
    return any(selector in names for selector in selectors)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _print_run_summary(runs: Sequence[ExperimentRun], csv_path: Path, json_path: Path) -> None:
    passed = sum(run.status == "passed" for run in runs)
    failed = sum(run.status == "failed" for run in runs)
    planned = sum(run.status == "planned" for run in runs)
    print(f"Summary: passed={passed}, failed={failed}, planned={planned}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    raise SystemExit(main())
