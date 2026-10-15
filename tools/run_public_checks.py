"""Run the public Windows checks without private campaign or kernel inputs."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TEST_FILES = (
    "test_config.py", "test_geometry.py", "test_orbit.py", "test_dynamics.py", "test_quickstart.py",
    "test_legacy_layout.py", "test_paired_benchmark.py", "test_desktop_tutorial.py",
    "test_scenario_config.py", "test_scenario_config_harmonics.py",
    "test_compact_recovery.py", "test_desktop_workflow.py", "test_large_workspace.py",
    "test_recorded_run_workspace.py", "test_result_summary.py",
    "test_run_workspace_resume.py", "test_tracking_console.py",
    "test_tracking_render_semantics.py", "test_workflow_polish.py",
    "test_workspace_appearance.py", "test_workspace_session.py", "test_station_catalog_filters.py",
    "test_pytest_desktop_app.py", "test_environment_readiness.py", "test_mission_overview.py",
)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junitxml", type=Path, help="Save a JUnit report")
    args = parser.parse_args()
    command = [sys.executable, "-m", "pytest", "-p", "tests.desktop_preferences"]
    command.extend(str(ROOT / "tests" / name) for name in TEST_FILES)
    if args.junitxml:
        report = args.junitxml.resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        command.append("--junitxml=" + str(report))
    command.extend(["-q", "-ra"])
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["QT_QPA_PLATFORM"] = "offscreen"
    if os.name == "nt":
        fonts = Path(env.get("WINDIR", "C:/Windows")) / "Fonts"
        if fonts.is_dir():
            env["QT_QPA_FONTDIR"] = str(fonts)
    with tempfile.TemporaryDirectory(prefix="lunar-od-public-") as temporary:
        # Every external-data override points into a known, empty temporary tree.
        # The scientific code and its skip policy are left unchanged.
        data = Path(temporary)
        for key, name in (("LUNAR_OD_PROJECT_ROOT", "project"),
                          ("LUNAR_OD_CAMPAIGN_ROOT", "campaign"),
                          ("LUNAR_OD_GRAVITY_ROOT", "gravity"),
                          ("LUNAR_OD_KERNEL_DIR", "kernels")):
            folder = data / name
            folder.mkdir()
            env[key] = str(folder)
        print("Public checks: committed fixtures only; external data isolated.", flush=True)
        return subprocess.call(command, cwd=ROOT, env=env)

if __name__ == "__main__":
    raise SystemExit(main())
