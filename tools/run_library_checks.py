"""Kernel-independent regression tier; explicit kernel integrations stay separate."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FILES = ("test_config.py", "test_geometry.py", "test_orbit.py", "test_dynamics.py",
         "test_estimators.py", "test_filters.py", "test_measurements.py", "test_quickstart.py",
         "test_paired_benchmark.py")
# These four existing integration tests load external kernels unconditionally.
# Their fixture-related names do not mean that their ephemerides are bundled.
KERNEL_INTEGRATIONS = (
    "tests/test_estimators.py::EstimatorTests::test_position_srif_reduces_cost_on_spice_fixture_arc",
    "tests/test_estimators.py::EstimatorTests::test_range_rate_srif_reduces_cost_on_spice_fixture_arc",
    "tests/test_measurements.py::MeasurementTests::test_position_measurement_generation_and_clean_residual_closure",
    "tests/test_measurements.py::MeasurementTests::test_range_rate_measurement_generation_and_clean_residual_closure",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junitxml", default="outputs/library.xml")
    parser.add_argument("--with-kernels", action="store_true")
    args = parser.parse_args()
    command = [sys.executable, "-m", "pytest", *("tests/" + name for name in FILES),
               "-q", "-ra", "--junitxml=" + args.junitxml]
    if not args.with_kernels:
        for case in KERNEL_INTEGRATIONS:
            command += ["--deselect", case]
        print("Library tier excludes four explicitly listed external-SPICE integration cases.", flush=True)
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
