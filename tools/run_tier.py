"""Run one verification tier of the Lunar OD test suite (WP7.2, REQ-SW-002).

    python tools/run_tier.py commit            # every push: L0 + L1 + L7, kernel-free
    python tools/run_tier.py pr                # every PR: L2 + kernel-backed L0/L1/L7
    python tools/run_tier.py nightly           # L3 + L4 + full regression incl. slow
    python tools/run_tier.py release           # L5 real data
    python tools/run_tier.py --list            # print every tier's selection

Levels and resource markers come from tests/levels.py (applied by tests/conftest.py).
Every tier runs with --strict-levels, so an unclassified test file stops the run with
the table row that fixes it. Kernel-backed tiers pass --require-kernels: a missing
kernel FAILS every `spice` test instead of skipping it (REQ-SW-003). Kernels come from
LUNAR_OD_KERNEL_DIR (CI: tools/fetch_kernels.py); see docs/assurance/test-levels.md.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

#: tier -> (pytest -m expression, kernels required, slow tests enabled, purpose)
TIERS: dict[str, tuple[str, bool, bool, str]] = {
    "commit": (
        "(analytic or derivative or contract_negative) and not spice and not campaign and not qt and not slow",
        False, False,
        "every push: L0 + L1 + L7 that need no kernels, campaign data or desktop"),
    "pr": (
        "(external_oracle or ((analytic or derivative or contract_negative) and spice)) "
        "and not campaign and not qt and not slow",
        True, False,
        "every pull request: L2 plus the kernel-dependent L0/L1/L7, on verified kernels"),
    "nightly": (
        "",
        True, True,
        "nightly: L3 + L4 and the full regression (every level, slow tests on)"),
    "release": (
        "real_data",
        True, True,
        "release: L5 real-data tests"),
}


def command_for(tier: str, extra: list[str] | None = None, junitxml: str | None = None) -> list[str]:
    expression, kernels, _slow, _purpose = TIERS[tier]
    command = [sys.executable, "-m", "pytest", "--strict-levels", "-ra"]
    if expression:
        command += ["-m", expression]
    if kernels:
        command.append("--require-kernels")
    if junitxml:
        command.append("--junitxml=" + junitxml)
    return command + list(extra or [])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tier", nargs="?", choices=sorted(TIERS))
    parser.add_argument("--list", action="store_true", help="print every tier's selection and exit")
    parser.add_argument("--junitxml", help="save a JUnit report")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER, help="extra pytest arguments after --")
    args = parser.parse_args(argv)
    if args.list or not args.tier:
        for name, (expression, kernels, slow, purpose) in TIERS.items():
            print(f"{name:8s} {purpose}\n         -m {expression or '(all tests)'!r}"
                  f"  kernels={'required' if kernels else 'none'}  slow={'on' if slow else 'off'}")
        return 0 if args.list else 2
    extra = [a for a in args.pytest_args if a != "--"]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    if TIERS[args.tier][2]:
        env["LUNAR_OD_RUN_SLOW_TESTS"] = "1"
    if args.junitxml:
        Path(args.junitxml).resolve().parent.mkdir(parents=True, exist_ok=True)
    command = command_for(args.tier, extra, args.junitxml)
    print(f"Tier {args.tier}: {TIERS[args.tier][3]}", flush=True)
    print(" ".join(command), flush=True)
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
