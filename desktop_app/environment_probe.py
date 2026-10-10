"""Machine-readable preflight entry point, isolated from the desktop's SPICE pool."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-dir")
    parser.add_argument("--duration-h", type=float)
    parser.add_argument("--state-step-s", type=float, default=600.0)
    parser.add_argument("--ephemeris-step-s", type=float, default=3600.0)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    from desktop_app.services.environment_readiness import inspect_environment
    report = inspect_environment(**vars(args))
    print("LUNAR_OD_ENVIRONMENT " + json.dumps(report, ensure_ascii=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
