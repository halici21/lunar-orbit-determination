"""Import GitHub Actions run history into results/test_runs/ for the desktop archive.

Read-only on GitHub (``gh run list``); writes one ``ci_<id>.json`` per finished
run, skipping files that already exist. Usage::

    python tools/import_ci_test_runs.py [--limit 200]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "test_runs"
FIELDS = "databaseId,workflowName,event,status,conclusion,headBranch,headSha,createdAt,updatedAt,url,displayTitle"


def _seconds(start, end):
    try:
        return round((datetime.fromisoformat(end.replace("Z", "+00:00"))
                      - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds(), 1)
    except (ValueError, AttributeError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args(argv)
    runs = json.loads(subprocess.run(["gh", "run", "list", "--limit", str(args.limit), "--json", FIELDS],
                                     cwd=ROOT, capture_output=True, text=True, check=True).stdout)
    OUT.mkdir(parents=True, exist_ok=True)
    written = 0
    for run in runs:
        if run.get("status") != "completed":
            continue
        path = OUT / f"ci_{run['databaseId']}.json"
        if path.exists():
            continue
        record = {"schema": "lunar-od-test-run-v1", "source": "ci", "workflow": run.get("workflowName"),
                  "event": run.get("event"), "conclusion": run.get("conclusion"), "title": run.get("displayTitle"),
                  "branch": run.get("headBranch"), "commit": run.get("headSha"), "url": run.get("url"),
                  "started_at": run.get("createdAt"), "finished_at": run.get("updatedAt"),
                  "duration_s": _seconds(run.get("createdAt"), run.get("updatedAt"))}
        path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        written += 1
    print(f"{written} new CI run(s) written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
