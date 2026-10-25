"""Phase 17-HARDEN: pin every test import to THIS worktree (FA-06).

The worktree-local ``pyproject.toml`` stops pytest from reading the workspace-root
ini (whose ``pythonpath`` pointed at the sibling ``python_port`` checkout).
This conftest restores the two import roots that ini used to provide, but for
this worktree: the repository root first, then ``desktop_app``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT / "studies" / "phase17" / "scripts", _ROOT / "desktop_app", _ROOT):        # inserted in reverse: ROOT ends first
    _entry = str(_path)
    if _entry in sys.path:
        sys.path.remove(_entry)
    sys.path.insert(0, _entry)

# HARDEN-D12: import the external campaign module through the guarded loader
# (it prepends a sibling checkout to sys.path before importing lunar_od).
# The module lives in <campaign>/06_scripts and itself imports
# phase6_scenario_comparison from this worktree's examples/, so both are made
# importable first, APPENDED so neither can shadow this worktree.  Without this
# the guard ran only when the caller exported PYTHONPATH; otherwise it failed
# with a swallowed ModuleNotFoundError and a later direct import left the stale
# sibling entries on sys.path.  A missing campaign is the only quiet skip.
from lunar_od.project_paths import campaign_root as _campaign_root  # noqa: E402
from lunar_od.project_paths import import_campaign_module as _import_campaign  # noqa: E402

_CAMPAIGN_SCRIPTS = _campaign_root() / "06_scripts"
if (_CAMPAIGN_SCRIPTS / "od_gravity_covariance_campaign.py").is_file():
    for _path in (_CAMPAIGN_SCRIPTS, _ROOT / "examples"):
        if str(_path) not in sys.path:
            sys.path.append(str(_path))
    _import_campaign()


# --- Test-run record for the desktop Arşiv › Test koşuları tab ---------------
# One small JSON per pytest session in results/test_runs/ (gitignored). It never
# affects the run: any failure here is swallowed. LUNAR_OD_RECORD_TESTS=0 skips it.
import time as _time

_SESSION_STARTED = [None]


def pytest_sessionstart(session):
    _SESSION_STARTED[0] = (_time.time(), _time.perf_counter())


def pytest_sessionfinish(session, exitstatus):
    import os
    if os.environ.get("LUNAR_OD_RECORD_TESTS", "1") == "0" or hasattr(session.config, "workerinput"):
        return
    try:
        import json
        import subprocess
        from datetime import datetime, timezone

        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        stats = getattr(reporter, "stats", {}) if reporter else {}
        count = lambda key: sum(1 for r in stats.get(key, ()) if getattr(r, "when", "call") in ("call", "setup", "teardown"))

        def git(*args):
            out = subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, text=True, timeout=5)
            return out.stdout.strip() if out.returncode == 0 else ""

        started_wall, started = _SESSION_STARTED[0] or (_time.time(), _time.perf_counter())
        record = {
            "schema": "lunar-od-test-run-v1", "source": "local",
            "started_at": datetime.fromtimestamp(started_wall, timezone.utc).isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(_time.perf_counter() - started, 1),
            "exit_status": int(exitstatus),
            "passed": count("passed"), "failed": count("failed"), "skipped": count("skipped"),
            "errors": count("error"), "xfailed": count("xfailed"), "xpassed": count("xpassed"),
            "collected": getattr(session, "testscollected", None),
            "args": [str(a) for a in session.config.invocation_params.args][:40],
            "commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "python": sys.version.split()[0],
            "failures": [r.nodeid for key in ("failed", "error") for r in stats.get(key, ())][:100],
        }
        folder = _ROOT / "results" / "test_runs"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (folder / f"local_{stamp}_{os.getpid()}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    except Exception:
        pass
