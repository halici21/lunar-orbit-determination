"""Phase 17-HARDEN: pin every test import to THIS worktree (FA-06).

The worktree-local ``pytest.ini`` stops pytest from reading the workspace-root
ini (whose ``pythonpath`` pointed at the sibling ``python_port`` checkout).
This conftest restores the two import roots that ini used to provide, but for
this worktree: the repository root first, then ``desktop_app``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT / "desktop_app", _ROOT):        # inserted in reverse: ROOT ends first
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
