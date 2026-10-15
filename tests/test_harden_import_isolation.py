"""Phase 17-HARDEN - cross-worktree import isolation.

Before HARDEN, a test run from this worktree picked up the workspace-root
pytest.ini, whose ``pythonpath`` put the SIBLING ``python_port`` checkout on
sys.path.  ``lunar_od`` happened to resolve locally (cwd first), but the
desktop-app modules (``app``, ``controllers.*``, ``services.*``) resolved from
the sibling, so those tests validated the wrong checkout.  FA-06 guarded only
``lunar_od``.  These tests guard every import root the suite relies on.

HARDEN-D12: the external campaign module runs ``sys.path.insert(0, <removed
sibling checkout>)`` (and ``<...>/examples``) at import.  conftest.py must bind it
through the guarded loader on a PLAIN ``pytest`` invocation; before the CD1
closure the guard ran only when the caller exported the campaign scripts on
PYTHONPATH, and otherwise a later direct import left the stale entries behind.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lunar_od.project_paths import campaign_root

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_SCRIPT = campaign_root() / "06_scripts" / "od_gravity_covariance_campaign.py"
needs_campaign = pytest.mark.skipif(not CAMPAIGN_SCRIPT.is_file(),
                                    reason="external od_covariance_campaign scripts not present")


def _sibling_entries() -> list[str]:
    siblings = [p for p in sys.path
                if p and "python_port" in Path(p).name.lower()
                and not Path(p).resolve().is_relative_to(ROOT)]
    siblings += [p for p in sys.path
                 if p and Path(p).resolve().parent == ROOT.parent
                 and Path(p).resolve() != ROOT and (Path(p) / "src" / "lunar_od").exists()]
    # anything INSIDE another python_port* checkout (e.g. <sibling>/examples)
    siblings += [p for p in sys.path
                 if p and not Path(p).resolve().is_relative_to(ROOT)
                 and any(part.lower().startswith("python_port") for part in Path(p).parts[:-1])]
    return siblings


@pytest.mark.parametrize("module", ["lunar_od", "app", "controllers.analysis_controller",
                                    "services.project_paths"])
def test_modules_resolve_inside_this_worktree(module):
    mod = importlib.import_module(module)
    assert Path(mod.__file__).resolve().is_relative_to(ROOT), (module, mod.__file__)


@needs_campaign
def test_campaign_module_is_bound_by_the_guard_at_session_start():
    mod = sys.modules.get("od_gravity_covariance_campaign")
    assert mod is not None, "conftest did not import the campaign through the guarded loader"
    for name in ("propagate_state", "generate_two_way_range_measurements"):
        src = sys.modules[getattr(mod, name).__module__].__file__
        assert Path(src).resolve().is_relative_to(ROOT), (name, src)


@needs_campaign
def test_direct_campaign_import_leaves_no_sibling_on_sys_path():
    """What later tests do (06_scripts on sys.path, plain import) must not reintroduce
    the campaign's stale sys.path entries."""
    scripts = str(CAMPAIGN_SCRIPT.parent)
    added = scripts not in sys.path
    if added:
        sys.path.insert(0, scripts)
    try:
        import od_gravity_covariance_campaign  # noqa: F401
        assert _sibling_entries() == []
    finally:
        if added:
            sys.path.remove(scripts)


def test_no_sibling_checkout_on_sys_path():
    assert _sibling_entries() == []


@needs_campaign
def test_plain_invocation_without_pythonpath_isolates():
    """The supported invocation (``python -m pytest`` in the worktree, no PYTHONPATH)
    must establish the isolation by itself."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
                        str(Path(__file__)), "-k", "not plain_invocation"],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


def test_pytest_uses_the_worktree_config(pytestconfig):
    assert pytestconfig.inipath is not None
    assert Path(pytestconfig.inipath).resolve() == ROOT / "pytest.ini"
    assert pytestconfig.getini("pythonpath") == []
