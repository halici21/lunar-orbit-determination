"""Phase 17-HARDEN - cross-worktree import isolation.

Before HARDEN, a test run from this worktree picked up the workspace-root
pytest.ini, whose ``pythonpath`` put the SIBLING ``python_port`` checkout on
sys.path.  ``lunar_od`` happened to resolve locally (cwd first), but the
desktop-app modules (``app``, ``controllers.*``, ``services.*``) resolved from
the sibling, so those tests validated the wrong checkout.  FA-06 guarded only
``lunar_od``.  These tests guard every import root the suite relies on.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("module", ["lunar_od", "app", "controllers.analysis_controller",
                                    "services.project_paths"])
def test_modules_resolve_inside_this_worktree(module):
    mod = importlib.import_module(module)
    assert Path(mod.__file__).resolve().is_relative_to(ROOT), (module, mod.__file__)


def test_no_sibling_checkout_on_sys_path():
    siblings = [p for p in sys.path
                if p and "python_port" in Path(p).name.lower()
                and not Path(p).resolve().is_relative_to(ROOT)]
    siblings += [p for p in sys.path
                 if p and Path(p).resolve().parent == ROOT.parent
                 and Path(p).resolve() != ROOT and (Path(p) / "lunar_od").exists()]
    assert siblings == []


def test_pytest_uses_the_worktree_config(pytestconfig):
    assert pytestconfig.inipath is not None
    assert Path(pytestconfig.inipath).resolve() == ROOT / "pytest.ini"
    assert pytestconfig.getini("pythonpath") == []
