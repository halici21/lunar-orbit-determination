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
