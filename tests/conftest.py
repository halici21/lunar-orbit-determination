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
    # Phase C4-I: the campaign module samples its Earth table at import time and
    # is a DE421-era record generator, so it is imported on DE421 and the
    # default ephemeris is restored afterwards.
    import os as _os

    from lunar_od.spice_loader import PLANETARY_EPHEMERIS_ENV as _EPH_ENV
    from lunar_od.spice_loader import load_spice_kernels as _load_kernels

    _eph_old = _os.environ.get(_EPH_ENV)
    _os.environ[_EPH_ENV] = "DE421"
    try:
        _import_campaign()
    finally:
        if _eph_old is None:
            _os.environ.pop(_EPH_ENV, None)
        else:
            _os.environ[_EPH_ENV] = _eph_old
        try:
            _load_kernels()
        except FileNotFoundError:
            pass


import os  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(scope="module")
def de421_ephemeris():
    """Phase C4-I: run a module that reproduces a DE421-era frozen record on DE421.

    The production default planetary ephemeris is DE440 since C4-I.  Records
    computed before it (MATLAB fixtures, campaign golden values) stay valid
    for the ephemeris they were computed with, so the modules that reproduce
    them select DE421 through the environment variable (which every
    ``load_spice_kernels`` call, including the study scripts', honours) and
    restore the default afterwards.
    """
    from lunar_od.spice_loader import PLANETARY_EPHEMERIS_ENV, load_spice_kernels

    old = os.environ.get(PLANETARY_EPHEMERIS_ENV)
    os.environ[PLANETARY_EPHEMERIS_ENV] = "DE421"
    try:
        load_spice_kernels()
    except FileNotFoundError:
        pass
    yield
    if old is None:
        os.environ.pop(PLANETARY_EPHEMERIS_ENV, None)
    else:
        os.environ[PLANETARY_EPHEMERIS_ENV] = old
    try:
        load_spice_kernels()
    except FileNotFoundError:
        pass
