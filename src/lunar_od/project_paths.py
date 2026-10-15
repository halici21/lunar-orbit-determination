"""Machine-independent locations of repository and project data (Phase 17-HARDEN).

The repository (this worktree) is ``repo_root()``.  Read-only campaign data
lives OUTSIDE the repository, in the project directory that contains it::

    <project>/                      LUNAR_OD_PROJECT_ROOT  (default: repo parent)
        od_covariance_campaign/     LUNAR_OD_CAMPAIGN_ROOT (default: <project>/...)
        python_port/data/gravity/   LUNAR_OD_GRAVITY_ROOT  (default: <project>/...)
        ephemeris_data.mat
        <this repository>/

Before Phase 17-HARDEN these were absolute ``C:/Users/.../Grad/...`` literals
scattered over examples and tests (defect HARDEN-D10), which only resolved on
one machine and silently pointed at a different checkout when the ``Grad``
junction moved.  Every caller now resolves through this module; environment
variables override the defaults, and nothing here falls back silently to a
second location.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT_ENV = "LUNAR_OD_PROJECT_ROOT"
CAMPAIGN_ROOT_ENV = "LUNAR_OD_CAMPAIGN_ROOT"
GRAVITY_ROOT_ENV = "LUNAR_OD_GRAVITY_ROOT"


def repo_root() -> Path:
    """The repository/worktree containing this package."""
    return Path(__file__).resolve().parents[2]


def artifacts_dir() -> Path:
    """The repository's committed artifacts directory (always inside the repo)."""
    return repo_root() / "studies" / "phase17" / "results"


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


def project_root() -> Path:
    """Directory holding the external campaign data and this repository."""
    return _env_path(PROJECT_ROOT_ENV, repo_root().parent)


def campaign_root() -> Path:
    """The read-only ``od_covariance_campaign`` data tree."""
    return _env_path(CAMPAIGN_ROOT_ENV, project_root() / "od_covariance_campaign")


def gravity_data_root() -> Path:
    """The lunar gravity-field coefficient files (``gl1800f/...``)."""
    return _env_path(GRAVITY_ROOT_ENV, project_root() / "python_port" / "data" / "gravity")


def ephemeris_mat() -> Path:
    """The MATLAB campaign ephemeris table (historical fixture source)."""
    return project_root() / "ephemeris_data.mat"


# ----------------------------------------------------------------------
# External campaign module (HARDEN-D12)
# ----------------------------------------------------------------------
CAMPAIGN_MODULE = "od_gravity_covariance_campaign"


class CrossCheckoutImportError(ImportError):
    """``lunar_od`` would be (or was) bound from a checkout other than this one."""


def _inside(path: str | os.PathLike, root: Path) -> bool:
    try:
        return Path(path).resolve().is_relative_to(root)
    except OSError:
        return False


def import_campaign_module():
    """Import the external read-only campaign module WITHOUT letting it rebind
    ``lunar_od`` to another checkout.

    ``od_gravity_covariance_campaign.py`` (outside the repository) runs
    ``sys.path.insert(0, <a sibling python_port checkout>)`` before importing
    ``lunar_od``.  If that sibling exists, a naive first import binds the
    campaign's -- and every later caller's -- ``lunar_od`` to it.  This loader
    (1) binds ``lunar_od`` to this repository first, (2) imports the campaign,
    (3) removes every ``sys.path`` entry the import added that lies outside the
    repository and is not the campaign script directory, and (4) verifies that
    the campaign's ``lunar_od`` objects come from this repository.
    """
    import importlib
    import sys

    import lunar_od  # this module's own package: already bound to THIS repository

    root = repo_root()
    if not _inside(lunar_od.__file__, root):
        raise CrossCheckoutImportError("lunar_od bound to %s, not %s" % (lunar_od.__file__, root))
    # The external campaign is a historical script tree, not an installed
    # package. Keep its two import roots confined to this guarded loader so
    # standalone study drivers work with the same overrides as pytest.
    scripts = campaign_root() / "06_scripts"
    if (scripts / (CAMPAIGN_MODULE + ".py")).is_file():
        for path in (scripts, root / "examples"):
            if str(path) not in sys.path:
                sys.path.append(str(path))
    module = importlib.import_module(CAMPAIGN_MODULE)
    sys.path[:] = [p for p in sys.path if not is_foreign_checkout_path(p)]
    for name in ("propagate_state", "generate_two_way_range_measurements"):
        obj = getattr(module, name, None)
        src = getattr(sys.modules.get(getattr(obj, "__module__", "")), "__file__", None)
        if src is not None and not _inside(src, root):
            raise CrossCheckoutImportError("%s.%s comes from %s, outside %s"
                                           % (CAMPAIGN_MODULE, name, src, root))
    return module


def is_foreign_checkout_path(path: str) -> bool:
    """A ``sys.path`` entry inside some OTHER ``python_port*`` checkout."""
    if not path:
        return False
    root = repo_root()
    if _inside(path, root):
        return False
    return any(part.lower().startswith("python_port") for part in Path(path).parts)
