"""Project path discovery — finds python_port root without hard-coding paths."""
from __future__ import annotations
from pathlib import Path


def _find_python_port() -> Path:
    """Walk up from this file to find the directory containing lunar_od/."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "lunar_od" / "__init__.py").exists():
            return parent
    # Fallback: assume desktop_app lives inside python_port
    return current.parents[2]


PYTHON_PORT: Path = _find_python_port()
RESULTS_DIR: Path = PYTHON_PORT / "results"
EXAMPLES_DIR: Path = PYTHON_PORT / "examples"
LUNAR_OD_DIR: Path = PYTHON_PORT / "lunar_od"
DESKTOP_APP_DIR: Path = Path(__file__).resolve().parents[1]
