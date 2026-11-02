"""WP7.6: pyproject extras are the requirement profiles, not a second copy that drifts."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
tomllib = pytest.importorskip("tomllib") if sys.version_info >= (3, 11) else pytest.importorskip("tomli")

PROFILES = {
    "accelerated": "requirements/accelerated.txt",
    "desktop": "desktop_app/requirements_desktop.txt",
    "dev": "requirements/dev.txt",
    "docs": "requirements/docs.txt",
}


def _requirement_lines(relative_path: str) -> list[str]:
    lines = []
    for raw in (ROOT / relative_path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and not line.startswith("-r"):
            lines.append(line)
    return lines


def test_extras_match_requirement_profiles():
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    assert set(extras) == set(PROFILES)
    for name, path in PROFILES.items():
        assert extras[name] == _requirement_lines(path), name


def test_windows_pins_satisfy_every_profile_floor():
    """The validated exact pins must sit inside the declared ranges."""
    from packaging.requirements import Requirement

    pins = {}
    for line in _requirement_lines("requirements/windows.txt"):
        requirement = Requirement(line)
        (spec,) = requirement.specifier
        pins[requirement.name.lower()] = spec.version
    floors = _requirement_lines("requirements.txt")
    for path in PROFILES.values():
        floors += _requirement_lines(path)
    for line in floors:
        requirement = Requirement(line)
        pinned = pins.get(requirement.name.lower())
        if pinned is not None:
            assert requirement.specifier.contains(pinned, prereleases=True), (line, pinned)
