"""Run provenance record for scenario runs (assessment CFG-02/CFG-04/CFG-10, WP1.6).

One JSON file per run names everything needed to reproduce it: source revision,
the normalized scenario config and its hash, the effective random seeds, the
SPICE kernels furnished at run time (with content hashes), the sampled
ephemeris identities and the package versions.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

RUN_PROVENANCE_SCHEMA = "lunar-od-run-provenance-v1"
_PACKAGES = ("numpy", "scipy", "spiceypy", "numba", "matplotlib", "lunar-od")
_REPO_ROOT = Path(__file__).resolve().parents[2]


def canonical_config_sha256(config_dict: dict[str, Any]) -> str:
    payload = json.dumps(config_dict, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_revision(root: Path = _REPO_ROOT) -> dict[str, Any]:
    """Git commit and dirty flag of the source tree, or None when unavailable."""
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty}


def _file_sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return "sha256:" + digest.hexdigest()
    except OSError:
        return None


def furnished_kernels() -> list[dict[str, Any]]:
    """Kernels in the SPICE pool, in load order, with content hashes."""
    import spiceypy as spice

    if not hasattr(spice, "ktotal"):  # a stubbed SPICE module has no pool to report
        return [{"file": "unrecorded", "type": None, "sha256": None}]
    kernels = []
    for index in range(spice.ktotal("ALL")):
        file_name, kind, _source, _handle = spice.kdata(index, "ALL")
        path = Path(file_name)
        kernels.append({"file": path.name, "type": kind, "sha256": _file_sha256(path)})
    return kernels


def earth_orientation_record(profile: str | None, policy: str | None, et_values) -> dict[str, Any]:
    """F2 WP2.4: the loaded Earth-orientation profile and, per epoch, the Earth PCK SPICE uses and whether its
    orientation is measured (``station_frames.earth_orientation_provenance``, without absolute paths)."""
    import spiceypy as spice

    from .spice_loader import earth_orientation

    record: dict[str, Any] = {"profile": earth_orientation(profile), "policy": policy}
    if not hasattr(spice, "ktotal"):  # a stubbed SPICE module has no pool to report
        record["epochs"] = "unrecorded"
        return record
    from .station_frames import earth_orientation_provenance

    record["epochs"] = []
    for et in et_values:
        prov = earth_orientation_provenance(float(et))
        record["epochs"].append({"et_s": float(et), "kernel": prov["kernel"],
                                 "last_datum_utc": prov["last_datum_utc"], "measured": bool(prov["measured"])})
    return record


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def run_provenance_record(
    config,
    *,
    measurement_seed: int | None,
    cold_start_seed: int,
    kernels: list[dict[str, Any]],
    ephemeris_identities: dict[str, Any],
    run_options: dict[str, Any] | None = None,
    earth_orientation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_dict = config.to_dict()
    record = {
        "schema": RUN_PROVENANCE_SCHEMA,
        "source": source_revision(),
        "scenario_config": config_dict,
        "scenario_config_sha256": canonical_config_sha256(config_dict),
        "seeds": {
            "measurement_seed": measurement_seed,
            "measurement_noise": bool(config.noise),
            "cold_start_seed": int(cold_start_seed),
        },
        "run_options": dict(run_options or {}),
        "spice_kernels": kernels,
        "ephemeris": {
            name: asdict(value) if is_dataclass(value) else value
            for name, value in ephemeris_identities.items()
        },
        "packages": package_versions(),
    }
    if earth_orientation is not None:
        record["earth_orientation"] = earth_orientation
    return record


def write_run_provenance(record: dict[str, Any], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    return path
