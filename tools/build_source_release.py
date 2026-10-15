"""Build a clean, tracked Windows source ZIP without changing record bytes.

python tools/build_source_release.py --version 0.1.0-alpha.2
Build the library wheel separately with pip wheel; publish only after validation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build(version):
    status = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"])
    if status.strip():
        raise ValueError("Build a release only from a clean committed checkout")
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    names = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z"]).decode().split("\0")
    destination = ROOT / "outputs/releases"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"lunar-od-v{version}-windows-source.zip"
    prefix = f"lunar-od-v{version}/"
    hashes = {}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            if not name:
                continue
            data = (ROOT / name).read_bytes()
            hashes[name] = hashlib.sha256(data).hexdigest()
            archive.writestr(prefix + name, data)
        archive.writestr(prefix + "SOURCE_REVISION.json", json.dumps({
            "git_commit": revision, "version": version, "source_files_sha256": hashes,
            "distribution": "Windows source preview; no Python runtime, standalone EXE or SPICE kernels",
        }, indent=2) + "\n")
    print(f"{path}: {len(hashes)} tracked files; source {revision}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    build(parser.parse_args().version)
