"""Build a relocatable Windows x64 desktop with its own CPython interpreter."""
from __future__ import annotations
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_URL = "https://www.python.org/ftp/python/3.13.2/python-3.13.2-embed-amd64.zip"
RUNTIME_SHA256 = "1e803610b140cbf69dfa2ceaaeb39651bef75a239c381289e827c30862a27b93"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def bundle_files(bundle):
    """Immutable distribution files; generated interpreter/JIT caches are runtime state."""
    return (p for p in sorted(bundle.rglob("*")) if p.is_file()
            and "__pycache__" not in p.relative_to(bundle).parts
            and p.suffix.lower() not in {".pyc", ".pyo", ".nbc", ".nbi"})


def bundle_hashes(bundle):
    return {p.relative_to(bundle).as_posix(): digest(p) for p in bundle_files(bundle)}


def build(version, development=False):
    if os.name != "nt" or sys.version_info[:2] != (3, 13):
        raise RuntimeError("Build on Windows x64 with Python 3.13")
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:-alpha\.\d+)?", version):
        raise ValueError("Invalid release version")
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip())
    if dirty and not development:
        raise ValueError("Commit the source before building a release")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    output = ROOT / "outputs" / "portable" / ("v" + version)
    bundle = output / "LunarOD"
    bundle.mkdir(parents=True, exist_ok=False)
    cache = ROOT / "outputs" / "download-cache"
    cache.mkdir(parents=True, exist_ok=True)
    runtime_zip = cache / RUNTIME_URL.rsplit("/", 1)[1]
    if not runtime_zip.exists():
        urllib.request.urlretrieve(RUNTIME_URL, runtime_zip)
    if digest(runtime_zip) != RUNTIME_SHA256:
        raise ValueError("CPython embeddable archive SHA256 mismatch")
    runtime = bundle / "runtime"
    with zipfile.ZipFile(runtime_zip) as archive:
        archive.extractall(runtime)
    site = runtime / "Lib" / "site-packages"
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-compile", "--only-binary=:all:",
                    "--target", str(site), "-r", str(ROOT / "requirements/windows.txt")], check=True)
    (runtime / "python313._pth").write_text(
        "python313.zip\n.\nLib/site-packages\n../app\n../app/src\nimport site\n", encoding="utf-8")
    app = bundle / "app"
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    source_hashes = {}
    roots = ("src/", "desktop_app/", "examples/", "fixtures/", "requirements/")
    singles = {"requirements.txt", "README.md", "LICENSE", "CITATION.cff",
               "tools/qualify_windows_portable.py", "docs/windows-quickstart.md"}
    for name in names:
        if name and (name.startswith(roots) or name in singles):
            destination = app / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
            source_hashes[name] = digest(destination)
    source = {"version": version, "git_commit": commit, "working_tree_dirty": dirty,
              "source_files_sha256": source_hashes, "runtime_url": RUNTIME_URL,
              "runtime_sha256": RUNTIME_SHA256,
              "distribution": "Windows x64 portable desktop with CPython; no SPICE kernels or external gravity data"}
    (app / "SOURCE_REVISION.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    (bundle / "LunarOD.cmd").write_text(
        '@echo off\r\nsetlocal\r\nif /i "%~1"=="--check" goto check\r\ncd /d "%~dp0app"\r\n'
        '"%~dp0runtime\\python.exe" -u "%~dp0app\\desktop_app\\main.py" %*\r\n'
        'if errorlevel 1 pause\r\nexit /b %errorlevel%\r\n:check\r\n'
        '"%~dp0runtime\\python.exe" -E "%~dp0app\\tools\\qualify_windows_portable.py" --bundle "%~dp0."\r\n'
        'exit /b %errorlevel%\r\n', encoding="ascii", newline="")
    (bundle / "Verify.cmd").write_text(
        '@echo off\r\nsetlocal\r\ncd /d "%~dp0"\r\n'
        '"%~dp0runtime\\python.exe" -E "%~dp0app\\tools\\qualify_windows_portable.py" --bundle "%~dp0."\r\n'
        'pause\r\n', encoding="ascii", newline="")
    (bundle / "README.txt").write_text(
        "Lunar OD " + version + " - Windows x64 portable preview\n\n"
        "Extract the ENTIRE archive to a writable folder, then double-click LunarOD.cmd.\n"
        "Python installation and pip are not required. Do not move the launcher alone.\n"
        "Choose Ornek > Verisiz sentetik egitimi baslat for the data-free tutorial.\n"
        "For SPICE scenarios select your kernel directory in Settings; kernels and gravity data are not bundled.\n"
        "Results are saved under app/results; copy them before replacing a version.\n"
        "Verify.cmd checks bundled imports, child processes, cancellation, desktop shutdown and a real tutorial solve.\n"
        "Redistributed CPython license is runtime/LICENSE.txt; third-party notices are in runtime/Lib/site-packages/*.dist-info.\n"
        "This alpha is a portable ZIP, not an installed or code-signed application.\n", encoding="utf-8")
    packages = {d.metadata["Name"]: d.version for d in metadata.distributions(path=[str(site)])}
    (bundle / "THIRD_PARTY_PACKAGES.json").write_text(json.dumps(packages, indent=2, sort_keys=True), encoding="utf-8")
    record = {"source": source, "packages": packages,
              "cache_policy": "Generated Python/Numba bytecode is excluded from immutable distribution files",
              "files_sha256": bundle_hashes(bundle)}
    (bundle / "BUNDLE_MANIFEST.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    # Qualification executes in a copy extracted from this exact ZIP.
    destination = ROOT / "outputs/releases"
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"lunar-od-v{version}-windows-x64-portable.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for path in bundle_files(bundle):
            stream.write(path, path.relative_to(output).as_posix())
    print(f"Built {archive} ({archive.stat().st_size:,} bytes)", flush=True)
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--development", action="store_true", help="Mark a local dirty-source test bundle; never publish it")
    args = parser.parse_args()
    build(args.version, args.development)
