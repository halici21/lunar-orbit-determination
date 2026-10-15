"""Build and qualify source, wheel and portable assets from a clean revision."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import venv
import zipfile

from build_source_release import build as build_source
from build_windows_portable import build as build_portable, digest

ROOT = Path(__file__).resolve().parents[1]


def run(command, **kwargs):
    subprocess.run([str(x) for x in command], check=True, **kwargs)


def release_version():
    value = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    return value.replace("a", "-alpha.")


def prepare(version, skip_public=False):
    if version != release_version():
        raise ValueError("Release version must match pyproject.toml")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise ValueError("Release source must be clean and committed")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run([sys.executable, ROOT / "tools/check_repository.py"], cwd=ROOT)
    if not skip_public:
        run([sys.executable, ROOT / "tools/run_public_checks.py", "--junitxml", "outputs/release-public.xml"], cwd=ROOT)
    destination = ROOT / "outputs/qualified-release"
    destination.mkdir(parents=True, exist_ok=False)
    wheel_dir = ROOT / "outputs/release-wheel"
    run([sys.executable, "-m", "pip", "wheel", "--no-deps", ROOT, "--wheel-dir", wheel_dir])
    wheels = list(wheel_dir.glob("lunar_orbit_determination-" + version.replace("-alpha.", "a") + "-*.whl"))
    if len(wheels) != 1:
        raise ValueError("Expected exactly one versioned wheel")
    wheel = destination / wheels[0].name
    shutil.copy2(wheels[0], wheel)
    source = build_source(version)
    portable = build_portable(version)
    for asset in (source, portable):
        shutil.copy2(asset, destination / asset.name)
    source, portable = destination / source.name, destination / portable.name
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLBACKEND"] = "Agg"
    with tempfile.TemporaryDirectory(prefix="lunar release & ") as temporary:
        scratch = Path(temporary)
        wheel_env = scratch / "wheel-env"
        venv.EnvBuilder(with_pip=True).create(wheel_env)
        python = wheel_env / "Scripts/python.exe"
        run([python, "-m", "pip", "install", wheel], cwd=scratch, env=env)
        run([python, "-m", "pip", "check"], cwd=scratch, env=env)
        check = scratch / "check_library_install.py"
        shutil.copy2(ROOT / "tools/check_library_install.py", check)
        run([python, check], cwd=scratch, env=env)
        with zipfile.ZipFile(source) as archive:
            archive.extractall(scratch / "source")
        source_root = scratch / "source" / f"lunar-od-v{version}"
        # The reference checkout can contain long immutable archive names.
        source_env = scratch / "source-env"
        venv.EnvBuilder(with_pip=True).create(source_env)
        source_python = source_env / "Scripts/python.exe"
        run([source_python, "-m", "pip", "install", "-r", source_root / "requirements/windows.txt"], env=env)
        run([source_python, "-m", "pip", "install", "-e", source_root], env=env)
        run([source_python, source_root / "tools/check_repository.py"], cwd=source_root, env=env)
        run([source_python, source_root / "examples/quickstart_od.py", "--output", scratch / "source-fit"], cwd=scratch, env=env)
        with zipfile.ZipFile(portable) as archive:
            archive.extractall(scratch / "portable")
        bundle = scratch / "portable/LunarOD"
        run([bundle / "runtime/python.exe", bundle / "app/tools/qualify_windows_portable.py",
             "--bundle", bundle], cwd=scratch, env=env)
        portable_report = json.loads((bundle / "qualification/verification.json").read_text(encoding="utf-8"))
    report = {"schema": "lunar-od-release-verification-v1", "version": version, "source_commit": commit,
        "public_checks": "previously verified exact source" if skip_public else "passed in release build",
        "wheel": "fresh environment / outside checkout / all imports / analytic circular orbit / pip check",
        "source": "fresh reference install in path with spaces and ampersand / record identity / real orbit fit",
        "portable": portable_report,
        "asset_sha256": {p.name: digest(p) for p in (wheel, source, portable)}}
    record_path = destination / f"lunar-od-v{version}-verification.json"
    record_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    assets = (wheel, source, portable, record_path)
    (destination / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.name}\n" for p in assets), encoding="utf-8")
    print(f"Qualified {version} from {commit}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=release_version())
    parser.add_argument("--skip-public", action="store_true", help="Local reuse of exact-revision public test evidence; CI never sets this")
    args = parser.parse_args()
    prepare(args.version, args.skip_public)
