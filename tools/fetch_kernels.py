"""Assemble the SHA-256-verified SPICE kernel set the kernel-backed CI tiers need (WP7.2).

    python tools/fetch_kernels.py --dest outputs/kernels
    python tools/fetch_kernels.py --dest outputs/kernels --source-dir %USERPROFILE%/Documents/mice/kernels
    python tools/fetch_kernels.py --make-archive outputs/lunar-od-kernels.zip --source-dir <validated dir>

Kernels are never committed. Each file of the CI set (:func:`ci_kernel_set`) is taken,
in this order, from: a valid copy already in ``--dest`` (the CI cache), ``--source-dir``,
the zip at ``LUNAR_OD_KERNEL_ARCHIVE_URL`` (optionally with ``LUNAR_OD_KERNEL_ARCHIVE_TOKEN``
as a bearer token), and finally the NAIF URL recorded in ``lunar_od.kernel_manifest``.
Every file must match its manifest SHA-256; anything missing or different exits 1
with the list, so a kernel tier fails with a clear message rather than skipping.

NAIF rotates some files (Earth predict/high-precision PCKs, versioned frame kernels),
so the NAIF fallback cannot rebuild the whole validated set; the archive (made once
from the validated local set with --make-archive) or a self-hosted runner with
LUNAR_OD_KERNEL_DIR is the dependable source.
"""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_URL_ENV = "LUNAR_OD_KERNEL_ARCHIVE_URL"
ARCHIVE_TOKEN_ENV = "LUNAR_OD_KERNEL_ARCHIVE_TOKEN"

#: Kernels the test suite furnishes outside the loader's profiles.
EXTRA_TEST_KERNELS = (
    "earthstns_itrf93_260814.bsp",   # tests/test_wp2_4_dsn_stations.py (NAIF station SPK oracle)
    "earth_latest_high_prec.bpc",    # tests/test_native_media_partials.py (manifest-pinned copy)
)


def ci_kernel_set() -> list[str]:
    """Every kernel of both ephemerides and both Earth-orientation profiles, plus the test extras."""
    from lunar_od.spice_loader import EARTH_ORIENTATION_KERNELS, PLANETARY_EPHEMERIS_KERNELS, required_kernels

    names: list[str] = []
    for ephemeris in PLANETARY_EPHEMERIS_KERNELS:
        for orientation in EARTH_ORIENTATION_KERNELS:
            names.extend(required_kernels(ephemeris, orientation))
    names.extend(EXTRA_TEST_KERNELS)
    return list(dict.fromkeys(names))


def _valid(path: Path) -> bool:
    from lunar_od.kernel_manifest import KERNEL_MANIFEST, kernel_sha256

    return path.is_file() and kernel_sha256(path) == KERNEL_MANIFEST[path.name][0]


def _download(url: str, token: str | None = None) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "lunar-od-kernel-fetch"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
        request.add_header("Accept", "application/octet-stream")
    with urllib.request.urlopen(request, timeout=600) as response:  # noqa: S310 - fixed https sources
        return response.read()


def assemble(dest: Path, source_dir: Path | None = None) -> list[str]:
    """Fill ``dest`` with the verified CI set; return the problems (empty when complete)."""
    from lunar_od.kernel_manifest import KERNEL_MANIFEST

    dest.mkdir(parents=True, exist_ok=True)
    names = ci_kernel_set()
    unknown = [n for n in names if n not in KERNEL_MANIFEST]
    if unknown:
        return [f"{n}: required but not in lunar_od.kernel_manifest" for n in unknown]
    missing = [n for n in names if not _valid(dest / n)]
    if missing:
        print(f"{len(names) - len(missing)} of {len(names)} kernels already verified in {dest}", flush=True)
    if missing and source_dir is not None:
        for name in list(missing):
            candidate = source_dir / name
            if _valid(candidate):
                shutil.copy2(candidate, dest / name)
                missing.remove(name)
                print(f"copied   {name} from {source_dir}", flush=True)
    archive_url = os.environ.get(ARCHIVE_URL_ENV, "").strip()
    if missing and archive_url:
        print(f"downloading kernel archive ({ARCHIVE_URL_ENV})", flush=True)
        try:
            payload = _download(archive_url, os.environ.get(ARCHIVE_TOKEN_ENV) or None)
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                members = {Path(m).name: m for m in archive.namelist() if not m.endswith("/")}
                for name in list(missing):
                    if name in members:
                        (dest / name).write_bytes(archive.read(members[name]))
                        if _valid(dest / name):
                            missing.remove(name)
                            print(f"archive  {name}", flush=True)
                        else:
                            (dest / name).unlink()
        except Exception as exc:  # report and fall through to NAIF
            print(f"kernel archive unavailable: {type(exc).__name__}: {exc}", flush=True)
    problems = []
    for name in list(missing):
        url = KERNEL_MANIFEST[name][1]
        try:
            data = _download(url)
        except Exception as exc:
            problems.append(f"{name}: not in --dest/--source-dir/archive; NAIF {url} -> {type(exc).__name__}: {exc}")
            continue
        with tempfile.NamedTemporaryFile(dir=dest, delete=False, suffix=".part") as handle:
            handle.write(data)
        part = Path(handle.name)
        target = dest / name
        part.replace(target)
        if _valid(target):
            print(f"naif     {name}", flush=True)
        else:
            target.unlink()
            problems.append(f"{name}: NAIF {url} serves different content than the manifest SHA-256 "
                            "(NAIF updates this file; use the validated archive)")
    return problems


def make_archive(out: Path, source_dir: Path) -> list[str]:
    """Zip the verified CI set from a validated local directory (to host as LUNAR_OD_KERNEL_ARCHIVE_URL)."""
    names = ci_kernel_set()
    problems = [f"{n}: missing or different in {source_dir}" for n in names if not _valid(source_dir / n)]
    if problems:
        return problems
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(source_dir / name, arcname=name)
    print(f"wrote {out} ({len(names)} kernels)")
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=ROOT / "outputs" / "kernels")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--make-archive", type=Path, metavar="ZIP")
    parser.add_argument("--list", action="store_true", help="print the CI kernel set and exit")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(ci_kernel_set()))
        return 0
    if args.make_archive:
        if args.source_dir is None:
            parser.error("--make-archive needs --source-dir")
        problems = make_archive(args.make_archive, args.source_dir)
    else:
        problems = assemble(args.dest, args.source_dir)
    if problems:
        print("\nSPICE kernel set INCOMPLETE; the kernel-backed tiers must fail, not skip (REQ-SW-003):",
              file=sys.stderr)
        for line in problems:
            print("  " + line, file=sys.stderr)
        print(f"\nProvide the validated set: set the repository secret {ARCHIVE_URL_ENV} to a zip made with\n"
              "  python tools/fetch_kernels.py --make-archive <zip> --source-dir <validated kernel dir>\n"
              "or run the tier on a runner whose LUNAR_OD_KERNEL_DIR holds the validated kernels.",
              file=sys.stderr)
        return 1
    print(f"kernel set verified in {args.dest if not args.make_archive else args.make_archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
