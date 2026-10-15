"""Read-only desktop preflight; scientific imports and SPICE run in a child process."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import runpy
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
KERNEL_KEY = "app_settings/kernel_directory"
PACKAGES = ("numpy", "scipy", "numba", "spiceypy", "matplotlib", "pandas",
            "pyqtgraph", "h5py", "PyQt5")


def apply_kernel_preference(settings, inherited: str | None) -> str:
    """Set this desktop process's override; clearing restores the inherited value."""
    selected = str(settings.value(KERNEL_KEY, "") or "").strip()
    effective = selected or inherited
    if effective:
        os.environ["LUNAR_OD_KERNEL_DIR"] = effective
    else:
        os.environ.pop("LUNAR_OD_KERNEL_DIR", None)
    return selected


def _runtime_checks():
    pins = {}
    for line in (ROOT / "requirements" / "windows.txt").read_text(encoding="utf-8").splitlines():
        if "==" in line and not line.startswith("#"):
            name, version = line.split("==", 1)
            pins[name] = version
    checks = []
    bits = struct.calcsize("P") * 8
    reference = sys.version_info[:2] == (3, 13) and bits == 64
    checks.append({"name": "Python", "status": "ok" if reference else "warning",
                   "detail": f"{platform.python_version()} / {bits} bit · doğrulanmış profil: 3.13 / 64 bit"})
    for name in PACKAGES:
        try:
            importlib.import_module("PyQt5.QtCore" if name == "PyQt5" else name)
            version = importlib.metadata.version(name)
            expected = pins.get(name)
            status = "ok" if version == expected else "warning"
            detail = f"{version} · referans: {expected}"
        except Exception as exc:
            status, detail = "error", f"İçe aktarılamadı: {exc}"
        checks.append({"name": name, "status": status, "detail": detail})
    return checks


def _check_spice(loader, directory, duration_h, state_step_s, ephemeris_step_s):
    import spiceypy as spice
    try:
        loader["load_spice_kernels"](directory)
        fixture = json.loads((ROOT / "fixtures" / "spice_snapshots.json").read_text(encoding="utf-8"))
        epoch = fixture["epoch_utc"]
        start = float(spice.str2et(epoch))
        # Match both runner grids, including a sample beyond a non-integral end.
        span = 0.0 if duration_h is None else duration_h * 3600.0
        end = max(math.ceil(span / step) * step for step in (state_step_s, ephemeris_step_s))
        offsets = sorted({0.0, end / 2.0, end})
        for offset in offsets:
            et = start + offset
            for body in ("EARTH", "SUN"):
                spice.spkezr(body, et, "J2000", "NONE", "MOON")
            spice.sxform("J2000", "ITRF93", et)
            for frame in ("MOON_PA_DE421", "MOON_PA_DE440"):
                spice.pxform("J2000", frame, et)
        return {"name": "SPICE yükleme / örnek epoklar", "status": "ok",
                "detail": f"UTC {epoch} · 0–{end / 3600:g} sa · {len(offsets)} epok. "
                          "Earth/Sun, ITRF93, DE421 ve DE440 sorguları geçti. "
                          "Bu kontrol bütün epokların kapsamını veya çözüm yakınsamasını doğrulamaz."}
    finally:
        spice.kclear()


def inspect_environment(*, kernel_dir=None, duration_h=None, state_step_s=600.0,
                        ephemeris_step_s=3600.0, output_dir=None):
    """Return a structured report without creating folders or changing source data."""
    checks = _runtime_checks()
    # Read the canonical loader without importing lunar_od's eager scientific API.
    # This also allows the missing-dependency report to retain the kernel inventory.
    loader = runpy.run_path(str(ROOT / "src" / "lunar_od" / "spice_loader.py"))
    candidates = [Path(kernel_dir).expanduser()] if kernel_dir else loader["default_kernel_candidates"]()
    directory = None
    try:
        directory = loader["resolve_kernel_dir"](candidates).resolve()
        checks.append({"name": "Kernel klasörü", "status": "ok", "detail": str(directory)})
    except OSError as exc:
        checks.append({"name": "Kernel klasörü", "status": "error", "detail": str(exc)})
    files = []
    for name in loader["REQUIRED_KERNELS"]:
        item = {"name": name, "status": "error", "detail": "Dosya eksik", "size_bytes": None}
        if directory is not None:
            path = directory / name
            try:
                with path.open("rb") as stream:
                    if not stream.read(1):
                        raise ValueError("Dosya boş")
                size = path.stat().st_size
                item.update(status="ok", detail=f"{size:,} bayt · okunabilir", size_bytes=size)
            except (OSError, ValueError) as exc:
                item["detail"] = str(exc)
        files.append(item)
    if output_dir:
        target = Path(output_dir).expanduser()
        if not target.is_absolute():
            target = ROOT / target
        target = target.resolve()
        ancestor = target
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        allowed = ancestor.is_dir() and os.access(ancestor, os.W_OK)
        checks.append({"name": "Çıktı klasörü", "status": "ok" if allowed else "error",
                       "detail": f"{target} · en yakın mevcut klasörde izin kontrolü; dosya yazılmadı"})
    if not any(c["status"] == "error" for c in checks + files):
        try:
            if (duration_h is not None and (not math.isfinite(duration_h) or duration_h <= 0)) or any(
                    not math.isfinite(s) or s <= 0 for s in (state_step_s, ephemeris_step_s)):
                raise ValueError("Süre ve örnek adımları sonlu ve pozitif olmalı.")
            checks.append(_check_spice(loader, directory, duration_h, state_step_s, ephemeris_step_s))
        except Exception as exc:
            checks.append({"name": "SPICE yükleme / örnek epoklar", "status": "error", "detail": str(exc)})
    else:
        checks.append({"name": "SPICE yükleme / örnek epoklar", "status": "pending",
                       "detail": "Eksik dosya veya bağımlılık nedeniyle çalıştırılmadı."})
    return {"schema": "lunar-od-environment-v1", "checked_at": datetime.now(timezone.utc).isoformat(),
            "ready": not any(c["status"] == "error" for c in checks + files),
            "python_executable": sys.executable, "kernel_dir": str(directory) if directory else None,
            "searched_paths": [str(p) for p in candidates], "duration_h": duration_h,
            "checks": checks, "kernels": files}
