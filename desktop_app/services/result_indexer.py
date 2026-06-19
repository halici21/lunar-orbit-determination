"""Indexes result folders and files under python_port/results/."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime

from .project_paths import RESULTS_DIR


def _mtime(p: Path) -> datetime:
    try:
        return datetime.fromtimestamp(p.stat().st_mtime)
    except OSError:
        return datetime.min


def index_result_folders(results_dir: Path = RESULTS_DIR) -> list[dict]:
    """Return list of folder-info dicts sorted by name.

    Each dict has keys: name, path, modified, csv_files, png_files.
    """
    folders: list[dict] = []
    if not results_dir.exists():
        return folders

    # Named sub-folders
    for entry in sorted(results_dir.iterdir()):
        if not entry.is_dir():
            continue
        csv_files = sorted(entry.glob("*.csv"))
        png_files = sorted(entry.glob("*.png"))
        folders.append({
            "name": entry.name,
            "path": entry,
            "modified": _mtime(entry),
            "csv_files": csv_files,
            "png_files": png_files,
        })

    # Files sitting directly in results root
    root_csvs = sorted(results_dir.glob("*.csv"))
    root_pngs = sorted(results_dir.glob("*.png"))
    if root_csvs or root_pngs:
        folders.insert(0, {
            "name": "(root)",
            "path": results_dir,
            "modified": _mtime(results_dir),
            "csv_files": root_csvs,
            "png_files": root_pngs,
        })

    return folders


def get_recent_files(results_dir: Path = RESULTS_DIR, limit: int = 60) -> list[Path]:
    """Return the most-recently-modified CSV and PNG files across all result dirs."""
    if not results_dir.exists():
        return []
    files = list(results_dir.rglob("*.csv")) + list(results_dir.rglob("*.png"))
    files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return files[:limit]


def count_results(results_dir: Path = RESULTS_DIR) -> dict:
    """Quick stat: total CSV/PNG counts and subfolder count."""
    if not results_dir.exists():
        return {"folders": 0, "csv": 0, "png": 0, "total": 0}
    folders = sum(1 for p in results_dir.iterdir() if p.is_dir())
    csvs = len(list(results_dir.rglob("*.csv")))
    pngs = len(list(results_dir.rglob("*.png")))
    return {"folders": folders, "csv": csvs, "png": pngs, "total": csvs + pngs}
