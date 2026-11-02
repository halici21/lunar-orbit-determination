"""Read-only archive sources: run folders, report/figure files and test-run records.

The archive never copies or edits anything it lists. Extra source folders are
the user's choice (stored in QSettings under SOURCES_KEY); a folder that has
disappeared is skipped, not an error.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from services.project_paths import PYTHON_PORT, RESULTS_DIR
from services.tracking_results import LoadCancelled

SOURCES_KEY = "archive/extra_sources"
TEST_RUNS_DIR = RESULTS_DIR / "test_runs"
REPORT_KINDS = {".png": "Grafik", ".svg": "Grafik", ".jpg": "Grafik", ".csv": "Tablo",
                ".json": "Kayıt", ".txt": "Metin", ".md": "Metin"}
_MAX_REPORTS = 8000


def extra_sources(settings):
    """The user's extra archive folders, as Paths (missing ones are kept: they may return)."""
    raw = settings.value(SOURCES_KEY, []) if settings is not None else []
    if isinstance(raw, str):
        raw = [raw] if raw else []
    return tuple(Path(p) for p in raw or () if isinstance(p, str) and p)


def source_label(path, roots):
    """Short, stable name for the root a file belongs to."""
    path = Path(path)
    for root in roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if Path(root) == RESULTS_DIR:
            return "Bu çalışma alanı"
        return f"{Path(root).parent.name}/{Path(root).name}"
    return Path(path).parent.name


def scan_archive_sources(folders, cancel=lambda: False, *, count_cache=None, missing_ok=False):
    """Index run artifacts across several folders; each entry records its source."""
    from services.run_archive import scan_run_archive
    roots = [Path(f) for f in folders]
    entries = []
    for index, folder in enumerate(roots):
        if cancel():
            raise LoadCancelled()
        if index and not folder.exists():
            continue  # an extra source that is offline right now
        found = scan_run_archive(folder, cancel, count_cache=count_cache, missing_ok=missing_ok and index == 0)
        entries.extend(replace(e, source=source_label(e.path, roots)) for e in found)
    return tuple(sorted(entries, key=lambda e: (e.created_at, str(e.path)), reverse=True))


def report_roots(extra=()):
    """Folders searched for reports and figures: this workspace, the study results, extras."""
    roots = [RESULTS_DIR, PYTHON_PORT / "studies" / "phase17" / "results", *extra]
    unique = []
    for root in roots:
        root = Path(root)
        if root.exists() and root not in unique:
            unique.append(root)
    return unique


@dataclass(frozen=True)
class ReportFile:
    path: Path
    source: str
    folder: str
    kind: str
    size: int
    modified: str


def scan_reports(roots, cancel=lambda: False):
    """List report-like files (figures, tables, records, notes) under the roots."""
    roots = [Path(r) for r in roots]
    files = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            if cancel():
                raise LoadCancelled()
            dirnames[:] = sorted(d for d in dirnames if not d.startswith((".", "__")) and d != "test_runs")
            for name in sorted(filenames):
                kind = REPORT_KINDS.get(Path(name).suffix.lower())
                if kind is None:
                    continue
                path = Path(dirpath) / name
                try:
                    stat = path.stat()
                except OSError:
                    continue
                folder = str(path.parent.relative_to(root)).replace("\\", "/")
                files.append(ReportFile(path, source_label(path, roots), "" if folder == "." else folder,
                    kind, stat.st_size, datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()))
                if len(files) >= _MAX_REPORTS:
                    return tuple(files)
    return tuple(sorted(files, key=lambda f: f.modified, reverse=True))


def load_test_runs(folder=None, cancel=lambda: False):
    """Read recorded pytest sessions and imported CI runs, newest first."""
    folder = Path(folder or TEST_RUNS_DIR)
    runs = []
    if not folder.exists():
        return ()
    for path in sorted(folder.glob("*.json")):
        if cancel():
            raise LoadCancelled()
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and record.get("schema") == "lunar-od-test-run-v1":
            runs.append(dict(record, _path=str(path)))
    return tuple(sorted(runs, key=lambda r: str(r.get("finished_at", "")), reverse=True))
