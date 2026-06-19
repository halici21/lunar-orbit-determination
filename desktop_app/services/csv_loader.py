"""CSV loading utilities using pandas (gracefully degrades if pandas absent)."""
from __future__ import annotations
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

try:
    import pandas as _pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def load_csv(path: Path) -> "Optional[pd.DataFrame]":
    if not HAS_PANDAS:
        return None
    try:
        return _pd.read_csv(path)
    except Exception:
        return None


def load_csv_page(path: Path, page_index: int, page_size: int) -> "Optional[pd.DataFrame]":
    """Load one display page from a CSV without materializing the whole file."""
    if not HAS_PANDAS:
        return None
    try:
        page_index = max(0, int(page_index))
        page_size = max(1, int(page_size))
        start = page_index * page_size
        skiprows = range(1, start + 1) if start > 0 else None
        return _pd.read_csv(path, skiprows=skiprows, nrows=page_size)
    except Exception:
        return None


def csv_summary(path: Path) -> dict:
    """Return {rows, cols, columns, numeric_cols, size_kb, error}."""
    base = {
        "rows": 0, "cols": 0, "columns": [],
        "numeric_cols": [], "size_kb": 0.0, "error": None,
    }
    try:
        base["size_kb"] = path.stat().st_size / 1024
    except OSError:
        pass

    if not HAS_PANDAS:
        base["error"] = "pandas not installed"
        return base

    try:
        header = _pd.read_csv(path, nrows=0)
        sample = _pd.read_csv(path, nrows=1000)
    except Exception:
        base["error"] = "failed to parse"
        return base

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            line_count = sum(1 for _ in fh)
        base["rows"] = max(0, line_count - 1)
    except OSError:
        base["rows"] = len(sample)

    base["cols"] = len(header.columns)
    base["columns"] = list(header.columns)
    base["numeric_cols"] = list(sample.select_dtypes(include="number").columns)
    return base


def detect_aggregate_csv(folder: Path) -> Optional[Path]:
    """Heuristic: find the 'aggregate' CSV in a result folder."""
    for p in folder.glob("*aggregate*.csv"):
        return p
    for p in folder.glob("*summary*.csv"):
        return p
    csvs = sorted(folder.glob("*.csv"))
    return csvs[0] if csvs else None
