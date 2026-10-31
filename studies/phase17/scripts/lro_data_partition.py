"""WP0.4: the LRO data partition. Hold-out days are refused unless explicitly unsealed.

The partition lives in ``studies/phase17/configs/lro_data_partition.json``. Every LRO tracking
reader calls :func:`require_readable_day` before it opens a day directory.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PARTITION_PATH = Path(__file__).resolve().parents[1] / "configs" / "lro_data_partition.json"


class HoldoutDataError(RuntimeError):
    """A sealed hold-out day was requested outside the predeclared one-time evaluation."""


def load_partition(path: Path = PARTITION_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_readable_day(day_dir: str, partition: dict | None = None) -> None:
    """Raise :class:`HoldoutDataError` for a sealed hold-out day (``day_dir`` ends in YYYYDDD)."""
    partition = load_partition() if partition is None else partition
    day = Path(day_dir).name
    if day in partition["excluded"]["days"]:
        raise HoldoutDataError(f"LRO day {day} is excluded: {partition['excluded']['reason']}")
    holdout = partition["holdout"]
    if day not in holdout["days"]:
        return
    unseal = holdout["unseal"]
    required = unseal["required_value"]
    given = os.environ.get(unseal["environment_variable"])
    if required is None or given != required:
        raise HoldoutDataError(
            f"LRO day {day} is a sealed hold-out day ({PARTITION_PATH.name}); it is read only by the "
            f"predeclared one-time evaluation (WP5.4) with {unseal['environment_variable']} set to its "
            "predeclaration path.")
