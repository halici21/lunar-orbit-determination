"""Measurement CSV ingestion helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .scenarios import MeasurementType


@dataclass(frozen=True)
class IngestedMeasurements:
    measurement_type: MeasurementType
    obs_data: np.ndarray
    station_names: tuple[str, ...]


POSITION_COLUMNS = ("t_s", "range_m", "az_rad", "el_rad", "station_id", "time_index")
RANGE_RATE_COLUMNS = ("t_s", "range_m", "range_rate_mps", "az_rad", "el_rad", "station_id", "time_index")


def read_measurement_csv(
    path,
    measurement_type: MeasurementType,
    *,
    station_names: Sequence[str] = (),
) -> IngestedMeasurements:
    """Read position or range-rate observations into the internal ObsData layout."""
    if measurement_type not in {"position", "range_rate"}:
        raise ValueError("measurement_type must be 'position' or 'range_rate'.")
    path = Path(path)
    station_names = tuple(station_names)
    rows: list[list[float]] = []

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Measurement CSV must have a header row.")
        _validate_headers(reader.fieldnames, measurement_type)
        for line_number, row in enumerate(reader, start=2):
            rows.append(_parse_row(row, measurement_type, station_names, line_number))

    if not rows:
        width = 6 if measurement_type == "position" else 7
        return IngestedMeasurements(measurement_type, np.zeros((0, width), dtype=float), station_names)
    return IngestedMeasurements(measurement_type, np.asarray(rows, dtype=float), station_names)


def _validate_headers(fieldnames: Sequence[str], measurement_type: MeasurementType) -> None:
    fields = set(fieldnames)
    required = set(POSITION_COLUMNS if measurement_type == "position" else RANGE_RATE_COLUMNS)
    if "station_name" in fields:
        required.discard("station_id")
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError(f"Measurement CSV missing required column(s): {', '.join(sorted(missing))}")


def _parse_row(
    row: dict[str, str],
    measurement_type: MeasurementType,
    station_names: tuple[str, ...],
    line_number: int,
) -> list[float]:
    station_id = _station_id(row, station_names, line_number)
    arc_id = row.get("arc_id", "")
    if measurement_type == "position":
        values = [
            _float(row, "t_s", line_number),
            _float(row, "range_m", line_number),
            _float(row, "az_rad", line_number),
            _float(row, "el_rad", line_number),
            float(station_id),
            _float(row, "time_index", line_number),
        ]
    else:
        values = [
            _float(row, "t_s", line_number),
            _float(row, "range_m", line_number),
            _float(row, "range_rate_mps", line_number),
            _float(row, "az_rad", line_number),
            _float(row, "el_rad", line_number),
            float(station_id),
            _float(row, "time_index", line_number),
        ]
    if arc_id not in {"", None}:
        values.append(_float(row, "arc_id", line_number))
    return values


def _station_id(row: dict[str, str], station_names: tuple[str, ...], line_number: int) -> int:
    station_name = row.get("station_name", "")
    if station_name:
        if not station_names:
            raise ValueError("station_name column requires station_names to be provided.")
        try:
            return station_names.index(station_name) + 1
        except ValueError as exc:
            raise ValueError(f"Unknown station_name {station_name!r} on line {line_number}.") from exc

    station_id = int(_float(row, "station_id", line_number))
    if station_id <= 0:
        raise ValueError(f"station_id must be one-based and positive on line {line_number}.")
    if station_names and station_id > len(station_names):
        raise ValueError(f"station_id out of range on line {line_number}.")
    return station_id


def _float(row: dict[str, str], column: str, line_number: int) -> float:
    try:
        return float(row[column])
    except KeyError as exc:
        raise ValueError(f"Missing column {column!r} on line {line_number}.") from exc
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value for {column!r} on line {line_number}.") from exc
