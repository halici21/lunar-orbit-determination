"""CCSDS Orbit Data Messages (CCSDS 502.0-B-3): KVN OEM and OPM writers (WP6.2).

Writes estimated trajectories and their covariances as Orbit Ephemeris Messages
(OEM: ephemeris data lines plus covariance blocks) and Orbit Parameter Messages
(OPM: one state vector plus its covariance), in Keyword = Value Notation (KVN).
A minimal reader for the same subset is provided for round-trip tests and for
inspecting written files; it is not a general ODM parser.

Library-to-CCSDS contract (stated in every written file as COMMENT lines):

* Library states are Moon-centred with SPICE ``J2000`` axes ("MCI" in this code
  base). SPICE ``J2000`` is the CCSDS/SANA ``EME2000`` frame, so the defaults are
  ``CENTER_NAME = MOON`` and ``REF_FRAME = EME2000``. Do **not** write the SANA
  name ``MCI``: in SANA it denotes the *Mars* Centred Inertial frame.
* Library epochs are SPICE ET, i.e. TDB seconds past J2000.0
  (2000-01-01T12:00:00 TDB). They are written with ``TIME_SYSTEM = TDB``; the
  calendar conversion is exact integer/rational arithmetic on the binary value
  of each epoch (TDB has no leap seconds), rounded to ``epoch_decimals`` digits.
* Library units are m, m/s, m^2...; ODM units are km, km/s, km^2, km^2/s and
  km^2/s^2. Numbers are written with 17 significant digits, so the kilometre
  value read back is bit-identical to the kilometre value written.
* Covariances are written as the lower triangle, row by row, in the order
  X, Y, Z, X_DOT, Y_DOT, Z_DOT (502.0-B-3 OEM Section 5.2.5, OPM Table 3-3).
  Only the 6x6 position/velocity block is written; augmented (bias) states are
  dropped by :func:`oem_segment_from_filter_result`.

Scenario time arguments are elapsed seconds; convert them with the scenario's
``et0_s`` (``epoch_et_s = et0_s + t_s``) before writing.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "CCSDS_ODM_VERSIONS",
    "OdmFormatError",
    "OemCovariance",
    "OemSegment",
    "OemMessage",
    "OpmMessage",
    "SUPPORTED_REF_FRAMES",
    "SUPPORTED_CENTER_NAMES",
    "et_to_tdb_calendar",
    "tdb_calendar_to_et",
    "oem_segment_from_filter_result",
    "format_oem",
    "format_opm",
    "write_oem",
    "write_opm",
    "read_oem",
    "read_opm",
]

#: Message versions this module writes and reads (502.0-B-2 and 502.0-B-3).
CCSDS_ODM_VERSIONS = ("2.0", "3.0")

#: Inertial SANA reference frames the writer accepts. The library's states are
#: SPICE J2000 axes, i.e. ``EME2000``; ``ICRF`` is accepted only for callers who
#: have rotated their states themselves (the ~20 mas frame bias is not applied
#: here). Rotating frames are rejected: this writer performs no frame transform.
SUPPORTED_REF_FRAMES = ("EME2000", "ICRF")

#: Origins accepted for ``CENTER_NAME``. The library's states are Moon-centred.
SUPPORTED_CENTER_NAMES = ("MOON", "EARTH", "EARTH-MOON BARYCENTER", "SUN",
                          "SOLAR SYSTEM BARYCENTER")

#: Only TDB is written: library epochs are SPICE ET (= TDB seconds past J2000).
TIME_SYSTEM = "TDB"

_J2000_SECONDS_FROM_MIDNIGHT = 43200
_J2000_DATE = date(2000, 1, 1)
_KM = 1.0e3
_STATE_KEYS = ("X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT")
_OPM_COV_KEYS = tuple(
    f"C{_STATE_KEYS[i]}_{_STATE_KEYS[j]}" for i in range(6) for j in range(i + 1)
)
_OEM_INTERPOLATION = ("HERMITE", "LAGRANGE", "LINEAR")
_EPOCH_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(\.\d+)?Z?$")
_DOY_EPOCH_RE = re.compile(r"^(\d{4})-(\d{3})T(\d{2}):(\d{2}):(\d{2})(\.\d+)?Z?$")
_NAME_RE = re.compile(r"^[^\s=].*$")
_SYMMETRY_RTOL = 1.0e-9


class OdmFormatError(ValueError):
    """Raised for an invalid ODM input to a writer or a malformed file on read."""


# ---------------------------------------------------------------------------
# Epochs
# ---------------------------------------------------------------------------

def et_to_tdb_calendar(et_s: float, *, decimals: int = 6) -> str:
    """Format SPICE ET (TDB s past J2000) as a CCSDS ``YYYY-MM-DDThh:mm:ss.f`` TDB string.

    Exact: the binary value of ``et_s`` is converted with rational arithmetic and
    rounded half-to-even once, to ``decimals`` fractional digits (0..9).
    """
    if not isinstance(decimals, int) or not 0 <= decimals <= 9:
        raise OdmFormatError(f"epoch decimals must be an int in 0..9, got {decimals!r}")
    value = float(et_s)
    if not math.isfinite(value):
        raise OdmFormatError(f"epoch must be finite, got {et_s!r}")
    scale = 10 ** decimals
    # Integer count of 10^-decimals seconds since 2000-01-01T00:00:00 TDB.
    ticks = round((Fraction(value) + _J2000_SECONDS_FROM_MIDNIGHT) * scale)
    days, tick_of_day = divmod(ticks, 86400 * scale)
    try:
        day = _J2000_DATE + timedelta(days=days)
    except OverflowError as exc:
        raise OdmFormatError(f"epoch {et_s!r} is outside the calendar range") from exc
    whole, frac = divmod(tick_of_day, scale)
    hh, rem = divmod(whole, 3600)
    mm, ss = divmod(rem, 60)
    text = f"{day.isoformat()}T{hh:02d}:{mm:02d}:{ss:02d}"
    if decimals:
        text += "." + f"{frac:0{decimals}d}"
    return text


def tdb_calendar_to_et(text: str) -> float:
    """Parse a CCSDS calendar or day-of-year TDB epoch to SPICE ET seconds (nearest double)."""
    value = text.strip()
    match = _EPOCH_RE.match(value)
    if match:
        year, month, dom, hh, mm, ss, frac = match.groups()
        try:
            day = date(int(year), int(month), int(dom))
        except ValueError as exc:
            raise OdmFormatError(f"invalid epoch {text!r}: {exc}") from exc
    else:
        match = _DOY_EPOCH_RE.match(value)
        if not match:
            raise OdmFormatError(f"invalid CCSDS epoch {text!r}")
        year, doy, hh, mm, ss, frac = match.groups()
        if not 1 <= int(doy) <= (366 if _is_leap(int(year)) else 365):
            raise OdmFormatError(f"invalid day of year in epoch {text!r}")
        day = date(int(year), 1, 1) + timedelta(days=int(doy) - 1)
    hh_i, mm_i, ss_i = int(hh), int(mm), int(ss)
    # TDB has no leap seconds: second 60 is invalid.
    if hh_i > 23 or mm_i > 59 or ss_i > 59:
        raise OdmFormatError(f"invalid time of day in TDB epoch {text!r}")
    seconds = Fraction((day - _J2000_DATE).days * 86400 + hh_i * 3600 + mm_i * 60 + ss_i)
    if frac:
        seconds += Fraction(int(frac[1:]), 10 ** (len(frac) - 1))
    return float(seconds - _J2000_SECONDS_FROM_MIDNIGHT)


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OemCovariance:
    """One OEM covariance block: epoch (ET s) and a 6x6 covariance in m, m/s units."""

    epoch_et_s: float
    covariance_m: np.ndarray
    cov_ref_frame: str | None = None


@dataclass(frozen=True)
class OemSegment:
    """One OEM segment: increasing epochs (ET s), states (N, 6) in m, m/s, covariances.

    ``usable_start_et_s`` / ``usable_stop_et_s`` default to the segment bounds and
    are written as USEABLE_START_TIME / USEABLE_STOP_TIME (the standard's spelling).
    """

    epochs_et_s: np.ndarray
    states_m: np.ndarray
    object_name: str
    object_id: str
    covariances: tuple[OemCovariance, ...] = ()
    center_name: str = "MOON"
    ref_frame: str = "EME2000"
    interpolation: str | None = None
    interpolation_degree: int | None = None
    usable_start_et_s: float | None = None
    usable_stop_et_s: float | None = None
    comments: tuple[str, ...] = ()


@dataclass(frozen=True)
class OemMessage:
    """An OEM as read back by :func:`read_oem` (values converted to m, m/s)."""

    version: str
    header: dict
    segments: tuple[OemSegment, ...]
    epoch_text: tuple[tuple[str, ...], ...] = field(default=())
    #: The kilometre values exactly as parsed from each segment's data lines.
    states_km: tuple[np.ndarray, ...] = field(default=())


@dataclass(frozen=True)
class OpmMessage:
    """An OPM state (m, m/s) with optional covariance, as written or read back."""

    epoch_et_s: float
    state_m: np.ndarray
    object_name: str
    object_id: str
    covariance_m: np.ndarray | None = None
    center_name: str = "MOON"
    ref_frame: str = "EME2000"
    cov_ref_frame: str | None = None
    version: str = "3.0"
    header: dict = field(default_factory=dict)
    epoch_text: str = ""
    #: The kilometre values exactly as parsed (``None`` for a message built in memory).
    state_km: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Validation and formatting helpers
# ---------------------------------------------------------------------------

def _num(value: float) -> str:
    """17 significant digits: the double read back is the double written."""
    return f"{float(value):.16e}"


def _check_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
        raise OdmFormatError(f"{name} must be a non-empty single-line string, got {value!r}")
    if not _NAME_RE.match(value.strip()):
        raise OdmFormatError(f"{name} {value!r} is not a valid KVN value")
    return value.strip()


def _check_frame(name: str, frame: str) -> str:
    frame = _check_text(name, frame).upper()
    if frame not in SUPPORTED_REF_FRAMES:
        raise OdmFormatError(
            f"{name} {frame!r} is not supported; this writer performs no frame "
            f"transformation and accepts only {SUPPORTED_REF_FRAMES} (library states "
            "are SPICE J2000 = EME2000). Note SANA 'MCI' is Mars-centred.")
    return frame


def _check_center(center: str) -> str:
    center = _check_text("CENTER_NAME", center).upper()
    if center not in SUPPORTED_CENTER_NAMES:
        raise OdmFormatError(
            f"CENTER_NAME {center!r} is not one of {SUPPORTED_CENTER_NAMES}")
    return center


def _check_version(version: str) -> str:
    if version not in CCSDS_ODM_VERSIONS:
        raise OdmFormatError(f"message version must be one of {CCSDS_ODM_VERSIONS}, got {version!r}")
    return version


def _check_state_rows(states, n_expected: int | None = None) -> np.ndarray:
    arr = np.asarray(states, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise OdmFormatError(f"states must have shape (N, 6) in m and m/s, got {arr.shape}")
    if n_expected is not None and arr.shape[0] != n_expected:
        raise OdmFormatError(f"{arr.shape[0]} states for {n_expected} epochs")
    if not np.all(np.isfinite(arr)):
        raise OdmFormatError("states contain non-finite values")
    return arr


def _check_covariance(covariance) -> np.ndarray:
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (6, 6):
        raise OdmFormatError(
            f"covariance must be the 6x6 position/velocity block, got shape {cov.shape}")
    if not np.all(np.isfinite(cov)):
        raise OdmFormatError("covariance contains non-finite values")
    scale = np.sqrt(np.abs(np.outer(np.diag(cov), np.diag(cov))))
    if np.any(np.abs(cov - cov.T) > _SYMMETRY_RTOL * np.maximum(scale, np.finfo(float).tiny)):
        raise OdmFormatError("covariance is not symmetric (relative tolerance 1e-9)")
    if np.any(np.diag(cov) < 0.0):
        raise OdmFormatError("covariance has a negative variance on the diagonal")
    return 0.5 * (cov + cov.T)


def _covariance_to_km(cov_m: np.ndarray) -> np.ndarray:
    # Position/velocity blocks: km^2, km^2/s, km^2/s^2 -> all are m-units / 1e6.
    return cov_m / (_KM * _KM)


def _lower_triangle_rows(cov_km: np.ndarray) -> list[str]:
    return [" ".join(_num(cov_km[i, j]) for j in range(i + 1)) for i in range(6)]


def _header_lines(kind: str, version: str, originator: str, creation_date: str | None,
                  message_id: str | None, comments: Sequence[str]) -> list[str]:
    created = creation_date or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    lines = [f"CCSDS_{kind}_VERS = {version}"]
    lines += [f"COMMENT {_check_comment(c)}" for c in comments]
    lines.append(f"CREATION_DATE = {_check_text('CREATION_DATE', created)}")
    lines.append(f"ORIGINATOR = {_check_text('ORIGINATOR', originator)}")
    if message_id is not None:
        if version == "2.0":
            raise OdmFormatError("MESSAGE_ID requires message version 3.0")
        lines.append(f"MESSAGE_ID = {_check_text('MESSAGE_ID', message_id)}")
    return lines


def _check_comment(comment: str) -> str:
    if "\n" in comment or "\r" in comment:
        raise OdmFormatError(f"COMMENT must be a single line, got {comment!r}")
    return comment.strip()


def _contract_comments(center: str, frame: str) -> list[str]:
    return [
        "Written by lunar_od.ccsds_odm (CCSDS 502.0-B-3 KVN).",
        f"States: {center}-centred, axes {frame} (library MCI = SPICE J2000 = EME2000).",
        "Epochs: SPICE ET = TDB seconds past J2000.0, written as TDB calendar.",
        "Units: km, km/s; covariance km**2, km**2/s, km**2/s**2.",
    ]


# ---------------------------------------------------------------------------
# OEM writer
# ---------------------------------------------------------------------------

def _validated_segment(seg: OemSegment) -> OemSegment:
    epochs = np.asarray(seg.epochs_et_s, dtype=float)
    if epochs.ndim != 1 or epochs.size < 1:
        raise OdmFormatError("segment epochs must be a non-empty 1-D array")
    if not np.all(np.isfinite(epochs)):
        raise OdmFormatError("segment epochs contain non-finite values")
    if epochs.size > 1 and not np.all(np.diff(epochs) > 0.0):
        raise OdmFormatError("OEM epochs must be strictly increasing within a segment")
    states = _check_state_rows(seg.states_m, epochs.size)
    frame = _check_frame("REF_FRAME", seg.ref_frame)
    center = _check_center(seg.center_name)
    start = float(epochs[0]) if seg.usable_start_et_s is None else float(seg.usable_start_et_s)
    stop = float(epochs[-1]) if seg.usable_stop_et_s is None else float(seg.usable_stop_et_s)
    if not (epochs[0] <= start <= stop <= epochs[-1]):
        raise OdmFormatError("useable start/stop must satisfy START <= USEABLE_START <= "
                             "USEABLE_STOP <= STOP")
    if seg.interpolation is not None:
        if seg.interpolation.upper() not in _OEM_INTERPOLATION:
            raise OdmFormatError(f"INTERPOLATION must be one of {_OEM_INTERPOLATION}")
        if seg.interpolation_degree is None or not (
                isinstance(seg.interpolation_degree, int) and 1 <= seg.interpolation_degree):
            raise OdmFormatError("INTERPOLATION requires a positive integer INTERPOLATION_DEGREE")
    elif seg.interpolation_degree is not None:
        raise OdmFormatError("INTERPOLATION_DEGREE given without INTERPOLATION")
    covs = []
    previous = -math.inf
    for cov in seg.covariances:
        e = float(cov.epoch_et_s)
        if not (math.isfinite(e) and epochs[0] <= e <= epochs[-1]):
            raise OdmFormatError(f"covariance epoch {e!r} lies outside the segment span")
        if e < previous:
            raise OdmFormatError("covariance epochs must be non-decreasing")
        previous = e
        cov_frame = None if cov.cov_ref_frame is None else _check_frame("COV_REF_FRAME", cov.cov_ref_frame)
        covs.append(OemCovariance(e, _check_covariance(cov.covariance_m), cov_frame))
    return OemSegment(
        epochs_et_s=epochs, states_m=states,
        object_name=_check_text("OBJECT_NAME", seg.object_name),
        object_id=_check_text("OBJECT_ID", seg.object_id),
        covariances=tuple(covs), center_name=center, ref_frame=frame,
        interpolation=None if seg.interpolation is None else seg.interpolation.upper(),
        interpolation_degree=seg.interpolation_degree,
        usable_start_et_s=start, usable_stop_et_s=stop,
        comments=tuple(_check_comment(c) for c in seg.comments))


def format_oem(segments: OemSegment | Iterable[OemSegment], *, originator: str,
               version: str = "3.0", creation_date: str | None = None,
               message_id: str | None = None, epoch_decimals: int = 6,
               header_comments: Sequence[str] = ()) -> str:
    """Return a KVN OEM (502.0-B-3, Section 5) for one or more segments.

    Each segment carries the 6-element states at its epochs and an optional
    covariance section (lower triangle, km**2 units). ``creation_date`` (UTC
    calendar string) defaults to now; pass it for byte-reproducible output.
    """
    version = _check_version(version)
    if isinstance(segments, OemSegment):
        segments = [segments]
    segs = [_validated_segment(s) for s in segments]
    if not segs:
        raise OdmFormatError("an OEM needs at least one segment")
    fmt = lambda e: et_to_tdb_calendar(e, decimals=epoch_decimals)  # noqa: E731
    lines = _header_lines("OEM", version, originator, creation_date, message_id,
                          list(header_comments))
    for seg in segs:
        lines += ["", "META_START"]
        lines += [f"COMMENT {c}" for c in _contract_comments(seg.center_name, seg.ref_frame)]
        lines += [f"COMMENT {c}" for c in seg.comments]
        lines += [
            f"OBJECT_NAME = {seg.object_name}",
            f"OBJECT_ID = {seg.object_id}",
            f"CENTER_NAME = {seg.center_name}",
            f"REF_FRAME = {seg.ref_frame}",
            f"TIME_SYSTEM = {TIME_SYSTEM}",
            f"START_TIME = {fmt(seg.epochs_et_s[0])}",
            f"USEABLE_START_TIME = {fmt(seg.usable_start_et_s)}",
            f"USEABLE_STOP_TIME = {fmt(seg.usable_stop_et_s)}",
            f"STOP_TIME = {fmt(seg.epochs_et_s[-1])}",
        ]
        if seg.interpolation is not None:
            lines.append(f"INTERPOLATION = {seg.interpolation}")
            lines.append(f"INTERPOLATION_DEGREE = {seg.interpolation_degree}")
        lines += ["META_STOP", ""]
        states_km = seg.states_m / _KM
        for epoch, row in zip(seg.epochs_et_s, states_km):
            lines.append(fmt(epoch) + " " + " ".join(_num(v) for v in row))
        if seg.covariances:
            lines += ["", "COVARIANCE_START"]
            for cov in seg.covariances:
                lines.append(f"EPOCH = {fmt(cov.epoch_et_s)}")
                lines.append(f"COV_REF_FRAME = {cov.cov_ref_frame or seg.ref_frame}")
                lines += _lower_triangle_rows(_covariance_to_km(cov.covariance_m))
            lines.append("COVARIANCE_STOP")
    return "\n".join(lines) + "\n"


def write_oem(path, segments, **kwargs) -> Path:
    """Write :func:`format_oem` output to ``path`` (UTF-8 subset: ASCII) and return it."""
    text = format_oem(segments, **kwargs)
    out = Path(path)
    out.write_text(text, encoding="ascii", newline="\n")
    return out


def oem_segment_from_filter_result(result, et0_s: float, *, object_name: str,
                                   object_id: str, covariance_stride: int = 1,
                                   **segment_kwargs) -> OemSegment:
    """Build an :class:`OemSegment` from a sequential filter result (UKF/EKF).

    Uses ``result.t_update_s`` (elapsed s; epochs become ``et0_s + t``),
    ``result.state_estimates`` and ``result.covariances``; only the leading 6x6
    position/velocity block of an augmented state is exported. Every
    ``covariance_stride``-th covariance is written (0 writes none).
    """
    t = np.asarray(result.t_update_s, dtype=float)
    states = np.asarray(result.state_estimates, dtype=float)
    if states.ndim != 2 or states.shape[1] < 6 or states.shape[0] != t.size:
        raise OdmFormatError("filter result states must be (N, >=6) matching t_update_s")
    epochs = float(et0_s) + t
    covs: list[OemCovariance] = []
    if covariance_stride:
        p = np.asarray(result.covariances, dtype=float)
        if p.ndim != 3 or p.shape[0] != t.size or p.shape[1] < 6 or p.shape[2] < 6:
            raise OdmFormatError("filter result covariances must be (N, >=6, >=6)")
        covs = [OemCovariance(float(epochs[k]), p[k, :6, :6])
                for k in range(0, t.size, int(covariance_stride))]
    return OemSegment(epochs_et_s=epochs, states_m=states[:, :6], object_name=object_name,
                      object_id=object_id, covariances=tuple(covs), **segment_kwargs)


# ---------------------------------------------------------------------------
# OPM writer
# ---------------------------------------------------------------------------

def format_opm(epoch_et_s: float, state_m, *, object_name: str, object_id: str,
               originator: str, covariance_m=None, center_name: str = "MOON",
               ref_frame: str = "EME2000", cov_ref_frame: str | None = None,
               version: str = "3.0", creation_date: str | None = None,
               message_id: str | None = None, epoch_decimals: int = 6,
               header_comments: Sequence[str] = ()) -> str:
    """Return a KVN OPM (502.0-B-3, Section 3) with a state vector and optional covariance."""
    version = _check_version(version)
    state = _check_state_rows(np.reshape(np.asarray(state_m, dtype=float), (1, -1))
                              if np.size(state_m) == 6 else state_m, 1)[0]
    frame = _check_frame("REF_FRAME", ref_frame)
    center = _check_center(center_name)
    epoch = et_to_tdb_calendar(epoch_et_s, decimals=epoch_decimals)
    lines = _header_lines("OPM", version, originator, creation_date, message_id,
                          list(header_comments))
    lines.append("")
    lines += [f"COMMENT {c}" for c in _contract_comments(center, frame)]
    lines += [
        f"OBJECT_NAME = {_check_text('OBJECT_NAME', object_name)}",
        f"OBJECT_ID = {_check_text('OBJECT_ID', object_id)}",
        f"CENTER_NAME = {center}",
        f"REF_FRAME = {frame}",
        f"TIME_SYSTEM = {TIME_SYSTEM}",
        "",
        f"EPOCH = {epoch}",
    ]
    units = ("km",) * 3 + ("km/s",) * 3
    state_km = state / _KM
    lines += [f"{k} = {_num(v)} [{u}]" for k, v, u in zip(_STATE_KEYS, state_km, units)]
    if covariance_m is not None:
        cov_km = _covariance_to_km(_check_covariance(covariance_m))
        cov_frame = frame if cov_ref_frame is None else _check_frame("COV_REF_FRAME", cov_ref_frame)
        lines += ["", f"COV_REF_FRAME = {cov_frame}"]
        for i in range(6):
            for j in range(i + 1):
                unit = ("km**2", "km**2/s", "km**2/s**2")[(i >= 3) + (j >= 3)]
                lines.append(f"C{_STATE_KEYS[i]}_{_STATE_KEYS[j]} = {_num(cov_km[i, j])} [{unit}]")
    elif cov_ref_frame is not None:
        raise OdmFormatError("COV_REF_FRAME given without a covariance")
    return "\n".join(lines) + "\n"


def write_opm(path, epoch_et_s: float, state_m, **kwargs) -> Path:
    """Write :func:`format_opm` output to ``path`` and return it."""
    out = Path(path)
    out.write_text(format_opm(epoch_et_s, state_m, **kwargs), encoding="ascii", newline="\n")
    return out


# ---------------------------------------------------------------------------
# Minimal reader (round-trip subset)
# ---------------------------------------------------------------------------

def _kvn(line: str, lineno: int) -> tuple[str, str]:
    key, sep, value = line.partition("=")
    if not sep:
        raise OdmFormatError(f"line {lineno}: expected KEY = VALUE, got {line!r}")
    value = value.strip()
    unit = re.search(r"\s*\[[^\]]*\]$", value)
    if unit:
        value = value[:unit.start()].strip()
    return key.strip().upper(), value


def _float(text: str, lineno: int) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise OdmFormatError(f"line {lineno}: {text!r} is not a number") from exc
    if not math.isfinite(value):
        raise OdmFormatError(f"line {lineno}: non-finite value {text!r}")
    return value


def _lines(source) -> list[str]:
    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source
                                    and _is_file(source)):
        source = Path(source).read_text(encoding="utf-8")
    return [line.strip() for line in str(source).splitlines()]


def _is_file(text: str) -> bool:
    try:
        return Path(text).is_file()
    except (OSError, ValueError):
        return False


def _check_time_system(value: str, lineno: int) -> None:
    if value.upper() != TIME_SYSTEM:
        raise OdmFormatError(
            f"line {lineno}: TIME_SYSTEM {value!r} not supported by this minimal reader (TDB only)")


def _read_header(lines, kind) -> tuple[str, dict, int]:
    idx = 0
    while idx < len(lines) and (not lines[idx] or lines[idx].startswith("COMMENT")):
        idx += 1
    if idx >= len(lines):
        raise OdmFormatError("empty message")
    key, version = _kvn(lines[idx], idx + 1)
    if key != f"CCSDS_{kind}_VERS":
        raise OdmFormatError(f"line {idx + 1}: expected CCSDS_{kind}_VERS, got {key}")
    _check_version(version)
    header: dict = {}
    idx += 1
    while idx < len(lines):
        line = lines[idx]
        if not line or line.startswith("COMMENT"):
            idx += 1
            continue
        if "=" not in line:
            break
        key, value = _kvn(line, idx + 1)
        if key not in ("CREATION_DATE", "ORIGINATOR", "MESSAGE_ID", "CLASSIFICATION"):
            break
        header[key] = value
        idx += 1
    for required in ("CREATION_DATE", "ORIGINATOR"):
        if required not in header:
            raise OdmFormatError(f"header is missing {required}")
    return version, header, idx


def _covariance_from_rows(rows: list[list[float]], lineno: int) -> np.ndarray:
    if [len(r) for r in rows] != [1, 2, 3, 4, 5, 6]:
        raise OdmFormatError(f"line {lineno}: covariance must be 6 lower-triangular rows of 1..6 values")
    cov = np.zeros((6, 6))
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            cov[i, j] = cov[j, i] = v
    return cov * (_KM * _KM)


def read_oem(source) -> OemMessage:
    """Read a KVN OEM written by :func:`format_oem` (path or text); values in m, m/s."""
    lines = _lines(source)
    version, header, idx = _read_header(lines, "OEM")
    segments: list[OemSegment] = []
    epoch_texts: list[tuple[str, ...]] = []
    states_km: list[np.ndarray] = []
    required = ("OBJECT_NAME", "OBJECT_ID", "CENTER_NAME", "REF_FRAME", "TIME_SYSTEM",
                "START_TIME", "STOP_TIME")
    n = len(lines)
    while idx < n:
        line = lines[idx]
        if not line or line.startswith("COMMENT"):
            idx += 1
            continue
        if line != "META_START":
            raise OdmFormatError(f"line {idx + 1}: expected META_START, got {line!r}")
        idx += 1
        meta: dict = {}
        while idx < n and lines[idx] != "META_STOP":
            if lines[idx] and not lines[idx].startswith("COMMENT"):
                if "=" not in lines[idx]:
                    raise OdmFormatError(f"line {idx + 1}: data before META_STOP")
                key, value = _kvn(lines[idx], idx + 1)
                meta[key] = value
            idx += 1
        if idx >= n:
            raise OdmFormatError("META_START without META_STOP")
        idx += 1
        missing = [k for k in required if k not in meta]
        if missing:
            raise OdmFormatError(f"segment metadata is missing {missing}")
        _check_time_system(meta["TIME_SYSTEM"], idx)
        epochs, texts, states = [], [], []
        covs: list[OemCovariance] = []
        while idx < n and lines[idx] != "META_START":
            line = lines[idx]
            if not line or line.startswith("COMMENT"):
                idx += 1
                continue
            if line == "COVARIANCE_START":
                idx += 1
                while idx < n and lines[idx] != "COVARIANCE_STOP":
                    if not lines[idx] or lines[idx].startswith("COMMENT"):
                        idx += 1
                        continue
                    key, value = _kvn(lines[idx], idx + 1)
                    if key != "EPOCH":
                        raise OdmFormatError(f"line {idx + 1}: covariance block must start with EPOCH")
                    cov_epoch = tdb_calendar_to_et(value)
                    idx += 1
                    cov_frame = None
                    if idx < n and lines[idx].upper().startswith("COV_REF_FRAME"):
                        cov_frame = _kvn(lines[idx], idx + 1)[1].upper()
                        idx += 1
                    rows = []
                    start_line = idx + 1
                    while len(rows) < 6 and idx < n:
                        if lines[idx] and not lines[idx].startswith("COMMENT"):
                            rows.append([_float(t, idx + 1) for t in lines[idx].split()])
                        idx += 1
                    covs.append(OemCovariance(cov_epoch, _covariance_from_rows(rows, start_line),
                                              cov_frame))
                if idx >= n:
                    raise OdmFormatError("COVARIANCE_START without COVARIANCE_STOP")
                idx += 1
                continue
            tokens = line.split()
            if len(tokens) not in (7, 10):
                raise OdmFormatError(
                    f"line {idx + 1}: ephemeris line needs epoch + 6 (or 9) values, got {len(tokens)}")
            texts.append(tokens[0])
            epochs.append(tdb_calendar_to_et(tokens[0]))
            states.append([_float(t, idx + 1) for t in tokens[1:7]])
            idx += 1
        if not epochs:
            raise OdmFormatError("segment has no ephemeris data lines")
        degree = meta.get("INTERPOLATION_DEGREE")
        segment = OemSegment(
            epochs_et_s=np.asarray(epochs), states_m=np.asarray(states) * _KM,
            object_name=meta["OBJECT_NAME"], object_id=meta["OBJECT_ID"],
            covariances=tuple(covs), center_name=meta["CENTER_NAME"].upper(),
            ref_frame=meta["REF_FRAME"].upper(), interpolation=meta.get("INTERPOLATION"),
            interpolation_degree=None if degree is None else int(degree),
            usable_start_et_s=tdb_calendar_to_et(meta.get("USEABLE_START_TIME", meta["START_TIME"])),
            usable_stop_et_s=tdb_calendar_to_et(meta.get("USEABLE_STOP_TIME", meta["STOP_TIME"])))
        if (tdb_calendar_to_et(meta["START_TIME"]) > epochs[0]
                or tdb_calendar_to_et(meta["STOP_TIME"]) < epochs[-1]):
            raise OdmFormatError("ephemeris lines fall outside START_TIME..STOP_TIME")
        segments.append(_validated_segment(segment))
        epoch_texts.append(tuple(texts))
        states_km.append(np.asarray(states))
    if not segments:
        raise OdmFormatError("OEM contains no segment")
    return OemMessage(version, header, tuple(segments), tuple(epoch_texts), tuple(states_km))


def read_opm(source) -> OpmMessage:
    """Read a KVN OPM written by :func:`format_opm` (path or text); values in m, m/s."""
    lines = _lines(source)
    version, header, idx = _read_header(lines, "OPM")
    values: dict = {}
    for lineno in range(idx, len(lines)):
        line = lines[lineno]
        if not line or line.startswith("COMMENT"):
            continue
        key, value = _kvn(line, lineno + 1)
        if key in values:
            raise OdmFormatError(f"line {lineno + 1}: duplicate keyword {key}")
        values[key] = (value, lineno + 1)
    required = ("OBJECT_NAME", "OBJECT_ID", "CENTER_NAME", "REF_FRAME", "TIME_SYSTEM",
                "EPOCH") + _STATE_KEYS
    missing = [k for k in required if k not in values]
    if missing:
        raise OdmFormatError(f"OPM is missing {missing}")
    _check_time_system(*values["TIME_SYSTEM"])
    state_km = np.array([_float(*values[k]) for k in _STATE_KEYS])
    state = state_km * _KM
    cov = None
    present = [k for k in _OPM_COV_KEYS if k in values]
    if present:
        if len(present) != len(_OPM_COV_KEYS):
            raise OdmFormatError("OPM covariance must contain all 21 lower-triangle terms")
        rows = [[_float(*values[f"C{_STATE_KEYS[i]}_{_STATE_KEYS[j]}"]) for j in range(i + 1)]
                for i in range(6)]
        cov = _check_covariance(_covariance_from_rows(rows, values[_OPM_COV_KEYS[0]][1]))
    cov_frame = values.get("COV_REF_FRAME", (None,))[0]
    return OpmMessage(
        epoch_et_s=tdb_calendar_to_et(values["EPOCH"][0]), state_m=state,
        object_name=values["OBJECT_NAME"][0], object_id=values["OBJECT_ID"][0],
        covariance_m=cov, center_name=_check_center(values["CENTER_NAME"][0]),
        ref_frame=_check_frame("REF_FRAME", values["REF_FRAME"][0]),
        cov_ref_frame=None if cov_frame is None else cov_frame.upper(),
        version=version, header=header, epoch_text=values["EPOCH"][0], state_km=state_km)
