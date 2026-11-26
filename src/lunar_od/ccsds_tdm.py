"""CCSDS Tracking Data Message (TDM, CCSDS 503.0-B-2) reader and writer, KVN form (F6 WP6.1, REQ-IO-001).

The reader is strict and fails closed: a keyword that is not in the standard's header table (3-2),
metadata table (3-3) or data-keyword table (3-5), a keyword out of the table order, a non-normative
enumerated value, a malformed time tag or number, a unit or any other token after a measurement
(KVN never displays units, 4.4), a duplicated keyword/time-tag pair (3.4.11) or a keyword whose
records are not chronological (3.4.10) raises :class:`TDMFormatError` naming the line.

Values keep the units of the standard (table 3-5): ``RANGE`` in ``RANGE_UNITS`` (km by default),
``ANGLE_1``/``ANGLE_2`` in deg, ``DOPPLER_*`` in km/s, frequencies in Hz. Measurement text is kept
verbatim, so a file read and written again reproduces its records exactly; text values of
enumerated metadata keywords are upper-cased (they are case-insensitive, 4.3.7).

The time system of the data time tags is the segment's ``TIME_SYSTEM``. Converting a tag to the
library's time basis (TDB seconds past an epoch ``et0``) is explicit: :func:`tdm_epochs_to_tdb_seconds`
supports UTC (TAI - UTC from the loaded leapseconds kernel), TAI, GPS, TT and TDB and refuses
every other system. Mapping records to the estimator's observation rows is
:func:`lunar_od.measurement_ingestion.read_measurement_tdm`.

XML TDMs are not read. Annex-E style examples used by the tests are written for this project.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "DATA_KEYWORD_UNITS",
    "TDMEpoch",
    "TDMFormatError",
    "TDMRecord",
    "TDMSegment",
    "TrackingDataMessage",
    "format_number",
    "format_tdm",
    "parse_tdm",
    "read_tdm",
    "tdm_epochs_to_tdb_seconds",
    "tdb_seconds_to_tdm_epoch",
    "write_tdm",
]

SUPPORTED_VERSIONS = ("1.0", "2.0")
MAX_LINE_LENGTH = 254
PARTICIPANT_INDICES = (1, 2, 3, 4, 5)

#: Header keywords in their mandatory order (table 3-2); COMMENT lines sit between the version and
#: CREATION_DATE (4.5.2a).
HEADER_KEYWORDS = ("CCSDS_TDM_VERS", "CREATION_DATE", "ORIGINATOR", "MESSAGE_ID")
_HEADER_MANDATORY = ("CCSDS_TDM_VERS", "CREATION_DATE", "ORIGINATOR")


def _indexed(stem: str) -> tuple[str, ...]:
    return tuple(f"{stem}_{n}" for n in PARTICIPANT_INDICES)


#: Metadata keywords in their mandatory order (table 3-3), META_START/META_STOP excluded.
METADATA_KEYWORDS: tuple[str, ...] = (
    "TRACK_ID", "DATA_TYPES", "TIME_SYSTEM", "START_TIME", "STOP_TIME",
    *_indexed("PARTICIPANT"),
    "MODE", "PATH", "PATH_1", "PATH_2",
    *_indexed("EPHEMERIS_NAME"),
    "TRANSMIT_BAND", "RECEIVE_BAND", "TURNAROUND_NUMERATOR", "TURNAROUND_DENOMINATOR",
    "TIMETAG_REF", "INTEGRATION_INTERVAL", "INTEGRATION_REF", "FREQ_OFFSET",
    "RANGE_MODE", "RANGE_MODULUS", "RANGE_UNITS", "ANGLE_TYPE", "REFERENCE_FRAME",
    "INTERPOLATION", "INTERPOLATION_DEGREE",
    "DOPPLER_COUNT_BIAS", "DOPPLER_COUNT_SCALE", "DOPPLER_COUNT_ROLLOVER",
    *_indexed("TRANSMIT_DELAY"), *_indexed("RECEIVE_DELAY"),
    "DATA_QUALITY",
    "CORRECTION_ANGLE_1", "CORRECTION_ANGLE_2", "CORRECTION_DOPPLER", "CORRECTION_MAG",
    "CORRECTION_RANGE", "CORRECTION_RCS", "CORRECTION_RECEIVE", "CORRECTION_TRANSMIT",
    "CORRECTION_ABERRATION_YEARLY", "CORRECTION_ABERRATION_DIURNAL",
    "CORRECTIONS_APPLIED",
)
_METADATA_RANK = {key: i for i, key in enumerate(METADATA_KEYWORDS)}

#: Normative enumerations of table 3-3 (text values are case-insensitive, 4.3.7).
METADATA_ENUMS: dict[str, tuple[str, ...]] = {
    "MODE": ("SEQUENTIAL", "SINGLE_DIFF"),
    "TIMETAG_REF": ("TRANSMIT", "RECEIVE"),
    "INTEGRATION_REF": ("START", "MIDDLE", "END"),
    "RANGE_MODE": ("COHERENT", "CONSTANT", "ONE_WAY"),
    "RANGE_UNITS": ("KM", "S", "RU"),
    "ANGLE_TYPE": ("AZEL", "RADEC", "XEYN", "XSYE"),
    "DOPPLER_COUNT_ROLLOVER": ("YES", "NO"),
    "DATA_QUALITY": ("RAW", "VALIDATED", "DEGRADED"),
    "CORRECTIONS_APPLIED": ("YES", "NO"),
}
#: Canonical spelling written back for the RANGE_UNITS values (the standard writes 'km', 's', 'RU').
_RANGE_UNITS_CANONICAL = {"KM": "km", "S": "s", "RU": "RU"}

#: SANA time-systems registry values accepted as TIME_SYSTEM (annex B); conversion supports fewer.
TIME_SYSTEMS = ("GMST", "GPS", "MET", "MRT", "SCLK", "TAI", "TCB", "TDB", "TCG", "TT", "UT1", "UTC")
_CONVERTIBLE_TIME_SYSTEMS = ("UTC", "TAI", "GPS", "TT", "TDB")

_METADATA_INTEGER = ("TURNAROUND_NUMERATOR", "TURNAROUND_DENOMINATOR", "INTERPOLATION_DEGREE",
                     "DOPPLER_COUNT_SCALE")
_METADATA_REAL = ("INTEGRATION_INTERVAL", "FREQ_OFFSET", "RANGE_MODULUS", "DOPPLER_COUNT_BIAS",
                  *_indexed("TRANSMIT_DELAY"), *_indexed("RECEIVE_DELAY"),
                  *(k for k in METADATA_KEYWORDS if k.startswith("CORRECTION_")))
_METADATA_POSITIVE = ("INTEGRATION_INTERVAL", "DOPPLER_COUNT_BIAS", "DOPPLER_COUNT_SCALE")
_METADATA_NON_NEGATIVE = ("RANGE_MODULUS", *_indexed("TRANSMIT_DELAY"), *_indexed("RECEIVE_DELAY"))
_METADATA_EPOCHS = ("START_TIME", "STOP_TIME")

#: Data keywords and their fixed units (table 3-5). ``_n`` keywords take n = 1..5.
DATA_KEYWORD_UNITS: dict[str, str] = {
    "ANGLE_1": "deg", "ANGLE_2": "deg", "CARRIER_POWER": "dBW", "CLOCK_BIAS": "s",
    "CLOCK_DRIFT": "s/s", "DOPPLER_COUNT": "n/a", "DOPPLER_INSTANTANEOUS": "km/s",
    "DOPPLER_INTEGRATED": "km/s", "DOR": "s", "MAG": "n/a", "PC_N0": "dBHz", "PR_N0": "dBHz",
    "PRESSURE": "hPa", "RANGE": "RANGE_UNITS", "RCS": "m**2", "RECEIVE_FREQ": "Hz",
    "RECEIVE_FREQ_n": "Hz", "RECEIVE_PHASE_CT_n": "n/a", "RHUMIDITY": "%", "STEC": "TECU",
    "TEMPERATURE": "K", "TRANSMIT_FREQ_n": "Hz", "TRANSMIT_FREQ_RATE_n": "Hz/s",
    "TRANSMIT_PHASE_CT_n": "n/a", "TROPO_DRY": "m", "TROPO_WET": "m", "VLBI_DELAY": "s",
}
#: Phase counts are digit strings of any length (4.3.11), not 16-digit doubles.
_PHASE_COUNT_STEMS = ("RECEIVE_PHASE_CT", "TRANSMIT_PHASE_CT")

_EPOCH_RE = re.compile(
    r"^(\d{4})-(?:(\d{2})-(\d{2})|(\d{3}))T(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)(Z?)$")
_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_FIXED_RE = re.compile(r"^[+-]?(\d+)\.(\d+)$")
_FLOAT_RE = re.compile(r"^[+-]?(\d)\.(\d+)(?:[eE]([+-]?\d+))?$")
_PHASE_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
_PATH_RE = re.compile(r"^[1-5](?:,[1-5])+$")
_KEYWORD_RE = re.compile(r"^[A-Z0-9_]+$")


class TDMFormatError(ValueError):
    """A TDM that does not follow CCSDS 503.0-B-2 (or uses a feature this reader refuses)."""


# ======================================================================
# values
# ======================================================================
@dataclass(frozen=True, order=True)
class TDMEpoch:
    """A CCSDS time tag ``YYYY-MM-DDThh:mm:ss[.d]`` (or day-of-year form), in the segment's time system.

    ``second`` may reach 60.x only for a UTC leap second. The text written back is the calendar form;
    a day-of-year tag reads to the same epoch.
    """

    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: float

    @property
    def seconds_of_day(self) -> float:
        return 3600.0 * self.hour + 60.0 * self.minute + self.second

    @classmethod
    def parse(cls, text: str) -> "TDMEpoch":
        m = _EPOCH_RE.match(text)
        if not m:
            raise TDMFormatError(f"time tag {text!r} is not YYYY-MM-DDThh:mm:ss[.d][Z] or YYYY-DDDThh:mm:ss[.d][Z]")
        year = int(m[1])
        try:
            if m[4] is not None:
                doy = int(m[4])
                if not 1 <= doy <= (366 if _is_leap(year) else 365):
                    raise ValueError
                d = date.fromordinal(date(year, 1, 1).toordinal() + doy - 1)
                month, day = d.month, d.day
            else:
                month, day = int(m[2]), int(m[3])
                date(year, month, day)
        except ValueError:
            raise TDMFormatError(f"time tag {text!r} names no calendar day") from None
        hour, minute, second = int(m[5]), int(m[6]), float(m[7])
        if hour > 23 or minute > 59 or second >= 61.0:
            raise TDMFormatError(f"time tag {text!r} has an out-of-range clock field")
        return cls(year, month, day, hour, minute, second)

    def format(self) -> str:
        if self.second == int(self.second):
            seconds = f"{int(self.second):02d}"
        else:
            # shortest positional text that reads back to the stored double
            ip, _, fp = _positional(self.second).partition(".")
            seconds = f"{int(ip):02d}.{fp}"
        return (f"{self.year:04d}-{self.month:02d}-{self.day:02d}T"
                f"{self.hour:02d}:{self.minute:02d}:{seconds}")

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.format()


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _positional(value: float) -> str:
    return np.format_float_positional(value, unique=True, trim="-")


def _check_number(text: str, *, phase_count: bool = False) -> float:
    """Validate a TDM numeric literal (4.3.2-4.3.5, 4.3.11) and return its value."""
    if phase_count:
        if not _PHASE_RE.match(text):
            raise TDMFormatError(f"phase count {text!r} is not a decimal digit string")
        return float(text)
    if _INTEGER_RE.match(text):
        value = int(text)
        if not -2**31 <= value <= 2**31 - 1:
            raise TDMFormatError(f"integer {text!r} is outside the 32-bit range of 4.3.2")
        return float(value)
    fixed = _FIXED_RE.match(text)
    if fixed:
        if len(fixed[1]) + len(fixed[2]) > 16:
            raise TDMFormatError(f"fixed-point value {text!r} has more than 16 digits (4.3.4)")
        return float(text)
    flt = _FLOAT_RE.match(text)
    if flt:
        if 1 + len(flt[2]) > 16:
            raise TDMFormatError(f"floating-point mantissa of {text!r} has more than 16 digits (4.3.5)")
        value = float(text)
        if not math.isfinite(value):
            raise TDMFormatError(f"value {text!r} is not a finite double")
        return value
    raise TDMFormatError(f"{text!r} is not a TDM integer, fixed-point or floating-point value")


def format_number(value: float) -> str:
    """Shortest text that reads back to ``value``, within the 16-digit limit of 4.3.4/4.3.5.

    A double needing 17 significant digits is written with 16 (relative change < 1e-16).
    """
    value = float(value)
    if not math.isfinite(value):
        raise TDMFormatError(f"{value!r} cannot be written to a TDM (4.3.5 excludes NaN and Inf)")
    if value == 0.0:
        return "0.0"
    text = repr(value)
    if "e" not in text and "E" not in text and "inf" not in text:
        if _FIXED_RE.match(text) and sum(c.isdigit() for c in text) <= 16:
            return text
    for digits in range(2, 17):          # at least one fractional mantissa digit (4.3.5b)
        candidate = f"{value:.{digits - 1}e}"
        if float(candidate) == value:
            return candidate
    return f"{value:.15e}"


# ======================================================================
# message model
# ======================================================================
@dataclass(frozen=True)
class TDMRecord:
    """One tracking data record ``keyword = timetag measurement``; ``measurement`` is the literal text."""

    keyword: str
    epoch: TDMEpoch
    measurement: str

    @property
    def value(self) -> float:
        return float(self.measurement)

    @classmethod
    def from_value(cls, keyword: str, epoch: TDMEpoch, value: float) -> "TDMRecord":
        return cls(keyword, epoch, format_number(value))


@dataclass(frozen=True)
class TDMSegment:
    """One metadata section and its data section.

    ``metadata`` holds ``(keyword, value)`` pairs in table 3-3 order, values as text (enumerations
    upper-case). Comments sit only at the start of each section (4.5.2).
    """

    metadata: tuple[tuple[str, str], ...]
    records: tuple[TDMRecord, ...]
    metadata_comments: tuple[str, ...] = ()
    data_comments: tuple[str, ...] = ()

    def get(self, keyword: str, default: str | None = None) -> str | None:
        for key, value in self.metadata:
            if key == keyword:
                return value
        return default

    @property
    def time_system(self) -> str:
        return self.get("TIME_SYSTEM")

    @property
    def participants(self) -> dict[int, str]:
        return {n: self.get(f"PARTICIPANT_{n}") for n in PARTICIPANT_INDICES
                if self.get(f"PARTICIPANT_{n}") is not None}

    @property
    def path(self) -> tuple[int, ...]:
        text = self.get("PATH")
        return tuple(int(p) for p in text.split(",")) if text else ()

    @property
    def range_units(self) -> str:
        return self.get("RANGE_UNITS", "km")

    def records_of(self, keyword: str) -> tuple[TDMRecord, ...]:
        return tuple(r for r in self.records if r.keyword == keyword)


@dataclass(frozen=True)
class TrackingDataMessage:
    """A KVN TDM: header plus one or more segments."""

    version: str
    creation_date: TDMEpoch
    originator: str
    segments: tuple[TDMSegment, ...]
    message_id: str | None = None
    comments: tuple[str, ...] = field(default=())


# ======================================================================
# reading
# ======================================================================
def read_tdm(path) -> TrackingDataMessage:
    """Read a KVN TDM file (ASCII); see :func:`parse_tdm`."""
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise TDMFormatError(f"{path}: a TDM contains only printable ASCII (4.2.1): {exc}") from None
    return parse_tdm(text)


def parse_tdm(text: str) -> TrackingDataMessage:
    """Parse KVN TDM text; raises :class:`TDMFormatError` on any departure from the standard."""
    lines = _lines(text)
    pos = 0

    def peek():
        return lines[pos] if pos < len(lines) else None

    # ---- header ----
    header: dict[str, str] = {}
    comments: list[str] = []
    order = 0
    while (item := peek()) is not None:
        lineno, key, value = item
        if key == "META_START":
            break
        pos += 1
        if key == "COMMENT":
            if order != 1:
                raise TDMFormatError(f"line {lineno}: header COMMENT only between CCSDS_TDM_VERS and CREATION_DATE (4.5.2a)")
            comments.append(value)
            continue
        if key not in HEADER_KEYWORDS:
            raise TDMFormatError(f"line {lineno}: {key!r} is not a TDM header keyword (table 3-2)")
        rank = HEADER_KEYWORDS.index(key) + 1
        if rank <= order or key in header:
            raise TDMFormatError(f"line {lineno}: header keyword {key} out of the table 3-2 order")
        if value is None or value == "":
            raise TDMFormatError(f"line {lineno}: {key} needs a value (4.3.1)")
        if not header and key != "CCSDS_TDM_VERS":
            raise TDMFormatError(f"line {lineno}: the first header line must be CCSDS_TDM_VERS (3.2.1)")
        header[key] = value
        order = rank
    for key in _HEADER_MANDATORY:
        if key not in header:
            raise TDMFormatError(f"mandatory header keyword {key} missing (table 3-2)")
    if header["CCSDS_TDM_VERS"] not in SUPPORTED_VERSIONS:
        raise TDMFormatError(f"CCSDS_TDM_VERS = {header['CCSDS_TDM_VERS']!r}; supported: {', '.join(SUPPORTED_VERSIONS)}")
    try:
        creation = TDMEpoch.parse(header["CREATION_DATE"])
    except TDMFormatError as exc:
        raise TDMFormatError(f"CREATION_DATE: {exc}") from None

    # ---- segments ----
    segments: list[TDMSegment] = []
    while peek() is not None:
        lineno, key, value = peek()
        if key != "META_START":
            raise TDMFormatError(f"line {lineno}: expected META_START, found {key!r}")
        segment, pos = _parse_segment(lines, pos)
        segments.append(segment)
    if not segments:
        raise TDMFormatError("a TDM has at least one segment (3.3.1.1)")
    return TrackingDataMessage(
        version=header["CCSDS_TDM_VERS"], creation_date=creation, originator=header["ORIGINATOR"],
        segments=tuple(segments), message_id=header.get("MESSAGE_ID"), comments=tuple(comments))


_BARE = ("META_START", "META_STOP", "DATA_START", "DATA_STOP")


def _lines(text: str) -> list[tuple[int, str, str | None]]:
    """(line number, keyword, value) for every non-blank line; value is None for the bare keywords."""
    out = []
    for lineno, line in enumerate(re.split(r"\r\n|\n\r|\r|\n", text), start=1):
        if len(line) > MAX_LINE_LENGTH:
            raise TDMFormatError(f"line {lineno}: longer than {MAX_LINE_LENGTH} characters (4.2.1)")
        if any(not (32 <= ord(c) <= 126) for c in line):
            raise TDMFormatError(f"line {lineno}: only printable ASCII and blanks are allowed (4.2.1)")
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "COMMENT" or stripped.startswith("COMMENT "):
            out.append((lineno, "COMMENT", stripped[8:] if len(stripped) > 7 else ""))
            continue
        if stripped in _BARE:
            out.append((lineno, stripped, None))
            continue
        key, eq, value = stripped.partition("=")
        key = key.strip()
        if not eq:
            raise TDMFormatError(f"line {lineno}: not a 'keyword = value' line")
        if not _KEYWORD_RE.match(key):
            raise TDMFormatError(f"line {lineno}: keyword {key!r} must be uppercase without blanks (4.2.6)")
        value = value.strip()
        if not value:
            raise TDMFormatError(f"line {lineno}: {key} needs a value (4.3.1)")
        out.append((lineno, key, value))
    return out


def _parse_segment(lines, pos):
    lineno0 = lines[pos][0]
    pos += 1
    meta: list[tuple[str, str]] = []
    meta_comments: list[str] = []
    rank = -1
    while True:
        if pos >= len(lines):
            raise TDMFormatError(f"metadata section opened on line {lineno0} has no META_STOP")
        lineno, key, value = lines[pos]
        pos += 1
        if key == "META_STOP":
            break
        if key == "COMMENT":
            if meta:
                raise TDMFormatError(f"line {lineno}: metadata COMMENT only right after META_START (4.5.2b)")
            meta_comments.append(value)
            continue
        if key not in _METADATA_RANK:
            raise TDMFormatError(f"line {lineno}: {key!r} is not a TDM metadata keyword (table 3-3)")
        if value is None:
            raise TDMFormatError(f"line {lineno}: {key} needs a value")
        if _METADATA_RANK[key] <= rank:
            raise TDMFormatError(f"line {lineno}: metadata keyword {key} repeated or out of the table 3-3 order (3.3.1.8)")
        rank = _METADATA_RANK[key]
        try:
            meta.append((key, _check_metadata_value(key, value)))
        except TDMFormatError as exc:
            raise TDMFormatError(f"line {lineno}: {exc}") from None
    _check_metadata(dict(meta), lineno0)
    meta_t = tuple(meta)

    if pos >= len(lines) or lines[pos][1] != "DATA_START":
        raise TDMFormatError(f"metadata section of line {lineno0} is not followed by DATA_START (3.3.1.3)")
    pos += 1
    data_comments: list[str] = []
    records: list[TDMRecord] = []
    probe = TDMSegment(meta_t, ())
    allowed = _allowed_data_keywords(probe)
    last_epoch: dict[str, TDMEpoch] = {}
    while True:
        if pos >= len(lines):
            raise TDMFormatError(f"data section of the segment on line {lineno0} has no DATA_STOP")
        lineno, key, value = lines[pos]
        pos += 1
        if key == "DATA_STOP":
            break
        if key == "COMMENT":
            if records:
                raise TDMFormatError(f"line {lineno}: data COMMENT only before the first record (4.5.2c)")
            data_comments.append(value)
            continue
        if value is None:
            raise TDMFormatError(f"line {lineno}: unexpected {key}")
        try:
            record = _parse_record(key, value, probe, allowed)
        except TDMFormatError as exc:
            raise TDMFormatError(f"line {lineno}: {exc}") from None
        previous = last_epoch.get(key)
        if previous is not None:
            if record.epoch == previous:
                raise TDMFormatError(f"line {lineno}: {key} at {value.split()[0]} repeats a keyword/time tag (3.4.11)")
            if record.epoch < previous:
                raise TDMFormatError(f"line {lineno}: {key} records are not in chronological order (3.4.10)")
        last_epoch[key] = record.epoch
        records.append(record)
    if not records:
        raise TDMFormatError(f"the data section of the segment on line {lineno0} has no tracking data record (3.4.1)")
    return TDMSegment(meta_t, tuple(records), tuple(meta_comments), tuple(data_comments)), pos


def _check_metadata_value(key: str, value: str) -> str:
    if key in METADATA_ENUMS:
        upper = value.upper()
        if upper not in METADATA_ENUMS[key]:
            raise TDMFormatError(f"{key} = {value!r}; allowed: {', '.join(METADATA_ENUMS[key])} (table 3-3)")
        return _RANGE_UNITS_CANONICAL[upper] if key == "RANGE_UNITS" else upper
    if key == "TIME_SYSTEM":
        upper = value.upper()
        if upper not in TIME_SYSTEMS:
            raise TDMFormatError(f"TIME_SYSTEM = {value!r} is not a SANA time system (annex B)")
        return upper
    if key in _METADATA_EPOCHS:
        TDMEpoch.parse(value)
        return value
    if key in ("PATH", "PATH_1", "PATH_2"):
        if not _PATH_RE.match(value):
            raise TDMFormatError(f"{key} = {value!r} is not a comma-separated list of participant indices 1-5")
        return value
    if key in _METADATA_INTEGER:
        if not _INTEGER_RE.match(value):
            raise TDMFormatError(f"{key} = {value!r} must be an integer")
        number = _check_number(value)
    elif key in _METADATA_REAL:
        number = _check_number(value)
    else:
        return value
    if key in _METADATA_POSITIVE and not number > 0.0:
        raise TDMFormatError(f"{key} = {value} must be positive")
    if key in _METADATA_NON_NEGATIVE and number < 0.0:
        raise TDMFormatError(f"{key} = {value} must be non-negative")
    return value


def _check_metadata(meta: dict[str, str], lineno0: int) -> None:
    where = f"metadata section of line {lineno0}"
    if "TIME_SYSTEM" not in meta:
        raise TDMFormatError(f"{where}: TIME_SYSTEM is mandatory (table 3-3)")
    participants = {n for n in PARTICIPANT_INDICES if f"PARTICIPANT_{n}" in meta}
    if not participants:
        raise TDMFormatError(f"{where}: at least one PARTICIPANT_n is mandatory (table 3-3)")
    for stem in ("EPHEMERIS_NAME", "TRANSMIT_DELAY", "RECEIVE_DELAY"):
        for n in PARTICIPANT_INDICES:
            if f"{stem}_{n}" in meta and n not in participants:
                raise TDMFormatError(f"{where}: {stem}_{n} names no PARTICIPANT_{n}")
    mode = meta.get("MODE")
    for key in ("PATH", "PATH_1", "PATH_2"):
        if key in meta:
            indices = [int(p) for p in meta[key].split(",")]
            missing = sorted(set(indices) - participants)
            if missing:
                raise TDMFormatError(f"{where}: {key} uses undefined participant(s) {missing}")
            if any(a == b for a, b in zip(indices, indices[1:])):
                raise TDMFormatError(f"{where}: {key} repeats a participant on consecutive legs")
    if mode == "SEQUENTIAL":
        if "PATH" not in meta or "PATH_1" in meta or "PATH_2" in meta:
            raise TDMFormatError(f"{where}: MODE = SEQUENTIAL takes PATH, not PATH_1/PATH_2 (3.3.2)")
    elif mode == "SINGLE_DIFF":
        if "PATH" in meta or "PATH_1" not in meta or "PATH_2" not in meta:
            raise TDMFormatError(f"{where}: MODE = SINGLE_DIFF takes PATH_1 and PATH_2 (3.3.2.5.3)")
    elif any(k in meta for k in ("PATH", "PATH_1", "PATH_2")):
        raise TDMFormatError(f"{where}: PATH keywords need MODE (3.3.2)")
    if any(k.startswith("CORRECTION_") for k in meta) and "CORRECTIONS_APPLIED" not in meta:
        raise TDMFormatError(f"{where}: CORRECTION_* requires CORRECTIONS_APPLIED (3.4.15.3)")
    if "INTERPOLATION" in meta and "INTERPOLATION_DEGREE" not in meta:
        raise TDMFormatError(f"{where}: INTERPOLATION requires INTERPOLATION_DEGREE (table 3-3)")
    if "REFERENCE_FRAME" in meta and meta.get("ANGLE_TYPE") != "RADEC":
        raise TDMFormatError(f"{where}: REFERENCE_FRAME applies only to ANGLE_TYPE = RADEC (table 3-3)")
    if "START_TIME" in meta and "STOP_TIME" in meta:
        if TDMEpoch.parse(meta["STOP_TIME"]) < TDMEpoch.parse(meta["START_TIME"]):
            raise TDMFormatError(f"{where}: STOP_TIME precedes START_TIME")
    if "DATA_TYPES" in meta:
        for item in meta["DATA_TYPES"].split(","):
            if _data_keyword_class(item.strip(), participants | {"n"}) is None:
                raise TDMFormatError(f"{where}: DATA_TYPES entry {item.strip()!r} is not a data keyword (table 3-5)")


def _data_keyword_class(keyword: str, participants) -> str | None:
    """The table 3-5 entry a data keyword instantiates, or None."""
    if keyword in DATA_KEYWORD_UNITS and not keyword.endswith("_n"):
        return keyword
    stem, _, index = keyword.rpartition("_")
    generic = f"{stem}_n"
    if generic in DATA_KEYWORD_UNITS and (index in {str(p) for p in participants}):
        return generic
    return None


def _allowed_data_keywords(segment: TDMSegment):
    participants = set(segment.participants)
    types = segment.get("DATA_TYPES")
    declared = None
    if types:
        declared = {t.strip() for t in types.split(",")}
    return participants, declared


def _parse_record(key: str, value: str, segment: TDMSegment, allowed) -> TDMRecord:
    participants, declared = allowed
    generic = _data_keyword_class(key, participants)
    if generic is None:
        if key in _METADATA_RANK or key in HEADER_KEYWORDS:
            raise TDMFormatError(f"{key} is a header/metadata keyword inside a data section")
        raise TDMFormatError(f"{key!r} is not a TDM data keyword (table 3-5) or names an undefined participant")
    if declared is not None and key not in declared and generic not in declared:
        raise TDMFormatError(f"{key} is not listed in this segment's DATA_TYPES")
    parts = value.split()
    if len(parts) != 2:
        if len(parts) > 2:
            raise TDMFormatError(f"{key}: extra token(s) {' '.join(parts[2:])!r} after the measurement; "
                                 f"KVN TDM records carry no units (4.4)")
        raise TDMFormatError(f"{key}: a record is 'timetag measurement' (3.4.1)")
    epoch_text, measurement = parts
    try:
        epoch = TDMEpoch.parse(epoch_text)
    except TDMFormatError as exc:
        raise TDMFormatError(f"{key}: {exc}") from None
    if epoch_text.endswith("Z") and segment.time_system != "UTC":
        raise TDMFormatError(f"{key}: the Z suffix marks UTC but TIME_SYSTEM = {segment.time_system} (4.3.9)")
    if epoch.second >= 60.0 and segment.time_system != "UTC":
        raise TDMFormatError(f"{key}: second {epoch.second} only exists in UTC (leap second)")
    stem = key.rsplit("_", 1)[0]
    _check_number(measurement, phase_count=stem in _PHASE_COUNT_STEMS)
    return TDMRecord(key, epoch, measurement)


# ======================================================================
# writing
# ======================================================================
def format_tdm(tdm: TrackingDataMessage) -> str:
    """KVN text of ``tdm``; the text is parsed back before it is returned, so an invalid message fails here."""
    out = [f"CCSDS_TDM_VERS = {tdm.version}"]
    out += [_comment(c) for c in tdm.comments]
    out.append(f"CREATION_DATE = {tdm.creation_date.format()}")
    out.append(f"ORIGINATOR = {tdm.originator}")
    if tdm.message_id is not None:
        out.append(f"MESSAGE_ID = {tdm.message_id}")
    width = max([len(k) for s in tdm.segments for k, _ in s.metadata] + [8])
    for segment in tdm.segments:
        out.append("")
        out.append("META_START")
        out += [_comment(c) for c in segment.metadata_comments]
        for key, value in sorted(segment.metadata, key=lambda kv: _METADATA_RANK.get(kv[0], -1)):
            out.append(f"{key:<{width}} = {value}")
        out.append("META_STOP")
        out.append("")
        out.append("DATA_START")
        out += [_comment(c) for c in segment.data_comments]
        kw_width = max(len(r.keyword) for r in segment.records) if segment.records else 0
        for record in segment.records:
            out.append(f"{record.keyword:<{kw_width}} = {record.epoch.format()} {record.measurement}")
        out.append("DATA_STOP")
    text = "\n".join(out) + "\n"
    parse_tdm(text)
    return text


def _comment(text: str) -> str:
    return f"COMMENT {text}" if text else "COMMENT"


def write_tdm(tdm: TrackingDataMessage, path) -> Path:
    """Write ``tdm`` as a KVN file (ASCII, LF line ends) and return the path."""
    path = Path(path)
    path.write_bytes(format_tdm(tdm).encode("ascii"))
    return path


# ======================================================================
# time systems
# ======================================================================
DAYSEC = 86400.0
TT_MINUS_TAI_S = 32.184
TAI_MINUS_GPS_S = 19.0
_J2000_ORDINAL = date(2000, 1, 1).toordinal()


def _seconds_past_j2000_noon(epoch: TDMEpoch) -> tuple[float, float]:
    """(whole-day part, seconds of day) of the formal calendar seconds past 2000-01-01T12:00."""
    days = date(epoch.year, epoch.month, epoch.day).toordinal() - _J2000_ORDINAL
    return days * DAYSEC - 43200.0, epoch.seconds_of_day


def tdm_epochs_to_tdb_seconds(epochs: Sequence[TDMEpoch], time_system: str, et0_s: float,
                              site=None) -> np.ndarray:
    """TDM time tags in ``time_system`` -> TDB seconds past ``et0`` (``et0`` in TDB seconds past J2000).

    UTC goes through :func:`lunar_od.station_time.utc_to_tdb_seconds` (needs the leapseconds kernel
    loaded); TAI and GPS (= TAI - 19 s) add 32.184 s to reach TT; TT -> TDB adds
    :func:`lunar_od.station_time.tdb_minus_tt` (geocentric, or at ``site``), solved at the TDB epoch
    by two fixed-point passes (the derivative of TDB - TT is < 1e-8, so the second pass is exact to
    < 1e-19 s). TDB tags are used as they are. Other systems raise :class:`TDMFormatError`.
    """
    from . import station_time

    system = time_system.upper()
    if system not in _CONVERTIBLE_TIME_SYSTEMS:
        raise TDMFormatError(f"TIME_SYSTEM = {time_system} has no conversion to TDB here; "
                             f"supported: {', '.join(_CONVERTIBLE_TIME_SYSTEMS)}")
    epochs = list(epochs)
    if not epochs:
        return np.zeros(0)
    et0 = float(et0_s)
    if system == "UTC":
        return np.asarray(station_time.utc_to_tdb_seconds(
            [(e.year, e.month, e.day, e.seconds_of_day) for e in epochs], et0, site), dtype=float)
    offset = {"TAI": TT_MINUS_TAI_S, "GPS": TT_MINUS_TAI_S + TAI_MINUS_GPS_S, "TT": 0.0, "TDB": 0.0}[system]
    parts = [_seconds_past_j2000_noon(e) for e in epochs]
    rel = np.array([(base - et0) + (sod + offset) for base, sod in parts], dtype=float)
    if system == "TDB":
        return rel
    t = rel.copy()
    for _ in range(2):
        t = rel + np.asarray(station_time.tdb_minus_tt(et0, t, site), dtype=float)
    return t


def tdb_seconds_to_tdm_epoch(t_s: float, et0_s: float) -> TDMEpoch:
    """TDB seconds past ``et0`` -> a TDB calendar time tag (inverse of the TDB branch above).

    The day boundary is found in whole seconds of ``et0`` first, so the fraction of a second keeps
    the resolution of ``t_s`` (about 1e-11 s at a week past ``et0``).
    """
    et0 = float(et0_s)
    whole = math.floor(et0)
    total = (et0 - whole) + float(t_s) + 43200.0           # seconds past 2000-01-01T00:00 minus `whole`
    days_whole, rem_whole = divmod(whole, 86400)
    sod = rem_whole + total
    shift = math.floor(sod / DAYSEC)
    sod -= shift * DAYSEC
    whole_sod = math.floor(sod)
    second = sod - whole_sod                                  # exact (Sterbenz)
    day_shift, whole_sod = divmod(int(whole_sod), 86400)
    d = date.fromordinal(_J2000_ORDINAL + int(days_whole) + shift + day_shift)
    hour, rest = divmod(whole_sod, 3600)
    minute, sec = divmod(rest, 60)
    seconds = min(sec + second, math.nextafter(sec + 1.0, 0.0))   # never round up into the next minute
    return TDMEpoch(d.year, d.month, d.day, hour, minute, seconds)


def with_records(segment: TDMSegment, records: Iterable[TDMRecord]) -> TDMSegment:
    """A copy of ``segment`` with ``records`` replaced (convenience for writers)."""
    return replace(segment, records=tuple(records))
