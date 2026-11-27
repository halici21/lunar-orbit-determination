"""Reader for low-speed Universal Tracking Data Format (UTDF) files (Phase F1).

UTDF is the GSFC ground-network tracking format used for LRO's USN/STDN S-band
tracking (PDS LRO-L-RSS-1-TRACKING-V1.0, ``data/trk``): one 75-byte big-endian
binary record per datum.  The layout follows ``label/lro_trk.fmt`` of that
archive and the GSFC Ground Network Tracking and Acquisition Data Handbook
(453-HDBK-GN).

Only the fields the radiometric chain needs are decoded; everything is kept in
raw units next to the converted value, so a conversion can always be checked.

F5 WP5.6 (RD-11, studies/phase17/configs/wp5_6_utdf_hardening_predeclaration.json) adds integrity checks
that report what is wrong with a reason code instead of accepting it silently: :func:`record_reason_codes`
(transmit/receive PAD, link type, band, time fields) and :func:`check_doppler_counts` (time order, gaps,
transmit-frequency changes, frozen counter, 48-bit rollover, reversal, implausible rate, count
discontinuity). Elapsed time is a TAI difference, so an interval across a UTC leap second has its SI
length, and the range-rate constants (transmit frequency, K, M) come from the record.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np

RECORD_BYTES = 75
FIXED_BYTES = b"\x0d\x0a\x01"

#: RTLT is carried in units of 1/256 ns.
RTLT_UNIT_S = 1.0 / 256.0e9

#: Validity flags in byte 51 (MSB first): sidelobe, destruct, range refraction
#: corrected, angle refraction corrected, angle data corrected, angle valid,
#: range valid, range-rate valid.
FLAG_SIDELOBE = 0x80
FLAG_DESTRUCT = 0x40
FLAG_RANGE_REFRACTION_CORRECTED = 0x20
FLAG_ANGLE_REFRACTION_CORRECTED = 0x10
FLAG_ANGLE_CORRECTED = 0x08
FLAG_ANGLE_VALID = 0x04
FLAG_RANGE_VALID = 0x02
FLAG_DOPPLER_VALID = 0x01

#: Speed of light [m/s] (the value the LRO studies use).
C_LIGHT_MPS = 299792458.0
#: The Doppler counter counts the Doppler (times M) plus this bias [Hz] (453-HDBK-GN 4.2.2.4(c)).
DOPPLER_BIAS_HZ = 2.4e8
#: Bytes 33-38 hold a 48-bit counter.
COUNTER_MODULUS = 2 ** 48

#: Frequency band (high nibble of byte 52) -> (turnaround ratio K, count multiplier M), 453-HDBK-GN 4.2.2.4(c).
BAND_CONSTANTS = {
    1: (1.0, 1000.0),             # VHF
    3: (240.0 / 221.0, 1000.0),   # S-band
    5: (880.0 / 749.0, 250.0),    # X-band
}

_J2000_NOON = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)


def _cal_s(y, m, d) -> int:
    return int((datetime(y, m, d, tzinfo=timezone.utc) - _J2000_NOON).total_seconds())


#: (UTC calendar seconds past 2000-01-01T12:00 at which the value starts, TAI - UTC [s]): the IERS steps,
#: in the form of the leapseconds kernel's DELTET/DELTA_AT (naif0012).
LEAP_TABLE = tuple((float(_cal_s(*ymd)), float(dat)) for ymd, dat in (
    ((1972, 1, 1), 10), ((1972, 7, 1), 11), ((1973, 1, 1), 12), ((1974, 1, 1), 13), ((1975, 1, 1), 14),
    ((1976, 1, 1), 15), ((1977, 1, 1), 16), ((1978, 1, 1), 17), ((1979, 1, 1), 18), ((1980, 1, 1), 19),
    ((1981, 7, 1), 20), ((1982, 7, 1), 21), ((1983, 7, 1), 22), ((1985, 7, 1), 23), ((1988, 1, 1), 24),
    ((1990, 1, 1), 25), ((1991, 1, 1), 26), ((1992, 7, 1), 27), ((1993, 7, 1), 28), ((1994, 7, 1), 29),
    ((1996, 1, 1), 30), ((1997, 7, 1), 31), ((1999, 1, 1), 32), ((2006, 1, 1), 33), ((2009, 1, 1), 34),
    ((2012, 7, 1), 35), ((2015, 7, 1), 36), ((2017, 1, 1), 37)))
#: The table is known to hold until this UTC instant (IERS Bulletin C: no leap second at the end of
#: December 2025 or June 2026). Later epochs need a caller-supplied ``tai_minus_utc``.
LEAP_TABLE_VALID_UNTIL_CAL_S = float(_cal_s(2026, 7, 1))

#: Reason codes (record level).
PAD_XMIT_RCV_MISMATCH = "PAD_XMIT_RCV_MISMATCH"
PAD_FILE_MISMATCH = "PAD_FILE_MISMATCH"
NOT_TWO_WAY = "NOT_TWO_WAY"
BAND_UNSUPPORTED = "BAND_UNSUPPORTED"
MICROSECONDS_OUT_OF_RANGE = "MICROSECONDS_OUT_OF_RANGE"
SECONDS_OUT_OF_YEAR = "SECONDS_OUT_OF_YEAR"
#: Reason codes (Doppler interval level).
TIME_NOT_INCREASING = "TIME_NOT_INCREASING"
GAP = "GAP"
FREQUENCY_CHANGED = "FREQUENCY_CHANGED"
COUNT_FROZEN = "COUNT_FROZEN"
COUNT_ROLLOVER = "COUNT_ROLLOVER"
COUNT_REVERSAL = "COUNT_REVERSAL"
RATE_OUT_OF_RANGE = "RATE_OUT_OF_RANGE"
RATE_DISCONTINUITY = "RATE_DISCONTINUITY"


class UTDFFormatError(ValueError):
    """A file or record that does not follow the UTDF layout."""


@dataclass(frozen=True)
class UTDFRecord:
    """One decoded UTDF datum (receive-time tagged, UTC)."""

    year: int
    seconds_of_year: int
    microseconds: int
    sic: int
    vid: int
    rtlt_raw: int
    doppler_count_raw: int
    transmit_frequency_10hz: int
    pad_xmit: int
    pad_rcv: int
    mode: int
    flags: int
    router: bytes
    #: raw byte 52: frequency band (high nibble) and data transmission type (low nibble)
    band_byte: int = field(default=0)
    #: raw bytes 53-54: tracker type, last-frame flag and sample rate
    tracking_mode: int = field(default=0)

    @property
    def is_leap_second(self) -> bool:
        """True for 23:59:60 Dec 31 of a year ending in a leap second (seconds of year = year length)."""
        return (self.seconds_of_year == _days_in_year(self.year) * 86400
                and _leap_step_at(_cal_s(self.year + 1, 1, 1)))

    @property
    def utc(self) -> datetime:
        """Receive time tag (UTC): seconds of year counted from Jan 1 00:00 (UTC calendar seconds)."""
        if self.is_leap_second:
            raise UTDFFormatError(f"{self.utc_iso} is a leap second, which datetime cannot represent; "
                                  "use utc_iso or tai_seconds_past_j2000.")
        start = datetime(self.year, 1, 1, tzinfo=timezone.utc)
        return start + timedelta(seconds=self.seconds_of_year, microseconds=self.microseconds)

    @property
    def utc_iso(self) -> str:
        """UTC tag as ``YYYY-MM-DDTHH:MM:SS.ffffff`` (second 60 at a year-end leap second)."""
        if self.is_leap_second:
            return f"{self.year:04d}-12-31T23:59:60.{self.microseconds:06d}"
        return self.utc.strftime("%Y-%m-%dT%H:%M:%S.%f")

    def tai_microseconds_past_j2000(self, tai_minus_utc: Callable[[float], float] | None = None) -> int:
        """TAI microseconds past 2000-01-01T12:00:00 TAI (integer, so differences are exact)."""
        cal = _cal_s(self.year, 1, 1) + self.seconds_of_year
        lookup = builtin_tai_minus_utc if tai_minus_utc is None else tai_minus_utc
        # 23:59:60 is one SI second after 23:59:59, still under the old TAI - UTC
        dat = lookup(float(cal - 1)) if self.is_leap_second else lookup(float(cal))
        if dat != round(dat):
            raise UTDFFormatError(f"TAI - UTC must be a whole number of seconds after 1972; got {dat}.")
        return (cal + int(round(dat))) * 1_000_000 + self.microseconds

    @property
    def tai_seconds_past_j2000(self) -> float:
        """TAI seconds past 2000-01-01T12:00:00 TAI, from the built-in leap-second table."""
        return self.tai_microseconds_past_j2000() / 1e6

    @property
    def frequency_band(self) -> int:
        """Byte 52 high nibble: 1 VHF, 2 UHF, 3 S, 4 C, 5 X, 6 Ku, 7 visible, 8 S up/Ku down."""
        return self.band_byte >> 4

    @property
    def data_type(self) -> int:
        """Byte 52 low nibble: 0 test, 2 simulated, 3 resubmit, 4 real time, 5 playback."""
        return self.band_byte & 0x0F

    @property
    def link_type(self) -> int:
        """SRE mode bits 6-5 (Table 4-2): 1 one-way, 2 two-way, 3 three-way, 0 not used."""
        return (self.mode >> 4) & 0x3

    @property
    def turnaround_ratio(self) -> float | None:
        """K of the record's band (240/221 S, 1 VHF, 880/749 X); None for a band without handbook constants."""
        c = BAND_CONSTANTS.get(self.frequency_band)
        return None if c is None else c[0]

    @property
    def doppler_multiplier(self) -> float | None:
        """M of the record's band (1000 S and VHF, 250 X); None for a band without handbook constants."""
        c = BAND_CONSTANTS.get(self.frequency_band)
        return None if c is None else c[1]

    @property
    def rtlt_s(self) -> float:
        """Round-trip light time [s] as measured (station and transponder delays included)."""
        return self.rtlt_raw * RTLT_UNIT_S

    @property
    def transmit_frequency_hz(self) -> float:
        """Station transmit (uplink) frequency [Hz]."""
        return 10.0 * self.transmit_frequency_10hz

    @property
    def range_valid(self) -> bool:
        return bool(self.flags & FLAG_RANGE_VALID) and self.rtlt_raw > 0

    @property
    def doppler_valid(self) -> bool:
        return bool(self.flags & FLAG_DOPPLER_VALID)


def decode_record(raw: bytes, *, century: int = 2000) -> UTDFRecord:
    """Decode one 75-byte record; raises :class:`UTDFFormatError` on a bad layout."""
    if len(raw) != RECORD_BYTES:
        raise UTDFFormatError(f"a UTDF record has {RECORD_BYTES} bytes; got {len(raw)}.")
    if raw[:3] != FIXED_BYTES:
        raise UTDFFormatError(f"fixed bytes 0D 0A 01 expected; got {raw[:3].hex(' ')}.")
    sic, vid = struct.unpack(">HH", raw[6:10])
    sec, usec = struct.unpack(">II", raw[10:18])
    rt_hi, rt_lo = struct.unpack(">IH", raw[26:32])
    dop_hi, dop_lo = struct.unpack(">IH", raw[32:38])
    # Binary count of 10 Hz (453-HDBK-GN 4.2); the PDS lro_trk.fmt labels it IEEE_REAL, which decodes the
    # LRO records to ~1.9e-31 instead of 2091.414 MHz.
    (ftx,) = struct.unpack(">I", raw[40:44])
    (mode,) = struct.unpack(">H", raw[48:50])
    return UTDFRecord(
        year=century + raw[5], seconds_of_year=sec, microseconds=usec, sic=sic, vid=vid,
        rtlt_raw=rt_hi * 65536 + rt_lo, doppler_count_raw=dop_hi * 65536 + dop_lo,
        transmit_frequency_10hz=int(ftx), pad_xmit=raw[45], pad_rcv=raw[47], mode=mode,
        flags=raw[50], router=bytes(raw[3:5]), band_byte=raw[51],
        tracking_mode=struct.unpack(">H", raw[52:54])[0],
    )


def read_utdf(path, *, century: int = 2000) -> list[UTDFRecord]:
    """All records of one UTDF file, in file order."""
    data = Path(path).read_bytes()
    if len(data) % RECORD_BYTES:
        raise UTDFFormatError(f"{path}: {len(data)} bytes is not a whole number of {RECORD_BYTES}-byte records.")
    return [decode_record(data[i:i + RECORD_BYTES], century=century)
            for i in range(0, len(data), RECORD_BYTES)]


def file_pad_id(path) -> int:
    """Receiver PAD ID from an archive file name ``LSUTDF_SSSSVV_PAD_YYYY_DDD_HHMM.TRK``."""
    parts = Path(path).stem.split("_")
    if len(parts) != 6 or parts[0].upper() != "LSUTDF":
        raise UTDFFormatError(f"not an LRO low-speed UTDF file name: {Path(path).name}")
    return int(parts[2])


# ======================================================================
# WP5.6: time, record and Doppler-count checks (RD-11)
# ======================================================================
def _days_in_year(year: int) -> int:
    return 366 if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0 else 365


def builtin_tai_minus_utc(utc_cal_s_past_j2000: float) -> float:
    """TAI - UTC [s] at a UTC calendar instant (seconds past 2000-01-01T12:00, no leap seconds counted)."""
    t = float(utc_cal_s_past_j2000)
    if t < LEAP_TABLE[0][0]:
        raise UTDFFormatError("TAI - UTC is integral only from 1972-01-01; earlier UTC is not supported.")
    if t >= LEAP_TABLE_VALID_UNTIL_CAL_S:
        raise UTDFFormatError("the built-in leap-second table is known to hold only until 2026-07-01; pass "
                              "tai_minus_utc (e.g. from the leapseconds kernel) for later epochs.")
    dat = LEAP_TABLE[0][1]
    for start, value in LEAP_TABLE:
        if t >= start:
            dat = value
    return dat


def _leap_step_at(cal_s: int) -> bool:
    """True if a positive leap second ends just before calendar instant ``cal_s`` (a TAI - UTC step)."""
    return any(start == float(cal_s) for start, _ in LEAP_TABLE[1:])


def record_reason_codes(record: UTDFRecord, *, file_pad: int | None = None) -> tuple[str, ...]:
    """Record-level integrity codes; an empty tuple means the record is clean."""
    codes = []
    if record.pad_xmit != record.pad_rcv:
        codes.append(PAD_XMIT_RCV_MISMATCH)
    if file_pad is not None and record.pad_rcv != file_pad:
        codes.append(PAD_FILE_MISMATCH)
    if record.link_type != 2:
        codes.append(NOT_TWO_WAY)
    if record.frequency_band not in BAND_CONSTANTS:
        codes.append(BAND_UNSUPPORTED)
    if record.microseconds >= 1_000_000:
        codes.append(MICROSECONDS_OUT_OF_RANGE)
    if record.seconds_of_year >= _days_in_year(record.year) * 86400 and not record.is_leap_second:
        codes.append(SECONDS_OUT_OF_YEAR)
    return tuple(codes)


@dataclass(frozen=True)
class DopplerInterval:
    """One counted-Doppler interval between two Doppler-valid records ``index_a`` -> ``index_b``."""

    index_a: int
    index_b: int
    #: elapsed SI time (TAI difference) [s]
    dt_s: float
    #: count difference, unwrapped across a 48-bit rollover
    dn: int
    #: dn / dt [Hz]
    rate_hz: float
    #: handbook average range rate [m/s] with the record's f_T, K and M (nan when not computable)
    range_rate_mps: float
    codes: tuple[str, ...]
    #: deviation of range_rate_mps from the neighbours' cubic [m/s] (nan when not tested)
    deviation_mps: float = math.nan

    @property
    def usable(self) -> bool:
        return not self.codes


def _range_rate(rate_hz: float, record: UTDFRecord) -> float:
    k, m = record.turnaround_ratio, record.doppler_multiplier
    if k is None or not math.isfinite(rate_hz):
        return math.nan
    return -C_LIGHT_MPS * (rate_hz - DOPPLER_BIAS_HZ) / (2.0 * record.transmit_frequency_hz * k * m)


def check_doppler_counts(records, *, max_gap_s: float = 60.0, max_range_rate_mps: float = 3000.0,
                         discontinuity_mps: float = 0.020, window: int = 4, min_neighbours: int = 6,
                         file_pad: int | None = None,
                         tai_minus_utc: Callable[[float], float] | None = None) -> list[DopplerInterval]:
    """Counted-Doppler intervals of one station's time-ordered stream, each with its reason codes.

    One interval per consecutive pair of the Doppler-valid records, in the given order. Flagged intervals are
    returned, not dropped; ``interval.usable`` is True only when no code applies. RATE_DISCONTINUITY: within a
    run of otherwise clean intervals, an interval whose range rate deviates by more than
    ``discontinuity_mps`` from a least-squares cubic through up to ``window`` clean neighbours on each side
    (at least ``min_neighbours``), flagged largest first and removed before the deviations are recomputed.
    """
    idx = [i for i, r in enumerate(records) if r.doppler_valid]
    tai = {i: records[i].tai_microseconds_past_j2000(tai_minus_utc) for i in idx}
    rcodes = {i: record_reason_codes(records[i], file_pad=file_pad) for i in idx}
    out: list[DopplerInterval] = []
    for a, b in zip(idx, idx[1:]):
        ra, rb = records[a], records[b]
        codes = list(dict.fromkeys(rcodes[a] + rcodes[b]))
        dt = (tai[b] - tai[a]) / 1e6
        if dt <= 0.0:
            codes.append(TIME_NOT_INCREASING)
        elif dt > max_gap_s:
            codes.append(GAP)
        if ra.transmit_frequency_10hz != rb.transmit_frequency_10hz:
            codes.append(FREQUENCY_CHANGED)
        dn = rb.doppler_count_raw - ra.doppler_count_raw
        count_code = None
        if dn == 0:
            count_code = COUNT_FROZEN
        elif dn < 0:
            wrapped = dn + COUNTER_MODULUS
            rr_w = _range_rate(wrapped / dt, ra) if dt > 0.0 else math.nan
            if math.isfinite(rr_w) and abs(rr_w) <= max_range_rate_mps:
                dn, count_code = wrapped, COUNT_ROLLOVER
            else:
                count_code = COUNT_REVERSAL
        rate = dn / dt if dt > 0.0 else math.nan
        rr = _range_rate(rate, ra)
        if count_code is not None:
            codes.append(count_code)
        elif math.isfinite(rr) and abs(rr) > max_range_rate_mps:
            codes.append(RATE_OUT_OF_RANGE)
        out.append(DopplerInterval(a, b, dt, dn, rate, rr, tuple(codes)))
    return _flag_discontinuities(out, tai, discontinuity_mps, window, min_neighbours)


def _flag_discontinuities(out, tai, threshold, window, min_neighbours):
    runs, cur = [], []
    for j, iv in enumerate(out):
        if iv.codes or not math.isfinite(iv.range_rate_mps):
            if cur:
                runs.append(cur)
            cur = []
        else:
            cur.append(j)
    if cur:
        runs.append(cur)
    t0 = next(iter(tai.values()), 0)
    dev = {}
    flagged = {}
    for run in runs:
        t = np.array([0.5 * ((tai[out[j].index_a] - t0) + (tai[out[j].index_b] - t0)) / 1e6 for j in run])
        y = np.array([out[j].range_rate_mps for j in run])
        active = list(range(len(run)))
        while True:
            best, best_d = None, 0.0
            for p, q in enumerate(active):
                nb = active[max(0, p - window):p] + active[p + 1:p + 1 + window]
                if len(nb) < min_neighbours:
                    dev.pop(run[q], None)
                    continue
                tt = (t[nb] - t[q]) / 60.0
                coef = np.polyfit(tt, y[nb], 3)
                d = float(y[q] - np.polyval(coef, 0.0))
                dev[run[q]] = d
                if abs(d) > threshold and abs(d) > abs(best_d):
                    best, best_d = q, d
            if best is None:
                break
            flagged[run[best]] = best_d
            dev.pop(run[best], None)
            active.remove(best)
    result = []
    for j, iv in enumerate(out):
        if j in flagged:
            result.append(DopplerInterval(iv.index_a, iv.index_b, iv.dt_s, iv.dn, iv.rate_hz, iv.range_rate_mps,
                                          iv.codes + (RATE_DISCONTINUITY,), flagged[j]))
        elif j in dev:
            result.append(DopplerInterval(iv.index_a, iv.index_b, iv.dt_s, iv.dn, iv.rate_hz, iv.range_rate_mps,
                                          iv.codes, dev[j]))
        else:
            result.append(iv)
    return result
