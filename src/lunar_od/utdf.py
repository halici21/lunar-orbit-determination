"""Reader for low-speed Universal Tracking Data Format (UTDF) files (Phase F1).

UTDF is the GSFC ground-network tracking format used for LRO's USN/STDN S-band
tracking (PDS LRO-L-RSS-1-TRACKING-V1.0, ``data/trk``): one 75-byte big-endian
binary record per datum.  The layout follows ``label/lro_trk.fmt`` of that
archive and the GSFC Ground Network Tracking and Acquisition Data Handbook
(453-HDBK-GN).

Only the fields the radiometric chain needs are decoded; everything is kept in
raw units next to the converted value, so a conversion can always be checked.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
    transmit_frequency_10hz: float
    pad_xmit: int
    pad_rcv: int
    mode: int
    flags: int
    router: bytes

    @property
    def utc(self) -> datetime:
        """Receive time tag (UTC): seconds of year counted from Jan 1 00:00."""
        start = datetime(self.year, 1, 1, tzinfo=timezone.utc)
        return start + timedelta(seconds=self.seconds_of_year, microseconds=self.microseconds)

    @property
    def rtlt_s(self) -> float:
        """Round-trip light time [s] as measured (station and transponder delays included)."""
        return self.rtlt_raw * RTLT_UNIT_S

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
    (ftx,) = struct.unpack(">f", raw[40:44])
    (mode,) = struct.unpack(">H", raw[48:50])
    return UTDFRecord(
        year=century + raw[5], seconds_of_year=sec, microseconds=usec, sic=sic, vid=vid,
        rtlt_raw=rt_hi * 65536 + rt_lo, doppler_count_raw=dop_hi * 65536 + dop_lo,
        transmit_frequency_10hz=float(ftx), pad_xmit=raw[45], pad_rcv=raw[47], mode=mode,
        flags=raw[50], router=bytes(raw[3:5]),
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
