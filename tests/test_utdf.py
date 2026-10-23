"""Phase F1: UTDF record decoding on synthetic records (no archive data in the repository)."""
from __future__ import annotations

import struct
from datetime import datetime, timezone

import pytest

from lunar_od.utdf import (
    FLAG_DOPPLER_VALID,
    FLAG_RANGE_VALID,
    RECORD_BYTES,
    UTDFFormatError,
    decode_record,
    file_pad_id,
    read_utdf,
)


def _record(*, year=9, sic=59, vid=1, sec=22377425, usec=250000, rtlt_raw=615107924736, dop=136859486617,
            ftx=0.0, pad=103, mode=0x2662, flags=0x0F) -> bytes:
    r = bytearray(RECORD_BYTES)
    r[0:3] = b"\x0d\x0a\x01"
    r[3:5] = b"DD"
    r[5] = year
    r[6:10] = struct.pack(">HH", sic, vid)
    r[10:18] = struct.pack(">II", sec, usec)
    r[26:32] = struct.pack(">IH", rtlt_raw >> 16, rtlt_raw & 0xFFFF)
    r[32:38] = struct.pack(">IH", dop >> 16, dop & 0xFFFF)
    r[40:44] = struct.pack(">f", ftx)
    r[45] = pad
    r[47] = pad
    r[48:50] = struct.pack(">H", mode)
    r[50] = flags
    return bytes(r)


def test_decode_fields_and_units():
    rec = decode_record(_record())
    assert (rec.year, rec.sic, rec.vid, rec.pad_rcv, rec.pad_xmit) == (2009, 59, 1, 103, 103)
    assert rec.rtlt_raw == 615107924736
    assert rec.rtlt_s == pytest.approx(615107924736 / 256e9, rel=0, abs=1e-15)
    assert rec.doppler_count_raw == 136859486617
    # seconds of year from Jan 1 00:00 UTC: 22377425 s = 258 d + 86225 s -> 2009-09-16 23:57:05.25,
    # four minutes before the 00:01 close time in the archive file name lsutdf_005901_103_2009_260_0001
    assert rec.utc == datetime(2009, 9, 16, 23, 57, 5, 250000, tzinfo=timezone.utc)
    assert rec.range_valid and rec.doppler_valid


def test_validity_flags():
    assert not decode_record(_record(flags=0x0F & ~FLAG_RANGE_VALID)).range_valid
    assert not decode_record(_record(rtlt_raw=0)).range_valid
    assert not decode_record(_record(flags=0x0F & ~FLAG_DOPPLER_VALID)).doppler_valid


def test_bad_layouts_raise(tmp_path):
    with pytest.raises(UTDFFormatError):
        decode_record(b"\x00" * RECORD_BYTES)
    with pytest.raises(UTDFFormatError):
        decode_record(_record()[:-1])
    p = tmp_path / "LSUTDF_005901_103_2009_260_0001.TRK"
    p.write_bytes(_record() + _record(sec=22377426) + b"\x0d")
    with pytest.raises(UTDFFormatError):
        read_utdf(p)


def test_read_file_and_name(tmp_path):
    p = tmp_path / "lsutdf_005901_105_2009_260_0146.trk"
    p.write_bytes(_record(pad=105) + _record(pad=105, sec=22377431))
    recs = read_utdf(p)
    assert [r.seconds_of_year for r in recs] == [22377425, 22377431]
    assert file_pad_id(p) == 105
    with pytest.raises(UTDFFormatError):
        file_pad_id(tmp_path / "other.trk")
