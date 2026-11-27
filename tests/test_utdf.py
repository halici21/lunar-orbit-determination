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
            ftx=209141411, pad=103, pad_xmit=None, mode=0x2662, flags=0x0F, band=0x34) -> bytes:
    r = bytearray(RECORD_BYTES)
    r[0:3] = b"\x0d\x0a\x01"
    r[3:5] = b"DD"
    r[5] = year
    r[6:10] = struct.pack(">HH", sic, vid)
    r[10:18] = struct.pack(">II", sec, usec)
    r[26:32] = struct.pack(">IH", rtlt_raw >> 16, rtlt_raw & 0xFFFF)
    r[32:38] = struct.pack(">IH", dop >> 16, dop & 0xFFFF)
    r[40:44] = struct.pack(">I", ftx)
    r[45] = pad if pad_xmit is None else pad_xmit
    r[47] = pad
    r[48:50] = struct.pack(">H", mode)
    r[50] = flags
    r[51] = band
    r[52:54] = struct.pack(">H", 0x1001)
    r[72:75] = b"\x04\x04\x04"
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
    assert rec.transmit_frequency_hz == pytest.approx(2091.41411e6, rel=0, abs=1e-3)


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


# ======================================================================
# F5 WP5.6 (RD-11, REQ-DATA-005): studies/phase17/configs/wp5_6_utdf_hardening_predeclaration.json
# ======================================================================
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from datetime import timedelta  # noqa: E402
from pathlib import Path  # noqa: E402

from lunar_od.utdf import (  # noqa: E402
    COUNTER_MODULUS,
    check_doppler_counts,
    record_reason_codes,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "utdf"
SLIP_FILE = "lsutdf_005901_103_2009_263_1105.trk"
CONTROL_FILE = "lsutdf_005901_103_2009_263_0722.trk"
C_LIGHT = 299792458.0
BIAS_HZ = 2.4e8
UTC = timezone.utc


def _fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _split(data: bytes) -> list[bytes]:
    return [data[i:i + RECORD_BYTES] for i in range(0, len(data), RECORD_BYTES)]


def _with_count(raw: bytes, count: int) -> bytes:
    r = bytearray(raw)
    r[32:38] = struct.pack(">IH", count >> 16, count & 0xFFFF)
    return bytes(r)


def _handbook_rdot(dn, dt, ft_hz, k, m):
    """453-HDBK-GN 4.2.2.4(c), written out independently of the module."""
    return -C_LIGHT * (dn / dt - BIAS_HZ) / (2.0 * ft_hz * k * m)


def _usable(ivs):
    return [iv for iv in ivs if not iv.codes]


# ---------------------------------------------------------------- U_G1
def test_utdf_decode_real_archived_record():
    prov = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf-8"))
    for name in (SLIP_FILE, CONTROL_FILE):
        data = _fixture_bytes(name)
        assert hashlib.sha256(data).hexdigest() == prov["files"][name]["sha256"]
        assert len(data) == prov["files"][name]["bytes"] and len(data) % RECORD_BYTES == 0
        for raw in _split(data):
            assert raw[:3] == b"\x0d\x0a\x01" and raw[72:75] == b"\x04\x04\x04"
        recs = read_utdf(FIXTURES / name)
        pad = file_pad_id(FIXTURES / name)
        assert pad == 103
        # file name: ..._2009_263_HHMM, day 263 of 2009 = 2009-09-20. HH:MM is the last record's minute,
        # truncated (amendment log, WP5.6 U_G1: the predeclared 'close time' upper bound failed on 0722,
        # whose last record is 07:22:20); each file spans 4 minutes
        hhmm = Path(name).stem.split("_")[-1]
        minute = datetime(2009, 9, 20, int(hhmm[:2]), int(hhmm[2:]), tzinfo=UTC)
        tags = [r.utc for r in recs]
        assert all(b > a for a, b in zip(tags, tags[1:]))
        assert minute - timedelta(minutes=5) <= tags[0] and tags[-1] < minute + timedelta(minutes=1)
        for r in recs:
            assert (r.year, r.sic, r.vid) == (2009, 59, 1)
            assert r.pad_xmit == r.pad_rcv == pad
            assert r.frequency_band == 3 and r.link_type == 2
            assert record_reason_codes(r, file_pad=pad) == ()
            assert r.transmit_frequency_hz == 2091414110.0
            if r.range_valid:
                assert 3.5e8 <= r.rtlt_s * C_LIGHT / 2.0 <= 4.1e8
        assert any(r.range_valid for r in recs) and any(r.doppler_valid for r in recs)


# ---------------------------------------------------------------- U_G2 / U_G3
def test_real_count_discontinuity_is_flagged():
    recs = read_utdf(FIXTURES / SLIP_FILE)
    ivs = check_doppler_counts(recs, file_pad=103)
    slip_start = datetime(2009, 9, 20, 11, 3, 15, tzinfo=UTC)
    flagged = [iv for iv in ivs if "RATE_DISCONTINUITY" in iv.codes]
    assert [recs[iv.index_a].utc for iv in flagged] == [slip_start]
    (slip,) = flagged
    assert recs[slip.index_b].utc - recs[slip.index_a].utc == timedelta(seconds=5)
    assert abs(abs(slip.deviation_mps) - 1.024) <= 0.05 * 1.024
    for iv in ivs:
        if iv is not slip:
            assert set(iv.codes) <= {"GAP"}, (recs[iv.index_a].utc, iv.codes)


def test_real_control_pass_has_no_codes():
    recs = read_utdf(FIXTURES / CONTROL_FILE)
    ivs = check_doppler_counts(recs, file_pad=103)
    assert len(ivs) >= 10
    assert [iv.codes for iv in ivs if iv.codes] == []


# ---------------------------------------------------------------- U_G4
def _control_stream():
    raws = [r for r in _split(_fixture_bytes(CONTROL_FILE)) if decode_record(r).doppler_valid]
    return raws, [decode_record(r).doppler_count_raw for r in raws]


def test_counter_rollover_is_flagged_and_unwrapped():
    raws, counts = _control_stream()
    base = check_doppler_counts([decode_record(r) for r in raws])
    k = len(raws) // 2
    off = COUNTER_MODULUS - (counts[k] + counts[k + 1]) // 2
    wrapped = [_with_count(r, (c + off) % COUNTER_MODULUS) for r, c in zip(raws, counts)]
    recs = [decode_record(r) for r in wrapped]
    assert recs[k + 1].doppler_count_raw < recs[k].doppler_count_raw  # the counter did wrap
    ivs = check_doppler_counts(recs)
    roll = [i for i, iv in enumerate(ivs) if "COUNT_ROLLOVER" in iv.codes]
    assert roll == [k] and ivs[k].codes == ("COUNT_ROLLOVER",)
    assert ivs[k].dn == counts[k + 1] - counts[k]
    assert ivs[k].range_rate_mps == base[k].range_rate_mps
    assert all(not iv.codes for i, iv in enumerate(ivs) if i != k)


def test_frozen_counter_with_valid_flag_is_flagged():
    raws, counts = _control_stream()
    k = len(raws) // 2
    raws = list(raws)
    raws[k + 1] = _with_count(raws[k + 1], counts[k])
    recs = [decode_record(r) for r in raws]
    assert recs[k + 1].doppler_valid
    ivs = check_doppler_counts(recs)
    assert "COUNT_FROZEN" in ivs[k].codes
    assert ivs[k].codes and ivs[k + 1].codes


def test_count_reversal_and_rate_out_of_range_are_flagged():
    raws, counts = _control_stream()
    k = len(raws) // 2
    rev = list(raws)
    rev[k + 1] = _with_count(rev[k + 1], counts[k] - 1000)
    ivs = check_doppler_counts([decode_record(r) for r in rev])
    assert "COUNT_REVERSAL" in ivs[k].codes and "COUNT_ROLLOVER" not in ivs[k].codes
    jump = [_with_count(r, c + (10**9 if i > k else 0)) for i, (r, c) in enumerate(zip(raws, counts))]
    ivs = check_doppler_counts([decode_record(r) for r in jump])
    assert "RATE_OUT_OF_RANGE" in ivs[k].codes
    assert [i for i, iv in enumerate(ivs) if iv.codes] == [k]


# ---------------------------------------------------------------- U_G5
def test_pad_xmit_rcv_mismatch_and_link_checks():
    assert record_reason_codes(decode_record(_record())) == ()
    bad = decode_record(_record(pad=103, pad_xmit=105))
    assert "PAD_XMIT_RCV_MISMATCH" in record_reason_codes(bad)
    three_way_zero = decode_record(_record(pad=103, pad_xmit=0))
    assert "PAD_XMIT_RCV_MISMATCH" in record_reason_codes(three_way_zero)
    assert "PAD_FILE_MISMATCH" in record_reason_codes(decode_record(_record(pad=105)), file_pad=103)
    assert record_reason_codes(decode_record(_record(pad=105)), file_pad=105) == ()
    three_way = decode_record(_record(mode=(0x2662 & ~0x30) | 0x30))
    assert three_way.link_type == 3 and "NOT_TWO_WAY" in record_reason_codes(three_way)
    assert decode_record(_record(mode=0x2662)).link_type == 2
    # a mismatched record taints both intervals that touch it
    stream = [decode_record(_record(sec=22377425 + 5 * i, usec=0, dop=136859486617 + 1291000000 * i,
                                    pad_xmit=(105 if i == 2 else None))) for i in range(5)]
    ivs = check_doppler_counts(stream)
    assert [("PAD_XMIT_RCV_MISMATCH" in iv.codes) for iv in ivs] == [False, True, True, False]


def test_record_time_field_checks():
    assert "MICROSECONDS_OUT_OF_RANGE" in record_reason_codes(decode_record(_record(usec=1_000_000)))
    assert "SECONDS_OUT_OF_YEAR" in record_reason_codes(decode_record(_record(year=9, sec=365 * 86400 + 5)))


# ---------------------------------------------------------------- U_G6
def _leap_stream(year2, tags_si, rate_hz=258_000_000, n0=10**11):
    """Records at (year two digits, seconds of year) with counts advancing at rate_hz per SI second."""
    return [decode_record(_record(year=y, sec=s, usec=0, dop=n0 + rate_hz * t, flags=0x0F))
            for (y, s), t in zip(year2, tags_si)]


def _utc_cal_past_j2000(dt: datetime) -> float:
    return (dt - datetime(2000, 1, 1, 12, tzinfo=UTC)).total_seconds()


def test_utdf_leap_second_crossing():
    # (a) year end 2016 (366 days): 23:59:50, 23:59:55, 23:59:60, 00:00:00, 00:00:05, 00:00:10
    d366 = 365 * 86400
    tags = [(16, d366 + 86390), (16, d366 + 86395), (16, 366 * 86400), (17, 0), (17, 5), (17, 10)]
    si = [0, 5, 10, 11, 16, 21]
    recs = _leap_stream(tags, si)
    assert [r.utc_iso for r in recs] == [
        "2016-12-31T23:59:50.000000", "2016-12-31T23:59:55.000000", "2016-12-31T23:59:60.000000",
        "2017-01-01T00:00:00.000000", "2017-01-01T00:00:05.000000", "2017-01-01T00:00:10.000000"]
    tai = [r.tai_seconds_past_j2000 for r in recs]
    assert [b - a for a, b in zip(tai, tai[1:])] == [5.0, 5.0, 1.0, 5.0, 5.0]
    expect0 = _utc_cal_past_j2000(datetime(2016, 12, 31, 23, 59, 50, tzinfo=UTC)) + 36.0
    assert tai[0] == expect0
    assert tai[3] == _utc_cal_past_j2000(datetime(2017, 1, 1, tzinfo=UTC)) + 37.0
    with pytest.raises(UTDFFormatError):
        recs[2].utc
    assert record_reason_codes(recs[2]) == ()
    ivs = check_doppler_counts(recs)
    assert [iv.dt_s for iv in ivs] == [5.0, 5.0, 1.0, 5.0, 5.0]
    assert all(iv.rate_hz == 258_000_000 for iv in ivs)
    # without the leap record, 23:59:55 -> 00:00:00 is 6 SI seconds (the datetime difference says 5)
    no_leap = [recs[0], recs[1], recs[3], recs[4]]
    ivs = check_doppler_counts(no_leap)
    assert ivs[1].dt_s == 6.0 and (no_leap[2].utc - no_leap[1].utc).total_seconds() == 5.0
    assert ivs[1].rate_hz == 258_000_000
    assert ivs[1].range_rate_mps == _handbook_rdot(258_000_000 * 6, 6.0, 2091414110.0, 240 / 221, 1000)

    # (b) mid-year 2015-06-30 (181 days before July 1): 23:59:55 -> 00:00:00 is 6 s; a repeated tag is flagged
    jul1 = 181 * 86400
    recs = _leap_stream([(15, jul1 - 5), (15, jul1), (15, jul1 + 5)], [0, 6, 11])
    ivs = check_doppler_counts(recs)
    assert [iv.dt_s for iv in ivs] == [6.0, 5.0]
    assert recs[1].utc_iso == "2015-07-01T00:00:00.000000"
    dup = _leap_stream([(15, jul1 - 5), (15, jul1), (15, jul1)], [0, 5, 6])
    assert "TIME_NOT_INCREASING" in check_doppler_counts(dup)[1].codes

    # (c) no leap second: dt is the datetime difference exactly
    recs = read_utdf(FIXTURES / CONTROL_FILE)
    valid = [r for r in recs if r.doppler_valid]
    for iv in check_doppler_counts(recs):
        assert iv.dt_s == (recs[iv.index_b].utc - recs[iv.index_a].utc).total_seconds()
    assert len(valid) >= 2

    # (e) beyond the table's horizon only with a caller-supplied TAI - UTC
    late = decode_record(_record(year=26, sec=200 * 86400, usec=0))
    with pytest.raises(UTDFFormatError):
        late.tai_seconds_past_j2000
    two = [late, decode_record(_record(year=26, sec=200 * 86400 + 5, usec=0, dop=136859486617 + 1291000000))]
    with pytest.raises(UTDFFormatError):
        check_doppler_counts(two)
    ivs = check_doppler_counts(two, tai_minus_utc=lambda utc_cal_s: 37.0)
    assert ivs[0].dt_s == 5.0


def test_builtin_leap_table_matches_the_leapseconds_kernel():
    from tests.test_estimator_lm_consistency import _SKIP_REASON, _SPICE_OK
    if not _SPICE_OK:
        pytest.skip(_SKIP_REASON)
    import spiceypy as spice

    from lunar_od import station_time as ST
    from lunar_od import utdf as U
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels()
    epochs, dat = ST._leap_table()
    assert [(float(e), float(d)) for e, d in zip(epochs, dat)] == [(float(e), float(d)) for e, d in U.LEAP_TABLE]
    # the year-end leap record against SPICE's own UTC parser
    rec = decode_record(_record(year=16, sec=366 * 86400, usec=500000))
    et = spice.str2et(rec.utc_iso)
    assert abs(spice.unitim(et, "TDB", "TAI") - rec.tai_seconds_past_j2000) <= 1e-6


# ---------------------------------------------------------------- U_G7
@pytest.mark.parametrize("band,k,m", [(0x34, 240 / 221, 1000), (0x14, 1.0, 1000), (0x54, 880 / 749, 250)])
def test_range_rate_uses_record_frequency_and_band(band, k, m):
    ftx = 719214560 if band == 0x54 else 209141411
    a = decode_record(_record(sec=100, usec=0, dop=10**11, ftx=ftx, band=band))
    b = decode_record(_record(sec=105, usec=0, dop=10**11 + 1_290_000_000, ftx=ftx, band=band))
    assert (a.turnaround_ratio, a.doppler_multiplier) == (k, m)
    (iv,) = check_doppler_counts([a, b], max_range_rate_mps=1e9)
    expect = _handbook_rdot(1_290_000_000, 5.0, 10.0 * ftx, k, m)
    assert math.isclose(iv.range_rate_mps, expect, rel_tol=4 * 2.220446049250313e-16)


def test_unsupported_band_and_frequency_change_are_flagged():
    a = decode_record(_record(sec=100, usec=0, band=0x44))      # C-band: no handbook K, M
    b = decode_record(_record(sec=105, usec=0, dop=136859486617 + 1291000000, band=0x44))
    assert a.turnaround_ratio is None and "BAND_UNSUPPORTED" in record_reason_codes(a)
    (iv,) = check_doppler_counts([a, b])
    assert "BAND_UNSUPPORTED" in iv.codes and math.isnan(iv.range_rate_mps)
    c = decode_record(_record(sec=100, usec=0))
    d = decode_record(_record(sec=105, usec=0, dop=136859486617 + 1291000000, ftx=209141412))
    (iv,) = check_doppler_counts([c, d])
    assert "FREQUENCY_CHANGED" in iv.codes
