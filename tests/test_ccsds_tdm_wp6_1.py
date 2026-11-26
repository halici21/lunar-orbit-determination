"""F6 WP6.1 (REQ-IO-001): CCSDS TDM (503.0-B-2) KVN reader/writer and TDM ingestion.

Oracles:
- the four hand-written examples in ``tests/fixtures/ccsds_tdm`` (annex E style, written for this
  project) with their values read by hand;
- round trip: parse -> write -> parse gives the same message, and writing is idempotent;
- time systems: an independent chain (SPICE ``unitim``/``str2et``; TDB - TT differs from ERFA's
  series by tens of microseconds, hence the 5e-5 s tolerance) and exact identities between TAI,
  GPS and TT tags (1e-9 s: double resolution of seconds past an epoch a few days away);
- ingestion: the existing CSV reader on the same rows, and rows -> TDM -> rows;
- an Orekit cross-read when ``orekit_jpype`` and its data are installed (skipped otherwise).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from lunar_od.ccsds_tdm import (
    DATA_KEYWORD_UNITS,
    _check_number,
    TDMEpoch,
    TDMFormatError,
    TDMRecord,
    TDMSegment,
    TrackingDataMessage,
    format_number,
    format_tdm,
    parse_tdm,
    read_tdm,
    tdb_seconds_to_tdm_epoch,
    tdm_epochs_to_tdb_seconds,
    write_tdm,
)
from lunar_od.measurement_ingestion import IngestedMeasurements, read_measurement_csv
from lunar_od.tdm_ingestion import ingested_to_tdm, read_measurement_tdm

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ccsds_tdm"
EXAMPLES = sorted(FIXTURES.glob("*.kvn"))
ET0 = 8.5e8                                   # TDB seconds past J2000, 2026-12-08


def _spice_ok() -> tuple[bool, str]:
    try:
        from lunar_od.spice_loader import load_spice_kernels
        load_spice_kernels()
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"SPICE kernels unavailable: {exc}"
    return True, ""


# ======================================================================
# reading the examples
# ======================================================================
def test_there_are_four_examples():
    assert [p.name for p in EXAMPLES] == [
        "media_meteo_utc.kvn", "one_way_doppler_tdb.kvn", "two_way_frequencies_tai.kvn",
        "two_way_range_angles_utc.kvn"]


def test_two_way_range_angles_example_values():
    tdm = read_tdm(FIXTURES / "two_way_range_angles_utc.kvn")
    assert (tdm.version, tdm.originator, tdm.message_id) == ("2.0", "LUNAR_OD", "WP61-E8-0001")
    assert tdm.creation_date == TDMEpoch(2026, 11, 25, 9, 30, 0.0)
    assert len(tdm.comments) == 2
    (seg,) = tdm.segments
    assert seg.time_system == "UTC"
    assert seg.participants == {1: "DSS-25", 2: "LRO"}
    assert seg.path == (1, 2, 1)
    assert seg.get("TURNAROUND_NUMERATOR") == "240"
    assert seg.metadata_comments == ("Two-way S-band pass, receive-time tags",)
    assert len(seg.data_comments) == 1
    assert len(seg.records) == 12
    ranges = seg.records_of("RANGE")
    assert [r.value for r in ranges] == [768801.234567, 768751.765432, 768702.3]
    # the day-of-year tag with Z reads to the same calendar epoch
    assert ranges[2].epoch == TDMEpoch(2026, 11, 25, 10, 2, 0.0)
    assert seg.records_of("DOPPLER_INSTANTANEOUS")[2].measurement == "-4.11657e-01"


def test_frequency_example_values_and_phase_count_text():
    (seg,) = read_tdm(FIXTURES / "two_way_frequencies_tai.kvn").segments
    assert seg.time_system == "TAI"
    assert [r.value for r in seg.records_of("RECEIVE_FREQ_1")] == [8430123456.789, 8430123456.912, 8430123457.035]
    assert seg.records_of("TRANSMIT_FREQ_RATE_1")[0].value == 0.25
    # phase counts keep every digit (4.3.11); only the float view rounds
    assert seg.records_of("TRANSMIT_PHASE_CT_1")[0].measurement == "123456789012345678901.25"
    assert seg.get("INTEGRATION_INTERVAL") == "1.0" and seg.get("INTEGRATION_REF") == "MIDDLE"


def test_media_example_has_two_segments_and_a_single_participant_segment():
    tdm = read_tdm(FIXTURES / "media_meteo_utc.kvn")
    assert len(tdm.segments) == 2
    assert tdm.segments[0].participants == {1: "DSS-25"}
    assert tdm.segments[0].get("MODE") is None
    assert {r.keyword for r in tdm.segments[0].records} == {"TROPO_DRY", "TROPO_WET", "PRESSURE", "RHUMIDITY",
                                                            "TEMPERATURE"}
    assert tdm.segments[1].records_of("CARRIER_POWER")[0].value == -150.25


def test_every_table_3_5_keyword_has_a_unit():
    assert len(DATA_KEYWORD_UNITS) == 27
    assert DATA_KEYWORD_UNITS["ANGLE_1"] == "deg" and DATA_KEYWORD_UNITS["DOPPLER_INTEGRATED"] == "km/s"


# ======================================================================
# round trip
# ======================================================================
@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_round_trip_reproduces_the_message(path, tmp_path):
    first = read_tdm(path)
    text = format_tdm(first)
    second = parse_tdm(text)
    assert second == first
    assert format_tdm(second) == text                       # writing is idempotent
    out = write_tdm(second, tmp_path / path.name)
    assert read_tdm(out) == first
    # measurement literals survive verbatim
    original = re.findall(r"^[ ]*[A-Z_0-9]+[ ]*=[ ]*\S+[ ]+(\S+)[ ]*\r?$", path.read_text(encoding="ascii"), re.M)
    assert original == [r.measurement for s in first.segments for r in s.records]


def test_line_terminators_and_blank_lines_are_insignificant():
    text = (FIXTURES / "one_way_doppler_tdb.kvn").read_text(encoding="ascii")
    lf = parse_tdm(text)
    assert parse_tdm(text.replace("\n", "\r\n")) == lf
    assert parse_tdm(text.replace("\n", "\n\n   \n")) == lf
    assert parse_tdm(text.replace(" = ", "=")) == lf


def test_enumerations_are_case_insensitive_and_written_upper_case():
    text = (FIXTURES / "one_way_doppler_tdb.kvn").read_text(encoding="ascii")
    seg = parse_tdm(text.replace("ANGLE_TYPE = AZEL", "ANGLE_TYPE = azel")
                    .replace("MODE = SEQUENTIAL", "MODE = Sequential")).segments[0]
    assert seg.get("ANGLE_TYPE") == "AZEL" and seg.get("MODE") == "SEQUENTIAL"
    assert seg.get("RANGE_UNITS") == "km"


def test_number_format_is_shortest_and_within_16_digits():
    rng = np.random.default_rng(20261125)
    values = np.concatenate([rng.normal(size=300) * 10.0 ** rng.integers(-30, 30, 300),
                             [0.1 + 0.2, 1e16, 1e-5, 384400.123, -2.5e-310, 7175000000.0, 5e-324]])
    for value in values:
        text = format_number(value)
        assert len(re.sub(r"[eE].*$", "", text).replace("-", "").replace(".", "")) <= 16, text
        _check_number(text)                                  # the reader's grammar accepts it
        back = float(text)
        if len(repr(float(value)).replace("-", "").replace(".", "").split("e")[0].lstrip("0")) <= 16:
            assert back == value
        else:                                              # 17-digit doubles: one rounding of 16 digits
            assert abs(back - value) <= 1e-15 * abs(value)
    with pytest.raises(TDMFormatError):
        format_number(float("nan"))


def test_epoch_text_round_trip():
    for text in ("2026-11-25T10:00:00", "2026-11-25T10:00:00.5", "2026-11-25T23:59:59.999999999",
                 "2028-366T00:00:01.25", "2016-12-31T23:59:60.5"):
        epoch = TDMEpoch.parse(text)
        assert TDMEpoch.parse(epoch.format()) == epoch
    assert TDMEpoch.parse("2028-366T00:00:00") == TDMEpoch(2028, 12, 31, 0, 0, 0.0)


def test_writer_fails_closed_on_an_invalid_message():
    seg = TDMSegment((("TIME_SYSTEM", "UTC"), ("PARTICIPANT_1", "DSS-25"), ("RANGE_UNIT", "km")),
                     (TDMRecord("RANGE", TDMEpoch(2026, 1, 1, 0, 0, 0.0), "1.0"),))
    tdm = TrackingDataMessage("2.0", TDMEpoch(2026, 1, 1, 0, 0, 0.0), "X", (seg,))
    with pytest.raises(TDMFormatError, match="RANGE_UNIT"):
        format_tdm(tdm)
    bad_value = TDMSegment((("TIME_SYSTEM", "UTC"), ("PARTICIPANT_1", "DSS-25")),
                           (TDMRecord("RANGE", TDMEpoch(2026, 1, 1, 0, 0, 0.0), "1.0 km"),))
    with pytest.raises(TDMFormatError, match="no units"):
        format_tdm(TrackingDataMessage("2.0", TDMEpoch(2026, 1, 1, 0, 0, 0.0), "X", (bad_value,)))


# ======================================================================
# negative tests: the reader fails closed
# ======================================================================
BASE = (FIXTURES / "one_way_doppler_tdb.kvn").read_text(encoding="ascii")


def _swap(old: str, new: str) -> str:
    assert old in BASE, old
    return BASE.replace(old, new, 1)


NEGATIVE = {
    "unknown header keyword": (_swap("ORIGINATOR = LUNAR_OD", "ORIGINATOR = LUNAR_OD\nOBJECT_NAME = LRO"),
                               "header keyword"),
    "header out of order": (_swap("CREATION_DATE = 2026-11-25T12:00:00Z\nORIGINATOR = LUNAR_OD",
                                  "ORIGINATOR = LUNAR_OD\nCREATION_DATE = 2026-11-25T12:00:00Z"), "order"),
    "missing ORIGINATOR": (_swap("ORIGINATOR = LUNAR_OD\n", ""), "ORIGINATOR"),
    "unsupported version": (_swap("CCSDS_TDM_VERS = 1.0", "CCSDS_TDM_VERS = 3.0"), "CCSDS_TDM_VERS"),
    "unknown metadata keyword": (_swap("ANGLE_TYPE = AZEL", "ANGLE_TYPE = AZEL\nANGLE_UNITS = deg"), "metadata keyword"),
    "metadata out of order": (_swap("TIME_SYSTEM = TDB\nPARTICIPANT_1 = LRO",
                                    "PARTICIPANT_1 = LRO\nTIME_SYSTEM = TDB"), "order"),
    "repeated metadata keyword": (_swap("RANGE_UNITS = km", "RANGE_UNITS = km\nRANGE_UNITS = km"), "repeated"),
    "missing TIME_SYSTEM": (_swap("TIME_SYSTEM = TDB\n", ""), "TIME_SYSTEM"),
    "unknown time system": (_swap("TIME_SYSTEM = TDB", "TIME_SYSTEM = LST"), "SANA"),
    "missing participants": (re.sub(r"PARTICIPANT_\d = \S+\n", "", _swap("MODE = SEQUENTIAL\nPATH = 1,2\n", "")),
                             "PARTICIPANT"),
    "non-normative RANGE_UNITS": (_swap("RANGE_UNITS = km", "RANGE_UNITS = m"), "RANGE_UNITS"),
    "non-normative ANGLE_TYPE": (_swap("ANGLE_TYPE = AZEL", "ANGLE_TYPE = HADEC"), "ANGLE_TYPE"),
    "PATH to an undefined participant": (_swap("PATH = 1,2", "PATH = 1,3"), "undefined participant"),
    "SEQUENTIAL without PATH": (_swap("PATH = 1,2\n", ""), "SEQUENTIAL"),
    "PATH without MODE": (_swap("MODE = SEQUENTIAL\n", ""), "MODE"),
    "CORRECTION without CORRECTIONS_APPLIED": (_swap("CORRECTIONS_APPLIED = YES\n", ""), "CORRECTIONS_APPLIED"),
    "unknown data keyword": (_swap("RANGE = 2026-11-25T12:00:00 384400.5", "RANGE_RATE = 2026-11-25T12:00:00 0.2"),
                             "data keyword"),
    "indexed keyword of an undefined participant": (
        _swap("DATA_START\n", "DATA_START\nRECEIVE_FREQ_3 = 2026-11-25T12:00:00 2.2e9\n"), "data keyword"),
    "unit after the value": (_swap("RANGE = 2026-11-25T12:00:00 384400.5", "RANGE = 2026-11-25T12:00:00 384400.5 [km]"),
                             "no units"),
    "unit glued to the value": (_swap("384400.5", "384400.5km"), "not a TDM"),
    "missing measurement": (_swap("RANGE = 2026-11-25T12:00:00 384400.5", "RANGE = 2026-11-25T12:00:00"),
                            "timetag measurement"),
    "duplicated keyword and time tag": (_swap("RANGE = 2026-11-25T12:00:30.25 384406.6",
                                              "RANGE = 2026-11-25T12:00:00 384406.6"), "3.4.11"),
    "non-chronological records": (_swap("RANGE = 2026-11-25T12:00:30.25 384406.6",
                                        "RANGE = 2026-11-25T11:00:00 384406.6"), "chronological"),
    "month 13": (_swap("RANGE = 2026-11-25T12:00:00", "RANGE = 2026-13-25T12:00:00"), "calendar day"),
    "day 366 of a common year": (_swap("RANGE = 2026-11-25T12:00:00", "RANGE = 2026-366T12:00:00"), "calendar day"),
    "hour 24": (_swap("RANGE = 2026-11-25T12:00:00", "RANGE = 2026-11-25T24:00:00"), "clock field"),
    "leap second outside UTC": (_swap("RANGE = 2026-11-25T12:00:00", "RANGE = 2026-11-25T11:59:60.5"), "leap second"),
    "Z suffix outside UTC": (_swap("RANGE = 2026-11-25T12:00:00", "RANGE = 2026-11-25T12:00:00Z"), "Z suffix"),
    "NaN value": (_swap("384400.5", "NaN"), "not a TDM"),
    "17-digit fixed value": (_swap("384400.5", "384400.12345678901"), "16 digits"),
    "comma decimal": (_swap("384400.5", "384400,5"), "not a TDM"),
    "lower-case keyword": (_swap("RANGE = 2026-11-25T12:00:00", "range = 2026-11-25T12:00:00"), "uppercase"),
    "tab character": (_swap("RANGE = 2026-11-25T12:00:00 ", "RANGE = 2026-11-25T12:00:00\t"), "ASCII"),
    "line over 254 characters": (_swap("ORIGINATOR = LUNAR_OD", "ORIGINATOR = " + "L" * 250), "254"),
    "comment after the first record": (_swap("DATA_STOP", "COMMENT late\nDATA_STOP"), "4.5.2c"),
    "comment after a metadata keyword": (_swap("META_STOP", "COMMENT late\nMETA_STOP"), "4.5.2b"),
    "header comment before the version": ("COMMENT first\n" + BASE, "header COMMENT"),
    "missing DATA_STOP": (_swap("DATA_STOP\n", ""), "DATA_STOP"),
    "empty data section": (re.sub(r"DATA_START\n.*DATA_STOP", "DATA_START\nDATA_STOP", BASE, flags=re.S),
                           "no tracking data record"),
    "no segment": (BASE.split("META_START")[0], "at least one segment"),
    "negative RANGE_MODULUS": (_swap("RANGE_UNITS = km", "RANGE_MODULUS = -1.0\nRANGE_UNITS = km"), "non-negative"),
    "STOP_TIME before START_TIME": (_swap("TIME_SYSTEM = TDB",
                                          "TIME_SYSTEM = TDB\nSTART_TIME = 2026-11-25T13:00:00\n"
                                          "STOP_TIME = 2026-11-25T12:00:00"), "STOP_TIME"),
    "DATA_TYPES does not list a record keyword": (
        _swap("TIME_SYSTEM = TDB", "DATA_TYPES = RANGE,ANGLE_1,ANGLE_2\nTIME_SYSTEM = TDB"), "DATA_TYPES"),
    "metadata keyword in the data section": (_swap("DATA_START\n", "DATA_START\nRANGE_UNITS = km\n"),
                                             "header/metadata keyword"),
}


@pytest.mark.parametrize("name", sorted(NEGATIVE))
def test_reader_fails_closed(name):
    text, message = NEGATIVE[name]
    with pytest.raises(TDMFormatError, match=re.escape(message)):
        parse_tdm(text)


def test_non_ascii_file_is_refused(tmp_path):
    path = tmp_path / "bad.kvn"
    path.write_bytes(BASE.replace("LUNAR_OD", "LUNAR_ÖD").encode("utf-8"))
    with pytest.raises(TDMFormatError, match="ASCII"):
        read_tdm(path)


# ======================================================================
# time systems
# ======================================================================
def test_tdb_tags_are_seconds_past_j2000_and_invert():
    assert tdm_epochs_to_tdb_seconds([TDMEpoch(2000, 1, 1, 12, 0, 0.0)], "TDB", 0.0)[0] == 0.0
    assert tdm_epochs_to_tdb_seconds([TDMEpoch(2000, 1, 2, 12, 0, 1.5)], "TDB", 86400.0)[0] == 1.5
    for t in (-90000.0, -0.25, 0.0, 123.456789, 604799.999999):
        epoch = tdb_seconds_to_tdm_epoch(t, ET0)
        assert abs(tdm_epochs_to_tdb_seconds([epoch], "TDB", ET0)[0] - t) < 1e-9
        assert TDMEpoch.parse(epoch.format()) == epoch


def test_tai_gps_and_tt_tags_agree():
    tai = TDMEpoch(2026, 11, 25, 10, 0, 0.0)
    tt = TDMEpoch(2026, 11, 25, 10, 0, 32.184)
    gps = TDMEpoch(2026, 11, 25, 9, 59, 41.0)
    values = [tdm_epochs_to_tdb_seconds([e], s, ET0)[0] for e, s in ((tai, "TAI"), (tt, "TT"), (gps, "GPS"))]
    assert max(values) - min(values) < 1e-9
    # TDB - TT is periodic with a 1.66 ms amplitude
    tdb_as_tt = tdm_epochs_to_tdb_seconds([tt], "TDB", ET0)[0]
    assert 0.0 < abs(values[1] - tdb_as_tt) < 1.7e-3


@pytest.mark.parametrize("system", ["UT1", "TCB", "SCLK", "MET"])
def test_unsupported_time_systems_are_refused(system):
    with pytest.raises(TDMFormatError, match="no conversion"):
        tdm_epochs_to_tdb_seconds([TDMEpoch(2026, 1, 1, 0, 0, 0.0)], system, ET0)


def test_tt_and_utc_against_the_spice_chain():
    ok, reason = _spice_ok()
    if not ok:
        pytest.skip(reason)
    import spiceypy as spice

    tt = TDMEpoch(2026, 11, 25, 10, 0, 32.184)
    tt_seconds = tdm_epochs_to_tdb_seconds([tt], "TDB", 0.0)[0]       # the tag as TT seconds past J2000
    ours = tdm_epochs_to_tdb_seconds([tt], "TT", ET0)[0] + ET0
    assert abs(ours - spice.unitim(tt_seconds, "TT", "TDB")) < 5e-5
    utc = TDMEpoch(2026, 11, 25, 9, 59, 23.0)                          # TAI - UTC = 37 s
    ours_utc = tdm_epochs_to_tdb_seconds([utc], "UTC", ET0)[0] + ET0
    assert abs(ours_utc - spice.str2et("2026-11-25T09:59:23.000")) < 5e-5
    assert abs(ours_utc - ours) < 1e-9                                         # same instant as the TT tag


# ======================================================================
# ingestion
# ======================================================================
TDB_META = """META_START
TIME_SYSTEM = TDB
PARTICIPANT_1 = {station}
PARTICIPANT_2 = {craft}
MODE = SEQUENTIAL
PATH = {path}
RANGE_UNITS = km
ANGLE_TYPE = AZEL
{extra}META_STOP
DATA_START
{records}DATA_STOP
"""
HEADER = "CCSDS_TDM_VERS = 2.0\nCREATION_DATE = 2026-11-25T00:00:00\nORIGINATOR = LUNAR_OD\n"
STATIONS = ("DSS-25", "DSS-63")
T0 = "2026-12-08T11:06:40"                    # = ET0 in TDB
T1 = "2026-12-08T11:07:40"


def _tdm(*segments) -> str:
    return HEADER + "".join(segments)


def _segment(records, *, station="DSS-25", craft="LRO", path="1,2,1", extra=""):
    return TDB_META.format(station=station, craft=craft, path=path, extra=extra, records=records)


RANGE_ANGLES = (f"RANGE = {T0} 768800.0\nANGLE_1 = {T0} 90.0\nANGLE_2 = {T0} 30.0\n"
                f"RANGE = {T1} 768750.0\nANGLE_1 = {T1} 90.5\nANGLE_2 = {T1} 30.5\n")


def test_two_way_round_trip_range_is_halved_and_rows_match_the_csv_reader(tmp_path):
    got = read_measurement_tdm(parse_tdm(_tdm(_segment(RANGE_ANGLES))), "position", station_names=STATIONS,
                               et0_s=ET0, two_way_convention="round_trip")
    csv = tmp_path / "rows.csv"
    csv.write_text("t_s,range_m,az_rad,el_rad,station_name,time_index\n"
                   f"0,384400000.0,{math.radians(90.0)!r},{math.radians(30.0)!r},DSS-25,1\n"
                   f"60,384375000.0,{math.radians(90.5)!r},{math.radians(30.5)!r},DSS-25,2\n", encoding="utf-8")
    expected = read_measurement_csv(csv, "position", station_names=STATIONS)
    assert got.measurement_type == "position" and got.station_names == STATIONS
    np.testing.assert_array_equal(got.obs_data, expected.obs_data)


def test_one_way_equivalent_keeps_the_value_and_segments_are_joined():
    ranges = f"RANGE = {T0} 384400.0\nRANGE = {T1} 384375.0\n"
    angles = f"ANGLE_1 = {T0} 90.0\nANGLE_2 = {T0} 30.0\nANGLE_1 = {T1} 90.5\nANGLE_2 = {T1} 30.5\n"
    other = f"RANGE = {T1} 384000.0\nANGLE_1 = {T1} 10.0\nANGLE_2 = {T1} 5.0\n"
    tdm = parse_tdm(_tdm(_segment(ranges), _segment(angles), _segment(other, station="DSS-63")))
    got = read_measurement_tdm(tdm, "position", station_names=STATIONS, et0_s=ET0,
                               two_way_convention="one_way_equivalent").obs_data
    assert got.shape == (3, 6)
    np.testing.assert_array_equal(got[:, [0, 1, 4, 5]], [[0.0, 384400000.0, 1, 1], [60.0, 384375000.0, 1, 2],
                                                        [60.0, 384000000.0, 2, 2]])


def test_range_rate_layout_from_one_way_doppler_example():
    got = read_measurement_tdm(FIXTURES / "one_way_doppler_tdb.kvn", "range_rate",
                               station_names=("DSS-35",), et0_s=ET0)
    t_first = tdm_epochs_to_tdb_seconds([TDMEpoch(2026, 11, 25, 12, 0, 0.0)], "TDB", ET0)[0]
    np.testing.assert_array_equal(got.obs_data[:, 0], [t_first, t_first + 30.25])
    np.testing.assert_allclose(got.obs_data[:, 1:], [
        [384400500.0, 201.5, math.radians(271.25), math.radians(12.5), 1, 1],
        [384406600.0, 201.6, math.radians(271.375), math.radians(12.625), 1, 2]], rtol=1e-15, atol=0)


def test_utc_example_ingests_with_kernels():
    ok, reason = _spice_ok()
    if not ok:
        pytest.skip(reason)
    got = read_measurement_tdm(FIXTURES / "two_way_range_angles_utc.kvn", "range_rate", station_names=("DSS-25",),
                               et0_s=ET0, two_way_convention="round_trip").obs_data
    np.testing.assert_allclose(np.diff(got[:, 0]), [60.0, 60.0], atol=1e-9)
    np.testing.assert_allclose(got[:, 1], [384400617.2835, 384375882.716, 384351150.0], rtol=1e-15)
    np.testing.assert_allclose(got[:, 2], [-206.1725, -206.0005, -205.8285], rtol=1e-15)   # sign kept


@pytest.mark.parametrize("layout", ["position", "range_rate"])
def test_rows_to_tdm_to_rows(layout):
    rng = np.random.default_rng(61)
    n = 7
    t = np.sort(rng.uniform(0.0, 7200.0, n))
    station = rng.integers(1, 3, n).astype(float)
    rows = [t, rng.uniform(3.6e8, 4.1e8, n)]
    if layout == "range_rate":
        rows.append(rng.normal(0.0, 900.0, n))
    rows += [rng.uniform(0.0, 2 * math.pi, n), rng.uniform(0.0, 1.4, n), station, np.arange(1.0, n + 1)]
    obs = np.column_stack(rows)
    tdm = ingested_to_tdm(IngestedMeasurements(layout, obs, STATIONS), spacecraft="LRO", et0_s=ET0,
                          originator="LUNAR_OD", creation_date="2026-11-25T00:00:00")
    back = read_measurement_tdm(parse_tdm(format_tdm(tdm)), layout, station_names=STATIONS, et0_s=ET0).obs_data
    np.testing.assert_allclose(back[:, 0], obs[:, 0], rtol=0, atol=1e-9)
    # km/deg conversions and the 16-digit limit of 4.3.5: a few roundings of 1.1e-16 each
    np.testing.assert_allclose(back[:, 1:], obs[:, 1:], rtol=2e-15, atol=0)


INGEST_NEGATIVE = {
    "unmapped data keyword": (_segment(RANGE_ANGLES + f"RECEIVE_FREQ_1 = {T0} 2.2e9\n"), "round_trip", "no column"),
    "two-way without a convention": (_segment(RANGE_ANGLES), None, "two_way_convention"),
    "convention on one-way data": (_segment(RANGE_ANGLES, path="2,1"), "round_trip", "one-way"),
    "incomplete epoch": (_segment(RANGE_ANGLES.replace(f"ANGLE_2 = {T1} 30.5\n", "")), "round_trip", "lacks ANGLE_2"),
    "range in seconds": (_segment(RANGE_ANGLES).replace("RANGE_UNITS = km", "RANGE_UNITS = s"), "round_trip",
                         "RANGE_UNITS"),
    "ambiguous range": (_segment(RANGE_ANGLES).replace("RANGE_UNITS = km", "RANGE_MODULUS = 32768.0\nRANGE_UNITS = km"),
                        "round_trip", "RANGE_MODULUS"),
    "transmit time tags": (_segment(RANGE_ANGLES).replace("PATH = 1,2,1", "PATH = 1,2,1\nTIMETAG_REF = TRANSMIT"),
                           "round_trip", "TIMETAG_REF"),
    "RADEC angles": (_segment(RANGE_ANGLES).replace("ANGLE_TYPE = AZEL", "ANGLE_TYPE = RADEC"), "round_trip",
                     "ANGLE_TYPE"),
    "station delay": (_segment(RANGE_ANGLES, extra="RECEIVE_DELAY_1 = 1.0e-06\n"), "round_trip", "RECEIVE_DELAY_1"),
    "receiver is not a station": (_segment(RANGE_ANGLES, station="DSS-99"), "round_trip", "station_names"),
    "three-way path": (_segment(RANGE_ANGLES, path="2,1,2,1"), "round_trip", "neither one-way nor two-way"),
    "degraded data": (_segment(RANGE_ANGLES, extra="DATA_QUALITY = DEGRADED\n"), "round_trip", "DEGRADED"),
    "corrections not applied": (_segment(RANGE_ANGLES, extra="CORRECTION_RANGE = 0.1\nCORRECTIONS_APPLIED = NO\n"),
                                "round_trip", "not applied"),
    "azimuth out of range": (_segment(RANGE_ANGLES.replace(f"ANGLE_1 = {T0} 90.0", f"ANGLE_1 = {T0} 360.0")),
                             "round_trip", "azimuth"),
}


@pytest.mark.parametrize("name", sorted(INGEST_NEGATIVE))
def test_ingestion_fails_closed(name):
    segment, convention, message = INGEST_NEGATIVE[name]
    tdm = parse_tdm(_tdm(segment))
    with pytest.raises(ValueError, match=message):
        read_measurement_tdm(tdm, "position", station_names=STATIONS, et0_s=ET0, two_way_convention=convention)


def test_two_spacecraft_are_refused():
    tdm = parse_tdm(_tdm(_segment(RANGE_ANGLES), _segment(RANGE_ANGLES, craft="KPLO", station="DSS-63")))
    with pytest.raises(ValueError, match="more than one spacecraft"):
        read_measurement_tdm(tdm, "position", station_names=STATIONS, et0_s=ET0, two_way_convention="round_trip")


def test_same_record_in_two_segments_is_refused():
    tdm = parse_tdm(_tdm(_segment(RANGE_ANGLES), _segment(RANGE_ANGLES)))
    with pytest.raises(ValueError, match="more than one segment"):
        read_measurement_tdm(tdm, "position", station_names=STATIONS, et0_s=ET0, two_way_convention="round_trip")


# ======================================================================
# Orekit cross-read
# ======================================================================
def _orekit():
    try:
        import orekit_jpype
    except ImportError:
        pytest.skip("Orekit cross-read needs orekit_jpype (Orekit's Python wrapper), which is not installed")
    try:
        orekit_jpype.initVM()
        from orekit_jpype.pyhelpers import setup_orekit_data
        setup_orekit_data(from_pip_library=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Orekit or its orekit-data is unavailable: {exc}")
    from org.orekit.data import DataSource
    from org.orekit.files.ccsds.ndm import ParserBuilder
    return ParserBuilder, DataSource


@pytest.mark.parametrize("name", ["two_way_range_angles_utc.kvn", "one_way_doppler_tdb.kvn"])
def test_orekit_reads_the_same_observations(name, tmp_path):  # pragma: no cover - needs Orekit
    ParserBuilder, DataSource = _orekit()
    ours = read_tdm(FIXTURES / name)
    # cross-read what our writer produced, not only our input
    written = write_tdm(ours, tmp_path / name)
    theirs = ParserBuilder().buildTdmParser().parseMessage(DataSource(str(written)))
    si = {"RANGE": 1000.0, "DOPPLER_INSTANTANEOUS": 1000.0, "ANGLE_1": math.pi / 180.0, "ANGLE_2": math.pi / 180.0}
    segments = list(theirs.getSegments())
    assert len(segments) == len(ours.segments)
    for mine, other in zip(ours.segments, segments):
        assert str(other.getMetadata().getTimeSystem().name()) == mine.time_system
        observations = list(other.getData().getObservations())
        assert len(observations) == len(mine.records)
        for record, obs in zip(mine.records, observations):
            assert str(obs.getType().name()) == record.keyword
            assert math.isclose(float(obs.getMeasurement()), record.value * si[record.keyword], rel_tol=1e-15)
