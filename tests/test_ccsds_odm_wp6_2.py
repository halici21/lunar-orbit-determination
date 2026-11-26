"""WP6.2 / REQ-IO-002: CCSDS 502.0-B KVN OEM and OPM writers with covariance.

Oracles:
* Round trip through the module's own minimal reader: the kilometre values read
  back are bit-identical to ``state_m / 1000`` (17 significant digits written).
* Hand-computed calendar epochs (J2000.0 = 2000-01-01T12:00:00 TDB) and the
  standard's covariance ordering, checked against independently written text.
* Orekit's ODM parser, only when Orekit (``orekit_jpype``) and its data are
  installed; otherwise skipped with the reason.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pytest

from lunar_od import ccsds_odm as odm

CREATED = "2026-11-25T00:00:00"
ET0 = 8.123456789012345e8  # TDB s past J2000, a non-round binary value


def _trajectory(n=7, step=60.0, seed=1):
    rng = np.random.default_rng(seed)
    epochs = ET0 + step * np.arange(n)
    states = np.column_stack([
        1.837e6 + rng.normal(0, 1e5, n), rng.normal(0, 1e5, n), rng.normal(0, 1e5, n),
        rng.normal(0, 1e3, n), 1.6e3 + rng.normal(0, 10, n), rng.normal(0, 10, n)])
    return epochs, states


def _covariance(seed=2):
    rng = np.random.default_rng(seed)
    scale = np.diag([30.0, 40.0, 50.0, 0.03, 0.02, 0.01])  # m, m/s sigmas
    a = rng.normal(size=(6, 6))
    corr = a @ a.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    return scale @ corr @ scale


def _oem_text(**kwargs):
    epochs, states = _trajectory()
    cov = _covariance()
    seg = odm.OemSegment(
        epochs, states, "LUNAR SAT", "2026-999A",
        covariances=(odm.OemCovariance(epochs[0], cov), odm.OemCovariance(epochs[3], 2 * cov)),
        interpolation="HERMITE", interpolation_degree=5)
    return odm.format_oem(seg, originator="LUNAR_OD", creation_date=CREATED, **kwargs), epochs, states, cov


# ---------------------------------------------------------------------------
# Epochs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("et, text", [
    (0.0, "2000-01-01T12:00:00.000000"),
    (-43200.0, "2000-01-01T00:00:00.000000"),
    (86400.0 * 366 - 43200.0, "2001-01-01T00:00:00.000000"),  # 2000 is a leap year
    (43199.9999994, "2000-01-01T23:59:59.999999"),  # rounding below a carry
    (43199.9999999, "2000-01-02T00:00:00.000000"),  # rounding carries to next day
])
def test_epoch_calendar_known_values(et, text):
    assert odm.et_to_tdb_calendar(et) == text


def test_epoch_round_trip_within_half_unit_of_written_precision():
    for decimals in (0, 3, 6, 9):
        for et in (ET0, -1.5e8 + 0.123456789, 1.0e9 + 0.5):
            text = odm.et_to_tdb_calendar(et, decimals=decimals)
            back = odm.tdb_calendar_to_et(text)
            assert abs(back - et) <= 0.5 * 10.0 ** -decimals + 1e-6 * 2.0 ** -20 * abs(et)
            # Re-formatting the parsed epoch is idempotent.
            assert odm.et_to_tdb_calendar(back, decimals=decimals) == text


def test_day_of_year_epoch_is_accepted():
    assert odm.tdb_calendar_to_et("2000-001T12:00:00") == 0.0
    assert odm.tdb_calendar_to_et("2000-366T12:00:00") == 365 * 86400.0


@pytest.mark.parametrize("bad", [
    "2000-01-01 12:00:00", "2000-13-01T00:00:00", "2000-01-01T24:00:00",
    "2016-12-31T23:59:60", "2001-366T00:00:00", "not an epoch"])
def test_invalid_epochs_rejected(bad):
    with pytest.raises(odm.OdmFormatError):
        odm.tdb_calendar_to_et(bad)


# ---------------------------------------------------------------------------
# OEM
# ---------------------------------------------------------------------------

def test_oem_round_trip_positions_bit_exact_in_km():
    text, epochs, states, cov = _oem_text()
    msg = odm.read_oem(text)
    seg = msg.segments[0]
    assert msg.version == "3.0"
    assert msg.header == {"CREATION_DATE": CREATED, "ORIGINATOR": "LUNAR_OD"}
    np.testing.assert_array_equal(msg.states_km[0], states / 1000.0)
    # Back in metres the only change is the km->m scaling: at most 1 ulp.
    assert np.all(np.abs(seg.states_m - states) <= np.spacing(np.abs(states)))
    assert np.all(np.abs(seg.epochs_et_s - epochs) <= 0.5e-6 + np.spacing(ET0))
    assert (seg.center_name, seg.ref_frame) == ("MOON", "EME2000")
    assert (seg.interpolation, seg.interpolation_degree) == ("HERMITE", 5)
    assert [c.cov_ref_frame for c in seg.covariances] == ["EME2000", "EME2000"]
    np.testing.assert_allclose(seg.covariances[0].covariance_m, cov, rtol=1e-15, atol=0)
    np.testing.assert_allclose(seg.covariances[1].covariance_m, 2 * cov, rtol=1e-15, atol=0)


def test_oem_metadata_and_time_system_are_explicit():
    text, epochs, *_ = _oem_text()
    for line in ("CCSDS_OEM_VERS = 3.0", "CENTER_NAME = MOON", "REF_FRAME = EME2000",
                 "TIME_SYSTEM = TDB", "META_START", "META_STOP",
                 "COVARIANCE_START", "COVARIANCE_STOP", "COV_REF_FRAME = EME2000",
                 f"START_TIME = {odm.et_to_tdb_calendar(epochs[0])}",
                 f"STOP_TIME = {odm.et_to_tdb_calendar(epochs[-1])}"):
        assert line in text.splitlines(), line
    assert "REF_FRAME = MCI" not in text


def test_oem_covariance_lower_triangle_order_and_units():
    # Element (i, j) = 10*(i+1) + (j+1) in km units, so the written rows can be
    # checked against the standard's row-wise lower-triangle order by eye.
    cov_km = np.array([[10.0 * max(i, j) + min(i, j) + 11.0 for j in range(6)] for i in range(6)])
    epochs, states = _trajectory(n=2)
    seg = odm.OemSegment(epochs, states, "S", "ID",
                         covariances=(odm.OemCovariance(epochs[0], cov_km * 1e6),))
    text = odm.format_oem(seg, originator="O", creation_date=CREATED)
    block = text.split("COV_REF_FRAME = EME2000\n")[1].split("COVARIANCE_STOP")[0].split("\n")[:6]
    rows = [[float(v) for v in line.split()] for line in block]
    assert rows == [[11.0], [21.0, 22.0], [31.0, 32.0, 33.0], [41.0, 42.0, 43.0, 44.0],
                    [51.0, 52.0, 53.0, 54.0, 55.0], [61.0, 62.0, 63.0, 64.0, 65.0, 66.0]]


def test_oem_output_is_deterministic_and_ascii():
    a, *_ = _oem_text()
    b, *_ = _oem_text()
    assert a == b
    a.encode("ascii")


def test_oem_multiple_segments_and_version_2():
    epochs, states = _trajectory()
    segs = [odm.OemSegment(epochs[:3], states[:3], "S", "ID"),
            odm.OemSegment(epochs[3:], states[3:], "S", "ID")]
    text = odm.format_oem(segs, originator="O", creation_date=CREATED, version="2.0")
    msg = odm.read_oem(text)
    assert msg.version == "2.0" and len(msg.segments) == 2
    np.testing.assert_array_equal(np.vstack(msg.states_km), states / 1000.0)


def test_oem_from_filter_result_drops_bias_states():
    epochs, states = _trajectory(n=4)
    cov = _covariance()
    aug_states = np.column_stack([states, np.full(4, 7.0)])
    aug_cov = np.zeros((4, 7, 7))
    aug_cov[:, :6, :6] = cov
    aug_cov[:, 6, 6] = 1.0
    result = SimpleNamespace(t_update_s=epochs - ET0, state_estimates=aug_states,
                             covariances=aug_cov)
    seg = odm.oem_segment_from_filter_result(result, ET0, object_name="S", object_id="ID",
                                             covariance_stride=2)
    assert seg.states_m.shape == (4, 6) and len(seg.covariances) == 2
    msg = odm.read_oem(odm.format_oem(seg, originator="O", creation_date=CREATED))
    np.testing.assert_array_equal(msg.states_km[0], states / 1000.0)


def test_write_and_read_oem_file(tmp_path):
    epochs, states = _trajectory()
    path = odm.write_oem(tmp_path / "t.oem", odm.OemSegment(epochs, states, "S", "ID"),
                         originator="O", creation_date=CREATED)
    assert b"\r\n" not in path.read_bytes()
    np.testing.assert_array_equal(odm.read_oem(path).states_km[0], states / 1000.0)


# ---------------------------------------------------------------------------
# OPM
# ---------------------------------------------------------------------------

def test_opm_round_trip_state_and_covariance():
    _, states = _trajectory(n=1)
    cov = _covariance()
    text = odm.format_opm(ET0, states[0], object_name="S", object_id="ID", originator="O",
                          covariance_m=cov, creation_date=CREATED)
    msg = odm.read_opm(text)
    np.testing.assert_array_equal(msg.state_km, states[0] / 1000.0)
    np.testing.assert_allclose(msg.covariance_m, cov, rtol=1e-15, atol=0)
    assert msg.epoch_text == odm.et_to_tdb_calendar(ET0)
    assert (msg.center_name, msg.ref_frame, msg.cov_ref_frame) == ("MOON", "EME2000", "EME2000")
    keys = re.findall(r"^(C[XYZ]\w*) =", text, re.M)
    assert keys == list(odm._OPM_COV_KEYS) and len(keys) == 21
    assert keys[:3] == ["CX_X", "CY_X", "CY_Y"] and keys[-1] == "CZ_DOT_Z_DOT"
    assert "X = " in text and "[km/s]" in text and "[km**2/s**2]" in text


def test_opm_without_covariance():
    _, states = _trajectory(n=1)
    msg = odm.read_opm(odm.format_opm(ET0, states[0], object_name="S", object_id="ID",
                                      originator="O", creation_date=CREATED))
    assert msg.covariance_m is None


# ---------------------------------------------------------------------------
# Negative tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mutate, match", [
    (lambda e, s, c: (e[::-1], s, c), "strictly increasing"),
    (lambda e, s, c: (e, s[:, :3], c), r"shape \(N, 6\)"),
    (lambda e, s, c: (e, np.where(np.eye(len(e), 6, dtype=bool), np.nan, s), c), "non-finite"),
    (lambda e, s, c: (e, s, c + np.triu(np.ones((6, 6)), 1)), "not symmetric"),
    (lambda e, s, c: (e, s, c - 2 * np.diag(np.diag(c))), "negative variance"),
    (lambda e, s, c: (e, s, c[:3, :3]), "6x6"),
])
def test_oem_invalid_inputs_rejected(mutate, match):
    epochs, states = _trajectory()
    e, s, c = mutate(epochs, states, _covariance())
    with pytest.raises(odm.OdmFormatError, match=match):
        odm.format_oem(odm.OemSegment(e, s, "S", "ID", covariances=(odm.OemCovariance(epochs[0], c),)),
                       originator="O", creation_date=CREATED)


@pytest.mark.parametrize("kwargs, match", [
    ({"ref_frame": "MCI"}, "Mars"),
    ({"ref_frame": "MOON_PA"}, "no frame transformation"),
    ({"center_name": "MARS"}, "CENTER_NAME"),
    ({"object_name": "two\nlines"}, "single-line"),
    ({"interpolation": "HERMITE"}, "INTERPOLATION_DEGREE"),
    ({"interpolation_degree": 3}, "without INTERPOLATION"),
    ({"usable_start_et_s": ET0 - 1.0}, "USEABLE"),
])
def test_oem_invalid_metadata_rejected(kwargs, match):
    epochs, states = _trajectory()
    with pytest.raises(odm.OdmFormatError, match=match):
        odm.format_oem(odm.OemSegment(epochs, states, **{"object_name": "S", "object_id": "ID", **kwargs}),
                       originator="O", creation_date=CREATED)


def test_covariance_epoch_outside_segment_rejected():
    epochs, states = _trajectory()
    seg = odm.OemSegment(epochs, states, "S", "ID",
                         covariances=(odm.OemCovariance(epochs[-1] + 1.0, _covariance()),))
    with pytest.raises(odm.OdmFormatError, match="outside the segment"):
        odm.format_oem(seg, originator="O", creation_date=CREATED)


def test_version_and_message_id_rules():
    epochs, states = _trajectory()
    seg = odm.OemSegment(epochs, states, "S", "ID")
    with pytest.raises(odm.OdmFormatError, match="version"):
        odm.format_oem(seg, originator="O", version="1.0")
    with pytest.raises(odm.OdmFormatError, match="MESSAGE_ID"):
        odm.format_oem(seg, originator="O", version="2.0", message_id="M1")
    assert "MESSAGE_ID = M1" in odm.format_oem(seg, originator="O", message_id="M1")


@pytest.mark.parametrize("edit, match", [
    (lambda t: t.replace("TIME_SYSTEM = TDB", "TIME_SYSTEM = UTC"), "TDB only"),
    (lambda t: t.replace("CCSDS_OEM_VERS = 3.0", "CCSDS_OPM_VERS = 3.0"), "CCSDS_OEM_VERS"),
    (lambda t: t.replace("META_STOP\n", ""), "before META_STOP"),
    (lambda t: t.replace("COVARIANCE_STOP\n", ""), "COVARIANCE_STOP"),
    (lambda t: t.replace("ORIGINATOR = LUNAR_OD\n", ""), "ORIGINATOR"),
    (lambda t: re.sub(r"^(\d{4}-\S+(?: \S+){5}) \S+$", r"\1", t, count=1, flags=re.M), "epoch \\+ 6"),
    (lambda t: re.sub(r"^\S+ \S+$", "1.0 2.0 3.0", t, count=1, flags=re.M), "lower-triangular"),
    (lambda t: t.replace("REF_FRAME = EME2000", "REF_FRAME = ITRF2000", 1), "not supported"),
])
def test_oem_reader_rejects_malformed_files(edit, match):
    text, *_ = _oem_text()
    with pytest.raises(odm.OdmFormatError, match=match):
        odm.read_oem(edit(text))


def test_opm_reader_rejects_partial_covariance():
    _, states = _trajectory(n=1)
    text = odm.format_opm(ET0, states[0], object_name="S", object_id="ID", originator="O",
                          covariance_m=_covariance(), creation_date=CREATED)
    with pytest.raises(odm.OdmFormatError, match="21"):
        odm.read_opm(re.sub(r"^CZ_DOT_Z_DOT = .*$", "", text, flags=re.M))
    with pytest.raises(odm.OdmFormatError, match="missing"):
        odm.read_opm(re.sub(r"^EPOCH = .*$", "", text, flags=re.M))


# ---------------------------------------------------------------------------
# Orekit cross-read (optional: runs only where orekit_jpype and a JVM exist)
# ---------------------------------------------------------------------------

def _orekit_parser_builder(tmp_path):
    """An Orekit ParserBuilder on an isolated DataContext (the default one is untouched).

    The context holds a one-entry UTC-TAI table, needed only to parse the UTC
    CREATION_DATE, and a stub Moon/Earth body: parsing data lines never queries
    a body ephemeris, so no JPL files are needed. Orekit's own TDB scale converts
    the epochs.
    """
    orekit_jpype = pytest.importorskip(
        "orekit_jpype", reason="Orekit (orekit_jpype) not installed; cross-read skipped")
    try:
        orekit_jpype.initVM()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Orekit JVM could not be started: {exc}")
    from jpype import JImplements, JOverride
    from java.io import File
    from org.orekit.data import DirectoryCrawler, LazyLoadedDataContext
    from org.orekit.files.ccsds.ndm import ParserBuilder
    from org.orekit.frames import FramesFactory

    data = tmp_path / "orekit-data"
    data.mkdir()
    (data / "tai-utc.dat").write_text(
        " 1972 JAN  1 =JD 2441317.5  TAI-UTC=  10.0       S + (MJD - 41317.) X 0.0      S\n",
        encoding="ascii")
    context = LazyLoadedDataContext()
    context.getDataProvidersManager().addProvider(DirectoryCrawler(File(str(data))))

    @JImplements("org.orekit.bodies.CelestialBody")
    class StubBody:
        def __init__(self, name):
            self.name = name

        @JOverride
        def getName(self):
            return self.name

        @JOverride
        def getGM(self):
            return 4.9028e12

        @JOverride
        def getInertiallyOrientedFrame(self):
            return FramesFactory.getEME2000()

        @JOverride
        def getBodyOrientedFrame(self):
            return FramesFactory.getEME2000()

        @JOverride
        def getPVCoordinates(self, *args):
            raise RuntimeError("stub body: no ephemeris")

        @JOverride
        def getPosition(self, *args):
            raise RuntimeError("stub body: no ephemeris")

    @JImplements("org.orekit.bodies.CelestialBodyLoader")
    class StubLoader:
        @JOverride
        def loadCelestialBody(self, name):
            return StubBody(name)

    for body in ("Moon", "Earth"):
        context.getCelestialBodies().addCelestialBodyLoader(body, StubLoader())
    return ParserBuilder(context), context


def _orekit_et(date, context):
    """Orekit date -> TDB seconds past J2000.0 measured on Orekit's own TDB scale."""
    from org.orekit.time import AbsoluteDate

    tdb = context.getTimeScales().getTDB()
    return float(date.offsetFrom(AbsoluteDate(2000, 1, 1, 12, 0, 0.0, tdb), tdb))


def test_orekit_reads_oem_positions_identically(tmp_path):
    builder, context = _orekit_parser_builder(tmp_path)
    from org.orekit.data import DataSource

    text, epochs, states, cov = _oem_text()
    path = tmp_path / "orekit.oem"
    path.write_text(text, encoding="ascii")
    oem = builder.buildOemParser().parseMessage(DataSource(str(path)))
    seg = oem.getSegments().get(0)
    meta = seg.getMetadata()
    assert str(meta.getTimeSystem()) == "TDB"
    assert str(meta.getCenter().getName()).upper() == "MOON"
    assert str(meta.getReferenceFrame().getName()) == "EME2000"
    lines = list(seg.getData().getEphemeridesDataLines())
    ours = odm.read_oem(text).segments[0]
    pos = np.array([[p.getPosition().getX(), p.getPosition().getY(), p.getPosition().getZ()]
                    for p in lines])
    vel = np.array([[p.getVelocity().getX(), p.getVelocity().getY(), p.getVelocity().getZ()]
                    for p in lines])
    # Same km digits, same x1000 scaling: position and velocity difference exactly 0.
    np.testing.assert_array_equal(pos, ours.states_m[:, :3])
    np.testing.assert_array_equal(vel, ours.states_m[:, 3:])
    et = np.array([_orekit_et(p.getDate(), context) for p in lines])
    assert np.all(np.abs(et - epochs) <= 0.5e-6 + 1e-9)
    covs = list(seg.getData().getCovarianceMatrices())
    assert len(covs) == 2
    m = covs[0].getCovarianceMatrix()
    orekit_cov = np.array([[m.getEntry(i, j) for j in range(6)] for i in range(6)])
    np.testing.assert_allclose(orekit_cov, cov, rtol=1e-15, atol=0)


def test_orekit_reads_opm_state_and_covariance(tmp_path):
    builder, context = _orekit_parser_builder(tmp_path)
    from org.orekit.data import DataSource

    _, states = _trajectory(n=1)
    cov = _covariance()
    text = odm.format_opm(ET0, states[0], object_name="S", object_id="ID", originator="O",
                          covariance_m=cov, creation_date=CREATED)
    path = tmp_path / "orekit.opm"
    path.write_text(text, encoding="ascii")
    opm = builder.buildOpmParser().parseMessage(DataSource(str(path)))
    pv = opm.getData().getStateVectorBlock().toTimeStampedPVCoordinates()
    pos = np.array([pv.getPosition().getX(), pv.getPosition().getY(), pv.getPosition().getZ()])
    np.testing.assert_array_equal(pos, odm.read_opm(text).state_m[:3])
    assert abs(_orekit_et(pv.getDate(), context) - ET0) <= 0.5e-6 + 1e-9
    m = opm.getData().getCovarianceBlock().getCovarianceMatrix()
    orekit_cov = np.array([[m.getEntry(i, j) for j in range(6)] for i in range(6)])
    np.testing.assert_allclose(orekit_cov, cov, rtol=1e-15, atol=0)


def test_export_example_demo_writes_readable_messages(tmp_path):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "examples" / "export_ccsds_odm.py"
    spec = importlib.util.spec_from_file_location("export_ccsds_odm", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--demo", "--output-dir", str(tmp_path), "--creation-date", CREATED]) == 0
    oem = odm.read_oem(tmp_path / "DEMO.oem")
    opm = odm.read_opm(tmp_path / "DEMO.opm")
    seg = oem.segments[0]
    assert seg.states_m.shape == (61, 6) and len(seg.covariances) == 7
    np.testing.assert_array_equal(opm.state_km, oem.states_km[0][0])
    # Circular orbit: radius constant to rounding.
    r = np.linalg.norm(seg.states_m[:, :3], axis=1)
    np.testing.assert_allclose(r, r[0], rtol=1e-14)
