"""F2 WP2.4 slice 1: Earth-orientation profiles and provenance (gates G1-G6 of
studies/phase17/configs/wp2_4_predeclaration.json).

The oracle for every rotation is SPICE itself with a single Earth PCK furnished (plus the LSK and the ITRF93
association FK): the profile must reproduce exactly the kernel SPICE is expected to give priority to.
"""
from __future__ import annotations

import numpy as np
import pytest

spice = pytest.importorskip("spiceypy")

from lunar_od import spice_loader as SL  # noqa: E402
from lunar_od import station_frames as SF  # noqa: E402

try:
    KERNEL_DIR = SL.resolve_kernel_dir()
    SL.required_kernel_paths(KERNEL_DIR, earth_orientation="MEASURED")
except FileNotFoundError as exc:  # pragma: no cover - environment without kernels
    pytest.skip(f"SPICE kernels unavailable: {exc}", allow_module_level=True)

#: Previous production kernel list (before WP2.4): the predict Earth PCK alone.
PREVIOUS = tuple(k for k in SL.required_kernels() if k != SL.EARTH_HIGH_PRECISION_PCK)
FRAME_BASE = ("naif0012.tls.txt", "earth_assoc_itrf93.tf.txt")


@pytest.fixture(autouse=True)
def _restore_default_pool():
    yield
    SL.load_spice_kernels()


def _furnish(names):
    spice.kclear()
    for name in names:
        spice.furnsh(str(KERNEL_DIR / name))


def _rotations(utc_dates):
    ets = [float(spice.str2et(d)) for d in utc_dates]
    return np.array([spice.sxform("ITRF93", "J2000", et) for et in ets])


def _monthly(first_year, first_month, count):
    out = []
    y, m = first_year, first_month
    for _ in range(count):
        out.append(f"{y:04d}-{m:02d}-02T00:00:00")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def test_profiles_and_selection(monkeypatch):
    """G6: argument > environment > default; unknown profile raises; every kernel is in the manifest."""
    from lunar_od.kernel_manifest import KERNEL_MANIFEST

    pred, meas = SL.required_kernels(earth_orientation="PREDICT"), SL.required_kernels(earth_orientation="MEASURED")
    assert pred == SL.required_kernels()
    i = PREVIOUS.index(SL.EARTH_PREDICT_PCK)
    assert pred == PREVIOUS[:i] + (SL.EARTH_HIGH_PRECISION_PCK,) + PREVIOUS[i:]
    assert meas == PREVIOUS[:i + 1] + (SL.EARTH_HIGH_PRECISION_PCK,) + PREVIOUS[i + 1:]
    monkeypatch.setenv(SL.EARTH_ORIENTATION_ENV, "measured")
    assert SL.required_kernels() == meas
    assert SL.required_kernels(earth_orientation="PREDICT") == pred
    with pytest.raises(ValueError, match="Earth-orientation profile"):
        SL.required_kernels(earth_orientation="IERS")
    for eph in SL.PLANETARY_EPHEMERIS_KERNELS:
        for profile in SL.EARTH_ORIENTATION_KERNELS:
            assert set(SL.required_kernels(eph, profile)) <= set(KERNEL_MANIFEST)


def test_default_profile_is_unchanged_where_the_predict_kernel_covers():
    """G1: 30 monthly epochs 2025-01..2027-06, bitwise equal to the previous kernel list."""
    dates = _monthly(2025, 1, 30)
    _furnish(PREVIOUS)
    before = _rotations(dates)
    SL.load_spice_kernels()
    assert np.array_equal(_rotations(dates), before)


def test_default_profile_gains_measured_pre_2025_orientation():
    """G2: pre-2025 epochs resolve under the default profile, bitwise equal to the high-precision kernel alone."""
    dates = ["2000-01-02T00:00:00", "2009-09-20T00:00:00", "2010-09-20T00:00:00", "2015-06-01T00:00:00",
             "2020-06-01T00:00:00", "2024-12-31T00:00:00"]
    _furnish(PREVIOUS)
    with pytest.raises(Exception):
        _rotations(dates[:1])
    _furnish(FRAME_BASE + (SL.EARTH_HIGH_PRECISION_PCK,))
    oracle = _rotations(dates)
    SL.load_spice_kernels()
    assert np.array_equal(_rotations(dates), oracle)


def test_measured_profile_gives_the_high_precision_kernel_priority():
    """G3: inside [2000-01-02, 2026-04-18] = high-precision kernel alone; 2026-04-20..2027-06 = predict alone."""
    inside = ["2000-01-02T00:00:00", "2010-09-20T00:00:00", "2025-06-01T00:00:00", "2025-10-01T00:00:00",
              "2026-01-21T00:00:00", "2026-03-01T00:00:00", "2026-04-18T00:00:00"]
    after = ["2026-04-20T00:00:00"] + _monthly(2026, 6, 13)
    _furnish(FRAME_BASE + (SL.EARTH_HIGH_PRECISION_PCK,))
    oracle_inside = _rotations(inside)
    _furnish(FRAME_BASE + (SL.EARTH_PREDICT_PCK,))
    oracle_after = _rotations(after)
    SL.load_spice_kernels(earth_orientation="MEASURED")
    assert np.array_equal(_rotations(inside), oracle_inside)
    assert np.array_equal(_rotations(after), oracle_after)


def _prov(utc):
    return SF.earth_orientation_provenance(float(spice.str2et(utc)))


def test_provenance_follows_the_kernel_spice_uses():
    """G4: last datum from the comments; kernel and measured flag per profile and epoch."""
    hp, pr = SL.EARTH_HIGH_PRECISION_PCK, SL.EARTH_PREDICT_PCK
    SL.load_spice_kernels()
    expected = {"2020-06-01T00:00:00": (hp, True), "2025-06-01T00:00:00": (pr, True),
                "2025-10-01T00:00:00": (pr, False)}
    for utc, (kernel, measured) in expected.items():
        p = _prov(utc)
        assert (p["kernel"], p["measured"]) == (kernel, measured), (utc, p)
    assert _prov("2025-06-01T00:00:00")["last_datum_utc"] == "2025 AUG 26 00:00:00"
    assert _prov("2020-06-01T00:00:00")["last_datum_utc"] == "2026 JAN 21 00:00:00"
    assert _prov("1990-01-01T00:00:00")["kernel"] is None
    SL.load_spice_kernels(earth_orientation="MEASURED")
    expected = {"2025-10-01T00:00:00": (hp, True), "2026-03-01T00:00:00": (hp, False),
                "2026-06-01T00:00:00": (pr, False)}
    for utc, (kernel, measured) in expected.items():
        p = _prov(utc)
        assert (p["kernel"], p["measured"]) == (kernel, measured), (utc, p)


def test_measured_orientation_requirement_fails_closed():
    """G5: predicted or uncovered epochs raise naming the epoch and kernel; measured ones pass; the old flag holds."""
    et = lambda utc: float(spice.str2et(utc))  # noqa: E731
    SL.load_spice_kernels()
    SF.require_measured_earth_orientation([et("2010-09-20T00:00:00"), et("2025-06-01T00:00:00")])
    with pytest.raises(SF.PredictedEarthOrientationError, match=SL.EARTH_PREDICT_PCK) as caught:
        SF.require_measured_earth_orientation([et("2025-06-01T00:00:00"), et("2025-10-01T00:00:00")])
    assert "2025-10-01" in str(caught.value)
    assert isinstance(caught.value, ValueError)
    with pytest.raises(SF.PredictedEarthOrientationError, match="no loaded Earth PCK"):
        SF.require_measured_earth_orientation(et("1990-01-01T00:00:00"))
    assert SF.is_predicted_earth_orientation(et("2027-03-08T00:00:00"))
    assert not SF.is_predicted_earth_orientation(et("2025-03-01T00:00:00"))
    SL.load_spice_kernels(earth_orientation="MEASURED")
    SF.require_measured_earth_orientation(et("2025-10-01T00:00:00"))
    with pytest.raises(SF.PredictedEarthOrientationError, match=SL.EARTH_HIGH_PRECISION_PCK):
        SF.require_measured_earth_orientation(et("2026-03-01T00:00:00"))
