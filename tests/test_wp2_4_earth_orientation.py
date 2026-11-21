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


def _kernel_facts(name):
    """(coverage start ET, coverage end ET, last datum ET, last datum UTC string) read from the kernel itself, so
    the gates follow the pinned high-precision release (amendment 2026-11-03: kernel update)."""
    spice.kclear()
    spice.furnsh(str(KERNEL_DIR / "naif0012.tls.txt"))
    path = str(KERNEL_DIR / name)
    cover = spice.cell_double(2000)
    spice.pckcov(path, SF.EARTH_ITRF93_PCK_BODY, cover)
    start, end = spice.wnfetd(cover, 0)[0], spice.wnfetd(cover, spice.wncard(cover) - 1)[1]
    utc = SF._last_datum_utc(path)
    return start, end, float(spice.str2et(utc + " UTC")), utc


HP_START, HP_END, HP_LAST, HP_LAST_UTC = _kernel_facts(SL.EARTH_HIGH_PRECISION_PCK)
PR_START, _, PR_LAST, PR_LAST_UTC = _kernel_facts(SL.EARTH_PREDICT_PCK)
DAY = 86400.0


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
    """G3: inside the high-precision coverage = that kernel alone; after its end = the predict kernel alone."""
    spice.kclear()
    spice.furnsh(str(KERNEL_DIR / "naif0012.tls.txt"))
    iso = lambda et: spice.et2utc(et, "ISOC", 0)  # noqa: E731
    inside = ["2000-01-02T00:00:00", "2010-09-20T00:00:00", "2025-06-01T00:00:00", "2025-10-01T00:00:00",
              iso(HP_LAST), iso(HP_LAST + 30 * DAY), iso(HP_END - DAY)]
    after = [iso(HP_END + DAY * k) for k in (1, 30, 90, 180, 365)]
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
    assert PR_LAST_UTC == "2025 AUG 26 00:00:00"
    SL.load_spice_kernels()
    iso = lambda et: spice.et2utc(et, "ISOC", 0)  # noqa: E731
    expected = {"2020-06-01T00:00:00": (hp, True), iso(PR_LAST - DAY): (pr, True), iso(PR_LAST + DAY): (pr, False)}
    for utc, (kernel, measured) in expected.items():
        p = _prov(utc)
        assert (p["kernel"], p["measured"]) == (kernel, measured), (utc, p)
    assert _prov(iso(PR_LAST - DAY))["last_datum_utc"] == PR_LAST_UTC
    assert _prov("2020-06-01T00:00:00")["last_datum_utc"] == HP_LAST_UTC
    assert _prov("1990-01-01T00:00:00")["kernel"] is None
    SL.load_spice_kernels(earth_orientation="MEASURED")
    expected = {iso(PR_LAST + DAY): (hp, True), iso(HP_LAST - DAY): (hp, True), iso(HP_LAST + DAY): (hp, False),
                iso(HP_END + DAY): (pr, False)}
    for utc, (kernel, measured) in expected.items():
        p = _prov(utc)
        assert (p["kernel"], p["measured"]) == (kernel, measured), (utc, p)


def test_measured_orientation_requirement_fails_closed():
    """G5: predicted or uncovered epochs raise naming the epoch and kernel; measured ones pass; the old flag holds."""
    et = lambda utc: float(spice.str2et(utc))  # noqa: E731
    SL.load_spice_kernels()
    SF.require_measured_earth_orientation([et("2010-09-20T00:00:00"), PR_LAST - DAY])
    with pytest.raises(SF.PredictedEarthOrientationError, match=SL.EARTH_PREDICT_PCK) as caught:
        SF.require_measured_earth_orientation([PR_LAST - DAY, et("2025-10-01T00:00:00")])
    assert "2025-10-01" in str(caught.value)
    assert isinstance(caught.value, ValueError)
    with pytest.raises(SF.PredictedEarthOrientationError, match="no loaded Earth PCK"):
        SF.require_measured_earth_orientation(et("1990-01-01T00:00:00"))
    assert SF.is_predicted_earth_orientation(et("2027-03-08T00:00:00"))
    assert not SF.is_predicted_earth_orientation(et("2025-03-01T00:00:00"))
    SL.load_spice_kernels(earth_orientation="MEASURED")
    SF.require_measured_earth_orientation([et("2025-10-01T00:00:00"), HP_LAST - DAY])
    with pytest.raises(SF.PredictedEarthOrientationError, match=SL.EARTH_HIGH_PRECISION_PCK):
        SF.require_measured_earth_orientation(HP_LAST + DAY)


# ---------------------------------------------------------------------------
# Slice 2: ScenarioConfig.earth_orientation, scenario runner and run provenance (G8-G11).
# ---------------------------------------------------------------------------
def test_scenario_earth_orientation_field(tmp_path):
    """G8: default None; allowed values; unknown raises; JSON round trip; schema; archive registry; identity."""
    import json

    from lunar_od.scenario_config import (
        SCENARIO_CONFIG_FIELDS_ADDED_AFTER_ARCHIVES,
        load_scenario_config_json,
        scenario_config_from_mapping,
        scenario_config_schema,
        scenario_earth_orientation_profile,
        scenario_measurement_identity,
        write_normalized_scenario_config,
    )

    base = {"name": "eo", "measurement_type": "position", "estimator_type": "bls_lm", "start_mode": "cold",
            "network": "single"}
    default = scenario_config_from_mapping(base)
    assert default.earth_orientation is None and scenario_earth_orientation_profile(default) is None
    assert "earth_orientation" not in scenario_measurement_identity(default)
    profiles = {"predict": "PREDICT", "measured": "MEASURED", "measured_only": "MEASURED"}
    for value, profile in profiles.items():
        cfg = scenario_config_from_mapping({**base, "earth_orientation": value})
        assert scenario_earth_orientation_profile(cfg) == profile
        assert scenario_measurement_identity(cfg)["earth_orientation"] == value
        path = write_normalized_scenario_config(cfg, tmp_path / f"{value}.json")
        assert load_scenario_config_json(path) == cfg
        assert json.loads(path.read_text(encoding="utf-8"))["earth_orientation"] == value
    with pytest.raises(ValueError, match="earth_orientation"):
        scenario_config_from_mapping({**base, "earth_orientation": "MEASURED"})
    assert scenario_config_schema()["properties"]["earth_orientation"]["enum"] == [None, *profiles]
    assert SCENARIO_CONFIG_FIELDS_ADDED_AFTER_ARCHIVES["earth_orientation"] is None


def _e2e(tmp_path, name, **fields):
    from tests.test_scenario_reproducibility_wp1_6 import _run_cli, _runner

    return _run_cli(_runner(), tmp_path, name, noise=True, measurement_seed=20261103, cold_start_seed=5, **fields)


def test_scenario_profiles_are_bitwise_identical_at_the_fixture_epoch(tmp_path):
    """G9 + G11: unset / 'predict' / 'measured' give identical results at 2027 (the predict kernel serves there);
    the provenance records the profile and the serving kernel at both span ends."""
    from tests.test_scenario_reproducibility_wp1_6 import _RESULT_COLUMNS

    row_u, prov_u = _e2e(tmp_path, "unset")
    row_p, prov_p = _e2e(tmp_path, "predict", earth_orientation="predict")
    row_m, prov_m = _e2e(tmp_path, "measured", earth_orientation="measured")
    for key in _RESULT_COLUMNS:
        assert row_u[key] == row_p[key] == row_m[key], key
    files = [k["file"] for k in prov_m["spice_kernels"]]
    assert files.index(SL.EARTH_PREDICT_PCK) < files.index(SL.EARTH_HIGH_PRECISION_PCK)
    files = [k["file"] for k in prov_u["spice_kernels"]]
    assert files.index(SL.EARTH_HIGH_PRECISION_PCK) < files.index(SL.EARTH_PREDICT_PCK)
    for prov, profile, policy in ((prov_u, "PREDICT", None), (prov_p, "PREDICT", "predict"),
                                  (prov_m, "MEASURED", "measured")):
        block = prov["earth_orientation"]
        assert (block["profile"], block["policy"]) == (profile, policy)
        assert len(block["epochs"]) == 2
        for epoch in block["epochs"]:
            assert epoch["kernel"] == SL.EARTH_PREDICT_PCK
            assert epoch["last_datum_utc"] == "2025 AUG 26 00:00:00"
            assert epoch["measured"] is False


def test_scenario_measured_only_fails_closed_at_a_predicted_epoch(tmp_path):
    """G10: 'measured_only' at the 2027 fixture epoch raises before propagation, naming the predict kernel."""
    from unittest import mock

    from tests.test_scenario_reproducibility_wp1_6 import _runner, _write

    runner = _runner()
    out = tmp_path / "mo"
    out.mkdir()
    cfg = _write(out, name="eo_mo", duration_h=1.0, sample_step_s=600.0, max_iter=3, output_dir=str(out),
                 earth_orientation="measured_only")
    with mock.patch.object(runner, "propagate_truth_with_ephemeris", side_effect=AssertionError("propagated")):
        with pytest.raises(SF.PredictedEarthOrientationError, match=SL.EARTH_PREDICT_PCK):
            runner.main([str(cfg)])
