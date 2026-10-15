"""Phase 17-HARDEN mutants A-F and the legacy-leak guard.

Each mutant re-introduces one defect class that Phase 17 actually shipped (or
could have shipped).  For each, a test asserts that the current code REJECTS
it, and a second assertion shows that the defect is physically real, i.e. that the check
would have fired on the original defect ("would it have failed before the
repair?  YES").

  A  Earth epoch offset of -504000 s (Phase 17-FXR)
  B  Earth table clamp at the end of coverage (Phase 17-FXR)
  C  ΔDOR / surrogate rows behind the Moon (Phase 17-R1O-D2)
  D  STM unpacked in C order (Phase 17-R1O); see also test_r1o_stm_layout.py
  E  wrong frame / centre metadata on an Earth provider
  F  wrong units (km) on an Earth provider
"""
from __future__ import annotations

import dataclasses
import inspect
import runpy
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "studies" / "phase17" / "scripts"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import phase17_r1m_core as M  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (M.CAMPAIGN_ROOT / "08_manifests").exists(),
    reason="read-only campaign data not available")

EPOCH_OFFSET_S = -504000.0                 # the FXR defect: phase7 ET0 - 504000 s
GC_OVERLAP_T = 7200.0 + 600.0 * np.arange(9)   # Goldstone+Canberra both see the Moon


@pytest.fixture(scope="module")
def spice():
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels(None, clear=True)
    et0, _ = M.campaign_epoch()
    return et0


@pytest.fixture(scope="module")
def stations():
    import od_gravity_covariance_campaign as C  # noqa: N811
    return {s.name: s for s in C.STATIONS}


def _spice_earth(et):
    import spiceypy as sp
    return np.array([np.asarray(sp.spkpos("EARTH", float(e), "J2000", "NONE", "MOON")[0]) * 1e3
                     for e in np.atleast_1d(et)])


def _earth_matches_measurement_epoch(pos_fn, et0, t, tol_m=1.0):
    """Physical oracle: Earth handed to the model == SPICE at et0 + t."""
    return float(np.max(np.linalg.norm(pos_fn(t) - _spice_earth(et0 + t), axis=1))) <= tol_m


def _small_arc(et0):
    t = np.arange(0.0, 601.0, 60.0)
    earth = M.EarthEphemeris(M.EARTH_FIXTURE_CANONICAL, et0, 0.0, 600.0)
    u = earth.pos(t) / np.linalg.norm(earth.pos(t), axis=1)[:, None]
    x = np.hstack([u * 1.838e6, np.zeros((t.size, 3))])
    return t, x


# ======================================================================
# A - Earth epoch offset
# ======================================================================
def test_mutant_a_epoch_offset_rejected_by_production_generator(spice, stations):
    from lunar_od.ephemeris import EphemerisEpochMismatchError
    from lunar_od.measurement_contracts import (
        generate_two_way_range_measurements_checked as generate_two_way_range_measurements)
    et0 = spice
    t, x = _small_arc(et0)
    st = (stations["Goldstone DSN"],)
    mutant = M.EarthEphemeris(M.EARTH_FIXTURE_CANONICAL, et0 + EPOCH_OFFSET_S, 0.0, 600.0)
    with pytest.raises(EphemerisEpochMismatchError, match="504000"):
        generate_two_way_range_measurements(t, x, st, np.ones((t.size, 1), bool),
                                            mutant.pos, mutant.vel, et0, noise=False)
    good = M.EarthEphemeris(M.EARTH_FIXTURE_CANONICAL, et0, 0.0, 600.0)
    generate_two_way_range_measurements(t, x, st, np.ones((t.size, 1), bool),
                                        good.pos, good.vel, et0, noise=False)


def test_mutant_a_would_have_failed_before_the_repair(spice):
    """YES: the offset Earth is physically wrong by far more than the oracle tolerance,
    and the pre-repair path had no epoch check.  (The byte-frozen generator itself still
    has none; the contract lives in lunar_od.measurement_contracts, which every
    production / Phase-17 caller must use -- see the structural test below.  An
    untagged callable is still accepted there, which is why every Phase-17 science
    builder passes TAGGED providers.)"""
    from lunar_od.ephemeris import require_ephemeris_contract
    et0 = spice
    t = np.array([0.0, 300.0, 600.0])
    mutant = M.EarthEphemeris(M.EARTH_FIXTURE_CANONICAL, et0 + EPOCH_OFFSET_S, 0.0, 600.0)
    assert not _earth_matches_measurement_epoch(mutant.pos, et0, t, tol_m=1e7)
    assert require_ephemeris_contract(lambda tt: mutant.pos(tt), et0) == "UNTAGGED"
    assert _earth_matches_measurement_epoch(
        M.EarthEphemeris(M.EARTH_FIXTURE_CANONICAL, et0, 0.0, 600.0).pos, et0, t)


def test_legacy_offset_fixture_is_flagged_not_silently_accepted(spice):
    from lunar_od.ephemeris import LegacyEphemerisWarning, require_ephemeris_contract
    et0 = spice
    with pytest.warns(M.SupersededEarthFixtureWarning):
        legacy = M.measurement_earth_ephemeris(M.EARTH_FIXTURE_LEGACY_OFFSET_CLAMPED, et0, 3600.0)
    assert legacy.legacy_reproduction_only
    with pytest.warns(LegacyEphemerisWarning):
        assert require_ephemeris_contract(legacy.pos, et0) == "LEGACY"
    assert not _earth_matches_measurement_epoch(legacy.pos, et0, np.array([0.0, 1800.0]), 1e7)


# ======================================================================
# B - clamp at the end of coverage
# ======================================================================
def test_mutant_b_clamp_replaced_by_coverage_error(spice):
    from lunar_od.ephemeris import sample_moon_centered_ephemeris, EphemerisCoverageError
    et0 = spice
    canon = M.EarthEphemeris(M.EARTH_FIXTURE_CANONICAL, et0, 0.0, 3600.0)
    canon.pos(np.array([3600.0]))
    with pytest.raises(M.EarthEphemerisCoverageError):
        canon.pos(np.array([3600.0 + 1e-6]))
    eph = sample_moon_centered_ephemeris(et0, np.arange(0.0, 3601.0, 60.0))
    eph.earth_position(3600.0 + 0.5 * eph.extrapolation_margin_s)
    for bad in (-2.0 * eph.extrapolation_margin_s, 3600.0 + 2.0 * eph.extrapolation_margin_s):
        with pytest.raises(EphemerisCoverageError):
            eph.earth_position(bad)
        with pytest.raises(EphemerisCoverageError):
            eph.sun_position(bad)
        with pytest.raises(EphemerisCoverageError):
            eph.earth_velocity(bad)


def test_mutant_b_would_have_failed_before_the_repair(spice):
    """YES: a clamped table freezes the Earth; the SPICE oracle sees it move ~1 km/s."""
    et0 = spice
    with pytest.warns(M.SupersededEarthFixtureWarning):
        legacy = M.measurement_earth_ephemeris(M.EARTH_FIXTURE_LEGACY_OFFSET_CLAMPED, et0, 60000.0)
    c = legacy.clamp_after_s
    assert np.isfinite(c)
    frozen = legacy.pos(np.array([c, c + 3600.0]))
    assert np.array_equal(frozen[0], frozen[1])
    moved = _spice_earth(et0 + np.array([c, c + 3600.0]))
    assert np.linalg.norm(moved[1] - moved[0]) > 1e5


# ======================================================================
# C - lunar occultation of ΔDOR / surrogate rows
# ======================================================================
def _gc_nom48(side, et0):
    t = GC_OVERLAP_T
    earth = M.earth_position_mci(t, M.EARTH_FIXTURE_CANONICAL)
    u = earth / np.linalg.norm(earth, axis=1)[:, None]
    from lunar_od.constants import R_MOON_M
    pos = side * u * (R_MOON_M + 100e3)
    stm = np.tile(np.eye(6).reshape(-1, order="F"), (t.size, 1))
    return t, np.hstack([pos, np.zeros((t.size, 3)), stm, np.zeros((t.size, 6))]), earth


def _builder_row_counts(rule, side, et0, stations):
    from lunar_od.delta_dor import QuasarDirection
    from phase17_r1o_core import build_ddor_arc
    import phase17_r1od_campaign as OD
    from phase17_r1o_r_requalification import earth_los_rows
    t, nom48, _ = _gc_nom48(side, et0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", M.SupersededVisibilityWarning)
        arc = build_ddor_arc(nom48, t, et0, ("Goldstone DSN", "Canberra DSN"),
                             sigma_angle_rad=5e-9, row_rule=rule)
        od = OD.build_ddor_rows(SimpleNamespace(t_grid=t, nom48=nom48),
                                stations["Goldstone DSN"], stations["Canberra DSN"],
                                QuasarDirection(2.1, -0.3), row_rule=rule)
        los, _, _ = earth_los_rows(nom48, t, row_rule=rule)
    return {"r1o_surrogate": arc.h_k.size, "r1od_production": od["n_dual_visible"],
            "earth_los": los.shape[0]}


def test_mutant_c_occulted_rows_rejected_by_every_builder(spice, stations):
    far = _builder_row_counts(M.DDOR_ROW_RULE_VISIBLE, -1.0, spice, stations)
    assert far == {"r1o_surrogate": 0, "r1od_production": 0, "earth_los": 0}
    near = _builder_row_counts(M.DDOR_ROW_RULE_VISIBLE, +1.0, spice, stations)
    assert min(near.values()) > 0            # the geometry is not vacuous


def test_mutant_c_would_have_caught_the_original_defect(spice, stations):
    """YES: the elevation-only rule (the code R1O / R1O-D / FXR ran) forms rows
    for a spacecraft that is behind the Moon from both stations."""
    far = _builder_row_counts(M.DDOR_ROW_RULE_LEGACY_ELEVATION_ONLY, -1.0, spice, stations)
    assert min(far.values()) > 0


def test_mutant_c_zero_moon_radius_is_detected(spice, stations, monkeypatch):
    """Disabling the occultation test in the SHARED contract makes the far side
    visible, so the check above is driven by the occultation test and by nothing else."""
    import lunar_od.constants as K
    from lunar_od.visibility import sample_j2000_to_itrf93_transforms
    t, nom48, earth = _gc_nom48(-1.0, spice)
    xf = sample_j2000_to_itrf93_transforms(spice, t)
    st = (stations["Goldstone DSN"], stations["Canberra DSN"])
    assert not M.station_visibility_mask(t, nom48[:, :3], st, earth, xf).any()
    monkeypatch.setattr(K, "R_MOON_M", 0.0)
    assert M.station_visibility_mask(t, nom48[:, :3], st, earth, xf).all()


# ======================================================================
# D - STM order (the full FD proof is tests/test_r1o_stm_layout.py)
# ======================================================================
def test_mutant_d_stm_c_order_is_detected():
    from lunar_od.constants import AU_M, MU_MOON_M3S2
    from lunar_od.dynamics import propagate_state, propagate_state_with_k_sensitivity
    from lunar_od.srp import SRPOptions
    srp = SRPOptions(k_srp_m2_per_kg=0.01)
    x0 = np.array([1.8377e6, 0.0, 2.0e5, 0.0, 1.5337e3, 4.0e2])
    t = np.array([0.0, 600.0])
    earth = lambda _t: np.array([3.844e8, 0.0, 0.0])          # noqa: E731
    sun = lambda _t: np.array([-AU_M, 0.0, 0.0])              # noqa: E731
    nom = propagate_state_with_k_sensitivity(t, x0, MU_MOON_M3S2, 0.0, 0.0, earth, sun,
                                             srp=srp, rtol=1e-12, atol=1e-13)
    fd = np.zeros((3, 6))
    for j, h in enumerate((1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3)):
        e = np.zeros(6); e[j] = h
        fd[:, j] = (propagate_state(t, x0 + e, MU_MOON_M3S2, 0.0, 0.0, earth, sun,
                                    srp=srp, rtol=1e-12, atol=1e-13)[-1, :3]
                    - propagate_state(t, x0 - e, MU_MOON_M3S2, 0.0, 0.0, earth, sun,
                                      srp=srp, rtol=1e-12, atol=1e-13)[-1, :3]) / (2 * h)
    phi_f = nom[-1, 6:42].reshape((6, 6), order="F")
    phi_c = nom[-1, 6:42].reshape(6, 6)
    rel = lambda a: np.max(np.abs(a[:3] - fd)) / np.max(np.abs(fd))  # noqa: E731
    assert rel(phi_f) < 1e-6
    assert rel(phi_c) > 1e-2                  # YES: the transposed STM fails FD


# ======================================================================
# E / F - frame, centre and unit metadata
# ======================================================================
@pytest.mark.parametrize("field,value,match", [
    ("frame", "ECLIPJ2000", "frame"),
    ("center", "EARTH", "centre"),
    ("units", "km, km/s", "units"),
])
def test_mutants_e_f_metadata_mismatch_rejected(spice, field, value, match):
    from lunar_od.ephemeris import EphemerisContractError, require_ephemeris_contract
    good = M.EarthEphemeris(M.EARTH_FIXTURE_CANONICAL, spice, 0.0, 600.0)
    assert require_ephemeris_contract(good.pos, spice) == "MATCHED"
    bad = dataclasses.replace(good, **{field: value})
    with pytest.raises(EphemerisContractError, match=match):
        require_ephemeris_contract(bad.pos, spice)


def test_mutant_f_km_provider_rejected_even_untagged(spice, stations):
    from lunar_od.ephemeris import EphemerisContractError
    from lunar_od.measurement_contracts import (
        generate_two_way_range_measurements_checked as generate_two_way_range_measurements)
    et0 = spice
    t, x = _small_arc(et0)
    good = M.EarthEphemeris(M.EARTH_FIXTURE_CANONICAL, et0, 0.0, 600.0)
    with pytest.raises(EphemerisContractError, match="Moon-Earth distance"):
        generate_two_way_range_measurements(
            t, x, (stations["Goldstone DSN"],), np.ones((t.size, 1), bool),
            lambda tt: good.pos(tt) / 1e3, lambda tt: good.vel(tt) / 1e3, et0, noise=False)


# ======================================================================
# Legacy leak (negative test)
# ======================================================================
GATED_SCRIPTS = (
    "phase17_r1o_comparison", "phase17_r1o_confirmation", "phase17_r1o_crosscheck",
    "phase17_r1o_ddor", "phase17_r1od_surrogate_bridge", "phase17_r1o_r_requalification",
    "phase17_r1od_anchor_convention_check", "phase17_r1od_jacobian_and_k",
    "phase17_r1o_celestial", "phase17_r1o_baseline", "phase17_r1od_campaign",
)


@pytest.mark.parametrize("script", GATED_SCRIPTS)
def test_historical_scripts_refuse_without_explicit_opt_in(script, monkeypatch):
    monkeypatch.delenv(M.LEGACY_REPRODUCTION_ENV, raising=False)
    with pytest.raises(M.LegacyReproductionRequired):
        runpy.run_path(str(EXAMPLES / (script + ".py")), run_name="__main__")


def test_every_row_builder_defaults_to_the_visibility_contract():
    from phase17_r1o_core import build_ddor_arc
    import phase17_r1od_campaign as OD
    from phase17_r1o_r_requalification import earth_los_rows
    for fn in (build_ddor_arc, OD.build_ddor_rows, earth_los_rows):
        assert inspect.signature(fn).parameters["row_rule"].default == M.DDOR_ROW_RULE_VISIBLE


def test_legacy_rule_always_warns_and_unknown_rule_raises():
    with pytest.warns(M.SupersededVisibilityWarning):
        M.require_row_rule(M.DDOR_ROW_RULE_LEGACY_ELEVATION_ONLY)
    with pytest.raises(ValueError):
        M.require_row_rule("ELEVATION_ONLY")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        M.require_row_rule(M.DDOR_ROW_RULE_VISIBLE)


def test_row_rule_names_are_single_sourced():
    import phase17_r1od2_core as D2
    import phase17_fxr_requalification as FXR
    assert D2.ROW_RULE_PRODUCTION_VISIBILITY == M.DDOR_ROW_RULE_VISIBLE
    assert D2.ROW_RULE_ELEVATION_ONLY == M.DDOR_ROW_RULE_LEGACY_ELEVATION_ONLY
    assert FXR._LEGACY_RULE == M.DDOR_ROW_RULE_LEGACY_ELEVATION_ONLY


def test_science_earth_provider_is_tagged_and_matched(spice):
    from lunar_od.ephemeris import require_ephemeris_contract
    earth = M.measurement_earth_ephemeris(M.EARTH_FIXTURE_CANONICAL, spice, 3600.0)
    assert require_ephemeris_contract(earth.pos, spice) == "MATCHED"
    assert require_ephemeris_contract(earth.vel, spice) == "MATCHED"


def test_archived_orphans_are_not_importable():
    assert not (EXAMPLES / "phase17_r1od_fd_convergence.py").exists()
    assert not (EXAMPLES / "phase17_r1od_qualification.py").exists()
    manifest = (ROOT / "archive" / "phase17_orphans" / "MANIFEST.sha256").read_text(encoding="utf-8")
    assert "f48fd6de0254ab55c31c0c02626a1a8e3ce70363d1234108da0d99967d21bbb6" in manifest
    assert "ec36d2327ba7ec56a9f9caa3116815e7e6d60d237bd0286956b83f980e04f13f" in manifest


def test_no_direct_calls_to_the_unchecked_range_generator():
    """Structural guard: the frozen generator is reached only through the contract layer.
    AST-based, so only real call sites count (not strings or comments)."""
    import ast
    offenders = []
    for path in sorted(EXAMPLES.glob("phase17_*.py")) + [ROOT / "src" / "lunar_od" / "scenarios.py"]:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                f = node.func
                name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
                if name == "generate_two_way_range_measurements":
                    offenders.append("%s:%d" % (path.name, node.lineno))
    assert offenders == []


def test_frozen_range_module_is_untouched_by_harden():
    import hashlib
    frozen = "bddff533c4a1773ba80dd8f96cbb6933baa516b40c0e71f82bb56c927880a995"
    assert hashlib.sha256((ROOT / "src" / "lunar_od" / "two_way_range.py").read_bytes()).hexdigest() == frozen
