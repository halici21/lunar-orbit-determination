"""WP7.5 / TST-06, TST-07: the Earth-provider contract without kernels or campaign data.

``test_harden_mutants`` exercises the same contract against the Phase 17 campaign
fixtures and is skipped wherever those are absent (CI).  These tests use synthetic
tagged providers so the negative cases always run.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

import lunar_od.measurement_contracts as MC
from lunar_od.ephemeris import (
    MOON_EARTH_DISTANCE_BOUNDS_M,
    EphemerisContractError,
    EphemerisEpochMismatchError,
    LegacyEphemerisWarning,
)

ET0 = 8.0e8
T_PASS = np.array([0.0, 300.0, 600.0])
EARTH_DISTANCE_M = 3.844e8


@dataclass(frozen=True)
class TaggedEarth:
    et_origin: float = ET0
    frame: str = "J2000"
    center: str = "MOON"
    units: str = "m, m/s"
    legacy_reproduction_only: bool = False
    scale: float = 1.0

    def pos(self, t):
        t = np.asarray(t, dtype=float).reshape(-1)
        return self.scale * np.column_stack(
            [EARTH_DISTANCE_M * np.ones_like(t), 1.0e3 * t, np.zeros_like(t)])

    def vel(self, t):
        t = np.asarray(t, dtype=float).reshape(-1)
        return self.scale * np.column_stack(
            [np.zeros_like(t), 1.0e3 * np.ones_like(t), np.zeros_like(t)])


def test_matched_provider_pair_accepted():
    good = TaggedEarth()
    assert MC.require_earth_provider_contract(good.pos, good.vel, ET0, T_PASS) == ("MATCHED", "MATCHED")


def test_untagged_metre_callables_accepted():
    good = TaggedEarth()
    assert MC.require_earth_provider_contract(
        lambda t: good.pos(t), lambda t: good.vel(t), ET0, T_PASS) == ("UNTAGGED", "UNTAGGED")


def test_epoch_origin_offset_rejected():
    """Phase 17-FXR: Earth seconds counted from an origin 504000 s before et0."""
    bad = TaggedEarth(et_origin=ET0 - 504000.0)
    with pytest.raises(EphemerisEpochMismatchError, match="504000"):
        MC.require_earth_provider_contract(bad.pos, bad.vel, ET0, T_PASS)


@pytest.mark.parametrize("field, value, match", [
    ("frame", "ECLIPJ2000", "frame"),
    ("center", "EARTH", "centre"),
    ("units", "km, km/s", "units"),
])
def test_wrong_metadata_rejected(field, value, match):
    bad = replace(TaggedEarth(), **{field: value})
    with pytest.raises(EphemerisContractError, match=match):
        MC.require_earth_provider_contract(bad.pos, bad.vel, ET0, T_PASS)


def test_velocity_provider_checked_independently():
    good, bad = TaggedEarth(), TaggedEarth(et_origin=ET0 + 1.0)
    with pytest.raises(EphemerisEpochMismatchError, match="get_earth_vel"):
        MC.require_earth_provider_contract(good.pos, bad.vel, ET0, T_PASS)


@pytest.mark.parametrize("scale", [1.0e-3, 1.0 / EARTH_DISTANCE_M, 2.0])
def test_untagged_non_physical_distance_rejected(scale):
    """km-for-m, a unit vector, and an Earth twice too far all fall outside the bounds."""
    bad = TaggedEarth(scale=scale)
    lo, hi = MOON_EARTH_DISTANCE_BOUNDS_M
    assert not lo <= scale * EARTH_DISTANCE_M <= hi
    with pytest.raises(EphemerisContractError, match="Moon-Earth distance"):
        MC.require_earth_provider_contract(lambda t: bad.pos(t), lambda t: bad.vel(t), ET0, T_PASS)


def test_non_finite_earth_position_rejected():
    good = TaggedEarth()

    def pos(t):
        p = good.pos(t)
        p[1, 0] = np.nan
        return p

    with pytest.raises(EphemerisContractError, match="Moon-Earth distance"):
        MC.require_earth_provider_contract(pos, good.vel, ET0, T_PASS)


def test_legacy_provider_accepted_with_warning_despite_offset():
    legacy = TaggedEarth(et_origin=ET0 - 504000.0, legacy_reproduction_only=True)
    with pytest.warns(LegacyEphemerisWarning):
        assert MC.require_earth_provider_contract(legacy.pos, legacy.vel, ET0, T_PASS) == ("LEGACY", "LEGACY")


def test_checked_generator_forwards_arguments_and_result_unchanged(monkeypatch):
    calls = []
    sentinel = object()

    def fake_generator(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(MC, "generate_two_way_range_measurements", fake_generator)
    good = TaggedEarth()
    state = np.zeros((T_PASS.size, 6))
    mask = np.ones((T_PASS.size, 1), bool)
    out = MC.generate_two_way_range_measurements_checked(
        T_PASS, state, ("st",), mask, good.pos, good.vel, ET0, noise=False, seed=7)
    assert out is sentinel
    ((args, kwargs),) = calls
    assert args[0] is T_PASS and args[1] is state and args[3] is mask
    assert args[4] == good.pos and args[5] == good.vel and args[6] == ET0
    assert kwargs == {"noise": False, "seed": 7}


def test_checked_generator_rejects_before_generating(monkeypatch):
    def must_not_run(*args, **kwargs):
        raise AssertionError("generator called despite a contract violation")

    monkeypatch.setattr(MC, "generate_two_way_range_measurements", must_not_run)
    bad = TaggedEarth(et_origin=ET0 - 504000.0)
    with pytest.raises(EphemerisEpochMismatchError):
        MC.generate_two_way_range_measurements_checked(
            T_PASS, np.zeros((3, 6)), ("st",), np.ones((3, 1), bool), bad.pos, bad.vel, ET0)
