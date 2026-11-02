"""F1/F2 LRO station provider is free of ulp(et0) epoch quantisation (G4e root cause, 2026-10-27).

A fake SPICE with an analytic rotating station and a linearly moving Earth: the provider must follow the exact
state at ``et0 + t`` for every ``t``, whereas reading SPICE at the rounded epoch alone steps by
velocity x ulp(et0) (~1e-4 m in 2009).
"""
import sys
import types

import numpy as np
import pytest

import phase_f1_lro_residuals as F1

OMEGA = 7.2921159e-5            # rad/s
EARTH_V_KMS = np.array([0.9, -0.3, 0.1])
EARTH_R0_KM = np.array([3.0e5, 2.0e5, 1.0e5])
ET0 = 3.06e8                    # 2009, ulp(ET0) = 5.96e-8 s


def _rot_t(t):
    c, s = np.cos(OMEGA * t), np.sin(OMEGA * t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot(et):
    # angle from (et - ET0), exact by Sterbenz: the fake is exact at the (rounded) epoch it is given, so the test
    # sees only the provider's quantisation, not cos() of a 2e4 rad argument
    return _rot_t(et - ET0)


def _sxform(_from, _to, et):
    r = _rot(et)
    w = OMEGA * np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    x = np.zeros((6, 6))
    x[:3, :3] = r
    x[3:, 3:] = r
    x[3:, :3] = w @ r
    return x


def _spkezr(_target, et, _frame, _abcorr, _obs):
    return np.concatenate([EARTH_R0_KM + EARTH_V_KMS * (et - ET0), EARTH_V_KMS]), 0.0


def _exact(r, t):
    """The station position at the exact epoch ET0 + t."""
    return _rot_t(t) @ r + (EARTH_R0_KM + EARTH_V_KMS * t) * 1e3


@pytest.fixture
def provider(monkeypatch):
    fake = types.SimpleNamespace(sxform=_sxform, spkezr=_spkezr)
    monkeypatch.setitem(sys.modules, "spiceypy", fake)
    setup = object.__new__(F1.Setup)
    setup.et0 = ET0
    r = np.array([6.378e6, 0.0, 0.0])

    def make(exact):
        monkeypatch.setattr(F1, "EXACT_EPOCH_STATIONS", exact)
        return setup._station(r)
    return make, r


@pytest.mark.parametrize("t", [12_345.678901234, 91_234.5, 160_001.25])
def test_station_position_follows_the_exact_epoch(provider, t):
    make, r = provider
    exact = make(True)
    quantised = make(False)
    ts = t + np.arange(-200, 201) * np.spacing(t) * 37.0      # crosses many ulp(et0) boundaries
    err_exact = max(np.linalg.norm(exact(x)[:3] - _exact(r, x)) for x in ts)
    err_quant = max(np.linalg.norm(quantised(x)[:3] - _exact(r, x)) for x in ts)
    # quantised provider: |v| x ulp(et0)/2 ~ 1e3 m/s x 3e-8 s; exact: a dt^2 (~1e-17 m) plus ulp of a 4e8 m
    # position (6e-8 m)
    assert err_quant > 1e-5
    assert err_exact < 3e-7


def test_exact_provider_is_continuous_across_an_epoch_rounding_boundary(provider):
    make, _ = provider
    exact = make(True)
    t = 91_234.5
    ts = t + np.arange(0, 4001) * np.spacing(t)
    x = np.array([exact(s)[:3] for s in ts])
    steps = np.linalg.norm(np.diff(x, axis=0), axis=1)
    # one ulp(t) of motion plus position rounding is ~1e-7 m; a quantisation jump would be ~6e-5 m
    assert steps.max() < 3e-7
    quantised = make(False)
    xq = np.array([quantised(s)[:3] for s in ts])
    assert np.linalg.norm(np.diff(xq, axis=0), axis=1).max() > 1e-5
