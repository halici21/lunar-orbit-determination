"""Phase D2: shadow-event segmented integration and PropagationProfile.

Default invariance, the segmentation mechanics on an analytic problem, and the
contact epochs on a synthetic eclipsing orbit.  The CANON-arc accuracy and the
estimator closure are qualified in studies/phase17/scripts/phase_d2_propagation.py.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from lunar_od.constants import AU_M, MU_MOON_M3S2, R_MOON_M
from lunar_od.dynamics import (
    PropagationProfile,
    _integrate,
    _integrate_segmented,
    propagate_augmented_state,
    propagate_state,
    propagate_state_with_k_sensitivity,
    shadow_contact_times,
)
from lunar_od.srp import SRPOptions, _illumination_for

R_ORBIT = R_MOON_M + 100e3
V_ORBIT = math.sqrt(MU_MOON_M3S2 / R_ORBIT)
PERIOD = 2.0 * math.pi * R_ORBIT / V_ORBIT
X0 = np.array([R_ORBIT, 0.0, 0.0, 0.0, V_ORBIT, 0.0])
SUN = np.array([AU_M, 0.0, 0.0])
ARGS = (MU_MOON_M3S2, 0.0, 0.0, lambda _t: np.array([384_400e3, 0.0, 0.0]), lambda _t: SUN)
SRP = SRPOptions(k_srp_m2_per_kg=0.01)
T = np.linspace(0.0, 1.5 * PERIOD, 200)
AUG0 = np.concatenate([X0, np.eye(6).reshape(-1, order="F")])


def _oscillator(_t, y):
    return np.array([y[1], -y[0]])


def test_profile_defaults():
    assert PropagationProfile().propagator_kwargs() == {"method": "DOP853", "shadow_events": True}


def test_segmented_without_cuts_is_the_single_run():
    t = np.linspace(0.0, 10.0, 37)
    y0 = np.array([1.0, 0.0])
    a = _integrate_segmented(t, y0, _oscillator, 1e-10, 1e-12, "DOP853", [], "osc")
    b = _integrate(t, y0, _oscillator, 1e-10, 1e-12, "DOP853", "osc")
    assert np.array_equal(a, b)


@pytest.mark.parametrize("method", ["DOP853", "ADAMS"])
@pytest.mark.parametrize("backward", [False, True])
def test_segmented_matches_analytic_with_cuts(method, backward):
    t = np.linspace(0.0, 10.0, 41)
    cuts = [1.234, 5.0, 7.77]          # 5.0 coincides with an output epoch
    if backward:
        t, cuts = t[::-1], cuts[::-1]
    y0 = np.array([math.cos(t[0]), -math.sin(t[0])])
    out = _integrate_segmented(t, y0, _oscillator, 1e-12, 1e-14, method, cuts, "osc")
    assert out.shape == (t.size, 2)
    assert np.max(np.abs(out[:, 0] - np.cos(t))) < 1e-9


def test_events_flag_is_inert_without_a_shadow():
    for srp in (None, SRPOptions(k_srp_m2_per_kg=0.01, shadow_model="NO_SHADOW")):
        a = propagate_state(T, X0, *ARGS, srp=srp)
        b = propagate_state(T, X0, *ARGS, srp=srp, shadow_events=True)
        assert np.array_equal(a, b)
        a = propagate_augmented_state(T, AUG0, *ARGS, srp=srp)
        b = propagate_augmented_state(T, AUG0, *ARGS, srp=srp, shadow_events=True)
        assert np.array_equal(a, b)
    assert shadow_contact_times(T, X0, *ARGS, srp=SRPOptions(k_srp_m2_per_kg=0.01, shadow_model="NO_SHADOW")) == []


def test_default_is_unchanged_by_the_new_keyword():
    a = propagate_state(T, X0, *ARGS, srp=SRP)
    b = propagate_state(T, X0, *ARGS, srp=SRP, shadow_events=False)
    assert np.array_equal(a, b)


def _regime(t, x):
    nu = _illumination_for(np.asarray(x[:3]), SUN, SRP)
    return 2 if nu >= 1.0 else (0 if nu <= 0.0 else 1)


def test_contacts_separate_illumination_regimes():
    contacts = shadow_contact_times(T, X0, *ARGS, srp=SRP, rtol=1e-12, atol=1e-12)
    # an equatorial orbit with the Sun in its plane crosses the shadow once per
    # revolution (outer and inner contact on entry and on exit); 1.5
    # revolutions from the sub-solar point give one full shadow and the next entry
    assert len(contacts) == 6
    assert contacts[4] - contacts[0] == pytest.approx(PERIOD, rel=1e-2)
    assert contacts == sorted(contacts)
    for tc in contacts:
        tt = np.array([0.0, tc - 0.05, tc + 0.05])
        x = propagate_state(tt, X0, *ARGS, srp=SRP, method="DOP853", rtol=1e-12, atol=1e-12, shadow_events=True)
        assert _regime(tt[1], x[1]) != _regime(tt[2], x[2])


def test_segmented_propagators_agree():
    kw = dict(srp=SRP, rtol=1e-11, atol=1e-12, **PropagationProfile().propagator_kwargs())
    s = propagate_state(T, X0, *ARGS, **kw)
    a = propagate_augmented_state(T, AUG0, *ARGS, **kw)
    k = propagate_state_with_k_sensitivity(T, X0, *ARGS, **kw)
    tight = propagate_state(T, X0, *ARGS, **(kw | dict(rtol=1e-13, atol=1e-14)))
    for x in (s, a, k):
        assert np.max(np.linalg.norm(x[:, :3] - tight[:, :3], axis=1)) < 1e-4
