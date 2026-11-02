"""The shadow-contact guard judges a pre-pass cut by its time offset from the final trajectory's contact.

2026-10-27: on the two-day N = 300 LRO arc the angle-only 1e-9 rad guard refused every tolerance (1.1e-9 rad at
rtol 1e-12) although the cut was ~1e-6 s from the contact. The guard must accept a mm-level along-track
difference, still refuse a cut with no contact nearby, and keep the angle test at a tangent.
"""
import numpy as np
import pytest
from scipy.optimize import brentq

from lunar_od import dynamics as dyn
from lunar_od.srp import SRPOptions, apparent_radii_and_separation

SUN = np.array([-1.496e11, 0.0, 0.0])        # shadow on the +x side of the Moon
R_ORBIT = 1.79e6
V_ORBIT = 1630.0
OMEGA = V_ORBIT / R_ORBIT


def _state(t):
    th = OMEGA * t
    return np.array([R_ORBIT * np.cos(th), R_ORBIT * np.sin(th), 0.0,
                     -V_ORBIT * np.sin(th), V_ORBIT * np.cos(th), 0.0])


@pytest.fixture
def setup():
    srp = SRPOptions(k_srp_m2_per_kg=0.01)
    radius = srp.occulting_body_radius_m

    def outer(t):
        a, b, g, _, _ = apparent_radii_and_separation(_state(t)[:3], SUN, np.zeros(3), radius)
        return g - a - b
    # the orbit enters penumbra a little before reaching the -y..+x shadow side
    ts = np.linspace(0.0, 2 * np.pi / OMEGA, 4001)
    vals = np.array([outer(t) for t in ts])
    k = int(np.flatnonzero(np.sign(vals[:-1]) != np.sign(vals[1:]))[0])
    t_c = brentq(outer, ts[k], ts[k + 1], xtol=1e-12)
    return dyn._shadow_contact_check(lambda t: SUN, srp), t_c, radius


def test_cut_at_the_contact_passes(setup):
    check, t_c, _ = setup
    check(t_c, _state(t_c))


def test_millimetre_trajectory_difference_passes_although_the_angle_exceeds_1e9(setup):
    check, t_c, radius = setup
    s = _state(t_c)
    s[:3] += 5e-3 * s[3:6] / np.linalg.norm(s[3:6])           # 5 mm along track: ~3e-6 s
    a, b, g, _, _ = apparent_radii_and_separation(s[:3], SUN, np.zeros(3), radius)
    assert abs(g - a - b) > 1e-9                               # the old angle-only guard refused this
    check(t_c, s)


def test_cut_without_a_nearby_contact_fails(setup):
    check, t_c, _ = setup
    with pytest.raises(RuntimeError, match="from the contact"):
        check(t_c + 10.0, _state(t_c + 10.0))


def test_tangent_without_rate_keeps_the_angle_test(setup):
    check, t_c, _ = setup
    s = _state(t_c)
    s[:3] *= 1.0 + 1e-6                                         # off the contact
    s[3:6] = 0.0                                                # no motion along the trajectory: rate = 0
    with pytest.raises(RuntimeError, match="Shadow contact inconsistent"):
        check(t_c, s)
