"""Phase 5 — Earth J2 (Moon-centered) verification.

Indirect is the default physical model:
    a_EarthJ2_rel = a_J2(sc rel Earth) - a_J2(Moon rel Earth)
Direct keeps only the spacecraft term (debug / sensitivity).  These tests pin
the physics, the Python<->Numba parity, the indirect<<direct magnitude, and the
bit-for-bit preservation of the Earth-off (Phase 4) behaviour.
"""
import unittest

import numpy as np

from lunar_od import (
    body_j2_acceleration,
    body_j2_gravity_gradient,
    dynamics_jacobian_a_matrix,
    f3body_moon,
    ode_fun_v3,
)
from lunar_od.dynamics import _J2000_TO_EARTH_BF as C_E, _earth_mode_int
from lunar_od.accelerated import f3body_rhs, ode42_rhs
from lunar_od.constants import (
    J2_EARTH_UNNORMALIZED as J2_E,
    R_EARTH_J2_REF_M as R_E,
    MU_EARTH_M3S2,
)

MU_M = 4902.800066163796e9
MU_E = 398600.4354360959e9
MU_S = 1.327124400419393e20
R_ME = np.array([-83446893.0, 354010875.0, 178558253.0])   # Earth rel Moon (m)
R_MS = np.array([1.40753701450e11, -4.21884124439e10, -1.82638284191e10])
MOON_R = 1_737_400.0


def _states():
    out = []
    for alt in (100e3, 2000e3):
        r0 = MOON_R + alt
        out += [
            np.array([r0, 0.0, 0.0, 0.0, np.sqrt(MU_M / r0), 0.0]),
            np.array([r0 * 0.6, r0 * 0.5, r0 * 0.62, -0.4, 0.7, 0.3]),
        ]
    return out


def _earth_accel_py(r_sc, mode):
    r_sc_e = r_sc - R_ME
    a = body_j2_acceleration(r_sc_e, MU_E, R_E, J2_E, C_E)
    if mode == "indirect":
        a = a - body_j2_acceleration(-R_ME, MU_E, R_E, J2_E, C_E)
    return a


def _a(state, **kw):
    return f3body_moon(state, MU_M, MU_E, MU_S, R_ME, R_MS, **kw)[3:]


class EarthJ2Tests(unittest.TestCase):
    # 1 -- helper sanity with Earth params
    def test_helper_sanity(self):
        a = body_j2_acceleration(_states()[0][:3] - R_ME, MU_E, R_E, J2_E, C_E)
        self.assertTrue(np.all(np.isfinite(a)) and np.linalg.norm(a) > 0.0)

    # 2 -- direct mode acceleration (composition is bitwise)
    def test_direct_mode_acceleration(self):
        for s in _states():
            a_on = _a(s, j2_earth=J2_E, earth_j2_mode="direct")
            a_off = _a(s)
            self.assertEqual(np.linalg.norm(a_on - (a_off + _earth_accel_py(s[:3], "direct"))), 0.0)

    # 3 -- indirect mode acceleration (machine precision: two-operation grouping
    #      (base+sc)-moon differs from base+(sc-moon) by FP association ~1e-16)
    def test_indirect_mode_acceleration(self):
        for s in _states():
            a_on = _a(s, j2_earth=J2_E, earth_j2_mode="indirect")
            a_off = _a(s)
            self.assertLess(np.linalg.norm(a_on - (a_off + _earth_accel_py(s[:3], "indirect"))), 1e-14)

    # 4 -- indirect form == direct term minus Moon term
    def test_indirect_equals_direct_minus_moon(self):
        for s in _states():
            direct = _earth_accel_py(s[:3], "direct")
            indirect = _earth_accel_py(s[:3], "indirect")
            moon_term = body_j2_acceleration(-R_ME, MU_E, R_E, J2_E, C_E)
            np.testing.assert_allclose(indirect, direct - moon_term, rtol=0, atol=0)

    # 5 -- j2_earth = 0 is a no-op (Py + Numba)
    def test_earth_zero_noop(self):
        for s in _states():
            self.assertEqual(np.linalg.norm(_a(s, j2_earth=0.0) - _a(s)), 0.0)
            an_off = np.asarray(f3body_rhs(s, MU_M, MU_E, MU_S, R_ME, R_MS))[3:]
            an_zero = np.asarray(f3body_rhs(s, MU_M, MU_E, MU_S, R_ME, R_MS,
                                            0.0, 0.0, np.eye(3), 0.0, 0.0, 0, np.eye(3)))[3:]
            self.assertEqual(np.linalg.norm(an_off - an_zero), 0.0)

    # 6 -- Python <-> Numba Earth acceleration parity (both modes)
    def test_python_numba_accel_parity(self):
        for mode, emode in (("indirect", 1), ("direct", 2)):
            worst = 0.0
            for s in _states():
                ap = _a(s, j2_earth=J2_E, earth_j2_mode=mode)
                an = np.asarray(f3body_rhs(s, MU_M, MU_E, MU_S, R_ME, R_MS,
                                           0.0, 0.0, np.eye(3), J2_E, R_E, emode, C_E))[3:]
                worst = max(worst, np.linalg.norm(ap - an) / np.linalg.norm(ap))
            self.assertLess(worst, 1e-12)

    # 7 -- Python <-> Numba Earth gradient / STM parity
    def test_python_numba_gradient_parity(self):
        PhiI = np.eye(6).reshape(-1, order="F")
        worst_x = worst_phi = 0.0
        for s in _states():
            xaug = np.concatenate([s, PhiI])
            dpy = ode_fun_v3(0.0, xaug, MU_M, MU_E, MU_S, lambda t: R_ME, lambda t: R_MS,
                             0.0, J2_E, "indirect")
            dnb = np.asarray(ode42_rhs(xaug, MU_M, MU_E, MU_S, R_ME, R_MS,
                                       0.0, 0.0, np.eye(3), J2_E, R_E, 1, C_E))
            worst_x = max(worst_x, np.linalg.norm(dpy[:6] - dnb[:6]) / np.linalg.norm(dpy[:6]))
            worst_phi = max(worst_phi, np.linalg.norm(dpy[6:] - dnb[6:]) / max(np.linalg.norm(dpy[6:]), 1e-30))
        self.assertLess(worst_x, 1e-12)
        self.assertLess(worst_phi, 1e-10)

    # 8 -- finite-difference gradient of the indirect acceleration
    def test_finite_difference_gradient(self):
        h = 1.0
        worst = 0.0
        for s in _states():
            r = s[:3]
            G = body_j2_gravity_gradient(r - R_ME, MU_E, R_E, J2_E, C_E)
            G_fd = np.zeros((3, 3))
            for j in range(3):
                e = np.zeros(3); e[j] = h
                G_fd[:, j] = (_earth_accel_py(r + e, "indirect") - _earth_accel_py(r - e, "indirect")) / (2 * h)
            worst = max(worst, np.linalg.norm(G_fd - G) / np.linalg.norm(G))
        self.assertLess(worst, 1e-4)

    # 9 -- Earth-off preserves Phase 4 behaviour bit-for-bit
    def test_earth_off_preserves_moon_j2(self):
        from lunar_od.dynamics import MOON_J2
        for s in _states():
            a_phase4 = _a(s, j2_moon=MOON_J2)
            a_with_earth_off = _a(s, j2_moon=MOON_J2, j2_earth=0.0)
            self.assertEqual(np.linalg.norm(a_phase4 - a_with_earth_off), 0.0)

    # 10 -- enabling Earth J2 adds only the Earth term (machine precision; FP
    #       association on the large base acceleration, see test 3)
    def test_only_earth_contribution_added(self):
        for s in _states():
            delta = _a(s, j2_earth=J2_E, earth_j2_mode="indirect") - _a(s)
            self.assertLess(np.linalg.norm(delta - _earth_accel_py(s[:3], "indirect")), 1e-14)

    # 11 -- indirect << direct (and sign sanity)
    def test_indirect_much_smaller_than_direct(self):
        for s in _states():
            direct = np.linalg.norm(_earth_accel_py(s[:3], "direct"))
            indirect = np.linalg.norm(_earth_accel_py(s[:3], "indirect"))
            self.assertLess(indirect, 0.5 * direct)   # near-cancellation; sign error -> ~2x

    # 12 -- mode mapping + invalid mode guard
    def test_mode_mapping(self):
        self.assertEqual(_earth_mode_int(0.0, "indirect"), 0)
        self.assertEqual(_earth_mode_int(J2_E, "indirect"), 1)
        self.assertEqual(_earth_mode_int(J2_E, "direct"), 2)
        with self.assertRaises(ValueError):
            f3body_moon(_states()[0], MU_M, MU_E, MU_S, R_ME, R_MS, j2_earth=J2_E, earth_j2_mode="bogus")


if __name__ == "__main__":
    unittest.main()
