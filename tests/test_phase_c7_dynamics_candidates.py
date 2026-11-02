"""Phase C7 dynamics candidates against finite differences and closed forms (gate G_UNIT)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "studies" / "phase17" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import phase_c7_dynamics_candidates as T  # noqa: E402
from lunar_od.constants import SPEED_OF_LIGHT_M_S  # noqa: E402

MU_E = 3.986004354360959e14
MU_M = 4.902800066163796e12
R = np.array([[1.20e6, -0.55e6, 1.17e6]])
D_EARTH = np.array([[3.1e8, 2.0e8, 0.6e8]])


def test_tide_acceleration_is_the_gradient_of_its_potential():
    for n, k in ((2, T.K2_GL1800F), (3, T.K3_GL1800F)):
        a = T.tide_acceleration(R, D_EARTH, MU_E, n, k)[0]
        h = 10.0
        fd = np.array([(T.tide_potential(R + h * e, D_EARTH, MU_E, n, k)[0]
                        - T.tide_potential(R - h * e, D_EARTH, MU_E, n, k)[0]) / (2 * h) for e in np.eye(3)[:, None, :]])
        assert np.max(np.abs(a - fd)) <= 1e-6 * np.linalg.norm(a)


def test_degree_two_earth_tide_size_at_50_km():
    r = np.array([[1788.0e3, 0.0, 0.0]])
    a = T.tide_acceleration(r, np.array([[3.844e8, 0.0, 0.0]]), MU_E, 2, T.K2_GL1800F)[0]
    want = T.K2_GL1800F * MU_E * T.R_REF_GL1800F_M ** 5 / (3.844e8 ** 3 * 1788.0e3 ** 4) * 3.0   # (n+1) P2(1) = 3
    assert abs(np.linalg.norm(a) / want - 1.0) < 1e-12
    assert a[0] < 0.0          # at the sub-Earth point the induced bulge pulls the spacecraft down


def test_pn_term_is_radial_on_a_circular_orbit():
    r = np.array([[1788.0e3, 0.0, 0.0]])
    v = np.array([[0.0, np.sqrt(MU_M / 1788.0e3), 0.0]])
    a = T.pn_acceleration(r, v, MU_M)[0]
    want = MU_M / (SPEED_OF_LIGHT_M_S ** 2 * 1788.0e3 ** 2) * (4.0 * MU_M / 1788.0e3 - v[0, 1] ** 2)
    assert abs(a[0] / want - 1.0) < 1e-12 and abs(a[1]) == 0.0 and abs(a[2]) == 0.0


def test_third_body_closed_form():
    d = np.array([[1.0e11, 0.0, 0.0]])
    r = np.array([[1.0e6, 0.0, 0.0]])
    a = T.third_body_acceleration(r, d, 3.0e14)[0]
    want = 3.0e14 * (1.0 / (1.0e11 - 1.0e6) ** 2 - 1.0 / 1.0e22)
    assert abs(a[0] / want - 1.0) < 1e-9
