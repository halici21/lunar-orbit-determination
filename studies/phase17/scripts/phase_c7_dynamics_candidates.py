"""Phase C7: lunar solid tide (k2, k3), Moon 1PN and planetary third-body accelerations (study-level; vectorised).

Predeclaration: studies/phase17/configs/phase_c7_dynamics_candidates_survey_predeclaration.json.
All inputs are Moon-centred J2000, SI; arrays are (n, 3).
"""
from __future__ import annotations

import numpy as np

from lunar_od.constants import SPEED_OF_LIGHT_M_S

#: GL1800F label (jggrx_1800f_sha.lbl lines 31-34): estimated Love numbers; the field excludes the permanent tide
K2_GL1800F = 0.024223
K3_GL1800F = 0.0163
R_REF_GL1800F_M = 1738.0e3


def _legendre(n: int, c):
    if n == 2:
        return 0.5 * (3.0 * c ** 2 - 1.0), 3.0 * c
    if n == 3:
        return 0.5 * (5.0 * c ** 3 - 3.0 * c), 0.5 * (15.0 * c ** 2 - 3.0)
    raise ValueError("degree 2 or 3")


def tide_potential(r, d, mu_j: float, n: int, k_n: float, radius: float = R_REF_GL1800F_M):
    r, d = np.atleast_2d(r), np.atleast_2d(d)
    rn, dn = np.linalg.norm(r, axis=1), np.linalg.norm(d, axis=1)
    c = np.einsum("ij,ij->i", r, d) / (rn * dn)
    return k_n * mu_j * radius ** (2 * n + 1) / (dn ** (n + 1) * rn ** (n + 1)) * _legendre(n, c)[0]


def tide_acceleration(r, d, mu_j: float, n: int, k_n: float, radius: float = R_REF_GL1800F_M):
    """Gradient of tide_potential w.r.t. the spacecraft position r; d = raising body position."""
    r, d = np.atleast_2d(r), np.atleast_2d(d)
    rn, dn = np.linalg.norm(r, axis=1), np.linalg.norm(d, axis=1)
    rh, dh = r / rn[:, None], d / dn[:, None]
    c = np.einsum("ij,ij->i", rh, dh)
    p, dp = _legendre(n, c)
    scale = k_n * mu_j * radius ** (2 * n + 1) / dn ** (n + 1) * rn ** -(n + 2)
    return scale[:, None] * (-(n + 1) * p[:, None] * rh + dp[:, None] * (dh - c[:, None] * rh))


def pn_acceleration(r, v, mu: float):
    """Schwarzschild term of the central body (IERS 2010 eq. 10.12 without Lense-Thirring / de Sitter)."""
    r, v = np.atleast_2d(r), np.atleast_2d(v)
    rn = np.linalg.norm(r, axis=1)
    v2 = np.einsum("ij,ij->i", v, v)
    rv = np.einsum("ij,ij->i", r, v)
    f = mu / (SPEED_OF_LIGHT_M_S ** 2 * rn ** 3)
    return f[:, None] * ((4.0 * mu / rn - v2)[:, None] * r + 4.0 * rv[:, None] * v)


def third_body_acceleration(r, d, mu_j: float):
    r, d = np.atleast_2d(r), np.atleast_2d(d)
    q = d - r
    return mu_j * (q / np.linalg.norm(q, axis=1)[:, None] ** 3 - d / np.linalg.norm(d, axis=1)[:, None] ** 3)
