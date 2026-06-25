"""Generic body-J2 force-model helpers (Phase 3).

These wrap the existing body-fixed zonal kernels (``zonal_j2_acceleration`` /
``zonal_j2_gravity_gradient``) with the inertial <-> body-fixed frame rotation,
so the J2 perturbation of *any* central body (Moon, Earth, ...) can be evaluated
with one code path given that body's GM, reference radius, unnormalized J2 and
constant inertial->body-fixed rotation.

Scope (Phase 3): defined and independently tested, but NOT yet wired into the
production propagation path (that is Phase 4); Earth J2 is not wired into any
propagation (that is Phase 5).

Contract
--------
- SI units: ``r_rel_inertial`` in metres, ``mu`` in m^3/s^2, ``radius_ref`` in
  metres; returns acceleration in m/s^2 and gravity-gradient in 1/s^2.
- ``j2`` is the **unnormalized** zonal coefficient (J2 = -C20), not C-bar(2,0).
- ``r_rel_inertial`` is the spacecraft position relative to the J2 body's centre,
  expressed in the inertial (propagation) axes.
- ``c_inertial_to_bf`` is the constant 3x3 rotation from inertial axes to the
  body's body-fixed (pole-aligned) frame.  Because J2 is axially symmetric, only
  the pole direction matters, so a constant (mean-pole) rotation suffices.

Indirect-term note (for Earth J2 in Phase 5): the indirect/relative formulation
``a_J2(sc rel body) - a_J2(centre rel body)`` is the responsibility of the call
site, not of these helpers; each helper evaluates a single direct J2 term.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .dynamics import zonal_j2_acceleration, zonal_j2_gravity_gradient

__all__ = ["body_j2_acceleration", "body_j2_gravity_gradient"]


def body_j2_acceleration(
    r_rel_inertial: ArrayLike,
    mu: float,
    radius_ref: float,
    j2: float,
    c_inertial_to_bf: ArrayLike,
) -> np.ndarray:
    """J2 perturbing acceleration (inertial axes) for one central body.

    Rotates ``r_rel_inertial`` into the body-fixed frame, evaluates the zonal J2
    acceleration there, and rotates the result back to inertial axes.  Uses the
    same numpy operations and order as the existing Moon-J2 production path, so
    with Moon parameters the result is bitwise identical to that path.
    """
    c = np.asarray(c_inertial_to_bf, dtype=float)
    r_bf = c @ np.asarray(r_rel_inertial, dtype=float)
    a_bf = zonal_j2_acceleration(r_bf, mu, radius_ref, j2)
    return c.T @ a_bf


def body_j2_gravity_gradient(
    r_rel_inertial: ArrayLike,
    mu: float,
    radius_ref: float,
    j2: float,
    c_inertial_to_bf: ArrayLike,
) -> np.ndarray:
    """J2 gravity-gradient tensor (inertial axes) for one central body.

    Returns ``C^T @ G_bf @ C`` where ``G_bf`` is the body-fixed analytic J2
    gravity gradient.  Matches the existing Moon-J2 STM assembly exactly for
    Moon parameters.
    """
    c = np.asarray(c_inertial_to_bf, dtype=float)
    r_bf = c @ np.asarray(r_rel_inertial, dtype=float)
    g_bf = zonal_j2_gravity_gradient(r_bf, mu, radius_ref, j2)
    return c.T @ g_bf @ c
