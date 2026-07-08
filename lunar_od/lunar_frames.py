"""Lunar body-fixed frame providers (Phase 13A).

Epoch-dependent J2000 -> MOON_PA rotation sampling for the lunar
spherical-harmonic gravity engine, plus a pure-numpy nearest-neighbour lookup
for the propagation hot path.

Why this exists
---------------
The constant ``dynamics._MCI_TO_MOON_BF`` (mean-pole) rotation is adequate for
the axially symmetric J2 term only.  Longitude-dependent (m > 0) spherical
harmonics need the epoch-dependent Moon orientation, because the Moon rotates
(synchronously, ~27.3 d): with a frozen frame the C22 field would be fixed in
inertial space and become unphysical within hours.  The GRAIL (GRGM/GL)
coefficient sets are referenced to the lunar principal-axis frame, so the
target frame is SPICE ``MOON_PA`` (kernels ``moon_080317.tf`` +
``moon_pa_de421_1900-2050.bpc``, both already in
``spice_loader.REQUIRED_KERNELS``).

Design (matches the ephemeris pre-sampling pattern)
---------------------------------------------------
- ``sample_moon_pa_rotations`` calls SPICE ONCE, before propagation, to build a
  dense rotation grid.  There are NO SPICE calls in the propagation hot path.
- ``nearest_rotation_at_time`` is the hot-path lookup: pure numpy, no SPICE, no
  kernels.  Nearest-neighbour on a dense grid (default cadence suggestion:
  60 s) keeps the returned matrix exactly orthonormal (it is a genuinely
  sampled rotation, not a blend) and is bit-identical between the Python and
  future Numba paths.  The 60 s default is a working assumption to be tested by
  the Phase 13C cadence preflight, not a validated scientific choice.
- Importing this module has NO side effects: kernels are loaded only inside
  ``sample_moon_pa_rotations`` (with ``clear=False``, so an already-furnished
  kernel pool is left intact), or not at all when ``load_kernels=False``.
- If the kernels cannot be found the sampler raises an explicit
  ``FileNotFoundError`` (via ``spice_loader``).  There is NO silent
  identity-frame fallback: an identity/fixed frame is acceptable only in tests
  and prototypes, never for real m > 0 propagation.

Rotation convention (contract)
------------------------------
``C = spice.pxform("J2000", "MOON_PA", et)`` maps a J2000 inertial vector to
the MOON_PA body-fixed frame:

    r_pa    = C @ r_j2000
    r_j2000 = C.T @ r_pa          (C is orthonormal: C^-1 = C^T)
    C.T @ C = I,  det(C) = +1     (proper rotation)

This is exactly the ``c_inertial_to_bf`` expected by
``gravity_harmonics.spherical_harmonic_acceleration`` (which computes
``r_bf = C @ r`` and returns ``C.T @ a_bf``).

Time convention: ``et0`` is a SPICE ET (TDB seconds past J2000), consistent
with the ephemeris layer (``et0 = (first_jd_TDB - 2451545.0) * 86400``);
``t_grid_s`` holds propagation-relative seconds, so row ``i`` is evaluated at
``et0 + t_grid_s[i]``.

Scope (Phase 13A): provider only.  No dynamics splice, no config fields, no
Numba kernel -- those are Phase 13B.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .spice_loader import load_spice_kernels

__all__ = [
    "sample_moon_pa_rotations",
    "nearest_rotation_at_time",
    "validate_rotation_matrix",
]


def _as_time_grid(t_grid_s: ArrayLike, name: str = "t_grid_s") -> np.ndarray:
    grid = np.asarray(t_grid_s, dtype=float)
    if grid.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array of seconds.")
    if grid.size == 0:
        raise ValueError(f"{name} must contain at least one epoch.")
    if not np.all(np.isfinite(grid)):
        raise ValueError(f"{name} must be finite.")
    if grid.size > 1 and not np.all(np.diff(grid) > 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    return grid


def sample_moon_pa_rotations(
    et0: float,
    t_grid_s: ArrayLike,
    *,
    frame: str = "MOON_PA",
    kernel_dir: os.PathLike[str] | str | None = None,
    load_kernels: bool = True,
) -> np.ndarray:
    """Sample C_j2000->MOON_PA rotation matrices on a time grid via SPICE.

    Parameters
    ----------
    et0 : SPICE ET (TDB seconds past J2000) of grid epoch zero.
    t_grid_s : (N,) propagation-relative seconds, strictly increasing; row i is
        evaluated at ``et0 + t_grid_s[i]``.
    frame : target body-fixed frame (default ``"MOON_PA"``; the GRAIL/GRGM
        coefficient frame).
    kernel_dir : optional explicit SPICE kernel directory; ``None`` uses the
        standard resolver chain (``LUNAR_OD_KERNEL_DIR`` env var,
        ``~/Documents/mice/kernels``, ``<root>/kernels``).
    load_kernels : when True (default), the required kernels are furnished via
        ``spice_loader.load_spice_kernels(..., clear=False)`` -- the existing
        kernel pool is NOT cleared.  Pass False if the caller already manages
        the pool (the frame/orientation kernels must then be loaded, otherwise
        SPICE raises).

    Returns
    -------
    (N, 3, 3) float64 array of proper rotation matrices with the convention
    ``r_pa = C[i] @ r_j2000``.

    Raises ``FileNotFoundError`` (explicit, no silent fallback) when the
    kernels cannot be resolved.
    """
    grid = _as_time_grid(t_grid_s)
    if load_kernels:
        load_spice_kernels(kernel_dir, clear=False)
    import spiceypy as spice

    rotations = np.zeros((grid.size, 3, 3), dtype=np.float64)
    for idx, rel_t_s in enumerate(grid):
        rotations[idx, :, :] = np.asarray(
            spice.pxform("J2000", frame, float(et0 + rel_t_s)), dtype=np.float64
        )
    return rotations


def nearest_rotation_at_time(
    rotation_grid: np.ndarray,
    t_grid_s: ArrayLike,
    t_s: float,
) -> np.ndarray:
    """Nearest-neighbour rotation lookup on a pre-sampled grid (hot path).

    Pure numpy: no SPICE calls, no kernel requirement.  Returns the (3, 3)
    matrix of the grid epoch closest to ``t_s``.  An exact midpoint tie
    resolves deterministically to the LOWER index (first minimum).

    Raises ``ValueError`` for: out-of-range ``t_s`` (no extrapolation),
    a non-monotonic grid, a grid/matrix length mismatch, or malformed inputs.
    """
    grid = _as_time_grid(t_grid_s)
    rotations = np.asarray(rotation_grid, dtype=float)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("rotation_grid must have shape (N, 3, 3).")
    if rotations.shape[0] != grid.size:
        raise ValueError(
            f"rotation_grid length {rotations.shape[0]} does not match "
            f"t_grid_s length {grid.size}."
        )
    t = float(t_s)
    if t < grid[0] or t > grid[-1]:
        raise ValueError(
            f"t_s={t} is outside the sampled rotation grid "
            f"[{grid[0]}, {grid[-1]}] (no extrapolation)."
        )
    idx = int(np.argmin(np.abs(grid - t)))   # first minimum -> lower index on tie
    return rotations[idx]


def validate_rotation_matrix(c: ArrayLike, *, atol: float = 1e-10) -> None:
    """Raise ``ValueError`` unless ``c`` is a proper rotation matrix.

    Checks: shape (3, 3), finite entries, orthonormality ``max|C^T C - I| <=
    atol`` and ``|det(C) - 1| <= atol`` (proper rotation, no reflection/scale).
    """
    matrix = np.asarray(c, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("rotation matrix must have shape (3, 3).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("rotation matrix must be finite.")
    ortho_err = float(np.max(np.abs(matrix.T @ matrix - np.eye(3))))
    if ortho_err > atol:
        raise ValueError(f"matrix is not orthonormal: max|C^T C - I| = {ortho_err:.3e}.")
    det_err = float(abs(np.linalg.det(matrix) - 1.0))
    if det_err > atol:
        raise ValueError(f"matrix is not a proper rotation: |det - 1| = {det_err:.3e}.")
