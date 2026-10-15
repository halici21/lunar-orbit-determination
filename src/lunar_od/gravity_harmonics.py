"""Production low-degree lunar spherical-harmonic gravity engine (Phase 11B).

Pines-formulation perturbing acceleration for a fully-normalized spherical
harmonic gravity field.  The Pines (Cartesian direction-cosine) formulation was
selected in Phase 11A because it has NO polar singularity: exact-pole evaluation
is clean and continuous, which the classical spherical-gradient formulation is
not (it loses the m=1 horizontal acceleration at the exact pole).  Polar low
lunar orbits are a primary use case, so this property is decisive.

Scope (Phase 11B)
-----------------
Acceleration-only, pure-Python reference engine, DEFAULT-OFF (nothing here is
wired into the production propagation path).  This module adds no dependency to
``dynamics.py`` / ``accelerated.py`` / ``force_models.py``; it is imported
explicitly by callers/tests via ``lunar_od.gravity_harmonics``.

Deliberately NOT in this phase (each is a later, gated step):
- Numba twin + Python<->Numba parity gate (Phase 11C).  The body-fixed core
  ``_pines_acceleration_bf`` is kept as a self-contained scalar-loop routine so
  it ports directly to the ``accelerated.py`` scalar-tuple njit pattern.
- Potential U(r) (energy sanity).  Acceleration is sufficient for all Phase 11B
  validation; potential is deferred to a later energy-sanity/validation phase.
- Coefficient-file loader (SHADR/.gfc) and real GRAIL models (Phase 12).  Real
  models carry their own GM and reference radius (GRAIL R_ref = 1738.0 km, NOT
  R_MOON_M = 1737.4 km); the engine always uses the model's own mu/r_ref.
- Gradient / STM (see policy below).

Units / conventions
--------------------
- SI everywhere: position in metres, ``mu`` in m^3/s^2, reference radius in
  metres; returned acceleration in m/s^2.
- Coefficients are FULLY NORMALIZED Cbar_nm / Sbar_nm, stored as square 2-D
  arrays indexed ``cbar[n, m]`` (rows n, columns m; the strict upper triangle
  m > n is unused).  The zonal relation is format-dependent:
  ``Jn = -sqrt(2n+1) * Cbar_n0``  (for J2: ``Cbar20 = -J2 / sqrt(5)``).
- Only the perturbation (degrees ``n >= 2``) is evaluated.  The ``n = 0``
  point-mass term and the ``n = 1`` centre-of-mass terms are IGNORED (the call
  site owns the point-mass term, exactly like the existing J2 helpers); a model
  may carry nonzero n=0/n=1 rows without affecting the output.

Frame policy
------------
``c_inertial_to_bf`` is the rotation from inertial (propagation) axes to the
body-fixed frame in which the coefficients are defined; ``None`` means identity.
A fixed / identity rotation is adequate ONLY for low-degree engine validation at
a single epoch.  Real ``m > 0`` (tesseral/sectoral) lunar-harmonics propagation
REQUIRES an epoch-dependent inertial->MOON_PA rotation, because the Moon rotates
(synchronous, ~27.3 d) and longitude-dependent terms otherwise freeze in
inertial space and become unphysical within hours.  The constant
``dynamics._MCI_TO_MOON_BF`` used by the production Moon-J2 path is only
pole-aligned and is NOT sufficient for tesseral/sectoral terms.  The
epoch-dependent provider (SPICE ``pxform("J2000", "MOON_PA", et)``, kernel
``moon_pa_de421_1900-2050.bpc``, pre-sampled onto the integration grid) is
Phase 12/13 scope.

Double-count policy (documented; enforcement is a later phase)
--------------------------------------------------------------
A harmonic model that includes ``Cbar20`` already contains the J2 term.  If such
a model is composed with the separate Moon-J2 path (``j2_moon != 0``), J2 is
counted twice.  When this engine is wired into the dynamics (Phase 12+), the
combination ``lunar harmonics active (n >= 2) AND j2_moon != 0`` MUST raise
``ValueError``.  Silently zeroing ``Cbar20`` or silently disabling ``j2_moon``
is forbidden -- force-model composition must stay explicit and auditable.
Earth J2 is a different body and remains composable with lunar harmonics.

Gradient / STM policy (documented; enforcement is a later phase)
----------------------------------------------------------------
This engine is acceleration-only: no gravity-gradient / Jacobian is implemented.
STM / OD estimator integration needs a harmonic gradient.  When harmonics are
later requested together with augmented (STM) dynamics without that gradient,
the code MUST raise an explicit error rather than silently propagating a
harmonic acceleration with a J2-level or missing gradient.  A silent
acceleration/gradient mismatch is forbidden.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "SphericalHarmonicGravityModel",
    "spherical_harmonic_acceleration",
    "spherical_harmonic_gravity_gradient",
    "cbar_n0_from_unnormalized_jn",
]

# Below this radius (m) the field evaluation is nonphysical (inside/at centre);
# guard rather than divide by zero.
_MIN_RADIUS_M = 1.0


def cbar_n0_from_unnormalized_jn(jn: float, n: int) -> float:
    """Fully-normalized zonal coefficient from an unnormalized zonal Jn.

    ``Cbar_n0 = -Jn / sqrt(2n + 1)``.  For n = 2 this is the format-dependent
    bridge ``Cbar20 = -J2 / sqrt(5)`` used by the C20/J2 equivalence test.
    """
    return -float(jn) / math.sqrt(2.0 * n + 1.0)


def _dabar_factor(n: int, m: int) -> float:
    """f_nm in  dAbar_nm/du = f_nm * Abar_n,m+1  (fully normalized; m=0 seam)."""
    if m == 0:
        return math.sqrt(n * (n + 1) / 2.0)
    return math.sqrt((n - m) * (n + m + 1.0))


def _column_recursion(n: int, m: int) -> tuple[float, float]:
    """(alpha, beta) of the normalized column recursion, valid for n >= m + 2:
    Abar_nm = alpha * u * Abar_(n-1)m - beta * Abar_(n-2)m."""
    alpha = math.sqrt((2.0 * n - 1.0) * (2.0 * n + 1.0) / ((n - m) * (n + m)))
    beta = math.sqrt(
        (2.0 * n + 1.0) * (n + m - 1.0) * (n - m - 1.0)
        / ((2.0 * n - 3.0) * (n + m) * (n - m))
    )
    return alpha, beta


@dataclass(frozen=True, eq=False)
class SphericalHarmonicGravityModel:
    """A fully-normalized spherical-harmonic gravity model (immutable).

    Fields
    ------
    mu_m3_s2 : gravitational parameter of the model (m^3/s^2).
    r_ref_m  : reference radius the coefficients are paired with (m).  Must be
               the model's own radius -- do NOT substitute a body shape radius.
    cbar, sbar : fully-normalized coefficient arrays, square, indexed [n, m].
                 ``sbar[:, 0]`` must be zero (order m=0 has no sine term).
    nmax, mmax : evaluation truncation (nmax >= 2, 0 <= mmax <= nmax).  These
                 select how much of the (possibly larger) coefficient array is
                 used, so truncation is done by constructing models with
                 different nmax/mmax from the same arrays.
    frame    : informational label for the body-fixed frame (e.g. "MOON_PA").
               The engine does NOT apply this; the caller supplies the rotation.
    metadata : free-form provenance dict (source, checksum, ...); preserved
               verbatim.  Phase 12's loader populates it.

    NOTE: the arrays are stored by reference; do not mutate them after
    construction (the model is otherwise treated as immutable).
    """

    mu_m3_s2: float
    r_ref_m: float
    cbar: np.ndarray
    sbar: np.ndarray
    nmax: int
    mmax: int
    frame: str = "MOON_PA"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        cbar = np.ascontiguousarray(self.cbar, dtype=float)
        sbar = np.ascontiguousarray(self.sbar, dtype=float)
        if cbar.ndim != 2 or cbar.shape[0] != cbar.shape[1]:
            raise ValueError("cbar must be a square 2-D array indexed [n, m].")
        if sbar.shape != cbar.shape:
            raise ValueError("cbar and sbar must have identical shape.")
        n_avail = cbar.shape[0] - 1
        nmax = int(self.nmax)
        mmax = int(self.mmax)
        if nmax < 2:
            raise ValueError("nmax must be >= 2 (n=0 point mass and n=1 CoM are excluded).")
        if nmax > n_avail:
            raise ValueError(f"nmax={nmax} exceeds coefficient arrays (n_avail={n_avail}).")
        if not (0 <= mmax <= nmax):
            raise ValueError("mmax must satisfy 0 <= mmax <= nmax.")
        if not np.all(sbar[:, 0] == 0.0):
            raise ValueError("sbar[:, 0] must be zero (order m=0 has no sine term).")
        if not (self.mu_m3_s2 > 0.0 and self.r_ref_m > 0.0):
            raise ValueError("mu_m3_s2 and r_ref_m must be positive.")
        object.__setattr__(self, "cbar", cbar)
        object.__setattr__(self, "sbar", sbar)
        object.__setattr__(self, "mu_m3_s2", float(self.mu_m3_s2))
        object.__setattr__(self, "r_ref_m", float(self.r_ref_m))
        object.__setattr__(self, "nmax", nmax)
        object.__setattr__(self, "mmax", mmax)


def _pines_acceleration_bf(
    r_bf: np.ndarray,
    mu: float,
    r_ref: float,
    cbar: np.ndarray,
    sbar: np.ndarray,
    nmax: int,
    mmax: int,
) -> np.ndarray:
    """Pines perturbing acceleration (n >= 2) in the body-fixed frame, m/s^2.

    Self-contained scalar-loop routine (no polar singularity): uses direction
    cosines s = x/r, t = y/r, u = z/r and the normalized derived Legendre
    functions Abar_nm(u); longitude enters only through Rm/Im = Re/Im[(s+i t)^m].
    Structured for a later direct port to the Numba scalar-tuple pattern.
    """
    x, y, z = float(r_bf[0]), float(r_bf[1]), float(r_bf[2])
    r = math.sqrt(x * x + y * y + z * z)
    if r < _MIN_RADIUS_M:
        raise ValueError("position radius is too small for a harmonic field evaluation.")
    s, t, u = x / r, y / r, z / r

    # Normalized derived Legendre Abar[n, m]; one extra column so Abar_n,m+1 = 0
    # is available for the derivative relation dAbar_nm/du = f_nm * Abar_n,m+1.
    ab = np.zeros((nmax + 1, nmax + 2))
    ab[0, 0] = 1.0
    ab[1, 0] = math.sqrt(3.0) * u
    ab[1, 1] = math.sqrt(3.0)
    for n in range(2, nmax + 1):
        ab[n, n] = math.sqrt((2.0 * n + 1.0) / (2.0 * n)) * ab[n - 1, n - 1]
        ab[n, n - 1] = u * math.sqrt(2.0 * n) * ab[n, n]
    for m in range(0, nmax - 1):
        for n in range(m + 2, nmax + 1):
            alpha, beta = _column_recursion(n, m)
            ab[n, m] = alpha * u * ab[n - 1, m] - beta * ab[n - 2, m]

    # Rm/Im = Re/Im[(s + i t)^m]
    rm = np.zeros(nmax + 1)
    im = np.zeros(nmax + 1)
    rm[0] = 1.0
    for m in range(1, nmax + 1):
        rm[m] = s * rm[m - 1] - t * im[m - 1]
        im[m] = s * im[m - 1] + t * rm[m - 1]

    gx = gy = gz = 0.0
    rho = r_ref / r
    kn = (mu / (r * r)) * rho * rho          # (mu/r^2)(R/r)^n starting at n = 2
    for n in range(2, nmax + 1):
        m_top = min(n, mmax)
        for m in range(0, m_top + 1):
            cnm = cbar[n, m]
            snm = sbar[n, m]
            if cnm == 0.0 and snm == 0.0:
                continue
            d = cnm * rm[m] + snm * im[m]
            if m == 0:
                e = f = 0.0
            else:
                e = cnm * rm[m - 1] + snm * im[m - 1]
                f = snm * rm[m - 1] - cnm * im[m - 1]
            abp = _dabar_factor(n, m) * ab[n, m + 1]      # dAbar_nm/du
            lam = ((n + m + 1.0) * ab[n, m] + u * abp) * d
            gx += kn * (m * ab[n, m] * e - s * lam)
            gy += kn * (m * ab[n, m] * f - t * lam)
            gz += kn * (abp * d - u * lam)
        kn *= rho

    return np.array([gx, gy, gz])


def spherical_harmonic_acceleration(
    r_inertial_m: ArrayLike,
    model: SphericalHarmonicGravityModel,
    c_inertial_to_bf: ArrayLike | None = None,
) -> np.ndarray:
    """Perturbing acceleration (degrees n >= 2) in inertial axes, m/s^2.

    Parameters
    ----------
    r_inertial_m : (3,) Moon-centered position in inertial (propagation) axes, m.
    model : the fully-normalized :class:`SphericalHarmonicGravityModel`.  Its own
        ``mu_m3_s2`` and ``r_ref_m`` are used (never substituted).
    c_inertial_to_bf : optional (3, 3) rotation from inertial to the body-fixed
        frame the coefficients are defined in.  ``None`` -> identity (adequate
        only for single-epoch low-degree validation; see the module frame
        policy for the epoch-dependent MOON_PA requirement).

    Returns
    -------
    (3,) perturbing acceleration in inertial axes.  The n=0 point mass and n=1
    centre-of-mass terms are excluded by design.
    """
    r = np.asarray(r_inertial_m, dtype=float).reshape(3)
    if c_inertial_to_bf is None:
        r_bf = r
        a_bf = _pines_acceleration_bf(
            r_bf, model.mu_m3_s2, model.r_ref_m, model.cbar, model.sbar,
            model.nmax, model.mmax,
        )
        return a_bf
    c = np.asarray(c_inertial_to_bf, dtype=float).reshape(3, 3)
    r_bf = c @ r
    a_bf = _pines_acceleration_bf(
        r_bf, model.mu_m3_s2, model.r_ref_m, model.cbar, model.sbar,
        model.nmax, model.mmax,
    )
    return c.T @ a_bf


def _pines_gradient_bf(
    r_bf: np.ndarray,
    mu: float,
    r_ref: float,
    cbar: np.ndarray,
    sbar: np.ndarray,
    nmax: int,
    mmax: int,
) -> np.ndarray:
    """Analytic Pines gravity gradient d a / d r (n >= 2), body-fixed, 1/s^2.

    Derived from the SAME Pines expansion as ``_pines_acceleration_bf`` -- same
    normalization, same Abar recursion, same truncation semantics -- so the
    gradient represents exactly the force model the acceleration evaluates.

    Chain rule.  With e = (s, t, u) = r / |r| and k_n = (mu / r^2) (R/r)^n, the
    acceleration is a_i = sum_nm k_n(r) g_i(e).  Since

        d r / d x_j  = e_j ,
        d e_k / d x_j = (delta_kj - e_k e_j) / r ,
        d k_n / d r  = -(n + 2) k_n / r ,

    the gradient separates cleanly into a radial part and a transverse part:

        G = (da/dr) e^T + (1/r) M (I - e e^T),      M_ik = sum_nm k_n dg_i/de_k.

    The per-term derivatives use only quantities the acceleration recursion
    already forms, plus one extra Abar column: with
    Q = dAbar_nm/du = f_nm Abar_n,m+1 the second derivative is
    Q' = f_nm f_n,m+1 Abar_n,m+2, so the Abar table is carried to m + 2.

    The longitude derivatives close on themselves because
    d(R_m)/ds = m R_(m-1), d(R_m)/dt = -m I_(m-1), d(I_m)/ds = m I_(m-1),
    d(I_m)/dt = m R_(m-1), which makes dD/ds = m E and dD/dt = m F with the same
    E and F the acceleration already builds.  The m = 0 and m = 1 seams need no
    special casing: every term that would reach R_(-1) / I_(-1) carries an
    explicit m (m - 1) factor and therefore vanishes.
    """
    x, y, z = float(r_bf[0]), float(r_bf[1]), float(r_bf[2])
    r = math.sqrt(x * x + y * y + z * z)
    if r < _MIN_RADIUS_M:
        raise ValueError("position radius is too small for a harmonic field evaluation.")
    s, t, u = x / r, y / r, z / r

    # Abar[n, m]; two spare columns so Abar_n,m+1 and Abar_n,m+2 are available.
    ab = np.zeros((nmax + 1, nmax + 3))
    ab[0, 0] = 1.0
    ab[1, 0] = math.sqrt(3.0) * u
    ab[1, 1] = math.sqrt(3.0)
    for n in range(2, nmax + 1):
        ab[n, n] = math.sqrt((2.0 * n + 1.0) / (2.0 * n)) * ab[n - 1, n - 1]
        ab[n, n - 1] = u * math.sqrt(2.0 * n) * ab[n, n]
    for m in range(0, nmax - 1):
        for n in range(m + 2, nmax + 1):
            alpha, beta = _column_recursion(n, m)
            ab[n, m] = alpha * u * ab[n - 1, m] - beta * ab[n - 2, m]

    rm = np.zeros(nmax + 1)
    im = np.zeros(nmax + 1)
    rm[0] = 1.0
    for m in range(1, nmax + 1):
        rm[m] = s * rm[m - 1] - t * im[m - 1]
        im[m] = s * im[m - 1] + t * rm[m - 1]

    # da/dr (radial) and M = da/de (transverse), accumulated over the field.
    dadr = np.zeros(3)
    mmat = np.zeros((3, 3))

    rho = r_ref / r
    kn = (mu / (r * r)) * rho * rho
    for n in range(2, nmax + 1):
        m_top = min(n, mmax)
        for m in range(0, m_top + 1):
            cnm = cbar[n, m]
            snm = sbar[n, m]
            if cnm == 0.0 and snm == 0.0:
                continue
            p = ab[n, m]
            q = _dabar_factor(n, m) * ab[n, m + 1]                       # dAbar/du
            # d2Abar/du2 = f_nm f_n,m+1 Abar_n,m+2.  The factor f_n,m+1 is only
            # defined for m + 1 <= n; at m = n the whole term is identically
            # zero because Abar_n,m+2 = 0, so it is skipped rather than
            # evaluated (sqrt of a negative argument otherwise).
            if m + 1 <= n:
                qp = _dabar_factor(n, m) * _dabar_factor(n, m + 1) * ab[n, m + 2]
            else:
                qp = 0.0

            d = cnm * rm[m] + snm * im[m]
            if m == 0:
                e = f = 0.0
            else:
                e = cnm * rm[m - 1] + snm * im[m - 1]
                f = snm * rm[m - 1] - cnm * im[m - 1]
            if m >= 2:
                g2 = cnm * rm[m - 2] + snm * im[m - 2]
                h2 = snm * rm[m - 2] - cnm * im[m - 2]
            else:
                g2 = h2 = 0.0

            w = (n + m + 1.0) * p + u * q
            lam = w * d
            wu = (n + m + 2.0) * q + u * qp
            mm1 = m * (m - 1.0)

            gx = m * p * e - s * lam
            gy = m * p * f - t * lam
            gz = q * d - u * lam

            # radial part
            radial = -(n + 2.0) * kn / r
            dadr[0] += radial * gx
            dadr[1] += radial * gy
            dadr[2] += radial * gz

            # transverse part: dg_i/d(s, t, u)
            mmat[0, 0] += kn * (mm1 * p * g2 - lam - s * m * w * e)
            mmat[0, 1] += kn * (mm1 * p * h2 - s * m * w * f)
            mmat[0, 2] += kn * (m * q * e - s * d * wu)

            mmat[1, 0] += kn * (mm1 * p * h2 - t * m * w * e)
            mmat[1, 1] += kn * (-mm1 * p * g2 - lam - t * m * w * f)
            mmat[1, 2] += kn * (m * q * f - t * d * wu)

            mmat[2, 0] += kn * (m * e * (q - u * w))
            mmat[2, 1] += kn * (m * f * (q - u * w))
            mmat[2, 2] += kn * (qp * d - lam - u * d * wu)
        kn *= rho

    evec = np.array([s, t, u])
    return np.outer(dadr, evec) + (mmat - np.outer(mmat @ evec, evec)) / r


def spherical_harmonic_gravity_gradient(
    r_inertial_m: ArrayLike,
    model: SphericalHarmonicGravityModel,
    c_inertial_to_bf: ArrayLike | None = None,
) -> np.ndarray:
    """Gravity gradient d a / d r of the perturbation (n >= 2), inertial axes.

    Companion of :func:`spherical_harmonic_acceleration`: same model, same
    truncation, same frame convention.  With a body-fixed rotation ``C`` the
    inertial gradient is the similarity transform ``C^T G_bf C`` (the rotation
    is state-independent at a fixed epoch, exactly as for the body-J2 path).

    Returns
    -------
    (3, 3) gradient in 1/s^2.  Symmetric and trace-free up to round-off, since
    the field is the gradient of a Laplace-harmonic potential.

    Notes
    -----
    Providing this function does NOT by itself enable harmonics in STM /
    estimator propagation: that composition stays fail-closed at the
    ``dynamics`` entry points until explicitly qualified and enabled.
    """
    r = np.asarray(r_inertial_m, dtype=float).reshape(3)
    if c_inertial_to_bf is None:
        return _pines_gradient_bf(
            r, model.mu_m3_s2, model.r_ref_m, model.cbar, model.sbar,
            model.nmax, model.mmax,
        )
    c = np.asarray(c_inertial_to_bf, dtype=float).reshape(3, 3)
    g_bf = _pines_gradient_bf(
        c @ r, model.mu_m3_s2, model.r_ref_m, model.cbar, model.sbar,
        model.nmax, model.mmax,
    )
    return c.T @ g_bf @ c
