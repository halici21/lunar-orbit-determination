"""Phase 11A -- Low-Degree Spherical Harmonics Prototype (Pines vs Cunningham).

Purpose
-------
Compare two candidate mathematical formulations for the future lunar
spherical-harmonic gravity engine on tiny, hand-written, fully-normalized
coefficient sets (nmax <= 3, plus an nmax=8 recursion cross-check):

1. **Pines formulation** (primary candidate): Cartesian direction-cosine
   method with normalized "derived" Legendre functions Abar_nm(u).  No polar
   singularity anywhere in the algorithm.
2. **Normalized Cunningham-style classical formulation** (reference /
   cross-check): spherical-gradient method with normalized associated
   Legendre functions Pbar_nm(sin phi) and the (r, phi, lambda) chain rule.
   NOTE on naming: this is the *classical spherical-gradient* algorithm the
   Phase 10/11A plan called "normalized Cunningham"; it is distinct from
   Cunningham's 1970 complex V/W recursion.  It carries the well-known
   1/cos(phi) polar singularity, handled here with an explicit clamp.

Scope / constraints (Phase 11A, approved)
-----------------------------------------
- Prototype ONLY.  No production module; ``lunar_od/dynamics.py``,
  ``accelerated.py``, ``force_models.py``, ``scenario_config.py`` untouched.
- No GRAIL file download, no coefficient-file loader.
- Fixed frame / identity rotation only.  **This is acceptable ONLY for a
  mathematical prototype**: real m > 0 (tesseral/sectoral) lunar harmonics
  propagation REQUIRES an epoch-dependent inertial->MOON_PA rotation
  (SPICE ``pxform("J2000", "MOON_PA", et)``; kernel
  ``moon_pa_de421_1900-2050.bpc`` is already available).  The constant
  ``_MCI_TO_MOON_BF`` used by the production Moon-J2 path is NOT sufficient
  for longitude-dependent terms; that rotation provider is Phase 12/13 scope.
- No commit / push.

Conventions
-----------
- SI units: position in metres, GM in m^3/s^2, reference radius in metres;
  returned acceleration in m/s^2.
- Coefficients are **fully normalized** Cbar_nm / Sbar_nm, stored as square
  2-D arrays indexed ``cbar[n, m]`` (rows n, columns m; upper triangle unused).
- Fully-normalized zonal relation: ``Jn = -sqrt(2n+1) * Cbar_n0``; for J2
  specifically ``Cbar20 = -J2 / sqrt(5)`` (NOT ``-sqrt(5) * J2``).
- Both engines evaluate ONLY the perturbation (degrees n >= 2).  The n = 0
  point-mass and n = 1 (centre-of-mass) rows are ignored by design; the call
  site owns the point-mass term, exactly like the existing J2 helpers.

Run:  python examples/phase11a_harmonics_prototype.py
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Relative clamp on the equatorial distance rho = sqrt(x^2+y^2) used by the
# classical (Cunningham-style) formulation to avoid division by zero at the
# exact pole.  This keeps the evaluation finite there but NOT correct: the
# m = 1 horizontal terms are simply lost (documented limitation, see report).
CUNNINGHAM_POLE_EPS = 1e-12


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _prepare_inputs(r_m, cbar, sbar, nmax, mmax, c_inertial_to_bf):
    """Validate/normalize common arguments; returns (r_bf, cbar, sbar, nmax, mmax, c)."""
    r = np.asarray(r_m, dtype=float).reshape(3)
    cbar = np.asarray(cbar, dtype=float)
    sbar = np.asarray(sbar, dtype=float)
    if cbar.shape != sbar.shape or cbar.ndim != 2 or cbar.shape[0] != cbar.shape[1]:
        raise ValueError("cbar/sbar must be equal square 2-D arrays indexed [n, m].")
    n_avail = cbar.shape[0] - 1
    nmax = n_avail if nmax is None else int(nmax)
    if nmax > n_avail:
        raise ValueError(f"nmax={nmax} exceeds coefficient arrays (n_avail={n_avail}).")
    mmax = nmax if mmax is None else min(int(mmax), nmax)
    if c_inertial_to_bf is None:
        c = None
        r_bf = r
    else:
        c = np.asarray(c_inertial_to_bf, dtype=float).reshape(3, 3)
        r_bf = c @ r
    return r_bf, cbar, sbar, nmax, mmax, c


def _dabar_factor(n: int, m: int) -> float:
    """f_nm with dAbar_nm/du = f_nm * Abar_n,m+1 (fully normalized; m=0 seam)."""
    if m == 0:
        return math.sqrt(n * (n + 1) / 2.0)
    return math.sqrt((n - m) * (n + m + 1.0))


def _column_recursion_coeffs(n: int, m: int) -> tuple[float, float]:
    """(alpha, beta) of the normalized column recursion, valid for n >= m+2:
    Xbar_nm = alpha * u * Xbar_(n-1)m - beta * Xbar_(n-2)m   (same for Abar and Pbar)."""
    alpha = math.sqrt((2.0 * n - 1.0) * (2.0 * n + 1.0) / ((n - m) * (n + m)))
    beta = math.sqrt(
        (2.0 * n + 1.0) * (n + m - 1.0) * (n - m - 1.0)
        / ((2.0 * n - 3.0) * (n + m) * (n - m))
    )
    return alpha, beta


# ---------------------------------------------------------------------------
# 1) Pines formulation (primary candidate)
# ---------------------------------------------------------------------------
def pines_acceleration(r_m, mu, r_ref, cbar, sbar, nmax=None, mmax=None,
                       c_inertial_to_bf=None):
    """Perturbing acceleration (n >= 2) via the normalized Pines formulation.

    Cartesian direction cosines s = x/r, t = y/r, u = z/r with normalized
    derived Legendre functions Abar_nm(u) = Nbar_nm * Pnm(u) / (1-u^2)^(m/2).
    Longitude enters through Rm/Im = Re/Im[(s + i t)^m]; latitude never appears
    as an angle, so there is NO polar singularity.

    Parameters: r_m [m] position relative to the body centre (inertial axes if
    ``c_inertial_to_bf`` given, else already body-fixed); mu [m^3/s^2];
    r_ref [m] reference radius paired with the coefficients; cbar/sbar
    fully-normalized coefficients ``[n, m]``.  Returns (3,) m/s^2.
    """
    r_bf, cbar, sbar, nmax, mmax, c = _prepare_inputs(
        r_m, cbar, sbar, nmax, mmax, c_inertial_to_bf)
    if nmax < 2:
        return np.zeros(3)

    x, y, z = r_bf
    r = math.sqrt(x * x + y * y + z * z)
    s, t, u = x / r, y / r, z / r

    # Normalized derived Legendre Abar[n, m]; one extra column so Abar_n,n+1=0
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
            alpha, beta = _column_recursion_coeffs(n, m)
            ab[n, m] = alpha * u * ab[n - 1, m] - beta * ab[n - 2, m]

    # Rm/Im = Re/Im[(s + i t)^m]  (Rm = cos^m(phi) cos(m lam) etc.)
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
        for m in range(0, min(n, mmax) + 1):
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

    a_bf = np.array([gx, gy, gz])
    return a_bf if c is None else c.T @ a_bf


# ---------------------------------------------------------------------------
# 2) Normalized Cunningham-style classical formulation (reference)
# ---------------------------------------------------------------------------
def cunningham_acceleration(r_m, mu, r_ref, cbar, sbar, nmax=None, mmax=None,
                            c_inertial_to_bf=None):
    """Perturbing acceleration (n >= 2) via the classical spherical-gradient
    formulation with fully-normalized Pbar_nm(sin phi) (plan name: "normalized
    Cunningham"; distinct from Cunningham's 1970 complex V/W recursion).

    Computes dU/dr, dU/dphi, dU/dlambda and maps to Cartesian with the chain
    rule.  The 1/rho^2 (= 1/(r cos phi))^2) and tan(phi) factors are SINGULAR
    at the poles; rho is clamped at ``CUNNINGHAM_POLE_EPS * r`` so the result
    stays finite there, but exact-pole values are NOT trustworthy for m > 0
    terms (the m = 1 horizontal contribution is lost entirely because it is
    multiplied by x = y = 0).  Documented prototype limitation.
    """
    r_bf, cbar, sbar, nmax, mmax, c = _prepare_inputs(
        r_m, cbar, sbar, nmax, mmax, c_inertial_to_bf)
    if nmax < 2:
        return np.zeros(3)

    x, y, z = r_bf
    r = math.sqrt(x * x + y * y + z * z)
    rho_xy = math.hypot(x, y)
    rho_eff = max(rho_xy, CUNNINGHAM_POLE_EPS * r)   # polar clamp (see docstring)

    sin_phi = z / r
    cos_phi = rho_eff / r
    tan_phi = sin_phi / cos_phi
    lam = math.atan2(y, x)

    # Normalized associated Legendre Pbar[n, m](sin phi); one extra column so
    # Pbar_n,n+1 = 0 is available for the derivative relation.
    p = np.zeros((nmax + 1, nmax + 2))
    p[0, 0] = 1.0
    p[1, 0] = math.sqrt(3.0) * sin_phi
    p[1, 1] = math.sqrt(3.0) * cos_phi
    for n in range(2, nmax + 1):
        p[n, n] = math.sqrt((2.0 * n + 1.0) / (2.0 * n)) * cos_phi * p[n - 1, n - 1]
        p[n, n - 1] = math.sqrt(2.0 * n + 1.0) * sin_phi * p[n - 1, n - 1]
    for m in range(0, nmax - 1):
        for n in range(m + 2, nmax + 1):
            alpha, beta = _column_recursion_coeffs(n, m)
            p[n, m] = alpha * sin_phi * p[n - 1, m] - beta * p[n - 2, m]

    du_dr = du_dphi = du_dlam = 0.0
    rho = r_ref / r
    pw = rho * rho                                    # (R/r)^n starting at n = 2
    for n in range(2, nmax + 1):
        for m in range(0, min(n, mmax) + 1):
            cnm = cbar[n, m]
            snm = sbar[n, m]
            if cnm == 0.0 and snm == 0.0:
                continue
            cml = math.cos(m * lam)
            sml = math.sin(m * lam)
            trig = cnm * cml + snm * sml
            trig_l = snm * cml - cnm * sml
            # dPbar_nm/dphi = f_nm * Pbar_n,m+1 - m * tan(phi) * Pbar_nm
            dp = _dabar_factor(n, m) * p[n, m + 1] - m * tan_phi * p[n, m]
            du_dr += -(n + 1.0) * pw * p[n, m] * trig
            du_dphi += pw * dp * trig
            du_dlam += pw * m * p[n, m] * trig_l
        pw *= rho

    du_dr *= mu / (r * r)
    du_dphi *= mu / r
    du_dlam *= mu / r

    # Chain rule (r, phi, lambda) -> (x, y, z); a = grad(U), U = +mu/r convention.
    ax = du_dr * (x / r) + du_dphi * (-x * z / (r * r * rho_eff)) \
        + du_dlam * (-y / (rho_eff * rho_eff))
    ay = du_dr * (y / r) + du_dphi * (-y * z / (r * r * rho_eff)) \
        + du_dlam * (x / (rho_eff * rho_eff))
    az = du_dr * (z / r) + du_dphi * (rho_eff / (r * r))

    a_bf = np.array([ax, ay, az])
    return a_bf if c is None else c.T @ a_bf


# ---------------------------------------------------------------------------
# Synthetic coefficient sets (NOT a real gravity model)
# ---------------------------------------------------------------------------
def cbar_n0_from_jn(jn: float, n: int) -> float:
    """Fully-normalized zonal from unnormalized Jn:  Cbar_n0 = -Jn / sqrt(2n+1)."""
    return -jn / math.sqrt(2.0 * n + 1.0)


def nominal_coefficients(nmax: int = 3):
    """Fixed synthetic nmax=3 set with loosely Moon-like magnitudes.

    Values are hand-picked for testing only (nonzero in every (n, m) slot so
    all recursion branches are exercised); they are NOT a real lunar model.
    """
    cbar = np.zeros((nmax + 1, nmax + 1))
    sbar = np.zeros((nmax + 1, nmax + 1))
    cbar[2, 0] = cbar_n0_from_jn(2.0346e-4, 2)   # ~ Moon J2 -> Cbar20 = -J2/sqrt(5)
    cbar[2, 1] = 1.2e-8
    sbar[2, 1] = -3.0e-9
    cbar[2, 2] = 3.47e-5
    sbar[2, 2] = 1.0e-6
    if nmax >= 3:
        cbar[3, 0] = cbar_n0_from_jn(8.46e-6, 3)  # ~ Moon J3 -> Cbar30 = -J3/sqrt(7)
        cbar[3, 1] = 2.6e-5
        sbar[3, 1] = 5.5e-6
        cbar[3, 2] = 1.4e-5
        sbar[3, 2] = 4.9e-6
        cbar[3, 3] = 1.2e-5
        sbar[3, 3] = -2.0e-6
    return cbar, sbar


def random_coefficients(nmax: int, scale: float = 1e-6, seed: int = 20260702):
    """Seeded random normalized coefficients for recursion cross-checks."""
    rng = np.random.default_rng(seed)
    cbar = np.zeros((nmax + 1, nmax + 1))
    sbar = np.zeros((nmax + 1, nmax + 1))
    for n in range(2, nmax + 1):
        for m in range(0, n + 1):
            cbar[n, m] = scale * rng.standard_normal()
            if m > 0:
                sbar[n, m] = scale * rng.standard_normal()
    return cbar, sbar


def _pos(lat_deg: float, lon_deg: float, radius_m: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    return radius_m * np.array(
        [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)]
    )


# ---------------------------------------------------------------------------
# Comparison battery / report
# ---------------------------------------------------------------------------
def main() -> int:
    from lunar_od.constants import J2_MOON_UNNORMALIZED, MU_MOON_M3S2, R_MOON_M
    from lunar_od.force_models import body_j2_acceleration

    mu, r_ref, j2 = MU_MOON_M3S2, R_MOON_M, J2_MOON_UNNORMALIZED
    r_llo = r_ref + 100e3
    funcs = {"Pines": pines_acceleration, "Cunningham": cunningham_acceleration}
    identity = np.eye(3)
    failures: list[str] = []
    results: dict[str, dict[str, str]] = {name: {} for name in funcs}

    def check(name: str, label: str, ok: bool, detail: str) -> None:
        results[name][label] = f"{'PASS' if ok else 'FAIL'}  {detail}"
        if not ok:
            failures.append(f"{name} / {label}: {detail}")

    zeros3 = (np.zeros((4, 4)), np.zeros((4, 4)))
    cbar_j2 = np.zeros((3, 3))
    cbar_j2[2, 0] = cbar_n0_from_jn(j2, 2)
    sbar_j2 = np.zeros((3, 3))
    cbar_nom, sbar_nom = nominal_coefficients(3)
    probe_positions = [
        _pos(0.0, 0.0, r_llo), _pos(0.0, 90.0, r_llo), _pos(35.0, 140.0, r_llo),
        _pos(-60.0, 250.0, r_llo), _pos(90.0, 0.0, r_llo), _pos(20.0, 300.0, r_ref + 2000e3),
    ]

    for name, func in funcs.items():
        # T1 -- zero coefficients (and ignored n=0/1 rows) -> exactly zero
        a0 = func(probe_positions[2], mu, r_ref, *zeros3)
        c01 = np.zeros((4, 4)); c01[0, 0] = 1.0; c01[1, 1] = 0.5; c01[1, 0] = 0.3
        a01 = func(probe_positions[2], mu, r_ref, c01, np.zeros((4, 4)))
        check(name, "T1 zero-coeff -> zero", float(np.max(np.abs(a0))) == 0.0
              and float(np.max(np.abs(a01))) == 0.0,
              f"|a|={np.max(np.abs(a0)):.1e}, n<2 rows |a|={np.max(np.abs(a01)):.1e}")

        # T2 -- C20-only vs existing Moon-J2 helper (< 1e-14 m/s^2)
        worst = 0.0
        for rr in probe_positions:
            a_e = func(rr, mu, r_ref, cbar_j2, sbar_j2, nmax=2)
            a_r = body_j2_acceleration(rr, mu, r_ref, j2, identity)
            worst = max(worst, float(np.linalg.norm(a_e - a_r)))
        check(name, "T2 C20-only == J2 helper", worst < 1e-14,
              f"worst |da| = {worst:.3e} m/s^2 (tol 1e-14)")

        # T3 -- wrong-sign C20 is caught (perturbation flips sign; ~2x vs ref)
        cbar_bad = np.zeros((3, 3)); cbar_bad[2, 0] = -cbar_j2[2, 0]
        r3 = probe_positions[2]
        a_bad = func(r3, mu, r_ref, cbar_bad, sbar_j2, nmax=2)
        a_ref = body_j2_acceleration(r3, mu, r_ref, j2, identity)
        rel = float(np.linalg.norm(a_bad - a_ref) / np.linalg.norm(a_ref))
        check(name, "T3 wrong-sign C20 caught", rel > 1.0,
              f"|a_wrong - a_ref|/|a_ref| = {rel:.3f} (expect ~2)")

        # T5 -- C22 longitude dependence + C20 axisymmetry
        cbar_c22 = np.zeros((3, 3)); cbar_c22[2, 2] = 3.47e-5
        a_l0 = func(_pos(0, 0, r_llo), mu, r_ref, cbar_c22, sbar_j2)
        a_l90 = func(_pos(0, 90, r_llo), mu, r_ref, cbar_c22, sbar_j2)
        a_l180 = func(_pos(0, 180, r_llo), mu, r_ref, cbar_c22, sbar_j2)
        rad0 = float(a_l0 @ _pos(0, 0, r_llo)) / r_llo
        rad90 = float(a_l90 @ _pos(0, 90, r_llo)) / r_llo
        rad180 = float(a_l180 @ _pos(0, 180, r_llo)) / r_llo
        ok22 = (not np.allclose(rad0, rad90, rtol=1e-6)
                and np.isclose(rad90, -rad0, rtol=1e-10)
                and np.isclose(rad180, rad0, rtol=1e-10))
        a_z0 = func(_pos(0, 0, r_llo), mu, r_ref, cbar_j2, sbar_j2)
        a_z90 = func(_pos(0, 90, r_llo), mu, r_ref, cbar_j2, sbar_j2)
        okz = np.isclose(np.linalg.norm(a_z0), np.linalg.norm(a_z90), rtol=1e-12)
        check(name, "T5 C22 longitude dep.", ok22 and okz,
              f"radial(0/90/180 deg) = {rad0:.3e}/{rad90:.3e}/{rad180:.3e}; "
              f"C20 axisym {'ok' if okz else 'BROKEN'}")

        # T6 -- C30 north-south asymmetry (odd zonal: az symmetric, ax antisym.)
        cbar_c30 = np.zeros((4, 4)); cbar_c30[3, 0] = cbar_n0_from_jn(8.46e-6, 3)
        a_n = func(_pos(45, 0, r_llo), mu, r_ref, cbar_c30, np.zeros((4, 4)))
        a_s = func(_pos(-45, 0, r_llo), mu, r_ref, cbar_c30, np.zeros((4, 4)))
        ok30 = (np.isclose(a_n[2], a_s[2], rtol=1e-12) and abs(a_n[2]) > 0.0
                and np.isclose(a_n[0], -a_s[0], rtol=1e-12))
        check(name, "T6 C30 N-S asymmetry", ok30,
              f"az(N)={a_n[2]:.3e} az(S)={a_s[2]:.3e} (equal => mirror sym. broken)")

        # T8 -- unit mistake (km input) gives grossly different magnitude
        r_ok = _pos(20, 40, r_llo)
        a_m = func(r_ok, mu, r_ref, cbar_nom, sbar_nom)
        a_km = func(r_ok / 1000.0, mu, r_ref, cbar_nom, sbar_nom)
        ratio = float(np.linalg.norm(a_km) / np.linalg.norm(a_m))
        check(name, "T8 unit-mistake canary", ratio > 1e6,
              f"|a(km)|/|a(m)| = {ratio:.2e} (>1e6 detectable)")

    # T4 -- pole behaviour --------------------------------------------------
    for pole in (_pos(90, 0, r_llo), _pos(-90, 0, r_llo)):
        a_p = pines_acceleration(pole, mu, r_ref, cbar_nom, sbar_nom)
        near = _pos(math.copysign(89.9999, pole[2]), 0, r_llo)
        a_np = pines_acceleration(near, mu, r_ref, cbar_nom, sbar_nom)
        cont = float(np.linalg.norm(a_p - a_np) / np.linalg.norm(a_np))
        check("Pines", f"T4 pole ({'N' if pole[2] > 0 else 'S'})",
              bool(np.all(np.isfinite(a_p))) and np.linalg.norm(a_p) > 0 and cont < 1e-4,
              f"finite, |a|={np.linalg.norm(a_p):.3e}, continuity vs 89.9999deg = {cont:.1e}")
    a_pole_c = cunningham_acceleration(_pos(90, 0, r_llo), mu, r_ref, cbar_nom, sbar_nom)
    a_pole_p = pines_acceleration(_pos(90, 0, r_llo), mu, r_ref, cbar_nom, sbar_nom)
    pole_rel = float(np.linalg.norm(a_pole_c - a_pole_p) / np.linalg.norm(a_pole_p))
    hor_c = float(np.hypot(a_pole_c[0], a_pole_c[1]))
    hor_p = float(np.hypot(a_pole_p[0], a_pole_p[1]))
    # Expected documented behaviour: finite (clamp works) but the m=1 horizontal
    # acceleration is essentially lost -> large relative error vs Pines.
    check("Cunningham", "T4 exact pole (documented)",
          bool(np.all(np.isfinite(a_pole_c))) and hor_p > 0.0
          and hor_c < 1e-2 * hor_p and pole_rel > 1e-3,
          f"finite but WRONG: horizontal |a| = {hor_c:.1e} vs Pines {hor_p:.3e} "
          f"(lost); rel. error vs Pines = {pole_rel:.2e} -- singularity is real")
    a_np_c = cunningham_acceleration(_pos(89.99, 0, r_llo), mu, r_ref, cbar_nom, sbar_nom)
    a_np_p = pines_acceleration(_pos(89.99, 0, r_llo), mu, r_ref, cbar_nom, sbar_nom)
    near_rel = float(np.linalg.norm(a_np_c - a_np_p) / np.linalg.norm(a_np_p))
    check("Cunningham", "T4 near pole 89.99deg", near_rel < 1e-9,
          f"rel diff vs Pines = {near_rel:.2e}")

    # T7 -- cross-formulation consistency ------------------------------------
    worst3 = 0.0
    for lat in (-80, -45, -20, 0, 20, 45, 80):
        for lon in range(0, 360, 45):
            for alt in (50e3, 500e3):
                rr = _pos(lat, lon, r_ref + alt)
                ap = pines_acceleration(rr, mu, r_ref, cbar_nom, sbar_nom)
                ac = cunningham_acceleration(rr, mu, r_ref, cbar_nom, sbar_nom)
                worst3 = max(worst3, float(np.linalg.norm(ap - ac) / np.linalg.norm(ap)))
    check("Pines", "T7 vs Cunningham (nmax=3)", worst3 < 1e-12,
          f"worst rel diff = {worst3:.2e} over 112-point grid")
    cbar8, sbar8 = random_coefficients(8)
    worst8 = 0.0
    for lat in (-70, -30, 0, 40, 75):
        for lon in (10, 100, 200, 305):
            rr = _pos(lat, lon, r_llo)
            ap = pines_acceleration(rr, mu, r_ref, cbar8, sbar8)
            ac = cunningham_acceleration(rr, mu, r_ref, cbar8, sbar8)
            worst8 = max(worst8, float(np.linalg.norm(ap - ac) / np.linalg.norm(ap)))
    check("Pines", "T7b vs Cunningham (nmax=8)", worst8 < 1e-11,
          f"worst rel diff = {worst8:.2e} (seeded random coeffs)")

    # Timing (informational; pure Python, no Numba) ---------------------------
    timing = {}
    for name, func in funcs.items():
        for label, (cb, sb) in (("nmax=3", (cbar_nom, sbar_nom)),
                                ("nmax=8", (cbar8, sbar8))):
            rr = _pos(30, 60, r_llo)
            n_eval = 2000
            t0 = time.perf_counter()
            for _ in range(n_eval):
                func(rr, mu, r_ref, cb, sb)
            timing[(name, label)] = (time.perf_counter() - t0) / n_eval * 1e6

    # Report ------------------------------------------------------------------
    print("=" * 78)
    print("Phase 11A -- Pines vs normalized Cunningham prototype comparison")
    print("=" * 78)
    for name in funcs:
        print(f"\n--- {name} ---")
        for label, line in results[name].items():
            print(f"  {label:32s} {line}")
    print("\n--- Timing (pure Python prototype, per call) ---")
    for (name, label), us in timing.items():
        print(f"  {name:11s} {label}: {us:8.1f} us")

    print("\n--- Recommendation ---")
    print("  Selected formulation: PINES")
    print("  - No polar singularity: exact-pole evaluation is clean (T4), which the")
    print("    classical formulation demonstrably is not (m=1 horizontal terms lost")
    print("    at the exact pole despite the rho clamp). Polar LLO is a primary")
    print("    use case, so this is decisive.")
    print("  - Same normalized column recursion, comparable cost and clarity;")
    print("    loop-and-scalar structure ports directly to the Numba pattern used")
    print("    in accelerated.py (Phase 4 scalar-core style).")
    print("  - Cunningham-style implementation is kept in this prototype as an")
    print("    independent cross-check oracle (T7/T7b agreement at 1e-12/1e-11).")
    print("\n--- Risks remaining before a production module (Phase 11B+) ---")
    print("  1. Epoch-dependent J2000->MOON_PA rotation REQUIRED for m>0 terms in")
    print("     real propagation (fixed frame here is prototype-only).")
    print("  2. J2/harmonics double-count guard (ValueError) at composition time.")
    print("  3. Numba twin + Python<->Numba parity tests (Phase 4 contract).")
    print("  4. Real-model GM/R_ref pairing (GRAIL R_ref=1738.0 km != R_MOON_M).")
    print("  5. High-degree (n>~150) recursion scaling out of scope (nmax<=64 plan).")

    print("\n" + "=" * 78)
    if failures:
        print(f"RESULT: FAIL ({len(failures)} failed check(s))")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("RESULT: ALL CHECKS PASS  (production code untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
