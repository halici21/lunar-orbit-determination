"""IERS Conventions (2010) solid Earth tide station displacement (Phase C3-I).

Phase C3 found the solid tide worth 0.23 m at DSN stations and 1.1 sigma of OD; production had none. This is
the conventional (tide-free) displacement of IERS 2010 section 7.1.1 (Mathews et al. 1997):

* step 1: degree-2 and degree-3 in-phase terms (eqs. 7.5, 7.6) with the latitude dependence of h2 and l2,
  the out-of-phase diurnal/semidiurnal terms (mantle anelasticity) and the L1 latitude-dependence terms;
* step 2: frequency-dependent corrections of the diurnal band (Table 7.3a) and the long-period band
  (Table 7.3b).

The step-1 corrections, the step-2 tables and the Doodson arguments are transcribed from pyTMD 3.0.9
(``pyTMD/predict/solid_earth.py``, ``pyTMD/astro.py``; MIT licence, Copyright (c) 2017 Tyler C Sutterley; see
LICENSES/pyTMD.txt). Inputs are Earth-fixed (ITRF93) Cartesian vectors in metres.
"""
from __future__ import annotations

import numpy as np

A_EARTH_M = 6378136.3
H2_0, L2_0, H3, L3 = 0.6078, 0.0847, 0.292, 0.015

#: Table 7.3a (reduced, as DEHANTTIDEINEL STEP2DIU): Doodson multipliers (tau, s, h, p, N', ps) and
#: corrections [mm]: radial in-phase, radial out-of-phase, transverse in-phase, transverse out-of-phase.
_DIURNAL = np.array([
    [1, -3, 0, 2, 0, 0, -0.01, 0.0, 0.0, 0.0], [1, -3, 2, 0, 0, 0, -0.01, 0.0, 0.0, 0.0],
    [1, -2, 0, 1, -1, 0, -0.02, 0.0, 0.0, 0.0], [1, -2, 0, 1, 0, 0, -0.08, 0.0, -0.01, 0.01],
    [1, -2, 2, -1, 0, 0, -0.02, 0.0, 0.0, 0.0], [1, -1, 0, 0, -1, 0, -0.10, 0.0, 0.0, 0.0],
    [1, -1, 0, 0, 0, 0, -0.51, 0.0, -0.02, 0.03], [1, -1, 2, 0, 0, 0, 0.01, 0.0, 0.0, 0.0],
    [1, 0, -2, 1, 0, 0, 0.01, 0.0, 0.0, 0.0], [1, 0, 0, -1, 0, 0, 0.02, 0.0, 0.0, 0.0],
    [1, 0, 0, 1, 0, 0, 0.06, 0.0, 0.0, 0.0], [1, 0, 0, 1, 1, 0, 0.01, 0.0, 0.0, 0.0],
    [1, 0, 2, -1, 0, 0, 0.01, 0.0, 0.0, 0.0], [1, 1, -3, 0, 0, 1, -0.06, 0.0, 0.0, 0.0],
    [1, 1, -2, 0, -1, 0, 0.01, 0.0, 0.0, 0.0], [1, 1, -2, 0, 0, 0, -1.23, -0.07, 0.06, 0.01],
    [1, 1, -1, 0, 0, -1, 0.02, 0.0, 0.0, 0.0], [1, 1, -1, 0, 0, 1, 0.04, 0.0, 0.0, 0.0],
    [1, 1, 0, 0, -1, 0, -0.22, 0.01, 0.01, 0.0], [1, 1, 0, 0, 0, 0, 12.00, -0.80, -0.67, -0.03],
    [1, 1, 0, 0, 1, 0, 1.73, -0.12, -0.10, 0.0], [1, 1, 0, 0, 2, 0, -0.04, 0.0, 0.0, 0.0],
    [1, 1, 1, 0, 0, -1, -0.50, -0.01, 0.03, 0.0], [1, 1, 1, 0, 0, 1, 0.01, 0.0, 0.0, 0.0],
    [1, 0, 1, 0, 1, -1, -0.01, 0.0, 0.0, 0.0], [1, 1, 2, -2, 0, 0, -0.01, 0.0, 0.0, 0.0],
    [1, 1, 2, 0, 0, 0, -0.11, 0.01, 0.01, 0.0], [1, 2, -2, 1, 0, 0, -0.01, 0.0, 0.0, 0.0],
    [1, 2, 0, -1, 0, 0, -0.02, 0.0, 0.0, 0.0], [1, 3, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0],
    [1, 3, 0, 0, 1, 0, 0.0, 0.0, 0.0, 0.0]])
#: Table 7.3b (reduced, as DEHANTTIDEINEL STEP2LON).
_LONG_PERIOD = np.array([
    [0, 0, 0, 0, 1, 0, 0.47, 0.23, 0.16, 0.07], [0, 0, 2, 0, 0, 0, -0.20, -0.12, -0.11, -0.05],
    [0, 1, 0, -1, 0, 0, -0.11, -0.08, -0.09, -0.04], [0, 2, 0, 0, 0, 0, -0.13, -0.11, -0.15, -0.07],
    [0, 2, 0, 0, 1, 0, -0.05, -0.05, -0.06, -0.03]])


def _poly(c, t):
    return sum(ci * t ** i for i, ci in enumerate(c))


def doodson_arguments(t_cent_tt: float, ut_hour: float) -> np.ndarray:
    """(tau, s, h, p, N', ps) [rad], IERS STEP2DIU/STEP2LON form: tau from the UT hour, the rest from TT
    Julian centuries since J2000."""
    t = t_cent_tt
    s = _poly([218.3164477, 481267.88123421, -1.5786e-3, 1.855835e-6, -1.53388e-8], t)
    lam = _poly([280.4606184, 36000.7700536, 3.8793e-4, -2.58e-8], t)
    tau = ut_hour * 15.0 - s + lam
    s = s + _poly([0.0, 1.396971278, 3.08889e-4, 2.1e-8, 7.0e-9], t)
    h = _poly([280.46645, 36000.7697489, 3.0322222e-4, 2.0e-8, -6.54e-9], t)
    p = _poly([83.3532465, 4069.0137287, -1.032172222e-2, -1.24991e-5, 5.263e-8], t)
    zns = _poly([234.95544499, 1934.13626197, -2.07561111e-3, -2.13944e-6, 1.65e-8], t)
    ps = _poly([282.93734098, 1.71945766667, 4.5688889e-4, -1.778e-8, -3.34e-9], t)
    return np.radians(np.mod(np.array([tau, s, h, p, zns, ps]), 360.0))


def _geo(r):
    rad = np.linalg.norm(r)
    sinphi = r[2] / rad
    cosphi = np.hypot(r[0], r[1]) / rad
    sinla = r[1] / (cosphi * rad)
    cosla = r[0] / (cosphi * rad)
    return rad, sinphi, cosphi, sinla, cosla


def _to_xyz(dr, dn, de, sinphi, cosphi, sinla, cosla):
    return np.array([dr * cosla * cosphi - de * sinla - dn * cosla * sinphi,
                     dr * sinla * cosphi + de * cosla - dn * sinla * sinphi,
                     dr * sinphi + dn * cosphi])


def _body_corrections(r, rb, f2):
    """Out-of-phase (eqs. 19-22) and latitude-dependence (eqs. 25-26) corrections of Mathews et al. (1997)
    for one body at ECEF ``rb`` with degree-2 factor ``f2``."""
    _, sp, cp, sl, cl = _geo(r)
    _, bsp, bcp, bsl, bcl = _geo(rb)
    sin2phi, cos2phi = 2.0 * sp * cp, cp * cp - sp * sp
    bsin2phi = 2.0 * bcp * bsp
    cos2la, sin2la = cl * cl - sl * sl, 2.0 * cl * sl
    bcos2la, bsin2la = bcl * bcl - bsl * bsl, 2.0 * bcl * bsl
    out = np.zeros(3)
    # out-of-phase, diurnal (dh2 = -0.0025, dl2 = -0.0007)
    dh, dl = -0.0025, -0.0007
    s_d = sl * bcl - cl * bsl
    c_d = cl * bcl + sl * bsl
    out += _to_xyz(-0.75 * dh * f2 * sin2phi * bsin2phi * s_d, -1.5 * dl * f2 * cos2phi * bsin2phi * s_d,
                   -1.5 * dl * f2 * sp * bsin2phi * c_d, sp, cp, sl, cl)
    # out-of-phase, semidiurnal (dh2 = -0.0022, dl2 = -0.0007)
    dh, dl = -0.0022, -0.0007
    s_s = sin2la * bcos2la - cos2la * bsin2la
    c_s = cos2la * bcos2la + sin2la * bsin2la
    out += _to_xyz(-0.75 * dh * f2 * cp * cp * bcp * bcp * s_s, 1.5 * dl * f2 * cp * sp * bcp * bcp * s_s,
                   -1.5 * dl * f2 * cp * bcp * bcp * c_s, sp, cp, sl, cl)
    # latitude dependence, diurnal (L1 = 0.0012)
    l1 = 0.0012
    out += _to_xyz(0.0, -1.5 * l1 * f2 * sp * sp * bsin2phi * c_d, 1.5 * l1 * f2 * sp * cos2phi * bsin2phi * s_d,
                   sp, cp, sl, cl)
    # latitude dependence, semidiurnal (L1 = 0.0024)
    l1 = 0.0024
    out += _to_xyz(0.0, -1.5 * l1 * f2 * sp * cp * bcp * bcp * c_s, -1.5 * l1 * f2 * sp * sp * cp * bcp * bcp * s_s,
                   sp, cp, sl, cl)
    return out


def _step2(r, args):
    _, sp, cp, sl, cl = _geo(r)
    sin2phi, cos2phi = 2.0 * sp * cp, cp * cp - sp * sp
    out = np.zeros(3)
    th = _DIURNAL[:, :6] @ args
    sph = np.sin(th) * cl + np.cos(th) * sl
    cph = np.cos(th) * cl - np.sin(th) * sl
    dr_ip, dr_op, dt_ip, dt_op = _DIURNAL[:, 6], _DIURNAL[:, 7], _DIURNAL[:, 8], _DIURNAL[:, 9]
    dr = sin2phi * (dr_ip * sph + dr_op * cph)
    dn = cos2phi * (dt_ip * sph + dt_op * cph)
    de = sp * (dt_ip * cph - dt_op * sph)
    out += _to_xyz(dr.sum(), dn.sum(), de.sum(), sp, cp, sl, cl)
    th = _LONG_PERIOD[:, :6] @ args
    dr_ip, dr_op, dt_ip, dt_op = (_LONG_PERIOD[:, k] for k in (6, 7, 8, 9))
    dr = (1.5 * sp * sp - 0.5) * (dt_ip * np.sin(th) + dr_ip * np.cos(th))
    dn = sin2phi * (dt_op * np.sin(th) + dr_op * np.cos(th))
    out += _to_xyz(dr.sum(), dn.sum(), 0.0, sp, cp, sl, cl)
    return out * 1e-3


def solid_tide_displacement_ecef(r_station_ecef_m, r_sun_ecef_m, r_moon_ecef_m, mjd_tt: float, ut_fraction: float,
                                 *, mass_ratio_sun: float, mass_ratio_moon: float) -> np.ndarray:
    """Conventional tide-free solid Earth tide displacement [m] (ECEF) of a station; ``mass_ratio_*`` = GM_body /
    GM_Earth; ``mjd_tt`` the TT Modified Julian Date, ``ut_fraction`` the UT fraction of the day."""
    r = np.asarray(r_station_ecef_m, dtype=float)
    rad = np.linalg.norm(r)
    rh = r / rad
    sinphi = r[2] / rad
    p2 = 1.5 * sinphi * sinphi - 0.5
    h2, l2 = H2_0 - 0.0006 * p2, L2_0 + 0.0002 * p2
    total = np.zeros(3)
    for rb_, ratio in ((r_sun_ecef_m, mass_ratio_sun), (r_moon_ecef_m, mass_ratio_moon)):
        rb = np.asarray(rb_, dtype=float)
        rbn = np.linalg.norm(rb)
        bh = rb / rbn
        u = float(bh @ rh)
        f2 = ratio * A_EARTH_M ** 4 / rbn ** 3
        f3 = ratio * A_EARTH_M ** 5 / rbn ** 4
        total += f2 * (h2 * rh * (1.5 * u * u - 0.5) + 3.0 * l2 * u * (bh - u * rh))
        total += f3 * (H3 * rh * (2.5 * u ** 3 - 1.5 * u) + L3 * (7.5 * u * u - 1.5) * (bh - u * rh))
        total += _body_corrections(r, rb, f2)
    t_cent = (mjd_tt - 51544.5) / 36525.0
    total += _step2(r, doodson_arguments(t_cent, 24.0 * (ut_fraction % 1.0)))
    return total


_GM: dict = {}


def _mass_ratios():
    if not _GM:
        import spiceypy as spice
        ge = spice.bodvrd("EARTH", "GM", 1)[1][0]
        _GM["sun"] = spice.bodvrd("SUN", "GM", 1)[1][0] / ge
        _GM["moon"] = spice.bodvrd("MOON", "GM", 1)[1][0] / ge
    return _GM["sun"], _GM["moon"]


def solid_tide_at_et(r_station_ecef_m, et_s: float) -> np.ndarray:
    """The displacement at TDB epoch ``et_s`` with SPICE Sun and Moon (geometric, ITRF93) and UT from UTC."""
    import spiceypy as spice
    from . import station_time as _st
    et = float(et_s)
    sun = np.asarray(spice.spkpos("SUN", et, "ITRF93", "NONE", "EARTH")[0]) * 1e3
    moon = np.asarray(spice.spkpos("MOON", et, "ITRF93", "NONE", "EARTH")[0]) * 1e3
    d_geo = float(_st.tdb_minus_tt(et, 0.0))
    tt = et - d_geo
    tai = tt - _st.TT_MINUS_TAI_S
    dat = float(_st.tai_minus_utc(tai))
    dat = float(_st.tai_minus_utc(tai - dat))
    utc = tai - dat
    ms, mm = _mass_ratios()
    return solid_tide_displacement_ecef(r_station_ecef_m, sun, moon, 51544.5 + tt / 86400.0,
                                        ((utc + 43200.0) / 86400.0) % 1.0, mass_ratio_sun=ms, mass_ratio_moon=mm)
