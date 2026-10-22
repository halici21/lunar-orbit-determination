"""Station time: UTC -> TAI -> TT -> TDB with the full TDB-TT series and its topocentric terms (Phase C2-I).

Phase C2 (``studies/phase17/reports/phase_c2_time_scale_materiality.md``) found that SPICE's one-term DELTET
model of TDB - TT is 12-16 us from the Fairhead-Bretagnon series (1.8 sigma of OD once station tags are
converted from UTC), and that the station-dependent part of TDB - TT, absent from production, is worth
0.6 sigma (its rate scales every station-measured interval). This module provides:

* :func:`dtdb` - a port of ERFA ``eraDtdb`` (same inputs, same algorithm; coefficients generated from the
  ERFA 2.0.1 C source into :mod:`lunar_od._erfa_dtdb_table`), and :func:`dtdb_rate`, its analytic derivative;
* :func:`tdb_minus_tt` / :func:`tdb_minus_tt_rate` - the same at a production epoch ``et0 + t`` (seconds of
  TDB past J2000), with the epoch kept in two parts so ``et0 + t`` is never formed;
* :func:`utc_to_tdb_seconds` - UTC station tags to TDB seconds past ``et0`` (TAI - UTC from the loaded
  leapseconds kernel, TT = TAI + 32.184 s, TDB = TT + :func:`dtdb` at the site).

``ut`` (the local-solar-time argument of the topocentric terms) is UTC, not UT1; |UT1 - UTC| < 0.9 s moves
the topocentric term by < 2e-10 s. Dates before 1972 (no integral TAI - UTC) are rejected.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from . import _erfa_dtdb_table as _T

DJ00 = 2451545.0
DJM = 365250.0
DAYSEC = 86400.0
D2PI = 6.283185307179586476925287
DD2R = 1.745329251994329576923691e-2
TT_MINUS_TAI_S = 32.184
_DT_DTMILL = 1.0 / (DJM * DAYSEC)          # d(Julian millennia) / d(second)
_DTSOL_DT = D2PI / DAYSEC                  # d(local solar time) / d(second), rad/s

_A = np.array([r[0] for r in _T.FAIRHD])
_F = np.array([r[1] for r in _T.FAIRHD])
_P = np.array([r[2] for r in _T.FAIRHD])
_GROUPS = tuple(slice(lo, hi + 1) for lo, hi in _T.GROUPS)
_CHUNK = 2048


@dataclass(frozen=True)
class Site:
    """ERFA dtdb site: east longitude [rad], distance from the Earth's spin axis u [km], distance north of
    the equatorial plane v [km]."""

    elong_rad: float
    u_km: float
    v_km: float

    @classmethod
    def from_itrf_m(cls, r_ecef_m) -> "Site":
        x, y, z = (float(c) for c in np.asarray(r_ecef_m, dtype=float).reshape(3))
        return cls(math.atan2(y, x), math.hypot(x, y) / 1e3, z / 1e3)


# ======================================================================
# eraDtdb and its derivative
# ======================================================================
def _fundamental(t):
    w = t / 3600.0
    return {name: np.fmod(c0 + c1 * w, 360.0) * DD2R for name, (c0, c1) in _T.FUNDAMENTAL.items()}


def _fundamental_rates():
    return {name: c1 / 3600.0 * DD2R * _DT_DTMILL for name, (_, c1) in _T.FUNDAMENTAL.items()}


def _series(t):
    """w0..w4 (the T**k group sums) and their T-derivatives, for 1-D t [Julian millennia]."""
    w = np.zeros((5, t.size))
    dw = np.zeros((5, t.size))
    for i0 in range(0, t.size, _CHUNK):
        tt = t[i0:i0 + _CHUNK, None]
        for k, g in enumerate(_GROUPS):
            arg = _F[g] * tt + _P[g]
            w[k, i0:i0 + _CHUNK] = np.sum(_A[g] * np.sin(arg), axis=1)
            dw[k, i0:i0 + _CHUNK] = np.sum(_A[g] * _F[g] * np.cos(arg), axis=1)
    return w, dw


def _evaluate(date1, date2, ut, elong, u, v, *, rate: bool):
    date1, date2, ut, elong, u, v = np.broadcast_arrays(*(np.asarray(x, dtype=float)
                                                         for x in (date1, date2, ut, elong, u, v)))
    shape = date1.shape
    date1, date2, ut, elong, u, v = (x.reshape(-1) for x in (date1, date2, ut, elong, u, v))
    t = ((date1 - DJ00) + date2) / DJM
    args = _fundamental(t)
    args["tsol"] = np.fmod(ut, 1.0) * D2PI + elong
    rates = _fundamental_rates()
    rates["tsol"] = _DTSOL_DT

    wt = np.zeros(t.size)
    dwt = np.zeros(t.size)
    for coef, uv, fn, mult in _T.TOPOCENTRIC:
        arg = 0.0
        darg = 0.0
        for name, m in mult:
            arg = arg + m * args[name]
            darg = darg + m * rates[name]
        scale = coef * (u if uv == "u" else v)
        if fn == "sin":
            wt = wt + scale * np.sin(arg)
            dwt = dwt + scale * np.cos(arg) * darg
        else:
            wt = wt + scale * np.cos(arg)
            dwt = dwt - scale * np.sin(arg) * darg

    w, dw = _series(t)
    wf = t * (t * (t * (t * w[4] + w[3]) + w[2]) + w[1]) + w[0]
    wj = np.zeros(t.size)
    dwj = np.zeros(t.size)
    for amp, f, ph in _T.JPL:
        wj = wj + amp * np.sin(f * t + ph)
        dwj = dwj + amp * f * np.cos(f * t + ph)
    wj = wj + _T.JPL_T2 * t * t
    dwj = dwj + 2.0 * _T.JPL_T2 * t
    if not rate:
        return (wt + wf + wj).reshape(shape)
    # d(wf)/dT = sum_k [k T^(k-1) w_k + T^k w_k']
    dwf = (dw[0] + w[1] + t * (dw[1] + 2.0 * w[2] + t * (dw[2] + 3.0 * w[3] + t * (dw[3] + 4.0 * w[4]
                                                                                    + t * dw[4]))))
    return (dwt + (dwf + dwj) * _DT_DTMILL).reshape(shape)


def dtdb(date1, date2, ut, elong=0.0, u=0.0, v=0.0):
    """TDB - TT [s]: ERFA ``eraDtdb(date1, date2, ut, elong, u, v)``, vectorised. ``date1 + date2`` is the TDB
    (TT acceptable) Julian date, ``ut`` the UT fraction of a day, ``elong`` [rad], ``u``/``v`` [km]."""
    return _evaluate(date1, date2, ut, elong, u, v, rate=False)


def dtdb_rate(date1, date2, ut, elong=0.0, u=0.0, v=0.0):
    """d(TDB - TT)/dt [s/s], the analytic derivative of :func:`dtdb` (local solar time advances at 2 pi per
    86400 s; d(UT)/d(TDB) differs from 1 by ~1e-10, neglected)."""
    return _evaluate(date1, date2, ut, elong, u, v, rate=True)


# ======================================================================
# leap seconds (the loaded leapseconds kernel)
# ======================================================================
_LEAP_CACHE: dict = {}


def _leap_table():
    """(UTC calendar seconds past J2000 at which each TAI - UTC value starts, TAI - UTC) from DELTET/DELTA_AT.
    The kernel's @dates are formal calendar seconds past 2000-01-01T12:00 (no leap seconds counted)."""
    import spiceypy as spice
    values = np.asarray(spice.gdpool("DELTET/DELTA_AT", 0, 400), dtype=float)
    key = values.tobytes()
    if key not in _LEAP_CACHE:
        _LEAP_CACHE.clear()
        _LEAP_CACHE[key] = (values[1::2].copy(), values[0::2].copy())
    return _LEAP_CACHE[key]


def tai_minus_utc(utc_calendar_s_past_j2000) -> np.ndarray:
    """TAI - UTC [s] at UTC calendar seconds past 2000-01-01T12:00 (formal, as the leapseconds kernel)."""
    epochs, dat = _leap_table()
    s = np.asarray(utc_calendar_s_past_j2000, dtype=float)
    if np.any(s < epochs[0]):
        raise ValueError("TAI - UTC is integral only from 1972-01-01; earlier UTC is not supported.")
    return dat[np.searchsorted(epochs, s, side="right") - 1]


# ======================================================================
# production epochs (seconds of TDB past J2000 = et0 + t)
# ======================================================================
def _two_part(et0_s: float, t_s):
    days = math.floor(float(et0_s) / DAYSEC)
    rem = float(et0_s) - days * DAYSEC                    # exact (Sterbenz)
    return DJ00 + days, rem, (rem + np.asarray(t_s, dtype=float)) / DAYSEC


def _site_args(site: Site | None):
    if site is None:
        return 0.0, 0.0, 0.0
    return site.elong_rad, site.u_km, site.v_km


def _ut_of_tdb(et0_s: float, t_s, d1, d2, geo):
    """UTC fraction of the day at TDB epoch et0 + t (TT = TDB - geo; one pass is exact to < 1e-15 s)."""
    rem = float(et0_s) - (d1 - DJ00) * DAYSEC
    tai_rel = rem + np.asarray(t_s, dtype=float) - geo - TT_MINUS_TAI_S     # TAI seconds past the et0 day
    base = (d1 - DJ00) * DAYSEC
    dat = tai_minus_utc(base + tai_rel)                                    # TAI read as UTC: off by < 1 min
    dat = tai_minus_utc(base + tai_rel - dat)                              # settles away from a leap instant
    utc_rel = tai_rel - dat
    return np.mod(utc_rel / DAYSEC + 0.5, 1.0)


def tdb_minus_tt(et0_s: float, t_s, site: Site | None = None):
    """TDB - TT [s] at TDB epoch ``et0 + t`` (seconds past J2000), geocentric or at ``site``."""
    d1, _, d2 = _two_part(et0_s, t_s)
    geo = dtdb(d1, d2, 0.0)
    if site is None:
        return geo
    return dtdb(d1, d2, _ut_of_tdb(et0_s, t_s, d1, d2, geo), *_site_args(site))


def tdb_minus_tt_rate(et0_s: float, t_s, site: Site | None = None):
    """d(TDB - TT)/dt [s/s] at TDB epoch ``et0 + t``, analytic, geocentric or at ``site``."""
    d1, _, d2 = _two_part(et0_s, t_s)
    if site is None:
        return dtdb_rate(d1, d2, 0.0)
    geo = dtdb(d1, d2, 0.0)
    return dtdb_rate(d1, d2, _ut_of_tdb(et0_s, t_s, d1, d2, geo), *_site_args(site))


# ======================================================================
# UTC tags
# ======================================================================
_ISO = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2}(?:\.\d*)?)\s*Z?\s*$")


def _mjd(year: int, month: int, day: int) -> int:
    """Modified Julian Date of 0h of a Gregorian calendar date (the eraCal2jd algorithm). C integer division
    truncates toward zero: (month - 14) / 12 is -1 for January and February and 0 otherwise."""
    my = -1 if month < 3 else 0
    iypmy = year + my
    return int((1461 * (iypmy + 4800)) // 4 + (367 * (month - 2 - 12 * my)) // 12
               - (3 * ((iypmy + 4900) // 100)) // 4 + day - 2432076)


def _parse_tag(tag) -> tuple[int, int, int, float]:
    if isinstance(tag, str):
        m = _ISO.match(tag)
        if not m:
            raise ValueError(f"UTC tag {tag!r} is not YYYY-MM-DDTHH:MM:SS[.fff][Z]")
        y, mo, d, hh, mm = (int(m.group(i)) for i in range(1, 6))
        sod = hh * 3600.0 + mm * 60.0 + float(m.group(6))
        return y, mo, d, sod
    y, mo, d, sod = tag
    return int(y), int(mo), int(d), float(sod)


def utc_to_tdb_seconds(utc_tags: Iterable, et0_s: float, site: Site | None = None) -> np.ndarray:
    """UTC station tags -> TDB seconds past ``et0`` (``et0`` itself is seconds of TDB past J2000).

    A tag is an ISO string ``YYYY-MM-DDTHH:MM:SS[.fff][Z]`` or a ``(year, month, day, seconds_of_day)`` tuple.
    TAI - UTC comes from the loaded leapseconds kernel (by UTC date), TT = TAI + 32.184 s, and TDB = TT +
    :func:`dtdb` evaluated at TT and at ``site`` (geocentric when None) with the UTC fraction of the day as
    ``ut``. The whole-day part is subtracted from ``et0`` exactly before the seconds of the day are added, so the
    result carries only the final double rounding. A tag inside a leap second (seconds_of_day >= 86400) is
    accepted; its ``ut`` is clipped to the end of the day.
    """
    tags = [_parse_tag(x) for x in utc_tags]
    if not tags:
        return np.zeros(0)
    et0 = float(et0_s)
    days = np.array([_mjd(y, mo, d) - 51544 for y, mo, d, _ in tags], dtype=float)   # days past 2000-01-01
    sod = np.array([s for *_, s in tags], dtype=float)
    midnight = days * DAYSEC - 43200.0                                             # UTC calendar s past J2000
    dat = tai_minus_utc(midnight)
    tt_rel = (midnight - et0) + (sod + dat + TT_MINUS_TAI_S)                       # TT seconds past et0
    d1, rem, _ = _two_part(et0, 0.0)
    d2 = (rem + tt_rel) / DAYSEC
    ut = np.minimum(sod, DAYSEC - 1e-9) / DAYSEC
    return tt_rel + dtdb(d1, d2, ut, *_site_args(site))


def spice_utc_chain_tdb_seconds(utc_tags: Sequence, et0_s: float) -> np.ndarray:
    """The same tags through SPICE's own chain (leap seconds + DELTET), for comparisons: TDB seconds past et0.
    TT is formed exactly as in :func:`utc_to_tdb_seconds`; SPICE's TDB - TT is its analytic K sin(E) model."""
    import spiceypy as spice
    k, eb = spice.gdpool("DELTET/K", 0, 1)[0], spice.gdpool("DELTET/EB", 0, 1)[0]
    m0, m1 = spice.gdpool("DELTET/M", 0, 2)
    tags = [_parse_tag(x) for x in utc_tags]
    et0 = float(et0_s)
    days = np.array([_mjd(y, mo, d) - 51544 for y, mo, d, _ in tags], dtype=float)
    sod = np.array([s for *_, s in tags], dtype=float)
    midnight = days * DAYSEC - 43200.0
    tt_rel = (midnight - et0) + (sod + tai_minus_utc(midnight) + TT_MINUS_TAI_S)
    m = m0 + m1 * (et0 + tt_rel)
    return tt_rel + k * np.sin(m + eb * np.sin(m))
