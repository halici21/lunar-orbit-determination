"""Station coordinate realizations and Earth-orientation provenance (Phase C3-I).

Phase C3 (``studies/phase17/reports/phase_c3_station_frame_materiality.md``) found that reading modern station
coordinates (ITRF 2020, the realization real DSN coordinates are published in) through SPICE's ITRF93
orientation misplaces a station by up to 0.33 m (1.4 sigma of OD), and that the production Earth PCK
(NAIF long-term predict kernel) drifts from IERS Earth orientation after its creation date (112 m at 2027).

* :func:`itrf2020_to_itrf93` - the 14-parameter EPSG transformation ITRF 2020 -> ITRF 93 (Altamimi et al. 2023,
  doi 10.1029/2023GL106373; distributed in PROJ ``data/ITRF2020``), position-vector convention, parameters
  evaluated at the epoch of the coordinates. Orekit's transcription of the same transformation omits the
  scale; ``include_scale=False`` reproduces it.
* :func:`predicted_earth_orientation_start_et` / :func:`is_predicted_earth_orientation` - whether an epoch lies
  in the predicted part of a loaded NAIF predict PCK (``earth_YYYY_YYMMDD_YYYY_predict.bpc``; YYMMDD is the
  creation date).
* :func:`earth_orientation_provenance` / :func:`require_measured_earth_orientation` (F2 WP2.4) - which loaded
  Earth PCK SPICE uses at an epoch (the last-loaded one that covers it) and whether that epoch lies on or before
  the kernel's last measured EOP datum (its comment field ``UTC Epoch of last datum``); the second fails closed.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

#: EPSG ITRF2020 -> ITRF93 at 2015.0: translation [m], scale [ppb], rotation [mas], and their rates per year.
_T = np.array([-65.8, 1.9, -71.3]) * 1e-3
_T_DOT = np.array([-2.8, -0.2, -2.3]) * 1e-3
_D_PPB, _D_DOT_PPB = 4.47, 0.12
_R_MAS = np.array([-3.36, -4.33, 0.75])
_R_DOT_MAS = np.array([-0.11, -0.19, 0.07])
_MAS = math.pi / (180.0 * 3600.0 * 1000.0)
#: 2015-01-01T12:00 TT in TDB seconds past J2000 (the parameters' reference epoch, as Orekit builds it);
#: TDB - TT (~1.6 ms) is far below the epoch resolution these rates need.
_ET_2015 = 5479.0 * 86400.0
_YEAR_S = 365.25 * 86400.0


def helmert_parameters(et_s: float) -> tuple[np.ndarray, float, np.ndarray]:
    """(translation [m], scale [-], rotation [rad]) of ITRF 2020 -> ITRF 93 at TDB epoch ``et_s``."""
    dt = (float(et_s) - _ET_2015) / _YEAR_S
    return _T + _T_DOT * dt, (_D_PPB + _D_DOT_PPB * dt) * 1e-9, (_R_MAS + _R_DOT_MAS * dt) * _MAS


def itrf2020_to_itrf93(r_m, et_s: float, *, include_scale: bool = True) -> np.ndarray:
    """ITRF 2020 position(s) [m] -> ITRF 93 at epoch ``et_s``: X93 = X + T + D X + R X, R = [[0, -r3, r2],
    [r3, 0, -r1], [-r2, r1, 0]] (position-vector convention)."""
    r = np.asarray(r_m, dtype=float)
    t, d, rot = helmert_parameters(et_s)
    r1, r2, r3 = rot
    m = np.array([[0.0, -r3, r2], [r3, 0.0, -r1], [-r2, r1, 0.0]])
    out = r + t + (r @ m.T)
    if include_scale:
        out = out + d * r
    return out


_PREDICT = re.compile(r"earth_\d{4}_(\d{2})(\d{2})(\d{2})_\d{4}_predict\.bpc$", re.IGNORECASE)


def predicted_earth_orientation_start_et() -> float | None:
    """Creation epoch (TDB seconds past J2000) of the loaded NAIF predict Earth PCK, or None if none is loaded.
    Later epochs use predicted, not measured, Earth orientation from that kernel."""
    import spiceypy as spice
    starts = []
    for i in range(spice.ktotal("PCK")):
        name = Path(spice.kdata(i, "PCK")[0]).name
        m = _PREDICT.search(name)
        if m:
            yy, mm, dd = m.groups()
            starts.append(float(spice.str2et(f"20{yy}-{mm}-{dd}T00:00:00")))
    return min(starts) if starts else None


def is_predicted_earth_orientation(et_s: float) -> bool:
    """True when the Earth PCK SPICE uses at ``et_s`` holds predicted, not measured, orientation there (real data
    needs final EOP / a high-precision PCK covering the data epochs). Through F2 WP2.4 this follows the kernel
    actually used (:func:`earth_orientation_provenance`); before, it compared with the predict kernel's creation
    date whatever kernel had priority. An epoch no loaded kernel covers is not reported as predicted."""
    prov = earth_orientation_provenance(et_s)
    return prov["kernel"] is not None and not prov["measured"]


#: NAIF body code of the ITRF93 Earth orientation in binary PCKs.
EARTH_ITRF93_PCK_BODY = 3000
_LAST_DATUM = re.compile(r"UTC Epoch of last datum:\s*(\d{4} [A-Z]{3} \d{2} \d{2}:\d{2}:\d{2})", re.IGNORECASE)


class PredictedEarthOrientationError(ValueError):
    """An epoch's Earth orientation is predicted (or not covered) where measured orientation is required."""


def _last_datum_utc(path: str) -> str | None:
    import spiceypy as spice

    handle = spice.dafopr(path)
    try:
        lines = spice.dafec(handle, 2000, 300)[1]
    finally:
        spice.dafcls(handle)
    m = _LAST_DATUM.search("\n".join(lines))
    return m.group(1) if m else None


def _covers(path: str, et_s: float) -> bool:
    import spiceypy as spice

    cover = spice.cell_double(2000)
    spice.pckcov(path, EARTH_ITRF93_PCK_BODY, cover)
    return any(a <= et_s <= b for a, b in (spice.wnfetd(cover, i) for i in range(spice.wncard(cover))))


def earth_orientation_provenance(et_s: float) -> dict:
    """The loaded Earth PCK SPICE uses for ITRF93 at ``et_s`` and whether its orientation there is measured.

    Binary PCKs are searched in reverse load order (the last-loaded covering file has priority in SPICE). Returns
    ``kernel`` (file name, or None when no loaded kernel covers ``et_s``), ``path``, ``last_datum_utc`` /
    ``last_datum_et`` (the kernel's last measured EOP datum, from its comments) and ``measured``
    (``et_s <= last_datum_et``; False when the kernel records no last datum).
    """
    import spiceypy as spice

    et = float(et_s)
    for i in reversed(range(spice.ktotal("PCK"))):
        path = spice.kdata(i, "PCK")[0]
        if not path.lower().endswith(".bpc") or not _covers(path, et):
            continue
        utc = _last_datum_utc(path)
        last_et = float(spice.str2et(utc + " UTC")) if utc else None
        return {"kernel": Path(path).name, "path": path, "last_datum_utc": utc, "last_datum_et": last_et,
                "measured": last_et is not None and et <= last_et}
    return {"kernel": None, "path": None, "last_datum_utc": None, "last_datum_et": None, "measured": False}


def require_measured_earth_orientation(et_values) -> None:
    """Fail closed (:class:`PredictedEarthOrientationError`) unless every epoch's Earth orientation is measured.

    Load the ``MEASURED`` profile (``spice_loader.load_spice_kernels(earth_orientation="MEASURED")``) and a
    high-precision Earth PCK whose last datum follows the data to satisfy this for recent epochs."""
    import spiceypy as spice

    for et in np.atleast_1d(np.asarray(et_values, dtype=float)).ravel():
        prov = earth_orientation_provenance(float(et))
        if not prov["measured"]:
            where = (f"kernel {prov['kernel']} holds predicted orientation after its last datum "
                     f"{prov['last_datum_utc']} UTC") if prov["kernel"] else "no loaded Earth PCK covers it"
            raise PredictedEarthOrientationError(
                f"Earth orientation at {spice.et2utc(float(et), 'ISOC', 3)} TDB-ET is not measured: {where}. "
                "Load a high-precision Earth PCK covering the data epochs with priority "
                "(earth_orientation='MEASURED').")
