"""DSN antenna coordinates with plate motion (F2 WP2.4 slice 4, MEAS-06, RD-09).

The table is DSN 810-005, module 301, Rev. P (May 2026), Table 2: Cartesian coordinates in ITRF93 at epoch 2003.0,
as quoted in the comments of NAIF's earthstns_itrf93_260814.bsp, with one plate velocity per complex. The SPK is the
qualification oracle of this table and is not needed at run time. ITRF93 is the frame of SPICE's Earth orientation, so
these coordinates take no Helmert transformation (``coordinate_frame='ITRF93'`` in
:func:`lunar_od.relativistic_light_time.displaced_station_state`).

:func:`dsn_station` returns an ordinary :class:`lunar_od.config.Station` at a chosen epoch, so every production path
consumes it unchanged. ``Station`` is static: the position is frozen at that epoch, which errs by less than 0.2 mm per
day. Choose the arc's start epoch. The source states an absolute accuracy of up to 0.1 m (1 sigma), from events such
as the 2019 Ridgecrest earthquake and integrated plate-velocity uncertainty.

The rounded tables in :mod:`lunar_od.config` are unchanged: synthetic baselines keep their stations.
"""
from __future__ import annotations

import math

import numpy as np

from .config import Station
from .geometry import WGS84_A_M, WGS84_INV_F

#: SPICE ET of the source's epoch "2003.0", taken as 2003 Jan 1 00:00:00 TDB as NAIF does.
EPOCH_ET_S = 94651200.0
_JULIAN_YEAR_S = 365.25 * 86400.0

_POSITIONS_M = {
    "DSS-13": (-2351112.659, -4655530.636, 3660912.728),
    "DSS-14": (-2353621.420, -4641341.472, 3677052.318),
    "DSS-15": (-2353538.958, -4641649.429, 3676669.984),
    "DSS-23": (-2354702.027, -4646969.709, 3669213.211),
    "DSS-24": (-2354906.711, -4646840.095, 3669242.325),
    "DSS-25": (-2355022.014, -4646953.204, 3669040.567),
    "DSS-26": (-2354890.797, -4647166.328, 3668871.755),
    "DSS-33": (-4461103.0, 2682649.0, -3674286.0),
    "DSS-34": (-4461147.093, 2682439.239, -3674393.133),
    "DSS-35": (-4461273.090, 2682568.925, -3674152.093),
    "DSS-36": (-4461168.415, 2682814.657, -3674083.901),
    "DSS-43": (-4460894.917, 2682361.507, -3674748.152),
    "DSS-45": (-4460935.578, 2682765.661, -3674380.982),
    "DSS-53": (4849338.209, -360657.812, 4114746.173),
    "DSS-54": (4849434.488, -360723.8999, 4114618.835),
    "DSS-55": (4849525.256, -360606.0932, 4114495.084),
    "DSS-56": (4849421.679, -360549.659, 4114646.987),
    "DSS-63": (4849092.518, -360180.3480, 4115109.251),
    "DSS-65": (4849339.634, -360427.6637, 4114750.733),
}

#: Plate velocity [m/yr] by the antenna number's first digit: Goldstone, Canberra, Madrid.
_VELOCITIES_M_PER_YR = {
    "1": (-0.0180, 0.0065, -0.0038), "2": (-0.0180, 0.0065, -0.0038),
    "3": (-0.0335, -0.0041, 0.0392), "4": (-0.0335, -0.0041, 0.0392),
    "5": (-0.0100, 0.0242, 0.0156), "6": (-0.0100, 0.0242, 0.0156),
}

#: The source gives this location as an estimate, good to a few metres.
ESTIMATED_LOCATIONS = frozenset({"DSS-33"})
DSN_STATION_NAMES = tuple(_POSITIONS_M)


def dsn_station_itrf93(name: str, et_s: float) -> tuple[np.ndarray, np.ndarray]:
    """ITRF93 position [m] and plate velocity [m/s] of antenna ``name`` (``'DSS-14'``) at TDB epoch ``et_s``."""
    if name not in _POSITIONS_M:
        raise ValueError(f"Unknown DSN antenna {name!r}; known: {DSN_STATION_NAMES}.")
    et = float(et_s)
    if not math.isfinite(et):
        raise ValueError(f"et_s must be finite; got {et_s!r}.")
    velocity = np.asarray(_VELOCITIES_M_PER_YR[name[4]], dtype=float) / _JULIAN_YEAR_S
    return np.asarray(_POSITIONS_M[name], dtype=float) + velocity * (et - EPOCH_ET_S), velocity


def _ecef_to_geodetic_wgs84(r_m: np.ndarray) -> tuple[float, float, float]:
    """Latitude [deg], longitude [deg] and height [m] by fixed-point iteration on the prime-vertical radius."""
    x, y, z = (float(v) for v in r_m)
    e2 = (2.0 - 1.0 / WGS84_INV_F) / WGS84_INV_F
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - e2))
    for _ in range(8):
        n = WGS84_A_M / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
        height = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1.0 - e2 * n / (n + height)))
    n = WGS84_A_M / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
    return math.degrees(lat), math.degrees(math.atan2(y, x)), p / math.cos(lat) - n


def dsn_station(name: str, et_s: float, *, sigma_range_m: float, sigma_angle_rad: float,
                sigma_range_rate_mps: float | None = None, allow_estimated: bool = False,
                color_rgb: tuple[float, float, float] = (0.85, 0.325, 0.098)) -> Station:
    """A :class:`Station` at the position of antenna ``name`` at ``et_s``, frozen there (see the module docstring)."""
    r, _velocity = dsn_station_itrf93(name, et_s)
    if name in ESTIMATED_LOCATIONS and not allow_estimated:
        raise ValueError(f"{name} is an estimated location, good to a few metres; pass allow_estimated=True to use it.")
    lat_deg, lon_deg, alt_m = _ecef_to_geodetic_wgs84(r)
    return Station(name=name, lat_deg=lat_deg, lon_deg=lon_deg, alt_m=alt_m, color_rgb=color_rgb,
                   sigma_range_m=sigma_range_m, sigma_angle_rad=sigma_angle_rad,
                   sigma_range_rate_mps=sigma_range_rate_mps)
