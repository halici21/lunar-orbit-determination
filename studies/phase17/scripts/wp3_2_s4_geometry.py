"""F3 WP3.2 S4: the gate geometries of the SRP light-time slice (predeclaration ``geometry``).

Predeclaration: studies/phase17/configs/wp3_2_s4_srp_light_time_predeclaration.json (cc7a454).

- Synthetic: S3's geometry (wp3_2_s3_geometry: Earth circle, Sun with a central Earth eclipse greatest at 3 h,
  orbits TERM and PLANE, T6H) with the Sun at the SSB: ``v_sun = 0`` and ``v_moon(t) = -d sun_e/dt`` (analytic),
  sampled into an SSBVelocityTable every 60 s from -2 h to 8 h.
- Real E2: S3's E2 arc with ``sample_ssb_velocities`` on its ephemeris grid.
"""
from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import wp3_2_s3_geometry as S3G  # noqa: E402
from lunar_od.constants import AU_M  # noqa: E402

T_TABLE = np.arange(-7200.0, 8 * 3600.0 + 0.5, 60.0)


def _circle_rate(radius, omega, phase, t):
    th = omega * float(t) + phase
    return radius * omega * np.array([-math.sin(th), math.cos(th) * math.cos(S3G.TILT),
                                      math.cos(th) * math.sin(S3G.TILT)])


def sun_e_rate(t):
    """d/dt of S3's Moon -> Sun vector ``sun_e``."""
    return _circle_rate(AU_M, S3G.W_SUN, S3G.PHI_E, t) + _circle_rate(384_400e3, S3G.W_EARTH, 0.0, t)


@lru_cache(maxsize=None)
def synthetic_table():
    from lunar_od.srp_light_time import SSBVelocityTable
    moon = np.vstack([-sun_e_rate(t) for t in T_TABLE])
    return SSBVelocityTable(t_s=T_TABLE.copy(), sun_vel_mps=np.zeros_like(moon), moon_vel_mps=moon)


def synthetic_options():
    from lunar_od.srp_light_time import SRPLightTimeOptions
    return SRPLightTimeOptions(ssb_velocities=synthetic_table())


@lru_cache(maxsize=None)
def e2_table():
    from lunar_od.srp_light_time import sample_ssb_velocities
    et0, _t, eph, _mus, _x0 = S3G.e2_case()
    return sample_ssb_velocities(et0, eph.t_ephem_s)


def e2_options():
    from lunar_od.srp_light_time import SRPLightTimeOptions
    return SRPLightTimeOptions(ssb_velocities=e2_table())
