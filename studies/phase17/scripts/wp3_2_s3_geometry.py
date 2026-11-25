"""F3 WP3.2 S3: the gate geometries of the Earth-shadow slice (predeclaration ``geometry``).

Predeclaration: studies/phase17/configs/wp3_2_s3_earth_shadow_predeclaration.json (a5954f1).

- Synthetic eclipse: S1's synthetic Earth (circle of 384400 km, sidereal month, 23.44 deg tilt) and a Sun on the
  coplanar circle ``sun_e(t) = circle(AU, sidereal year, PHI_E, t) + earth(t)`` with
  ``PHI_E = (W_EARTH - W_SUN) * T_GE``, so the central eclipse is greatest at ``T_GE = 3 h``.  Orbits: ``TERM``
  (terminator, no lunar shadow) and ``PLANE`` (Sun in the orbit plane), 100 km circular polar, built as S1's
  ``_orbit`` with ``sun_e(0)``.
- Real E2: WP3.1's total lunar eclipse of 2026-03-03, arc ``t_ge - 3 h .. t_ge + 3 h`` every 300 s, a
  MoonCenteredEphemeris on a 60 s grid with 2 h margin, DE440 GMs, orbit PLANE at the arc start.
"""
from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for extra in (ROOT, HERE):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from lunar_od.constants import AU_M, MU_EARTH_M3S2, MU_MOON_M3S2, MU_SUN_M3S2, R_MOON_M  # noqa: E402

DAY_S = 86400.0
TILT = math.radians(23.44)
W_EARTH = 2.0 * math.pi / (27.321661 * DAY_S)
W_SUN = 2.0 * math.pi / (365.25636 * DAY_S)
T_GE = 3 * 3600.0
PHI_E = (W_EARTH - W_SUN) * T_GE
T6H = np.arange(0.0, 6 * 3600.0 + 1.0, 300.0)


def _circle(radius, omega, phase, t):
    th = omega * float(t) + phase
    return radius * np.array([math.cos(th), math.sin(th) * math.cos(TILT), math.sin(th) * math.sin(TILT)])


def earth(t):
    """S1's synthetic Moon -> Earth vector."""
    return _circle(384_400e3, W_EARTH, 0.0, t)


def sun_e(t):
    """Moon -> Sun with a central eclipse greatest at T_GE."""
    return _circle(AU_M, W_SUN, PHI_E, t) + earth(t)


BASE_E = (MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2, earth, sun_e)


def orbit(sun_position, sun_in_plane: bool) -> np.ndarray:
    """S1's ``_orbit``: 100 km circular polar orbit; Sun in its plane (PLANE) or terminator (TERM)."""
    a = R_MOON_M + 100e3
    z = np.array([0.0, 0.0, 1.0])
    s = np.asarray(sun_position, dtype=float)
    s = s / np.linalg.norm(s)
    h = np.cross(s, z) if sun_in_plane else np.cross(np.cross(s, z), z)
    h /= np.linalg.norm(h)
    return np.concatenate([a * z, math.sqrt(MU_MOON_M3S2 / a) * np.cross(h, z)])


TERM = orbit(sun_e(0.0), sun_in_plane=False)
PLANE = orbit(sun_e(0.0), sun_in_plane=True)


class SyntheticEclipseEphemeris:
    earth_position = staticmethod(earth)
    sun_position = staticmethod(sun_e)


# ----------------------------------------------------------------------
# real eclipses (SPICE)
# ----------------------------------------------------------------------
ECLIPSES = {"E2": "2026-03-03", "E3": "2026-08-28"}


@lru_cache(maxsize=None)
def greatest_eclipse_et(name: str) -> float:
    from lunar_od.spice_loader import load_spice_kernels
    import wp3_1_materiality_survey as survey
    load_spice_kernels()
    return survey.greatest_eclipse_et(ECLIPSES[name])


@lru_cache(maxsize=None)
def e2_case():
    """``(et0, t, ephemeris, (mu_moon, mu_earth, mu_sun), x0_plane)`` of the real E2 arc."""
    from lunar_od.ephemeris import sample_moon_centered_ephemeris
    from lunar_od.ephemeris_generation import gravitational_parameters
    et0 = greatest_eclipse_et("E2") - T_GE
    eph = sample_moon_centered_ephemeris(et0, np.arange(-7200.0, T6H[-1] + 7200.0 + 0.5, 60.0))
    mus = gravitational_parameters("DE440")
    return et0, T6H.copy(), eph, mus, orbit(eph.sun_position(0.0), sun_in_plane=True)
