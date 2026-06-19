"""SPICE ephemeris sampling and interpolation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PchipInterpolator


@dataclass(frozen=True)
class MoonCenteredEphemeris:
    """Moon-centered J2000 ephemeris samples and PCHIP interpolants."""

    t_ephem_s: np.ndarray
    earth_pos_m: np.ndarray
    sun_pos_m: np.ndarray
    earth_vel_mps: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "t_ephem_s", np.asarray(self.t_ephem_s, dtype=float).reshape(-1))
        object.__setattr__(self, "earth_pos_m", _as_n_by_3(self.earth_pos_m, "earth_pos_m"))
        object.__setattr__(self, "sun_pos_m", _as_n_by_3(self.sun_pos_m, "sun_pos_m"))
        object.__setattr__(self, "earth_vel_mps", _as_n_by_3(self.earth_vel_mps, "earth_vel_mps"))

        n = self.t_ephem_s.size
        if self.earth_pos_m.shape[0] != n or self.sun_pos_m.shape[0] != n or self.earth_vel_mps.shape[0] != n:
            raise ValueError("Ephemeris arrays must have the same number of rows as t_ephem_s.")

        # Build interpolants once at construction — reused across all RHS calls.
        object.__setattr__(self, "_ep", PchipInterpolator(self.t_ephem_s, self.earth_pos_m, axis=0))
        object.__setattr__(self, "_sp", PchipInterpolator(self.t_ephem_s, self.sun_pos_m, axis=0))
        object.__setattr__(self, "_ev", PchipInterpolator(self.t_ephem_s, self.earth_vel_mps, axis=0))

    def earth_position(self, t_s: ArrayLike) -> np.ndarray:
        return self._ep(t_s)

    def sun_position(self, t_s: ArrayLike) -> np.ndarray:
        return self._sp(t_s)

    def earth_velocity(self, t_s: ArrayLike) -> np.ndarray:
        return self._ev(t_s)


def perturb_moon_centered_ephemeris(
    ephemeris: MoonCenteredEphemeris,
    *,
    earth_position_bias_m: ArrayLike = (0.0, 0.0, 0.0),
    earth_velocity_bias_mps: ArrayLike = (0.0, 0.0, 0.0),
    sun_position_bias_m: ArrayLike = (0.0, 0.0, 0.0),
) -> MoonCenteredEphemeris:
    """Return a deterministic ephemeris perturbation for mismatch campaigns."""
    earth_position_bias = np.asarray(earth_position_bias_m, dtype=float).reshape(3)
    earth_velocity_bias = np.asarray(earth_velocity_bias_mps, dtype=float).reshape(3)
    sun_position_bias = np.asarray(sun_position_bias_m, dtype=float).reshape(3)
    return MoonCenteredEphemeris(
        t_ephem_s=ephemeris.t_ephem_s.copy(),
        earth_pos_m=ephemeris.earth_pos_m + earth_position_bias,
        sun_pos_m=ephemeris.sun_pos_m + sun_position_bias,
        earth_vel_mps=ephemeris.earth_vel_mps + earth_velocity_bias,
    )


def _as_n_by_3(value: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3).")
    return array


def sample_moon_centered_ephemeris(et0: float, t_ephem_s: ArrayLike) -> MoonCenteredEphemeris:
    """Sample Earth/Sun Moon-centered J2000 ephemerides using loaded SPICE kernels.

    This mirrors the MATLAB runner:

    - `cspice_spkpos('EARTH', ..., 'J2000', 'NONE', 'MOON')`
    - `cspice_spkpos('SUN', ..., 'J2000', 'NONE', 'MOON')`
    - `cspice_spkezr('EARTH', ..., 'J2000', 'NONE', 'MOON')`

    Units are converted from SPICE km and km/s to m and m/s.
    """
    import spiceypy as spice

    t_ephem_s = np.asarray(t_ephem_s, dtype=float).reshape(-1)
    earth_pos_rows = []
    sun_pos_rows = []
    earth_vel_rows = []

    for t_s in t_ephem_s:
        et = float(et0 + t_s)
        earth_pos_km, _ = spice.spkpos("EARTH", et, "J2000", "NONE", "MOON")
        sun_pos_km, _ = spice.spkpos("SUN", et, "J2000", "NONE", "MOON")
        earth_state_km, _ = spice.spkezr("EARTH", et, "J2000", "NONE", "MOON")

        earth_pos_rows.append(np.asarray(earth_pos_km, dtype=float) * 1000.0)
        sun_pos_rows.append(np.asarray(sun_pos_km, dtype=float) * 1000.0)
        earth_vel_rows.append(np.asarray(earth_state_km[3:6], dtype=float) * 1000.0)

    return MoonCenteredEphemeris(
        t_ephem_s=t_ephem_s,
        earth_pos_m=np.vstack(earth_pos_rows),
        sun_pos_m=np.vstack(sun_pos_rows),
        earth_vel_mps=np.vstack(earth_vel_rows),
    )
