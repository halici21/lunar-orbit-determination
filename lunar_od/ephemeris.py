"""SPICE ephemeris sampling and interpolation helpers.

Phase 17-HARDEN ephemeris contract
----------------------------------
Measurement generators receive the Earth as a callable of SCENARIO seconds next
to a separately passed ``et0``.  Phase 17-FXR found that an untagged callable
whose t = 0 was 504000 s earlier than ``et0`` passed through silently.  Any
provider that DOES carry metadata (an object exposing ``et_origin`` and
optionally ``frame``, ``center``, ``units``, ``legacy_reproduction_only``, or a
bound method of such an object) is now validated by
:func:`require_ephemeris_contract`: a mismatched epoch origin, frame, centre or
unit label raises instead of producing a well-formed observable of the wrong
geometry.  :class:`MoonCenteredEphemeris` carries that metadata and refuses
queries beyond a documented extrapolation margin.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PchipInterpolator

#: Moon->Earth distance bounds [m] (perigee ~356 000 km, apogee ~407 000 km, with
#: margin).  A Moon-centred metre-valued Earth position outside them is a unit,
#: centre or epoch-contract error, never physics.
MOON_EARTH_DISTANCE_BOUNDS_M = (3.40e8, 4.20e8)
#: Qualified PCHIP extrapolation allowed past a sampled ephemeris grid: it covers
#: light-time pre-roll (two-way Earth-Moon light time < 3 s) and nothing more.
DEFAULT_EPHEMERIS_EXTRAPOLATION_MARGIN_S = 10.0


class EphemerisContractError(ValueError):
    """An ephemeris provider's epoch/frame/centre/units/coverage contract is violated."""


class EphemerisEpochMismatchError(EphemerisContractError):
    """The provider's epoch origin differs from the measurement model's et0."""


class EphemerisCoverageError(EphemerisContractError):
    """An ephemeris was queried outside its declared coverage."""


class LegacyEphemerisWarning(UserWarning):
    """A provider explicitly marked LEGACY_REPRODUCTION_ONLY was accepted."""


def require_ephemeris_contract(
    provider,
    et0_s: float,
    name: str = "ephemeris",
    *,
    frame: str = "J2000",
    center: str = "MOON",
) -> str:
    """Validate an epoch-tagged ephemeris provider against the model's contract.

    Returns ``"MATCHED"``, ``"LEGACY"`` (explicitly marked reproduction-only
    provider, accepted with a warning) or ``"UNTAGGED"`` (a bare callable: no
    metadata to check, so it is accepted unchanged -- tagging is the caller's
    responsibility, and the Phase 17 science builders always tag).
    """
    owner = getattr(provider, "__self__", provider)
    origin = getattr(owner, "et_origin", None)
    if origin is None:
        return "UNTAGGED"
    if bool(getattr(owner, "legacy_reproduction_only", False)):
        warnings.warn(
            f"{name}: LEGACY_REPRODUCTION_ONLY provider accepted (epoch origin {origin!r} "
            f"vs et0 {et0_s!r}); not valid for current science",
            LegacyEphemerisWarning, stacklevel=3)
        return "LEGACY"
    got_frame = getattr(owner, "frame", frame)
    got_center = getattr(owner, "center", center)
    units = getattr(owner, "units", None)
    if str(got_frame).upper() != frame.upper():
        raise EphemerisContractError(f"{name}: frame {got_frame!r}, model requires {frame!r}")
    if str(got_center).upper() != center.upper():
        raise EphemerisContractError(f"{name}: centre {got_center!r}, model requires {center!r}")
    if units is not None and not str(units).replace(" ", "").lower().startswith("m,"):
        raise EphemerisContractError(f"{name}: units {units!r}, model requires metres (m, m/s)")
    if float(origin) != float(et0_s):
        raise EphemerisEpochMismatchError(
            f"{name}: epoch origin ET {float(origin)!r} differs from the measurement model's "
            f"et0 {float(et0_s)!r} by {float(origin) - float(et0_s)!r} s")
    return "MATCHED"


def require_moon_earth_distance(earth_pos_mci_m: ArrayLike, name: str = "Earth position") -> None:
    """Physical plausibility of a Moon-centred Earth position history in metres."""
    r = np.linalg.norm(np.asarray(earth_pos_mci_m, dtype=float).reshape(-1, 3), axis=1)
    lo, hi = MOON_EARTH_DISTANCE_BOUNDS_M
    bad = ~np.isfinite(r) | (r < lo) | (r > hi)
    if np.any(bad):
        raise EphemerisContractError(
            f"{name}: Moon-Earth distance {r[bad][:3].tolist()} m outside {MOON_EARTH_DISTANCE_BOUNDS_M} "
            "(wrong units, wrong centre, or a non-physical provider)")


@dataclass(frozen=True)
class MoonCenteredEphemeris:
    """Moon-centered J2000 ephemeris samples and PCHIP interpolants.

    ``et0_s`` (optional, set by :func:`sample_moon_centered_ephemeris`) is the
    SPICE ET of ``t = 0``.  Queries more than ``extrapolation_margin_s`` outside
    the sampled grid raise :class:`EphemerisCoverageError` instead of silently
    extrapolating the PCHIP interpolant.
    """

    t_ephem_s: np.ndarray
    earth_pos_m: np.ndarray
    sun_pos_m: np.ndarray
    earth_vel_mps: np.ndarray
    et0_s: float | None = None
    extrapolation_margin_s: float = DEFAULT_EPHEMERIS_EXTRAPOLATION_MARGIN_S
    frame: str = "J2000"
    center: str = "MOON"
    units: str = "m, m/s"

    @property
    def et_origin(self) -> float | None:
        return self.et0_s

    def _covered(self, t_s: ArrayLike) -> ArrayLike:
        t = np.asarray(t_s, dtype=float)
        lo = self.t_ephem_s[0] - self.extrapolation_margin_s
        hi = self.t_ephem_s[-1] + self.extrapolation_margin_s
        if np.any((t < lo) | (t > hi)) or not np.all(np.isfinite(t)):
            raise EphemerisCoverageError(
                f"ephemeris queried at t={np.atleast_1d(t)[:3].tolist()} s outside "
                f"[{lo!r}, {hi!r}] s (grid {self.t_ephem_s[0]!r}..{self.t_ephem_s[-1]!r}, "
                f"margin {self.extrapolation_margin_s!r} s)")
        return t_s

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
        return self._ep(self._covered(t_s))

    def sun_position(self, t_s: ArrayLike) -> np.ndarray:
        return self._sp(self._covered(t_s))

    def earth_velocity(self, t_s: ArrayLike) -> np.ndarray:
        return self._ev(self._covered(t_s))


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
        et0_s=ephemeris.et0_s,
        extrapolation_margin_s=ephemeris.extrapolation_margin_s,
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
        et0_s=float(et0),
    )
