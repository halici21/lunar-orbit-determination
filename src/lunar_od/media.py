"""Tropospheric and ionospheric media delay for radiometric observables (Phase C5-I).

Phase C5 (``studies/phase17/reports/phase_c5_media_materiality.md``) found the media worth 2-12 m of two-way range
(66 sigma of OD) for the hydrostatic troposphere, 4.8 sigma for a nominal wet delay and 3.5 sigma for the X-band
ionosphere; production had none. This module is the calibration interface: zenith hydrostatic and wet delays and
the vertical TEC are inputs (DSN media calibrations in phase F), mapped to the line of sight; a default climatology
serves synthetic work.

* :func:`zhd_iers` - IERS Conventions 2010 eq. 9.11 (Davis et al. 1985 form of Saastamoinen);
* :func:`niell_mapping` - Niell (1996) hydrostatic (seasonal, height-corrected) and wet mapping functions, with
  the coefficients and algorithm as carried in Orekit 13.1.9 ``NiellMappingFunctionModel``;
* :func:`black_eisner_mapping` - kept to reproduce the C5 survey;
* :func:`single_layer_mapping`, :func:`iono_delay_m` - thin-shell ionosphere;
* :class:`MediaModel` - the per-pass configuration and :meth:`MediaModel.leg_delays`.

The troposphere is non-dispersive (same delay for group and phase); the ionosphere delays the group (range,
delta-DOR) and advances the phase (counted Doppler) by the same amount.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

#: Niell (1996) coefficients at latitudes 15, 30, 45, 60, 75 deg (Orekit NiellMappingFunctionModel).
_LAT_DEG = np.array([15.0, 30.0, 45.0, 60.0, 75.0])
_AH_AVG = np.array([0.0012769934, 0.001268323, 0.0012465397, 0.0012196049, 0.0012045996])
_BH_AVG = np.array([0.0029153695, 0.0029152299, 0.0029288445, 0.0029022565, 0.0029024912])
_CH_AVG = np.array([0.062610505, 0.062837393, 0.063721774, 0.063824265, 0.064258455])
_AH_AMP = np.array([0.0, 1.2709626e-5, 2.6523662e-5, 3.4000452e-5, 4.1202191e-5])
_BH_AMP = np.array([0.0, 2.1414979e-5, 3.0160779e-5, 7.2562722e-5, 1.1723375e-4])
_CH_AMP = np.array([0.0, 9.01284e-5, 4.3497037e-5, 8.4795348e-4, 0.0017037206])
_AW = np.array([5.8021897e-4, 5.6794847e-4, 5.8118019e-4, 5.9727542e-4, 6.1641693e-4])
_BW = np.array([0.0014275268, 0.0015138625, 0.0014572752, 0.0015007428, 0.0017599082])
_CW = np.array([0.043472961, 0.04672951, 0.043908931, 0.044626982, 0.054736038])
_HT = (2.53e-5, 5.49e-3, 1.14e-3)
_DOY_MIN = 28.0
_YEAR_D = 365.25
IONO_K = 40.3e16          # m Hz^2 per TECU
C_LIGHT_MPS = 299792458.0


def standard_pressure_hpa(h_m: float) -> float:
    """Standard-atmosphere pressure [hPa] at height ``h_m``."""
    return 1013.25 * (1.0 - 2.2557e-5 * h_m) ** 5.2568


def zhd_iers(p_hpa: float, lat_rad: float, h_m: float) -> float:
    """Zenith hydrostatic delay [m], IERS Conventions 2010 eq. 9.11."""
    return 0.0022768 * p_hpa / (1.0 - 0.00266 * math.cos(2.0 * lat_rad) - 0.28e-6 * h_m)


def _continued_fraction(a: float, b: float, c: float, el: float) -> float:
    s = math.sin(el)
    return (1.0 + a / (1.0 + b / (1.0 + c))) / (s + a / (s + b / (s + c)))


def niell_mapping(el_rad: float, lat_rad: float, h_m: float, doy: float) -> tuple[float, float]:
    """Niell (1996) (hydrostatic, wet) mapping factors at elevation ``el_rad``, geodetic latitude ``lat_rad``,
    height ``h_m`` and day of year ``doy`` (fractional)."""
    lat = min(max(abs(math.degrees(lat_rad)), 15.0), 75.0)
    dmin = _DOY_MIN + (183.0 if lat_rad < 0.0 else 0.0)
    cosd = math.cos(2.0 * math.pi * (doy - dmin) / _YEAR_D)

    def node(avg, amp=None):
        v = float(np.interp(lat, _LAT_DEG, avg))
        return v if amp is None else v - float(np.interp(lat, _LAT_DEG, amp)) * cosd
    ah, bh, ch = node(_AH_AVG, _AH_AMP), node(_BH_AVG, _BH_AMP), node(_CH_AVG, _CH_AMP)
    mh = _continued_fraction(ah, bh, ch, el_rad)
    mh += (1.0 / math.sin(el_rad) - _continued_fraction(*_HT, el_rad)) * max(0.0, h_m) / 1000.0
    mw = _continued_fraction(node(_AW), node(_BW), node(_CW), el_rad)
    return mh, mw


def black_eisner_mapping(el_rad: float) -> float:
    return 1.001 / math.sqrt(0.002001 + math.sin(el_rad) ** 2)


def single_layer_mapping(el_rad: float, layer_height_m: float = 350e3, earth_radius_m: float = 6371e3) -> float:
    """Thin-shell obliquity STEC / VTEC."""
    return 1.0 / math.sqrt(1.0 - (earth_radius_m * math.cos(el_rad) / (earth_radius_m + layer_height_m)) ** 2)


def iono_delay_m(stec_tecu: float, freq_hz: float) -> float:
    """First-order ionospheric delay [m] (group delay; the phase is advanced by the same amount)."""
    return IONO_K * stec_tecu / freq_hz ** 2


def default_vtec_tecu(local_time_h: float) -> float:
    """The C5 climatology: 5 TECU at night, 30 TECU at 14 h local time (cosine)."""
    return 5.0 + 25.0 * max(0.0, math.cos(2.0 * math.pi * (local_time_h - 14.0) / 24.0))


@dataclass(frozen=True)
class MediaSite:
    lat_rad: float
    lon_rad: float
    h_m: float

    @classmethod
    def from_station(cls, st) -> "MediaSite":
        return cls(float(st.lat_rad), float(st.lon_rad), float(st.alt_m))

    @property
    def up_ecef(self) -> np.ndarray:
        return np.array([math.cos(self.lat_rad) * math.cos(self.lon_rad),
                         math.cos(self.lat_rad) * math.sin(self.lon_rad), math.sin(self.lat_rad)])


@dataclass(frozen=True)
class MediaModel:
    """Media configuration of a pass.

    ``zhd_m`` None -> IERS ZHD at the standard-atmosphere pressure of the station height; ``vtec_tecu`` a number,
    ``'DEFAULT_CLIMATOLOGY'`` or None (no ionosphere); ``mapping`` 'NIELL' | 'BLACK_EISNER'.
    """

    troposphere: bool = True
    zhd_m: float | None = None
    zwd_m: float = 0.15
    mapping: str = "NIELL"
    vtec_tecu: float | str | None = "DEFAULT_CLIMATOLOGY"
    layer_height_m: float = 350e3
    earth_radius_m: float = 6371e3
    uplink_hz: float = 7.2e9
    downlink_hz: float = 7.2e9 * 880.0 / 749.0

    def __post_init__(self):
        if self.mapping not in ("NIELL", "BLACK_EISNER"):
            raise ValueError(f"mapping must be 'NIELL' or 'BLACK_EISNER'; got {self.mapping!r}.")
        if isinstance(self.vtec_tecu, str) and self.vtec_tecu != "DEFAULT_CLIMATOLOGY":
            raise ValueError("vtec_tecu must be a number, 'DEFAULT_CLIMATOLOGY' or None.")

    def to_metadata(self) -> dict:
        return asdict(self)

    @classmethod
    def from_metadata(cls, d: dict) -> "MediaModel":
        return cls(**d)

    def leg_delays(self, site: MediaSite, et: float, los_j2000, uplink: bool) -> tuple[float, float, float]:
        """(group [m], phase [m], elevation [rad]) of one leg at its ground event epoch ``et`` (TDB s past J2000),
        ``los_j2000`` from the station towards the spacecraft."""
        import spiceypy as spice
        up = np.asarray(spice.pxform("ITRF93", "J2000", float(et))) @ site.up_ecef
        los = np.asarray(los_j2000, dtype=float)
        el = math.asin(min(1.0, max(-1.0, float(np.dot(los / np.linalg.norm(los), up)))))
        tropo = 0.0
        if self.troposphere:
            zhd = self.zhd_m if self.zhd_m is not None else zhd_iers(standard_pressure_hpa(site.h_m), site.lat_rad,
                                                                     site.h_m)
            if self.mapping == "NIELL":
                mh, mw = niell_mapping(el, site.lat_rad, site.h_m, self.day_of_year(et))
            else:
                mh = mw = black_eisner_mapping(el)
            tropo = zhd * mh + self.zwd_m * mw
        iono = 0.0
        if self.vtec_tecu is not None:
            if self.vtec_tecu == "DEFAULT_CLIMATOLOGY":
                lt = ((float(et) - 69.184 + 43200.0) / 3600.0 + math.degrees(site.lon_rad) / 15.0) % 24.0
                vtec = default_vtec_tecu(lt)
            else:
                vtec = float(self.vtec_tecu)
            stec = vtec * single_layer_mapping(el, self.layer_height_m, self.earth_radius_m)
            iono = iono_delay_m(stec, self.uplink_hz if uplink else self.downlink_hz)
        return tropo + iono, tropo - iono, el

    def day_of_year(self, et: float) -> float:
        """UTC day of year (1 at 00:00 on 1 January), fractional."""
        import spiceypy as spice
        s = spice.et2utc(float(et), "D", 6)                 # 'YYYY-DDD // HH:MM:SS.ffffff'
        date, clock = s.split("//")
        doy = int(date.strip().split("-")[1])
        h, m, sec = clock.strip().split(":")
        return doy + (int(h) * 3600 + int(m) * 60 + float(sec)) / 86400.0

    def leg_delay_partials(self, site: MediaSite, et: float, los_j2000, uplink: bool):
        """Group/phase LOS and explicit-epoch derivatives of leg_delays."""
        from .media_partials import model_partials
        return model_partials(self,site,et,los_j2000,uplink)
