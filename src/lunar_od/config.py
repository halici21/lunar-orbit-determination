"""Configuration data for the Python Lunar OD port."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import geodetic_to_ecef_wgs84


@dataclass(frozen=True)
class Station:
    name: str
    lat_deg: float
    lon_deg: float
    alt_m: float
    color_rgb: tuple[float, float, float]
    sigma_range_m: float
    sigma_angle_rad: float
    sigma_range_rate_mps: float | None = None
    bias: tuple[float, ...] = ()

    @property
    def lat_rad(self) -> float:
        return float(np.deg2rad(self.lat_deg))

    @property
    def lon_rad(self) -> float:
        return float(np.deg2rad(self.lon_deg))

    @property
    def r_ecef_m(self) -> np.ndarray:
        x_m, y_m, z_m = geodetic_to_ecef_wgs84(self.lat_deg, self.lon_deg, self.alt_m)
        return np.array([float(x_m), float(y_m), float(z_m)], dtype=float)


def _range_sigma(name: str) -> float:
    return 94.0 if "ITU" in name else 5.0


def _angle_sigma(name: str) -> float:
    return float(np.deg2rad(0.005 if "ITU" in name else 0.001))


def _rr_sigma(name: str) -> float:
    return 1e-3 if "ITU" in name else 1e-4


def _station(
    name: str,
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    color_rgb: tuple[float, float, float],
    include_range_rate: bool,
) -> Station:
    return Station(
        name=name,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        alt_m=alt_m,
        color_rgb=color_rgb,
        sigma_range_m=_range_sigma(name),
        sigma_angle_rad=_angle_sigma(name),
        sigma_range_rate_mps=_rr_sigma(name) if include_range_rate else None,
    )


POSITION_ONLY_STATION_DEFS = (
    ("ITU Ayazaga", 41.101, 29.023, 100.0, (0.0, 0.447, 0.741)),
    ("Goldstone DSN", 35.30, -116.81, 969.67, (0.85, 0.325, 0.098)),
    ("Madrid DSN", 40.43, -4.25, 833.73, (0.85, 0.325, 0.098)),
    ("Canberra DSN", -35.40, 148.98, 680.00, (0.85, 0.325, 0.098)),
    ("Dongara KGS", -29.04, 115.35, 251.10, (0.494, 0.184, 0.556)),
    ("Chuuk KGS", 7.25, 151.47, 50.00, (0.494, 0.184, 0.556)),
    ("Svalbard KGS", 78.23, 15.39, 498.52, (0.494, 0.184, 0.556)),
    ("Malargue ESA", -35.78, -69.40, 1550.00, (0.466, 0.674, 0.188)),
    ("New Norcia ESA", -31.05, 116.19, 252.26, (0.466, 0.674, 0.188)),
    ("Evpatoria RUS", 45.19, 33.21, 100.00, (0.635, 0.078, 0.184)),
    ("Ussuriisk RUS", 44.02, 131.76, 100.00, (0.635, 0.078, 0.184)),
    ("Bear Lakes RUS", 55.87, 37.95, 100.00, (0.635, 0.078, 0.184)),
    ("Byalalu ISRO", 12.90, 77.37, 25.00, (0.929, 0.694, 0.125)),
)


RANGE_RATE_STATION_DEFS = (
    ("ITU Ayazaga", 41.101, 29.023, 100.0, (0.0, 0.447, 0.741)),
    ("Goldstone DSN", 35.30, -116.81, 969.67, (0.85, 0.325, 0.098)),
    ("Madrid DSN", 40.43, -4.25, 833.73, (0.85, 0.325, 0.098)),
    ("Canberra DSN", -35.40, 148.98, 680.00, (0.85, 0.325, 0.098)),
    ("Daejeon KGS", 36.38, 127.35, 102.00, (0.494, 0.184, 0.556)),
    ("Dongara KGS", -29.04, 115.35, 251.10, (0.494, 0.184, 0.556)),
    ("Chuuk KGS", 7.25, 151.47, 50.00, (0.494, 0.184, 0.556)),
    ("Svalbard KGS", 78.23, 15.39, 498.52, (0.494, 0.184, 0.556)),
    ("Malargue ESA", -35.78, -69.40, 1550.00, (0.466, 0.674, 0.188)),
    ("Cebreros ESA", 40.45, -4.37, 794.10, (0.466, 0.674, 0.188)),
    ("New Norcia ESA", -31.05, 116.19, 252.26, (0.466, 0.674, 0.188)),
    ("Evpatoria RUS", 45.19, 33.21, 100.00, (0.635, 0.078, 0.184)),
    ("Ussuriisk RUS", 44.02, 131.76, 100.00, (0.635, 0.078, 0.184)),
    ("Bear Lakes RUS", 55.87, 37.95, 100.00, (0.635, 0.078, 0.184)),
    ("Byalalu ISRO", 12.90, 77.37, 25.00, (0.929, 0.694, 0.125)),
)


def position_only_stations() -> tuple[Station, ...]:
    return tuple(_station(*station_def, include_range_rate=False) for station_def in POSITION_ONLY_STATION_DEFS)


def range_rate_stations() -> tuple[Station, ...]:
    return tuple(_station(*station_def, include_range_rate=True) for station_def in RANGE_RATE_STATION_DEFS)
