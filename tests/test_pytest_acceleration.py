from __future__ import annotations

import numpy as np

from lunar_od import ACCELERATION_BACKEND, NUMBA_AVAILABLE, is_occulted_by_moon, range_rate_stations
from lunar_od.accelerated import visibility_mask_ecef


def test_acceleration_backend_flag_is_consistent():
    assert ACCELERATION_BACKEND in {"numpy", "numba"}
    assert NUMBA_AVAILABLE is (ACCELERATION_BACKEND == "numba")


def test_visibility_mask_ecef_matches_reference_loop():
    stations = range_rate_stations()[:4]
    station_ecef = np.vstack([station.r_ecef_m for station in stations])
    station_lat = np.array([station.lat_rad for station in stations])
    station_lon = np.array([station.lon_rad for station in stations])

    # Synthetic ECEF spacecraft/Moon geometry chosen to exercise both
    # elevation rejection and occultation logic without SPICE dependencies.
    r_moon_ecef = np.array(
        [
            [3.8e8, 0.0, 0.0],
            [3.8e8, 1.0e6, -2.0e6],
            [3.8e8, -2.0e6, 1.0e6],
            [3.8e8, 3.0e6, 0.5e6],
            [3.8e8, -1.0e6, -1.0e6],
        ],
        dtype=float,
    )
    r_ecef = np.array(
        [
            [3.82e8, 1.0e6, 0.2e6],
            [3.78e8, 2.0e6, -1.5e6],
            [3.84e8, -1.5e6, 1.2e6],
            [3.60e8, 3.1e6, 0.4e6],
            [3.90e8, -2.0e6, -0.8e6],
        ],
        dtype=float,
    )

    min_elev_rad = np.deg2rad(5.0)
    radius_moon_m = 1_737_400.0

    got = visibility_mask_ecef(
        r_ecef,
        r_moon_ecef,
        station_ecef,
        station_lat,
        station_lon,
        min_elev_rad,
        radius_moon_m,
    )
    ref = _reference_visibility_mask(
        r_ecef,
        r_moon_ecef,
        stations,
        min_elev_rad,
        radius_moon_m,
    )

    np.testing.assert_array_equal(got, ref)


def _reference_visibility_mask(r_ecef, r_moon_ecef, stations, min_elev_rad, radius_moon_m):
    out = np.zeros((r_ecef.shape[0], len(stations)), dtype=bool)
    for station_idx, station in enumerate(stations):
        station_ecef = np.asarray(station.r_ecef_m, dtype=float).reshape(3)
        rho_vec = r_ecef - station_ecef
        range_m = np.linalg.norm(rho_vec, axis=1)
        zenith_unit = np.array(
            [
                np.cos(station.lat_rad) * np.cos(station.lon_rad),
                np.cos(station.lat_rad) * np.sin(station.lon_rad),
                np.sin(station.lat_rad),
            ],
            dtype=float,
        )
        sin_el = np.divide(
            rho_vec @ zenith_unit,
            range_m,
            out=np.zeros(r_ecef.shape[0], dtype=float),
            where=range_m > 0.0,
        )
        in_view = np.arcsin(np.clip(sin_el, -1.0, 1.0)) > min_elev_rad
        if not np.any(in_view):
            continue
        idx = np.where(in_view)[0]
        occulted = is_occulted_by_moon(
            station_ecef,
            r_ecef[idx, :],
            r_moon_ecef[idx, :],
            radius_moon_m,
        )
        out[idx[~occulted], station_idx] = True
    return out
