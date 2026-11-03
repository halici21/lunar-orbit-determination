"""F2 WP2.4 slice 4: DSN station catalogue with plate motion, gates G16-G23.

Oracles: NAIF's station SPK earthstns_itrf93_260814.bsp (independent carrier of DSN 810-005 301 Rev. P) and the
published test case of the IERS DEHANTTIDEINEL routine.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from lunar_od import dsn_stations as dsn
from lunar_od.kernel_manifest import KERNEL_MANIFEST
from lunar_od.solid_tide import solid_tide_displacement_ecef

SPK = "earthstns_itrf93_260814.bsp"
SPK_SHA256 = "007eb0aad5dd022c6d61bdbfb2395680bd1eb31c49a8637d0b0a1cfbea737c6b"
SIGMAS = dict(sigma_range_m=1.0, sigma_angle_rad=1e-4, sigma_range_rate_mps=3e-5)
YEAR_S = 365.25 * 86400.0


def _id(name):
    return str(399000 + int(name.split("-")[1]))


def _et(spice, year):
    return spice.str2et(f"{year}-01-01T00:00:00")


@pytest.fixture(scope="module")
def spice():
    spiceypy = pytest.importorskip("spiceypy")
    from lunar_od.spice_loader import load_spice_kernels, resolve_kernel_dir
    try:
        load_spice_kernels(None, clear=False)
        path = resolve_kernel_dir() / SPK
    except FileNotFoundError:
        pytest.skip("SPICE kernels not available")
    if not path.is_file():
        pytest.skip(f"{SPK} not available")
    spiceypy.furnsh(str(path))
    yield spiceypy
    spiceypy.unload(str(path))


def test_g16_table_matches_the_naif_spk(spice):
    assert len(dsn.DSN_STATION_NAMES) == 19
    assert {int(_id(n)) for n in dsn.DSN_STATION_NAMES} == set(spice.spkobj(str(_spk_path())))
    for year in (1990, 2003, 2010, 2027, 2040):
        et = _et(spice, year)
        for name in dsn.DSN_STATION_NAMES:
            state = np.asarray(spice.spkezr(_id(name), et, "ITRF93", "NONE", "EARTH")[0]) * 1e3
            r, v = dsn.dsn_station_itrf93(name, et)
            assert np.max(np.abs(r - state[:3])) < 1e-3, (name, year)
            assert np.max(np.abs(v - state[3:])) < 1e-12, (name, year)


def _spk_path():
    from lunar_od.spice_loader import resolve_kernel_dir

    return resolve_kernel_dir() / SPK


def test_g17_station_reproduces_the_catalogue_position():
    for name in dsn.DSN_STATION_NAMES:
        station = dsn.dsn_station(name, 8.0e8, allow_estimated=True, **SIGMAS)
        assert station.name == name
        assert np.max(np.abs(station.r_ecef_m - dsn.dsn_station_itrf93(name, 8.0e8)[0])) < 1e-6


@pytest.fixture(scope="module")
def arc(spice):
    from lunar_od.ephemeris import sample_moon_centered_ephemeris

    et0 = spice.str2et("2027-03-02T00:00:00")
    t = np.arange(0.0, 3600.0 + 1.0, 60.0)
    eph = sample_moon_centered_ephemeris(et0, np.arange(-600.0, 3600.0 + 601.0, 60.0))
    return dict(et0=et0, t=t, earth_pos=eph.earth_position(t), earth_vel=eph.earth_velocity(t))


def test_g18_production_chain_matches_the_spk_in_j2000(spice, arc):
    from lunar_od.two_way_range import make_exact_sxform_station_state_provider

    for name in ("DSS-14", "DSS-43", "DSS-63"):
        station = dsn.dsn_station(name, arc["et0"], **SIGMAS)
        provider = make_exact_sxform_station_state_provider(station, arc["et0"], arc["t"], arc["earth_pos"],
                                                            arc["earth_vel"])
        for k, t in enumerate(arc["t"]):
            state = np.asarray(provider.state(float(t)), dtype=float)
            geocentric = np.r_[state[:3] - arc["earth_pos"][k], state[3:] - arc["earth_vel"][k]]
            oracle = np.asarray(spice.spkezr(_id(name), arc["et0"] + t, "J2000", "NONE", "EARTH")[0]) * 1e3
            assert np.linalg.norm(geocentric[:3] - oracle[:3]) < 1e-3, (name, t)
            assert np.linalg.norm(geocentric[3:] - oracle[3:]) < 1e-7, (name, t)


def test_g19_freezing_at_the_arc_start_errs_below_the_documented_limit():
    for name in dsn.DSN_STATION_NAMES:
        drift = dsn.dsn_station_itrf93(name, 8.0e8 + 86400.0)[0] - dsn.dsn_station_itrf93(name, 8.0e8)[0]
        assert np.linalg.norm(drift) < 0.2e-3


def _ssb_context(spice, arc, station):
    from lunar_od import relativistic_light_time as rlt
    from lunar_od.two_way_range import make_exact_sxform_station_state_provider

    provider = make_exact_sxform_station_state_provider(station, arc["et0"], arc["t"], arc["earth_pos"],
                                                        arc["earth_vel"])
    return rlt.exact_epoch_station_state(lambda t: np.asarray(provider.state(t), dtype=float), arc["et0"], arc["t"],
                                         arc["earth_pos"], arc["earth_vel"])


def test_g20_plate_motion_is_material(spice, arc):
    """Epoch-2003 coordinates against the arc-epoch position: the size of ignoring plate motion at 2027."""
    from lunar_od import relativistic_light_time as rlt

    et0 = arc["et0"]
    for name, expected in (("DSS-14", 0.47), ("DSS-43", 1.25), ("DSS-63", 0.74)):
        shift = np.linalg.norm(dsn.dsn_station_itrf93(name, et0)[0] - dsn.dsn_station_itrf93(name, dsn.EPOCH_ET_S)[0])
        assert shift == pytest.approx(expected, rel=0.01), name

    t = arc["t"]
    r0 = 1737.4e3 + 100e3
    # A circular-enough Moon orbit from the two-body speed: this gate measures station geometry, not dynamics.
    w = math.sqrt(4.9028e12 / r0) / r0
    history = np.c_[r0 * np.cos(w * t), r0 * np.sin(w * t), np.zeros(t.size),
                    -r0 * w * np.sin(w * t), r0 * w * np.cos(w * t), np.zeros(t.size)]
    sc = rlt.SpacecraftHistory(t, history)
    eph = rlt.BarycentricEphemeris(et0)
    model = rlt.DSN_LIGHT_TIME_MODEL
    now = _ssb_context(spice, arc, dsn.dsn_station("DSS-43", et0, **SIGMAS))
    then = _ssb_context(spice, arc, dsn.dsn_station("DSS-43", dsn.EPOCH_ET_S, **SIGMAS))
    d_range, d_doppler = [], []
    for t3 in np.arange(600.0, 3001.0, 300.0):
        a = rlt.solve_two_way(float(t3), now, sc, eph, model)
        b = rlt.solve_two_way(float(t3), then, sc, eph, model)
        d_range.append(0.5 * rlt.C_LIGHT_MPS * (a.round_trip_s - b.round_trip_s))
        d_doppler.append(rlt.counted_doppler(float(t3), 60.0, now, sc, eph, model)[0]
                         - rlt.counted_doppler(float(t3), 60.0, then, sc, eph, model)[0])
    assert 0.2 <= np.max(np.abs(d_range)) <= 1.3, d_range
    assert 1e-6 <= np.max(np.abs(d_doppler)) <= 2e-4, d_doppler


def test_g21_solid_tide_reproduces_the_iers_test_case():
    # DEHANTTIDEINEL header test case, 2009-04-13 0h UTC (TAI-UTC = 34 s). Values transcribed from memory of that
    # header, the file itself could not be fetched: see the pre-registration's survey note.
    station = np.array([4075578.385, 931852.890, 4801570.154])
    sun = np.array([137859926952.015, 54228127881.4350, 23509422341.6960])
    moon = np.array([-179996231.920342, -312468450.131567, -169288918.592160])
    mjd_tt = 54934.0 + (32.184 + 34.0) / 86400.0
    tide = solid_tide_displacement_ecef(station, sun, moon, mjd_tt, 0.0, mass_ratio_sun=332946.0482,
                                        mass_ratio_moon=0.0123000371)
    expected = np.array([0.07700420357108126, 0.06304056321824968, 0.05516568152597247])
    assert np.max(np.abs(tide - expected)) < 1e-6, tide


def test_g22_fail_closed():
    with pytest.raises(ValueError, match="Unknown DSN antenna"):
        dsn.dsn_station_itrf93("DSS-99", 8.0e8)
    with pytest.raises(ValueError, match="estimated"):
        dsn.dsn_station("DSS-33", 8.0e8, **SIGMAS)
    assert dsn.dsn_station("DSS-33", 8.0e8, allow_estimated=True, **SIGMAS).name == "DSS-33"
    with pytest.raises(ValueError, match="finite"):
        dsn.dsn_station_itrf93("DSS-14", float("nan"))


def test_g23_manifest_records_the_spk_and_the_loader_does_not_furnish_it():
    from lunar_od import spice_loader
    from lunar_od.kernel_manifest import NAIF_GENERIC

    assert KERNEL_MANIFEST[SPK] == (SPK_SHA256, NAIF_GENERIC + "spk/stations/" + SPK)
    source = spice_loader.__file__
    with open(source, encoding="utf-8") as handle:
        assert "earthstns" not in handle.read()


def test_plate_rate_is_the_published_complex_velocity():
    # The rate is per Julian year, one velocity per complex: two antennas of a complex share it.
    _, v14 = dsn.dsn_station_itrf93("DSS-14", 8.0e8)
    _, v25 = dsn.dsn_station_itrf93("DSS-25", 8.0e8)
    assert np.array_equal(v14, v25)
    assert v14 * YEAR_S == pytest.approx([-0.0180, 0.0065, -0.0038], rel=1e-12)
