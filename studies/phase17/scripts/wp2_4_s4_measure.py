"""Measured margins of the WP2.4 slice 4 gates and the offset of the rounded scenario stations from the real antennas."""
import json
import math
import sys

import numpy as np
import spiceypy as sp

from lunar_od import dsn_stations as dsn
from lunar_od import relativistic_light_time as rlt
from lunar_od.config import POSITION_ONLY_STATION_DEFS, Station
from lunar_od.ephemeris import sample_moon_centered_ephemeris
from lunar_od.solid_tide import solid_tide_displacement_ecef
from lunar_od.spice_loader import load_spice_kernels, resolve_kernel_dir
from lunar_od.two_way_range import make_exact_sxform_station_state_provider

load_spice_kernels(None, clear=False)
sp.furnsh(str(resolve_kernel_dir() / "earthstns_itrf93_260814.bsp"))
S = dict(sigma_range_m=1.0, sigma_angle_rad=1e-4, sigma_range_rate_mps=3e-5)
out = {}
pid = lambda n: str(399000 + int(n[4:]))

g16p = g16v = 0.0
for y in (1990, 2003, 2010, 2027, 2040):
    et = sp.str2et(f"{y}-01-01T00:00:00")
    for n in dsn.DSN_STATION_NAMES:
        s = np.asarray(sp.spkezr(pid(n), et, "ITRF93", "NONE", "EARTH")[0]) * 1e3
        r, v = dsn.dsn_station_itrf93(n, et)
        g16p, g16v = max(g16p, np.abs(r - s[:3]).max()), max(g16v, np.abs(v - s[3:]).max())
out["G16"] = {"position_m": g16p, "velocity_mps": g16v}
out["G17_m"] = max(np.abs(dsn.dsn_station(n, 8e8, allow_estimated=True, **S).r_ecef_m - dsn.dsn_station_itrf93(n, 8e8)[0]).max()
                   for n in dsn.DSN_STATION_NAMES)

et0 = sp.str2et("2027-03-02T00:00:00")
t = np.arange(0.0, 3601.0, 60.0)
eph = sample_moon_centered_ephemeris(et0, np.arange(-600.0, 4201.0, 60.0))
ep, ev = eph.earth_position(t), eph.earth_velocity(t)
g18p = g18v = 0.0
for n in ("DSS-14", "DSS-43", "DSS-63"):
    prov = make_exact_sxform_station_state_provider(dsn.dsn_station(n, et0, **S), et0, t, ep, ev)
    for k, tk in enumerate(t):
        st = np.asarray(prov.state(float(tk)), dtype=float)
        o = np.asarray(sp.spkezr(pid(n), et0 + tk, "J2000", "NONE", "EARTH")[0]) * 1e3
        g18p = max(g18p, np.linalg.norm(st[:3] - ep[k] - o[:3]))
        g18v = max(g18v, np.linalg.norm(st[3:] - ev[k] - o[3:]))
out["G18"] = {"position_m": g18p, "velocity_mps": g18v}
out["G19_m_per_day"] = max(np.linalg.norm(dsn.dsn_station_itrf93(n, 8e8 + 86400)[0] - dsn.dsn_station_itrf93(n, 8e8)[0])
                           for n in dsn.DSN_STATION_NAMES)
out["G20_station_shift_m"] = {n: float(np.linalg.norm(dsn.dsn_station_itrf93(n, et0)[0] - dsn.dsn_station_itrf93(n, dsn.EPOCH_ET_S)[0]))
                              for n in ("DSS-14", "DSS-43", "DSS-63")}

r0 = 1737.4e3 + 100e3
w = math.sqrt(4.9028e12 / r0) / r0
h = np.c_[r0 * np.cos(w * t), r0 * np.sin(w * t), np.zeros(t.size), -r0 * w * np.sin(w * t), r0 * w * np.cos(w * t), np.zeros(t.size)]
sc, be, m = rlt.SpacecraftHistory(t, h), rlt.BarycentricEphemeris(et0), rlt.DSN_LIGHT_TIME_MODEL


def ctx(st):
    p = make_exact_sxform_station_state_provider(st, et0, t, ep, ev)
    return rlt.exact_epoch_station_state(lambda x: np.asarray(p.state(x), dtype=float), et0, t, ep, ev)


def effect(a_station, b_station):
    a, b = ctx(a_station), ctx(b_station)
    dr, dd = [], []
    for t3 in np.arange(600.0, 3001.0, 300.0):
        dr.append(0.5 * rlt.C_LIGHT_MPS * (rlt.solve_two_way(float(t3), a, sc, be, m).round_trip_s
                                           - rlt.solve_two_way(float(t3), b, sc, be, m).round_trip_s))
        dd.append(rlt.counted_doppler(float(t3), 60.0, a, sc, be, m)[0] - rlt.counted_doppler(float(t3), 60.0, b, sc, be, m)[0])
    return float(np.abs(dr).max()), float(np.abs(dd).max())


out["G20_DSS43"] = dict(zip(("range_m", "doppler_mps"), effect(dsn.dsn_station("DSS-43", et0, **S), dsn.dsn_station("DSS-43", dsn.EPOCH_ET_S, **S))))

st = np.array([4075578.385, 931852.890, 4801570.154])
tide = solid_tide_displacement_ecef(st, np.array([137859926952.015, 54228127881.4350, 23509422341.6960]),
                                    np.array([-179996231.920342, -312468450.131567, -169288918.592160]),
                                    54934.0 + 66.184 / 86400.0, 0.0, mass_ratio_sun=332946.0482, mass_ratio_moon=0.0123000371)
out["G21_max_abs_m"] = float(np.abs(tide - np.array([0.07700420357108126, 0.06304056321824968, 0.05516568152597247])).max())

# Diagnostic: the rounded scenario stations against the antennas they stand for.
defs = {d[0]: d for d in POSITION_ONLY_STATION_DEFS}
out["rounded_scenario_stations"] = {}
for label, antenna in (("Goldstone DSN", "DSS-14"), ("Madrid DSN", "DSS-63"), ("Canberra DSN", "DSS-43")):
    d = defs[label]
    rounded = Station(d[0], d[1], d[2], d[3], d[4], 1.0, 1e-4, 3e-5)
    real = dsn.dsn_station(antenna, et0, **S)
    nearest = min((n for n in dsn.DSN_STATION_NAMES if n != "DSS-33"),
                  key=lambda n: np.linalg.norm(dsn.dsn_station_itrf93(n, et0)[0] - rounded.r_ecef_m))
    rng, dop = effect(rounded, real)
    out["rounded_scenario_stations"][label] = {
        "antenna": antenna, "offset_km": float(np.linalg.norm(rounded.r_ecef_m - real.r_ecef_m) / 1e3),
        "nearest_antenna": nearest,
        "nearest_offset_km": float(np.linalg.norm(rounded.r_ecef_m - dsn.dsn_station_itrf93(nearest, et0)[0]) / 1e3),
        "two_way_range_change_m": rng, "doppler_change_mps": dop}
print(json.dumps(out, indent=1))
json.dump(out, open(sys.argv[1], "w"), indent=1)
