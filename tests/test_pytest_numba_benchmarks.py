from __future__ import annotations

from time import perf_counter

import numpy as np
import pytest

from lunar_od import NUMBA_AVAILABLE, range_rate_stations
from lunar_od.accelerated import geometric_range_rate_observables, position_observables
from lunar_od.geometry import ecef2razel_sez
from lunar_od.radiometrics import instantaneous_geometric_range_rate


def test_numba_position_observables_match_and_speed_up(capsys):
    if not NUMBA_AVAILABLE:
        pytest.skip("Numba is not installed in this environment.")

    state, obs_pos, _, earth_pos, earth_vel, xforms, stations = _synthetic_measurement_case()

    # Warm up JIT compilation so the benchmark measures execution, not compile time.
    position_observables(state, obs_pos[:8], earth_pos, xforms, stations)

    ref_s, ref = _best_time(
        _reference_position_observables,
        state,
        obs_pos,
        earth_pos,
        xforms,
        stations,
        repeats=3,
    )
    fast_s, got = _best_time(
        position_observables,
        state,
        obs_pos,
        earth_pos,
        xforms,
        stations,
        repeats=5,
    )

    np.testing.assert_allclose(got, ref, rtol=0.0, atol=2e-7)
    speedup = ref_s / fast_s
    with capsys.disabled():
        print(
            "\nBENCH position_observables "
            f"n_obs={obs_pos.shape[0]} python_loop_s={ref_s:.6f} "
            f"numba_s={fast_s:.6f} speedup={speedup:.2f}x"
        )
    assert speedup > 1.0


def test_numba_geometric_range_rate_observables_match_and_speed_up(capsys):
    if not NUMBA_AVAILABLE:
        pytest.skip("Numba is not installed in this environment.")

    state, _, obs_rr, earth_pos, earth_vel, xforms, stations = _synthetic_measurement_case()

    # Warm up JIT compilation so the benchmark measures execution, not compile time.
    geometric_range_rate_observables(state, obs_rr[:8], earth_pos, earth_vel, xforms, stations)

    ref_s, ref = _best_time(
        _reference_range_rate_observables,
        state,
        obs_rr,
        earth_pos,
        earth_vel,
        xforms,
        stations,
        repeats=3,
    )
    fast_s, got = _best_time(
        geometric_range_rate_observables,
        state,
        obs_rr,
        earth_pos,
        earth_vel,
        xforms,
        stations,
        repeats=5,
    )

    np.testing.assert_allclose(got, ref, rtol=0.0, atol=2e-7)
    speedup = ref_s / fast_s
    with capsys.disabled():
        print(
            "\nBENCH geometric_range_rate_observables "
            f"n_obs={obs_rr.shape[0]} python_loop_s={ref_s:.6f} "
            f"numba_s={fast_s:.6f} speedup={speedup:.2f}x"
        )
    assert speedup > 1.0


def _synthetic_measurement_case(n_steps: int = 1500, n_stations: int = 4):
    stations = range_rate_stations()[:n_stations]
    t_s = np.arange(n_steps, dtype=float) * 60.0
    theta = 2.0 * np.pi * t_s / (12.0 * 3600.0)

    state = np.zeros((n_steps, 6), dtype=float)
    state[:, 0] = 3.84e8 + 1.6e6 * np.cos(theta)
    state[:, 1] = 2.0e7 + 1.2e6 * np.sin(theta)
    state[:, 2] = 5.0e6 * np.sin(0.5 * theta)
    state[:, 3] = -1.6e6 * np.sin(theta) * (2.0 * np.pi / (12.0 * 3600.0))
    state[:, 4] = 1.2e6 * np.cos(theta) * (2.0 * np.pi / (12.0 * 3600.0))
    state[:, 5] = 5.0e6 * np.cos(0.5 * theta) * (np.pi / (12.0 * 3600.0))

    earth_pos = np.zeros((n_steps, 3), dtype=float)
    earth_vel = np.zeros((n_steps, 3), dtype=float)
    xforms = np.zeros((n_steps, 6, 6), dtype=float)
    xforms[:, :, :] = np.eye(6, dtype=float)

    obs_pos = np.zeros((n_steps * n_stations, 6), dtype=float)
    obs_rr = np.zeros((n_steps * n_stations, 7), dtype=float)
    row = 0
    for ti in range(n_steps):
        for si in range(n_stations):
            station_id = si + 1
            time_index = ti + 1
            obs_pos[row, :] = [t_s[ti], 0.0, 0.0, 0.0, station_id, time_index]
            obs_rr[row, :] = [t_s[ti], 0.0, 0.0, 0.0, 0.0, station_id, time_index]
            row += 1

    return state, obs_pos, obs_rr, earth_pos, earth_vel, xforms, stations


def _reference_position_observables(state_history, obs_data, earth_pos, xforms, stations):
    out = np.zeros((obs_data.shape[0], 3), dtype=float)
    for obs_idx in range(obs_data.shape[0]):
        station_id = int(obs_data[obs_idx, 4]) - 1
        time_idx = int(obs_data[obs_idx, 5]) - 1
        station = stations[station_id]
        r_sat_eci = state_history[time_idx, :3] - earth_pos[time_idx, :]
        r_sat_ecef = xforms[time_idx, :3, :3] @ r_sat_eci
        rho_ecef = r_sat_ecef - station.r_ecef_m
        az, el, range_m = ecef2razel_sez(rho_ecef, station.lat_rad, station.lon_rad)
        out[obs_idx, :] = [range_m, az, el]
    return out


def _reference_range_rate_observables(state_history, obs_data, earth_pos, earth_vel, xforms, stations):
    out = np.zeros((obs_data.shape[0], 4), dtype=float)
    for obs_idx in range(obs_data.shape[0]):
        station_id = int(obs_data[obs_idx, 5]) - 1
        time_idx = int(obs_data[obs_idx, 6]) - 1
        station = stations[station_id]
        state_eci = state_history[time_idx, :] - np.concatenate(
            [earth_pos[time_idx, :], earth_vel[time_idx, :]]
        )
        state_ecef = xforms[time_idx, :, :] @ state_eci
        rho_ecef = state_ecef[:3] - station.r_ecef_m
        v_ecef = state_ecef[3:]
        az, el, range_m = ecef2razel_sez(rho_ecef, station.lat_rad, station.lon_rad)
        rr_mps = instantaneous_geometric_range_rate(rho_ecef, v_ecef)
        out[obs_idx, :] = [range_m, rr_mps, az, el]
    return out


def _best_time(func, *args, repeats: int):
    best_s = float("inf")
    best_value = None
    for _ in range(repeats):
        t0 = perf_counter()
        value = func(*args)
        elapsed_s = perf_counter() - t0
        if elapsed_s < best_s:
            best_s = elapsed_s
            best_value = value
    return best_s, best_value
