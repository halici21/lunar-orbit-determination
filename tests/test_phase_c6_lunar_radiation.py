"""Phase C6 lunar radiation pressure model against closed-form Lambertian-sphere fluxes (gate G_UNIT)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "studies" / "phase17" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import phase_c6_lunar_radiation as L  # noqa: E402
from lunar_od.constants import AU_M, SPEED_OF_LIGHT_M_S  # noqa: E402
from lunar_od.srp import solar_pressure_at  # noqa: E402

R = L.R_MOON_M
SUN = np.array([AU_M, 0.0, 0.0])
S_1AU = solar_pressure_at(AU_M) * SPEED_OF_LIGHT_M_S


def test_uniform_temperature_sphere_flux_is_sigma_t4_r2_over_r2():
    temp = 250.0
    lum = L.STEFAN_BOLTZMANN * temp ** 4 / np.pi
    for r in (R + 50e3, 10 * R):
        pos = np.array([[0.3, -0.5, 0.81]]) / np.linalg.norm([0.3, -0.5, 0.81]) * r
        a = L.acceleration_kernel(pos, SUN, radiance_fn=lambda nrm, s: np.full(nrm.shape[:-1], lum))[0]
        want = L.STEFAN_BOLTZMANN * temp ** 4 * (R / r) ** 2 / SPEED_OF_LIGHT_M_S
        assert abs(np.linalg.norm(a) / want - 1.0) < 1e-3
        assert np.linalg.norm(np.cross(a, pos[0])) / np.linalg.norm(a) / r < 1e-9


def test_zero_phase_lambert_albedo_far_field():
    model = L.LunarRadiation(thermal=False)
    r = 500 * R                                    # far field: finite-distance terms are O(R/r)
    a = L.acceleration_kernel(np.array([[r, 0.0, 0.0]]), SUN, model)[0]
    want = 2.0 / 3.0 * model.albedo * S_1AU * (R / r) ** 2 / SPEED_OF_LIGHT_M_S
    assert abs(a[0] / want - 1.0) < 1e-2


def test_over_the_subsolar_point_the_acceleration_is_radial_and_outward():
    a = L.acceleration_kernel(np.array([[R + 50e3, 0.0, 0.0]]), SUN)[0]
    assert a[0] > 0.0
    assert np.linalg.norm(a[1:]) <= 1e-6 * np.linalg.norm(a)


def test_night_side_is_the_100_k_floor_and_dayside_is_near_the_solar_flux():
    day = L.acceleration_kernel(np.array([[R + 50e3, 0.0, 0.0]]), SUN)[0]
    night = L.acceleration_kernel(np.array([[-(R + 50e3), 0.0, 0.0]]), SUN)[0]
    srp = solar_pressure_at(AU_M)
    assert 0.5 < np.linalg.norm(day) / srp < 1.2
    want_night = 0.95 * L.STEFAN_BOLTZMANN * 100.0 ** 4 * (R / (R + 50e3)) ** 2 / SPEED_OF_LIGHT_M_S
    assert abs(np.linalg.norm(night) / want_night - 1.0) < 1e-3


def test_forced_columns_match_the_closed_form_oscillator():
    """A3 integrator: x'' = -n^2 x + U (constant forcing) from rest, and the local transitions."""
    import phase_c6_forced_columns as FC
    n = 2.0 * np.pi / 7200.0
    u = np.array([0.6, 0.0, 0.8])
    grid = np.arange(0.0, 3 * 7200.0 + 1.0, 20.0)
    r0 = np.array([1.8e6, 0.0, 0.0])
    v0 = np.array([0.0, 1.8e6 * n, 0.0])
    c, s = np.cos(n * grid), np.sin(n * grid)
    states = np.hstack([np.outer(c, r0) + np.outer(s / n, v0), np.outer(-n * s, r0) + np.outer(c, v0)])
    forcing = np.repeat(u[None, :, None], grid.size, axis=0)
    out, trans = FC.forced_columns(grid, states, forcing, lambda t, r: -n ** 2 * np.eye(3), n_sub=2,
                                   local_transitions=True)
    want_r = np.outer((1.0 - c) / n ** 2, u)
    want_v = np.outer(s / n, u)
    assert np.max(np.abs(out[:, :3, 0] - want_r)) <= 1e-9 * np.max(np.abs(want_r))
    assert np.max(np.abs(out[:, 3:, 0] - want_v)) <= 1e-9 * np.max(np.abs(want_v))
    ct, st = np.cos(n * 20.0), np.sin(n * 20.0)
    want = np.block([[ct * np.eye(3), st / n * np.eye(3)], [-n * st * np.eye(3), ct * np.eye(3)]])
    assert np.max(np.abs(trans - want[None]) / (np.abs(want[None]) + 1.0)) < 1e-10      # RK4 truncation 5e-11
