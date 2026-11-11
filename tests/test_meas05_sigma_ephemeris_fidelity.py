"""MEAS-05 / X-04: fidelity of the fast UKF sigma propagator's ephemeris.

``make_fast_sigma_propagator`` interpolates the Moon-centred Earth and Sun
linearly on the 600 s grid; the batch path uses cubic Hermite (Earth) and PCHIP
(Sun).  The Earth position error reaches ~117 m at mid-grid, but the
perturbation it causes is a ~1e-9 m/s^2 tidal-acceleration error.  Measured on
DE440 (2024-03-01, 100 km polar-ish orbit) against the batch-path physics
integrated with DOP853 at rtol 1e-13: 4.6e-6 m after 60 s (a sigma step),
4.3e-5 m after 600 s, 9.5e-4 m after 1 h, 4.1e-2 m after 1 day.  These bounds
include the fixed-step RK4 error.  The sigma propagator's error is therefore
immaterial against any tracking noise, and the linear interpolation is kept;
this test pins the bound so a future change cannot silently degrade it.
"""
from __future__ import annotations

import numpy as np
import pytest

X0 = np.array([1.8374e6, 0.0, 2.0e5, 0.0, 1.6e3, 0.0])


@pytest.fixture(scope="module")
def de440_case():
    spiceypy = pytest.importorskip("spiceypy")
    from lunar_od.spice_loader import load_spice_kernels
    try:
        load_spice_kernels(None, clear=False)
    except FileNotFoundError:
        pytest.skip("SPICE kernels not available")
    from lunar_od.ephemeris import sample_moon_centered_ephemeris
    from lunar_od.ephemeris_generation import gravitational_parameters

    et0 = spiceypy.str2et("2024-03-01T00:00:00")
    ephemeris = sample_moon_centered_ephemeris(et0, np.arange(0.0, 86400.0 + 600.0, 600.0))
    return ephemeris, gravitational_parameters()


def test_linear_earth_interpolation_error_is_what_meas05_reported(de440_case):
    ephemeris, _ = de440_case
    t = ephemeris.t_ephem_s
    linear = 0.5 * (ephemeris.earth_pos_m[:-1] + ephemeris.earth_pos_m[1:])
    hermite = ephemeris.earth_position(t[:-1] + 300.0)
    worst = np.max(np.linalg.norm(linear - hermite, axis=1))
    assert 80.0 < worst < 160.0          # measured 116.9 m


@pytest.mark.parametrize("span_s, bound_m", [(60.0, 2e-5), (600.0, 2e-4), (3600.0, 5e-3), (86400.0, 0.2)])
def test_fast_sigma_propagator_matches_the_batch_ephemeris_path(de440_case, span_s, bound_m):
    from lunar_od.dynamics import make_fast_sigma_propagator, propagate_state

    ephemeris, (mu_moon, mu_earth, mu_sun) = de440_case
    propagate = make_fast_sigma_propagator(ephemeris, mu_moon, mu_earth, mu_sun)
    if propagate is None:
        pytest.skip("numba unavailable")
    fast = propagate(0.0, span_s, X0)
    reference = np.asarray(propagate_state(
        np.array([0.0, span_s]), X0, mu_moon, mu_earth, mu_sun,
        ephemeris.earth_position, ephemeris.sun_position,
        rtol=1e-13, atol=1e-12, method="DOP853"))[-1]
    # Bounds are ~4-5x the measured gaps (module docstring).
    assert np.linalg.norm(fast[:3] - reference[:3]) < bound_m
