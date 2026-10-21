"""Phase 17-HARDEN-D18 / D19: interpolators no longer extrapolate without a bound.

D18 -- ``accelerated.make_lerp_vec3`` / ``rk4_6state`` (fast UKF sigma
propagation) raise ``EphemerisCoverageError`` beyond the HARDEN-D15 10 s
ephemeris margin; the njit kernels are untouched.

D19 -- the generic history interpolators in ``radiometrics`` extrapolate the edge
interval by at most the same 10 s light-time pre-roll margin unless a caller
passes an explicit ``max_extrapolation_s`` (the test-side VODE ephemeris
callables do); the counted-Doppler slope helper and
``filters._interp_pass_values`` accept only closed support plus the two-ULP
representation allowance.

Every accepted request must return exactly the legacy bytes; the legacy
formulas are written out here independently of the code under test.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import lunar_od.accelerated as accelerated
import lunar_od.filters as filters
import lunar_od.radiometrics as radiometrics
from lunar_od.ephemeris import DEFAULT_EPHEMERIS_EXTRAPOLATION_MARGIN_S, EphemerisCoverageError
from lunar_od.history_domain import HistoryDomainError, require_supported_epoch

H_S = 60.0
T_GRID = np.arange(0.0, 1200.0 + 0.5 * H_S, H_S)
RNG = np.random.default_rng(20261021)
STATES = np.cumsum(RNG.normal(size=(T_GRID.size, 6)), axis=0) * 1e3
PHI = RNG.normal(size=(T_GRID.size, 6, 6))


def _legacy_edge_linear(t_grid, values, t):
    """origin/main radiometrics._interp_1d_linear_extrap edge branch, per column."""
    i0, i1 = (0, 1) if t <= t_grid[0] else (t_grid.size - 2, t_grid.size - 1)
    dt = t_grid[i1] - t_grid[i0]
    return np.array([float(values[i0, c] + (values[i1, c] - values[i0, c]) / dt * (t - t_grid[i0]))
                     for c in range(values.shape[1])])


def _bitwise(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


# ======================================================================
# D19 -- radiometrics generic interpolators
# ======================================================================
def test_d19_ledger_reproduction_now_raises_with_diagnostics():
    with pytest.raises(HistoryDomainError) as excinfo:
        radiometrics._interp_state(T_GRID, STATES, T_GRID[-1] + 100.0)
    err = excinfo.value
    assert err.consumer == "radiometrics._interp_vector"
    assert err.support_start_s == T_GRID[0] and err.support_end_s == T_GRID[-1]
    assert err.required_post_roll_s == pytest.approx(100.0)
    assert err.required_pre_roll_s == 0.0
    assert "refusing to extrapolate" in str(err)


@pytest.mark.parametrize("t", [T_GRID[-1] + 1.5 * H_S, T_GRID[0] - 1.5 * H_S, T_GRID[-1] + 1e4])
def test_d19_every_generic_helper_fails_closed(t):
    with pytest.raises(HistoryDomainError):
        radiometrics._interp_vector(T_GRID, STATES, t)
    with pytest.raises(HistoryDomainError):
        radiometrics._interp_matrix(T_GRID, PHI, t)
    with pytest.raises(HistoryDomainError):
        radiometrics._interp_state(T_GRID, STATES, t)
    with pytest.raises(HistoryDomainError):
        radiometrics.interp_state_history(T_GRID, STATES, t)
    with pytest.raises(HistoryDomainError):
        radiometrics._interp_state_transition_position(T_GRID, PHI, t)


@pytest.mark.parametrize("t", [
    T_GRID[-1] + 0.0426,              # ΔDOR station B: one Earth-diameter light time
    T_GRID[0] - 2.7,                  # M3 uplink t1: one round-trip light time
    T_GRID[-1] + 10.0,                # exactly the margin: accepted
    T_GRID[0] - 10.0,
    T_GRID[-1],                       # endpoints keep their legacy edge branch
    T_GRID[0],
])
def test_d19_accepted_edge_requests_return_the_legacy_bytes(t):
    assert _bitwise(radiometrics._interp_vector(T_GRID, STATES, t), _legacy_edge_linear(T_GRID, STATES, t))
    assert _bitwise(radiometrics._interp_state(T_GRID, STATES, t), _legacy_edge_linear(T_GRID, STATES, t))
    flat = PHI.reshape(T_GRID.size, -1)
    assert _bitwise(radiometrics._interp_state_transition_position(T_GRID, PHI, t),
                    _legacy_edge_linear(T_GRID, flat, t).reshape(6, 6)[:3, :])


def test_d19_interior_requests_are_untouched():
    for t in (1.0, 437.25, T_GRID[-1] - 1e-9):
        assert _bitwise(radiometrics._interp_vector(T_GRID, STATES, t),
                        np.array([float(np.interp(t, T_GRID, STATES[:, c])) for c in range(6)]))


def test_d19_default_bound_is_the_d15_ephemeris_margin():
    assert radiometrics.GENERIC_HISTORY_MAX_EXTRAPOLATION_S == DEFAULT_EPHEMERIS_EXTRAPOLATION_MARGIN_S == 10.0
    for t in (T_GRID[0] - 10.0 - 1e-6, T_GRID[-1] + 10.0 + 1e-6):
        with pytest.raises(HistoryDomainError):
            radiometrics._interp_vector(T_GRID, STATES, t)
        with pytest.raises(HistoryDomainError):
            radiometrics._interp_state(T_GRID, STATES, t)


def test_d19_explicit_opt_in_extends_the_bound_and_keeps_the_legacy_bytes():
    t = T_GRID[-1] + 1.46 * H_S                        # worst measured VODE overrun
    with pytest.raises(HistoryDomainError):
        radiometrics._interp_state(T_GRID, STATES, t)
    got = radiometrics._interp_state(T_GRID, STATES, t, max_extrapolation_s=300.0)
    assert _bitwise(got, _legacy_edge_linear(T_GRID, STATES, t))
    with pytest.raises(HistoryDomainError):
        radiometrics._interp_state(T_GRID, STATES, T_GRID[-1] + 300.5, max_extrapolation_s=300.0)
    with pytest.raises(ValueError):
        radiometrics._interp_vector(T_GRID, STATES, t, max_extrapolation_s=-1.0)


@pytest.mark.parametrize("fn,arr", [
    (radiometrics._interp_vector, STATES),
    (radiometrics._interp_state, STATES),
    (radiometrics._interp_state_transition_position, PHI),
])
def test_d19_non_finite_epochs_raise_history_domain_error(fn, arr):
    for t in (math.nan, math.inf, -math.inf):
        with pytest.raises(HistoryDomainError):
            fn(T_GRID, arr, t)


def test_d19_counted_doppler_slope_helper_is_strict():
    vals = STATES[:, :3]
    end = float(T_GRID[-1])
    near = math.nextafter(end, math.inf)               # one ULP: representation allowance
    value, slope = radiometrics._interp_array_and_slope(T_GRID, vals, near)
    assert np.all(np.isfinite(value)) and np.all(np.isfinite(slope))
    with pytest.raises(HistoryDomainError):
        radiometrics._interp_array_and_slope(T_GRID, vals, end + 1e-3)
    with pytest.raises(HistoryDomainError):
        radiometrics._interp_array_and_slope(T_GRID, vals, -1e-3)


def test_d19_vode_ephemeris_callable_over_a_grid_that_just_covers_the_arc():
    """The motivating need: VODE probes the RHS past the last output epoch."""
    from lunar_od.dynamics import propagate_state

    earth = np.zeros((T_GRID.size, 6))
    earth[:, 0] = 3.84e8 + 1.0e3 * T_GRID
    earth[:, 3] = 1.0e3

    def get_earth_pos(t):
        return np.array([radiometrics._interp_state(T_GRID, earth, float(ti), max_extrapolation_s=300.0)[:3]
                         for ti in np.atleast_1d(np.asarray(t, dtype=float))])

    def get_sun_pos(t):
        return np.tile([1.496e11, 0.0, 0.0], (np.atleast_1d(t).size, 1))

    r0 = 1837.4e3
    out = propagate_state(T_GRID, [r0, 0.0, 0.0, 0.0, math.sqrt(4902.800066e9 / r0), 0.0],
                          4902.800066e9, 3.986004418e14, 0.0, get_earth_pos, get_sun_pos)
    assert out.shape == (T_GRID.size, 6) and np.all(np.isfinite(out))


def test_d19_require_supported_epoch_contract():
    require_supported_epoch(5.0, 0.0, 10.0, history_name="h", model_context="m", consumer="c")
    require_supported_epoch(12.0, 0.0, 10.0, history_name="h", model_context="m", consumer="c",
                            max_extrapolation_s=2.0)
    with pytest.raises(HistoryDomainError) as excinfo:
        require_supported_epoch(12.5, 0.0, 10.0, history_name="h", model_context="m", consumer="c",
                                max_extrapolation_s=2.0)
    assert excinfo.value.support_end_s == 10.0           # diagnostics quote the true support
    assert excinfo.value.required_post_roll_s == pytest.approx(2.5)
    with pytest.raises(ValueError):
        require_supported_epoch(5.0, 0.0, 10.0, history_name="h", model_context="m", consumer="c",
                                max_extrapolation_s=-1.0)


# ======================================================================
# D19 -- filters
# ======================================================================
def test_d19_filters_pass_values_is_strict_and_bitwise_inside():
    targets = np.array([0.0, 30.0, 599.5, T_GRID[-1]])
    got = filters._interp_pass_values(T_GRID, STATES, targets)
    want = np.column_stack([np.interp(targets, T_GRID, STATES[:, c]) for c in range(6)])
    assert _bitwise(got, want)
    rot = PHI[:, :3, :3]
    assert filters._interp_pass_values(T_GRID, rot, targets).shape == (4, 3, 3)
    filters._interp_pass_values(T_GRID, STATES, np.array([math.nextafter(T_GRID[-1], math.inf)]))
    for bad in (T_GRID[-1] + 1e-3, T_GRID[0] - 1e-3, math.nan):
        with pytest.raises(HistoryDomainError):
            filters._interp_pass_values(T_GRID, STATES, np.array([10.0, bad]))


def test_d19_dead_unbounded_filters_helper_removed():
    assert not hasattr(filters, "_interp_1d_linear_extrap")


# ======================================================================
# D18 -- accelerated ephemeris lerp / fast RK4
# ======================================================================
EPH_T = np.arange(0.0, 3600.0 + 30.0, 60.0)
EARTH = np.column_stack([3.84e8 + 1.0e3 * EPH_T, 2.0e2 * EPH_T, np.full(EPH_T.size, 1.0e6)])
SUN = np.tile([1.496e11, 0.0, 0.0], (EPH_T.size, 1))


def test_d18_margin_is_the_d15_ephemeris_margin():
    assert accelerated.EPHEMERIS_GRID_EXTRAPOLATION_MARGIN_S == DEFAULT_EPHEMERIS_EXTRAPOLATION_MARGIN_S == 10.0


@pytest.mark.parametrize("t", [EPH_T[-1] + 1e4, EPH_T[0] - 1e4, EPH_T[-1] + 10.001, math.nan])
def test_d18_ledger_reproduction_now_raises(t):
    with pytest.raises(EphemerisCoverageError):
        accelerated.make_lerp_vec3(EPH_T, EARTH)(t)


@pytest.mark.parametrize("t", [EPH_T[-1] + 10.0, EPH_T[0] - 10.0, EPH_T[-1] + 4.0, 1234.5])
def test_d18_accepted_queries_return_the_kernel_bytes(t):
    got = accelerated.make_lerp_vec3(EPH_T, EARTH)(t)
    if accelerated.NUMBA_AVAILABLE:
        assert _bitwise(got, accelerated._lerp_vec3_numba(float(t), EPH_T, EARTH))
    assert np.all(np.isfinite(got))


def _rk4(t0, tf):
    r0 = 1837.4e3
    y0 = np.array([r0, 0.0, 0.0, 0.0, math.sqrt(4902.800066e9 / r0), 0.0])
    return accelerated.rk4_6state(y0, t0, tf, 4902.800066e9, 3.986004418e14, 1.32712440018e20,
                                  EPH_T, EARTH, SUN, dt=10.0)


def test_d18_rk4_rejects_intervals_leaving_the_margined_grid():
    for t0, tf in ((3500.0, 3700.0), (100.0, -20.0), (-20.0, 60.0)):
        with pytest.raises(EphemerisCoverageError):
            _rk4(t0, tf)


def test_d18_rk4_accepted_intervals_are_the_kernel_bytes():
    for t0, tf in ((0.0, 60.0), (3540.0, 3610.0), (600.0, 540.0), (-10.0, 50.0)):
        got = _rk4(t0, tf)
        r0 = 1837.4e3
        y0 = np.array([r0, 0.0, 0.0, 0.0, math.sqrt(4902.800066e9 / r0), 0.0])
        want = accelerated._rk4_6state_numba(
            y0, float(t0), float(tf), 10.0, 4902.800066e9, 3.986004418e14, 1.32712440018e20,
            EPH_T, EARTH, SUN, 0.0, 0.0, np.eye(3), 0.0, 0.0, 0, np.eye(3))
        assert _bitwise(got, want)


def test_d18_fast_sigma_propagator_inherits_the_guard():
    from types import SimpleNamespace

    from lunar_od.dynamics import make_fast_sigma_propagator

    eph = SimpleNamespace(t_ephem_s=EPH_T, earth_pos_m=EARTH, sun_pos_m=SUN)
    prop = make_fast_sigma_propagator(eph, 4902.800066e9, 3.986004418e14, 1.32712440018e20)
    if prop is None:
        pytest.skip("numba unavailable: the fast sigma propagator is not built")
    r0 = 1837.4e3
    x = np.array([r0, 0.0, 0.0, 0.0, math.sqrt(4902.800066e9 / r0), 0.0])
    assert np.all(np.isfinite(prop(0.0, 60.0, x)))
    with pytest.raises(EphemerisCoverageError):
        prop(3600.0, 3700.0, x)
