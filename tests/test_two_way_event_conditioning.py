"""Phase 17C - conditioning of the two-way event solve.

Phase 17B traced the measurement resolution floor to one line: the round-trip
light time was recovered by differencing two arc-relative event epochs,

    round_trip_light_time_s = t3 - t1

with t3 and t1 both of order 1e4 s and their difference of order 1 s. That
subtraction quantises the observable at

    q = c * ulp(t_event) / 2

which was measured at 2.726e-04 m for an event epoch near 14157 s, and which
doubles every time the arc crosses a binary exponent boundary.

The solver's own converged local delays were already available at that point.
The identity is exact:

    t1  = t2u - uplink_lt
    t2u = t2d - transponder_delay
    t2d = t3  - downlink_lt
    =>  t3 - t1 == downlink_lt + transponder_delay + uplink_lt

These tests pin the repaired behaviour. Before the fix they fail; the mechanism
they protect is the quantisation law, not a loose error bound.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lunar_od.two_way_range import (
    TwoWayRangeConfig,
    make_exact_sxform_station_state_provider,
    solve_two_way_range_events,
    two_way_range_from_solution,
)

pytest.importorskip("spiceypy")

C_LIGHT = 299792458.0

#: Arc-relative receive epochs. 6000 s and 9600 s straddle the 8192 s binade
#: boundary where the old quantum doubled.
RECEIVE_EPOCHS_S = (6000.0, 9600.0, 14160.0)


@pytest.fixture(scope="module")
def fixture():
    """A short lunar arc with real station geometry, built once."""
    import spiceypy as spice

    from lunar_od.config import position_only_stations
    from lunar_od.constants import J2_MOON_UNNORMALIZED, MU_MOON_M3S2
    from lunar_od.dynamics import propagate_state
    from lunar_od.spice_loader import load_spice_kernels

    load_spice_kernels(None, clear=True)
    et0 = 857806356.99998                     # the frozen arc's own epoch
    t_grid = np.arange(0.0, 15000.0, 60.0)
    x0 = np.array([-1746513.5335792408, -277083.98513177276, 518734.13974716066,
                   -138.53981552770264, -1206.7373147449678, -1085.0511650068934])

    def earth_at(_t):
        return np.array([384_400e3, 0.0, 0.0])

    def sun_at(_t):
        return np.array([149.6e9, 0.0, 0.0])

    states = propagate_state(
        t_grid, x0, MU_MOON_M3S2, 0.0, 0.0, earth_at, sun_at,
        rtol=1e-13, atol=1e-16, j2_moon=J2_MOON_UNNORMALIZED,
    )
    station = [s for s in position_only_stations() if "Goldstone" in s.name][0]
    n = t_grid.size
    earth_pos = np.tile(np.array([384_400e3, 0.0, 0.0]), (n, 1))
    earth_vel = np.zeros((n, 3))
    provider = make_exact_sxform_station_state_provider(
        station, float(et0), t_grid, earth_pos, earth_vel)
    cfg = TwoWayRangeConfig(tolerance_s=1e-13, equation_tolerance_s=1e-14,
                            max_iter=400)
    return t_grid, states, provider, cfg


def _range_at(fixture, t_rx, states=None):
    t_grid, nominal, provider, cfg = fixture
    sol = solve_two_way_range_events(
        t_rx, provider, t_grid, nominal if states is None else states, cfg)
    return sol, two_way_range_from_solution(sol, cfg)


def _perturbed(states, scale_m):
    """Shift the whole history along +x by a known amount."""
    out = states.copy()
    out[:, 0] += scale_m
    return out


# ======================================================================
# The identity the repair rests on
# ======================================================================
@pytest.mark.parametrize("t_rx", RECEIVE_EPOCHS_S)
def test_round_trip_light_time_equals_the_sum_of_local_delays(fixture, t_rx):
    """t3 - t1 == downlink + transponder + uplink, exactly.

    Both sides are computed here; the point is that the solver must REPORT the
    well-conditioned one. Pre-fix the reported value is the epoch difference and
    disagrees with the local sum by a fraction of ulp(t3).
    """
    sol, _ = _range_at(fixture, t_rx)
    local_sum = (sol.downlink_light_time_s + sol.transponder_delay_s
                 + sol.uplink_light_time_s)
    assert sol.round_trip_light_time_s == pytest.approx(local_sum, abs=1e-15), (
        "round_trip_light_time_s is not the sum of the solver's own local "
        "delays; it is being recovered by differencing large event epochs"
    )


@pytest.mark.parametrize("t_rx", RECEIVE_EPOCHS_S)
def test_reported_light_time_is_not_quantised_to_the_event_epoch(fixture, t_rx):
    """The reported round-trip light time must not be a multiple of ulp(t3).

    An epoch difference lands exactly on the ulp(t3) lattice. A sum of local
    delays essentially never does.
    """
    sol, _ = _range_at(fixture, t_rx)
    q = math.ulp(abs(sol.t3_s))
    residue = sol.round_trip_light_time_s / q
    assert abs(residue - round(residue)) > 1e-6, (
        f"round_trip_light_time_s is an exact multiple of ulp(t3)={q:.3e} s, "
        "which is the signature of the epoch-difference formulation"
    )


# ======================================================================
# The quantisation law itself
# ======================================================================
@pytest.mark.parametrize("t_rx", RECEIVE_EPOCHS_S)
def test_range_resolves_below_the_old_event_epoch_quantum(fixture, t_rx):
    """The observable must respond to perturbations far below c*ulp(t3)/2.

    This is the mechanism test. Pre-fix the range is dead to any state change
    smaller than that quantum; post-fix it must resolve at least two orders
    finer.
    """
    t_grid, states, provider, cfg = fixture
    sol0, r0 = _range_at(fixture, t_rx)
    old_quantum = C_LIGHT * math.ulp(abs(sol0.t3_s)) / 2.0
    probe = old_quantum / 100.0
    _, r = _range_at(fixture, t_rx, _perturbed(states, probe))
    assert r != r0, (
        f"range did not move for a {probe:.3e} m shift, i.e. it is still "
        f"quantised at the old {old_quantum:.3e} m event-epoch scale"
    )


@pytest.mark.parametrize("t_rx", RECEIVE_EPOCHS_S)
def test_range_response_is_linear_well_below_the_old_quantum(fixture, t_rx):
    """Halving a sub-quantum perturbation must halve the response.

    A staircase cannot do this; a well-conditioned observable can.
    """
    t_grid, states, provider, cfg = fixture
    sol0, r0 = _range_at(fixture, t_rx)
    old_quantum = C_LIGHT * math.ulp(abs(sol0.t3_s)) / 2.0
    big = old_quantum / 50.0
    _, r_big = _range_at(fixture, t_rx, _perturbed(states, big))
    _, r_half = _range_at(fixture, t_rx, _perturbed(states, big / 2.0))
    d_big, d_half = r_big - r0, r_half - r0
    assert d_big != 0.0 and d_half != 0.0
    assert d_half / d_big == pytest.approx(0.5, rel=0.05)


def test_the_quantum_no_longer_doubles_across_a_binade(fixture):
    """The old floor doubled at 8192 s. The repaired one must not.

    6000 s and 9600 s sit on opposite sides of that boundary, so the old
    formulation gave them quanta in a 1:2 ratio.
    """
    t_grid, states, _, _ = fixture
    increments = {}
    for t_rx in (6000.0, 9600.0):
        sol0, r0 = _range_at(fixture, t_rx)
        old_quantum = C_LIGHT * math.ulp(abs(sol0.t3_s)) / 2.0
        probe = old_quantum / 100.0
        _, r = _range_at(fixture, t_rx, _perturbed(states, probe))
        increments[t_rx] = abs(r - r0) / probe        # response per metre
    ratio = increments[9600.0] / increments[6000.0]
    # Both are now ordinary O(1) range sensitivities, not epoch quanta.
    assert 0.2 < ratio < 5.0, (
        f"response ratio across the 8192 s binade is {ratio:.3f}; a value near "
        "the old 1:2 quantum ratio means the epoch coupling survives"
    )


# ======================================================================
# The repair must not have changed the physics
# ======================================================================
@pytest.mark.parametrize("t_rx", RECEIVE_EPOCHS_S)
def test_event_equations_still_close(fixture, t_rx):
    sol, _ = _range_at(fixture, t_rx)
    assert sol.converged
    assert sol.uplink_equation_residual_s <= 1e-14
    assert sol.downlink_equation_residual_s <= 1e-14


@pytest.mark.parametrize("t_rx", RECEIVE_EPOCHS_S)
def test_event_ordering_preserved(fixture, t_rx):
    sol, _ = _range_at(fixture, t_rx)
    assert sol.t1_s < sol.t2u_s <= sol.t2d_s < sol.t3_s


@pytest.mark.parametrize("t_rx", RECEIVE_EPOCHS_S)
def test_range_still_matches_the_epoch_form_within_the_old_quantum(fixture, t_rx):
    """Physics parity: the repair may only move the answer by about the amount
    of precision the old form was throwing away."""
    sol, r = _range_at(fixture, t_rx)
    epoch_form_range = 0.5 * C_LIGHT * (
        (sol.t3_s - sol.t1_s) - sol.transponder_delay_s)
    old_quantum = C_LIGHT * math.ulp(abs(sol.t3_s)) / 2.0
    assert abs(r - epoch_form_range) <= 2.0 * old_quantum


@pytest.mark.parametrize("t_rx", RECEIVE_EPOCHS_S)
def test_light_times_are_physically_sane(fixture, t_rx):
    sol, _ = _range_at(fixture, t_rx)
    assert 1.0 < sol.uplink_light_time_s < 2.0
    assert 1.0 < sol.downlink_light_time_s < 2.0
    assert sol.uplink_light_time_s > 0.0 and sol.downlink_light_time_s > 0.0
