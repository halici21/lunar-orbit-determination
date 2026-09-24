"""Q1-F08: the light-time UPDATE criterion must stay attainable at large epochs.

Historical defect
-----------------
The update criterion compared ``light_time_tolerance_s`` (1e-10 s) against
``abs(new_t2 - t2)``, a difference of two *absolute* epochs. Both operands are
first rounded onto the representable grid of spacing ``ulp(t)``, and their
difference is then exact by the Sterbenz lemma, so the residual can only take the
values ``0, ulp(t), 2*ulp(t), ...``. Once ``ulp(t) > 1e-10`` -- that is, from
``t = 2^19 s = 524_288 s = 145.64 h`` -- the tolerance falls strictly between zero
and one ulp, so an iterate that converges without becoming exactly stationary can
never be accepted, no matter how many iterations are allowed.

The repair evaluates the same movement on the LOCAL light time, where
``ulp(tau) ~ 2e-16 s``. **No tolerance was changed.**

These tests protect that property. They deliberately do not assert on solver
internals: they assert the observable consequence -- that a solve at an epoch
above the historical boundary succeeds with its physical residuals inside the
unchanged tolerance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from examples import r1_r4_long_arc_qualification as harness
from lunar_od.radiometrics import (
    RangeRatePhysicsConfig,
    resolve_counted_doppler_station_state_provider,
    two_way_counted_doppler_observable,
)
from lunar_od.two_way_range import TwoWayRangeConfig, solve_two_way_range_events

#: The epoch at which the historical defect first became observable in the
#: Phase-1 long-arc campaign: G20 / H5 / Tc = 100 s / cadence = 60 s.
Q1_F08_HISTORICAL_FAILURE_EPOCH_S = 542190.0

#: Onset of the historical failure surface. Derived, not measured:
#: ulp(t) > 1e-10 s  <=>  2^(e-52) > 1e-10  <=>  e >= 19.
Q1_F08_BOUNDARY_S = 2.0**19

#: Accepted contract values. Neither is redefined here; they are imported as
#: expectations so that a silent change to either fails this test.
UPDATE_TOLERANCE_S = 1e-10
EQUATION_TOLERANCE_S = 1e-11


@pytest.fixture(scope="module")
def g20_long_arc():
    """The exact geometry and arc that exhibited the historical failure."""
    geometry = {g.geometry_id: g for g in harness.phase1_geometries()}["G20"]
    station = harness._station(geometry)
    t_grid, states, _visibility = harness.truth_arc(
        geometry, harness.HORIZONS_S["H5"], 60.0
    )
    return {
        "station": station,
        "t_grid": t_grid,
        "states": states,
        "earth_pos": harness._earth_pos(t_grid),
        "earth_vel": harness._earth_vel(t_grid),
        "et0": harness.spice_epoch(),
    }


def _provider(arc, count_interval_s, delay_s):
    config = RangeRatePhysicsConfig(
        mode="two_way_counted_doppler",
        count_interval_s=count_interval_s,
        counted_doppler_model="four_event_delay",
        transponder_delay_s=delay_s,
    )
    provider = resolve_counted_doppler_station_state_provider(
        config,
        arc["station"],
        arc["t_grid"],
        arc["earth_pos"],
        arc["earth_vel"],
        et0_s=arc["et0"],
    )
    return config, provider


def test_q1f08_boundary_arithmetic_is_why_this_test_exists():
    """The derivation, as arithmetic. No production code involved.

    Above 2^19 s a one-ulp iterate movement cannot satisfy the accepted update
    tolerance. This is the property the repair had to remove, and it is asserted
    here so that the reason for the other tests survives independently of them.
    """
    below = math.nextafter(Q1_F08_BOUNDARY_S, 0.0)
    assert math.ulp(below) <= UPDATE_TOLERANCE_S, (
        "below 2^19 a one-ulp movement must remain acceptable; "
        f"ulp={math.ulp(below)!r}"
    )
    assert math.ulp(Q1_F08_BOUNDARY_S) > UPDATE_TOLERANCE_S, (
        "at and above 2^19 a one-ulp movement exceeds the update tolerance; "
        f"ulp={math.ulp(Q1_F08_BOUNDARY_S)!r}"
    )
    # The historical failure epoch really is in the affected binade.
    assert Q1_F08_HISTORICAL_FAILURE_EPOCH_S >= Q1_F08_BOUNDARY_S


@pytest.mark.parametrize("delay_s", [0.0, 1e-4, 1e-3])
def test_q1f08_historical_failure_point_converges(g20_long_arc, delay_s):
    """The exact point that fail-closed in the campaign must now solve."""
    config, provider = _provider(g20_long_arc, 100.0, delay_s)
    solution = solve_two_way_range_events(
        Q1_F08_HISTORICAL_FAILURE_EPOCH_S,
        provider,
        g20_long_arc["t_grid"],
        g20_long_arc["states"],
        TwoWayRangeConfig(transponder_delay_s=delay_s),
    )
    assert solution.converged
    assert solution.uplink_equation_residual_s <= EQUATION_TOLERANCE_S
    assert solution.downlink_equation_residual_s <= EQUATION_TOLERANCE_S
    assert (
        solution.t1_s
        < solution.t2u_s
        <= solution.t2d_s
        < Q1_F08_HISTORICAL_FAILURE_EPOCH_S
    )
    assert np.isfinite([solution.t1_s, solution.t2u_s, solution.t2d_s]).all()


@pytest.mark.parametrize(
    "receive_time_s",
    [
        Q1_F08_BOUNDARY_S - 1.0,
        Q1_F08_BOUNDARY_S,
        Q1_F08_BOUNDARY_S + 1.0,
        Q1_F08_HISTORICAL_FAILURE_EPOCH_S,
        604000.0,
    ],
)
def test_q1f08_update_criterion_is_attainable_across_the_boundary(
    g20_long_arc, receive_time_s
):
    """Crossing 2^19 must not change whether a solve is accepted."""
    config, provider = _provider(g20_long_arc, 100.0, 1e-3)
    solution = solve_two_way_range_events(
        receive_time_s,
        provider,
        g20_long_arc["t_grid"],
        g20_long_arc["states"],
        TwoWayRangeConfig(transponder_delay_s=1e-3),
    )
    assert solution.converged, (
        f"update criterion unattainable at t3={receive_time_s!r} s "
        f"(ulp={math.ulp(receive_time_s)!r} s, tolerance={UPDATE_TOLERANCE_S!r} s)"
    )
    assert solution.downlink_equation_residual_s <= EQUATION_TOLERANCE_S


def test_q1f08_r3_two_event_path_is_repaired_too(g20_long_arc):
    """The defect lived in the R3 solver as well; both must be covered."""
    config = RangeRatePhysicsConfig(
        mode="two_way_counted_doppler", count_interval_s=100.0
    )
    provider = resolve_counted_doppler_station_state_provider(
        config,
        g20_long_arc["station"],
        g20_long_arc["t_grid"],
        g20_long_arc["earth_pos"],
        g20_long_arc["earth_vel"],
        et0_s=g20_long_arc["et0"],
    )
    value = two_way_counted_doppler_observable(
        Q1_F08_HISTORICAL_FAILURE_EPOCH_S,
        g20_long_arc["station"],
        g20_long_arc["t_grid"],
        g20_long_arc["states"],
        g20_long_arc["earth_pos"],
        g20_long_arc["earth_vel"],
        None,
        config,
        station_state_provider=provider,
    )
    assert np.isfinite(value)


def test_q1f08_repair_did_not_move_the_accepted_tolerances():
    """The repair changed the represented quantity, not the contract."""
    config = RangeRatePhysicsConfig(mode="two_way_counted_doppler")
    assert config.light_time_tolerance_s == UPDATE_TOLERANCE_S
    assert config.light_time_equation_tolerance_s == EQUATION_TOLERANCE_S
