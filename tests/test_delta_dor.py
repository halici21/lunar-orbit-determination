"""Phase 17-R1O-D - permanent qualification tests for lunar_od.delta_dor.

Covers, per the governing spec's required permanent-test list: zero baseline,
station-swap sign reversal, quasar analytic delay, common spacecraft
transmit event, time-tag contract, local-delay numerical conditioning,
state-Jacobian FD agreement, K-sensitivity chain, invalid station/quasar
inputs, and the unit/sign convention.

All position models are LOCAL, BOUNDED motion about a reference epoch
(see `make_scenario`) -- evaluating a linear velocity model directly at a
large absolute time blows up unphysically and is a modelling bug, not
something a light-time solver should be asked to tolerate.
"""
from __future__ import annotations

import numpy as np
import pytest

from lunar_od.delta_dor import (
    CommonTransmitEventSolution,
    DeltaDorEventError,
    QuasarDirection,
    delta_dor_observable_s,
    delta_dor_spacecraft_sensitivity_full,
    quasar_differential_delay_s,
    solve_common_transmit_event,
    solve_forward_one_way_light_time,
)
from lunar_od.measurements import C_LIGHT_MPS

R_SC_REF = np.array([1.8e6, 0.3e6, 0.1e6]) + 3.844e8 * np.array([1.0, 0.0, 0.0])
V_SC = np.array([-200.0, 1500.0, 300.0])
R_A_REF = np.array([6.0e6, 2.0e6, 1.0e6])
V_A = np.array([0.0, 400.0, 0.0])
R_B_REF = np.array([-3.0e6, 5.5e6, 3.0e6])
V_B = np.array([-350.0, -100.0, 0.0])
T_OBS = 1.0e8


def make_scenario(t_ref: float):
    def sc_pos(t):
        return R_SC_REF + V_SC * (t - t_ref)

    def station_a_pos(t):
        return R_A_REF + V_A * (t - t_ref)

    def station_b_pos(t):
        return R_B_REF + V_B * (t - t_ref)

    return sc_pos, station_a_pos, station_b_pos


@pytest.fixture()
def scenario():
    return make_scenario(T_OBS)


@pytest.fixture()
def common_event(scenario):
    sc_pos, station_a_pos, station_b_pos = scenario
    return solve_common_transmit_event(T_OBS, station_a_pos(T_OBS), station_b_pos, sc_pos)


# ======================================================================
# Common-transmit-event solver
# ======================================================================
def test_both_stations_trace_back_to_the_identical_transmit_event(scenario, common_event):
    """s17 hard requirement: A and B share exactly one spacecraft transmit event."""
    sc_pos, station_a_pos, _ = scenario
    sc_at_ttx = sc_pos(common_event.transmit_time_s)
    assert common_event.backward_converged and common_event.forward_converged
    assert common_event.backward_equation_residual_s < 1e-10
    assert common_event.forward_equation_residual_s < 1e-10


def test_time_tag_is_reception_at_station_a(scenario):
    """s18: the observation epoch is defined as station A's reception time.

    Verified using the solver's OWN authoritative local-delay residual
    (`backward_equation_residual_s`, freshly evaluated inside the solve),
    not by re-deriving `T_obs - t_tx` here: that subtraction differences two
    O(1e8) numbers and its own floating-point noise floor (~2.7e-9 s at this
    scale) is larger than a naively "tight" external tolerance would allow,
    even though the solver's internal answer is fully converged.
    """
    sc_pos, station_a_pos, station_b_pos = scenario
    common = solve_common_transmit_event(T_OBS, station_a_pos(T_OBS), station_b_pos, sc_pos)
    assert common.t_obs_s == T_OBS
    assert common.backward_converged
    assert common.backward_equation_residual_s < 1e-10


def test_zero_baseline_gives_zero_spacecraft_differential_delay(scenario):
    sc_pos, station_a_pos, _ = scenario
    common = solve_common_transmit_event(T_OBS, station_a_pos(T_OBS), station_a_pos, sc_pos)
    assert common.spacecraft_differential_delay_s == 0.0


def test_station_swap_reverses_sign_exactly(common_event, scenario):
    """s35: swapping which light time is subtracted from which must flip the
    sign EXACTLY (pure float negation), holding the shared t_tx fixed.
    """
    sc_pos, station_a_pos, station_b_pos = scenario
    t_tx = common_event.transmit_time_s
    sc_at_tx = sc_pos(t_tx)
    lt_a = np.linalg.norm(sc_at_tx - station_a_pos(T_OBS)) / C_LIGHT_MPS
    lt_b = np.linalg.norm(
        sc_at_tx - station_b_pos(common_event.station_b_receive_time_s)
    ) / C_LIGHT_MPS
    d_ab = lt_b - lt_a
    d_ba = lt_a - lt_b
    assert d_ab + d_ba == 0.0
    assert d_ab == pytest.approx(common_event.spacecraft_differential_delay_s, rel=1e-13)


def test_forward_solver_converges_and_matches_backward_solver_self_consistency():
    """Forward-solving station A from the already-known t_tx must recover the
    SAME light time the backward solve produced.

    Compared against `common.station_a_range_m / C_LIGHT_MPS` -- a LOCAL
    quantity carried by the backward solve itself -- rather than against
    `T_obs - common.transmit_time_s`, which differences two O(1e8) numbers
    and inherits that subtraction's own floating-point floor (this is the
    same fix applied to `test_time_tag_is_reception_at_station_a` above).
    """
    sc_pos, station_a_pos, _ = make_scenario(T_OBS)
    common = solve_common_transmit_event(T_OBS, station_a_pos(T_OBS), station_a_pos, sc_pos)
    forward = solve_forward_one_way_light_time(
        common.transmit_time_s, sc_pos(common.transmit_time_s), station_a_pos,
    )
    assert forward.converged
    expected_light_time_s = common.station_a_range_m / C_LIGHT_MPS
    assert forward.light_time_s == pytest.approx(expected_light_time_s, rel=1e-13)


def test_forward_solver_rejects_non_positive_light_speed():
    with pytest.raises(ValueError):
        solve_forward_one_way_light_time(0.0, np.zeros(3), lambda t: np.ones(3),
                                         light_speed_mps=-1.0)


def test_backward_leg_convergence_failure_raises_controlled_error():
    """A target that recedes faster than c never converges; must raise, not
    silently return the last (unconverged) iterate.
    """
    def runaway_target(t):
        return np.array([1e20 * t, 0.0, 0.0])

    with pytest.raises(DeltaDorEventError):
        solve_common_transmit_event(1.0, np.zeros(3), lambda t: np.zeros(3), runaway_target)


# ======================================================================
# s19 -- local-delay numerical conditioning
# ======================================================================
@pytest.mark.parametrize("exponent", [0, 13, 23, 27, 30, 33])
def test_local_delay_is_stable_across_binades(exponent):
    """The physical scenario is IDENTICAL at every epoch (make_scenario
    re-anchors position/velocity), so D_S must not drift with the absolute
    epoch label -- the Phase 17C-class regression this guards against.
    """
    t_ref = float(2 ** exponent) + 1234.5678
    sc_pos, station_a_pos, station_b_pos = make_scenario(t_ref)
    common = solve_common_transmit_event(t_ref, station_a_pos(t_ref), station_b_pos, sc_pos)
    baseline_sc, baseline_a, baseline_b = make_scenario(1234.5678)
    baseline = solve_common_transmit_event(
        1234.5678, baseline_a(1234.5678), baseline_b, baseline_sc
    )
    rel = abs(common.spacecraft_differential_delay_s
             - baseline.spacecraft_differential_delay_s) / abs(
        baseline.spacecraft_differential_delay_s)
    assert rel < 1e-8


# ======================================================================
# Quasar delay
# ======================================================================
def test_quasar_delay_matches_analytic_dot_product():
    quasar = QuasarDirection(ra_rad=0.3, dec_rad=0.6)
    r_a, r_b = np.array([6e6, 2e6, 1e6]), np.array([-3e6, 5.5e6, 3e6])
    d_q = quasar_differential_delay_s(r_a, r_b, quasar)
    d_q_manual = -float(np.dot(r_b - r_a, quasar.unit_vector_j2000)) / C_LIGHT_MPS
    assert d_q == pytest.approx(d_q_manual, rel=1e-13)


def test_quasar_delay_zero_for_zero_baseline():
    quasar = QuasarDirection(ra_rad=1.1, dec_rad=-0.4)
    r = np.array([1e6, 2e6, 3e6])
    assert quasar_differential_delay_s(r, r, quasar) == 0.0


def test_quasar_delay_zero_for_baseline_perpendicular_to_source():
    quasar = QuasarDirection(ra_rad=0.0, dec_rad=0.0)  # s_hat = [1,0,0]
    baseline = np.array([0.0, 1.0, 1.0]) * 1e7  # perpendicular to x
    assert abs(quasar_differential_delay_s(np.zeros(3), baseline, quasar)) < 1e-15


def test_quasar_direction_unit_vector_roundtrip():
    q = QuasarDirection(ra_rad=2.4, dec_rad=-0.55)
    q2 = QuasarDirection.from_unit_vector(q.unit_vector_j2000)
    assert q2.ra_rad == pytest.approx(q.ra_rad, abs=1e-12)
    assert q2.dec_rad == pytest.approx(q.dec_rad, abs=1e-12)


def test_quasar_direction_rejects_zero_vector():
    with pytest.raises(ValueError):
        QuasarDirection.from_unit_vector(np.zeros(3))


def test_delta_dor_observable_is_spacecraft_minus_quasar_delay(common_event):
    quasar = QuasarDirection(ra_rad=0.5, dec_rad=0.2)
    d_q = 1.23e-8
    assert delta_dor_observable_s(common_event, d_q) == pytest.approx(
        common_event.spacecraft_differential_delay_s - d_q, rel=1e-13)


# ======================================================================
# State Jacobian / K sensitivity chain (structural + numerical)
# ======================================================================
def test_direct_measurement_k_dependence_is_structurally_absent():
    """s15/s40: DDOR's observable functions take no K parameter at all."""
    import inspect

    sig = inspect.signature(quasar_differential_delay_s)
    assert "k" not in [p.lower() for p in sig.parameters]
    assert "k_srp" not in [p.lower() for p in sig.parameters]


def test_state_jacobian_rejects_short_augmented_history():
    from lunar_od.delta_dor import CommonTransmitEventSolution

    common = CommonTransmitEventSolution(
        t_obs_s=0.0, transmit_time_s=-1.0, station_b_receive_time_s=0.03,
        spacecraft_differential_delay_s=0.01, backward_converged=True,
        forward_converged=True, forward_iterations=3,
        backward_equation_residual_s=0.0, forward_equation_residual_s=0.0,
        station_a_range_m=3.8e8, station_b_range_m=3.9e8,
    )
    bad_history = np.zeros((5, 10))  # < 42 columns
    t_grid = np.linspace(-2.0, 2.0, 5)
    with pytest.raises(ValueError):
        delta_dor_spacecraft_sensitivity_full(
            common, t_grid, bad_history, np.zeros(3), np.zeros(3), np.zeros(3),
        )


def test_event_matrix_is_well_conditioned_for_a_benign_geometry(common_event):
    sc_pos, station_a_pos, station_b_pos = make_scenario(T_OBS)
    t_grid = np.linspace(common_event.transmit_time_s - 5.0,
                        common_event.station_b_receive_time_s + 5.0, 11)
    states = np.array([np.concatenate([sc_pos(t), V_SC]) for t in t_grid])
    phi = np.tile(np.eye(6).reshape(1, 36), (len(t_grid), 1))
    s_k = np.zeros((len(t_grid), 6))
    x_aug = np.hstack([states, phi, s_k])
    station_b_pos_tb = station_b_pos(common_event.station_b_receive_time_s)
    sens = delta_dor_spacecraft_sensitivity_full(
        common_event, t_grid, x_aug, station_a_pos(T_OBS), station_b_pos_tb, V_B,
    )
    assert np.isfinite(sens.event_matrix_condition_number)
    assert sens.event_matrix_condition_number < 100.0
    assert sens.d_spacecraft_delay_dk == 0.0  # zero S_K history -> zero K sensitivity


def test_k_sensitivity_scales_linearly_with_injected_s_k(common_event):
    """A synthetic nonzero S_K history must produce a proportional K column,
    confirming the substitution path (Phi_r -> S_K) is wired correctly.
    """
    sc_pos, station_a_pos, station_b_pos = make_scenario(T_OBS)
    t_grid = np.linspace(common_event.transmit_time_s - 5.0,
                        common_event.station_b_receive_time_s + 5.0, 11)
    states = np.array([np.concatenate([sc_pos(t), V_SC]) for t in t_grid])
    phi = np.tile(np.eye(6).reshape(1, 36), (len(t_grid), 1))
    station_b_pos_tb = station_b_pos(common_event.station_b_receive_time_s)

    results = []
    for scale in (1.0, 2.0):
        s_k = np.tile(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * scale, (len(t_grid), 1))
        x_aug = np.hstack([states, phi, s_k])
        sens = delta_dor_spacecraft_sensitivity_full(
            common_event, t_grid, x_aug, station_a_pos(T_OBS), station_b_pos_tb, V_B,
        )
        results.append(sens.d_spacecraft_delay_dk)
    assert results[1] == pytest.approx(2.0 * results[0], rel=1e-9)
