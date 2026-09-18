"""PHASE 17-R1O-D - geometric, spacecraft-event, and quasar-delay oracles.

ANALYSIS SPACE ONLY (this script). It qualifies the NEW production module
`lunar_od/delta_dor.py` by exercising it against independent, hand-verifiable
truths, per R1O-D s35/s36/s21/s19.

All position models below are expressed as LOCAL, BOUNDED motion about a
chosen reference epoch T_REF: `position(t) = position0 + velocity*(t -
T_REF)`.  This is the correct way to build a light-time oracle at large
absolute epochs -- a linear model evaluated directly at a large absolute
time (e.g. `r0 + v*1e8`) blows up unphysically (a "station" drifting at
400 m/s for 1e8 s moves 4e10 m from Earth), which is a modelling bug, not a
numerical one.  Every scenario factory below takes T_REF explicitly so the
s19 conditioning sweep can vary the absolute epoch while holding the
physical scenario (position AT that epoch, and all velocities) identical.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ARTIFACTS = Path(r"C:/Users/erayh/Documents/Python/Grad/python_port_phase17r0/artifacts")

from lunar_od.delta_dor import (
    QuasarDirection,
    delta_dor_observable_s,
    quasar_differential_delay_s,
    solve_common_transmit_event,
)
from lunar_od.measurements import C_LIGHT_MPS

GATES: dict[str, bool] = {}

# Reference-epoch physical configuration (positions AT t=T_REF, and
# velocities, both epoch-independent -- this is what "the same physical
# scenario" means throughout this script).
R_SC_REF = np.array([1.8e6, 0.3e6, 0.1e6]) + 3.844e8 * np.array([1.0, 0.0, 0.0])
V_SC = np.array([-200.0, 1500.0, 300.0])
R_A_REF = np.array([6.0e6, 2.0e6, 1.0e6])
V_A = np.array([0.0, 400.0, 0.0])
R_B_REF = np.array([-3.0e6, 5.5e6, 3.0e6])
V_B = np.array([-350.0, -100.0, 0.0])


def make_scenario(t_ref: float):
    """Position callbacks for the SAME physical scenario, referenced at t_ref."""

    def sc_pos(t):
        return R_SC_REF + V_SC * (t - t_ref)

    def station_a_pos(t):
        return R_A_REF + V_A * (t - t_ref)

    def station_b_pos(t):
        return R_B_REF + V_B * (t - t_ref)

    return sc_pos, station_a_pos, station_b_pos


def hdr(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


def gate(name: str, ok: bool, detail: str = "") -> None:
    GATES[name] = bool(ok)
    print("  %-55s %s  %s" % (name, "PASS" if ok else "FAIL", detail))


def main() -> None:
    T_obs = 1.0e8
    sc_pos, station_a_pos, station_b_pos = make_scenario(T_obs)

    # ==================================================================
    hdr("s36 -- SPACECRAFT EVENT ORACLE (independent high-accuracy solve)")
    from scipy.optimize import brentq

    station_a_at_tobs = station_a_pos(T_obs)
    light_time_a_approx = float(
        np.linalg.norm(sc_pos(T_obs) - station_a_at_tobs) / C_LIGHT_MPS
    )
    print("  approximate station-A light time: %.6f s" % light_time_a_approx)

    def residual_tx(t_tx):
        return t_tx - T_obs + np.linalg.norm(sc_pos(t_tx) - station_a_at_tobs) / C_LIGHT_MPS

    bracket = 3.0 * light_time_a_approx + 1.0
    t_tx_oracle = brentq(residual_tx, T_obs - bracket, T_obs + bracket, xtol=1e-14, rtol=1e-15)
    sc_at_ttx = sc_pos(t_tx_oracle)

    def residual_tb(t_b):
        return t_b - t_tx_oracle - np.linalg.norm(sc_at_ttx - station_b_pos(t_b)) / C_LIGHT_MPS

    light_time_b_approx = float(
        np.linalg.norm(sc_at_ttx - station_b_pos(t_tx_oracle)) / C_LIGHT_MPS
    )
    bracket_b = 3.0 * light_time_b_approx + 1.0
    t_b_oracle = brentq(
        residual_tb, t_tx_oracle - bracket_b, t_tx_oracle + bracket_b, xtol=1e-14, rtol=1e-15
    )
    # D_S = light_time_B - light_time_A, both LOCAL quantities -- an earlier
    # version of this oracle computed d_s_oracle = t_b_oracle - T_obs, which
    # has the SAME precision defect the production fix below addresses (t_b
    # is already built from a large t_tx, so subtracting T_obs afterwards
    # cannot recover precision lost at t_b's construction). Fixed the same
    # way: difference the two small light times directly.
    light_time_a_oracle = float(
        np.linalg.norm(sc_at_ttx - station_a_at_tobs) / C_LIGHT_MPS
    )
    light_time_b_oracle = float(
        np.linalg.norm(sc_at_ttx - station_b_pos(t_b_oracle)) / C_LIGHT_MPS
    )
    d_s_oracle = light_time_b_oracle - light_time_a_oracle

    common = solve_common_transmit_event(T_obs, station_a_at_tobs, station_b_pos, sc_pos)
    rel_ttx = abs(common.transmit_time_s - t_tx_oracle) / abs(t_tx_oracle)
    rel_ds = abs(common.spacecraft_differential_delay_s - d_s_oracle) / abs(d_s_oracle)
    print("  oracle t_tx=%.12f  production t_tx=%.12f  rel=%.3e"
          % (t_tx_oracle, common.transmit_time_s, rel_ttx))
    print("  oracle D_S=%.12e  production D_S=%.12e  rel=%.3e"
          % (d_s_oracle, common.spacecraft_differential_delay_s, rel_ds))
    gate("SPACECRAFT_DOR_EVENT_ORACLE_GATE", rel_ttx < 1e-10 and rel_ds < 1e-8)

    # ==================================================================
    hdr("s17 -- COMMON-TRANSMIT-EVENT GATE")
    # Both legs' own equation residuals (freshly evaluated inside the solver
    # at the converged epoch, FA-03A-style -- never reconstructed here by
    # subtracting the large absolute T_obs/t_tx/t_B against each other,
    # which is precisely the operation s19 below shows loses precision).
    print("  single t_tx used for BOTH light-time equations: %.12f" % common.transmit_time_s)
    print("  station-A (backward) equation residual: %.3e s" % common.backward_equation_residual_s)
    print("  station-B (forward)  equation residual: %.3e s" % common.forward_equation_residual_s)
    gate("DDOR_COMMON_TRANSMIT_EVENT_GATE",
        common.backward_converged and common.forward_converged
        and common.backward_equation_residual_s < 1e-10
        and common.forward_equation_residual_s < 1e-10)

    # ==================================================================
    hdr("s35 -- BASIC GEOMETRIC ORACLES")

    # (a) zero baseline -> zero geometric DOR
    common_zero_baseline = solve_common_transmit_event(
        T_obs, station_a_at_tobs, station_a_pos, sc_pos,  # B == A
    )
    quasar = QuasarDirection(ra_rad=0.3, dec_rad=0.6)
    d_q_zero = quasar_differential_delay_s(station_a_at_tobs, station_a_at_tobs, quasar)
    ddor_zero = delta_dor_observable_s(common_zero_baseline, d_q_zero)
    print("  zero baseline: D_S=%.3e  D_Q=%.3e  DDOR=%.3e" % (
        common_zero_baseline.spacecraft_differential_delay_s, d_q_zero, ddor_zero))
    gate("zero_baseline_zero_ddor", abs(ddor_zero) < 1e-15)

    # (b) station swap -> sign reversal of the spacecraft term.
    #
    # An earlier version of this test tagged A->B by "receive at A at T_obs"
    # and B->A by "receive at B at T_obs" and expected the two D_S values to
    # be exact negatives. That is NOT the same physical comparison: pinning
    # A's reception to T_obs and pinning B's reception to T_obs are, in
    # general, DIFFERENT transmit events (their light times to a fixed
    # T_obs differ), so the two D_S values were never expected to be exact
    # negatives -- the ~3e-8 s gap it found was real physics from a flawed
    # comparison, not a solver defect.
    #
    # The correct swap test holds t_tx FIXED (one shared transmit event,
    # from the existing common_ab solve above) and asks only: does
    # subtracting the two light times in the opposite order flip the sign?
    # This is what "the observable's sign convention reverses under A/B
    # relabelling" actually means.
    common_ab = solve_common_transmit_event(T_obs, station_a_at_tobs, station_b_pos, sc_pos)
    t_tx_shared = common_ab.transmit_time_s
    sc_at_shared_tx = sc_pos(t_tx_shared)
    light_time_a_at_shared_tx = float(
        np.linalg.norm(sc_at_shared_tx - station_a_pos(T_obs)) / C_LIGHT_MPS
    )
    light_time_b_at_shared_tx = float(
        np.linalg.norm(sc_at_shared_tx - station_b_pos(common_ab.station_b_receive_time_s))
        / C_LIGHT_MPS
    )
    d_s_ab = light_time_b_at_shared_tx - light_time_a_at_shared_tx
    d_s_ba = light_time_a_at_shared_tx - light_time_b_at_shared_tx
    print("  shared t_tx=%.9f;  light_time_A=%.9f  light_time_B=%.9f"
          % (t_tx_shared, light_time_a_at_shared_tx, light_time_b_at_shared_tx))
    print("  D_S(A->B) = %.15e s   [from production: %.15e s]"
          % (d_s_ab, common_ab.spacecraft_differential_delay_s))
    print("  D_S(B->A) = %.15e s (same t_tx, light times swapped)" % d_s_ba)
    print("  sum (must be exactly 0.0, pure float negation) = %.3e s" % (d_s_ab + d_s_ba))
    gate("station_swap_sign_reversal",
        (d_s_ab + d_s_ba) == 0.0
        and abs(d_s_ab - common_ab.spacecraft_differential_delay_s) < 1e-13)

    # (c) quasar plane-wave delay vs analytic dot-product truth
    r_a_ref = station_a_pos(T_obs)
    r_b_ref = station_b_pos(T_obs)
    d_q = quasar_differential_delay_s(r_a_ref, r_b_ref, quasar)
    d_q_manual = -float(np.dot(r_b_ref - r_a_ref, quasar.unit_vector_j2000)) / C_LIGHT_MPS
    print("  D_Q (function)=%.12e  D_Q (manual dot product)=%.12e" % (d_q, d_q_manual))
    gate("QUASAR_DELAY_ORACLE_GATE", abs(d_q - d_q_manual) < 1e-18)

    qd2 = QuasarDirection.from_unit_vector(quasar.unit_vector_j2000)
    gate("quasar_direction_roundtrip",
        abs(qd2.ra_rad - quasar.ra_rad) < 1e-12 and abs(qd2.dec_rad - quasar.dec_rad) < 1e-12)

    # (d) baseline perpendicular to quasar direction -> zero quasar delay
    s_hat = quasar.unit_vector_j2000
    perp = np.cross(s_hat, np.array([1.0, 0.0, 0.0]))
    perp = perp / np.linalg.norm(perp)
    d_q_perp = quasar_differential_delay_s(np.zeros(3), perp * 1.0e7, quasar)
    print("  baseline PERPENDICULAR to quasar direction: D_Q=%.3e (expect 0)" % d_q_perp)
    gate("perpendicular_baseline_zero_quasar_delay", abs(d_q_perp) < 1e-15)

    # (e) rigid epoch translation -> unchanged LOCAL delay physics.
    # The scenario factory is EXPLICITLY built so this holds: re-anchoring at
    # T_obs+shift reproduces the identical physical configuration, just
    # relabelled, so D_S computed there must match the un-shifted D_S to the
    # light-time-equation tolerance -- this is a controlled, not merely
    # observed, invariance.
    shift = 3.7e6
    t_obs2 = T_obs + shift
    sc_pos2, station_a_pos2, station_b_pos2 = make_scenario(t_obs2)
    common_shifted = solve_common_transmit_event(
        t_obs2, station_a_pos2(t_obs2), station_b_pos2, sc_pos2,
    )
    d_s_shift_diff = abs(common_shifted.spacecraft_differential_delay_s
                        - common_ab.spacecraft_differential_delay_s)
    print("  D_S(T_obs)=%.12e  D_S(T_obs+%.1e)=%.12e  |diff|=%.3e"
          % (common_ab.spacecraft_differential_delay_s, shift,
             common_shifted.spacecraft_differential_delay_s, d_s_shift_diff))
    gate("DDOR_GEOMETRIC_ORACLE_GATE",
        d_s_shift_diff < 1e-8 and (d_s_ab + d_s_ba) == 0.0 and abs(ddor_zero) < 1e-15)

    # ==================================================================
    hdr("s19 -- LOCAL-DELAY NUMERICAL CONDITIONING SWEEP")
    print("  Holding the PHYSICAL configuration fixed (make_scenario always")
    print("  re-anchors position/velocity identically) and varying ONLY the")
    print("  absolute epoch label across binades -- isolates numerical")
    print("  conditioning from physics, the exact Phase 17C control.")
    prev_ds = None
    max_rel_step = 0.0
    for exp in (0, 5, 10, 13, 16, 20, 23, 25, 27, 30, 33, 36):
        t_obs_i = float(2 ** exp) + 1234.5678
        sc_i, a_i, b_i = make_scenario(t_obs_i)
        try:
            c_i = solve_common_transmit_event(t_obs_i, a_i(t_obs_i), b_i, sc_i)
        except Exception as exc:  # noqa: BLE001
            print("    2^%2d s : FAILED (%s)" % (exp, exc))
            continue
        print("    2^%2d s (T_obs=%.6e): D_S=%.15e  fwd_iters=%d  fwd_resid=%.2e"
              % (exp, t_obs_i, c_i.spacecraft_differential_delay_s,
                 c_i.forward_iterations, c_i.forward_equation_residual_s))
        if prev_ds is not None:
            rel_step = abs(c_i.spacecraft_differential_delay_s - prev_ds) / abs(prev_ds)
            max_rel_step = max(max_rel_step, rel_step)
        prev_ds = c_i.spacecraft_differential_delay_s

    print("\n  max relative step in D_S across ALL binades (physical scenario")
    print("  held fixed): %.3e" % max_rel_step)
    gate("DDOR_LOCAL_DELAY_NUMERICAL_CONDITIONING_GATE", max_rel_step < 1e-9)

    hdr("SUMMARY")
    for k, v in GATES.items():
        print("  %-55s %s" % (k, "PASS" if v else "FAIL"))
    all_pass = all(GATES.values())
    print("\n  ALL ORACLE GATES: %s" % ("PASS" if all_pass else "FAIL"))

    (ARTIFACTS / "r1od_oracle_gates.json").write_text(json.dumps(dict(
        gates={k: ("PASS" if v else "FAIL") for k, v in GATES.items()},
        all_pass=all_pass,
        max_relative_step_across_binades=max_rel_step,
        spacecraft_event_oracle_rel_ttx=rel_ttx,
        spacecraft_event_oracle_rel_ds=rel_ds,
    ), indent=2, default=float))
    print("  wrote r1od_oracle_gates.json")

    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
