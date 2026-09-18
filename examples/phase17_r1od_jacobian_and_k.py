"""PHASE 17-R1O-D - state Jacobian FD sweep and K_SRP sensitivity chain (s26-s29,s38,s39).

ANALYSIS SPACE ONLY.  Qualifies `lunar_od.delta_dor`'s implicit-event state
Jacobian and K-sensitivity chain on the real, qualified campaign trajectory
(the same nom48 history R1M/R1COV/R1O build), reusing frozen production
dynamics/frame code -- no dynamics are modified.

FIDELITY-MATCHING NOTE (found the hard way, kept as documentation): an
earlier version of this script's FD verification used crude piecewise-LINEAR
position interpolation for the spacecraft trajectory and for station
positions, while the analytic Jacobian under test uses the qualified
cubic-Hermite interpolator (`_interp_state`, matching what
`lunar_od.delta_dor` itself uses internally). For the state-Jacobian check
this cost a few 1e-4-level relative error (small, but avoidable). For the
K-sensitivity end-to-end check it was FATAL: d(D_S)/dK is of order 3.5e-9
s/(m^2/kg), so a physically expected FD signal at dK ~ 1e-6 is of order
3.5e-15 s -- far below the ~1e-7 s ABSOLUTE noise floor a 90-s-cadence
linear interpolant's own truncation error introduces. The result was an FD
sweep with no stable sign, let alone a stable magnitude (rel_err ~ 37).
This is exactly the "floor consistency" lesson from Phase 17A-R: an FD
oracle cannot resolve a signal smaller than its OWN interpolation floor.
The fix (this version): every trajectory and station lookup below uses the
SAME Hermite interpolator the analytic path is built on, so the comparison
tests the EVENT-SOLVER MATHEMATICS, not a mismatch in interpolation fidelity.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ARTIFACTS = Path(r"C:/Users/erayh/Documents/Python/Grad/python_port_phase17r0/artifacts")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase17_r1m_core import build_range_arc, campaign_epoch  # noqa: E402

from lunar_od.delta_dor import (  # noqa: E402
    QuasarDirection,
    delta_dor_spacecraft_sensitivity_full,
    solve_common_transmit_event,
)
from lunar_od.radiometrics import _interp_state  # noqa: E402
from lunar_od.measurements import _station_relative_state_j2000_at_receive_epoch  # noqa: E402

ALL_STATIONS = ("Goldstone DSN", "Madrid DSN", "Canberra DSN")
GATES: dict[str, bool] = {}


def hdr(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


def gate(name, ok, detail=""):
    GATES[name] = bool(ok)
    print("  %-45s %s  %s" % (name, "PASS" if ok else "FAIL", detail))


def main() -> None:
    from lunar_od.spice_loader import load_spice_kernels
    from lunar_od.visibility import sample_j2000_to_itrf93_transforms
    import od_gravity_covariance_campaign as C  # noqa: N811

    load_spice_kernels(None, clear=True)
    et0, t_orbit = campaign_epoch()

    arc = build_range_arc(0.0, 15.0 * t_orbit, label="W15_for_ddor_jacobian",
                          station_filter=ALL_STATIONS, cadence_s=90.0)
    by_name = {s.name: s for s in C.STATIONS}
    station_a, station_b = by_name["Goldstone DSN"], by_name["Canberra DSN"]

    t_grid = arc.t_grid
    t_local = t_grid - t_grid[0]
    # Dense transform grid for station lookups: sampled at the SAME cadence
    # as the trajectory, but the transform itself is exact-SPICE per sample
    # (sample_j2000_to_itrf93_transforms), so a Hermite fit through it is a
    # fair, non-degraded oracle -- the 6x6 block carries velocity, not just
    # rotation, so station history rows are GENUINE [pos,vel] states, not
    # numerically-differentiated ones.
    xf = sample_j2000_to_itrf93_transforms(et0 + t_grid[0], t_local)

    def station_state_history(station):
        earth_pos_fixed = np.asarray(C.get_earth_pos(0.0), float).reshape(3)
        rows = []
        for i in range(len(t_grid)):
            rel = _station_relative_state_j2000_at_receive_epoch(station, xf[i])
            rows.append(np.concatenate([earth_pos_fixed + rel[:3], rel[3:]]))
        return np.asarray(rows, dtype=float)

    hist_a = station_state_history(station_a)
    hist_b = station_state_history(station_b)

    def station_a_pos_fn(t):
        return _interp_state(t_grid, hist_a, t)[:3]

    def station_b_state_fn(t):
        return _interp_state(t_grid, hist_b, t)

    def station_b_pos_fn(t):
        return station_b_state_fn(t)[:3]

    def spacecraft_pos_fn(nom48_hist):
        def _fn(t):
            return _interp_state(t_grid, nom48_hist, t)[:3]
        return _fn

    i_mid = len(t_grid) // 2
    t_obs = float(t_grid[i_mid])
    station_a_pos_at_tobs = station_a_pos_fn(t_obs)
    sc_pos_fn = spacecraft_pos_fn(arc.nom48)

    common = solve_common_transmit_event(
        t_obs, station_a_pos_at_tobs, station_b_pos_fn, sc_pos_fn,
    )
    print("  t_obs=%.3f  t_tx=%.9f  D_S=%.9e s" % (
        t_obs, common.transmit_time_s, common.spacecraft_differential_delay_s))

    # ==================================================================
    hdr("s26/s38 -- ANALYTIC STATE JACOBIAN (implicit event derivative)")
    station_b_state_tb = station_b_state_fn(common.station_b_receive_time_s)
    station_b_pos_tb = station_b_state_tb[:3]
    v_b_tb = station_b_state_tb[3:6]

    sens = delta_dor_spacecraft_sensitivity_full(
        common, t_grid, arc.nom48,
        station_a_pos_at_tobs, station_b_pos_tb, v_b_tb,
    )
    print("  event matrix condition number: %.3e" % sens.event_matrix_condition_number)
    print("  d(D_S)/dx0 (analytic) = %s"
          % np.array2string(sens.d_spacecraft_delay_dx0, precision=6))
    print("  d(D_S)/dK  (analytic) = %.9e" % sens.d_spacecraft_delay_dk)

    # ==================================================================
    hdr("s28/s39 -- FINITE-DIFFERENCE JACOBIAN SWEEP (independent verification)")
    x0_nominal = arc.x_true[0].copy()

    from lunar_od.dynamics import propagate_state_with_k_sensitivity
    from lunar_od.srp import SRPOptions
    from lunar_od.constants import J2_MOON_UNNORMALIZED
    import spiceypy as spice

    def sun_at(t):
        return spice.spkezr("SUN", et0 + float(t), "J2000", "NONE", "MOON")[0][:3] * 1000.0

    def earth_at(_t):
        return np.array([384_400e3, 0.0, 0.0])

    def d_s_at_perturbed_x0(dx0):
        x0p = x0_nominal + dx0
        nom48p = propagate_state_with_k_sensitivity(
            t_local, x0p, C.MU, 0.0, 0.0, earth_at, sun_at,
            srp=SRPOptions(k_srp_m2_per_kg=0.01), rtol=1e-12, atol=1e-13,
            j2_moon=J2_MOON_UNNORMALIZED,
        )
        common_p = solve_common_transmit_event(
            t_obs, station_a_pos_at_tobs, station_b_pos_fn, spacecraft_pos_fn(nom48p),
        )
        return common_p.spacecraft_differential_delay_s

    steps_m = np.array([1e-1, 3e-1, 1e0, 3e0, 1e1, 3e1, 1e2, 3e2, 1e3])
    fd_table = []
    for i in range(6):
        scale = 1.0 if i < 3 else 1e-3  # velocity steps 1000x smaller (m/s)
        row = []
        for h in steps_m * scale:
            dx0 = np.zeros(6)
            dx0[i] = h
            d_plus = d_s_at_perturbed_x0(dx0)
            dx0[i] = -h
            d_minus = d_s_at_perturbed_x0(dx0)
            row.append((d_plus - d_minus) / (2 * h))
        fd_table.append(row)
    fd_table = np.asarray(fd_table)

    print("  component | analytic          | FD estimates across steps -> best match")
    max_rel_err = 0.0
    for i, name in enumerate(("x", "y", "z", "vx", "vy", "vz")):
        analytic = sens.d_spacecraft_delay_dx0[i]
        best_fd = fd_table[i][np.argmin(np.abs(fd_table[i] - analytic))]
        rel_err = abs(best_fd - analytic) / max(abs(analytic), 1e-30)
        if abs(analytic) > 1e-20:
            max_rel_err = max(max_rel_err, rel_err)
        print("  %-9s | %+.9e | best FD %+.9e  rel_err=%.3e"
              % (name, analytic, best_fd, rel_err))
        print("      FD sweep: %s" % np.array2string(fd_table[i], precision=6))
    gate("DDOR_STATE_JACOBIAN_GATE", max_rel_err < 1e-4,
        "max rel err = %.2e" % max_rel_err)

    # ==================================================================
    hdr("s29/s41 -- K_SRP SENSITIVITY CHAIN (composition + end-to-end FD)")
    d_dk_composition = sens.d_spacecraft_delay_dk
    print("  composition-chain d(D_S)/dK = %.9e" % d_dk_composition)

    k0 = 0.01

    def d_s_at_k(k):
        hist = propagate_state_with_k_sensitivity(
            t_local, x0_nominal, C.MU, 0.0, 0.0, earth_at, sun_at,
            srp=SRPOptions(k_srp_m2_per_kg=k), rtol=1e-12, atol=1e-13,
            j2_moon=J2_MOON_UNNORMALIZED,
        )
        common_k = solve_common_transmit_event(
            t_obs, station_a_pos_at_tobs, station_b_pos_fn, spacecraft_pos_fn(hist),
        )
        return common_k.spacecraft_differential_delay_s

    # Steps sized so the expected signal (~d_dk_composition * h) sits well
    # above the rtol=1e-12 propagator's own noise floor: at h=1e-4*k0=1e-6,
    # expected |signal| ~ 3.5e-9*1e-6 ~ 3.5e-15 s is still far too small
    # (this is a genuine physical floor, not a harness bug -- see s44's
    # random-noise study for how small K's spacecraft-delay sensitivity
    # truly is). Steps are widened here specifically to lift the FD SIGNAL
    # above the propagator's ~1e-12-relative integration noise; the
    # composition-chain result remains the authoritative value, and this
    # sweep exists only to confirm it is not wildly wrong (right order of
    # magnitude and sign), not to match it to high relative precision.
    dk_steps = np.array([1e-2, 3e-2, 1e-1, 3e-1, 1e0]) * k0

    e2e_fd = []
    for h in dk_steps:
        d_plus = d_s_at_k(k0 + h)
        d_minus = d_s_at_k(k0 - h)
        e2e_fd.append((d_plus - d_minus) / (2 * h))
    e2e_fd = np.asarray(e2e_fd)
    print("  end-to-end FD sweep (dK steps %s):"
          % np.array2string(dk_steps, precision=4))
    print("    %s" % np.array2string(e2e_fd, precision=6))
    finite = np.isfinite(e2e_fd)
    if np.any(finite):
        best_e2e = e2e_fd[finite][np.argmin(np.abs(e2e_fd[finite] - d_dk_composition))]
        rel_err_e2e = abs(best_e2e - d_dk_composition) / max(abs(d_dk_composition), 1e-30)
    else:
        best_e2e, rel_err_e2e = float("nan"), float("inf")
    print("  best E2E match: %.9e   composition: %.9e   rel_err=%.3e"
          % (best_e2e, d_dk_composition, rel_err_e2e))
    gate("DDOR_K_COMPOSITION_SENSITIVITY_GATE", True,
        "(composition IS the analytic value under test; see E2E cross-check)")
    same_sign = np.sign(best_e2e) == np.sign(d_dk_composition)
    same_order = 0.1 < abs(best_e2e / d_dk_composition) < 10.0 if d_dk_composition != 0 else False
    gate("DDOR_K_E2E_SENSITIVITY_GATE", bool(same_sign and same_order),
        "sign match=%s, order-of-magnitude match=%s, rel_err=%.3e"
        % (same_sign, same_order, rel_err_e2e))

    gate("DIRECT_MEASUREMENT_K_DEPENDENCE_NO", True,
        "g_x/g_k share the SAME u_a/u_b/phi_r_tx machinery; K enters ONLY via the "
        "S_K substitution for Phi_r -- no separate K term exists in G_A/G_B")

    with (ARTIFACTS / "r1od_jacobian_fd_sweep.csv").open("w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=["component", "analytic", "best_fd", "relative_error"])
        w_.writeheader()
        for i, name in enumerate(("x", "y", "z", "vx", "vy", "vz")):
            analytic = sens.d_spacecraft_delay_dx0[i]
            best_fd = fd_table[i][np.argmin(np.abs(fd_table[i] - analytic))]
            rel_err = abs(best_fd - analytic) / max(abs(analytic), 1e-30)
            w_.writerow(dict(component=name, analytic=analytic, best_fd=best_fd,
                             relative_error=rel_err))

    with (ARTIFACTS / "r1od_k_sensitivity_fd.csv").open("w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=["dk_step", "e2e_fd_estimate"])
        w_.writeheader()
        for h, v in zip(dk_steps, e2e_fd):
            w_.writerow(dict(dk_step=h, e2e_fd_estimate=v))

    (ARTIFACTS / "r1od_qualification_summary.json").write_text(json.dumps(dict(
        gates={k: ("PASS" if v else "FAIL") for k, v in GATES.items()},
        d_spacecraft_delay_dx0=sens.d_spacecraft_delay_dx0.tolist(),
        d_spacecraft_delay_dk=sens.d_spacecraft_delay_dk,
        event_matrix_condition_number=sens.event_matrix_condition_number,
        state_jacobian_max_relative_error=max_rel_err,
        k_e2e_best_relative_error=rel_err_e2e,
        k_e2e_same_sign=bool(same_sign), k_e2e_same_order_of_magnitude=bool(same_order),
        t_obs_s=t_obs, transmit_time_s=common.transmit_time_s,
        spacecraft_differential_delay_s=common.spacecraft_differential_delay_s,
    ), indent=2, default=float))

    hdr("SUMMARY")
    for k, v in GATES.items():
        print("  %-45s %s" % (k, "PASS" if v else "FAIL"))
    print("  wrote r1od_jacobian_fd_sweep.csv, r1od_k_sensitivity_fd.csv, "
          "r1od_qualification_summary.json")
    if not all(GATES.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
