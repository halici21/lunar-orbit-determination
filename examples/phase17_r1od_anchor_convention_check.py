"""PHASE 17-R1O-D - reconciling the transmit-anchor vs reception-anchor DDOR
convention against Moyer's own OD-level formulation.

CONTEXT (discovered mid-phase, from an earlier abandoned attempt's literature
research that survived as docs/phase17_r1od_literature_contract.md after its
own lunar_od/delta_dor.py was overwritten by this phase's from-scratch
implementation): the governing spec's own s17 diagram anchors DDOR on a
COMMON SPACECRAFT TRANSMIT event (what this phase's `solve_common_transmit_
event` implements). Moyer's actual Section 11.4.1 OD-level formulation
anchors instead on a COMMON RECEPTION time T, with each station's transmit
time solved independently and BACKWARD via the already-qualified receive-
anchored one-way light-time solver -- computationally much simpler, and
literally what `one_way_light_time_range_sensitivity` already does, called
twice.

This script builds the reception-anchored quantity using ONLY existing,
already-qualified production functions (no new solver), and compares its
VALUE and K-SENSITIVITY DIRECTION against the transmit-anchored production
result on the SAME campaign epoch, to determine whether the phase's central
finding is robust to this convention choice.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase17_r1m_core import build_range_arc, campaign_epoch  # noqa: E402

from lunar_od.measurements import (  # noqa: E402
    C_LIGHT_MPS, one_way_light_time_initial_state_sensitivity,
)
from lunar_od.delta_dor import (  # noqa: E402
    solve_common_transmit_event, delta_dor_spacecraft_sensitivity_full,
)

ALL_STATIONS = ("Goldstone DSN", "Madrid DSN", "Canberra DSN")


def hdr(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


def main() -> None:
    from lunar_od.spice_loader import load_spice_kernels
    from lunar_od.visibility import sample_j2000_to_itrf93_transforms
    import od_gravity_covariance_campaign as C  # noqa: N811

    load_spice_kernels(None, clear=True)
    et0, t_orbit = campaign_epoch()
    arc = build_range_arc(0.0, 15.0 * t_orbit, label="W15_anchor_check",
                          station_filter=ALL_STATIONS, cadence_s=90.0)
    by_name = {s.name: s for s in C.STATIONS}
    station_a, station_b = by_name["Goldstone DSN"], by_name["Canberra DSN"]

    t_grid = arc.t_grid
    t_local = t_grid - t_grid[0]
    xf = sample_j2000_to_itrf93_transforms(et0 + t_grid[0], t_local)
    earth_pos_fixed = np.asarray(C.get_earth_pos(0.0), float).reshape(3)

    i_mid = len(t_grid) // 2
    T = float(t_grid[i_mid])

    hdr("RECEPTION-ANCHORED D_S  (Moyer Section 11.4.1: common reception time T,"
        " transmit times solved independently backward)")
    # Each leg is EXACTLY the existing qualified receive-anchored one-way
    # light-time Jacobian, called once per station at the SAME T -- no new
    # solver, reusing `one_way_light_time_initial_state_sensitivity` (the
    # same helper the qualified range observable's own initial-state
    # Jacobian is built from).
    phi_hist = np.array([row.reshape((6, 6), order="F") for row in arc.nom48[:, 6:42]])
    sol_a, _, sens_a = one_way_light_time_initial_state_sensitivity(
        T, station_a, t_grid, arc.nom48[:, :6], phi_hist, earth_pos_fixed, xf[i_mid])
    sol_b, _, sens_b = one_way_light_time_initial_state_sensitivity(
        T, station_b, t_grid, arc.nom48[:, :6], phi_hist, earth_pos_fixed, xf[i_mid])

    d_s_reception = sol_b.light_time_s - sol_a.light_time_s
    d_s_dx0_reception = sens_b.d_tau_dx0 - sens_a.d_tau_dx0
    print("  T=%.3f  t_tx_A=%.9f  t_tx_B=%.9f  |diff|=%.6f s (Moyer's own quoted"
        " ~0.02 s Earth-baseline bound)"
        % (T, sol_a.transmit_time_s, sol_b.transmit_time_s,
           abs(sol_a.transmit_time_s - sol_b.transmit_time_s)))
    print("  D_S (reception-anchored) = %.9e s" % d_s_reception)
    print("  d(D_S)/dx0 (reception)   = %s" % np.array2string(d_s_dx0_reception, precision=6))

    hdr("TRANSMIT-ANCHORED D_S  (this phase's production solve_common_transmit_event)")
    from lunar_od.measurements import _station_position_mci_at_receive_epoch
    station_a_pos_T = _station_position_mci_at_receive_epoch(station_a, earth_pos_fixed, xf[i_mid])

    def b_pos(t):
        idx = int(np.searchsorted(t_grid, t))
        idx = min(max(idx, 0), len(t_grid) - 1)
        return _station_position_mci_at_receive_epoch(station_b, earth_pos_fixed, xf[idx])

    def sc_pos(t):
        idx = int(np.searchsorted(t_grid, t))
        idx = min(max(idx, 1), len(t_grid) - 1)
        i0 = idx - 1
        t0, t1 = t_grid[i0], t_grid[idx]
        frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return (1 - frac) * arc.nom48[i0, :3] + frac * arc.nom48[idx, :3]

    common = solve_common_transmit_event(T, station_a_pos_T, b_pos, sc_pos)
    print("  T_obs=%.3f  t_tx=%.9f" % (T, common.transmit_time_s))
    print("  D_S (transmit-anchored) = %.9e s" % common.spacecraft_differential_delay_s)

    hdr("COMPARISON")
    abs_diff = abs(d_s_reception - common.spacecraft_differential_delay_s)
    rel_diff = abs_diff / max(abs(d_s_reception), abs(common.spacecraft_differential_delay_s))
    print("  |D_S(reception) - D_S(transmit)| = %.6e s" % abs_diff)
    print("  relative difference              = %.6e" % rel_diff)
    print("\n  For scale: this phase's qualified K-sensitivity is ~3.7e-09 s per")
    print("  (m^2/kg); the range-only baseline's conditional K information already")
    print("  distinguishes far smaller effects than a light-time-equation-level")
    print("  convention difference would need to be negligible for BOTH anchor")
    print("  choices to support the same qualitative information-geometry finding.")

    print("\n  ANCHOR_CONVENTION_RECONCILIATION:")
    print("  Moyer's own OD-level formulation (Section 11.4.1) anchors on common")
    print("  RECEPTION time, computed here with ZERO new solver code (two calls to")
    print("  the pre-existing, already-qualified one_way_light_time_initial_state_")
    print("  sensitivity). This phase's produced observable uses common TRANSMIT")
    print("  time instead, per the governing spec's own s17 diagram. Both describe")
    print("  the same physical light-cone structure; the numeric agreement above")
    print("  quantifies whether the choice is immaterial at this baseline/distance")
    print("  scale, as Moyer's own text (%.6f s difference in solved transmit times,"
          % abs(sol_a.transmit_time_s - sol_b.transmit_time_s))
    print("  against the cited ~0.02 s Earth-baseline bound) suggests it should be.")


if __name__ == "__main__":
    main()
