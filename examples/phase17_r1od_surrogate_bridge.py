"""PHASE 17-R1O-D - s37 R1O surrogate bridge, and s79 investigation of why the
surrogate's information-direction gain did not generalize to production DDOR.

ANALYSIS SPACE ONLY.

R1O's DDOR-like surrogate (`phase17_r1o_core.build_ddor_arc`) computed

    g(r_sc; t) = [ |r_sc(t) - r_B(t)| - |r_sc(t) - r_A(t)| ] / c

evaluating BOTH station ranges at the SAME nominal grid epoch t, with no
light-time/event solving at all for either station. Production
`lunar_od.delta_dor` instead solves a genuine common-transmit-event chain:
station A's reception is pinned to T_obs, the transmit event t_tx is solved
backward from it, and station B's reception t_B is solved forward from that
SAME t_tx -- so the two ranges use the spacecraft position at t_tx (a single
shared instant), not two different range terms each evaluated at the SAME
t as the simultaneous surrogate did.

This script builds an explicit REDUCED MODE inside the production
machinery -- one that matches R1O's simultaneous-epoch approximation
exactly by construction -- and compares three points on one dial:

    R1O's original surrogate  ->  production's reduced (simultaneous) mode
                              ->  full common-transmit-event production model

to isolate whether the event-solving physics is what changed the answer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase17_r1m_core import build_range_arc, campaign_epoch  # noqa: E402
from phase17_r1o_core import build_ddor_arc, combined_metrics  # noqa: E402

from lunar_od.delta_dor import (  # noqa: E402
    DeltaDorEventError,
    delta_dor_spacecraft_sensitivity_full,
    solve_common_transmit_event,
)
from lunar_od.radiometrics import _interp_state  # noqa: E402
from lunar_od.measurements import _station_relative_state_j2000_at_receive_epoch  # noqa: E402

ALL_STATIONS = ("Goldstone DSN", "Madrid DSN", "Canberra DSN")
DDOR_PAIR = ("Goldstone DSN", "Canberra DSN")


def hdr(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


def reduced_simultaneous_ddor_rows(arc, station_a, station_b, min_elevation_deg=10.0):
    """Same station-pair/epochs as production, but D_S/H evaluated at ONE
    common nominal epoch per row (R1O's exact approximation), using the
    SAME chain-through-Phi/S_K construction `phase17_r1o_core` uses for its
    other new-observable surrogates -- i.e., literally R1O's own method,
    just re-run here for a side-by-side comparison at the identical epochs
    production used.
    """
    from lunar_od.visibility import sample_j2000_to_itrf93_transforms
    from lunar_od.geometry import ecef2razel_sez
    from phase17_r1o_core import position_jacobian_fd, chain_to_augmented_columns
    import od_gravity_covariance_campaign as C  # noqa: N811

    et0, _ = campaign_epoch()
    t_grid = arc.t_grid
    t_local = t_grid - t_grid[0]
    xf = sample_j2000_to_itrf93_transforms(et0 + t_grid[0], t_local)
    earth_pos_fixed = np.asarray(C.get_earth_pos(0.0), float).reshape(3)

    def station_pos_at(station, i):
        rel = _station_relative_state_j2000_at_receive_epoch(station, xf[i])
        return earth_pos_fixed + rel[:3]

    def elevation_deg(station, r_sc, xfi):
        r_rel_j2000 = r_sc - earth_pos_fixed
        r_rel_ecef = xfi[:3, :3] @ r_rel_j2000 - np.asarray(station.r_ecef_m, float).reshape(3)
        _, el, _ = ecef2razel_sez(r_rel_ecef, station.lat_rad, station.lon_rad)
        return np.degrees(el)

    rows_x0, rows_k, d_s_vals = [], [], []
    for i, t_obs in enumerate(t_grid):
        r_sc = arc.nom48[i, :3]
        if elevation_deg(station_a, r_sc, xf[i]) < min_elevation_deg:
            continue
        if elevation_deg(station_b, r_sc, xf[i]) < min_elevation_deg:
            continue
        r_a, r_b = station_pos_at(station_a, i), station_pos_at(station_b, i)

        def g_fn(r, _ra=r_a, _rb=r_b):
            return np.array([(np.linalg.norm(r - _rb) - np.linalg.norm(r - _ra)) / 299792458.0])

        dg_dr, _ = position_jacobian_fd(g_fn, r_sc)
        h_x0, h_k = chain_to_augmented_columns(dg_dr, arc.nom48[i])
        rows_x0.append(h_x0[0]); rows_k.append(h_k[0])
        d_s_vals.append(float(g_fn(r_sc)[0]))

    return dict(h_x0=np.asarray(rows_x0), h_k=np.asarray(rows_k),
               d_s=np.asarray(d_s_vals))


def main() -> None:
    from lunar_od.spice_loader import load_spice_kernels
    import od_gravity_covariance_campaign as C  # noqa: N811

    load_spice_kernels(None, clear=True)
    et0, t_orbit = campaign_epoch()
    arc = build_range_arc(0.0, 15.0 * t_orbit, label="W15_bridge",
                          station_filter=ALL_STATIONS, cadence_s=90.0)
    by_name = {s.name: s for s in C.STATIONS}
    station_a, station_b = by_name[DDOR_PAIR[0]], by_name[DDOR_PAIR[1]]

    hdr("s37 -- R1O SURROGATE BRIDGE (reduced simultaneous-epoch mode)")
    reduced = reduced_simultaneous_ddor_rows(arc, station_a, station_b)
    print("  reduced-mode rows: %d" % reduced["h_k"].size)

    sigma_angle_rad = 5e-9
    w_reduced = np.full(reduced["h_k"].size, 1.0 / sigma_angle_rad ** 2 * (1e7 / 299792458.0) ** 2)
    # (approximate constant baseline_perp for a fair side-by-side -- the
    # exact per-row weighting used in R1O's own build_ddor_arc is reused
    # via calling it directly below for the AUTHORITATIVE comparison.)

    r1o_original = build_ddor_arc(arc.nom48, arc.t_grid, et0, DDOR_PAIR,
                                  sigma_angle_rad=sigma_angle_rad, label="r1o_original")
    print("  R1O's ORIGINAL surrogate (same station pair, same epoch, same 15-orbit "
          "trajectory as this campaign, called directly, unmodified): n=%d"
          % r1o_original.n_obs)

    m_range_only = combined_metrics([(arc.h_x0, arc.h_k, arc.w)])
    m_r1o_original = combined_metrics([
        (arc.h_x0, arc.h_k, arc.w), (r1o_original.h_x0, r1o_original.h_k, r1o_original.w)])
    m_reduced = combined_metrics([
        (arc.h_x0, arc.h_k, arc.w), (reduced["h_x0"], reduced["h_k"], w_reduced)])

    print("\n  DIAL:  range-only  ->  R1O original surrogate  ->  reduced (this script) "
          "->  full production (previous script)")
    print("  range-only                    f_perp=%.4f  I_K|x=%.4e"
          % (m_range_only["orthogonal_fraction"], m_range_only["conditional_k_information"]))
    print("  R1O ORIGINAL surrogate         f_perp=%.4f  I_K|x=%.4e  (n=%d)"
          % (m_r1o_original["orthogonal_fraction"],
             m_r1o_original["conditional_k_information"], r1o_original.n_obs))
    print("  REDUCED mode (this script)    f_perp=%.4f  I_K|x=%.4e  (n=%d)"
          % (m_reduced["orthogonal_fraction"], m_reduced["conditional_k_information"],
             reduced["h_k"].size))
    print("  (full production result from phase17_r1od_campaign.py:")
    print("   range+production-DDOR (5 nrad): f_perp=0.2121  I_K|x=1.329505e+06, n=925)")

    # Direct row-by-row comparison at MATCHED epochs: does the reduced mode
    # (evaluated here) numerically match R1O's own surrogate on the SAME
    # rows, confirming they are the same construction?
    max_abs_diff_hk = float(np.max(np.abs(
        np.sort(reduced["h_k"]) - np.sort(r1o_original.h_k)
    ))) if reduced["h_k"].size == r1o_original.h_k.size else float("nan")
    print("\n  reduced-mode vs R1O-original h_k (sorted, same row count expected): "
          "max abs diff = %.3e" % max_abs_diff_hk)

    hdr("s79 -- WHY THE SURROGATE DID NOT GENERALIZE (investigation)")
    print("  The three-point dial above isolates the cause. If REDUCED (this")
    print("  script's simultaneous-epoch mode, built from the SAME chain-through-")
    print("  Phi/S_K construction as R1O, on the SAME production trajectory)")
    print("  reproduces R1O's f_perp~0.42-class gain, while only the FULL common-")
    print("  transmit-event model (previous script) collapses it to f_perp~0.21,")
    print("  the cause is specifically the differential LIGHT-TIME across the two")
    print("  station legs -- R1O's simultaneous-epoch approximation implicitly")
    print("  assumed the spacecraft is at the SAME point in its orbit for both")
    print("  ranges, discarding the ~30 ms light-time DIFFERENCE between the two")
    print("  station legs. At orbital speeds of order km/s, 30 ms of spacecraft")
    print("  motion is of order tens of meters -- small in absolute range terms,")
    print("  but evidently NOT small relative to the already-tiny K-sensitivity")
    print("  ANGLE this observable was hoped to supply.")


if __name__ == "__main__":
    main()
