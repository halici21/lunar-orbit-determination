"""PHASE 17-R1O - COV/entry gate reproduction, window selection, O0/O1 controls.

ANALYSIS SPACE ONLY.

WINDOW SELECTION (declared from a geometric probe, before any K result was
inspected -- s19/s27's "before inspecting final K metrics" discipline applied
to picking a DDOR-capable window rather than to landmark placement):

R1M's 9-orbit (~17.7 h) campaign window never gives simultaneous >10-deg
elevation at ANY DSN station pair -- Goldstone and Madrid never rise above
~17-23 deg in that window, so no real baseline pair is ever dual-visible at
the qualified 10-deg threshold used everywhere else in this project.  That
window covers less than one Earth sidereal day, so it only samples part of
the diurnal Earth-Moon geometry cycle.

Extending to 15 orbits (~29.4 h, just over one sidereal day) was tested
BEFORE any DDOR information result was computed, purely as a visibility
count: Goldstone-Canberra reaches 72 simultaneous >10-deg samples at 180-s
cadence; Goldstone-Madrid and Madrid-Canberra reach 0 at 10 deg and only a
handful at <=5 deg.  R1O therefore uses:

  * the ORIGINAL G1 arc (2.0 orbits, Canberra only) for continuity with the
    R1COV/R1M qualification numbers (the entry-gate reproduction, s8), and

  * a new 15-orbit, three-station window ("W15") as the COMMON campaign for
    every cross-observable comparison in this phase, with Goldstone-Canberra
    as the DDOR baseline.  This is a REAL geometric fact about when a DDOR
    pass could be scheduled in this orbit, not a result chosen to favour any
    observable.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase17_r1cov_core import exact_covariance, scale_matrix  # noqa: E402
from phase17_r1m_core import build_range_arc, campaign_epoch  # noqa: E402
from phase17_r1o_core import combined_metrics, relative_error  # noqa: E402

ARTIFACTS = Path(r"C:/Users/erayh/Documents/Python/Grad/python_port_phase17r0/artifacts")
W15_ORBITS = 15.0
ALL_STATIONS = ("Goldstone DSN", "Madrid DSN", "Canberra DSN")
OUT: dict = {}


def hdr(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


def main() -> None:
    from lunar_od.spice_loader import load_spice_kernels
    from lunar_od.estimators import (
        _safe_covariance_from_information, _square_root_covariance_from_design,
    )

    load_spice_kernels(None, clear=True)
    et0, t_orbit = campaign_epoch()

    # ---------------- s10 qualified-covariance reproduction gate ------
    hdr("s10  QUALIFIED K COVARIANCE REPRODUCTION GATE")
    g1 = build_range_arc(0.0, 2.0 * t_orbit, label="G1")
    h = np.hstack([g1.h_x0, g1.h_k[:, None]])
    scale = scale_matrix()
    cov_prod, _ = _square_root_covariance_from_design(
        h, g1.w, np.zeros((7, 7)), scale)
    sig_prod = float(np.sqrt(cov_prod[6, 6]))
    _, sig_exact = exact_covariance(h, g1.w, None)
    info_old = scale.T @ (h.T @ (g1.w[:, None] * h)) @ scale
    sig_old = float(np.sqrt(
        (scale @ _safe_covariance_from_information(info_old) @ scale.T)[6, 6]))
    rel = relative_error(sig_prod, sig_exact)
    gate = rel < 1e-6
    print("  G1 qualified square-root sigma_K = %.10e" % sig_prod)
    print("  G1 exact oracle       sigma_K     = %.10e" % sig_exact)
    print("  G1 OLD floored path    sigma_K     = %.10e  (confirmed NOT active, "
          "ratio %.2fx)" % (sig_old, sig_prod / sig_old))
    print("  relative error vs exact oracle     = %.3e" % rel)
    print("  QUALIFIED_K_COVARIANCE_REPRODUCTION_GATE = %s" % ("PASS" if gate else "FAIL"))
    if not gate:
        raise SystemExit("R1COV covariance contract not reproducible -- STOP per s10.")
    OUT["qualified_k_covariance_reproduction_gate"] = "PASS"

    # ---------------- baseline G1 (continuity control) -----------------
    hdr("BASELINE_SIGMA_K reference (G1, 2.0 orbits, Canberra)")
    m_g1 = combined_metrics([(g1.h_x0, g1.h_k, g1.w)])
    for k, v in m_g1.items():
        print("  %-32s %s" % (k, v))
    OUT["g1_baseline"] = m_g1

    # ---------------- window-selection probe (documented) --------------
    hdr("WINDOW SELECTION -- dual-visibility probe (decided before any K result)")
    from lunar_od.visibility import sample_j2000_to_itrf93_transforms
    from phase17_r1o_core import station_elevation_deg
    import od_gravity_covariance_campaign as C  # noqa: N811

    probe = build_range_arc(0.0, W15_ORBITS * t_orbit, label="probe",
                            station_filter=ALL_STATIONS, cadence_s=180.0)
    t_local = probe.t_grid - probe.t_grid[0]
    xf = sample_j2000_to_itrf93_transforms(et0 + probe.t_grid[0], t_local)
    by_name = {s.name: s for s in C.STATIONS}
    els = {n: [] for n in by_name}
    for i, t_abs in enumerate(probe.t_grid):
        r_sc = probe.nom48[i, :3]
        earth_pos = C.get_earth_pos(t_abs)
        for n, s in by_name.items():
            els[n].append(station_elevation_deg(s, r_sc, earth_pos, xf[i]))
    names = list(by_name)
    probe_rows = []
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = np.array(els[names[i]]), np.array(els[names[j]])
            for thr in (0, 5, 10):
                cnt = int(((a > thr) & (b > thr)).sum())
                probe_rows.append(dict(station_a=names[i], station_b=names[j],
                                       elevation_threshold_deg=thr,
                                       simultaneous_visible_count=cnt))
                print("  %s / %s  thr=%2d deg  simultaneous=%d"
                      % (names[i], names[j], thr, cnt))
    with (ARTIFACTS / "r1o_window_probe.csv").open("w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(probe_rows[0].keys()))
        w_.writeheader(); w_.writerows(probe_rows)
    OUT["window_probe"] = probe_rows

    # ---------------- O0 -- range control on W15 ------------------------
    hdr("O0 -- RANGE CONTROL on W15 (15 orbits, all 3 DSN stations)")
    w15_range = build_range_arc(0.0, W15_ORBITS * t_orbit, label="W15_range",
                                station_filter=ALL_STATIONS, cadence_s=CADENCE_FOR_W15)
    m_o0 = combined_metrics([(w15_range.h_x0, w15_range.h_k, w15_range.w)])
    for k, v in m_o0.items():
        print("  %-32s %s" % (k, v))
    OUT["o0_range_w15"] = m_o0

    # ---------------- O1 -- range+Doppler control (carried forward) ----
    hdr("O1 -- RANGE + COUNTED DOPPLER CONTROL  (NOT_RERUN_CARRIED_FORWARD_FROM_R1G)")
    print("  R1G's own combined range+four-event-counted-Doppler information")
    print("  characterization (SR-UKF architecture, G6/G7) is carried forward")
    print("  rather than rebuilt, per the same NOT_RERUN policy R1M applied to")
    print("  this identical data (R1M s23). Rebuilding the four-event Doppler")
    print("  K-sensitivity column from scratch would require wiring together")
    print("  production endpoint sensitivities not exposed as a single K-column")
    print("  helper (unlike range's _two_way_range_k_srp_column); doing so was")
    print("  judged lower priority than the two genuinely new Tier-1 observables")
    print("  this phase exists to test (s21 marks this control secondary).")
    r1g_doppler_only_i_kk = 247.5
    r1g_range_doppler_i_kk = 260.8
    r1g_weakest_k_component = 1.0  # R1G's own reported value, ~1.0 for both
    print("\n  R1G G6  (Doppler-only)         I_KK = %.1f  rank 6/7  weakest_K ~ %.4f"
          % (r1g_doppler_only_i_kk, r1g_weakest_k_component))
    print("  R1G G7  (range + Doppler)      I_KK = %.1f  rank 6/7  weakest_K ~ %.4f"
          % (r1g_range_doppler_i_kk, r1g_weakest_k_component))
    gain = r1g_range_doppler_i_kk / r1g_doppler_only_i_kk
    print("  combined/Doppler-only I_KK ratio = %.3fx  (magnitude gain only;" % gain)
    print("  weakest mode stays ~fully K -- R1G reports no rotation, consistent")
    print("  with R1M's f_perp saturation finding for range alone)")
    OUT["o1_range_doppler_carried_forward"] = dict(
        source="R1G", doppler_only_i_kk=r1g_doppler_only_i_kk,
        range_doppler_i_kk=r1g_range_doppler_i_kk,
        weakest_k_component=r1g_weakest_k_component,
        gain_ratio=gain, rank="6/7",
        classification="ADDS_INFORMATION_MAGNITUDE_ONLY (R1G-reported, not re-derived)")

    (ARTIFACTS / "r1o_baseline.json").write_text(json.dumps(OUT, indent=2, default=float))
    print("\n  wrote r1o_baseline.json")

    with (ARTIFACTS / "r1o_baseline.csv").open("w", newline="") as fh:
        rows = [dict(case="G1_continuity", **m_g1), dict(case="O0_range_W15", **m_o0)]
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w_.writeheader(); w_.writerows(rows)
    print("  wrote r1o_baseline.csv")


CADENCE_FOR_W15 = 90.0

if __name__ == "__main__":
    main()
