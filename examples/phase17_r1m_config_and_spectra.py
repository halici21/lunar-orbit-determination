"""PHASE 17-R1M - frozen configuration record and full singular spectra (s53).

ANALYSIS SPACE ONLY.

Two deliverables the main scripts do not produce:

`r1m_config.json`   every input that would have to be reproduced to repeat
                    this study, including the Git state it was run against,
                    so the numbers are traceable to a commit rather than to a
                    session.

`r1m_multi_arc_singular_values.csv`
                    the COMPLETE singular spectrum of every information matrix
                    in the study, not just the smallest value. s21 forbids
                    tuning a rank tolerance to manufacture 7/7, and the honest
                    way to respect that is to publish the whole spectrum and
                    let the reader see where -- if anywhere -- a defensible
                    cut could fall.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase17_r1m_core import (  # noqa: E402
    ATOL, CADENCE_S, K_INITIAL, K_TRUTH, K_WRONG, RTOL, SCALE_K, SCALE_POS,
    SCALE_VEL, build_range_arc, campaign_epoch, information_matrix,
    scale_matrix, spectrum,
)
from phase17_r1m_multi_arc_study import (  # noqa: E402
    ALL_STATIONS, CAMPAIGNS, WINDOW_ORBITS, model_a_information,
    model_b_information,
)

REPO = Path(r"C:/Users/erayh/Documents/Python/Grad/python_port_phase17r0")
ARTIFACTS = REPO / "artifacts"
SINGLE_CASES = (("G0", 1.3), ("G1", 2.0), ("G2", 3.0), ("G3", 5.0))


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True).stdout.strip()


def main() -> None:
    from lunar_od.spice_loader import load_spice_kernels

    load_spice_kernels(None, clear=True)
    et0, t_orbit = campaign_epoch()

    cfg = dict(
        phase="PHASE 17-R1M",
        scope="ANALYSIS_ONLY_NO_PRODUCTION_CHANGE",
        git=dict(head=git("rev-parse", "HEAD"),
                 branch=git("rev-parse", "--abbrev-ref", "HEAD"),
                 main=git("rev-parse", "main"),
                 origin_main=git("rev-parse", "origin/main"),
                 dirty_tracked_files=git("status", "--porcelain",
                                         "--untracked-files=no").splitlines()),
        provenance="SYNTHETIC_CAMPAIGN_TRUTH",
        truth=dict(k_srp_truth_m2_per_kg=K_TRUTH,
                   k_srp_wrong_fixed=K_WRONG, k_srp_solve_initial=K_INITIAL),
        epoch=dict(campaign_et0=et0, orbit_period_s=t_orbit),
        sampling=dict(cadence_s=CADENCE_S, integrator_rtol=RTOL,
                      integrator_atol=ATOL),
        scaling=dict(position_m=SCALE_POS, velocity_m_s=SCALE_VEL,
                     k_srp_m2_per_kg=SCALE_K),
        measurement=dict(family="two_way_converged_event_range",
                         four_event_counted_doppler_substituted=False,
                         geometric_range_rate_substituted=False,
                         noise=False, min_elevation_deg=10.0,
                         tolerance_s=1e-13, equation_tolerance_s=1e-14),
        schedule=dict(window_orbits=WINDOW_ORBITS, campaigns=CAMPAIGNS,
                      stations=list(ALL_STATIONS),
                      single_arc_cases={k: v for k, v in SINGLE_CASES}),
        frozen_production_files=[
            "lunar_od/dynamics.py", "lunar_od/two_way_range.py",
            "lunar_od/radiometrics.py",
            "lunar_od/two_way_counted_doppler_reference.py",
            "lunar_od/estimators.py", "lunar_od/filters.py"],
    )
    (ARTIFACTS / "r1m_config.json").write_text(json.dumps(cfg, indent=2,
                                                          default=float))
    print("wrote r1m_config.json  (head %s, main %s)"
          % (cfg["git"]["head"][:12], cfg["git"]["main"][:12]))

    rows = []

    def emit(case, model, unknowns, n_obs, sp):
        sv = np.asarray(sp["singular_values"], float)
        for i, v in enumerate(sv):
            rows.append(dict(case=case, model=model, unknowns=unknowns,
                             observations=n_obs, index=i, singular_value=v,
                             ratio_to_largest=float(v / sv[0]),
                             is_k_dominant_mode=bool(
                                 i == len(sv) - 1
                                 and sp["weakest_k_component"] > 0.9)))

    print("\nsingle contiguous arcs")
    for label, orbits in SINGLE_CASES:
        arc = build_range_arc(0.0, orbits * t_orbit, label=label)
        sp = spectrum(information_matrix(
            np.hstack([arc.h_x0, arc.h_k[:, None]]), arc.w, scale_matrix()))
        emit(label, "single_contiguous", 7, arc.n_obs, sp)
        print("  %-3s sv[0]=%.6e sv[-1]=%.6e cond=%.3e weakK=%.12f"
              % (label, sp["largest"], sp["smallest"], sp["condition"],
                 sp["weakest_k_component"]))

    print("\nmulti-arc campaigns")
    window_s = WINDOW_ORBITS * t_orbit
    for name, starts in CAMPAIGNS.items():
        windows = [(s * t_orbit, s * t_orbit + window_s) for s in starts]
        arcs = [build_range_arc(w0, window_s, label="%s_a%d" % (name, i),
                                station_filter=ALL_STATIONS)
                for i, (w0, _) in enumerate(windows)]
        a = model_a_information(arcs)
        long_arc = build_range_arc(0.0, windows[-1][1], label="%s_long" % name,
                                   station_filter=ALL_STATIONS)
        b = model_b_information(long_arc, windows)
        emit(name, "A_independent_arcs", a["n_unknown"], a["n_obs"], a["spectrum"])
        emit(name, "B_continuity_linked", 7, b["n_obs"], b["spectrum"])
        print("  %s  A cond=%.3e weakK=%.9f   B cond=%.3e weakK=%.9f"
              % (name, a["spectrum"]["condition"],
                 a["spectrum"]["weakest_k_component"],
                 b["spectrum"]["condition"],
                 b["spectrum"]["weakest_k_component"]))

    path = ARTIFACTS / "r1m_multi_arc_singular_values.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote r1m_multi_arc_singular_values.csv (%d rows)" % len(rows))


if __name__ == "__main__":
    main()
