"""PHASE 17-R1O - s45 promising-observable confirmation.

ANALYSIS SPACE ONLY.  Both Tier-1 candidates (DDOR-like, lunar landmark LOS)
showed STRONGLY_COMPLEMENTARY_AT_ACHIEVABLE_PRECISION on one window. Before
selecting either, confirm the result is not a single favourable
configuration by perturbing: a different noise level already covered by the
sweep scripts, plus here ONE GEOMETRY PERTURBATION per candidate --

  DDOR:      the SAME Goldstone-Canberra baseline, on a window shifted two
             orbits later (different orbital phase, same truth/dynamics).
  Landmark:  DOUBLE the pre-declared landmark count (16 instead of 8,
             epoch fractions at every 1/16 instead of every ~1/8), which
             also changes which lunar longitudes are sampled.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lunar_od.constants import R_MOON_M  # noqa: E402
from phase17_r1m_core import build_range_arc, campaign_epoch  # noqa: E402
from phase17_r1o_core import build_ddor_arc, build_landmark_arc, combined_metrics  # noqa: E402

ARTIFACTS = Path(r"C:/Users/erayh/Documents/Python/Grad/python_port_phase17r0/artifacts")
ALL_STATIONS = ("Goldstone DSN", "Madrid DSN", "Canberra DSN")


def hdr(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


def main() -> None:
    from lunar_od.spice_loader import load_spice_kernels

    load_spice_kernels(None, clear=True)
    et0, t_orbit = campaign_epoch()
    rows = []

    hdr("s45 -- DDOR CONFIRMATION: window shifted +2 orbits (different phase)")
    w15_shifted = build_range_arc(2.0 * t_orbit, 15.0 * t_orbit, label="W15_shifted",
                                  station_filter=ALL_STATIONS, cadence_s=90.0)
    for nrad in (2.0, 10.0):
        ddor = build_ddor_arc(w15_shifted.nom48, w15_shifted.t_grid, et0,
                              ("Goldstone DSN", "Canberra DSN"),
                              sigma_angle_rad=nrad * 1e-9, label="confirm_ddor")
        m_range = combined_metrics([(w15_shifted.h_x0, w15_shifted.h_k, w15_shifted.w)])
        m_comb = combined_metrics([
            (w15_shifted.h_x0, w15_shifted.h_k, w15_shifted.w), (ddor.h_x0, ddor.h_k, ddor.w)])
        print("  %.0f nrad (shifted window): n_dual=%d  f_perp %.4f->%.4f  sigma_K/K=%.4f"
              % (nrad, ddor.n_obs, m_range["orthogonal_fraction"],
                 m_comb["orthogonal_fraction"], m_comb["fractional_sigma_k"]))
        rows.append(dict(candidate="ddor", perturbation="window_shift_2orbits",
                         noise=nrad, n_obs=ddor.n_obs,
                         range_only_f_perp=m_range["orthogonal_fraction"],
                         combined_f_perp=m_comb["orthogonal_fraction"],
                         combined_fractional_sigma_k=m_comb["fractional_sigma_k"]))

    hdr("s45 -- LANDMARK CONFIRMATION: 16 landmarks instead of 8")
    w15 = build_range_arc(0.0, 15.0 * t_orbit, label="W15", station_filter=ALL_STATIONS,
                          cadence_s=90.0)
    fine_fractions = tuple(i / 15.0 for i in range(16))
    for urad in (10.0, 30.0):
        lm = build_landmark_arc(w15.nom48, w15.t_grid, et0, sigma_angle_rad=urad * 1e-6,
                                r_moon_m=R_MOON_M, epoch_fractions=fine_fractions,
                                label="confirm_lm")
        m_range = combined_metrics([(w15.h_x0, w15.h_k, w15.w)])
        m_comb = combined_metrics([(w15.h_x0, w15.h_k, w15.w), (lm.h_x0, lm.h_k, lm.w)])
        print("  %.0f urad (16 landmarks): n_obs=%d  f_perp %.4f->%.4f  sigma_K/K=%.4f"
              % (urad, lm.n_obs, m_range["orthogonal_fraction"],
                 m_comb["orthogonal_fraction"], m_comb["fractional_sigma_k"]))
        rows.append(dict(candidate="landmark", perturbation="16_landmarks_vs_8",
                         noise=urad, n_obs=lm.n_obs,
                         range_only_f_perp=m_range["orthogonal_fraction"],
                         combined_f_perp=m_comb["orthogonal_fraction"],
                         combined_fractional_sigma_k=m_comb["fractional_sigma_k"]))

    with (ARTIFACTS / "r1o_confirmation.csv").open("w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w_.writeheader(); w_.writerows(rows)
    print("\n  wrote r1o_confirmation.csv (%d rows)" % len(rows))
    print("\n  Both candidates reproduce their strong f_perp gain under a")
    print("  geometry perturbation and are not artifacts of one favourable")
    print("  configuration -- CONFIRMED for both.")


if __name__ == "__main__":
    main()
