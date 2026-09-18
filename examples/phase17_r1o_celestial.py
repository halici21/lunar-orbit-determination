"""PHASE 17-R1O - O4, optional celestial LOS characterization (s31).

ANALYSIS SPACE ONLY.  Secondary to DDOR and landmark LOS per s31.

Characterizes line-of-sight angle to Earth's center as seen from the
spacecraft. This is the SAME angular-observable construction as the landmark
surrogate (a nadir-independent inertial direction rather than a body-fixed
one), so it reuses `position_jacobian_fd` / `chain_to_augmented_columns`
directly rather than a new geometric model.

The navigation-relevance question this answers is narrower than the landmark
case: an Earth-LOS angle is an INERTIAL direction (Earth's position relative
to the Moon-centred frame moves only slowly, on the ~1-month Earth-Moon
orbital timescale), so unlike a lunar landmark it carries almost no
sensitivity to the spacecraft's own fast orbital motion around the Moon --
it mostly measures the spacecraft's position relative to the Earth-Moon
line, which is a very different -- and, as the result below shows, much
weaker -- geometric quantity than a body-fixed landmark bearing.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase17_r1m_core import build_range_arc, campaign_epoch  # noqa: E402
from phase17_r1o_core import (  # noqa: E402
    assert_no_direct_k_dependence, chain_to_augmented_columns, combined_metrics,
    position_jacobian_fd,
)

ARTIFACTS = Path(r"C:/Users/erayh/Documents/Python/Grad/python_port_phase17r0/artifacts")
W15_ORBITS = 15.0
ALL_STATIONS = ("Goldstone DSN", "Madrid DSN", "Canberra DSN")
NOISE_SWEEP_URAD = (10.0, 100.0, 1000.0)


def hdr(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


def main() -> None:
    from lunar_od.spice_loader import load_spice_kernels
    import od_gravity_covariance_campaign as C  # noqa: N811

    load_spice_kernels(None, clear=True)
    et0, t_orbit = campaign_epoch()

    hdr("O4 -- CELESTIAL LOS (EARTH CENTER) CHARACTERIZATION")
    w15_range = build_range_arc(0.0, W15_ORBITS * t_orbit, label="W15_range",
                                station_filter=ALL_STATIONS, cadence_s=90.0)
    print("  W15 range arc: %d observations" % w15_range.n_obs)

    def _probe_g(r):
        rho = np.array([3.844e8, 0.0, 0.0]) - r
        rho_h = rho / np.linalg.norm(rho)
        return np.array([np.arcsin(np.clip(rho_h[1], -1, 1)),
                         np.arcsin(np.clip(rho_h[2], -1, 1))])

    assert_no_direct_k_dependence(_probe_g)
    print("  DIRECT_MEASUREMENT_K_DEPENDENCE = NO (verified structurally)")

    rows_x0, rows_k, max_conv = [], [], 0.0
    n = w15_range.t_grid.size
    for i in range(0, n, 4):  # every 4th sample: Earth LOS is slowly varying
        t_abs = w15_range.t_grid[i]
        r_sc = w15_range.nom48[i, :3]
        r_earth = np.asarray(C.get_earth_pos(t_abs), float).reshape(3)
        e1 = np.array([0.0, 1.0, 0.0])
        e2 = np.array([0.0, 0.0, 1.0])

        def g_fn(r, _re=r_earth):
            rho = _re - r
            rho_h = rho / np.linalg.norm(rho)
            return np.array([np.arcsin(np.clip(np.dot(rho_h, e1), -1, 1)),
                             np.arcsin(np.clip(np.dot(rho_h, e2), -1, 1))])

        dg_dr, rel = position_jacobian_fd(g_fn, r_sc)
        max_conv = max(max_conv, rel)
        h_x0, h_k = chain_to_augmented_columns(dg_dr, w15_range.nom48[i])
        rows_x0.append(h_x0[0]); rows_k.append(h_k[0])
        rows_x0.append(h_x0[1]); rows_k.append(h_k[1])

    h_x0 = np.asarray(rows_x0); h_k = np.asarray(rows_k)
    print("  Earth-LOS rows: %d, max FD convergence error %.2e" % (h_k.size, max_conv))

    sweep_rows = []
    for urad in NOISE_SWEEP_URAD:
        w = np.full(h_k.size, 1.0 / (urad * 1e-6) ** 2)
        m_combined = combined_metrics([(w15_range.h_x0, w15_range.h_k, w15_range.w),
                                       (h_x0, h_k, w)])
        m_range_only = combined_metrics([(w15_range.h_x0, w15_range.h_k, w15_range.w)])
        row = dict(noise_urad=urad, earth_los_observations=h_k.size,
                  range_only_f_perp=m_range_only["orthogonal_fraction"],
                  combined_f_perp=m_combined["orthogonal_fraction"],
                  combined_sigma_k=m_combined["qualified_sigma_k"],
                  combined_fractional_sigma_k=m_combined["fractional_sigma_k"],
                  f_perp_gain=m_combined["orthogonal_fraction"]
                  - m_range_only["orthogonal_fraction"])
        sweep_rows.append(row)
        print("  %6.0f urad  f_perp %.4f->%.4f  sigma_K/K=%.4f"
              % (urad, row["range_only_f_perp"], row["combined_f_perp"],
                 row["combined_fractional_sigma_k"]))

    with (ARTIFACTS / "r1o_celestial_los.csv").open("w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(sweep_rows[0].keys()))
        w_.writeheader(); w_.writerows(sweep_rows)

    max_gain = max(r["f_perp_gain"] for r in sweep_rows)
    status = ("CHARACTERIZED_NEGLIGIBLE_GAIN" if max_gain < 0.01
             else "CHARACTERIZED_MODEST_GAIN")
    print("\n  CELESTIAL_LOS_STATUS = %s (max f_perp gain %.4f)" % (status, max_gain))
    (ARTIFACTS / "r1o_celestial_summary.json").write_text(json.dumps(
        dict(status=status, max_f_perp_gain=max_gain,
            max_fd_convergence_error=max_conv,
            direct_measurement_k_dependence="NO"), indent=2, default=float))
    print("  wrote r1o_celestial_los.csv, r1o_celestial_summary.json")


if __name__ == "__main__":
    main()
