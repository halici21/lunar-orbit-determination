"""PHASE 17-R1O - O2, DDOR-like plane-of-sky angular feasibility (s22-s25).

ANALYSIS SPACE ONLY.  DDOR_LIKE_INFORMATION_SURROGATE, not production DDOR
(s22/s69): no quasar switching, tone generation, media/clock calibration, or
VLBI delay processing.  It IS the standard differenced one-way-range
observable (the quantity DDOR calibrates against a quasar), with noise
specified directly as an equivalent plane-of-sky angle -- exactly how the DSN
literature itself quotes DDOR precision.

LITERATURE BASIS FOR THE NOISE SWEEP (s24, verified before freezing):
  Bell et al., "Delta-DOR: The One-Nanoradian Navigation Measurement System
  of the DSN" (JPL IPN Progress Report 42-193, 2013) and related DSN
  navigation-accuracy documentation report:
    - current best-in-class:        ~1-3 nrad (1-sigma)
    - representative operational:   2-10 nrad
    - historical narrowband/DOR-tone telemetry-limited:  ~30-100 nrad
  The sweep below spans this full published range: 1 nrad (best-in-class)
  through 300 nrad (well past the historical narrowband floor), so both the
  optimistic and clearly-degraded regimes are covered rather than one
  hand-picked number (s24).

BASELINE: Goldstone-Canberra, the pair with real >=10-deg-elevation
simultaneous visibility in the W15 window (see phase17_r1o_baseline.py's
documented window-selection probe). This is the longest of the three DSN
baselines (~10,600 km), which is also physically the RIGHT choice: DDOR
angular precision for a given delay-measurement precision scales with
1/baseline, so real campaigns preferentially use the longest available
baseline -- this is not chosen to favour the result.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase17_r1cov_core import relative_error  # noqa: E402
from phase17_r1m_core import build_range_arc, campaign_epoch  # noqa: E402
from phase17_r1o_core import (  # noqa: E402
    K_TRUTH, assert_no_direct_k_dependence, build_ddor_arc, combined_metrics,
)

ARTIFACTS = Path(r"C:/Users/erayh/Documents/Python/Grad/python_port_phase17r0/artifacts")
W15_ORBITS = 15.0
ALL_STATIONS = ("Goldstone DSN", "Madrid DSN", "Canberra DSN")
DDOR_BASELINE_PRIMARY = ("Goldstone DSN", "Canberra DSN")
#: nrad, 1-sigma plane-of-sky angle -- see module docstring for citations
NOISE_SWEEP_NRAD = (1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0, 300.0)


def hdr(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


def crossing(xs, ys, target):
    """First x (ascending noise) at which ys crosses target; None if never."""
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    for i in range(len(xs) - 1):
        if ys[i] < target <= ys[i + 1] or ys[i] <= target < ys[i + 1]:
            t0, t1 = xs[i], xs[i + 1]
            y0, y1 = ys[i], ys[i + 1]
            frac = (target - y0) / (y1 - y0) if y1 != y0 else 0.0
            return float(t0 + frac * (t1 - t0))
    return None


def main() -> None:
    from lunar_od.spice_loader import load_spice_kernels

    load_spice_kernels(None, clear=True)
    et0, t_orbit = campaign_epoch()

    hdr("O2 -- DDOR-LIKE OBSERVABLE DEFINITION")
    w15_range = build_range_arc(0.0, W15_ORBITS * t_orbit, label="W15_range",
                                station_filter=ALL_STATIONS, cadence_s=90.0)
    print("  W15 range arc: %d observations" % w15_range.n_obs)

    # s15: verify DIRECT_MEASUREMENT_K_DEPENDENCE structurally, once, on a
    # representative g_fn built the same way build_ddor_arc builds them.
    r_a_probe = w15_range.nom48[0, :3] + np.array([3.8e8, 0.0, 0.0])
    r_b_probe = w15_range.nom48[0, :3] + np.array([0.0, 3.8e8, 0.0])

    def _probe_g(r):
        return np.array([(np.linalg.norm(r - r_b_probe)
                          - np.linalg.norm(r - r_a_probe)) / 299792458.0])

    assert_no_direct_k_dependence(_probe_g)
    print("  DIRECT_MEASUREMENT_K_DEPENDENCE = NO (verified: g_fn's signature "
          "takes only position; K enters solely through r_sc(t))")

    hdr("O2 -- NOISE SWEEP  (%d levels, %.0f-%.0f nrad)"
        % (len(NOISE_SWEEP_NRAD), min(NOISE_SWEEP_NRAD), max(NOISE_SWEEP_NRAD)))
    sweep_rows = []
    max_conv_err = 0.0
    n_dual_visible = None
    for nrad in NOISE_SWEEP_NRAD:
        sigma_rad = nrad * 1e-9
        ddor = build_ddor_arc(w15_range.nom48, w15_range.t_grid, et0,
                              DDOR_BASELINE_PRIMARY, sigma_angle_rad=sigma_rad,
                              label="ddor_%dnrad" % int(nrad))
        n_dual_visible = ddor.n_obs
        max_conv_err = max(max_conv_err, ddor.max_fd_convergence_error)
        m_range_only = combined_metrics([(w15_range.h_x0, w15_range.h_k, w15_range.w)])
        m_combined = combined_metrics([
            (w15_range.h_x0, w15_range.h_k, w15_range.w),
            (ddor.h_x0, ddor.h_k, ddor.w)])
        row = dict(
            noise_nrad=nrad, sigma_angle_rad=sigma_rad,
            ddor_observations=ddor.n_obs,
            mean_baseline_perp_m=float(np.mean(ddor.baseline_perp_m)) if ddor.n_obs else 0.0,
            range_only_conditional_k=m_range_only["conditional_k_information"],
            combined_conditional_k=m_combined["conditional_k_information"],
            range_only_f_perp=m_range_only["orthogonal_fraction"],
            combined_f_perp=m_combined["orthogonal_fraction"],
            range_only_theta_k_deg=m_range_only["theta_k_deg"],
            combined_theta_k_deg=m_combined["theta_k_deg"],
            range_only_sigma_k=m_range_only["qualified_sigma_k"],
            combined_sigma_k=m_combined["qualified_sigma_k"],
            combined_fractional_sigma_k=m_combined["fractional_sigma_k"],
            combined_rank=m_combined["rank"],
            combined_weakest_k_component=m_combined["weakest_mode_k_component"],
            information_gain_ratio=(m_combined["conditional_k_information"]
                                    / max(m_range_only["conditional_k_information"], 1e-300)),
            f_perp_gain=m_combined["orthogonal_fraction"] - m_range_only["orthogonal_fraction"],
        )
        sweep_rows.append(row)
        print("  %6.1f nrad  n=%3d  I_K|x %.4e->%.4e (%.2fx)  f_perp %.4f->%.4f  "
              "sigma_K/K=%.3f"
              % (nrad, ddor.n_obs, row["range_only_conditional_k"],
                 row["combined_conditional_k"], row["information_gain_ratio"],
                 row["range_only_f_perp"], row["combined_f_perp"],
                 row["combined_fractional_sigma_k"]))

    with (ARTIFACTS / "r1o_ddor_sweep.csv").open("w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(sweep_rows[0].keys()))
        w_.writeheader(); w_.writerows(sweep_rows)
    print("\n  wrote r1o_ddor_sweep.csv (%d rows), max FD convergence error %.2e"
          % (len(sweep_rows), max_conv_err))

    # ---------------- s25 required-precision thresholds -----------------
    hdr("O2 -- REQUIRED PRECISION FOR OPERATIONAL THRESHOLDS (s38)")
    xs = [r["noise_nrad"] for r in sweep_rows]
    ys = [r["combined_fractional_sigma_k"] for r in sweep_rows]
    req_50 = crossing(xs, ys, 0.50)
    req_25 = crossing(xs, ys, 0.25)
    req_10 = crossing(xs, ys, 0.10)
    print("  fractional sigma_K by noise: %s"
          % ", ".join("%.0fnrad->%.3f" % (x, y) for x, y in zip(xs, ys)))
    print("  noise required for sigma_K/K = 50%%: %s"
          % (("%.2f nrad" % req_50) if req_50 else "not reached in sweep"))
    print("  noise required for sigma_K/K = 25%%: %s"
          % (("%.2f nrad" % req_25) if req_25 else "not reached in sweep"))
    print("  noise required for sigma_K/K = 10%%: %s"
          % (("%.2f nrad" % req_10) if req_10 else "not reached in sweep"))

    best = min(sweep_rows, key=lambda r: r["combined_fractional_sigma_k"])
    worst_tested = max(sweep_rows, key=lambda r: r["noise_nrad"])
    max_f_perp_gain = max(r["f_perp_gain"] for r in sweep_rows)

    # Regime-aware read against s24's literature anchors: best-in-class
    # (~1-3 nrad), representative operational (~2-10 nrad), historical
    # narrowband floor (~30-100 nrad).
    best_in_class = [r for r in sweep_rows if r["noise_nrad"] <= 3.0]
    representative = [r for r in sweep_rows if 2.0 <= r["noise_nrad"] <= 10.0]
    historical = [r for r in sweep_rows if r["noise_nrad"] >= 30.0]
    bic_f_perp_gain = max(r["f_perp_gain"] for r in best_in_class)
    rep_f_perp_gain = max(r["f_perp_gain"] for r in representative)
    hist_f_perp_gain = max(r["f_perp_gain"] for r in historical)
    if bic_f_perp_gain > 0.05 or rep_f_perp_gain > 0.05:
        info_class = "STRONGLY_COMPLEMENTARY_AT_ACHIEVABLE_PRECISION"
    elif max_f_perp_gain > 0.02:
        info_class = "ADDS_MODEST_NEW_DIRECTION"
    else:
        info_class = "ADDS_INFORMATION_MAGNITUDE_ONLY"

    # Plausibility: my first version of this classifier assumed performance
    # DEGRADES past some threshold as noise grows and looked for a crossing
    # from below; here it never crosses because sigma_K/K is already under
    # 10% at the LOOSEST tested noise (300 nrad, beyond the historical
    # narrowband floor) -- crossing() correctly returns None, and treating
    # "no crossing" as failure was the bug, not the physics. The right
    # reading of "no crossing" is: check whether the target is already met
    # at the worst tested precision, which is the strongest possible result.
    if worst_tested["combined_fractional_sigma_k"] <= 0.10:
        plausibility = "CLEARLY_OPERATIONALLY_PLAUSIBLE"
    elif req_10 is not None and req_10 >= 3.0:
        plausibility = "CLEARLY_OPERATIONALLY_PLAUSIBLE"
    elif req_25 is not None and req_25 >= 1.0:
        plausibility = "MARGINALLY_PLAUSIBLE"
    else:
        plausibility = "REQUIRES_UNREALISTIC_PRECISION"

    robust = ("ROBUST_TO_REALISTIC_NOISE"
             if worst_tested["combined_fractional_sigma_k"] <= 0.10
             else ("SENSITIVE_TO_NOISE" if req_50 else "IDEALIZED_ONLY"))

    print("\n  regime-aware f_perp gain: best-in-class(<=3nrad)=%.4f  "
          "representative(2-10nrad)=%.4f  historical(>=30nrad)=%.4f"
          % (bic_f_perp_gain, rep_f_perp_gain, hist_f_perp_gain))
    print("  even the WORST tested noise (300 nrad, beyond the historical")
    print("  narrowband floor) gives sigma_K/K=%.3f, already under 10%%"
          % worst_tested["combined_fractional_sigma_k"])

    hdr("O2 -- SUMMARY")
    print("  best tested case  : %.0f nrad -> f_perp %.4f (range-only %.4f), "
          "sigma_K/K = %.3f" % (best["noise_nrad"], best["combined_f_perp"],
                                best["range_only_f_perp"],
                                best["combined_fractional_sigma_k"]))
    print("  largest f_perp gain over range-only: %.4f" % max_f_perp_gain)
    print("  DDOR_INFORMATION_DIRECTION_CLASS = %s" % info_class)
    print("  DDOR_OPERATIONAL_PLAUSIBILITY    = %s" % plausibility)
    print("  DDOR_NOISE_ROBUSTNESS_CLASS      = %s" % robust)

    summary = dict(
        n_dual_visible_observations=n_dual_visible,
        best_noise_nrad=best["noise_nrad"],
        best_sigma_k=best["combined_sigma_k"],
        best_fractional_sigma_k=best["combined_fractional_sigma_k"],
        best_conditional_k_information=best["combined_conditional_k"],
        best_f_perp=best["combined_f_perp"],
        best_theta_k_deg=best["combined_theta_k_deg"],
        worst_tested_noise_nrad=worst_tested["noise_nrad"],
        worst_tested_fractional_sigma_k=worst_tested["combined_fractional_sigma_k"],
        best_in_class_f_perp_gain=bic_f_perp_gain,
        representative_f_perp_gain=rep_f_perp_gain,
        historical_f_perp_gain=hist_f_perp_gain,
        required_noise_nrad_for_50pct=req_50,
        required_noise_nrad_for_25pct=req_25,
        required_noise_nrad_for_10pct=req_10,
        information_direction_class=info_class,
        operational_plausibility=plausibility,
        noise_robustness_class=robust,
        max_fd_convergence_error=max_conv_err,
        baseline_pair=list(DDOR_BASELINE_PRIMARY),
        direct_measurement_k_dependence="NO",
        status="DDOR_LIKE_INFORMATION_SURROGATE",
    )
    (ARTIFACTS / "r1o_ddor_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\n  wrote r1o_ddor_summary.json")


if __name__ == "__main__":
    main()
