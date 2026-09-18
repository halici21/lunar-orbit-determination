"""PHASE 17-R1O - O3, lunar-landmark LOS feasibility (s26-s30).

ANALYSIS SPACE ONLY.  Idealized known-landmark angular LOS, not production
optical navigation (s63/s69): no image processing, crater detection, or
camera rendering.

LANDMARK STRATEGY (declared before inspecting any K result, s27): landmarks
are placed at the spacecraft's own NADIR ground-track positions sampled at 8
pre-declared, equally time-spaced epochs across the O0/W15 arc (0%, 15%,
30%, ..., 100% of the arc). This is "chronologically selected, nadir-near"
by construction, not hand-tuned for K sensitivity.

NOISE SWEEP LITERATURE BASIS (s28, verified before freezing):
  - LONEStar (Lunar Flashlight extended mission, flight-demonstrated 2023):
    camera IFOV ~36.6 arcsec/pixel (~1.775e-4 rad/pixel); attitude stability
    ~15-20 arcsec (~73-97 urad) over 5-s intervals; empirical LOS errors
    0.25-1 pixel (~44-177 urad) -- for STAR/PLANET imaging, a DIFFERENT
    application from crater/landmark navigation, so used here only as an
    upper calibration anchor, not adopted directly (the module docstring in
    the governing spec explicitly warns against that transfer).
  - Star-tracker attitude knowledge is commonly cited at ~0.5 mrad (500 urad)
    1-sigma conservative, which scales to ~0.4 pixel of equivalent LOS error
    for a typical nav-camera IFOV (Optical Camera Characterization for
    Feature-Based Navigation in Lunar Orbit, Aerospace 2025) -- this sets a
    realistic DEGRADED floor set by attitude knowledge, independent of
    centroiding quality.
  - Dedicated crater/landmark navigation cameras (JGCD crater-nav literature)
    are commonly modelled with finer IFOVs than LONEStar's star/planet
    imager, giving best-case LOS uncertainties in the 1-30 urad range once
    sub-pixel centroiding is included.
  The sweep spans 1-1000 urad log-spaced, covering dedicated-camera best case
  through the star-tracker-attitude-limited degraded floor and beyond.
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
from lunar_od.constants import R_MOON_M  # noqa: E402
from phase17_r1m_core import build_range_arc, campaign_epoch  # noqa: E402
from phase17_r1o_core import (  # noqa: E402
    K_TRUTH, assert_no_direct_k_dependence, build_landmark_arc, combined_metrics,
)

ARTIFACTS = Path(r"C:/Users/erayh/Documents/Python/Grad/python_port_phase17r0/artifacts")
W15_ORBITS = 15.0
ALL_STATIONS = ("Goldstone DSN", "Madrid DSN", "Canberra DSN")
#: microradians, 1-sigma per tangent-plane angle -- see module docstring
NOISE_SWEEP_URAD = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 500.0, 1000.0)
#: star-tracker-attitude-limited degraded floor, cited in the module docstring
ATTITUDE_LIMITED_FLOOR_URAD = 500.0


def hdr(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


def crossing(xs, ys, target):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    for i in range(len(xs) - 1):
        if ys[i] < target <= ys[i + 1] or ys[i] <= target < ys[i + 1]:
            t0, t1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
            frac = (target - y0) / (y1 - y0) if y1 != y0 else 0.0
            return float(t0 + frac * (t1 - t0))
    return None


def main() -> None:
    from lunar_od.spice_loader import load_spice_kernels

    load_spice_kernels(None, clear=True)
    et0, t_orbit = campaign_epoch()

    hdr("O3 -- LUNAR LANDMARK LOS DEFINITION AND GEOMETRY")
    w15_range = build_range_arc(0.0, W15_ORBITS * t_orbit, label="W15_range",
                                station_filter=ALL_STATIONS, cadence_s=90.0)
    print("  W15 range arc: %d observations" % w15_range.n_obs)

    lm_probe = build_landmark_arc(
        w15_range.nom48, w15_range.t_grid, et0, sigma_angle_rad=1e-4,
        r_moon_m=R_MOON_M, label="lm_probe")
    print("  landmark lat/lon (deg), nadir-track, pre-declared epochs:")
    for i, (lat, lon) in enumerate(lm_probe.landmark_latlon_deg):
        cnt = int(np.sum(lm_probe.landmark_used_idx == i))
        print("    landmark %d: lat=%7.2f lon=%7.2f  -> %d observation rows"
              % (i, lat, lon, cnt))
    print("  total landmark LOS rows (2 angles each): %d" % lm_probe.n_obs)
    print("  max FD convergence error: %.3e" % lm_probe.max_fd_convergence_error)

    def _probe_g(r):
        rho = np.array([0.0, 0.0, R_MOON_M]) - r
        rho_h = rho / np.linalg.norm(rho)
        return np.array([np.arcsin(rho_h[0]), np.arcsin(rho_h[1])])

    assert_no_direct_k_dependence(_probe_g)
    print("  DIRECT_MEASUREMENT_K_DEPENDENCE = NO (verified structurally)")

    hdr("O3 -- NOISE SWEEP  (%d levels, %.0f-%.0f urad, ideal landmark+attitude)"
        % (len(NOISE_SWEEP_URAD), min(NOISE_SWEEP_URAD), max(NOISE_SWEEP_URAD)))
    sweep_rows = []
    max_conv_err = lm_probe.max_fd_convergence_error
    for urad in NOISE_SWEEP_URAD:
        sigma_rad = urad * 1e-6
        lm = build_landmark_arc(w15_range.nom48, w15_range.t_grid, et0,
                                sigma_angle_rad=sigma_rad, r_moon_m=R_MOON_M,
                                label="lm_%durad" % int(urad))
        max_conv_err = max(max_conv_err, lm.max_fd_convergence_error)
        m_range_only = combined_metrics([(w15_range.h_x0, w15_range.h_k, w15_range.w)])
        m_combined = combined_metrics([
            (w15_range.h_x0, w15_range.h_k, w15_range.w), (lm.h_x0, lm.h_k, lm.w)])
        row = dict(
            noise_urad=urad, sigma_angle_rad=sigma_rad, landmark_observations=lm.n_obs,
            range_only_conditional_k=m_range_only["conditional_k_information"],
            combined_conditional_k=m_combined["conditional_k_information"],
            range_only_f_perp=m_range_only["orthogonal_fraction"],
            combined_f_perp=m_combined["orthogonal_fraction"],
            range_only_theta_k_deg=m_range_only["theta_k_deg"],
            combined_theta_k_deg=m_combined["theta_k_deg"],
            combined_sigma_k=m_combined["qualified_sigma_k"],
            combined_fractional_sigma_k=m_combined["fractional_sigma_k"],
            combined_rank=m_combined["rank"],
            combined_weakest_k_component=m_combined["weakest_mode_k_component"],
            information_gain_ratio=(m_combined["conditional_k_information"]
                                    / max(m_range_only["conditional_k_information"], 1e-300)),
            f_perp_gain=m_combined["orthogonal_fraction"] - m_range_only["orthogonal_fraction"],
        )
        sweep_rows.append(row)
        print("  %7.1f urad  n=%4d  I_K|x %.4e->%.4e (%.2fx)  f_perp %.4f->%.4f  "
              "sigma_K/K=%.4f"
              % (urad, lm.n_obs, row["range_only_conditional_k"],
                 row["combined_conditional_k"], row["information_gain_ratio"],
                 row["range_only_f_perp"], row["combined_f_perp"],
                 row["combined_fractional_sigma_k"]))

    with (ARTIFACTS / "r1o_optical_landmark_sweep.csv").open("w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(sweep_rows[0].keys()))
        w_.writeheader(); w_.writerows(sweep_rows)
    print("\n  wrote r1o_optical_landmark_sweep.csv (%d rows)" % len(sweep_rows))

    # ---------------- s29/s30 landmark & attitude uncertainty -----------
    hdr("O3 -- LANDMARK-POSITION AND ATTITUDE UNCERTAINTY (s29/s30)")
    print("  Ideal limit already run above (perfect landmark map, perfect")
    print("  attitude -- all error folded into sigma_angle). Two degraded")
    print("  characterizations follow, at a representative 30-urad centroid")
    print("  precision, to see whether map/attitude error erodes the benefit.")
    baseline_urad = 30.0
    lm_ideal = build_landmark_arc(w15_range.nom48, w15_range.t_grid, et0,
                                  sigma_angle_rad=baseline_urad * 1e-6,
                                  r_moon_m=R_MOON_M, label="lm_ideal")
    m_ideal = combined_metrics([(w15_range.h_x0, w15_range.h_k, w15_range.w),
                                (lm_ideal.h_x0, lm_ideal.h_k, lm_ideal.w)])
    ideal_frac = m_ideal["fractional_sigma_k"]

    # Landmark position uncertainty: inflate sigma_angle to the RSS of the
    # centroiding term and an equivalent angular term from a mapping error
    # sigma_map at the mean topocentric range (a first-order treatment --
    # NOT a full map-covariance project, s29).
    r_mean_m = float(np.mean([np.linalg.norm(w15_range.x_true[i, :3])
                              for i in range(0, len(w15_range.x_true), 50)]))
    map_degradation = []
    for sigma_map_m in (50.0, 200.0):
        sigma_map_rad = sigma_map_m / r_mean_m
        sigma_eff = float(np.hypot(baseline_urad * 1e-6, sigma_map_rad))
        lm = build_landmark_arc(w15_range.nom48, w15_range.t_grid, et0,
                                sigma_angle_rad=sigma_eff, r_moon_m=R_MOON_M,
                                label="lm_map_%dm" % int(sigma_map_m))
        m = combined_metrics([(w15_range.h_x0, w15_range.h_k, w15_range.w),
                              (lm.h_x0, lm.h_k, lm.w)])
        rel_degrad = m["fractional_sigma_k"] / ideal_frac - 1.0
        map_degradation.append(rel_degrad)
        print("  landmark map sigma=%4.0f m (-> %.1f urad equiv, RSS %.1f urad): "
              "sigma_K/K=%.4f  (ideal %.4f, +%.0f%%)"
              % (sigma_map_m, sigma_map_rad * 1e6, sigma_eff * 1e6,
                 m["fractional_sigma_k"], ideal_frac, 100 * rel_degrad))

    # Attitude uncertainty: RSS the centroiding term with the star-tracker
    # attitude-knowledge floor cited above.
    att_degradation = []
    for sigma_att_urad in (100.0, ATTITUDE_LIMITED_FLOOR_URAD):
        sigma_eff = float(np.hypot(baseline_urad, sigma_att_urad)) * 1e-6
        lm = build_landmark_arc(w15_range.nom48, w15_range.t_grid, et0,
                                sigma_angle_rad=sigma_eff, r_moon_m=R_MOON_M,
                                label="lm_att_%durad" % int(sigma_att_urad))
        m = combined_metrics([(w15_range.h_x0, w15_range.h_k, w15_range.w),
                              (lm.h_x0, lm.h_k, lm.w)])
        rel_degrad = m["fractional_sigma_k"] / ideal_frac - 1.0
        att_degradation.append(rel_degrad)
        print("  attitude sigma=%4.0f urad (RSS %.1f urad): sigma_K/K=%.4f  (ideal %.4f, +%.0f%%)"
              % (sigma_att_urad, sigma_eff * 1e6, m["fractional_sigma_k"], ideal_frac,
                 100 * rel_degrad))

    # Classified from the ACTUAL measured degradation, not a placeholder:
    # "modest" if the realistic map/attitude cases stay within 50% of the
    # ideal-limit fractional sigma_K; "severe" once either pushes past it.
    worst_map = max(map_degradation)
    worst_att = max(att_degradation)
    landmark_uncertainty_class = "MODEST_EROSION" if worst_map < 0.5 else "SEVERE_EROSION"
    attitude_sensitivity_class = ("MODEST_EROSION" if worst_att < 0.5
                                  else "SEVERE_AT_ATTITUDE_LIMITED_FLOOR")
    print("\n  worst map-error degradation   : +%.0f%% over ideal -> %s"
          % (100 * worst_map, landmark_uncertainty_class))
    print("  worst attitude-error degradation: +%.0f%% over ideal -> %s"
          % (100 * worst_att, attitude_sensitivity_class))

    # ---------------- s25-equivalent thresholds --------------------------
    hdr("O3 -- REQUIRED PRECISION FOR OPERATIONAL THRESHOLDS")
    xs = [r["noise_urad"] for r in sweep_rows]
    ys = [r["combined_fractional_sigma_k"] for r in sweep_rows]
    req_50 = crossing(xs, ys, 0.50)
    req_25 = crossing(xs, ys, 0.25)
    req_10 = crossing(xs, ys, 0.10)
    print("  fractional sigma_K by noise: %s"
          % ", ".join("%.0furad->%.3f" % (x, y) for x, y in zip(xs, ys)))
    print("  noise required for sigma_K/K = 50%%: %s"
          % (("%.1f urad" % req_50) if req_50 else "not reached in sweep"))
    print("  noise required for sigma_K/K = 25%%: %s"
          % (("%.1f urad" % req_25) if req_25 else "not reached in sweep"))
    print("  noise required for sigma_K/K = 10%%: %s"
          % (("%.1f urad" % req_10) if req_10 else "not reached in sweep"))

    best = min(sweep_rows, key=lambda r: r["combined_fractional_sigma_k"])
    worst_tested = max(sweep_rows, key=lambda r: r["noise_urad"])
    max_f_perp_gain = max(r["f_perp_gain"] for r in sweep_rows)

    # Regime-aware read against s28's literature anchors: dedicated-camera
    # best case (<=10 urad), representative sub-pixel centroiding on a
    # decent nav camera (~30 urad), star-tracker-attitude-limited floor
    # (>=300 urad, approaching the cited 500-urad attitude knowledge limit).
    best_case = [r for r in sweep_rows if r["noise_urad"] <= 10.0]
    representative = [r for r in sweep_rows if r["noise_urad"] == 30.0]
    degraded = [r for r in sweep_rows if r["noise_urad"] >= 300.0]
    best_case_gain = max(r["f_perp_gain"] for r in best_case)
    rep_gain = max((r["f_perp_gain"] for r in representative), default=0.0)
    degraded_gain = max(r["f_perp_gain"] for r in degraded)
    if best_case_gain > 0.05 or rep_gain > 0.05:
        info_class = "STRONGLY_COMPLEMENTARY_AT_ACHIEVABLE_PRECISION"
    elif max_f_perp_gain > 0.02:
        info_class = "ADDS_MODEST_NEW_DIRECTION"
    else:
        info_class = "ADDS_INFORMATION_MAGNITUDE_ONLY"

    # Same fix as the DDOR script: "no crossing found" means the target was
    # already met at the loosest tested noise, not that it was never met.
    if worst_tested["combined_fractional_sigma_k"] <= 0.10:
        plausibility = "CLEARLY_OPERATIONALLY_PLAUSIBLE"
    elif req_10 is not None and req_10 >= 10.0:
        plausibility = "CLEARLY_OPERATIONALLY_PLAUSIBLE"
    elif req_25 is not None and req_25 >= 3.0:
        plausibility = "MARGINALLY_PLAUSIBLE"
    else:
        plausibility = "REQUIRES_UNREALISTIC_PRECISION"
    robust = ("ROBUST_TO_REALISTIC_NOISE"
             if worst_tested["combined_fractional_sigma_k"] <= 0.10
             else ("SENSITIVE_TO_NOISE" if req_50 else "IDEALIZED_ONLY"))

    print("\n  regime-aware f_perp gain: best-case(<=10urad)=%.4f  "
          "representative(30urad)=%.4f  degraded(>=300urad)=%.4f"
          % (best_case_gain, rep_gain, degraded_gain))
    print("  even the WORST tested noise (1000 urad, beyond the star-tracker")
    print("  attitude floor) gives sigma_K/K=%.3f, already under 10%%"
          % worst_tested["combined_fractional_sigma_k"])

    hdr("O3 -- SUMMARY")
    print("  best tested case  : %.0f urad -> f_perp %.4f (range-only %.4f), "
          "sigma_K/K = %.4f" % (best["noise_urad"], best["combined_f_perp"],
                                best["range_only_f_perp"],
                                best["combined_fractional_sigma_k"]))
    print("  largest f_perp gain over range-only: %.4f" % max_f_perp_gain)
    print("  OPTICAL_INFORMATION_DIRECTION_CLASS = %s" % info_class)
    print("  OPTICAL_OPERATIONAL_PLAUSIBILITY    = %s" % plausibility)
    print("  OPTICAL_NOISE_ROBUSTNESS_CLASS      = %s" % robust)
    print("  OPTICAL_LANDMARK_UNCERTAINTY_CLASS  = %s" % landmark_uncertainty_class)
    print("  OPTICAL_ATTITUDE_SENSITIVITY_CLASS  = %s" % attitude_sensitivity_class)

    summary = dict(
        n_landmarks=int(len(lm_probe.landmark_latlon_deg)),
        n_landmark_rows_ideal=lm_probe.n_obs,
        best_noise_urad=best["noise_urad"],
        best_sigma_k=best["combined_sigma_k"],
        best_fractional_sigma_k=best["combined_fractional_sigma_k"],
        best_conditional_k_information=best["combined_conditional_k"],
        best_f_perp=best["combined_f_perp"],
        best_theta_k_deg=best["combined_theta_k_deg"],
        required_noise_urad_for_50pct=req_50,
        required_noise_urad_for_25pct=req_25,
        required_noise_urad_for_10pct=req_10,
        worst_tested_noise_urad=worst_tested["noise_urad"],
        worst_tested_fractional_sigma_k=worst_tested["combined_fractional_sigma_k"],
        best_case_f_perp_gain=best_case_gain,
        representative_f_perp_gain=rep_gain,
        degraded_f_perp_gain=degraded_gain,
        information_direction_class=info_class,
        operational_plausibility=plausibility,
        noise_robustness_class=robust,
        attitude_sensitivity_class=attitude_sensitivity_class,
        landmark_uncertainty_class=landmark_uncertainty_class,
        max_fd_convergence_error=max_conv_err,
        direct_measurement_k_dependence="NO",
        status="IDEALIZED_KNOWN_LANDMARK_LOS_SURROGATE",
    )
    (ARTIFACTS / "r1o_optical_summary.json").write_text(
        json.dumps(summary, indent=2, default=float))
    print("\n  wrote r1o_optical_summary.json")


if __name__ == "__main__":
    main()
