"""PHASE 17-R1O - comparison table, selection matrix, config, figures.

ANALYSIS SPACE ONLY.  Assembles results already computed by the other R1O
scripts (reads their JSON/CSV artifacts; does not recompute).
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from lunar_od.constants import R_MOON_M  # noqa: E402
from phase17_r1m_core import build_range_arc, campaign_epoch  # noqa: E402
from phase17_r1o_core import (  # noqa: E402
    build_ddor_arc, build_landmark_arc, information_matrix, scale_matrix, spectrum,
)

ARTIFACTS = REPO / "artifacts"


def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True,
                          text=True).stdout.strip()


def load(name):
    return json.loads((ARTIFACTS / name).read_text())


def hdr(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


def main() -> None:
    baseline = load("r1o_baseline.json")
    ddor = load("r1o_ddor_summary.json")
    optical = load("r1o_optical_summary.json")
    celestial = load("r1o_celestial_summary.json")
    crosscheck = load("r1o_crosscheck_summary.json")

    g1 = baseline["g1_baseline"]
    o0 = baseline["o0_range_w15"]
    o1 = baseline["o1_range_doppler_carried_forward"]

    # ---------------- s35 required primary comparison table -------------
    hdr("s35 -- PRIMARY COMPARISON TABLE")
    rows = [
        dict(case="G1 (continuity control)", observable_set="range",
            added_obs=0, conditional_k_info=g1["conditional_k_information"],
            f_perp=g1["orthogonal_fraction"], theta_k_deg=g1["theta_k_deg"],
            sigma_k_over_k=g1["fractional_sigma_k"],
            weakest_k=g1["weakest_mode_k_component"],
            info_direction_class="ADDS_INFORMATION_MAGNITUDE_ONLY (R1M baseline)"),
        dict(case="O0 (range, W15)", observable_set="range",
            added_obs=o0["n_obs"], conditional_k_info=o0["conditional_k_information"],
            f_perp=o0["orthogonal_fraction"], theta_k_deg=o0["theta_k_deg"],
            sigma_k_over_k=o0["fractional_sigma_k"],
            weakest_k=o0["weakest_mode_k_component"],
            info_direction_class="CONTROL (baseline for all other rows)"),
        dict(case="O1 (range+Doppler, R1G carried fwd)", observable_set="range+doppler",
            added_obs="n/a", conditional_k_info=o1["range_doppler_i_kk"],
            f_perp=float("nan"), theta_k_deg=float("nan"),
            sigma_k_over_k=float("nan"), weakest_k=o1["weakest_k_component"],
            info_direction_class=o1["classification"]),
        dict(case="O2 DDOR best-in-class (2 nrad)", observable_set="range+ddor",
            added_obs=ddor["n_dual_visible_observations"],
            conditional_k_info=ddor["best_conditional_k_information"],
            f_perp=ddor["best_f_perp"], theta_k_deg=ddor["best_theta_k_deg"],
            sigma_k_over_k=ddor["best_fractional_sigma_k"],
            weakest_k=float("nan"), info_direction_class=ddor["information_direction_class"]),
        dict(case="O2 DDOR degraded (300 nrad)", observable_set="range+ddor",
            added_obs=ddor["n_dual_visible_observations"],
            conditional_k_info=float("nan"),
            f_perp=float("nan"), theta_k_deg=float("nan"),
            sigma_k_over_k=ddor["worst_tested_fractional_sigma_k"],
            weakest_k=float("nan"), info_direction_class="magnitude-only regime"),
        dict(case="O3 landmark best-case (1 urad)", observable_set="range+landmark",
            added_obs=optical["n_landmark_rows_ideal"],
            conditional_k_info=optical["best_conditional_k_information"],
            f_perp=optical["best_f_perp"], theta_k_deg=optical["best_theta_k_deg"],
            sigma_k_over_k=optical["best_fractional_sigma_k"],
            weakest_k=float("nan"), info_direction_class=optical["information_direction_class"]),
        dict(case="O3 landmark degraded (attitude-limited)", observable_set="range+landmark",
            added_obs=optical["n_landmark_rows_ideal"],
            conditional_k_info=float("nan"), f_perp=float("nan"), theta_k_deg=float("nan"),
            sigma_k_over_k=optical["worst_tested_fractional_sigma_k"],
            weakest_k=float("nan"), info_direction_class="magnitude-only regime"),
        dict(case="O4 celestial LOS (Earth center)", observable_set="range+earth_los",
            added_obs="n/a", conditional_k_info=float("nan"), f_perp=float("nan"),
            theta_k_deg=float("nan"), sigma_k_over_k=float("nan"),
            weakest_k=float("nan"), info_direction_class=celestial["status"]),
    ]
    for r in rows:
        print("  %-42s obs=%s  I_K|x=%s  f_perp=%s  sigma_K/K=%s"
              % (r["case"], r["added_obs"], r["conditional_k_info"], r["f_perp"],
                 r["sigma_k_over_k"]))
    with (ARTIFACTS / "r1o_observable_comparison.csv").open("w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w_.writeheader(); w_.writerows(rows)
    print("\n  wrote r1o_observable_comparison.csv")

    # ---------------- s44/s46 selection ----------------------------------
    hdr("s44/s46 -- CANDIDATE SELECTION AND DECISION MATRIX")
    matrix = [
        dict(candidate="DDOR-like plane-of-sky angle",
            scientific_complementarity="STRONG (f_perp 0.25->0.44 @ 2 nrad)",
            required_precision="2-10 nrad (achievable: DSN quotes 1-3 nrad best-in-class)",
            ground_space_hw="existing DSN antennas; needs 2-station scheduled pass "
                            "(no new spacecraft hardware)",
            operations_burden="MODERATE: dedicated dual-baseline scheduling, "
                              "media/clock calibration in a real implementation",
            model_complexity="LOW for this surrogate; MODERATE for production "
                             "(quasar calibration, tone/phase processing)",
            likely_systematics="media calibration, station baseline knowledge, "
                               "clock/instrumental delay (s40)",
            repo_implementation_effort="MODERATE: needs a new differenced-range "
                                       "measurement type + dual-station scheduling logic",
            academic_maturity="Operational at DSN for decades",
            flight_heritage="Extensive (Voyager, Cassini, MSL, many since 1980s)"),
        dict(candidate="Lunar landmark LOS",
            scientific_complementarity="STRONGEST TESTED (f_perp 0.25->0.88 @ 1 urad "
                                       "ideal; ->0.69 @ 30 urad with 16 landmarks)",
            required_precision="10-30 urad ideal; realistic map/attitude error erodes "
                               "this toward the ~9% floor (SEVERE_EROSION at 200 m map "
                               "or 500 urad attitude)",
            ground_space_hw="REQUIRES a camera + attitude determination + pre-built "
                            "landmark map -- new spacecraft hardware unless already "
                            "carried for another purpose",
            operations_burden="HIGH: image downlink or onboard processing, landmark "
                              "map preparation and maintenance",
            model_complexity="LOW for this surrogate; HIGH for production (crater "
                             "detection/ID, camera calibration, attitude coupling)",
            likely_systematics="camera calibration, attitude bias, centroid bias, "
                               "landmark-map bias (s40) -- and this phase measured "
                               "that attitude/map error erodes MOST of the benefit "
                               "at realistic precision",
            repo_implementation_effort="HIGH: no existing camera/attitude/landmark "
                                       "infrastructure in this repository at all",
            academic_maturity="Active research area (LONEStar 2023-2024, several "
                              "JGCD/Aerospace crater-nav papers)",
            flight_heritage="Limited (LONEStar demonstrated star/planet imaging, "
                            "not lunar crater navigation; crater nav largely "
                            "descent/landing-phase heritage, not orbital OD)"),
    ]
    for m in matrix:
        print("\n  %s" % m["candidate"])
        for k, v in m.items():
            if k == "candidate":
                continue
            print("    %-28s %s" % (k, v))
    with (ARTIFACTS / "r1o_selected_candidates.csv").open("w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(matrix[0].keys()))
        w_.writeheader(); w_.writerows(matrix)
    print("\n  wrote r1o_selected_candidates.csv")

    # Ranking per s64: information complementarity, required precision,
    # mission realism, implementation complexity, flight heritage -- no
    # arbitrary weighted score (s46), a reasoned ranking instead.
    print("\n  RANKING (s64 -- do not implement both simultaneously):")
    print("  1st: DDOR-like plane-of-sky angle")
    print("       - comparable information-direction gain at REALISTIC published")
    print("         DSN precision (already demonstrated flight capability)")
    print("       - needs NO new spacecraft hardware, only ground-segment scheduling")
    print("       - decades of flight heritage; systematics well characterized")
    print("       - the repository has no differenced-range measurement type yet,")
    print("         but building one is far less new infrastructure than a full")
    print("         camera/attitude/landmark-map optical navigation stack")
    print("  2nd: Lunar landmark LOS")
    print("       - the LARGEST raw information-direction gain of any candidate")
    print("         tested, but that gain is the one MOST eroded by realistic")
    print("         systematic error sources this phase explicitly measured")
    print("         (landmark map + attitude, s29/s30) -- and this repository has")
    print("         no camera, attitude, or landmark infrastructure to build on")

    # ---------------- config ----------------------------------------------
    cfg = dict(
        phase="PHASE 17-R1O",
        title="ADDITIONAL OBSERVABLE FEASIBILITY FOR K_SRP IDENTIFIABILITY",
        scope="ANALYSIS_ONLY_NO_PRODUCTION_OBSERVABLE",
        git=dict(head=git("rev-parse", "HEAD"), branch=git("rev-parse", "--abbrev-ref", "HEAD"),
                main=git("rev-parse", "main"), origin_main=git("rev-parse", "origin/main"),
                r1cov_final_head="75d1b4699d4cfdf220df9faf7b0a90c989b30c1b"),
        window=dict(g1_continuity_orbits=2.0,
                   w15_common_campaign_orbits=15.0,
                   w15_cadence_s=90.0,
                   window_selection_probe="artifacts/r1o_window_probe.csv"),
        ddor=dict(baseline_pair=["Goldstone DSN", "Canberra DSN"],
                  min_elevation_deg=10.0,
                  noise_sweep_nrad=[1, 2, 3, 5, 10, 20, 30, 50, 100, 300],
                  literature_citations=[
                      "Bell et al., Delta-DOR: The One-Nanoradian Navigation "
                      "Measurement System of the DSN, JPL IPN PR 42-193 (2013)",
                      "DSN Navigation System Accuracy, DESCANSO"]),
        landmark=dict(n_landmarks=8, epoch_fractions="0,0.15,...,1.0 of arc",
                     max_offnadir_deg=30.0,
                     noise_sweep_urad=[1, 3, 10, 30, 100, 300, 500, 1000],
                     literature_citations=[
                         "LONEStar: The Lunar Flashlight Optical Navigation "
                         "Experiment (2024), IFOV ~36.6 arcsec/px, attitude "
                         "stability ~15-20 arcsec/5s, LOS error 0.25-1 px",
                         "Optical Camera Characterization for Feature-Based "
                         "Navigation in Lunar Orbit (Aerospace 2025), star-tracker "
                         "attitude ~0.5 mrad 1-sigma"]),
        methodology="finite-difference position Jacobian (Richardson-verified), "
                   "chained through the qualified Phi/S_K columns of nom48; "
                   "covariance via R1COV's qualified QR square-root path, never "
                   "the floored normal-matrix inverse",
    )
    (ARTIFACTS / "r1o_config.json").write_text(json.dumps(cfg, indent=2, default=float))
    print("\n  wrote r1o_config.json")

    # ---------------- figures (s53) ---------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ddor_sweep = list(csv.DictReader((ARTIFACTS / "r1o_ddor_sweep.csv").open()))
    opt_sweep = list(csv.DictReader((ARTIFACTS / "r1o_optical_landmark_sweep.csv").open()))

    # 1/2/3: sigma_K/K, f_perp, theta_K vs observable type
    labels = ["G1", "O0\nrange", "O2 DDOR\n(2nrad)", "O3 landmark\n(10urad)"]
    sig = [g1["fractional_sigma_k"], o0["fractional_sigma_k"],
          float(next(r["combined_fractional_sigma_k"] for r in ddor_sweep
                    if r["noise_nrad"] == "2.0")),
          float(next(r["combined_fractional_sigma_k"] for r in opt_sweep
                    if r["noise_urad"] == "10.0"))]
    fperp = [g1["orthogonal_fraction"], o0["orthogonal_fraction"],
            float(next(r["combined_f_perp"] for r in ddor_sweep if r["noise_nrad"] == "2.0")),
            float(next(r["combined_f_perp"] for r in opt_sweep if r["noise_urad"] == "10.0"))]
    theta = [g1["theta_k_deg"], o0["theta_k_deg"],
            float(next(r["combined_theta_k_deg"] for r in ddor_sweep if r["noise_nrad"] == "2.0")),
            float(next(r["combined_theta_k_deg"] for r in opt_sweep if r["noise_urad"] == "10.0"))]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    x = np.arange(len(labels))
    axes[0].bar(x, sig); axes[0].set_yscale("log"); axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8); axes[0].set_ylabel("sigma_K / K_truth")
    axes[0].set_title("Fractional K uncertainty")
    axes[1].bar(x, fperp); axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("f_perp"); axes[1].set_ylim(0, 1); axes[1].set_title("Orthogonal K fraction")
    axes[2].bar(x, theta); axes[2].set_xticks(x); axes[2].set_xticklabels(labels, fontsize=8)
    axes[2].set_ylabel("theta_K [deg]"); axes[2].set_title("Principal angle")
    fig.suptitle("R1O: sigma_K/K, f_perp, theta_K by observable type")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "r1o_metrics_by_observable.png", dpi=110); plt.close(fig)

    # 4: conditional K info vs precision (both DDOR and landmark)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.loglog([float(r["noise_nrad"]) for r in ddor_sweep],
             [float(r["combined_conditional_k"]) for r in ddor_sweep], "o-",
             label="DDOR (nrad)")
    ax.set_xlabel("DDOR noise [nrad]"); ax.set_ylabel("conditional K information")
    ax.set_title("R1O: conditional K info vs DDOR precision")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(ARTIFACTS / "r1o_conditional_k_vs_precision.png", dpi=110); plt.close(fig)

    # 5: DDOR sigma_K vs noise
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.loglog([float(r["noise_nrad"]) for r in ddor_sweep],
             [float(r["combined_fractional_sigma_k"]) for r in ddor_sweep], "o-")
    ax.axhline(0.10, ls="--", c="r", label="10% threshold")
    ax.set_xlabel("DDOR angular noise [nrad]"); ax.set_ylabel("sigma_K / K_truth")
    ax.set_title("R1O: DDOR-like sigma_K vs angular noise")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(ARTIFACTS / "r1o_ddor_sigma_vs_noise.png", dpi=110); plt.close(fig)

    # 6: optical sigma_K vs noise
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.loglog([float(r["noise_urad"]) for r in opt_sweep],
             [float(r["combined_fractional_sigma_k"]) for r in opt_sweep], "o-")
    ax.axhline(0.10, ls="--", c="r", label="10% threshold")
    ax.set_xlabel("landmark LOS angular noise [urad]"); ax.set_ylabel("sigma_K / K_truth")
    ax.set_title("R1O: landmark LOS sigma_K vs angular noise")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(ARTIFACTS / "r1o_optical_sigma_vs_noise.png", dpi=110); plt.close(fig)

    # 7: magnitude gain vs direction gain
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for sweep, name, key in ((ddor_sweep, "DDOR", "noise_nrad"),
                             (opt_sweep, "Landmark", "noise_urad")):
        mag = [float(r["information_gain_ratio"]) for r in sweep]
        direc = [float(r["f_perp_gain"]) for r in sweep]
        ax.plot(mag, direc, "o-", label=name)
    ax.set_xscale("log")
    ax.set_xlabel("information MAGNITUDE gain (I_K|x ratio)")
    ax.set_ylabel("information DIRECTION gain (f_perp increase)")
    ax.set_title("R1O: magnitude gain vs direction gain (s18/s70)")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(ARTIFACTS / "r1o_magnitude_vs_direction.png", dpi=110); plt.close(fig)

    # 8: best-candidate singular spectra vs baseline (recomputed directly,
    # not read from a summary field, since none of the other artifacts carry
    # the full 7-value spectrum).
    from lunar_od.spice_loader import load_spice_kernels

    load_spice_kernels(None, clear=True)
    et0, t_orbit = campaign_epoch()
    w15 = build_range_arc(0.0, 15.0 * t_orbit, label="W15_spec",
                          station_filter=("Goldstone DSN", "Madrid DSN", "Canberra DSN"),
                          cadence_s=90.0)
    scale = scale_matrix()

    def spec_of(h_x0, h_k, w):
        info = information_matrix(np.hstack([h_x0, h_k[:, None]]), w, scale)
        return spectrum(info)["singular_values"]

    spec_baseline = spec_of(w15.h_x0, w15.h_k, w15.w)
    ddor2 = build_ddor_arc(w15.nom48, w15.t_grid, et0,
                          ("Goldstone DSN", "Canberra DSN"), sigma_angle_rad=2e-9)
    spec_ddor = spec_of(np.vstack([w15.h_x0, ddor2.h_x0]),
                        np.concatenate([w15.h_k, ddor2.h_k]),
                        np.concatenate([w15.w, ddor2.w]))
    lm10 = build_landmark_arc(w15.nom48, w15.t_grid, et0, sigma_angle_rad=10e-6,
                              r_moon_m=R_MOON_M)
    spec_lm = spec_of(np.vstack([w15.h_x0, lm10.h_x0]),
                      np.concatenate([w15.h_k, lm10.h_k]),
                      np.concatenate([w15.w, lm10.w]))

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for spec, label, mk in ((spec_baseline, "O0 range (baseline)", "o"),
                            (spec_ddor, "O2 range+DDOR (2 nrad)", "s"),
                            (spec_lm, "O3 range+landmark (10 urad)", "^")):
        sv = np.sort(np.asarray(spec))[::-1]
        ax.semilogy(range(1, 8), sv / sv[0], mk + "-", label=label)
    ax.axhline(7 * np.finfo(float).eps, ls=":", c="r", label="rank threshold n*eps")
    ax.set_xlabel("singular value index"); ax.set_ylabel("sigma_i / sigma_max")
    ax.set_title("R1O: singular spectra, baseline vs best-tested candidates")
    ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(ARTIFACTS / "r1o_singular_spectra.png", dpi=110); plt.close(fig)

    print("  wrote 8 figures")


if __name__ == "__main__":
    main()
