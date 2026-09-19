"""PHASE 17-R1O-OPT - production information geometry and error sweeps (s41-s52).

ANALYSIS SPACE ONLY.  Runs the production optical measurement model against
the qualified campaign trajectory and answers the phase's primary question:
does an OD-level lunar landmark image measurement ROTATE K_SRP's information
direction, or only add magnitude?

CADENCE IS AN ANALYSIS-GRID CHOICE, DECLARED HERE AND WHY
----------------------------------------------------------
The 90 s cadence inherited from the radiometric campaign is a poor sampling
grid for a narrow-field camera: the ~34 km FOV footprint at 105 km altitude
is crossed in ~21 s, so at 90 s sampling the ONLY in-FOV frames are the eight
epochs that DEFINE the pre-declared landmarks (mean off-boresight exactly
0.000 deg -- a pure sampling artifact).  A 10 s grid is used for the headline
runs so the production model is not judged on 16 design rows, and the full
90/30/10 s family is reported below so the choice is visible rather than
buried.  This changes the SAMPLING of the trajectory, not the landmark
selection rule (still the pre-declared R1O rule) and not the physics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ARTIFACTS = Path(r"C:/Users/erayh/Documents/Python/Grad/python_port_phase17r0/artifacts")

from phase17_r1m_core import build_range_arc, campaign_epoch  # noqa: E402
from phase17_r1o_core import combined_metrics  # noqa: E402
from phase17_r1o_opt_core import (  # noqa: E402
    build_production_optical_arc, nuisance_solved_k_sigma,
    omitted_variable_k_shift,
)

from lunar_od.lunar_landmark_optical import CameraIntrinsics, DEFAULT_CAMERA  # noqa: E402

W15_ORBITS = 15.0
HEADLINE_CADENCE_S = 10.0
ARCSEC = np.pi / (180.0 * 3600.0)
IFOV_RAD = DEFAULT_CAMERA.pixel_pitch_m / DEFAULT_CAMERA.focal_length_m
K_TRUTH = 0.01

RESULTS: dict = {}


def hdr(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


def main() -> None:
    from lunar_od.spice_loader import load_spice_kernels

    load_spice_kernels(None, clear=True)
    et0, t_orbit = campaign_epoch()

    # ------------------------------------------------------------------
    hdr("s41 -- CADENCE FAMILY (the observation-count confound, made visible)")
    cadence_rows = []
    arcs = {}
    for cad in (90.0, 30.0, HEADLINE_CADENCE_S):
        w15 = build_range_arc(0.0, W15_ORBITS * t_orbit, label=f"W15_cad{cad:.0f}",
                              cadence_s=cad)
        prod = build_production_optical_arc(
            w15.nom48, w15.t_grid, et0, sigma_centroid_px=1e-6 / IFOV_RAD)
        rb = (w15.h_x0, w15.h_k, w15.w)
        m_r = combined_metrics([rb])
        m_c = combined_metrics([rb, (prod.h_x0, prod.h_k, prod.w)])
        arcs[cad] = (w15, rb)
        cadence_rows.append(dict(
            cadence_s=cad, traj_samples=len(w15.t_grid), visible_obs=prod.n_visible_obs,
            mean_off_boresight_deg=prod.mean_off_boresight_deg,
            f_perp_range_only=m_r["orthogonal_fraction"],
            f_perp_combined=m_c["orthogonal_fraction"],
            sigma_k_frac_combined=m_c["fractional_sigma_k"]))
        print("  cadence=%5.0f s  samples=%6d  visible=%4d  mean off-boresight=%6.3f deg"
              "  f_perp %.4f -> %.4f   sigma_K/K=%.4f%%"
              % (cad, len(w15.t_grid), prod.n_visible_obs, prod.mean_off_boresight_deg,
                 m_r["orthogonal_fraction"], m_c["orthogonal_fraction"],
                 100 * m_c["fractional_sigma_k"]))
    print("\n  Observation count rises 6x across this family while f_perp stays flat:")
    print("  that is the signature of information MAGNITUDE accumulating without")
    print("  information DIRECTION rotating.")
    RESULTS["cadence_family"] = cadence_rows

    w15, range_block = arcs[HEADLINE_CADENCE_S]
    m_range_only = combined_metrics([range_block])
    RESULTS["range_only"] = {k: float(v) for k, v in m_range_only.items()}

    # ------------------------------------------------------------------
    hdr("s42/s43 -- PRIMARY INFORMATION GEOMETRY, LITERATURE-ANCHORED NOISE")
    print("  range-only baseline: f_perp=%.6f  theta_K=%.4f deg  sigma_K/K=%.4f%%"
          % (m_range_only["orthogonal_fraction"], m_range_only["theta_k_deg"],
             100 * m_range_only["fractional_sigma_k"]))
    print()
    print("  centroiding sigma | source                       | f_perp   | theta_K  | sigma_K/K")
    noise_cases = [
        (0.1, "dedicated-camera best case"),
        (0.25, "LONEStar best in-flight"),
        (1.0, "LONEStar worst in-flight"),
        (2.5, "Federici et al. 2025 nominal"),
    ]
    noise_rows = []
    for sigma_px, source in noise_cases:
        prod = build_production_optical_arc(
            w15.nom48, w15.t_grid, et0, sigma_centroid_px=sigma_px)
        m = combined_metrics([range_block, (prod.h_x0, prod.h_k, prod.w)])
        noise_rows.append(dict(sigma_px=sigma_px, source=source,
                               f_perp=m["orthogonal_fraction"], theta_k_deg=m["theta_k_deg"],
                               sigma_k_frac=m["fractional_sigma_k"], n_rows=prod.n_rows))
        print("  %6.2f px         | %-28s | %.6f | %7.4f | %8.4f%%"
              % (sigma_px, source, m["orthogonal_fraction"], m["theta_k_deg"],
                 100 * m["fractional_sigma_k"]))
    RESULTS["noise_sweep"] = noise_rows

    # An idealized, physically unreachable precision, to show the LIMIT.
    prod_ideal = build_production_optical_arc(
        w15.nom48, w15.t_grid, et0, sigma_centroid_px=1e-3)
    m_ideal = combined_metrics([range_block, (prod_ideal.h_x0, prod_ideal.h_k, prod_ideal.w)])
    print("\n  0.001 px (unreachable, shows the asymptote): f_perp=%.6f  sigma_K/K=%.6f%%"
          % (m_ideal["orthogonal_fraction"], 100 * m_ideal["fractional_sigma_k"]))
    RESULTS["unreachable_limit"] = {k: float(v) for k, v in m_ideal.items()}

    # OPTICAL BLOCK ALONE.  f_perp of the combined system can exceed BOTH
    # constituent blocks (a genuine mixture effect), so the constituent value
    # must be reported too or the combined number cannot be interpreted.
    prod_solo = build_production_optical_arc(
        w15.nom48, w15.t_grid, et0, sigma_centroid_px=0.25)
    m_solo = combined_metrics([(prod_solo.h_x0, prod_solo.h_k, prod_solo.w)])
    print()
    print("  OPTICAL BLOCK ALONE (0.25 px, %d rows): f_perp=%.6f  theta_K=%.4f deg"
          % (prod_solo.n_rows, m_solo["orthogonal_fraction"], m_solo["theta_k_deg"]))
    RESULTS["optical_only"] = {k: float(v) for k, v in m_solo.items()}

    f_perp_best = max([r["f_perp"] for r in noise_rows]
                      + [m_ideal["orthogonal_fraction"]])
    sigma_best = min([r["sigma_k_frac"] for r in noise_rows])
    improvement = m_range_only["fractional_sigma_k"] / sigma_best

    print()
    print("  SUMMARY OF THE PRIMARY QUESTION")
    print("    f_perp   range-only                     %.6f" % m_range_only["orthogonal_fraction"])
    print("    f_perp   optical block alone            %.6f" % m_solo["orthogonal_fraction"])
    print("    f_perp   best combined (any noise case) %.6f" % f_perp_best)
    print("    sigma_K/K range-only                    %.4f%%"
          % (100 * m_range_only["fractional_sigma_k"]))
    print("    sigma_K/K best combined                 %.4f%%" % (100 * sigma_best))
    print("    operational improvement factor          %.3fx" % improvement)
    print()
    print("    R1O published claim, for comparison:    f_perp 0.8757, sigma_K/K 0.19%")
    print("      (shown NOT as a target but because this phase exists to test it;")
    print("       s40 establishes that it rests on a transposed Phi)")
    print()
    # The classification threshold is stated explicitly rather than applied
    # silently.  R1O decision case for this observable was a STEP CHANGE in
    # K identifiability (an 18x sigma_K improvement and f_perp ~0.88, i.e. K
    # nearly orthogonal to the state subspace).  The production model is
    # measured against that decision question, not against an arbitrary cut.
    step_change = improvement >= 3.0 and f_perp_best >= 0.60
    marginal = (not step_change) and (
        f_perp_best > 1.05 * m_range_only["orthogonal_fraction"] or improvement > 1.10)
    classification = ("DIRECTION_ROTATING_STEP_CHANGE" if step_change
                      else "MARGINAL_DIRECTION_GAIN" if marginal
                      else "MAGNITUDE_ONLY")
    print("  OPTICAL_INFORMATION_CLASS = %s" % classification)
    print("    criterion: a STEP CHANGE requires >=3x sigma_K improvement AND")
    print("    f_perp >= 0.60; a MARGINAL gain is any measurable improvement")
    print("    below that; MAGNITUDE_ONLY is no directional gain at all.")
    RESULTS["information_class"] = classification
    RESULTS["improvement_factor"] = float(improvement)
    RESULTS["f_perp_best_combined"] = float(f_perp_best)

    # ------------------------------------------------------------------
    hdr("s44/s45 -- RANDOM ATTITUDE KNOWLEDGE ERROR (literature-anchored sweep)")
    print("  Random per-observation attitude error IS measurement noise, so it")
    print("  enters the weight in quadrature (pixel domain).  Bias does NOT --")
    print("  see s46 for the two distinct bias formulas.")
    print()
    print("  attitude sigma      | px equiv | source                    | f_perp   | sigma_K/K")
    att_cases = [
        (8.0 * ARCSEC, "BCT XACT in-orbit (MinXSS)"),
        (10.0 * ARCSEC, "Apollo sextant RMS sighting"),
        (40.0 * ARCSEC, "compact tracker pixel scale"),
        (120.0 * ARCSEC, "coarse tracker, 3-sigma roll"),
        (500e-6, "R1O conservative 0.5 mrad floor"),
    ]
    att_rows = []
    for sigma_att, source in att_cases:
        prod = build_production_optical_arc(
            w15.nom48, w15.t_grid, et0, sigma_centroid_px=0.25,
            attitude_random_sigma_rad=sigma_att)
        m = combined_metrics([range_block, (prod.h_x0, prod.h_k, prod.w)])
        att_rows.append(dict(sigma_att_rad=sigma_att, source=source,
                             sigma_px_effective=prod.sigma_px_effective,
                             f_perp=m["orthogonal_fraction"],
                             sigma_k_frac=m["fractional_sigma_k"]))
        print("  %7.2f arcsec       | %8.3f | %-25s | %.6f | %8.4f%%"
              % (sigma_att / ARCSEC, prod.sigma_px_effective, source,
                 m["orthogonal_fraction"], 100 * m["fractional_sigma_k"]))
    RESULTS["attitude_random_sweep"] = att_rows

    # ------------------------------------------------------------------
    hdr("s46 -- ATTITUDE / BORESIGHT BIAS: THE TWO DISTINCT FORMULAS")
    print("  R1O-D established that an UNMODELED coherent error and a SOLVED-FOR")
    print("  nuisance parameter are NOT the same question and must not share a")
    print("  formula.  Both are computed here for the same physical bias.")
    prod_b = build_production_optical_arc(
        w15.nom48, w15.t_grid, et0, sigma_centroid_px=0.25)
    h_x0_all = np.vstack([range_block[0], prod_b.h_x0])
    h_k_all = np.concatenate([range_block[1], prod_b.h_k])
    w_all = np.concatenate([range_block[2], prod_b.w])
    h_att_all = np.vstack([np.zeros((range_block[0].shape[0], 3)), prod_b.h_attitude_bias])

    bias_rows = []
    print()
    print("  bias magnitude  | Case B: K point shift      | Case C: sigma_K/K when solved")
    for bias_arcsec in (1.0, 8.0, 40.0, 120.0):
        b = np.full(3, bias_arcsec * ARCSEC / np.sqrt(3.0))  # isotropic, |b| = bias_arcsec
        shift = omitted_variable_k_shift(h_x0_all, h_k_all, w_all, h_att_all, b)
        sigma_solved = nuisance_solved_k_sigma(
            h_x0_all, h_k_all, w_all, h_att_all,
            np.full(3, bias_arcsec * ARCSEC))
        bias_rows.append(dict(bias_arcsec=bias_arcsec, k_shift=shift,
                              k_shift_frac=shift / K_TRUTH,
                              sigma_k_frac_solved=sigma_solved / K_TRUTH))
        print("  %6.1f arcsec    | %+.6e (%8.4f%% of K) | %8.4f%%"
              % (bias_arcsec, shift, 100 * shift / K_TRUTH, 100 * sigma_solved / K_TRUTH))
    RESULTS["attitude_bias"] = bias_rows

    # ------------------------------------------------------------------
    hdr("s47/s48 -- LANDMARK CATALOG (MAP) ERROR: RANDOM AND COHERENT")
    print("  RANDOM per-observation map error (identification jitter):")
    map_rows = []
    for map_m in (50.0, 200.0):
        prod = build_production_optical_arc(
            w15.nom48, w15.t_grid, et0, sigma_centroid_px=0.25,
            map_random_sigma_m=map_m)
        m = combined_metrics([range_block, (prod.h_x0, prod.h_k, prod.w)])
        map_rows.append(dict(map_sigma_m=map_m, sigma_px_effective=prod.sigma_px_effective,
                             f_perp=m["orthogonal_fraction"],
                             sigma_k_frac=m["fractional_sigma_k"]))
        print("    %6.1f m -> %8.3f px equivalent | f_perp=%.6f | sigma_K/K=%8.4f%%"
              % (map_m, prod.sigma_px_effective, m["orthogonal_fraction"],
                 100 * m["fractional_sigma_k"]))
    RESULTS["map_random_sweep"] = map_rows

    print("\n  COHERENT per-landmark catalog error (the physically correct model:")
    print("  a catalog entry is wrong the SAME way in every image of it):")
    n_lm = prod_b.h_landmark_bias.shape[1]
    h_lm_all = np.vstack([np.zeros((range_block[0].shape[0], n_lm)), prod_b.h_landmark_bias])
    map_bias_rows = []
    print("  map bias    | Case B: K point shift       | Case C: sigma_K/K when solved")
    rng = np.random.default_rng(20260919)
    direction = rng.normal(size=n_lm)
    direction /= np.linalg.norm(direction) / np.sqrt(n_lm / 3.0)  # per-landmark |b| = 1 m
    for map_m in (50.0, 200.0):
        b = direction * map_m
        shift = omitted_variable_k_shift(h_x0_all, h_k_all, w_all, h_lm_all, b)
        sigma_solved = nuisance_solved_k_sigma(
            h_x0_all, h_k_all, w_all, h_lm_all, np.full(n_lm, map_m))
        map_bias_rows.append(dict(map_sigma_m=map_m, k_shift=shift,
                                  k_shift_frac=shift / K_TRUTH,
                                  sigma_k_frac_solved=sigma_solved / K_TRUTH))
        print("  %6.1f m     | %+.6e (%8.4f%% of K) | %8.4f%%"
              % (map_m, shift, 100 * shift / K_TRUTH, 100 * sigma_solved / K_TRUTH))
    RESULTS["map_bias"] = map_bias_rows

    # ------------------------------------------------------------------
    hdr("s49 -- CAMERA CALIBRATION PERTURBATION (focal length)")
    print("  A focal-length calibration error rescales the image plane.  Evaluated")
    print("  as a MODEL perturbation (the measurement is formed with the wrong f),")
    print("  not as noise.")
    cal_rows = []
    for rel_err in (1e-4, 1e-3, 1e-2):
        cam = CameraIntrinsics(
            focal_length_m=DEFAULT_CAMERA.focal_length_m * (1.0 + rel_err),
            pixel_pitch_m=DEFAULT_CAMERA.pixel_pitch_m,
            sensor_width_px=DEFAULT_CAMERA.sensor_width_px,
            sensor_height_px=DEFAULT_CAMERA.sensor_height_px)
        prod = build_production_optical_arc(
            w15.nom48, w15.t_grid, et0, sigma_centroid_px=0.25, camera=cam)
        m = combined_metrics([range_block, (prod.h_x0, prod.h_k, prod.w)])
        cal_rows.append(dict(focal_rel_error=rel_err, f_perp=m["orthogonal_fraction"],
                             sigma_k_frac=m["fractional_sigma_k"], n_rows=prod.n_rows))
        print("    df/f=%.0e | rows=%4d | f_perp=%.6f | sigma_K/K=%8.4f%%"
              % (rel_err, prod.n_rows, m["orthogonal_fraction"],
                 100 * m["fractional_sigma_k"]))
    RESULTS["camera_calibration"] = cal_rows

    # ------------------------------------------------------------------
    hdr("s50 -- COMBINED REALISTIC CASE (pre-declared component values)")
    print("  Components declared from the literature contract BEFORE running:")
    print("    centroiding        0.25 px  (LONEStar best in-flight)")
    print("    attitude knowledge 10 arcsec (Apollo sextant / good compact tracker)")
    print("    catalog, random    50 m")
    prod_real = build_production_optical_arc(
        w15.nom48, w15.t_grid, et0, sigma_centroid_px=0.25,
        attitude_random_sigma_rad=10.0 * ARCSEC, map_random_sigma_m=50.0,
        label="combined_realistic")
    m_real = combined_metrics([range_block, (prod_real.h_x0, prod_real.h_k, prod_real.w)])
    print("\n  effective sigma = %.4f px" % prod_real.sigma_px_effective)
    print("  f_perp=%.6f  theta_K=%.4f deg  sigma_K/K=%.4f%%"
          % (m_real["orthogonal_fraction"], m_real["theta_k_deg"],
             100 * m_real["fractional_sigma_k"]))
    print("  range-only for comparison:      sigma_K/K=%.4f%%"
          % (100 * m_range_only["fractional_sigma_k"]))
    RESULTS["combined_realistic"] = {k: float(v) for k, v in m_real.items()}
    RESULTS["combined_realistic"]["sigma_px_effective"] = float(prod_real.sigma_px_effective)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "r1o_opt_campaign.json").write_text(json.dumps(RESULTS, indent=2, default=float))
    print("\n  wrote artifacts/r1o_opt_campaign.json")


if __name__ == "__main__":
    main()
