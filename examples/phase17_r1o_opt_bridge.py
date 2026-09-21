"""PHASE 17-R1O-OPT - the R1O surrogate bridge (s39), and the defect it exposed.

ANALYSIS SPACE ONLY.  Answers the question this phase exists to answer: does
Phase 17-R1O's headline landmark-LOS result survive a production measurement
model?  The bridge is built by CONSTRUCTION, not inference -- the surrogate
and the production model are evaluated on the IDENTICAL trajectory, the
IDENTICAL pre-declared landmarks, and the IDENTICAL epochs, and the
differences between them are introduced one at a time.

THE STATE-TRANSITION-MATRIX TRANSPOSITION (found here, reported in s40)
-----------------------------------------------------------------------
While qualifying the production Jacobian against re-propagated trajectories
(`phase17_r1o_opt_jacobian_and_k.py`), the analytic state Jacobian
disagreed with the end-to-end finite difference by factors of 1e2-1e8.  The
cause is NOT in the production model: `lunar_od.dynamics` packs the 6x6 STM
column-major (`np.eye(6).reshape(-1, order="F")`) and every PRODUCTION
consumer unflattens it with `order="F"`, but Phase 17-R1O's analysis helper
`phase17_r1o_core.chain_to_augmented_columns` unflattens with NumPy's
default C order -- i.e. it uses Phi^T.  Phi is not symmetric, so every
surrogate state-design row R1O built is wrong; the K column (built from the
plain 6-vector S_K, which needs no reshape) is unaffected.

This script therefore reports the surrogate BOTH ways: exactly as R1O
published it, and with the transposition corrected.  No attempt is made to
reproduce any particular number -- the production model is authoritative and
the corrected surrogate is reported wherever it lands (R1O-OPT s3).

SUPERSEDED BY PHASE 17-R1O-R -- THIS SCRIPT NO LONGER REPRODUCES ITS OWN s40
-----------------------------------------------------------------------------
Phase 17-R1O-R repaired `chain_to_augmented_columns` (commit 75ba49a).  STEP 1
below calls `build_landmark_arc`, which now unflattens Phi correctly, so it no
longer reproduces the f_perp ~ 0.8766 this script reported and its
"as published" label is no longer accurate: STEP 1 and STEP 2 now agree.

To reproduce this script's published output, check out commit 887efa6 or
earlier.  For the authoritative historical-vs-corrected comparison, use
`examples/phase17_r1o_r_requalification.py`, which reproduces the defective
path explicitly (by feeding the repaired helper vec_F(Phi^T)) instead of
depending on the helper still being broken, and which runs on R1O's own
three-station published configuration rather than this script's single-station
baseline.
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
from phase17_r1o_core import (  # noqa: E402
    build_landmark_arc, combined_metrics, position_jacobian_fd,
    _landmark_latlon_from_nadir,
)
from phase17_r1o_opt_core import (  # noqa: E402
    PREDECLARED_EPOCH_FRACTIONS, build_production_optical_arc,
)

from lunar_od.constants import R_MOON_M  # noqa: E402
from lunar_od.lunar_landmark_optical import DEFAULT_CAMERA  # noqa: E402

W15_ORBITS = 15.0


def hdr(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


def corrected_landmark_arc(nom48, t_grid_s, et0, *, sigma_angle_rad, r_moon_m,
                           epoch_fractions=PREDECLARED_EPOCH_FRACTIONS,
                           max_offnadir_deg=30.0):
    """R1O's `build_landmark_arc`, identical in physics, with the ONE change
    of unflattening Phi column-major.  Deliberately duplicated here rather
    than editing R1O's committed file: this phase reports what R1O published
    AND what it should have been, and rewriting the published script in place
    would destroy the first of those.
    """
    from lunar_od.lunar_frames import moon_pa_de440_rotation_at_et

    latlon = _landmark_latlon_from_nadir(nom48, t_grid_s, et0, epoch_fractions)
    lat_r, lon_r = np.radians(latlon[:, 0]), np.radians(latlon[:, 1])
    landmarks_pa = np.stack([
        r_moon_m * np.cos(lat_r) * np.cos(lon_r),
        r_moon_m * np.cos(lat_r) * np.sin(lon_r),
        r_moon_m * np.sin(lat_r),
    ], axis=1)

    rows_x0, rows_k = [], []
    for i, t_abs in enumerate(t_grid_s):
        r_sc = nom48[i, :3]
        nadir_hat = -r_sc / np.linalg.norm(r_sc)
        ref = np.array([0.0, 0.0, 1.0]) if abs(nadir_hat[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(nadir_hat, ref)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(nadir_hat, e1)

        c_pa_t = moon_pa_de440_rotation_at_et(et0 + t_abs).T
        phi = nom48[i, 6:42].reshape((6, 6), order="F")      # <-- the one change
        s_k = nom48[i, 42:48]
        for li in range(landmarks_pa.shape[0]):
            r_lm_j2000 = c_pa_t @ landmarks_pa[li]
            rho = r_lm_j2000 - r_sc
            rho_hat = rho / np.linalg.norm(rho)
            offnadir_deg = np.degrees(np.arccos(np.clip(np.dot(rho_hat, nadir_hat), -1.0, 1.0)))
            if offnadir_deg > max_offnadir_deg:
                continue

            def g_fn(r, _rlm=r_lm_j2000, _e1=e1, _e2=e2):
                rho_ = _rlm - r
                rho_h = rho_ / np.linalg.norm(rho_)
                return np.array([np.arcsin(np.clip(np.dot(rho_h, _e1), -1.0, 1.0)),
                                 np.arcsin(np.clip(np.dot(rho_h, _e2), -1.0, 1.0))])

            dg_dr, _ = position_jacobian_fd(g_fn, r_sc)
            h_x0 = dg_dr @ phi[:3, :]
            h_k = dg_dr @ s_k[:3]
            for row in range(2):
                rows_x0.append(h_x0[row])
                rows_k.append(float(h_k[row]))

    h_x0 = np.asarray(rows_x0, dtype=float)
    h_k = np.asarray(rows_k, dtype=float)
    w = np.full(h_k.size, 1.0 / sigma_angle_rad ** 2)
    return h_x0, h_k, w


def main() -> None:
    from lunar_od.spice_loader import load_spice_kernels

    load_spice_kernels(None, clear=True)
    et0, t_orbit = campaign_epoch()
    w15 = build_range_arc(0.0, W15_ORBITS * t_orbit, label="W15_range_bridge",
                          cadence_s=90.0)
    range_block = (w15.h_x0, w15.h_k, w15.w)
    m_range_only = combined_metrics([range_block])

    hdr("BASELINE -- RANGE ONLY (production two-way range, unaffected by s40)")
    print("  n_obs=%d  f_perp=%.6f  theta_K=%.4f deg  sigma_K/K=%.4f%%"
          % (m_range_only["n_obs"], m_range_only["orthogonal_fraction"],
             m_range_only["theta_k_deg"], 100 * m_range_only["fractional_sigma_k"]))

    results = {"range_only": m_range_only}

    # ------------------------------------------------------------------
    hdr("STEP 1 -- R1O's LANDMARK SURROGATE, EXACTLY AS PUBLISHED (Phi^T)")
    sigma_1urad = 1e-6
    lm_pub = build_landmark_arc(w15.nom48, w15.t_grid, et0,
                                sigma_angle_rad=sigma_1urad, r_moon_m=R_MOON_M,
                                label="as_published")
    m_pub = combined_metrics([range_block, (lm_pub.h_x0, lm_pub.h_k, lm_pub.w)])
    print("  landmark rows=%d at sigma=1 urad" % lm_pub.n_obs)
    print("  f_perp=%.6f  theta_K=%.4f deg  sigma_K/K=%.4f%%"
          % (m_pub["orthogonal_fraction"], m_pub["theta_k_deg"],
             100 * m_pub["fractional_sigma_k"]))
    print("  (R1O published headline: f_perp=0.8757, sigma_K/K=0.19%)")
    results["surrogate_as_published"] = m_pub

    # ------------------------------------------------------------------
    hdr("STEP 2 -- THE SAME SURROGATE WITH Phi UNFLATTENED CORRECTLY")
    h_x0_c, h_k_c, w_c = corrected_landmark_arc(
        w15.nom48, w15.t_grid, et0, sigma_angle_rad=sigma_1urad, r_moon_m=R_MOON_M)
    m_corr = combined_metrics([range_block, (h_x0_c, h_k_c, w_c)])
    print("  landmark rows=%d at sigma=1 urad" % h_k_c.size)
    print("  f_perp=%.6f  theta_K=%.4f deg  sigma_K/K=%.4f%%"
          % (m_corr["orthogonal_fraction"], m_corr["theta_k_deg"],
             100 * m_corr["fractional_sigma_k"]))
    k_col_identical = bool(np.array_equal(h_k_c, lm_pub.h_k))
    print("  K column identical to the published run (S_K needs no reshape): %s"
          % k_col_identical)
    results["surrogate_corrected"] = m_corr

    # ------------------------------------------------------------------
    hdr("STEP 3 -- PRODUCTION OPTICAL MODEL, IDEAL LIMIT")
    # The ideal limit: perfect attitude, perfect catalog, and a centroiding
    # sigma chosen to match the surrogate 1 urad angular noise so the two
    # are compared at EQUAL measurement precision, not at equal labels.
    ifov_rad = DEFAULT_CAMERA.pixel_pitch_m / DEFAULT_CAMERA.focal_length_m
    sigma_px_equiv = sigma_1urad / ifov_rad
    print("  1 urad at IFOV %.3f urad/px  ->  sigma = %.6f px"
          % (ifov_rad * 1e6, sigma_px_equiv))
    prod = build_production_optical_arc(
        w15.nom48, w15.t_grid, et0, sigma_centroid_px=sigma_px_equiv,
        label="production_ideal")
    m_prod = combined_metrics([range_block, (prod.h_x0, prod.h_k, prod.w)])
    print("  production rows=%d  (visible obs=%d, mean range=%.3f km,"
          " mean off-boresight=%.3f deg)"
          % (prod.n_rows, prod.n_visible_obs, prod.mean_range_m / 1e3,
             prod.mean_off_boresight_deg))
    print("  f_perp=%.6f  theta_K=%.4f deg  sigma_K/K=%.4f%%"
          % (m_prod["orthogonal_fraction"], m_prod["theta_k_deg"],
             100 * m_prod["fractional_sigma_k"]))
    results["production_ideal"] = m_prod

    # ------------------------------------------------------------------
    hdr("STEP 4 -- WHAT DIFFERS, ISOLATED")
    print("  The surrogate and the production model differ in exactly three ways:")
    print("    (i)   Phi transposition            -> step 1 vs step 2")
    print("    (ii)  observable definition        -> boresight-relative arcsin angles")
    print("          vs a pinhole image-plane (u, v) pair")
    print("    (iii) visibility gate              -> a 30 deg off-nadir cone vs the")
    print("          camera own %.3f deg half-FOV + far-side occultation test"
          % np.degrees(DEFAULT_CAMERA.half_fov_x_rad))
    print()
    print("  row counts: published surrogate %d, corrected surrogate %d, production %d"
          % (lm_pub.n_obs, h_k_c.size, prod.n_rows))
    print()
    print("  f_perp:   range-only               %.6f" % m_range_only["orthogonal_fraction"])
    print("            + surrogate as published %.6f" % m_pub["orthogonal_fraction"])
    print("            + surrogate corrected    %.6f" % m_corr["orthogonal_fraction"])
    print("            + production optical     %.6f" % m_prod["orthogonal_fraction"])

    transposition_effect = (m_pub["orthogonal_fraction"] - m_corr["orthogonal_fraction"])
    model_effect = (m_corr["orthogonal_fraction"] - m_prod["orthogonal_fraction"])
    print()
    print("  attributable to the Phi transposition alone : %+.6f f_perp" % transposition_effect)
    print("  attributable to the measurement model+gate  : %+.6f f_perp" % model_effect)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}
    out["_attribution"] = {
        "phi_transposition_delta_f_perp": float(transposition_effect),
        "measurement_model_delta_f_perp": float(model_effect),
        "k_column_unaffected_by_transposition": k_col_identical,
    }
    (ARTIFACTS / "r1o_opt_bridge.json").write_text(json.dumps(out, indent=2))
    print("\n  wrote artifacts/r1o_opt_bridge.json")


if __name__ == "__main__":
    main()
