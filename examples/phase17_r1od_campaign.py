"""PHASE 17-R1O-D - production DDOR campaign: info geometry, noise, systematics.

ANALYSIS SPACE ONLY.  Builds real production ΔDOR observation rows (via
`lunar_od.delta_dor`, the qualified event solver/Jacobian/K-chain) across the
W15 campaign window, on the Goldstone-Canberra pair (s30/s32), and reproduces
R1O's information-geometry comparison using R1COV's qualified square-root
covariance -- never the floored normal-matrix path (s42).
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
from phase17_r1o_core import combined_metrics  # noqa: E402
from phase17_r1cov_core import scale_matrix, square_root_covariance  # noqa: E402

from lunar_od.delta_dor import (  # noqa: E402
    QuasarDirection,
    DeltaDorEventError,
    delta_dor_spacecraft_sensitivity_full,
    quasar_differential_delay_s,
    solve_common_transmit_event,
)
from lunar_od.radiometrics import _interp_state  # noqa: E402
from lunar_od.measurements import _station_relative_state_j2000_at_receive_epoch  # noqa: E402

ALL_STATIONS = ("Goldstone DSN", "Madrid DSN", "Canberra DSN")
DDOR_PAIR = ("Goldstone DSN", "Canberra DSN")
ARTIFACTS = Path(r"C:/Users/erayh/Documents/Python/Grad/python_port_phase17r0/artifacts")


def hdr(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


def build_ddor_rows(arc, station_a, station_b, quasar, min_elevation_deg=10.0):
    """Production ΔDOR design rows: h_x0 (M,6), h_k (M,), D_S (M,), used t (M,)."""
    from lunar_od.visibility import sample_j2000_to_itrf93_transforms
    from lunar_od.geometry import ecef2razel_sez
    import od_gravity_covariance_campaign as C  # noqa: N811

    et0, _ = campaign_epoch()
    t_grid = arc.t_grid
    t_local = t_grid - t_grid[0]
    xf = sample_j2000_to_itrf93_transforms(et0 + t_grid[0], t_local)
    earth_pos_fixed = np.asarray(C.get_earth_pos(0.0), float).reshape(3)

    def station_hist(station):
        rows = []
        for i in range(len(t_grid)):
            rel = _station_relative_state_j2000_at_receive_epoch(station, xf[i])
            rows.append(np.concatenate([earth_pos_fixed + rel[:3], rel[3:]]))
        return np.asarray(rows, dtype=float)

    hist_a, hist_b = station_hist(station_a), station_hist(station_b)

    def a_pos(t):
        return _interp_state(t_grid, hist_a, t)[:3]

    def b_state(t):
        return _interp_state(t_grid, hist_b, t)

    def b_pos(t):
        return b_state(t)[:3]

    def sc_pos(t):
        return _interp_state(t_grid, arc.nom48, t)[:3]

    def elevation_deg(station, r_sc, earth_pos, xfi):
        r_rel_j2000 = r_sc - earth_pos
        r_rel_ecef = xfi[:3, :3] @ r_rel_j2000 - np.asarray(station.r_ecef_m, float).reshape(3)
        _, el, _ = ecef2razel_sez(r_rel_ecef, station.lat_rad, station.lon_rad)
        return np.degrees(el)

    rows_x0, rows_k, d_s_vals, t_used, cond_numbers = [], [], [], [], []
    n_attempted, n_dual_visible, n_converged = 0, 0, 0
    for i, t_obs in enumerate(t_grid):
        r_sc_i = arc.nom48[i, :3]
        el_a = elevation_deg(station_a, r_sc_i, earth_pos_fixed, xf[i])
        el_b = elevation_deg(station_b, r_sc_i, earth_pos_fixed, xf[i])
        if el_a < min_elevation_deg or el_b < min_elevation_deg:
            continue
        n_dual_visible += 1
        n_attempted += 1
        station_a_at_tobs = a_pos(t_obs)
        try:
            common = solve_common_transmit_event(float(t_obs), station_a_at_tobs, b_pos, sc_pos)
        except DeltaDorEventError:
            continue
        station_b_state_tb = b_state(common.station_b_receive_time_s)
        try:
            sens = delta_dor_spacecraft_sensitivity_full(
                common, t_grid, arc.nom48,
                station_a_at_tobs, station_b_state_tb[:3], station_b_state_tb[3:6],
            )
        except DeltaDorEventError:
            continue
        n_converged += 1
        rows_x0.append(sens.d_spacecraft_delay_dx0)
        rows_k.append(sens.d_spacecraft_delay_dk)
        d_s_vals.append(common.spacecraft_differential_delay_s)
        t_used.append(float(t_obs))
        cond_numbers.append(sens.event_matrix_condition_number)

    return dict(
        h_x0=np.asarray(rows_x0, float), h_k=np.asarray(rows_k, float),
        d_s=np.asarray(d_s_vals, float), t_used=np.asarray(t_used, float),
        cond_numbers=np.asarray(cond_numbers, float),
        n_dual_visible=n_dual_visible, n_converged=n_converged,
    )


def main() -> None:
    from lunar_od.spice_loader import load_spice_kernels
    import od_gravity_covariance_campaign as C  # noqa: N811

    load_spice_kernels(None, clear=True)
    et0, t_orbit = campaign_epoch()

    hdr("s30/s32 -- GOLDSTONE-CANBERRA DUAL VISIBILITY REPRODUCTION (production solver)")
    arc = build_range_arc(0.0, 15.0 * t_orbit, label="W15", station_filter=ALL_STATIONS,
                          cadence_s=90.0)
    by_name = {s.name: s for s in C.STATIONS}
    station_a, station_b = by_name[DDOR_PAIR[0]], by_name[DDOR_PAIR[1]]
    quasar = QuasarDirection(ra_rad=2.1, dec_rad=-0.3)

    ddor = build_ddor_rows(arc, station_a, station_b, quasar)
    print("  dual-visible (10deg mask) epochs: %d" % ddor["n_dual_visible"])
    print("  common-event solves converged:    %d" % ddor["n_converged"])
    print("  event-matrix condition: min=%.3e max=%.3e"
          % (ddor["cond_numbers"].min(), ddor["cond_numbers"].max()))
    gate_gc = ddor["n_converged"] >= 50  # comparable to R1O's surrogate n=143 order
    print("  GOLDSTONE_CANBERRA_DUAL_VISIBILITY_REPRODUCTION = %s"
          % ("PASS" if gate_gc else "FAIL"))

    hdr("s36-s40 -- PRODUCTION INFORMATION GEOMETRY  (range-only vs range+production-DDOR)")
    scale = scale_matrix()
    m_range_only = combined_metrics([(arc.h_x0, arc.h_k, arc.w)])
    print("  range-only: n=%d  I_K|x=%.6e  f_perp=%.4f  theta_K=%.2f  sigma_K/K=%.4f"
          % (m_range_only["n_obs"], m_range_only["conditional_k_information"],
             m_range_only["orthogonal_fraction"], m_range_only["theta_k_deg"],
             m_range_only["fractional_sigma_k"]))

    sigma_angle_rad = 5e-9  # representative operational DDOR precision (R1O s24 anchor)
    sigma_delay_rows = []
    et0v, _ = campaign_epoch()
    from lunar_od.visibility import sample_j2000_to_itrf93_transforms
    t_local = arc.t_grid - arc.t_grid[0]
    xf_full = sample_j2000_to_itrf93_transforms(et0v + arc.t_grid[0], t_local)
    earth_pos_fixed = np.asarray(C.get_earth_pos(0.0), float).reshape(3)

    def station_pos_at(station, i):
        rel = _station_relative_state_j2000_at_receive_epoch(station, xf_full[i])
        return earth_pos_fixed + rel[:3]

    idx_lookup = {t: i for i, t in enumerate(arc.t_grid)}
    for t_obs in ddor["t_used"]:
        i = idx_lookup[t_obs]
        r_a, r_b = station_pos_at(station_a, i), station_pos_at(station_b, i)
        s_hat = quasar.unit_vector_j2000
        los = (r_a - arc.nom48[i, :3]) / np.linalg.norm(r_a - arc.nom48[i, :3])
        baseline = r_b - r_a
        b_perp = float(np.linalg.norm(baseline - np.dot(baseline, los) * los))
        sigma_delay_rows.append(max(b_perp, 1.0) * sigma_angle_rad / 299792458.0)
    w_ddor = 1.0 / np.asarray(sigma_delay_rows) ** 2

    m_combined = combined_metrics([
        (arc.h_x0, arc.h_k, arc.w), (ddor["h_x0"], ddor["h_k"], w_ddor)])
    print("  range+production-DDOR (5 nrad): n=%d  I_K|x=%.6e  f_perp=%.4f  theta_K=%.2f  "
          "sigma_K/K=%.4f"
          % (m_combined["n_obs"], m_combined["conditional_k_information"],
             m_combined["orthogonal_fraction"], m_combined["theta_k_deg"],
             m_combined["fractional_sigma_k"]))

    f_perp_gain = m_combined["orthogonal_fraction"] - m_range_only["orthogonal_fraction"]
    info_gain_ratio = (m_combined["conditional_k_information"]
                       / max(m_range_only["conditional_k_information"], 1e-300))
    if f_perp_gain > 0.05:
        info_class = "STRONG_DIRECTION_GAIN"
    elif f_perp_gain > 0.02:
        info_class = "MODEST_DIRECTION_GAIN"
    else:
        info_class = "MAGNITUDE_ONLY"
    print("  f_perp gain = %.4f   info-magnitude gain ratio = %.3fx" % (f_perp_gain, info_gain_ratio))
    print("  PRODUCTION_DDOR_INFORMATION_DIRECTION_GATE class = %s" % info_class)

    # s42 covariance cross-check on THIS combined system specifically
    from phase17_r1o_core import orthogonal_decomposition, whitened
    a_x, b_k = whitened(np.vstack([arc.h_x0, ddor["h_x0"]]),
                        np.concatenate([arc.h_k, ddor["h_k"]]),
                        np.concatenate([arc.w, w_ddor]))
    dec = orthogonal_decomposition(a_x, b_k)
    schur_sigma = 1.0 / np.sqrt(dec["i_k_given_x"])
    qr_sigma = m_combined["qualified_sigma_k"]
    cc_rel = abs(qr_sigma - schur_sigma) / schur_sigma
    print("  R1COV_DDOR_COVARIANCE_PATH_GATE: QR=%.6e  Schur=%.6e  rel=%.3e  %s"
          % (qr_sigma, schur_sigma, cc_rel, "PASS" if cc_rel < 1e-6 else "FAIL"))

    hdr("s31/s44 -- RANDOM NOISE STUDY")
    noise_rows = []
    for nrad in (1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0):
        sig_r = nrad * 1e-9
        w_i = []
        for t_obs in ddor["t_used"]:
            i = idx_lookup[t_obs]
            r_a, r_b = station_pos_at(station_a, i), station_pos_at(station_b, i)
            s_hat = quasar.unit_vector_j2000
            los = (r_a - arc.nom48[i, :3]) / np.linalg.norm(r_a - arc.nom48[i, :3])
            baseline = r_b - r_a
            b_perp = float(np.linalg.norm(baseline - np.dot(baseline, los) * los))
            w_i.append(1.0 / (max(b_perp, 1.0) * sig_r / 299792458.0) ** 2)
        m_i = combined_metrics([(arc.h_x0, arc.h_k, arc.w),
                                (ddor["h_x0"], ddor["h_k"], np.asarray(w_i))])
        noise_rows.append(dict(noise_nrad=nrad, f_perp=m_i["orthogonal_fraction"],
                               fractional_sigma_k=m_i["fractional_sigma_k"],
                               conditional_k_info=m_i["conditional_k_information"]))
        print("  %6.1f nrad: f_perp=%.4f  sigma_K/K=%.4f" % (
            nrad, m_i["orthogonal_fraction"], m_i["fractional_sigma_k"]))

    with (ARTIFACTS / "r1od_random_noise_study.csv").open("w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(noise_rows[0].keys()))
        w_.writeheader(); w_.writerows(noise_rows)

    hdr("s32/s45/s46 -- SYSTEMATIC RESIDUAL / SESSION-BIAS STUDY")
    # s46 asks for THREE distinct cases, which require TWO different and
    # NOT-interchangeable formulas -- an earlier version of this script
    # conflated them (used the joint-solve correlation for both):
    #
    #   Case B (unmodeled bias): a constant, uncalibrated DDOR session bias
    #   is PRESENT in the data but NOT solved for. The standard
    #   omitted-variable-bias result applies: for a 7-parameter normal
    #   system N_7 = H_7^T W H_7 and a bias-direction column n_bias =
    #   H_7^T W @ (1 on DDOR rows, 0 on range rows), the resulting shift in
    #   every parameter (K included) is delta_theta = solve(N_7, n_bias) *
    #   bias_s. This is NOT the same quantity as any covariance entry --
    #   it is a BIAS in the point estimate, not an uncertainty.
    #
    #   Case C (bias as consider/nuisance parameter): bias IS included as
    #   an 8th solved-for parameter. Its effect is to INFLATE sigma_K
    #   (never bias the point estimate, by construction of a jointly-solved
    #   least-squares system) -- this is what the augmented covariance's
    #   [K,K] entry directly gives, via the SAME qualified square-root path.
    ones_ddor = np.ones(len(ddor["t_used"]))
    h_full_range = np.hstack([arc.h_x0, arc.h_k[:, None]])
    h_full_ddor = np.hstack([ddor["h_x0"], ddor["h_k"][:, None]])
    w_all = np.concatenate([arc.w, w_ddor])
    h_all_7 = np.vstack([h_full_range, h_full_ddor])
    bias_direction_col = np.concatenate([np.zeros(len(arc.w)), ones_ddor])

    n_7 = h_all_7.T @ (w_all[:, None] * h_all_7)
    n_bias = h_all_7.T @ (w_all * bias_direction_col)
    delta_theta_per_unit_bias = np.linalg.solve(n_7, n_bias)
    delta_k_per_unit_bias = float(delta_theta_per_unit_bias[6])
    print("  Case B (omitted-variable): d(K)/d(session_bias) = %.6e (m^2/kg)/s"
          % delta_k_per_unit_bias)

    h_aug = np.hstack([h_all_7, bias_direction_col[:, None]])
    scale8 = np.diag(list(np.diag(scale)) + [1.0])
    from lunar_od.estimators import _square_root_covariance_from_design
    cov8 = _square_root_covariance_from_design(h_aug, w_all, np.zeros((8, 8)), scale8)[0]
    sigma_k_case_c = float(np.sqrt(cov8[6, 6]))
    sigma_k_case_a = m_combined["qualified_sigma_k"]
    print("  Case A (perfectly calibrated): sigma_K = %.6e" % sigma_k_case_a)
    print("  Case C (bias solved as nuisance param): sigma_K = %.6e  (inflation %.2fx)"
          % (sigma_k_case_c, sigma_k_case_c / sigma_k_case_a))

    session_bias_s_values = [0.0, 1e-9, 1e-8, 1e-7]  # seconds, literature-plausible residual scale
    sys_rows = []
    for bias_s in session_bias_s_values:
        induced_k_shift = delta_k_per_unit_bias * bias_s
        frac_of_truth = abs(induced_k_shift) / 0.01
        sys_rows.append(dict(session_bias_s=bias_s,
                             case_b_induced_k_shift_m2_per_kg=induced_k_shift,
                             case_b_shift_fraction_of_k_truth=frac_of_truth,
                             case_a_sigma_k=sigma_k_case_a,
                             case_c_sigma_k_with_bias_nuisance=sigma_k_case_c))
        print("  bias=%.1e s: Case-B induced dK=%.4e m^2/kg (%.3f%% of K_truth)"
              % (bias_s, induced_k_shift, 100 * frac_of_truth))

    with (ARTIFACTS / "r1od_systematic_residual_study.csv").open("w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(sys_rows[0].keys()))
        w_.writeheader(); w_.writerows(sys_rows)

    summary = dict(
        n_dual_visible=int(ddor["n_dual_visible"]), n_converged=int(ddor["n_converged"]),
        goldstone_canberra_dual_visibility_reproduction="PASS" if gate_gc else "FAIL",
        range_only=m_range_only, range_ddor_5nrad=m_combined,
        f_perp_gain=f_perp_gain, information_direction_class=info_class,
        r1cov_ddor_covariance_path_gate="PASS" if cc_rel < 1e-6 else "FAIL",
        covariance_crosscheck_rel_error=cc_rel,
    )
    (ARTIFACTS / "r1od_campaign_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\n  wrote r1od_random_noise_study.csv, r1od_systematic_residual_study.csv, "
          "r1od_campaign_summary.json")


if __name__ == "__main__":
    main()
