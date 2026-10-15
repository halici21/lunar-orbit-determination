"""PHASE 17-FXR s27-s34 - requalify R1M, R1COV, R1O, R1O-D, R1O-OPT and R1O-R
on the epoch-consistent Earth fixture.

ANALYSIS SPACE ONLY.  No production file is modified.

Every task runs the SAME historical analysis code twice, changing only the
Earth ephemeris the measurement geometry sees:

  LEGACY     EARTH_FIXTURE_LEGACY_OFFSET_CLAMPED -- must reproduce the committed
             historical artifact of that phase EXACTLY, or the task is not
             re-running what was published and its corrected arm is void
  CANONICAL  EARTH_FIXTURE_CANONICAL -- the corrected result

The propagated truth, STM and dx/dK (nom48) do not depend on the Earth fixture
(mu_earth = 0); each task checks that too, so every delta below is geometry.

R1O-D gets a third arm.  As published it froze the Earth centre at the arc
start for all 15 orbits (a second, smaller Earth-epoch inconsistency).  The
arms are LEGACY/FROZEN (reproduction), CANONICAL/FROZEN (only the 5.83 d
offset repaired) and CANONICAL/MEASUREMENT_EPOCH (fully consistent).

Run:  PYTHONPATH="<worktree>;<od_covariance_campaign>/06_scripts"
      python studies/phase17/scripts/phase17_fxr_requalification.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

#: Phase 17-HARDEN: FXR requalified R1O/R1O-D with the elevation-only ΔDOR /
#: Earth-LOS row rule (lunar occultation untested).  The row builders now
#: default to the production visibility contract, so reproducing the published
#: FXR artifacts requires naming the superseded rule explicitly.  The
#: authoritative corrected numbers are Phase 17-R1O-D2's (see
#: studies/phase17/reports/phase17_authoritative_status.md).
_LEGACY_RULE = "ELEVATION_ONLY_HISTORICAL_R1OD"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
ARTIFACTS = ROOT / "studies" / "phase17" / "results"

DSN3 = ("Goldstone DSN", "Madrid DSN", "Canberra DSN")
DDOR_PAIR = ("Goldstone DSN", "Canberra DSN")
K_TRUTH = 0.01
C_LIGHT = 299792458.0
FIXTURES = ("LEGACY", "CANONICAL")

#: GEO s-Jacobian characterization: largest |f_perp(analytic) - f_perp(FD design)|
#: measured on the canonical fixture (studies/phase17/results/phase17_geo_jacobian_diagnosis.json).
JACOBIAN_F_PERP_BOUND = 3.77e-06
#: Relative sigma_K/K analytic-vs-FD difference measured by GEO (same artifact).
JACOBIAN_SIGMA_REL_BOUND = 1.6e-05
#: A classification margin is only trusted if it exceeds the measured Jacobian
#: effect by this factor (other configurations were not FD-checked by GEO).
MATERIALITY_SAFETY_FACTOR = 100.0


def _init() -> None:
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels(None, clear=True)
    warnings.simplefilter("ignore")


def _fx(name: str) -> str:
    import phase17_r1m_core as M
    return {"LEGACY": M.EARTH_FIXTURE_LEGACY_OFFSET_CLAMPED,
            "CANONICAL": M.EARTH_FIXTURE_CANONICAL,
            "OFFSET_ONLY": M.EARTH_FIXTURE_DIAG_OFFSET_UNCLAMPED,
            "CLAMP_ONLY": M.EARTH_FIXTURE_DIAG_CORRECT_EPOCH_CLAMPED}[name]


def _sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, float))).hexdigest()


def _clean(d):
    if isinstance(d, dict):
        return {k: _clean(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_clean(v) for v in d]
    if isinstance(d, np.ndarray):
        return d.tolist()
    if isinstance(d, np.generic):
        return d.item()
    return d


def _arc_info(arc) -> dict:
    """R1M single-arc row (identical arithmetic to phase17_r1m_multi_arc_study)."""
    from phase17_r1cov_core import condition_numbers, square_root_covariance
    from phase17_r1m_core import (information_matrix, orthogonal_decomposition,
                                  scale_matrix, spectrum, whitened)

    a_x, b_k = whitened(arc.h_x0, arc.h_k, arc.w)
    dec = orthogonal_decomposition(a_x, b_k)
    h = np.hstack([arc.h_x0, arc.h_k[:, None]])
    info = information_matrix(h, arc.w, scale_matrix())
    sp = spectrum(info)
    cn = condition_numbers(h, arc.w, None, scale_matrix())
    sr = square_root_covariance(h, arc.w, None, scale_matrix())
    return dict(
        observations=int(arc.n_obs), station=arc.station_name,
        k_column_norm=dec["k_column_norm"], projected_norm=dec["projected_norm"],
        orthogonal_norm=dec["orthogonal_norm"], orthogonal_fraction=dec["orthogonal_fraction"],
        raw_i_kk=dec["i_kk"], conditional_k_information=dec["i_k_given_x"],
        smallest_scaled_singular_value=sp["smallest"], scaled_condition=sp["condition"],
        rank=sp["rank_default"], weakest_mode_k_component=sp["weakest_k_component"],
        cond_design=cn["cond_design"], cond_information=cn["cond_information"],
        design_rank=int(sr.rank_estimate), design_rank_margin=float(sr.rank_margin),
        qr_sigma_k=float(sr.sigma_k),
        schur_sigma_k=float(1.0 / np.sqrt(dec["i_k_given_x"])) if dec["i_k_given_x"] > 0
        else float("inf"),
    )


# ======================================================================
# R1M single arcs + R1COV requalification (G0..G3)
# ======================================================================
def task_r1m_single(fx_name: str) -> dict:
    from lunar_od.estimators import _square_root_covariance_from_design
    from phase17_r1cov_core import K_INDEX, exact_covariance, floored_covariance, relative_error
    from phase17_r1cov_core import scale_matrix as cov_scale
    from phase17_r1m_core import build_range_arc, campaign_epoch

    fx = _fx(fx_name)
    _, t_orbit = campaign_epoch()
    rows, cov_rows, shas = [], [], {}
    for label, orbits in (("G0", 1.3), ("G1", 2.0), ("G2", 3.0), ("G3", 5.0)):
        arc = build_range_arc(0.0, orbits * t_orbit, label=label, earth_fixture=fx)
        shas[label] = _sha(arc.nom48)
        rows.append(dict(case=label, arc_orbits=orbits, **_arc_info(arc)))
        if label == "G2":
            continue
        h = np.hstack([arc.h_x0, arc.h_k[:, None]])
        scale = cov_scale()
        for pname, psig in (("data_only", None), ("broad_k_sigma_1.0", 1.0),
                            ("moderate_k_sigma_0.01", 0.01)):
            pinv = None
            if psig is not None:
                pinv = np.zeros((7, 7))
                pinv[K_INDEX, K_INDEX] = 1.0 / psig ** 2
            cov_new, _ = _square_root_covariance_from_design(
                h, arc.w, pinv if pinv is not None else np.zeros((7, 7)), scale)
            sig_new = float(np.sqrt(cov_new[K_INDEX, K_INDEX]))
            _, fl = floored_covariance(h, arc.w, pinv, scale)
            _, sig_ex = exact_covariance(h, arc.w, pinv)
            cov_rows.append(dict(case=label, prior=pname, old_sigma_k=fl["sigma_k"],
                                 qualified_sigma_k=sig_new, oracle_sigma_k=sig_ex,
                                 qualified_relative_error=relative_error(sig_new, sig_ex),
                                 old_overconfidence_factor=sig_ex / fl["sigma_k"],
                                 sigma_k_over_k_truth=sig_new / K_TRUTH,
                                 floored_modes=fl["floored_modes"],
                                 floor_inflation=fl["floor_inflation"]))
    return _clean(dict(task="r1m_single", fixture=fx_name, single_arc=rows,
                       r1cov_requalification=cov_rows, nom48_sha=shas))


# ======================================================================
# R1M multi-arc (M1-M3, contiguous controls) and fairness (F2/F3)
# ======================================================================
def _window_counts(arc, windows) -> list[int]:
    t = np.asarray(arc.obs[:, 0], float) + arc.t_grid[0]
    return [int(np.sum((t >= a - 1e-6) & (t <= b + 1e-6))) for a, b in windows]


def task_r1m_multi(fx_name: str) -> dict:
    from lunar_od.estimators import _square_root_covariance_from_design
    from phase17_r1cov_core import K_INDEX, exact_covariance, floored_covariance
    from phase17_r1cov_core import scale_matrix as cov_scale
    from phase17_r1m_addendum import FAIR_CAMPAIGNS, FAIR_WINDOW_ORBITS
    from phase17_r1m_core import (build_range_arc, campaign_epoch, orthogonal_decomposition,
                                  whitened)
    from phase17_r1m_multi_arc_study import (ALL_STATIONS, CAMPAIGNS, WINDOW_ORBITS,
                                             model_a_information, model_b_information)

    fx = _fx(fx_name)
    _, t_orbit = campaign_epoch()
    multi, controls, fair, m3_cov = [], [], [], None

    def both(name, starts, window_orbits):
        window_s = window_orbits * t_orbit
        windows = [(s * t_orbit, s * t_orbit + window_s) for s in starts]
        arcs = [build_range_arc(w0, window_s, label="%s_a%d" % (name, i),
                                station_filter=ALL_STATIONS, earth_fixture=fx)
                for i, (w0, _) in enumerate(windows)]
        a = model_a_information(arcs)
        long_arc = build_range_arc(0.0, windows[-1][1], label="%s_long" % name,
                                   station_filter=ALL_STATIONS, earth_fixture=fx)
        b = model_b_information(long_arc, windows)
        per_arc = []
        for arc in arcs:
            ax, bk = whitened(arc.h_x0, arc.h_k, arc.w)
            per_arc.append(orthogonal_decomposition(ax, bk)["i_k_given_x"])
        return windows, arcs, long_arc, a, b, per_arc

    for name, starts in CAMPAIGNS.items():
        windows, arcs, long_arc, a, b, per_arc = both(name, starts, WINDOW_ORBITS)
        sum_rel = abs(a["conditional_k"] - sum(per_arc)) / max(sum(per_arc), 1e-300)
        for model, res in (("A_independent_arcs", a), ("B_continuity_linked", b)):
            multi.append(dict(
                case=name, model=model, arcs=len(starts),
                elapsed_orbits=round((windows[-1][1] - windows[0][0]) / t_orbit, 4),
                observations=res["n_obs"], unknowns=res["n_unknown"],
                rank=res["spectrum"]["rank_default"],
                weakest_mode_k_component=res["spectrum"]["weakest_k_component"],
                conditional_k_information=res["conditional_k"],
                orthogonal_fraction=res["decomposition"]["orthogonal_fraction"],
                sigma_k_data_only=(1.0 / np.sqrt(res["conditional_k"])
                                   if res["conditional_k"] > 0 else float("inf")),
                per_window_obs_model_a=[int(x.n_obs) for x in arcs],
                per_window_obs_model_b=_window_counts(long_arc, windows),
                schur_sum_rule_rel=sum_rel))
        if name == "M3":
            keep = np.zeros(long_arc.obs.shape[0], bool)
            t_obs = np.asarray(long_arc.obs[:, 0], float)
            for w0, w1 in windows:
                keep |= (t_obs >= w0 - 1e-6) & (t_obs <= w1 + 1e-6)
            h = np.hstack([long_arc.h_x0[keep], long_arc.h_k[keep][:, None]])
            w = long_arc.w[keep]
            cov, _ = _square_root_covariance_from_design(h, w, np.zeros((7, 7)), cov_scale())
            _, sig_ex = exact_covariance(h, w, None)
            _, fl = floored_covariance(h, w, None, cov_scale())
            s = float(np.sqrt(cov[K_INDEX, K_INDEX]))
            m3_cov = dict(observations=int(keep.sum()), qualified_sigma_k=s,
                          oracle_sigma_k=sig_ex, old_sigma_k=fl["sigma_k"],
                          fractional=s / K_TRUTH)
        contig = build_range_arc(0.0, len(starts) * WINDOW_ORBITS * t_orbit,
                                 label="%s_contig" % name, station_filter=ALL_STATIONS,
                                 earth_fixture=fx)
        ax, bk = whitened(contig.h_x0, contig.h_k, contig.w)
        dec = orthogonal_decomposition(ax, bk)
        controls.append(dict(case=name, observations=int(contig.n_obs),
                             contiguous_conditional_k=dec["i_k_given_x"],
                             contiguous_f_perp=dec["orthogonal_fraction"],
                             linked_conditional_k=b["conditional_k"],
                             linked_over_contiguous=b["conditional_k"]
                             / max(dec["i_k_given_x"], 1e-300)))

    for name, starts in FAIR_CAMPAIGNS.items():
        windows, arcs, long_arc, a, b, per_arc = both(name, starts, FAIR_WINDOW_ORBITS)
        for model, res in (("A_independent_arcs", a), ("B_continuity_linked", b)):
            ck = res["conditional_k"]
            sig = 1.0 / np.sqrt(ck) if ck > 0 else float("inf")
            fair.append(dict(
                case=name, model=model, arcs=len(starts),
                elapsed_orbits=round((windows[-1][1] - windows[0][0]) / t_orbit, 4),
                observations=res["n_obs"], unknowns=res["n_unknown"],
                conditional_k_information=ck, sigma_k_data_only=sig,
                sigma_k_over_k_truth=sig / K_TRUTH,
                orthogonal_fraction=res["decomposition"]["orthogonal_fraction"],
                weakest_mode_k_component=res["spectrum"]["weakest_k_component"],
                rank=res["spectrum"]["rank_default"],
                per_window_obs_model_a=[int(x.n_obs) for x in arcs]))
    return _clean(dict(task="r1m_multi", fixture=fx_name, multi_arc=multi,
                       elapsed_control=controls, fairness=fair, m3_model_b_covariance=m3_cov))


# ======================================================================
# R1O (surrogates, corrected STM) and R1O-R STM equivalence
# ======================================================================
def task_r1o(fx_name: str) -> dict:
    import od_gravity_covariance_campaign as C  # noqa: N811
    from lunar_od.constants import R_MOON_M
    from phase17_r1m_core import build_range_arc, campaign_epoch
    from phase17_r1o_core import build_ddor_arc, build_landmark_arc, combined_metrics
    from phase17_r1o_r_requalification import (_assert_historical_equivalence,
                                               bridge_reduced_rows, earth_los_rows,
                                               historical_nom48)

    fx = _fx(fx_name)
    et0, t_orbit = campaign_epoch()
    w15 = build_range_arc(0.0, 15.0 * t_orbit, label="W15_range", station_filter=DSN3,
                          cadence_s=90.0, earth_fixture=fx)
    nom = w15.nom48
    rb = (w15.h_x0, w15.h_k, w15.w)
    out = dict(task="r1o", fixture=fx_name, nom48_sha=_sha(nom))
    out["stm_historical_equivalence"] = _assert_historical_equivalence(nom, historical_nom48(nom))
    out["range_only"] = combined_metrics([rb])
    lm = build_landmark_arc(nom, w15.t_grid, et0, sigma_angle_rad=1e-6, r_moon_m=R_MOON_M,
                            label="lm")
    out["landmark_1urad"] = combined_metrics([rb, (lm.h_x0, lm.h_k, lm.w)])
    out["landmark_only_1urad"] = combined_metrics([(lm.h_x0, lm.h_k, lm.w)])
    out["landmark_rows_sha"] = _sha(np.hstack([lm.h_x0, lm.h_k[:, None]]))
    for nrad in (1.0, 5.0):
        d = build_ddor_arc(nom, w15.t_grid, et0, DDOR_PAIR, sigma_angle_rad=nrad * 1e-9,
                           label="ddor", earth_fixture=fx,
                           row_rule=_LEGACY_RULE)   # FXR as published (HARDEN)
        out["ddor_surrogate_%gnrad" % nrad] = combined_metrics([rb, (d.h_x0, d.h_k, d.w)])
    ex, ek, _ = earth_los_rows(nom, w15.t_grid, earth_fixture=fx, row_rule=_LEGACY_RULE)
    # weight expressions copied verbatim from phase17_r1o_r_requalification (bitwise)
    w_e = np.full(ek.size, 1.0 / (1.0 * 1e-6) ** 2)
    out["earth_los_1urad"] = combined_metrics([rb, (ex, ek, w_e)])
    by = {s.name: s for s in C.STATIONS}
    bx, bk = bridge_reduced_rows(nom, w15.t_grid, et0, by[DDOR_PAIR[0]], by[DDOR_PAIR[1]],
                                 earth_fixture=fx)
    sig_b = 5.0 * 1e-9
    w_b = np.full(bk.size, 1.0 / sig_b ** 2 * (1e7 / 299792458.0) ** 2)
    out["bridge_surrogate_5nrad"] = combined_metrics([rb, (bx, bk, w_b)])
    return _clean(out)


# ======================================================================
# R1O-D production DDOR
# ======================================================================
def task_r1od(args) -> dict:
    fx_name, anchor = args
    import od_gravity_covariance_campaign as C  # noqa: N811
    from lunar_od.delta_dor import QuasarDirection
    from lunar_od.estimators import _square_root_covariance_from_design
    from phase17_r1m_core import build_range_arc, campaign_epoch
    from phase17_r1o_core import combined_metrics
    from phase17_r1cov_core import scale_matrix
    import phase17_r1od_campaign as D

    fx = _fx(fx_name)
    anchor_const = {"FROZEN": D.EARTH_ANCHOR_FROZEN_ARC_START,
                    "MEASUREMENT_EPOCH": D.EARTH_ANCHOR_MEASUREMENT_EPOCH}[anchor]
    _, t_orbit = campaign_epoch()
    arc = build_range_arc(0.0, 15.0 * t_orbit, label="W15", station_filter=DSN3,
                          cadence_s=90.0, earth_fixture=fx)
    by = {s.name: s for s in C.STATIONS}
    sa, sb = by[DDOR_PAIR[0]], by[DDOR_PAIR[1]]
    ddor = D.build_ddor_rows(arc, sa, sb, QuasarDirection(ra_rad=2.1, dec_rad=-0.3),
                             earth_fixture=fx, earth_anchor=anchor_const,
                             row_rule=_LEGACY_RULE)   # FXR as published (HARDEN)
    rb = (arc.h_x0, arc.h_k, arc.w)
    m_r = combined_metrics([rb])
    noise = []
    for nrad in (1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0):
        w_i = D.ddor_delay_weights(arc, ddor, sa, sb, nrad * 1e-9)
        m_i = combined_metrics([rb, (ddor["h_x0"], ddor["h_k"], w_i)])
        noise.append(dict(noise_nrad=nrad, f_perp=m_i["orthogonal_fraction"],
                          fractional_sigma_k=m_i["fractional_sigma_k"],
                          conditional_k_info=m_i["conditional_k_information"],
                          theta_k_deg=m_i["theta_k_deg"]))
    w5 = D.ddor_delay_weights(arc, ddor, sa, sb, 5e-9)
    m_c = combined_metrics([rb, (ddor["h_x0"], ddor["h_k"], w5)])
    m_ddor_only = combined_metrics([(ddor["h_x0"], ddor["h_k"], w5)])
    gain = m_c["orthogonal_fraction"] - m_r["orthogonal_fraction"]
    cls = ("STRONG_DIRECTION_GAIN" if gain > 0.05 else
           "MODEST_DIRECTION_GAIN" if gain > 0.02 else "MAGNITUDE_ONLY")
    # s45/s46 Case B (omitted-variable bias) and Case C (bias as nuisance)
    h7 = np.vstack([np.hstack([arc.h_x0, arc.h_k[:, None]]),
                    np.hstack([ddor["h_x0"], ddor["h_k"][:, None]])])
    w_all = np.concatenate([arc.w, w5])
    bias_col = np.concatenate([np.zeros(len(arc.w)), np.ones(len(ddor["t_used"]))])
    n7 = h7.T @ (w_all[:, None] * h7)
    dk_dbias = float(np.linalg.solve(n7, h7.T @ (w_all * bias_col))[6])
    scale8 = np.diag(list(np.diag(scale_matrix())) + [1.0])
    cov8 = _square_root_covariance_from_design(np.hstack([h7, bias_col[:, None]]), w_all,
                                               np.zeros((8, 8)), scale8)[0]
    return _clean(dict(
        task="r1od", fixture=fx_name, anchor=anchor, nom48_sha=_sha(arc.nom48),
        n_dual_visible=ddor["n_dual_visible"], n_converged=ddor["n_converged"],
        range_only=m_r, range_ddor_5nrad=m_c, ddor_only_5nrad=m_ddor_only,
        f_perp_gain=gain, information_direction_class=cls, noise_study=noise,
        below_range_at_every_precision=bool(all(r["f_perp"] < m_r["orthogonal_fraction"]
                                                for r in noise)),
        case_b_dk_per_second_bias=dk_dbias,
        case_b_shift_fraction_at_1ns=abs(dk_dbias * 1e-9) / K_TRUTH,
        case_a_sigma_k=m_c["qualified_sigma_k"],
        case_c_sigma_k=float(np.sqrt(cov8[6, 6])),
    ))


# ======================================================================
# R1O-OPT production optical
# ======================================================================
def task_r1oopt(fx_name: str) -> dict:
    from lunar_od.lunar_landmark_optical import DEFAULT_CAMERA
    from phase17_r1m_core import build_range_arc, campaign_epoch
    from phase17_r1o_core import combined_metrics
    from phase17_r1o_opt_core import build_production_optical_arc

    fx = _fx(fx_name)
    et0, t_orbit = campaign_epoch()
    ifov = DEFAULT_CAMERA.pixel_pitch_m / DEFAULT_CAMERA.focal_length_m
    out = dict(task="r1oopt", fixture=fx_name, cadence_family=[])
    for cad in (90.0, 10.0):
        w15 = build_range_arc(0.0, 15.0 * t_orbit, label="W15_cad%.0f" % cad, cadence_s=cad,
                              earth_fixture=fx)
        prod = build_production_optical_arc(w15.nom48, w15.t_grid, et0,
                                            sigma_centroid_px=1e-6 / ifov)
        rb = (w15.h_x0, w15.h_k, w15.w)
        m_r = combined_metrics([rb])
        m_c = combined_metrics([rb, (prod.h_x0, prod.h_k, prod.w)])
        out["cadence_family"].append(dict(
            cadence_s=cad, range_obs=int(w15.n_obs), station=w15.station_name,
            visible_obs=prod.n_visible_obs, f_perp_range_only=m_r["orthogonal_fraction"],
            f_perp_combined=m_c["orthogonal_fraction"],
            sigma_k_frac_combined=m_c["fractional_sigma_k"]))
    out["station"] = w15.station_name
    out["nom48_sha"] = _sha(w15.nom48)
    rb = (w15.h_x0, w15.h_k, w15.w)
    m_r = combined_metrics([rb])
    out["range_only"] = m_r
    noise = []
    for sigma_px in (0.1, 0.25, 1.0, 2.5):
        prod = build_production_optical_arc(w15.nom48, w15.t_grid, et0, sigma_centroid_px=sigma_px)
        m = combined_metrics([rb, (prod.h_x0, prod.h_k, prod.w)])
        noise.append(dict(sigma_px=sigma_px, f_perp=m["orthogonal_fraction"],
                          theta_k_deg=m["theta_k_deg"], sigma_k_frac=m["fractional_sigma_k"],
                          conditional_k_information=m["conditional_k_information"],
                          n_rows=prod.n_rows))
    out["noise_sweep"] = noise
    ideal = build_production_optical_arc(w15.nom48, w15.t_grid, et0, sigma_centroid_px=1e-3)
    out["unreachable_limit"] = combined_metrics([rb, (ideal.h_x0, ideal.h_k, ideal.w)])
    solo = build_production_optical_arc(w15.nom48, w15.t_grid, et0, sigma_centroid_px=0.25)
    out["optical_only"] = combined_metrics([(solo.h_x0, solo.h_k, solo.w)])
    out["optical_rows_sha"] = _sha(np.hstack([solo.h_x0, solo.h_k[:, None], solo.w[:, None]]))
    f_best = max([r["f_perp"] for r in noise] + [out["unreachable_limit"]["orthogonal_fraction"]])
    s_best = min(r["sigma_k_frac"] for r in noise)
    improvement = m_r["fractional_sigma_k"] / s_best
    step = improvement >= 3.0 and f_best >= 0.60
    marginal = (not step) and (f_best > 1.05 * m_r["orthogonal_fraction"] or improvement > 1.10)
    out.update(f_perp_best_combined=f_best, sigma_k_frac_best=s_best,
               improvement_factor=improvement,
               information_class=("DIRECTION_ROTATING_STEP_CHANGE" if step else
                                  "MARGINAL_DIRECTION_GAIN" if marginal else "MAGNITUDE_ONLY"),
               f_perp_ratio_best_over_range=f_best / m_r["orthogonal_fraction"])
    return _clean(out)


TASKS = (
    [("task_r1m_single", f) for f in FIXTURES]
    + [("task_r1m_multi", f) for f in FIXTURES]
    + [("task_r1o", f) for f in FIXTURES + ("OFFSET_ONLY", "CLAMP_ONLY")]
    + [("task_r1od", ("LEGACY", "FROZEN")), ("task_r1od", ("CANONICAL", "FROZEN")),
       ("task_r1od", ("CANONICAL", "MEASUREMENT_EPOCH"))]
    # s17-style attribution of the R1O-D reclassification (Earth at measurement epoch)
    + [("task_r1od", (f, "MEASUREMENT_EPOCH")) for f in ("LEGACY", "OFFSET_ONLY", "CLAMP_ONLY")]
    + [("task_r1oopt", f) for f in FIXTURES]
)


def _run(spec):
    name, arg = spec
    t0 = time.time()
    res = globals()[name](arg)
    res["elapsed_s"] = time.time() - t0
    return res


# ======================================================================
# Legacy-reproduction checks against the COMMITTED historical artifacts
# ======================================================================
def _eq(a, b) -> bool:
    return a == b


def repro_r1m(single: dict, multi: dict) -> dict:
    hist = json.loads((ARTIFACTS / "r1m_multi_arc_results.json").read_text())
    keys = ("observations", "orthogonal_fraction", "conditional_k_information", "rank",
            "weakest_mode_k_component", "raw_i_kk", "k_column_norm")
    bad = []
    for h, r in zip(hist["single_arc"], single["single_arc"]):
        bad += ["single %s %s" % (h["case"], k) for k in keys if not _eq(h[k], r[k])]
    for h in hist["multi_arc"]:
        r = next(x for x in multi["multi_arc"] if x["case"] == h["case"] and x["model"] == h["model"])
        bad += ["multi %s %s %s" % (h["case"], h["model"], k)
                for k in ("observations", "orthogonal_fraction", "conditional_k_information", "rank")
                if not _eq(h[k], r[k])]
    for h in hist["elapsed_control"]:
        r = next(x for x in multi["elapsed_control"] if x["case"] == h["case"])
        bad += ["control %s %s" % (h["case"], k) for k in
                ("observations", "contiguous_conditional_k", "linked_over_contiguous")
                if not _eq(h[k], r[k])]
    with (ARTIFACTS / "r1m_model_a_fairness.csv").open() as fh:
        for h in csv.DictReader(fh):
            r = next(x for x in multi["fairness"] if x["case"] == h["case"] and x["model"] == h["model"])
            bad += ["fair %s %s %s" % (h["case"], h["model"], k)
                    for k in ("conditional_k_information", "orthogonal_fraction")
                    if repr(r[k]) != h[k]]
            bad += ["fair %s %s observations" % (h["case"], h["model"])] if int(h["observations"]) != r["observations"] else []
    return dict(exact=not bad, mismatches=bad)


def repro_r1cov(single: dict, multi: dict) -> dict:
    bad = []
    with (ARTIFACTS / "r1cov_requalification.csv").open() as fh:
        for h in csv.DictReader(fh):
            r = next(x for x in single["r1cov_requalification"]
                     if x["case"] == h["case"] and x["prior"] == h["prior"])
            bad += ["%s %s %s" % (h["case"], h["prior"], k)
                    for k in ("old_sigma_k", "qualified_sigma_k", "oracle_sigma_k")
                    if repr(r[k]) != h[k]]
    hist = json.loads((ARTIFACTS / "r1cov_requalification.json").read_text())
    m3 = multi["m3_model_b_covariance"]
    bad += [k for k, hk in (("qualified_sigma_k", "best_existing_sigma_k"),
                            ("old_sigma_k", "best_existing_old_sigma_k"),
                            ("observations", "best_existing_observations"))
            if m3[k] != hist[hk]]
    return dict(exact=not bad, mismatches=bad)


def repro_r1o(r: dict) -> dict:
    hist = json.loads((ARTIFACTS / "r1o_r_requalification.json").read_text())
    pairs = [("range_only", hist["range_only"]),
             ("landmark_1urad", hist["landmark"]["corrected"]),
             ("ddor_surrogate_1nrad", hist["ddor"]["1nrad"]["corrected"]),
             ("ddor_surrogate_5nrad", hist["ddor"]["5nrad"]["corrected"]),
             ("earth_los_1urad", hist["earth_los"]["corrected"]),
             ("bridge_surrogate_5nrad", hist["r1od_bridge_surrogate"]["corrected"])]
    bad = ["%s %s" % (k, f) for k, h in pairs
           for f in ("orthogonal_fraction", "conditional_k_information", "fractional_sigma_k",
                     "n_obs") if r[k][f] != h[f]]
    return dict(exact=not bad, mismatches=bad)


def repro_r1od(r: dict) -> dict:
    hist = json.loads((ARTIFACTS / "r1od_campaign_summary.json").read_text())
    bad = ["%s %s" % (k, f) for k in ("range_only", "range_ddor_5nrad")
           for f in ("orthogonal_fraction", "conditional_k_information", "fractional_sigma_k",
                     "n_obs") if r[k][f] != hist[k][f]]
    bad += [k for k in ("n_dual_visible", "n_converged", "f_perp_gain",
                        "information_direction_class") if r[k] != hist[k]]
    deltas = {}
    with (ARTIFACTS / "r1od_random_noise_study.csv").open() as fh:
        for h, x in zip(csv.DictReader(fh), r["noise_study"]):
            for k in ("f_perp", "fractional_sigma_k"):
                if repr(x[k]) != h[k]:
                    tag = "noise %s %s" % (h["noise_nrad"], k)
                    bad.append(tag)
                    deltas[tag] = abs(x[k] - float(h[k]))
    with (ARTIFACTS / "r1od_systematic_residual_study.csv").open() as fh:
        row = [h for h in csv.DictReader(fh) if h["session_bias_s"] == "1e-09"][0]
    if repr(r["case_b_dk_per_second_bias"] * 1e-9) != row["case_b_induced_k_shift_m2_per_kg"]:
        bad.append("case_b_shift")
    if repr(r["case_c_sigma_k"]) != row["case_c_sigma_k_with_bias_nuisance"]:
        bad.append("case_c_sigma")
    # The published 5-nrad noise row and the published 5-nrad headline (summary
    # json) were computed by two different float expressions in the historical
    # script (python pow vs numpy power) and differ from EACH OTHER by 1 ulp.
    # The refactored weight helper reproduces the headline bit for bit.
    return dict(exact=not bad, mismatches=bad, abs_deltas=deltas,
                max_abs_delta=max(deltas.values()) if deltas else 0.0,
                headline_5nrad_bitwise=r["range_ddor_5nrad"]["orthogonal_fraction"]
                == hist["range_ddor_5nrad"]["orthogonal_fraction"])


def repro_r1oopt(r: dict) -> dict:
    hist = json.loads((ARTIFACTS / "r1o_opt_campaign.json").read_text())
    bad = ["range_only %s" % f for f in ("orthogonal_fraction", "fractional_sigma_k", "n_obs")
           if r["range_only"][f] != hist["range_only"][f]]
    for h, x in zip(hist["noise_sweep"], r["noise_sweep"]):
        bad += ["noise %s %s" % (h["sigma_px"], k) for k in ("f_perp", "sigma_k_frac")
                if x[k] != h[k]]
    bad += ["optical_only %s" % f for f in ("orthogonal_fraction", "n_obs")
            if r["optical_only"][f] != hist["optical_only"][f]]
    bad += [k for k in ("information_class", "improvement_factor", "f_perp_best_combined")
            if r[k] != hist[k]]
    for h in hist["cadence_family"]:
        x = next((c for c in r["cadence_family"] if c["cadence_s"] == h["cadence_s"]), None)
        if x is not None:
            bad += ["cadence %s %s" % (h["cadence_s"], k)
                    for k in ("f_perp_range_only", "f_perp_combined", "visible_obs") if x[k] != h[k]]
    return dict(exact=not bad, mismatches=bad)


# ======================================================================
# Classification and materiality
# ======================================================================
def materiality(margin_f_perp: float | None = None, margin_sigma_rel: float | None = None) -> dict:
    need_f = MATERIALITY_SAFETY_FACTOR * JACOBIAN_F_PERP_BOUND
    need_s = MATERIALITY_SAFETY_FACTOR * JACOBIAN_SIGMA_REL_BOUND
    ok = True
    if margin_f_perp is not None:
        ok = ok and abs(margin_f_perp) > need_f
    if margin_sigma_rel is not None:
        ok = ok and abs(margin_sigma_rel) > need_s
    return dict(margin_f_perp=margin_f_perp, margin_sigma_rel=margin_sigma_rel,
                required_f_perp=need_f, required_sigma_rel=need_s,
                status="CLASSIFIABLE" if ok else "JACOBIAN_LIMITED")


def r1m_verdict(single: dict, multi: dict) -> dict:
    s = {r["case"]: r for r in single["single_arc"]}
    mm = {(r["case"], r["model"]): r for r in multi["multi_arc"]}
    ff = {(r["case"], r["model"]): r for r in multi["fairness"]}
    f3a = ff[("F3", "A_independent_arcs")]["conditional_k_information"]
    g3 = s["G3"]["conditional_k_information"]
    m3b = mm[("M3", "B_continuity_linked")]["conditional_k_information"]
    all_rows = (list(s.values()) + list(mm.values()) + list(ff.values()))
    rank_def = all(r["rank"] < (r.get("unknowns") or 7) for r in all_rows)
    weak_k = min(r["weakest_mode_k_component"] for r in all_rows)
    fmax = max(r["orthogonal_fraction"] for r in all_rows)
    linked_best = min(r["sigma_k_data_only"] for (c, m), r in list(mm.items()) + list(ff.items())
                      if m == "B_continuity_linked") / K_TRUTH
    b_over_a = [mm[(c, "B_continuity_linked")]["conditional_k_information"]
                / max(mm[(c, "A_independent_arcs")]["conditional_k_information"], 1e-300)
                for c in ("M1", "M2", "M3")]
    b_over_a += [ff[(c, "B_continuity_linked")]["conditional_k_information"]
                 / max(ff[(c, "A_independent_arcs")]["conditional_k_information"], 1e-300)
                 for c in ("F2", "F3")]
    return dict(
        ordering_f3a_lt_g3_lt_m3b=bool(f3a < g3 < m3b),
        f3a=f3a, g3=g3, m3b=m3b,
        pure_multi_arc_worse_than_contiguous=bool(f3a < g3),
        linked_beats_pure_everywhere=bool(all(x > 1.0 for x in b_over_a)),
        linked_over_pure_ratios=b_over_a,
        normal_matrix_rank_deficient_everywhere=rank_def,
        min_weakest_mode_k_component=weak_k,
        max_f_perp=fmax,
        best_linked_sigma_k_frac=linked_best,
        practically_identifiable=bool(linked_best < 0.10),
    )


def main() -> None:
    _init()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(TASKS), initializer=_init) as ex:
        results = list(ex.map(_run, TASKS))
    print("all tasks done in %.0f s" % (time.time() - t0))
    R = {}
    for r in results:
        key = r["task"] + ("_" + r["anchor"] if "anchor" in r else "")
        R.setdefault(key, {})[r["fixture"]] = r
        print("  %-28s %-10s %6.0f s" % (key, r["fixture"], r["elapsed_s"]))

    L, K = "LEGACY", "CANONICAL"
    out = dict(phase="17-FXR", tasks=R)
    repro = dict(
        R1M=repro_r1m(R["r1m_single"][L], R["r1m_multi"][L]),
        R1COV=repro_r1cov(R["r1m_single"][L], R["r1m_multi"][L]),
        R1O=repro_r1o(R["r1o"][L]),
        R1OD=repro_r1od(R["r1od_FROZEN"][L]),
        R1OOPT=repro_r1oopt(R["r1oopt"][L]),
    )
    out["legacy_reproduction"] = repro
    print("\nLEGACY REPRODUCTION vs committed artifacts")
    for k, v in repro.items():
        print("  %-8s exact=%s %s" % (k, v["exact"], v["mismatches"][:6]))

    # nom48 must not depend on the Earth fixture
    truth_same = {
        t: R[t][L]["nom48_sha"] == R[t][K]["nom48_sha"]
        for t in ("r1m_single", "r1o", "r1oopt")}
    out["nominal_trajectory_fixture_independent"] = truth_same

    # ---------------- R1M ----------------
    v_l = r1m_verdict(R["r1m_single"][L], R["r1m_multi"][L])
    v_c = r1m_verdict(R["r1m_single"][K], R["r1m_multi"][K])
    out["r1m_verdict"] = dict(legacy=v_l, canonical=v_c)

    # ---------------- R1O ----------------
    def r1o_gains(r):
        base = r["range_only"]["orthogonal_fraction"]
        return {k: r[k]["orthogonal_fraction"] - base
                for k in ("landmark_1urad", "ddor_surrogate_1nrad", "ddor_surrogate_5nrad",
                          "earth_los_1urad", "bridge_surrogate_5nrad")}
    g_l, g_c = r1o_gains(R["r1o"][L]), r1o_gains(R["r1o"][K])
    out["r1o_gains"] = dict(legacy=g_l, canonical=g_c)
    out["r1o_materiality"] = {k: materiality(0.02 - v) for k, v in g_c.items()}

    # ---------------- R1O-D ----------------
    od = {("LEGACY", "FROZEN"): R["r1od_FROZEN"][L], ("CANONICAL", "FROZEN"): R["r1od_FROZEN"][K],
          ("CANONICAL", "MEASUREMENT_EPOCH"): R["r1od_MEASUREMENT_EPOCH"][K]}
    out["r1od_materiality"] = {"%s/%s" % k: materiality(0.02 - v["f_perp_gain"])
                               for k, v in od.items()}
    att = {f: R["r1od_MEASUREMENT_EPOCH"][f] for f in ("LEGACY", "OFFSET_ONLY", "CLAMP_ONLY",
                                                       "CANONICAL")}
    g = {f: v["f_perp_gain"] for f, v in att.items()}
    out["r1od_attribution"] = dict(
        gains=g, classes={f: v["information_direction_class"] for f, v in att.items()},
        ddor_only_f_perp={f: v["ddor_only_5nrad"]["orthogonal_fraction"] for f, v in att.items()},
        range_f_perp={f: v["range_only"]["orthogonal_fraction"] for f, v in att.items()},
        epoch_repair_given_no_clamp=g["CANONICAL"] - g["OFFSET_ONLY"],
        epoch_repair_given_clamp=g["CLAMP_ONLY"] - g["LEGACY"],
        clamp_repair_given_correct_epoch=g["CANONICAL"] - g["CLAMP_ONLY"],
        clamp_repair_given_offset_epoch=g["OFFSET_ONLY"] - g["LEGACY"],
        frozen_anchor_effect_legacy=R["r1od_FROZEN"]["LEGACY"]["f_perp_gain"] - g["LEGACY"],
        frozen_anchor_effect_canonical=R["r1od_FROZEN"]["CANONICAL"]["f_perp_gain"] - g["CANONICAL"])
    r1o_att = {f: r1o_gains(R["r1o"][f]) for f in ("LEGACY", "OFFSET_ONLY", "CLAMP_ONLY",
                                                   "CANONICAL")}
    out["r1o_attribution"] = r1o_att

    # ---------------- R1O-OPT ----------------
    op = R["r1oopt"]
    out["r1oopt_materiality"] = {
        f: materiality(op[f]["f_perp_best_combined"] - 1.05 * op[f]["range_only"]["orthogonal_fraction"],
                       (op[f]["improvement_factor"] - 1.10) / 1.10)
        for f in (L, K)}

    (ARTIFACTS / "phase17_fxr_requalification.json").write_text(
        json.dumps(out, indent=1, default=float))
    print("\nwrote studies/phase17/results/phase17_fxr_requalification.json")

    # ---------------- printed summary ----------------
    print("\nR1M single arcs (f_perp, I_K|x, QR sigma_K/K, n, station)")
    for f in (L, K):
        for r in R["r1m_single"][f]["single_arc"]:
            print("  %-9s %s  n=%4d f=%.6f I=%.4e sig/K=%9.4f  %s  cond(A)=%.2e rank(info)=%d design_rank=%d"
                  % (f, r["case"], r["observations"], r["orthogonal_fraction"],
                     r["conditional_k_information"], r["qr_sigma_k"] / K_TRUTH, r["station"],
                     r["cond_design"], r["rank"], r["design_rank"]))
    print("\nR1M multi-arc")
    for f in (L, K):
        for r in R["r1m_multi"][f]["multi_arc"] + R["r1m_multi"][f]["fairness"]:
            print("  %-9s %s %-20s n=%4d f=%.6f I=%.4e sig/K=%.4f rank=%d windowsA=%s"
                  % (f, r["case"], r["model"], r["observations"], r["orthogonal_fraction"],
                     r["conditional_k_information"],
                     (r["sigma_k_data_only"] / K_TRUTH), r["rank"],
                     r.get("per_window_obs_model_a")))
        for r in R["r1m_multi"][f]["elapsed_control"]:
            print("  %-9s %s contiguous n=%d I=%.4e f=%.6f linked/contig=%.3f"
                  % (f, r["case"], r["observations"], r["contiguous_conditional_k"],
                     r["contiguous_f_perp"], r["linked_over_contiguous"]))
        print("  %-9s M3B qualified covariance %s" % (f, R["r1m_multi"][f]["m3_model_b_covariance"]))
    print("  verdict legacy   :", v_l)
    print("  verdict canonical:", v_c)
    print("\nR1COV")
    for f in (L, K):
        for r in R["r1m_single"][f]["r1cov_requalification"]:
            print("  %-9s %s %-22s old=%.4e qual=%.6e oracle=%.6e rel=%.1e sig/K=%.4f"
                  % (f, r["case"], r["prior"], r["old_sigma_k"], r["qualified_sigma_k"],
                     r["oracle_sigma_k"], r["qualified_relative_error"], r["sigma_k_over_k_truth"]))
    print("\nR1O (corrected STM)")
    for f in (L, K):
        r = R["r1o"][f]
        for k in ("range_only", "landmark_1urad", "ddor_surrogate_1nrad", "ddor_surrogate_5nrad",
                  "earth_los_1urad", "bridge_surrogate_5nrad"):
            m = r[k]
            print("  %-9s %-24s n=%5d f=%.6f th=%7.3f I=%.4e sig/K=%.4f%%"
                  % (f, k, m["n_obs"], m["orthogonal_fraction"], m["theta_k_deg"],
                     m["conditional_k_information"], 100 * m["fractional_sigma_k"]))
        print("  %-9s STM equivalence %s" % (f, r["stm_historical_equivalence"]))
    print("  gains legacy   :", g_l)
    print("  gains canonical:", g_c)
    print("\nR1O-D")
    for k, r in od.items():
        print("  %-30s n_ddor=%d range f=%.6f  +DDOR5 f=%.6f gain=%+.6f sig/K %.4f%% -> %.4f%%  %s  below@all=%s  CaseB(1ns)=%.4f"
              % ("/".join(k), r["n_converged"], r["range_only"]["orthogonal_fraction"],
                 r["range_ddor_5nrad"]["orthogonal_fraction"], r["f_perp_gain"],
                 100 * r["range_only"]["fractional_sigma_k"],
                 100 * r["range_ddor_5nrad"]["fractional_sigma_k"],
                 r["information_direction_class"], r["below_range_at_every_precision"],
                 r["case_b_shift_fraction_at_1ns"]))
        print("      noise:", [(x["noise_nrad"], round(x["f_perp"], 4)) for x in r["noise_study"]])
    print("\nR1O-D ATTRIBUTION (Earth at measurement epoch):")
    for k, v in out["r1od_attribution"].items():
        print("  %-36s %s" % (k, v))
    print("\nR1O ATTRIBUTION (gains vs range-only):")
    for f, v in out["r1o_attribution"].items():
        print("  %-12s %s" % (f, {k: round(x, 5) for k, x in v.items()}))
    print("\nR1O-OPT")
    for f in (L, K):
        r = op[f]
        print("  %-9s station=%s range n=%d f=%.6f sig/K=%.4f%% | opt-only f=%.6f | best comb f=%.6f"
              " | improvement=%.4fx | %s"
              % (f, r["station"], r["range_only"]["n_obs"], r["range_only"]["orthogonal_fraction"],
                 100 * r["range_only"]["fractional_sigma_k"], r["optical_only"]["orthogonal_fraction"],
                 r["f_perp_best_combined"], r["improvement_factor"], r["information_class"]))
        print("      noise:", [(x["sigma_px"], round(x["f_perp"], 6), round(100 * x["sigma_k_frac"], 4))
                               for x in r["noise_sweep"]])
    print("  optical rows fixture-independent:",
          op[L]["optical_rows_sha"] == op[K]["optical_rows_sha"])
    print("\nnominal trajectory fixture-independent:", truth_same)


if __name__ == "__main__":
    main()
