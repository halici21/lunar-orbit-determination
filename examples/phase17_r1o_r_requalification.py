"""PHASE 17-R1O-R s13-s22 -- historical reproduction and corrected requalification.

ANALYSIS SPACE ONLY.  No production file is modified by this script.

WHAT IT DOES
------------
Phase 17-R1O published a cross-observable ranking whose surrogate state
Jacobians were built from Phi^T, because `chain_to_augmented_columns`
unflattened the column-major STM block with NumPy's default C order
(s7/s8, oracle in `phase17_r1o_r_stm_oracle.py`).  That single line is now
repaired.  This script answers the only question the repair raises:

    what should Phase 17-R1O actually have concluded?

It runs every affected analysis TWICE on ONE arc, changing nothing but the
STM unpacking order:

  HISTORICAL  the defective path, reproduced exactly (s14) -- proving this
              script really is re-running what produced the published numbers
  CORRECTED   the identical analysis with Phi unflattened column-major (s15)

HOW THE HISTORICAL PATH IS REPRODUCED WITHOUT UN-REPAIRING THE CODE
-------------------------------------------------------------------
For a column-major flattening v of Phi,

    v[i + 6j] = Phi[i, j]                  (pack,  order="F")
    C-order unpack of v                 == Phi^T

so the defective helper returned Phi^T.  Feeding the REPAIRED helper a row
whose [6:42] block has been replaced by vec_F(Phi^T) makes it return Phi^T
too -- identical arithmetic, reached honestly.  `_assert_historical_equivalence`
below verifies this against the literal old expression rather than assuming it.

The arc, trajectory, landmark set, station pair, epochs, noise levels, K truth
and sampling are IDENTICAL between the two runs (s15).  Only Phi's layout
differs, so every delta reported here is attributable to the defect alone.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from phase17_r1m_core import (  # noqa: E402
    K_TRUTH, build_range_arc, campaign_epoch,
)
from phase17_r1o_core import (  # noqa: E402
    build_ddor_arc, build_landmark_arc, chain_to_augmented_columns,
    combined_metrics, position_jacobian_fd,
)

ARTIFACTS = HERE.parent / "artifacts"

# ---- R1O's OWN published configuration, copied from its scripts -----------
# phase17_r1o_landmark.py s53/s87, phase17_r1o_ddor.py s48/s83,
# phase17_r1o_celestial.py s58: all three published headlines were computed on
# ONE shared arc -- 15 orbits, all three DSN stations, 90 s cadence.
W15_ORBITS = 15.0
ALL_STATIONS = ("Goldstone DSN", "Madrid DSN", "Canberra DSN")
CADENCE_S = 90.0
DDOR_PAIR = ("Goldstone DSN", "Canberra DSN")

LANDMARK_HEADLINE_URAD = 1.0      # R1O's published landmark headline
DDOR_HEADLINE_NRAD = 1.0          # R1O's published DDOR headline
DDOR_PRODUCTION_NRAD = 5.0        # the level R1O-D qualified production DDOR at
EARTH_LOS_URAD = 1.0              # negative control, at the same optimistic level


def hdr(t: str) -> None:
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


# ======================================================================
# s14 -- exact reproduction of the historical (defective) path
# ======================================================================
def historical_nom48(nom48: np.ndarray) -> np.ndarray:
    """nom48 with each Phi block replaced by vec_F(Phi^T).

    Makes the REPAIRED helper reproduce the defective result bit for bit,
    without reintroducing the defect into the shared analysis code.
    """
    out = np.array(nom48, dtype=float, copy=True)
    for i in range(out.shape[0]):
        phi = out[i, 6:42].reshape((6, 6), order="F")
        out[i, 6:42] = phi.T.reshape(-1, order="F")
    return out


def _assert_historical_equivalence(nom48: np.ndarray, hist48: np.ndarray) -> dict:
    """Verify the reproduction against the LITERAL pre-repair expression.

    The defective line was:  phi = nom48_row[6:42].reshape(6, 6)   (C order)
    followed by             h_x0 = dg_dr @ phi[:3, :].
    """
    rng = np.random.default_rng(20260922)
    dg_dr = rng.normal(size=(2, 3))
    worst_x0, worst_k = 0.0, 0.0
    for i in (0, nom48.shape[0] // 3, nom48.shape[0] // 2, nom48.shape[0] - 1):
        phi_defective = np.asarray(nom48[i, 6:42], dtype=float).reshape(6, 6)
        expected_x0 = dg_dr @ phi_defective[:3, :]
        expected_k = dg_dr @ np.asarray(nom48[i, 42:48], dtype=float)[:3]
        got_x0, got_k = chain_to_augmented_columns(dg_dr, hist48[i])
        worst_x0 = max(worst_x0, float(np.max(np.abs(got_x0 - expected_x0))))
        worst_k = max(worst_k, float(np.max(np.abs(got_k - expected_k))))
    return {"max_abs_state_diff": worst_x0, "max_abs_k_diff": worst_k,
            "bitwise": worst_x0 == 0.0 and worst_k == 0.0}


# ======================================================================
# s19 -- Earth-center LOS negative control (R1O's own construction)
# ======================================================================
def earth_los_rows(nom48: np.ndarray, t_grid: np.ndarray) -> tuple:
    import od_gravity_covariance_campaign as C  # noqa: N811

    rows_x0, rows_k, max_conv = [], [], 0.0
    for i in range(0, t_grid.size, 4):      # R1O sampled every 4th epoch
        r_sc = nom48[i, :3]
        r_earth = np.asarray(C.get_earth_pos(t_grid[i]), float).reshape(3)
        e1 = np.array([0.0, 1.0, 0.0])
        e2 = np.array([0.0, 0.0, 1.0])

        def g_fn(r, _re=r_earth):
            rho = _re - r
            rho_h = rho / np.linalg.norm(rho)
            return np.array([np.arcsin(np.clip(np.dot(rho_h, e1), -1, 1)),
                             np.arcsin(np.clip(np.dot(rho_h, e2), -1, 1))])

        dg_dr, rel = position_jacobian_fd(g_fn, r_sc)
        max_conv = max(max_conv, rel)
        h_x0, h_k = chain_to_augmented_columns(dg_dr, nom48[i])
        rows_x0.append(h_x0[0]); rows_k.append(h_k[0])
        rows_x0.append(h_x0[1]); rows_k.append(h_k[1])
    return np.asarray(rows_x0), np.asarray(rows_k), max_conv


# ======================================================================
# s20 -- R1O-D bridge, surrogate side only (production DDOR untouched)
# ======================================================================
def bridge_reduced_rows(nom48: np.ndarray, t_grid: np.ndarray, et0: float,
                        station_a, station_b, min_elevation_deg: float = 10.0):
    """R1O-D's `reduced_simultaneous_ddor_rows`, re-run through whichever
    nom48 it is handed.  Production ΔDOR is NOT touched by this phase.
    """
    from lunar_od.geometry import ecef2razel_sez
    from lunar_od.measurements import _station_relative_state_j2000_at_receive_epoch
    from lunar_od.visibility import sample_j2000_to_itrf93_transforms
    import od_gravity_covariance_campaign as C  # noqa: N811

    t_local = t_grid - t_grid[0]
    xf = sample_j2000_to_itrf93_transforms(et0 + t_grid[0], t_local)
    earth_pos_fixed = np.asarray(C.get_earth_pos(0.0), float).reshape(3)

    def station_pos_at(station, i):
        rel = _station_relative_state_j2000_at_receive_epoch(station, xf[i])
        return earth_pos_fixed + rel[:3]

    def elevation_deg(station, r_sc, xfi):
        r_rel = xfi[:3, :3] @ (r_sc - earth_pos_fixed) \
            - np.asarray(station.r_ecef_m, float).reshape(3)
        _, el, _ = ecef2razel_sez(r_rel, station.lat_rad, station.lon_rad)
        return np.degrees(el)

    rows_x0, rows_k = [], []
    for i in range(t_grid.size):
        r_sc = nom48[i, :3]
        if elevation_deg(station_a, r_sc, xf[i]) < min_elevation_deg:
            continue
        if elevation_deg(station_b, r_sc, xf[i]) < min_elevation_deg:
            continue
        r_a, r_b = station_pos_at(station_a, i), station_pos_at(station_b, i)

        def g_fn(r, _ra=r_a, _rb=r_b):
            return np.array([(np.linalg.norm(r - _rb)
                              - np.linalg.norm(r - _ra)) / 299792458.0])

        dg_dr, _ = position_jacobian_fd(g_fn, r_sc)
        h_x0, h_k = chain_to_augmented_columns(dg_dr, nom48[i])
        rows_x0.append(h_x0[0]); rows_k.append(h_k[0])
    return np.asarray(rows_x0), np.asarray(rows_k)


def _fmt(m: dict) -> str:
    return ("f_perp=%.6f  theta_K=%.4f deg  I_K|x=%.6e  sigma_K/K=%.6f%%"
            % (m["orthogonal_fraction"], m["theta_k_deg"],
               m["conditional_k_information"], 100 * m["fractional_sigma_k"]))


def _row(label: str, m: dict, source: str, classification: str) -> dict:
    return dict(
        case=label, conditional_k_information=m["conditional_k_information"],
        f_perp=m["orthogonal_fraction"], theta_k_deg=m["theta_k_deg"],
        sigma_k_frac=m["fractional_sigma_k"], n_obs=m["n_obs"],
        state_jacobian_source=source, classification=classification)


def main() -> None:
    from lunar_od.spice_loader import load_spice_kernels
    from lunar_od.constants import R_MOON_M
    import od_gravity_covariance_campaign as C  # noqa: N811

    load_spice_kernels(None, clear=True)
    et0, t_orbit = campaign_epoch()

    hdr("PHASE 17-R1O-R -- SHARED ARC (R1O's own published configuration)")
    print("  15 orbits, all three DSN stations, 90 s cadence "
          "(phase17_r1o_landmark.py s53/s87)")
    w15 = build_range_arc(0.0, W15_ORBITS * t_orbit, label="W15_range",
                          station_filter=ALL_STATIONS, cadence_s=CADENCE_S)
    print("  range observations: %d   trajectory samples: %d"
          % (w15.n_obs, w15.t_grid.size))

    nom_corrected = w15.nom48
    nom_historical = historical_nom48(nom_corrected)

    equiv = _assert_historical_equivalence(nom_corrected, nom_historical)
    print("  historical-path equivalence vs the literal pre-repair expression:")
    print("    max |state column diff| = %.3e   max |K column diff| = %.3e   "
          "bitwise=%s" % (equiv["max_abs_state_diff"], equiv["max_abs_k_diff"],
                          equiv["bitwise"]))
    if not equiv["bitwise"]:
        raise SystemExit("historical reproduction is not exact -- STOP (s14)")

    range_block = (w15.h_x0, w15.h_k, w15.w)
    m_range = combined_metrics([range_block])
    print("\n  RANGE-ONLY BASELINE (production Jacobian, unaffected): %s"
          % _fmt(m_range))
    print("  R1O published range-only f_perp = 0.2523, sigma_K/K = 9.27%")

    results = {"range_only": m_range, "historical_equivalence": equiv}
    table = [_row("range only (production two-way range)", m_range,
                  "production (order=F, always correct)", "baseline")]

    # ------------------------------------------------------------------
    hdr("s13 -- K-COLUMN INVARIANCE UNDER THE REPAIR")
    lm_hist = build_landmark_arc(nom_historical, w15.t_grid, et0,
                                 sigma_angle_rad=LANDMARK_HEADLINE_URAD * 1e-6,
                                 r_moon_m=R_MOON_M, label="lm_hist")
    lm_corr = build_landmark_arc(nom_corrected, w15.t_grid, et0,
                                 sigma_angle_rad=LANDMARK_HEADLINE_URAD * 1e-6,
                                 r_moon_m=R_MOON_M, label="lm_corr")
    k_bitwise = bool(np.array_equal(lm_hist.h_k, lm_corr.h_k))
    state_changed = not np.allclose(lm_hist.h_x0, lm_corr.h_x0)
    print("  landmark K column identical (bitwise): %s" % k_bitwise)
    print("  landmark state columns changed:        %s" % state_changed)
    print("  reason: S_K is a 6-vector in columns [42:48]; it is never")
    print("          reshaped, so no storage-order convention applies to it.")
    results["k_column_invariance"] = {
        "landmark_k_bitwise_identical": k_bitwise,
        "landmark_state_columns_changed": bool(state_changed),
        "max_abs_k_diff": float(np.max(np.abs(lm_hist.h_k - lm_corr.h_k))),
    }
    if not k_bitwise:
        raise SystemExit("K column changed under the repair -- STOP (s13)")

    # ------------------------------------------------------------------
    hdr("s14/s15/s16 -- LANDMARK: HISTORICAL vs CORRECTED (1 urad)")
    m_lm_hist = combined_metrics([range_block,
                                  (lm_hist.h_x0, lm_hist.h_k, lm_hist.w)])
    m_lm_corr = combined_metrics([range_block,
                                  (lm_corr.h_x0, lm_corr.h_k, lm_corr.w)])
    print("  landmark rows: %d (identical in both runs)" % lm_hist.n_obs)
    print("  HISTORICAL (Phi^T, as published): %s" % _fmt(m_lm_hist))
    print("  R1O published headline:           f_perp=0.8757, sigma_K/K=0.19%")
    print("  CORRECTED  (Phi,  order='F'):     %s" % _fmt(m_lm_corr))
    d_stm = m_lm_corr["orthogonal_fraction"] - m_lm_hist["orthogonal_fraction"]
    print("\n  delta f_perp attributable to the STM repair alone: %+.6f" % d_stm)
    results["landmark"] = {"historical": m_lm_hist, "corrected": m_lm_corr,
                           "delta_f_perp_stm": d_stm, "n_rows": lm_hist.n_obs}
    table.append(_row("landmark surrogate 1 urad -- HISTORICAL", m_lm_hist,
                      "R1O surrogate (Phi^T -- DEFECTIVE)", "superseded"))
    table.append(_row("landmark surrogate 1 urad -- CORRECTED", m_lm_corr,
                      "R1O surrogate (Phi, order=F)", "corrected"))

    # ------------------------------------------------------------------
    hdr("s17/s18 -- DDOR SURROGATE: HISTORICAL vs CORRECTED")
    results["ddor"] = {}
    for nrad in (DDOR_HEADLINE_NRAD, DDOR_PRODUCTION_NRAD):
        sig = nrad * 1e-9
        d_hist = build_ddor_arc(nom_historical, w15.t_grid, et0, DDOR_PAIR,
                                sigma_angle_rad=sig, label="ddor_hist")
        d_corr = build_ddor_arc(nom_corrected, w15.t_grid, et0, DDOR_PAIR,
                                sigma_angle_rad=sig, label="ddor_corr")
        m_h = combined_metrics([range_block, (d_hist.h_x0, d_hist.h_k, d_hist.w)])
        m_c = combined_metrics([range_block, (d_corr.h_x0, d_corr.h_k, d_corr.w)])
        k_same = bool(np.array_equal(d_hist.h_k, d_corr.h_k))
        print("  %.0f nrad, rows=%d, K column bitwise identical=%s"
              % (nrad, d_hist.n_obs, k_same))
        print("    HISTORICAL: %s" % _fmt(m_h))
        print("    CORRECTED : %s" % _fmt(m_c))
        print("    delta f_perp from STM repair: %+.6f"
              % (m_c["orthogonal_fraction"] - m_h["orthogonal_fraction"]))
        results["ddor"]["%gnrad" % nrad] = {
            "historical": m_h, "corrected": m_c, "k_bitwise": k_same,
            "n_rows": d_hist.n_obs}
        table.append(_row("DDOR surrogate %g nrad -- HISTORICAL" % nrad, m_h,
                          "R1O surrogate (Phi^T -- DEFECTIVE)", "superseded"))
        table.append(_row("DDOR surrogate %g nrad -- CORRECTED" % nrad, m_c,
                          "R1O surrogate (Phi, order=F)", "corrected"))

    # ------------------------------------------------------------------
    hdr("s19 -- EARTH-CENTER LOS NEGATIVE CONTROL: HISTORICAL vs CORRECTED")
    e_hx_h, e_hk_h, conv_h = earth_los_rows(nom_historical, w15.t_grid)
    e_hx_c, e_hk_c, conv_c = earth_los_rows(nom_corrected, w15.t_grid)
    w_e = np.full(e_hk_h.size, 1.0 / (EARTH_LOS_URAD * 1e-6) ** 2)
    m_e_h = combined_metrics([range_block, (e_hx_h, e_hk_h, w_e)])
    m_e_c = combined_metrics([range_block, (e_hx_c, e_hk_c, w_e)])
    print("  Earth-LOS rows: %d   K column bitwise identical: %s"
          % (e_hk_h.size, bool(np.array_equal(e_hk_h, e_hk_c))))
    print("  HISTORICAL: %s" % _fmt(m_e_h))
    print("  CORRECTED : %s" % _fmt(m_e_c))
    results["earth_los"] = {"historical": m_e_h, "corrected": m_e_c,
                            "n_rows": int(e_hk_h.size),
                            "fd_conv_hist": conv_h, "fd_conv_corr": conv_c}
    table.append(_row("Earth-center LOS %g urad -- HISTORICAL" % EARTH_LOS_URAD,
                      m_e_h, "R1O surrogate (Phi^T -- DEFECTIVE)", "superseded"))
    table.append(_row("Earth-center LOS %g urad -- CORRECTED" % EARTH_LOS_URAD,
                      m_e_c, "R1O surrogate (Phi, order=F)", "corrected"))

    # ------------------------------------------------------------------
    hdr("s20 -- R1O-D BRIDGE, SURROGATE SIDE ONLY")
    by_name = {s.name: s for s in C.STATIONS}
    st_a, st_b = by_name[DDOR_PAIR[0]], by_name[DDOR_PAIR[1]]
    b_hx_h, b_hk_h = bridge_reduced_rows(nom_historical, w15.t_grid, et0, st_a, st_b)
    b_hx_c, b_hk_c = bridge_reduced_rows(nom_corrected, w15.t_grid, et0, st_a, st_b)
    sig_b = DDOR_PRODUCTION_NRAD * 1e-9
    w_b = np.full(b_hk_h.size, 1.0 / sig_b ** 2 * (1e7 / 299792458.0) ** 2)
    m_b_h = combined_metrics([range_block, (b_hx_h, b_hk_h, w_b)])
    m_b_c = combined_metrics([range_block, (b_hx_c, b_hk_c, w_b)])
    print("  reduced-mode rows: %d   K column bitwise identical: %s"
          % (b_hk_h.size, bool(np.array_equal(b_hk_h, b_hk_c))))
    print("  HISTORICAL: %s" % _fmt(m_b_h))
    print("  CORRECTED : %s" % _fmt(m_b_c))
    print("  NOTE: production ΔDOR (lunar_od/delta_dor.py) is NOT touched by")
    print("        this phase -- only the surrogate side of the bridge moves.")
    results["r1od_bridge_surrogate"] = {
        "historical": m_b_h, "corrected": m_b_c, "n_rows": int(b_hk_h.size)}

    # ------------------------------------------------------------------
    hdr("s22 -- CORRECTED CROSS-OBSERVABLE TABLE")
    print("  %-46s %14s %9s %9s %11s" %
          ("case", "I_K|x", "f_perp", "theta_K", "sigma_K/K"))
    for r in table:
        print("  %-46s %14.6e %9.6f %9.4f %10.4f%%"
              % (r["case"], r["conditional_k_information"], r["f_perp"],
                 r["theta_k_deg"], 100 * r["sigma_k_frac"]))
    results["table"] = table

    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "r1o_r_requalification.json").write_text(
        json.dumps(results, indent=2, default=float))
    print("\nwrote %s" % (ARTIFACTS / "r1o_r_requalification.json"))


if __name__ == "__main__":
    main()
