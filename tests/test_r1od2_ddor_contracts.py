"""Phase 17-R1O-D2 - contracts of the corrected-geometry ΔDOR analysis harness.

Protects what later phases will build on:

* the D2 range and ΔDOR builders reproduce the FXR / R1O-D paths bit for bit
  (so D2 numbers are comparable with the published FXR reclassification);
* the D2 reference ΔDOR row rule requires BOTH stations to actually see the
  spacecraft (elevation AND lunar occultation), which the historical R1O-D rule
  did not;
* the legacy campaign Earth table cannot leak into D2 science;
* a nuisance prior enters the direction metric as information (a partly
  constrained session bias absorbs less of the K signature than a free one);
* operational subsets keep each normal point's parent session.

Requires the frozen campaign data and SPICE kernels; skipped where absent.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import phase17_r1m_core as M  # noqa: E402

CAMPAIGN_SCRIPTS = M.CAMPAIGN_ROOT / "06_scripts"
pytestmark = pytest.mark.skipif(
    not (CAMPAIGN_SCRIPTS / "od_gravity_covariance_campaign.py").exists(),
    reason="frozen od_covariance_campaign data/scripts not present in this checkout",
)


@pytest.fixture(scope="module")
def canon():
    from lunar_od.spice_loader import load_spice_kernels
    if str(CAMPAIGN_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(CAMPAIGN_SCRIPTS))
    load_spice_kernels(None, clear=True)
    import phase17_r1od2_core as D2
    scn = D2.canonical_scenario()
    nom = D2.propagate_nom48(scn)
    rb = D2.build_range_block(scn, nom)
    rows_hist = D2.build_ddor_rows(scn, rb, "GC", D2.ROW_RULE_ELEVATION_ONLY)
    rows = D2.build_ddor_rows(scn, rb, "GC", D2.ROW_RULE_PRODUCTION_VISIBILITY)
    return D2, scn, rb, rows_hist, rows


def test_range_block_reproduces_fxr_canonical_arc(canon):
    D2, scn, rb, _, _ = canon
    ref = M.build_range_arc(0.0, scn.duration_s, label="ref", station_filter=D2.DSN3, cadence_s=90.0)
    for k in ("h_x0", "h_k", "w", "obs"):
        np.testing.assert_array_equal(getattr(rb, k), getattr(ref, k))
    assert rb.earth.fixture == M.EARTH_FIXTURE_CANONICAL
    assert D2.earth_epoch_audit(scn, rb)["passed"]


def test_historical_rows_and_weights_reproduce_r1od_campaign(canon):
    D2, scn, rb, rows_hist, _ = canon
    import phase17_r1od_campaign as OD
    from lunar_od.delta_dor import QuasarDirection
    ref = M.build_range_arc(0.0, scn.duration_s, label="ref", station_filter=D2.DSN3, cadence_s=90.0)
    by = D2.stations_by_name()
    sa, sb = by["Goldstone DSN"], by["Canberra DSN"]
    hist = OD.build_ddor_rows(ref, sa, sb, QuasarDirection(2.1, -0.3),
                              earth_anchor=OD.EARTH_ANCHOR_MEASUREMENT_EPOCH)
    np.testing.assert_array_equal(rows_hist.h_x0, hist["h_x0"])
    np.testing.assert_array_equal(rows_hist.h_k, hist["h_k"])
    np.testing.assert_array_equal(D2.ddor_weights(rows_hist, D2.SEMANTICS_ANGULAR, 5e-9),
                                  OD.ddor_delay_weights(ref, hist, sa, sb, 5e-9))
    assert rows_hist.n == 263


def test_reference_rows_need_both_stations_to_see_the_spacecraft(canon):
    D2, _, rb, rows_hist, rows = canon
    from lunar_od.constants import R_MOON_M
    from phase17_r1o_core import station_position_mci
    by = D2.stations_by_name()
    assert set(rows.idx) <= set(rows_hist.idx)
    assert rows.n < rows_hist.n
    ia, ib = D2.DSN3.index("Goldstone DSN"), D2.DSN3.index("Canberra DSN")
    assert np.all(rb.vis[rows.idx, ia] & rb.vis[rows.idx, ib])

    def occulted(i, station):
        r_st = station_position_mci(station, rows_hist.earth_pos_hist[i], rb.xf[i])
        d = rb.nom48[i, :3] - r_st
        tt = np.clip(-np.dot(r_st, d) / np.dot(d, d), 0.0, 1.0)
        return np.linalg.norm(r_st + tt * d) < R_MOON_M

    occ = [i for i in rows.idx if occulted(i, by["Goldstone DSN"]) or occulted(i, by["Canberra DSN"])]
    assert occ == []
    hist_occ = [i for i in rows_hist.idx if occulted(i, by["Goldstone DSN"]) or occulted(i, by["Canberra DSN"])]
    assert len(hist_occ) > 0          # the historical rule did admit occulted epochs


def test_legacy_earth_table_cannot_leak(canon):
    D2 = canon[0]
    import od_gravity_covariance_campaign as C  # noqa: N811
    saved = C.get_earth_pos, C.get_earth_vel
    try:
        D2.install_legacy_earth_guard()
        with pytest.raises(D2.LegacyEarthLeakError):
            C.get_earth_pos(0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(D2.LegacyEarthLeakError):
                M.earth_position_mci(0.0, M.EARTH_FIXTURE_LEGACY_OFFSET_CLAMPED)
        np.testing.assert_array_equal(M.earth_position_mci(0.0), M.earth_position_mci(0.0))
    finally:
        C.get_earth_pos, C.get_earth_vel = saved


def test_nuisance_prior_enters_direction_metric(canon):
    D2, _, rb, _, rows = canon
    w = D2.ddor_weights(rows, D2.SEMANTICS_ANGULAR, 5e-9)
    blocks = [(rb.h_x0, rb.h_k, rb.w), (rows.h_x0, rows.h_k, w)]
    nuis = D2.session_nuisance_columns(rb.n_obs, rows)
    f0 = D2.info_metrics(blocks)["f_perp"]
    f_free = D2.info_metrics(blocks, nuisance=nuis)["f_perp"]
    f_prior = D2.info_metrics(blocks, nuisance=nuis, nuisance_prior_sigma=0.033e-9)["f_perp"]
    f_tight = D2.info_metrics(blocks, nuisance=nuis, nuisance_prior_sigma=1e-15)["f_perp"]
    assert f_free <= f_prior <= f0
    assert f_tight == pytest.approx(f0, rel=1e-6)      # a known bias removes nothing


def test_operational_subset_keeps_parent_sessions(canon):
    D2, _, rb, _, rows = canon
    sel = D2.ops_selection(rows, "OPS_EVERY_WINDOW", 0.5 * float(rb.t_grid[-1]))
    sub = D2.subset(rows, sel)
    assert sub.n == sel.size and sub.n % 3 == 0
    assert sub.n_sessions == sub.n // 3
    for s in range(sub.n_sessions):
        assert len(set(rows.session[sel][sub.session == s])) == 1


def test_design_metrics_match_the_fxr_metric(canon):
    D2, _, rb, _, _ = canon
    from phase17_r1o_core import combined_metrics
    m = D2.info_metrics([(rb.h_x0, rb.h_k, rb.w)])
    ref = combined_metrics([(rb.h_x0, rb.h_k, rb.w)])
    assert m["f_perp"] == pytest.approx(ref["orthogonal_fraction"], rel=1e-12)
    assert m["sigma_k"] == pytest.approx(ref["qualified_sigma_k"], rel=1e-12)
    assert m["design_rank"] == 7            # full rank from the DESIGN, not rank(H^T W H)
