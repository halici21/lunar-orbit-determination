"""Phase 17-R1O-CD1: contract-level invariants of the counted-Doppler characterization.

Science numbers live in artifacts/phase17_r1o_cd1_*; these tests pin the
preregistered infrastructure: predeclaration hash, orbit/epoch matrix, HF-primary
dynamics, Doppler schedule construction, count-interval mapping, event ownership,
noise/weight units, nuisance-column ownership, range-baseline recomputation, the
independent rebuild and the classification rules.
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "studies" / "phase17" / "scripts"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import phase17_r1o_cd1_campaign as C  # noqa: E402
import phase17_r1o_cd1_core as CD  # noqa: E402
import phase17_r1od3_core as D3  # noqa: E402

PRE = CD.load_predeclaration(C.PREDECLARATION_SHA256)
GRID = D3.load_grid()


# ----------------------------------------------------------------------
# predeclaration and matrix
# ----------------------------------------------------------------------
def test_predeclaration_hash_is_pinned_and_tampering_is_refused(tmp_path):
    assert PRE["canonical_sha256"] == C.PREDECLARATION_SHA256 == CD.canonical_sha256(PRE)
    assert PRE["written_before_any_cd1_k_metric"] is True
    for mutate in (lambda d: d["population"].__setitem__("POP(X)", "fraction(delta_X > 0) >= 0.50"),
                   lambda d: d["noise"].__setitem__("PRIMARY", "sigma_D = 1e-6 m/s"),
                   lambda d: d["epoch_grid"]["k"].pop()):
        bad = json.loads(CD.PRE_PATH.read_text(encoding="utf-8"))
        mutate(bad)
        p = tmp_path / "pre.json"
        p.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(CD.PredeclarationMismatch):
            CD.load_predeclaration(C.PREDECLARATION_SHA256, p)


def test_matrix_is_five_frozen_orbits_by_24_d3_epochs():
    assert set(C.ORBITS) == set(PRE["orbits"]["set"])
    assert len(C.ORBITS) == 5 and C.EPOCHS == tuple(range(24)) == tuple(PRE["epoch_grid"]["k"])
    d2 = json.loads((CD.ARTIFACTS.parent / "configs" / "phase17_r1od2_predeclared_grid.json").read_text(encoding="utf-8"))
    got = {g["case_id"]: hashlib.sha256(json.dumps(g["x0"]).encode()).hexdigest() for g in d2["geometries"]
           if g["case_id"] in C.ORBITS}
    assert got == {k: v["x0_sha256"] for k, v in PRE["orbits"]["set"].items()}
    assert GRID["canonical_sha256"] in PRE["epoch_grid"]["source"]
    assert D3.epoch_et(0, GRID) == GRID["stage_A_epoch_grid"]["canonical_et0"]          # canonical epoch exactly


def test_primary_dynamics_is_high_fidelity_and_the_fd_wrapper_matches_it():
    src = inspect.getsource(C.run_cell)
    assert 'if dyn == "HIGH_FIDELITY":' in src and "propagate_fidelity(scn)" in src
    assert PRE["dynamics"]["PRIMARY"].startswith("HIGH_FIDELITY")
    import phase17_r1od2_fidelity as F
    fsrc, wsrc = inspect.getsource(F.propagate_fidelity), inspect.getsource(C.propagate_hf)
    for token in ("MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2", "shadow_model=scn.shadow_model", "j2_moon=0.0",
                  "harmonic_model=C._M[NMAX]", "harmonic_stm_opt_in=True", "G.sun_mci(scn.et0 + float(t))"):
        assert token in fsrc and token in wsrc
    assert "rtol=1e-12, atol=1e-13" in fsrc
    assert inspect.signature(C.propagate_hf).parameters["rtol"].default == 1e-12
    assert inspect.signature(C.propagate_hf).parameters["atol"].default == 1e-13


# ----------------------------------------------------------------------
# schedule construction and count intervals (synthetic)
# ----------------------------------------------------------------------
def test_visibility_windows_and_count_interval_brackets():
    assert CD.visibility_windows(np.array([0, 1, 1, 0, 1, 1, 1, 0], bool)) == [(1, 2), (4, 6)]
    assert CD.visibility_windows(np.array([1, 1], bool)) == [(0, 1)]
    t = np.arange(0.0, 900.0 + 1, 90.0)
    assert CD.count_interval_nodes(5, 60.0, t) == (4, 6)
    assert CD.count_interval_nodes(5, 10.0, t) == (4, 6)
    assert CD.count_interval_nodes(5, 540.0, t) == (2, 8)


def test_count_interval_mapping_never_overlaps_and_matches_the_predeclaration():
    assert CD.TC_PRIMARY_S == PRE["count_interval"]["PRIMARY_S"] == 60.0
    assert CD.TC_CADENCE_S[60.0] == PRE["count_interval"]["PRIMARY_CADENCE_S"] == 90.0
    for tc, cad in CD.TC_CADENCE_S.items():
        assert cad >= tc and (cad / CD.GRID_CADENCE_S).is_integer()
        assert CD.doppler_config(tc).count_interval_s == tc
        assert CD.doppler_config(tc).mode == "two_way_counted_doppler"
        assert CD.doppler_config(tc).output_unit == "mps_equivalent"
        assert not CD.doppler_config(tc).four_event_enabled                      # production default R3
    assert {str(int(k)): v for k, v in CD.TC_CADENCE_S.items() if k != 60.0} == PRE["count_interval"]["sensitivity"]


def test_k_history_places_dxdk_in_the_first_stm_column_only():
    rng = np.random.default_rng(1)
    nom = rng.normal(size=(5, 48))
    kh = CD.k_history(nom)
    assert kh.shape == (5, 42)
    np.testing.assert_array_equal(kh[:, :6], nom[:, :6])
    np.testing.assert_array_equal(kh[:, 6:12], nom[:, 42:48])
    assert not kh[:, 12:].any()


# ----------------------------------------------------------------------
# noise and weights
# ----------------------------------------------------------------------
def test_noise_units_and_202e_scintillation_values():
    assert CD.SIGMA_D60_PRIMARY_MPS == 1.0e-4
    assert CD.SIGMA_D60_BRACKET_MPS == tuple(PRE["noise"]["bracket_60s_mps"])
    assert CD.sigma_tc(1e-4, 60.0, 1.0) == 1e-4
    assert CD.sigma_tc(1e-4, 540.0, 1.0) == pytest.approx(1e-4 / 9)
    assert CD.sigma_tc(1e-4, 10.0, 0.175) == pytest.approx(1e-4 * 6 ** 0.175)
    s = CD.scintillation_sigma_mps(np.array([180.0, 120.0, 90.0, 30.0, 10.0]), 60.0)
    c, f = 299792458.0, 7.2e9 * 880.0 / 749.0
    base = np.sqrt(0.53 * 2.7e-6 * c ** 2 / (f ** 2 * 60.0 ** 0.35))
    np.testing.assert_allclose(s[:3], base, rtol=1e-12)                          # no sine factor above 90 deg
    np.testing.assert_allclose(s[3:], base / np.sin(np.radians([30.0, 10.0])) ** 1.225, rtol=1e-12)
    assert s[0] == pytest.approx(2.07e-5, rel=1e-2) and s[4] == pytest.approx(1.77e-4, rel=1e-2)
    lb = CD.operational_lower_bound_sigma(np.array([180.0, 10.0]))
    assert np.all(lb >= CD.FTS_FLOOR_60S_HIGH_MPS) and np.all(lb >= s[[0, 4]])


# ----------------------------------------------------------------------
# nuisance ownership (synthetic rows)
# ----------------------------------------------------------------------
def _rows():
    t = np.array([0.0, 90.0, 180.0, 1000.0, 1090.0, 5000.0, 5090.0, 5180.0, 5270.0])
    ses = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    n = t.size
    rng = np.random.default_rng(3)
    return CD.DopplerRows(60.0, 90.0, t, np.arange(n), np.array([0, 0, 0, 1, 1, 2, 2, 2, 2]), ses,
                          rng.normal(size=(n, 6)), rng.normal(size=n), np.zeros(n), np.full(n, 90.0),
                          session_station=[0, 1, 2], session_window=[(0.0, 180.0), (1000.0, 1090.0), (5000.0, 5270.0)])


def test_nuisance_columns_are_owned_by_their_pass_and_zero_on_range_rows():
    r = _rows()
    assert CD.nuisance_columns(4, r, "D0_IDEAL") is None
    b = CD.nuisance_columns(4, r, "D1_SESSION_CONSTANT_BIAS")
    assert b.shape == (13, 3) and not b[:4].any()
    for j in range(3):
        np.testing.assert_array_equal(np.where(b[4:, j] == 1)[0], np.where(r.session == j)[0])
    d = CD.nuisance_columns(4, r, "D2_SESSION_BIAS_PLUS_LINEAR_DRIFT")
    assert d.shape == (13, 6) and not d[:4].any()
    np.testing.assert_array_equal(d[:, :3], b)
    for j in range(3):
        sel = r.session == j
        drift = d[4:, 3 + j]
        assert not drift[~sel].any()
        assert drift[sel].min() == -1.0 and drift[sel].max() == 1.0
    with pytest.raises(ValueError):
        CD.nuisance_columns(4, r, "D9")


def test_subset_renumbers_sessions_chronologically():
    r = _rows()
    s = r.subset(r.session != 1)
    assert s.n == 7 and s.n_sessions == 2
    assert s.session.tolist() == [0, 0, 0, 1, 1, 1, 1]
    assert s.session_station == [0, 2] and s.session_window == [(0.0, 180.0), (5000.0, 5270.0)]


def test_regular_session_takes_the_centred_hour_of_long_passes_only():
    t = np.arange(0.0, 9000.0 + 1, 90.0)
    n = t.size
    r = CD.DopplerRows(60.0, 90.0, t, np.arange(n), np.zeros(n, int), np.zeros(n, int), np.zeros((n, 6)),
                       np.zeros(n), np.zeros(n), np.zeros(n), session_station=[0], session_window=[(0.0, 9000.0)])
    m = CD.regular_session_mask(r)
    assert t[m].min() >= 4500.0 - 1800.0 and t[m].max() <= 4500.0 + 1800.0 and m.sum() == 41
    short = CD.DopplerRows(60.0, 90.0, t[:20], np.arange(20), np.zeros(20, int), np.zeros(20, int), np.zeros((20, 6)),
                           np.zeros(20), np.zeros(20), np.zeros(20), session_station=[0], session_window=[(0.0, 1710.0)])
    assert not CD.regular_session_mask(short).any()


# ----------------------------------------------------------------------
# classification rules
# ----------------------------------------------------------------------
def test_population_rule_boundaries():
    v = [0.03] * 50 + [0.01] * 25 + [-0.01] * 25                  # 75% positive, 50% > 0.02
    p = C.population(v)
    assert p["frac_pos"] == 0.75 and p["frac_modest_or_strong"] == 0.50 and p["POP"]
    assert not C.population([0.03] * 49 + [0.01] * 26 + [-0.01] * 25)["POP"]
    assert not C.population([0.03] * 60 + [-0.01] * 40)["POP"]      # 60% positive
    assert not C.population([0.0] * 100)["POP"]                      # zero is not positive


def test_decision_tree_order_and_partition():
    yes, no = dict(POP=True, frac_modest_or_strong=0.6), dict(POP=False, frac_modest_or_strong=0.3)
    rare = dict(POP=False, frac_modest_or_strong=0.05)
    assert C.decision(yes, yes, True) == "COUNTED_DOPPLER_ROBUSTLY_ADDS_K_SRP_DIRECTION"
    assert C.decision(yes, yes, False) == "COUNTED_DOPPLER_ROBUSTLY_IMPROVES_DIRECTION_BUT_K_REMAINS_WEAK"
    assert C.decision(no, no, False) == "COUNTED_DOPPLER_GAIN_IS_GEOMETRY_DEPENDENT"
    assert C.decision(yes, no, True) == "COUNTED_DOPPLER_K_INFORMATION_IS_SYSTEMATICALLY_FRAGILE"
    assert C.decision(rare, rare, False) == \
        "COUNTED_DOPPLER_ADDS_INFORMATION_MAGNITUDE_WITHOUT_BREAKING_K_STATE_DEGENERACY"
    assert C.decision(dict(POP=False, frac_modest_or_strong=0.10), rare, False) == \
        "COUNTED_DOPPLER_GAIN_IS_GEOMETRY_DEPENDENT"
    labels = {PRE["decision_tree"][k].split(":")[0] for k in "ABCDE"}
    assert labels == {C.decision(yes, yes, True), C.decision(yes, yes, False), C.decision(no, no, False),
                      C.decision(yes, no, True), C.decision(rare, rare, False)}


def test_nuisance_and_fidelity_classes():
    assert C.nuisance_retention_class([]) == "NOT_APPLICABLE_NO_IDEAL_GAIN"
    assert C.nuisance_retention_class([0.8, 0.9, 0.75]) == "RETAINED"
    assert C.nuisance_retention_class([0.3, 0.5]) == "PARTIALLY_RETAINED"
    assert C.nuisance_retention_class([0.1, 0.2, -0.5]) == "LOST"
    assert C.fidelity_class(True, False, 0.0) == "FIDELITY_CHANGES_POPULATION_VERDICT"
    assert C.fidelity_class(False, False, 0.03) == "FIDELITY_MATERIAL_VERDICT_STABLE"
    assert C.fidelity_class(True, True, 0.01) == "FIDELITY_NEGLIGIBLE"


def test_relative_and_absolute_thresholds_are_the_phase17_ones():
    import phase17_r1od2_core as D2
    assert D2.relative_gain_class(0.0501) == "STRONG_DIRECTION_GAIN"
    assert D2.relative_gain_class(0.05) == "MODEST_DIRECTION_GAIN"
    assert D2.relative_gain_class(0.02) == "MAGNITUDE_ONLY"
    assert D2.relative_gain_class(-0.0201) == "DIRECTION_LOSS"
    assert D2.absolute_class(0.5) == "STRONG_K_SEPARATION"
    assert D2.absolute_class(0.35) == "MODERATE_K_SEPARATION"
    assert D2.absolute_class(0.3499) == "WEAK_K_SEPARATION"


# ----------------------------------------------------------------------
# real geometry: canonical epoch, SIMPLIFIED (fast)
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def canon():
    import phase17_r1m_core as M
    if not (M.CAMPAIGN_ROOT / "08_manifests").exists():
        pytest.skip("read-only campaign data not available")
    C._init()
    D2 = D3.D2
    scn = C.scenario("CANON", M.campaign_initial_state().tolist(), 0)
    rb = D2.build_range_block(scn, D2.propagate_nom48(scn))
    st = C.stations()
    rows = CD.build_doppler_rows(scn, rb, 60.0, st, with_values=False)
    return SimpleNamespace(D2=D2, scn=scn, rb=rb, st=st, rows=rows)


def test_range_baseline_is_recomputed_and_reproduces_d3(canon):
    ref = {int(r["k"]): float(r["range_f_perp"]) for r in
           csv.DictReader((CD.ARTIFACTS / "phase17_r1od3_primary_results.csv").open()) if r["strategy"] == "GC_ONLY"}
    f0 = canon.D2.info_metrics([(canon.rb.h_x0, canon.rb.h_k, canon.rb.w)])["f_perp"]
    assert abs(f0 - ref[0]) <= 1e-12
    assert np.all(canon.rb.w == 1.0 / 25.0)


def test_doppler_rows_obey_the_schedule_contract(canon):
    r, rb = canon.rows, canon.rb
    assert r.n > 7 and r.n_candidates == r.n + r.n_dropped_visibility + r.n_dropped_domain
    assert np.all(np.diff(r.t_mid) >= 0)
    assert len({(int(n), int(s)) for n, s in zip(r.node, r.station)}) == r.n       # one row per node x station
    for i in range(r.n):
        lo, hi = CD.count_interval_nodes(int(r.node[i]), 60.0, rb.t_grid)
        assert np.all(rb.vis[lo:hi + 1, r.station[i]])
        w0, w1 = r.session_window[r.session[i]]
        assert w0 <= rb.t_grid[lo] and rb.t_grid[hi] <= w1
        assert r.session_station[r.session[i]] == r.station[i]
    first = [int(np.where(r.session == j)[0][0]) for j in range(r.n_sessions)]
    assert first == sorted(first)                                                  # chronological numbering


def test_doppler_rows_are_the_production_jacobian_and_match_the_four_event_model(canon):
    from lunar_od.radiometrics import RangeRatePhysicsConfig, two_way_counted_doppler_initial_state_jacobian
    from lunar_od.two_way_counted_doppler import four_event_counted_doppler_initial_state_jacobian
    r, rb, pg = canon.rows, canon.rb, canon.rb.pass_geo
    cfg4 = RangeRatePhysicsConfig(mode="two_way_counted_doppler", count_interval_s=60.0,
                                  counted_doppler_model="four_event_delay")
    kh = CD.k_history(rb.nom48)
    for i in (r.n // 5, r.n // 2, r.n - 1):
        args = (float(r.t_mid[i]), canon.st[r.station[i]], rb.t_grid)
        tail = (pg.earth_pos_mci_m, pg.earth_vel_mci_mps, pg.x_j2000_to_itrf93)
        h = two_way_counted_doppler_initial_state_jacobian(*args, rb.nom48[:, :42], *tail, CD.doppler_config(60.0),
                                                           et0_s=pg.et0_s)
        np.testing.assert_array_equal(h, r.h_x0[i])
        h4 = four_event_counted_doppler_initial_state_jacobian(*args, rb.nom48[:, :42], *tail, cfg4, et0_s=pg.et0_s)
        k4 = four_event_counted_doppler_initial_state_jacobian(*args, kh, *tail, cfg4, et0_s=pg.et0_s)[0]
        assert np.linalg.norm(h4 - r.h_x0[i]) <= 1e-9 * np.linalg.norm(r.h_x0[i])
        assert abs(k4 - r.h_k[i]) <= 1e-9 * np.max(np.abs(r.h_k))


def test_independent_rebuild_matches_the_metric_wrapper(canon):
    D2, rb, r = canon.D2, canon.rb, canon.rows
    w = np.full(r.n, 1.0 / CD.SIGMA_D60_PRIMARY_MPS ** 2)
    for model, prior, wr in (("D0_IDEAL", None, False), ("D0_IDEAL", None, True),
                             ("D1_SESSION_CONSTANT_BIAS", None, True), ("D2_SESSION_BIAS_PLUS_LINEAR_DRIFT", None, True),
                             ("D1_SESSION_CONSTANT_BIAS", 1e-4, True)):
        m = CD.metrics(D2, rb, r, w, model, prior, with_range=wr)
        assert abs(CD.independent_f_perp(D2, rb, r, w, model, prior, with_range=wr) - m["f_perp"]) <= 1e-10
    # a tight nuisance prior approaches the ideal case, a free nuisance cannot add direction
    f0 = CD.metrics(D2, rb, r, w, "D0_IDEAL", None)["f_perp"]
    f_tight = CD.metrics(D2, rb, r, w, "D1_SESSION_CONSTANT_BIAS", 1e-9)["f_perp"]
    f_free = CD.metrics(D2, rb, r, w, "D1_SESSION_CONSTANT_BIAS", None)["f_perp"]
    assert abs(f_tight - f0) < 1e-4 and f_free <= f0 + 1e-12


def test_regression_single_row_pass_gets_no_drift_column_and_wrapper_matches_lstsq():
    """CD1-IBF-01: a single-row pass once produced an all-zero D2 drift column, which made
    the QR projector in D2.info_metrics remove a spurious direction (rebuild diff ~1e-5)."""
    import phase17_r1od2_core as D2
    t = np.array([0.0, 90.0, 180.0, 3000.0, 6000.0, 6090.0, 6180.0])
    ses = np.array([0, 0, 0, 1, 2, 2, 2])                                   # pass 1 has one row
    n = t.size
    rng = np.random.default_rng(11)
    r = CD.DopplerRows(60.0, 90.0, t, np.arange(n), ses, ses, rng.normal(size=(n, 6)) * 1e-3,
                       rng.normal(size=n), np.zeros(n), np.full(n, 90.0),
                       session_station=[0, 1, 2], session_window=[(0.0, 180.0), (3000.0, 3000.0), (6000.0, 6180.0)])
    nu = CD.nuisance_columns(20, r, "D2_SESSION_BIAS_PLUS_LINEAR_DRIFT")
    assert nu.shape == (27, 5) and nu.any(axis=0).all()                    # 3 biases + 2 drifts, no zero column
    rb = SimpleNamespace(h_x0=rng.normal(size=(20, 6)) * 1e-3, h_k=rng.normal(size=20), w=np.ones(20), n_obs=20)
    w = np.ones(n)
    m = CD.metrics(D2, rb, r, w, "D2_SESSION_BIAS_PLUS_LINEAR_DRIFT", None)
    assert m["n_drift_columns"] == 2
    assert abs(m["f_perp"] - CD.independent_f_perp(D2, rb, r, w, "D2_SESSION_BIAS_PLUS_LINEAR_DRIFT", None)) <= 1e-12
