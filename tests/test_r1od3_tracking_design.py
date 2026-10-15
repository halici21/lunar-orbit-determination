"""Phase 17-R1O-D3: contract-level invariants of the tracking-design infrastructure.

Science numbers live in artifacts/phase17_r1od3_*; these tests pin bookkeeping
(grid hash, epochs, sessions, schedule patterns, resource rule, row/bias
ownership, weights) and one canonical reproduction under the F1 model.
"""
from __future__ import annotations

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

import phase17_r1od3_core as D3  # noqa: E402

GRID = D3.load_grid()


# ----------------------------------------------------------------------
# grid and epochs
# ----------------------------------------------------------------------
def test_grid_hash_is_pinned_and_tampering_is_refused(tmp_path):
    assert GRID["canonical_sha256"] == D3.GRID_SHA256
    bad = json.loads(D3.GRID_PATH.read_text(encoding="utf-8"))
    bad["calibration"]["nominal_bias_prior_ns"] = 0.05
    p = tmp_path / "grid.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(D3.GridHashMismatch):
        D3.load_grid(p)


def test_uniform_epoch_grid_includes_the_canonical_epoch_exactly():
    g = GRID["stage_A_epoch_grid"]
    ets = [D3.epoch_et(k, GRID) for k in range(g["count"])]
    assert g["count"] == 24 and ets[0] == g["canonical_et0"] == 857806356.99998
    np.testing.assert_allclose(np.diff(ets), g["synodic_month_s"] / 24, rtol=0, atol=1e-6)
    assert abs(360.0 / g["count"] - g["spacing_deg_synodic_phase"]) < 1e-12
    with pytest.raises(ValueError):
        D3.epoch_et(24, GRID)
    # D2's quarter epochs are grid points 6, 12, 18
    d2 = json.loads((D3.ARTIFACTS.parent / "configs" / "phase17_r1od2_predeclared_grid.json").read_text(encoding="utf-8"))
    d2et = {x["case_id"]: x["et0"] for x in d2["geometries"]}
    for k, cid in ((6, "E1_FIRST_QUARTER_PLUS"), (12, "E2_HALF_SYNODIC"), (18, "E3_THREE_QUARTER")):
        assert abs(D3.epoch_et(k, GRID) - d2et[cid]) < 1e-3


# ----------------------------------------------------------------------
# sessions and patterns (synthetic rows: bookkeeping only)
# ----------------------------------------------------------------------
def _rows(pair, runs, t0=0.0, seed=0):
    """runs = list of (start_s, n_rows) contiguous 90-s runs."""
    t, sess = [], []
    for j, (s, n) in enumerate(runs):
        t += [s + 90.0 * i for i in range(n)]
        sess += [j] * n
    rng = np.random.default_rng(seed)
    n = len(t)
    return SimpleNamespace(pair=pair, t_used=np.asarray(t) + t0, session=np.asarray(sess), n_sessions=len(runs),
                           n=n, h_x0=rng.normal(size=(n, 6)), h_k=rng.normal(size=n))


def test_session_construction_min_window_centring_and_normal_points():
    rows = _rows("GC", [(0.0, 12), (5000.0, 20), (20000.0, 80)])     # 18 min, 30 min, 120 min windows
    ss = D3.build_sessions(rows, GRID)
    assert len(ss) == 2                                              # the 18-min window does not qualify
    short, long = ss
    assert (short.start, short.end) == (5000.0, 5000.0 + 20 * 90.0)  # whole window
    assert abs((long.end - long.start) - 3600.0) < 1e-9              # capped at 1 h ...
    assert abs(0.5 * (long.start + long.end) - 0.5 * (long.window_start + long.window_end)) < 1e-9  # ... centred
    for s in ss:
        assert len(s.np_index) == 3
        assert set(s.np_index) <= set(s.row_index)
        t = rows.t_used[list(s.np_index)]
        assert np.all((t >= s.start) & (t <= s.end))


def test_schedule_patterns_are_chronological_and_deterministic():
    rows = _rows("GC", [(0.0, 20), (10000.0, 20), (40000.0, 20), (90000.0, 20)])
    ss = D3.build_sessions(rows, GRID)
    mid = 45000.0
    assert D3.select_pattern(ss, "FIRST_SESSION", mid) == [ss[0]]
    assert D3.select_pattern(ss, "LAST_SESSION", mid) == [ss[-1]]
    assert D3.select_pattern(ss, "MID_SESSION", mid) == [ss[2]]
    assert D3.select_pattern(ss, "FIRST_PLUS_LAST", mid) == [ss[0], ss[-1]]
    assert D3.select_pattern(ss, "ALL_FEASIBLE_SESSIONS", mid) == ss
    assert D3.select_pattern(ss[:1], "FIRST_PLUS_LAST", mid) is None
    assert D3.select_pattern([], "FIRST_SESSION", mid) is None


def test_goldstone_resource_rule_keeps_gc_and_drops_overlapping_gm():
    gc = D3.build_sessions(_rows("GC", [(0.0, 20), (10000.0, 20)]), GRID)
    gm = D3.build_sessions(_rows("GM", [(1000.0, 20), (1800.0 + 5000.0, 20), (11800.0, 20)]), GRID)
    # gm[0] overlaps gc[0]; gm[2] touches gc[1] end exactly (closed interval -> conflict)
    assert gm[2].start == gc[1].end
    g_k, m_k, raw = D3.resolve_goldstone(gc, gm)
    assert g_k == gc and m_k == [gm[1]] and raw == 2
    sched = D3.stack("GC_PLUS_GM", "X", 0, g_k + m_k, {"GC": _rows("GC", [(0.0, 20), (10000.0, 20)]),
                                                       "GM": _rows("GM", [(1000.0, 20), (6800.0, 20), (11800.0, 20)])})
    assert not D3.double_booked(sched)


# ----------------------------------------------------------------------
# ownership, bias columns, weights
# ----------------------------------------------------------------------
def _joint():
    rgc = _rows("GC", [(0.0, 20), (10000.0, 20)], seed=1)
    rgm = _rows("GM", [(5000.0, 20)], seed=2)
    ss = D3.build_sessions(rgc, GRID) + D3.build_sessions(rgm, GRID)
    return D3.stack("GC_PLUS_GM", "X", 7, ss, {"GC": rgc, "GM": rgm}), {"GC": rgc, "GM": rgm}


def test_stacking_preserves_row_ownership():
    s, rows = _joint()
    assert s.n == 9 and s.n_sessions == 3
    assert list(s.baseline) == ["GC"] * 3 + ["GM"] * 3 + ["GC"] * 3          # chronological across baselines
    for p in ("GC", "GM"):
        sel = s.baseline == p
        np.testing.assert_array_equal(s.h_x0[sel], rows[p].h_x0[s.parent_row[sel]])
        np.testing.assert_array_equal(s.h_k[sel], rows[p].h_k[s.parent_row[sel]])
    assert list(s.normal_point) == [0, 1, 2] * 3


def test_bias_columns_are_owned_by_exactly_their_session():
    s, _ = _joint()
    n_r = 50
    b = D3.bias_columns(n_r, s)
    assert b.shape == (n_r + 9, 3)
    assert not b[:n_r].any()
    np.testing.assert_array_equal(b[n_r:].sum(axis=1), np.ones(9))
    for j in range(3):
        np.testing.assert_array_equal(np.where(b[n_r:, j] == 1)[0], np.where(s.session == j)[0])


def test_session_label_permutation_changes_the_design_as_expected():
    """Relabelling sessions permutes bias columns (metrics invariant); moving ONE normal
    point into another session changes the nuisance design (and the K metric)."""
    import phase17_r1od2_core as D2
    s, _ = _joint()
    rng = np.random.default_rng(5)
    n_r = 40
    rb = SimpleNamespace(h_x0=rng.normal(size=(n_r, 6)), h_k=rng.normal(size=n_r), w=np.ones(n_r), n_obs=n_r)
    w = np.ones(s.n)                              # synthetic O(1) units throughout
    blocks = [(rb.h_x0, rb.h_k, rb.w), (s.h_x0, s.h_k, w)]
    base = D2.info_metrics(blocks, nuisance=D3.bias_columns(n_r, s), nuisance_prior_sigma=1.0)
    perm = np.array([2, 0, 1])
    s2 = D3.Schedule(**{**s.__dict__, "session": perm[s.session]})
    b1, b2 = D3.bias_columns(n_r, s), D3.bias_columns(n_r, s2)
    np.testing.assert_array_equal(b2[:, perm], b1)
    assert D2.info_metrics(blocks, nuisance=b2, nuisance_prior_sigma=1.0)["f_perp"] == pytest.approx(
        base["f_perp"], rel=1e-10)
    moved = s.session.copy()
    moved[0] = 1                                    # one GC normal point wrongly owned by the GM session
    s3 = D3.Schedule(**{**s.__dict__, "session": moved})
    assert not np.array_equal(D3.bias_columns(n_r, s3), b1)
    free = D2.info_metrics(blocks, nuisance=b1)["f_perp"]
    assert abs(D2.info_metrics(blocks, nuisance=D3.bias_columns(n_r, s3))["f_perp"] - free) > 1e-6


def test_primary_weight_is_the_constant_dsn_random_delay():
    w = D3.delay_weight(4, GRID)
    np.testing.assert_allclose(w, 1.0 / (0.048e-9) ** 2)
    assert GRID["noise"]["primary"]["semantics"] == "CONSTANT_DELAY_SIGMA"
    assert D3.calibration_prior("C0_IDEAL", GRID) == (False, None)
    assert D3.calibration_prior("C2_FREE", GRID) == (True, None)
    use, prior = D3.calibration_prior("C1_x2.0", GRID)
    assert use and prior == pytest.approx(2 * 0.0317e-9)


# ----------------------------------------------------------------------
# canonical reproduction under the current F1 model (real geometry)
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def canonical():
    import phase17_r1m_core as M
    if not (M.CAMPAIGN_ROOT / "08_manifests").exists():
        pytest.skip("read-only campaign data not available")
    import phase17_r1od3_campaign as C
    C._init()
    return C.run_geometry(dict(label="T_E00", k=0, x0=M.campaign_initial_state().tolist(), orbit="CANON",
                               dynamics="SIMPLIFIED", cals=["C0_IDEAL", D3.C1_NOMINAL, "C2_FREE"],
                               patterns="PRIMARY"))


def test_canonical_d2_result_reproduces_under_f1(canonical):
    assert canonical["range_f_perp"] == pytest.approx(0.0915024012983624, abs=1e-9)
    assert canonical["diagnostics"]["canonical_dense_angular_gc_f_perp"] == pytest.approx(0.172343, abs=1e-6)


def test_canonical_schedules_obey_every_contract(canonical):
    c = canonical["contract"]
    assert c["earth_audit_passed"] and all(c["rows_dual_visible"].values())
    assert c["normal_points_attached"] and c["ownership_consistent"] and not c["joint_double_booked"]
    assert canonical["rebuild"] and max(x["abs_diff"] for x in canonical["rebuild"]) < 1e-10
