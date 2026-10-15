"""Phase 17-R1O-D4: contract-level invariants of the GM narrow-window confirmation.

Science numbers live in artifacts/phase17_r1od4_*; these tests pin the preregistered
rules (hash, grid, epoch convention, window criterion, schedules, orbits, dynamics,
calibration mapping) and one canonical anchor reproduction against D3.
"""
from __future__ import annotations

import csv
import hashlib
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
import phase17_r1od4_campaign as C  # noqa: E402

PRE = C.load_predeclaration()
GRID = D3.load_grid()


# ----------------------------------------------------------------------
# predeclaration
# ----------------------------------------------------------------------
def test_predeclaration_hash_is_pinned_and_tampering_is_refused(tmp_path):
    assert PRE["canonical_sha256"] == C.PREDECLARATION_SHA256
    assert PRE["written_before_any_d4_k_metric"] is True
    for mutate in (lambda d: d["window"].__setitem__("required_consecutive_points", 3),
                   lambda d: d["thresholds"]["relative"].__setitem__("STRONG_DIRECTION_GAIN", "delta > 0.04"),
                   lambda d: d["phase_grid"]["phases_deg"].append(170.0)):
        bad = json.loads(C.PRE_PATH.read_text(encoding="utf-8"))
        mutate(bad)
        p = tmp_path / "pre.json"
        p.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(C.PredeclarationMismatch):
            C.load_predeclaration(p)


def test_campaign_uses_only_predeclared_schedules_and_calibrations():
    assert C.SCHEDULES == ("FIRST_SESSION", "ALL_FEASIBLE_SESSIONS")
    assert C.CAL_MAP == PRE["calibration"]["d3_core_names"]
    assert set(C.CAL_MAP.values()) == set(D3.CALIBRATIONS)
    assert PRE["baseline"]["primary"].startswith("GM_ONLY")


# ----------------------------------------------------------------------
# grid and epochs
# ----------------------------------------------------------------------
def test_dense_grid_is_22_points_5_deg_including_d3_anchors():
    ph = PRE["phase_grid"]["phases_deg"]
    assert len(ph) == PRE["phase_grid"]["count"] == 22
    assert ph[0] == 60.0 and ph[-1] == 165.0
    np.testing.assert_array_equal(np.diff(ph), 5.0)
    assert set(PRE["phase_grid"]["d3_anchors_deg"]) == {75.0, 90.0, 105.0, 135.0, 150.0} <= set(ph)


def test_phase_epoch_convention_equals_d3_k_at_multiples_of_15_deg():
    assert C.CANONICAL_ET0 == GRID["stage_A_epoch_grid"]["canonical_et0"]
    assert C.SYNODIC_S == GRID["stage_A_epoch_grid"]["synodic_month_s"]
    for k in range(4, 12):
        assert C.phase_et(15.0 * k) == D3.epoch_et(k, GRID)
    # independent route: phase / 360 of a synodic month
    for p in PRE["phase_grid"]["phases_deg"]:
        assert abs(C.phase_et(p) - (C.CANONICAL_ET0 + p / 360.0 * C.SYNODIC_S)) < 1e-6
    assert abs(C.phase_et(95.0) - C.phase_et(90.0) - C.SYNODIC_S / 72) < 1e-6


# ----------------------------------------------------------------------
# window criterion
# ----------------------------------------------------------------------
PH = [60.0 + 5 * i for i in range(22)]


def _series(strong_idx):
    return [0.06 if i in strong_idx else 0.01 for i in range(22)]


def test_seven_consecutive_strong_points_qualify_as_a_window():
    w = C.window_report(PH, _series(set(range(3, 10))), PRE)       # 75..105
    assert len(w) == 1 and w[0]["n_points"] == 7 and w[0]["width_deg"] == 30.0
    assert w[0]["start_deg"] == 75.0 and w[0]["end_deg"] == 105.0 and w[0]["overlaps_candidate"]
    assert w[0]["duration_days"] == pytest.approx(PRE["window"]["duration_days_per_30deg"])


def test_regression_three_consecutive_5deg_points_do_not_qualify():
    """The D3 k = 5-7 signature on the dense grid (75/80/85 or 75/90/105 isolated) is NOT a window."""
    assert C.window_report(PH, _series({3, 4, 5}), PRE) == []
    assert C.window_report(PH, _series({3, 6, 9}), PRE) == []
    assert C.window_report(PH, _series(set(range(3, 9))), PRE) == []                 # 6 points (25 deg)


def test_boundary_and_invalid_cell_rules():
    assert C.windows([True] * 7, 7) == [(0, 6)]
    assert C.windows([True] * 6 + [False] + [True] * 6, 7) == []                 # fragment
    s = _series(set(range(0, 7)))
    s[3] = None                                                                   # infeasible is NOT STRONG
    assert C.window_report(PH, s, PRE) == []
    assert not C.is_strong(0.05) and C.is_strong(0.0500001)
    # a window entirely outside [75, 150] is reported but does not overlap
    w = C.window_report(PH, _series(set(range(15, 22))), PRE)                     # 135..165
    assert w and w[0]["overlaps_candidate"]
    w = C.window_report([200.0 + 5 * i for i in range(22)], _series(set(range(0, 7))), PRE)
    assert w and not w[0]["overlaps_candidate"]


# ----------------------------------------------------------------------
# schedules
# ----------------------------------------------------------------------
def _rows(runs, seed=0):
    t, sess = [], []
    for j, (s, n) in enumerate(runs):
        t += [s + 90.0 * i for i in range(n)]
        sess += [j] * n
    rng = np.random.default_rng(seed)
    n = len(t)
    return SimpleNamespace(pair="GM", t_used=np.asarray(t), session=np.asarray(sess), n_sessions=len(runs), n=n,
                           h_x0=rng.normal(size=(n, 6)), h_k=rng.normal(size=n))


def test_first_session_is_the_chronologically_earliest_qualifying_window():
    rows = _rows([(0.0, 10), (5000.0, 20), (30000.0, 60), (80000.0, 20)])       # first window too short (15 min)
    ss = D3.build_sessions(rows, GRID)
    first = D3.select_pattern(ss, "FIRST_SESSION", 1e9)
    assert first == [ss[0]] and ss[0].window_start == 5000.0
    allp = D3.select_pattern(ss, "ALL_FEASIBLE_SESSIONS", 1e9)
    assert allp == ss and len(ss) == 3
    s = D3.stack("GM_ONLY", "FIRST_SESSION", -1, first, {"GM": rows})
    assert s.n == 3 and s.n_sessions == 1
    np.testing.assert_array_equal(s.h_x0, rows.h_x0[s.parent_row])
    b = D3.bias_columns(17, s)
    assert b.shape == (20, 1) and not b[:17].any() and b[17:].all()


# ----------------------------------------------------------------------
# frozen orbits and dynamics
# ----------------------------------------------------------------------
def test_orbit_initial_states_are_frozen():
    d2 = json.loads((D3.ARTIFACTS.parent / "configs" / "phase17_r1od2_predeclared_grid.json").read_text(encoding="utf-8"))
    got = {g["case_id"]: hashlib.sha256(json.dumps(g["x0"]).encode()).hexdigest() for g in d2["geometries"]
           if g["case_id"] in PRE["orbits"]["set"]}
    assert got == {k: v["x0_sha256"] for k, v in PRE["orbits"]["set"].items()}
    assert len(got) == 5


def test_high_fidelity_configuration_is_the_d3_stage_c_propagator():
    import inspect
    import phase17_r1od2_fidelity as F
    src = inspect.getsource(C.run_cell)
    assert "propagate_fidelity" in src and '"HIGH_FIDELITY"' in src
    assert callable(F.propagate_fidelity)
    assert "matrix" in PRE["dynamics"] and PRE["dynamics"]["matrix"].startswith("FULL")


# ----------------------------------------------------------------------
# canonical anchor reproduction (real geometry)
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def anchor90():
    import phase17_r1m_core as M
    if not (M.CAMPAIGN_ROOT / "08_manifests").exists():
        pytest.skip("read-only campaign data not available")
    C._init()
    return C.run_cell(dict(orbit="CANON", x0=M.campaign_initial_state().tolist(), phase=90.0, dynamics="SIMPLIFIED"))


def test_d3_anchor_90deg_reproduces_exactly(anchor90):
    ref = {(r["pattern"], r["calibration"]): float(r["delta_f_perp"]) for r in
           csv.DictReader((D3.ARTIFACTS / "phase17_r1od3_primary_results.csv").open())
           if r["strategy"] == "GM_ONLY" and r["k"] == "6" and r["pattern"] in C.SCHEDULES}
    for sched in C.SCHEDULES:
        for name, core in C.CAL_MAP.items():
            assert abs(anchor90["cells"][sched]["cal"][name]["delta_f_perp"] - ref[(sched, core)]) <= 1e-12


def test_anchor_cell_contracts_and_independent_rebuild(anchor90):
    assert anchor90["diag"]["earth_audit_passed"] and anchor90["diag"]["all_gm_rows_dual_visible"]
    for c in anchor90["cells"].values():
        assert c["feasible"] and c["np_visible"] and c["ownership_ok"]
        assert max(c["rebuild"].values()) <= 1e-10
    assert anchor90["cells"]["FIRST_SESSION"]["n_sessions"] == 1


def test_r2_calibration_fate_labels():
    w = C.window_report(PH, _series(set(range(3, 12))), PRE)[0]                  # 75..115 nominal
    assert C.r2_status(PH, w, _series(set(range(2, 13))), PRE) == "PERSISTS"
    assert C.r2_status(PH, w, _series(set(range(3, 10))), PRE) == "SHRINKS"
    assert C.r2_status(PH, w, _series({3, 4, 5, 8, 9, 10}), PRE) == "FRAGMENTS"
    assert C.r2_status(PH, w, _series({6}), PRE) == "DISAPPEARS"
