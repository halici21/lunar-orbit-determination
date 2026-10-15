"""Phase 17-R1O-CD1 closure: HARDEN-D23 materiality and the rebuild numerical floor.

Pins the committed closure evidence (artifacts/phase17_r1o_cd1_closure*.json)
against the committed CD1 decision, and qualifies the exact rational f_perp oracle
and the analysis-only reference-Earth seam on their own.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "studies" / "phase17" / "scripts"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import phase17_r1o_cd1_closure as X  # noqa: E402

CLOSURE = json.loads((ROOT / "studies" / "phase17" / "results" / "phase17_r1o_cd1_closure.json").read_text(encoding="utf-8"))
SUMMARY = json.loads((ROOT / "studies" / "phase17" / "results" / "phase17_r1o_cd1_summary.json").read_text(encoding="utf-8"))
RAW = json.loads(X.RAW_PATH.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# exact oracle
# ----------------------------------------------------------------------
def test_exact_oracle_matches_a_closed_form_projection():
    # b = (1, 1, 1) against span{(1, 0, 0), (0, 1, 0)}: residual (0, 0, 1), f = 1/sqrt(3)
    a = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    b = np.ones(3)
    assert abs(float(X.exact_f_perp(a, b, np.ones(3))) - 3 ** -0.5) < 1e-30


def test_exact_oracle_is_weight_and_column_scale_aware():
    rng = np.random.default_rng(1729)
    a, b, w = rng.normal(size=(40, 4)), rng.normal(size=40), rng.uniform(0.5, 2.0, size=40)
    sw = np.sqrt(w)
    q, _ = np.linalg.qr(sw[:, None] * a)
    bb = sw * b
    f_float = np.linalg.norm(bb - q @ (q.T @ bb)) / np.linalg.norm(bb)
    f_exact = float(X.exact_f_perp(a, b, w))
    assert abs(f_exact - f_float) < 1e-13
    # projection is invariant to column scaling: exactly for power-of-two scales (the
    # scaled float64 inputs are exact), to input rounding for decimal ones
    assert X.exact_f_perp(a * 2.0 ** np.array([20, -10, 0, 40]), b, w) == X.exact_f_perp(a, b, w)
    assert abs(float(X.exact_f_perp(a * np.array([1e6, 1e-3, 1.0, 1e12]), b, w)) - f_exact) < 1e-15


def test_exact_oracle_refuses_an_exactly_singular_design():
    a = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    with pytest.raises(ZeroDivisionError):
        X.exact_f_perp(a, np.array([1.0, 0.0, 0.0]), np.ones(3))


# ----------------------------------------------------------------------
# HARDEN-D23 materiality evidence
# ----------------------------------------------------------------------
def test_d23_covers_the_full_primary_population_and_reproduces_cd1():
    d = CLOSURE["d23"]
    assert d["cells"] == 240 and d["feasible_cells"] == 240 and len(RAW) == 240
    assert {(c["dynamics"]) for c in RAW} == {"HIGH_FIDELITY", "SIMPLIFIED"}
    assert d["same_rows_all"] is True
    # the production rebuild inside the closure reproduces the committed CD1 numbers exactly
    assert d["production_anchor_max_abs_diff"] == 0.0
    # reference and linear Earth coincide at the grid nodes (same SPICE source)
    assert d["earth_pos_diff_at_grid_nodes_max_m"] == 0.0


def test_d23_does_not_move_any_cd1_decision():
    d = CLOSURE["d23"]
    assert d["HARDEN_D23_CD1_MATERIALITY_GATE"] == "PASS"
    assert d["class_or_sign_changes"] == [] and d["decision_fields_identical"] is True
    for side in ("verdict_production", "verdict_reference"):
        assert d[side]["PRIMARY_CLASS"] == SUMMARY["PRIMARY_CLASS"]
        assert d[side]["NUISANCE_RETENTION_CLASS"] == SUMMARY["NUISANCE_RETENTION_CLASS"]
        assert d[side]["FIDELITY_EFFECT_CLASS"] == SUMMARY["FIDELITY_EFFECT_CLASS"]
    assert d["f_perp_abs_diff_max"] * 1e4 < d["closest_decision_margin"]


def test_d23_observable_scale_is_recorded_not_the_harden_estimate():
    # the end-to-end four-event count-rate effect is ~1e-3 m/s, three orders above
    # the HARDEN receive-endpoint estimate (~1.5e-6 m/s); the information metrics
    # are insensitive because the partials move only at the 1e-7 level
    d = CLOSURE["d23"]
    assert 1e-3 < d["observable_diff_max_mps"] < 3e-3
    assert d["state_partial_rel_diff_max"] < 1e-6 and d["k_partial_diff_over_colmax_max"] < 1e-8
    assert 2.0 < d["earth_pos_diff_max_m"] < 3.0


def test_reference_earth_seam_restores_the_production_resolver():
    import lunar_od.radiometrics as RM
    original = RM.resolve_counted_doppler_station_state_provider
    with X.reference_earth(dict(nodes=set())):
        assert RM.resolve_counted_doppler_station_state_provider is not original
    assert RM.resolve_counted_doppler_station_state_provider is original


# ----------------------------------------------------------------------
# numerical floor (CD1 section 32)
# ----------------------------------------------------------------------
def test_rebuild_exceedance_is_in_the_check_route_not_the_metric():
    floors = {(f["orbit"], f["k"]): f for f in CLOSURE["floor"]}
    assert set(floors) == set(X.ORACLE_CELLS)
    exc = floors[("O2_FULL_SUN_HIGH_BETA", 6)]
    assert abs(exc["wrapper_minus_independent"] - 1.4586e-10) < 1e-14     # the section-32 comparison
    for f in floors.values():
        assert f["err_wrapper"] < 1e-14          # the CD1 metric agrees with the exact oracle
        assert f["err_svd"] < 1e-14
        assert f["err_independent"] > 1e3 * f["err_wrapper"]
    assert exc["err_independent"] > 1e-10          # the strict exceedance is the lstsq route's own error
