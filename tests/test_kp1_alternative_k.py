"""Phase 17-KP1: alternative K_SRP parameterizations - contract and oracle tests.

Pins the predeclaration, the basis definitions, the exact composition of
parameter sensitivities (against a closed-form linear system), the
basis-invariance of the information geometry, the exact rational oracle, and
the committed campaign evidence.
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

import phase17_kp1_core as KP  # noqa: E402

PINNED = "4f7f2632706f3c07e2444043d827c79ba2d5f9aa79ac36d4c6a6ab8ec906e495"
SUMMARY_PATH = ROOT / "studies" / "phase17" / "results" / "phase17_kp1_summary.json"


# ----------------------------------------------------------------------
# predeclaration
# ----------------------------------------------------------------------
def test_predeclaration_is_pinned_and_tampering_is_refused(tmp_path):
    pre = KP.load_predeclaration(PINNED)
    assert pre["written_before_any_kp1_separation_metric"] is True
    assert set(pre["candidates"]) == set(KP.MODELS)
    assert pre["measurements"]["doppler_values_used"] is False
    bad = dict(pre)
    bad["thresholds"] = dict(pre["thresholds"], separation="STRONG >= 0.40")
    p = tmp_path / "pre.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(KP.PredeclarationMismatch):
        KP.load_predeclaration(PINNED, p)


def test_campaign_pins_the_same_predeclaration():
    src = (EXAMPLES / "phase17_kp1_campaign.py").read_text(encoding="utf-8")
    assert 'PREDECLARATION_SHA256 = "%s"' % PINNED in src


# ----------------------------------------------------------------------
# bases
# ----------------------------------------------------------------------
@pytest.mark.parametrize("model,truth", [("M0_SCALAR", [1.0]), ("M1_TWO_SEGMENT", [1.0, 1.0]),
                                         ("M2_LINEAR_DRIFT", [1.0, 0.0]), ("M3_THREE_SEGMENT", [1.0, 1.0, 1.0])])
def test_every_model_contains_the_constant_truth_exactly(model, truth):
    # H_bank @ M @ truth must be exactly the scalar-K column (bank column 0)
    e_k = np.zeros(len(KP.BANK))
    e_k[0] = 1.0
    assert np.array_equal(KP.model_matrix(model) @ np.asarray(truth), e_k)


def test_boundaries_and_drift_basis():
    t = np.arange(0.0, 90.0 * 1177, 90.0)
    b2, b3 = KP.boundaries(t, 2), KP.boundaries(t, 3)
    assert b2[0] == 0 and b2[-1] == t.size - 1 and abs(t[b2[1]] - t[-1] / 2) <= 45.0
    assert abs(t[b3[1]] - t[-1] / 3) <= 45.0 and abs(t[b3[2]] - 2 * t[-1] / 3) <= 45.0
    s = KP.drift_interval_values(t)
    assert s.size == t.size - 1 and -1.0 < s.min() < -0.999 and 0.999 < s.max() < 1.0
    assert abs(s.mean()) < 1e-12


# ----------------------------------------------------------------------
# exact composition against a closed-form linear system
# ----------------------------------------------------------------------
def _double_integrator(t, g0):
    """x' = v, v' = K g0 (constant): Phi = [[I, tI], [0, I]], dx/dK = [g0 t^2/2; g0 t]."""
    n = t.size
    nom = np.zeros((n, 48))
    for i, ti in enumerate(t):
        phi = np.eye(6)
        phi[:3, 3:] = ti * np.eye(3)
        nom[i, 6:42] = phi.reshape(-1, order="F")
        nom[i, 42:45] = 0.5 * g0 * ti ** 2
        nom[i, 45:48] = g0 * ti
    return nom


def _window_closed_form(t, g0, a, b):
    lo = np.clip(t, a, b)
    # position: integral_a^lo (t - tau) g0 dtau ; velocity: (lo - a) g0
    pos = ((t * (lo - a) - 0.5 * (lo ** 2 - a ** 2)))[:, None] * g0
    vel = (lo - a)[:, None] * g0
    return np.hstack([pos, vel])


def test_window_sensitivities_match_the_closed_form():
    t = np.arange(0.0, 90.0 * 301, 90.0)
    g0 = np.array([3e-7, -1e-7, 2e-7])
    bank = KP.sensitivity_bank(_double_integrator(t, g0), t)
    b2, b3 = bank["boundaries"]["two"], bank["boundaries"]["three"]
    scale = np.max(np.abs(bank["K"]))
    for name, (a, b) in (("SEG2_A", (b2[0], b2[1])), ("SEG3_A", (b3[0], b3[1])), ("SEG3_B", (b3[1], b3[2]))):
        ref = _window_closed_form(t, g0, t[a], t[b])
        assert np.max(np.abs(bank[name] - ref)) <= 1e-12 * scale, name


def test_drift_staircase_matches_the_closed_form():
    t = np.arange(0.0, 90.0 * 301, 90.0)
    g0 = np.array([3e-7, -1e-7, 2e-7])
    bank = KP.sensitivity_bank(_double_integrator(t, g0), t)
    s = KP.drift_interval_values(t)
    ref = np.zeros((t.size, 6))
    for i in range(t.size - 1):                      # sum of exact interval windows weighted by s_i
        ref += s[i] * _window_closed_form(t, g0, t[i], t[i + 1])
    assert np.max(np.abs(bank["DRIFT"] - ref)) <= 1e-11 * np.max(np.abs(bank["K"]))


def test_slot_history_packs_columns_in_stm_column_order():
    nom = np.random.default_rng(3).normal(size=(5, 48))
    cols = [np.full((5, 6), float(j + 1)) for j in range(3)]
    h = KP.slot_history(nom, cols)
    for j in range(3):
        phi = h[0, 6:42].reshape((6, 6), order="F")
        assert np.all(phi[:, j] == j + 1)
    assert np.all(h[:, 6 + 18:42] == 0) and np.array_equal(h[:, :6], nom[:, :6])


# ----------------------------------------------------------------------
# information geometry
# ----------------------------------------------------------------------
def _toy():
    rng = np.random.default_rng(11)
    c = rng.normal(size=(200, 8))
    hk = 0.3 * c @ rng.normal(size=8) + rng.normal(size=200)
    p = np.column_stack([hk] + [0.5 * hk + rng.normal(size=200) for _ in range(4)])
    return c, p


def test_scalar_model_sine_is_f_perp():
    c, p = _toy()
    q, _ = np.linalg.qr(c)
    b = p[:, 0]
    f = np.linalg.norm(b - q @ (q.T @ b)) / np.linalg.norm(b)
    assert abs(KP.block_metrics(c, p, "M0_SCALAR")["s_min"] - f) < 1e-14


@pytest.mark.parametrize("model", ["M1_TWO_SEGMENT", "M2_LINEAR_DRIFT", "M3_THREE_SEGMENT"])
def test_principal_sines_are_basis_invariant(model):
    c, p = _toy()
    hp = p @ KP.model_matrix(model)
    t = np.random.default_rng(5).normal(size=(hp.shape[1], hp.shape[1])) + 3 * np.eye(hp.shape[1])
    s1 = KP.independent_sines(c, hp)
    s2 = KP.independent_sines(c @ np.diag(np.arange(1.0, 9.0)), hp @ t)
    assert np.max(np.abs(np.sort(s1) - np.sort(s2))) < 1e-10
    m = KP.block_metrics(c, p, model)
    assert np.max(np.abs(np.sort(s1) - np.sort(m["sines"]))) < 1e-12
    assert m["s_min"] <= KP.block_metrics(c, p, "M0_SCALAR")["s_min"] + 1e-15   # block worst <= any member


def test_variation_sine_is_zero_when_the_variation_lies_in_state_plus_constant():
    c, p = _toy()
    p = p.copy()
    p[:, 1] = 0.7 * p[:, 0] + c @ np.arange(8.0)          # SEG2_A column inside span(c, h_K)
    assert KP.block_metrics(c, p, "M1_TWO_SEGMENT")["s_var_min"] < 1e-12


def test_exact_oracle_agrees_with_float_on_a_well_conditioned_problem():
    c, p = _toy()
    hp = p @ KP.model_matrix("M3_THREE_SEGMENT")
    ex = [float(x) for x in KP.exact_sines(c, hp)]
    fl = sorted(KP.block_metrics(c, p, "M3_THREE_SEGMENT")["sines"], reverse=True)
    assert max(abs(a - b) for a, b in zip(ex, fl)) < 1e-13


# ----------------------------------------------------------------------
# committed campaign evidence
# ----------------------------------------------------------------------
@pytest.mark.skipif(not SUMMARY_PATH.exists(), reason="KP1 campaign summary not yet produced")
def test_committed_campaign_gates_and_decision_are_consistent():
    s = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert s["predeclaration_sha256"] == PINNED
    g = s["gates"]
    assert g["ALL_PASS"] is True and g["bank_k_bitwise"] is True
    assert g["anchors_max_abs_diff"] <= 1e-10 and g["rebuild_max_abs_diff"] <= 1e-8
    assert g["exact_oracle_max_abs_diff"] <= 1e-12 and g["fd_worst"] <= 1e-4
    prim = s["table"]["HIGH_FIDELITY"]["RPD_D1FREE"]
    classes = {m: prim[m]["CLASS"] for m in KP.MODELS if m != "M0_SCALAR"}
    assert classes == s["per_model_class"]
    expected = ("ALTERNATIVE_K_PARAMETERIZATION_ADDS_DISTINGUISHABLE_DIRECTIONS"
                if "DISTINGUISHABLE" in classes.values() else
                "ALTERNATIVE_K_PARAMETERIZATION_REVEALS_ONLY_PARTIAL_SEPARATION"
                if "PARTIAL" in classes.values() else
                "ADDITIONAL_K_PARAMETERS_INCREASE_DEGENERACY_WITHOUT_NEW_INFORMATION")
    assert s["PRIMARY_CLASS"] == expected
    assert prim["M0_SCALAR"]["n"] == 120
