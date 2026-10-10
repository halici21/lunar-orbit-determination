"""Phase 17-KP2: nonlinear M3 K_SRP recovery - contract and oracle tests.

Pins the predeclaration, the segmented-propagation chaining (against a closed-form
double integrator with piecewise-constant forcing), the STM packing, the
variation basis and recovery metrics, and the committed campaign evidence.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import phase17_kp2_core as K2  # noqa: E402

SUMMARY = ROOT / "artifacts" / "phase17_kp2_summary.json"


def chi_threshold():
    return K2.CHI2_2_999


def _pinned():
    src = (EXAMPLES / "phase17_kp2_campaign.py").read_text(encoding="utf-8")
    return src.split('PREDECLARATION_SHA256 = "')[1].split('"')[0]


def test_predeclaration_is_pinned_and_tampering_is_refused(tmp_path):
    pre = K2.load_predeclaration(_pinned())
    assert pre["written_before_any_high_fidelity_recovery_result"] is True
    assert pre["truth"]["truths_fractional_segment_deviation"] == {k: list(v) for k, v in K2.TRUTHS.items()}
    bad = dict(pre, decision=dict(pre["decision"], A="relaxed"))
    p = tmp_path / "pre.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(K2.PredeclarationMismatch):
        K2.load_predeclaration(_pinned(), p)


def test_truths_share_the_common_level_and_span_the_variation_subspace():
    for name, f in K2.TRUTHS.items():
        assert abs(sum(f)) < 1e-15, name
    q = K2.Q_VAR
    assert np.allclose(q.T @ q, np.eye(2)) and np.allclose(q.sum(axis=0), 0.0)
    v1, v2 = q.T @ np.array(K2.TRUTHS["T1_STEP"]), q.T @ np.array(K2.TRUTHS["T2_MIDDLE"])
    assert abs(np.linalg.det(np.column_stack([v1, v2]))) > 1e-3       # T1 and T2 span both directions


# ----------------------------------------------------------------------
# segmented chaining against a closed form
# ----------------------------------------------------------------------
G0 = np.array([2e-7, -1e-7, 3e-7])


def _segment_closed_form(t_slice, x_start, k):
    """x'' = k g0: exact state, STM [[I, tau I], [0, I]] and dx/dk from the segment start."""
    tau = np.asarray(t_slice, float) - float(t_slice[0])
    n = tau.size
    out = np.zeros((n, 48))
    for i, s in enumerate(tau):
        out[i, :3] = x_start[:3] + x_start[3:] * s + 0.5 * k * G0 * s ** 2
        out[i, 3:6] = x_start[3:] + k * G0 * s
        phi = np.eye(6)
        phi[:3, 3:] = s * np.eye(3)
        out[i, 6:42] = phi.reshape(-1, order="F")
        out[i, 42:45] = 0.5 * G0 * s ** 2
        out[i, 45:48] = G0 * s
    return out


def test_segmented_chaining_matches_the_closed_form(monkeypatch):
    t = np.arange(0.0, 90.0 * 301, 90.0)
    scn = SimpleNamespace(t_grid=t)
    monkeypatch.setattr(K2, "_prop_segment", lambda scn, ts, xs, k, dyn: _segment_closed_form(ts, xs, k))
    ks = np.array([1.1, 0.9, 1.0]) * 0.01
    x0 = np.array([1e6, 2e5, -3e5, 10.0, -20.0, 5.0])
    X, PHI, S = K2.propagate_segmented(scn, x0, ks, "ANY")
    br = K2.boundaries(t)
    edges = [t[br[0]], t[br[1]], t[br[2]], t[br[3]]]
    for i in range(3):
        a, b = edges[i], edges[i + 1]
        lo = np.clip(t, a, b)
        # dx/dK_i = integral_a^lo (t - tau) g0 dtau ; velocity = (lo - a) g0
        pos = ((t * (lo - a) - 0.5 * (lo ** 2 - a ** 2)))[:, None] * G0
        vel = (lo - a)[:, None] * G0
        assert np.max(np.abs(S[i] - np.hstack([pos, vel]))) <= 1e-9 * np.max(np.abs(S[i])), i
    for n_idx in (0, 150, 300):
        phi = np.eye(6)
        phi[:3, 3:] = t[n_idx] * np.eye(3)
        assert np.allclose(PHI[n_idx], phi, atol=1e-9)
    # state continuity across the segment restarts: velocity gains k_j g0 per unit time
    v_end = x0[3:] + G0 * sum(ks[i] * (edges[i + 1] - edges[i]) for i in range(3))
    assert np.allclose(X[-1, 3:], v_end, rtol=0, atol=1e-12) and np.allclose(X[0], x0)


def test_nom48_packs_the_stm_column_major():
    rng = np.random.default_rng(2)
    X, PHI = rng.normal(size=(4, 6)), rng.normal(size=(4, 6, 6))
    n = K2.nom48(X, PHI, np.zeros((4, 6)))
    for i in range(4):
        assert np.array_equal(n[i, 6:42].reshape((6, 6), order="F"), PHI[i])


# ----------------------------------------------------------------------
# recovery metrics
# ----------------------------------------------------------------------
def test_recovery_metrics_on_a_known_posterior():
    f_true = np.array(K2.TRUTHS["T1_STEP"])
    f_hat = f_true + np.array([0.01, -0.005, 0.002])
    cov = np.eye(12) * 1e-6
    cov[6:9, 6:9] = np.diag([0.01, 0.012, 0.011]) ** 2 * K2.K0 ** 2
    th = np.concatenate([np.zeros(6), K2.K0 * (1 + f_hat), np.zeros(3)])
    m = K2.recovery_metrics(dict(theta=th.tolist(), cov=cov.tolist(), model="M3", converged=True, n_eval=5),
                            np.zeros(6), f_true)
    cf = np.diag([0.01, 0.012, 0.011]) ** 2
    cv = K2.Q_VAR.T @ cf @ K2.Q_VAR
    e = K2.Q_VAR.T @ (f_hat - f_true)
    assert abs(m["chi2_var_err"] - e @ np.linalg.solve(cv, e)) < 1e-9
    vh = K2.Q_VAR.T @ f_hat
    assert abs(m["chi2_var_detect"] - vh @ np.linalg.solve(cv, vh)) < 1e-9
    assert abs(m["level_err"] - np.mean(f_hat - f_true)) < 1e-15


# ----------------------------------------------------------------------
# committed campaign evidence
# ----------------------------------------------------------------------
@pytest.mark.skipif(not SUMMARY.exists(), reason="KP2 summary not yet produced")
def test_committed_campaign_evidence_is_internally_consistent():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["predeclaration_sha256"] == _pinned()
    for gate in ("KP1_ANCHOR", "NOISE_FREE", "M0_BASELINE", "NO_JOB_ERRORS"):
        assert s["gates"][gate] == "PASS", gate
    # the two frozen verification failures are reported as computed, with their post-hoc adjudication pinned
    assert s["gates"]["FD"] == "FAIL" and s["gates"]["PRODUCTION_BLS"] == "FAIL"
    post = json.loads((ROOT / "artifacts" / "phase17_kp2_posthoc_verification.json").read_text(encoding="utf-8"))
    assert post["fd_tight"]["worst"] <= 1e-4                          # FD exceedance = oracle integration floor
    assert post["bls_stationarity"]["status"] == "Converged"
    assert max(abs(x) for x in post["bls_stationarity"]["diff_over_sigma"]) <= 0.02   # same LS solution
    basin = json.loads((ROOT / "artifacts" / "phase17_kp2_posthoc_basin.json").read_text(encoding="utf-8"))
    assert basin["metrics"]["converged"] and basin["metrics"]["chi2_var_err"] <= chi_threshold()
    chi = K2.CHI2_2_999
    for t in s["trials"]:
        assert t["recovered"] == bool(t["converged"] and t["chi2_var_err"] <= chi and t["chi2_var_detect"] >= chi)
    expected = ("M3_VARIATION_NONLINEARLY_RECOVERABLE"
                if s["R_sep"] >= 0.80 and s["FP"] <= 0.20 and s["CONS"] >= 0.80 else
                "M3_VARIATION_PARTIALLY_RECOVERABLE" if s["R_sep"] >= 0.30 else
                "M3_LINEAR_SEPARATION_DOES_NOT_TRANSLATE_TO_NONLINEAR_RECOVERY")
    assert s["PRIMARY_CLASS"] == expected
    assert s["COMMON_LEVEL"]["status"] == "NOT_CLAIMED"
    assert len(s["trials"]) == 30
