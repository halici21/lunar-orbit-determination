"""Phase F3 core: linearised Kalman filter + RTS smoother with DMC, on closed-form synthetic systems.

The system is a 3-D harmonic oscillator (x'' = -n^2 x + K u + w), so Phi(t, t0), the constant-acceleration
sensitivity S(t) and the exact augmented [r, v, w] transition are known in closed form.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "studies" / "phase17" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import phase_f3_core as C  # noqa: E402

N = 2.0 * np.pi / 7200.0
U = np.array([0.6, 0.0, 0.8])
TAU = 1800.0


def phi_of(t):
    c, s = np.cos(N * t), np.sin(N * t)
    out = np.zeros((6, 6))
    out[:3, :3] = c * np.eye(3)
    out[:3, 3:] = s / N * np.eye(3)
    out[3:, :3] = -N * s * np.eye(3)
    out[3:, 3:] = c * np.eye(3)
    return out


def s_of(t):
    """Columns: K (constant acceleration along U), range bias (no dynamics)."""
    out = np.zeros((6, 2))
    out[:3, 0] = (1.0 - np.cos(N * t)) / N ** 2 * U
    out[3:, 0] = np.sin(N * t) / N * U
    return out


def reference(t_end):
    grid = np.arange(0.0, t_end + 20.0, 20.0)
    return C.Reference(grid, np.array([phi_of(t) for t in grid]), np.array([s_of(t) for t in grid]))


def rows_for(epochs, rng):
    """Range-like rows y = e . r + b, as batch partials w.r.t. [x0, K, b]."""
    out = []
    for t in epochs:
        e = rng.normal(size=3)
        e /= np.linalg.norm(e)
        hx = e @ phi_of(t)[:3]
        hc = np.array([e @ s_of(t)[:3, 0], 1.0])
        out.append((np.concatenate([hx, hc]), e))
    return out


SCALES_NO_W = [1.0] * 3 + [1e-3] * 3 + [1e-6, 1.0]
SCALES_W = [1.0] * 3 + [1e-3] * 3 + [1e-6] * 3 + [1e-6, 1.0]


def test_sigma_zero_smoother_equals_batch():
    rng = np.random.default_rng(1)
    epochs = np.arange(60.0, 3 * 7200.0, 60.0)
    rows = rows_for(epochs, rng)
    truth = np.array([30.0, -20.0, 10.0, 0.02, -0.01, 0.03, 2e-6, 5.0])
    h0 = np.array([h for h, _ in rows])
    r = h0 @ truth + rng.normal(size=len(rows))
    prior_mean = np.zeros(8)
    prior_cov = np.diag([1e4 ** 2] * 3 + [10.0 ** 2] * 3 + [1e-5 ** 2, 10.0 ** 2])
    ref = reference(epochs[-1])
    out = C.smooth(ref, 0.0, epochs, [(h[None, :], np.array([ri])) for h, ri in zip(h0, r)], prior_mean,
                   prior_cov, None, TAU, SCALES_NO_W)
    want, cov = C.batch_oracle(h0, r, prior_mean, prior_cov)
    sig = np.sqrt(np.diag(cov))
    assert np.max(np.abs(out["x"][0] - want) / sig) < 1e-6
    # the normal matrix has condition ~1e13 (10 km prior vs 1 m rows): agreement to 1e-4 is the double-precision floor
    np.testing.assert_allclose(np.sqrt(np.diag(out["cov"][0.0])), sig, rtol=1e-4)


def test_transition_matches_exact_augmented_exponential():
    ref = reference(1200.0)
    sigma = 1e-6
    f, q = C.transition(ref, 100.0, 700.0, sigma, TAU)
    a = np.zeros((9, 9))
    a[:3, 3:6] = np.eye(3)
    a[3:6, :3] = -N ** 2 * np.eye(3)
    a[3:6, 6:9] = np.eye(3)
    a[6:9, 6:9] = -np.eye(3) / TAU
    exact = expm(a * 600.0)
    row_scale = np.abs(exact).max(axis=1, keepdims=True)
    assert np.max(np.abs(f[:9, :9] - exact) / row_scale) < 1e-3
    # the constant-parameter column composes exactly: S(t_b) - Phi(t_b, t_a) S(t_a)
    np.testing.assert_allclose(f[:6, 9:], s_of(700.0) - phi_of(700.0) @ np.linalg.inv(phi_of(100.0)) @ s_of(100.0),
                               rtol=1e-6, atol=1e-9)
    # process noise against van Loan with the full A (gravity gradient included)
    qc = np.zeros((9, 9))
    qc[6:9, 6:9] = 2.0 * sigma ** 2 / TAU * np.eye(3)
    big = np.zeros((18, 18))
    big[:9, :9] = -a
    big[:9, 9:] = qc
    big[9:, 9:] = a.T
    e = expm(big * 600.0)
    q_exact = e[9:, 9:].T @ e[:9, 9:]
    for i in (0, 3, 6):
        assert abs(q[i, i] / q_exact[i, i] - 1.0) < 0.05


def test_dmc_smoother_is_consistent_and_beats_no_dmc():
    rng = np.random.default_rng(7)
    sigma = 3e-6
    dt = 10.0
    t_end = 3 * 7200.0
    a = np.zeros((9, 9))
    a[:3, 3:6] = np.eye(3)
    a[3:6, :3] = -N ** 2 * np.eye(3)
    a[3:6, 6:9] = np.eye(3)
    a[6:9, 6:9] = -np.eye(3) / TAU
    qc = np.zeros((9, 9))
    qc[6:9, 6:9] = 2.0 * sigma ** 2 / TAU * np.eye(3)
    big = np.zeros((18, 18))
    big[:9, :9] = -a
    big[:9, 9:] = qc
    big[9:, 9:] = a.T
    e = expm(big * dt)
    fd = e[9:, 9:].T
    qd = fd @ e[:9, 9:]
    lq = np.linalg.cholesky(0.5 * (qd + qd.T) + 1e-30 * np.eye(9))
    k_true, b_true = 2e-6, 3.0
    xk = np.concatenate([[50.0, -40.0, 20.0, 0.03, 0.01, -0.02], rng.normal(size=3) * sigma])
    times = np.arange(0.0, t_end + dt, dt)
    states = []
    for _ in times:
        states.append(xk.copy())
        # exact zero-order effect of the constant acceleration over dt is added through S
        xk = fd @ xk + lq @ rng.normal(size=9)
    states = np.array(states)
    # constant part K u enters as S(t) K (linear superposition)
    pos = states[:, :3] + np.array([s_of(t)[:3, 0] for t in times]) * k_true
    epochs = np.arange(60.0, t_end, 60.0)
    rows = []
    truth_pos = []
    for t in epochs:
        i = int(round(t / dt))
        ev = rng.normal(size=3)
        ev /= np.linalg.norm(ev)
        h = np.concatenate([ev @ phi_of(t)[:3], [ev @ s_of(t)[:3, 0], 1.0]])
        rows.append((h[None, :], np.array([ev @ pos[i] + b_true + rng.normal()])))
        truth_pos.append(pos[i])
    truth_pos = np.array(truth_pos)
    prior_mean = np.zeros(8)
    prior_cov = np.diag([1e3 ** 2] * 3 + [1.0 ** 2] * 3 + [1e-5 ** 2, 10.0 ** 2])
    ref = reference(t_end)

    def pos_err(out):
        return np.sqrt(np.mean(np.sum((out["x"][1:, :3] - truth_pos) ** 2, axis=1)))

    with_dmc = C.smooth(ref, 0.0, epochs, rows, prior_mean, prior_cov, sigma, TAU, SCALES_W)
    no_dmc = C.smooth(ref, 0.0, epochs, rows, prior_mean, prior_cov, None, TAU, SCALES_NO_W)
    assert 0.8 < with_dmc["nis"].mean() < 1.25
    assert no_dmc["nis"].mean() > 2.0
    assert pos_err(with_dmc) < 0.5 * pos_err(no_dmc)
