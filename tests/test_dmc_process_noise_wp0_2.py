"""WP0.2 (EST-09): DMC process noise finite and accurate for every gap length.

The van Loan block exp([[-A, Qc], [0, A^T]] dt) contains +beta on the diagonal of -A,
so it grows like exp(dt / tau) and the product phi @ e[:9, 9:] cancels: at dt/tau = 30
the result lost its accuracy, at ~144 (a one-day gap with tau = 600 s) it was no longer
positive semidefinite, at ~700 it overflowed.  Oracle: the closed-form integrals of the
Gauss-Markov / double-integrator model, Q = q_c int_0^T g(s) g(s)^T ds with
g = Phi(s) e_w = [(s - (1 - e^{-bs})/b)/b, (1 - e^{-bs})/b, e^{-bs}], q_c = 2 sigma^2 b,
evaluated in 60-digit decimal arithmetic (no cancellation at any T).
"""
from __future__ import annotations

from decimal import Decimal, getcontext

import numpy as np
import pytest

from lunar_od.dmc import DMCOptions, dmc_process_noise


def _closed_form_q(sigma: float, tau: float, dt: float) -> np.ndarray:
    getcontext().prec = 60
    b, t = 1 / Decimal(tau), Decimal(dt)
    qc = 2 * Decimal(sigma) ** 2 * b
    e1, e2 = (-b * t).exp(), (-2 * b * t).exp()
    i_e = (1 - e1) / b                              # int e^{-bs}
    i_e2 = (1 - e2) / (2 * b)                       # int e^{-2bs}
    i_se = (1 - e1 * (1 + b * t)) / b**2            # int s e^{-bs}
    i_1me2 = t - 2 * i_e + i_e2                     # int (1 - e)^2
    i_s1me = t**2 / 2 - i_se                        # int s (1 - e)
    q_ww = qc * i_e2
    q_vw = qc / b * (i_e - i_e2)
    q_vv = qc / b**2 * i_1me2
    q_rw = qc / b * (i_se - (i_e - i_e2) / b)
    q_rv = qc / b**2 * (i_s1me - i_1me2 / b)
    q_rr = qc / b**2 * (t**3 / 3 - 2 * i_s1me / b + i_1me2 / b**2)
    axis = np.array([[q_rr, q_rv, q_rw], [q_rv, q_vv, q_vw], [q_rw, q_vw, q_ww]], dtype=float)
    return np.kron(axis, np.eye(3))


@pytest.mark.parametrize("dt_over_tau", [1e-3, 0.5, 1.0, 30.0, 144.0, 700.0, 5000.0])
def test_dmc_q_matches_closed_form_at_every_gap(dt_over_tau):
    options = DMCOptions(sigma_mps2=1e-8, tau_s=600.0)
    dt = dt_over_tau * options.tau_s
    q = dmc_process_noise(options, dt)
    reference = _closed_form_q(options.sigma_mps2, options.tau_s, dt)
    assert np.all(np.isfinite(q))
    np.testing.assert_allclose(q, reference, rtol=1e-9, atol=1e-12 * np.max(np.abs(reference)))
    scale = np.sqrt(np.diag(reference))
    assert np.min(np.linalg.eigvalsh(q / scale[:, None] / scale[None, :])) > -1e-9


def test_dmc_q_at_zero_gap_is_zero():
    np.testing.assert_array_equal(dmc_process_noise(DMCOptions(1e-8, 600.0), 0.0), np.zeros((9, 9)))
