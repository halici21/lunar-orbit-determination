"""WP4.8b (EST-21 / RD-12): DMC empirical acceleration in RTN axes with per-axis sigma.

Before WP4.8b the Gauss-Markov acceleration w was inertial with one sigma for all
three axes, so an along-track drag-like or SRP-like error could not be given its own
size.  With ``axes="rtn"`` the acceleration is C(r, v) w, C = [R T N]
(R = r/|r|, N = r x v/|r x v|, T = N x R), and ``sigma_axes_mps2`` sets one sigma per
axis.  Oracles: central finite differences (the basis Jacobian and the 9x9 STM, which
now carries d(Cw)/dr and d(Cw)/dv), the rotation identity C Q_iso C^T = Q_iso, and
w = 0 reducing to the inertial propagation exactly.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from lunar_od.dmc import (
    DMCOptions,
    dmc_process_noise,
    propagate_dmc_state,
    rtn_acceleration_and_jacobian,
)
from tests.test_dmc_process_noise_wp0_2 import _closed_form_q
from tests.test_ekf_smoother import MU_MOON, _earth, _sun

MU_EARTH = 3.986004418e14
TOL = dict(rtol=1e-12, atol=1e-13)


def _state9(w=(3e-6, -2e-6, 1e-6)):
    r0 = 1737.4e3 + 100e3
    return np.array([r0, 30e3, -20e3, -15.0, math.sqrt(MU_MOON / r0), 4.0, *w])


def _basis(r, v):
    rh = r / np.linalg.norm(r)
    h = np.cross(r, v)
    nh = h / np.linalg.norm(h)
    return np.column_stack([rh, np.cross(nh, rh), nh])


def test_rtn_basis_is_orthonormal_and_right_handed():
    x = _state9()
    c = _basis(x[:3], x[3:6])
    a, _, _ = rtn_acceleration_and_jacobian(x[:3], x[3:6], x[6:9])
    np.testing.assert_allclose(c.T @ c, np.eye(3), atol=1e-15)
    assert np.linalg.det(c) == pytest.approx(1.0, abs=1e-15)
    np.testing.assert_allclose(a, c @ x[6:9], rtol=1e-15, atol=1e-22)
    assert c[:, 1] @ x[3:6] > 0.0                      # T points along the velocity


def test_rtn_acceleration_jacobian_matches_central_differences():
    x = _state9()
    r, v, w = x[:3], x[3:6], x[6:9]
    _, j_r, j_v = rtn_acceleration_and_jacobian(r, v, w)
    for jac, base, h in ((j_r, 0, 1.0), (j_v, 3, 1e-3)):
        for k in range(3):
            plus, minus = x.copy(), x.copy()
            plus[base + k] += h
            minus[base + k] -= h
            fd = (_basis(plus[:3], plus[3:6]) @ w - _basis(minus[:3], minus[3:6]) @ w) / (2 * h)
            np.testing.assert_allclose(jac[:, k], fd, rtol=1e-6, atol=1e-9 * np.linalg.norm(w) / h)


def test_rtn_stm_matches_central_finite_differences():
    options = DMCOptions(sigma_mps2=5e-6, tau_s=1800.0, axes="rtn")
    x0 = _state9()
    t = [0.0, 3600.0]
    phi = propagate_dmc_state(t, x0, options, MU_MOON, MU_EARTH, 0.0, _earth, _sun, **TOL)[-1, 9:].reshape(
        9, 9, order="F")
    steps = [1.0] * 3 + [1e-3] * 3 + [1e-6] * 3
    for j, h in enumerate(steps):
        plus, minus = x0.copy(), x0.copy()
        plus[j] += h
        minus[j] -= h
        f = [propagate_dmc_state(t, x, options, MU_MOON, MU_EARTH, 0.0, _earth, _sun, stm=False, **TOL)[-1]
             for x in (plus, minus)]
        column = (f[0] - f[1]) / (2.0 * h)
        assert np.linalg.norm(column - phi[:, j]) <= 1e-6 * np.linalg.norm(phi[:, j]), j


def test_rtn_and_inertial_differ_only_through_the_rotation():
    """w = 0: identical trajectories; w != 0: the RTN run equals an inertial run only if
    w is re-expressed, so the two genuinely differ (the option is not a no-op)."""
    t = np.arange(0.0, 3601.0, 600.0)
    zero = _state9(w=(0.0, 0.0, 0.0))
    rtn = DMCOptions(5e-6, 1800.0, axes="rtn")
    inertial = DMCOptions(5e-6, 1800.0)
    a = propagate_dmc_state(t, zero, rtn, MU_MOON, MU_EARTH, 0.0, _earth, _sun, stm=False, **TOL)
    b = propagate_dmc_state(t, zero, inertial, MU_MOON, MU_EARTH, 0.0, _earth, _sun, stm=False, **TOL)
    np.testing.assert_array_equal(a, b)
    # with the STM the r/v block agrees (d(Cw)/dx vanishes at w = 0); the w columns are C
    pa = propagate_dmc_state(t, zero, rtn, MU_MOON, MU_EARTH, 0.0, _earth, _sun, **TOL)[-1, 9:].reshape(9, 9, order="F")
    pb = propagate_dmc_state(t, zero, inertial, MU_MOON, MU_EARTH, 0.0, _earth, _sun, **TOL)[-1, 9:].reshape(9, 9, order="F")
    np.testing.assert_allclose(pa[:6, :6], pb[:6, :6], rtol=1e-9, atol=1e-9)
    x = _state9()
    a = propagate_dmc_state(t, x, rtn, MU_MOON, MU_EARTH, 0.0, _earth, _sun, stm=False, **TOL)
    b = propagate_dmc_state(t, x, inertial, MU_MOON, MU_EARTH, 0.0, _earth, _sun, stm=False, **TOL)
    assert np.linalg.norm(a[-1, :3] - b[-1, :3]) > 1.0


def test_isotropic_rtn_q_is_the_rotated_inertial_q():
    """Isotropic sigma: the r/v and w blocks are rotation-invariant (C q I C^T = q I); the
    cross blocks to w carry C, because w is now expressed in RTN coordinates."""
    x = _state9()
    c = _basis(x[:3], x[3:6])
    for dt in (60.0, 1800.0, 86400.0):
        q_i = dmc_process_noise(DMCOptions(5e-6, 1800.0), dt)
        q_r = dmc_process_noise(DMCOptions(5e-6, 1800.0, axes="rtn"), dt, state6=x[:6])
        atol = 1e-14 * np.max(np.abs(q_i))
        np.testing.assert_allclose(q_r[:6, :6], q_i[:6, :6], rtol=1e-12, atol=atol)
        np.testing.assert_allclose(q_r[6:, 6:], q_i[6:, 6:], rtol=1e-12, atol=atol)
        np.testing.assert_allclose(q_r[:6, 6:], q_i[:6, 6:] @ c, rtol=1e-12, atol=atol)
        np.testing.assert_allclose(q_r, q_r.T, rtol=0.0, atol=0.0)


def test_along_track_only_sigma_puts_q_along_t():
    x = _state9()
    options = DMCOptions(0.0, 1800.0, axes="rtn", sigma_axes_mps2=(0.0, 5e-6, 0.0))
    q = dmc_process_noise(options, 1800.0, state6=x[:6])
    t_hat = _basis(x[:3], x[3:6])[:, 1]
    iso = dmc_process_noise(DMCOptions(5e-6, 1800.0), 1800.0)
    for block in (slice(0, 3), slice(3, 6)):
        np.testing.assert_allclose(q[block, block], iso[block, block][0, 0] * np.outer(t_hat, t_hat),
                                   rtol=1e-12, atol=1e-14 * iso[block, block][0, 0])
    np.testing.assert_allclose(np.diag(q[6:, 6:]), [0.0, iso[6, 6], 0.0], rtol=1e-12, atol=0.0)


def test_per_axis_sigma_in_inertial_axes():
    q = dmc_process_noise(DMCOptions(0.0, 600.0, sigma_axes_mps2=(1e-6, 2e-6, 3e-6)), 300.0)
    unit = dmc_process_noise(DMCOptions(1.0, 600.0), 300.0)
    for k, s in enumerate((1e-6, 2e-6, 3e-6)):
        idx = [k, 3 + k, 6 + k]
        np.testing.assert_allclose(q[np.ix_(idx, idx)], s**2 * unit[np.ix_(idx, idx)], rtol=1e-13)


def test_invalid_rtn_options_rejected():
    with pytest.raises(ValueError):
        DMCOptions(1e-6, 600.0, axes="lvlh")
    with pytest.raises(ValueError):
        DMCOptions(1e-6, 600.0, sigma_axes_mps2=(1e-6, -1e-6, 0.0))
    with pytest.raises(ValueError):
        dmc_process_noise(DMCOptions(1e-6, 600.0, axes="rtn"), 60.0)    # needs state6


@pytest.mark.parametrize("sigma", [1e-12, 1e-8, 1e-4, 1e-2, 1.0])
def test_q_accuracy_is_independent_of_sigma(sigma):
    """van Loan (beta*dt <= 1) evaluated at sigma itself reached 2.8e-9 at 1 m/s^2; Q is now
    evaluated at a reference sigma and scaled (Q is linear in sigma^2)."""
    for dt in (60.0, 1800.0, 86400.0):
        q = dmc_process_noise(DMCOptions(sigma, 1800.0), dt)
        ref = _closed_form_q(sigma, 1800.0, dt)
        np.testing.assert_allclose(q, ref, rtol=1e-12, atol=1e-14 * np.max(np.abs(ref)))
