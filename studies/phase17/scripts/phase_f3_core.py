"""Phase F3 core: a linearised Kalman filter + RTS smoother with DMC on deviations from a batch reference.

Predeclaration: studies/phase17/configs/phase_f3_lro_dmc_smoother_predeclaration.json.  No LRO code here; the
driver (phase_f3_lro_dmc_smoother.py) supplies the reference and the tests use closed-form systems.

Filter state (physical units): [dx (6), w (3, DMC only), dc (nc)], where dx is the deviation of the Moon-centred
J2000 state from the reference at the epoch, w the Gauss-Markov acceleration and dc the deviation of the constant
parameters.  The reference supplies Phi(t, t0) and the constant-parameter sensitivity S(t) = dx(t)/dc on a grid;
both are cubic-spline interpolated.  A batch measurement row [H0_x, H0_c] (partials w.r.t. the epoch state and the
constants) becomes the local row [H0_x Phi(t)^-1, 0, H0_c - H0_x Phi(t)^-1 S(t)].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from lunar_od.dmc import DMCOptions, dmc_process_noise

SUBSTEP_S = 20.0


@dataclass
class Reference:
    """Phi(t, t0) (k, 6, 6) and S(t) (k, 6, nc) on ``grid`` (k,), spline-interpolated."""

    grid: np.ndarray
    phi: np.ndarray
    s: np.ndarray

    def __post_init__(self):
        k = self.grid.size
        self.nc = self.s.shape[2]
        self._phi = CubicSpline(self.grid, self.phi.reshape(k, 36), axis=0)
        self._s = CubicSpline(self.grid, self.s.reshape(k, 6 * self.nc), axis=0)

    def at(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        return self._phi(t).reshape(-1, 6, 6), self._s(t).reshape(-1, 6, self.nc)


def substep_matrices(phi1_phi0inv, ds, h, sigma, tau, n):
    """One sub-step [dx, w, dc]: Phi_xx from the reference, w coupling by the trapezoid rule, van Loan Q."""
    f = np.eye(n)
    f[:6, :6] = phi1_phi0inv
    q = None
    if sigma is not None:
        e = np.exp(-h / tau)
        b = np.zeros((6, 3))
        b[3:6] = np.eye(3)
        f[:6, 6:9] = 0.5 * h * (phi1_phi0inv @ b + b * e)
        f[6:9, 6:9] = e * np.eye(3)
        f[:6, 9:] = ds
        q = np.zeros((n, n))
        q[:9, :9] = dmc_process_noise(DMCOptions(sigma, tau), h)
    else:
        f[:6, 6:] = ds
    return f, q


def transition(ref: Reference, t_a: float, t_b: float, sigma, tau: float, substep: float = SUBSTEP_S):
    """(F, Q) from t_a to t_b, composed over equal sub-steps no longer than ``substep``.
    ``sigma`` None: no w in the state (the sigma = 0 control)."""
    nw = 0 if sigma is None else 3
    n = 6 + nw + ref.nc
    m = max(1, int(np.ceil((t_b - t_a) / substep - 1e-9)))
    ts = np.linspace(t_a, t_b, m + 1)
    phi, s = ref.at(ts)
    inv = np.linalg.inv(phi)
    f_tot, q_tot = np.eye(n), np.zeros((n, n))
    for i in range(m):
        p10 = phi[i + 1] @ inv[i]
        ds = s[i + 1] - p10 @ s[i]
        f, q = substep_matrices(p10, ds, ts[i + 1] - ts[i], sigma, tau, n)
        f_tot = f @ f_tot
        q_tot = f @ q_tot @ f.T + (q if q is not None else 0.0)
    return f_tot, 0.5 * (q_tot + q_tot.T)


def local_rows(ref: Reference, t: float, h0: np.ndarray, nw: int):
    """Batch rows h0 (m, 6 + nc) at epoch t -> filter rows (m, 6 + nw + nc)."""
    phi, s = ref.at(t)
    hx = np.linalg.solve(phi[0].T, h0[:, :6].T).T     # H0_x Phi^-1
    hc = h0[:, 6:] - hx @ s[0]
    return np.hstack([hx, np.zeros((h0.shape[0], nw)), hc])


def smooth(ref: Reference, t0: float, epochs, rows, prior_mean, prior_cov, sigma, tau, scales,
           keep_cov_at=()):
    """Forward Kalman filter and RTS smoother.

    epochs: sorted epochs (> t0).  rows: per epoch, None or (h0 (m, 6 + nc), r (m,)) with unit weights; scalar
    updates in order.  prior_mean / prior_cov: [dx0, dc] at t0 (w gets N(0, sigma^2) when sigma is not None).
    scales: physical size of one unit per filter-state component (the filter works in scaled units).
    Returns smoothed states at t0 and every epoch, the smoothed covariance at t0 and the NIS of every update.
    """
    nw = 0 if sigma is None else 3
    nc = ref.nc
    n = 6 + nw + nc
    d = np.asarray(scales, float)
    m0 = np.zeros(n)
    p0 = np.zeros((n, n))
    m0[:6], m0[6 + nw:] = prior_mean[:6], prior_mean[6:]
    idx = np.r_[0:6, 6 + nw:n]
    p0[np.ix_(idx, idx)] = prior_cov
    if nw:
        p0[6:9, 6:9] = sigma ** 2 * np.eye(3)
    x = m0 / d
    p = p0 / np.outer(d, d)
    times = [t0] + list(epochs)
    n_ep = len(times)
    xf = np.empty((n_ep, n))
    pf = np.empty((n_ep, n, n))
    xp = np.empty((n_ep, n))
    pp = np.empty((n_ep, n, n))
    fs = np.empty((n_ep, n, n))
    nis = []
    for k, t in enumerate(times):
        if k:
            f, q = transition(ref, times[k - 1], t, sigma, tau)
            f = f * (d[None, :] / d[:, None])
            q = q / np.outer(d, d)
            x = f @ x
            p = f @ p @ f.T + q
            p = 0.5 * (p + p.T)
            fs[k] = f
        xp[k], pp[k] = x, p
        rk = rows[k - 1] if k else None
        if rk is not None:
            h = local_rows(ref, t, rk[0], nw) * d[None, :]
            for hi, ri in zip(h, rk[1]):
                ph = p @ hi
                s = hi @ ph + 1.0
                g = ph / s
                nu = ri - hi @ x
                nis.append(nu * nu / s)
                x = x + g * nu
                a = np.eye(n) - np.outer(g, hi)
                p = a @ p @ a.T + np.outer(g, g)
        xf[k], pf[k] = x, p
    xs = xf.copy()
    ps_next = pf[-1]
    keep = {}
    for k in range(n_ep - 2, -1, -1):
        c = np.linalg.solve(pp[k + 1], fs[k + 1] @ pf[k]).T
        xs[k] = xf[k] + c @ (xs[k + 1] - xp[k + 1])
        ps_k = pf[k] + c @ (ps_next - pp[k + 1]) @ c.T
        if k == 0 or times[k] in keep_cov_at:
            keep[times[k]] = ps_k * np.outer(d, d)
        ps_next = ps_k
    return dict(times=np.array(times), x=xs * d[None, :], x_filtered=xf * d[None, :], cov=keep,
                nis=np.array(nis), n_state=n, nw=nw)


def batch_oracle(h0, r, prior_mean, prior_cov):
    """sigma = 0 reference: weighted least squares with the prior, in epoch form (unit-weight rows)."""
    info = np.linalg.inv(prior_cov)
    a = h0.T @ h0 + info
    b = h0.T @ r + info @ prior_mean
    cov = np.linalg.inv(a)
    return cov @ b, cov
