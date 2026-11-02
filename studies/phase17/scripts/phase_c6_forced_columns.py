"""Phase C6 (A3): sensitivity columns of state-independent forcings by direct variational integration.

F3's cached reference stores Phi(t, t0) from finite differences. Over 3.6 days cond(Phi) reaches ~1e12, so any
product Phi(t) Phi(tau)^-1 (the C6 quadrature, and F3's local transitions) loses all accuracy after about a day.
This module never inverts Phi: it integrates

    d/dt X = A(t) X + [0; F(t)],   X(t0) = 0,     A = [[0, I], [G(t, r_ref(t)), 0]]

along the cached trajectory with RK4 sub-steps, G the full gravity gradient (Moon point mass + GL1800F harmonics
in MOON_PA_DE440 + Earth and Sun third body). States between the 20 s grid nodes come from a cubic Hermite spline;
forcings are interpolated linearly. Each column of F is one forcing (m/s^2 per unit parameter).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicHermiteSpline


def sub_nodes(grid, states, n_sub: int = 10):
    """RK4 sub-step node times (stages included) and the Hermite-interpolated spacecraft positions there."""
    grid = np.asarray(grid, float)
    states = np.asarray(states, float)
    pos = CubicHermiteSpline(grid, states[:, :3], states[:, 3:6], axis=0)
    sub = np.concatenate([np.linspace(grid[i], grid[i + 1], 2 * n_sub + 1)[:-1] for i in range(grid.size - 1)]
                         + [grid[-1:]])
    return sub, pos(sub)


def forced_columns(grid, states, forcings, gradient, n_sub: int = 10, local_transitions: bool = False,
                   g_nodes=None):
    """X on the grid, (k, 6, nf); gradient(t, r) -> (3, 3).

    ``forcings`` is either (k, 3, nf) on the grid (interpolated linearly to the sub-step nodes) or (m, 3, nf)
    already evaluated at the sub_nodes() times (A4: exact at shadow transitions). ``g_nodes`` may pass the
    gradients at the sub-step nodes from an earlier call. With ``local_transitions`` also returns the (k-1, 6, 6)
    transitions over each grid interval.
    """
    grid = np.asarray(grid, float)
    states = np.asarray(states, float)
    forcings = np.asarray(forcings, float)
    k, nf = grid.size, forcings.shape[2]
    sub, r_sub = sub_nodes(grid, states, n_sub)
    # gradient at every sub-step node (shared between neighbouring RK4 stages)
    if g_nodes is None:
        g_nodes = np.array([gradient(t, r) for t, r in zip(sub, r_sub)])
    if forcings.shape[0] == sub.size:
        f_nodes = forcings
    else:
        f_nodes = np.stack([np.stack([np.interp(sub, grid, forcings[:, a, j]) for a in range(3)], axis=1)
                            for j in range(nf)], axis=2)                      # (m, 3, nf)
    out = np.zeros((k, 6, nf))
    trans = np.zeros((k - 1, 6, 6)) if local_transitions else None

    def rhs(x, g, f):
        d = np.empty_like(x)
        d[:3] = x[3:]
        d[3:] = g @ x[:3] + f
        return d

    x = np.zeros((6, nf))
    e = np.eye(6)
    zero_f = np.zeros((3, 6))
    for i in range(k - 1):
        dt = (grid[i + 1] - grid[i]) / n_sub
        phi = e.copy()
        for s in range(n_sub):
            j0 = i * 2 * n_sub + 2 * s
            g0, g1, g2 = g_nodes[j0], g_nodes[j0 + 1], g_nodes[j0 + 2]
            f0, f1, f2 = f_nodes[j0], f_nodes[j0 + 1], f_nodes[j0 + 2]
            k1 = rhs(x, g0, f0)
            k2 = rhs(x + 0.5 * dt * k1, g1, f1)
            k3 = rhs(x + 0.5 * dt * k2, g1, f1)
            k4 = rhs(x + dt * k3, g2, f2)
            x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            if local_transitions:
                q1 = rhs(phi, g0, zero_f)
                q2 = rhs(phi + 0.5 * dt * q1, g1, zero_f)
                q3 = rhs(phi + 0.5 * dt * q2, g1, zero_f)
                q4 = rhs(phi + dt * q3, g2, zero_f)
                phi = phi + dt / 6.0 * (q1 + 2 * q2 + 2 * q3 + q4)
        out[i + 1] = x
        if local_transitions:
            trans[i] = phi
    return (out, trans) if local_transitions else out


def lunar_gradient(et0: float, model, mu_earth: float, mu_sun: float, earth_fn, sun_fn):
    """gradient(t, r) for F2's force model: Moon point mass + harmonics (MOON_PA_DE440 at et0 + t) + 3rd bodies."""
    from lunar_od.accelerated import pines_gradient_bf_fast
    from lunar_od.dynamics import moon_pa_de440_rotation_at_et, point_mass_gravity_gradient, \
        third_body_gravity_gradient

    def gradient(t, r):
        c = moon_pa_de440_rotation_at_et(et0 + float(t))
        g_bf = pines_gradient_bf_fast(c @ r, model.mu_m3_s2, model.r_ref_m, model.cbar, model.sbar,
                                      model.nmax, model.mmax)
        return (point_mass_gravity_gradient(r, model.mu_m3_s2) + c.T @ g_bf @ c
                + third_body_gravity_gradient(r, earth_fn(t), mu_earth)
                + third_body_gravity_gradient(r, sun_fn(t), mu_sun))
    return gradient


class LocalDynamics:
    """F3d: accurate local transitions Phi(t_b, t_a) from gravity gradients at the sub_nodes() times.

    G is interpolated linearly between the nodes (1 s apart for F3's 20 s grid and n_sub = 10); Phi(t_b, t_a)
    is RK4 from the identity with steps <= ``step`` s. Results are memoised by (t_a, t_b).
    """

    def __init__(self, grid, g_nodes, n_sub: int = 10, step: float = 1.0):
        grid = np.asarray(grid, float)
        d = np.diff(grid)
        assert np.allclose(d, d[0], rtol=0.0, atol=1e-9), "F3's grid is uniform"
        self.t0, self.dt = float(grid[0]), float(d[0]) / (2 * n_sub)
        self.g = np.asarray(g_nodes, float)
        assert self.g.shape[0] == 2 * n_sub * (grid.size - 1) + 1
        self.step = step
        self.cache = {}

    def gradient(self, t: float) -> np.ndarray:
        x = (t - self.t0) / self.dt
        i = min(max(int(np.floor(x)), 0), self.g.shape[0] - 2)
        f = x - i
        return (1.0 - f) * self.g[i] + f * self.g[i + 1]

    def phi(self, t_a: float, t_b: float) -> np.ndarray:
        key = (float(t_a), float(t_b))
        out = self.cache.get(key)
        if out is not None:
            return out
        m = max(1, int(np.ceil((t_b - t_a) / self.step - 1e-9)))
        h = (t_b - t_a) / m
        p = np.eye(6)

        def rhs(x, g):
            return np.vstack([x[3:], g @ x[:3]])
        for i in range(m):
            t = t_a + i * h
            g0, g1, g2 = self.gradient(t), self.gradient(t + 0.5 * h), self.gradient(t + h)
            k1 = rhs(p, g0)
            k2 = rhs(p + 0.5 * h * k1, g1)
            k3 = rhs(p + 0.5 * h * k2, g1)
            k4 = rhs(p + h * k3, g2)
            p = p + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.cache[key] = p
        return p
