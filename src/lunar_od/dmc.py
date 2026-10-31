"""Dynamic model compensation (DMC): a first-order Gauss-Markov empirical acceleration.

The spacecraft state is augmented to ``[r, v, w]`` with

    r' = v,  v' = a_model(r, t) + w,  w' = -w / tau + u,   E[u u^T] = (2 sigma^2 / tau) I dt

(Tapley, Schutz & Born 2004, section 4.9). ``w`` absorbs accelerations the force
model misses; ``sigma`` is its steady-state standard deviation per axis and
``tau`` its correlation time.

The force model is ``dynamics.f3body_moon`` and its gradient
``dynamics.dynamics_jacobian_a_matrix`` (Moon point mass, Earth/Sun third body,
lunar/Earth J2, cannonball SRP), so with ``w = 0`` the trajectory is the one
``propagate_augmented_state`` integrates. Lunar spherical harmonics are not
supported here. ``dynamics.py`` itself is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from .dynamics import dynamics_jacobian_a_matrix, f3body_moon

DMC_SIZE = 3


@dataclass(frozen=True)
class DMCOptions:
    """Gauss-Markov empirical acceleration: steady-state sigma [m/s^2] per axis, correlation time [s]."""

    sigma_mps2: float
    tau_s: float

    def __post_init__(self) -> None:
        if not (self.sigma_mps2 >= 0.0 and np.isfinite(self.sigma_mps2)):
            raise ValueError("sigma_mps2 must be finite and non-negative.")
        if not (self.tau_s > 0.0 and np.isfinite(self.tau_s)):
            raise ValueError("tau_s must be finite and positive.")


#: Above this beta*dt the closed form replaces van Loan, whose -A block grows as exp(+beta*dt).
_DMC_CLOSED_FORM_MIN_BETA_DT = 1.0


def dmc_process_noise(options: DMCOptions, dt_s: float) -> np.ndarray:
    """Exact discrete process noise of [r, v, w] for the Gauss-Markov model (9x9).

    The kinematic coupling r' = v, v' = w is exact; the gravity gradient is left
    out of Q, as is standard (its effect over one step is second order).

    Short steps use van Loan's matrix exponential. Its -A block grows as
    exp(beta*dt), losing precision near beta*dt ~ 30 and overflowing past ~700
    (assessment EST-09), so longer steps use the closed-form integrals of
    Phi(s) G q G^T Phi(s)^T, which contain only exp(-beta*dt).
    """
    dt = abs(float(dt_s))
    beta = 1.0 / options.tau_s
    if beta * dt > _DMC_CLOSED_FORM_MIN_BETA_DT:
        return _dmc_process_noise_closed_form(options.sigma_mps2, beta, dt)
    a = np.zeros((9, 9))
    a[0:3, 3:6] = np.eye(3)
    a[3:6, 6:9] = np.eye(3)
    a[6:9, 6:9] = -beta * np.eye(3)
    qc = np.zeros((9, 9))
    qc[6:9, 6:9] = 2.0 * options.sigma_mps2**2 * beta * np.eye(3)
    big = np.zeros((18, 18))
    big[:9, :9] = -a
    big[:9, 9:] = qc
    big[9:, 9:] = a.T
    e = expm(big * dt)
    phi = e[9:, 9:].T
    q = phi @ e[:9, 9:]
    return 0.5 * (q + q.T)


def _dmc_process_noise_closed_form(sigma_mps2: float, beta: float, dt: float) -> np.ndarray:
    """Per-axis integrals of g_i(s) g_j(s), g = [(beta s - 1 + e^{-beta s}) / beta^2, (1 - e^{-beta s}) / beta, e^{-beta s}]."""
    x = beta * dt
    e1 = np.exp(-x)
    e2 = np.exp(-2.0 * x)
    i_rr = (x**3 / 3.0 - x**2 + x + 0.5 - 0.5 * e2 - 2.0 * x * e1) / beta**5
    i_rv = (0.5 * x**2 - x + 0.5 - e1 + x * e1 + 0.5 * e2) / beta**4
    i_rw = (0.5 * (1.0 - e2) - x * e1) / beta**3
    i_vv = (x - 2.0 * (1.0 - e1) + 0.5 * (1.0 - e2)) / beta**3
    i_vw = ((1.0 - e1) - 0.5 * (1.0 - e2)) / beta**2
    i_ww = 0.5 * (1.0 - e2) / beta
    per_axis = 2.0 * sigma_mps2**2 * beta * np.array([[i_rr, i_rv, i_rw],
                                                      [i_rv, i_vv, i_vw],
                                                      [i_rw, i_vw, i_ww]])
    return np.kron(per_axis, np.eye(3))


def propagate_dmc_state(
    t_eval_s: ArrayLike,
    state9: ArrayLike,
    options: DMCOptions,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    *,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    j2_moon: float = 0.0,
    srp=None,
    stm: bool = True,
) -> np.ndarray:
    """Propagate [r, v, w] (and its 9x9 STM, column-major) to ``t_eval_s``.

    Returns shape ``(len(t_eval_s), 90)`` with ``stm=True`` (state 9 + Phi 81),
    ``(len(t_eval_s), 9)`` otherwise. Integrator: DOP853.
    """
    t_eval = np.asarray(t_eval_s, dtype=float).reshape(-1)
    x0 = np.asarray(state9, dtype=float).reshape(-1)
    if x0.size != 9:
        raise ValueError("state9 must be [r(3), v(3), w(3)].")
    beta = 1.0 / options.tau_s
    force = dict(j2_moon=j2_moon, srp=srp)

    def rhs(t, y):
        r_e = np.asarray(get_earth_pos(float(t)), dtype=float).reshape(-1)[:3]
        r_s = np.asarray(get_sun_pos(float(t)), dtype=float).reshape(-1)[:3]
        x6 = y[:6]
        dx = f3body_moon(x6, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2, r_e, r_s, **force)
        out = np.empty_like(y)
        out[:3] = dx[:3]
        out[3:6] = dx[3:6] + y[6:9]
        out[6:9] = -beta * y[6:9]
        if stm:
            a9 = np.zeros((9, 9))
            a9[:6, :6] = dynamics_jacobian_a_matrix(x6, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
                                                    r_e, r_s, **force)
            a9[3:6, 6:9] = np.eye(3)
            a9[6:9, 6:9] = -beta * np.eye(3)
            phi = y[9:].reshape(9, 9, order="F")
            out[9:] = (a9 @ phi).reshape(-1, order="F")
        return out

    y0 = np.concatenate([x0, np.eye(9).reshape(-1, order="F")]) if stm else x0
    if t_eval.size == 1 or np.all(t_eval == t_eval[0]):
        return np.tile(y0, (t_eval.size, 1))
    sol = solve_ivp(rhs, (float(t_eval[0]), float(t_eval[-1])), y0, method="DOP853",
                    t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"DMC propagation failed: {sol.message}")
    return sol.y.T
