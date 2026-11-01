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
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from .dynamics import dynamics_jacobian_a_matrix, f3body_moon

DMC_SIZE = 3
_SIGMA_REFERENCE_MPS2 = 1e-6


@dataclass(frozen=True)
class DMCOptions:
    """Gauss-Markov empirical acceleration: steady-state sigma [m/s^2] per axis, correlation time [s]."""

    sigma_mps2: float
    tau_s: float
    # WP4.8b: ``axes="rtn"`` makes the acceleration C(r, v) w with C = [R T N]
    # (R = r/|r|, N = r x v/|r x v|, T = N x R); ``sigma_axes_mps2`` gives one steady-state
    # sigma per axis (R, T, N or x, y, z) and replaces ``sigma_mps2``.
    axes: Literal["inertial", "rtn"] = "inertial"
    sigma_axes_mps2: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if not (self.sigma_mps2 >= 0.0 and np.isfinite(self.sigma_mps2)):
            raise ValueError("sigma_mps2 must be finite and non-negative.")
        if not (self.tau_s > 0.0 and np.isfinite(self.tau_s)):
            raise ValueError("tau_s must be finite and positive.")
        if self.axes not in ("inertial", "rtn"):
            raise ValueError("axes must be 'inertial' or 'rtn'.")
        if self.sigma_axes_mps2 is not None:
            sig = np.asarray(self.sigma_axes_mps2, dtype=float)
            if sig.shape != (3,) or not np.all(np.isfinite(sig)) or np.any(sig < 0.0):
                raise ValueError("sigma_axes_mps2 must be three finite non-negative values.")


def _skew(a: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


def rtn_acceleration_and_jacobian(
    r: ArrayLike, v: ArrayLike, w_rtn: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """a = C(r, v) w with C = [R T N]; returns (a, da/dr, da/dv)."""
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    w = np.asarray(w_rtn, dtype=float)
    r_norm = np.linalg.norm(r)
    r_hat = r / r_norm
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    n_hat = h / h_norm
    t_hat = np.cross(n_hat, r_hat)
    eye = np.eye(3)
    d_rhat_dr = (eye - np.outer(r_hat, r_hat)) / r_norm
    p_n = (eye - np.outer(n_hat, n_hat)) / h_norm
    d_nhat_dr = p_n @ (-_skew(v))                    # dh/dr = -[v]x
    d_nhat_dv = p_n @ _skew(r)                       # dh/dv = [r]x
    # d(N x R) = -[R]x dN + [N]x dR
    d_that_dr = -_skew(r_hat) @ d_nhat_dr + _skew(n_hat) @ d_rhat_dr
    d_that_dv = -_skew(r_hat) @ d_nhat_dv
    a = w[0] * r_hat + w[1] * t_hat + w[2] * n_hat
    j_r = w[0] * d_rhat_dr + w[1] * d_that_dr + w[2] * d_nhat_dr
    j_v = w[1] * d_that_dv + w[2] * d_nhat_dv
    return a, j_r, j_v


def _rtn_basis(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    r_hat = r / np.linalg.norm(r)
    h = np.cross(r, v)
    n_hat = h / np.linalg.norm(h)
    return np.column_stack([r_hat, np.cross(n_hat, r_hat), n_hat])


def dmc_process_noise(options: DMCOptions, dt_s: float, *, state6: ArrayLike | None = None) -> np.ndarray:
    """Exact discrete process noise of [r, v, w] for the Gauss-Markov model (van Loan, 9x9).

    With ``axes="rtn"`` (``state6`` required) w is in RTN coordinates; the r and v
    blocks are rotated to the inertial frame with C at ``state6`` (held over the step).

    The kinematic coupling r' = v, v' = w is exact; the gravity gradient is left
    out of Q, as is standard (its effect over one step is second order).

    WP0.2 (EST-09): the van Loan block carries +beta on the diagonal of -A and grows
    like exp(dt / tau), so phi @ e[:9, 9:] cancelled (inaccurate at dt/tau ~ 30, not
    PSD at ~144, overflow at ~700).  It is now evaluated on h = dt / 2^k <= tau and
    doubled exactly, Q(2h) = Phi(h) Q(h) Phi(h)^T + Q(h), Phi(2h) = Phi(h)^2.
    """
    if options.axes == "rtn" and state6 is None:
        raise ValueError("RTN DMC process noise needs the state (state6).")
    # WP4.8b: Q is linear in sigma^2, and van Loan's accuracy depends on the size of
    # the Qc block against A (5e-8 relative at sigma = 1e-4 m/s^2 over one day, 3e-5 at
    # 1 m/s^2), so Q is evaluated at a well-scaled reference sigma and scaled per axis.
    sig = (np.full(3, options.sigma_mps2) if options.sigma_axes_mps2 is None
           else np.asarray(options.sigma_axes_mps2, dtype=float))
    reference = _gauss_markov_q(DMCOptions(_SIGMA_REFERENCE_MPS2, options.tau_s), dt_s)
    scale = np.tile(sig / _SIGMA_REFERENCE_MPS2, 3)
    q = scale[:, None] * reference * scale[None, :]
    if options.axes == "rtn":
        x = np.asarray(state6, dtype=float).reshape(-1)
        rot = np.eye(9)
        rot[0:3, 0:3] = rot[3:6, 3:6] = _rtn_basis(x[:3], x[3:6])
        q = rot @ q @ rot.T
    return 0.5 * (q + q.T)


def _gauss_markov_q(options: DMCOptions, dt_s: float) -> np.ndarray:
    dt = abs(float(dt_s))
    beta = 1.0 / options.tau_s
    doublings = max(0, int(np.ceil(np.log2(dt * beta)))) if dt * beta > 1.0 else 0
    phi, q = _van_loan(options, beta, dt / 2.0**doublings)
    for _ in range(doublings):
        q = phi @ q @ phi.T + q
        phi = phi @ phi
    return 0.5 * (q + q.T)


def _van_loan(options: DMCOptions, beta: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
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
    return phi, 0.5 * (q + q.T)


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
    rtn = options.axes == "rtn"

    def rhs(t, y):
        r_e = np.asarray(get_earth_pos(float(t)), dtype=float).reshape(-1)[:3]
        r_s = np.asarray(get_sun_pos(float(t)), dtype=float).reshape(-1)[:3]
        x6 = y[:6]
        dx = f3body_moon(x6, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2, r_e, r_s, **force)
        out = np.empty_like(y)
        out[:3] = dx[:3]
        if rtn:
            accel, j_r, j_v = rtn_acceleration_and_jacobian(y[:3], y[3:6], y[6:9])
        else:
            accel = y[6:9]
        out[3:6] = dx[3:6] + accel
        out[6:9] = -beta * y[6:9]
        if stm:
            a9 = np.zeros((9, 9))
            a9[:6, :6] = dynamics_jacobian_a_matrix(x6, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
                                                    r_e, r_s, **force)
            if rtn:
                a9[3:6, 0:3] += j_r
                a9[3:6, 3:6] += j_v
                a9[3:6, 6:9] = _rtn_basis(y[:3], y[3:6])
            else:
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
