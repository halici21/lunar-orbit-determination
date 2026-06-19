"""Moon-centered dynamics and STM helpers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

try:
    from .accelerated import (
        f3body_rhs as _f3body_rhs_fast,
        ode42_rhs as _ode42_rhs_fast,
        rk4_6state as _rk4_6state_fast,
    )
    _FAST_DYNAMICS = True
except Exception:
    _FAST_DYNAMICS = False

# ---------------------------------------------------------------------------
# J2 support — Moon's mean-pole rotation frame
# ---------------------------------------------------------------------------
# IAU 2006 mean pole: RA₀ = 269.9949°, Dec₀ = 66.5392°.  Since J2 is axially
# symmetric a fixed (mean) pole rotation suffices; libration (~0.04°) produces
# < 0.1 % error in the J2 acceleration for low lunar orbits.
MOON_J2:  float = 2.0346e-4   # IAU/GRAIL J2 coefficient
MOON_R_M: float = 1_737_400.0  # Moon mean radius (m)


def _build_moon_j2_rotation() -> np.ndarray:
    """Rotation matrix from J2000 MCI → Moon mean-pole (body-fixed) frame."""
    ra = np.radians(269.9949)
    dc = np.radians(66.5392)
    z = np.array([np.cos(dc) * np.cos(ra), np.cos(dc) * np.sin(ra), np.sin(dc)])
    # x_bf: MCI x-axis projected perpendicular to z (z_x ≈ 0 for Moon)
    e1 = np.array([1.0, 0.0, 0.0])
    x = e1 - np.dot(e1, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.array([x, y, z])


_MCI_TO_MOON_BF: np.ndarray = _build_moon_j2_rotation()


def _vec3(value: ArrayLike, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size != 3:
        raise ValueError(f"{name} must have exactly 3 elements.")
    return vector


def _state6(value: ArrayLike) -> np.ndarray:
    state = np.asarray(value, dtype=float).reshape(-1)
    if state.size != 6:
        raise ValueError("State vector must have exactly 6 elements.")
    return state


def point_mass_acceleration(r_sc_m: ArrayLike, mu_m3_s2: float) -> np.ndarray:
    """Central point-mass acceleration with MATLAB's near-zero safeguard."""
    r_sc_m = _vec3(r_sc_m, "r_sc_m")
    distance_m = float(np.linalg.norm(r_sc_m))
    if distance_m < 1e-3:
        return np.zeros(3, dtype=float)
    return -mu_m3_s2 * r_sc_m / distance_m**3


def zonal_j2_acceleration(
    r_body_fixed_m: ArrayLike,
    mu_m3_s2: float,
    reference_radius_m: float,
    j2: float,
) -> np.ndarray:
    """J2 perturbing acceleration for a body-fixed z-axis gravity field."""
    r_body_fixed_m = _vec3(r_body_fixed_m, "r_body_fixed_m")
    distance_m = float(np.linalg.norm(r_body_fixed_m))
    if distance_m < 1e-3 or j2 == 0.0:
        return np.zeros(3, dtype=float)
    if reference_radius_m <= 0.0:
        raise ValueError("reference_radius_m must be positive.")

    x_m, y_m, z_m = r_body_fixed_m
    r2 = distance_m**2
    z2_over_r2 = (z_m**2) / r2
    scale = -1.5 * float(j2) * float(mu_m3_s2) * float(reference_radius_m) ** 2 / distance_m**5
    return scale * np.array(
        [
            x_m * (1.0 - 5.0 * z2_over_r2),
            y_m * (1.0 - 5.0 * z2_over_r2),
            z_m * (3.0 - 5.0 * z2_over_r2),
        ],
        dtype=float,
    )


def zonal_j2_gravity_gradient(
    r_body_fixed_m: ArrayLike,
    mu_m3_s2: float,
    reference_radius_m: float,
    j2: float,
    *,
    step_m: float = 1.0,  # retained for backward compatibility; unused in analytic form
) -> np.ndarray:
    """Analytic J₂ gravity-gradient tensor (body-fixed frame, z = pole axis).

    G[i,j] = ∂a_J2_i/∂r_j  where a_J2 = zonal_j2_acceleration(r, mu, R, j2).
    Returns a 3×3 symmetric matrix.
    """
    r = _vec3(r_body_fixed_m, "r_body_fixed_m")
    x, y, z = r[0], r[1], r[2]
    r2 = float(np.dot(r, r))
    if r2 < 1e-6 or j2 == 0.0:
        return np.zeros((3, 3), dtype=float)

    r4 = r2 * r2
    A = -1.5 * float(j2) * float(mu_m3_s2) * float(reference_radius_m) ** 2 / (r2 ** 2.5)

    z2r2 = z * z / r2   # z²/r²
    z2r4 = z * z / r4   # z²/r⁴

    G = np.empty((3, 3), dtype=float)
    base = 1.0 - 5.0 * z2r2
    G[0, 0] = A * (base - 5.0 * x * x / r2 + 35.0 * x * x * z2r4)
    G[1, 1] = A * (base - 5.0 * y * y / r2 + 35.0 * y * y * z2r4)
    G[2, 2] = A * (3.0 - 30.0 * z2r2 + 35.0 * z * z * z2r4)
    G[0, 1] = A * x * y * (-5.0 / r2 + 35.0 * z2r4)
    G[0, 2] = A * x * z * (-15.0 / r2 + 35.0 * z2r4)
    G[1, 2] = A * y * z * (-15.0 / r2 + 35.0 * z2r4)
    G[1, 0] = G[0, 1]
    G[2, 0] = G[0, 2]
    G[2, 1] = G[1, 2]
    return G


def third_body_acceleration(
    r_sc_m: ArrayLike,
    r_third_body_m: ArrayLike,
    mu_third_body_m3_s2: float,
) -> np.ndarray:
    """Indirect-term third-body perturbing acceleration."""
    r_sc_m = _vec3(r_sc_m, "r_sc_m")
    r_third_body_m = _vec3(r_third_body_m, "r_third_body_m")

    d_sc_body_m = r_third_body_m - r_sc_m
    dist_sc_body_m = float(np.linalg.norm(d_sc_body_m))
    dist_origin_body_m = float(np.linalg.norm(r_third_body_m))

    return mu_third_body_m3_s2 * (
        d_sc_body_m / dist_sc_body_m**3 - r_third_body_m / dist_origin_body_m**3
    )


def f3body_moon(
    state_mci: ArrayLike,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    r_moon_earth_m: ArrayLike,
    r_moon_sun_m: ArrayLike,
    *,
    j2_moon: float = 0.0,
) -> np.ndarray:
    """6-state derivative matching MATLAB `f3body_moon.m`.

    Pass ``j2_moon=MOON_J2`` to include the lunar J2 oblateness perturbation
    (transforms to/from the Moon mean-pole body-fixed frame via ``_MCI_TO_MOON_BF``).
    """
    state_mci = _state6(state_mci)
    r_sc_m = state_mci[:3]
    v_sc_mps = state_mci[3:]

    a_total_mps2 = (
        point_mass_acceleration(r_sc_m, mu_moon_m3_s2)
        + third_body_acceleration(r_sc_m, r_moon_earth_m, mu_earth_m3_s2)
        + third_body_acceleration(r_sc_m, r_moon_sun_m, mu_sun_m3_s2)
    )
    if j2_moon:
        r_bf = _MCI_TO_MOON_BF @ r_sc_m
        a_j2_bf = zonal_j2_acceleration(r_bf, mu_moon_m3_s2, MOON_R_M, j2_moon)
        a_total_mps2 = a_total_mps2 + _MCI_TO_MOON_BF.T @ a_j2_bf
    return np.concatenate([v_sc_mps, a_total_mps2])


def point_mass_gravity_gradient(r_sc_m: ArrayLike, mu_m3_s2: float) -> np.ndarray:
    """Derivative of point-mass acceleration with respect to position."""
    r_sc_m = _vec3(r_sc_m, "r_sc_m")
    distance_m = float(np.linalg.norm(r_sc_m))
    if distance_m < 1e-3:
        return np.zeros((3, 3), dtype=float)

    identity = np.eye(3)
    rr_t = np.outer(r_sc_m, r_sc_m)
    return -mu_m3_s2 / distance_m**3 * identity + 3.0 * mu_m3_s2 / distance_m**5 * rr_t


def third_body_gravity_gradient(
    r_sc_m: ArrayLike,
    r_third_body_m: ArrayLike,
    mu_third_body_m3_s2: float,
) -> np.ndarray:
    """Derivative of indirect third-body acceleration with respect to spacecraft position."""
    r_sc_m = _vec3(r_sc_m, "r_sc_m")
    r_third_body_m = _vec3(r_third_body_m, "r_third_body_m")

    d_sc_body_m = r_third_body_m - r_sc_m
    distance_m = float(np.linalg.norm(d_sc_body_m))
    if distance_m < 1e-3:
        return np.zeros((3, 3), dtype=float)

    identity = np.eye(3)
    dd_t = np.outer(d_sc_body_m, d_sc_body_m)
    return (
        -mu_third_body_m3_s2 / distance_m**3 * identity
        + 3.0 * mu_third_body_m3_s2 / distance_m**5 * dd_t
    )


def dynamics_jacobian_a_matrix(
    state_mci: ArrayLike,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    r_moon_earth_m: ArrayLike,
    r_moon_sun_m: ArrayLike,
    *,
    j2_moon: float = 0.0,
) -> np.ndarray:
    """Build the 6x6 variational A matrix used by `odeFun_v3.m`."""
    state_mci = _state6(state_mci)
    r_sc_m = state_mci[:3]

    g_total = (
        point_mass_gravity_gradient(r_sc_m, mu_moon_m3_s2)
        + third_body_gravity_gradient(r_sc_m, r_moon_earth_m, mu_earth_m3_s2)
        + third_body_gravity_gradient(r_sc_m, r_moon_sun_m, mu_sun_m3_s2)
    )
    if j2_moon:
        r_bf = _MCI_TO_MOON_BF @ r_sc_m
        g_j2_bf = zonal_j2_gravity_gradient(r_bf, mu_moon_m3_s2, MOON_R_M, j2_moon)
        g_total = g_total + _MCI_TO_MOON_BF.T @ g_j2_bf @ _MCI_TO_MOON_BF

    return np.block(
        [
            [np.zeros((3, 3)), np.eye(3)],
            [g_total, np.zeros((3, 3))],
        ]
    )


def ode_fun_v3(
    t_s: float,
    state_aug_mci: ArrayLike,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    j2_moon: float = 0.0,
) -> np.ndarray:
    """42-state derivative matching MATLAB `odeFun_v3.m`.

    The STM is stored and flattened in column-major order, matching MATLAB's
    `Phi(:)` convention.
    """
    state_aug_mci = np.asarray(state_aug_mci, dtype=float).reshape(-1)
    if state_aug_mci.size != 42:
        raise ValueError("Augmented v3 state must have 42 elements = [x(6); Phi(36)].")

    x_mci = state_aug_mci[:6]
    phi = state_aug_mci[6:].reshape((6, 6), order="F")
    r_moon_earth_m = _vec3(get_earth_pos(float(t_s)), "r_moon_earth_m")
    r_moon_sun_m = _vec3(get_sun_pos(float(t_s)), "r_moon_sun_m")

    x_dot = f3body_moon(
        x_mci,
        mu_moon_m3_s2,
        mu_earth_m3_s2,
        mu_sun_m3_s2,
        r_moon_earth_m,
        r_moon_sun_m,
        j2_moon=j2_moon,
    )
    a_matrix = dynamics_jacobian_a_matrix(
        x_mci,
        mu_moon_m3_s2,
        mu_earth_m3_s2,
        mu_sun_m3_s2,
        r_moon_earth_m,
        r_moon_sun_m,
        j2_moon=j2_moon,
    )
    phi_dot = a_matrix @ phi
    return np.concatenate([x_dot, phi_dot.reshape(-1, order="F")])


def _propagate_vode(
    t_eval_s: np.ndarray,
    y0: np.ndarray,
    rhs_fn: Callable,
    rtol: float,
    atol: float,
) -> np.ndarray:
    """VODE Adams-12 multi-step integrator — faster than DOP853 for smooth orbits.

    Uses fewer RHS evaluations (~3-4×) by reusing solution history across steps.
    Accuracy is comparable to DOP853 at the same tolerances.
    """
    from scipy.integrate import ode as _scipy_ode

    solver = _scipy_ode(rhs_fn).set_integrator(
        "vode", method="adams", rtol=rtol, atol=atol, nsteps=50000, order=12
    )
    solver.set_initial_value(y0, t_eval_s[0])
    result = np.empty((len(t_eval_s), len(y0)))
    result[0] = y0.copy()
    for i in range(1, len(t_eval_s)):
        result[i] = solver.integrate(t_eval_s[i])
        if not solver.successful():
            raise RuntimeError(f"VODE Adams integration failed at t={t_eval_s[i]:.1f} s")
    return result


def propagate_state(
    t_eval_s: ArrayLike,
    state0_mci: ArrayLike,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    *,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    method: str = "ADAMS",
    j2_moon: float = 0.0,
) -> np.ndarray:
    """Propagate the 6-state dynamics at requested epochs.

    Returns an array with shape `(len(t_eval_s), 6)`.

    ``method`` may be ``"ADAMS"`` (VODE Adams-12, default, faster for smooth
    orbits), ``"DOP853"``, or any other ``solve_ivp``-compatible method name.
    Pass ``j2_moon=MOON_J2`` to include the lunar oblateness perturbation (forces
    the Python code path; the Numba kernel does not support J2).
    """
    from scipy.integrate import solve_ivp

    t_eval_s = np.asarray(t_eval_s, dtype=float).reshape(-1)
    if t_eval_s.size == 0:
        raise ValueError("t_eval_s must contain at least one epoch.")

    state0_mci = _state6(state0_mci)

    if _FAST_DYNAMICS:
        _j2 = float(j2_moon)
        _mr = MOON_R_M if j2_moon else 0.0
        _cbf = _MCI_TO_MOON_BF
        def rhs(t_s: float, state: np.ndarray) -> np.ndarray:
            return _f3body_rhs_fast(
                state, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
                get_earth_pos(float(t_s)), get_sun_pos(float(t_s)),
                _j2, _mr, _cbf,
            )
    else:
        def rhs(t_s: float, state: np.ndarray) -> np.ndarray:
            return f3body_moon(
                state, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
                get_earth_pos(float(t_s)), get_sun_pos(float(t_s)),
                j2_moon=j2_moon,
            )

    if method.upper() == "ADAMS":
        return _propagate_vode(t_eval_s, state0_mci, rhs, rtol, atol)

    solution = solve_ivp(
        rhs,
        (float(t_eval_s[0]), float(t_eval_s[-1])),
        state0_mci,
        method=method,
        t_eval=t_eval_s,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"State propagation failed: {solution.message}")

    return solution.y.T


def propagate_augmented_state(
    t_eval_s: ArrayLike,
    state_aug0_mci: ArrayLike,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    get_earth_pos: Callable[[float], ArrayLike],
    get_sun_pos: Callable[[float], ArrayLike],
    *,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    method: str = "ADAMS",
    j2_moon: float = 0.0,
) -> np.ndarray:
    """Propagate the 42-state dynamics at requested epochs.

    Returns an array with shape `(len(t_eval_s), 42)`.

    Defaults to ``"ADAMS"`` (VODE Adams-12) which is ~2.3× faster than DOP853
    for smooth lunar orbits.  Pass ``method="DOP853"`` for the classical path.
    Pass ``j2_moon=MOON_J2`` to include the lunar oblateness perturbation (forces
    the Python code path).
    """
    from scipy.integrate import solve_ivp

    t_eval_s = np.asarray(t_eval_s, dtype=float).reshape(-1)
    if t_eval_s.size == 0:
        raise ValueError("t_eval_s must contain at least one epoch.")

    state_aug0_mci = np.asarray(state_aug0_mci, dtype=float).reshape(-1)
    if state_aug0_mci.size != 42:
        raise ValueError("Initial augmented state must have 42 elements.")

    if _FAST_DYNAMICS:
        _j2 = float(j2_moon)
        _mr = MOON_R_M if j2_moon else 0.0
        _cbf = _MCI_TO_MOON_BF
        def rhs(t_s: float, state_aug: np.ndarray) -> np.ndarray:
            return _ode42_rhs_fast(
                state_aug, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
                get_earth_pos(float(t_s)), get_sun_pos(float(t_s)),
                _j2, _mr, _cbf,
            )
    else:
        def rhs(t_s: float, state_aug: np.ndarray) -> np.ndarray:
            return ode_fun_v3(
                t_s, state_aug, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
                get_earth_pos, get_sun_pos, j2_moon,
            )

    if method.upper() == "ADAMS":
        return _propagate_vode(t_eval_s, state_aug0_mci, rhs, rtol, atol)

    solution = solve_ivp(
        rhs,
        (float(t_eval_s[0]), float(t_eval_s[-1])),
        state_aug0_mci,
        method=method,
        t_eval=t_eval_s,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"Augmented propagation failed: {solution.message}")

    return solution.y.T


def make_fast_sigma_propagator(
    ephemeris,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    *,
    rk4_dt_s: float = 10.0,
):
    """Return a fast (t0, t1, state6) → state6 callable using numba RK4.

    Uses pre-sampled ephemeris grids + binary-search linear interpolation.
    ~250× faster than scipy DOP853 for short intervals (UKF sigma steps).
    Position accuracy ~3 mm over 60 s — sufficient when measurement noise
    is ≥ 10 m.

    Returns ``None`` when numba is unavailable.
    """
    if not _FAST_DYNAMICS:
        return None
    t_grid     = np.asarray(ephemeris.t_ephem_s, dtype=np.float64)
    earth_grid = np.asarray(ephemeris.earth_pos_m, dtype=np.float64)
    sun_grid   = np.asarray(ephemeris.sun_pos_m, dtype=np.float64)
    mu_m = float(mu_moon_m3_s2)
    mu_e = float(mu_earth_m3_s2)
    mu_s = float(mu_sun_m3_s2)
    dt   = float(rk4_dt_s)

    def _propagate(t0: float, t1: float, state6: np.ndarray) -> np.ndarray:
        if np.isclose(t0, t1):
            return state6.copy()
        return _rk4_6state_fast(
            np.asarray(state6, dtype=np.float64),
            float(t0), float(t1), mu_m, mu_e, mu_s,
            t_grid, earth_grid, sun_grid, dt=dt,
        )

    return _propagate


def propagate_truth_with_ephemeris(
    t_eval_s: ArrayLike,
    state0_mci: ArrayLike,
    mu_moon_m3_s2: float,
    mu_earth_m3_s2: float,
    mu_sun_m3_s2: float,
    ephemeris,
    *,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    method: str = "DOP853",
    j2_moon: float = 0.0,
) -> np.ndarray:
    """Propagate truth dynamics using Moon-centered ephemeris interpolants."""
    return propagate_state(
        t_eval_s,
        state0_mci,
        mu_moon_m3_s2,
        mu_earth_m3_s2,
        mu_sun_m3_s2,
        ephemeris.earth_position,
        ephemeris.sun_position,
        rtol=rtol,
        atol=atol,
        method=method,
        j2_moon=j2_moon,
    )
