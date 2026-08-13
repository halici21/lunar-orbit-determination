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

# Lunar spherical harmonics (Phase 13B splice) — default-off, additive.
try:
    from .accelerated import f3body_harmonics_rhs as _f3body_harmonics_rhs_fast
    _FAST_HARMONICS = True
except Exception:
    _FAST_HARMONICS = False
from .gravity_harmonics import (
    SphericalHarmonicGravityModel,
    spherical_harmonic_acceleration,
)
from .lunar_frames import nearest_rotation_at_time

# ---------------------------------------------------------------------------
# J2 support — Moon's mean-pole rotation frame
# ---------------------------------------------------------------------------
# IAU 2006 mean pole: RA₀ = 269.9949°, Dec₀ = 66.5392°.  Since J2 is axially
# symmetric a fixed (mean) pole rotation suffices; libration (~0.04°) produces
# < 0.1 % error in the J2 acceleration for low lunar orbits.
# MOON_J2 / MOON_R_M are sourced from the centralized constants module and
# re-exported here under their legacy names for backward compatibility.
from .constants import (
    J2_MOON_UNNORMALIZED as MOON_J2,   # IAU/GRAIL J2 coefficient
    R_MOON_M as MOON_R_M,              # Moon mean radius (m)
    R_EARTH_J2_REF_M,                  # EGM96 Earth J2 reference radius (m)
)


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

# ---------------------------------------------------------------------------
# Earth J2 frame (Phase 5).  First-phase constant approximation:
#   - constant J2000 mean-equator pole (identity rotation): the J2000 z-axis is
#     Earth's mean pole at J2000.0;
#   - precession / nutation / diurnal Earth orientation are neglected;
#   - this is NOT a high-fidelity Earth-orientation model.  It is sufficient to
#     validate the magnitude of the Earth-J2 effect and the architecture; a
#     SPICE epoch-dependent Earth orientation can be added as a separate phase.
# ---------------------------------------------------------------------------
_J2000_TO_EARTH_BF: np.ndarray = np.eye(3)
_ALLOWED_EARTH_J2_MODES = ("indirect", "direct")


def _earth_mode_int(j2_earth: float, earth_j2_mode: str) -> int:
    """Map (j2_earth, mode) to the Numba ``earth_mode`` int: 0=off/1=indirect/2=direct."""
    if not j2_earth:
        return 0
    if earth_j2_mode == "indirect":
        return 1
    if earth_j2_mode == "direct":
        return 2
    raise ValueError("earth_j2_mode must be 'indirect' or 'direct'.")


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
    j2_earth: float = 0.0,
    earth_j2_mode: str = "indirect",
    harmonic_model: SphericalHarmonicGravityModel | None = None,
    c_inertial_to_bf_harmonic: ArrayLike | None = None,
) -> np.ndarray:
    """6-state derivative matching MATLAB `f3body_moon.m`.

    Pass ``j2_moon=MOON_J2`` to include the lunar J2 oblateness perturbation
    (transforms to/from the Moon mean-pole body-fixed frame via ``_MCI_TO_MOON_BF``).

    Pass ``j2_earth=J2_EARTH_UNNORMALIZED`` to include Earth's J2 perturbation in
    the Moon-centered frame.  ``earth_j2_mode='indirect'`` (default) uses the
    physically correct relative form
    ``a_J2(sc rel Earth) - a_J2(Moon rel Earth)``; ``'direct'`` keeps only the
    spacecraft term and is for debug / sensitivity studies only.

    Pass ``harmonic_model`` (with ``c_inertial_to_bf_harmonic``, the epoch's
    inertial -> body-fixed rotation resolved by the CALLER, exactly like the
    third-body positions) to add the lunar spherical-harmonic perturbation
    (n >= 2, acceleration-only).  This function is time-agnostic and applies no
    guards: composition rules (J2 double-count ban, the m > 0 epoch-rotation
    requirement, the STM refusal) are enforced at the propagation entry points.
    A model containing C20 must NOT be combined with ``j2_moon != 0``.
    """
    state_mci = _state6(state_mci)
    r_sc_m = state_mci[:3]
    v_sc_mps = state_mci[3:]

    a_total_mps2 = (
        point_mass_acceleration(r_sc_m, mu_moon_m3_s2)
        + third_body_acceleration(r_sc_m, r_moon_earth_m, mu_earth_m3_s2)
        + third_body_acceleration(r_sc_m, r_moon_sun_m, mu_sun_m3_s2)
    )
    if j2_earth:
        r_moon_earth_m = _vec3(r_moon_earth_m, "r_moon_earth_m")
        r_sc_earth_m = r_sc_m - r_moon_earth_m            # Earth -> spacecraft
        a_total_mps2 = a_total_mps2 + body_j2_acceleration(
            r_sc_earth_m, mu_earth_m3_s2, R_EARTH_J2_REF_M, j2_earth, _J2000_TO_EARTH_BF
        )
        if earth_j2_mode == "indirect":
            a_total_mps2 = a_total_mps2 - body_j2_acceleration(
                -r_moon_earth_m, mu_earth_m3_s2, R_EARTH_J2_REF_M, j2_earth, _J2000_TO_EARTH_BF
            )
        elif earth_j2_mode != "direct":
            raise ValueError("earth_j2_mode must be 'indirect' or 'direct'.")
    if j2_moon:
        a_total_mps2 = a_total_mps2 + body_j2_acceleration(
            r_sc_m, mu_moon_m3_s2, MOON_R_M, j2_moon, _MCI_TO_MOON_BF
        )
    if harmonic_model is not None:
        a_total_mps2 = a_total_mps2 + spherical_harmonic_acceleration(
            r_sc_m, harmonic_model, c_inertial_to_bf_harmonic
        )
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
    j2_earth: float = 0.0,
    earth_j2_mode: str = "indirect",
) -> np.ndarray:
    """Build the 6x6 variational A matrix used by `odeFun_v3.m`.

    The Earth-J2 indirect term ``-a_J2(Moon rel Earth)`` is state-independent, so
    only the spacecraft term contributes to the gravity gradient (the same
    gradient applies for ``'indirect'`` and ``'direct'`` modes).
    """
    state_mci = _state6(state_mci)
    r_sc_m = state_mci[:3]

    g_total = (
        point_mass_gravity_gradient(r_sc_m, mu_moon_m3_s2)
        + third_body_gravity_gradient(r_sc_m, r_moon_earth_m, mu_earth_m3_s2)
        + third_body_gravity_gradient(r_sc_m, r_moon_sun_m, mu_sun_m3_s2)
    )
    if j2_earth:
        r_sc_earth_m = r_sc_m - _vec3(r_moon_earth_m, "r_moon_earth_m")
        g_total = g_total + body_j2_gravity_gradient(
            r_sc_earth_m, mu_earth_m3_s2, R_EARTH_J2_REF_M, j2_earth, _J2000_TO_EARTH_BF
        )
    if j2_moon:
        g_total = g_total + body_j2_gravity_gradient(
            r_sc_m, mu_moon_m3_s2, MOON_R_M, j2_moon, _MCI_TO_MOON_BF
        )

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
    j2_earth: float = 0.0,
    earth_j2_mode: str = "indirect",
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
        j2_earth=j2_earth,
        earth_j2_mode=earth_j2_mode,
    )
    a_matrix = dynamics_jacobian_a_matrix(
        x_mci,
        mu_moon_m3_s2,
        mu_earth_m3_s2,
        mu_sun_m3_s2,
        r_moon_earth_m,
        r_moon_sun_m,
        j2_moon=j2_moon,
        j2_earth=j2_earth,
        earth_j2_mode=earth_j2_mode,
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


def _harmonic_tesseral_active(model: SphericalHarmonicGravityModel) -> bool:
    """True when the model evaluates any nonzero m > 0 coefficient.

    Respects the model's own nmax/mmax truncation: coefficients outside it are
    never evaluated, so they do not trigger the epoch-rotation requirement.
    """
    if model.mmax < 1:
        return False
    cb = model.cbar[: model.nmax + 1, 1 : model.mmax + 1]
    sb = model.sbar[: model.nmax + 1, 1 : model.mmax + 1]
    return bool(np.any(cb != 0.0) or np.any(sb != 0.0))


def _prepare_harmonic_context(
    harmonic_model: SphericalHarmonicGravityModel,
    harmonic_rotation,
    j2_moon: float,
    t_eval_s: np.ndarray,
):
    """Validate the harmonics composition ONCE at propagation setup.

    Enforced rules (hard ``ValueError``, no silent fixes):
    - J2 double-count ban: a model whose (2, 0) coefficient is nonzero already
      contains J2, so ``j2_moon`` must be 0 (neither is silently altered).
      Earth J2 is a different body and stays freely composable.
    - m > 0 frame rule: models with nonzero tesseral/sectoral coefficients need
      an epoch-dependent rotation grid ``(t_grid_s, rotation_grid)`` (sampled
      from MOON_PA); a constant matrix is physical only for zonal-only models.
    - Rotation-grid coverage: the grid must span the whole propagation window
      (recommendation: sample ``[t0 - margin, T + margin]`` with
      ``margin = max(2 * cadence, 120 s)``); out-of-range lookups stay hard
      errors, never silent clamps.

    Returns ``(rot_const, rot_t, rot_grid, cbar, sbar, mu, r_ref, nmax, mmax)``
    with either ``rot_const`` or the grid pair set.  Raw arrays/scalars are
    extracted here exactly once (the dataclass never crosses into Numba).
    """
    if float(harmonic_model.cbar[2, 0]) != 0.0 and j2_moon:
        raise ValueError(
            "lunar harmonics model includes a nonzero C20 (J2) term; combining it "
            "with j2_moon != 0 would count J2 twice. Set j2_moon=0 or use a model "
            "without C20 (neither is altered silently)."
        )
    tesseral = _harmonic_tesseral_active(harmonic_model)
    if harmonic_rotation is None:
        raise ValueError(
            "harmonic_rotation is required with harmonic_model: pass a constant "
            "(3, 3) inertial->body matrix for a zonal-only model, or an "
            "epoch-dependent (t_grid_s, rotation_grid) pair sampled from MOON_PA "
            "(mandatory for m > 0 coefficients)."
        )
    if isinstance(harmonic_rotation, (tuple, list)) and len(harmonic_rotation) == 2:
        rot_t = np.asarray(harmonic_rotation[0], dtype=float)
        rot_grid = np.asarray(harmonic_rotation[1], dtype=float)
        if rot_t.ndim != 1 or rot_t.size == 0:
            raise ValueError("harmonic rotation t_grid_s must be a non-empty 1-D array.")
        if rot_grid.shape != (rot_t.size, 3, 3):
            raise ValueError(
                "harmonic rotation grid must have shape (N, 3, 3) matching t_grid_s."
            )
        if rot_t.size > 1 and not np.all(np.diff(rot_t) > 0.0):
            raise ValueError("harmonic rotation t_grid_s must be strictly increasing.")
        t_lo, t_hi = float(t_eval_s[0]), float(t_eval_s[-1])
        if rot_t[0] > t_lo or rot_t[-1] < t_hi:
            raise ValueError(
                f"harmonic rotation grid [{rot_t[0]:.1f}, {rot_t[-1]:.1f}] s does not "
                f"cover the propagation window [{t_lo:.1f}, {t_hi:.1f}] s; sample it "
                "with a margin, e.g. [t0 - m, T + m] with m = max(2*cadence, 120 s)."
            )
        rot_const = None
    else:
        rot_const = np.asarray(harmonic_rotation, dtype=float)
        if rot_const.shape != (3, 3):
            raise ValueError(
                "harmonic_rotation must be a (3, 3) matrix or a "
                "(t_grid_s, rotation_grid) pair."
            )
        if tesseral:
            raise ValueError(
                "harmonic model has nonzero m > 0 (tesseral/sectoral) coefficients; "
                "a constant rotation freezes the Moon's longitude and is not "
                "physical. Provide an epoch-dependent (t_grid_s, rotation_grid) "
                "sampled from MOON_PA (lunar_frames.sample_moon_pa_rotations)."
            )
        rot_t = None
        rot_grid = None
    return (
        rot_const, rot_t, rot_grid,
        harmonic_model.cbar, harmonic_model.sbar,
        float(harmonic_model.mu_m3_s2), float(harmonic_model.r_ref_m),
        int(harmonic_model.nmax), int(harmonic_model.mmax),
    )


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
    j2_earth: float = 0.0,
    earth_j2_mode: str = "indirect",
    harmonic_model: SphericalHarmonicGravityModel | None = None,
    harmonic_rotation=None,
) -> np.ndarray:
    """Propagate the 6-state dynamics at requested epochs.

    Returns an array with shape `(len(t_eval_s), 6)`.

    ``method`` may be ``"ADAMS"`` (VODE Adams-12, default, faster for smooth
    orbits), ``"DOP853"``, or any other ``solve_ivp``-compatible method name.
    Pass ``j2_moon=MOON_J2`` to include the lunar oblateness perturbation, and
    ``j2_earth=J2_EARTH_UNNORMALIZED`` (``earth_j2_mode='indirect'`` by default)
    to include Earth's J2 in the Moon-centered frame.

    Lunar spherical harmonics (Phase 13B, acceleration-only, default-off): pass
    ``harmonic_model`` plus ``harmonic_rotation`` — either a constant (3, 3)
    inertial->body matrix (zonal-only models) or an epoch-dependent
    ``(t_grid_s, rotation_grid)`` pair from
    ``lunar_frames.sample_moon_pa_rotations`` (mandatory for m > 0 terms).  The
    grid must cover the propagation window with a margin (suggested
    ``max(2 * cadence, 120 s)``).  A model containing C20 must be used with
    ``j2_moon=0`` (enforced here with ``ValueError`` — J2 is never counted
    twice and nothing is silently altered); Earth J2 remains composable.
    """
    from scipy.integrate import solve_ivp

    t_eval_s = np.asarray(t_eval_s, dtype=float).reshape(-1)
    if t_eval_s.size == 0:
        raise ValueError("t_eval_s must contain at least one epoch.")

    state0_mci = _state6(state0_mci)
    _emode = _earth_mode_int(j2_earth, earth_j2_mode)
    _h_ctx = None
    if harmonic_model is not None:
        _h_ctx = _prepare_harmonic_context(
            harmonic_model, harmonic_rotation, j2_moon, t_eval_s
        )

    if _FAST_DYNAMICS:
        _j2 = float(j2_moon)
        _mr = MOON_R_M if j2_moon else 0.0
        _cbf = _MCI_TO_MOON_BF
        _j2e = float(j2_earth)
        _er = R_EARTH_J2_REF_M if j2_earth else 0.0
        _cbfe = _J2000_TO_EARTH_BF
        def rhs(t_s: float, state: np.ndarray) -> np.ndarray:
            return _f3body_rhs_fast(
                state, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
                get_earth_pos(float(t_s)), get_sun_pos(float(t_s)),
                _j2, _mr, _cbf, _j2e, _er, _emode, _cbfe,
            )
    else:
        def rhs(t_s: float, state: np.ndarray) -> np.ndarray:
            return f3body_moon(
                state, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
                get_earth_pos(float(t_s)), get_sun_pos(float(t_s)),
                j2_moon=j2_moon, j2_earth=j2_earth, earth_j2_mode=earth_j2_mode,
            )

    if _h_ctx is not None:
        # Harmonics-on RHS overrides the closures above; the harmonics-off
        # production paths above stay byte-identical and are never entered
        # with a model.  The rotation lookup is the SAME Python function for
        # both paths (exact parity); only the force evaluation dispatches.
        (_rc, _rt, _rg, _hcb, _hsb, _hmu, _hrr, _hn, _hm) = _h_ctx

        def _rotation_at(t_s: float) -> np.ndarray:
            if _rc is not None:
                return _rc
            return nearest_rotation_at_time(_rg, _rt, float(t_s))

        if _FAST_DYNAMICS and _FAST_HARMONICS:
            _j2h = float(j2_moon)
            _mrh = MOON_R_M if j2_moon else 0.0
            _j2eh = float(j2_earth)
            _erh = R_EARTH_J2_REF_M if j2_earth else 0.0

            def rhs(t_s: float, state: np.ndarray) -> np.ndarray:
                return _f3body_harmonics_rhs_fast(
                    state, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
                    get_earth_pos(float(t_s)), get_sun_pos(float(t_s)),
                    _j2h, _mrh, _MCI_TO_MOON_BF, _j2eh, _erh, _emode,
                    _J2000_TO_EARTH_BF,
                    _hcb, _hsb, _hmu, _hrr, _hn, _hm, _rotation_at(t_s),
                )
        else:
            def rhs(t_s: float, state: np.ndarray) -> np.ndarray:
                return f3body_moon(
                    state, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
                    get_earth_pos(float(t_s)), get_sun_pos(float(t_s)),
                    j2_moon=j2_moon, j2_earth=j2_earth, earth_j2_mode=earth_j2_mode,
                    harmonic_model=harmonic_model,
                    c_inertial_to_bf_harmonic=_rotation_at(t_s),
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


def _propagate_augmented_with_harmonics(
    t_eval_s, state_aug0_mci, mu_moon, mu_earth, mu_sun,
    get_earth_pos, get_sun_pos, harmonic_model, h_ctx, *,
    rtol, atol, method, j2_moon, j2_earth, earth_j2_mode,
):
    """42-state RHS with a MATCHED harmonic acceleration and gradient.

    The same ``harmonic_model`` supplies both, so trajectory and variational
    fidelity cannot diverge, and the same rotation is applied to both.
    """
    from scipy.integrate import solve_ivp
    from .gravity_harmonics import spherical_harmonic_gravity_gradient

    rot_const, rot_t, rot_grid = h_ctx[0], h_ctx[1], h_ctx[2]

    def _rotation_at(t_s: float) -> np.ndarray:
        if rot_const is not None:
            return rot_const
        return nearest_rotation_at_time(rot_grid, rot_t, float(t_s))

    state_aug0_mci = np.asarray(state_aug0_mci, dtype=float).reshape(-1)
    if state_aug0_mci.size != 42:
        raise ValueError("Initial augmented state must have 42 elements.")

    def rhs(t_s: float, state_aug: np.ndarray) -> np.ndarray:
        t_s = float(t_s)
        x_mci = state_aug[:6]
        phi = state_aug[6:].reshape((6, 6), order="F")
        r_earth = _vec3(get_earth_pos(t_s), "r_moon_earth_m")
        r_sun = _vec3(get_sun_pos(t_s), "r_moon_sun_m")
        c_bf = _rotation_at(t_s)
        x_dot = f3body_moon(
            x_mci, mu_moon, mu_earth, mu_sun, r_earth, r_sun,
            j2_moon=j2_moon, j2_earth=j2_earth, earth_j2_mode=earth_j2_mode,
            harmonic_model=harmonic_model, c_inertial_to_bf_harmonic=c_bf,
        )
        a_matrix = dynamics_jacobian_a_matrix(
            x_mci, mu_moon, mu_earth, mu_sun, r_earth, r_sun,
            j2_moon=j2_moon, j2_earth=j2_earth, earth_j2_mode=earth_j2_mode,
        )
        a_matrix[3:6, 0:3] = a_matrix[3:6, 0:3] + spherical_harmonic_gravity_gradient(
            x_mci[:3], harmonic_model, c_bf
        )
        phi_dot = a_matrix @ phi
        return np.concatenate([x_dot, phi_dot.reshape(-1, order="F")])

    if method.upper() == "ADAMS":
        return _propagate_vode(t_eval_s, state_aug0_mci, rhs, rtol, atol)
    solution = solve_ivp(
        rhs, (float(t_eval_s[0]), float(t_eval_s[-1])), state_aug0_mci,
        method=method, t_eval=t_eval_s, rtol=rtol, atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"Augmented harmonic propagation failed: {solution.message}")
    return solution.y.T


_GENERIC_PA_FRAMES = {"MOON_PA", "MOON_ME"}


def require_explicit_pa_realization(model: SphericalHarmonicGravityModel) -> str:
    """Return the model's explicit PA realization frame, or fail closed.

    A generic ``MOON_PA`` label is NOT acceptable for OD use.  SPICE resolves
    the generic alias according to whichever lunar frame kernel was furnished
    last (``moon_080317.tf`` -> MOON_PA_DE421, ``moon_de440_220930.tf`` ->
    MOON_PA_DE440), so a model carrying only the generic label can silently be
    evaluated in a realization that is not the one its coefficients were solved
    in.  The realization must therefore be stated explicitly, e.g.
    ``MOON_PA_DE440``.
    """
    frame = str(getattr(model, "frame", "") or "").strip().upper()
    if frame in _GENERIC_PA_FRAMES or not frame:
        raise ValueError(
            f"lunar harmonics model carries the generic body-fixed frame label "
            f"{frame or '(empty)'!r}, which does not identify a principal-axes "
            "realization. SPICE resolves the generic MOON_PA alias by kernel "
            "load order, so this could silently evaluate the field in the wrong "
            "realization. Set an explicit realization (e.g. 'MOON_PA_DE440') "
            "matching the coefficient product."
        )
    return frame


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
    j2_earth: float = 0.0,
    earth_j2_mode: str = "indirect",
    harmonic_model: SphericalHarmonicGravityModel | None = None,
    harmonic_rotation=None,
    harmonic_stm_opt_in: bool = False,
) -> np.ndarray:
    """Propagate the 42-state dynamics at requested epochs.

    Returns an array with shape `(len(t_eval_s), 42)`.

    Defaults to ``"ADAMS"`` (VODE Adams-12) which is ~2.3× faster than DOP853
    for smooth lunar orbits.  Pass ``method="DOP853"`` for the classical path.
    Pass ``j2_moon=MOON_J2`` for the lunar oblateness perturbation, and
    ``j2_earth=J2_EARTH_UNNORMALIZED`` (``earth_j2_mode='indirect'``) for Earth's
    J2 in the Moon-centered frame.

    Lunar spherical harmonics in the STM are **explicit opt-in only** and are
    still fail-closed by default.  Passing ``harmonic_model`` without
    ``harmonic_stm_opt_in=True`` raises, exactly as before.  With the opt-in the
    matched analytic Pines gradient drives the variational equations, and three
    further conditions are enforced (each a hard error, never a silent
    downgrade):

    - the model must declare an explicit principal-axes realization
      (``MOON_PA_DE440``, not the generic ``MOON_PA`` alias);
    - ``harmonic_rotation`` must be supplied and must satisfy the same
      composition rules as the 6-state path (J2 double-count ban, epoch-grid
      requirement for m > 0, grid coverage);
    - the SAME model instance drives both the acceleration and the gradient, so
      trajectory/variational degree, order and coefficients cannot diverge.

    This does not change any default: with ``harmonic_model=None`` the function
    behaves exactly as before.
    """
    from scipy.integrate import solve_ivp

    t_eval_s = np.asarray(t_eval_s, dtype=float).reshape(-1)
    if harmonic_model is not None:
        if not harmonic_stm_opt_in:
            raise ValueError(
                "lunar harmonics in the STM are explicit opt-in only; pass "
                "harmonic_stm_opt_in=True to use the matched analytic Pines "
                "gradient, or use 6-state propagation. (Refusing rather than "
                "silently falling back to a J2-level gradient.)"
            )
        require_explicit_pa_realization(harmonic_model)
        _h_ctx = _prepare_harmonic_context(
            harmonic_model, harmonic_rotation, j2_moon, t_eval_s
        )
        return _propagate_augmented_with_harmonics(
            t_eval_s, state_aug0_mci, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
            get_earth_pos, get_sun_pos, harmonic_model, _h_ctx,
            rtol=rtol, atol=atol, method=method,
            j2_moon=j2_moon, j2_earth=j2_earth, earth_j2_mode=earth_j2_mode,
        )

    if t_eval_s.size == 0:
        raise ValueError("t_eval_s must contain at least one epoch.")

    state_aug0_mci = np.asarray(state_aug0_mci, dtype=float).reshape(-1)
    if state_aug0_mci.size != 42:
        raise ValueError("Initial augmented state must have 42 elements.")

    _emode = _earth_mode_int(j2_earth, earth_j2_mode)

    if _FAST_DYNAMICS:
        _j2 = float(j2_moon)
        _mr = MOON_R_M if j2_moon else 0.0
        _cbf = _MCI_TO_MOON_BF
        _j2e = float(j2_earth)
        _er = R_EARTH_J2_REF_M if j2_earth else 0.0
        _cbfe = _J2000_TO_EARTH_BF
        def rhs(t_s: float, state_aug: np.ndarray) -> np.ndarray:
            return _ode42_rhs_fast(
                state_aug, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
                get_earth_pos(float(t_s)), get_sun_pos(float(t_s)),
                _j2, _mr, _cbf, _j2e, _er, _emode, _cbfe,
            )
    else:
        def rhs(t_s: float, state_aug: np.ndarray) -> np.ndarray:
            return ode_fun_v3(
                t_s, state_aug, mu_moon_m3_s2, mu_earth_m3_s2, mu_sun_m3_s2,
                get_earth_pos, get_sun_pos, j2_moon, j2_earth, earth_j2_mode,
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
    j2_moon: float = 0.0,
    j2_earth: float = 0.0,
    earth_j2_mode: str = "indirect",
):
    """Return a fast (t0, t1, state6) → state6 callable using numba RK4.

    Uses pre-sampled ephemeris grids + binary-search linear interpolation.
    ~250× faster than scipy DOP853 for short intervals (UKF sigma steps).
    Position accuracy ~3 mm over 60 s — sufficient when measurement noise
    is ≥ 10 m.  Pass ``j2_moon`` / ``j2_earth`` to include the J2 perturbations
    in the sigma propagation as well.

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
    _j2 = float(j2_moon)
    _mr = MOON_R_M if j2_moon else 0.0
    _cbf = _MCI_TO_MOON_BF
    _j2e = float(j2_earth)
    _er = R_EARTH_J2_REF_M if j2_earth else 0.0
    _emode = _earth_mode_int(j2_earth, earth_j2_mode)
    _cbfe = _J2000_TO_EARTH_BF

    def _propagate(t0: float, t1: float, state6: np.ndarray) -> np.ndarray:
        if np.isclose(t0, t1):
            return state6.copy()
        return _rk4_6state_fast(
            np.asarray(state6, dtype=np.float64),
            float(t0), float(t1), mu_m, mu_e, mu_s,
            t_grid, earth_grid, sun_grid, dt=dt,
            j2_moon=_j2, moon_r=_mr, c_bf=_cbf,
            j2_earth=_j2e, earth_r=_er, earth_mode=_emode, c_bf_earth=_cbfe,
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
    j2_earth: float = 0.0,
    earth_j2_mode: str = "indirect",
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
        j2_earth=j2_earth,
        earth_j2_mode=earth_j2_mode,
    )


# ---------------------------------------------------------------------------
# Generic Body-J2 helpers (Phase 4).  Imported at module bottom so force_models
# can resolve zonal_j2_* (defined above) without an import cycle; f3body_moon /
# dynamics_jacobian_a_matrix reference these at call time, by which point the
# module is fully loaded.
# ---------------------------------------------------------------------------
from .force_models import (  # noqa: E402
    body_j2_acceleration,
    body_j2_gravity_gradient,
)
