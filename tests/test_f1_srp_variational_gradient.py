"""Phase 17-F1: the SRP position gradient in the variational equations.

Small permanent regressions; the full qualification harnesses and their
artifacts are examples/phase17_f1_{local_gradient_oracle,variational_oracle,
parity,materiality}.py and artifacts/phase17_f1_*.json.

Oracles never use the code under test: finite differences / integrals of the
PRODUCTION acceleration, and a fixed-step RK4 re-propagation of the PRODUCTION
force (identical step sequence for every perturbed run, so differencing it has
no adaptive-step noise).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from lunar_od.constants import AU_M, J2_MOON_UNNORMALIZED, MU_MOON_M3S2, R_MOON_M
from lunar_od.dynamics import (
    dynamics_jacobian_a_matrix,
    f3body_moon,
    propagate_augmented_state,
    propagate_state,
    propagate_state_with_k_sensitivity,
)
from lunar_od.srp import (
    SRPOptions,
    _illumination_for,
    solar_pressure_at,
    srp_acceleration_kernel,
    srp_acceleration_position_gradient,
    srp_acceleration_with_lunar_shadow,
)

ROOT = Path(__file__).resolve().parents[1]
SUN_DIR = np.array([-0.83, 0.52, 0.19]) / np.linalg.norm([-0.83, 0.52, 0.19])
SUN = SUN_DIR * AU_M * 1.0167
N_PERP = np.cross(SUN_DIR, [0.2, 0.9, -0.4])
N_PERP /= np.linalg.norm(N_PERP)
DIST = R_MOON_M + 100e3


def _penumbra_point(nu_target: float) -> np.ndarray:
    opts = SRPOptions(k_srp_m2_per_kg=0.01)
    axis = -DIST * SUN_DIR
    lo, hi = 0.0, 3 * R_MOON_M
    lo_nu = _illumination_for(axis + lo * N_PERP, SUN, opts)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (_illumination_for(axis + mid * N_PERP, SUN, opts) < nu_target) == (lo_nu < nu_target):
            lo = mid
        else:
            hi = mid
    return axis + 0.5 * (lo + hi) * N_PERP


def _integral_check(r, opts, half=200.0, nodes=32):
    """int G e ds over [-L, L] vs a(r+Le) - a(r-Le): no differencing of close values."""
    x, w = np.polynomial.legendre.leggauss(nodes)
    worst = 0.0
    for j in range(3):
        e = np.zeros(3)
        e[j] = 1.0
        lhs = half * sum(wi * (srp_acceleration_position_gradient(r + half * t * e, SUN, opts) @ e)
                         for wi, t in zip(w, x))
        rhs = (srp_acceleration_with_lunar_shadow(r + half * e, SUN, opts)
               - srp_acceleration_with_lunar_shadow(r - half * e, SUN, opts))
        worst = max(worst, float(np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs)))
    return worst


# ----------------------------------------------------------------------
# local gradient
# ----------------------------------------------------------------------
@pytest.mark.parametrize("nu_target", [0.1, 0.5, 0.9, 0.005, 0.995])
def test_penumbra_gradient_matches_production_acceleration(nu_target):
    r = _penumbra_point(nu_target)
    opts = SRPOptions(k_srp_m2_per_kg=0.01)
    assert 0.0 < _illumination_for(r, SUN, opts) < 1.0
    assert _integral_check(r, opts) < 1e-6


def test_full_sun_gradient_is_the_distance_direction_tensor():
    """Full light: G = K (P nu / d)(I - 3 u u^T) with nu = 1, and it matches FD."""
    r = DIST * SUN_DIR + 0.3 * DIST * N_PERP
    k = 0.037
    opts = SRPOptions(k_srp_m2_per_kg=k)
    q = r - SUN
    d = np.linalg.norm(q)
    u = q / d
    closed = k * solar_pressure_at(d) / d * (np.eye(3) - 3 * np.outer(u, u))
    g = srp_acceleration_position_gradient(r, SUN, opts)
    np.testing.assert_allclose(g, closed, rtol=1e-12, atol=0)
    h = 1e4                      # smooth on the 1.5e11 m scale; roundoff-limited only
    fd = np.column_stack([(srp_acceleration_with_lunar_shadow(r + h * e, SUN, opts)
                           - srp_acceleration_with_lunar_shadow(r - h * e, SUN, opts)) / (2 * h)
                          for e in np.eye(3)])
    assert np.linalg.norm(fd - g) / np.linalg.norm(g) < 1e-6


def test_no_shadow_has_no_illumination_term():
    r = _penumbra_point(0.5)
    g_ns = srp_acceleration_position_gradient(r, SUN, SRPOptions(k_srp_m2_per_kg=0.01, shadow_model="NO_SHADOW"))
    q = r - SUN
    d = np.linalg.norm(q)
    u = q / d
    np.testing.assert_allclose(g_ns, 0.01 * solar_pressure_at(d) / d * (np.eye(3) - 3 * np.outer(u, u)),
                               rtol=1e-12, atol=0)


def test_k_zero_umbra_and_inactive_give_exactly_zero():
    umbra = -DIST * SUN_DIR
    assert _illumination_for(umbra, SUN, SRPOptions(k_srp_m2_per_kg=0.01)) == 0.0
    assert not srp_acceleration_position_gradient(umbra, SUN, SRPOptions(k_srp_m2_per_kg=0.01)).any()
    pen = _penumbra_point(0.5)
    assert not srp_acceleration_position_gradient(pen, SUN, SRPOptions(k_srp_m2_per_kg=0.0)).any()
    assert not srp_acceleration_position_gradient(pen, SUN, SRPOptions(enabled=False)).any()


def test_gradient_is_linear_in_k_and_finite_at_the_contacts():
    pen = _penumbra_point(0.5)
    g1 = srp_acceleration_position_gradient(pen, SUN, SRPOptions(k_srp_m2_per_kg=0.01))
    g3 = srp_acceleration_position_gradient(pen, SUN, SRPOptions(k_srp_m2_per_kg=0.03))
    np.testing.assert_allclose(g3, 3 * g1, rtol=1e-12)
    for nu in (1e-9, 1 - 1e-9):
        g = srp_acceleration_position_gradient(_penumbra_point(nu), SUN, SRPOptions(k_srp_m2_per_kg=0.01))
        assert np.all(np.isfinite(g))
        assert np.linalg.norm(g) < 0.05 * np.linalg.norm(g1)   # grad nu -> 0 at tangency


# ----------------------------------------------------------------------
# the A matrix and the propagation entry points
# ----------------------------------------------------------------------
def test_a_matrix_adds_exactly_the_srp_gradient_in_the_position_block():
    r = _penumbra_point(0.5)
    x = np.concatenate([r, [10.0, 1600.0, 5.0]])
    srp = SRPOptions(k_srp_m2_per_kg=0.01)
    args = (x, MU_MOON_M3S2, 0.0, 0.0, np.array([3.844e8, 0, 0]), SUN)
    a0 = dynamics_jacobian_a_matrix(*args, j2_moon=J2_MOON_UNNORMALIZED)
    a1 = dynamics_jacobian_a_matrix(*args, j2_moon=J2_MOON_UNNORMALIZED, srp=srp)
    diff = a1 - a0
    # (A_grav + G) - A_grav cancels to eps * |A_grav| ~ 1e-22 s^-2 (G is ~1e-6 of A_grav)
    np.testing.assert_allclose(diff[3:6, 0:3], srp_acceleration_position_gradient(r, SUN, srp),
                               rtol=0, atol=4 * np.finfo(float).eps * np.abs(a0[3:6, 0:3]).max())
    diff[3:6, 0:3] = 0.0
    assert not diff.any()                       # no velocity dependence, nothing else moves
    np.testing.assert_array_equal(
        dynamics_jacobian_a_matrix(*args, j2_moon=J2_MOON_UNNORMALIZED, srp=SRPOptions(enabled=False)), a0)


def test_state_only_propagation_never_evaluates_the_gradient(monkeypatch):
    """F1 is a variational correction: the 6-state truth path must not touch it."""
    import lunar_od.srp as srp_mod

    def boom(*_a, **_k):
        raise AssertionError("state-only propagation evaluated the SRP gradient")
    monkeypatch.setattr(srp_mod, "srp_acceleration_position_gradient", boom)
    x0 = np.concatenate([_penumbra_point(0.5), [0.0, 1600.0, 0.0]])
    propagate_state(np.array([0.0, 60.0]), x0, MU_MOON_M3S2, 0.0, 0.0,
                    lambda _t: np.array([3.844e8, 0, 0]), lambda _t: SUN,
                    srp=SRPOptions(k_srp_m2_per_kg=0.01))


# ----------------------------------------------------------------------
# STM and S_K against a fixed-step RK4 re-propagation, over one eclipse orbit
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def eclipse_orbit():
    examples = ROOT / "examples"
    if str(examples) not in sys.path:
        sys.path.insert(0, str(examples))
    import phase17_r1m_core as M
    if not (M.CAMPAIGN_ROOT / "08_manifests").exists():
        pytest.skip("read-only campaign data not available")
    import phase17_geo_core as G
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels(None, clear=True)
    et0, t_orbit = M.campaign_epoch()
    x0 = M.campaign_initial_state()
    t = np.arange(0.0, t_orbit + 45.0, 90.0)
    sun = lambda tt: G.sun_mci(et0 + float(tt))  # noqa: E731
    return dict(t=t, x0=x0, mu=G.mu_moon(), sun=sun, earth=np.array([384_400e3, 0.0, 0.0]))


def _rk4(case, x0, k, dt=2.0):
    srp = SRPOptions(k_srp_m2_per_kg=k)

    def f(tt, y):
        return f3body_moon(y, case["mu"], 0.0, 0.0, case["earth"], case["sun"](tt),
                           j2_moon=J2_MOON_UNNORMALIZED, srp=srp)
    y, out, tt = np.asarray(x0, float).copy(), [np.asarray(x0, float)], 0.0
    for _ in range(1, case["t"].size):
        for _ in range(int(90.0 / dt)):
            k1 = f(tt, y)
            k2 = f(tt + dt / 2, y + dt / 2 * k1)
            k3 = f(tt + dt / 2, y + dt / 2 * k2)
            k4 = f(tt + dt, y + dt * k3)
            y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            tt += dt
        out.append(y)
    return np.array(out)


def _old_model(case, k):
    """Analysis-only replica of the PRE-F1 variational equations (negative control)."""
    import lunar_od.dynamics as dyn
    srp = SRPOptions(k_srp_m2_per_kg=k)

    def rhs(tt, y):
        x, phi, s = y[:6], y[6:42].reshape((6, 6), order="F"), y[42:48]
        r_sun = case["sun"](tt)
        a = dynamics_jacobian_a_matrix(x, case["mu"], 0.0, 0.0, case["earth"], r_sun, j2_moon=J2_MOON_UNNORMALIZED)
        dfdk = np.concatenate([np.zeros(3), srp_acceleration_kernel(x[:3], r_sun, srp)])
        return np.concatenate([f3body_moon(x, case["mu"], 0.0, 0.0, case["earth"], r_sun,
                                           j2_moon=J2_MOON_UNNORMALIZED, srp=srp),
                               (a @ phi).reshape(-1, order="F"), a @ s + dfdk])
    y0 = np.concatenate([case["x0"], np.eye(6).reshape(-1, order="F"), np.zeros(6)])
    return dyn._propagate_vode(case["t"], y0, rhs, 1e-13, 1e-14)


def _new_model(case, k):
    return propagate_state_with_k_sensitivity(
        case["t"], case["x0"], case["mu"], 0.0, 0.0, lambda _t: case["earth"], case["sun"],
        srp=SRPOptions(k_srp_m2_per_kg=k), rtol=1e-13, atol=1e-14, j2_moon=J2_MOON_UNNORMALIZED)


def _stm_err(case, hist):
    phi = np.array([row.reshape((6, 6), order="F") for row in hist[:, 6:42]])
    worst = 0.0
    for j in range(6):
        h = 10.0 if j < 3 else 1e-2
        e = np.zeros(6)
        e[j] = h
        fd = (_rk4(case, case["x0"] + e, 0.01) - _rk4(case, case["x0"] - e, 0.01)) / (2 * h)
        an = phi[:, :, j]
        worst = max(worst, float(np.max(np.abs(fd - an)) / np.max(np.abs(an))))
    return worst


def test_corrected_stm_matches_repropagation_and_the_old_one_does_not(eclipse_orbit):
    new = _stm_err(eclipse_orbit, _new_model(eclipse_orbit, 0.01))
    old = _stm_err(eclipse_orbit, _old_model(eclipse_orbit, 0.01))
    assert new <= 1e-6, new
    assert old >= 10 * new, (old, new)       # HARDEN-D05 would have been caught


def test_corrected_k_sensitivity_matches_k_repropagation(eclipse_orbit):
    dk = 1e-2
    fd = (_rk4(eclipse_orbit, eclipse_orbit["x0"], 0.01 + dk)
          - _rk4(eclipse_orbit, eclipse_orbit["x0"], 0.01 - dk)) / (2 * dk)
    s_k = _new_model(eclipse_orbit, 0.01)[:, 42:48]
    err = float(np.max(np.abs(fd - s_k)) / np.max(np.abs(s_k)))
    assert err <= 1e-5, err


def test_42_state_fast_and_reference_paths_carry_the_same_gradient(eclipse_orbit):
    import lunar_od.dynamics as dyn
    case = eclipse_orbit
    aug0 = np.concatenate([case["x0"], np.eye(6).reshape(-1, order="F")])
    kw = dict(rtol=1e-13, atol=1e-14, j2_moon=J2_MOON_UNNORMALIZED, srp=SRPOptions(k_srp_m2_per_kg=0.01))
    fast = propagate_augmented_state(case["t"], aug0, case["mu"], 0.0, 0.0, lambda _t: case["earth"],
                                     case["sun"], **kw)
    saved = dyn._FAST_DYNAMICS
    dyn._FAST_DYNAMICS = False
    try:
        ref = propagate_augmented_state(case["t"], aug0, case["mu"], 0.0, 0.0, lambda _t: case["earth"],
                                        case["sun"], **kw)
    finally:
        dyn._FAST_DYNAMICS = saved
    k48 = _new_model(case, 0.01)
    scale = np.max(np.abs(k48[:, 6:42]))
    assert np.max(np.abs(fast[:, 6:42] - k48[:, 6:42])) / scale < 1e-8
    assert np.max(np.abs(ref[:, 6:42] - k48[:, 6:42])) / scale < 1e-8


def test_f1_refreeze_is_additive_and_history_is_preserved():
    """The protected dynamics.py hash moved only through an additive, documented
    F1 re-freeze; earlier provenance records are untouched."""
    import hashlib
    examples = ROOT / "examples"
    if str(examples) not in sys.path:
        sys.path.insert(0, str(examples))
    import r2_measurement_fidelity_validation as r2
    path = "lunar_od/dynamics.py"
    assert r2.PHASE17_HARDEN_REFROZEN_CURRENT_TREE_SHA256[path].startswith("b369899a")
    assert r2.CLOSURE_REFROZEN_CURRENT_TREE_SHA256[path].startswith("d0f7ab02")
    assert r2.PHASE17_F1_REFROZEN_CURRENT_TREE_SHA256[path].startswith("c6e7975c")
    # Phase D2 (owner-authorized 2026-10-23) re-froze dynamics.py after F1.
    assert r2.PHASE_D2_REFROZEN_CURRENT_TREE_SHA256[path].startswith("2f531a04")
    # The user-requested numerical-core correction is a later additive identity.
    new = r2.NUMERICAL_CORE_REFROZEN_CURRENT_TREE_SHA256[path]
    assert r2.expected_current_tree_sha256(path) == new
    assert hashlib.sha256((ROOT / r2.current_source_path(path)).read_bytes()).hexdigest() == new
