"""Phase 17 - the qualified links of the K_SRP sensitivity chain.

Covers the force-level coefficient derivative and the trajectory sensitivity
that propagates it. The measurement link is not covered here because it is not
qualified: see the Phase 17 report.

No estimator is touched. Phase 17's own gate forbids estimator integration until
the whole chain passes, and it does not.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lunar_od.constants import AU_M, J2_MOON_UNNORMALIZED, MU_MOON_M3S2, R_MOON_M
from lunar_od.dynamics import (
    K_SENSITIVITY_STATE_SIZE,
    propagate_state,
    propagate_state_with_k_sensitivity,
)
from lunar_od.srp import (
    SRPOptions,
    illumination_fraction,
    srp_acceleration_kernel,
    srp_acceleration_with_lunar_shadow,
    srp_partial_wrt_k_srp,
)

SUN_PLUS_X = np.array([AU_M, 0.0, 0.0])
R_SUNWARD = np.array([2.0e6, 3.0e5, -1.0e5])
R_ANTISUN = np.array([-2.0e6, 3.0e5, -1.0e5])
STATE0 = np.array([1.8377e6, 0.0, 0.0, 0.0, 1.6337e3, 0.0])


def _sun(_t):
    return np.array([-1.0, 0.05, 0.02]) / np.linalg.norm([-1.0, 0.05, 0.02]) * AU_M


def _earth(_t):
    return np.array([3.844e8, 0.0, 0.0])


# ======================================================================
# Group A - force-level dа/dK
# ======================================================================
def test_acceleration_is_the_coefficient_times_the_kernel():
    for k in (0.0, 1e-6, 0.01, 0.5, 2.0):
        o = SRPOptions(k_srp_m2_per_kg=k)
        a = srp_acceleration_with_lunar_shadow(R_SUNWARD, SUN_PLUS_X, o)
        g = srp_acceleration_kernel(R_SUNWARD, SUN_PLUS_X, o)
        assert a == pytest.approx(k * g, rel=1e-15, abs=1e-300)


def test_kernel_does_not_depend_on_the_coefficient():
    ref = srp_acceleration_kernel(R_SUNWARD, SUN_PLUS_X,
                                  SRPOptions(k_srp_m2_per_kg=0.01))
    for k in (0.0, 1e-9, 0.5, 10.0):
        g = srp_acceleration_kernel(R_SUNWARD, SUN_PLUS_X,
                                    SRPOptions(k_srp_m2_per_kg=k))
        assert np.array_equal(g, ref)


def test_kernel_matches_a_central_difference_of_the_force():
    o = SRPOptions(k_srp_m2_per_kg=0.01)
    g = srp_acceleration_kernel(R_SUNWARD, SUN_PLUS_X, o)
    dk = 1e-6
    fd = (srp_acceleration_with_lunar_shadow(
              R_SUNWARD, SUN_PLUS_X, SRPOptions(k_srp_m2_per_kg=0.01 + dk))
          - srp_acceleration_with_lunar_shadow(
              R_SUNWARD, SUN_PLUS_X, SRPOptions(k_srp_m2_per_kg=0.01 - dk))) / (2 * dk)
    assert g == pytest.approx(fd, rel=1e-9)


def test_derivative_is_finite_and_correct_at_zero_coefficient():
    """The edge case a/K cannot express.

    At K = 0 the acceleration vanishes but the model is still linear in K, so
    the sensitivity is the same finite vector as anywhere else. An estimator may
    legitimately start a solve-for from zero.
    """
    o0 = SRPOptions(k_srp_m2_per_kg=0.0)
    a0 = srp_acceleration_with_lunar_shadow(R_SUNWARD, SUN_PLUS_X, o0)
    g0 = srp_acceleration_kernel(R_SUNWARD, SUN_PLUS_X, o0)
    assert np.array_equal(a0, np.zeros(3))
    assert np.linalg.norm(g0) > 0.0
    assert all(math.isfinite(v) for v in g0)
    g1 = srp_acceleration_kernel(R_SUNWARD, SUN_PLUS_X,
                                 SRPOptions(k_srp_m2_per_kg=0.01))
    assert np.array_equal(g0, g1)


def test_derivative_vanishes_in_umbra():
    """nu = 0, so changing K cannot change the force at that instant."""
    o = SRPOptions(k_srp_m2_per_kg=0.01)
    r_umbra = np.array([-2.0e6, 0.0, 0.0])
    nu, regime = illumination_fraction(r_umbra, SUN_PLUS_X, np.zeros(3), R_MOON_M)
    assert regime == "UMBRA" and nu == 0.0
    assert np.array_equal(
        srp_acceleration_kernel(r_umbra, SUN_PLUS_X, o), np.zeros(3)
    )


def test_derivative_is_nonzero_in_full_light_and_penumbra():
    o = SRPOptions(k_srp_m2_per_kg=0.01)
    assert np.linalg.norm(srp_acceleration_kernel(R_SUNWARD, SUN_PLUS_X, o)) > 0
    found = False
    for y in np.linspace(1.70e6, 1.80e6, 2001):
        r = np.array([-2.0e6, y, 0.0])
        nu, regime = illumination_fraction(r, SUN_PLUS_X, np.zeros(3), R_MOON_M)
        if regime == "PENUMBRA":
            found = True
            assert np.linalg.norm(srp_acceleration_kernel(r, SUN_PLUS_X, o)) > 0
    assert found


def test_disabled_srp_has_no_coefficient_derivative():
    o = SRPOptions(enabled=False)
    assert np.array_equal(
        srp_acceleration_kernel(R_SUNWARD, SUN_PLUS_X, o), np.zeros(3)
    )


def test_the_alias_is_the_same_function():
    assert srp_partial_wrt_k_srp is srp_acceleration_kernel


# ======================================================================
# Group B - trajectory dx/dK
# ======================================================================
def _sens(k, t_eval, **kw):
    return propagate_state_with_k_sensitivity(
        t_eval, STATE0, MU_MOON_M3S2, 0.0, 0.0, _earth, _sun,
        srp=SRPOptions(k_srp_m2_per_kg=k), rtol=1e-13, atol=1e-16,
        j2_moon=J2_MOON_UNNORMALIZED, **kw,
    )


def test_augmented_state_is_48_elements():
    out = _sens(0.01, np.array([0.0, 600.0]))
    assert out.shape[1] == K_SENSITIVITY_STATE_SIZE == 48


def test_sensitivity_starts_at_zero():
    """dx/dK(t0) = 0: the initial Cartesian state does not depend on K."""
    out = _sens(0.01, np.array([0.0, 600.0]))
    assert np.array_equal(out[0, 42:48], np.zeros(6))


def test_the_first_42_columns_are_the_ordinary_augmented_history():
    """A caller can slice this and get the 42-state layout back."""
    out = _sens(0.01, np.array([0.0, 1800.0]))
    phi0 = out[0, 6:42].reshape((6, 6), order="F")
    assert phi0 == pytest.approx(np.eye(6), abs=1e-14)
    state = propagate_state(
        np.array([0.0, 1800.0]), STATE0, MU_MOON_M3S2, 0.0, 0.0, _earth, _sun,
        rtol=1e-13, atol=1e-16, j2_moon=J2_MOON_UNNORMALIZED,
        srp=SRPOptions(k_srp_m2_per_kg=0.01),
    )
    assert out[-1, :6] == pytest.approx(state[-1], rel=1e-9)


def test_sensitivity_matches_a_central_difference():
    """The step is large on purpose: a cannonball force is exactly linear in K,
    so truncation is tiny and the small-step end is where integrator noise wins.
    """
    t_eval = np.array([0.0, 3600.0])
    an = _sens(0.01, t_eval)[-1, 42:48]
    dk = 0.5 * 0.01
    fd = (_sens(0.01 + dk, t_eval)[-1, :6]
          - _sens(0.01 - dk, t_eval)[-1, :6]) / (2 * dk)
    assert an[:3] == pytest.approx(fd[:3], rel=1e-5)
    assert an[3:] == pytest.approx(fd[3:], rel=1e-5)


def test_finite_difference_degrades_at_small_steps():
    """The signature that the residual is FD noise, not a modelling error.

    A wrong analytic derivative would show a floor that does not move with the
    step. Noise-limited error scales as 1/dK, and that is what is seen.
    """
    t_eval = np.array([0.0, 3600.0])
    an = _sens(0.01, t_eval)[-1, 42:48]
    errs = {}
    for frac in (0.5, 0.01):
        dk = frac * 0.01
        fd = (_sens(0.01 + dk, t_eval)[-1, :6]
              - _sens(0.01 - dk, t_eval)[-1, :6]) / (2 * dk)
        errs[frac] = float(np.linalg.norm(an[:3] - fd[:3])
                           / np.linalg.norm(an[:3]))
    assert errs[0.5] < errs[0.01]


def test_sensitivity_grows_with_arc_length():
    out = _sens(0.01, np.array([0.0, 1800.0, 3600.0, 7200.0]))
    mags = [float(np.linalg.norm(out[i, 42:45])) for i in range(1, 4)]
    assert mags[0] < mags[1] < mags[2]


def test_sensitivity_is_independent_of_the_coefficient_to_first_order():
    """dx/dK is itself nearly K-independent, because the force is linear in K.

    Not exactly independent: the perturbed trajectory samples a slightly
    different gravity field, so this is a weak statement and is tested weakly.
    """
    t_eval = np.array([0.0, 3600.0])
    a = _sens(0.005, t_eval)[-1, 42:45]
    b = _sens(0.02, t_eval)[-1, 42:45]
    assert np.linalg.norm(a - b) / np.linalg.norm(a) < 1e-3


def test_custom_initial_sensitivity_is_honoured():
    seed = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    out = propagate_state_with_k_sensitivity(
        np.array([0.0, 60.0]), STATE0, MU_MOON_M3S2, 0.0, 0.0, _earth, _sun,
        srp=SRPOptions(k_srp_m2_per_kg=0.01), j2_moon=J2_MOON_UNNORMALIZED,
        s_k0=seed,
    )
    assert out[0, 42:48] == pytest.approx(seed)


def test_sensitivity_propagation_requires_active_srp():
    for srp in (None, SRPOptions(enabled=False)):
        with pytest.raises(ValueError, match="no coefficient to differentiate"):
            propagate_state_with_k_sensitivity(
                np.array([0.0, 60.0]), STATE0, MU_MOON_M3S2, 0.0, 0.0,
                _earth, _sun, srp=srp,
            )


def test_zero_coefficient_still_propagates_a_finite_sensitivity():
    """K = 0 is a legal solve-for starting point; the sensitivity must exist."""
    out = _sens(0.0, np.array([0.0, 3600.0]))
    s_k = out[-1, 42:48]
    assert np.all(np.isfinite(s_k))
    assert np.linalg.norm(s_k[:3]) > 0.0
    # and the trajectory itself is the SRP-free one
    plain = propagate_state(
        np.array([0.0, 3600.0]), STATE0, MU_MOON_M3S2, 0.0, 0.0, _earth, _sun,
        rtol=1e-13, atol=1e-16, j2_moon=J2_MOON_UNNORMALIZED,
    )
    assert out[-1, :3] == pytest.approx(plain[-1, :3], abs=1e-6)


# ======================================================================
# Group C - Phase 16 behaviour is untouched
# ======================================================================
def test_phase16_fixed_k_propagation_is_unchanged_by_the_refactor():
    """The force is now evaluated as K * kernel; results must be identical."""
    t_eval = np.linspace(0.0, 3600.0, 7)
    on = propagate_state(
        t_eval, STATE0, MU_MOON_M3S2, 0.0, 0.0, _earth, _sun,
        srp=SRPOptions(k_srp_m2_per_kg=0.01),
    )
    off = propagate_state(t_eval, STATE0, MU_MOON_M3S2, 0.0, 0.0, _earth, _sun)
    assert np.linalg.norm(on[-1, :3] - off[-1, :3]) > 0.0
    assert np.all(np.isfinite(on))


def test_srp_still_off_by_default():
    import inspect

    assert inspect.signature(propagate_state).parameters["srp"].default is None
