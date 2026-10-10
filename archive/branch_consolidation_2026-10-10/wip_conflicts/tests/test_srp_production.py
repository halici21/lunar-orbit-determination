"""Phase 16 - production cannonball SRP with a conical lunar shadow.

The parity group imports the frozen Phase 13 campaign oracle by path and
compares against it. The oracle is never modified and never calls production, so
the comparison is between two independent implementations of the same model
rather than a function checked against itself.

Everything else here is about what production refuses to do: run SRP by
default, accept an enabled configuration with no coefficient, take a number
from the reference spacecraft that reports it as unknown, or treat a screening
range as a validator.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from lunar_od.constants import (
    AU_M,
    MU_EARTH_M3S2,
    MU_MOON_M3S2,
    MU_SUN_M3S2,
    R_MOON_M,
    R_SUN_M,
    SOLAR_IRRADIANCE_1AU_W_M2,
    SOLAR_PRESSURE_1AU_N_M2,
    SPEED_OF_LIGHT_M_S,
)
from lunar_od.dynamics import f3body_moon, propagate_state
from lunar_od.srp import (
    SRPConfigurationError,
    SRPOptions,
    apparent_radii_and_separation,
    illumination_fraction,
    solar_pressure_at,
    srp_acceleration,
    srp_acceleration_with_lunar_shadow,
)

PHASE13_REFERENCE = Path(
    "C:/Users/erayh/Documents/Python/Grad/od_covariance_campaign/06_scripts/"
    "phase13_srp_reference.py"
)
PHASE14_DOCUMENT = Path(
    "C:/Users/erayh/Documents/Python/Grad/od_covariance_campaign/03_data/"
    "phase14_reference_spacecraft_configuration_v1.json"
)

#: Phase 13's parametric screening value. Used here ONLY to reproduce Phase 13,
#: never as a production default. Nothing in lunar_od/ contains this number.
PHASE13_PARAMETRIC_K_SRP = 0.01

# A representative Moon-centered geometry: ~100 km lunar orbit, Sun roughly
# along -X at one astronomical unit.
R_SC = np.array([1.8e6, 3.0e5, -2.0e5])
R_SUN = np.array([-1.0, 0.05, 0.02]) / np.linalg.norm([-1.0, 0.05, 0.02]) * AU_M


def _load_phase13_oracle():
    spec = importlib.util.spec_from_file_location(
        "phase13_srp_reference_oracle", PHASE13_REFERENCE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


requires_oracle = pytest.mark.skipif(
    not PHASE13_REFERENCE.exists(),
    reason="Phase 13 campaign oracle is not present in this checkout",
)


# ======================================================================
# Group A - activation semantics
# ======================================================================
def test_srp_is_off_by_default_in_f3body_moon():
    base = f3body_moon(
        np.concatenate([R_SC, [0.0, 1.6e3, 0.0]]),
        MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
        np.array([3.8e8, 0.0, 0.0]), R_SUN,
    )
    explicit_none = f3body_moon(
        np.concatenate([R_SC, [0.0, 1.6e3, 0.0]]),
        MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
        np.array([3.8e8, 0.0, 0.0]), R_SUN, srp=None,
    )
    assert np.array_equal(base, explicit_none)


def test_enabling_srp_without_a_coefficient_fails_closed():
    with pytest.raises(SRPConfigurationError, match="no default"):
        SRPOptions()


def test_the_failure_message_names_the_alternatives_it_refuses():
    with pytest.raises(SRPConfigurationError) as exc:
        SRPOptions(enabled=True)
    message = str(exc.value)
    assert "UNKNOWN" in message
    assert "screening envelope" in message


def test_disabled_srp_needs_no_coefficient():
    options = SRPOptions(enabled=False)
    assert not options.is_active
    assert np.array_equal(
        srp_acceleration_with_lunar_shadow(R_SC, R_SUN, options), np.zeros(3)
    )


def test_production_srp_module_contains_no_hardcoded_coefficient():
    """No default, no screening endpoint, no Phase 13 parametric value.

    Checked over the parsed numeric literals rather than the source text, so
    that discussing a value in a docstring is not mistaken for using one.
    """
    import ast

    source = Path(
        importlib.util.find_spec("lunar_od.srp").origin
    ).read_text(encoding="utf-8")
    literals = {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, float)
    }
    for forbidden in (0.01, 0.00864, 0.048358660031063726, 0.02379):
        assert forbidden not in literals, forbidden


def test_production_dynamics_never_loads_a_reference_configuration():
    for name in ("lunar_od.dynamics", "lunar_od.srp", "lunar_od.constants"):
        source = Path(
            importlib.util.find_spec(name).origin
        ).read_text(encoding="utf-8")
        assert "reference_config" not in source, name
        assert "load_reference_configuration" not in source, name


# ======================================================================
# Group B - coefficient validation
# ======================================================================
def test_positive_finite_k_is_accepted():
    assert SRPOptions(k_srp_m2_per_kg=0.02).require_k() == 0.02


def test_zero_k_is_accepted():
    options = SRPOptions(k_srp_m2_per_kg=0.0)
    assert options.is_active
    assert options.require_k() == 0.0


def test_negative_k_is_rejected():
    with pytest.raises(SRPConfigurationError, match="non-negative"):
        SRPOptions(k_srp_m2_per_kg=-1e-3)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_k_is_rejected(bad):
    with pytest.raises(SRPConfigurationError, match="finite"):
        SRPOptions(k_srp_m2_per_kg=bad)


@pytest.mark.parametrize("bad", ["0.01", None, True, [0.01]])
def test_wrong_type_k_is_rejected(bad):
    with pytest.raises(SRPConfigurationError):
        SRPOptions(k_srp_m2_per_kg=bad)


def test_k_outside_the_screening_envelope_is_not_rejected():
    """The Phase 15 envelope is not a physical bound and must not act as one.

    0.06 lies above its upper endpoint of 0.048359 and must still be accepted:
    that endpoint excludes the solar arrays, so it bounds nothing.
    """
    assert SRPOptions(k_srp_m2_per_kg=0.06).require_k() == 0.06
    assert SRPOptions(k_srp_m2_per_kg=1e-6).require_k() == 1e-6


def test_invalid_shadow_model_is_rejected():
    with pytest.raises(SRPConfigurationError, match="shadow_model"):
        SRPOptions(k_srp_m2_per_kg=0.01, shadow_model="BINARY_CYLINDRICAL")


def test_invalid_k_source_is_rejected():
    with pytest.raises(SRPConfigurationError, match="k_srp_source"):
        SRPOptions(k_srp_m2_per_kg=0.01, k_srp_source="MEASURED_BY_SOMEONE")


def test_cr_and_area_compose_into_the_product():
    options = SRPOptions.from_cr_and_area_to_mass(1.5, 0.02)
    assert options.require_k() == pytest.approx(0.03, rel=1e-15)
    assert "C_R * A/m" in options.provenance


# ======================================================================
# Group C - force physics
# ======================================================================
def test_solar_pressure_constant_matches_s_over_c():
    assert SOLAR_PRESSURE_1AU_N_M2 == pytest.approx(
        SOLAR_IRRADIANCE_1AU_W_M2 / SPEED_OF_LIGHT_M_S, rel=1e-15
    )
    assert SOLAR_PRESSURE_1AU_N_M2 == pytest.approx(4.539807e-6, rel=1e-6)


def test_speed_of_light_agrees_with_the_measurement_domain():
    """One number, two declarations, proven equal so they cannot drift."""
    from lunar_od.measurements import C_LIGHT_MPS

    assert SPEED_OF_LIGHT_M_S == C_LIGHT_MPS


def test_force_is_anti_solar_with_the_sun_at_plus_x():
    """Sun at +X relative to the spacecraft: acceleration must point at -X."""
    r_sc = np.zeros(3)
    r_sun = np.array([AU_M, 0.0, 0.0])
    a = srp_acceleration(r_sc, r_sun, 0.01)
    assert a[0] < 0.0
    assert abs(a[1]) < 1e-30 and abs(a[2]) < 1e-30
    unit = a / np.linalg.norm(a)
    assert unit == pytest.approx(np.array([-1.0, 0.0, 0.0]), abs=1e-15)


def test_magnitude_at_one_au_is_p_times_k():
    a = srp_acceleration(np.zeros(3), np.array([AU_M, 0.0, 0.0]), 0.01)
    assert np.linalg.norm(a) == pytest.approx(
        SOLAR_PRESSURE_1AU_N_M2 * 0.01, rel=1e-14
    )


def test_inverse_square_sun_distance_scaling():
    near = solar_pressure_at(0.5 * AU_M)
    at_au = solar_pressure_at(AU_M)
    far = solar_pressure_at(2.0 * AU_M)
    assert near / at_au == pytest.approx(4.0, rel=1e-14)
    assert far / at_au == pytest.approx(0.25, rel=1e-14)


def test_sun_distance_is_not_pinned_to_one_au():
    """A fixed-1-AU shortcut would make these two equal. They must not be."""
    a_close = srp_acceleration(np.zeros(3), np.array([0.98 * AU_M, 0, 0]), 0.01)
    a_far = srp_acceleration(np.zeros(3), np.array([1.02 * AU_M, 0, 0]), 0.01)
    ratio = np.linalg.norm(a_close) / np.linalg.norm(a_far)
    assert ratio == pytest.approx((1.02 / 0.98) ** 2, rel=1e-12)
    assert ratio > 1.08


def test_acceleration_is_linear_in_k_at_fixed_geometry():
    a1 = srp_acceleration(R_SC, R_SUN, 0.01)
    a2 = srp_acceleration(R_SC, R_SUN, 0.03)
    assert np.linalg.norm(a2) / np.linalg.norm(a1) == pytest.approx(3.0, rel=1e-14)
    assert a2 == pytest.approx(3.0 * a1, rel=1e-14)


def test_zero_k_gives_exactly_zero_acceleration():
    assert np.array_equal(srp_acceleration(R_SC, R_SUN, 0.0), np.zeros(3))


def test_zero_illumination_gives_exactly_zero_acceleration():
    assert np.array_equal(
        srp_acceleration(R_SC, R_SUN, 0.01, illumination=0.0), np.zeros(3)
    )


def test_units_would_catch_a_kilometre_metre_confusion():
    """A km/m slip in the Sun distance moves the magnitude by 10^6."""
    correct = np.linalg.norm(srp_acceleration(np.zeros(3), np.array([AU_M, 0, 0]), 0.01))
    slipped = np.linalg.norm(
        srp_acceleration(np.zeros(3), np.array([AU_M / 1000.0, 0, 0]), 0.01)
    )
    assert slipped / correct == pytest.approx(1e6, rel=1e-9)
    assert 4.5e-8 < correct < 4.6e-8      # a physically sane 1 AU, K=0.01 value


def test_coincident_sun_and_spacecraft_is_rejected():
    with pytest.raises(SRPConfigurationError, match="coincident"):
        srp_acceleration(R_SC, R_SC, 0.01)


# ======================================================================
# Group D - shadow
# ======================================================================
def _sunward(distance_m: float) -> np.ndarray:
    """A spacecraft position on the Sun side of the Moon."""
    return np.array([distance_m, 0.0, 0.0])


SUN_PLUS_X = np.array([AU_M, 0.0, 0.0])


def test_full_light_on_the_sunward_side():
    nu, regime = illumination_fraction(_sunward(2.0e6), SUN_PLUS_X, np.zeros(3), R_MOON_M)
    assert nu == 1.0
    assert regime == "FULL_LIGHT"


def test_umbra_directly_behind_the_moon():
    nu, regime = illumination_fraction(
        _sunward(-2.0e6), SUN_PLUS_X, np.zeros(3), R_MOON_M
    )
    assert nu == 0.0
    assert regime == "UMBRA"


def test_a_penumbra_exists_and_is_partial():
    """Sweep the terminator until a strictly partial illumination appears."""
    found = []
    for y in np.linspace(1.70e6, 1.80e6, 4001):
        nu, regime = illumination_fraction(
            np.array([-2.0e6, y, 0.0]), SUN_PLUS_X, np.zeros(3), R_MOON_M
        )
        if 1e-9 < nu < 1.0 - 1e-9:
            found.append((y, nu, regime))
    assert found, "no penumbra found along the terminator sweep"
    assert all(r == "PENUMBRA" for _, _, r in found)


def test_illumination_is_bounded_and_finite_everywhere_on_a_fine_sweep():
    for y in np.linspace(0.0, 4.0e6, 20001):
        nu, _ = illumination_fraction(
            np.array([-2.0e6, y, 0.0]), SUN_PLUS_X, np.zeros(3), R_MOON_M
        )
        assert 0.0 <= nu <= 1.0
        assert math.isfinite(nu)


def test_illumination_is_continuous_through_the_penumbra():
    """The conical increment must shrink with the sampling step.

    That scaling is what separates a continuous model from a binary one, whose
    step stays at a full unit however finely it is sampled.
    """
    ys_coarse = np.linspace(1.70e6, 1.80e6, 401)
    ys_fine = np.linspace(1.70e6, 1.80e6, 4001)
    steps = []
    for ys in (ys_coarse, ys_fine):
        nu = np.array([
            illumination_fraction(
                np.array([-2.0e6, y, 0.0]), SUN_PLUS_X, np.zeros(3), R_MOON_M
            )[0]
            for y in ys
        ])
        steps.append(float(np.max(np.abs(np.diff(nu)))))
    coarse, fine = steps
    assert fine < coarse
    assert fine / coarse == pytest.approx(0.1, rel=0.5)


def test_all_four_shadow_transitions_are_traversed():
    """A full sweep across the shadow must visit light, penumbra and umbra."""
    regimes = []
    for y in np.linspace(2.2e6, -2.2e6, 6001):
        _, regime = illumination_fraction(
            np.array([-2.0e6, y, 0.0]), SUN_PLUS_X, np.zeros(3), R_MOON_M
        )
        if not regimes or regimes[-1] != regime:
            regimes.append(regime)
    assert regimes[0] == "FULL_LIGHT" and regimes[-1] == "FULL_LIGHT"
    assert "UMBRA" in regimes
    assert regimes.count("PENUMBRA") == 2, regimes


def test_near_tangency_stays_in_domain():
    """The exact contact geometries must not produce a domain error or NaN."""
    a_s, a_o, _, d_sun, d_occ = apparent_radii_and_separation(
        _sunward(-2.0e6), SUN_PLUS_X, np.zeros(3), R_MOON_M
    )
    for gamma_target in (a_s + a_o, abs(a_s - a_o)):
        for delta in (-1e-14, 0.0, 1e-14):
            y = math.tan(gamma_target + delta) * 2.0e6
            nu, _ = illumination_fraction(
                np.array([-2.0e6, y, 0.0]), SUN_PLUS_X, np.zeros(3), R_MOON_M
            )
            assert math.isfinite(nu) and 0.0 <= nu <= 1.0
    assert d_sun > 0 and d_occ > 0


def test_annular_geometry_is_reachable():
    """Far from a small occultor the Moon fits inside the solar disk."""
    nu, regime = illumination_fraction(
        np.array([-4.0e9, 0.0, 0.0]), SUN_PLUS_X, np.zeros(3), R_MOON_M
    )
    assert regime == "ANNULAR"
    assert 0.0 < nu < 1.0


def test_no_shadow_model_ignores_the_occultation():
    umbral = _sunward(-2.0e6)
    with_shadow = srp_acceleration_with_lunar_shadow(
        umbral, SUN_PLUS_X, SRPOptions(k_srp_m2_per_kg=0.01)
    )
    without = srp_acceleration_with_lunar_shadow(
        umbral, SUN_PLUS_X,
        SRPOptions(k_srp_m2_per_kg=0.01, shadow_model="NO_SHADOW"),
    )
    assert np.array_equal(with_shadow, np.zeros(3))
    assert np.linalg.norm(without) > 0.0


# ======================================================================
# Group E - Phase 13 oracle parity
# ======================================================================
@requires_oracle
def test_oracle_constants_match():
    oracle = _load_phase13_oracle()
    assert SOLAR_PRESSURE_1AU_N_M2 == pytest.approx(oracle.P_SRP_1AU_N_M2, rel=1e-15)
    assert AU_M == oracle.AU_M
    assert R_SUN_M == oracle.R_SUN_M
    assert R_MOON_M == pytest.approx(oracle.R_MOON_M, rel=1e-15)


@requires_oracle
@pytest.mark.parametrize("k", [0.0, 1e-6, PHASE13_PARAMETRIC_K_SRP, 0.00864, 0.048359, 0.5])
def test_acceleration_parity_full_light(k):
    oracle = _load_phase13_oracle()
    mine = srp_acceleration(R_SC, R_SUN, k)
    theirs = oracle.srp_acceleration(R_SC, R_SUN, 1.0, k, 1.0)
    abs_err = float(np.linalg.norm(mine - theirs))
    rel_err = abs_err / max(float(np.linalg.norm(theirs)), 1e-300)
    assert abs_err <= 1e-24
    assert rel_err <= 1e-14


@requires_oracle
@pytest.mark.parametrize("d_au", [0.95, 0.9898, 1.0, 1.05])
def test_acceleration_parity_over_sun_distance(d_au):
    oracle = _load_phase13_oracle()
    r_sun = np.array([-1.0, 0.1, 0.0])
    r_sun = r_sun / np.linalg.norm(r_sun) * d_au * AU_M
    mine = srp_acceleration(R_SC, r_sun, PHASE13_PARAMETRIC_K_SRP)
    theirs = oracle.srp_acceleration(R_SC, r_sun, 1.0, PHASE13_PARAMETRIC_K_SRP, 1.0)
    assert mine == pytest.approx(theirs, rel=1e-14, abs=1e-24)


@requires_oracle
def test_illumination_parity_across_a_full_shadow_sweep():
    oracle = _load_phase13_oracle()
    worst_nu = 0.0
    regimes_checked = set()
    for y in np.linspace(2.2e6, -2.2e6, 4001):
        r_sc = np.array([-2.0e6, y, 0.0])
        mine, regime = illumination_fraction(r_sc, SUN_PLUS_X, np.zeros(3), R_MOON_M)
        theirs, oracle_regime = oracle.illumination_fraction(
            r_sc, SUN_PLUS_X, np.zeros(3), oracle.R_MOON_M
        )
        worst_nu = max(worst_nu, abs(mine - theirs))
        assert regime == oracle_regime
        regimes_checked.add(regime)
    assert worst_nu <= 1e-15
    assert {"FULL_LIGHT", "PENUMBRA", "UMBRA"} <= regimes_checked


@requires_oracle
def test_shadowed_acceleration_parity():
    oracle = _load_phase13_oracle()
    options = SRPOptions(k_srp_m2_per_kg=PHASE13_PARAMETRIC_K_SRP)
    for y in np.linspace(2.0e6, -2.0e6, 501):
        r_sc = np.array([-2.0e6, y, 0.0])
        mine = srp_acceleration_with_lunar_shadow(r_sc, SUN_PLUS_X, options)
        nu, _ = oracle.illumination_fraction(
            r_sc, SUN_PLUS_X, np.zeros(3), oracle.R_MOON_M
        )
        theirs = oracle.srp_acceleration(
            r_sc, SUN_PLUS_X, 1.0, PHASE13_PARAMETRIC_K_SRP, nu
        )
        assert mine == pytest.approx(theirs, rel=1e-14, abs=1e-24)


@requires_oracle
def test_direction_parity_with_the_sun_at_plus_x():
    oracle = _load_phase13_oracle()
    mine = srp_acceleration(np.zeros(3), SUN_PLUS_X, 0.01)
    theirs = oracle.srp_acceleration(np.zeros(3), SUN_PLUS_X, 1.0, 0.01, 1.0)
    assert np.sign(mine[0]) == np.sign(theirs[0]) == -1.0
    assert mine == pytest.approx(theirs, rel=1e-15, abs=1e-30)


# ======================================================================
# Group F - production integration
# ======================================================================
def _sun_at(_t):
    return R_SUN


def _earth_at(_t):
    return np.array([3.8e8, 0.0, 0.0])


STATE0 = np.array([1.8377e6, 0.0, 0.0, 0.0, 1.6337e3, 0.0])
T_EVAL = np.linspace(0.0, 3600.0, 13)


def test_propagation_with_srp_absent_matches_the_old_call():
    without = propagate_state(
        T_EVAL, STATE0, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
        _earth_at, _sun_at,
    )
    explicit_none = propagate_state(
        T_EVAL, STATE0, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
        _earth_at, _sun_at, srp=None,
    )
    assert np.array_equal(without, explicit_none)


def test_propagation_with_disabled_srp_matches_the_old_call():
    without = propagate_state(
        T_EVAL, STATE0, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
        _earth_at, _sun_at,
    )
    disabled = propagate_state(
        T_EVAL, STATE0, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
        _earth_at, _sun_at, srp=SRPOptions(enabled=False),
    )
    assert np.array_equal(without, disabled)


def test_propagation_with_zero_k_matches_srp_off_to_integrator_precision():
    off = propagate_state(
        T_EVAL, STATE0, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
        _earth_at, _sun_at,
    )
    zero_k = propagate_state(
        T_EVAL, STATE0, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
        _earth_at, _sun_at, srp=SRPOptions(k_srp_m2_per_kg=0.0),
    )
    assert np.max(np.abs(off[:, :3] - zero_k[:, :3])) < 1e-6


def test_propagation_with_srp_moves_the_trajectory():
    off = propagate_state(
        T_EVAL, STATE0, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
        _earth_at, _sun_at,
    )
    on = propagate_state(
        T_EVAL, STATE0, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
        _earth_at, _sun_at,
        srp=SRPOptions(k_srp_m2_per_kg=PHASE13_PARAMETRIC_K_SRP),
    )
    displacement = np.linalg.norm(on[-1, :3] - off[-1, :3])
    assert 1e-3 < displacement < 10.0


def test_propagated_displacement_grows_with_k():
    off = propagate_state(
        T_EVAL, STATE0, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
        _earth_at, _sun_at,
    )
    displacements = []
    for k in (0.005, 0.01, 0.02):
        on = propagate_state(
            T_EVAL, STATE0, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
            _earth_at, _sun_at, srp=SRPOptions(k_srp_m2_per_kg=k),
        )
        displacements.append(np.linalg.norm(on[-1, :3] - off[-1, :3]))
    assert displacements[0] < displacements[1] < displacements[2]
    # Near-linear over this short arc, but not asserted as exact: the perturbed
    # trajectory samples a slightly different gravity field.
    assert displacements[2] / displacements[0] == pytest.approx(4.0, rel=0.05)


def test_f3body_moon_srp_term_is_purely_additive():
    state = np.concatenate([R_SC, [0.0, 1.6e3, 0.0]])
    base = f3body_moon(
        state, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
        _earth_at(0), R_SUN,
    )
    with_srp = f3body_moon(
        state, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
        _earth_at(0), R_SUN, srp=SRPOptions(k_srp_m2_per_kg=0.01),
    )
    expected = srp_acceleration_with_lunar_shadow(
        R_SC, R_SUN, SRPOptions(k_srp_m2_per_kg=0.01)
    )
    assert with_srp[:3] == pytest.approx(base[:3], rel=0, abs=0)
    assert (with_srp[3:] - base[3:]) == pytest.approx(expected, rel=1e-13)


# ======================================================================
# Group G - reference-configuration interoperability
# ======================================================================
requires_config = pytest.mark.skipif(
    not PHASE14_DOCUMENT.exists(),
    reason="Phase 14 configuration artifact is not present in this checkout",
)


@requires_config
def test_the_reference_configuration_still_cannot_supply_a_point_k():
    from lunar_od.reference_config import (
        UnknownParameterValueError,
        load_reference_configuration,
    )

    config = load_reference_configuration(PHASE14_DOCUMENT)
    with pytest.raises(UnknownParameterValueError):
        config.require("srp.K_SRP")


@requires_config
def test_no_production_helper_bypasses_the_unknown_protection():
    """There is no from_reference_configuration constructor, by design."""
    assert not hasattr(SRPOptions, "from_reference_configuration")
    assert not hasattr(SRPOptions, "from_config")


@requires_config
def test_explicit_runtime_k_coexists_with_the_frozen_configuration():
    from lunar_od.reference_config import fingerprint, load_reference_configuration

    config = load_reference_configuration(PHASE14_DOCUMENT)
    before = fingerprint(config)
    options = SRPOptions(
        k_srp_m2_per_kg=PHASE13_PARAMETRIC_K_SRP,
        k_srp_source="CAMPAIGN_PARAMETRIC",
        provenance="Phase 13 parametric screening value; not a spacecraft property",
    )
    assert options.require_k() == PHASE13_PARAMETRIC_K_SRP
    assert fingerprint(config) == before
    assert config.parameter("srp.K_SRP").value is None


@requires_config
def test_the_frozen_configuration_cannot_be_mutated_to_carry_k():
    from lunar_od.reference_config import load_reference_configuration

    config = load_reference_configuration(PHASE14_DOCUMENT)
    # The section mapping is read-only, so a coefficient cannot be inserted.
    with pytest.raises((AttributeError, TypeError)):
        config.sections["srp"]["K_SRP"] = object()
    # And the parameter itself is frozen against ordinary assignment.
    # (object.__setattr__ deliberately bypasses that — it is how frozen
    # dataclasses populate their own fields — so it is not the thing to assert.)
    with pytest.raises((AttributeError, TypeError)):
        config.parameter("srp.K_SRP").value = 0.01
    assert config.parameter("srp.K_SRP").value is None


@requires_config
def test_the_screening_envelope_stays_metadata(caplog):
    """Reading the envelope is explicit and does not configure anything."""
    from lunar_od.reference_config import load_reference_configuration

    config = load_reference_configuration(PHASE14_DOCUMENT)
    envelope = config.interval("K_SRP_body_only_screening_envelope")
    assert envelope.semantics == "SCREENING_ENVELOPE"
    # Using an endpoint is possible, but only by naming it as such.
    options = SRPOptions(
        k_srp_m2_per_kg=envelope.upper,
        k_srp_source="SCREENING_ENVELOPE_ENDPOINT",
    )
    assert options.k_srp_source == "SCREENING_ENVELOPE_ENDPOINT"


# ======================================================================
# Group H - metadata and provenance
# ======================================================================
def test_runtime_k_provenance_is_retained():
    options = SRPOptions(
        k_srp_m2_per_kg=PHASE13_PARAMETRIC_K_SRP,
        k_srp_source="CAMPAIGN_PARAMETRIC",
        provenance="Phase 13 screening value",
    )
    assert options.k_srp_source == "CAMPAIGN_PARAMETRIC"
    assert "Phase 13" in options.provenance


def test_options_are_frozen():
    options = SRPOptions(k_srp_m2_per_kg=0.01)
    with pytest.raises((AttributeError, TypeError)):
        options.k_srp_m2_per_kg = 0.02


def test_default_source_label_says_the_caller_supplied_it():
    assert SRPOptions(k_srp_m2_per_kg=0.01).k_srp_source == "CALLER_SUPPLIED_FIXED"


# ======================================================================
# Group I - default invariance against the actual pre-Phase-16 code
# ======================================================================
PRE_PHASE16_DYNAMICS = Path(
    "C:/Users/erayh/Documents/Python/Grad/od_covariance_campaign/00_recovery/"
    "phase14_base_snapshot/worktree_files/lunar_od/dynamics.py"
)
#: The Phase 13 freeze recorded this hash for lunar_od/dynamics.py, so the
#: snapshot below is provably the module as it stood before SRP existed.
PRE_PHASE16_DYNAMICS_SHA256 = (
    "64411c997e9372237c70009fbce8f50e917476f3581c7e1b020714f5abb67304"
)

requires_pre_phase16 = pytest.mark.skipif(
    not PRE_PHASE16_DYNAMICS.exists(),
    reason="pre-Phase-16 dynamics snapshot is not present in this checkout",
)


def _load_pre_phase16_dynamics():
    import hashlib
    import sys

    raw = PRE_PHASE16_DYNAMICS.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == PRE_PHASE16_DYNAMICS_SHA256, (
        "the snapshot is not the frozen pre-Phase-16 module"
    )
    spec = importlib.util.spec_from_file_location(
        "lunar_od._dynamics_pre_phase16", PRE_PHASE16_DYNAMICS
    )
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "lunar_od"
    sys.modules["lunar_od._dynamics_pre_phase16"] = module
    spec.loader.exec_module(module)
    return module


@requires_pre_phase16
def test_propagation_is_bit_identical_to_the_pre_phase16_module():
    """The strongest available statement of default invariance.

    Not "equivalent to the new code with SRP switched off" — compared against
    the module as it actually was before this phase, verified by the hash the
    Phase 13 freeze recorded for it.
    """
    old = _load_pre_phase16_dynamics()
    args = (T_EVAL, STATE0, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
            _earth_at, _sun_at)
    before = old.propagate_state(*args)
    after = propagate_state(*args)
    assert np.array_equal(before, after)


@requires_pre_phase16
def test_force_evaluation_is_bit_identical_to_the_pre_phase16_module():
    old = _load_pre_phase16_dynamics()
    state = np.concatenate([R_SC, [0.0, 1.6e3, 0.0]])
    args = (state, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
            _earth_at(0), R_SUN)
    assert np.array_equal(
        old.f3body_moon(*args, j2_moon=2.0346e-4),
        f3body_moon(*args, j2_moon=2.0346e-4),
    )


@requires_pre_phase16
def test_augmented_propagation_is_bit_identical_to_the_pre_phase16_module():
    old = _load_pre_phase16_dynamics()
    aug0 = np.concatenate([STATE0, np.eye(6).reshape(-1, order="F")])
    args = (T_EVAL, aug0, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
            _earth_at, _sun_at)
    assert np.array_equal(
        old.propagate_augmented_state(*args),
        __import__("lunar_od.dynamics", fromlist=["x"]).propagate_augmented_state(*args),
    )


@requires_pre_phase16
def test_srp_on_differs_from_the_pre_phase16_module():
    """The control for the three tests above: they would pass trivially if the
    new code simply ignored ``srp``."""
    old = _load_pre_phase16_dynamics()
    args = (T_EVAL, STATE0, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
            _earth_at, _sun_at)
    before = old.propagate_state(*args)
    after = propagate_state(
        *args, srp=SRPOptions(k_srp_m2_per_kg=PHASE13_PARAMETRIC_K_SRP)
    )
    assert not np.array_equal(before, after)
    assert np.linalg.norm(after[-1, :3] - before[-1, :3]) > 1e-3
