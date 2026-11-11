"""F3 WP3.2 S1b: the opt-in lunar tide reaches every propagation of a batch solve (B_G1-B_G3).

Predeclaration: studies/phase17/configs/wp3_2_s1_lunar_tide_predeclaration.json (bdafb53).
"""
from __future__ import annotations

import numpy as np
import pytest

from lunar_od import estimators as E
from lunar_od.constants import J2_MOON_UNNORMALIZED, MU_EARTH_M3S2, MU_MOON_M3S2, MU_SUN_M3S2
from lunar_od.dynamics import PropagationProfile, propagate_state
from lunar_od.lunar_tide import LunarTideOptions
from lunar_od.measurements import PassGeometry
from lunar_od.propagation_contract import model_options, propagator_options, resolve_profile
from lunar_od.srp import SRPOptions
from tests import test_filters as T
from tests import test_wp3_2_lunar_tide as G

TIDE = LunarTideOptions(static_field_convention="TIDE_FREE")
PROPAGATORS = ("propagate_state", "propagate_augmented_state", "propagate_state_with_k_sensitivity")


# ----------------------------------------------------------------------
# B_G1: the options dicts
# ----------------------------------------------------------------------
def _historical_model_options(srp, profile):
    """The pre-WP3.2 body of propagation_contract.model_options, restated as the oracle."""
    resolved = resolve_profile(srp, profile)
    out = {} if srp is None else {"srp": srp}
    if resolved is not None:
        out["propagation_profile"] = resolved
    return out


def _historical_propagator_options(srp, profile):
    out = _historical_model_options(srp, profile)
    resolved = out.pop("propagation_profile", None)
    if resolved is not None:
        out.update(resolved.propagator_kwargs())
    return out


@pytest.mark.parametrize("srp", [None, SRPOptions(k_srp_m2_per_kg=0.01), SRPOptions(enabled=False)])
@pytest.mark.parametrize("profile", [None, PropagationProfile(), PropagationProfile("ADAMS", False)])
def test_options_without_the_tide_are_historical_and_with_it_add_only_the_tide(srp, profile):
    for fn, oracle in ((model_options, _historical_model_options),
                       (propagator_options, _historical_propagator_options)):
        expected = oracle(srp, profile)
        assert fn(srp, profile) == expected
        assert fn(srp, profile, None) == expected
        with_tide = fn(srp, profile, TIDE)
        assert with_tide["lunar_tide"] is TIDE
        assert {k: v for k, v in with_tide.items() if k != "lunar_tide"} == expected


@pytest.mark.parametrize("bad", [True, "TIDE_FREE", {"static_field_convention": "TIDE_FREE"}])
def test_options_refuse_a_tide_that_is_not_options(bad):
    for fn in (model_options, propagator_options):
        with pytest.raises(TypeError, match="LunarTideOptions"):
            fn(None, None, bad)


# ----------------------------------------------------------------------
# B_G2: every propagation of a solve carries the tide
# ----------------------------------------------------------------------
def _spy(monkeypatch):
    calls = []
    for name in PROPAGATORS:
        original = getattr(E, name)

        def capture(*args, _original=original, _name=name, **kw):
            calls.append((_name, kw.copy()))
            return _original(*args, **kw)
        monkeypatch.setattr(E, name, capture)
    return calls


def _check(calls, tide):
    assert calls
    for name, kw in calls:
        if tide is None:
            assert "lunar_tide" not in kw, name
        else:
            assert kw.get("lunar_tide") is tide, name


@pytest.mark.parametrize("tide", [None, TIDE])
@pytest.mark.parametrize("profile", [None, PropagationProfile()])
@pytest.mark.parametrize("measurement", ["position", "range_rate"])
@pytest.mark.parametrize("solver", ["srif", "bls_lm"])
def test_position_and_range_rate_solves_forward_the_tide_everywhere(monkeypatch, solver, measurement, profile, tide):
    from tests.test_ukf_force_consistency import fixture
    t, x, pg, obs, earth, sun, srp = fixture()
    if measurement == "range_rate":
        _, x, t, _, pg, obs, earth, sun = T._synthetic_range_rate_case(120.0)
    calls = _spy(monkeypatch)
    kw = {} if tide is None else {"lunar_tide": tide}
    getattr(E, f"estimate_{measurement}_{solver}")(
        t, obs, x + np.array([10.0, -20.0, 15.0, 0.01, -0.01, 0.005]), pg, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
        earth, sun, max_iter=2, return_posterior=True, srp=srp, propagation_profile=profile,
        rtol=1e-12, atol=1e-13, **kw)
    _check(calls, tide)


@pytest.mark.parametrize("tide", [None, TIDE])
@pytest.mark.parametrize("solve_for_k", [False, True])
@pytest.mark.parametrize("solver", ["srif", "bls_lm"])
def test_two_way_range_solves_forward_the_tide_everywhere(monkeypatch, solver, solve_for_k, tide):
    from lunar_od.spice_loader import load_spice_kernels
    from tests.test_k_srp_bls_solve_for import _MU_MOON, _fixture
    load_spice_kernels()
    f = _fixture(duration_s=1800.0, step_s=60.0)   # 3 epochs leave the two-way posterior rank-deficient
    calls = _spy(monkeypatch)
    kw = {} if tide is None else {"lunar_tide": tide}
    if solve_for_k:
        kw["k_srp_prior_sigma"] = 0.01          # SRIF requires an initial K uncertainty
    getattr(E, f"estimate_two_way_range_{solver}")(
        f["t_pass"], f["obs"], f["x_true0"], f["pass_geo"], _MU_MOON, MU_EARTH_M3S2, MU_SUN_M3S2, f["gep"], f["gsp"],
        max_iter=1, return_posterior=True, srp=SRPOptions(k_srp_m2_per_kg=0.01), solve_for_k_srp=solve_for_k,
        propagation_profile=PropagationProfile(), **kw)
    _check(calls, tide)
    if solve_for_k:
        assert any(name == "propagate_state_with_k_sensitivity" for name, _ in calls)


# ----------------------------------------------------------------------
# B_G3: noiseless recovery with the tide, and the bias without it
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def tide_truth_case():
    t = G.T6H
    truth = propagate_state(t, G.X0, *G.BASE, j2_moon=J2_MOON_UNNORMALIZED, lunar_tide=TIDE,
                            method="DOP853", rtol=1e-12, atol=1e-13)
    stations = tuple(T._synthetic_position_station(*q) for q in [(0., 0., 0.), (0., 90., 0.), (45., -30., 500.)])
    pg = PassGeometry(t_s=t, earth_pos_mci_m=np.zeros((t.size, 3)), earth_vel_mci_mps=np.zeros((t.size, 3)),
                      x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t.size, axis=0), stations=stations,
                      measurement_type="position")
    return t, pg, T._build_clean_position_observations(t, truth, pg)


def _solve(solver, case, tide):
    t, pg, obs = case
    start = G.X0 + np.array([100.0, -60.0, 80.0, 0.1, -0.06, 0.08])
    kw = {} if tide is None else {"lunar_tide": tide}
    x_hat, _status, _stats = getattr(E, f"estimate_position_{solver}")(
        t, obs, start, pg, *G.BASE, j2_moon=J2_MOON_UNNORMALIZED, propagation_profile=PropagationProfile(),
        rtol=1e-12, atol=1e-13, **kw)
    return float(np.linalg.norm(x_hat[:3] - G.X0[:3])), float(np.linalg.norm(x_hat[3:6] - G.X0[3:6]))


@pytest.mark.parametrize("solver", ["bls_lm", "srif"])
def test_noiseless_tide_truth_is_recovered_with_the_tide_and_biased_without_it(solver, tide_truth_case):
    dr, dv = _solve(solver, tide_truth_case, TIDE)
    dr_free, _ = _solve(solver, tide_truth_case, None)
    assert dr < 1e-3, dr
    assert dv < 1e-6, dv
    assert dr_free >= 100 * dr, (dr_free, dr)


def test_wp3_2_s1_refreeze_of_estimators_is_additive_and_history_is_preserved():
    """estimators.py moved through the additive WP3.2 S1 table; WP4.5b's record is untouched.

    WP3.2 S2 re-froze it again; tests/test_wp3_2_lunar_radiation_transport.py checks the current identity.
    """
    import hashlib
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    if str(root / "examples") not in sys.path:
        sys.path.insert(0, str(root / "examples"))
    import r2_measurement_fidelity_validation as r2
    path = "lunar_od/estimators.py"
    assert r2.F4_WP4_5B_REFROZEN_CURRENT_TREE_SHA256[path].startswith("ee18f3c9")
    assert r2.F3_WP3_2_S1_REFROZEN_CURRENT_TREE_SHA256[path].startswith("0ed77589")
    assert r2.expected_current_tree_sha256(path) == r2.F3_WP3_2_S2_REFROZEN_CURRENT_TREE_SHA256[path]
    assert hashlib.sha256((root / r2.current_source_path(path)).read_bytes()).hexdigest() != (
        r2.F3_WP3_2_S1_REFROZEN_CURRENT_TREE_SHA256[path])
