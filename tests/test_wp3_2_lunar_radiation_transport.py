"""F3 WP3.2 S2b: the opt-in LRP reaches every propagation of a batch solve (LB_G1-LB_G3).

Predeclaration: studies/phase17/configs/wp3_2_s2_lunar_radiation_predeclaration.json (da5a4a0).
"""
from __future__ import annotations

import numpy as np
import pytest

from lunar_od import estimators as E
from lunar_od.constants import J2_MOON_UNNORMALIZED, MU_EARTH_M3S2, MU_MOON_M3S2, MU_SUN_M3S2
from lunar_od.dynamics import PropagationProfile, propagate_state
from lunar_od.lunar_radiation import LunarRadiationOptions
from lunar_od.lunar_tide import LunarTideOptions
from lunar_od.measurements import PassGeometry
from lunar_od.propagation_contract import model_options, propagator_options
from lunar_od.srp import SRPOptions
from tests import test_filters as T
from tests import test_wp3_2_lunar_tide as G
from tests import test_wp3_2_lunar_tide_transport as S1

LRP = LunarRadiationOptions()
TIDE = LunarTideOptions(static_field_convention="TIDE_FREE")
NOS = SRPOptions(k_srp_m2_per_kg=0.05, shadow_model="NO_SHADOW")


# ----------------------------------------------------------------------
# LB_G1: the options dicts
# ----------------------------------------------------------------------
@pytest.mark.parametrize("tide", [None, TIDE])
@pytest.mark.parametrize("srp", [None, SRPOptions(k_srp_m2_per_kg=0.01), SRPOptions(enabled=False)])
@pytest.mark.parametrize("profile", [None, PropagationProfile(), PropagationProfile("ADAMS", False)])
def test_options_without_lrp_are_the_s1_dicts_and_with_it_add_only_lrp(srp, profile, tide):
    for fn, oracle in ((model_options, S1._historical_model_options),
                       (propagator_options, S1._historical_propagator_options)):
        expected = oracle(srp, profile)
        if tide is not None:
            expected = {**expected, "lunar_tide": tide}
        assert fn(srp, profile, tide) == expected
        assert fn(srp, profile, tide, None) == expected
        with_lrp = fn(srp, profile, tide, LRP)
        assert with_lrp["lunar_radiation"] is LRP
        assert {k: v for k, v in with_lrp.items() if k != "lunar_radiation"} == expected


@pytest.mark.parametrize("bad", [True, "C6", {"albedo": 0.12}])
def test_options_refuse_an_lrp_that_is_not_options(bad):
    for fn in (model_options, propagator_options):
        with pytest.raises(TypeError, match="LunarRadiationOptions"):
            fn(None, None, None, bad)


# ----------------------------------------------------------------------
# LB_G2: every propagation of a solve carries the LRP
# ----------------------------------------------------------------------
def _check(calls, lrp):
    assert calls
    for name, kw in calls:
        if lrp is None:
            assert "lunar_radiation" not in kw, name
        else:
            assert kw.get("lunar_radiation") is lrp, name


@pytest.mark.parametrize("lrp", [None, LRP])
@pytest.mark.parametrize("profile", [None, PropagationProfile()])
@pytest.mark.parametrize("measurement", ["position", "range_rate"])
@pytest.mark.parametrize("solver", ["srif", "bls_lm"])
def test_position_and_range_rate_solves_forward_lrp_everywhere(monkeypatch, solver, measurement, profile, lrp):
    from tests.test_ukf_force_consistency import fixture
    t, x, pg, obs, earth, sun, srp = fixture()            # srp: CONICAL_PENUMBRA, K = 0.1 (active)
    if measurement == "range_rate":
        _, x, t, _, pg, obs, earth, sun = T._synthetic_range_rate_case(120.0)
    calls = S1._spy(monkeypatch)
    kw = {} if lrp is None else {"lunar_radiation": lrp}
    getattr(E, f"estimate_{measurement}_{solver}")(
        t, obs, x + np.array([10.0, -20.0, 15.0, 0.01, -0.01, 0.005]), pg, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
        earth, sun, max_iter=2, return_posterior=True, srp=srp, propagation_profile=profile,
        rtol=1e-12, atol=1e-13, **kw)
    _check(calls, lrp)


@pytest.mark.parametrize("lrp", [None, LRP])
@pytest.mark.parametrize("solve_for_k", [False, True])
@pytest.mark.parametrize("solver", ["srif", "bls_lm"])
def test_two_way_range_solves_forward_lrp_everywhere(monkeypatch, solver, solve_for_k, lrp):
    from lunar_od.spice_loader import load_spice_kernels
    from tests.test_k_srp_bls_solve_for import _MU_MOON, _fixture
    load_spice_kernels()
    f = _fixture(duration_s=1800.0, step_s=60.0)
    calls = S1._spy(monkeypatch)
    kw = {} if lrp is None else {"lunar_radiation": lrp}
    if solve_for_k:
        kw["k_srp_prior_sigma"] = 0.01
    getattr(E, f"estimate_two_way_range_{solver}")(
        f["t_pass"], f["obs"], f["x_true0"], f["pass_geo"], _MU_MOON, MU_EARTH_M3S2, MU_SUN_M3S2, f["gep"], f["gsp"],
        max_iter=1, return_posterior=True, srp=SRPOptions(k_srp_m2_per_kg=0.01), solve_for_k_srp=solve_for_k,
        propagation_profile=PropagationProfile(), **kw)
    _check(calls, lrp)
    if solve_for_k:
        assert any(name == "propagate_state_with_k_sensitivity" for name, _ in calls)


def test_two_way_solves_refuse_an_lrp_that_is_not_options():
    from lunar_od.spice_loader import load_spice_kernels
    from tests.test_k_srp_bls_solve_for import _MU_MOON, _fixture
    load_spice_kernels()
    f = _fixture(duration_s=1800.0, step_s=60.0)
    for solver in ("srif", "bls_lm"):
        with pytest.raises(TypeError, match="LunarRadiationOptions"):
            getattr(E, f"estimate_two_way_range_{solver}")(
                f["t_pass"], f["obs"], f["x_true0"], f["pass_geo"], _MU_MOON, MU_EARTH_M3S2, MU_SUN_M3S2, f["gep"],
                f["gsp"], max_iter=1, srp=SRPOptions(k_srp_m2_per_kg=0.01),
                propagation_profile=PropagationProfile(), lunar_radiation="C6")


# ----------------------------------------------------------------------
# LB_G3: noiseless recovery with the LRP, and the bias without it
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def lrp_truth_case():
    return make_lrp_truth_case()


def make_lrp_truth_case():
    t = G.T6H
    truth = propagate_state(t, G.X0, *G.BASE, j2_moon=J2_MOON_UNNORMALIZED, srp=NOS, lunar_radiation=LRP,
                            method="DOP853", rtol=1e-12, atol=1e-13)
    stations = tuple(T._synthetic_position_station(*q) for q in [(0., 0., 0.), (0., 90., 0.), (45., -30., 500.)])
    pg = PassGeometry(t_s=t, earth_pos_mci_m=np.zeros((t.size, 3)), earth_vel_mci_mps=np.zeros((t.size, 3)),
                      x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t.size, axis=0), stations=stations,
                      measurement_type="position")
    return t, pg, T._build_clean_position_observations(t, truth, pg)


def lrp_recovery(solver, case, lrp):
    t, pg, obs = case
    start = G.X0 + np.array([100.0, -60.0, 80.0, 0.1, -0.06, 0.08])
    kw = {} if lrp is None else {"lunar_radiation": lrp}
    x_hat, _status, _stats = getattr(E, f"estimate_position_{solver}")(
        t, obs, start, pg, *G.BASE, j2_moon=J2_MOON_UNNORMALIZED, srp=NOS, propagation_profile=PropagationProfile(),
        rtol=1e-12, atol=1e-13, **kw)
    return float(np.linalg.norm(x_hat[:3] - G.X0[:3])), float(np.linalg.norm(x_hat[3:6] - G.X0[3:6]))


@pytest.mark.parametrize("solver", ["bls_lm", "srif"])
def test_noiseless_lrp_truth_is_recovered_with_lrp_and_biased_without_it(solver, lrp_truth_case):
    dr, dv = lrp_recovery(solver, lrp_truth_case, LRP)
    dr_free, _ = lrp_recovery(solver, lrp_truth_case, None)
    assert dr < 1e-3, dr
    assert dv < 1e-6, dv
    assert dr_free >= 100 * dr, (dr_free, dr)


def test_wp3_2_s2_refreeze_of_estimators_is_additive_and_history_is_preserved():
    """estimators.py moved only through the additive WP3.2 S2 table; the S1 and WP4.5b records are untouched."""
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
    new = r2.F3_WP3_2_S2_REFROZEN_CURRENT_TREE_SHA256[path]
    assert r2.expected_current_tree_sha256(path) == new
    assert hashlib.sha256((root / r2.current_source_path(path)).read_bytes()).hexdigest() == new
