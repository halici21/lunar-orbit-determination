"""F3 WP3.2 S4b: the opt-in SRP light-time package reaches every propagation of a batch solve (LTB_G1-LTB_G3).

Predeclaration: studies/phase17/configs/wp3_2_s4_srp_light_time_predeclaration.json (cc7a454, errata e7e261d
before any result).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from lunar_od import estimators as E
from lunar_od.constants import J2_MOON_UNNORMALIZED, MU_EARTH_M3S2, MU_MOON_M3S2, MU_SUN_M3S2
from lunar_od.dynamics import PropagationProfile, propagate_state
from lunar_od.earth_shadow import EarthShadowOptions
from lunar_od.lunar_j2_model import use_lunar_j2_model
from lunar_od.lunar_radiation import LunarRadiationOptions
from lunar_od.lunar_tide import LunarTideOptions
from lunar_od.measurements import PassGeometry
from lunar_od.propagation_contract import model_options, propagator_options
from lunar_od.srp import SRPOptions
from lunar_od.srp_light_time import SRPLightTimeOptions, SSBVelocityTable
from tests import test_filters as T
from tests import test_wp3_2_lunar_tide_transport as S1


@pytest.fixture(autouse=True, scope="module")
def _legacy_lunar_j2_model():
    """The synthetic gates run on the legacy mean-pole lunar J2 (predeclaration ``lunar_j2``), as S1-S3."""
    with use_lunar_j2_model("iau_mean_pole_legacy"):
        yield


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "studies" / "phase17" / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "studies" / "phase17" / "scripts"))

import wp3_2_s3_geometry as GEO  # noqa: E402
import wp3_2_s4_geometry as GLT  # noqa: E402

LT = GLT.synthetic_options()
ES = EarthShadowOptions()
LRP = LunarRadiationOptions()
TIDE = LunarTideOptions(static_field_convention="TIDE_FREE")
PEN = SRPOptions(k_srp_m2_per_kg=0.05)


def _wide_options():
    """A velocity table wide enough for any fixture arc used below (constant SSB velocities)."""
    t = np.array([-1e7, 0.0, 1e7])
    return SRPLightTimeOptions(ssb_velocities=SSBVelocityTable(
        t_s=t, sun_vel_mps=np.tile([0.0, 12.0, 0.0], (3, 1)), moon_vel_mps=np.tile([2.9e4, 3.0e3, 1.0e3], (3, 1))))


# ----------------------------------------------------------------------
# LTB_G1: the options dicts
# ----------------------------------------------------------------------
@pytest.mark.parametrize("es", [None, ES])
@pytest.mark.parametrize("lrp", [None, LRP])
@pytest.mark.parametrize("tide", [None, TIDE])
@pytest.mark.parametrize("srp", [None, SRPOptions(k_srp_m2_per_kg=0.01), SRPOptions(enabled=False)])
@pytest.mark.parametrize("profile", [None, PropagationProfile(), PropagationProfile("ADAMS", False)])
def test_options_without_light_time_are_the_s3_dicts_and_with_it_add_only_it(srp, profile, tide, lrp, es):
    for fn, oracle in ((model_options, S1._historical_model_options),
                       (propagator_options, S1._historical_propagator_options)):
        expected = oracle(srp, profile)
        for key, value in (("lunar_tide", tide), ("lunar_radiation", lrp), ("earth_shadow", es)):
            if value is not None:
                expected = {**expected, key: value}
        assert fn(srp, profile, tide, lrp, es) == expected
        assert fn(srp, profile, tide, lrp, es, None) == expected
        with_lt = fn(srp, profile, tide, lrp, es, LT)
        assert with_lt["srp_light_time"] is LT
        assert {k: v for k, v in with_lt.items() if k != "srp_light_time"} == expected


@pytest.mark.parametrize("bad", [True, "CN+S", {"ssb_velocities": None}])
def test_options_refuse_a_light_time_that_is_not_options(bad):
    for fn in (model_options, propagator_options):
        with pytest.raises(TypeError, match="SRPLightTimeOptions"):
            fn(None, None, None, None, None, bad)


# ----------------------------------------------------------------------
# LTB_G2: every propagation of a solve carries the package
# ----------------------------------------------------------------------
def _check(calls, srp_light_time):
    assert calls
    for name, kw in calls:
        if srp_light_time is None:
            assert "srp_light_time" not in kw, name
        else:
            assert kw.get("srp_light_time") is srp_light_time, name


WIDE = _wide_options()


@pytest.mark.parametrize("srp_light_time", [None, WIDE], ids=["off", "on"])
@pytest.mark.parametrize("profile", [None, PropagationProfile()])
@pytest.mark.parametrize("measurement", ["position", "range_rate"])
@pytest.mark.parametrize("solver", ["srif", "bls_lm"])
def test_position_and_range_rate_solves_forward_the_light_time_everywhere(
        monkeypatch, solver, measurement, profile, srp_light_time):
    from tests.test_ukf_force_consistency import fixture
    t, x, pg, obs, earth, sun, srp = fixture()            # srp: CONICAL_PENUMBRA, K = 0.1 (active)
    if measurement == "range_rate":
        _, x, t, _, pg, obs, earth, sun = T._synthetic_range_rate_case(120.0)
    calls = S1._spy(monkeypatch)
    kw = {} if srp_light_time is None else {"srp_light_time": srp_light_time}
    getattr(E, f"estimate_{measurement}_{solver}")(
        t, obs, x + np.array([10.0, -20.0, 15.0, 0.01, -0.01, 0.005]), pg, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
        earth, sun, max_iter=2, return_posterior=True, srp=srp, propagation_profile=profile,
        rtol=1e-12, atol=1e-13, **kw)
    _check(calls, srp_light_time)


@pytest.mark.parametrize("srp_light_time", [None, WIDE], ids=["off", "on"])
@pytest.mark.parametrize("solve_for_k", [False, True])
@pytest.mark.parametrize("solver", ["srif", "bls_lm"])
def test_two_way_range_solves_forward_the_light_time_everywhere(monkeypatch, solver, solve_for_k, srp_light_time):
    from lunar_od.spice_loader import load_spice_kernels
    from tests.test_k_srp_bls_solve_for import _MU_MOON, _fixture
    load_spice_kernels()
    f = _fixture(duration_s=1800.0, step_s=60.0)
    calls = S1._spy(monkeypatch)
    kw = {} if srp_light_time is None else {"srp_light_time": srp_light_time}
    if solve_for_k:
        kw["k_srp_prior_sigma"] = 0.01
    getattr(E, f"estimate_two_way_range_{solver}")(
        f["t_pass"], f["obs"], f["x_true0"], f["pass_geo"], _MU_MOON, MU_EARTH_M3S2, MU_SUN_M3S2, f["gep"], f["gsp"],
        max_iter=1, return_posterior=True, srp=SRPOptions(k_srp_m2_per_kg=0.01), solve_for_k_srp=solve_for_k,
        propagation_profile=PropagationProfile(), **kw)
    _check(calls, srp_light_time)
    if solve_for_k:
        assert any(name == "propagate_state_with_k_sensitivity" for name, _ in calls)


def test_two_way_solves_refuse_a_light_time_that_is_not_options():
    from lunar_od.spice_loader import load_spice_kernels
    from tests.test_k_srp_bls_solve_for import _MU_MOON, _fixture
    load_spice_kernels()
    f = _fixture(duration_s=1800.0, step_s=60.0)
    for solver in ("srif", "bls_lm"):
        with pytest.raises(TypeError, match="SRPLightTimeOptions"):
            getattr(E, f"estimate_two_way_range_{solver}")(
                f["t_pass"], f["obs"], f["x_true0"], f["pass_geo"], _MU_MOON, MU_EARTH_M3S2, MU_SUN_M3S2, f["gep"],
                f["gsp"], max_iter=1, srp=SRPOptions(k_srp_m2_per_kg=0.01),
                propagation_profile=PropagationProfile(), srp_light_time="CN+S")


# ----------------------------------------------------------------------
# LTB_G3: noiseless recovery with the package, and the bias without it
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def light_time_truth_case():
    return make_light_time_truth_case()


def make_light_time_truth_case():
    t = GEO.T6H
    truth = propagate_state(t, GEO.PLANE, *GEO.BASE_E, j2_moon=J2_MOON_UNNORMALIZED, srp=PEN, srp_light_time=LT,
                            method="DOP853", shadow_events=True, rtol=1e-12, atol=1e-13)
    stations = tuple(T._synthetic_position_station(*q) for q in [(0., 0., 0.), (0., 90., 0.), (45., -30., 500.)])
    pg = PassGeometry(t_s=t, earth_pos_mci_m=np.zeros((t.size, 3)), earth_vel_mci_mps=np.zeros((t.size, 3)),
                      x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t.size, axis=0), stations=stations,
                      measurement_type="position")
    return t, pg, T._build_clean_position_observations(t, truth, pg)


def light_time_recovery(solver, case, srp_light_time):
    t, pg, obs = case
    start = GEO.PLANE + np.array([100.0, -60.0, 80.0, 0.1, -0.06, 0.08])
    kw = {} if srp_light_time is None else {"srp_light_time": srp_light_time}
    x_hat, _status, _stats = getattr(E, f"estimate_position_{solver}")(
        t, obs, start, pg, *GEO.BASE_E, j2_moon=J2_MOON_UNNORMALIZED, srp=PEN,
        propagation_profile=PropagationProfile(), rtol=1e-12, atol=1e-13, **kw)
    return float(np.linalg.norm(x_hat[:3] - GEO.PLANE[:3])), float(np.linalg.norm(x_hat[3:6] - GEO.PLANE[3:6]))


@pytest.mark.parametrize("solver", ["bls_lm", "srif"])
def test_noiseless_light_time_truth_is_recovered_with_it(solver, light_time_truth_case):
    dr, dv = light_time_recovery(solver, light_time_truth_case, LT)
    assert dr < 1e-3, dr
    assert dv < 1e-6, dv


def test_noiseless_light_time_truth_is_biased_without_it(light_time_truth_case):
    dr, _ = light_time_recovery("bls_lm", light_time_truth_case, LT)
    dr_free, _ = light_time_recovery("bls_lm", light_time_truth_case, None)
    assert dr_free >= 100 * dr, (dr_free, dr)
