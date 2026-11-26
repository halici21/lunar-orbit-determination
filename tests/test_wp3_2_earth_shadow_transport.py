"""F3 WP3.2 S3b: the opt-in Earth shadow reaches every propagation of a batch solve (EB_G1-EB_G3).

Predeclaration: studies/phase17/configs/wp3_2_s3_earth_shadow_predeclaration.json (a5954f1, errata before any
result).
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
from tests import test_filters as T
from tests import test_wp3_2_lunar_tide_transport as S1


@pytest.fixture(autouse=True, scope="module")
def _legacy_lunar_j2_model():
    """The synthetic gates run on the legacy mean-pole lunar J2 (predeclaration ``lunar_j2``), as S1/S2."""
    with use_lunar_j2_model("iau_mean_pole_legacy"):
        yield


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "studies" / "phase17" / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "studies" / "phase17" / "scripts"))

import wp3_2_s3_geometry as GEO  # noqa: E402

ES = EarthShadowOptions()
LRP = LunarRadiationOptions()
TIDE = LunarTideOptions(static_field_convention="TIDE_FREE")
NOS = SRPOptions(k_srp_m2_per_kg=0.05, shadow_model="NO_SHADOW")


# ----------------------------------------------------------------------
# EB_G1: the options dicts
# ----------------------------------------------------------------------
@pytest.mark.parametrize("lrp", [None, LRP])
@pytest.mark.parametrize("tide", [None, TIDE])
@pytest.mark.parametrize("srp", [None, SRPOptions(k_srp_m2_per_kg=0.01), SRPOptions(enabled=False)])
@pytest.mark.parametrize("profile", [None, PropagationProfile(), PropagationProfile("ADAMS", False)])
def test_options_without_earth_shadow_are_the_s2_dicts_and_with_it_add_only_it(srp, profile, tide, lrp):
    for fn, oracle in ((model_options, S1._historical_model_options),
                       (propagator_options, S1._historical_propagator_options)):
        expected = oracle(srp, profile)
        if tide is not None:
            expected = {**expected, "lunar_tide": tide}
        if lrp is not None:
            expected = {**expected, "lunar_radiation": lrp}
        assert fn(srp, profile, tide, lrp) == expected
        assert fn(srp, profile, tide, lrp, None) == expected
        with_es = fn(srp, profile, tide, lrp, ES)
        assert with_es["earth_shadow"] is ES
        assert {k: v for k, v in with_es.items() if k != "earth_shadow"} == expected


@pytest.mark.parametrize("bad", [True, "WGS84", {"radius_m": 6378137.0}])
def test_options_refuse_an_earth_shadow_that_is_not_options(bad):
    for fn in (model_options, propagator_options):
        with pytest.raises(TypeError, match="EarthShadowOptions"):
            fn(None, None, None, None, bad)


# ----------------------------------------------------------------------
# EB_G2: every propagation of a solve carries the Earth shadow
# ----------------------------------------------------------------------
def _check(calls, earth_shadow):
    assert calls
    for name, kw in calls:
        if earth_shadow is None:
            assert "earth_shadow" not in kw, name
        else:
            assert kw.get("earth_shadow") is earth_shadow, name


@pytest.mark.parametrize("earth_shadow", [None, ES])
@pytest.mark.parametrize("profile", [None, PropagationProfile()])
@pytest.mark.parametrize("measurement", ["position", "range_rate"])
@pytest.mark.parametrize("solver", ["srif", "bls_lm"])
def test_position_and_range_rate_solves_forward_the_earth_shadow_everywhere(
        monkeypatch, solver, measurement, profile, earth_shadow):
    from tests.test_ukf_force_consistency import fixture
    t, x, pg, obs, earth, sun, srp = fixture()            # srp: CONICAL_PENUMBRA, K = 0.1 (active)
    if measurement == "range_rate":
        _, x, t, _, pg, obs, earth, sun = T._synthetic_range_rate_case(120.0)
    calls = S1._spy(monkeypatch)
    kw = {} if earth_shadow is None else {"earth_shadow": earth_shadow}
    getattr(E, f"estimate_{measurement}_{solver}")(
        t, obs, x + np.array([10.0, -20.0, 15.0, 0.01, -0.01, 0.005]), pg, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
        earth, sun, max_iter=2, return_posterior=True, srp=srp, propagation_profile=profile,
        rtol=1e-12, atol=1e-13, **kw)
    _check(calls, earth_shadow)


@pytest.mark.parametrize("earth_shadow", [None, ES])
@pytest.mark.parametrize("solve_for_k", [False, True])
@pytest.mark.parametrize("solver", ["srif", "bls_lm"])
def test_two_way_range_solves_forward_the_earth_shadow_everywhere(monkeypatch, solver, solve_for_k, earth_shadow):
    from lunar_od.spice_loader import load_spice_kernels
    from tests.test_k_srp_bls_solve_for import _MU_MOON, _fixture
    load_spice_kernels()
    f = _fixture(duration_s=1800.0, step_s=60.0)
    calls = S1._spy(monkeypatch)
    kw = {} if earth_shadow is None else {"earth_shadow": earth_shadow}
    if solve_for_k:
        kw["k_srp_prior_sigma"] = 0.01
    getattr(E, f"estimate_two_way_range_{solver}")(
        f["t_pass"], f["obs"], f["x_true0"], f["pass_geo"], _MU_MOON, MU_EARTH_M3S2, MU_SUN_M3S2, f["gep"], f["gsp"],
        max_iter=1, return_posterior=True, srp=SRPOptions(k_srp_m2_per_kg=0.01), solve_for_k_srp=solve_for_k,
        propagation_profile=PropagationProfile(), **kw)
    _check(calls, earth_shadow)
    if solve_for_k:
        assert any(name == "propagate_state_with_k_sensitivity" for name, _ in calls)


def test_two_way_solves_refuse_an_earth_shadow_that_is_not_options():
    from lunar_od.spice_loader import load_spice_kernels
    from tests.test_k_srp_bls_solve_for import _MU_MOON, _fixture
    load_spice_kernels()
    f = _fixture(duration_s=1800.0, step_s=60.0)
    for solver in ("srif", "bls_lm"):
        with pytest.raises(TypeError, match="EarthShadowOptions"):
            getattr(E, f"estimate_two_way_range_{solver}")(
                f["t_pass"], f["obs"], f["x_true0"], f["pass_geo"], _MU_MOON, MU_EARTH_M3S2, MU_SUN_M3S2, f["gep"],
                f["gsp"], max_iter=1, srp=SRPOptions(k_srp_m2_per_kg=0.01),
                propagation_profile=PropagationProfile(), earth_shadow="WGS84")


# ----------------------------------------------------------------------
# EB_G3: noiseless recovery with the Earth shadow, and the bias without it
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def earth_shadow_truth_case():
    return make_earth_shadow_truth_case()


def make_earth_shadow_truth_case():
    t = GEO.T6H
    truth = propagate_state(t, GEO.TERM, *GEO.BASE_E, j2_moon=J2_MOON_UNNORMALIZED, srp=NOS, earth_shadow=ES,
                            method="DOP853", shadow_events=True, rtol=1e-12, atol=1e-13)
    stations = tuple(T._synthetic_position_station(*q) for q in [(0., 0., 0.), (0., 90., 0.), (45., -30., 500.)])
    pg = PassGeometry(t_s=t, earth_pos_mci_m=np.zeros((t.size, 3)), earth_vel_mci_mps=np.zeros((t.size, 3)),
                      x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t.size, axis=0), stations=stations,
                      measurement_type="position")
    return t, pg, T._build_clean_position_observations(t, truth, pg)


def earth_shadow_recovery(solver, case, earth_shadow):
    t, pg, obs = case
    start = GEO.TERM + np.array([100.0, -60.0, 80.0, 0.1, -0.06, 0.08])
    kw = {} if earth_shadow is None else {"earth_shadow": earth_shadow}
    x_hat, _status, _stats = getattr(E, f"estimate_position_{solver}")(
        t, obs, start, pg, *GEO.BASE_E, j2_moon=J2_MOON_UNNORMALIZED, srp=NOS,
        propagation_profile=PropagationProfile(), rtol=1e-12, atol=1e-13, **kw)
    return float(np.linalg.norm(x_hat[:3] - GEO.TERM[:3])), float(np.linalg.norm(x_hat[3:6] - GEO.TERM[3:6]))


@pytest.mark.parametrize("solver", ["bls_lm", "srif"])
def test_noiseless_earth_shadow_truth_is_recovered_with_it_and_biased_without_it(solver, earth_shadow_truth_case):
    dr, dv = earth_shadow_recovery(solver, earth_shadow_truth_case, ES)
    dr_free, _ = earth_shadow_recovery(solver, earth_shadow_truth_case, None)
    assert dr < 1e-3, dr
    assert dv < 1e-6, dv
    assert dr_free >= 100 * dr, (dr_free, dr)


# ----------------------------------------------------------------------
# E_G9: the protected-file re-freeze is additive
# ----------------------------------------------------------------------
def test_wp3_2_s3_refreeze_is_additive_and_history_is_preserved():
    """dynamics.py and estimators.py moved through a new F3_WP3_2_S3 table resolved first; the 2026-11-21
    merge table and the path-migration history keep every earlier identity."""
    import hashlib
    import json
    if str(ROOT / "examples") not in sys.path:
        sys.path.insert(0, str(ROOT / "examples"))
    import r2_measurement_fidelity_validation as r2
    migration = json.loads((ROOT / "docs" / "development" / "path-migration.json").read_text(encoding="utf-8"))
    entries = {}

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("lunar_od/dynamics.py", "lunar_od/estimators.py") and isinstance(value, dict) \
                        and "original_sha256" in value:
                    entries[key] = value
                walk(value)
    walk(migration)
    for path, old in (("lunar_od/dynamics.py", "bcdd8fa1"), ("lunar_od/estimators.py", "e5884773")):
        previous = r2.F3_WP3_2_MAIN_MERGE_2026_11_21_REFROZEN_CURRENT_TREE_SHA256[path]
        assert previous.startswith(old)
        new = r2.F3_WP3_2_S3_REFROZEN_CURRENT_TREE_SHA256[path]
        assert r2.expected_current_tree_sha256(path) == new != previous
        assert hashlib.sha256((ROOT / r2.current_source_path(path)).read_bytes()).hexdigest() == new
        assert entries[path]["original_sha256"] == new
        assert entries[path]["refrozen_history_sha256"][-1] == previous
