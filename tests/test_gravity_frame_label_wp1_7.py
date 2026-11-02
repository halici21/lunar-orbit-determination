"""WP1.7 (DYN-12, CFG-07, X-13): gravity-model frame label and stale scenario guards.

Before WP1.7 the loader labelled every model with the generic ``MOON_PA`` alias,
which the STM path refuses (``require_explicit_pa_realization``), so a loaded model
could reach the variational equations only through ``dataclasses.replace``; the
scenario builder did not pass its configured realization either.  The scenario
parser also refused BLS/SRIF with harmonics "because the gradient is not
implemented", although the analytic Pines gradient exists; and the schema advertised
``True`` for two fields the parser always rejects.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lunar_od.dynamics import require_explicit_pa_realization
from lunar_od.gravity_model_loader import LOADER_FRAMES, load_lunar_gravity_model
from lunar_od.scenario_config import (
    ScenarioConfig,
    scenario_config_from_mapping,
    scenario_config_schema,
    scenario_lunar_gravity_model,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "gravity" / "synthetic_norm_sha.tab"
BASE = {"name": "wp17", "measurement_type": "position", "estimator_type": "ukf", "start_mode": "cold",
        "network": "multi"}


@pytest.mark.parametrize("frame", ["MOON_PA_DE421", "MOON_PA_DE440"])
def test_loaded_gravity_model_accepted_by_stm_path(frame):
    model = load_lunar_gravity_model(FIXTURE, nmax=3, frame=frame)
    assert require_explicit_pa_realization(model) == frame


def test_default_label_stays_generic_and_refused():
    model = load_lunar_gravity_model(FIXTURE, nmax=3)
    assert model.frame == "MOON_PA"
    with pytest.raises(ValueError):
        require_explicit_pa_realization(model)
    with pytest.raises(ValueError):
        load_lunar_gravity_model(FIXTURE, nmax=3, frame="MOON_ME")
    assert LOADER_FRAMES == ("MOON_PA", "MOON_PA_DE421", "MOON_PA_DE440")


@pytest.mark.parametrize("frame", ["MOON_PA_DE421", "MOON_PA_DE440"])
def test_scenario_model_carries_the_configured_realization(frame):
    config = ScenarioConfig(**BASE, enable_lunar_harmonics=True, lunar_gravity_model_path=str(FIXTURE),
                            lunar_gravity_nmax=3, lunar_gravity_frame=frame)
    assert require_explicit_pa_realization(scenario_lunar_gravity_model(config)) == frame


@pytest.mark.parametrize("estimator", ["bls_lm", "srif"])
def test_batch_harmonics_refusal_no_longer_claims_a_missing_gradient(estimator):
    payload = {**BASE, "estimator_type": estimator, "enable_lunar_harmonics": True,
               "lunar_gravity_model_path": str(FIXTURE), "lunar_gravity_nmax": 3}
    with pytest.raises(ValueError) as excinfo:
        scenario_config_from_mapping(payload)
    assert "gradient not implemented" not in str(excinfo.value)
    assert "not yet consumed" in str(excinfo.value)


def test_schema_admits_only_the_accepted_value_of_rejected_switches():
    props = scenario_config_schema()["properties"]
    for name in ("enable_lunar_harmonics", "enable_earth_j2"):
        assert props[name]["enum"] == [False]
        assert props[name]["default"] is False
