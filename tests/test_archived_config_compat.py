"""Archived desktop runs stay reloadable when ScenarioConfig gains fields (archive written before the field)."""
from __future__ import annotations

from dataclasses import fields

from lunar_od import scenario_config_from_mapping
from lunar_od.scenario_config import SCENARIO_CONFIG_FIELDS_ADDED_AFTER_ARCHIVES, ScenarioConfig
from services.archived_scenario import effective_config_differences

SAVED = {"name": "a", "measurement_type": "position", "estimator_type": "bls_lm", "start_mode": "cold",
         "network": "single"}


def test_archive_predating_added_fields_compares_with_their_original_values():
    current = scenario_config_from_mapping(SAVED).to_dict()
    effective = {k: v for k, v in current.items() if k not in SCENARIO_CONFIG_FIELDS_ADDED_AFTER_ARCHIVES}
    assert effective_config_differences(effective, current) == []


def test_added_field_with_a_non_default_current_value_is_still_a_difference():
    name = next(iter(SCENARIO_CONFIG_FIELDS_ADDED_AFTER_ARCHIVES))
    current = scenario_config_from_mapping(SAVED).to_dict()
    effective = {k: v for k, v in current.items() if k != name}
    current[name] = 12345
    assert effective_config_differences(effective, current) == [name]


def test_other_missing_fields_are_differences():
    current = scenario_config_from_mapping(SAVED).to_dict()
    effective = {k: v for k, v in current.items() if k != "noise"}
    assert effective_config_differences(effective, current) == ["noise"]


def test_registered_values_are_the_dataclass_defaults():
    defaults = {f.name: f.default for f in fields(ScenarioConfig)}
    for name, value in SCENARIO_CONFIG_FIELDS_ADDED_AFTER_ARCHIVES.items():
        assert defaults[name] == value, name
