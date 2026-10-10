"""Phase 15 - reference-configuration infrastructure.

The tests are built around a synthetic document rather than the Phase 14
artifact, so that they exercise the schema rather than one frozen spacecraft.
The real Phase 14 document is loaded by a separate group that skips when the
campaign directory is not present.

The recurring subject is what the loader refuses to do: replace an unknown with
a number, average two disagreeing publications, promote a screening range to a
bound, or hash a runtime timestamp into a scientific fingerprint.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from lunar_od.reference_config import (
    DEFAULT_REFERENCE_CONFIGURATION,
    Derivation,
    Interval,
    ParameterValue,
    ReferenceConfigurationError,
    ReferenceConfigurationRegistry,
    SourceDiscrepancy,
    UnknownParameterStatusError,
    UnknownParameterValueError,
    UnsupportedConfigurationSchemaError,
    canonical_json,
    configuration_snapshot,
    fingerprint,
    legacy_phase14_fingerprint,
    parse_reference_configuration,
    run_checks,
    scientific_projection,
    summarize_reference_configuration,
    to_document,
    verify_fingerprint,
)

PHASE14_DOCUMENT = Path(
    "C:/Users/erayh/Documents/Python/Grad/od_covariance_campaign/03_data/"
    "phase14_reference_spacecraft_configuration_v1.json"
)
PHASE14_ANCESTOR_FINGERPRINT = (
    "775e4231bb556819067d60d289c0737a66ae0e6bd562e0e817b55c7f526368df"
)

_TURNAROUND = 880.0 / 749.0
_UPLINK_HZ = 7.2e9
_DOWNLINK_HZ = _UPLINK_HZ * _TURNAROUND


def _synthetic_document() -> dict:
    """A minimal generation-1 document exercising every semantic feature."""
    return {
        "schema_version": 1,
        "identity": {
            "configuration_id": {
                "value": "TEST-REF-v1", "unit": "-",
                "status": "OWNER_FROZEN_REFERENCE", "source_id": "TEST_SELF",
            },
            "configuration_class": {
                "value": "THESIS_REFERENCE_SPACECRAFT", "unit": "-",
                "status": "OWNER_FROZEN_REFERENCE", "source_id": "TEST_SELF",
            },
        },
        "dynamics": {
            "mass_kg": {
                "value": 200.0, "unit": "kg",
                "status": "PUBLIC_SPACECRAFT_SOURCED", "source_id": "TEST_PRIMARY",
            },
            "mass_alternative_kg": {
                "value": 250.0, "unit": "kg",
                "status": "PUBLIC_SPACECRAFT_SOURCED",
                "source_id": "TEST_SECONDARY",
            },
            "C_R": {
                "value": None, "unit": "-", "status": "UNKNOWN",
                "unknown_reason": "NOT_PUBLIC",
                "blocks": ["a point K_SRP"],
            },
        },
        "srp": {
            "K_SRP": {
                "value": None, "unit": "m^2/kg", "status": "UNKNOWN",
                "unknown_reason": "BLOCKED_BY_DEPENDENCY",
                "blocks": ["a point SRP magnitude"],
            },
        },
        "rf": {
            "uplink_frequency_hz": {
                "value": _UPLINK_HZ, "unit": "Hz",
                "status": "OWNER_FROZEN_REFERENCE", "source_id": "TEST_RF",
            },
            "downlink_frequency_hz": {
                "value": _DOWNLINK_HZ, "unit": "Hz",
                "status": "DERIVED_FROM_SOURCED_VALUES", "source_id": "TEST_RF",
                "derivation": "uplink_frequency_hz * turnaround_ratio",
            },
            "turnaround_ratio": {
                "value": _TURNAROUND, "unit": "-",
                "status": "PUBLIC_RF_HARDWARE_SOURCED", "source_id": "TEST_RF",
            },
            "transponder_rx_band_hz": {
                "value": [7.145e9, 7.235e9], "unit": "Hz",
                "status": "PUBLIC_RF_HARDWARE_SOURCED", "source_id": "TEST_RF",
            },
            "transponder_tx_band_hz": {
                "value": [8.4e9, 8.5e9], "unit": "Hz",
                "status": "PUBLIC_RF_HARDWARE_SOURCED", "source_id": "TEST_RF",
            },
            "spacecraft_tx_power_w": {
                "value": 3.8, "unit": "W",
                "status": "PUBLIC_RF_HARDWARE_SOURCED", "source_id": "TEST_RF",
            },
            "spacecraft_tx_power_dbw": {
                "value": 10.0 * math.log10(3.8), "unit": "dBW",
                "status": "DERIVED_FROM_SOURCED_VALUES", "source_id": "TEST_RF",
                "derivation": "10*log10(P_W)",
            },
            "spacecraft_EIRP_dbw": {
                "value": None, "unit": "dBW", "status": "UNKNOWN",
                "unknown_reason": "BLOCKED_BY_DEPENDENCY",
                "blocks": ["downlink C/N0", "any thermal Doppler sigma"],
            },
        },
        "navigation": {
            "sigma_doppler_mps": {
                "value": 2.120e-6, "unit": "m/s",
                "status": "OWNER_FROZEN_REFERENCE", "source_id": "TEST_NAV",
                "semantics": "GROUND_FREQUENCY_STANDARD_COMPONENT",
            },
        },
        "intervals": {
            "K_SRP_body_only_screening_envelope": {
                "lower": 0.00864, "upper": 0.048359, "unit": "m^2/kg",
                "semantics": "SCREENING_ENVELOPE",
                "justification": "body-only geometry; arrays excluded",
                "source_ids": ["TEST_PRIMARY"],
                "excludes": ["solar array area"],
            },
        },
        "discrepancies": [
            {
                "parameter": "dynamics.mass_kg", "unit": "kg",
                "canonical_value": 200.0, "canonical_source_id": "TEST_PRIMARY",
                "conflicting_value": 250.0,
                "conflicting_source_id": "TEST_SECONDARY",
                "reason_for_canonical_selection": "primary source",
            },
        ],
        "provenance": {
            "sources": [
                {"source_id": "TEST_SELF", "title": "self", "organization": "t"},
                {"source_id": "TEST_PRIMARY", "title": "primary",
                 "organization": "t"},
                {"source_id": "TEST_SECONDARY", "title": "secondary",
                 "organization": "t"},
                {"source_id": "TEST_RF", "title": "radio", "organization": "t"},
                {"source_id": "TEST_NAV", "title": "nav", "organization": "t"},
            ],
        },
        "limitations": {"class": "synthetic test fixture, not a spacecraft"},
    }


@pytest.fixture()
def document() -> dict:
    return _synthetic_document()


@pytest.fixture()
def config(document):
    return parse_reference_configuration(document)


# ======================================================================
# loading
# ======================================================================
def test_valid_document_loads(config):
    assert config.configuration_id == "TEST-REF-v1"
    assert config.schema_version == 1
    assert config.parameter_count == 15


def test_malformed_document_is_rejected():
    with pytest.raises(ReferenceConfigurationError):
        parse_reference_configuration({"identity": {}})


def test_bare_number_parameter_is_rejected(document):
    document["dynamics"]["mass_kg"] = 200.0
    with pytest.raises(ReferenceConfigurationError, match="carries no provenance"):
        parse_reference_configuration(document)


def test_unsupported_schema_version_is_rejected(document):
    document["schema_version"] = 99
    with pytest.raises(UnsupportedConfigurationSchemaError, match="not supported"):
        parse_reference_configuration(document)


def test_non_integer_schema_version_is_rejected(document):
    document["schema_version"] = "1"
    with pytest.raises(UnsupportedConfigurationSchemaError):
        parse_reference_configuration(document)


def test_missing_schema_version_means_generation_zero_not_current(document):
    """Absence is a named generation, not a default of the current one."""
    del document["schema_version"]
    config = parse_reference_configuration(document)
    assert config.schema_version == 0
    assert config.schema_version != 1


def test_generation_zero_unknown_outside_the_overlay_is_rejected(document):
    """A generation-0 document cannot introduce an unknown the overlay never saw.

    Generation 0 keeps its unknown reasons in prose, so the only ones loadable
    are the ones the overlay transcribed. An unfamiliar unknown must fail rather
    than load with no stated consequence.
    """
    del document["schema_version"]
    document["dynamics"]["some_new_property"] = {
        "value": None, "unit": "kg", "status": "UNKNOWN",
    }
    with pytest.raises(ReferenceConfigurationError, match="generation-0 overlay"):
        parse_reference_configuration(document)


def test_missing_section_is_rejected(document):
    del document["navigation"]
    with pytest.raises(ReferenceConfigurationError, match="missing required section"):
        parse_reference_configuration(document)


# ======================================================================
# UNKNOWN semantics
# ======================================================================
def test_unknown_survives_loading(config):
    p = config.parameter("rf.spacecraft_EIRP_dbw")
    assert p.status == "UNKNOWN"
    assert p.value is None
    assert p.unknown_reason == "BLOCKED_BY_DEPENDENCY"
    assert p.blocks


def test_unknown_is_not_converted_to_zero(config):
    for path in config.unknown_parameters():
        value = config.parameter(path).value
        assert value is None
        assert value != 0
        assert not isinstance(value, float)


def test_reading_an_unknown_raises_rather_than_defaulting(config):
    with pytest.raises(UnknownParameterValueError, match="UNKNOWN"):
        config.require("rf.spacecraft_EIRP_dbw")


def test_unknown_error_names_what_is_blocked(config):
    with pytest.raises(UnknownParameterValueError, match="thermal Doppler sigma"):
        config.require("rf.spacecraft_EIRP_dbw", context="link budget")


def test_unknown_with_a_value_is_rejected():
    with pytest.raises(UnknownParameterStatusError, match="must carry value None"):
        ParameterValue(value=0.0, unit="dBW", status="UNKNOWN",
                       unknown_reason="NOT_PUBLIC", blocks=("x",))


def test_unknown_without_a_stated_consequence_is_rejected():
    with pytest.raises(UnknownParameterStatusError, match="must state what it blocks"):
        ParameterValue(value=None, unit="dBW", status="UNKNOWN",
                       unknown_reason="NOT_PUBLIC")


def test_unknown_with_an_uncontrolled_reason_is_rejected():
    with pytest.raises(UnknownParameterStatusError, match="unknown_reason"):
        ParameterValue(value=None, unit="dBW", status="UNKNOWN",
                       unknown_reason="dunno", blocks=("x",))


def test_document_unknown_without_reason_is_rejected(document):
    del document["rf"]["spacecraft_EIRP_dbw"]["unknown_reason"]
    with pytest.raises(ReferenceConfigurationError, match="unknown_reason"):
        parse_reference_configuration(document)


def test_unknown_point_with_screening_envelope_is_valid(config):
    assert not config.parameter("srp.K_SRP").is_known
    envelope = config.interval("K_SRP_body_only_screening_envelope")
    assert envelope.semantics == "SCREENING_ENVELOPE"


# ======================================================================
# parameter status
# ======================================================================
def test_invalid_status_is_rejected():
    with pytest.raises(UnknownParameterStatusError, match="controlled vocabulary"):
        ParameterValue(value=1.0, unit="kg", status="PROBABLY_FINE",
                       source_id="S")


def test_mission_sourced_is_reserved():
    with pytest.raises(UnknownParameterStatusError, match="reserved"):
        ParameterValue(value=1.0, unit="kg", status="MISSION_SOURCED",
                       source_id="S")


def test_known_parameter_without_a_source_is_rejected():
    with pytest.raises(UnknownParameterStatusError, match="source_id"):
        ParameterValue(value=1.0, unit="kg", status="PUBLIC_SPACECRAFT_SOURCED")


def test_parameter_without_a_unit_is_rejected():
    with pytest.raises(UnknownParameterStatusError, match="unit"):
        ParameterValue(value=1.0, unit="  ", status="PUBLIC_SPACECRAFT_SOURCED",
                       source_id="S")


def test_public_rf_hardware_status_is_available(config):
    assert config.parameter("rf.turnaround_ratio").status == (
        "PUBLIC_RF_HARDWARE_SOURCED"
    )


# ======================================================================
# derivations
# ======================================================================
def test_downlink_is_uplink_times_turnaround(config):
    f_up = config.require("rf.uplink_frequency_hz")
    ratio = config.require("rf.turnaround_ratio")
    f_dn = config.require("rf.downlink_frequency_hz")
    assert f_dn == pytest.approx(f_up * ratio, rel=1e-12)
    assert f_dn == pytest.approx(7.2e9 * 880.0 / 749.0, rel=1e-12)


def test_contradictory_frequency_plan_is_rejected(document):
    document["rf"]["downlink_frequency_hz"]["value"] = 8.3e9
    with pytest.raises(ReferenceConfigurationError):
        parse_reference_configuration(document)


def test_downlink_outside_the_declared_transmit_band_is_rejected(document):
    document["rf"]["transponder_tx_band_hz"]["value"] = [8.6e9, 8.7e9]
    with pytest.raises(ReferenceConfigurationError):
        parse_reference_configuration(document)


def test_dbw_derivation_is_rechecked(document):
    document["rf"]["spacecraft_tx_power_dbw"]["value"] = 12.0
    with pytest.raises(ReferenceConfigurationError, match="10 log10"):
        parse_reference_configuration(document)


def test_derived_parameter_must_carry_a_formula():
    with pytest.raises(ReferenceConfigurationError, match="Derivation"):
        ParameterValue(value=1.0, unit="Hz",
                       status="DERIVED_FROM_SOURCED_VALUES", source_id="S")


def test_derivation_records_its_formula(config):
    d = config.parameter("rf.downlink_frequency_hz").derivation
    assert d is not None and "turnaround" in d.formula


def test_derived_is_distinguishable_from_sourced(config):
    assert config.parameter("rf.downlink_frequency_hz").is_derived
    assert not config.parameter("rf.uplink_frequency_hz").is_derived


# ======================================================================
# provenance
# ======================================================================
def test_dangling_source_id_is_rejected(document):
    document["dynamics"]["mass_kg"]["source_id"] = "NO_SUCH_SOURCE"
    with pytest.raises(ReferenceConfigurationError, match="dangling"):
        parse_reference_configuration(document)


def test_every_known_parameter_cites_a_resolvable_source(config):
    for path, p in config.walk():
        if p.is_known:
            assert p.source_id in config.sources, path


def test_duplicate_source_registration_is_rejected(document):
    document["provenance"]["sources"].append(
        {"source_id": "TEST_RF", "title": "dup", "organization": "t"}
    )
    with pytest.raises(ReferenceConfigurationError, match="more than once"):
        parse_reference_configuration(document)


def test_units_must_match_the_name_suffix(document):
    document["rf"]["spacecraft_tx_power_w"]["unit"] = "dBW"
    with pytest.raises(ReferenceConfigurationError, match="unit"):
        parse_reference_configuration(document)


# ======================================================================
# interval semantics
# ======================================================================
def test_screening_envelope_must_name_what_it_excludes():
    with pytest.raises(ReferenceConfigurationError, match="excludes"):
        Interval(lower=0.0, upper=1.0, unit="m^2/kg",
                 semantics="SCREENING_ENVELOPE", justification="x")


def test_physical_bound_cannot_exclude_a_contributor():
    with pytest.raises(ReferenceConfigurationError, match="cannot exclude"):
        Interval(lower=1.0, upper=2.0, unit="-", semantics="PHYSICAL_BOUND",
                 justification="x", excludes=("arrays",))


def test_interval_semantics_must_come_from_the_vocabulary():
    with pytest.raises(ReferenceConfigurationError, match="semantics"):
        Interval(lower=0.0, upper=1.0, unit="-", semantics="ROUGHLY",
                 justification="x")


def test_unordered_interval_is_rejected():
    with pytest.raises(ReferenceConfigurationError, match="not ordered"):
        Interval(lower=2.0, upper=1.0, unit="-", semantics="PHYSICAL_BOUND",
                 justification="x")


def test_screening_envelope_is_not_relabelled_as_a_bound(document):
    document["intervals"]["K_SRP_body_only_screening_envelope"]["semantics"] = (
        "PHYSICAL_BOUND"
    )
    with pytest.raises(ReferenceConfigurationError):
        parse_reference_configuration(document)


def test_interval_position_helper(config):
    envelope = config.interval("K_SRP_body_only_screening_envelope")
    assert envelope.contains(0.01)
    assert 0.0 < envelope.position_of(0.01) < 0.10


# ======================================================================
# source discrepancy
# ======================================================================
def test_both_disagreeing_masses_survive(config):
    d = config.discrepancy("dynamics.mass_kg")
    assert d is not None
    assert d.canonical_value == 200.0
    assert d.conflicting_value == 250.0
    assert config.require("dynamics.mass_kg") == 200.0
    assert config.require("dynamics.mass_alternative_kg") == 250.0


def test_no_averaging_occurs(config):
    d = config.discrepancy("dynamics.mass_kg")
    assert d.midpoint == 225.0
    assert d.canonical_value != d.midpoint
    for _, p in config.walk():
        assert p.value != 225.0


def test_a_computed_canonical_value_is_rejected(document):
    """The averaging signature, caught structurally rather than arithmetically.

    A mean is reported by no source, so it can only enter as a derived value.
    Comparing a stored canonical against its own midpoint would be vacuous —
    that equality is impossible once the two values differ — so the invariant
    enforced instead is that a canonical must carry a reporting source's status.
    """
    document["discrepancies"][0]["canonical_value"] = 225.0
    document["discrepancies"][0]["canonical_status"] = "DERIVED_FROM_SOURCED_VALUES"
    document["dynamics"]["mass_kg"]["value"] = 225.0
    with pytest.raises(ReferenceConfigurationError, match="averaging"):
        parse_reference_configuration(document)


def test_canonical_value_must_match_its_own_parameter(document):
    document["discrepancies"][0]["canonical_value"] = 199.0
    with pytest.raises(ReferenceConfigurationError):
        parse_reference_configuration(document)


def test_discrepancy_requires_a_reason():
    with pytest.raises(ReferenceConfigurationError, match="reason"):
        SourceDiscrepancy(parameter="p", unit="kg", canonical_value=1.0,
                          canonical_source_id="A", conflicting_value=2.0,
                          conflicting_source_id="B",
                          reason_for_canonical_selection="")


def test_agreeing_sources_are_not_a_discrepancy():
    with pytest.raises(ReferenceConfigurationError, match="not a discrepancy"):
        SourceDiscrepancy(parameter="p", unit="kg", canonical_value=1.0,
                          canonical_source_id="A", conflicting_value=1.0,
                          conflicting_source_id="B",
                          reason_for_canonical_selection="r")


# ======================================================================
# sigma semantics
# ======================================================================
def test_doppler_sigma_declares_it_is_only_one_component(config):
    p = config.parameter("navigation.sigma_doppler_mps")
    assert p.semantics == "GROUND_FREQUENCY_STANDARD_COMPONENT"
    assert p.semantics != "TOTAL_DOPPLER_NOISE"


def test_sigma_without_declared_semantics_is_rejected(document):
    del document["navigation"]["sigma_doppler_mps"]["semantics"]
    with pytest.raises(ReferenceConfigurationError, match="semantics"):
        parse_reference_configuration(document)


def test_sigma_cannot_claim_total_while_a_component_is_blocked(document):
    document["navigation"]["sigma_doppler_mps"]["semantics"] = (
        "TOTAL_DOPPLER_NOISE"
    )
    with pytest.raises(ReferenceConfigurationError, match="overclaimed|total"):
        parse_reference_configuration(document)


# ======================================================================
# fingerprinting
# ======================================================================
def test_fingerprint_is_deterministic(document):
    a = fingerprint(parse_reference_configuration(copy.deepcopy(document)))
    b = fingerprint(parse_reference_configuration(copy.deepcopy(document)))
    assert a == b and len(a) == 64


def test_fingerprint_is_invariant_to_key_order(document):
    shuffled = json.loads(json.dumps(document))
    shuffled["rf"] = dict(reversed(list(shuffled["rf"].items())))
    shuffled = dict(reversed(list(shuffled.items())))
    assert (fingerprint(parse_reference_configuration(shuffled))
            == fingerprint(parse_reference_configuration(document)))


def test_fingerprint_is_invariant_to_whitespace(document):
    compact = json.loads(json.dumps(document, separators=(",", ":")))
    pretty = json.loads(json.dumps(document, indent=4))
    assert (fingerprint(parse_reference_configuration(compact))
            == fingerprint(parse_reference_configuration(pretty)))


def test_fingerprint_is_invariant_to_runtime_metadata(document):
    base = fingerprint(parse_reference_configuration(copy.deepcopy(document)))
    noisy = copy.deepcopy(document)
    noisy["generated_utc"] = "2026-09-05T12:00:00Z"
    noisy["runtime_seconds"] = 91.32
    noisy["hostname"] = "some-machine"
    noisy["source_path"] = "C:/somewhere/else.json"
    assert fingerprint(parse_reference_configuration(noisy)) == base


def test_fingerprint_changes_when_a_scientific_value_changes(document):
    base = fingerprint(parse_reference_configuration(copy.deepcopy(document)))
    changed = copy.deepcopy(document)
    changed["dynamics"]["mass_kg"]["value"] = 201.0
    changed["discrepancies"][0]["canonical_value"] = 201.0
    assert fingerprint(parse_reference_configuration(changed)) != base


def test_fingerprint_changes_when_a_status_changes(document):
    base = fingerprint(parse_reference_configuration(copy.deepcopy(document)))
    changed = copy.deepcopy(document)
    changed["dynamics"]["mass_alternative_kg"]["status"] = "PARAMETRIC"
    assert fingerprint(parse_reference_configuration(changed)) != base


def test_fingerprint_changes_when_interval_semantics_change(document):
    base = fingerprint(parse_reference_configuration(copy.deepcopy(document)))
    changed = copy.deepcopy(document)
    changed["intervals"]["K_SRP_body_only_screening_envelope"]["excludes"] = [
        "solar array area", "something else"
    ]
    assert fingerprint(parse_reference_configuration(changed)) != base


def test_scientific_projection_excludes_the_origin(config):
    projection = scientific_projection(config)
    blob = canonical_json(projection)
    assert "file_name" not in blob
    assert "generated_utc" not in blob


def test_verify_fingerprint_raises_on_mismatch(config):
    with pytest.raises(ReferenceConfigurationError, match="fingerprints as"):
        verify_fingerprint(config, "0" * 64)


def test_round_trip_through_a_document_preserves_the_fingerprint(config):
    reloaded = parse_reference_configuration(to_document(config))
    assert fingerprint(reloaded) == fingerprint(config)
    assert reloaded.unknown_parameters() == config.unknown_parameters()


def test_round_trip_preserves_unknown_reasons(config):
    reloaded = parse_reference_configuration(to_document(config))
    for path in config.unknown_parameters():
        assert (reloaded.parameter(path).unknown_reason
                == config.parameter(path).unknown_reason)
        assert reloaded.parameter(path).blocks == config.parameter(path).blocks


# ======================================================================
# registry and opt-in
# ======================================================================
def test_there_is_no_default_configuration():
    assert DEFAULT_REFERENCE_CONFIGURATION is None


def test_a_fresh_registry_is_empty():
    assert ReferenceConfigurationRegistry().ids() == ()


def test_unregistered_id_raises_and_explains(tmp_path):
    registry = ReferenceConfigurationRegistry()
    with pytest.raises(ReferenceConfigurationError, match="opt-in"):
        registry.get("LTB-IRIS-DSN34X-v1")


def test_registration_reads_identity_from_the_document(tmp_path, document):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    registry = ReferenceConfigurationRegistry()
    entry = registry.register_file(path)
    assert entry.configuration_id == "TEST-REF-v1"
    assert entry.configuration_class == "THESIS_REFERENCE_SPACECRAFT"
    assert registry.ids() == ("TEST-REF-v1",)
    assert registry.get("TEST-REF-v1").configuration_id == "TEST-REF-v1"


def test_registry_refuses_to_shadow_an_id(tmp_path, document):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(document), encoding="utf-8")
    b.write_text(json.dumps(document), encoding="utf-8")
    registry = ReferenceConfigurationRegistry()
    registry.register_file(a)
    with pytest.raises(ReferenceConfigurationError, match="refusing to shadow"):
        registry.register_file(b)


def test_registry_entry_exposes_class_without_implying_truth(tmp_path, document):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    registry = ReferenceConfigurationRegistry()
    entry = registry.register_file(path)
    assert entry.configuration_class != "MISSION_CONFIGURATION"


def test_mission_configuration_class_is_rejected_by_validation(document):
    document["identity"]["configuration_class"]["value"] = "MISSION_CONFIGURATION"
    with pytest.raises(ReferenceConfigurationError):
        parse_reference_configuration(document)


# ======================================================================
# production invariance
# ======================================================================
def test_importing_the_package_reads_no_files_and_registers_nothing():
    import lunar_od.reference_config as rc

    assert rc.registered_ids() == ()
    assert rc.DEFAULT_REFERENCE_CONFIGURATION is None


def test_production_modules_do_not_import_reference_config():
    """No production module may pull the configuration layer in implicitly."""
    import importlib
    import sys

    for name in ("lunar_od.dynamics", "lunar_od.force_models",
                 "lunar_od.estimators", "lunar_od.filters",
                 "lunar_od.measurements", "lunar_od.scenarios"):
        module = importlib.import_module(name)
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "reference_config" not in source, name
    assert "lunar_od.reference_config" in sys.modules or True


# ======================================================================
# reporting helpers
# ======================================================================
def test_snapshot_carries_id_fingerprint_and_schema(config):
    snap = configuration_snapshot(config)
    assert snap["REFERENCE_CONFIGURATION_ID"] == "TEST-REF-v1"
    assert snap["REFERENCE_CONFIGURATION_FINGERPRINT"] == fingerprint(config)
    assert snap["CONFIGURATION_SCHEMA_VERSION"] == 1


def test_snapshot_contains_no_runtime_metadata(config):
    blob = json.dumps(configuration_snapshot(config))
    for volatile in ("utc", "timestamp", "elapsed", "hostname", "path"):
        assert volatile not in blob.casefold()


def test_summary_distinguishes_the_categories(config):
    text = summarize_reference_configuration(config)
    assert "UNKNOWN" in text
    assert "SCREENING_ENVELOPE" in text
    assert "never averaged" in text
    assert "225.0" in text and "the average" in text


# ======================================================================
# the real Phase 14 document
# ======================================================================
requires_phase14 = pytest.mark.skipif(
    not PHASE14_DOCUMENT.exists(),
    reason="Phase 14 campaign artifact is not present in this checkout",
)


@pytest.fixture()
def phase14_config():
    from lunar_od.reference_config import load_reference_configuration

    return load_reference_configuration(
        PHASE14_DOCUMENT,
        expected_ancestor_fingerprint=PHASE14_ANCESTOR_FINGERPRINT,
    )


@requires_phase14
def test_phase14_document_loads_and_validates(phase14_config):
    assert phase14_config.configuration_id == "LTB-IRIS-DSN34X-v1"
    assert phase14_config.configuration_class == "PUBLIC_SPACECRAFT_REFERENCE"
    assert phase14_config.schema_version == 0
    assert phase14_config.parameter_count == 69
    assert len(phase14_config.unknown_parameters()) == 13


@requires_phase14
def test_phase14_ancestor_fingerprint_is_reproduced():
    document = json.loads(PHASE14_DOCUMENT.read_text(encoding="utf-8"))
    assert legacy_phase14_fingerprint(document) == PHASE14_ANCESTOR_FINGERPRINT


@requires_phase14
def test_phase14_wrong_ancestor_fingerprint_is_rejected():
    from lunar_od.reference_config import load_reference_configuration

    with pytest.raises(ReferenceConfigurationError, match="not the document"):
        load_reference_configuration(
            PHASE14_DOCUMENT, expected_ancestor_fingerprint="0" * 64
        )


@requires_phase14
def test_phase14_k_srp_point_is_unknown_and_envelope_is_screening(phase14_config):
    assert phase14_config.parameter("srp.K_SRP").status == "UNKNOWN"
    with pytest.raises(UnknownParameterValueError):
        phase14_config.require("srp.K_SRP")
    envelope = phase14_config.interval("K_SRP_body_only_screening_envelope")
    assert envelope.semantics == "SCREENING_ENVELOPE"
    assert "solar array area" in envelope.excludes
    assert envelope.lower == pytest.approx(0.008640, rel=1e-9)
    assert envelope.upper == pytest.approx(0.048358660031063726, rel=1e-9)


@requires_phase14
def test_phase14_c_r_bound_is_physical_but_k_srp_is_not(phase14_config):
    assert phase14_config.interval("C_R_physical_bound").semantics == (
        "PHYSICAL_BOUND"
    )
    assert phase14_config.interval(
        "K_SRP_body_only_screening_envelope"
    ).semantics == "SCREENING_ENVELOPE"


@requires_phase14
def test_phase14_parametric_k_srp_sits_low_in_the_envelope(phase14_config):
    """Phase 13's screening value is inside the envelope, near its floor."""
    envelope = phase14_config.interval("K_SRP_body_only_screening_envelope")
    assert envelope.contains(0.01)
    assert envelope.position_of(0.01) < 0.05


@requires_phase14
def test_phase14_mass_discrepancy_is_preserved(phase14_config):
    assert phase14_config.require("dynamics.mass_kg") == 200.0
    d = phase14_config.discrepancy("dynamics.mass_kg")
    assert d.conflicting_value == 250.0
    assert d.canonical_value != d.midpoint


@requires_phase14
def test_phase14_eirp_and_thermal_chain_remain_unknown(phase14_config):
    for path in ("rf.spacecraft_EIRP_dbw", "rf.spacecraft_tx_antenna_gain_dbi",
                 "rf.spacecraft_g_over_t_db_per_k"):
        assert not phase14_config.is_known(path)
    assert "thermal" in " ".join(
        phase14_config.parameter("rf.spacecraft_EIRP_dbw").blocks
    ).casefold()


@requires_phase14
def test_phase14_frequency_plan_closes(phase14_config):
    f_up = phase14_config.require("rf.uplink_frequency_hz")
    f_dn = phase14_config.require("rf.downlink_frequency_hz")
    ratio = phase14_config.require("rf.turnaround_ratio")
    assert f_dn == pytest.approx(f_up * ratio, rel=1e-12)
    lo, hi = phase14_config.require("rf.transponder_tx_band_hz")
    assert lo <= f_dn <= hi


@requires_phase14
def test_phase14_all_semantic_checks_pass(phase14_config):
    results = run_checks(phase14_config)
    assert results
    assert [r for r in results if not r.passed] == []


@requires_phase14
def test_phase14_round_trips_to_generation_one(phase14_config):
    reloaded = parse_reference_configuration(to_document(phase14_config))
    assert reloaded.schema_version == 1
    assert reloaded.parameter_count == phase14_config.parameter_count
    assert reloaded.unknown_parameters() == phase14_config.unknown_parameters()
    for path, p in phase14_config.walk():
        assert reloaded.parameter(path).value == p.value, path
        assert reloaded.parameter(path).status == p.status, path
