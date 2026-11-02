"""WP1.5: the force identity covers the executed ephemeris (DYN-10), disabled Earth J2
(DYN-11) and the shadow events that actually run (DYN-16)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from lunar_od.dynamics import PropagationProfile
from lunar_od.ephemeris import MoonCenteredEphemeris, perturb_moon_centered_ephemeris
from lunar_od.force_contract import (
    EPHEMERIS_FORCE_CONTRACT_SCHEMA_VERSION,
    FORCE_CONTRACT_SCHEMA_VERSION,
    NOT_APPLICABLE,
    ForceContractError,
    bind_spec_to_runtime,
    ephemeris_execution_contract,
)
from lunar_od.propagation_contract import numerical_metadata
from lunar_od.scenario_config import force_execution_spec_from_scenario_config, scenario_config_from_mapping
from lunar_od.srp import SRPOptions

_PAYLOAD = {"name": "wp1_5", "measurement_type": "range_rate", "estimator_type": "bls_lm",
            "start_mode": "cold", "network": "multi"}


def _ephemeris(step_s: float = 600.0) -> MoonCenteredEphemeris:
    t = np.arange(0.0, 7200.0 + step_s, step_s)
    phase = 2.6617e-6 * t
    earth = 3.844e8 * np.column_stack([np.cos(phase), np.sin(phase), np.zeros_like(t)])
    earth_vel = 3.844e8 * 2.6617e-6 * np.column_stack([-np.sin(phase), np.cos(phase), np.zeros_like(t)])
    sun = np.tile([1.496e11, 0.0, 0.0], (t.size, 1))
    return MoonCenteredEphemeris(t_ephem_s=t, earth_pos_m=earth, sun_pos_m=sun, earth_vel_mps=earth_vel,
                                 et0_s=3.0e8)


def _spec(ephemeris=None, **overrides):
    execution = None if ephemeris is None else ephemeris_execution_contract(ephemeris, planetary_spk=("de440.bsp",))
    return force_execution_spec_from_scenario_config(
        scenario_config_from_mapping({**_PAYLOAD, **overrides}), ephemeris_execution=execution)


def test_legacy_specs_keep_schema_v1_and_fingerprint():
    spec = _spec()
    assert spec.contract().schema_version == FORCE_CONTRACT_SCHEMA_VERSION
    assert "ephemeris_execution" not in spec.contract().to_canonical_payload()


def test_ephemeris_identity_selects_schema_v3():
    contract = _spec(_ephemeris()).contract()
    assert contract.schema_version == EPHEMERIS_FORCE_CONTRACT_SCHEMA_VERSION
    payload = contract.to_canonical_payload()["ephemeris_execution"]
    assert payload["planetary_spk"] == ["de440.bsp"]
    assert payload["grid_step_s"] == 600.0 and payload["earth_interpolation"] == "hermite"


def test_fingerprint_differs_for_perturbed_ephemeris():
    base = _ephemeris()
    shifted = perturb_moon_centered_ephemeris(base, earth_position_bias_m=(1e-3, 0.0, 0.0))
    assert _spec(base).fingerprint() != _spec(shifted).fingerprint()
    assert _spec(base).fingerprint() == _spec(_ephemeris()).fingerprint()


def test_fingerprint_differs_for_grid_step_and_interpolation():
    base = _ephemeris()
    assert _spec(base).fingerprint() != _spec(_ephemeris(300.0)).fingerprint()
    pchip = dataclasses.replace(base, earth_interpolation="legacy_pchip")
    assert _spec(base).fingerprint() != _spec(pchip).fingerprint()


def test_v3_hashes_disabled_earth_j2_as_not_applicable():
    a = dataclasses.replace(_spec(_ephemeris()), earth_j2_mode="identity_experimental")
    b = dataclasses.replace(_spec(_ephemeris()), earth_j2_mode="something_else")
    assert a.contract().earth_j2.mode == NOT_APPLICABLE
    assert a.fingerprint() == b.fingerprint()


def test_earth_j2_contract_requires_runtime_binding():
    spec = dataclasses.replace(_spec(), enable_earth_j2=True)
    with pytest.raises(ForceContractError, match="Earth J2"):
        bind_spec_to_runtime(spec, spec.propagation_kwargs(gravity_only=True), context="WP1.5")


def test_numerical_metadata_reports_executed_shadow_events():
    profile = PropagationProfile()
    no_shadow = SRPOptions(enabled=True, k_srp_m2_per_kg=0.01, shadow_model="NO_SHADOW")
    meta = numerical_metadata(no_shadow, profile, 1e-11, 1e-12)
    assert meta["requested_shadow_events"] is True and meta["shadow_events"] is False
    conical = SRPOptions(enabled=True, k_srp_m2_per_kg=0.01, shadow_model="CONICAL_PENUMBRA")
    assert numerical_metadata(conical, profile, 1e-11, 1e-12)["shadow_events"] is True
