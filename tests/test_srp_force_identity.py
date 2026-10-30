"""Scientific SRP identity, runtime binding and legacy identity preservation."""
from dataclasses import replace
import pytest
from lunar_od.scenario_config import (scenario_config_from_mapping,
    force_execution_spec_from_scenario_config,force_runtime_kwargs,scenario_force_model_preflight)
from lunar_od.force_contract import (bind_spec_to_runtime,ForceContractError,
    ConsumerRole,ConsumerReadiness,evaluate_force_model_parity)
from lunar_od.srp import SRPOptions


def config(**overrides):
    return scenario_config_from_mapping(dict(name='srp_identity',measurement_type='position',
        estimator_type='srif',start_mode='formal',network='multi',**overrides))


def spec(**options):return force_execution_spec_from_scenario_config(config(**options))


def test_disabled_srp_keeps_the_historical_v1_physics_identity():
    original=spec();disabled=spec(srp=dict(enabled=False))
    assert original.fingerprint()==disabled.fingerprint()
    assert original.contract().schema_version.endswith('.v1')
    assert 'solar_radiation_pressure' not in original.contract().to_canonical_payload()


@pytest.mark.parametrize('changed',[dict(k_srp_m2_per_kg=.02),dict(shadow_model='NO_SHADOW'),
    dict(occulting_body_radius_m=1.8e6)])
def test_every_active_srp_primitive_changes_the_force_identity(changed):
    a=spec(srp=dict(k_srp_m2_per_kg=.01))
    options=dict(k_srp_m2_per_kg=.01)
    options.update(changed)
    b=spec(srp=options)
    assert a.fingerprint()!=b.fingerprint()
    assert a.contract().schema_version.endswith('.v2')


def test_numerical_profile_is_separate_from_force_physics():
    assert spec(srp=dict(k_srp_m2_per_kg=.01)).fingerprint()==spec(srp=dict(k_srp_m2_per_kg=.01),
        propagation_profile=dict(method='DOP853',shadow_events=True)).fingerprint()


def test_srp_binding_rejects_omitted_or_changed_runtime_force():
    s=spec(srp=dict(k_srp_m2_per_kg=.01));runtime=force_runtime_kwargs(s)
    bind_spec_to_runtime(s,runtime,context='test')
    for value in [None,SRPOptions(k_srp_m2_per_kg=.02),SRPOptions(enabled=False)]:
        with pytest.raises(ForceContractError,match='SRP'):
            bind_spec_to_runtime(s,{**runtime,'srp':value},context='test')
    with pytest.raises(ForceContractError,match='SRP'):
        s.propagation_kwargs()


def test_unexecuted_static_srp_constants_cannot_be_reported_as_runtime_physics():
    s=spec(srp=dict(k_srp_m2_per_kg=.01))
    s=replace(s,solar_radiation_pressure=replace(s.solar_radiation_pressure,solar_radius_m=7e8))
    with pytest.raises(ForceContractError,match='constant'):
        force_runtime_kwargs(s)


def test_srp_fast_role_is_rejected_but_the_shared_profile_path_is_allowed():
    c=config(srp=dict(k_srp_m2_per_kg=.01));s=force_execution_spec_from_scenario_config(c)
    with pytest.raises(ForceContractError,match='ukf_fast_sigma'):
        evaluate_force_model_parity(s.contract(),s.contract())
    decision=scenario_force_model_preflight(c)
    assert decision.match and decision.schema_version.endswith('.v2')
    assert decision.manifest['estimator']['consumer_capabilities']['posterior_covariance']=='pending_common_profile_validation'
    with pytest.raises(ForceContractError,match='ESTIMATOR_STATE'):
        evaluate_force_model_parity(s.contract(),s.contract(),estimator_roles=(ConsumerRole.TRUTH_STATE,))


@pytest.mark.parametrize('container',[tuple,iter])
def test_role_iterables_cannot_drop_the_capability_check(container):
    contract=spec().contract()
    blocked=replace(contract,consumer_capabilities={r:ConsumerReadiness.UNSUPPORTED for r in ConsumerRole})
    with pytest.raises(ForceContractError,match='unsupported'):
        evaluate_force_model_parity(contract,blocked,estimator_roles=container([ConsumerRole.ESTIMATOR_STATE]))


def test_active_srp_cannot_be_mislabeled_as_the_old_contract_schema():
    contract=spec(srp=dict(k_srp_m2_per_kg=.01)).contract()
    with pytest.raises(ForceContractError,match='v2'):
        replace(contract,schema_version='r0b.force-model-contract.v1')
