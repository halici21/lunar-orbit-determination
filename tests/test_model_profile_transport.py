"""Force and numerical options must reach all nominal/candidate/posterior paths."""
from dataclasses import replace
import numpy as np
import pytest
from lunar_od import estimators as E, scenarios as S, filters as F
from lunar_od.dynamics import PropagationProfile
from lunar_od.srp import SRPOptions
from lunar_od.scenario_config import scenario_config_from_mapping
from tests.test_ukf_force_consistency import fixture
from tests import test_filters as T
from lunar_od import dynamics as D
from lunar_od.constants import MU_MOON_M3S2, J2_MOON_UNNORMALIZED


@pytest.mark.parametrize('measurement',['position','range_rate'])
@pytest.mark.parametrize('solver',['srif','bls_lm'])
def test_nominal_candidate_and_posterior_use_the_same_force_and_tolerances(monkeypatch,measurement,solver):
    t,x,pg,obs,earth,sun,srp=fixture()
    if measurement=='range_rate':
        _,x,t,_,pg,obs,earth,sun=T._synthetic_range_rate_case(120.)
    calls=[]
    for name in ['propagate_state','propagate_augmented_state']:
        original=getattr(E,name)
        def capture(*args,_original=original,**kw):
            calls.append(kw.copy());return _original(*args,**kw)
        monkeypatch.setattr(E,name,capture)
    estimator=getattr(E,f'estimate_{measurement}_{solver}')
    initial=x+np.array([10.,-20.,15.,.01,-.01,.005])
    estimator(t,obs,initial,pg,4.902800066e12,0.,0.,earth,sun,
        max_iter=2,return_posterior=True,srp=srp,propagation_profile=PropagationProfile(),
        rtol=1e-12,atol=1e-13)
    assert len(calls)>=3
    for kw in calls:
        assert kw.get('srp')==srp
        assert kw.get('method')=='DOP853' and kw.get('shadow_events') is True
        assert kw['rtol']==1e-12 and kw['atol']==1e-13


def test_json_round_trip_retains_the_explicit_scientific_profile():
    payload=dict(name='profile',measurement_type='position',estimator_type='srif',
        start_mode='formal',network='multi',srp=dict(k_srp_m2_per_kg=.01),
        propagation_profile=dict(method='DOP853',shadow_events=True))
    config=scenario_config_from_mapping(payload)
    assert config.srp.k_srp_m2_per_kg==.01
    assert config.propagation_profile==PropagationProfile()
    assert scenario_config_from_mapping(config.to_dict())==config


def test_fast_sigma_provider_cannot_silently_override_an_explicit_profile():
    t,x,pg,obs,earth,sun,_=fixture()
    with pytest.raises(ValueError,match='profile|Profile'):
        F.run_lunar_ukf(t,obs,x,np.eye(6),pg,4.902800066e12,0.,0.,earth,sun,
            propagation_profile=PropagationProfile(),fast_sigma_propagator=lambda t0,t1,x:x)


@pytest.mark.parametrize('mode',['hot','formal','sqrt_formal'])
def test_handoffs_keep_the_same_srp_and_profile(monkeypatch,mode):
    _,x,_,_,earth,sun,srp=fixture();calls=[]
    for name in ['propagate_state','propagate_augmented_state']:
        original=getattr(S,name)
        def capture(*args,_original=original,**kw):
            calls.append(kw.copy());return _original(*args,**kw)
        monkeypatch.setattr(S,name,capture)
    args=(86400.,86410.,4.902800066e12,0.,0.,earth,sun)
    kw=dict(rtol=1e-12,atol=1e-13,srp=srp,propagation_profile=PropagationProfile())
    if mode=='hot':S._propagate_hot_start(x,*args,**kw)
    elif mode=='formal':S._propagate_formal_handoff(x,np.eye(6),*args,None,**kw)
    else:S._propagate_sqrt_information_handoff(x,np.eye(6),*args,None,**kw)
    assert calls and all(c.get('srp')==srp and c.get('method')=='DOP853' for c in calls)


def earth(t):return np.tile([384400e3,0.,0.],(np.size(np.asarray(t)),1))
def sun(t):return np.tile([149.6e9,0.,0.],(np.size(np.asarray(t)),1))


def profiled_arcs():
    t,x,pg,_,_,_,srp=fixture()
    times=np.arange(0.,301.,60.)
    h=D.propagate_state(times,x,MU_MOON_M3S2,0.,0.,earth,sun,srp=srp,
        j2_moon=J2_MOON_UNNORMALIZED,rtol=1e-12,atol=1e-13,
        **PropagationProfile().propagator_kwargs())
    arcs=[]
    for n,indices in enumerate([slice(0,3),slice(3,6)],1):
        a=replace(pg,t_s=times[indices])
        obs=T._build_clean_position_observations(a.t_s,h[indices],a)
        arcs.append(S.PreparedArc(n,0,2,a.t_s,h[indices],obs,a))
    return tuple(arcs),srp


@pytest.mark.parametrize('mode',['cold','hot','formal','sqrt_formal'])
def test_two_arc_scenario_uses_the_profile_end_to_end(mode):
    arcs,srp=profiled_arcs()
    result=S.run_batch_arc_sequence(arcs,'position',mode,'srif',MU_MOON_M3S2,0.,0.,earth,sun,
        max_iter=3,srp=srp,propagation_profile=PropagationProfile(),
        j2_moon=J2_MOON_UNNORMALIZED,rtol=1e-12,atol=1e-13)
    assert result.numerical_profile['method']=='DOP853'
    assert result.numerical_profile['error_control']=='fixed_caller_tolerances'
    assert result.derivative_validation_profile=='pending_common_profile_derivative_qualification'
    assert len(result.arc_results)==2
    for arc in result.arc_results:
        assert arc.final_position_error_m<1e-3
        assert np.all(np.isfinite(arc.estimated_state))


def test_profiled_process_workers_match_the_serial_scenario():
    arcs,srp=profiled_arcs()
    args=(arcs,'position','cold','srif',MU_MOON_M3S2,0.,0.,earth,sun)
    kw=dict(max_iter=2,srp=srp,propagation_profile=PropagationProfile(),
        j2_moon=J2_MOON_UNNORMALIZED,rtol=1e-12,atol=1e-13)
    serial=S.run_batch_arc_sequence(*args,**kw)
    process=S.run_batch_arc_sequence(*args,parallel=True,**kw)
    assert serial.numerical_profile==process.numerical_profile
    for a,b in zip(serial.arc_results,process.arc_results):
        np.testing.assert_allclose(a.estimated_state,b.estimated_state,rtol=0,atol=1e-10)


def test_cli_truth_and_estimator_receive_the_same_srp_profile(tmp_path):
    import contextlib,json
    from tests.test_r0b_f1_executed_physics import _RunnerHarness,runner,_config
    from lunar_od.scenario_config import scenario_config_from_mapping
    cfg=_config(srp=dict(k_srp_m2_per_kg=.01),
        propagation_profile=dict(method='DOP853',shadow_events=True))
    with contextlib.ExitStack() as stack:
        h=_RunnerHarness(stack,str(tmp_path))
        result=runner.run_configured_scenario(cfg,manifest_dir=tmp_path)
        for call in [h.truth_spy.call_args,h.batch_spy.call_args]:
            assert call.kwargs['srp'].k_srp_m2_per_kg==.01
            assert call.kwargs['propagation_profile']==PropagationProfile()
        numerical=json.loads((tmp_path/f'{cfg.name}_numerical_profile.json').read_text())
        assert numerical['truth']['method']=='DOP853'
        assert numerical['force_manifest_sha256']==result.force_contract_manifest_sha256


@pytest.mark.parametrize('options',[dict(method='TYPO'),dict(method='ADAMS',shadow_events=True),
    dict(shadow_events=1),dict(ignored_option=True)])
def test_invalid_numerical_combinations_fail_before_execution(options):
    with pytest.raises(ValueError,match='profile|Profile'):
        scenario_config_from_mapping(dict(name='invalid',measurement_type='position',
            estimator_type='srif',start_mode='cold',network='multi',propagation_profile=options))


@pytest.mark.parametrize('solver',['srif','bls_lm'])
@pytest.mark.parametrize('profile',[None,PropagationProfile()])
def test_direct_two_way_profile_none_preserves_the_historical_contract(monkeypatch,solver,profile):
    from tests.test_k_srp_bls_solve_for import _fixture,_MU_MOON
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels()
    f=_fixture(duration_s=120.,step_s=60.)
    class Captured(Exception):pass
    calls=[]
    def capture(*args,**kw):
        calls.append(kw);raise Captured
    monkeypatch.setattr(E,'propagate_augmented_state',capture)
    with pytest.raises(Captured):
        getattr(E,f'estimate_two_way_range_{solver}')(f['t_pass'],f['obs'],f['x_true0'],f['pass_geo'],
            _MU_MOON,0.,0.,f['gep'],f['gsp'],max_iter=1,srp=SRPOptions(k_srp_m2_per_kg=.01),
            propagation_profile=profile)
    assert calls[0].get('method','ADAMS')==('ADAMS' if profile is None else 'DOP853')
    assert calls[0].get('shadow_events',False)==(profile is not None)
