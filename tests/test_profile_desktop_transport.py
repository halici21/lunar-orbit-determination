"""Actual headless worker dispatch and cache identity across model variants."""
import json
from unittest import mock
from tests.test_pytest_desktop_app import (
    _app,_analysis_controller,_spec,_base_params,_patched_worker_env)


def test_desktop_force_variants_have_distinct_truth_and_observation_caches(tmp_path):
    _app();ac=_analysis_controller()
    profile=dict(method='DOP853',shadow_events=True)
    variants=[dict(srp=dict(k_srp_m2_per_kg=k),propagation_profile=profile) for k in [.01,.02,.02]]
    worker=ac._AnalysisWorker(_spec('srif',variants),_base_params(output_dir=str(tmp_path)))
    with _patched_worker_env() as spies:
        with mock.patch('lunar_od.build_measurement_arcs',return_value=(object(),)) as arcs:
            worker._do_run()
        assert arcs.call_count==2
        assert spies['truth'].call_count==2
        assert spies['batch'].call_count==3
        assert [c.kwargs['srp'].k_srp_m2_per_kg for c in spies['batch'].call_args_list]==[.01,.02,.02]
        for c in spies['truth'].call_args_list+spies['batch'].call_args_list:
            assert c.kwargs['propagation_profile'].method=='DOP853'
    profiles=list(tmp_path.glob('*_numerical_profile.json'))
    assert len(profiles)==3
    for p in profiles:
        saved=json.loads(p.read_text())
        assert saved['truth']['method']=='DOP853'
        assert saved['force_manifest_sha256'].startswith('sha256:')


def test_desktop_tolerance_variants_do_not_share_a_numerically_different_truth(tmp_path):
    _app();ac=_analysis_controller()
    profile=dict(method='DOP853',shadow_events=True)
    variants=[dict(rtol=r,propagation_profile=profile) for r in [1e-10,1e-12]]
    worker=ac._AnalysisWorker(_spec('srif',variants),_base_params(output_dir=str(tmp_path)))
    with _patched_worker_env() as spies:
        worker._do_run()
        assert spies['truth'].call_count==2
        assert [c.kwargs['rtol'] for c in spies['truth'].call_args_list]==[1e-10,1e-12]
