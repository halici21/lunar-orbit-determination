"""The F2 adapter preserves nuisance/empirical columns and historical evidence."""
import numpy as np
import pytest
import phase_f2_lro_od as F2


def test_f2_adapter_preserves_all_fifteen_columns_and_prior_rows():
    target=np.linspace(1.,2.,15);target[6]=-1.
    a=np.diag(np.linspace(1.,2.,15))
    class Problem:
        def residual(self,p,partials=True):
            return a@(target-p),a if partials else None,np.zeros((2,6))
    def augment(r,h,p):
        # A conditional calibration prior on one empirical coefficient.
        return np.r_[r,3*(target[11]-p[11])],None if h is None else np.vstack([h,np.eye(15)[11]*3])
    initial=np.ones(15);initial[6]=.01
    out=F2._fit_trf(Problem(),initial,augment,np.ones(15),15)
    expected=target.copy();expected[6]=0.
    np.testing.assert_allclose(out[0],expected,rtol=0,atol=1e-7)
    assert out[6]=='STATIONARY'
    assert out[9]['rank']==15
    assert out[10]['data_rows']==15 and out[10]['prior_rows']==1
    assert np.asarray(out[10]['residual_jacobian_whitened']).shape==(16,15)


def test_problem_pool_closes_on_solver_error(monkeypatch):
    flags=[]
    class Problem:
        def __init__(self,*a,**k):pass
        def attach_pool(self,workers):flags.append(('attached',workers))
        def close_pool(self):flags.append(('closed',None))
    def fail(*a,**k):raise ValueError('diagnostic')
    monkeypatch.setattr(F2,'Problem',Problem);monkeypatch.setattr(F2,'_solve_problem',fail)
    with pytest.raises(ValueError,match='diagnostic'):F2.solve(2,1.,workers=2)
    assert flags==[('attached',2),('closed',None)]


def test_new_run_cannot_overwrite_historical_record(tmp_path,monkeypatch):
    monkeypatch.setattr(F2.F,'REPO',tmp_path)
    old=tmp_path/'studies/phase17/results/record.json';old.parent.mkdir(parents=True);old.write_text('historical')
    actual=F2._write_result(old,'new')
    assert old.read_text()=='historical'
    assert actual.read_text()=='new'
    assert actual==tmp_path/'outputs/numerical-core/trf/record.json'


def test_historical_or_mixed_parts_cannot_reuse_current_gates():
    with pytest.raises(ValueError, match='do not match'):
        F2._require_current_runs({'S1': {'numerical_identity': {}}, 'S2': {}})
    F2._require_current_runs({'S1': {'numerical_identity': F2.numerical_identity()}})


def test_legacy_stationary_label_cannot_pass_the_new_convergence_gate():
    assert not F2._certified_stationary({'solver': 'legacy', 'status': 'STATIONARY'})
    assert not F2._certified_stationary({'solver': 'trf', 'status': 'STATIONARY'})
    assert F2._certified_stationary({'solver': 'trf', 'status': 'STATIONARY',
                                   'numerical_diagnostics': {'stationary': True}})
