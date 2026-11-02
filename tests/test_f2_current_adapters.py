"""Current F2h/i studies must use the certified solver without losing physics."""
import json
import numpy as np
import pytest
import phase_f2_lro_od as F2


@pytest.mark.parametrize('entry', ['h', 'i'])
def test_current_studies_preserve_priors_parameters_and_use_trf(tmp_path,monkeypatch,entry):
    calls=[]
    def fit(*args,**kwargs):
        calls.append((args,kwargs))
        return {'est_positions':[[0.,0.,0.]],'status':'MaxIter','postfit_rms_m':1.,
                'doppler_postfit_rms_mmps':1.,'solver':'trf','numerical_diagnostics':{'stationary':False}}
    monkeypatch.setattr(F2,'solve',fit)
    monkeypatch.setattr(F2.F,'REPO',tmp_path)
    monkeypatch.setattr(F2,'SOLVER_BACKEND','trf')
    monkeypatch.setattr(F2,'_workers_arg',lambda:2)
    if entry=='h':F2.main_f2h()
    else:F2.main_f2i_part('s1r')
    args,kw=calls[0]
    assert args[0]==300 and args[2]==F2.F2H_PRIORS
    assert args[9]==4
    assert F2.F2H_PRIORS[13][1]==1e-11 and F2.F2H_PRIORS[14][1]==1e-11
    if entry=='i':
        assert kw['step_limit'] is None and not kw['gn_stationarity']
    assert not F2._certified_stationary({'status':'Converged','solver':'legacy'})


def test_legacy_controls_are_not_silently_ignored_by_trf():
    with pytest.raises(ValueError,match='legacy F2i'):
        F2.solve(2,1.,solver='trf',step_limit=(200.,.2))


def test_irreducible_residual_cannot_trigger_false_convergence():
    target=np.zeros(15);target[0]=1.;target[6]=-1.
    class Problem:
        def residual(self,p,partials=True):
            return (np.r_[target-p,1000.],np.vstack([np.eye(15),np.zeros((1,15))]) if partials else None,
                    np.zeros((2,6)))
    initial=np.zeros(15);initial[6]=.01
    scale=np.array([1e3]*3+[1.]*3+[1e-3]+[1.]*3+[1e-3]+[1e-9]*4)
    result=F2._fit_trf(Problem(),initial,lambda r,h,p:(r,h),scale,15)
    assert result[6]=='STATIONARY'
    np.testing.assert_allclose(result[0][[0,6]],[1.,0.],atol=1e-7)
    assert result[9]['gn_decrement']<=.1
