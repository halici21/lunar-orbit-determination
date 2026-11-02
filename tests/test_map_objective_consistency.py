"""An exact nonlinear MAP counterexample for the production acceptance loops.

h(x) = (x[0]^2, x[1:]), y=(4,0,...), prior mean=(1,0,...),
P0=I/2. The positive MAP root solves x^3-3x-1=0, not h(x)=y.
Only geometry and propagation adapters are replaced; production QR/LM,
prior terms, candidate acceptance and termination execute unchanged.
"""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.optimize import brentq
from lunar_od import estimators as E
from lunar_od.radiometrics import RangeRatePhysicsConfig


@pytest.mark.parametrize('measurement',['position','range_rate'])
@pytest.mark.parametrize('solver',['srif','bls_lm'])
@pytest.mark.parametrize('prior_kind',['covariance','sqrt_information'])
def test_candidate_acceptance_and_final_cost_use_the_explicit_map_objective(monkeypatch,measurement,solver,prior_kind):
    m=6 if measurement=='position' else 8
    target=np.zeros(m);target[0]=4.
    mean=np.zeros(6);mean[0]=1.
    def augmented(t,x,*args,**kw):return np.tile(x,(len(t),1))
    def state(t,x,*args,**kw):return np.tile(x,(len(t),1))
    def values(hist):
        x=hist[0,:6];return np.r_[x[0]**2,x[1:],np.zeros(m-6)]
    def jac(hist):
        j=np.zeros((m,6));j[:6,:]=np.eye(6);j[0,0]=2*hist[0,0];return j
    monkeypatch.setattr(E,'propagate_augmented_state',augmented)
    monkeypatch.setattr(E,'propagate_state',state)
    monkeypatch.setattr(E,f'_{measurement}_weight_diagonal',lambda *a:np.ones(m))
    monkeypatch.setattr(E,f'_{measurement}_residual_from_h',lambda obs,h:target-h)
    if measurement=='position':
        monkeypatch.setattr(E,'compute_position_residuals_analytic',lambda hist,*a:(None,values(hist),None))
        monkeypatch.setattr(E,'_position_initial_state_jacobian',lambda obs,hist,*a:jac(hist))
    else:
        monkeypatch.setattr(E,'_range_rate_nominal_and_initial_jacobian',
            lambda t,obs,x,pg,mu,me,ms,ge,gs,hist,*a,**kw:(values(hist),jac(hist)))
        monkeypatch.setattr(E,'compute_range_rate_residuals',lambda hist,*a:(None,values(hist)))
    pg=SimpleNamespace(stations=(),range_rate_physics=RangeRatePhysicsConfig())
    prior={'prior_covariance':.5*np.eye(6)} if prior_kind=='covariance' else {'prior_sqrt_information':np.sqrt(2.)*np.eye(6)}
    fit,status,stats=getattr(E,f'estimate_{measurement}_{solver}')(
        [0.,1.],np.zeros((2,6 if measurement=='position' else 7)),mean,pg,
        0.,0.,0.,lambda t:np.zeros(3),lambda t:np.zeros(3),
        max_iter=40,tol_cost_stability=1e-12,return_posterior=True,**prior)
    expected=brentq(lambda x:x**3-3*x-1,1.,2.)
    assert fit[0]==pytest.approx(expected,abs=2e-6),(fit,status)
    explicit_objective=(4.-fit[0]**2)**2+2.*float((fit-mean)@(fit-mean))
    assert stats.final_cost==pytest.approx(explicit_objective,rel=1e-10,abs=1e-12)
    expected_information=np.diag([4.*fit[0]**2+2.]+[3.]*5)
    np.testing.assert_allclose(stats.posterior_information,expected_information,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(stats.posterior_covariance,np.linalg.inv(expected_information),rtol=1e-12,atol=1e-12)
