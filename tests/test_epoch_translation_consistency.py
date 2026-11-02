"""A half second remains a half second under a shift of scenario origin."""
import numpy as np
import pytest
from lunar_od.filters import _propagate_local_state_history
from lunar_od.scenarios import _propagate_hot_start
from lunar_od.scenarios import _propagate_formal_handoff, _propagate_sqrt_information_handoff
from lunar_od.dynamics import make_fast_sigma_propagator
from lunar_od.ephemeris import MoonCenteredEphemeris


@pytest.mark.parametrize('origin',[0.,86400.,100000.,1e9])
def test_local_history_and_handoff_preserve_elapsed_time(origin):
    x=np.array([1.8374e6,0.,0.,0.,1000.,0.])
    earth=lambda t:np.array([384400e3,0.,0.])
    sun=lambda t:np.array([1.5e11,0.,0.])
    grid=origin+np.array([-.5,0.,.5])
    truth=np.tile(x,(3,1));truth[:,:3]+=(grid-origin)[:,None]*x[3:]
    h=_propagate_local_state_history(grid,origin,x,0.,0.,0.,earth,sun,1e-12,1e-13)
    np.testing.assert_allclose(h,truth,rtol=0,atol=1e-4)
    handoff=_propagate_hot_start(x,origin,origin+.5,0.,0.,0.,earth,sun,rtol=1e-12,atol=1e-13)
    np.testing.assert_allclose(handoff,truth[-1],rtol=0,atol=1e-4)


@pytest.mark.parametrize('origin',[0.,86400.,100000.])
def test_fast_sigma_does_not_skip_a_subsecond_step(origin):
    grid=origin+np.array([-1.,1.]);x=np.array([1.8374e6,0.,0.,0.,1000.,0.])
    ep=MoonCenteredEphemeris(grid,np.tile([384400e3,0.,0.],(2,1)),
                             np.tile([1.5e11,0.,0.],(2,1)),np.zeros((2,3)))
    prop=make_fast_sigma_propagator(ep,0.,0.,0.)
    if prop is None:pytest.skip('Numba unavailable')
    expected=x.copy();expected[:3]+=.5*x[3:]
    np.testing.assert_allclose(prop(origin,origin+.5,x),expected,rtol=0,atol=1e-6)


@pytest.mark.parametrize('origin',[0.,86400.,100000.,1e9])
@pytest.mark.parametrize('process_noise',[False,True])
def test_formal_handoffs_match_the_analytic_free_motion_covariance(origin,process_noise):
    x=np.array([1.8374e6,0.,0.,0.,1000.,0.])
    covariance=np.diag([100.,200.,300.,.01,.02,.03])
    information=np.diag(1./np.sqrt(np.diag(covariance)))
    q=np.eye(6)*1e-4 if process_noise else None
    earth=lambda t:np.array([384400e3,0.,0.])
    sun=lambda t:np.array([1.5e11,0.,0.])
    args=(origin,origin+.5,0.,0.,0.,earth,sun,q)
    actual_x,actual_p=_propagate_formal_handoff(x,covariance,*args,rtol=1e-12,atol=1e-13)
    sqrt_x,sqrt_r,sqrt_p=_propagate_sqrt_information_handoff(x,information,*args,rtol=1e-12,atol=1e-13)
    phi=np.block([[np.eye(3),.5*np.eye(3)],[np.zeros((3,3)),np.eye(3)]])
    expected_p=phi@covariance@phi.T+(q if q is not None else 0.)
    for actual in [actual_x,sqrt_x]:
        np.testing.assert_allclose(actual,phi@x,rtol=0,atol=1e-4)
    for actual in [actual_p,sqrt_p,np.linalg.inv(sqrt_r.T@sqrt_r)]:
        np.testing.assert_allclose(actual,expected_p,rtol=0,atol=1e-7)
