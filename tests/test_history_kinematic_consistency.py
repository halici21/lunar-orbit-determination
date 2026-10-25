"""Position, velocity and variational polynomials must share boundary semantics."""
import numpy as np
import pytest
from lunar_od.ephemeris import MoonCenteredEphemeris,EphemerisCoverageError
from lunar_od.radiometrics import _interp_state,_interp_state_transition_position
from lunar_od.history_domain import HistoryDomainError

T=np.array([0.,5.,10.])
S=np.column_stack([T*T,np.zeros((3,2)),2*T,np.zeros((3,2))])

@pytest.mark.parametrize('t',[-.1,0.,2.5,5.,9.9,10.,10.1])
def test_state_velocity_is_position_derivative_even_at_edges(t):
    h=1e-4
    v=(_interp_state(T,S,t+h)[:3]-_interp_state(T,S,t-h)[:3])/(2*h)
    np.testing.assert_allclose(_interp_state(T,S,t)[3:6],v,rtol=0,atol=1e-7)

@pytest.mark.parametrize('t',[-.1,0.,2.5,10.,10.1])
def test_stm_and_state_use_the_same_polynomial(t):
    phi=np.tile(np.eye(6),(3,1,1));phi[:,0,0]=T*T;phi[:,3,0]=2*T
    h=.01
    fd=(_interp_state(T,S+h*phi[:,:,0],t)[:3]-_interp_state(T,S-h*phi[:,:,0],t)[:3])/(2*h)
    np.testing.assert_allclose(_interp_state_transition_position(T,phi,t)[:,0],fd,rtol=0,atol=1e-8)

def test_earth_ephemeris_derivative_and_nodes():
    e=MoonCenteredEphemeris(T,S[:,:3],S[:,:3],S[:,3:6])
    q=np.array([0.,2.5,5.,9.9,10.])
    np.testing.assert_allclose(e.earth_position(q)[:,0],q*q,rtol=0,atol=1e-12)
    np.testing.assert_allclose(e.earth_velocity(q)[:,0],2*q,rtol=0,atol=1e-12)
    np.testing.assert_allclose(e._ep.derivative()(q),e.earth_velocity(q),rtol=0,atol=1e-12)

def test_coverage_does_not_expand():
    with pytest.raises(HistoryDomainError):_interp_state(T,S,21.)
    e=MoonCenteredEphemeris(T,S[:,:3],S[:,:3],S[:,3:6],extrapolation_margin_s=0.)
    with pytest.raises(EphemerisCoverageError):e.earth_position(10.001)


def test_cubic_polynomial_uses_the_same_continuation_for_state_and_earth():
    t = np.array([0., 2., 7.])
    r = np.column_stack([t**3, -2*t**3+t, 3*t**2])
    v = np.column_stack([3*t**2, -6*t**2+1, 6*t])
    ep = MoonCenteredEphemeris(t, r, r, v)
    for q in [-.1, 0., 1., 2., 7., 7.1]:
        expected = np.array([q**3, -2*q**3+q, 3*q**2, 3*q**2, -6*q**2+1, 6*q])
        np.testing.assert_allclose(_interp_state(t, np.hstack([r,v]), q), expected, atol=1e-10)
        np.testing.assert_allclose(ep.earth_position(q), expected[:3], atol=1e-10)
        np.testing.assert_allclose(ep.earth_velocity(q), expected[3:], atol=1e-10)
