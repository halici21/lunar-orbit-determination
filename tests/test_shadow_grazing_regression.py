"""Independent circular-trajectory oracle for paired roots within one solver step."""
import math
import numpy as np
import pytest
from scipy.optimize import brentq
from lunar_od.constants import AU_M,MU_MOON_M3S2,R_MOON_M
from lunar_od.dynamics import shadow_contact_times
from lunar_od.srp import SRPOptions,apparent_radii_and_separation

R=R_MOON_M+100e3
W=math.sqrt(MU_MOON_M3S2/R**3)
MID=math.pi/W
def state(t):
    return np.array([R*math.cos(W*t),R*math.sin(W*t),0.,-R*W*math.sin(W*t),R*W*math.cos(W*t),0.])

def geometry(duration):
    def at_beta(beta,t):
        sun=AU_M*np.array([math.cos(beta),0.,math.sin(beta)])
        a,b,g,*_=apparent_radii_and_separation(state(t)[:3],sun,np.zeros(3),R_MOON_M)
        return g-a-b
    beta=brentq(lambda b:at_beta(b,MID-duration/2),1.0,1.4,xtol=1e-14)
    sun=AU_M*np.array([math.cos(beta),0.,math.sin(beta)])
    return sun,lambda t:at_beta(beta,t)

@pytest.mark.parametrize('duration',[1.,5.,18.9413577,60.])
@pytest.mark.parametrize('phase',[0.,7.,19.,31.])
@pytest.mark.parametrize('backward',[False,True])
def test_grazing_pair_matches_analytic_roots(duration,phase,backward):
    sun,q=geometry(duration)
    expected=np.array([brentq(q,MID-100,MID,xtol=1e-10),brentq(q,MID,MID+100,xtol=1e-10)])
    span=[MID-180+phase,MID+180+phase]
    if backward:span.reverse();expected=expected[::-1]
    found=shadow_contact_times(span,state(span[0]),MU_MOON_M3S2,0.,0.,lambda t:np.array([384400e3,0.,0.]),lambda t:sun,
                              srp=SRPOptions(k_srp_m2_per_kg=0.),rtol=1e-12,atol=1e-13)
    assert len(found)==2
    np.testing.assert_allclose(found,expected,rtol=0,atol=1e-3)

def test_tangent_is_not_a_spurious_eclipse_pair():
    sun,q=geometry(0.)
    span=[MID-180,MID+180]
    found=shadow_contact_times(span,state(span[0]),MU_MOON_M3S2,0.,0.,lambda t:np.array([384400e3,0.,0.]),lambda t:sun,
                              srp=SRPOptions(k_srp_m2_per_kg=0.),rtol=1e-12,atol=1e-13)
    assert len(found)<=1
    if found:assert abs(found[0]-MID)<1e-3
