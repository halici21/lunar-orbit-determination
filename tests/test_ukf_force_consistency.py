"""Force switches must reach every state, STM and local-observable path."""
import math
import numpy as np
import pytest
from lunar_od import filters as F, dynamics as D, PassGeometry
from lunar_od.srp import SRPOptions
from lunar_od.constants import MU_MOON_M3S2,R_MOON_M,AU_M
from tests import test_filters as T


def fixture():
    r=R_MOON_M+1e5;x=np.array([r,3e4,-2e4,-15.,math.sqrt(MU_MOON_M3S2/r),4.])
    t=np.array([0.,60.,120.]);earth=lambda t:np.tile([384400e3,0.,0.],(np.size(np.asarray(t)),1))
    sun=lambda t:np.tile([AU_M,0.,0.],(np.size(np.asarray(t)),1))
    srp=SRPOptions(k_srp_m2_per_kg=.1)
    truth=D.propagate_state(t,x,MU_MOON_M3S2,0.,0.,earth,sun,srp=srp,rtol=1e-12,atol=1e-13)
    stations=tuple(T._synthetic_position_station(*q) for q in [(0.,0.,0.),(0.,90.,0.),(45.,-30.,500.)])
    pg=PassGeometry(t_s=t,earth_pos_mci_m=np.zeros((3,3)),earth_vel_mci_mps=np.zeros((3,3)),x_j2000_to_itrf93=np.repeat(np.eye(6)[None,:,:],3,axis=0),stations=stations,measurement_type='position')
    obs=T._build_clean_position_observations(t,truth,pg)
    return t,x,pg,obs,earth,sun,srp


def test_fixed_srp_reaches_the_stm_predictor(monkeypatch):
    t,x,pg,obs,earth,sun,srp=fixture();calls=[];original=F.propagate_augmented_state
    def capture(*args,**kw):
        calls.append(kw.get('srp'));return original(*args,**kw)
    monkeypatch.setattr(F,'propagate_augmented_state',capture)
    F.run_lunar_ukf(t,obs,x,np.diag([80.**2]*3+[.08**2]*3),pg,MU_MOON_M3S2,0.,0.,earth,sun,
                    srp=srp,use_stm_linearization=True,rtol=1e-12,atol=1e-13)
    assert calls and all(p==srp for p in calls)


def test_local_ode_history_carries_srp_and_j2(monkeypatch):
    t,x,pg,obs,earth,sun,srp=fixture();calls=[];original=F.propagate_state
    def capture(*args,**kw):
        calls.append(kw);return original(*args,**kw)
    monkeypatch.setattr(F,'propagate_state',capture)
    F._propagate_local_state_history(np.array([59.,60.,61.]),60.,x,MU_MOON_M3S2,0.,0.,earth,sun,
                                      1e-12,1e-13,srp=srp,j2_moon=2e-4)
    assert calls and all(c.get('srp')==srp and c.get('j2_moon')==2e-4 for c in calls)


def test_unverified_fast_provider_cannot_override_an_active_srp_model():
    t,x,pg,obs,earth,sun,srp=fixture()
    with pytest.raises(ValueError,match='SRP'):
        F.run_lunar_ukf(t,obs,x,np.eye(6),pg,MU_MOON_M3S2,0.,0.,earth,sun,srp=srp,
                        fast_sigma_propagator=lambda t0,t1,x:x)
