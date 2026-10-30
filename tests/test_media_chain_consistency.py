"""Value and Jacobian differentiate the same group/phase delay model."""
import numpy as np
import pytest
from lunar_od import relativistic_light_time as R


class AnalyticMedia:
    def leg_delays(self,site,et,los,uplink):
        los=np.asarray(los)
        return (1000.+.001*et+np.array([1e-6,-2e-6,.5e-6])@los,
                1000.+.002*et+.00005*et**2+np.array([-2e-6,1e-6,1e-6])@los,1.)


class Epoch:
    et0_s=0.


def setup(delta=None):
    p=np.array([1.8374e6,1e6,2e5,10.,1000.,-20.])
    if delta is not None:p+=delta
    t=np.array([-40.,40.]);states=np.array([np.r_[p[:3]+ti*p[3:],p[3:]] for ti in t])
    phi=np.array([np.block([[np.eye(3),ti*np.eye(3)],[np.zeros((3,3)),np.eye(3)]]) for ti in t])
    h=np.hstack([states,phi.transpose(0,2,1).reshape(2,36)])
    return R.SpacecraftHistory(t,h)


def station(t):return np.array([384400e3+t,20*t,0.,1.,20.,0.])


def solve(sc,t=0.):
    return R.solve_two_way(t,station,sc,Epoch(),R.LightTimeModel(),media=AnalyticMedia(),media_site=object())


@pytest.mark.parametrize('step',[.01,.1,1.])
def test_range_clock_partial_has_the_group_media_chain(step):
    sc=setup();fd=(R.two_way_range_m(solve(sc,step))-R.two_way_range_m(solve(sc,-step)))/(2*step)
    assert R.two_way_range_clock_partial(solve(sc))==pytest.approx(fd,abs=1e-5)


@pytest.mark.parametrize('column,step',[(0,10.),(1,10.),(2,10.),(4,10.),(4,100.)])
def test_range_state_partial_has_the_media_chain(column,step):
    dx=np.zeros(6);dx[column]=step
    fd=(R.two_way_range_m(solve(setup(dx)))-R.two_way_range_m(solve(setup(-dx))))/(2*step)
    assert R.two_way_range_partial(solve(setup()),setup())[column]==pytest.approx(fd,abs=1e-8)


@pytest.mark.parametrize('step',[.01,.1,1.])
def test_counted_clock_partial_uses_phase_media(step):
    def dop(t):return R.counted_doppler(t,20.,station,setup(),Epoch(),R.LightTimeModel(),media=AnalyticMedia(),media_site=object(),with_clock_partial=True)
    fd=(dop(step)[0]-dop(-step)[0])/(2*step)
    assert dop(0.)[3]==pytest.approx(fd,abs=5e-6)


def test_counted_velocity_partial_uses_phase_media():
    def dop(dx):return R.counted_doppler(0.,20.,station,setup(dx),Epoch(),R.LightTimeModel(),media=AnalyticMedia(),media_site=object(),with_partial=True)
    dx=np.zeros(6);dx[4]=1.
    fd=(dop(dx)[0]-dop(-dx)[0])/2
    assert dop(None)[1][4]==pytest.approx(fd,abs=2e-8)


class SiteDependentMedia(AnalyticMedia):
    def leg_delays(self,site,et,los,uplink):
        g,p,e=super().leg_delays(site,et,los,uplink)
        return site*g,site*p,e


def dor(sc,t=0.):
    station_b=lambda t:station(t)+np.array([0.,2e6,0.,0.,0.,0.])
    return R.delta_dor_station_time(t,station,station_b,sc,Epoch(),R.LightTimeModel(),
                                    media=SiteDependentMedia(),media_sites=(1.,2.))


@pytest.mark.parametrize('column',[0,1,4])
def test_delta_dor_measured_value_and_partial_use_the_same_media(column):
    step=100. if column<3 else 10.
    dx=np.zeros(6);dx[column]=step
    fd=(dor(setup(dx)).delay_s-dor(setup(-dx)).delay_s)/(2*step)
    assert R.delta_dor_partial(dor(setup()),setup())[column]==pytest.approx(fd,abs=2e-16)


@pytest.mark.parametrize('step',[.1,1.])
def test_delta_dor_measured_clock_partial_has_the_media_chain(step):
    fd=(dor(setup(),step).delay_s-dor(setup(),-step).delay_s)/(2*step)+1.
    ca,cb=R.delta_dor_clock_partials(dor(setup()))
    assert ca==pytest.approx(fd,abs=2e-13)
    assert cb==-1.


@pytest.mark.parametrize('observable',['range','doppler'])
def test_vacuum_keeps_additional_empirical_sensitivity_columns(observable):
    # The F2 empirical adapter extends position_sensitivity without changing
    # the underlying six/seven-column history layout or n_par marker.
    base=setup()
    class Extended(R.SpacecraftHistory):
        def position_sensitivity(self,t):
            return np.hstack([super().position_sensitivity(t),np.ones((3,4))])
    sc=Extended(base.t_grid,base.hist)
    if observable=='range':
        sol=R.solve_two_way(0.,station,sc,Epoch(),R.LightTimeModel())
        partial=R.two_way_range_partial(sol,sc)
    else:
        _,partial,_=R.counted_doppler(0.,20.,station,sc,Epoch(),R.LightTimeModel(),with_partial=True)
    assert partial.shape==(10,)
    np.testing.assert_allclose(partial[6:],np.sum(partial[:3]),rtol=1e-8,atol=1e-14)
