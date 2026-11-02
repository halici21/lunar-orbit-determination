"""Differentiate the same midpoint TT/TDB interval factor used in the value."""
import numpy as np
import pytest
from lunar_od import relativistic_light_time as R


@pytest.mark.parametrize('step', [.01, .1])
def test_clock_partial_includes_interval_rate_curvature(step, monkeypatch):
    class Epoch:
        et0_s = 0.
    # Analytic clock relation D(t)=a*t^2/2. Amplified curvature resolves the
    # neglected chain term above range subtraction roundoff, without SPICE.
    a = 1e-10
    monkeypatch.setattr(R._station_time, 'tdb_minus_tt_rate', lambda et, t, site: a*t)
    sc = R.SpacecraftHistory([-10.,10.], np.tile([3.84e8,0.,0.,0.,0.,0.],(2,1)))
    station = lambda t: np.array([6.4e6,0.,0.,0.,0.,0.])
    model = R.LightTimeModel(interval_time_scale='TT', station_time='FULL_TOPOCENTRIC')
    def solve(t):
        return R.solve_two_way(t,station,sc,Epoch(),model,site=object())
    sol = solve(.5)
    expected = -.5*R.C_LIGHT_MPS*sol.round_trip_s*a
    fd = (R.two_way_range_m(solve(.5+step))-R.two_way_range_m(solve(.5-step)))/(2*step)
    assert expected == pytest.approx(fd, abs=5e-6)
    assert R.two_way_range_clock_partial(sol) == pytest.approx(expected, abs=1e-9)


def test_state_partial_includes_the_midpoint_dependency_on_light_time(monkeypatch):
    class Epoch:
        et0_s = 0.
    a = 1e-6
    monkeypatch.setattr(R._station_time, 'tdb_minus_tt_rate', lambda et, t, site: a*t)
    t = np.array([-10., 10.])
    state = np.tile([3.84e8,0.,0.,0.,0.,0.], (2,1))
    history = np.hstack([state, np.tile(np.eye(6).reshape(-1, order='F'), (2,1))])
    sc = R.SpacecraftHistory(t, history)
    model = R.LightTimeModel(interval_time_scale='TT', station_time='FULL_TOPOCENTRIC')
    sol = R.solve_two_way(.5,lambda t:np.array([6.4e6,0.,0.,0.,0.,0.]),sc,Epoch(),model,site=object())
    expected = 1-sol.tdb_minus_tt_rate+.5*sol.round_trip_s*a
    assert R.two_way_range_partial(sol,sc)[0] == pytest.approx(expected, abs=1e-12)
