"""A moving delay body contributes even when both link endpoints are fixed."""
import numpy as np
import pytest
from lunar_od import relativistic_light_time as R


class MovingBody:
    # Deliberately amplified GM resolves the derivative above range roundoff;
    # this is an analytic diagnostic, not an Earth gravity model.
    velocity = np.array([3e4, 5e3, 0.])

    def moon_state(self, t):
        return np.zeros(6)

    def body_position(self, name, t):
        return self.velocity * t

    def body_velocity(self, name, t):
        return self.velocity

    def gm(self, name):
        return 3.986e17


@pytest.mark.parametrize('step', [.01, .1])
def test_clock_partial_differentiates_the_moving_shapiro_body(step):
    eph = MovingBody()
    sc = R.SpacecraftHistory([-10., 10.], np.tile([3.84e8, 1e7, 0., 0., 0., 0.], (2, 1)))
    station = lambda t: np.array([6.4e6, 0., 0., 0., 0., 0.])
    model = R.LightTimeModel(frame='SSB', shapiro_bodies=('EARTH',))
    def solve(t):
        return R.solve_two_way(t, station, sc, eph, model)
    fd = (R.two_way_range_m(solve(.5 + step)) - R.two_way_range_m(solve(.5 - step))) / (2*step)
    assert abs(fd) > 1e-3
    assert R.two_way_range_clock_partial(solve(.5)) == pytest.approx(fd, abs=5e-6)
