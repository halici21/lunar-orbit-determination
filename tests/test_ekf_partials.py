"""Phase D6: analytic local measurement partials for the EKF against finite-difference oracles."""

import math
from dataclasses import replace

import numpy as np
import pytest

from lunar_od import PassGeometry, RangeRatePhysicsConfig, propagate_augmented_state
from lunar_od.ekf import run_lunar_ekf
from lunar_od.ekf_partials import analytic_measurement_jacobian
from lunar_od.filters import _measurement_context, _measurement_residual, _resolve_ukf_bias_config
from lunar_od.radiometrics import LEGACY_INTERPOLATED_STATION_METHOD
from tests.test_ekf_smoother import MU_MOON, _add_noise, _arc, _earth, _sun
from tests.test_filters import _build_clean_two_way_range_rate_observations, _synthetic_range_rate_station

TOL = dict(rtol=1e-12, atol=1e-13)


def counted_doppler_arc(duration_s=3601.0, count_interval_s=20.0):
    """D3's arc observed by two-way counted Doppler; every count window lies inside the pass grid."""
    r0 = 1737.4e3 + 100e3
    x_true0 = np.array([r0, 30e3, -20e3, -15.0, math.sqrt(MU_MOON / r0), 4.0])
    t_pass = np.arange(0.0, duration_s, 60.0)
    truth = propagate_augmented_state(t_pass, np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")]),
                                      MU_MOON, 0.0, 0.0, _earth, _sun, **TOL)[:, :6]
    pass_geo = PassGeometry(
        t_s=t_pass, earth_pos_mci_m=np.zeros((t_pass.size, 3)), earth_vel_mci_mps=np.zeros((t_pass.size, 3)),
        x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass.size, axis=0),
        stations=tuple(_synthetic_range_rate_station(*s) for s in ((0.0, 0.0, 0.0), (0.0, 90.0, 0.0),
                                                                   (45.0, -30.0, 500.0))),
        measurement_type="range_rate",
        range_rate_physics=RangeRatePhysicsConfig(mode="two_way_counted_doppler", count_interval_s=count_interval_s,
                                                  station_state_method=LEGACY_INTERPOLATED_STATION_METHOD))
    obs = _build_clean_two_way_range_rate_observations(t_pass, truth, pass_geo, MU_MOON, _earth, _sun,
                                                       edge_margin_s=(60.0, 60.0))
    return x_true0, t_pass, truth, pass_geo, obs


def richardson_jacobian(measurement_type, row, pass_geo, x, h_pos=10.0, h_vel=1e-2):
    """Richardson-extrapolated central difference of the shared measurement function (D6 oracle)."""
    bias_cfg = _resolve_ukf_bias_config(measurement_type, 6, len(pass_geo.stations), None)
    _, _, fn, _, _ = _measurement_context(measurement_type, row, pass_geo, MU_MOON, 0.0, 0.0, _earth, _sun,
                                          TOL["rtol"], TOL["atol"], bias_cfg)
    residual = _measurement_residual(measurement_type)

    def central(j, step):
        plus, minus = x.copy(), x.copy()
        plus[j] += step
        minus[j] -= step
        return np.asarray(residual(np.asarray(fn(plus)), np.asarray(fn(minus)))) / (2.0 * step)

    cols = []
    for j in range(6):
        step = h_pos if j < 3 else h_vel
        cols.append((4.0 * central(j, step / 2.0) - central(j, step)) / 3.0)
    return np.column_stack(cols)


def _relative_row_errors(h, ref):
    return np.linalg.norm(h - ref, axis=1) / np.linalg.norm(ref, axis=1)


@pytest.mark.parametrize("measurement_type", ["position", "range_rate", "counted_doppler"])
def test_analytic_partials_match_the_richardson_oracle(measurement_type):
    if measurement_type == "counted_doppler":
        _, _, truth, pass_geo, obs = counted_doppler_arc(duration_s=1201.0)
        kind = "range_rate"
    else:
        _, _, truth, pass_geo, obs = _arc(measurement_type, duration_s=1201.0)
        kind = measurement_type
    time_col = 5 if kind == "position" else 6
    rng = np.random.default_rng(0)
    for row in obs[::4]:
        x = truth[int(row[time_col]) - 1] + rng.normal(0.0, 1.0, 6) * [20, 20, 20, 0.02, 0.02, 0.02]
        h = analytic_measurement_jacobian(kind, row, pass_geo, x, MU_MOON, 0.0, 0.0, _earth, _sun, **TOL)
        err = _relative_row_errors(h, richardson_jacobian(kind, row, pass_geo, x))
        if measurement_type == "counted_doppler":
            assert err[1] <= 1e-4 and np.all(err[[0, 2, 3]] <= 1e-6), err
        else:
            assert np.all(err <= 1e-6), err


@pytest.mark.parametrize("measurement_type", ["position", "range_rate", "counted_doppler"])
def test_analytic_ekf_matches_the_finite_difference_ekf(measurement_type):
    if measurement_type == "counted_doppler":
        x_true0, t_pass, _, pass_geo, clean = counted_doppler_arc(duration_s=601.0)
        kind = "range_rate"
    else:
        x_true0, t_pass, _, pass_geo, clean = _arc(measurement_type)
        kind = measurement_type
    rng = np.random.default_rng(1)
    obs = _add_noise(clean, kind, rng)
    p0 = np.diag([20.0**2] * 3 + [0.02**2] * 3)
    x_guess = x_true0 + rng.multivariate_normal(np.zeros(6), p0)
    args = (t_pass, obs, x_guess, p0, pass_geo, MU_MOON, 0.0, 0.0, _earth, _sun)
    fd = run_lunar_ekf(*args, smooth=True, **TOL)
    an = run_lunar_ekf(*args, smooth=True, measurement_jacobian="analytic", **TOL)
    sigma = np.sqrt(np.diag(fd.final_covariance))
    assert np.all(np.abs(an.final_state - fd.final_state) <= 0.01 * sigma)
    sigma0 = np.sqrt(np.diag(fd.smoothed.covariances[0]))
    assert np.all(np.abs(an.smoothed.state_estimates[0] - fd.smoothed.state_estimates[0]) <= 0.01 * sigma0)


def test_analytic_bias_columns_match_the_finite_difference_ekf():
    x_true0, t_pass, _, pass_geo, obs = _arc("position")
    obs[:, 1:4] += np.array([20.0, 1.2e-5, -8.0e-6])
    x_guess = np.concatenate([x_true0, np.zeros(3)])
    p0 = np.diag([5.0**2] * 3 + [0.01**2] * 3 + [100.0**2, 1e-4**2, 1e-4**2])
    args = (t_pass, obs, x_guess, p0, pass_geo, MU_MOON, 0.0, 0.0, _earth, _sun)
    fd = run_lunar_ekf(*args, **TOL)
    an = run_lunar_ekf(*args, measurement_jacobian="analytic", **TOL)
    sigma = np.sqrt(np.diag(fd.final_covariance))
    assert np.all(np.abs(an.final_state - fd.final_state) <= 0.01 * sigma)


def test_analytic_partials_reject_observables_they_do_not_differentiate_exactly():
    x_true0, t_pass, _, pass_geo, obs = counted_doppler_arc(duration_s=601.0)
    args = (t_pass, obs, x_true0, np.eye(6), MU_MOON, 0.0, 0.0, _earth, _sun)
    taylor = replace(pass_geo, range_rate_physics=replace(pass_geo.range_rate_physics, local_state_model="taylor3"))
    apparent = replace(pass_geo, companion_geometry="apparent_one_way")
    for geo in (taylor, apparent):
        with pytest.raises(ValueError):
            run_lunar_ekf(*args[:4], geo, *args[4:], measurement_jacobian="analytic")
    with pytest.raises(ValueError):
        run_lunar_ekf(*args[:4], pass_geo, *args[4:], measurement_jacobian="exact")
