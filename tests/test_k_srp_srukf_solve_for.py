"""Phase 17-R - permanent guards for K_SRP solve-for in the SR-UKF.

The two-way-range UKF path remains unsupported (unchanged from before this
phase: run_lunar_ukf explicitly rejects measurement_type="two_way_range").
K_SRP is integrated through the qualified counted-Doppler range_rate path
instead (Phase 17A-R's own qualified observable).

K occupies state index 6 (x0_mci/p0 must be 7-dimensional): each sigma point
propagates its OWN K through propagate_state's srp= kwarg (s35), and a
negative-K sigma point is refused, never clipped (s18/s36).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("spiceypy")

from lunar_od.config import Station
from lunar_od.dynamics import propagate_state_with_k_sensitivity
from lunar_od.filters import (
    UnscentedTransformConfig,
    _range_rate_measurement_from_state,
    run_lunar_ukf,
)
from lunar_od.measurements import PassGeometry
from lunar_od.radiometrics import RangeRatePhysicsConfig
from lunar_od.srp import SRPOptions

_MU_MOON = 4902.800066e9
_ET_UTC = "2027-03-02 00:00:00"


def setup_module(_module):
    from lunar_od.spice_loader import load_spice_kernels

    load_spice_kernels()


def _station(lat, lon, alt):
    return Station(name="S", lat_deg=lat, lon_deg=lon, alt_m=alt,
                  color_rgb=(0.0, 0.0, 0.0), sigma_range_m=1.0,
                  sigma_angle_rad=1e-5, sigma_range_rate_mps=1e-4)


def _fixture(k_truth=0.01, duration_s=600.0, step_s=60.0):
    import spiceypy as spice

    et0 = float(spice.str2et(_ET_UTC))

    def get_earth_pos(t):
        return np.tile(np.array([384_400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))

    def get_sun_pos(t):
        return np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))

    t_pass = np.arange(0.0, duration_s + 1.0, step_s)
    r0 = 1737.4e3 + 100e3
    x_true0 = np.array([r0, 30e3, -20e3, -15.0, math.sqrt(_MU_MOON / r0), 4.0])
    srp = SRPOptions(k_srp_m2_per_kg=k_truth)
    nom = propagate_state_with_k_sensitivity(
        t_pass, x_true0, _MU_MOON, 0.0, 0.0, get_earth_pos, get_sun_pos,
        srp=srp, rtol=1e-12, atol=1e-13)
    x_truth = nom[:, :6]

    stations = (_station(0.0, 0.0, 0.0), _station(0.0, 90.0, 0.0),
               _station(45.0, -30.0, 500.0))
    pass_geo = PassGeometry(
        t_s=t_pass, earth_pos_mci_m=np.zeros((t_pass.size, 3)),
        earth_vel_mci_mps=np.zeros((t_pass.size, 3)),
        x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass.size, axis=0),
        stations=stations, measurement_type="range_rate",
        range_rate_physics=RangeRatePhysicsConfig(
            mode="two_way_counted_doppler", count_interval_s=20.0,
            counted_doppler_model="four_event_delay"),
        et0_s=et0,
    )
    pre_margin, post_margin = 12.5, 10.5
    rows = []
    for time_idx, t_s in enumerate(t_pass, start=1):
        if t_s < t_pass[0] + pre_margin or t_s > t_pass[-1] - post_margin:
            continue
        for sid in range(1, len(stations) + 1):
            rows.append([t_s, 0.0, 0.0, 0.0, 0.0, sid, time_idx])
    obs = np.asarray(rows, dtype=float)
    for i, row in enumerate(obs):
        ti = int(row[6]) - 1
        h = _range_rate_measurement_from_state(
            x_truth[ti], row, pass_geo, _MU_MOON, 0.0, 0.0, get_earth_pos,
            get_sun_pos, 1e-12, 1e-13)
        obs[i, 1:5] = h
    return dict(t_pass=t_pass, x_true0=x_true0, x_truth=x_truth, obs=obs,
                pass_geo=pass_geo, gep=get_earth_pos, gsp=get_sun_pos,
                srp=srp, k_truth=k_truth)


@pytest.fixture(scope="module")
def fx():
    return _fixture()


# ======================================================================
# default invariance, checked bitwise
# ======================================================================
def test_default_srukf_is_bitwise_unaffected_by_new_kwargs(fx):
    x0 = fx["x_true0"] + np.array([20.0, -15.0, 10.0, 0.01, -0.008, 0.005])
    p0 = np.diag([80.0 ** 2] * 3 + [0.08 ** 2] * 3)
    args = (fx["t_pass"], fx["obs"], x0, p0, fx["pass_geo"], _MU_MOON, 0.0, 0.0,
            fx["gep"], fx["gsp"])
    kw = dict(config=UnscentedTransformConfig(alpha=0.35), rtol=1e-12, atol=1e-13)

    r_old = run_lunar_ukf(*args, **kw)
    r_new = run_lunar_ukf(*args, **kw, srp=None, solve_for_k_srp=False)

    assert np.array_equal(r_old.final_state, r_new.final_state)
    assert np.array_equal(r_old.final_covariance, r_new.final_covariance)


# ======================================================================
# s13 explicit opt-in / dimension requirement
# ======================================================================
def test_solve_for_k_without_srp_is_rejected(fx):
    x0 = np.concatenate([fx["x_true0"], [0.01]])
    p0 = np.diag([80.0 ** 2] * 3 + [0.08 ** 2] * 3 + [0.005 ** 2])
    with pytest.raises(ValueError, match="solve_for_k_srp"):
        run_lunar_ukf(fx["t_pass"], fx["obs"], x0, p0, fx["pass_geo"], _MU_MOON,
                     0.0, 0.0, fx["gep"], fx["gsp"], solve_for_k_srp=True)


def test_solve_for_k_requires_a_seven_dimensional_state(fx):
    with pytest.raises(ValueError, match="7-element"):
        run_lunar_ukf(
            fx["t_pass"], fx["obs"], fx["x_true0"].copy(),
            np.diag([80.0 ** 2] * 3 + [0.08 ** 2] * 3), fx["pass_geo"],
            _MU_MOON, 0.0, 0.0, fx["gep"], fx["gsp"],
            srp=fx["srp"], solve_for_k_srp=True,
        )


def test_two_way_range_measurement_type_remains_unsupported_with_k(fx):
    """K_SRP integration must not silently open the pre-existing UKF
    two-way-range restriction; it goes through counted Doppler only."""
    x0 = np.concatenate([fx["x_true0"], [0.01]])
    p0 = np.diag([80.0 ** 2] * 3 + [0.08 ** 2] * 3 + [0.005 ** 2])
    with pytest.raises(ValueError, match="two_way_range"):
        run_lunar_ukf(
            fx["t_pass"], fx["obs"], x0, p0, fx["pass_geo"], _MU_MOON, 0.0, 0.0,
            fx["gep"], fx["gsp"], measurement_type="two_way_range",
            srp=fx["srp"], solve_for_k_srp=True,
        )


# ======================================================================
# s36 - negative-K sigma points are refused, never clipped
# ======================================================================
def test_a_covariance_producing_negative_k_sigma_points_is_refused(fx):
    x0 = np.concatenate([fx["x_true0"], [0.001]])
    # UnscentedTransformConfig's default alpha=1e-3 makes the sigma spread
    # ~ alpha*sqrt(n)*sigma -- negligible unless alpha is large enough to be
    # a meaningful spread choice; alpha=0.5 with sigma_K=0.01 against
    # K0=0.001 pushes a sigma point well past zero.
    p0 = np.diag([80.0 ** 2] * 3 + [0.08 ** 2] * 3 + [0.01 ** 2])
    with pytest.raises(ValueError, match="negative K_SRP"):
        run_lunar_ukf(
            fx["t_pass"], fx["obs"], x0, p0, fx["pass_geo"], _MU_MOON, 0.0, 0.0,
            fx["gep"], fx["gsp"], srp=fx["srp"], solve_for_k_srp=True,
            config=UnscentedTransformConfig(alpha=0.5),
        )


# ======================================================================
# basic recovery
# ======================================================================
def test_srukf_recovers_k_from_a_wrong_initial_value(fx):
    x0 = np.concatenate([
        fx["x_true0"] + np.array([20.0, -15.0, 10.0, 0.01, -0.008, 0.005]),
        [1.2 * fx["k_truth"]],
    ])
    p0 = np.diag([80.0 ** 2] * 3 + [0.08 ** 2] * 3 + [0.005 ** 2])
    result = run_lunar_ukf(
        fx["t_pass"], fx["obs"], x0, p0, fx["pass_geo"], _MU_MOON, 0.0, 0.0,
        fx["gep"], fx["gsp"],
        process_noise=np.diag([0.01 ** 2] * 3 + [1e-5 ** 2] * 3 + [0.0]),
        config=UnscentedTransformConfig(alpha=0.35), rtol=1e-12, atol=1e-13,
        srp=fx["srp"], solve_for_k_srp=True,
    )
    assert result.final_state[6] != pytest.approx(1.2 * fx["k_truth"], rel=1e-9)
    assert result.final_state[6] >= 0.0
    assert abs(result.final_state[6] - fx["k_truth"]) < abs(
        1.2 * fx["k_truth"] - fx["k_truth"])
    assert np.all(np.linalg.eigvalsh(result.final_covariance) > 0.0)


# ======================================================================
# Near-truth convergence, the regime where the counted-Doppler observable
# is closest to linear in the state and in K (s63's spirit: a UKF that
# cannot approach truth in the EASY near-linear regime cannot be trusted in
# the nonlinear one). A true common BLS/SRIF-vs-UKF Gaussian oracle would
# require a BLS/SRIF estimator built for the counted-Doppler observable,
# which this phase does not add (BLS/SRIF K_SRP solve-for was integrated for
# two-way range only); the campaign report's BLS-SRIF common-posterior test
# already discharges s32 on that observable.
# ======================================================================
def test_srukf_converges_closely_starting_near_truth(fx):
    x0_orbit = fx["x_true0"] + np.array([2.0, -1.0, 1.0, 0.001, -0.001, 0.0005])
    x0 = np.concatenate([x0_orbit, [1.05 * fx["k_truth"]]])
    p0 = np.diag([50.0 ** 2] * 3 + [0.05 ** 2] * 3 + [0.003 ** 2])
    result = run_lunar_ukf(
        fx["t_pass"], fx["obs"], x0, p0, fx["pass_geo"], _MU_MOON, 0.0, 0.0,
        fx["gep"], fx["gsp"],
        process_noise=np.diag([0.001 ** 2] * 3 + [1e-6 ** 2] * 3 + [0.0]),
        config=UnscentedTransformConfig(alpha=0.35), rtol=1e-12, atol=1e-13,
        srp=fx["srp"], solve_for_k_srp=True,
    )
    assert abs(result.final_state[6] - fx["k_truth"]) < abs(
        1.05 * fx["k_truth"] - fx["k_truth"]), (
        "SR-UKF failed to move K closer to truth from a near-truth start"
    )
