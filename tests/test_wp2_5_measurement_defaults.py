"""F2 WP2.5 (MEAS-09, MEAS-10): measurement defaults, gates in studies/phase17/configs/wp2_5_predeclaration.json.

The oracle of the Jacobians is a Richardson extrapolation of central differences of re-propagated arcs, with its
convergence order measured rather than the best step picked.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lunar_od.config import Station
from lunar_od.measurements import (
    compute_position_residuals,
    compute_position_residuals_analytic,
    generate_position_measurements,
    normalize_jacobian_model,
    position_initial_state_jacobian_from_augmented_history,
)

STATIONS = (
    Station("Goldstone", 35.4267, -116.8900, 1000.0, (0, 0, 0), 2.0, 1e-4, 1e-4),
    Station("Canberra", -35.4014, 148.9817, 690.0, (0, 0, 0), 2.0, 1e-4, 1e-4),
)
FIRST_ORDER = "analytic_first_order_light_time"
IMPLICIT = "implicit_light_time"
EXACT = "analytic_exact_geometric"


@pytest.fixture(scope="module")
def arc():
    spice = pytest.importorskip("spiceypy")
    from lunar_od.spice_loader import load_spice_kernels
    try:
        load_spice_kernels(None, clear=False)
    except FileNotFoundError:
        pytest.skip("SPICE kernels not available")
    from lunar_od.dynamics import propagate_augmented_state
    from lunar_od.ephemeris import sample_moon_centered_ephemeris
    from lunar_od.ephemeris_generation import gravitational_parameters
    from lunar_od.geometry import ecef2razel_sez

    et0 = spice.str2et("2027-03-02T00:00:00")
    t = np.arange(0.0, 3600.0 + 1.0, 30.0)
    eph = sample_moon_centered_ephemeris(et0, np.arange(-600.0, 3600.0 + 601.0, 60.0))
    gm = gravitational_parameters()
    r0 = 1737.4e3 + 100e3
    x0 = np.array([r0, 30e3, -20e3, -15.0, np.sqrt(gm[0] / r0), 4.0])

    def propagate(x_initial, augmented=False):
        y0 = np.concatenate([x_initial, np.eye(6).reshape(-1, order="F")])
        hist = propagate_augmented_state(t, y0, *gm, eph.earth_position, eph.sun_position, rtol=1e-12, atol=1e-13)
        return hist if augmented else hist[:, :6]

    aug = propagate(x0, augmented=True)
    earth = eph.earth_position(t)
    vis = np.zeros((t.size, len(STATIONS)), dtype=bool)
    for k in range(t.size):
        rot = np.asarray(spice.pxform("J2000", "ITRF93", et0 + t[k]))
        sc_ecef = rot @ (aug[k, :3] - earth[k])
        for j, st in enumerate(STATIONS):
            vis[k, j] = ecef2razel_sez(sc_ecef - st.r_ecef_m, st.lat_rad, st.lon_rad)[1] >= np.radians(10.0)
    return dict(et0=et0, t=t, eph=eph, gm=gm, x0=x0, propagate=propagate, aug=aug, vis=vis)


def _position(arc, jacobian_model=None, profile="one_way_light_time"):
    eph = arc["eph"]
    return generate_position_measurements(arc["t"], arc["aug"][:, :6], STATIONS, arc["vis"], eph.earth_position,
                                          eph.earth_velocity, arc["et0"], noise=False, arc_id=1,
                                          measurement_model_profile=profile, jacobian_model=jacobian_model)[:2]


def _analytic(arc, obs, geo):
    _, _, h_tilde = compute_position_residuals_analytic(arc["aug"][:, :6], obs, geo)
    return np.asarray(position_initial_state_jacobian_from_augmented_history(obs, arc["aug"], h_tilde, geo)).reshape(-1, 3, 6)


def _central(arc, obs, geo, pos_step, vel_step):
    def predicted(x):
        return compute_position_residuals(arc["propagate"](x), obs, geo)[1].reshape(-1, 3)

    d = np.zeros((obs.shape[0], 3, 6))
    for j in range(6):
        e = np.zeros(6)
        e[j] = pos_step if j < 3 else vel_step
        delta = predicted(arc["x0"] + e) - predicted(arc["x0"] - e)
        delta[:, 1] = (delta[:, 1] + np.pi) % (2.0 * np.pi) - np.pi
        d[:, :, j] = delta / (2.0 * e[j])
    return d


def _relative(a, b):
    return np.linalg.norm(a - b, axis=2) / np.linalg.norm(b, axis=2)


def _richardson(arc, obs, geo):
    d1 = _central(arc, obs, geo, 20.0, 0.02)
    d2 = _central(arc, obs, geo, 10.0, 0.01)
    return (4.0 * d2 - d1) / 3.0


def test_g1_central_differences_converge_to_the_default_jacobian_at_second_order(arc):
    obs, geo = _position(arc)
    h = _analytic(arc, obs, geo)
    errors = [np.median(_relative(_central(arc, obs, geo, 1e3 * s, 1.0 * s), h).max(axis=1)) for s in (8.0, 4.0, 2.0)]
    for coarse, fine in zip(errors, errors[1:]):
        assert 3.0 <= coarse / fine <= 5.0, errors


def test_g2_default_jacobian_matches_richardson_and_first_order_does_not(arc):
    obs, geo = _position(arc)
    reference = _richardson(arc, obs, geo)
    assert np.max(_relative(_analytic(arc, obs, geo), reference)) < 2e-6
    first = replace(geo, jacobian_model=FIRST_ORDER)
    assert np.max(_relative(_analytic(arc, obs, first), reference)) > 0.01


@pytest.mark.parametrize("profile", ["one_way_light_time", "one_way_light_time_aberrated_local_mci",
                                     "one_way_light_time_ssb", "one_way_light_time_ssb_aberrated"])
def test_g3_light_time_profiles_default_to_implicit(profile):
    for label in (None, EXACT):
        assert normalize_jacobian_model(label, measurement_model_profile=profile) == IMPLICIT
    assert normalize_jacobian_model(FIRST_ORDER, measurement_model_profile=profile) == FIRST_ORDER


def test_g3_companion_and_geometric_resolution():
    for label in (None, EXACT):
        assert normalize_jacobian_model(label, companion_geometry="apparent_one_way") == IMPLICIT
    assert normalize_jacobian_model(None) == EXACT
    assert normalize_jacobian_model(EXACT) == EXACT
    with pytest.raises(ValueError, match="implicit_light_time"):
        normalize_jacobian_model(IMPLICIT)


def test_g4_observations_do_not_depend_on_the_jacobian_model(arc):
    assert np.array_equal(_position(arc)[0], _position(arc, FIRST_ORDER)[0])


@pytest.mark.parametrize("estimator", ["bls_lm", "srif"])
def test_g5_estimators_recover_the_truth(arc, estimator):
    import lunar_od.estimators as est

    run = getattr(est, f"estimate_position_{estimator}")
    start = arc["x0"] + np.array([80.0, -60.0, 40.0, 0.02, -0.01, 0.015])

    def errors(model):
        obs, geo = _position(arc, model)
        x_hat, status, _ = run(arc["t"], obs, start, geo, *arc["gm"], arc["eph"].earth_position,
                               arc["eph"].sun_position, rtol=1e-12, atol=1e-13)
        return np.linalg.norm(x_hat[:3] - arc["x0"][:3]), np.linalg.norm(x_hat[3:] - arc["x0"][3:]), status

    pos, vel, status = errors(None)
    pos_first, vel_first, _ = errors(FIRST_ORDER)
    assert pos < 0.01 and vel < 1e-5, (pos, vel, status)
    assert pos < 1e-3 or pos <= 1.5 * pos_first, (pos, pos_first)
    assert vel <= 1.5 * max(vel_first, 1e-6), (vel, vel_first)


def test_g6_posterior_sigmas_match_the_richardson_design(arc):
    from lunar_od.estimators import estimate_position_bls_lm

    obs, geo = _position(arc)
    h = _richardson(arc, obs, geo)
    station = STATIONS[1]
    weights = np.array([1.0 / station.sigma_range_m, 1.0 / station.sigma_angle_rad, 1.0 / station.sigma_angle_rad])
    design = (h * weights[None, :, None]).reshape(-1, 6)
    assert np.all(obs[:, 4] == 2)                          # Canberra only on this arc
    reference = np.sqrt(np.diag(np.linalg.inv(design.T @ design)))

    def sigmas(model):
        o, g = _position(arc, model)
        _, _, stats = estimate_position_bls_lm(arc["t"], o, arc["x0"], g, *arc["gm"], arc["eph"].earth_position,
                                               arc["eph"].sun_position, rtol=1e-12, atol=1e-13,
                                               return_posterior=True)
        return np.sqrt(np.diag(stats.posterior_covariance))

    assert np.max(np.abs(sigmas(None) / reference - 1.0)) < 0.01, sigmas(None) / reference


def test_g7_scenario_label_says_what_runs(arc):
    from lunar_od.scenario_config import scenario_config_from_mapping

    config = scenario_config_from_mapping({"name": "wp2_5", "start_mode": "cold", "network": "single",
                                           "measurement_type": "position", "estimator_type": "bls_lm",
                                           "measurement_model_profile": "one_way_light_time"})
    assert config.jacobian_model == EXACT                  # the field and its cached-package identity stay
    _, geo = _position(arc, config.jacobian_model, config.measurement_model_profile)
    assert geo.jacobian_model == IMPLICIT
