"""F2 WP2.1 (MEAS-01, X-07): barycentric DSN counted Doppler in the production estimators.

Oracles, each independent of the code under test:

* one-way light time against SPICE ``spkezr(..., 'CN', ...)`` (converged Newtonian in the SSB frame; the
  2e-11 s tolerance is SPICE's own epoch-quantisation floor, see the test);
* the Shapiro closed form (Moyer eq. 8-38) against a numerical line integral of 2GM/(c^3 r);
* the production Jacobian against central differences of re-propagated arcs;
* BLS-LM recovers the truth from barycentric data, while the Moon-centred model leaves a systematic misfit.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from lunar_od.config import Station
from lunar_od.radiometrics import (
    BARYCENTRIC_DSN_COUNTED_DOPPLER_MODEL,
    BARYCENTRIC_DSN_COUNTED_DOPPLER_MODEL_VERSION,
    LEGACY_INTERPOLATED_STATION_METHOD,
    RangeRatePhysicsConfig,
    two_way_counted_doppler_initial_state_jacobian,
    two_way_counted_doppler_observable,
)

C = 299792458.0
COUNT_S = 60.0
BARY = RangeRatePhysicsConfig(mode="two_way_counted_doppler", count_interval_s=COUNT_S,
                              counted_doppler_model=BARYCENTRIC_DSN_COUNTED_DOPPLER_MODEL)
MOON_CENTRED = replace(BARY, counted_doppler_model="single_bounce_exact_station")
STATIONS = (
    Station("Goldstone", 35.4267, -116.8900, 1000.0, (0, 0, 0), 2.0, 1e-4, 1e-4),
    Station("Canberra", -35.4014, 148.9817, 690.0, (0, 0, 0), 2.0, 1e-4, 1e-4),
)


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

    return dict(spice=spice, et0=et0, t=t, eph=eph, gm=gm, x0=x0, propagate=propagate,
                aug=propagate(x0, augmented=True))


def _doppler(arc, history, t_mid, cfg=BARY, station=STATIONS[0]):
    return two_way_counted_doppler_observable(t_mid, station, arc["t"], history, arc["eph"].earth_position(arc["t"]),
                                              arc["eph"].earth_velocity(arc["t"]), np.zeros((arc["t"].size, 6, 6)),
                                              cfg, et0_s=arc["et0"])


def test_one_way_light_time_matches_spice_converged_newtonian(arc):
    from lunar_od import relativistic_light_time as rlt

    spice, et0 = arc["spice"], arc["et0"]
    model = rlt.LightTimeModel(frame="SSB")
    eph = rlt.BarycentricEphemeris(et0)
    moon = rlt.SpacecraftHistory(arc["t"], np.zeros((arc["t"].size, 6)))   # the "spacecraft" is the Moon centre

    def earth_from_moon(t_s):
        return np.asarray(spice.spkezr("EARTH", et0 + t_s, "J2000", "NONE", "MOON")[0]) * 1e3

    for t3 in (600.0, 1800.0, 3000.0):
        _, _, leg = rlt.one_way_range(t3, earth_from_moon, moon, eph, model)
        lt_spice = spice.spkezr("MOON", et0 + t3, "J2000", "CN", "EARTH")[1]
        # SPICE evaluates SSB states at a rounded et: ~30 km/s x ulp(et) ~ 3.6 mm (1.2e-11 s) of epoch
        # quantisation, which this module removes. 1 ps is therefore below the oracle's own floor.
        assert abs(leg.light_time_s - lt_spice) < 2e-11


def test_shapiro_closed_form_matches_the_line_integral(arc):
    from lunar_od import relativistic_light_time as rlt

    eph = rlt.BarycentricEphemeris(arc["et0"])
    model = rlt.LightTimeModel(frame="SSB", shapiro_bodies=("SUN",))
    t_rx = 1800.0
    r_rx = eph.body_position("EARTH", t_rx) + np.array([6.4e6, 0.0, 0.0])
    r_tx = eph.moon_state(t_rx)[:3] + np.array([1.8e6, 0.0, 0.0])
    sun = eph.body_position("SUN", t_rx)
    d = r_rx - r_tx
    length = float(np.linalg.norm(d))
    integral, _ = quad(lambda s: 1.0 / np.linalg.norm(r_tx + s * d / length - sun), 0.0, length,
                       epsabs=0.0, epsrel=1e-13)
    expected = 2.0 * eph.gm("SUN") / C ** 3 * integral
    delay, _, _ = rlt.shapiro_delay_s(model, eph, r_tx, r_rx, t_rx)
    assert delay == pytest.approx(expected, rel=1e-10)
    assert 1e-9 < delay < 1e-7                      # tens of ns: the Sun term is material for ranging


def test_barycentric_model_moves_counted_doppler_by_the_c1_materiality(arc):
    hist = arc["aug"][:, :6]
    diffs = np.array([_doppler(arc, hist, t) - _doppler(arc, hist, t, MOON_CENTRED) for t in (900.0, 1800.0, 2700.0)])
    assert np.all(np.abs(diffs) < 2e-4)             # C1: -68...+87 um/s over the tested geometries
    assert np.max(np.abs(diffs)) > 1e-6             # and not negligible against 1e-4 m/s noise


def test_production_jacobian_matches_central_differences(arc):
    t_mid = 1800.0
    analytic = two_way_counted_doppler_initial_state_jacobian(
        t_mid, STATIONS[0], arc["t"], arc["aug"], arc["eph"].earth_position(arc["t"]),
        arc["eph"].earth_velocity(arc["t"]), np.zeros((arc["t"].size, 6, 6)), BARY, et0_s=arc["et0"])
    # Steps sit above the re-propagation noise floor (~2e-9 m/s of counted Doppler at rtol 1e-12, which
    # grows as 1/h); the qualified Moon-centred Jacobian shows the same floor at these steps.
    steps = (10.0, 10.0, 10.0, 1e-2, 1e-2, 1e-2)
    numeric = np.zeros(6)
    for j, h in enumerate(steps):
        dx = np.zeros(6)
        dx[j] = h
        plus = _doppler(arc, arc["propagate"](arc["x0"] + dx), t_mid)
        minus = _doppler(arc, arc["propagate"](arc["x0"] - dx), t_mid)
        numeric[j] = (plus - minus) / (2.0 * h)
    assert np.max(np.abs(analytic - numeric)) < 1e-6 * np.linalg.norm(analytic)


def test_bls_lm_recovers_truth_from_barycentric_data_and_moon_centred_misfit(arc):
    from lunar_od.estimators import estimate_range_rate_bls_lm
    from lunar_od.measurements import compute_range_rate_residuals, generate_range_rate_measurements

    eph, t, gm = arc["eph"], arc["t"], arc["gm"]
    truth = arc["aug"][:, :6]
    vis = np.ones((t.size, len(STATIONS)), dtype=bool)
    obs, pass_geo = generate_range_rate_measurements(t, truth, STATIONS, vis, eph.earth_position,
                                                     eph.earth_velocity, arc["et0"], noise=False, arc_id=1,
                                                     range_rate_physics=BARY)
    assert obs.shape[0] > 100
    assert pass_geo.measurement_metadata["counted_doppler_model_version"] == \
        BARYCENTRIC_DSN_COUNTED_DOPPLER_MODEL_VERSION

    x_start = arc["x0"] + np.array([80.0, -60.0, 40.0, 0.02, -0.01, 0.015])
    x_hat, status, _ = estimate_range_rate_bls_lm(t, obs, x_start, pass_geo, *gm, eph.earth_position,
                                                  eph.sun_position, rtol=1e-12, atol=1e-13)
    assert np.linalg.norm(x_hat[:3] - arc["x0"][:3]) < 1e-2, status
    assert np.linalg.norm(x_hat[3:] - arc["x0"][3:]) < 1e-5, status

    # X-07 evidence: the Moon-centred model, at the truth, misfits the same data systematically.
    _, h_moon = compute_range_rate_residuals(truth, obs, replace(pass_geo, range_rate_physics=MOON_CENTRED))
    misfit = obs[:, 2] - h_moon[:, 1]
    assert np.sqrt(np.mean(misfit ** 2)) > 1e-6


@pytest.mark.parametrize("overrides", [
    {"station_state_method": LEGACY_INTERPOLATED_STATION_METHOD},
    {"output_unit": "hz"},
    {"station_clock_drift": 1e-12},
    {"mode": "geometric_instantaneous"},
])
def test_unsupported_barycentric_options_fail_closed(overrides):
    with pytest.raises(ValueError, match="barycentric_dsn"):
        replace(BARY, **overrides)


def test_scenario_loader_rejects_barycentric_with_the_ukf():
    from lunar_od.scenario_config import scenario_config_from_mapping

    with pytest.raises(ValueError, match="barycentric_dsn.*ukf"):
        scenario_config_from_mapping({"name": "wp2_1", "start_mode": "cold", "network": "single",
                                      "measurement_type": "range_rate", "estimator_type": "ukf",
                                      "range_rate_physics": "two_way_counted_doppler",
                                      "counted_doppler_model": BARYCENTRIC_DSN_COUNTED_DOPPLER_MODEL})
