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


# ---------------------------------------------------------------------------
# Slice 2: barycentric two-way range selectable from a scenario
# (measurement_model_profile='two_way_light_time_ssb').
# ---------------------------------------------------------------------------
def _two_way_payload(**overrides):
    payload = {"name": "wp2_1_two_way", "measurement_type": "two_way_range", "estimator_type": "bls_lm",
               "start_mode": "cold", "network": "multi", "measurement_model_profile": "two_way_light_time_ssb"}
    payload.update(overrides)
    return payload


def test_scenario_accepts_the_barycentric_two_way_range_profile():
    from lunar_od.scenario_config import scenario_config_from_mapping

    config = scenario_config_from_mapping(_two_way_payload())
    assert config.measurement_model_profile == "two_way_light_time_ssb"


@pytest.mark.parametrize("overrides, match", [
    ({"measurement_type": "position"}, "two_way_light_time_ssb"),
    ({"measurement_type": "range_rate"}, "two_way_light_time_ssb"),
    ({"two_way_range_convention": "raw_half_round_trip"}, "delay_calibrated"),
    ({"estimator_type": "ukf"}, "two_way_range"),
])
def test_scenario_rejects_unsupported_barycentric_two_way_range(overrides, match):
    from lunar_od.scenario_config import scenario_config_from_mapping

    with pytest.raises(ValueError, match=match):
        scenario_config_from_mapping(_two_way_payload(**overrides))


def test_scenario_arc_builder_routes_two_way_range_to_the_barycentric_model(arc):
    from lunar_od.estimators import estimate_two_way_range_bls_lm
    from lunar_od.scenarios import build_measurement_arcs
    from lunar_od.two_way_range import TwoWayRangeConfig, compute_two_way_range_residuals

    eph, t, gm = arc["eph"], arc["t"], arc["gm"]
    truth = arc["aug"][:, :6]
    arcs = build_measurement_arcs("two_way_range", t, truth, [0], [t.size - 1],
                                  np.ones((t.size, len(STATIONS)), dtype=bool), STATIONS, eph.earth_position,
                                  eph.earth_velocity, arc["et0"], noise=False,
                                  measurement_model_profile="two_way_light_time_ssb",
                                  two_way_range=TwoWayRangeConfig())
    assert len(arcs) == 1
    prepared = arcs[0]
    assert prepared.pass_geo.measurement_model_profile == "two_way_light_time_ssb"

    x_start = arc["x0"] + np.array([80.0, -60.0, 40.0, 0.02, -0.01, 0.015])
    x_hat, status, _ = estimate_two_way_range_bls_lm(prepared.t_pass_s, prepared.obs_data, x_start,
                                                     prepared.pass_geo, *gm, eph.earth_position,
                                                     eph.sun_position, rtol=1e-12, atol=1e-13)
    assert np.linalg.norm(x_hat[:3] - arc["x0"][:3]) < 1e-2, status

    # The Moon-centred M3 model at the truth misfits the same data by the barycentric terms:
    # Sun Shapiro ~ 2GM/c^3 * r12/AU ~ 7.6 m, the SSB-frame term <= (v_SSB/c)^2 * rho ~ 4.5 m,
    # TT interval ~ 0.1 m and Earth Shapiro ~ 0.04 m.
    moon_geo = replace(prepared.pass_geo, measurement_model_profile=None)
    residuals, _ = compute_two_way_range_residuals(truth, prepared.obs_data, moon_geo)
    assert 7.0 < float(np.mean(residuals)) < 12.5

    # Decomposition: the Shapiro share equals the oracle-tested closed form within 0.6 m of 7.6 m.
    from lunar_od import relativistic_light_time as rlt
    no_shapiro = rlt.LightTimeModel(frame="SSB", interval_time_scale="TT")
    _, stations, eph_b = rlt._pass_context(prepared.pass_geo)
    kw = rlt.pass_station_kwargs(prepared.pass_geo)
    sc = rlt.SpacecraftHistory(prepared.pass_geo.t_s, truth)
    row = prepared.obs_data[prepared.obs_data.shape[0] // 2]
    sid = int(row[2]) - 1
    full = rlt.two_way_range_m(rlt.solve_two_way(row[0], stations[sid], sc, eph_b, rlt.DSN_LIGHT_TIME_MODEL, **kw[sid]))
    bare = rlt.two_way_range_m(rlt.solve_two_way(row[0], stations[sid], sc, eph_b, no_shapiro, **kw[sid]))
    assert 7.0 < full - bare < 8.2


# ---------------------------------------------------------------------------
# Slice 3: barycentric one-way light time for the position observables
# (measurement_model_profile='one_way_light_time_ssb' / '..._ssb_aberrated').
# ---------------------------------------------------------------------------
def _moon_centre_position_row(arc, profile, k=60):
    """Generate one Goldstone row with the 'spacecraft' at the Moon centre, at grid index ``k``."""
    from lunar_od.measurements import generate_position_measurements

    t = arc["t"]
    vis = np.zeros((t.size, 1), dtype=bool)
    vis[k, 0] = True
    _obs, pass_geo, clean = generate_position_measurements(
        t, np.zeros((t.size, 6)), STATIONS[:1], vis, arc["eph"].earth_position, arc["eph"].earth_velocity,
        arc["et0"], noise=False, measurement_model_profile=profile)
    assert clean.shape[0] == 1
    return clean[0], pass_geo


@pytest.mark.parametrize("profile, abcorr", [("one_way_light_time_ssb", "CN"),
                                             ("one_way_light_time_ssb_aberrated", "CN+S")])
def test_ssb_position_observable_matches_spice(arc, profile, abcorr):
    """G9/G10: range = c x (SPICE CN light time + Moyer 8-38 Sun/Earth Shapiro); az/el = SPICE apparent direction."""
    from lunar_od.geometry import ecef2razel_sez, wrap_to_pi

    spice = arc["spice"]
    row, pass_geo = _moon_centre_position_row(arc, profile)
    et_rx = arc["et0"] + row[0]
    station = STATIONS[0]
    obspos_km = np.asarray(station.r_ecef_m, dtype=float) / 1000.0
    state, lt = spice.spkcpo("MOON", et_rx, "J2000", "OBSERVER", abcorr, obspos_km, "EARTH", "ITRF93")
    rho_j2000 = np.asarray(state[:3]) * 1000.0
    rot = np.asarray(spice.pxform("J2000", "ITRF93", et_rx))
    az_ref, el_ref, _rng = ecef2razel_sez(rot @ rho_j2000, station.lat_rad, station.lon_rad)

    # Shapiro closed form, geometry straight from SPICE (bodies at the receive epoch, as Moyer 8-38 uses).
    r_tx = np.asarray(spice.spkpos("MOON", et_rx - lt, "J2000", "NONE", "SSB")[0]) * 1000.0
    r_rx = (np.asarray(spice.spkpos("EARTH", et_rx, "J2000", "NONE", "SSB")[0]) * 1000.0
            + np.asarray(spice.pxform("ITRF93", "J2000", et_rx)) @ (obspos_km * 1000.0))
    r12 = np.linalg.norm(r_rx - r_tx)
    shapiro = 0.0
    for body in ("SUN", "EARTH"):
        b = np.asarray(spice.spkpos(body, et_rx, "J2000", "NONE", "SSB")[0]) * 1000.0
        r1, r2 = np.linalg.norm(r_tx - b), np.linalg.norm(r_rx - b)
        gm = float(spice.bodvrd(body, "GM", 1)[1][0]) * 1e9
        shapiro += 2.0 * gm / C ** 3 * np.log((r1 + r2 + r12) / (r1 + r2 - r12))
    assert 7.0 < C * shapiro < 8.5          # the Sun term dominates (C1: 7.3-7.8 m)

    if abcorr == "CN":
        assert abs(row[1] - C * (lt + shapiro)) < 0.01
    assert abs(wrap_to_pi(row[2] - az_ref)) < 1e-9
    assert abs(row[3] - el_ref) < 1e-9
    assert pass_geo.measurement_model_profile == profile


def test_ssb_position_bls_recovers_truth_and_moon_centred_misfit(arc):
    """G11/G12: BLS-LM recovers the truth from SSB position data; the Moon-centred profile misfits it."""
    from lunar_od.estimators import estimate_position_bls_lm
    from lunar_od.measurements import compute_position_residuals, generate_position_measurements

    eph, t, gm = arc["eph"], arc["t"], arc["gm"]
    truth = arc["aug"][:, :6]
    vis = np.ones((t.size, len(STATIONS)), dtype=bool)
    obs, pass_geo, _ = generate_position_measurements(t, truth, STATIONS, vis, eph.earth_position,
                                                      eph.earth_velocity, arc["et0"], noise=False, arc_id=1,
                                                      measurement_model_profile="one_way_light_time_ssb")
    assert obs.shape[0] > 100
    x_start = arc["x0"] + np.array([80.0, -60.0, 40.0, 0.02, -0.01, 0.015])
    x_hat, status, _ = estimate_position_bls_lm(t, obs, x_start, pass_geo, *gm, eph.earth_position,
                                                eph.sun_position, rtol=1e-12, atol=1e-13)
    assert np.linalg.norm(x_hat[:3] - arc["x0"][:3]) < 1e-2, status

    # G12 (amended after the result, see the predeclaration): the Moon-centred model treats the MCI frame as
    # inertial, so it misses the Moon's SSB motion over the light time, -(v_Moon,SSB . u) tau (up to ~38 km,
    # geometry dependent). The remainder is the slice-2 budget: Sun Shapiro ~7.6 m + second order ~4 m.
    from lunar_od import relativistic_light_time as rlt

    moon_geo = replace(pass_geo, measurement_model_profile="one_way_light_time")
    residuals, _ = compute_position_residuals(truth, obs, moon_geo)
    misfit = np.asarray(residuals).reshape(-1, 3)[:, 0]
    bary = rlt.BarycentricEphemeris(arc["et0"])
    first_order = np.empty(obs.shape[0])
    for i, row in enumerate(obs):
        k, station = int(row[5]) - 1, STATIONS[int(row[4]) - 1]
        los = truth[k, :3] - (eph.earth_position(np.array([row[0]]))[0]
                              + pass_geo.x_j2000_to_itrf93[k, :3, :3].T @ station.r_ecef_m)
        first_order[i] = -(bary.moon_state(row[0])[3:6] @ los) / C
    assert np.max(np.abs(misfit)) > 5000.0
    assert np.max(np.abs(misfit - first_order)) < 15.0


def _position_payload(**overrides):
    payload = {"name": "wp2_1_position", "measurement_type": "position", "estimator_type": "bls_lm",
               "start_mode": "cold", "network": "multi", "measurement_model_profile": "one_way_light_time_ssb"}
    payload.update(overrides)
    return payload


@pytest.mark.parametrize("profile", ["one_way_light_time_ssb", "one_way_light_time_ssb_aberrated"])
def test_scenario_accepts_the_ssb_position_profiles(profile):
    from lunar_od.scenario_config import scenario_config_from_mapping

    assert scenario_config_from_mapping(
        _position_payload(measurement_model_profile=profile)).measurement_model_profile == profile


@pytest.mark.parametrize("overrides", [{"measurement_type": "range_rate"}, {"estimator_type": "ukf"}])
def test_scenario_rejects_unsupported_ssb_position_profiles(overrides):
    from lunar_od.scenario_config import scenario_config_from_mapping

    with pytest.raises(ValueError):
        scenario_config_from_mapping(_position_payload(**overrides))
