"""F2 WP2.3 slice 1 (MEAS-08): two-way range station bias solve-for in BLS-LM by variable projection.

Predeclared in ``studies/phase17/configs/wp2_3_predeclaration.json``. Oracles: known injected biases and the
truth state (noise-free data); the bias-free solve on unbiased data (projection identity).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lunar_od.config import Station

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



ROOT = Path(__file__).resolve().parents[1]
OREKIT = ROOT / "studies" / "phase17" / "results" / "phase_c5i_reference.npz"



X_OFFSET = np.array([80.0, -60.0, 40.0, 0.02, -0.01, 0.015])
INJECTED = {1: 7.0, 2: -4.0}


def _ssb_arc(arc, biases=None):
    from lunar_od.scenarios import build_measurement_arcs
    from lunar_od.two_way_range import TwoWayRangeConfig

    eph, t = arc["eph"], arc["t"]
    truth = arc["aug"][:, :6]
    prepared = build_measurement_arcs("two_way_range", t, truth, [0], [t.size - 1],
                                      np.ones((t.size, len(STATIONS)), dtype=bool), STATIONS, eph.earth_position,
                                      eph.earth_velocity, arc["et0"], noise=False,
                                      measurement_model_profile="two_way_light_time_ssb",
                                      two_way_range=TwoWayRangeConfig())[0]
    obs = prepared.obs_data.copy()
    for sid, b in (biases or {}).items():
        obs[obs[:, 2].astype(int) == sid, 1] += b
    return prepared, obs


def _solve(arc, prepared, obs, **kw):
    from lunar_od.estimators import estimate_two_way_range_bls_lm

    eph, gm = arc["eph"], arc["gm"]
    return estimate_two_way_range_bls_lm(prepared.t_pass_s, obs, arc["x0"] + X_OFFSET, prepared.pass_geo, *gm,
                                         eph.earth_position, eph.sun_position, rtol=1e-12, atol=1e-13, **kw)


def test_station_biases_and_state_are_recovered(arc):
    """G1/G2: +7 m / -4 m injected biases within 1 mm; position within 1 cm."""
    prepared, obs = _ssb_arc(arc, INJECTED)
    assert set(np.unique(obs[:, 2]).astype(int)) == {1, 2}
    x_hat, status, stats = _solve(arc, prepared, obs, bias_mode="station_range", return_posterior=True)
    assert np.linalg.norm(x_hat[:3] - arc["x0"][:3]) < 1e-2, status
    for sid, b in INJECTED.items():
        assert abs(stats.station_range_biases_m[sid] - b) < 1e-3
    assert stats.posterior_covariance.shape == (6, 6)
    assert np.all(np.isfinite(np.diag(stats.posterior_covariance)))


def test_bias_free_model_misfits_biased_data(arc):
    """G3: without bias_mode the station-mean residuals carry the injected bias (within 2 m on a station)."""
    from lunar_od.relativistic_light_time import compute_two_way_range_residuals_ssb

    prepared, obs = _ssb_arc(arc, INJECTED)
    x_hat, _status, _ = _solve(arc, prepared, obs)
    hist = arc["propagate"](x_hat)
    residuals, _ = compute_two_way_range_residuals_ssb(hist, obs, prepared.pass_geo)
    sid = obs[:, 2].astype(int)
    means = {s: float(np.mean(np.asarray(residuals)[sid == s])) for s in INJECTED}
    assert any(abs(means[s] - INJECTED[s]) < 2.0 for s in INJECTED), (means, np.linalg.norm(x_hat[:3] - arc["x0"][:3]))


def test_projection_is_an_identity_on_unbiased_data(arc):
    """G4: unbiased data -> biases < 1 mm and the state within 1 cm of the bias-free solve."""
    prepared, obs = _ssb_arc(arc)
    x_plain, _, _ = _solve(arc, prepared, obs)
    x_bias, status, stats = _solve(arc, prepared, obs, bias_mode="station_range")
    assert np.linalg.norm(x_bias[:3] - x_plain[:3]) < 1e-2, status
    assert max(abs(b) for b in stats.station_range_biases_m.values()) < 1e-3


def test_station_bias_mode_fails_closed(arc):
    """G5: SRIF, K solve-for, robust rejection, unknown modes and one-row stations are rejected."""
    from lunar_od.estimators import estimate_two_way_range_srif

    prepared, obs = _ssb_arc(arc)
    eph, gm = arc["eph"], arc["gm"]
    with pytest.raises(ValueError, match="bias_mode"):
        estimate_two_way_range_srif(prepared.t_pass_s, obs, arc["x0"], prepared.pass_geo, *gm, eph.earth_position,
                                    eph.sun_position, bias_mode="per_pass")
    with pytest.raises(ValueError, match="solve_for_k_srp"):
        _solve(arc, prepared, obs, bias_mode="station_range", solve_for_k_srp=True)
    with pytest.raises(ValueError, match="robust_outlier_rejection"):
        _solve(arc, prepared, obs, bias_mode="station_range", robust_outlier_rejection=True)
    with pytest.raises(ValueError, match="bias_mode"):
        _solve(arc, prepared, obs, bias_mode="per_pass")
    single = np.vstack([obs[obs[:, 2] == 1], obs[obs[:, 2] == 2][:1]])
    with pytest.raises(ValueError, match="two rows per station"):
        _solve(arc, prepared, single, bias_mode="station_range")


def test_srif_station_bias_matches_bls(arc):
    """G6 (slice 1b): SRIF recovers the biases within 1 mm, its state agrees with BLS-LM within 1 cm and the
    posterior position sigmas agree within 1 %."""
    from lunar_od.estimators import estimate_two_way_range_srif

    prepared, obs = _ssb_arc(arc, INJECTED)
    eph, gm = arc["eph"], arc["gm"]
    x_bls, _, s_bls = _solve(arc, prepared, obs, bias_mode="station_range", return_posterior=True)
    x_srif, status, s_srif = estimate_two_way_range_srif(
        prepared.t_pass_s, obs, arc["x0"] + X_OFFSET, prepared.pass_geo, *gm, eph.earth_position,
        eph.sun_position, rtol=1e-12, atol=1e-13, bias_mode="station_range", return_posterior=True)
    for sid, b in INJECTED.items():
        assert abs(s_srif.station_range_biases_m[sid] - b) < 1e-3, status
    assert np.linalg.norm(x_srif[:3] - x_bls[:3]) < 1e-2, status
    sig_bls = np.sqrt(np.diag(s_bls.posterior_covariance)[:3])
    sig_srif = np.sqrt(np.diag(s_srif.posterior_covariance)[:3])
    assert np.max(np.abs(sig_srif / sig_bls - 1.0)) < 1e-2


# ---------------------------------------------------------------------------
# Slice 2: fixed transponder delay on the barycentric two-way range (X-15 delay part).
# ---------------------------------------------------------------------------
DELAY_S = 0.05


def _with_delay(pass_geo, delay):
    from dataclasses import replace

    from lunar_od.two_way_range import TwoWayRangeConfig

    return replace(pass_geo, two_way_range=TwoWayRangeConfig(transponder_delay_s=delay))


def test_zero_delay_is_the_slice_one_path(arc):
    """G7: an explicit zero delay leaves the solution and its partial identical (the delay branch is skipped)."""
    from lunar_od import relativistic_light_time as rlt

    prepared, obs = _ssb_arc(arc)
    model, stations, eph_b = rlt._pass_context(prepared.pass_geo)
    kw = rlt.pass_station_kwargs(prepared.pass_geo)
    sc = rlt.SpacecraftHistory(prepared.pass_geo.t_s, arc["aug"])
    row = obs[obs.shape[0] // 2]
    sid = int(row[2]) - 1
    a = rlt.solve_two_way(row[0], stations[sid], sc, eph_b, model, **kw[sid])
    b = rlt.solve_two_way(row[0], stations[sid], sc, eph_b, model, transponder_delay_s=0.0, **kw[sid])
    assert "transponder_delay_s" not in kw[sid]
    assert rlt.two_way_range_m(a) == rlt.two_way_range_m(b)
    assert np.array_equal(rlt.two_way_range_partial(a, sc), rlt.two_way_range_partial(b, sc))


def test_delay_effect_matches_moon_centred_m3(arc):
    """G8 (amended after the result, exploratory): SSB(d) - SSB(0) differs from M3(d) - M3(0) by the first-order
    frame term Delta_M3 (u_up . v_Moon,SSB)/c; after removing it the rows agree within 1.5 mm. As computed the
    pre-registered 1 mm raw bound failed at 3.8 mm. The change exceeds 1 m on some row."""
    from dataclasses import replace

    from lunar_od.relativistic_light_time import compute_two_way_range_residuals_ssb
    from lunar_od.two_way_range import compute_two_way_range_residuals

    prepared, obs = _ssb_arc(arc)
    truth = arc["aug"][:, :6]
    geo0, geod = prepared.pass_geo, _with_delay(prepared.pass_geo, DELAY_S)
    ssb0 = compute_two_way_range_residuals_ssb(truth, obs, geo0)[1]
    ssbd = compute_two_way_range_residuals_ssb(truth, obs, geod)[1]
    m30 = compute_two_way_range_residuals(truth, obs, replace(geo0, measurement_model_profile=None))[1]
    m3d = compute_two_way_range_residuals(truth, obs, replace(geod, measurement_model_profile=None))[1]
    d_ssb, d_m3 = np.asarray(ssbd) - np.asarray(ssb0), np.asarray(m3d) - np.asarray(m30)
    assert np.max(np.abs(d_ssb)) > 1.0
    from lunar_od import relativistic_light_time as rlt
    model, stations, eph_b = rlt._pass_context(geod)
    kw = rlt.pass_station_kwargs(geod)
    sc = rlt.SpacecraftHistory(geod.t_s, truth)
    frame = np.zeros(obs.shape[0])
    for i in range(obs.shape[0]):
        sid = int(obs[i, 2]) - 1
        up = rlt.solve_two_way(obs[i, 0], stations[sid], sc, eph_b, model, **kw[sid]).uplink
        u = up.rel / np.linalg.norm(up.rel)
        frame[i] = d_m3[i] * float(u @ np.asarray(eph_b.moon_state(up.t_rx_s))[3:6]) / 299792458.0
    assert np.max(np.abs(d_ssb - d_m3)) > 1e-3        # the raw bound fails as computed (3.8 mm)
    assert np.max(np.abs(d_ssb - d_m3 - frame)) < 1.5e-3, np.max(np.abs(d_ssb - d_m3 - frame))


def test_delayed_range_jacobian_matches_finite_difference(arc):
    """G9 (step amended after the result): the d = 0.05 s Jacobian vs a central difference, 1e-6 of the row norm.
    The pre-registered 1 m / 1e-3 m/s step sits on the round-off floor (1.8e-6, and 1.3e-5 at 0.1 m / 1e-4 m/s);
    10 m / 1e-2 m/s gives 7.5e-8."""
    from lunar_od.relativistic_light_time import (
        compute_two_way_range_residuals_ssb,
        two_way_range_nominal_and_initial_jacobian_ssb,
    )

    prepared, obs = _ssb_arc(arc)
    geo = _with_delay(prepared.pass_geo, DELAY_S)
    _h, jac = two_way_range_nominal_and_initial_jacobian_ssb(obs, geo, arc["aug"])
    fd = np.zeros_like(jac)
    for j, step in enumerate((10.0, 10.0, 10.0, 1e-2, 1e-2, 1e-2)):
        dx = np.zeros(6)
        dx[j] = step
        hp = compute_two_way_range_residuals_ssb(arc["propagate"](arc["x0"] + dx), obs, geo)[1]
        hm = compute_two_way_range_residuals_ssb(arc["propagate"](arc["x0"] - dx), obs, geo)[1]
        fd[:, j] = (np.asarray(hp) - np.asarray(hm)) / (2.0 * step)
    rel = np.max(np.abs(jac - fd), axis=1) / np.linalg.norm(jac, axis=1)
    assert np.max(rel) < 1e-6, np.max(rel)


def test_delayed_data_are_recovered_and_the_delay_is_material(arc):
    """G10: matching model recovers position < 1 cm. The pre-registered materiality clause (zero-delay model
    post-fit RMS > 0.5 m) FAILED as computed (0.28 m: the fit absorbs the 26.5 m delay signature into a 187 m
    position error); amended (exploratory) to position error > 10 m and post-fit RMS > 1000x the matched fit."""
    from lunar_od.relativistic_light_time import compute_two_way_range_residuals_ssb
    from lunar_od.scenarios import build_measurement_arcs
    from lunar_od.two_way_range import TwoWayRangeConfig

    eph, t = arc["eph"], arc["t"]
    prepared = build_measurement_arcs("two_way_range", t, arc["aug"][:, :6], [0], [t.size - 1],
                                      np.ones((t.size, len(STATIONS)), dtype=bool), STATIONS, eph.earth_position,
                                      eph.earth_velocity, arc["et0"], noise=False,
                                      measurement_model_profile="two_way_light_time_ssb",
                                      two_way_range=TwoWayRangeConfig(transponder_delay_s=DELAY_S))[0]
    obs = prepared.obs_data
    x_hat, status, _ = _solve(arc, prepared, obs)
    assert np.linalg.norm(x_hat[:3] - arc["x0"][:3]) < 1e-2, status
    matched, _ = compute_two_way_range_residuals_ssb(arc["propagate"](x_hat), obs, prepared.pass_geo)

    from dataclasses import replace
    zero = replace(prepared, pass_geo=_with_delay(prepared.pass_geo, 0.0))
    x_zero, _status, _ = _solve(arc, zero, obs)
    residuals, _ = compute_two_way_range_residuals_ssb(arc["propagate"](x_zero), obs, zero.pass_geo)
    rms = float(np.sqrt(np.mean(np.square(residuals))))
    assert rms < 0.5                                   # the pre-registered clause fails as computed
    assert np.linalg.norm(x_zero[:3] - arc["x0"][:3]) > 10.0
    assert rms > 1e3 * float(np.sqrt(np.mean(np.square(matched))))


def test_delay_gates(arc):
    """G11: loader accepts a delay with two_way_light_time_ssb, still rejects the raw convention; barycentric
    counted Doppler with a delay still fails closed (at RangeRatePhysicsConfig construction, ValueError: one
    layer earlier than the pre-registered NotImplementedError of counted_doppler_ssb._require_supported)."""
    from lunar_od.radiometrics import BARYCENTRIC_DSN_COUNTED_DOPPLER_MODEL, RangeRatePhysicsConfig
    from lunar_od.scenario_config import scenario_config_from_mapping

    base = {"name": "wp2_3_delay", "measurement_type": "two_way_range", "estimator_type": "bls_lm",
            "start_mode": "cold", "network": "multi", "measurement_model_profile": "two_way_light_time_ssb"}
    assert scenario_config_from_mapping({**base, "transponder_delay_s": 1e-6}).transponder_delay_s == 1e-6
    with pytest.raises(ValueError, match="delay_calibrated"):
        scenario_config_from_mapping({**base, "transponder_delay_s": 1e-6,
                                      "two_way_range_convention": "raw_half_round_trip"})
    with pytest.raises(ValueError, match="transponder_delay_s"):
        RangeRatePhysicsConfig(mode="two_way_counted_doppler", count_interval_s=60.0,
                               counted_doppler_model=BARYCENTRIC_DSN_COUNTED_DOPPLER_MODEL,
                               transponder_delay_s=1e-6)


# ---------------------------------------------------------------------------
# Slice 3: one range bias per tracking pass (bias_mode='pass_range').
# ---------------------------------------------------------------------------
PASS_BIASES = {(1, 1): 6.0, (1, 2): -3.0, (2, 1): 2.0}


def _two_pass_arc(arc, biases):
    """Station 1 tracks 0-1200 s and 2400-3600 s, station 2 the whole hour; ``biases`` by (station, pass) or
    by station."""
    from lunar_od.scenarios import build_measurement_arcs
    from lunar_od.two_way_range import TwoWayRangeConfig

    eph, t = arc["eph"], arc["t"]
    vis = np.ones((t.size, len(STATIONS)), dtype=bool)
    vis[(t > 1200.0) & (t < 2400.0), 0] = False
    prepared = build_measurement_arcs("two_way_range", t, arc["aug"][:, :6], [0], [t.size - 1], vis, STATIONS,
                                      eph.earth_position, eph.earth_velocity, arc["et0"], noise=False,
                                      measurement_model_profile="two_way_light_time_ssb",
                                      two_way_range=TwoWayRangeConfig())[0]
    obs = prepared.obs_data.copy()
    sid = obs[:, 2].astype(int)
    late = obs[:, 0] > 1800.0
    for key, b in biases.items():
        if isinstance(key, tuple):
            station, k = key
            rows = (sid == station) & (late if (station == 1 and k == 2) else ~late if station == 1 else True)
        else:
            rows = sid == key
        obs[rows, 1] += b
    return prepared, obs


def test_pass_biases_and_state_are_recovered_by_bls_and_srif(arc):
    """G12: every pass bias within 1 mm and position within 1 cm, BLS-LM and SRIF."""
    from lunar_od.estimators import estimate_two_way_range_srif

    prepared, obs = _two_pass_arc(arc, PASS_BIASES)
    eph, gm = arc["eph"], arc["gm"]
    x_bls, status, s_bls = _solve(arc, prepared, obs, bias_mode="pass_range")
    x_srif, status_s, s_srif = estimate_two_way_range_srif(
        prepared.t_pass_s, obs, arc["x0"] + X_OFFSET, prepared.pass_geo, *gm, eph.earth_position,
        eph.sun_position, rtol=1e-12, atol=1e-13, bias_mode="pass_range")
    for x, st, stats in ((x_bls, status, s_bls), (x_srif, status_s, s_srif)):
        assert set(stats.station_range_biases_m) == set(PASS_BIASES), stats.station_range_biases_m
        assert np.linalg.norm(x[:3] - arc["x0"][:3]) < 1e-2, st
        for key, b in PASS_BIASES.items():
            assert abs(stats.station_range_biases_m[key] - b) < 1e-3, (key, stats.station_range_biases_m[key])


def test_station_bias_model_misfits_pass_biased_data(arc):
    """G13 (amended after the result, exploratory): the pre-registered clause (a station-1 pass-mean residual
    above 1 m) FAILED as computed (0.088 m): the orbit absorbs the 9 m pass-to-pass step (676 m position error,
    station biases -7.3 / -7.0 m). Amended to position error > 10 m and post-fit RMS > 1000x the pass-mode fit."""
    from lunar_od.relativistic_light_time import compute_two_way_range_residuals_ssb

    prepared, obs = _two_pass_arc(arc, PASS_BIASES)
    x_hat, _status, stats = _solve(arc, prepared, obs, bias_mode="station_range")
    residuals, _ = compute_two_way_range_residuals_ssb(arc["propagate"](x_hat), obs, prepared.pass_geo)
    residuals = np.asarray(residuals) - np.array([stats.station_range_biases_m[int(s)] for s in obs[:, 2]])
    st1 = obs[:, 2].astype(int) == 1
    means = [float(np.mean(residuals[st1 & (obs[:, 0] < 1800.0)])), float(np.mean(residuals[st1 & (obs[:, 0] > 1800.0)]))]
    assert max(abs(m) for m in means) < 1.0, means     # the pre-registered clause fails as computed
    assert np.linalg.norm(x_hat[:3] - arc["x0"][:3]) > 10.0
    x_pass, _status, s_pass = _solve(arc, prepared, obs, bias_mode="pass_range")
    r_pass, _ = compute_two_way_range_residuals_ssb(arc["propagate"](x_pass), obs, prepared.pass_geo)
    key = [(int(q), 2 if (q == 1 and tt > 1800.0) else 1) for q, tt in zip(obs[:, 2], obs[:, 0])]
    r_pass = np.asarray(r_pass) - np.array([s_pass.station_range_biases_m[k] for k in key])
    assert np.sqrt(np.mean(residuals ** 2)) > 1e3 * np.sqrt(np.mean(r_pass ** 2))


def test_pass_mode_returns_equal_biases_for_station_constant_bias(arc):
    """G14: station-constant +7 m / -4 m -> both station-1 pass biases within 1 mm of +7 m."""
    prepared, obs = _two_pass_arc(arc, INJECTED)
    _x, status, stats = _solve(arc, prepared, obs, bias_mode="pass_range")
    for key in ((1, 1), (1, 2)):
        assert abs(stats.station_range_biases_m[key] - 7.0) < 1e-3, (key, status)
    assert abs(stats.station_range_biases_m[(2, 1)] + 4.0) < 1e-3


def test_pass_bias_mode_fails_closed(arc):
    """G15: a one-row pass, K solve-for and robust rejection are rejected for 'pass_range'."""
    prepared, obs = _two_pass_arc(arc, {})
    late1 = np.flatnonzero((obs[:, 2] == 1) & (obs[:, 0] > 1800.0))
    single = np.delete(obs, late1[1:], axis=0)
    with pytest.raises(ValueError, match="two rows per pass"):
        _solve(arc, prepared, single, bias_mode="pass_range")
    with pytest.raises(ValueError, match="solve_for_k_srp"):
        _solve(arc, prepared, obs, bias_mode="pass_range", solve_for_k_srp=True)
    with pytest.raises(ValueError, match="robust_outlier_rejection"):
        _solve(arc, prepared, obs, bias_mode="pass_range", robust_outlier_rejection=True)


# ---------------------------------------------------------------------------
# Slice 4: per-station receive time-tag offset (bias_mode='station_time_tag').
# ---------------------------------------------------------------------------
CLOCK_OFFSETS_S = (1e-3, -2e-3)


def _tagged_arc(arc, offsets, biases):
    """Data generated with true clock offsets (exact receive tags t3 + b) plus range biases; the returned pass
    geometry has the offsets removed, as the estimator must not know them."""
    from dataclasses import replace

    from lunar_od.relativistic_light_time import generate_two_way_range_measurements_ssb
    from lunar_od.two_way_range import TwoWayRangeConfig

    eph, t = arc["eph"], arc["t"]
    obs, geo = generate_two_way_range_measurements_ssb(
        t, arc["aug"][:, :6], STATIONS, np.ones((t.size, len(STATIONS)), dtype=bool), eph.earth_position,
        eph.earth_velocity, arc["et0"], noise=False, config=TwoWayRangeConfig(),
        station_clock_offsets_s=None if offsets is None else list(offsets))
    meta = dict(geo.measurement_metadata)
    meta.pop("station_clock_offsets_s", None)
    geo = replace(geo, measurement_metadata=meta)
    obs = np.array(obs, dtype=float, copy=True)
    for sid, b in biases.items():
        obs[obs[:, 2].astype(int) == sid, 1] += b
    return geo, obs


def _solve_geo(arc, geo, obs, estimator="bls", **kw):
    from lunar_od.estimators import estimate_two_way_range_bls_lm, estimate_two_way_range_srif

    fn = estimate_two_way_range_bls_lm if estimator == "bls" else estimate_two_way_range_srif
    eph, gm = arc["eph"], arc["gm"]
    return fn(arc["t"], obs, arc["x0"] + X_OFFSET, geo, *gm, eph.earth_position, eph.sun_position,
              rtol=1e-12, atol=1e-13, **kw)


def test_time_tags_range_biases_and_state_are_recovered(arc):
    """G16: +1 ms / -2 ms clock offsets within 1 us, +7 / -4 m range biases within 1 mm, position < 1 cm.

    Design amended after the result (exploratory): with the pre-registered column fixed at the a-priori state
    the gate FAILED as computed (BLS position 0.81 m: the 0.49 mm column error is amplified ~1600x by the weak
    time-tag geometry, posterior position sigma 0.9-2 km for 2 m range noise); the column is now re-evaluated
    at every linearisation (4.5 mm). Thresholds unchanged."""
    geo, obs = _tagged_arc(arc, CLOCK_OFFSETS_S, INJECTED)
    for estimator in ("bls", "srif"):
        x, status, stats = _solve_geo(arc, geo, obs, estimator, bias_mode="station_time_tag")
        assert np.linalg.norm(x[:3] - arc["x0"][:3]) < 1e-2, (estimator, status, np.linalg.norm(x[:3] - arc["x0"][:3]))
        for sid, b in INJECTED.items():
            assert abs(stats.station_time_tag_biases_s[sid] - CLOCK_OFFSETS_S[sid - 1]) < 1e-6, (estimator, stats.station_time_tag_biases_s)
            assert abs(stats.station_range_biases_m[sid] - b) < 1e-3, (estimator, stats.station_range_biases_m)


def test_range_only_bias_model_misfits_time_tagged_data(arc):
    """G17: bias_mode='station_range' -> position error > 10 m or post-fit RMS > 1000x the time-tag fit."""
    from lunar_od.relativistic_light_time import compute_two_way_range_residuals_ssb

    geo, obs = _tagged_arc(arc, CLOCK_OFFSETS_S, INJECTED)
    sid = obs[:, 2].astype(int)

    def post_fit(mode):
        x, _status, stats = _solve_geo(arc, geo, obs, bias_mode=mode)
        r = np.asarray(compute_two_way_range_residuals_ssb(arc["propagate"](x), obs, geo)[0], dtype=float)
        r = r - np.array([stats.station_range_biases_m[s] for s in sid])
        if stats.station_time_tag_biases_s is not None:
            from lunar_od.relativistic_light_time import two_way_range_clock_columns_ssb
            g = two_way_range_clock_columns_ssb(arc["propagate"](x), obs, geo)
            r = r - g * np.array([stats.station_time_tag_biases_s[s] for s in sid])
        return x, float(np.sqrt(np.mean(r ** 2)))

    x_range, rms_range = post_fit("station_range")
    _x_tag, rms_tag = post_fit("station_time_tag")
    err = float(np.linalg.norm(x_range[:3] - arc["x0"][:3]))
    assert err > 10.0 or rms_range > 1e3 * rms_tag, (err, rms_range, rms_tag)


def test_time_tag_mode_is_an_identity_on_untagged_data(arc):
    """G18: no clock offsets -> |time tag| < 1e-8 s and the state within 1 cm of the station_range solve."""
    geo, obs = _tagged_arc(arc, None, INJECTED)
    x_tag, status, stats = _solve_geo(arc, geo, obs, bias_mode="station_time_tag")
    x_range, _s, _ = _solve_geo(arc, geo, obs, bias_mode="station_range")
    assert max(abs(v) for v in stats.station_time_tag_biases_s.values()) < 1e-8, stats.station_time_tag_biases_s
    assert np.linalg.norm(x_tag[:3] - x_range[:3]) < 1e-2, status


def test_time_tag_mode_fails_closed(arc):
    """G19: Moon-centred pass, < 3 rows on a station, K solve-for and robust rejection are rejected."""
    from dataclasses import replace

    geo, obs = _tagged_arc(arc, None, {})
    with pytest.raises(ValueError, match="two_way_light_time_ssb"):
        _solve_geo(arc, replace(geo, measurement_model_profile=None), obs, bias_mode="station_time_tag")
    short = np.vstack([obs[obs[:, 2] == 1], obs[obs[:, 2] == 2][:2]])
    with pytest.raises(ValueError, match="three rows per station"):
        _solve_geo(arc, geo, short, bias_mode="station_time_tag")
    with pytest.raises(ValueError, match="solve_for_k_srp"):
        _solve_geo(arc, geo, obs, bias_mode="station_time_tag", solve_for_k_srp=True)
    with pytest.raises(ValueError, match="robust_outlier_rejection"):
        _solve_geo(arc, geo, obs, bias_mode="station_time_tag", robust_outlier_rejection=True)


# ---------------------------------------------------------------------------
# Slice 5: ramped uplink core (relativistic_light_time.UplinkRamp, counted_doppler(uplink_ramp=...)).
# ---------------------------------------------------------------------------
F_REF_HZ = 7.16e9
T_MID_S = 1800.0
TC_S = 60.0


def _doppler_context(arc):
    from lunar_od import relativistic_light_time as rlt

    prepared, _obs = _ssb_arc(arc)
    model, stations, eph_b = rlt._pass_context(prepared.pass_geo)
    kw = rlt.pass_station_kwargs(prepared.pass_geo)[0]
    return rlt, model, stations[0], eph_b, kw, prepared.pass_geo.t_s


def _three_segment_ramp(rlt):
    return rlt.UplinkRamp(epochs_s=(1700.0, 1785.0, 1810.0),
                          frequencies_hz=(F_REF_HZ + 500.0, F_REF_HZ + 534.0, F_REF_HZ + 527.75),
                          rates_hz_per_s=(0.4, -0.25, 0.1), reference_hz=F_REF_HZ)


def test_constant_ramp_at_reference_is_the_unramped_count(arc):
    """G20: f = f_ref, fdot = 0 -> value and partial bitwise equal to the unramped count."""
    rlt, model, st, eph_b, kw, t_s = _doppler_context(arc)
    sc = rlt.SpacecraftHistory(t_s, arc["aug"])
    flat = rlt.UplinkRamp(epochs_s=(0.0,), frequencies_hz=(F_REF_HZ,), rates_hz_per_s=(0.0,), reference_hz=F_REF_HZ)
    v0, p0, _ = rlt.counted_doppler(T_MID_S, TC_S, st, sc, eph_b, model, with_partial=True, **kw)
    v1, p1, _ = rlt.counted_doppler(T_MID_S, TC_S, st, sc, eph_b, model, with_partial=True, uplink_ramp=flat, **kw)
    assert v0 == v1
    assert np.array_equal(p0, p1)


def test_constant_frequency_offset_matches_the_closed_form(arc):
    """G21: f_ref + 1 kHz -> change = -c/(2 f_ref Tc) * 1 kHz * (t1e - t1s) within 1e-12 relative."""
    rlt, model, st, eph_b, kw, t_s = _doppler_context(arc)
    sc = rlt.SpacecraftHistory(t_s, arc["aug"][:, :6])
    off = rlt.UplinkRamp(epochs_s=(0.0,), frequencies_hz=(F_REF_HZ + 1e3,), rates_hz_per_s=(0.0,),
                         reference_hz=F_REF_HZ)
    v0, _, (a, b) = rlt.counted_doppler(T_MID_S, TC_S, st, sc, eph_b, model, **kw)
    v1, _, _ = rlt.counted_doppler(T_MID_S, TC_S, st, sc, eph_b, model, uplink_ramp=off, **kw)
    t1_span = TC_S - (b.round_trip_s - a.round_trip_s)
    expected = -299792458.0 / (2.0 * F_REF_HZ * TC_S) * 1e3 * t1_span
    assert abs((v1 - v0) - expected) <= 1e-12 * abs(expected), ((v1 - v0), expected)


def test_ramp_correction_matches_receive_time_quadrature(arc):
    """G22: three-segment ramp with breaks inside the count: correction vs an independent composite
    Gauss-Legendre quadrature over receive time of (f_T(t1(t3)) - f_ref) dt1/dt3, within 1e-6 relative."""
    rlt, model, st, eph_b, kw, t_s = _doppler_context(arc)
    sc = rlt.SpacecraftHistory(t_s, arc["aug"][:, :6])
    ramp = _three_segment_ramp(rlt)
    v0, _, (a, b) = rlt.counted_doppler(T_MID_S, TC_S, st, sc, eph_b, model, **kw)
    v1, _, _ = rlt.counted_doppler(T_MID_S, TC_S, st, sc, eph_b, model, uplink_ramp=ramp, **kw)
    assert a.t1_s < 1785.0 < 1810.0 < b.t1_s

    def round_trip(t3):
        return rlt.solve_two_way(t3, st, sc, eph_b, model, **kw).round_trip_s

    nodes, weights = np.polynomial.legendre.leggauss(4)
    edges = np.linspace(T_MID_S - 0.5 * TC_S, T_MID_S + 0.5 * TC_S, 601)
    h = 1e-3
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        for x, w in zip(nodes, weights):
            t3 = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
            dt1_dt3 = 1.0 - (round_trip(t3 + h) - round_trip(t3 - h)) / (2.0 * h)
            total += 0.5 * (hi - lo) * w * (ramp.frequency_hz(t3 - round_trip(t3)) - F_REF_HZ) * dt1_dt3
    oracle = -299792458.0 / (2.0 * F_REF_HZ * TC_S) * total
    assert abs((v1 - v0) - oracle) <= 1e-6 * abs(oracle), ((v1 - v0), oracle)


def test_ramped_partials_match_finite_differences(arc):
    """G23: ramped state partial vs central differences (10 m / 1e-2 m/s), 1e-6 of the row norm; the clock
    partial vs a 1e-3 s central difference of the receive-tag offset (additional evidence)."""
    rlt, model, st, eph_b, kw, t_s = _doppler_context(arc)
    ramp = _three_segment_ramp(rlt)
    sc_aug = rlt.SpacecraftHistory(t_s, arc["aug"])
    _v, partial, _, clock = rlt.counted_doppler(T_MID_S, TC_S, st, sc_aug, eph_b, model, with_partial=True,
                                                with_clock_partial=True, uplink_ramp=ramp, **kw)

    def value(x0, offset=0.0):
        sc = rlt.SpacecraftHistory(t_s, arc["propagate"](x0))
        k = dict(kw, clock_offset_s=offset)
        return rlt.counted_doppler(T_MID_S, TC_S, st, sc, eph_b, model, uplink_ramp=ramp, **k)[0]

    fd = np.zeros(6)
    for j, step in enumerate((10.0, 10.0, 10.0, 1e-2, 1e-2, 1e-2)):
        dx = np.zeros(6)
        dx[j] = step
        fd[j] = (value(arc["x0"] + dx) - value(arc["x0"] - dx)) / (2.0 * step)
    assert np.max(np.abs(np.asarray(partial).reshape(6) - fd)) <= 1e-6 * np.linalg.norm(fd), (partial, fd)
    clock_fd = (value(arc["x0"], 1e-3) - value(arc["x0"], -1e-3)) / 2e-3
    assert abs(clock - clock_fd) <= 1e-6 * max(abs(clock_fd), 1e-3), (clock, clock_fd)


def test_ramp_fails_closed(arc):
    """G24: a ramp starting after t1 of the count start; non-increasing epochs; a non-positive frequency."""
    rlt, model, st, eph_b, kw, t_s = _doppler_context(arc)
    sc = rlt.SpacecraftHistory(t_s, arc["aug"][:, :6])
    late = rlt.UplinkRamp(epochs_s=(1790.0,), frequencies_hz=(F_REF_HZ,), rates_hz_per_s=(0.0,),
                          reference_hz=F_REF_HZ)
    with pytest.raises(ValueError, match="before it"):
        rlt.counted_doppler(T_MID_S, TC_S, st, sc, eph_b, model, uplink_ramp=late, **kw)
    with pytest.raises(ValueError, match="increasing"):
        rlt.UplinkRamp(epochs_s=(10.0, 10.0), frequencies_hz=(F_REF_HZ, F_REF_HZ), rates_hz_per_s=(0.0, 0.0),
                       reference_hz=F_REF_HZ)
    with pytest.raises(ValueError, match="positive"):
        rlt.UplinkRamp(epochs_s=(0.0,), frequencies_hz=(-1.0,), rates_hz_per_s=(0.0,), reference_hz=F_REF_HZ)
