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
