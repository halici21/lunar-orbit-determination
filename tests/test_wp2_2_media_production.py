"""F2 WP2.2 (MEAS-06): troposphere + ionosphere media in the production barycentric two-way range.

Predeclared in ``studies/phase17/configs/wp2_2_predeclaration.json``. Oracles independent of the code under test:

* the Niell mapping functions and the single-layer ionosphere mapping against Orekit 13.1.9 reference values
  (``studies/phase17/results/phase_c5i_reference.npz``, produced by ``phase_c5i_reference_sandbox.py``);
* BLS-LM recovers the truth from media-bearing data, and the vacuum model misfits it by the media delay.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lunar_od.media import niell_mapping, single_layer_mapping

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


def test_niell_mapping_matches_orekit():
    """G1: 8400 Orekit points (elevation x latitude x height x 12 dates), |diff| < 1e-9."""
    ref = np.load(OREKIT, allow_pickle=False)
    worst = 0.0
    for el, lat, h, doy, mh, mw in ref["g2"]:
        ours = niell_mapping(float(el), float(lat), float(h), float(doy))
        worst = max(worst, abs(ours[0] - mh), abs(ours[1] - mw))
    assert worst < 1e-9


@pytest.mark.parametrize("height_m", [350000, 450000])
def test_single_layer_mapping_matches_orekit(height_m):
    """G2: thin-shell ionosphere mapping, relative difference < 1e-12."""
    ref = np.load(OREKIT, allow_pickle=False)
    ours = np.array([single_layer_mapping(float(e), float(height_m), 6371e3) for e in ref["g3_el"]])
    assert np.max(np.abs(ours / ref[f"g3_{height_m}"] - 1.0)) < 1e-12


def _elevation_mask(arc, truth, min_el_deg=10.0):
    from lunar_od.geometry import ecef2razel_sez

    spice, t = arc["spice"], arc["t"]
    earth = arc["eph"].earth_position(t)
    vis = np.zeros((t.size, len(STATIONS)), dtype=bool)
    for k in range(t.size):
        rot = np.asarray(spice.pxform("J2000", "ITRF93", arc["et0"] + t[k]))
        sc_ecef = rot @ (truth[k, :3] - earth[k])
        for j, st in enumerate(STATIONS):
            _az, el, _rng = ecef2razel_sez(sc_ecef - st.r_ecef_m, st.lat_rad, st.lon_rad)
            vis[k, j] = el >= np.radians(min_el_deg)
    return vis


def test_media_two_way_range_route_recovery_and_materiality(arc):
    """G3/G4: the arc builder carries the media model; BLS-LM recovers the truth; vacuum misfits by the media."""
    from lunar_od.estimators import estimate_two_way_range_bls_lm
    from lunar_od.scenarios import build_measurement_arcs
    from lunar_od.two_way_range import TwoWayRangeConfig, compute_two_way_range_residuals

    eph, t, gm = arc["eph"], arc["t"], arc["gm"]
    truth = arc["aug"][:, :6]
    vis = _elevation_mask(arc, truth)
    assert vis.sum() > 20
    arcs = build_measurement_arcs("two_way_range", t, truth, [0], [t.size - 1], vis, STATIONS,
                                  eph.earth_position, eph.earth_velocity, arc["et0"], noise=False,
                                  measurement_model_profile="two_way_light_time_ssb",
                                  two_way_range=TwoWayRangeConfig(), media_model="default_climatology")
    prepared = arcs[0]
    meta = prepared.pass_geo.measurement_metadata
    assert meta["media"]["troposphere"] is True

    x_start = arc["x0"] + np.array([80.0, -60.0, 40.0, 0.02, -0.01, 0.015])
    x_hat, status, _ = estimate_two_way_range_bls_lm(prepared.t_pass_s, prepared.obs_data, x_start,
                                                     prepared.pass_geo, *gm, eph.earth_position,
                                                     eph.sun_position, rtol=1e-12, atol=1e-13)
    assert np.linalg.norm(x_hat[:3] - arc["x0"][:3]) < 1e-2, status

    vacuum_meta = {k: v for k, v in meta.items() if k != "media"}
    vacuum_geo = replace(prepared.pass_geo, measurement_metadata=vacuum_meta)
    residuals, _ = compute_two_way_range_residuals(truth, prepared.obs_data, vacuum_geo)
    assert 2.4 < float(np.mean(residuals)) < 20.0


def _payload(**overrides):
    payload = {"name": "wp2_2", "measurement_type": "two_way_range", "estimator_type": "bls_lm",
               "start_mode": "cold", "network": "multi", "measurement_model_profile": "two_way_light_time_ssb",
               "media_model": "default_climatology"}
    payload.update(overrides)
    return payload


def test_scenario_accepts_media_on_the_barycentric_two_way_range():
    from lunar_od.scenario_config import scenario_config_from_mapping, scenario_measurement_identity

    config = scenario_config_from_mapping(_payload())
    assert config.media_model == "default_climatology"
    assert scenario_measurement_identity(config)["media_model"] == "default_climatology"
    default = scenario_config_from_mapping(_payload(media_model="none"))
    assert "media_model" not in scenario_measurement_identity(default)


@pytest.mark.parametrize("overrides", [
    {"measurement_model_profile": "geometric_instantaneous"},
    {"measurement_type": "position", "measurement_model_profile": "one_way_light_time_ssb"},
    {"media_model": "dsn_calibration"},
])
def test_scenario_rejects_unsupported_media(overrides):
    from lunar_od.scenario_config import scenario_config_from_mapping

    with pytest.raises(ValueError, match="media_model"):
        scenario_config_from_mapping(_payload(**overrides))


def test_arc_builder_rejects_media_off_the_barycentric_route():
    from lunar_od.scenarios import build_measurement_arcs

    with pytest.raises(ValueError, match="media_model"):
        build_measurement_arcs("position", np.arange(3.0), np.zeros((3, 6)), [0], [2], np.ones((3, 1), bool),
                               STATIONS[:1], None, None, 0.0, media_model="default_climatology")
