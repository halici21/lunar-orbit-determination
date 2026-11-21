"""F4 WP4.2-R: BLS-LM must not stop on a damping exhaustion reached at the loosened early-iteration tolerance.

Gates in studies/phase17/configs/wp4_2r_predeclaration.json. The oracle is the cost at the truth: a least-squares
solve that stopped at its minimum cannot score worse than the truth on the same data.
"""
from __future__ import annotations

import numpy as np
import pytest

from lunar_od.config import Station

STATIONS = (
    Station("Goldstone", 35.4267, -116.8900, 1000.0, (0, 0, 0), 2.0, 1e-4, 1e-4),
    Station("Canberra", -35.4014, 148.9817, 690.0, (0, 0, 0), 2.0, 1e-4, 1e-4),
    Station("Madrid", 40.4314, -4.2481, 834.0, (0, 0, 0), 2.0, 1e-4, 1e-4),
)
OFFSET = np.array([80.0, -60.0, 40.0, 0.02, -0.01, 0.015])
KINDS = ("position", "range_rate", "two_way_range")


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
    from lunar_od.visibility import is_occulted_by_moon

    et0 = spice.str2et("2027-03-02T00:00:00")
    t = np.arange(0.0, 6.0 * 3600.0 + 1.0, 60.0)
    eph = sample_moon_centered_ephemeris(et0, np.arange(-600.0, t[-1] + 601.0, 60.0))
    gm = gravitational_parameters()
    r0 = 1737.4e3 + 100e3
    x0 = np.array([r0, 30e3, -20e3, -15.0, np.sqrt(gm[0] / r0), 4.0])

    def propagate(x_initial):
        y0 = np.concatenate([x_initial, np.eye(6).reshape(-1, order="F")])
        return propagate_augmented_state(t, y0, *gm, eph.earth_position, eph.sun_position, rtol=1e-12, atol=1e-13)

    aug = propagate(x0)
    earth = eph.earth_position(t)
    vis = np.zeros((t.size, len(STATIONS)), dtype=bool)
    for k in range(t.size):
        rot = np.asarray(spice.pxform("J2000", "ITRF93", et0 + t[k]))
        for j, st in enumerate(STATIONS):
            if ecef2razel_sez(rot @ (aug[k, :3] - earth[k]) - st.r_ecef_m, st.lat_rad, st.lon_rad)[1] < np.radians(10.0):
                continue
            station_mci = earth[k] + rot.T @ st.r_ecef_m
            vis[k, j] = not bool(np.asarray(is_occulted_by_moon(station_mci, aug[k:k + 1, :3], np.zeros(3), 1737.4e3))[0])
    return dict(et0=et0, t=t, eph=eph, gm=gm, x0=x0, propagate=propagate, aug=aug, vis=vis)


def _data(arc, kind, noise):
    from lunar_od.measurements import generate_position_measurements, generate_range_rate_measurements
    from lunar_od.two_way_range import generate_two_way_range_measurements

    eph, rng = arc["eph"], np.random.default_rng(2000)
    args = (arc["t"], arc["aug"][:, :6], STATIONS, arc["vis"], eph.earth_position, eph.earth_velocity, arc["et0"])
    if kind == "position":
        return generate_position_measurements(*args, noise=noise, rng=rng, arc_id=1,
                                              measurement_model_profile="one_way_light_time")[:2]
    if kind == "range_rate":
        return generate_range_rate_measurements(*args, noise=noise, rng=rng, arc_id=1, companion_geometry="apparent_one_way")
    return generate_two_way_range_measurements(*args, noise=noise, rng=rng, arc_id=1)[:2]


def _cost(arc, kind, x, obs, geo):
    from lunar_od.measurements import compute_position_residuals, compute_range_rate_residuals, measurement_sigma_vector
    from lunar_od.two_way_range import compute_two_way_range_residuals

    residuals = {"position": compute_position_residuals, "range_rate": compute_range_rate_residuals,
                 "two_way_range": compute_two_way_range_residuals}[kind]
    r = np.asarray(residuals(arc["propagate"](x)[:, :6], obs, geo)[0]) / measurement_sigma_vector(obs, geo)
    return float(r @ r)


def _fit(arc, kind, obs, geo):
    import lunar_od.estimators as est

    return getattr(est, f"estimate_{kind}_bls_lm")(arc["t"], obs, arc["x0"] + OFFSET, geo, *arc["gm"],
                                                    arc["eph"].earth_position, arc["eph"].sun_position,
                                                    rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("kind, bound_m", [("position", 1e-3), ("range_rate", 1e-3), ("two_way_range", 0.02)])
def test_g1_noise_free_fits_recover_the_truth(arc, kind, bound_m):
    obs, geo = _data(arc, kind, noise=False)
    x_hat, status, _ = _fit(arc, kind, obs, geo)
    assert status != "DampingLimit"
    assert np.linalg.norm(np.asarray(x_hat[:3]) - arc["x0"][:3]) < bound_m, status


@pytest.mark.parametrize("kind", KINDS)
def test_g2_noisy_fits_end_at_or_below_the_cost_at_the_truth(arc, kind):
    obs, geo = _data(arc, kind, noise=True)
    x_hat, status, _ = _fit(arc, kind, obs, geo)
    assert status != "DampingLimit"
    assert _cost(arc, kind, np.asarray(x_hat[:6]), obs, geo) <= _cost(arc, kind, arc["x0"], obs, geo), status


def test_g4_a_fit_that_converged_before_moves_by_a_negligible_fraction_of_sigma():
    """The WP2.5 1 h arc, seed 1004: J-Stab at iteration 14 before the fix; estimate recorded before the code change.

    Amended after the result (amendment log, exploratory): the owner chose (B), no loosening, which changes every
    BLS-LM fit, so the bitwise clause cannot hold. Measured: same status and iteration count, 4.6 mm, 1.7e-6 sigma.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    import inspect

    import test_wp2_5_measurement_defaults as wp25
    import lunar_od.estimators as est
    from lunar_od.measurements import generate_position_measurements

    pytest.importorskip("spiceypy")
    try:
        one_hour = inspect.unwrap(wp25.arc)()
    except FileNotFoundError:
        pytest.skip("SPICE kernels not available")
    eph, t = one_hour["eph"], one_hour["t"]
    obs, geo = generate_position_measurements(t, one_hour["aug"][:, :6], wp25.STATIONS, one_hour["vis"],
                                              eph.earth_position, eph.earth_velocity, one_hour["et0"], noise=True,
                                              rng=np.random.default_rng(1004), arc_id=1,
                                              measurement_model_profile="one_way_light_time")[:2]
    x_hat, status, stats = est.estimate_position_bls_lm(t, obs, one_hour["x0"] + OFFSET, geo, *one_hour["gm"],
                                                        eph.earth_position, eph.sun_position, rtol=1e-12, atol=1e-13,
                                                        return_posterior=True)
    before = ["0x1.c095ec6f39ee4p+20", "0x1.d3689f89c5625p+14", "-0x1.35e07541b13a4p+14",
              "-0x1.ddb057b58d05bp+3", "0x1.98603593aba7fp+10", "0x1.fffd5a7dd5d6dp+1"]
    assert (status, stats.iterations) == ("J-Stab", 14)
    sigma = np.sqrt(np.diag(np.asarray(stats.posterior_covariance)[:6, :6]))
    moved = np.abs(np.asarray(x_hat[:6]) - np.array([float.fromhex(v) for v in before])) / sigma
    assert np.max(moved) < 1e-4, moved
