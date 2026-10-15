"""Phase 17-HARDEN boundary, coverage, force-model and consistency regressions.

Every oracle here is independent of the code under test: its own segment-sphere
geometry, its own SEZ elevation, SPICE directly, or central differences.
"""
from __future__ import annotations

import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "studies" / "phase17" / "scripts"
ARTIFACTS = ROOT / "studies" / "phase17" / "results"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import phase17_r1m_core as M  # noqa: E402

from lunar_od.constants import AU_M, R_MOON_M  # noqa: E402

needs_campaign = pytest.mark.skipif(not (M.CAMPAIGN_ROOT / "08_manifests").exists(),
                                    reason="read-only campaign data not available")
SEED = 20260926


@pytest.fixture(scope="module")
def et0():
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels(None, clear=True)
    return M.campaign_epoch()[0]


@pytest.fixture(scope="module")
def stations():
    import od_gravity_covariance_campaign as C  # noqa: N811
    return {s.name: s for s in C.STATIONS}


def _spice_earth(et):
    import spiceypy as sp
    return np.array([np.asarray(sp.spkpos("EARTH", float(e), "J2000", "NONE", "MOON")[0]) * 1e3
                     for e in np.atleast_1d(et)])


def _segment_hits_sphere(a, b, r):
    """Independent oracle: does segment a->b pass strictly inside |x| < r?"""
    d = b - a
    s = float(np.clip(-np.dot(a, d) / np.dot(d, d), 0.0, 1.0))
    return float(np.linalg.norm(a + s * d)) < r


# ======================================================================
# Coverage, nonzero start, long arc
# ======================================================================
@needs_campaign
def test_nonzero_arc_start_and_long_arc_keep_the_measurement_epoch(et0):
    t_orbit = M.campaign_epoch()[1]
    for t0, span in ((5000.0, 3600.0), (0.0, 15.0 * t_orbit)):
        earth = M.measurement_earth_ephemeris(M.EARTH_FIXTURE_CANONICAL, et0 + t0, span)
        t = np.array([0.0, 0.5 * span, span])
        err = np.linalg.norm(earth.pos(t) - _spice_earth(et0 + t0 + t), axis=1)
        assert float(err.max()) < 1e-3
        with pytest.raises(M.EarthEphemerisCoverageError):
            earth.pos(np.array([span * (1 + 1e-12) + 1e-6]))
        with pytest.raises(M.EarthEphemerisCoverageError):
            earth.pos(np.array([-1e-6]))


@needs_campaign
def test_moon_centered_ephemeris_carries_its_epoch(et0):
    from lunar_od.ephemeris import require_ephemeris_contract, sample_moon_centered_ephemeris
    eph = sample_moon_centered_ephemeris(et0, np.arange(0.0, 7201.0, 60.0))
    assert eph.et_origin == et0
    assert require_ephemeris_contract(eph.earth_position, et0) == "MATCHED"
    t = np.array([30.0, 3630.0, 7170.0])
    assert float(np.max(np.linalg.norm(eph.earth_position(t).reshape(-1, 3)
                                       - _spice_earth(et0 + t), axis=1))) < 1.0


# ======================================================================
# Elevation and lunar-limb boundaries
# ======================================================================
def _sez_to_ecef(lat, lon):
    sl, cl, so, co = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
    s = np.array([sl * co, sl * so, -cl])
    e = np.array([-so, co, 0.0])
    z = np.array([cl * co, cl * so, sl])
    return s, e, z


@needs_campaign
@pytest.mark.parametrize("delta_deg,visible", [(+1e-4, True), (-1e-4, False)])
def test_elevation_mask_boundary(et0, stations, delta_deg, visible):
    from lunar_od.visibility import sample_j2000_to_itrf93_transforms
    st = stations["Goldstone DSN"]
    t = np.array([0.0, 60.0])
    earth = M.earth_position_mci(t, M.EARTH_FIXTURE_CANONICAL)
    xf = sample_j2000_to_itrf93_transforms(et0, t)
    s, e, z = _sez_to_ecef(st.lat_rad, st.lon_rad)
    el = math.radians(10.0 + delta_deg)
    los = math.cos(el) * (-s) + math.sin(el) * z          # due north
    r_ecef = np.asarray(st.r_ecef_m, float).reshape(3) + 5.0e6 * los
    pos = np.array([earth[i] + xf[i][:3, :3].T @ r_ecef for i in range(t.size)])
    vis = M.station_visibility_mask(t, pos, (st,), earth, xf)
    assert bool(vis.all()) is visible


@needs_campaign
@pytest.mark.parametrize("offset_m", [-2000.0, -50.0, +50.0, +2000.0])
def test_lunar_limb_boundary_against_independent_oracle(et0, stations, offset_m):
    """Spacecraft whose station line of sight passes offset_m above/below the limb."""
    from lunar_od.measurements import _station_relative_state_j2000_at_receive_epoch
    from lunar_od.visibility import sample_j2000_to_itrf93_transforms
    st = stations["Goldstone DSN"]
    t = 7200.0 + np.array([0.0, 60.0])
    earth = M.earth_position_mci(t, M.EARTH_FIXTURE_CANONICAL)
    xf = sample_j2000_to_itrf93_transforms(et0, t)
    pos = []
    for i in range(t.size):
        r_st = earth[i] + _station_relative_state_j2000_at_receive_epoch(st, xf[i])[:3]
        dist = np.linalg.norm(r_st)
        u = -r_st / dist                                      # station -> Moon centre
        n = np.cross(u, [0.0, 0.0, 1.0]); n /= np.linalg.norm(n)
        alpha = math.asin((R_MOON_M + offset_m) / dist)       # impact parameter R + offset
        v = math.cos(alpha) * u + math.sin(alpha) * n
        pos.append(r_st + v * (dist * math.cos(alpha) + 2.0e6))   # well past the limb
        assert _segment_hits_sphere(r_st, pos[-1], R_MOON_M) is (offset_m < 0)
    vis = M.station_visibility_mask(t, np.array(pos), (st,), earth, xf, min_elevation_deg=-90.0)
    assert bool(vis.all()) is (offset_m > 0)


def test_seeded_random_occultation_agrees_with_independent_oracle():
    from lunar_od.visibility import is_occulted_by_moon
    rng = np.random.default_rng(SEED)
    n_checked = 0
    for _ in range(4000):
        a = rng.normal(size=3); a *= rng.uniform(1.2, 50.0) * R_MOON_M / np.linalg.norm(a)
        b = rng.normal(size=3); b *= rng.uniform(1.01, 5.0) * R_MOON_M / np.linalg.norm(b)
        d = b - a
        s = float(np.clip(-np.dot(a, d) / np.dot(d, d), 0.0, 1.0))
        if abs(np.linalg.norm(a + s * d) - R_MOON_M) < 1e-3 * R_MOON_M:
            continue                                           # skip the 1e-9 guard band
        prod = bool(is_occulted_by_moon(a, b, np.zeros(3), R_MOON_M)[0])
        assert prod is _segment_hits_sphere(a, b, R_MOON_M)
        n_checked += 1
    assert n_checked > 3500


# ======================================================================
# Frames and station velocity
# ======================================================================
@needs_campaign
def test_station_velocity_is_the_time_derivative_of_station_position(et0, stations):
    from lunar_od.measurements import _station_relative_state_j2000_at_receive_epoch
    from lunar_od.visibility import sample_j2000_to_itrf93_transforms
    st = stations["Canberra DSN"]
    h = 1.0
    xf = sample_j2000_to_itrf93_transforms(et0 + 3000.0, np.array([-h, 0.0, h]))
    for k in range(3):
        rot = xf[k][:3, :3]
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(rot) - 1.0) < 1e-12
    s = [_station_relative_state_j2000_at_receive_epoch(st, xf[k]) for k in range(3)]
    fd = (s[2][:3] - s[0][:3]) / (2 * h)
    assert np.linalg.norm(fd - s[1][3:]) < 1e-5 * np.linalg.norm(s[1][3:])
    assert 300.0 < np.linalg.norm(s[1][3:]) < 470.0          # |omega x r| at Canberra


# ======================================================================
# SRP and shadow
# ======================================================================
def _shadow_scan():
    from lunar_od.srp import SRPOptions, illumination_fraction, srp_acceleration_kernel
    sun = np.array([AU_M, 0.0, 0.0])
    opts = SRPOptions(k_srp_m2_per_kg=0.01)
    rho = np.linspace(R_MOON_M - 30e3, R_MOON_M + 30e3, 601)
    pos = [np.array([-1.838e6, r, 0.0]) for r in rho]
    nu = np.array([illumination_fraction(p, sun, np.zeros(3), R_MOON_M)[0] for p in pos])
    kern = np.array([srp_acceleration_kernel(p, sun, opts) for p in pos])
    return rho, pos, nu, kern, sun, opts


def test_shadow_entry_exit_is_continuous_bounded_and_monotone():
    rho, _, nu, kern, _, _ = _shadow_scan()
    assert np.all((nu >= 0.0) & (nu <= 1.0))
    assert nu[0] == 0.0 and nu[-1] == 1.0                     # umbra ... full light
    assert np.all(np.diff(nu) >= -1e-15)                      # monotone exit
    assert np.max(np.abs(np.diff(nu))) < 0.05                 # no step at contact
    assert np.all(kern[nu == 0.0] == 0.0)                     # umbra: no force


def test_srp_is_linear_in_k_and_vanishes_at_k_zero():
    from lunar_od.srp import srp_acceleration
    _, pos, nu, kern, sun, _ = _shadow_scan()
    i = int(np.argmin(np.abs(nu - 0.5)))
    for k in (0.0, 0.01, 0.037):
        a = srp_acceleration(pos[i], sun, k, nu[i])
        assert np.allclose(a, k * kern[i], rtol=1e-14, atol=0.0)
    assert np.all(srp_acceleration(pos[i], sun, 0.0, nu[i]) == 0.0)


def test_srp_position_gradient_root_cause_and_its_f1_closure():
    """HARDEN-D05: the pre-F1 variational A matrix omitted d(a_SRP)/dr; that
    gradient is negligible in full light and large only in penumbra, which is
    where the range-Jacobian STM error came from.  Phase 17-F1 closed it: A now
    takes ``srp`` and adds exactly that gradient (tests/test_f1_srp_variational_gradient.py
    holds the STM / S_K qualification)."""
    from lunar_od.dynamics import dynamics_jacobian_a_matrix
    from lunar_od.srp import srp_acceleration_position_gradient
    assert "srp" in inspect.signature(dynamics_jacobian_a_matrix).parameters
    rho, pos, nu, _, sun, opts = _shadow_scan()
    from lunar_od.srp import srp_acceleration_kernel

    def grad(p):
        g = np.zeros((3, 3))
        for j in range(3):
            e = np.zeros(3); e[j] = 0.5
            g[:, j] = (srp_acceleration_kernel(p + e, sun, opts)
                       - srp_acceleration_kernel(p - e, sun, opts)) / 1.0
        return opts.k_srp_m2_per_kg * g
    pen = int(np.argmin(np.abs(nu - 0.5)))
    g_pen = np.linalg.norm(grad(pos[pen]), 2)
    g_full = np.linalg.norm(grad(np.array([+1.838e6, 0.0, 0.0])), 2)
    g_grav = 4.9028e12 / 1.838e6 ** 3
    assert 1e-13 < g_pen < 1e-9
    assert g_pen > 1e4 * g_full
    assert g_pen < 1e-3 * g_grav                              # small, but secular
    # F1: the production analytic gradient is that same quantity
    g_an = srp_acceleration_position_gradient(pos[pen], sun, opts)
    assert np.linalg.norm(grad(pos[pen]) - g_an) / np.linalg.norm(g_an) < 1e-4   # 2nd-order FD, 0.5 m


def test_range_jacobian_root_cause_evidence_is_consistent():
    dec = json.loads((ARTIFACTS / "phase17_harden_range_jacobian_decomposition.json").read_text())
    pred = json.loads((ARTIFACTS / "phase17_harden_range_jacobian_srp_gradient_prediction.json")
                      .read_text())
    by = {(e["variant"], e["rtol"]): e for e in dec}
    for e in dec:                                             # event derivative exact
        for col in e["columns"].values():
            for key, v in col.items():
                if key.startswith("h="):
                    assert v["lin_rel"] < 1e-6
    worst = lambda e: max(max(v["stm_pos_rel"] for k, v in c.items() if k.startswith("h="))  # noqa: E731
                          for c in e["columns"].values())
    assert worst(by[("SRP_CONICAL_PENUMBRA", 1e-13)]) > 5 * worst(by[("SRP_NO_SHADOW", 1e-13)])
    assert worst(by[("SRP_CONICAL_PENUMBRA", 1e-13)]) > 5 * worst(by[("NO_SRP_K0", 1e-13)])
    best = next(e for e in pred if e["rtol"] == 1e-14 and e["h_pos"] == 10.0
                and e["shadow"] == "CONICAL_PENUMBRA")
    corr = {k: v["correlation"] for k, v in best["columns"].items()}
    assert sum(c >= 0.99 for c in corr.values()) >= 4
    assert min(corr.values()) > 0.7
    ctrl = next(e for e in pred if e["shadow"] == "NO_SHADOW")
    assert max(v["predicted_rel"] for v in ctrl["columns"].values()) < 1e-9
    # closure: with the FD oracle integrated at rtol 3e-15 the prediction explains
    # every column; the looser-rtol residual scaled with rtol (oracle, not physics)
    conv = json.loads((ARTIFACTS / "phase17_harden_range_jacobian_oracle_convergence.json").read_text())
    tight = conv["oracle_tolerance_study_h10_dt1"]["rtol_3e-15"]["columns"]
    assert min(v["correlation"] for v in tight.values()) > 0.98
    assert max(v["residual_after_prediction_rel"] for v in tight.values()) < 2e-7
    loose = conv["oracle_tolerance_study_h10_dt1"]["rtol_1e-13"]["columns"]
    assert loose["y0"]["residual_after_prediction_rel"] > 20 * tight["y0"]["residual_after_prediction_rel"]
    quad = conv["quadrature_study_rtol_1e-14_h10"]
    assert abs(quad["dt_0.1"]["columns"]["y0"]["residual_after_prediction_rel"]
               - quad["dt_1.0"]["columns"]["y0"]["residual_after_prediction_rel"]) < 5e-8


# ======================================================================
# Third body
# ======================================================================
@needs_campaign
def test_third_body_is_direct_minus_indirect_at_the_right_epoch(et0):
    from lunar_od.dynamics import point_mass_acceleration, third_body_acceleration
    mu_e = 3.986004418e14
    r_e = _spice_earth(et0 + 1234.0)[0]
    rng = np.random.default_rng(SEED)
    for _ in range(20):
        r = rng.normal(size=3); r *= rng.uniform(1.75e6, 5e6) / np.linalg.norm(r)
        a = third_body_acceleration(r, r_e, mu_e)
        ref = point_mass_acceleration(r - r_e, mu_e) - point_mass_acceleration(-r_e, mu_e)
        assert np.allclose(a, ref, rtol=1e-12, atol=1e-20)
        d = np.linalg.norm(r_e); u = r_e / d
        tidal = mu_e / d ** 3 * (3.0 * np.dot(r, u) * u - r)
        assert np.linalg.norm(a - tidal) < 0.05 * np.linalg.norm(tidal)
    assert 3.40e8 < np.linalg.norm(r_e) < 4.20e8


# ======================================================================
# Light time and cross-measurement consistency
# ======================================================================
def test_common_transmit_event_satisfies_both_light_time_equations():
    from lunar_od.delta_dor import solve_common_transmit_event
    c = 299792458.0
    ra = np.array([3.8e8, 1.0e6, 0.0])
    rb = lambda t: np.array([3.8e8, -4.0e6, 1.0e6]) + np.array([0.0, 300.0, 0.0]) * t  # noqa: E731
    sc = lambda t: np.array([1.8e6, 0.0, 0.0]) + np.array([0.0, 1600.0, 0.0]) * t      # noqa: E731
    sol = solve_common_transmit_event(100.0, ra, rb, sc)
    assert sol.backward_converged and sol.forward_converged
    t_tx, t_b = sol.transmit_time_s, sol.station_b_receive_time_s
    assert abs((100.0 - t_tx) - np.linalg.norm(ra - sc(t_tx)) / c) < 1e-12
    assert abs((t_b - t_tx) - np.linalg.norm(rb(t_b) - sc(t_tx)) / c) < 1e-12


def test_cross_measurement_geometry_and_derivatives_are_consistent():
    x = json.loads((ARTIFACTS / "phase17_harden_cross_measurement.json").read_text())
    assert all(v == 0.0 for v in x["at_grid_nodes"].values())
    assert x["orientation_transform_same_epoch"] and x["spacecraft_nominal_same"]
    # mid-node range-vs-Doppler difference is exactly the documented linear-Earth
    # interpolation of counted Doppler (R3-P22), not an epoch or frame error
    mid = x["at_mid_nodes"]
    assert abs(mid["range_vs_doppler_pos_m"] - x["earth_interp_error_mid_node_m"]["linear_doppler"]) < 1e-6
    assert mid["range_vs_ddor_pos_m"] < 1e-3
    d = json.loads((ARTIFACTS / "phase17_harden_derivatives.json").read_text())
    for chain in ("counted_doppler", "ddor", "optical"):
        for col, v in d[chain].items():
            bound = 1e-3 if col == "K_SRP" else 1e-5
            assert min(v["rel_err"]) < bound, (chain, col, v)
