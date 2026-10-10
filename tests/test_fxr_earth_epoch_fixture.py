"""Phase 17-FXR - permanent protection for the Earth-epoch contract of the
Phase 17 range-analysis fixture (`examples/phase17_r1m_core.build_range_arc`).

Up to FXR the fixture handed the production two-way range model an Earth
position table whose t = 0 was 504000 s (5.83 d) BEFORE the epoch the same
model used for the Earth rotation, Sun and spacecraft state, and which held the
Earth constant after ~4 orbits.  The production model cannot see this: it gets
Earth as an untagged callable of scenario seconds.  These tests therefore check
BEHAVIOUR, not source text: the Earth states actually stored in the pass
geometry are compared with SPICE at the epoch the production model uses for
the rotation (`pass_geo.et0_s + t`).

The superseded fixture stays reproducible, but only on explicit request; its
tests pin the published baseline so the provenance path cannot rot.

Requires the frozen campaign data (od_covariance_campaign) and SPICE kernels;
skipped where they are absent.
"""

from __future__ import annotations

import hashlib
import inspect
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import phase17_r1m_core as M  # noqa: E402

CAMPAIGN_SCRIPTS = M.CAMPAIGN_ROOT / "06_scripts"
pytestmark = pytest.mark.skipif(
    not (CAMPAIGN_SCRIPTS / "od_gravity_covariance_campaign.py").exists()
    or not (M.CAMPAIGN_ROOT / "08_manifests").exists(),
    reason="frozen od_covariance_campaign data/scripts not present in this checkout",
)

DSN3 = ("Goldstone DSN", "Madrid DSN", "Canberra DSN")
PUBLISHED_OFFSET_S = 504000.0
#: T_OBS[-1] of the campaign table: arange(0, 4*T_ORBIT + 30, 60)[-1]
LEGACY_TABLE_END_S = 28260.0


def _spice_earth(et: float) -> np.ndarray:
    import spiceypy as spice
    return np.asarray(spice.spkezr("EARTH", float(et), "J2000", "NONE", "MOON")[0]) * 1e3


def _schedule_sha(obs: np.ndarray) -> str:
    """Observation schedule (receive tag, station, grid index): visibility only."""
    return hashlib.sha256(np.ascontiguousarray(obs[:, [0, 2, 3]])).hexdigest()


@pytest.fixture(scope="module")
def spice_loaded():
    from lunar_od.spice_loader import load_spice_kernels
    if str(CAMPAIGN_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(CAMPAIGN_SCRIPTS))
    load_spice_kernels(None, clear=True)
    return True


@pytest.fixture(scope="module")
def w15(spice_loaded):
    """The published range baseline configuration under both fixtures."""
    _, t_orbit = M.campaign_epoch()
    arcs = {}
    for fx in (M.EARTH_FIXTURE_CANONICAL, M.EARTH_FIXTURE_LEGACY_OFFSET_CLAMPED):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", M.SupersededEarthFixtureWarning)
            arcs[fx] = M.build_range_arc(0.0, 15.0 * t_orbit, label="W15", station_filter=DSN3,
                                         cadence_s=90.0, earth_fixture=fx)
    return arcs


@pytest.fixture(scope="module")
def w15_pre_f1(spice_loaded):
    """The same arcs under the PRE-F1 variational equations (SRP position gradient
    omitted), reproduced for the historical-value checks only.  ANALYSIS-ONLY: the
    gradient is zeroed in-process for the build; production has no such switch."""
    import lunar_od.srp as srp_mod
    original = srp_mod.srp_acceleration_position_gradient
    srp_mod.srp_acceleration_position_gradient = lambda *_a, **_k: np.zeros((3, 3))
    try:
        _, t_orbit = M.campaign_epoch()
        arcs = {}
        for fx in (M.EARTH_FIXTURE_CANONICAL, M.EARTH_FIXTURE_LEGACY_OFFSET_CLAMPED):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", M.SupersededEarthFixtureWarning)
                arcs[fx] = M.build_range_arc(0.0, 15.0 * t_orbit, label="W15", station_filter=DSN3,
                                             cadence_s=90.0, earth_fixture=fx)
    finally:
        srp_mod.srp_acceleration_position_gradient = original
    return arcs


def _metrics(arc):
    from phase17_r1cov_core import scale_matrix, square_root_covariance
    a_x, b_k = M.whitened(arc.h_x0, arc.h_k, arc.w)
    dec = M.orthogonal_decomposition(a_x, b_k)
    sr = square_root_covariance(np.hstack([arc.h_x0, arc.h_k[:, None]]), arc.w, None,
                                scale_matrix())
    return dec["orthogonal_fraction"], sr.sigma_k / M.K_TRUTH


# ----------------------------------------------------------------------
# The contract itself (canonical fixture)
# ----------------------------------------------------------------------
def test_canonical_earth_position_epoch_equals_earth_rotation_epoch(w15):
    arc = w15[M.EARTH_FIXTURE_CANONICAL]
    pg = arc.pass_geo
    et0, _ = M.campaign_epoch()
    assert pg.et0_s == et0                       # rotation origin = campaign epoch
    assert arc.earth.et_origin == pg.et0_s       # Earth origin = rotation origin
    expected = np.array([_spice_earth(pg.et0_s + t) for t in pg.t_s])
    np.testing.assert_array_equal(pg.earth_pos_mci_m, expected[:, :3])
    np.testing.assert_array_equal(pg.earth_vel_mci_mps, expected[:, 3:])


def test_canonical_windowed_arc_keeps_one_epoch(spice_loaded):
    """Windows starting after the campaign epoch (R1M multi-arc) must shift the
    Earth with the rotation; the legacy table restarted at its own t = 0."""
    _, t_orbit = M.campaign_epoch()
    arc = M.build_range_arc(2.5 * t_orbit, 0.5 * t_orbit, label="win",
                            station_filter=DSN3, earth_fixture=M.EARTH_FIXTURE_CANONICAL)
    pg = arc.pass_geo
    assert arc.earth.et_origin == pg.et0_s == M.campaign_epoch()[0] + arc.t_grid[0]
    for i in (0, pg.t_s.size // 2, pg.t_s.size - 1):
        np.testing.assert_array_equal(pg.earth_pos_mci_m[i],
                                      _spice_earth(pg.et0_s + pg.t_s[i])[:3])


def test_canonical_coverage_spans_arc_and_never_clamps(w15):
    arc = w15[M.EARTH_FIXTURE_CANONICAL]
    t = arc.pass_geo.t_s
    assert arc.earth.t_min_s <= t[0] and arc.earth.t_max_s >= t[-1]
    assert arc.earth.clamp_after_s == float("inf")
    pos = arc.pass_geo.earth_pos_mci_m
    after = t > LEGACY_TABLE_END_S
    # the Earth keeps moving (~1 km/s relative to the Moon) through the whole arc
    assert np.all(np.linalg.norm(np.diff(pos[after], axis=0), axis=1) > 1e4)


def test_canonical_ephemeris_fails_loudly_outside_coverage(spice_loaded):
    et0, _ = M.campaign_epoch()
    eph = M.measurement_earth_ephemeris(M.EARTH_FIXTURE_CANONICAL, et0, 1000.0)
    np.testing.assert_array_equal(eph.pos([0.0, 1000.0]),
                                  np.array([_spice_earth(et0)[:3], _spice_earth(et0 + 1000.0)[:3]]))
    with pytest.raises(M.EarthEphemerisCoverageError):
        eph.pos([1000.5])
    with pytest.raises(M.EarthEphemerisCoverageError):
        eph.vel([-1.0])


def test_canonical_is_the_default_and_legacy_is_explicit_only(spice_loaded):
    sig = inspect.signature(M.build_range_arc)
    assert sig.parameters["earth_fixture"].default == M.EARTH_FIXTURE_CANONICAL
    et0, _ = M.campaign_epoch()
    with pytest.warns(M.SupersededEarthFixtureWarning):
        M.measurement_earth_ephemeris(M.EARTH_FIXTURE_LEGACY_OFFSET_CLAMPED, et0, 100.0)
    with pytest.raises(ValueError):
        M.measurement_earth_ephemeris("SOMETHING_ELSE", et0, 100.0)


def test_station_placement_helper_uses_measurement_epoch(spice_loaded):
    et0, _ = M.campaign_epoch()
    t = np.array([0.0, 50000.0, 100000.0])
    np.testing.assert_array_equal(M.earth_position_mci(t),
                                  np.array([_spice_earth(et0 + x)[:3] for x in t]))
    np.testing.assert_array_equal(M.earth_velocity_mci(t),
                                  np.array([_spice_earth(et0 + x)[3:] for x in t]))


# ----------------------------------------------------------------------
# The superseded fixture: still exactly what was published
# ----------------------------------------------------------------------
def test_legacy_fixture_reproduces_the_offset_and_the_clamp(w15):
    arc = w15[M.EARTH_FIXTURE_LEGACY_OFFSET_CLAMPED]
    pg = arc.pass_geo
    assert pg.et0_s - arc.earth.et_origin == PUBLISHED_OFFSET_S
    # at a table node the legacy Earth IS SPICE 5.83 d early, not at the rotation epoch
    np.testing.assert_array_equal(pg.earth_pos_mci_m[0], _spice_earth(pg.et0_s - PUBLISHED_OFFSET_S)[:3])
    assert np.linalg.norm(pg.earth_pos_mci_m[0] - _spice_earth(pg.et0_s)[:3]) > 1e8
    frozen = pg.earth_pos_mci_m[pg.t_s >= LEGACY_TABLE_END_S]
    assert frozen.shape[0] > 100
    assert np.all(frozen == frozen[0])
    assert arc.earth.clamp_after_s == LEGACY_TABLE_END_S


def test_legacy_fixture_reproduces_published_range_baseline(w15, w15_pre_f1):
    """R1O / R1O-D / R1O-R / GEO-G0 published: 782 obs, f_perp 0.2523, sigma_K/K 9.27%.

    The published digits were computed before Phase 17-F1 added the SRP position
    gradient to the variational equations; they reproduce exactly under that
    (analysis-only) pre-F1 model.  With the corrected STM the same arc moves in
    the 5th significant digit."""
    arc = w15[M.EARTH_FIXTURE_LEGACY_OFFSET_CLAMPED]
    assert arc.n_obs == 782
    assert _schedule_sha(arc.obs) == (
        "7358d8d3b70fd6ef985cbb412be491dcef77d85911bc51760a7a4720fd8d415f")
    f_perp, sig = _metrics(w15_pre_f1[M.EARTH_FIXTURE_LEGACY_OFFSET_CLAMPED])
    assert f_perp == pytest.approx(0.2522919965229832, rel=1e-9)
    assert sig == pytest.approx(0.09273513250655349, rel=1e-9)
    f_perp, sig = _metrics(arc)                  # post-F1
    assert f_perp == pytest.approx(0.25228239732530355, rel=1e-9)
    assert sig == pytest.approx(0.0927353425945735, rel=1e-9)


def test_canonical_fixture_gives_the_fxr_range_baseline(w15, w15_pre_f1):
    """FXR / GEO-G0 consistent: 960 obs, f_perp 0.0915, sigma_K/K 25.4%.

    FXR's 0.091508 reproduces exactly under the pre-F1 variational model; the
    authoritative post-F1 value (docs/phase17_authoritative_status.md) is 0.091502."""
    arc = w15[M.EARTH_FIXTURE_CANONICAL]
    assert arc.n_obs == 960
    assert _schedule_sha(arc.obs) == (
        "40c6434121c6d533dadda3b58d9f62fa8bb9790bffb80c0f867d27dc8cdc2542")
    f_perp, sig = _metrics(w15_pre_f1[M.EARTH_FIXTURE_CANONICAL])
    assert f_perp == pytest.approx(0.09150755457311016, rel=1e-9)
    assert sig == pytest.approx(0.25402136377136003, rel=1e-9)
    f_perp, sig = _metrics(arc)                  # post-F1, authoritative
    assert f_perp == pytest.approx(0.0915024012983624, rel=1e-9)
    assert sig == pytest.approx(0.2540230025569859, rel=1e-9)


def test_truth_trajectory_does_not_depend_on_earth_fixture(w15):
    """mu_earth = 0: only the measurement geometry may change between fixtures."""
    a = w15[M.EARTH_FIXTURE_CANONICAL]
    b = w15[M.EARTH_FIXTURE_LEGACY_OFFSET_CLAMPED]
    np.testing.assert_array_equal(a.nom48, b.nom48)
