"""Evidence tests for the 2026-10-31 independent assessment.

Each ``xfail(strict=True)`` test asserts the behaviour the assessment says the
code should have. While the finding stands the test fails (XFAIL); a fix turns
it into XPASS, which strict mode reports as a failure so the marker is removed
together with the fix. An XPASS before any fix means the finding was wrong and
is withdrawn from docs/validation/independent-assessment-2026-10-31.md.

Every finding test has an unmarked control that shows the oracle and fixture
behave as expected on an easy case, so an XFAIL cannot come from a broken test.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.linalg import expm

from lunar_od import LEGACY_INTERPOLATED_STATION_METHOD, RangeRatePhysicsConfig, solve_two_way_light_time
from lunar_od.dmc import DMCOptions, dmc_process_noise
from lunar_od.dynamics import propagate_augmented_state, propagate_state
from lunar_od.filters import _interp_pass_transforms, _interp_pass_values
from lunar_od.force_contract import ForceExecutionSpec, ForceModelContract
from lunar_od.lunar_frames import nearest_rotation_at_time
from lunar_od.scenarios import _ukf_default_initial_covariance
from lunar_od.thesis_matrix import THESIS_COLD_START_SIGMA_POS_M, THESIS_COLD_START_SIGMA_VEL_MPS

EARTH_RATE_RAD_S = 7.2921150e-5
MOON_RATE_RAD_S = 2.6617e-6
MU_MOON = 4902.800066163796e9
MU_EARTH = 398600.4354360959e9
MU_SUN = 132712440041.9393e9


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def _earth_sxform(t: float) -> np.ndarray:
    """6x6 J2000->rotating transform for a uniform spin about z (sxform layout)."""
    r = _rot_z(EARTH_RATE_RAD_S * t)
    omega = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]) * EARTH_RATE_RAD_S
    xf = np.zeros((6, 6))
    xf[:3, :3] = r
    xf[3:, 3:] = r
    xf[3:, :3] = -r @ omega
    return xf


# --- MEAS-04: UKF/EKF counted Doppler interpolates the J2000->ITRF93 transform linearly ---

def _interpolated_station_error(grid_step_s: float) -> tuple[float, float]:
    t_grid = np.arange(0.0, 1800.0 + grid_step_s, grid_step_s)
    xforms = np.array([_earth_sxform(t) for t in t_grid])
    t_mid = t_grid[:-1] + 0.5 * grid_step_s
    interp = _interp_pass_transforms(t_grid, xforms, t_mid)
    station_itrf = np.array([6.378e6, 0.0, 0.0])
    ortho = max(np.linalg.norm(x[:3, :3] @ x[:3, :3].T - np.eye(3)) for x in interp)
    station_err = max(
        np.linalg.norm(x[:3, :3].T @ station_itrf - _earth_sxform(t)[:3, :3].T @ station_itrf)
        for x, t in zip(interp, t_mid)
    )
    return ortho, station_err


def test_control_linear_transform_interpolation_exact_on_nodes():
    t_grid = np.arange(0.0, 600.0, 90.0)
    xforms = np.array([_earth_sxform(t) for t in t_grid])
    np.testing.assert_array_equal(_interp_pass_values(t_grid, xforms, t_grid), xforms)


def test_control_rotation_transform_interpolation_exact_on_nodes():
    t_grid = np.arange(0.0, 600.0, 90.0)
    xforms = np.array([_earth_sxform(t) for t in t_grid])
    np.testing.assert_array_equal(_interp_pass_transforms(t_grid, xforms, t_grid), xforms)


# MEAS-04 closed by WP1.1: the UKF/EKF resample the transform with _interp_pass_transforms.
@pytest.mark.parametrize("grid_step_s", [60.0, 90.0, 240.0])
def test_meas04_ukf_interpolated_station_transform_is_orthonormal_and_mm_accurate(grid_step_s):
    ortho, station_err = _interpolated_station_error(grid_step_s)
    assert ortho < 1e-12
    assert station_err < 1e-3


def test_meas04_ukf_interpolated_transform_rate_block_matches_sxform():
    t_grid = np.arange(0.0, 1800.0 + 90.0, 90.0)
    xforms = np.array([_earth_sxform(t) for t in t_grid])
    t_mid = t_grid[:-1] + 45.0
    station_itrf = np.array([6.378e6, 0.0, 0.0, 0.0, 0.0, 0.0])
    for x, t in zip(_interp_pass_transforms(t_grid, xforms, t_mid), t_mid):
        got = np.linalg.solve(x, station_itrf)
        want = np.linalg.solve(_earth_sxform(t), station_itrf)
        assert np.linalg.norm(got[:3] - want[:3]) < 1e-3
        assert np.linalg.norm(got[3:] - want[3:]) < 1e-6


def test_meas04_rotation_interpolation_against_spice_sxform():
    """Real J2000->ITRF93 (precession, nutation, polar motion) on 60 s and 240 s grids."""
    spice = pytest.importorskip("spiceypy")
    from lunar_od.spice_loader import load_spice_kernels
    try:
        load_spice_kernels()
    except Exception as exc:  # pragma: no cover - kernel-less environments
        pytest.skip(f"SPICE kernels unavailable: {exc}")
    et0 = spice.str2et("2027-03-08T00:00:00")  # inside the loaded Earth BPC
    station_itrf = np.array([-2.3533e6, -4.6416e6, 3.6762e6, 0.0, 0.0, 0.0])
    for step in (60.0, 240.0):
        t_grid = np.arange(0.0, 1800.0 + step, step)
        xforms = np.array([spice.sxform("J2000", "ITRF93", et0 + t) for t in t_grid])
        t_mid = t_grid[:-1] + 0.5 * step
        for x, t in zip(_interp_pass_transforms(t_grid, xforms, t_mid), t_mid):
            want = np.linalg.solve(spice.sxform("J2000", "ITRF93", et0 + t), station_itrf)
            got = np.linalg.solve(x, station_itrf)
            assert np.linalg.norm(got[:3] - want[:3]) < 1e-3
            assert np.linalg.norm(got[3:] - want[3:]) < 1e-6


# --- DYN-05: nearest-sample lunar orientation makes the force a step function ---

def _moon_rotation_grid(step_s: float):
    t_grid = np.arange(0.0, 600.0 + step_s, step_s)
    return t_grid, np.array([_rot_z(MOON_RATE_RAD_S * t) for t in t_grid])


def _rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0)))


def test_control_nearest_rotation_returns_grid_sample():
    t_grid, rotations = _moon_rotation_grid(60.0)
    np.testing.assert_array_equal(nearest_rotation_at_time(rotations, t_grid, 120.0), rotations[2])


@pytest.mark.xfail(strict=True, reason="DYN-05: orientation jumps by omega*cadence at every grid midpoint")
def test_dyn05_lunar_orientation_continuous_across_grid_midpoint():
    t_grid, rotations = _moon_rotation_grid(60.0)
    mid = 90.0
    before = nearest_rotation_at_time(rotations, t_grid, mid - 1e-6)
    after = nearest_rotation_at_time(rotations, t_grid, mid + 1e-6)
    assert _rotation_angle(before, after) < 1e-9


# --- EST-09: DMC van Loan process noise over long gaps ---

def _dmc_a(tau_s: float) -> np.ndarray:
    a = np.zeros((9, 9))
    a[0:3, 3:6] = np.eye(3)
    a[3:6, 6:9] = np.eye(3)
    a[6:9, 6:9] = -np.eye(3) / tau_s
    return a


def _dmc_q_substepped(options: DMCOptions, dt_s: float, pieces: int) -> np.ndarray:
    """Independent reference: compose short van Loan steps, Q <- Phi Q Phi^T + Q_h."""
    h = dt_s / pieces
    phi_h = expm(_dmc_a(options.tau_s) * h)
    q_h = dmc_process_noise(options, h)
    q = np.zeros((9, 9))
    for _ in range(pieces):
        q = phi_h @ q @ phi_h.T + q_h
    return q


def _relative_difference(q: np.ndarray, reference: np.ndarray) -> float:
    return float(np.max(np.abs(q - reference)) / np.max(np.abs(reference)))


def test_control_dmc_q_matches_substepped_reference_short_gap():
    options = DMCOptions(sigma_mps2=1e-8, tau_s=600.0)
    q = dmc_process_noise(options, 600.0)
    assert _relative_difference(q, _dmc_q_substepped(options, 600.0, 8)) < 1e-9


@pytest.mark.xfail(strict=True, reason="EST-09: van Loan block grows like exp(dt/tau) and cancels")
@pytest.mark.parametrize("gap_over_tau", [30.0, 144.0])
def test_est09_dmc_q_long_gap_finite_psd_and_accurate(gap_over_tau):
    options = DMCOptions(sigma_mps2=1e-8, tau_s=600.0)
    dt = gap_over_tau * options.tau_s
    q = dmc_process_noise(options, dt)
    reference = _dmc_q_substepped(options, dt, int(gap_over_tau))
    assert np.all(np.isfinite(q))
    assert np.min(np.linalg.eigvalsh(q)) >= -1e-12 * np.max(np.abs(reference))
    assert _relative_difference(q, reference) < 1e-8


# --- EST-19: UKF default initial covariance vs the injected cold-start error ---

@pytest.mark.xfail(strict=True, reason="EST-19: UKF P0 is 100 m / 0.1 m/s; cold start injects 180 m / 0.03 m/s")
def test_est19_ukf_default_p0_matches_injected_cold_start_sigma():
    p0 = _ukf_default_initial_covariance(6, "position", None, 1)
    sigma = np.sqrt(np.diag(p0))
    np.testing.assert_allclose(sigma[:3], THESIS_COLD_START_SIGMA_POS_M, rtol=0.01)
    np.testing.assert_allclose(sigma[3:6], THESIS_COLD_START_SIGMA_VEL_MPS, rtol=0.01)


# --- EST-01: SRIF scores the current point (42-state) and the candidate (6-state) differently ---

def _earth_pos(t):
    return np.array([3.844e8, 0.0, 0.0])


def _sun_pos(t):
    return np.array([1.496e11, 0.0, 0.0])


def _low_lunar_orbit_state() -> np.ndarray:
    r = 1_737_400.0 + 100_000.0
    v = np.sqrt(MU_MOON / r)
    inc = np.deg2rad(85.0)
    return np.array([r, 0.0, 0.0, 0.0, v * np.cos(inc), v * np.sin(inc)])


def _six_vs_augmented_position_difference(rtol: float, atol: float) -> float:
    t = np.linspace(0.0, 4 * 3600.0, 61)
    x0 = _low_lunar_orbit_state()
    six = np.asarray(propagate_state(t, x0, MU_MOON, MU_EARTH, MU_SUN, _earth_pos, _sun_pos, rtol=rtol, atol=atol))
    aug0 = np.concatenate([x0, np.eye(6).reshape(-1)])
    aug = np.asarray(
        propagate_augmented_state(t, aug0, MU_MOON, MU_EARTH, MU_SUN, _earth_pos, _sun_pos, rtol=rtol, atol=atol)
    )
    return float(np.max(np.linalg.norm(six[:, :3] - aug[:, :3], axis=1)))


def test_control_six_and_augmented_paths_agree_at_tight_tolerance():
    assert _six_vs_augmented_position_difference(1e-13, 1e-13) < 1e-2


@pytest.mark.xfail(strict=True, reason="EST-01: at estimator tolerances the two paths differ by more than 1 mm")
@pytest.mark.parametrize("rtol, atol", [(1e-10, 1e-11), (1e-8, 1e-10)])
def test_est01_six_and_augmented_paths_agree_at_estimator_tolerance(rtol, atol):
    assert _six_vs_augmented_position_difference(rtol, atol) < 1e-3


# --- MEAS-03: counted-Doppler round-trip solver convergence tests use epoch differences ---

class _Station:
    name = "Synthetic Station"
    lat_rad = 0.0
    lon_rad = 0.0
    sigma_range_m = 1.0
    sigma_angle_rad = 1e-6
    sigma_range_rate_mps = 1e-4
    bias = ()

    @property
    def r_ecef_m(self):
        return np.zeros(3)


def _solve_count(arc_start_s: float, samples: int = 200) -> int:
    t_grid = arc_start_s + np.arange(-240.0, 241.0, 20.0)
    local = t_grid - arc_start_s
    states = np.zeros((t_grid.size, 6))
    states[:, 0] = 3.844e8 + 1600.0 * local
    states[:, 3] = 1600.0
    zeros = np.zeros((t_grid.size, 3))
    xforms = np.repeat(np.eye(6)[None], t_grid.size, axis=0)
    config = RangeRatePhysicsConfig(station_state_method=LEGACY_INTERPOLATED_STATION_METHOD)
    rng = np.random.default_rng(20261031)
    solved = 0
    for offset in rng.uniform(-100.0, 100.0, samples):
        solution = solve_two_way_light_time(
            arc_start_s + offset, _Station(), t_grid, states, zeros, zeros, xforms, config
        )
        # Production callers raise RoundTripLightTimeConvergenceError when this is False.
        solved += int(solution.converged)
    return solved


def test_control_counted_doppler_solver_converges_near_zero_epoch():
    assert _solve_count(0.0) == 200


@pytest.mark.xfail(strict=True, reason="MEAS-03: epoch-difference residuals hit ulp(t) on long arcs")
@pytest.mark.parametrize("arc_start_s", [3.0e5, 2.5e6])
def test_meas03_counted_doppler_solver_converges_on_long_arcs(arc_start_s):
    assert _solve_count(arc_start_s) == 200


# --- DYN-10: the force fingerprint does not identify the executed ephemeris ---

def _field_names(cls) -> set[str]:
    names = set()
    for field in dataclasses.fields(cls):
        names.add(field.name.lower())
        if dataclasses.is_dataclass(field.type) and not isinstance(field.type, str):
            names |= _field_names(field.type)
    return names


@pytest.mark.xfail(strict=True, reason="DYN-10: no ephemeris kernel, cadence or interpolation identity is hashed")
def test_dyn10_force_identity_records_ephemeris_kernel_and_interpolation():
    names = _field_names(ForceModelContract) | _field_names(ForceExecutionSpec)
    assert any("kernel" in n or "spk" in n for n in names)
    assert any("interpolation" in n for n in names)
