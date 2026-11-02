"""WP1.3 (DYN-03, DYN-04, DYN-05): lunar orientation used by the gravity field.

DYN-05 is repaired: the sampled-grid harmonic path interpolates the rotation
geodesically (SLERP) instead of taking the nearest sample, so the force is no
longer a step function of time.  Oracle: SPICE ``pxform`` MOON_PA_DE440 at the
exact epoch, and the DIRECT exact-epoch propagation mode.

DYN-03 and DYN-04 are quantified, not changed: the default point-mass J2 uses
the IAU constant mean pole and the (J2, R_MOON_M) pair, which every frozen J2
baseline was computed with.  The strict xfails record the open defects until
the owner decides on the default.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from lunar_od import dynamics as dyn
from lunar_od.constants import J2_MOON_UNNORMALIZED, R_MOON_M
from lunar_od.lunar_frames import interpolated_rotation_at_time, nearest_rotation_at_time

ET0 = 820497669.18392
MOON_RATE_RAD_S = 2.6617e-6
GRGM1200A_TAB = (
    Path(__file__).resolve().parents[2] / "_archive" / "data" / "gggrx_1200a_sha.tab"
)


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    """Rotation angle between two rotations; chord form, accurate near zero."""
    return float(2.0 * np.arcsin(np.linalg.norm(a - b) / (2.0 * np.sqrt(2.0))))


def _uniform_grid(step_s: float = 60.0):
    t_grid = np.arange(0.0, 600.0 + step_s, step_s)
    return t_grid, np.array([_rot_z(MOON_RATE_RAD_S * t) for t in t_grid])


# --- DYN-05: continuous orientation -------------------------------------------

def test_interpolation_is_exact_for_a_uniform_rotation():
    """Analytic oracle: a constant-rate spin is reproduced at every epoch."""
    t_grid, rotations = _uniform_grid()
    for t in (0.0, 0.5, 29.999, 30.0, 90.0, 431.7, 600.0):
        got = interpolated_rotation_at_time(rotations, t_grid, t)
        np.testing.assert_allclose(got, _rot_z(MOON_RATE_RAD_S * t), rtol=0.0, atol=1e-15)


def test_interpolation_returns_grid_samples_bitwise_and_is_continuous():
    t_grid, rotations = _uniform_grid()
    for k in (0, 3, len(t_grid) - 1):
        np.testing.assert_array_equal(
            interpolated_rotation_at_time(rotations, t_grid, t_grid[k]), rotations[k])
    mid = 90.0
    before = interpolated_rotation_at_time(rotations, t_grid, mid - 1e-6)
    after = interpolated_rotation_at_time(rotations, t_grid, mid + 1e-6)
    assert _angle(before, after) < 1e-9
    # The legacy nearest lookup jumps by omega * cadence here.
    jump = _angle(nearest_rotation_at_time(rotations, t_grid, mid - 1e-6),
                  nearest_rotation_at_time(rotations, t_grid, mid + 1e-6))
    assert jump == pytest.approx(MOON_RATE_RAD_S * 60.0, rel=1e-6)


def test_interpolation_refuses_extrapolation_and_malformed_grids():
    t_grid, rotations = _uniform_grid()
    with pytest.raises(ValueError, match="no extrapolation"):
        interpolated_rotation_at_time(rotations, t_grid, 600.1)
    with pytest.raises(ValueError, match="does not match"):
        interpolated_rotation_at_time(rotations[:-1], t_grid, 10.0)


@pytest.fixture(scope="module")
def spice():
    spiceypy = pytest.importorskip("spiceypy")
    from lunar_od.spice_loader import load_spice_kernels
    try:
        load_spice_kernels(None, clear=False)
    except FileNotFoundError:
        pytest.skip("SPICE kernels not available")
    return spiceypy


def _pa_grid(spice, t_grid):
    return np.array([spice.pxform("J2000", "MOON_PA_DE440", ET0 + t) for t in t_grid])


def test_interpolated_moon_pa_matches_spice_at_grid_midpoints(spice):
    t_grid = np.arange(0.0, 3601.0, 60.0)
    rotations = _pa_grid(spice, t_grid)
    worst_slerp = worst_nearest = 0.0
    for t in t_grid[:-1] + 30.0:
        exact = np.asarray(spice.pxform("J2000", "MOON_PA_DE440", ET0 + t))
        worst_slerp = max(worst_slerp, _angle(interpolated_rotation_at_time(rotations, t_grid, t), exact))
        worst_nearest = max(worst_nearest, _angle(nearest_rotation_at_time(rotations, t_grid, t), exact))
    assert worst_slerp < 1e-10          # measured ~1e-12 rad (angular-acceleration term)
    assert worst_nearest > 7e-5         # omega * 30 s = 8.0e-5 rad: what DYN-05 removed


def test_grid_mode_propagation_tracks_the_exact_epoch_mode(spice):
    """Integration oracle: DIRECT mode queries pxform at every RHS epoch."""
    from lunar_od.gravity_model_loader import load_lunar_gravity_model
    from lunar_od.project_paths import gravity_data_root

    path = gravity_data_root() / "gl1800f" / "jggrx_1800f_sha.tab"
    if not path.is_file():
        pytest.skip("GL1800F gravity field not installed")
    model = dataclasses.replace(load_lunar_gravity_model(str(path), nmax=8, mmax=8),
                                frame="MOON_PA_DE440")

    def far(position):
        return lambda t: np.tile(position, (np.size(np.asarray(t)), 1))

    earth, sun = far(np.array([384400e3, 0.0, 0.0])), far(np.array([149.6e9, 0.0, 0.0]))
    x0 = np.array([1.8374e6, 0.0, 2.0e5, 0.0, 1.6e3, 0.0])
    t = np.array([0.0, 3600.0])
    t_grid = np.arange(-600.0, 4201.0, 60.0)
    direct = np.asarray(dyn.propagate_state(t, x0, 4902.800066e9, 0.0, 0.0, earth, sun,
                                            harmonic_model=model, harmonic_epoch_et0=ET0))
    grid = np.asarray(dyn.propagate_state(t, x0, 4902.800066e9, 0.0, 0.0, earth, sun,
                                          harmonic_model=model,
                                          harmonic_rotation=(t_grid, _pa_grid(spice, t_grid))))
    # Nearest-sample lookup gave 5.2e-4 m here; SLERP gives 9.1e-5 m, with the
    # rotation itself exact to ~1e-12 rad (the remainder is integration path).
    assert np.linalg.norm(direct[-1, :3] - grid[-1, :3]) < 2e-4


# --- DYN-03 / DYN-04: quantified, default unchanged ------------------------------

def test_dyn03_iau_mean_pole_is_1p5_degrees_from_the_de440_figure_axis(spice):
    pole = dyn._MCI_TO_MOON_BF[2]
    start, stop = spice.str2et("2020-01-01"), spice.str2et("2038-08-01")   # one nodal cycle
    angles = [
        np.degrees(np.arccos(np.clip(np.asarray(spice.pxform("J2000", "MOON_PA_DE440", et))[2] @ pole, -1, 1)))
        for et in np.linspace(start, stop, 200)
    ]
    assert 1.40 < min(angles) and max(angles) < 1.70     # measured 1.47..1.62 deg


def _grail_c20_pair():
    if not GRGM1200A_TAB.is_file():
        pytest.skip("GRGM1200A coefficient file not available")
    with GRGM1200A_TAB.open(encoding="ascii") as handle:
        r_ref_km = float(handle.readline().split(",")[0])
        for line in handle:
            fields = [f.strip() for f in line.split(",")]
            if fields[0] == "2" and fields[1] == "0":
                return -float(fields[2]) * np.sqrt(5.0), r_ref_km * 1e3
    raise AssertionError("C20 record not found")


def test_dyn04_grail_pair_is_read_from_the_coefficient_file():
    j2, r_ref = _grail_c20_pair()
    assert r_ref == 1738.0e3
    assert j2 == pytest.approx(2.0322356e-4, rel=1e-7)


@pytest.mark.xfail(strict=True, reason="DYN-04: (J2_MOON_UNNORMALIZED, R_MOON_M) is not GRAIL's "
                   "(C20, R_ref) pair; J2*R^2 is 4.7e-4 high (default kept for frozen baselines)")
def test_dyn04_default_j2_r2_product_matches_grail_c20():
    j2, r_ref = _grail_c20_pair()
    assert J2_MOON_UNNORMALIZED * R_MOON_M**2 == pytest.approx(j2 * r_ref**2, rel=1e-6)


@pytest.mark.xfail(strict=True, reason="DYN-03: default J2 uses the IAU constant mean pole, "
                   "1.5 deg from MOON_PA_DE440 (default kept for frozen baselines)")
def test_dyn03_default_j2_acceleration_matches_the_pa_pole_j2(spice):
    from lunar_od.dynamics import f3body_moon

    r = np.array([1.8374e6, 0.0, 2.0e5])
    state = np.concatenate([r, np.zeros(3)])
    earth, sun = np.array([384400e3, 0.0, 0.0]), np.array([149.6e9, 0.0, 0.0])   # GM 0: unused
    a_default = (f3body_moon(state, 4902.800066e9, 0.0, 0.0, earth, sun, j2_moon=J2_MOON_UNNORMALIZED)[3:]
                 - f3body_moon(state, 4902.800066e9, 0.0, 0.0, earth, sun, j2_moon=0.0)[3:])
    z = np.asarray(spice.pxform("J2000", "MOON_PA_DE440", ET0))[2]
    rn = np.linalg.norm(r)
    sin_phi = (r @ z) / rn
    k = 1.5 * J2_MOON_UNNORMALIZED * 4902.800066e9 * R_MOON_M**2 / rn**4
    a_pa = k * ((5.0 * sin_phi**2 - 1.0) * r / rn - 2.0 * sin_phi * z)
    np.testing.assert_allclose(a_default, a_pa, rtol=1e-3)
