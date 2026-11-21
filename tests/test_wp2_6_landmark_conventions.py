"""F2 WP2.6 slice 1 (MEAS-13): landmark body frame, nadir attitude Jacobian, pixel origin.

Gates G1-G6 in studies/phase17/configs/wp2_6_predeclaration.json. The "before" values were recorded from the
module at the pre-registration commit (tests/fixtures/wp2_6_landmark_before.json).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lunar_od.lunar_landmark_optical import (DEFAULT_CAMERA, CameraIntrinsics, landmark_inertial_position_m,
                                             landmark_surface_outward_normal_j2000, nadir_pointing_camera_frame,
                                             perturbed_camera_frame, pinhole_position_jacobian, pinhole_project)

R_MOON = 1737.4e3
BEFORE = json.loads((Path(__file__).parent / "fixtures" / "wp2_6_landmark_before.json").read_text())


def _hex(values):
    return [float(v).hex() for v in np.ravel(values)]


def _vec(hexes):
    return np.array([float.fromhex(h) for h in hexes])


@pytest.fixture(scope="module")
def spice_loaded():
    pytest.importorskip("spiceypy")
    from lunar_od.spice_loader import load_spice_kernels
    try:
        load_spice_kernels(None, clear=False)
    except FileNotFoundError:
        pytest.skip("SPICE kernels not available")


def _survey_geometries():
    """The survey's 40 geometries (studies/phase17/scripts/wp2_6_survey.py): landmark within the half-FOV."""
    rng = np.random.default_rng(6)
    out = []
    for _ in range(40):
        lm_dir = rng.standard_normal(3)
        lm_dir /= np.linalg.norm(lm_dir)
        tilt = rng.standard_normal(3)
        tilt -= tilt @ lm_dir * lm_dir
        tilt /= np.linalg.norm(tilt)
        out.append(((R_MOON + 100e3) * lm_dir + rng.uniform(2e3, 14e3) * tilt, R_MOON * lm_dir))
    return out


def _coupled_fd(r_sc, r_lm, bias, step=1.0):
    fd = np.zeros((2, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = step
        for sign in (1.0, -1.0):
            r = r_sc + sign * e
            p = pinhole_project(r, r_lm, perturbed_camera_frame(nadir_pointing_camera_frame(r), bias))
            fd[:, j] += sign * np.array([p.u_px, p.v_px]) / (2.0 * step)
    return fd


@pytest.mark.parametrize("bias", [np.zeros(3), np.array([1e-3, -6e-4, 8e-4])])
def test_g1_nadir_from_state_jacobian_matches_fd_with_the_attitude_rebuilt(bias):
    worst = 0.0
    for r_sc, r_lm in _survey_geometries():
        c = perturbed_camera_frame(nadir_pointing_camera_frame(r_sc), bias)
        analytic = pinhole_position_jacobian(r_sc, r_lm, c, attitude_source="nadir_from_state",
                                             attitude_bias_rad=bias)
        fd = _coupled_fd(r_sc, r_lm, bias)
        worst = max(worst, np.linalg.norm(analytic - fd) / np.linalg.norm(fd))
    assert worst < 1e-6, worst


def test_g2_external_attitude_jacobian_is_unchanged(spice_loaded):
    for row in BEFORE["rows"]:
        r_sc, r_lm = _vec(row["r_sc"]), _vec(row["r_lm_pa"])
        c = nadir_pointing_camera_frame(r_sc)
        assert _hex(pinhole_position_jacobian(r_sc, r_lm, c)) == row["jacobian"]
        assert _hex(pinhole_position_jacobian(r_sc, r_lm, c, DEFAULT_CAMERA, attitude_source="external")) == \
            row["jacobian"]


def test_g3_body_frames(spice_loaded):
    import spiceypy as spice

    et = BEFORE["et"]
    for row in BEFORE["rows"]:
        lat, lon = row["lat"], row["lon"]
        pa = landmark_inertial_position_m(lat, lon, R_MOON, et, body_frame="MOON_PA_DE440")
        assert _hex(pa) == row["r_lm_pa"]
        assert _hex(landmark_surface_outward_normal_j2000(lat, lon, et, body_frame="MOON_PA_DE440")) == row["n_pa"]
        r_body = R_MOON * np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
        me = landmark_inertial_position_m(lat, lon, R_MOON, et)
        assert _hex(me) == _hex(np.asarray(spice.pxform("J2000", "MOON_ME_DE440_ME421", et)).T @ r_body)
        normal = landmark_surface_outward_normal_j2000(lat, lon, et)
        assert np.max(np.abs(normal - me / np.linalg.norm(me))) < 1e-15
        assert 100.0 < np.linalg.norm(me - pa) < 1000.0


@pytest.mark.parametrize("frame", ["MOON_ME", "MOON_PA", "IAU_MOON", "J2000", ""])
def test_g4_other_frames_fail_closed(spice_loaded, frame):
    with pytest.raises(ValueError):
        landmark_inertial_position_m(0.1, 0.2, R_MOON, BEFORE["et"], body_frame=frame)
    with pytest.raises(ValueError):
        landmark_surface_outward_normal_j2000(0.1, 0.2, BEFORE["et"], body_frame=frame)


def test_g5_pixel_origin(spice_loaded):
    centre = CameraIntrinsics(pixel_origin="center")
    assert centre.principal_point_px == ((2048 - 1) / 2.0, (2048 - 1) / 2.0)
    with pytest.raises(ValueError):
        CameraIntrinsics(pixel_origin="middle")
    for row in BEFORE["rows"]:
        r_sc, r_lm = _vec(row["r_sc"]), _vec(row["r_lm_pa"])
        c = nadir_pointing_camera_frame(r_sc)
        corner = pinhole_project(r_sc, r_lm, c, CameraIntrinsics(pixel_origin="corner"))
        assert _hex([corner.u_px, corner.v_px]) == row["uv"] and corner.within_fov == row["within_fov"]
        default = pinhole_project(r_sc, r_lm, c)
        assert _hex([default.u_px, default.v_px]) == row["uv"]
        moved = pinhole_project(r_sc, r_lm, c, centre)
        assert abs((corner.u_px - moved.u_px) - 0.5) < 1e-12 and abs((corner.v_px - moved.v_px) - 0.5) < 1e-12


def test_g6_nadir_from_state_rejects_a_foreign_attitude():
    r_sc, r_lm = _survey_geometries()[0]
    foreign = perturbed_camera_frame(nadir_pointing_camera_frame(r_sc), np.array([5e-3, 0.0, 0.0]))
    with pytest.raises(ValueError):
        pinhole_position_jacobian(r_sc, r_lm, foreign, attitude_source="nadir_from_state")
    with pytest.raises(ValueError):
        pinhole_position_jacobian(r_sc, r_lm, foreign, attitude_source="star_tracker")
