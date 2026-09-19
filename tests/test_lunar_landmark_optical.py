"""Permanent tests for `lunar_od.lunar_landmark_optical` (Phase 17-R1O-OPT).

These protect the production optical measurement model: the pinhole
projection and its analytic Jacobian, the camera attitude chain, the
geometric visibility gate, and the K_SRP chain-rule composition.  The
Jacobian test in particular is a regression guard for the state-transition
reshape-order defect documented in the phase report (§40): a transposed Phi
produces state columns wrong by orders of magnitude, and a chain-rule test
with a known Phi catches it immediately.
"""
from __future__ import annotations

import numpy as np
import pytest

from lunar_od.lunar_landmark_optical import (
    DEFAULT_CAMERA,
    CameraIntrinsics,
    OpticalGeometryError,
    landmark_geometrically_visible,
    landmark_optical_state_and_k_sensitivity,
    nadir_pointing_camera_frame,
    optical_one_way_light_time_s,
    perturbed_camera_frame,
    pinhole_position_jacobian,
    pinhole_project,
    small_angle_rotation,
)

IDENTITY = np.eye(3)


# ----------------------------------------------------------------------
# Camera intrinsics
# ----------------------------------------------------------------------
def test_focal_length_in_pixels_matches_metric_definition():
    cam = CameraIntrinsics(focal_length_m=0.035, pixel_pitch_m=5.5e-6)
    assert cam.focal_length_px == pytest.approx(0.035 / 5.5e-6)


def test_principal_point_is_sensor_centre():
    cam = CameraIntrinsics(sensor_width_px=2048, sensor_height_px=1024)
    assert cam.principal_point_px == (1024.0, 512.0)


def test_half_fov_follows_from_sensor_and_focal_length():
    cam = CameraIntrinsics()
    expected = np.arctan((cam.sensor_width_px / 2.0) / cam.focal_length_px)
    assert cam.half_fov_x_rad == pytest.approx(expected)


def test_ground_sample_distance_scales_linearly_with_range():
    cam = CameraIntrinsics()
    assert cam.ground_sample_distance_m(200_000.0) == pytest.approx(
        2.0 * cam.ground_sample_distance_m(100_000.0))


# ----------------------------------------------------------------------
# Pinhole projection
# ----------------------------------------------------------------------
def test_boresight_landmark_projects_to_principal_point():
    proj = pinhole_project(np.zeros(3), np.array([0.0, 0.0, 1e5]), IDENTITY)
    u0, v0 = DEFAULT_CAMERA.principal_point_px
    assert proj.u_px == pytest.approx(u0)
    assert proj.v_px == pytest.approx(v0)
    assert proj.in_front_of_camera
    assert proj.off_boresight_rad == pytest.approx(0.0, abs=1e-12)


def test_off_axis_projection_matches_exact_tangent():
    theta = np.radians(2.0)
    r_lm = 1e5 * np.array([np.sin(theta), 0.0, np.cos(theta)])
    proj = pinhole_project(np.zeros(3), r_lm, IDENTITY)
    u0, _ = DEFAULT_CAMERA.principal_point_px
    assert proj.u_px - u0 == pytest.approx(
        DEFAULT_CAMERA.focal_length_px * np.tan(theta), rel=1e-12)


def test_landmark_behind_camera_is_flagged_not_projected():
    proj = pinhole_project(np.zeros(3), np.array([0.0, 0.0, -1e5]), IDENTITY)
    assert not proj.in_front_of_camera
    assert not proj.within_fov
    assert np.isnan(proj.u_px)


def test_range_is_frame_independent():
    r_lm = np.array([1e4, 2e4, 9e4])
    rot = small_angle_rotation(np.array([1e-3, -2e-3, 3e-4]))
    a = pinhole_project(np.zeros(3), r_lm, IDENTITY)
    b = pinhole_project(np.zeros(3), r_lm, rot)
    assert a.range_m == pytest.approx(b.range_m)


# ----------------------------------------------------------------------
# Analytic Jacobian
# ----------------------------------------------------------------------
def _central_fd(r_sc, r_lm, c_ci, step):
    out = np.zeros((2, 3))
    for i in range(3):
        dr = np.zeros(3)
        dr[i] = step
        p = pinhole_project(r_sc + dr, r_lm, c_ci)
        m = pinhole_project(r_sc - dr, r_lm, c_ci)
        out[:, i] = (np.array([p.u_px, p.v_px]) - np.array([m.u_px, m.v_px])) / (2 * step)
    return out


def test_analytic_jacobian_matches_central_finite_difference():
    r_sc = np.array([-1.7e6, -2.8e5, 5.2e5])
    c_ci = nadir_pointing_camera_frame(r_sc)
    r_lm = r_sc + np.array([5e4, 2e4, -8e4])
    analytic = pinhole_position_jacobian(r_sc, r_lm, c_ci)
    # Step scaled to the OBSERVATION geometry (~1e5 m range), not to |r_sc|.
    fd = _central_fd(r_sc, r_lm, c_ci, 1.0)
    assert np.max(np.abs(analytic - fd)) / np.max(np.abs(analytic)) < 1e-8


def test_jacobian_finite_difference_converges_quadratically():
    r_sc = np.array([-1.7e6, -2.8e5, 5.2e5])
    c_ci = nadir_pointing_camera_frame(r_sc)
    r_lm = r_sc + np.array([5e4, 2e4, -8e4])
    analytic = pinhole_position_jacobian(r_sc, r_lm, c_ci)
    denom = np.max(np.abs(analytic))
    err_coarse = np.max(np.abs(_central_fd(r_sc, r_lm, c_ci, 1000.0) - analytic)) / denom
    err_fine = np.max(np.abs(_central_fd(r_sc, r_lm, c_ci, 100.0) - analytic)) / denom
    assert 30.0 < err_coarse / err_fine < 300.0


def test_jacobian_rejects_landmark_behind_camera():
    with pytest.raises(OpticalGeometryError):
        pinhole_position_jacobian(np.zeros(3), np.array([0.0, 0.0, -1e5]), IDENTITY)


def test_jacobian_is_linear_in_focal_length():
    r_sc = np.zeros(3)
    r_lm = np.array([1e4, 2e4, 9e4])
    cam2 = CameraIntrinsics(focal_length_m=2 * DEFAULT_CAMERA.focal_length_m,
                            pixel_pitch_m=DEFAULT_CAMERA.pixel_pitch_m)
    j1 = pinhole_position_jacobian(r_sc, r_lm, IDENTITY, DEFAULT_CAMERA)
    j2 = pinhole_position_jacobian(r_sc, r_lm, IDENTITY, cam2)
    assert np.allclose(j2, 2.0 * j1, rtol=1e-12)


# ----------------------------------------------------------------------
# Attitude chain
# ----------------------------------------------------------------------
def test_nadir_frame_is_orthonormal_and_right_handed():
    c = nadir_pointing_camera_frame(np.array([-1.7e6, -2.8e5, 5.2e5]))
    assert np.allclose(c @ c.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(c) == pytest.approx(1.0)


def test_nadir_frame_boresight_points_at_body_centre():
    r_sc = np.array([-1.7e6, -2.8e5, 5.2e5])
    c = nadir_pointing_camera_frame(r_sc)
    assert np.allclose(c[2, :], -r_sc / np.linalg.norm(r_sc))


def test_nadir_frame_rejects_zero_position():
    with pytest.raises(OpticalGeometryError):
        nadir_pointing_camera_frame(np.zeros(3))


def test_small_angle_rotation_first_order_shift_of_boresight():
    a = np.array([50e-6, -30e-6, 80e-6])
    shifted = small_angle_rotation(a) @ np.array([0.0, 0.0, 1.0])
    # R = I - skew(a) so R v = v - (a x v); for v = z_hat that is (-ay, ax, 1).
    assert shifted[0] == pytest.approx(-a[1])
    assert shifted[1] == pytest.approx(a[0])


def test_small_angle_rotation_orthonormal_to_second_order():
    a = np.array([50e-6, -30e-6, 80e-6])
    r = small_angle_rotation(a)
    residual = np.max(np.abs(r @ r.T - np.eye(3)))
    assert residual < 10.0 * float(np.dot(a, a))


def test_zero_attitude_error_leaves_frame_unchanged():
    c = nadir_pointing_camera_frame(np.array([-1.7e6, -2.8e5, 5.2e5]))
    assert np.allclose(perturbed_camera_frame(c, np.zeros(3)), c)


def test_attitude_bias_moves_the_image_by_the_expected_pixel_count():
    r_sc = np.array([0.0, 0.0, 1e5])
    r_lm = np.zeros(3)                       # exactly on the boresight
    c = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]])  # looking down -z
    tilt = 1e-4                               # rad, about the camera x axis
    proj0 = pinhole_project(r_sc, r_lm, c)
    proj1 = pinhole_project(r_sc, r_lm, perturbed_camera_frame(c, np.array([tilt, 0.0, 0.0])))
    shift_px = np.hypot(proj1.u_px - proj0.u_px, proj1.v_px - proj0.v_px)
    assert shift_px == pytest.approx(DEFAULT_CAMERA.focal_length_px * tilt, rel=1e-3)


# ----------------------------------------------------------------------
# Visibility gate
# ----------------------------------------------------------------------
def test_landmark_directly_below_spacecraft_is_visible():
    r_lm = np.array([0.0, 0.0, 1_737_400.0])
    n_hat = np.array([0.0, 0.0, 1.0])
    r_sc = r_lm + n_hat * 105_000.0
    vis = landmark_geometrically_visible(r_sc, r_lm, n_hat,
                                         nadir_pointing_camera_frame(r_sc))
    assert vis.visible and vis.reason == "VISIBLE"


def test_far_side_landmark_is_occluded():
    r_sc = np.array([0.0, 0.0, 1_842_400.0])
    r_lm = np.array([0.0, 0.0, -1_737_400.0])
    n_hat = np.array([0.0, 0.0, -1.0])
    vis = landmark_geometrically_visible(r_sc, r_lm, n_hat,
                                         nadir_pointing_camera_frame(r_sc))
    assert not vis.visible and vis.reason == "OCCULTED_FAR_SIDE"
    assert vis.off_nadir_from_landmark_deg > 90.0


def test_near_side_landmark_outside_field_of_view_is_rejected():
    # A landmark on the near side but far enough along the surface to fall
    # outside the ~9.1 deg half-FOV of a nadir-pointing camera.
    r_moon = 1_737_400.0
    r_sc = np.array([0.0, 0.0, r_moon + 105_000.0])
    ang = np.radians(15.0)
    n_hat = np.array([np.sin(ang), 0.0, np.cos(ang)])
    r_lm = r_moon * n_hat
    vis = landmark_geometrically_visible(r_sc, r_lm, n_hat,
                                         nadir_pointing_camera_frame(r_sc))
    assert not vis.visible and vis.reason == "OUTSIDE_FOV"


# ----------------------------------------------------------------------
# K_SRP chain composition
# ----------------------------------------------------------------------
def test_state_and_k_sensitivity_is_the_documented_chain_rule():
    rng = np.random.default_rng(17)
    dg_dr = rng.normal(size=(2, 3))
    phi = rng.normal(size=(6, 6))
    s_k = rng.normal(size=6)
    d_x0, d_k = landmark_optical_state_and_k_sensitivity(dg_dr, phi, s_k)
    assert np.allclose(d_x0, dg_dr @ phi[:3, :])
    assert np.allclose(d_k, dg_dr @ s_k[:3])


def test_k_sensitivity_uses_only_position_rows_of_the_sensitivity_vector():
    """K enters ONLY through the spacecraft position, so perturbing the
    velocity rows of S_K must not change dg/dK.  This is the structural
    statement of DIRECT_MEASUREMENT_K_DEPENDENCE = NO."""
    rng = np.random.default_rng(23)
    dg_dr = rng.normal(size=(2, 3))
    phi = np.eye(6)
    s_k = rng.normal(size=6)
    s_k_other = s_k.copy()
    s_k_other[3:] += 1e6
    _, d_k_a = landmark_optical_state_and_k_sensitivity(dg_dr, phi, s_k)
    _, d_k_b = landmark_optical_state_and_k_sensitivity(dg_dr, phi, s_k_other)
    assert np.allclose(d_k_a, d_k_b)


def test_transposed_state_transition_matrix_is_detectably_different():
    """Regression guard for the §40 defect: an asymmetric Phi must give a
    materially different answer when transposed, so a wrong reshape order
    cannot pass silently."""
    rng = np.random.default_rng(41)
    dg_dr = rng.normal(size=(2, 3))
    phi = rng.normal(size=(6, 6))
    s_k = rng.normal(size=6)
    d_x0, _ = landmark_optical_state_and_k_sensitivity(dg_dr, phi, s_k)
    d_x0_t, _ = landmark_optical_state_and_k_sensitivity(dg_dr, phi.T, s_k)
    assert not np.allclose(d_x0, d_x0_t)


# ----------------------------------------------------------------------
# Optical light time
# ----------------------------------------------------------------------
def test_optical_light_time_is_range_over_c():
    assert optical_one_way_light_time_s(299_792_458.0) == pytest.approx(1.0)


def test_optical_light_time_at_lunar_orbit_scale_is_sub_millisecond():
    assert optical_one_way_light_time_s(300_000.0) < 1.1e-3
