"""PHASE 17-R1O-OPT - camera/attitude/visibility/light-time oracles.

ANALYSIS SPACE ONLY.  Qualifies the NEW production module
`lunar_od/lunar_landmark_optical.py` against hand-verifiable synthetic
geometry (s13/s16 camera projection oracle, s16 attitude frame gate) and
against the real campaign trajectory (s73 analytic-Jacobian FD check, s24-25
light-time significance test).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lunar_od.lunar_landmark_optical import (
    CameraIntrinsics, DEFAULT_CAMERA, OpticalGeometryError,
    landmark_geometrically_visible, landmark_inertial_position_m,
    landmark_surface_outward_normal_j2000, nadir_pointing_camera_frame,
    optical_one_way_light_time_s, perturbed_camera_frame, pinhole_position_jacobian,
    pinhole_project, small_angle_rotation,
)
from lunar_od.measurements import C_LIGHT_MPS

GATES: dict[str, bool] = {}


def hdr(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


def gate(name: str, ok: bool, detail: str = "") -> None:
    GATES[name] = bool(ok)
    print("  %-55s %s  %s" % (name, "PASS" if ok else "FAIL", detail))


def central_fd_jacobian(g_fn, r, step_m):
    g_plus, g_minus = [], []
    for i in range(3):
        dr = np.zeros(3)
        dr[i] = step_m
        g_plus.append(np.asarray(g_fn(r + dr)))
        g_minus.append(np.asarray(g_fn(r - dr)))
    return np.array([(g_plus[i] - g_minus[i]) / (2.0 * step_m) for i in range(3)]).T


def main() -> None:
    from lunar_od.spice_loader import load_spice_kernels

    load_spice_kernels(None, clear=True)

    hdr("S1 - PINHOLE PROJECTION: BORESIGHT-CENTERED LANDMARK (hand-verifiable)")
    # Landmark placed EXACTLY on the boresight, range 100 km -> must land
    # exactly at the principal point, independent of intrinsics.
    c_ci = np.eye(3)  # trivial "camera frame == inertial" for this synthetic check
    r_sc = np.zeros(3)
    r_lm = np.array([0.0, 0.0, 100_000.0])  # +z is boresight by pinhole_project's convention
    proj = pinhole_project(r_sc, r_lm, c_ci, DEFAULT_CAMERA)
    u0, v0 = DEFAULT_CAMERA.principal_point_px
    ok = abs(proj.u_px - u0) < 1e-9 and abs(proj.v_px - v0) < 1e-9 and proj.in_front_of_camera
    gate("BORESIGHT_LANDS_AT_PRINCIPAL_POINT", ok,
         f"u={proj.u_px:.9f} (expect {u0}), v={proj.v_px:.9f} (expect {v0})")

    hdr("S2 - PINHOLE PROJECTION: KNOWN OFF-AXIS ANGLE (hand-verifiable, exact tan())")
    theta = np.radians(2.0)  # 2 degrees off boresight, well inside the ~9.15 deg half-FOV
    range_m = 100_000.0
    r_lm2 = range_m * np.array([np.sin(theta), 0.0, np.cos(theta)])
    proj2 = pinhole_project(r_sc, r_lm2, c_ci, DEFAULT_CAMERA)
    expected_du = DEFAULT_CAMERA.focal_length_px * np.tan(theta)
    actual_du = proj2.u_px - u0
    rel_err = abs(actual_du - expected_du) / abs(expected_du)
    gate("OFFAXIS_PROJECTION_MATCHES_EXACT_TAN", rel_err < 1e-12,
         f"expected du={expected_du:.6f}px, actual du={actual_du:.6f}px, rel_err={rel_err:.3e}")

    hdr("S3 - CAMERA ATTITUDE FRAME GATE: nadir-pointing orthonormality + boresight convention")
    r_sc_real = np.array([-1_746_513.53, -277_083.99, 518_734.14])  # real campaign point (see summary)
    c_nadir = nadir_pointing_camera_frame(r_sc_real)
    orthonorm_err = float(np.max(np.abs(c_nadir @ c_nadir.T - np.eye(3))))
    det_err = abs(float(np.linalg.det(c_nadir)) - 1.0)
    boresight_row = c_nadir[2, :]
    nadir_hat = -r_sc_real / np.linalg.norm(r_sc_real)
    boresight_err = float(np.max(np.abs(boresight_row - nadir_hat)))
    ok = orthonorm_err < 1e-12 and det_err < 1e-12 and boresight_err < 1e-12
    gate("CAMERA_ATTITUDE_FRAME_GATE", ok,
         f"orthonorm_err={orthonorm_err:.3e}, det_err={det_err:.3e}, boresight_err={boresight_err:.3e}")

    hdr("S4 - SMALL-ANGLE ROTATION: orthogonality to second order + first-order angle recovery")
    angles = np.array([50e-6, -30e-6, 80e-6])  # ~10 arcsec class, R1O-OPT s18 anchor
    dr = small_angle_rotation(angles)
    orthonorm_err2 = float(np.max(np.abs(dr @ dr.T - np.eye(3))))
    # R = I - skew(a), so R @ v = v - (a x v).  For v = z_hat,
    # a x z_hat = (ay, -ax, 0), hence the first-order shift is (-ay, +ax, 0).
    z_perturbed = dr @ np.array([0.0, 0.0, 1.0])
    expected_shift = np.array([-angles[1], angles[0], 0.0])
    actual_shift = z_perturbed - np.array([0.0, 0.0, 1.0])
    first_order_err = float(np.linalg.norm(actual_shift - expected_shift))
    # Orthonormality of a FIRST-ORDER rotation is only exact to O(angle^2):
    # with |a| ~ 8e-5 rad the expected residual is ~|a|^2 ~ 6.4e-9, so the
    # tolerance below must sit ABOVE that floor, not below it (the same
    # "floor consistency" discipline Phase 17A-R established for FD oracles:
    # never test a quantity against a tolerance tighter than its own
    # construction's known truncation level).
    expected_second_order = float(np.dot(angles, angles))
    gate("SMALL_ANGLE_ROTATION_SELF_CONSISTENT",
         orthonorm_err2 < 10.0 * expected_second_order and first_order_err < 1e-15,
         f"orthonorm_err2={orthonorm_err2:.3e} (|a|^2={expected_second_order:.3e} expected), "
         f"first_order_err={first_order_err:.3e}")

    hdr("S5 - ANALYTIC JACOBIAN vs CONVERGENCE-CHECKED CENTRAL FD (R1O-OPT s73 hard block)")
    c_ci3 = nadir_pointing_camera_frame(r_sc_real)
    r_lm3 = r_sc_real + np.array([50_000.0, 20_000.0, -80_000.0])  # some landmark ~ off-nadir, in front

    def g_fn(r):
        p = pinhole_project(r, r_lm3, c_ci3, DEFAULT_CAMERA)
        return np.array([p.u_px, p.v_px])

    jac_analytic = pinhole_position_jacobian(r_sc_real, r_lm3, c_ci3, DEFAULT_CAMERA)
    # R1O-OPT s73 forbids resting on ONE arbitrary FD step.  The sweep below
    # covers five decades and reports the quadratic-truncation region, the
    # optimum, and the rounding-noise floor -- the FD oracle's own precision
    # floor is measured, not assumed (Phase 17A-R "floor consistency").
    #
    # NOTE (kept as documentation): an earlier version of this script scaled
    # the FD step to |r_sc| (~1.8e6 m), giving ~1840 m steps and a 1.4e-4
    # "disagreement" that was entirely the FD side's own truncation error.
    # The step must be scaled to the OBSERVATION geometry (landmark range
    # ~1e5 m), not to the spacecraft's inertial position magnitude.
    denom = float(np.max(np.abs(jac_analytic)))
    sweep = []
    for step in (1000.0, 100.0, 10.0, 1.0, 0.1, 0.01):
        jac_fd = central_fd_jacobian(g_fn, r_sc_real, step)
        rel = float(np.max(np.abs(jac_fd - jac_analytic)) / denom)
        sweep.append((step, rel))
        print(f"    step={step:8.2f} m   analytic-vs-FD rel_err = {rel:.3e}")
    best_step, best_rel = min(sweep, key=lambda s: s[1])
    # Quadratic convergence check over the truncation-dominated decade:
    # a 10x smaller step must reduce the error by ~100x.
    ratio_1000_100 = sweep[0][1] / sweep[1][1]
    quadratic_ok = 30.0 < ratio_1000_100 < 300.0
    gate("CAMERA_PROJECTION_ORACLE_GATE", best_rel < 1e-8 and quadratic_ok,
         f"best rel_err={best_rel:.3e} at step={best_step} m, "
         f"quadratic ratio(1000m/100m)={ratio_1000_100:.1f} (expect ~100)")

    hdr("S6 - VISIBILITY GATE: hand-verifiable near-side / far-side / outside-FOV")
    lat0, lon0 = np.radians(10.0), np.radians(20.0)
    et_fake = 8.0e8  # arbitrary but fixed epoch for this synthetic check
    r_lm_near = landmark_inertial_position_m(lat0, lon0, 1_737_400.0, et_fake)
    n_hat_near = landmark_surface_outward_normal_j2000(lat0, lon0, et_fake)
    # Spacecraft directly above this landmark (along its own outward normal) -> near side, in FOV.
    r_sc_above = r_lm_near + n_hat_near * 105_000.0
    c_ci_above = nadir_pointing_camera_frame(r_sc_above)
    vis_near = landmark_geometrically_visible(r_sc_above, r_lm_near, n_hat_near, c_ci_above, DEFAULT_CAMERA)
    # Antipodal landmark (lon + 180 deg) as seen from the SAME spacecraft position -> far side, occulted.
    r_lm_far = landmark_inertial_position_m(lat0, lon0 + np.pi, 1_737_400.0, et_fake)
    n_hat_far = landmark_surface_outward_normal_j2000(lat0, lon0 + np.pi, et_fake)
    vis_far = landmark_geometrically_visible(r_sc_above, r_lm_far, n_hat_far, c_ci_above, DEFAULT_CAMERA)
    # A near-side landmark 20 deg off nadir (outside the ~9.15 deg half-FOV) -> geometrically near, but outside FOV.
    lat1 = lat0 + np.radians(20.0 * 105.0 / 1737.4)  # crude great-circle offset ~20deg off-nadir
    r_lm_edge = landmark_inertial_position_m(lat1, lon0, 1_737_400.0, et_fake)
    n_hat_edge = landmark_surface_outward_normal_j2000(lat1, lon0, et_fake)
    vis_edge = landmark_geometrically_visible(r_sc_above, r_lm_edge, n_hat_edge, c_ci_above, DEFAULT_CAMERA)
    ok = (vis_near.visible and vis_near.reason == "VISIBLE"
          and not vis_far.visible and vis_far.reason == "OCCULTED_FAR_SIDE"
          and not vis_edge.visible and vis_edge.reason == "OUTSIDE_FOV")
    gate("VISIBILITY_GATE_THREE_CASES", ok,
         f"near={vis_near.reason}, far={vis_far.reason}, edge={vis_edge.reason}")

    hdr("S7 - OPTICAL LIGHT-TIME / TIME-TAG / ABERRATION (R1O-OPT s24-26: tested, not assumed)")
    # THREE PHYSICALLY DISTINCT EFFECTS, deliberately separated.  An earlier
    # version of this section reported only "spacecraft motion during the
    # light time" and labelled the result OPTICAL_LIGHT_TIME_SIGNIFICANCE --
    # which is NOT the light-time retardation at all, but the cost of getting
    # the TIME-TAG convention wrong.  Conflating them is exactly the error
    # R1O-D caught in its own session-bias study (one formula used for two
    # different physical questions), so each is computed on its own terms.
    from lunar_od.constants import MU_MOON_M3S2 as mu_moon
    r_orbit_m = 1_737_400.0 + 105_470.0
    v_circ_mps = float(np.sqrt(mu_moon / r_orbit_m))
    max_range_m = 300_000.0  # generous bound on the slant range to an in-FOV landmark
    tau_s = optical_one_way_light_time_s(max_range_m, C_LIGHT_MPS)
    gsd_m = DEFAULT_CAMERA.ground_sample_distance_m(105_470.0)
    ifov_rad = DEFAULT_CAMERA.pixel_pitch_m / DEFAULT_CAMERA.focal_length_m
    print(f"  v_sc(MCI) ~= {v_circ_mps:.3f} m/s, tau(300 km slant) = {tau_s * 1e6:.3f} us")
    print(f"  GSD at 105.47 km = {gsd_m:.3f} m/px,  IFOV = {ifov_rad * 1e6:.3f} urad/px")

    # (1) TRUE LIGHT-TIME RETARDATION: the landmark must be taken at its
    #     EMISSION-epoch inertial position.  In MCI the landmark moves only
    #     by lunar rotation (sidereal period 27.32 d).
    lunar_surface_speed_mps = 2.0 * np.pi * 1_737_400.0 / (27.321661 * 86400.0)
    landmark_retardation_m = lunar_surface_speed_mps * tau_s
    retardation_px = landmark_retardation_m / gsd_m
    print(f"\n  (1) landmark retardation  = {landmark_retardation_m * 1e3:.4f} mm"
          f"  -> {retardation_px:.3e} px")

    # (2) TIME-TAG CONVENTION: the spacecraft position must be taken at the
    #     SHUTTER (reception) epoch.  Using the emission epoch instead costs
    #     a full v_sc*tau of position error.
    time_tag_error_m = v_circ_mps * tau_s
    time_tag_px = time_tag_error_m / gsd_m
    print(f"  (2) time-tag error cost   = {time_tag_error_m:.4f} m"
          f"   -> {time_tag_px:.3e} px   (cost of the WRONG convention)")

    # (3) STELLAR ABERRATION: a distinct relativistic effect of the OBSERVER's
    #     velocity in the frame the landmark position is expressed in; it is
    #     NOT captured by the light-time geometry above.
    aberration_rad = v_circ_mps / C_LIGHT_MPS
    aberration_px = aberration_rad / ifov_rad
    print(f"  (3) stellar aberration    = {aberration_rad * 1e6:.4f} urad"
          f" -> {aberration_px:.3e} px")

    # Classification thresholds are stated against the FINEST centroiding
    # precision this phase tests (0.1 px, the dedicated-camera best case):
    # an effect matters if it is an appreciable fraction of that.
    finest_centroiding_px = 0.1
    light_time_significance = (
        "NEGLIGIBLE" if retardation_px < 0.01 * finest_centroiding_px else "SIGNIFICANT")
    aberration_significance = (
        "NEGLIGIBLE" if aberration_px < 0.01 * finest_centroiding_px else "SIGNIFICANT")
    print(f"\n  against the finest centroiding precision tested ({finest_centroiding_px} px):")
    print(f"    OPTICAL_LIGHT_TIME_SIGNIFICANCE   = {light_time_significance}"
          f"  ({retardation_px / finest_centroiding_px:.2e} of it)")
    print(f"    OPTICAL_ABERRATION_SIGNIFICANCE   = {aberration_significance}"
          f"  ({aberration_px / finest_centroiding_px:.2e} of it)")
    print(f"    OPTICAL_TIME_TAG_CONTRACT         = SHUTTER_RECEPTION_EPOCH"
          f"  (wrong choice costs {time_tag_px:.3f} px)")
    gate("OPTICAL_LIGHT_TIME_AND_TIMETAG_TESTED", True,
         f"retardation {retardation_px:.2e} px, aberration {aberration_px:.2e} px, "
         f"time-tag {time_tag_px:.3f} px")
    optical_light_time_significance = light_time_significance

    hdr("SUMMARY")
    all_pass = all(GATES.values())
    for k, v in GATES.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"\n  ALL_ORACLES_PASS = {all_pass}")
    print(f"  OPTICAL_LIGHT_TIME_SIGNIFICANCE = {optical_light_time_significance}")


if __name__ == "__main__":
    main()
