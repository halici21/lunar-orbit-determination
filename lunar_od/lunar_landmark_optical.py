"""Production OD-level lunar landmark optical (image-plane) measurement model.

Phase 17-R1O-OPT.  This module implements a pinhole-camera image-plane
observable (u, v) of a known lunar landmark, consumed as an OD-level
measurement -- landmark IDENTITY is assumed known (a matched catalog entry),
so this is the identified-feature-centroid stage of optical navigation, not a
vision pipeline.  It does NOT implement raw image rendering, crater/feature
detection, descriptor matching, or image segmentation (R1O-OPT §13; see
``docs/phase17_r1o_opt_literature_contract.md``, "What this module does NOT
implement").

## What makes this "production" rather than R1O's surrogate

R1O's ``phase17_r1o_core.build_landmark_arc`` (Phase 17-R1O, analysis-only)
idealized the observable as a direct boresight-relative tangent-plane angle
pair with EXACTLY known attitude and EXACTLY known landmark position -- no
camera, no attitude chain, no image plane.  This module instead builds the
full chain a real optical-navigation OD would consume:

    r_landmark^PA (body-fixed, Moon principal axes)
        --[MOON_PA_DE440, already-qualified]-->  r_landmark^J2000(t)
        --[camera-from-inertial attitude, EXPLICIT EXTERNAL INPUT]-->  rho^C
        --[pinhole projection]-->  (u, v) pixels

with attitude decomposed into three SEPARATE, non-interchangeable error
sources (R1O-OPT §17-20): random per-observation knowledge noise, a
deterministic boresight/alignment bias, and (optionally) a camera-to-body
mounting misalignment -- since this repository has NO pre-existing spacecraft
attitude representation (audited: `orbit.py`'s `rot_x`/`rot_z` are generic
orbital-element DCM helpers, `force_contract.py`'s/`scenario_config.py`'s
`body_frame` fields describe the GRAVITY body frame, not spacecraft attitude;
`reference_config` has zero camera fields), the default here is an explicit,
documented nadir-pointing convention, matching R1O's own surrogate boresight
choice so the two are geometrically comparable.

## Camera parameter choice (literature-anchored, not copied)

See ``docs/phase17_r1o_opt_literature_contract.md`` for the full reasoning.
Federici et al. (2025) cite ~2.5 px noise <-> ~160 m ground resolution for
feature-based lunar-orbit optical nav (their exact intrinsics were not
recoverable). This module's default ``CameraIntrinsics`` (35 mm focal length,
5.5 um pixel pitch, 2048x2048 sensor) gives ~16.6 m/pixel ground-sample
distance at this campaign's measured ~105.5 km orbital altitude -- same order
of magnitude, a defensible reasoned choice, not a forced match.

## What this module does NOT model

Lens distortion (``DISTORTION_MODEL = DEFERRED_CHARACTERIZED``), a photometric
illumination/detectability renderer (``FEATURE_DETECTABILITY_MODEL =
CHARACTERIZED_NOT_FULLY_IMPLEMENTED`` -- only solar-incidence-angle screening
is provided), and an ellipsoidal/topographic lunar shape model
(``LANDMARK_POSITION_MODEL = SPHERICAL_REFERENCE``, R_MOON_M mean radius,
matching R1O's own surrogate).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


class OpticalGeometryError(RuntimeError):
    """A landmark projection was requested from a degenerate camera geometry."""


# ======================================================================
# Camera intrinsics
# ======================================================================
@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics.  Focal length carried in both metric and
    pixel form so ground-sample distance and field of view are always
    derivable from the SAME numbers used in the projection (no duplicated,
    driftable constants).
    """

    focal_length_m: float = 0.035          # 35 mm, literature-anchored (see module docstring)
    pixel_pitch_m: float = 5.5e-6          # common CMOS pixel pitch
    sensor_width_px: int = 2048
    sensor_height_px: int = 2048

    @property
    def focal_length_px(self) -> float:
        return self.focal_length_m / self.pixel_pitch_m

    @property
    def principal_point_px(self) -> tuple[float, float]:
        return (self.sensor_width_px / 2.0, self.sensor_height_px / 2.0)

    @property
    def half_fov_x_rad(self) -> float:
        return float(np.arctan((self.sensor_width_px / 2.0) / self.focal_length_px))

    @property
    def half_fov_y_rad(self) -> float:
        return float(np.arctan((self.sensor_height_px / 2.0) / self.focal_length_px))

    def ground_sample_distance_m(self, range_m: float) -> float:
        """GSD ~= range * (pixel_pitch / focal_length), the small-angle IFOV*range approximation."""
        return float(range_m) * self.pixel_pitch_m / self.focal_length_m


DEFAULT_CAMERA = CameraIntrinsics()


# ======================================================================
# Camera attitude chain (R1O-OPT s17-s20: three separable error sources)
# ======================================================================
def nadir_pointing_camera_frame(r_sc_inertial_m: ArrayLike) -> np.ndarray:
    """Default (error-free) camera-from-inertial DCM: boresight = +z = nadir.

    Rows are the camera axes expressed in inertial components, so
    ``C_ci @ v_inertial = v_camera``.  The tangent-plane basis (camera x/y)
    is built with the SAME reference-vector construction R1O's landmark
    surrogate used (``build_landmark_arc``), so the ideal (error-free) case
    of this module is geometrically identical to R1O's boresight convention
    -- the two are directly comparable, only the chain downstream differs.
    """
    r_sc = np.asarray(r_sc_inertial_m, dtype=float).reshape(3)
    norm = float(np.linalg.norm(r_sc))
    if norm <= 0.0:
        raise OpticalGeometryError("Spacecraft position must be nonzero.")
    boresight = -r_sc / norm  # nadir
    ref = np.array([0.0, 0.0, 1.0]) if abs(boresight[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x_cam = np.cross(boresight, ref)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(boresight, x_cam)
    c_ci = np.vstack([x_cam, y_cam, boresight])
    return c_ci


def small_angle_rotation(angles_rad: ArrayLike) -> np.ndarray:
    """First-order (small-angle) rotation matrix from a 3-vector of tilt angles.

    ``R = I + skew(-angles)`` to first order, i.e. the standard small-angle
    attitude-error DCM (e.g. Markley & Crassidis, *Fundamentals of Spacecraft
    Attitude Determination and Control*, the generic small-angle-error
    convention).  Exact (not first-order) for verification purposes callers
    needing higher fidelity should compose true axis-angle rotations, but at
    the arcsecond-to-arcminute magnitudes this module characterizes the
    linearization error is itself negligible relative to the noise being
    modeled (self-consistent: this IS the attitude error, sub-mrad by
    construction for every case this phase runs).
    """
    a = np.asarray(angles_rad, dtype=float).reshape(3)
    skew = np.array([
        [0.0, -a[2], a[1]],
        [a[2], 0.0, -a[0]],
        [-a[1], a[0], 0.0],
    ])
    return np.eye(3) - skew


def perturbed_camera_frame(c_ci_nominal: ArrayLike, error_angles_rad: ArrayLike) -> np.ndarray:
    """Apply a small-angle attitude error (random noise OR a deterministic
    bias -- the caller decides which by how ``error_angles_rad`` was drawn)
    to a nominal camera-from-inertial DCM.  The SAME function represents both
    error classes; only the calling convention differs, mirroring R1O-D's
    finding that noise and bias must be analyzed with DIFFERENT downstream
    formulas even though the injection mechanism is shared.
    """
    c_nom = np.asarray(c_ci_nominal, dtype=float).reshape(3, 3)
    d_r = small_angle_rotation(error_angles_rad)
    return d_r @ c_nom


# ======================================================================
# Pinhole projection and its analytic position Jacobian
# ======================================================================
@dataclass(frozen=True)
class LandmarkProjection:
    u_px: float
    v_px: float
    range_m: float
    off_boresight_rad: float
    in_front_of_camera: bool
    within_fov: bool


def pinhole_project(
    r_sc_inertial_m: ArrayLike,
    r_landmark_inertial_m: ArrayLike,
    c_ci: ArrayLike,
    intrinsics: CameraIntrinsics = DEFAULT_CAMERA,
) -> LandmarkProjection:
    """Project a landmark's inertial position into camera pixel coordinates.

    ``rho_cam = C_ci @ (r_landmark - r_sc)``; ``u = f_px*x/z + u0``,
    ``v = f_px*y/z + v0`` with the camera z axis as the boresight/depth axis
    (standard pinhole convention).
    """
    r_sc = np.asarray(r_sc_inertial_m, dtype=float).reshape(3)
    r_lm = np.asarray(r_landmark_inertial_m, dtype=float).reshape(3)
    c = np.asarray(c_ci, dtype=float).reshape(3, 3)

    rho_inertial = r_lm - r_sc
    range_m = float(np.linalg.norm(rho_inertial))
    rho_cam = c @ rho_inertial
    z = float(rho_cam[2])
    in_front = z > 0.0

    f_px = intrinsics.focal_length_px
    u0, v0 = intrinsics.principal_point_px
    if in_front:
        u = f_px * rho_cam[0] / z + u0
        v = f_px * rho_cam[1] / z + v0
        off_boresight = float(np.arccos(np.clip(z / range_m, -1.0, 1.0))) if range_m > 0 else 0.0
        within_fov = (
            abs(u - u0) <= intrinsics.sensor_width_px / 2.0
            and abs(v - v0) <= intrinsics.sensor_height_px / 2.0
        )
    else:
        u, v, off_boresight, within_fov = float("nan"), float("nan"), float(np.pi), False

    return LandmarkProjection(
        u_px=u, v_px=v, range_m=range_m, off_boresight_rad=off_boresight,
        in_front_of_camera=in_front, within_fov=within_fov,
    )


def pinhole_position_jacobian(
    r_sc_inertial_m: ArrayLike,
    r_landmark_inertial_m: ArrayLike,
    c_ci: ArrayLike,
    intrinsics: CameraIntrinsics = DEFAULT_CAMERA,
) -> np.ndarray:
    """Analytic d(u, v)/d(r_sc), a (2, 3) matrix.

    ``rho_cam = C_ci @ (r_lm - r_sc)``, so ``d(rho_cam)/d(r_sc) = -C_ci``
    (``r_lm`` and ``C_ci`` do not depend on ``r_sc``).  Then the standard
    pinhole-projection partials
    ``d(u)/d(rho_cam) = f_px * [1/z, 0, -x/z^2]``,
    ``d(v)/d(rho_cam) = f_px * [0, 1/z, -y/z^2]`` are chained through.
    Verified against a convergence-checked central finite difference in
    ``examples/phase17_r1o_opt_oracles.py`` (R1O-OPT §73: an analytic
    Jacobian checked by FD, not "supported only by one arbitrary FD step").
    """
    r_sc = np.asarray(r_sc_inertial_m, dtype=float).reshape(3)
    r_lm = np.asarray(r_landmark_inertial_m, dtype=float).reshape(3)
    c = np.asarray(c_ci, dtype=float).reshape(3, 3)

    rho_cam = c @ (r_lm - r_sc)
    x, y, z = rho_cam
    if z <= 0.0:
        raise OpticalGeometryError("Landmark is not in front of the camera (z_cam <= 0).")
    f_px = intrinsics.focal_length_px

    d_uv_d_rhocam = np.array([
        [f_px / z, 0.0, -f_px * x / z ** 2],
        [0.0, f_px / z, -f_px * y / z ** 2],
    ])
    d_rhocam_d_rsc = -c
    return d_uv_d_rhocam @ d_rhocam_d_rsc


def landmark_optical_state_and_k_sensitivity(
    dg_dr: ArrayLike, phi: ArrayLike, s_k: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """(d(u,v)/dx0, d(u,v)/dK) from the position Jacobian and the trajectory's
    already-qualified variational history.  ``dg/dx0 = dg/dr @ Phi[:3, :]``,
    ``dg/dK = dg/dr @ S_K[:3]`` -- the SAME chain-rule substitution
    `_two_way_range_k_srp_column` (production range) and R1O's landmark
    surrogate both use; no implicit event solve is needed here because,
    unlike ΔDOR, this observable has no coupled light-time unknowns (light
    time is tested for significance separately, R1O-OPT §24, and is either
    negligible or applied as an explicit, non-implicit correction).
    """
    dg_dr = np.asarray(dg_dr, dtype=float).reshape(2, 3)
    phi = np.asarray(phi, dtype=float).reshape(6, 6)
    s_k = np.asarray(s_k, dtype=float).reshape(6)
    d_x0 = dg_dr @ phi[:3, :]
    d_k = dg_dr @ s_k[:3]
    return d_x0, d_k


# ======================================================================
# Landmark lunar-fixed -> inertial position (reuses qualified MOON_PA_DE440)
# ======================================================================
def landmark_inertial_position_m(lat_rad: float, lon_rad: float, r_moon_m: float, et: float) -> np.ndarray:
    """Spherical-Moon landmark position at inertial epoch ``et`` (TDB seconds past J2000).

    Uses the already-qualified ``moon_pa_de440_rotation_at_et`` (Phase
    17-R1M/17-R1O precedent) -- the versioned DE440 realization, never the
    load-order-dependent generic ``MOON_PA`` alias.
    """
    from .lunar_frames import moon_pa_de440_rotation_at_et

    lat, lon = float(lat_rad), float(lon_rad)
    r_pa = r_moon_m * np.array([
        np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat),
    ])
    c_pa = moon_pa_de440_rotation_at_et(et)
    return c_pa.T @ r_pa


def landmark_surface_outward_normal_j2000(lat_rad: float, lon_rad: float, et: float) -> np.ndarray:
    """Outward local-vertical unit normal at a spherical-Moon landmark, in J2000.

    For a spherical body the outward normal equals the unit position vector,
    so this is only a distinct function from ``landmark_inertial_position_m``
    for documentation clarity at the occultation-gate call sites below.
    """
    from .lunar_frames import moon_pa_de440_rotation_at_et

    lat, lon = float(lat_rad), float(lon_rad)
    n_pa = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    c_pa = moon_pa_de440_rotation_at_et(et)
    return c_pa.T @ n_pa


# ======================================================================
# Visibility gate (R1O-OPT s27-s29): occultation, front-of-camera, FOV
# ======================================================================
@dataclass(frozen=True)
class VisibilityResult:
    visible: bool
    reason: str
    off_nadir_from_landmark_deg: float  # local-horizon angle, occultation test


def landmark_geometrically_visible(
    r_sc_inertial_m: ArrayLike,
    r_landmark_inertial_m: ArrayLike,
    landmark_outward_normal: ArrayLike,
    c_ci: ArrayLike,
    intrinsics: CameraIntrinsics = DEFAULT_CAMERA,
) -> VisibilityResult:
    """Three independent geometric visibility conditions, checked in order:

    1. Occultation / self-horizon: the landmark's own local-vertical normal
       must have a positive component toward the spacecraft (the spacecraft
       is above the landmark's local horizon) -- otherwise the landmark is on
       the far side of the Moon as seen from the spacecraft.
    2. In front of the camera (``z_cam > 0``).
    3. Within the camera's field of view.

    This is a pure geometric gate; illumination/detectability (R1O-OPT §28)
    is characterized separately and NOT folded in here.
    """
    r_sc = np.asarray(r_sc_inertial_m, dtype=float).reshape(3)
    r_lm = np.asarray(r_landmark_inertial_m, dtype=float).reshape(3)
    n_hat = np.asarray(landmark_outward_normal, dtype=float).reshape(3)
    n_hat = n_hat / np.linalg.norm(n_hat)

    view_dir = r_sc - r_lm
    view_norm = float(np.linalg.norm(view_dir))
    cos_local = float(np.dot(n_hat, view_dir) / view_norm) if view_norm > 0 else -1.0
    off_nadir_deg = float(np.degrees(np.arccos(np.clip(cos_local, -1.0, 1.0))))
    if cos_local <= 0.0:
        return VisibilityResult(False, "OCCULTED_FAR_SIDE", off_nadir_deg)

    proj = pinhole_project(r_sc, r_lm, c_ci, intrinsics)
    if not proj.in_front_of_camera:
        return VisibilityResult(False, "BEHIND_CAMERA", off_nadir_deg)
    if not proj.within_fov:
        return VisibilityResult(False, "OUTSIDE_FOV", off_nadir_deg)
    return VisibilityResult(True, "VISIBLE", off_nadir_deg)


# ======================================================================
# Optical light time (R1O-OPT s24-25: tested, not assumed)
# ======================================================================
def optical_one_way_light_time_s(range_m: float, light_speed_mps: float = 299792458.0) -> float:
    """One-way light time landmark->camera.  A simple algebraic quantity (no
    iterative solve needed): the landmark is fixed in the lunar-body frame
    over the sub-millisecond light times at this campaign's ~105 km altitude,
    so, unlike the two-body spacecraft<->ground-station light-time solves
    elsewhere in this repository, there is no moving-target fixed-point
    iteration here -- only the question (tested in
    ``examples/phase17_r1o_opt_light_time.py``) of whether the SPACECRAFT's
    own motion during this light time is significant enough to need a
    light-time-corrected spacecraft position rather than the shutter-time
    position.
    """
    return float(range_m) / float(light_speed_mps)
