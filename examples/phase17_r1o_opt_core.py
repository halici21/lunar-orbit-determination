"""PHASE 17-R1O-OPT - shared fixtures for the PRODUCTION optical measurement.

ANALYSIS SPACE ONLY.  Nothing here modifies production code; it EXECUTES the
new production module `lunar_od/lunar_landmark_optical.py` against the
already-qualified campaign trajectory (Phase 17-R1M/17-R1COV/17-R1O) and the
already-qualified MOON_PA_DE440 frame path.

LANDMARK SELECTION IS PRE-DECLARED (R1O-OPT s29, hard block s75)
----------------------------------------------------------------
The landmark set is inherited VERBATIM from Phase 17-R1O's own
`_landmark_latlon_from_nadir`: nadir-ground-track points at the fixed epoch
fractions (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0) of the arc, spherical
Moon.  That rule was fixed and committed in a PRIOR phase, before this
phase's production model existed and therefore before any production K
result could have influenced it.  This is the strongest available form of
pre-declaration -- not a promise about ordering within this phase, but a
selection rule that is literally imported from already-committed code.
`LANDMARK_SELECTION_NOT_K_TUNED = YES`.

ERROR SOURCES ARE NOT INTERCHANGEABLE (R1O-OPT s17-s20, s30-s33)
----------------------------------------------------------------
This builder accepts RANDOM per-observation errors (which legitimately
inflate the measurement-noise variance and therefore the information weight)
and, separately, returns the machinery needed to treat COHERENT errors --
attitude/boresight bias and per-landmark catalog bias -- with the two
DISTINCT formulas R1O-D established: an omitted-variable point-estimate
shift, and a nuisance-parameter covariance inflation.  A coherent error is
NEVER folded into the noise variance here; doing so is the exact conflation
R1O-D caught mid-phase and corrected.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from phase17_r1o_core import _landmark_latlon_from_nadir  # noqa: E402  (pre-declared rule)
from phase17_r1m_core import K_TRUTH  # noqa: E402

from lunar_od.constants import R_MOON_M  # noqa: E402
from lunar_od.lunar_landmark_optical import (  # noqa: E402
    CameraIntrinsics, DEFAULT_CAMERA, landmark_geometrically_visible,
    landmark_inertial_position_m, landmark_optical_state_and_k_sensitivity,
    landmark_surface_outward_normal_j2000, nadir_pointing_camera_frame,
    perturbed_camera_frame, pinhole_position_jacobian, pinhole_project,
)

# Pre-declared landmark epoch fractions, inherited verbatim from Phase 17-R1O.
PREDECLARED_EPOCH_FRACTIONS = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0)


@dataclass(frozen=True)
class OpticalArc:
    """Design rows for the production image-plane (u, v) observable."""

    label: str
    landmark_latlon_deg: np.ndarray     # (L, 2)
    t_used_s: np.ndarray                # (M,) absolute campaign seconds, one per SCALAR row
    landmark_used_idx: np.ndarray       # (M,) which landmark produced each scalar row
    h_x0: np.ndarray                    # (M, 6)  d(pixel)/dx0
    h_k: np.ndarray                     # (M,)    d(pixel)/dK
    w: np.ndarray                       # (M,)    1/sigma_px^2
    sigma_px_effective: float
    n_visible_obs: int                  # number of (landmark, epoch) pairs that passed the gate
    mean_range_m: float
    mean_off_boresight_deg: float
    # Coherent-error design columns (NOT folded into w):
    h_attitude_bias: np.ndarray         # (M, 3) d(pixel)/d(attitude bias angles)
    h_landmark_bias: np.ndarray         # (M, 3L) d(pixel)/d(each landmark's PA position error)

    @property
    def n_rows(self) -> int:
        return int(self.h_k.size)


def build_production_optical_arc(
    nom48: np.ndarray,
    t_grid_s: np.ndarray,
    et0: float,
    *,
    sigma_centroid_px: float,
    attitude_random_sigma_rad: float = 0.0,
    map_random_sigma_m: float = 0.0,
    camera: CameraIntrinsics = DEFAULT_CAMERA,
    epoch_fractions: tuple[float, ...] = PREDECLARED_EPOCH_FRACTIONS,
    label: str = "optical",
    attitude_bias_rad: np.ndarray | None = None,
) -> OpticalArc:
    """Build (u, v) pixel design rows through the full production chain.

    RANDOM error sources (per-observation independent) inflate sigma_px in
    quadrature -- legitimate, because independent per-observation errors ARE
    measurement noise.  COHERENT sources are not touched here; their design
    columns are returned separately for the two bias formulas.

    ``attitude_bias_rad``, if given, perturbs the NOMINAL camera frame used
    to evaluate the measurement -- i.e. it moves the model, exactly as a real
    uncorrected boresight misalignment would.  It does NOT change ``w``.
    """
    latlon = _landmark_latlon_from_nadir(nom48, t_grid_s, et0, epoch_fractions)
    lat_r, lon_r = np.radians(latlon[:, 0]), np.radians(latlon[:, 1])
    n_landmarks = latlon.shape[0]
    ifov_rad = camera.pixel_pitch_m / camera.focal_length_m

    rows_x0: list[np.ndarray] = []
    rows_k: list[float] = []
    rows_att: list[np.ndarray] = []
    rows_lm: list[np.ndarray] = []
    t_used: list[float] = []
    lm_used: list[int] = []
    ranges: list[float] = []
    off_bores: list[float] = []
    n_visible = 0

    for i, t_abs in enumerate(t_grid_s):
        r_sc = nom48[i, :3]
        et = et0 + t_abs
        c_ci_nominal = nadir_pointing_camera_frame(r_sc)
        c_ci = (perturbed_camera_frame(c_ci_nominal, attitude_bias_rad)
                if attitude_bias_rad is not None else c_ci_nominal)

        for li in range(n_landmarks):
            r_lm = landmark_inertial_position_m(lat_r[li], lon_r[li], R_MOON_M, et)
            n_hat = landmark_surface_outward_normal_j2000(lat_r[li], lon_r[li], et)
            vis = landmark_geometrically_visible(r_sc, r_lm, n_hat, c_ci, camera)
            if not vis.visible:
                continue
            n_visible += 1

            proj = pinhole_project(r_sc, r_lm, c_ci, camera)
            dg_dr = pinhole_position_jacobian(r_sc, r_lm, c_ci, camera)
            # Phi MUST be unflattened column-major: `lunar_od.dynamics` packs
            # it as `np.eye(6).reshape(-1, order="F")` and propagates it that
            # way, and every production consumer (`delta_dor.py`,
            # `two_way_range_event_sensitivity`, `accelerated.py`) reshapes
            # with order="F".  Phase 17-R1O's own analysis helper
            # `chain_to_augmented_columns` used the DEFAULT (C) order, i.e.
            # Phi^T -- see this phase's report §40 for the end-to-end FD
            # evidence and the consequences for R1O's published surrogate
            # numbers.
            h_x0, h_k = landmark_optical_state_and_k_sensitivity(
                dg_dr, nom48[i, 6:42].reshape((6, 6), order="F"), nom48[i, 42:48])

            # Coherent-error design columns.
            #
            # Attitude bias: d(pixel)/d(bias angle).  The bias rotates the
            # camera frame, so rho_cam -> (I - skew(a)) rho_cam, giving
            # d(rho_cam)/da = -skew'(rho_cam) = +skew(rho_cam) at a = 0
            # (since -skew(a) v = -(a x v) = +(v x a) = skew(v) a).
            rho_cam = c_ci @ (r_lm - r_sc)
            z = float(rho_cam[2])
            f_px = camera.focal_length_px
            d_uv_d_rhocam = np.array([
                [f_px / z, 0.0, -f_px * rho_cam[0] / z ** 2],
                [0.0, f_px / z, -f_px * rho_cam[1] / z ** 2],
            ])
            skew_rho = np.array([
                [0.0, -rho_cam[2], rho_cam[1]],
                [rho_cam[2], 0.0, -rho_cam[0]],
                [-rho_cam[1], rho_cam[0], 0.0],
            ])
            h_att = d_uv_d_rhocam @ skew_rho          # (2, 3)

            # Landmark catalog bias: d(pixel)/d(landmark inertial position),
            # which is +C_ci into the same projection partials (opposite sign
            # to the spacecraft-position Jacobian, since rho = r_lm - r_sc).
            h_lm_local = d_uv_d_rhocam @ c_ci          # (2, 3)
            h_lm_full = np.zeros((2, 3 * n_landmarks))
            h_lm_full[:, 3 * li:3 * li + 3] = h_lm_local

            for row in range(2):
                rows_x0.append(h_x0[row])
                rows_k.append(float(h_k[row]))
                rows_att.append(h_att[row])
                rows_lm.append(h_lm_full[row])
                t_used.append(t_abs)
                lm_used.append(li)
            ranges.append(proj.range_m)
            off_bores.append(np.degrees(proj.off_boresight_rad))

    if not rows_x0:
        empty = np.zeros((0, 6))
        return OpticalArc(
            label=label, landmark_latlon_deg=latlon, t_used_s=np.zeros(0),
            landmark_used_idx=np.zeros(0, int), h_x0=empty, h_k=np.zeros(0),
            w=np.zeros(0), sigma_px_effective=float("nan"), n_visible_obs=0,
            mean_range_m=float("nan"), mean_off_boresight_deg=float("nan"),
            h_attitude_bias=np.zeros((0, 3)),
            h_landmark_bias=np.zeros((0, 3 * n_landmarks)))

    mean_range = float(np.mean(ranges))
    # RANDOM error budget, combined in quadrature IN THE PIXEL DOMAIN.
    sigma_att_px = attitude_random_sigma_rad / ifov_rad
    # A random per-observation map error of sigma_map metres subtends
    # sigma_map / range radians at the camera.
    sigma_map_px = (map_random_sigma_m / mean_range) / ifov_rad if mean_range > 0 else 0.0
    sigma_px = float(np.sqrt(sigma_centroid_px ** 2 + sigma_att_px ** 2 + sigma_map_px ** 2))

    n_rows = len(rows_x0)
    return OpticalArc(
        label=label,
        landmark_latlon_deg=latlon,
        t_used_s=np.asarray(t_used, dtype=float),
        landmark_used_idx=np.asarray(lm_used, dtype=int),
        h_x0=np.asarray(rows_x0, dtype=float),
        h_k=np.asarray(rows_k, dtype=float),
        w=np.full(n_rows, 1.0 / sigma_px ** 2),
        sigma_px_effective=sigma_px,
        n_visible_obs=n_visible,
        mean_range_m=mean_range,
        mean_off_boresight_deg=float(np.mean(off_bores)),
        h_attitude_bias=np.asarray(rows_att, dtype=float),
        h_landmark_bias=np.asarray(rows_lm, dtype=float),
    )


# ======================================================================
# The two DISTINCT coherent-error formulas (R1O-D precedent, s34/s36)
# ======================================================================
def omitted_variable_k_shift(
    h_x0: np.ndarray, h_k: np.ndarray, w: np.ndarray,
    h_bias: np.ndarray, bias_value: np.ndarray,
) -> float:
    """Case B: an UNMODELED coherent error shifts the POINT ESTIMATE.

    Standard omitted-variable bias: with design ``H = [H_x, h_K]`` and an
    unmodeled systematic contribution ``H_b b`` to the residuals, the induced
    parameter shift is ``(H^T W H)^-1 H^T W H_b b``; the K component of that
    shift is returned.  This is NOT the covariance-inflation formula below --
    R1O-D found the two differ by more than two orders of magnitude in
    practical impact, and using one for the other hides the real risk.
    """
    h = np.hstack([h_x0, h_k[:, None]])
    normal = h.T @ (w[:, None] * h)
    residual_from_bias = h_bias @ np.asarray(bias_value, float).reshape(-1)   # (N,)
    rhs = h.T @ (w * residual_from_bias)                                       # (7,)
    shift = np.linalg.solve(normal, rhs)
    return float(shift[-1])


def nuisance_solved_k_sigma(
    h_x0: np.ndarray, h_k: np.ndarray, w: np.ndarray,
    h_bias: np.ndarray, bias_prior_sigma: np.ndarray,
) -> float:
    """Case C: the coherent error is SOLVED FOR, inflating K's covariance only.

    Builds the joint system ``[H_x, h_K, H_b]`` with a prior on the bias
    block, recovers the covariance through the R1COV-qualified square-root
    (QR) path -- never the normal matrix, per R1O-OPT hard block s74 -- and
    returns the marginal K sigma.
    """
    from phase17_r1cov_core import square_root_covariance, scale_matrix

    n_bias = h_bias.shape[1]
    prior_sigma = np.asarray(bias_prior_sigma, float).reshape(-1)
    if prior_sigma.size == 1:
        prior_sigma = np.full(n_bias, float(prior_sigma[0]))

    h_full = np.hstack([h_x0, h_k[:, None], h_bias])
    base_scale = np.diag(scale_matrix())            # (7,) state+K scaling
    scale = np.diag(np.concatenate([base_scale, prior_sigma]))
    # Prior: infinite (uninformative) on state+K, ``prior_sigma`` on the bias
    # block -- expressed as an inverse-covariance block with zeros on the
    # state/K diagonal.
    prior_inv = np.zeros((7 + n_bias, 7 + n_bias))
    prior_inv[7:, 7:] = np.diag(1.0 / prior_sigma ** 2)
    sr = square_root_covariance(h_full, w, prior_inv, scale)
    return float(sr.sigma_k)


def fractional(sigma_k: float) -> float:
    return float(sigma_k / K_TRUTH)
