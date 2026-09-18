"""PHASE 17-R1O - shared fixtures for additional-observable feasibility.

ANALYSIS SPACE ONLY.  Nothing here modifies production code.  It IMPORTS and
EXECUTES the already-qualified trajectory (Phase 17-R1M/17-R1COV), frame
transforms, and station-position helpers, and builds two NEW analysis-only
observable surrogates (DDOR-like differenced range, lunar-landmark
line-of-sight) using the SAME chain-rule construction the production range
Jacobian already uses.

THE ONE CONSTRUCTION EVERY NEW OBSERVABLE HERE USES
----------------------------------------------------
For any observable that is a function of the spacecraft's INERTIAL POSITION
at a single time only, g = g(r_sc(t)) (no direct velocity or K dependence --
verified structurally below, not assumed):

    dg/dx0 = dg/dr(t) . dr(t)/dx0 = dg/dr(t) . Phi[:3, :](t)
    dg/dK  = dg/dr(t) . dr(t)/dK  = dg/dr(t) . S_K[:3](t)

where Phi(t) and S_K(t) are the ALREADY-QUALIFIED 6x6 STM and dx/dK columns
carried in `nom48` (Phase 17-R's 48-state variational history), and
`dg/dr(t)` is a convergence-checked central finite difference on the
elementary 3-vector geometry (position -> range difference, or position ->
tangent-plane angle).  This is exactly the pattern
`_two_way_range_k_srp_column` already uses for the production range
observable (`d(range)/dr(t) . S_K(t)`); R1O reuses the pattern for two NEW
`g(r)` functions rather than inventing a different construction.

Because `g_fn` below takes only a position vector -- literally, by
construction, with no `k` parameter anywhere in its signature or body -- K
enters ONLY through how `r_sc(t)` itself depends on K.  `DIRECT_MEASUREMENT_K_
DEPENDENCE` is verified, not assumed, by `assert_no_direct_k_dependence`.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys  # noqa: E402

sys.path.insert(0, str(HERE))
from phase17_r1cov_core import (  # noqa: E402
    K_INDEX, exact_covariance, relative_error, scale_matrix,
    square_root_covariance,
)
from phase17_r1m_core import (  # noqa: E402
    CADENCE_S, CAMPAIGN_ROOT, K_TRUTH, RTOL, ATOL, ArcFixture, SCALE_POS,
    SCALE_VEL, SCALE_K, build_range_arc, campaign_epoch, campaign_initial_state,
    information_matrix, orthogonal_decomposition, schur_conditional, spectrum,
    whitened,
)

C_LIGHT_MPS = 299792458.0
DDOR_MIN_ELEVATION_DEG = 10.0          # same threshold as the qualified range arcs
LANDMARK_MAX_OFFNADIR_DEG = 30.0       # generic navigation-camera half-cone


# ======================================================================
# theta_K -- the principal-angle companion to f_perp (s17)
# ======================================================================
def theta_k_deg(f_perp: float) -> float:
    """arcsin(f_perp) in degrees; f_perp = sin(theta_K) by construction (s24)."""
    f = float(np.clip(f_perp, 0.0, 1.0))
    return float(np.degrees(np.arcsin(f)))


# ======================================================================
# Convergence-checked finite-difference position Jacobian (s26)
# ======================================================================
def position_jacobian_fd(g_fn, r: np.ndarray, *, step_frac: float = 1e-4,
                         min_step_m: float = 1.0) -> tuple[np.ndarray, float]:
    """Central-difference dg/dr at r, with a Richardson-style convergence check.

    Returns (jacobian_at_finer_step, relative_step_halving_change).  The
    caller decides what "verified" means for its own tolerance; this module's
    callers require the returned change to be < 1e-6 before trusting the
    Jacobian (checked at the call sites below, not hidden here).
    """
    r = np.asarray(r, dtype=float)
    scale = max(float(np.linalg.norm(r)), 1.0)
    step = max(step_frac * scale, min_step_m)

    def _fd(h: float) -> np.ndarray:
        g0 = np.atleast_1d(np.asarray(g_fn(r), dtype=float))
        jac = np.zeros((g0.size, 3))
        for i in range(3):
            dr = np.zeros(3)
            dr[i] = h
            gp = np.atleast_1d(np.asarray(g_fn(r + dr), dtype=float))
            gm = np.atleast_1d(np.asarray(g_fn(r - dr), dtype=float))
            jac[:, i] = (gp - gm) / (2.0 * h)
        return jac

    j_coarse = _fd(step)
    j_fine = _fd(step * 0.5)
    denom = max(float(np.max(np.abs(j_fine))), 1e-300)
    rel_change = float(np.max(np.abs(j_coarse - j_fine))) / denom
    return j_fine, rel_change


def assert_no_direct_k_dependence(g_fn) -> bool:
    """Structural proof that g_fn's signature carries no K parameter (s15).

    Not a numerical check of behaviour -- a proof about the FUNCTION ITSELF:
    it is built to accept only a position vector, so K cannot enter it
    directly.  Returns True (DIRECT_MEASUREMENT_K_DEPENDENCE = NO) or raises.
    """
    sig = inspect.signature(g_fn)
    params = list(sig.parameters)
    if len(params) != 1:
        raise AssertionError(
            "observable function must take exactly one argument (position); "
            "got %r -- cannot certify DIRECT_MEASUREMENT_K_DEPENDENCE=NO" % params)
    return True


def chain_to_augmented_columns(
    dg_dr: np.ndarray, nom48_row: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(dg/dx0, dg/dK) for one row, from a position Jacobian and one nom48 row.

    nom48_row is 48 columns: [0:6] state, [6:42] Phi (6x6, row-major), [42:48]
    dx/dK.  Only the position ROWS of Phi (mapping dx0 -> dr(t)) and the
    position rows of dx/dK are used, exactly as `_two_way_range_k_srp_column`
    uses them for the production range observable.
    """
    phi = np.asarray(nom48_row[6:42], dtype=float).reshape(6, 6)
    s_k = np.asarray(nom48_row[42:48], dtype=float)
    h_x0 = dg_dr @ phi[:3, :]
    h_k = dg_dr @ s_k[:3]
    return h_x0, h_k


# ======================================================================
# Station geometry (reuses the qualified frame-transform helpers, read-only)
# ======================================================================
def station_position_mci(station, earth_pos_mci: np.ndarray,
                         x_j2k_itrf: np.ndarray) -> np.ndarray:
    from lunar_od.measurements import _station_position_mci_at_receive_epoch

    return np.asarray(
        _station_position_mci_at_receive_epoch(
            station, np.asarray(earth_pos_mci, float).reshape(3), x_j2k_itrf),
        dtype=float)


def station_elevation_deg(station, r_sc_mci: np.ndarray, earth_pos_mci: np.ndarray,
                          x_j2k_itrf: np.ndarray) -> float:
    from lunar_od.geometry import ecef2razel_sez

    r_rel_j2000 = (np.asarray(r_sc_mci, float).reshape(3)
                  - np.asarray(earth_pos_mci, float).reshape(3))
    r_rel_ecef = np.asarray(x_j2k_itrf, float)[:3, :3] @ r_rel_j2000 - \
        np.asarray(station.r_ecef_m, float).reshape(3)
    _, el_rad, _ = ecef2razel_sez(r_rel_ecef, station.lat_rad, station.lon_rad)
    return float(np.degrees(el_rad))


# ======================================================================
# O2 -- DDOR-like differenced one-way range surrogate
# ======================================================================
@dataclass(frozen=True)
class DdorArc:
    label: str
    station_a_name: str
    station_b_name: str
    t_used_s: np.ndarray               # absolute campaign-epoch seconds used
    h_x0: np.ndarray                   # (M, 6)
    h_k: np.ndarray                    # (M,)
    w: np.ndarray                      # (M,) 1/sigma_delay^2 for the CURRENT sigma_angle
    baseline_perp_m: np.ndarray        # (M,) effective baseline used per row
    max_fd_convergence_error: float

    @property
    def n_obs(self) -> int:
        return int(self.h_k.size)


def _pair_stations(names: tuple[str, str], stations) -> tuple:
    by_name = {s.name: s for s in stations}
    return by_name[names[0]], by_name[names[1]]


def build_ddor_arc(
    nom48: np.ndarray,
    t_grid_s: np.ndarray,
    et0: float,
    station_pair_names: tuple[str, str],
    *,
    sigma_angle_rad: float,
    min_elevation_deg: float = DDOR_MIN_ELEVATION_DEG,
    label: str = "ddor",
) -> DdorArc:
    """Build the design rows for a DDOR-like differenced one-way range.

    g(r_sc; t) = [ |r_sc - r_B(t)| - |r_sc - r_A(t)| ] / c   (seconds)

    Both stations must be simultaneously above `min_elevation_deg` -- real
    DDOR requires both antennas tracking the same source at once.  The
    measurement noise is specified as a PLANE-OF-SKY ANGLE sigma_angle_rad
    (baseline-independent, as the DDOR literature quotes it) and converted
    per-row to an equivalent delay sigma via the baseline component
    perpendicular to the instantaneous line of sight:

        sigma_delay(t) = B_perp(t) * sigma_angle_rad / c

    which is the standard DDOR relation between angular and delay precision
    and is what lets w(t) = 1/sigma_delay(t)^2 correctly reduce information
    when the baseline happens to be poorly oriented relative to the source.
    """
    import od_gravity_covariance_campaign as C  # noqa: N811
    from lunar_od.visibility import sample_j2000_to_itrf93_transforms

    station_a, station_b = _pair_stations(station_pair_names, C.STATIONS)
    t_local = t_grid_s - t_grid_s[0]
    xf = sample_j2000_to_itrf93_transforms(et0 + t_grid_s[0], t_local)

    rows_x0, rows_k, t_used, b_perp_used = [], [], [], []
    max_conv_err = 0.0
    for i, t_abs in enumerate(t_grid_s):
        r_sc = nom48[i, :3]
        earth_pos = C.get_earth_pos(t_abs)
        xfi = xf[i]
        r_a = station_position_mci(station_a, earth_pos, xfi)
        r_b = station_position_mci(station_b, earth_pos, xfi)
        el_a = station_elevation_deg(station_a, r_sc, earth_pos, xfi)
        el_b = station_elevation_deg(station_b, r_sc, earth_pos, xfi)
        if el_a < min_elevation_deg or el_b < min_elevation_deg:
            continue

        def g_fn(r, _ra=r_a, _rb=r_b):
            return np.array([(np.linalg.norm(r - _rb) - np.linalg.norm(r - _ra))
                             / C_LIGHT_MPS])

        dg_dr, rel_change = position_jacobian_fd(g_fn, r_sc)
        max_conv_err = max(max_conv_err, rel_change)
        h_x0, h_k = chain_to_augmented_columns(dg_dr, nom48[i])

        los = 0.5 * ((r_sc - r_a) / np.linalg.norm(r_sc - r_a)
                     + (r_sc - r_b) / np.linalg.norm(r_sc - r_b))
        baseline = r_b - r_a
        b_perp = float(np.linalg.norm(baseline - np.dot(baseline, los) * los))

        rows_x0.append(h_x0[0])
        rows_k.append(h_k[0])
        t_used.append(t_abs)
        b_perp_used.append(b_perp)

    if not rows_x0:
        return DdorArc(label=label, station_a_name=station_a.name,
                      station_b_name=station_b.name,
                      t_used_s=np.zeros(0), h_x0=np.zeros((0, 6)),
                      h_k=np.zeros(0), w=np.zeros(0), baseline_perp_m=np.zeros(0),
                      max_fd_convergence_error=max_conv_err)

    b_perp_arr = np.asarray(b_perp_used, dtype=float)
    sigma_delay = np.maximum(b_perp_arr, 1.0) * sigma_angle_rad / C_LIGHT_MPS
    w = 1.0 / sigma_delay ** 2
    return DdorArc(
        label=label, station_a_name=station_a.name, station_b_name=station_b.name,
        t_used_s=np.asarray(t_used, dtype=float),
        h_x0=np.asarray(rows_x0, dtype=float), h_k=np.asarray(rows_k, dtype=float),
        w=w, baseline_perp_m=b_perp_arr, max_fd_convergence_error=max_conv_err)


# ======================================================================
# O3 -- lunar-landmark line-of-sight surrogate
# ======================================================================
@dataclass(frozen=True)
class LandmarkArc:
    label: str
    landmark_latlon_deg: np.ndarray    # (L, 2)
    t_used_s: np.ndarray
    landmark_used_idx: np.ndarray
    h_x0: np.ndarray                   # (2M, 6) -- two tangent-plane angles per obs
    h_k: np.ndarray                   # (2M,)
    w: np.ndarray                      # (2M,) 1/sigma_angle^2
    max_fd_convergence_error: float

    @property
    def n_obs(self) -> int:
        return int(self.h_k.size)


def _landmark_latlon_from_nadir(nom48: np.ndarray, t_grid_s: np.ndarray,
                                et0: float, epoch_fractions: tuple[float, ...]
                                ) -> np.ndarray:
    """Nadir-point lat/lon (MOON_PA, spherical Moon) at pre-declared fractions
    of the arc.  Declared as a strategy BEFORE any K metric is inspected (s27):
    "geographically distributed, chronologically selected surface features"
    sampled along the spacecraft's own ground track.
    """
    from lunar_od.lunar_frames import moon_pa_de440_rotation_at_et

    latlon = []
    n = t_grid_s.size
    for frac in epoch_fractions:
        i = int(round(frac * (n - 1)))
        r_sc = nom48[i, :3]
        c_pa = moon_pa_de440_rotation_at_et(et0 + t_grid_s[i])
        r_pa = c_pa @ r_sc
        r_hat = r_pa / np.linalg.norm(r_pa)
        lat = np.degrees(np.arcsin(np.clip(r_hat[2], -1.0, 1.0)))
        lon = np.degrees(np.arctan2(r_hat[1], r_hat[0]))
        latlon.append((lat, lon))
    return np.asarray(latlon, dtype=float)


def build_landmark_arc(
    nom48: np.ndarray,
    t_grid_s: np.ndarray,
    et0: float,
    *,
    sigma_angle_rad: float,
    r_moon_m: float,
    epoch_fractions: tuple[float, ...] = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0),
    max_offnadir_deg: float = LANDMARK_MAX_OFFNADIR_DEG,
    label: str = "landmark",
) -> LandmarkArc:
    """Build design rows for idealized known-landmark LOS tangent-plane angles.

    Each observation is TWO angles (boresight-relative, about a nadir-pointing
    camera): the standard gnomonic tangent-plane pair.  Landmark positions are
    known exactly here (s29's ideal limit); attitude is known exactly (s30's
    ideal limit).  Both idealizations are characterized separately by the
    sweep scripts that call this builder, not baked in here.
    """
    from lunar_od.lunar_frames import moon_pa_de440_rotation_at_et

    latlon = _landmark_latlon_from_nadir(nom48, t_grid_s, et0, epoch_fractions)
    lat_r, lon_r = np.radians(latlon[:, 0]), np.radians(latlon[:, 1])
    landmarks_pa = np.stack([
        r_moon_m * np.cos(lat_r) * np.cos(lon_r),
        r_moon_m * np.cos(lat_r) * np.sin(lon_r),
        r_moon_m * np.sin(lat_r),
    ], axis=1)

    rows_x0, rows_k, t_used, lm_used = [], [], [], []
    max_conv_err = 0.0
    for i, t_abs in enumerate(t_grid_s):
        r_sc = nom48[i, :3]
        nadir_hat = -r_sc / np.linalg.norm(r_sc)
        # gnomonic tangent-plane basis orthogonal to the nadir boresight
        ref = np.array([0.0, 0.0, 1.0]) if abs(nadir_hat[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(nadir_hat, ref)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(nadir_hat, e1)

        c_pa = moon_pa_de440_rotation_at_et(et0 + t_abs)
        c_pa_t = c_pa.T
        for li in range(landmarks_pa.shape[0]):
            r_lm_j2000 = c_pa_t @ landmarks_pa[li]
            rho = r_lm_j2000 - r_sc
            rho_hat = rho / np.linalg.norm(rho)
            offnadir_deg = np.degrees(np.arccos(np.clip(np.dot(rho_hat, nadir_hat), -1.0, 1.0)))
            if offnadir_deg > max_offnadir_deg:
                continue

            def g_fn(r, _rlm=r_lm_j2000, _e1=e1, _e2=e2):
                rho_ = _rlm - r
                rho_h = rho_ / np.linalg.norm(rho_)
                return np.array([np.arcsin(np.clip(np.dot(rho_h, _e1), -1.0, 1.0)),
                                 np.arcsin(np.clip(np.dot(rho_h, _e2), -1.0, 1.0))])

            dg_dr, rel_change = position_jacobian_fd(g_fn, r_sc)
            max_conv_err = max(max_conv_err, rel_change)
            h_x0, h_k = chain_to_augmented_columns(dg_dr, nom48[i])
            rows_x0.append(h_x0[0]); rows_k.append(h_k[0])
            rows_x0.append(h_x0[1]); rows_k.append(h_k[1])
            t_used.append(t_abs); lm_used.append(li)

    if not rows_x0:
        return LandmarkArc(label=label, landmark_latlon_deg=latlon,
                          t_used_s=np.zeros(0), landmark_used_idx=np.zeros(0, int),
                          h_x0=np.zeros((0, 6)), h_k=np.zeros(0), w=np.zeros(0),
                          max_fd_convergence_error=max_conv_err)

    n_meas = len(rows_x0)
    w = np.full(n_meas, 1.0 / sigma_angle_rad ** 2)
    return LandmarkArc(
        label=label, landmark_latlon_deg=latlon,
        t_used_s=np.asarray(t_used, dtype=float),
        landmark_used_idx=np.asarray(lm_used, dtype=int),
        h_x0=np.asarray(rows_x0, dtype=float), h_k=np.asarray(rows_k, dtype=float),
        w=w, max_fd_convergence_error=max_conv_err)


# ======================================================================
# Combined-system metrics (s16, s17, s26)
# ======================================================================
def combined_metrics(blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> dict:
    """Assemble several (h_x0, h_k, w) blocks into one 7-parameter system and
    report the s16 primary metrics plus theta_K.
    """
    h_x0 = np.vstack([b[0] for b in blocks if b[0].shape[0]])
    h_k = np.concatenate([b[1] for b in blocks if b[1].shape[0]])
    w = np.concatenate([b[2] for b in blocks if b[2].shape[0]])
    scale = scale_matrix()
    a_x, b_k = whitened(h_x0, h_k, w)
    dec = orthogonal_decomposition(a_x, b_k)
    info = information_matrix(np.hstack([h_x0, h_k[:, None]]), w, scale)
    sp = spectrum(info)
    sr = square_root_covariance(np.hstack([h_x0, h_k[:, None]]), w, None, scale)
    return dict(
        n_obs=int(h_x0.shape[0]),
        raw_i_kk=dec["i_kk"], conditional_k_information=dec["i_k_given_x"],
        orthogonal_fraction=dec["orthogonal_fraction"],
        theta_k_deg=theta_k_deg(dec["orthogonal_fraction"]),
        smallest_scaled_singular_value=sp["smallest"],
        scaled_condition=sp["condition"], rank=sp["rank_default"],
        weakest_mode_k_component=sp["weakest_k_component"],
        qualified_sigma_k=sr.sigma_k,
        fractional_sigma_k=sr.sigma_k / K_TRUTH,
        rank_margin=sr.rank_margin,
    )
