"""PHASE 17-GEO - geometry construction, descriptors, arcs and metrics.

ANALYSIS SPACE ONLY.  No production file is modified.  Every physical quantity
here is computed by the production modules (dynamics, srp, two_way_range,
visibility, estimators); this module only arranges geometries and reads the
resulting design matrices.

FRAMES
------
Orbit geometry is specified in the LUNAR MEAN-EQUATOR frame (LME): the frame
`lunar_od.dynamics._MCI_TO_MOON_BF` rotates MCI (J2000 axes, Moon-centred) into,
and the frame whose z axis the production J2 term uses as the lunar pole.  So an
"inclination" here is the inclination the dynamics actually sees.

STM CONTRACT (R1O-R)
--------------------
Every unpacking of the propagated Phi block uses order="F".  Nothing in this
module reshapes columns [6:42] at all: state columns come from the production
two-way range Jacobian, which already honours the contract.

THE FIXTURE-EPOCH REPAIR (GEO s14/s15)
--------------------------------------
The frozen R1M/R1O fixture `phase17_r1m_core.build_range_arc` takes the Sun and
Earth rotation from the manifest epoch but the Earth's POSITION from the campaign
module, whose table starts 504000 s (5.83 d) earlier and is clamped after 4
orbits.  `build_geo_range_arc` below uses ONE epoch for all three and exact
SPICE Earth states over the whole arc.  G0 reports both, so the effect of the
repair is measured on the canonical orbit before any geometry is varied.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from phase17_r1cov_core import square_root_covariance  # noqa: E402
from phase17_r1m_core import (  # noqa: E402
    ATOL, K_TRUTH, RTOL, campaign_epoch, campaign_initial_state,
    orthogonal_decomposition, scale_matrix, whitened,
)

C_LIGHT = 299_792_458.0


# ======================================================================
# Frames and orbital elements
# ======================================================================
def mci_to_lme() -> np.ndarray:
    from lunar_od.dynamics import _MCI_TO_MOON_BF
    return np.asarray(_MCI_TO_MOON_BF, dtype=float)


def mu_moon() -> float:
    from lunar_od.constants import MU_MOON_M3S2
    return float(MU_MOON_M3S2)


def r_moon() -> float:
    from lunar_od.constants import R_MOON_M
    return float(R_MOON_M)


def elements_from_state(x_mci: np.ndarray) -> dict:
    """Osculating Keplerian elements in the LME frame (degrees, metres)."""
    mu = mu_moon()
    c = mci_to_lme()
    r = c @ np.asarray(x_mci[:3], float)
    v = c @ np.asarray(x_mci[3:6], float)
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    e_vec = np.cross(v, h) / mu - r / rn
    e = float(np.linalg.norm(e_vec))
    a = 1.0 / (2.0 / rn - vn * vn / mu)
    inc = math.degrees(math.acos(max(-1.0, min(1.0, h[2] / hn))))
    n_vec = np.cross([0.0, 0.0, 1.0], h)
    nn = np.linalg.norm(n_vec)
    raan = math.degrees(math.atan2(n_vec[1], n_vec[0])) % 360.0 if nn > 0 else 0.0
    return dict(a_m=float(a), e=e, inc_deg=inc, raan_deg=raan,
                period_s=float(2 * math.pi * math.sqrt(a ** 3 / mu)),
                r_now_m=float(rn), h_hat_mci=(c.T @ (h / hn)).tolist())


def state_from_elements(a_m: float, e: float, inc_deg: float, raan_deg: float,
                        argp_deg: float, nu_deg: float) -> np.ndarray:
    """Cartesian MCI state from LME-frame Keplerian elements."""
    mu = mu_moon()
    i, O, w, nu = (math.radians(x) for x in (inc_deg, raan_deg, argp_deg, nu_deg))
    p = a_m * (1 - e * e)
    r_pf = p / (1 + e * math.cos(nu)) * np.array([math.cos(nu), math.sin(nu), 0.0])
    v_pf = math.sqrt(mu / p) * np.array([-math.sin(nu), e + math.cos(nu), 0.0])

    def rz(t):
        return np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])

    def rx(t):
        return np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])

    q = rz(O) @ rx(i) @ rz(w)
    c = mci_to_lme()
    return np.concatenate([c.T @ (q @ r_pf), c.T @ (q @ v_pf)])


def orbit_normal_lme(inc_deg: float, raan_deg: float) -> np.ndarray:
    i, O = math.radians(inc_deg), math.radians(raan_deg)
    return np.array([math.sin(i) * math.sin(O), -math.sin(i) * math.cos(O), math.cos(i)])


# ======================================================================
# Ephemerides (one epoch for everything)
# ======================================================================
def sun_mci(et: float) -> np.ndarray:
    import spiceypy as spice
    return np.asarray(spice.spkezr("SUN", float(et), "J2000", "NONE", "MOON")[0][:3]) * 1e3


def earth_state_mci(et: float) -> np.ndarray:
    import spiceypy as spice
    return np.asarray(spice.spkezr("EARTH", float(et), "J2000", "NONE", "MOON")[0]) * 1e3


def beta_deg(h_hat_mci: np.ndarray, et: float) -> float:
    """Signed solar beta: angle of the Moon->Sun direction above the orbit plane."""
    s = sun_mci(et)
    s /= np.linalg.norm(s)
    return float(math.degrees(math.asin(max(-1.0, min(1.0, float(np.dot(h_hat_mci, s)))))))


def earth_view_deg(h_hat_mci: np.ndarray, et: float) -> float:
    """Angle between the orbit normal and the Moon->Earth line.

    0 or 180 deg = orbit seen FACE-ON from Earth; 90 deg = EDGE-ON
    (the Slojkowski 2014 convention for the LRO face-on/edge-on cycle).
    """
    e = earth_state_mci(et)[:3]
    e /= np.linalg.norm(e)
    return float(math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(h_hat_mci, e)))))))


def raan_for_beta(inc_deg: float, beta_target_deg: float, et: float,
                  n_grid: int = 3601) -> list[float]:
    """All LME RAANs (deg) at which a circular orbit of this inclination has the
    target beta at epoch ``et``.  Deterministic root bracketing on a 0.1 deg grid,
    refined by bisection.  Returns [] when the beta is unreachable."""
    c = mci_to_lme()
    s = sun_mci(et)
    s /= np.linalg.norm(s)

    def f(raan):
        n = c.T @ orbit_normal_lme(inc_deg, raan)
        return math.degrees(math.asin(max(-1.0, min(1.0, float(np.dot(n, s)))))) - beta_target_deg

    grid = np.linspace(0.0, 360.0, n_grid)
    vals = np.array([f(g) for g in grid])
    roots = []
    for k in range(len(grid) - 1):
        if vals[k] == 0.0:
            roots.append(float(grid[k]))
        elif vals[k] * vals[k + 1] < 0:
            lo, hi = grid[k], grid[k + 1]
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if f(lo) * f(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            roots.append(float(0.5 * (lo + hi)))
    return sorted(set(round(r, 9) for r in roots))


# ======================================================================
# Propagation (the canonical GEO force model, frozen = G0's)
# ======================================================================
def propagate48(t_grid: np.ndarray, x0: np.ndarray, et0: float, k_srp: float = K_TRUTH,
                rtol: float = RTOL, atol: float = ATOL,
                shadow_model: str = "CONICAL_PENUMBRA") -> np.ndarray:
    """Moon point mass + lunar J2 + cannonball SRP (conical penumbra).

    IDENTICAL to phase17_r1m_core.build_range_arc's truth: mu_earth = mu_sun = 0
    (no third-body gravity), which is the canonical baseline's force model and is
    frozen for GEO (s17).  The Sun position feeds SRP only.
    """
    from lunar_od.constants import J2_MOON_UNNORMALIZED
    from lunar_od.dynamics import propagate_state_with_k_sensitivity
    from lunar_od.srp import SRPOptions

    t_grid = np.asarray(t_grid, float)
    return propagate_state_with_k_sensitivity(
        t_grid - t_grid[0], np.asarray(x0, float), mu_moon(), 0.0, 0.0,
        lambda _t: np.array([384_400e3, 0.0, 0.0]),
        lambda t: sun_mci(et0 + t_grid[0] + float(t)),
        srp=SRPOptions(k_srp_m2_per_kg=float(k_srp), shadow_model=shadow_model),
        rtol=rtol, atol=atol, j2_moon=J2_MOON_UNNORMALIZED)


# ======================================================================
# Physical descriptors (geometry, NOT K metrics)
# ======================================================================
def srp_history(nom48: np.ndarray, t_grid: np.ndarray, et0: float) -> dict:
    """Production illumination, SRP kernel and its RTN split along a trajectory."""
    from lunar_od.srp import SRPOptions, _illumination_for, srp_acceleration_kernel

    opts = SRPOptions(k_srp_m2_per_kg=1.0)
    nu, g_norm, rtn = [], [], []
    for i, t in enumerate(t_grid):
        r = nom48[i, :3]
        v = nom48[i, 3:6]
        s = sun_mci(et0 + t)
        nu.append(_illumination_for(r, s, opts))
        g = srp_acceleration_kernel(r, s, opts)
        g_norm.append(float(np.linalg.norm(g)))
        rh = r / np.linalg.norm(r)
        nh = np.cross(r, v)
        nh /= np.linalg.norm(nh)
        th = np.cross(nh, rh)
        rtn.append([float(g @ rh), float(g @ th), float(g @ nh)])
    return dict(nu=np.asarray(nu), g_norm=np.asarray(g_norm), g_rtn=np.asarray(rtn))


def eclipse_stats(nu: np.ndarray, t_grid: np.ndarray, threshold: float = 0.5) -> dict:
    """Shadow history from the production illumination factor.

    An eclipse is a maximal run of samples with nu < threshold (0.5 = half the
    solar disk hidden; penumbra is continuous, so the threshold only defines
    event counting, never the force).  Fractions are time-weighted on the grid.
    """
    nu = np.asarray(nu, float)
    dt = float(np.median(np.diff(t_grid)))
    dark = nu < threshold
    entries = int(np.sum(~dark[:-1] & dark[1:]))
    exits = int(np.sum(dark[:-1] & ~dark[1:]))
    runs, cur = [], 0
    for d in dark:
        if d:
            cur += 1
        elif cur:
            runs.append(cur * dt)
            cur = 0
    if cur:
        runs.append(cur * dt)
    return dict(
        sunlit_fraction=float(np.mean(nu >= 0.999)),
        eclipse_fraction=float(np.mean(dark)),
        penumbra_fraction=float(np.mean((nu > 0.0) & (nu < 0.999))),
        mean_illumination=float(np.mean(nu)),
        eclipse_entries=entries, eclipse_exits=exits,
        n_eclipses=len(runs),
        mean_eclipse_duration_s=float(np.mean(runs)) if runs else 0.0,
        min_eclipse_duration_s=float(np.min(runs)) if runs else 0.0,
        max_eclipse_duration_s=float(np.max(runs)) if runs else 0.0,
    )


def altitude_stats(nom48: np.ndarray) -> dict:
    rn = np.linalg.norm(nom48[:, :3], axis=1) - r_moon()
    return dict(alt_mean_km=float(rn.mean() / 1e3), alt_min_km=float(rn.min() / 1e3),
                alt_max_km=float(rn.max() / 1e3))


# ======================================================================
# Epoch-consistent range arc
# ======================================================================
@dataclass
class GeoArc:
    label: str
    t_grid: np.ndarray
    nom48: np.ndarray
    obs: np.ndarray
    pass_geo: object
    h_x0: np.ndarray
    h_k: np.ndarray
    w: np.ndarray
    stations: tuple
    extra: dict = field(default_factory=dict)

    @property
    def n_obs(self) -> int:
        return int(self.h_k.size)


def _earth_fns(et0: float, earth_mode: str = "CONSISTENT"):
    """Earth state providers.

    CONSISTENT        SPICE at et0 + t over the whole arc (the GEO fixture).
    CAMPAIGN_TABLE    the frozen campaign module's table (reproduces the published
                      fixture's Earth exactly: 5.83 d earlier epoch, clamped after
                      4 orbits) -- fixture-attribution diagnostic only.
    OFFSET_ONLY       SPICE at the campaign ET0 + t, NOT clamped.
    CLAMP_ONLY        SPICE at et0 + t, clamped at the campaign table's last epoch.
    """
    if earth_mode == "CAMPAIGN_TABLE":
        import od_gravity_covariance_campaign as C  # noqa: N811
        return C.get_earth_pos, C.get_earth_vel
    if earth_mode in ("OFFSET_ONLY", "CLAMP_ONLY"):
        import od_gravity_covariance_campaign as C  # noqa: N811
        base = C.ET0 if earth_mode == "OFFSET_ONLY" else et0
        t_max = float("inf") if earth_mode == "OFFSET_ONLY" else float(C.T_OBS[-1])

        def pos_m(t):
            tt = np.minimum(np.atleast_1d(np.asarray(t, float)), t_max)
            return np.array([earth_state_mci(base + x)[:3] for x in tt])

        def vel_m(t):
            tt = np.minimum(np.atleast_1d(np.asarray(t, float)), t_max)
            return np.array([earth_state_mci(base + x)[3:] for x in tt])

        return pos_m, vel_m

    def pos(t):
        tt = np.atleast_1d(np.asarray(t, float))
        return np.array([earth_state_mci(et0 + x)[:3] for x in tt])

    def vel(t):
        tt = np.atleast_1d(np.asarray(t, float))
        return np.array([earth_state_mci(et0 + x)[3:] for x in tt])

    return pos, vel


def build_geo_range_arc(x0: np.ndarray, et0: float, duration_s: float, *,
                        label: str, cadence_s: float = 90.0,
                        station_names=("Goldstone DSN", "Madrid DSN", "Canberra DSN"),
                        k_srp: float = K_TRUTH, min_elevation_deg: float = 10.0,
                        nom48: np.ndarray | None = None,
                        shadow_model: str = "CONICAL_PENUMBRA",
                        earth_mode: str = "CONSISTENT") -> GeoArc:
    """Production two-way range arc with ONE consistent epoch (see module doc).

    Mirrors phase17_r1m_core.build_range_arc line for line -- same propagator,
    tolerances, light-time config, visibility, Jacobian and K column -- except
    that the Earth state comes from SPICE at the same epoch as the Sun and the
    Earth rotation, over the whole arc.
    """
    import spiceypy as spice

    import od_gravity_covariance_campaign as C  # noqa: N811
    from lunar_od.estimators import _two_way_range_k_srp_column, _two_way_range_weight_diagonal
    from lunar_od.two_way_range import (
        TwoWayRangeConfig, generate_two_way_range_measurements,
        two_way_range_nominal_and_initial_jacobian,
    )
    from lunar_od.visibility import (
        VisibilityConfig, analyze_visibility_gap_with_transforms,
        sample_j2000_to_itrf93_transforms,
    )

    t_grid = np.arange(0.0, duration_s + 0.5 * cadence_s, cadence_s)
    if nom48 is None:
        nom48 = propagate48(t_grid, x0, et0, k_srp, shadow_model=shadow_model)
    x_true = nom48[:, :6]
    earth_pos, earth_vel = _earth_fns(et0, earth_mode)
    xf = sample_j2000_to_itrf93_transforms(et0, t_grid)
    vis_cfg = VisibilityConfig(
        r_moon_mean_m=r_moon(), earth_rotation_rad_s=7.292115e-5,
        epoch_utc=spice.et2utc(et0, "ISOC", 3), min_elevation_deg=min_elevation_deg)
    by_name = {s.name: i for i, s in enumerate(C.STATIONS)}
    idx = [by_name[n] for n in station_names]
    stations = tuple(C.STATIONS[i] for i in idx)
    _, _, vis, _ = analyze_visibility_gap_with_transforms(
        t_grid, x_true, stations, earth_pos, xf, 0.0, vis_cfg)
    vis = np.asarray(vis, dtype=bool).reshape(t_grid.size, len(stations))
    tight = TwoWayRangeConfig(tolerance_s=1e-13, equation_tolerance_s=1e-14, max_iter=200)
    obs, pass_geo = generate_two_way_range_measurements(
        t_grid, x_true, stations, vis, earth_pos, earth_vel, et0,
        noise=False, rng=None, config=tight)
    if obs.shape[0] == 0:
        return GeoArc(label, t_grid, nom48, obs, pass_geo, np.zeros((0, 6)), np.zeros(0),
                      np.zeros(0), stations, dict(vis=vis))
    _, h_x0 = two_way_range_nominal_and_initial_jacobian(obs, pass_geo, nom48[:, :42])
    h_k = _two_way_range_k_srp_column(obs, nom48, h_x0)
    w = _two_way_range_weight_diagonal(obs, pass_geo)
    return GeoArc(label, t_grid, nom48, obs, pass_geo, np.asarray(h_x0, float),
                  np.asarray(h_k, float), np.asarray(w, float), stations, dict(vis=vis))


def predicted_range(arc: GeoArc, nom48_alt: np.ndarray) -> np.ndarray:
    """Production nominal range for a different trajectory history, same events."""
    from lunar_od.two_way_range import two_way_range_nominal_and_initial_jacobian
    z, _ = two_way_range_nominal_and_initial_jacobian(arc.obs, arc.pass_geo, nom48_alt[:, :42])
    return np.asarray(z, float).reshape(-1)


# ======================================================================
# Information metrics -- QR only, never the normal matrix (s25)
# ======================================================================
def matched_indices(n_obs: int, n_match: int) -> np.ndarray:
    """Predeclared deterministic time-stratified subset: evenly spaced ranks of
    the time-ordered observation rows (observation rows are already time ordered
    per station; the arc builder emits them in epoch order)."""
    if n_match >= n_obs:
        return np.arange(n_obs)
    return np.unique(np.round(np.linspace(0, n_obs - 1, n_match)).astype(int))


def information_metrics(h_x0: np.ndarray, h_k: np.ndarray, w: np.ndarray,
                        k_truth: float = K_TRUTH) -> dict:
    """Direction (f_perp, theta_K) and magnitude (I_K|x, sigma_K) of K information.

    f_perp: QR projector on the whitened state block (r1m_core.orthogonal_decomposition).
    sigma_K: R1COV square-root covariance (QR of the scaled whitened design).
    Diagnostics: singular values of the scaled whitened DESIGN matrix itself
    (its squares are the normal-matrix eigenvalues; the normal matrix is never
    formed), K share of the weakest right-singular vector, and K/state
    correlations from the qualified covariance.
    """
    h_x0 = np.asarray(h_x0, float)
    h_k = np.asarray(h_k, float)
    w = np.asarray(w, float)
    a_x, b_k = whitened(h_x0, h_k, w)
    dec = orthogonal_decomposition(a_x, b_k)
    scale = scale_matrix()
    h_full = np.hstack([h_x0, h_k[:, None]])
    a_scaled = np.sqrt(w)[:, None] * (h_full @ scale)
    _, sv, vt = np.linalg.svd(a_scaled, full_matrices=False)
    sr = square_root_covariance(h_full, w, None, scale)
    cov = np.asarray(sr.covariance, float)
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    corr_k = [float(cov[-1, j] / (sd[-1] * sd[j])) if sd[-1] > 0 and sd[j] > 0 else float("nan")
              for j in range(6)]
    f = dec["orthogonal_fraction"]
    return dict(
        n_obs=int(h_k.size),
        k_column_norm=dec["k_column_norm"],
        f_perp=f,
        theta_k_deg=float(math.degrees(math.asin(min(1.0, max(0.0, f))))),
        raw_k_information=dec["i_kk"],
        conditional_k_information=dec["i_k_given_x"],
        sigma_k=float(sr.sigma_k),
        sigma_k_frac=float(sr.sigma_k / k_truth),
        design_singular_values=[float(s) for s in sv],
        smallest_design_sv=float(sv[-1]),
        design_condition=float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
        weakest_mode_k_component=float(abs(vt[-1, -1])),
        k_state_correlations=corr_k,
        max_abs_k_state_correlation=float(np.nanmax(np.abs(corr_k))),
        state_column_norms=[float(np.linalg.norm(a_x[:, j])) for j in range(6)],
        rank_margin=float(sr.rank_margin),
    )
