"""PHASE 17-R1O-D2 - corrected-geometry production ΔDOR robustness: shared core.

ANALYSIS SPACE ONLY.  No production file is modified.  Every physical quantity
comes from production code: `lunar_od.delta_dor` (common-transmit event solver
and implicit-event Jacobian/K chain), `lunar_od.two_way_range` (range design),
`lunar_od.visibility` (elevation mask + lunar occultation), `lunar_od.dynamics`
(truth, STM, dx/dK) and the Phase 17-FXR canonical Earth fixture.

EARTH
-----
Every arc uses `phase17_r1m_core.measurement_earth_ephemeris(EARTH_FIXTURE_CANONICAL,
et0, t_max)`: exact SPICE Earth at the SAME ET origin the production range model
uses for the Earth rotation, coverage-checked.  ΔDOR stations are placed on that
same ephemeris at each measurement-grid epoch (Earth position AND velocity).
`install_legacy_earth_guard()` makes any call into the superseded campaign
Earth table raise, so the published fixture cannot leak into new science.

ΔDOR ROW RULE
-------------
R1O-D/FXR accepted a ΔDOR epoch when both stations saw the spacecraft above 10
deg ELEVATION; it never tested lunar occultation, so 93 of the 263 canonical
FXR rows were formed with the spacecraft behind the Moon for at least one
station.  D2's reference rule is PRODUCTION_VISIBILITY: both stations visible
by `analyze_visibility_gap_with_transforms` (the same elevation + occultation
test the range arc uses).  The historical rule is kept, named, for D0.

INFORMATION METRICS
-------------------
Direction (f_perp, theta_K) by a QR projector; magnitude (I(K|x), sigma_K) by
R1COV's square-root covariance; ALL rank / weakest-mode statements from the SVD
of the whitened, scaled DESIGN matrix -- never from rank(H^T W H), which R1COV
showed is squared-condition-number artefact (brief s3).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import phase17_r1m_core as M  # noqa: E402

C_LIGHT = 299792458.0
DSN3 = ("Goldstone DSN", "Madrid DSN", "Canberra DSN")
PAIRS = {
    "GC": ("Goldstone DSN", "Canberra DSN"),   # DSN standard (DSN-210 s2.1.4)
    "GM": ("Goldstone DSN", "Madrid DSN"),     # DSN standard
    "MC": ("Madrid DSN", "Canberra DSN"),      # physically possible, not a DSN ΔDOR baseline
}
ROW_RULE_ELEVATION_ONLY = "ELEVATION_ONLY_HISTORICAL_R1OD"
ROW_RULE_PRODUCTION_VISIBILITY = "PRODUCTION_VISIBILITY_ELEVATION_AND_OCCULTATION"
SEMANTICS_ANGULAR = "ANGULAR_SIGMA_PROJECTED_BY_BPERP"   # R1O-D/FXR convention
SEMANTICS_DELAY = "CONSTANT_DELAY_SIGMA"                 # DSN-210 error-budget units
SCALE = np.diag([1e6, 1e6, 1e6, 1e3, 1e3, 1e3, 1e-2])
NUISANCE_SCALE_S = 1e-10
K_INDEX = 6


class LegacyEarthLeakError(RuntimeError):
    """The superseded campaign Earth table was touched by D2 science code."""


def install_legacy_earth_guard() -> None:
    """Make the campaign module's Earth-table accessors raise (D2 s8)."""
    import od_gravity_covariance_campaign as C  # noqa: N811

    def _blocked(*_a, **_k):
        raise LegacyEarthLeakError("legacy campaign Earth table used in Phase 17-R1O-D2 science")

    C.get_earth_pos = _blocked
    C.get_earth_vel = _blocked


def stations_by_name() -> dict:
    import od_gravity_covariance_campaign as C  # noqa: N811
    return {s.name: s for s in C.STATIONS}


# ======================================================================
# Scenarios
# ======================================================================
@dataclass(frozen=True)
class Scenario:
    case_id: str
    family: str
    et0: float
    x0: tuple
    duration_s: float
    k_srp: float = M.K_TRUTH
    shadow_model: str = "CONICAL_PENUMBRA"
    cadence_s: float = 90.0
    note: str = ""

    @property
    def x0_array(self) -> np.ndarray:
        return np.asarray(self.x0, float)

    @property
    def t_grid(self) -> np.ndarray:
        return np.arange(0.0, self.duration_s + 0.5 * self.cadence_s, self.cadence_s)


def canonical_scenario() -> Scenario:
    et0, t_orbit = M.campaign_epoch()
    return Scenario("CANON", "canonical", et0, tuple(M.campaign_initial_state().tolist()),
                    15.0 * t_orbit, note="Phase 17-FXR canonical (= GEO G0_consistent)")


# ======================================================================
# Truth + range (canonical Earth fixture)
# ======================================================================
def propagate_nom48(scn: Scenario) -> np.ndarray:
    import phase17_geo_core as G
    return G.propagate48(scn.t_grid, scn.x0_array, scn.et0, scn.k_srp,
                         shadow_model=scn.shadow_model)


@dataclass
class RangeBlock:
    t_grid: np.ndarray
    nom48: np.ndarray
    earth: object
    xf: np.ndarray
    vis: np.ndarray          # (N, 3) production visibility, DSN3 order
    obs: np.ndarray
    pass_geo: object
    h_x0: np.ndarray
    h_k: np.ndarray
    w: np.ndarray

    @property
    def n_obs(self) -> int:
        return int(self.h_k.size)


def build_range_block(scn: Scenario, nom48: np.ndarray) -> RangeBlock:
    """Production two-way range, DSN3, on the canonical Earth fixture.

    Mirrors `phase17_geo_core.build_geo_range_arc` line for line; the only
    difference is that the Earth is the FXR epoch-tagged, coverage-checked
    `EarthEphemeris` rather than an ad-hoc SPICE closure.
    """
    import spiceypy as spice

    from lunar_od.estimators import _two_way_range_k_srp_column, _two_way_range_weight_diagonal
    from lunar_od.two_way_range import (TwoWayRangeConfig, generate_two_way_range_measurements,
                                        two_way_range_nominal_and_initial_jacobian)
    from lunar_od.visibility import (VisibilityConfig, analyze_visibility_gap_with_transforms,
                                     sample_j2000_to_itrf93_transforms)
    from lunar_od.constants import R_MOON_M

    t_grid = scn.t_grid
    x_true = nom48[:, :6]
    earth = M.measurement_earth_ephemeris(M.EARTH_FIXTURE_CANONICAL, scn.et0, float(t_grid[-1]))
    xf = sample_j2000_to_itrf93_transforms(scn.et0, t_grid)
    vis_cfg = VisibilityConfig(r_moon_mean_m=R_MOON_M, earth_rotation_rad_s=7.292115e-5,
                               epoch_utc=spice.et2utc(scn.et0, "ISOC", 3), min_elevation_deg=10.0)
    by = stations_by_name()
    stations = tuple(by[n] for n in DSN3)
    _, _, vis, _ = analyze_visibility_gap_with_transforms(
        t_grid, x_true, stations, earth.pos, xf, 0.0, vis_cfg)
    vis = np.asarray(vis, dtype=bool).reshape(t_grid.size, len(stations))
    tight = TwoWayRangeConfig(tolerance_s=1e-13, equation_tolerance_s=1e-14, max_iter=200)
    obs, pass_geo = generate_two_way_range_measurements(
        t_grid, x_true, stations, vis, earth.pos, earth.vel, scn.et0,
        noise=False, rng=None, config=tight)
    _, h_x0 = two_way_range_nominal_and_initial_jacobian(obs, pass_geo, nom48[:, :42])
    h_k = _two_way_range_k_srp_column(obs, nom48, h_x0)
    w = _two_way_range_weight_diagonal(obs, pass_geo)
    return RangeBlock(t_grid, nom48, earth, xf, vis, obs, pass_geo, np.asarray(h_x0, float),
                      np.asarray(h_k, float), np.asarray(w, float))


def earth_epoch_audit(scn: Scenario, rb: RangeBlock) -> dict:
    """s8/s19: Earth handed to the measurement model == SPICE at the measurement epoch."""
    pg = rb.pass_geo
    idx = np.unique(np.linspace(0, pg.t_s.size - 1, 25).round().astype(int))
    exp = np.array([M._spice_earth_state_mci(pg.et0_s + pg.t_s[i]) for i in idx])
    import spiceypy as spice
    sun_ok = all(np.array_equal(
        np.asarray(spice.spkezr("SUN", scn.et0 + float(rb.t_grid[i]), "J2000", "NONE", "MOON")[0][:3]) * 1e3,
        np.asarray(spice.spkezr("SUN", pg.et0_s + float(rb.t_grid[i]), "J2000", "NONE", "MOON")[0][:3]) * 1e3)
        for i in idx[:5])
    return dict(
        earth_fixture=rb.earth.fixture,
        earth_et_origin=rb.earth.et_origin, rotation_et_origin=float(pg.et0_s),
        scenario_et0=scn.et0, spacecraft_epoch_et0=scn.et0,
        sun_et_origin=scn.et0,
        max_earth_pos_diff_m=float(np.max(np.abs(pg.earth_pos_mci_m[idx] - exp[:, :3]))),
        max_earth_vel_diff_mps=float(np.max(np.abs(pg.earth_vel_mci_mps[idx] - exp[:, 3:]))),
        coverage=[rb.earth.t_min_s, rb.earth.t_max_s], grid_end=float(rb.t_grid[-1]),
        clamp_after=rb.earth.clamp_after_s,
        sun_same_scenario_epoch=bool(sun_ok),
        quasar_frame_contract="fixed J2000 unit vector (synthetic ICRF approximation; epoch-independent)",
        passed=bool(rb.earth.fixture == M.EARTH_FIXTURE_CANONICAL
                    and rb.earth.et_origin == pg.et0_s == scn.et0
                    and np.max(np.abs(pg.earth_pos_mci_m[idx] - exp[:, :3])) == 0.0
                    and np.max(np.abs(pg.earth_vel_mci_mps[idx] - exp[:, 3:])) == 0.0
                    and rb.earth.clamp_after_s == float("inf") and sun_ok),
    )


# ======================================================================
# Production ΔDOR rows
# ======================================================================
@dataclass
class DdorRows:
    pair: str
    row_rule: str
    t_used: np.ndarray        # (M,) grid epochs used (receive tag at station A)
    idx: np.ndarray           # (M,) grid indices
    h_x0: np.ndarray          # (M, 6)
    h_k: np.ndarray           # (M,)
    d_s: np.ndarray           # (M,) spacecraft differential delay [s]
    b_perp: np.ndarray        # (M,) baseline component perpendicular to station-A LOS [m]
    baseline: np.ndarray      # (M, 3) r_B - r_A at the grid epoch, J2000 [m]
    s_sc_geo: np.ndarray      # (M, 3) geocentric spacecraft unit vector, J2000
    session: np.ndarray       # (M,) session index (contiguous dual-visible run)
    n_candidates: int
    n_converged: int
    cond_max: float
    earth_pos_hist: np.ndarray = field(repr=False, default=None)
    hist_a: np.ndarray = field(repr=False, default=None)
    hist_b: np.ndarray = field(repr=False, default=None)

    @property
    def n(self) -> int:
        return int(self.h_k.size)

    @property
    def n_sessions(self) -> int:
        return int(self.session.max() + 1) if self.n else 0


def sessions_from_indices(idx: np.ndarray) -> np.ndarray:
    """Contiguous runs of grid indices (gap > 1 grid step starts a new session)."""
    if idx.size == 0:
        return np.zeros(0, int)
    return np.concatenate([[0], np.cumsum(np.diff(idx) > 1)]).astype(int)


def build_ddor_rows(scn: Scenario, rb: RangeBlock, pair: str, row_rule: str,
                    min_elevation_deg: float = 10.0) -> DdorRows:
    """Production ΔDOR design rows on the canonical Earth (Earth at measurement epoch).

    With ROW_RULE_ELEVATION_ONLY this reproduces
    `phase17_r1od_campaign.build_ddor_rows(..., earth_anchor=MEASUREMENT_EPOCH)`
    arithmetic exactly (checked by D0).
    """
    from lunar_od.delta_dor import (DeltaDorEventError, delta_dor_spacecraft_sensitivity_full,
                                    solve_common_transmit_event)
    from lunar_od.geometry import ecef2razel_sez
    from lunar_od.measurements import _station_relative_state_j2000_at_receive_epoch
    from lunar_od.radiometrics import _interp_state

    by = stations_by_name()
    na, nb = PAIRS[pair]
    sa, sb = by[na], by[nb]
    ia, ib = DSN3.index(na), DSN3.index(nb)
    t_grid, nom48, xf = rb.t_grid, rb.nom48, rb.xf
    ep = rb.earth.pos(t_grid)
    ev = rb.earth.vel(t_grid)

    def hist(station):
        rows = []
        for i in range(t_grid.size):
            rel = _station_relative_state_j2000_at_receive_epoch(station, xf[i])
            rows.append(np.concatenate([ep[i] + rel[:3], ev[i] + rel[3:]]))
        return np.asarray(rows, float)

    hist_a, hist_b = hist(sa), hist(sb)

    def elev(station, r_sc, e, xfi):
        r_rel = xfi[:3, :3] @ (r_sc - e) - np.asarray(station.r_ecef_m, float).reshape(3)
        _, el, _ = ecef2razel_sez(r_rel, station.lat_rad, station.lon_rad)
        return np.degrees(el)

    rows_x0, rows_k, ds, used, conds = [], [], [], [], []
    n_cand = 0
    for i, t_obs in enumerate(t_grid):
        r_sc = nom48[i, :3]
        if row_rule == ROW_RULE_ELEVATION_ONLY:
            if elev(sa, r_sc, ep[i], xf[i]) < min_elevation_deg \
                    or elev(sb, r_sc, ep[i], xf[i]) < min_elevation_deg:
                continue
        elif row_rule == ROW_RULE_PRODUCTION_VISIBILITY:
            if not (rb.vis[i, ia] and rb.vis[i, ib]):
                continue
        else:
            raise ValueError(row_rule)
        n_cand += 1
        a_tobs = _interp_state(t_grid, hist_a, t_obs)[:3]
        try:
            common = solve_common_transmit_event(
                float(t_obs), a_tobs, lambda t: _interp_state(t_grid, hist_b, t)[:3],
                lambda t: _interp_state(t_grid, nom48, t)[:3])
        except DeltaDorEventError:
            continue
        b_state = _interp_state(t_grid, hist_b, common.station_b_receive_time_s)
        try:
            sens = delta_dor_spacecraft_sensitivity_full(common, t_grid, nom48, a_tobs,
                                                         b_state[:3], b_state[3:6])
        except DeltaDorEventError:
            continue
        rows_x0.append(sens.d_spacecraft_delay_dx0)
        rows_k.append(sens.d_spacecraft_delay_dk)
        ds.append(common.spacecraft_differential_delay_s)
        used.append(i)
        conds.append(sens.event_matrix_condition_number)

    idx = np.asarray(used, int)
    # geometry per row, at the grid epoch (the historical weight convention)
    r_a = ep[idx] + np.array([_station_relative_state_j2000_at_receive_epoch(sa, xf[i])[:3] for i in idx]).reshape(-1, 3)
    r_b = ep[idx] + np.array([_station_relative_state_j2000_at_receive_epoch(sb, xf[i])[:3] for i in idx]).reshape(-1, 3)
    base = r_b - r_a
    # per-row scalar arithmetic, exactly as phase17_r1od_campaign.ddor_delay_weights
    b_perp = np.zeros(idx.size)
    for n, i in enumerate(idx):
        los = (r_a[n] - nom48[i, :3]) / np.linalg.norm(r_a[n] - nom48[i, :3])
        b_perp[n] = float(np.linalg.norm(base[n] - np.dot(base[n], los) * los))
    s_geo = nom48[idx, :3] - ep[idx]
    s_geo /= np.linalg.norm(s_geo, axis=1)[:, None]
    return DdorRows(pair, row_rule, t_grid[idx], idx, np.asarray(rows_x0, float).reshape(-1, 6),
                    np.asarray(rows_k, float), np.asarray(ds, float), b_perp, base, s_geo,
                    sessions_from_indices(idx), n_cand, len(idx),
                    float(max(conds)) if conds else float("nan"), ep, hist_a, hist_b)


def ddor_weights(rows: DdorRows, semantics: str, sigma: float) -> np.ndarray:
    """ANGULAR: sigma [rad] -> sigma_tau = max(B_perp,1)*sigma/c (R1O-D arithmetic).
    DELAY: sigma [s] used directly (DSN-210 units)."""
    if semantics == SEMANTICS_ANGULAR:
        sig = np.array([max(float(b), 1.0) * sigma / C_LIGHT for b in rows.b_perp])
        return 1.0 / sig ** 2
    if semantics == SEMANTICS_DELAY:
        return np.full(rows.n, 1.0 / sigma ** 2)
    raise ValueError(semantics)


def ops_selection(rows: DdorRows, mode: str, t_mid_split: float,
                  min_window_s: float = 1200.0, cadence_s: float = 90.0) -> np.ndarray:
    """Operational-session realizations (predeclared, DSN-210 s2.1.4 / CB13):
    three normal points per session at 1/6, 1/2, 5/6 of the dual-visible window.

    OPS_EVERY_WINDOW : one session in every window >= min_window_s
    OPS_TWO_SESSIONS : the longest qualifying window whose midpoint falls in each
                       half of the arc (split at t_mid_split)
    """
    wins = []
    for s in range(rows.n_sessions):
        k = np.where(rows.session == s)[0]
        dur = (rows.t_used[k[-1]] - rows.t_used[k[0]]) + cadence_s
        if dur >= min_window_s:
            wins.append((s, k, dur, 0.5 * (rows.t_used[k[0]] + rows.t_used[k[-1]])))
    if mode == "OPS_TWO_SESSIONS":
        pick = []
        for half in (0, 1):
            cand = [wv for wv in wins if (wv[3] < t_mid_split) == (half == 0)]
            if cand:
                pick.append(max(cand, key=lambda wv: wv[2]))
        wins = pick
    elif mode != "OPS_EVERY_WINDOW":
        raise ValueError(mode)
    sel = []
    for _, k, _, _ in wins:
        for frac in (1 / 6, 1 / 2, 5 / 6):
            sel.append(int(k[int(round(frac * (k.size - 1)))]))
    return np.array(sorted(set(sel)), int)


def subset(rows: DdorRows, sel: np.ndarray) -> DdorRows:
    """Row subset that KEEPS each row's parent-window session (renumbered 0..q-1),
    so a session's normal points share one session bias."""
    sess = np.unique(rows.session[sel], return_inverse=True)[1] if sel.size else np.zeros(0, int)
    return DdorRows(rows.pair, rows.row_rule + "|subset", rows.t_used[sel], rows.idx[sel],
                    rows.h_x0[sel], rows.h_k[sel], rows.d_s[sel], rows.b_perp[sel],
                    rows.baseline[sel], rows.s_sc_geo[sel], np.asarray(sess, int),
                    rows.n_candidates, int(sel.size), rows.cond_max)


# ======================================================================
# Information metrics (design-matrix based)
# ======================================================================
def _assemble(blocks):
    h_x0 = np.vstack([b[0] for b in blocks if len(b[1])])
    h_k = np.concatenate([b[1] for b in blocks if len(b[1])])
    w = np.concatenate([b[2] for b in blocks if len(b[1])])
    return h_x0, h_k, w


def info_metrics(blocks, nuisance: np.ndarray | None = None,
                 nuisance_prior_sigma: float | None = None) -> dict:
    """blocks: [(h_x0, h_k, w), ...]; nuisance: (N_total, q) extra columns (seconds/s).

    f_perp: K column vs span(state [+ nuisance]); sigma_K: R1COV QR covariance
    (nuisance columns appended AFTER K, scaled by 0.1 ns, optional prior);
    rank / weakest mode: SVD of the whitened scaled design of state+K(+nuisance).
    """
    from phase17_r1cov_core import RankDeficientCovariance, numerical_rank, square_root_covariance

    h_x0, h_k, w = _assemble(blocks)
    sw = np.sqrt(w)
    q = 0 if nuisance is None else nuisance.shape[1]
    h_full = np.hstack([h_x0, h_k[:, None]] + ([nuisance] if q else []))
    scale = np.diag(list(np.diag(SCALE)) + [NUISANCE_SCALE_S] * q)
    a = sw[:, None] * (h_full @ scale)
    a_other = sw[:, None] * np.hstack([h_x0] + ([nuisance] if q else []))
    b_k = sw * h_k
    if q and nuisance_prior_sigma is not None:
        # a nuisance prior is information: append its square-root pseudo-rows to
        # BOTH the design (SVD / rank) and the projector, so a partly-constrained
        # bias can only absorb part of the K signature (K has no prior row)
        prior_rows = np.hstack([np.zeros((q, 7)), np.eye(q) / nuisance_prior_sigma])
        a = np.vstack([a, prior_rows @ scale])
        a_other = np.vstack([a_other, np.hstack([np.zeros((q, 6)), np.eye(q) / nuisance_prior_sigma])])
        b_k = np.concatenate([b_k, np.zeros(q)])
    _, sv, vt = np.linalg.svd(a, full_matrices=False)
    rank, ratio, thr = numerical_rank(np.linalg.qr(a, mode="r"))
    qm, _ = np.linalg.qr(a_other)
    b_perp = b_k - qm @ (qm.T @ b_k)
    nk, npp = float(np.linalg.norm(b_k)), float(np.linalg.norm(b_perp))
    f = npp / nk if nk > 0 else float("nan")
    prior_inv = None
    if q and nuisance_prior_sigma is not None:
        prior_inv = np.zeros((7 + q, 7 + q))
        prior_inv[7:, 7:] = np.eye(q) / nuisance_prior_sigma ** 2
    try:
        sr = square_root_covariance(h_full, w, prior_inv, scale)
        cov = np.asarray(sr.covariance, float)
        sd = np.sqrt(np.clip(np.diag(cov), 0, None))
        corr = cov[K_INDEX] / (sd[K_INDEX] * sd)
        sigma_k = float(sr.sigma_k)
        corr_state = [float(c) for c in corr[:6]]
        corr_nuis = float(np.max(np.abs(corr[7:]))) if q else float("nan")
    except RankDeficientCovariance:
        sigma_k, corr_state, corr_nuis = float("inf"), [float("nan")] * 6, float("nan")
    wv = vt[-1]
    return dict(
        n_obs=int(h_k.size), n_nuisance=q,
        f_perp=f, theta_k_deg=math.degrees(math.asin(min(1.0, max(0.0, f)))),
        i_kk=nk ** 2, conditional_k_information=npp ** 2, k_column_norm=nk,
        sigma_k=sigma_k, sigma_k_frac=sigma_k / M.K_TRUTH,
        design_singular_values=[float(s) for s in sv],
        design_condition=float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
        design_rank=int(rank), design_rank_margin=float(ratio / thr) if thr > 0 else float("inf"),
        weakest_right_singular_vector=[float(x) for x in wv],
        weakest_mode_k_component=float(abs(wv[K_INDEX])),
        k_state_correlations=corr_state,
        max_abs_k_state_correlation=float(np.nanmax(np.abs(corr_state))),
        max_abs_k_nuisance_correlation=corr_nuis,
    )


def relative_gain_class(delta: float) -> str:
    """R1O-D thresholds (gain > 0.05 strong, > 0.02 modest), extended with a symmetric loss class."""
    if delta > 0.05:
        return "STRONG_DIRECTION_GAIN"
    if delta > 0.02:
        return "MODEST_DIRECTION_GAIN"
    if delta >= -0.02:
        return "MAGNITUDE_ONLY"
    return "DIRECTION_LOSS"


def historical_r1od_class(delta: float) -> str:
    return ("STRONG_DIRECTION_GAIN" if delta > 0.05 else
            "MODEST_DIRECTION_GAIN" if delta > 0.02 else "MAGNITUDE_ONLY")


def absolute_class(f: float) -> str:
    """GEO absolute gates (f_perp >= 0.50 strong, >= 0.35 moderate)."""
    if f >= 0.50:
        return "STRONG_K_SEPARATION"
    if f >= 0.35:
        return "MODERATE_K_SEPARATION"
    return "WEAK_K_SEPARATION"


def omitted_bias_k_shift(blocks, delta_full: np.ndarray) -> float:
    """K shift of the weighted LS solution when an unmodelled delay vector
    delta (length = total rows; zeros on range rows) is present in the data."""
    h_x0, h_k, w = _assemble(blocks)
    sw = np.sqrt(w)
    # scaled columns (the design is conditioned ~1e8 only after scaling; an
    # unscaled SVD cutoff could silently discard the weak K direction)
    a = sw[:, None] * (np.hstack([h_x0, h_k[:, None]]) @ SCALE)
    sol, *_ = np.linalg.lstsq(a, sw * delta_full, rcond=None)
    return float(sol[K_INDEX] * SCALE[K_INDEX, K_INDEX])


def session_nuisance_columns(n_range: int, rows: DdorRows) -> np.ndarray:
    q = rows.n_sessions
    out = np.zeros((n_range + rows.n, q))
    for s in range(q):
        out[n_range + np.where(rows.session == s)[0], s] = 1.0
    return out


# ======================================================================
# Quasar geometry and calibration systematics (CCSDS eq. 25/26/32)
# ======================================================================
def quasar_direction(s_sc: np.ndarray, sep_rad: float, pa_rad: float) -> np.ndarray:
    """Unit vector at angular separation `sep` from s_sc, position angle `pa`
    measured from celestial north (J2000 +z) through east."""
    s = s_sc / np.linalg.norm(s_sc)
    z = np.array([0.0, 0.0, 1.0])
    east = np.cross(z, s)
    east /= np.linalg.norm(east)
    north = np.cross(s, east)
    return math.cos(sep_rad) * s + math.sin(sep_rad) * (math.cos(pa_rad) * north
                                                        + math.sin(pa_rad) * east)


def session_quasars(rows: DdorRows, sep_rad: float, pa_rad: float) -> np.ndarray:
    """One synthetic quasar per session, placed relative to the geocentric
    spacecraft direction at the session's middle row (GEOMETRY_ONLY_SYNTHETIC_QUASAR)."""
    q = np.zeros((rows.n_sessions, 3))
    for s in range(rows.n_sessions):
        k = np.where(rows.session == s)[0]
        q[s] = quasar_direction(rows.s_sc_geo[k[len(k) // 2]], sep_rad, pa_rad)
    return q


def systematic_delay_vectors(rows: DdorRows, xf_rows: np.ndarray, quasars: np.ndarray,
                             eps_bl_m: float = 0.02, eps_ut1_m: float = 0.02,
                             eps_pm_m: float = 0.01, eps_q_rad: float = 0.75e-9,
                             r_earth_m: float = 6378137.0) -> dict:
    """Per-row ΔDOR delay errors for unit-sigma systematic components.

    delta_tau = -(dB . (s_sc - s_q))/c   baseline error (station location, UT1, PM)
    delta_tau = -(B . ds_q)/c            quasar coordinate error (per session)
    Station-location errors are ITRF vectors (rotated to J2000 per row);
    UT1 / polar motion are small rotations of the whole baseline.
    Returns {group: [vectors]} each vector a 1-sigma realization of an
    independent component.
    """
    rot = np.array([x[:3, :3] for x in xf_rows])        # J2000 -> ITRF93
    d = rows.s_sc_geo - quasars[rows.session]
    out = {"station_location": [], "earth_orientation_ut1": [], "polar_motion": [],
           "quasar_coordinate": []}
    for j in range(3):
        e = np.zeros(3)
        e[j] = eps_bl_m
        db = np.einsum("nji,j->ni", rot, e)            # R^T e
        out["station_location"].append(-np.sum(db * d, axis=1) / C_LIGHT)
    for axis, key, eps in ((2, "earth_orientation_ut1", eps_ut1_m),
                           (0, "polar_motion", eps_pm_m), (1, "polar_motion", eps_pm_m)):
        e = np.zeros(3)
        e[axis] = 1.0
        ax = np.einsum("nji,j->ni", rot, e)
        db = (eps / r_earth_m) * np.cross(ax, rows.baseline)
        out[key].append(-np.sum(db * d, axis=1) / C_LIGHT)
    for s in range(rows.n_sessions):
        qs = quasars[s]
        t1 = np.cross([0.0, 0.0, 1.0], qs)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(qs, t1)
        m = rows.session == s
        for t in (t1, t2):
            v = np.zeros(rows.n)
            v[m] = -(rows.baseline[m] @ (eps_q_rad * t)) / C_LIGHT
            out["quasar_coordinate"].append(v)
    return out


# ======================================================================
# Independent surrogate (R1O arithmetic, general Earth)
# ======================================================================
def surrogate_rows(rb: RangeBlock, rows: DdorRows, sigma_angle_rad: float):
    """R1O `build_ddor_arc` arithmetic on the SAME epochs as the production rows:
    g = (|r - r_B| - |r - r_A|)/c at the grid epoch, FD position Jacobian chained
    through the column-major STM; weight from B_perp to the MEAN line of sight."""
    import od_gravity_covariance_campaign as C  # noqa: N811
    from phase17_r1o_core import (chain_to_augmented_columns, position_jacobian_fd,
                                  station_position_mci)

    by = {s.name: s for s in C.STATIONS}
    na, nb = PAIRS[rows.pair]
    sa, sb = by[na], by[nb]
    hx, hk, bp = [], [], []
    for i in rows.idx:
        r_sc = rb.nom48[i, :3]
        e = rows.earth_pos_hist[i]
        r_a = station_position_mci(sa, e, rb.xf[i])
        r_b = station_position_mci(sb, e, rb.xf[i])

        def g_fn(r, _ra=r_a, _rb=r_b):
            return np.array([(np.linalg.norm(r - _rb) - np.linalg.norm(r - _ra)) / C_LIGHT])

        dg_dr, _ = position_jacobian_fd(g_fn, r_sc)
        h_x0, h_k = chain_to_augmented_columns(dg_dr, rb.nom48[i])
        los = 0.5 * ((r_sc - r_a) / np.linalg.norm(r_sc - r_a)
                     + (r_sc - r_b) / np.linalg.norm(r_sc - r_b))
        base = r_b - r_a
        bp.append(float(np.linalg.norm(base - np.dot(base, los) * los)))
        hx.append(h_x0[0])
        hk.append(h_k[0])
    bp = np.asarray(bp)
    w = 1.0 / (np.maximum(bp, 1.0) * sigma_angle_rad / C_LIGHT) ** 2
    return np.asarray(hx, float).reshape(-1, 6), np.asarray(hk, float), w


# ======================================================================
# Geometry descriptors (not K metrics)
# ======================================================================
def geometry_descriptors(scn: Scenario, rb: RangeBlock) -> dict:
    import phase17_geo_core as G

    i_mid = rb.t_grid.size // 2
    et_mid = scn.et0 + float(rb.t_grid[i_mid])
    sun = G.sun_mci(et_mid)
    earth = rb.earth.pos([rb.t_grid[i_mid]])[0]
    el = G.elements_from_state(scn.x0_array)
    ang = lambda u, v: math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))))))  # noqa: E731
    return dict(
        sun_moon_earth_deg=ang(sun, earth),
        earth_los_vs_srp_direction_deg=ang(earth, -sun),
        beta_deg=G.beta_deg(np.asarray(el["h_hat_mci"]), et_mid),
        earth_view_deg=G.earth_view_deg(np.asarray(el["h_hat_mci"]), et_mid),
        moon_geocentric_declination_deg=math.degrees(math.asin(-earth[2] / np.linalg.norm(earth))),
        inc_lme_deg=el["inc_deg"], alt_km=(el["a_m"] - G.r_moon()) / 1e3,
        period_s=el["period_s"], et_mid=et_mid,
        range_obs=int(rb.n_obs),
    )


def pair_geometry(rows: DdorRows, rb: RangeBlock) -> dict:
    """Baseline length/direction and plane-of-sky projection vs the K signature."""
    if rows.n == 0:
        return dict(n_rows=0)
    base = rows.baseline
    s = rows.s_sc_geo
    bp = base - np.sum(base * s, axis=1)[:, None] * s          # plane-of-sky baseline
    sk = rb.nom48[rows.idx, 42:45]                              # dr/dK at the row epoch
    skp = sk - np.sum(sk * s, axis=1)[:, None] * s               # plane-of-sky K displacement
    cosang = np.sum(bp * skp, axis=1) / (np.linalg.norm(bp, axis=1) * np.linalg.norm(skp, axis=1))
    return dict(
        n_rows=rows.n, n_sessions=rows.n_sessions,
        dual_visible_duration_h=float(rows.n * 90.0 / 3600.0),
        baseline_length_km=float(np.mean(np.linalg.norm(base, axis=1)) / 1e3),
        projected_baseline_km_mean=float(np.mean(np.linalg.norm(bp, axis=1)) / 1e3),
        b_perp_station_a_los_km_mean=float(np.mean(rows.b_perp) / 1e3),
        k_displacement_plane_of_sky_fraction=float(np.mean(np.linalg.norm(skp, axis=1) / np.linalg.norm(sk, axis=1))),
        abs_cos_baseline_vs_k_plane_of_sky_mean=float(np.mean(np.abs(cosang))),
        baseline_vs_k_plane_of_sky_angle_deg_median=float(np.degrees(np.median(np.arccos(np.clip(np.abs(cosang), 0, 1))))),
    )
