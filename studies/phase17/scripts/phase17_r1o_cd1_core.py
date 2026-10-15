"""PHASE 17-R1O-CD1 - production counted-Doppler K_SRP complementarity: core.

ANALYSIS SPACE ONLY.  The observable is the production two-way counted Doppler
(`lunar_od.radiometrics.two_way_counted_doppler_*`, R3 exact-station model; it is
the four-event model at zero transponder delay, R4-P08), evaluated with the
production event solver, exact event-epoch station states, linear Earth
interpolation (R3-P22) and the FA-03B history-domain guards.  Nothing here edits
production code.

Rows
----
One counted-Doppler row per (count midpoint on the 90 s grid, DSN3 station) - the
production candidate rule of `measurements.generate_range_rate_measurements` -
restricted to count intervals that are FULLY inside one visibility window: every
grid node from floor(t_start) to ceil(t_end) is visible (elevation + lunar
occultation, the production `analyze_visibility_gap_with_transforms` mask).
Candidates whose events leave the history support raise HistoryDomainError and are
dropped, as in production.

K column
--------
The production Jacobian takes the 6x6 STM from columns 6:42.  Replacing the first
STM column by dx/dK (nom48[:, 42:48]) and zeroing the rest makes column 0 of the
same production Jacobian equal to dy/dK (the HARDEN / F1 qualified construction).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parents[2] / "studies" / "phase17" / "results"
PRE_PATH = ARTIFACTS.parent / "configs" / "phase17_r1o_cd1_predeclaration.json"

GRID_CADENCE_S = 90.0
TC_PRIMARY_S = 60.0
# count interval -> midpoint cadence (s); cadence >= Tc so counts never overlap
TC_CADENCE_S = {10.0: 90.0, 60.0: 90.0, 540.0: 540.0}
SIGMA_D60_PRIMARY_MPS = 1.0e-4          # repository DSN station sigma_range_rate_mps
SIGMA_D60_BRACKET_MPS = (1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3)
TC_NOISE_SCALINGS = {"SCINT_T^-0.175": 0.175, "WHITE_PHASE_T^-1": 1.0}
# 202E Eq (30)/(31), X-up/X-down, and the repository H-maser 60 s floor (high)
C_BAND_XX = 2.7e-6
FTS_FLOOR_60S_HIGH_MPS = 2.120e-6
NUISANCE_MODELS = ("D0_IDEAL", "D1_SESSION_CONSTANT_BIAS", "D2_SESSION_BIAS_PLUS_LINEAR_DRIFT")
NUISANCE_PRIORS_MPS = (None, 1.0e-5, 1.0e-4, 1.0e-3)       # None = FREE
DOPPLER_NUIS_UNIT_MPS = 1.0e-4


class PredeclarationMismatch(SystemExit):
    """The CD1 predeclaration on disk is not the committed one."""


def canonical_sha256(d: dict) -> str:
    body = {k: v for k, v in d.items() if k != "canonical_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_predeclaration(pinned: str, path: Path = PRE_PATH) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    got = canonical_sha256(d)
    if not (got == d.get("canonical_sha256") == pinned):
        raise PredeclarationMismatch("CD1 predeclaration hash mismatch: file %s stored %s pinned %s"
                                     % (got, d.get("canonical_sha256"), pinned))
    return d


# ======================================================================
# Doppler rows
# ======================================================================
@dataclass
class DopplerRows:
    tc_s: float
    cadence_s: float
    t_mid: np.ndarray            # (N,) count midpoint, ground receive time [s, arc-local]
    node: np.ndarray             # (N,) grid index of the midpoint
    station: np.ndarray          # (N,) index into DSN3
    session: np.ndarray          # (N,) pass id (station, contiguous visibility window), global numbering
    h_x0: np.ndarray             # (N, 6) dy/dx0 [m/s per m, m/s per m/s]
    h_k: np.ndarray              # (N,) dy/dK [m/s per (m^2/kg)]
    y: np.ndarray                # (N,) observable, m/s equivalent
    sep_deg: np.ndarray          # (N,) Sun-Earth-spacecraft angle at the midpoint
    n_candidates: int = 0
    n_dropped_visibility: int = 0
    n_dropped_domain: int = 0
    session_station: list = field(default_factory=list)
    session_window: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.h_k.size)

    @property
    def n_sessions(self) -> int:
        return len(self.session_station)

    def subset(self, mask: np.ndarray) -> "DopplerRows":
        """Rows under a boolean mask; sessions renumbered in first-appearance order."""
        mask = np.asarray(mask, bool)
        old = self.session[mask]
        keep = list(dict.fromkeys(old.tolist()))
        remap = {o: i for i, o in enumerate(keep)}
        return DopplerRows(self.tc_s, self.cadence_s, self.t_mid[mask], self.node[mask], self.station[mask],
                           np.array([remap[o] for o in old], int), self.h_x0[mask], self.h_k[mask],
                           self.y[mask], self.sep_deg[mask], self.n_candidates, self.n_dropped_visibility,
                           self.n_dropped_domain, [self.session_station[o] for o in keep],
                           [self.session_window[o] for o in keep])


def visibility_windows(vis_col: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of visible grid nodes: (first, last) inclusive."""
    out, start = [], None
    for i, v in enumerate(list(vis_col) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    return out


def count_interval_nodes(k: int, tc_s: float, t_grid: np.ndarray) -> tuple[int, int]:
    """Grid nodes bracketing the count interval [t_k - Tc/2, t_k + Tc/2]."""
    t = float(t_grid[k])
    lo = int(math.floor((t - 0.5 * tc_s) / GRID_CADENCE_S + 1e-9))
    hi = int(math.ceil((t + 0.5 * tc_s) / GRID_CADENCE_S - 1e-9))
    return lo, hi


def doppler_config(tc_s: float):
    from lunar_od.radiometrics import RangeRatePhysicsConfig
    return RangeRatePhysicsConfig(mode="two_way_counted_doppler", count_interval_s=float(tc_s))


def k_history(nom48: np.ndarray) -> np.ndarray:
    """42-column history whose first STM column is dx/dK (HARDEN/F1 construction)."""
    fake = np.array(nom48[:, :42], dtype=float, copy=True)
    fake[:, 6:42] = 0.0
    fake[:, 6:12] = nom48[:, 42:48]
    return fake


def sep_angle_deg(et: float, s_hat_geo: np.ndarray, earth_pos_mci: np.ndarray) -> float:
    import phase17_geo_core as G
    sun_geo = G.sun_mci(et) - earth_pos_mci
    c = float(np.dot(sun_geo / np.linalg.norm(sun_geo), s_hat_geo))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def build_doppler_rows(scn, rb, tc_s: float, stations, with_values: bool = True) -> DopplerRows:
    """All-visible production counted Doppler at the declared (Tc, cadence)."""
    from lunar_od.history_domain import HistoryDomainError
    from lunar_od.radiometrics import (resolve_counted_doppler_station_state_provider,
                                       two_way_counted_doppler_initial_state_jacobian,
                                       two_way_counted_doppler_observable)

    pg = rb.pass_geo
    t_grid = rb.t_grid
    cfg = doppler_config(tc_s)
    step = int(round(TC_CADENCE_S[tc_s] / GRID_CADENCE_S))
    hk_hist = k_history(rb.nom48)
    x42 = rb.nom48[:, :42]
    x6 = rb.nom48[:, :6]
    rows = dict(t=[], node=[], st=[], ses=[], hx=[], hk=[], y=[], sep=[])
    sess_station, sess_window = [], []
    n_cand = n_vis = n_dom = 0
    for s_idx, st in enumerate(stations):
        prov = resolve_counted_doppler_station_state_provider(cfg, st, t_grid, pg.earth_pos_mci_m,
                                                              pg.earth_vel_mci_mps, et0_s=pg.et0_s)
        for (a, b) in visibility_windows(rb.vis[:, s_idx]):
            sid = len(sess_station)
            sess_station.append(s_idx)
            sess_window.append((float(t_grid[a]), float(t_grid[b])))
            for k in range(a, b + 1):
                if k % step:
                    continue
                n_cand += 1
                lo, hi = count_interval_nodes(k, tc_s, t_grid)
                if lo < a or hi > b:
                    n_vis += 1
                    continue
                t = float(t_grid[k])
                try:
                    hx = two_way_counted_doppler_initial_state_jacobian(
                        t, st, t_grid, x42, pg.earth_pos_mci_m, pg.earth_vel_mci_mps, pg.x_j2000_to_itrf93,
                        cfg, et0_s=pg.et0_s, station_state_provider=prov)
                    hk = two_way_counted_doppler_initial_state_jacobian(
                        t, st, t_grid, hk_hist, pg.earth_pos_mci_m, pg.earth_vel_mci_mps, pg.x_j2000_to_itrf93,
                        cfg, et0_s=pg.et0_s, station_state_provider=prov)[0]
                    y = (two_way_counted_doppler_observable(
                        t, st, t_grid, x6, pg.earth_pos_mci_m, pg.earth_vel_mci_mps, pg.x_j2000_to_itrf93,
                        cfg, et0_s=pg.et0_s, station_state_provider=prov) if with_values else float("nan"))
                except HistoryDomainError:
                    n_dom += 1
                    continue
                r_geo = rb.nom48[k, :3] - pg.earth_pos_mci_m[k]
                rows["t"].append(t); rows["node"].append(k); rows["st"].append(s_idx); rows["ses"].append(sid)
                rows["hx"].append(np.asarray(hx, float)); rows["hk"].append(float(hk)); rows["y"].append(y)
                rows["sep"].append(sep_angle_deg(pg.et0_s + t, r_geo / np.linalg.norm(r_geo), pg.earth_pos_mci_m[k]))
    order = np.argsort(np.asarray(rows["t"]), kind="stable")
    out = DopplerRows(tc_s, TC_CADENCE_S[tc_s], np.asarray(rows["t"])[order], np.asarray(rows["node"], int)[order],
                      np.asarray(rows["st"], int)[order], np.asarray(rows["ses"], int)[order],
                      np.asarray(rows["hx"]).reshape(-1, 6)[order], np.asarray(rows["hk"])[order],
                      np.asarray(rows["y"])[order], np.asarray(rows["sep"])[order], n_cand, n_vis, n_dom,
                      sess_station, sess_window)
    # drop sessions with no surviving row, renumber in chronological order of first row
    return out.subset(np.ones(out.n, bool))


def regular_session_mask(rows: DopplerRows, session_s: float = 3600.0) -> np.ndarray:
    """SECONDARY schedule: per pass lasting >= session_s, the centred session_s of rows."""
    mask = np.zeros(rows.n, bool)
    for j, (w0, w1) in enumerate(rows.session_window):
        if w1 - w0 < session_s:
            continue
        mid = 0.5 * (w0 + w1)
        sel = (rows.session == j) & (rows.t_mid >= mid - 0.5 * session_s) & (rows.t_mid <= mid + 0.5 * session_s)
        mask |= sel
    return mask


# ======================================================================
# Noise
# ======================================================================
def sigma_tc(sigma60: float, tc_s: float, exponent: float) -> float:
    """Declared count-interval scaling of a 60 s sigma: sigma(T) = sigma60 (60/T)^exponent."""
    return float(sigma60 * (TC_PRIMARY_S / tc_s) ** exponent)


def scintillation_sigma_mps(sep_deg: np.ndarray, tc_s: float) -> np.ndarray:
    """DSN 810-005 202E Eq (30), two-way X-up/X-down (C_band 2.7e-6, Eq 31), sigma_V in m/s."""
    from lunar_od.radiometrics import C_LIGHT_MPS, DEFAULT_X_BAND_TURNAROUND_RATIO, DEFAULT_X_BAND_UPLINK_HZ
    f_c = DEFAULT_X_BAND_UPLINK_HZ * DEFAULT_X_BAND_TURNAROUND_RATIO
    base = 0.53 * C_BAND_XX * C_LIGHT_MPS ** 2 / (f_c ** 2 * tc_s ** 0.35)
    sep = np.radians(np.asarray(sep_deg, float))
    fac = np.where(np.asarray(sep_deg) <= 90.0, np.sin(sep) ** 2.45, 1.0)
    return np.sqrt(base / fac)


def operational_lower_bound_sigma(sep_deg: np.ndarray) -> np.ndarray:
    """Link-independent 60 s terms only: 202E scintillation RSS the H-maser high floor.
    Thermal noise is PARAMETRIC (no spacecraft RF hardware) and is NOT included, so
    this is a LOWER bound on the white noise (optimistic weighting)."""
    return np.hypot(scintillation_sigma_mps(sep_deg, TC_PRIMARY_S), FTS_FLOOR_60S_HIGH_MPS)


# ======================================================================
# Nuisance columns (Doppler rows only; range rows carry zeros)
# ======================================================================
def nuisance_columns(n_range: int, rows: DopplerRows, model: str) -> np.ndarray | None:
    """Per-pass constant bias (D1) or bias + normalised linear drift (D2), in m/s.

    Drift regressor: (t - t_mid_pass) / half_span_pass over the pass's rows, in [-1, 1],
    so a drift parameter is the drift's range-rate contribution at the pass edge.
    """
    if model == "D0_IDEAL":
        return None
    if model not in ("D1_SESSION_CONSTANT_BIAS", "D2_SESSION_BIAS_PLUS_LINEAR_DRIFT"):
        raise ValueError(model)
    q1 = rows.n_sessions
    bias = np.zeros((n_range + rows.n, q1))
    drift = []
    for j in range(q1):
        sel = np.where(rows.session == j)[0]
        bias[n_range + sel, j] = 1.0
        if model == "D2_SESSION_BIAS_PLUS_LINEAR_DRIFT":
            t = rows.t_mid[sel]
            mid, half = 0.5 * (t.min() + t.max()), 0.5 * (t.max() - t.min())
            # IMPLEMENTATION_BUG_FIX CD1-IBF-01: a drift is undefined on a single-row
            # pass (half == 0).  The former all-zero column made the wrapper's QR
            # projector remove a spurious direction; such passes carry no drift column.
            if half > 0:
                col = np.zeros(n_range + rows.n)
                col[n_range + sel] = (t - mid) / half
                drift.append(col)
    return np.column_stack([bias] + drift) if drift else bias


def metrics(D2, rb, rows: DopplerRows, w_d: np.ndarray, model: str, prior_mps, *, with_range: bool = True) -> dict:
    """D2.info_metrics on [range; Doppler] (+ nuisance).  Nuisance parameters are
    expressed in units of DOPPLER_NUIS_UNIT_MPS so the fixed D2 nuisance design
    scale (1e-10) gives O(1) scaled columns; f_perp is invariant to that choice."""
    blocks = ([(rb.h_x0, rb.h_k, rb.w)] if with_range else []) + [(rows.h_x0, rows.h_k, w_d)]
    n_range = rb.n_obs if with_range else 0
    nu = nuisance_columns(n_range, rows, model)
    kappa = DOPPLER_NUIS_UNIT_MPS / D2.NUISANCE_SCALE_S
    if nu is None:
        return D2.info_metrics(blocks)
    prior = None if prior_mps is None else prior_mps / kappa
    m = D2.info_metrics(blocks, nuisance=nu * kappa, nuisance_prior_sigma=prior)
    m.update(nuisance_correlations(D2, blocks, nu * kappa, prior, model, rows.n_sessions))
    return m


def nuisance_correlations(D2, blocks, nu, prior, model, q1) -> dict:
    from phase17_r1cov_core import RankDeficientCovariance, square_root_covariance
    h_x0, h_k, w = D2._assemble(blocks)
    q = nu.shape[1]
    h_full = np.hstack([h_x0, h_k[:, None], nu])
    scale = np.diag(list(np.diag(D2.SCALE)) + [D2.NUISANCE_SCALE_S] * q)
    prior_inv = None
    if prior is not None:
        prior_inv = np.zeros((7 + q, 7 + q))
        prior_inv[7:, 7:] = np.eye(q) / prior ** 2
    try:
        cov = np.asarray(square_root_covariance(h_full, w, prior_inv, scale).covariance, float)
    except RankDeficientCovariance:
        return dict(max_abs_k_bias_corr=float("nan"), max_abs_k_drift_corr=float("nan"))
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    corr = np.abs(cov[D2.K_INDEX, 7:] / (sd[D2.K_INDEX] * sd[7:]))
    out = dict(max_abs_k_bias_corr=float(np.max(corr[:q1])),
               bias_sigma_mps_median=float(np.median(sd[7:7 + q1])) * DOPPLER_NUIS_UNIT_MPS)
    out["max_abs_k_drift_corr"] = float(np.max(corr[q1:])) if q > q1 else float("nan")
    out["n_drift_columns"] = int(q - q1) if model == "D2_SESSION_BIAS_PLUS_LINEAR_DRIFT" else 0
    return out


def independent_f_perp(D2, rb, rows: DopplerRows, w_d: np.ndarray, model: str, prior_mps,
                       with_range: bool = True) -> float:
    """Independent route: whitened least squares (numpy lstsq) of the K column on
    [state | nuisance] (+ prior pseudo-rows); no call into D2.info_metrics."""
    hx = [rb.h_x0] if with_range else []
    hk = [rb.h_k] if with_range else []
    ww = [rb.w] if with_range else []
    h_x = np.vstack(hx + [rows.h_x0])
    h_k = np.concatenate(hk + [rows.h_k])
    w = np.concatenate(ww + [w_d])
    n_range = rb.n_obs if with_range else 0
    nu = nuisance_columns(n_range, rows, model)
    cols = [h_x * np.array([1e6, 1e6, 1e6, 1e3, 1e3, 1e3])]
    if nu is not None:
        cols.append(nu * DOPPLER_NUIS_UNIT_MPS)
    a = np.sqrt(w)[:, None] * np.hstack(cols)
    b = np.sqrt(w) * h_k
    if nu is not None and prior_mps is not None:
        q = nu.shape[1]
        a = np.vstack([a, np.hstack([np.zeros((q, 6)), np.eye(q) * DOPPLER_NUIS_UNIT_MPS / prior_mps])])
        b = np.concatenate([b, np.zeros(q)])
    coef, *_ = np.linalg.lstsq(a, b, rcond=None)
    return float(np.linalg.norm(b - a @ coef) / np.linalg.norm(b))
