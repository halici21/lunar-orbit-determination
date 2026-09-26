"""PHASE 17-R1O-D3 - predeclared DeltaDOR tracking-design characterization: core.

ANALYSIS SPACE ONLY.  No production file is modified.  Every physical quantity
comes from the qualified production chain through phase17_r1od2_core:
F1 variational equations (propagate_nom48), checked canonical Earth and the
elevation + occultation visibility contract (build_range_block), production
common-transmit DeltaDOR rows (build_ddor_rows, PRODUCTION_VISIBILITY) and the
design-matrix information metrics (info_metrics).

What is new here is BOOKKEEPING, and it is kept auditable:
  * sessions  : deterministic, from contiguous dual-visibility windows only;
  * schedules : chronological patterns, never chosen with K information;
  * stacking  : every scheduled row carries (epoch, baseline, session, normal point,
                parent-row index); one bias column per (baseline, session), owning
                exactly that session's rows;
  * resources : Goldstone complex-level no-overlap between GC and GM sessions.

The predeclared grid (artifacts/phase17_r1od3_predeclared_grid.json) is
verified against its pinned canonical hash before anything runs.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ARTIFACTS = HERE.parent / "artifacts"
GRID_PATH = ARTIFACTS / "phase17_r1od3_predeclared_grid.json"
GRID_SHA256 = "a651c4ded63d0fac7a375ee7b6c2bc20c84eabefbb6958e9846ab323c8720003"

import phase17_r1od2_core as D2  # noqa: E402

CADENCE_S = 90.0
FRACTIONS = (1.0 / 6.0, 0.5, 5.0 / 6.0)
PATTERNS_SINGLE = ("FIRST_SESSION", "MID_SESSION", "LAST_SESSION", "FIRST_PLUS_LAST", "ALL_FEASIBLE_SESSIONS")
PATTERNS_JOINT = {"FIRST_GC_PLUS_FIRST_GM": "FIRST_SESSION",
                  "FIRST_LAST_GC_PLUS_FIRST_LAST_GM": "FIRST_PLUS_LAST",
                  "ALL_GC_PLUS_ALL_GM": "ALL_FEASIBLE_SESSIONS"}
PRIMARY_PATTERN = {"GC_ONLY": "ALL_FEASIBLE_SESSIONS", "GM_ONLY": "ALL_FEASIBLE_SESSIONS",
                   "GC_PLUS_GM": "ALL_GC_PLUS_ALL_GM"}
CALIBRATIONS = ("C0_IDEAL", "C1_x0.5", "C1_x1.0", "C1_x2.0", "C2_FREE")
C1_NOMINAL = "C1_x1.0"


class GridHashMismatch(SystemExit):
    """The predeclared grid on disk is not the committed one."""


def canonical_sha256(grid: dict) -> str:
    body = {k: v for k, v in grid.items() if k != "canonical_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_grid(path: Path = GRID_PATH) -> dict:
    grid = json.loads(Path(path).read_text(encoding="utf-8"))
    got = canonical_sha256(grid)
    if not (got == grid.get("canonical_sha256") == GRID_SHA256):
        raise GridHashMismatch("D3 predeclared grid hash mismatch: file %s, stored %s, pinned %s"
                               % (got, grid.get("canonical_sha256"), GRID_SHA256))
    return grid


def epoch_et(k: int, grid: dict) -> float:
    g = grid["stage_A_epoch_grid"]
    if not 0 <= k < g["count"]:
        raise ValueError("epoch index %d outside 0..%d" % (k, g["count"] - 1))
    return float(g["canonical_et0"]) + k * float(g["synodic_month_s"]) / g["count"]


# ======================================================================
# Sessions
# ======================================================================
@dataclass(frozen=True)
class Session:
    pair: str                 # "GC" or "GM"
    window_id: int            # contiguous dual-visibility run index within the pair's rows
    window_start: float       # s (arc-local)
    window_end: float
    start: float              # session interval (<= 1 h, inside the window)
    end: float
    row_index: tuple          # indices into the pair's DdorRows of the session's rows
    np_index: tuple           # indices into the pair's DdorRows of the 3 normal points

    @property
    def hours(self) -> float:
        return (self.end - self.start) / 3600.0

    @property
    def mid(self) -> float:
        return 0.5 * (self.window_start + self.window_end)


def build_sessions(rows, grid: dict) -> list[Session]:
    """Qualifying sessions of one pair, chronological.  Visibility-only: no K input."""
    sr = grid["session_rules"]
    out = []
    for w in range(rows.n_sessions):
        k = np.where(rows.session == w)[0]
        t0, t1 = float(rows.t_used[k[0]]), float(rows.t_used[k[-1]])
        w_start, w_end = t0, t1 + CADENCE_S
        if w_end - w_start < sr["min_window_s"]:
            continue
        if w_end - w_start >= sr["session_max_s"]:
            c = 0.5 * (w_start + w_end)
            a, b = c - 0.5 * sr["session_max_s"], c + 0.5 * sr["session_max_s"]
            k = k[(rows.t_used[k] >= a) & (rows.t_used[k] + CADENCE_S <= b)]
        else:
            a, b = w_start, w_end
        nps = sorted({int(k[int(round(f * (k.size - 1)))]) for f in FRACTIONS})
        out.append(Session(rows.pair, int(w), w_start, w_end, float(a), float(b),
                           tuple(int(i) for i in k), tuple(nps)))
    return sorted(out, key=lambda s: s.window_start)


def select_pattern(sessions: list[Session], pattern: str, t_arc_mid: float) -> list[Session] | None:
    """Chronological pattern; None when the pattern is infeasible for this geometry."""
    if not sessions:
        return None
    if pattern == "FIRST_SESSION":
        return [sessions[0]]
    if pattern == "LAST_SESSION":
        return [sessions[-1]]
    if pattern == "MID_SESSION":
        return [min(sessions, key=lambda s: (abs(s.mid - t_arc_mid), s.window_start))]
    if pattern == "FIRST_PLUS_LAST":
        return [sessions[0], sessions[-1]] if len(sessions) >= 2 else None
    if pattern == "ALL_FEASIBLE_SESSIONS":
        return list(sessions)
    raise ValueError(pattern)


def overlaps(a: Session, b: Session) -> bool:
    """Closed-interval overlap of two session intervals."""
    return a.start <= b.end and b.start <= a.end


def resolve_goldstone(gc: list[Session], gm: list[Session]) -> tuple[list[Session], list[Session], int]:
    """GC priority: keep every GC session; drop GM sessions overlapping a kept GC session.
    Returns (gc, gm_kept, raw_conflicts)."""
    kept, raw = [], 0
    for s in gm:
        hit = any(overlaps(s, g) for g in gc)
        raw += int(hit)
        if not hit:
            kept.append(s)
    return list(gc), kept, raw


# ======================================================================
# Stacked schedules with auditable ownership
# ======================================================================
@dataclass
class Schedule:
    strategy: str
    pattern: str
    epoch_id: int
    h_x0: np.ndarray
    h_k: np.ndarray
    t: np.ndarray
    baseline: np.ndarray          # (M,) "GC"/"GM"
    session: np.ndarray           # (M,) global session id 0..q-1 (chronological across baselines)
    normal_point: np.ndarray      # (M,) 0..2 within the session
    parent_row: np.ndarray        # (M,) index into the pair's DdorRows
    sessions: list = field(default_factory=list)
    raw_conflicts: int = 0
    gm_dropped: int = 0

    @property
    def n(self) -> int:
        return int(self.h_k.size)

    @property
    def n_sessions(self) -> int:
        return len(self.sessions)

    @property
    def session_hours(self) -> float:
        return float(sum(s.hours for s in self.sessions))


def stack(strategy: str, pattern: str, epoch_id: int, chosen: list[Session], rows_by_pair: dict,
          raw_conflicts: int = 0, gm_dropped: int = 0) -> Schedule:
    chosen = sorted(chosen, key=lambda s: (s.start, s.pair))
    hx, hk, t, bl, se, npt, par = [], [], [], [], [], [], []
    for sid, s in enumerate(chosen):
        rows = rows_by_pair[s.pair]
        for j, i in enumerate(s.np_index):
            hx.append(rows.h_x0[i]); hk.append(rows.h_k[i]); t.append(rows.t_used[i])
            bl.append(s.pair); se.append(sid); npt.append(j); par.append(i)
    return Schedule(strategy, pattern, epoch_id, np.asarray(hx, float).reshape(-1, 6), np.asarray(hk, float),
                    np.asarray(t, float), np.asarray(bl), np.asarray(se, int), np.asarray(npt, int),
                    np.asarray(par, int), chosen, raw_conflicts, gm_dropped)


def build_schedules(epoch_id: int, rows_by_pair: dict, sessions_by_pair: dict, t_arc_mid: float) -> dict:
    """All predeclared (strategy, pattern) schedules; infeasible ones map to None."""
    out = {}
    for strategy, pair in (("GC_ONLY", "GC"), ("GM_ONLY", "GM")):
        for p in PATTERNS_SINGLE:
            ch = select_pattern(sessions_by_pair[pair], p, t_arc_mid)
            out[(strategy, p)] = None if ch is None else stack(strategy, p, epoch_id, ch, rows_by_pair)
    for jp, single in PATTERNS_JOINT.items():
        gc = select_pattern(sessions_by_pair["GC"], single, t_arc_mid)
        gm = select_pattern(sessions_by_pair["GM"], single, t_arc_mid)
        if gc is None or gm is None:
            out[("GC_PLUS_GM", jp)] = None
            continue
        gc_k, gm_k, raw = resolve_goldstone(gc, gm)
        out[("GC_PLUS_GM", jp)] = stack("GC_PLUS_GM", jp, epoch_id, gc_k + gm_k, rows_by_pair,
                                        raw_conflicts=raw, gm_dropped=len(gm) - len(gm_k))
    return out


def bias_columns(n_range: int, sched: Schedule) -> np.ndarray:
    """(n_range + M, q): column j is 1 exactly on the rows of global session j."""
    q = sched.n_sessions
    out = np.zeros((n_range + sched.n, q))
    for j in range(q):
        out[n_range + np.where(sched.session == j)[0], j] = 1.0
    return out


def double_booked(sched: Schedule) -> bool:
    gc = [s for s in sched.sessions if s.pair == "GC"]
    gm = [s for s in sched.sessions if s.pair == "GM"]
    return any(overlaps(a, b) for a in gc for b in gm)


# ======================================================================
# Metrics
# ======================================================================
def delay_weight(n: int, grid: dict) -> np.ndarray:
    return np.full(n, 1.0 / (grid["noise"]["primary"]["sigma_ns"] * 1e-9) ** 2)


def calibration_prior(cal: str, grid: dict):
    """(use_nuisance, prior_sigma_s or None)."""
    nominal = grid["calibration"]["nominal_bias_prior_ns"] * 1e-9
    if cal == "C0_IDEAL":
        return False, None
    if cal.startswith("C1_x"):
        return True, float(cal[4:]) * nominal
    if cal == "C2_FREE":
        return True, None
    raise ValueError(cal)


def nuisance_posterior(blocks, nuis: np.ndarray, prior_sigma) -> dict:
    """Bias posterior sigmas and K-bias correlations from the R1COV QR covariance
    (same scaling / prior convention as D2.info_metrics)."""
    from phase17_r1cov_core import RankDeficientCovariance, square_root_covariance
    h_x0, h_k, w = D2._assemble(blocks)
    q = nuis.shape[1]
    h_full = np.hstack([h_x0, h_k[:, None], nuis])
    scale = np.diag(list(np.diag(D2.SCALE)) + [D2.NUISANCE_SCALE_S] * q)
    prior_inv = None
    if prior_sigma is not None:
        prior_inv = np.zeros((7 + q, 7 + q))
        prior_inv[7:, 7:] = np.eye(q) / prior_sigma ** 2
    try:
        cov = np.asarray(square_root_covariance(h_full, w, prior_inv, scale).covariance, float)
    except RankDeficientCovariance:
        return dict(bias_sigma_ns=[float("inf")] * q, max_abs_k_bias_corr=float("nan"))
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    corr = cov[D2.K_INDEX, 7:] / (sd[D2.K_INDEX] * sd[7:])
    return dict(bias_sigma_ns=[float(s * 1e9) for s in sd[7:]],
                max_abs_k_bias_corr=float(np.max(np.abs(corr))) if q else float("nan"))


KEYS = ("n_obs", "f_perp", "theta_k_deg", "conditional_k_information", "i_kk", "k_column_norm", "sigma_k",
        "sigma_k_frac", "design_singular_values", "design_condition", "weakest_mode_k_component",
        "max_abs_k_state_correlation", "max_abs_k_nuisance_correlation")


def schedule_metrics(rb, sched: Schedule, cal: str, grid: dict, range_f: float) -> dict:
    blocks = [(rb.h_x0, rb.h_k, rb.w), (sched.h_x0, sched.h_k, delay_weight(sched.n, grid))]
    use, prior = calibration_prior(cal, grid)
    nuis = bias_columns(rb.n_obs, sched) if use else None
    m = D2.info_metrics(blocks, nuisance=nuis, nuisance_prior_sigma=prior)
    d = m["f_perp"] - range_f
    out = {k: m[k] for k in KEYS}
    out.update(delta_f_perp=d, relative_class=D2.relative_gain_class(d),
               absolute_class=D2.absolute_class(m["f_perp"]))
    if use:
        out.update(nuisance_posterior(blocks, nuis, prior))
    return out


def pair_row_contract(rows, rb, pair: str) -> bool:
    """Every row of the pair is dual-visible under the production contract."""
    ia, ib = (D2.DSN3.index(n) for n in D2.PAIRS[pair])
    return bool(np.all(rb.vis[rows.idx, ia] & rb.vis[rows.idx, ib]))


def projected_baseline_km(rows, idx: np.ndarray) -> float:
    base, s = rows.baseline[idx], rows.s_sc_geo[idx]
    bp = base - np.sum(base * s, axis=1)[:, None] * s
    return float(np.mean(np.linalg.norm(bp, axis=1)) / 1e3)


def independent_f_perp(rb, sched: Schedule, cal: str, grid: dict) -> float:
    """Independent rebuild of combined f_perp from component blocks, by a DIFFERENT
    route than D2.info_metrics: explicit normal-equation-free projection with
    numpy.linalg.lstsq on the whitened state (+ nuisance, + prior pseudo-rows)."""
    w = np.concatenate([rb.w, delay_weight(sched.n, grid)])
    h_x = np.vstack([rb.h_x0, sched.h_x0])
    h_k = np.concatenate([rb.h_k, sched.h_k])
    use, prior = calibration_prior(cal, grid)
    cols = [h_x * np.diag(D2.SCALE)[:6]]
    if use:
        cols.append(bias_columns(rb.n_obs, sched) * D2.NUISANCE_SCALE_S)
    a = np.sqrt(w)[:, None] * np.hstack(cols)
    b = np.sqrt(w) * h_k
    if use and prior is not None:
        q = sched.n_sessions
        a = np.vstack([a, np.hstack([np.zeros((q, 6)), np.eye(q) * D2.NUISANCE_SCALE_S / prior])])
        b = np.concatenate([b, np.zeros(q)])
    coef, *_ = np.linalg.lstsq(a, b, rcond=None)
    r = b - a @ coef
    return float(np.linalg.norm(r) / np.linalg.norm(b))
