"""Phase 13G — real lunar gravity orbit-effect validation campaign.

Physical-effect VALIDATION of the real GRAIL models on the baseline LLO
(phase6 orbit: a = R_MOON + 100 km, e = 0.01, i = 45 deg, perilune ~81.6 km).
Phase 12B verified that the machinery works; this campaign asks whether the
orbit-level effects are physically correct, decomposable and interpretable.

NON-GOALS (Phase 13G-b scope): no filters, no OD, no estimators, no STM, no
scenario-runner threading, no sensitivity sweeps (13G-c), no production-code
changes.

Truth-model note: every number here is an INTERNAL model-vs-model difference
(truth-model hierarchy Level 1) under one shared integrator, one shared
DE421-derived .mat translational ephemeris and identical tolerances; nothing
is an accuracy claim against external truth.

Stages:
  --baseline  GRGM660PRIM + MOON_PA_DE421: V0 point-mass+3rd-body,
              V1 classical J2, V2 real C20-only (bridge), V3 C20+C22,
              V4 zonal-only(64), V5 tesseral-only(64, C20 removed;
              diagnostic decomposition), V6 full nmax ladder 8..128;
              1 orbit + 1 day; Cartesian + RTN + osculating-element metrics.
  --compare   GL1800F + MOON_PA_DE440 comparison points: cross-model at
              nmax=64 and the 64->128->256 convergence tail (256 runtime-gated).
  --frames    INTENTIONALLY-WRONG frame diagnostics (flagged diagnostic=True):
              frozen-at-t0 grid ("Moon not rotating"), constant mean-pole grid,
              wrong PA pairing (660PM coefficients on a MOON_PA_DE440 grid).
              The m>0+constant-matrix production guard is deliberately emulated
              around via constant-valued GRIDS; never a bypass of physics
              guards for a "valid" run.
  --sensitivity  13G-c1 one-factor axes (8 cases, GRGM660PRIM primary):
              altitude 100/200/500 km circular (i=45), inclination
              0/30/60/90 deg (100 km circular), eccentric 80x500 km (i=45,
              perilune-window |nu|<30 deg metric); per case V1/V2/V3/
              zonal@64 + full ladder 8..128; GL1800F only on S1/S7/S8
              (64/128, 256 only on S8 runtime-gated).  The "orbit" window
              uses each case's own period.
  --sevenday  13G-c2 selected seven-day confirmation (L1=S1 anchor,
              L2=S7 polar, L3=S8 eccentric, L4=S3 control; GRGM660PRIM
              J2/32/64/128 everywhere, zonal@64 on L2/L3; GL1800F only
              L2@128 and L3@128/256-gated).  Every scientific case uses ONE
              uninterrupted continuous propagate_state call (the chunked
              state-handoff methodology was REJECTED by the continuous-vs-
              chunked preflight — multistep integrator restart artifact —
              and is retained only as recorded evidence).  Surface crossing
              is detected on the sampled output grid and truncates the
              post-processing arrays at the first sampled crossing;
              comparisons use the minimum common valid horizon.  Atomic
              per-case store writes; a rerun rebuilds the sevenday stage
              cleanly (an interrupted model case reruns continuous from
              scratch; no mid-trajectory resume).
  (default: --baseline)
Refused here: --plots.

Reuse: loader/profile/rotation/propagation helpers are imported from the
committed phase12b script; initial state/ephemeris from phase6.  Script-local
additions (deliberately NOT production): make_variant mask factory, rv2coe,
RTN decomposition, element drift summary, frozen/constant rotation pairs.

Outputs (git-ignored, summary only; dense trajectories stay in memory):
  results/phase13g/phase13g_store.json         cumulative store
  results/phase13g/phase13g_runs.csv           per-run safety/runtime metrics
  results/phase13g/phase13g_comparisons.csv    pairwise Cartesian+RTN metrics
  results/phase13g/phase13g_element_drift.csv  osculating element summaries
  results/phase13g/phase13g_report.md
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))

from lunar_od.constants import (  # noqa: E402
    J2_MOON_UNNORMALIZED, MU_EARTH_M3S2, MU_MOON_M3S2, MU_SUN_M3S2, R_MOON_M,
)
from lunar_od.dynamics import _MCI_TO_MOON_BF, propagate_state  # noqa: E402
from lunar_od.gravity_harmonics import SphericalHarmonicGravityModel  # noqa: E402
from lunar_od.gravity_model_loader import (  # noqa: E402
    describe_model, load_lunar_gravity_model, resolve_gravity_dir,
)
from lunar_od.orbit import coe2rv  # noqa: E402
from phase12b_real_grail_validation import (  # noqa: E402
    MODELS as GRAIL_FILES, _CountingGetter, diff_metrics, load_kernel_profile,
    rotation_pair, run_case, truncate_model,
)
from phase6_scenario_comparison import initial_state, load_ephemeris  # noqa: E402

MU_M, MU_E, MU_S = MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2
J2 = J2_MOON_UNNORMALIZED
J2000_JD = 2451545.0
OUT = ROOT / "results" / "phase13g"
STORE_PATH = OUT / "phase13g_store.json"
RUNS_CSV = OUT / "phase13g_runs.csv"
COMP_CSV = OUT / "phase13g_comparisons.csv"
ELEM_CSV = OUT / "phase13g_element_drift.csv"
MD_PATH = OUT / "phase13g_report.md"
SENS_RUNS_CSV = OUT / "phase13g_sensitivity_runs.csv"
SENS_COMP_CSV = OUT / "phase13g_sensitivity_comparisons.csv"
SENS_ELEM_CSV = OUT / "phase13g_sensitivity_elements.csv"
SENS_MD_PATH = OUT / "phase13g_sensitivity_report.md"
SEVEN_RUNS_CSV = OUT / "phase13g_sevenday_runs.csv"
SEVEN_COMP_CSV = OUT / "phase13g_sevenday_comparisons.csv"
SEVEN_ELEM_CSV = OUT / "phase13g_sevenday_elements.csv"
SEVEN_DAILY_CSV = OUT / "phase13g_sevenday_daily.csv"
SEVEN_MD_PATH = OUT / "phase13g_sevenday_report.md"

CADENCE_S = 60.0                    # Phase 13C accepted default
OPT_RUNTIME_GATE_S = 300.0          # projected cap for optional nmax=256
LADDER_660 = (8, 16, 32, 64, 128)   # GRGM660PRIM full-model ladder
DECOMP_NMAX = 64                    # zonal/tesseral decomposition degree

TRUTH_NOTE = (
    "All differences are internal model-vs-model comparisons (truth-model "
    "hierarchy Level 1) under one shared integrator, one shared DE421-derived "
    ".mat translational ephemeris (also for the GL1800F+DE440 rotation runs — "
    "declared mix, effect measured at m-level in Phase 12B) and identical "
    "tolerances; nothing here is accuracy against external truth."
)


# ---------------------------------------------------------------------------
# Script-local model decomposition (production loader/engine untouched)
# ---------------------------------------------------------------------------
def make_variant(
    model: SphericalHarmonicGravityModel, mode: str, nmax: int | None = None
) -> SphericalHarmonicGravityModel:
    """Masked copy of a real model for physical decomposition.

    Modes: ``c20_only`` (only Cbar[2,0]; mmax=0), ``c20_c22`` (degree-2 m=0
    and m=2 terms), ``zonal_only`` (all m>0 zeroed; mmax=0),
    ``tesseral_only`` (ALL m=0 zeroed including C20 — diagnostic
    decomposition; runs with j2_moon=0 so no double-count arises).
    ``dataclasses.replace`` re-runs the production shape/sign guards.
    """
    base = truncate_model(model, nmax) if nmax is not None else model
    cbar = base.cbar.copy()
    sbar = base.sbar.copy()
    if mode == "c20_only":
        keep = cbar[2, 0]
        cbar = np.zeros((3, 3)); sbar = np.zeros((3, 3))
        cbar[2, 0] = keep
        new_nmax, new_mmax = 2, 0
    elif mode == "c20_c22":
        c20, c22, s22 = cbar[2, 0], cbar[2, 2], sbar[2, 2]
        cbar = np.zeros((3, 3)); sbar = np.zeros((3, 3))
        cbar[2, 0], cbar[2, 2], sbar[2, 2] = c20, c22, s22
        new_nmax, new_mmax = 2, 2
    elif mode == "zonal_only":
        cbar[:, 1:] = 0.0
        sbar[:, :] = 0.0
        new_nmax, new_mmax = base.nmax, 0
    elif mode == "tesseral_only":
        cbar[:, 0] = 0.0            # removes every zonal term INCLUDING C20
        new_nmax, new_mmax = base.nmax, base.mmax
    else:
        raise ValueError(f"unknown variant mode {mode!r}.")
    metadata = dict(base.metadata)
    metadata["variant"] = mode
    return dataclasses.replace(
        base, cbar=cbar, sbar=sbar, nmax=new_nmax, mmax=new_mmax, metadata=metadata
    )


# ---------------------------------------------------------------------------
# Script-local osculating elements (production has coe2rv only, no inverse)
# ---------------------------------------------------------------------------
_E_TOL = 1e-9
_INC_TOL = 1e-9


def rv2coe(r_m, v_mps, mu_m3_s2: float) -> dict:
    """Osculating classical elements; NaN for angles undefined at singularities.

    Near-circular: nu/argp -> NaN, use ``u`` (argument of latitude).
    Near-equatorial: raan/u -> NaN, ``true_longitude`` stays defined.
    """
    r = np.asarray(r_m, dtype=float).reshape(3)
    v = np.asarray(v_mps, dtype=float).reshape(3)
    rn = float(np.linalg.norm(r))
    h = np.cross(r, v)
    hn = float(np.linalg.norm(h))
    node = np.cross([0.0, 0.0, 1.0], h)
    nn = float(np.linalg.norm(node))
    e_vec = np.cross(v, h) / mu_m3_s2 - r / rn
    ecc = float(np.linalg.norm(e_vec))
    energy = float(v @ v) / 2.0 - mu_m3_s2 / rn
    sma = -mu_m3_s2 / (2.0 * energy)
    inc = math.acos(max(-1.0, min(1.0, h[2] / hn)))

    def _angle_between(a, b, sign_ref):
        cosang = float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
        ang = math.acos(max(-1.0, min(1.0, cosang)))
        return 2.0 * math.pi - ang if sign_ref < 0.0 else ang

    equatorial = nn < _INC_TOL
    circular = ecc < _E_TOL
    raan = math.nan if equatorial else math.atan2(node[1], node[0]) % (2.0 * math.pi)
    argp = (
        math.nan if (equatorial or circular)
        else _angle_between(node, e_vec, e_vec[2])
    )
    nu = math.nan if circular else _angle_between(e_vec, r, float(r @ v))
    u = math.nan if equatorial else _angle_between(node, r, r[2])
    true_longitude = math.atan2(r[1], r[0]) % (2.0 * math.pi)
    return {
        "a_m": sma, "e": ecc, "i_rad": inc, "raan_rad": raan,
        "argp_rad": argp, "nu_rad": nu, "u_rad": u,
        "true_longitude_rad": true_longitude,
    }


_ELEMENTS = ("a_m", "e", "i_rad", "raan_rad", "argp_rad", "u_rad")
_ANGLE_ELEMENTS = {"i_rad", "raan_rad", "argp_rad", "u_rad"}


def element_series(traj: np.ndarray, mu: float) -> dict[str, np.ndarray]:
    rows = [rv2coe(state[:3], state[3:], mu) for state in traj]
    return {key: np.array([row[key] for row in rows]) for key in _ELEMENTS}


def element_drift_summary(
    t_s: np.ndarray, series: dict[str, np.ndarray], period_s: float | None = None
) -> list[dict]:
    """Per element: initial/final/min/max, secular drift per day, short-period
    amplitude.  Drift uses ORBIT-AVERAGED differencing (mean over the first
    orbital period vs mean over the last) whenever the window holds >= 2
    periods — a plain linear fit leaks short-period signal into the slope
    (t*sin(wt) does not integrate to zero even over whole periods).  Shorter
    windows fall back to the endpoint difference.  Angles are unwrapped first;
    any NaN in a series yields NaN statistics."""
    t_s = np.asarray(t_s, dtype=float)
    span = float(t_s[-1] - t_s[0])
    averaged = period_s is not None and span >= 2.0 * period_s
    out = []
    for key, values in series.items():
        if np.any(~np.isfinite(values)):
            out.append({"element": key, "initial": math.nan, "final": math.nan,
                        "min": math.nan, "max": math.nan,
                        "drift_per_day": math.nan, "short_period_amp": math.nan})
            continue
        work = np.unwrap(values) if key in _ANGLE_ELEMENTS else values
        if averaged:
            head = t_s <= t_s[0] + period_s
            tail = t_s >= t_s[-1] - period_s
            dt = float(t_s[tail].mean() - t_s[head].mean())
            slope = float(work[tail].mean() - work[head].mean()) / dt
        else:
            slope = float(work[-1] - work[0]) / span
        resid = work - (work.mean() + slope * (t_s - t_s.mean()))
        out.append({
            "element": key,
            "initial": float(work[0]),
            "final": float(work[-1]),
            "min": float(work.min()),
            "max": float(work.max()),
            "drift_per_day": float(slope * 86400.0),
            "short_period_amp": float((resid.max() - resid.min()) / 2.0),
        })
    return out


# ---------------------------------------------------------------------------
# Script-local RTN decomposition of a trajectory difference
# ---------------------------------------------------------------------------
def rtn_basis(r_ref: np.ndarray, v_ref: np.ndarray) -> np.ndarray:
    r_hat = r_ref / np.linalg.norm(r_ref)
    w = np.cross(r_ref, v_ref)
    w_hat = w / np.linalg.norm(w)
    s_hat = np.cross(w_hat, r_hat)
    return np.vstack([r_hat, s_hat, w_hat])       # rows: R, T(along), N(cross)


def rtn_series(traj: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """(N, 3) radial/along-track/cross-track components of traj-ref positions."""
    out = np.empty((ref.shape[0], 3))
    for k in range(ref.shape[0]):
        basis = rtn_basis(ref[k, :3], ref[k, 3:])
        out[k] = basis @ (traj[k, :3] - ref[k, :3])
    return out

def rtn_summary(traj: np.ndarray, ref: np.ndarray) -> dict:
    comps = rtn_series(traj, ref)
    rms = np.sqrt(np.mean(comps ** 2, axis=0))
    return {
        "final_radial_m": float(comps[-1, 0]),
        "final_along_m": float(comps[-1, 1]),
        "final_cross_m": float(comps[-1, 2]),
        "max_abs_along_m": float(np.abs(comps[:, 1]).max()),
        "rms_radial_m": float(rms[0]),
        "rms_along_m": float(rms[1]),
        "rms_cross_m": float(rms[2]),
    }


# ---------------------------------------------------------------------------
# Sensitivity orbit set (13G-c1): one-factor-at-a-time, 8 cases, no cross
# product.  States built with the production coe2rv (RAAN/argp/nu fixed at the
# baseline style values).
# ---------------------------------------------------------------------------
SENSITIVITY_CASES = (
    {"case": "S1_alt100_i45", "axis": "altitude", "alt_km": 100.0, "incl_deg": 45.0},
    {"case": "S2_alt200_i45", "axis": "altitude", "alt_km": 200.0, "incl_deg": 45.0},
    {"case": "S3_alt500_i45", "axis": "altitude", "alt_km": 500.0, "incl_deg": 45.0},
    {"case": "S4_alt100_i0", "axis": "inclination", "alt_km": 100.0, "incl_deg": 0.0},
    {"case": "S5_alt100_i30", "axis": "inclination", "alt_km": 100.0, "incl_deg": 30.0},
    {"case": "S6_alt100_i60", "axis": "inclination", "alt_km": 100.0, "incl_deg": 60.0},
    {"case": "S7_alt100_i90", "axis": "inclination", "alt_km": 100.0, "incl_deg": 90.0},
    {"case": "S8_ecc80x500_i45", "axis": "eccentric",
     "peri_alt_km": 80.0, "apo_alt_km": 500.0, "incl_deg": 45.0},
)
# GL1800F runs ONLY on these selected cases (no full-matrix duplication)
GL1800F_CASES = ("S1_alt100_i45", "S7_alt100_i90", "S8_ecc80x500_i45")
PERILUNE_HALF_WIDTH_DEG = 30.0


def case_state(spec: dict) -> np.ndarray:
    """Initial 6-state for one sensitivity case (circular or peri/apo pair)."""
    if "peri_alt_km" in spec:
        r_peri = R_MOON_M + spec["peri_alt_km"] * 1e3
        r_apo = R_MOON_M + spec["apo_alt_km"] * 1e3
        a, e = 0.5 * (r_peri + r_apo), (r_apo - r_peri) / (r_apo + r_peri)
    else:
        a, e = R_MOON_M + spec["alt_km"] * 1e3, 0.0
    r, v = coe2rv(a, e, math.radians(spec["incl_deg"]), math.radians(30.0),
                  math.radians(20.0), math.radians(10.0), MU_M)
    return np.concatenate([r, v])


def build_sensitivity_matrix() -> dict:
    """Pure run-plan builder (no data, no SPICE) — the anti-explosion guard.

    Per case, GRGM660PRIM gets at most 9 runs (V1/V2/V3/V4 + the 5-step
    ladder); GL1800F appears only on the GL1800F_CASES subset."""
    g660_runs = ("v1_j2", "v2_c20", "v3_c20c22", "v4_zonal64") + tuple(
        f"v6_full{n}" for n in LADDER_660
    )
    return {
        spec["case"]: {
            "axis": spec["axis"],
            "g660_runs": g660_runs,
            "gl1800f": spec["case"] in GL1800F_CASES,
        }
        for spec in SENSITIVITY_CASES
    }


def perilune_mask_from(traj: np.ndarray, half_width_deg: float) -> np.ndarray:
    """Epochs with |true anomaly| < half width, from a reference trajectory."""
    nus = np.array([rv2coe(s[:3], s[3:], MU_M)["nu_rad"] for s in traj])
    wrapped = (nus + np.pi) % (2.0 * np.pi) - np.pi
    with np.errstate(invalid="ignore"):
        return np.abs(wrapped) < math.radians(half_width_deg)


# ---------------------------------------------------------------------------
# Frame-diagnostic rotation pairs (INTENTIONALLY WRONG; grid-valued so the
# production m>0+constant-matrix guard is emulated around, not weakened)
# ---------------------------------------------------------------------------
def frozen_pair(pair):
    """Real time grid, every matrix frozen to the t=0 rotation ("Moon not rotating")."""
    t_grid, rots = pair
    return (t_grid, np.repeat(rots[:1], t_grid.size, axis=0))


def constant_pair(t_grid: np.ndarray, c: np.ndarray):
    """Real time grid, every matrix equal to a fixed matrix (e.g. mean pole)."""
    c = np.asarray(c, dtype=float).reshape(3, 3)
    return (t_grid, np.repeat(c[None, :, :], t_grid.size, axis=0))


# ---------------------------------------------------------------------------
# Campaign infrastructure
# ---------------------------------------------------------------------------
def window_setup(window: str, s0: np.ndarray | None = None):
    """Windows for one initial state (default: the phase6 baseline state, so
    the baseline/compare/frames stages behave exactly as in 13G-b).  The
    "orbit" window length comes from the CASE's own osculating period (a
    500 km or eccentric case must not reuse the 100 km baseline period).
    day1 is sampled at 120 s (denser than 13C's 600 s) so the osculating-
    element series resolves short-period content; sampling density does not
    change the integration itself."""
    if s0 is None:
        s0 = initial_state()
    if window == "orbit":
        a0 = rv2coe(s0[:3], s0[3:], MU_M)["a_m"]
        period = 2.0 * math.pi * math.sqrt(a0 ** 3 / MU_M)
        t_end = math.ceil(period / 60.0) * 60.0
        out_step = 60.0
    elif window == "day1":
        t_end = 86400.0
        out_step = 120.0
    else:
        raise ValueError(f"unknown window {window!r}.")
    teval = np.arange(0.0, t_end + 1.0, out_step)
    eph, first_jd = load_ephemeris(t_end + 600.0)
    et0 = (first_jd - J2000_JD) * 86400.0
    return teval, t_end, eph, et0, s0


def load_real_model(key: str, nmax: int) -> SphericalHarmonicGravityModel:
    tab_path = resolve_gravity_dir() / key / GRAIL_FILES[key]["tab"]
    if not tab_path.is_file():
        raise FileNotFoundError(
            f"real GRAIL file not found: {tab_path} — Phase 13G needs the "
            f"Phase 12B data layout under data/gravity/ (never committed)."
        )
    t0 = time.perf_counter()
    model = load_lunar_gravity_model(tab_path, nmax=nmax)
    print(f"[load] {key} nmax={nmax} in {time.perf_counter() - t0:.2f} s "
          f"-> {describe_model(model)}")
    return model


class WindowRunner:
    """Runs cases for one window, collecting run/comparison/element rows."""

    def __init__(self, window: str, stage: str, s0: np.ndarray | None = None,
                 case: str = "", axis: str = ""):
        self.window = window
        self.stage = stage
        self.case = case
        self.axis = axis
        self.teval, self.t_end, self.eph, self.et0, self.s0 = window_setup(window, s0)
        a0 = rv2coe(self.s0[:3], self.s0[3:], MU_M)["a_m"]
        self.period_s = 2.0 * math.pi * math.sqrt(a0 ** 3 / MU_M)
        self.trajs: dict[str, np.ndarray] = {}
        self.run_rows: list[dict] = []
        self.comp_rows: list[dict] = []
        self.elem_rows: list[dict] = []

    def pair(self, frame: str):
        return rotation_pair(self.et0, self.t_end, CADENCE_S, frame)

    def run(self, name: str, *, model=None, rotation=None, rotation_label="",
            j2_moon=0.0, variant="", model_key="", nmax=None,
            diagnostic=False, intentionally_wrong=False, elements=True) -> dict:
        res = run_case(self.eph.earth_position, self.eph.sun_position,
                       self.s0, self.teval,
                       j2_moon=j2_moon, harmonic_model=model,
                       harmonic_rotation=rotation)
        traj = res["traj"]
        self.trajs[name] = traj
        radii = np.linalg.norm(traj[:, :3], axis=1)
        row = {
            "stage": self.stage, "case": self.case, "axis": self.axis,
            "window": self.window, "run": name,
            "model": model_key, "variant": variant,
            "nmax": nmax if nmax is not None else (model.nmax if model else ""),
            "mmax": model.mmax if model else "",
            "rotation": rotation_label, "j2_moon": j2_moon,
            "cadence_s": CADENCE_S if rotation is not None else "",
            "runtime_s": round(res["runtime_s"], 3),
            "rhs_evals": res["rhs_evals"],
            "min_altitude_m": round(res["min_altitude_m"], 1),
            "max_altitude_m": round(float(radii.max()) - R_MOON_M, 1),
            "surface_crossing": res["surface_crossing"],
            "nan": False,
            "diagnostic": diagnostic,
            "intentionally_wrong": intentionally_wrong,
        }
        self.run_rows.append(row)
        if elements:
            for erow in element_drift_summary(
                self.teval, element_series(traj, MU_M), period_s=self.period_s
            ):
                self.elem_rows.append({
                    "stage": self.stage, "case": self.case, "axis": self.axis,
                    "window": self.window, "run": name,
                    "diagnostic": diagnostic, **erow,
                })
        flag = " DIAG" if diagnostic else ""
        print(f"  [{self.window}] {name:26s} rt {res['runtime_s']:6.2f} s  "
              f"rhs {res['rhs_evals']:6d}  min_alt "
              f"{res['min_altitude_m'] / 1e3:6.2f} km{flag}"
              f"{'  SURFACE-CROSSING!' if res['surface_crossing'] else ''}")
        return res

    def compare(self, name: str, run_a: str, reference: str, note: str,
                diagnostic=False, intentionally_wrong=False,
                perilune_mask: np.ndarray | None = None) -> dict:
        traj, ref = self.trajs[run_a], self.trajs[reference]
        row = {
            "stage": self.stage, "case": self.case, "axis": self.axis,
            "window": self.window, "comparison": name,
            "run": run_a, "reference": reference,
            **diff_metrics(traj, ref), **rtn_summary(traj, ref),
            "diagnostic": diagnostic, "intentionally_wrong": intentionally_wrong,
            "note": note,
        }
        if perilune_mask is not None and perilune_mask.any():
            dp = np.linalg.norm(traj[:, :3] - ref[:, :3], axis=1)
            peri_rms = float(np.sqrt(np.mean(dp[perilune_mask] ** 2)))
            row["perilune_rms_dpos_m"] = peri_rms
            row["perilune_n_epochs"] = int(perilune_mask.sum())
            if row["rms_dpos_m"] > 0.0:
                row["perilune_ratio"] = peri_rms / row["rms_dpos_m"]
        self.comp_rows.append(row)
        print(f"  [{self.window}] {name}: final dpos "
              f"{row['final_dpos_m']:.3e} m (R {row['final_radial_m']:+.2e} "
              f"T {row['final_along_m']:+.2e} N {row['final_cross_m']:+.2e})"
              f"{'  [INTENTIONALLY WRONG]' if intentionally_wrong else ''}")
        return row


# ---------------------------------------------------------------------------
# Stage: baseline (GRGM660PRIM + MOON_PA_DE421)
# ---------------------------------------------------------------------------
def run_baseline() -> dict:
    big = load_real_model("grgm660prim", max(LADDER_660))
    variants = {
        "c20_only": make_variant(big, "c20_only"),
        "c20_c22": make_variant(big, "c20_c22"),
        "zonal_only": make_variant(big, "zonal_only", nmax=DECOMP_NMAX),
        "tesseral_only": make_variant(big, "tesseral_only", nmax=DECOMP_NMAX),
    }
    stage: dict = {"model": describe_model(big), "windows": {}, "kepler_check": None}
    for window in ("orbit", "day1"):
        wr = WindowRunner(window, "baseline")
        load_kernel_profile("de421")
        pair = wr.pair("MOON_PA_DE421")

        wr.run("v0_pointmass", model_key="grgm660prim", variant="none")
        wr.run("v1_j2", j2_moon=J2, model_key="grgm660prim", variant="classical_j2")
        wr.run("v2_c20", model=variants["c20_only"], rotation=pair,
               rotation_label="MOON_PA_DE421@60s", model_key="grgm660prim",
               variant="c20_only")
        wr.run("v3_c20c22", model=variants["c20_c22"], rotation=pair,
               rotation_label="MOON_PA_DE421@60s", model_key="grgm660prim",
               variant="c20_c22")
        wr.run("v4_zonal64", model=variants["zonal_only"], rotation=pair,
               rotation_label="MOON_PA_DE421@60s", model_key="grgm660prim",
               variant="zonal_only")
        wr.run("v5_tesseral64", model=variants["tesseral_only"], rotation=pair,
               rotation_label="MOON_PA_DE421@60s", model_key="grgm660prim",
               variant="tesseral_only", diagnostic=True)
        for nmax in LADDER_660:
            wr.run(f"v6_full{nmax}", model=truncate_model(big, nmax),
                   rotation=pair, rotation_label="MOON_PA_DE421@60s",
                   model_key="grgm660prim", variant="full", nmax=nmax)

        wr.compare("j2_effect", "v1_j2", "v0_pointmass",
                   "classical Moon-J2 signal above point-mass+3rd-body")
        wr.compare("c20_bridge", "v2_c20", "v1_j2",
                   "internal consistency: real C20-only vs classical J2 "
                   "(GRAIL C20 vs constants J2 ~0.12% + PA-vs-mean-pole frame)")
        wr.compare("c22_effect", "v3_c20c22", "v2_c20",
                   "real C22/S22 signal (longitude-dependent)")
        wr.compare("tesseral_from_full_minus_zonal", "v6_full64", "v4_zonal64",
                   "tesseral contribution incl. zonal-tesseral coupling")
        wr.compare("tesseral_direct", "v5_tesseral64", "v0_pointmass",
                   "direct tesseral-only signal (C20 removed; diagnostic "
                   "decomposition)", diagnostic=True)
        wr.compare("j2only_model_error", "v1_j2", "v6_full64",
                   "HEADLINE: error of a classical-J2-only propagation vs "
                   "full real model nmax=64")
        for lo, hi in zip(LADDER_660[:-1], LADDER_660[1:]):
            wr.compare(f"ladder_{lo}_vs_{hi}", f"v6_full{lo}", f"v6_full{hi}",
                       f"truncation step nmax {lo}->{hi}")

        stage["windows"][window] = {
            "t_end_s": wr.t_end, "n_epochs": int(wr.teval.size),
            "runs": wr.run_rows, "comparisons": wr.comp_rows,
            "elements": wr.elem_rows,
        }

        if window == "orbit" and stage["kepler_check"] is None:
            stage["kepler_check"] = _kepler_energy_check(wr)
    return stage


def _kepler_energy_check(wr: WindowRunner) -> dict:
    """Integrator sanity: Moon-only two-body run must conserve energy."""
    try:
        traj = propagate_state(wr.teval, wr.s0, MU_M, 0.0, 0.0,
                               wr.eph.earth_position, wr.eph.sun_position,
                               method="ADAMS")
    except Exception as exc:                      # noqa: BLE001 (optional check)
        return {"ran": False, "reason": f"{type(exc).__name__}: {exc}"}
    r = np.linalg.norm(traj[:, :3], axis=1)
    v2 = np.sum(traj[:, 3:] ** 2, axis=1)
    energy = v2 / 2.0 - MU_M / r
    rel_drift = float(abs(energy[-1] - energy[0]) / abs(energy[0]))
    return {"ran": True, "rel_energy_drift_1orbit": rel_drift}


# ---------------------------------------------------------------------------
# Stage: compare (GL1800F + MOON_PA_DE440 comparison points)
# ---------------------------------------------------------------------------
def run_compare() -> dict:
    big660 = load_real_model("grgm660prim", 64)
    big1800 = load_real_model("gl1800f", 256)
    stage: dict = {"windows": {}}
    for window in ("orbit", "day1"):
        wr = WindowRunner(window, "compare")

        load_kernel_profile("de421")
        pair421 = wr.pair("MOON_PA_DE421")
        wr.run("g660_full64", model=truncate_model(big660, 64), rotation=pair421,
               rotation_label="MOON_PA_DE421@60s", model_key="grgm660prim",
               variant="full", nmax=64)

        load_kernel_profile("de440")
        pair440 = wr.pair("MOON_PA_DE440")
        for nmax in (64, 128):
            wr.run(f"g1800_full{nmax}", model=truncate_model(big1800, nmax),
                   rotation=pair440, rotation_label="MOON_PA_DE440@60s",
                   model_key="gl1800f", variant="full", nmax=nmax)
        rt128 = wr.run_rows[-1]["runtime_s"]
        projected = rt128 * 4.0
        if projected <= OPT_RUNTIME_GATE_S:
            wr.run("g1800_full256", model=truncate_model(big1800, 256),
                   rotation=pair440, rotation_label="MOON_PA_DE440@60s",
                   model_key="gl1800f", variant="full", nmax=256)
            wr.compare("g1800_ladder_128_vs_256", "g1800_full128",
                       "g1800_full256", "high-degree convergence tail")
        else:
            print(f"  [{window}] g1800_full256 SKIPPED "
                  f"(projected {projected:.0f} s > {OPT_RUNTIME_GATE_S:.0f} s)")

        wr.compare("cross_model_nmax64", "g660_full64", "g1800_full64",
                   "GRGM660PRIM@64 (MOON_PA_DE421) vs GL1800F@64 "
                   "(MOON_PA_DE440); mixes coefficient difference + DE421/"
                   "DE440 orientation + shared DE421 .mat ephemeris — the "
                   "three effects are NOT separated")
        wr.compare("g1800_ladder_64_vs_128", "g1800_full64", "g1800_full128",
                   "truncation step nmax 64->128")

        stage["windows"][window] = {
            "t_end_s": wr.t_end, "n_epochs": int(wr.teval.size),
            "runs": wr.run_rows, "comparisons": wr.comp_rows,
            "elements": wr.elem_rows,
        }
    return stage


# ---------------------------------------------------------------------------
# Stage: frames (INTENTIONALLY WRONG diagnostics, clearly flagged)
# ---------------------------------------------------------------------------
def run_frames() -> dict:
    big = load_real_model("grgm660prim", 64)
    stage: dict = {"windows": {}}
    for window in ("orbit", "day1"):
        wr = WindowRunner(window, "frames")

        load_kernel_profile("de440")
        pair440 = wr.pair("MOON_PA_DE440")     # for the wrong-pairing diagnostic

        load_kernel_profile("de421")
        pair421 = wr.pair("MOON_PA_DE421")
        frozen = frozen_pair(pair421)
        meanpole = constant_pair(pair421[0], _MCI_TO_MOON_BF)

        for nmax in (8, 64):
            model_n = truncate_model(big, nmax)
            correct = f"g660_n{nmax}_correct"
            wr.run(correct, model=model_n, rotation=pair421,
                   rotation_label="MOON_PA_DE421@60s",
                   model_key="grgm660prim", variant="full", nmax=nmax)
            wr.run(f"g660_n{nmax}_frozen", model=model_n, rotation=frozen,
                   rotation_label="FROZEN@t0 (diagnostic)",
                   model_key="grgm660prim", variant="full", nmax=nmax,
                   diagnostic=True, intentionally_wrong=True, elements=False)
            wr.run(f"g660_n{nmax}_wrongpair", model=model_n, rotation=pair440,
                   rotation_label="MOON_PA_DE440 (wrong pairing, diagnostic)",
                   model_key="grgm660prim", variant="full", nmax=nmax,
                   diagnostic=True, intentionally_wrong=True, elements=False)
            if nmax == 64:
                wr.run("g660_n64_meanpole", model=model_n, rotation=meanpole,
                       rotation_label="CONSTANT mean-pole (diagnostic)",
                       model_key="grgm660prim", variant="full", nmax=64,
                       diagnostic=True, intentionally_wrong=True, elements=False)

            wr.compare(f"frame_frozen_n{nmax}", f"g660_n{nmax}_frozen", correct,
                       "cost of ignoring lunar rotation entirely",
                       diagnostic=True, intentionally_wrong=True)
            wr.compare(f"frame_wrongpair_n{nmax}", f"g660_n{nmax}_wrongpair",
                       correct,
                       "cost of the DE421<->DE440 PA mismatch alone",
                       diagnostic=True, intentionally_wrong=True)
        wr.compare("frame_meanpole_n64", "g660_n64_meanpole",
                   "g660_n64_correct",
                   "cost of a constant mean-pole frame (13C physical frame "
                   "effect, now with the real model)",
                   diagnostic=True, intentionally_wrong=True)

        stage["windows"][window] = {
            "t_end_s": wr.t_end, "n_epochs": int(wr.teval.size),
            "runs": wr.run_rows, "comparisons": wr.comp_rows,
            "elements": wr.elem_rows,
        }
    return stage


# ---------------------------------------------------------------------------
# Stage: sensitivity (13G-c1 — altitude / inclination / eccentricity axes)
# ---------------------------------------------------------------------------
def run_sensitivity() -> dict:
    plan = build_sensitivity_matrix()
    big660 = load_real_model("grgm660prim", max(LADDER_660))
    big1800 = load_real_model("gl1800f", 256)
    variants = {
        "v2_c20": make_variant(big660, "c20_only"),
        "v3_c20c22": make_variant(big660, "c20_c22"),
        "v4_zonal64": make_variant(big660, "zonal_only", nmax=DECOMP_NMAX),
    }
    ladder_models = {n: truncate_model(big660, n) for n in LADDER_660}
    stage: dict = {"plan_note": "V0/V5 dropped (13G-b proved decomposition "
                   "linearity, coupling ~4%); tesseral is difference-based",
                   "skip_128": None, "cases": {}}
    skip_128 = False

    for spec in SENSITIVITY_CASES:
        case_id, axis = spec["case"], spec["axis"]
        s0 = case_state(spec)
        print(f"[sensitivity] === {case_id} ({axis}) ===")
        case_data: dict = {"spec": {k: v for k, v in spec.items()}, "windows": {}}
        for window in ("orbit", "day1"):
            wr = WindowRunner(window, "sensitivity", s0=s0, case=case_id, axis=axis)
            load_kernel_profile("de421")
            pair421 = wr.pair("MOON_PA_DE421")

            wr.run("v1_j2", j2_moon=J2, model_key="grgm660prim",
                   variant="classical_j2")
            for vname in ("v2_c20", "v3_c20c22", "v4_zonal64"):
                wr.run(vname, model=variants[vname], rotation=pair421,
                       rotation_label="MOON_PA_DE421@60s",
                       model_key="grgm660prim", variant=vname[3:])
            for nmax in LADDER_660:
                if nmax == 128 and skip_128:
                    print(f"  [{window}] v6_full128 SKIPPED (runtime guard)")
                    continue
                res = wr.run(f"v6_full{nmax}", model=ladder_models[nmax],
                             rotation=pair421, rotation_label="MOON_PA_DE421@60s",
                             model_key="grgm660prim", variant="full", nmax=nmax)
                if nmax == 128 and window == "day1" and res["runtime_s"] > 60.0:
                    skip_128 = True
                    stage["skip_128"] = (f"day1 full@128 took {res['runtime_s']:.1f} s "
                                         f"> 60 s at {case_id}; remaining 128 runs skipped")

            peri_mask = None
            if axis == "eccentric" and "v6_full64" in wr.trajs:
                peri_mask = perilune_mask_from(wr.trajs["v6_full64"],
                                               PERILUNE_HALF_WIDTH_DEG)

            def cmp(name, a, b, note):
                wr.compare(name, a, b, note, perilune_mask=peri_mask)

            cmp("j2only_model_error", "v1_j2", "v6_full64",
                "HEADLINE per regime: classical-J2-only vs full real model @64")
            cmp("c20_bridge", "v2_c20", "v1_j2",
                "real C20-only vs classical J2 (coefficient + frame difference)")
            cmp("c22_effect", "v3_c20c22", "v2_c20", "real C22/S22 signal")
            cmp("zonal_beyond_c20", "v4_zonal64", "v2_c20",
                "zonal terms n>=3 contribution")
            cmp("tesseral_contribution", "v6_full64", "v4_zonal64",
                "difference-based tesseral contribution (incl. coupling; "
                "direct-path cross-check done in 13G-b)")
            for lo, hi in zip(LADDER_660[:-1], LADDER_660[1:]):
                if f"v6_full{hi}" in wr.trajs:
                    cmp(f"ladder_{lo}_vs_{hi}", f"v6_full{lo}", f"v6_full{hi}",
                        f"truncation step nmax {lo}->{hi}")

            if plan[case_id]["gl1800f"]:
                load_kernel_profile("de440")
                pair440 = wr.pair("MOON_PA_DE440")
                for nmax in (64, 128):
                    wr.run(f"g1800_full{nmax}", model=truncate_model(big1800, nmax),
                           rotation=pair440, rotation_label="MOON_PA_DE440@60s",
                           model_key="gl1800f", variant="full", nmax=nmax)
                cmp("cross_model_nmax64", "v6_full64", "g1800_full64",
                    "GRGM660PRIM@64 (DE421) vs GL1800F@64 (DE440); coefficient+"
                    "orientation+shared-.mat effects NOT separated")
                cmp("g1800_ladder_64_vs_128", "g1800_full64", "g1800_full128",
                    "GL1800F truncation step 64->128")
                if case_id == "S8_ecc80x500_i45":
                    rt128 = wr.run_rows[-1]["runtime_s"]
                    projected = rt128 * 4.0
                    if projected <= OPT_RUNTIME_GATE_S:
                        wr.run("g1800_full256",
                               model=truncate_model(big1800, 256),
                               rotation=pair440,
                               rotation_label="MOON_PA_DE440@60s",
                               model_key="gl1800f", variant="full", nmax=256)
                        cmp("g1800_ladder_128_vs_256", "g1800_full128",
                            "g1800_full256", "high-degree tail at perilune")
                    else:
                        print(f"  [{window}] g1800_full256 SKIPPED "
                              f"(projected {projected:.0f} s > gate)")

            case_data["windows"][window] = {
                "t_end_s": wr.t_end, "n_epochs": int(wr.teval.size),
                "period_s": round(wr.period_s, 1),
                "runs": wr.run_rows, "comparisons": wr.comp_rows,
                "elements": wr.elem_rows,
            }
        stage["cases"][case_id] = case_data
    return stage


# ---------------------------------------------------------------------------
# Seven-day confirmation (13G-c2): matrix, chunked runner, preflight gate,
# trailing-orbit daily means, growth profile, atomic store writes.
# All results are "selected multi-day confirmation - Level-1 internal
# model-vs-model"; nothing here is estimator performance.
# ---------------------------------------------------------------------------
SEVENDAY_T_END_S = 7.0 * 86400.0            # exactly 604800 s
SEVENDAY_CHUNK_S = 86400.0                  # 7 x 1-day state-handoff chunks
SEVENDAY_OUT_STEP_S = {"L1": 300.0, "L2": 300.0, "L3": 60.0, "L4": 300.0}
SEVENDAY_CASES = (
    {"id": "L1", "case_ref": "S1_alt100_i45", "circular": True,
     "zonal": False, "gl": ()},
    {"id": "L2", "case_ref": "S7_alt100_i90", "circular": True,
     "zonal": True, "gl": (128,)},
    {"id": "L3", "case_ref": "S8_ecc80x500_i45", "circular": False,
     "zonal": True, "gl": (128, 256)},
    {"id": "L4", "case_ref": "S3_alt500_i45", "circular": True,
     "zonal": False, "gl": ()},
)
GL256_GATE_S = 420.0        # rough empirical projection gate, not a guarantee
GROWTH_D1_MIN_M = 1.0       # below this, growth ratios are indeterminate
PREFLIGHT_ABS_GATE_M = 0.05     # per-model continuous-vs-chunked final dpos
PREFLIGHT_REL_GATE = 0.01       # comparison artifact vs measured separation
GROWTH_LABEL = "descriptive classification, not a fitted dynamical law"
CROSSING_LIMITATION = (
    "Surface crossing is detected on the sampled output grid rather than "
    "through a continuous root-finding event. The numerical solver may have "
    "evaluated states beyond the first sampled crossing before the "
    "uninterrupted propagation call returned; those states are excluded "
    "from scientific interpretation."
)


def build_sevenday_matrix() -> dict:
    """Pure seven-day run plan (no data, no SPICE) — the size guard.

    Per case: GRGM660PRIM J2-only + full 32/64/128 (+ zonal@64 on L2/L3);
    GL1800F only where listed (256 only on L3, runtime-gated).  Total <= 21.
    """
    plan = {}
    for spec in SEVENDAY_CASES:
        runs = ["j2_only", "full32", "full64", "full128"]
        if spec["zonal"]:
            runs.append("zonal64")
        runs += [f"g1800_{n}" for n in spec["gl"]]
        plan[spec["id"]] = {
            "case_ref": spec["case_ref"], "circular": spec["circular"],
            "out_step_s": SEVENDAY_OUT_STEP_S[spec["id"]],
            "runs": runs,
        }
    total = sum(len(p["runs"]) for p in plan.values())
    if total > 21:
        raise RuntimeError(f"seven-day matrix exploded: {total} runs > 21.")
    plan["_total_runs"] = total
    return plan


def run_case_chunked(get_earth, get_sun, s0, t_end, out_step, *,
                     j2_moon=0.0, model=None, rotation=None,
                     chunk_s=SEVENDAY_CHUNK_S) -> dict:
    """Chunked state-handoff propagation with sampled-grid crossing policy.

    Retained only as a diagnostic/evidence tool (used by
    ``run_chunk_preflight``); the seven-day campaign itself uses
    ``run_case_continuous_sevenday`` and never calls this function. Each
    invocation of this function applies the same ``chunk_s`` day-boundary
    restart scheme to itself for internal fair comparison. Surface crossing
    is detected on the SAMPLED grid: scientific metrics are truncated at the
    first sampled crossing and that invocation's later chunks are skipped.
    (No claim is made that the solver never evaluated sub-surface states in
    the chunk containing the crossing.)
    """
    ge = _CountingGetter(get_earth)
    t_parts, x_parts, chunk_runtimes = [], [], []
    state = np.asarray(s0, dtype=float)
    status = "complete"
    n_chunks = int(round(t_end / chunk_s))
    t0 = 0.0
    for k in range(n_chunks):
        t1 = min(t0 + chunk_s, t_end)
        teval = np.arange(t0, t1 + out_step / 2.0, out_step)
        tic = time.perf_counter()
        traj = propagate_state(teval, state, MU_M, MU_E, MU_S, ge, get_sun,
                               method="ADAMS", j2_moon=j2_moon,
                               harmonic_model=model,
                               harmonic_rotation=rotation)
        chunk_runtimes.append(round(time.perf_counter() - tic, 3))
        if not np.all(np.isfinite(traj)):
            status = "integrator_failure"
            break
        seg_t = teval if k == 0 else teval[1:]
        seg_x = traj if k == 0 else traj[1:]
        t_parts.append(seg_t)
        x_parts.append(seg_x)
        state = traj[-1]
        t0 = t1
        alt = np.linalg.norm(seg_x[:, :3], axis=1) - R_MOON_M
        if np.any(alt <= 0.0):
            status = "surface_crossing_detected"
            break
    t = np.concatenate(t_parts) if t_parts else np.array([])
    traj = np.vstack(x_parts) if x_parts else np.empty((0, 6))
    result = {"t": t, "traj": traj,
              "runtime_s": float(sum(chunk_runtimes)),
              "chunk_runtimes_s": chunk_runtimes,
              "rhs_evals": int(ge.calls), "status": status,
              "first_crossing_t_s": None, "validity": "valid"}
    if status == "surface_crossing_detected":
        result.update(truncate_at_first_crossing(t, traj))
    if result["traj"].shape[0]:
        radii = np.linalg.norm(result["traj"][:, :3], axis=1)
        result["min_altitude_m"] = float(radii.min() - R_MOON_M)
        result["max_altitude_m"] = float(radii.max() - R_MOON_M)
    return result


def truncate_at_first_crossing(t: np.ndarray, traj: np.ndarray) -> dict:
    """Cut scientific metrics at the FIRST sampled altitude<=0 epoch;
    post-crossing samples are excluded from physical interpretation."""
    alt = np.linalg.norm(traj[:, :3], axis=1) - R_MOON_M
    idx = np.nonzero(alt <= 0.0)[0]
    if idx.size == 0:
        return {"t": t, "traj": traj, "first_crossing_t_s": None,
                "validity": "valid"}
    first = int(idx[0])
    return {"t": t[:first], "traj": traj[:first],
            "first_crossing_t_s": float(t[first]),
            "validity": "invalid_after_first_crossing"}


# -- continuous-vs-chunked numerical preflight (mandatory gate) --------------
def evaluate_chunk_preflight(per_model: dict, comparison: dict) -> dict:
    """Pure gate: chunk restart artifact must be negligible per model AND
    relative to the comparison signal we intend to measure."""
    abs_ok = all(m["final_dpos_m"] <= PREFLIGHT_ABS_GATE_M
                 for m in per_model.values())
    sep = comparison["continuous_final_dpos_m"]
    rel_limit = PREFLIGHT_REL_GATE * sep if sep > 0 else PREFLIGHT_ABS_GATE_M
    rel_ok = comparison["artifact_final_dpos_m"] <= rel_limit
    return {"pass": bool(abs_ok and rel_ok),
            "abs_limit_m": PREFLIGHT_ABS_GATE_M,
            "rel_limit_m": rel_limit,
            "rule": (f"per-model continuous-vs-chunked final dpos <= "
                     f"{PREFLIGHT_ABS_GATE_M} m AND comparison artifact <= "
                     f"{PREFLIGHT_REL_GATE:.0%} of the measured separation"),
            "per_model_pass": abs_ok, "comparison_pass": rel_ok}


def run_chunk_preflight(eph, s0, pair, model128, out_step=300.0) -> dict:
    """Continuous 2-day vs 2x1-day chunked, for J2-only and full@128, on the
    L1/baseline-like state.  Also checks the artifact ON the comparison
    (J2 vs full128) itself.  Campaign must not start if the gate fails."""
    t_end = 2.0 * 86400.0
    teval = np.arange(0.0, t_end + out_step / 2.0, out_step)
    variants = {"j2_only": {"j2_moon": J2},
                "full128": {"harmonic_model": model128,
                            "harmonic_rotation": pair}}
    per_model, trajs = {}, {}
    for name, kw in variants.items():
        cont = run_case(eph.earth_position, eph.sun_position, s0, teval, **kw)
        chunk = run_case_chunked(
            eph.earth_position, eph.sun_position, s0, t_end, out_step,
            j2_moon=kw.get("j2_moon", 0.0), model=kw.get("harmonic_model"),
            rotation=kw.get("harmonic_rotation"))
        if chunk["status"] != "complete":
            raise RuntimeError(f"preflight chunked run failed: {chunk['status']}")
        dp = np.linalg.norm(cont["traj"][:, :3] - chunk["traj"][:, :3], axis=1)
        dv = np.linalg.norm(cont["traj"][:, 3:] - chunk["traj"][:, 3:], axis=1)
        coe_c = rv2coe(cont["traj"][-1, :3], cont["traj"][-1, 3:], MU_M)
        coe_h = rv2coe(chunk["traj"][-1, :3], chunk["traj"][-1, 3:], MU_M)
        per_model[name] = {
            "final_dpos_m": float(dp[-1]), "max_dpos_m": float(dp.max()),
            "final_dvel_mps": float(dv[-1]),
            "final_da_m": abs(coe_c["a_m"] - coe_h["a_m"]),
            "final_de": abs(coe_c["e"] - coe_h["e"]),
            "final_di_rad": abs(coe_c["i_rad"] - coe_h["i_rad"]),
        }
        trajs[name] = (cont["traj"], chunk["traj"])
    dc = np.linalg.norm(trajs["j2_only"][0][:, :3]
                        - trajs["full128"][0][:, :3], axis=1)
    dh = np.linalg.norm(trajs["j2_only"][1][:, :3]
                        - trajs["full128"][1][:, :3], axis=1)
    comparison = {"continuous_final_dpos_m": float(dc[-1]),
                  "chunked_final_dpos_m": float(dh[-1]),
                  "artifact_final_dpos_m": float(abs(dc[-1] - dh[-1])),
                  "artifact_max_dpos_m": float(np.max(np.abs(dc - dh)))}
    gate = evaluate_chunk_preflight(per_model, comparison)
    return {"window_s": t_end, "out_step_s": out_step,
            "per_model": per_model, "comparison": comparison, "gate": gate}


# -- trailing one-orbit daily element means (13G-b methodology preserved) ----
def trailing_window_mask(t: np.ndarray, t_k: float, period_s: float,
                         leading: bool = False) -> np.ndarray:
    if leading:                      # day-0 reference: mean over [0, T]
        return (t >= -1e-9) & (t <= period_s + 1e-9)
    return (t >= t_k - period_s - 1e-9) & (t <= t_k + 1e-9)


def daily_element_rows(t, traj, period_s, *, run: str, case_id: str,
                       circular: bool) -> list[dict]:
    """Per day-boundary trailing ONE-ORBIT means (calendar-day means would
    leak short-period signal as the boundary phase drifts).  Circular cases
    report eccentricity magnitude and nonsingular e*cos/e*sin components
    only — NO argument-of-perilune interpretation; argp evolution is
    reported only for the eccentric case."""
    boundaries = [0.0] + [k * 86400.0 for k in range(1, 8)]
    coes = [rv2coe(s[:3], s[3:], MU_M) for s in traj]
    a = np.array([c["a_m"] for c in coes])
    e = np.array([c["e"] for c in coes])
    inc = np.array([c["i_rad"] for c in coes])
    raan = np.unwrap(np.array([c["raan_rad"] for c in coes]))
    argp = np.array([c["argp_rad"] for c in coes])
    varpi = raan + argp                      # NaN-safe: NaN argp -> NaN varpi
    hcomp = np.where(np.isnan(varpi), 0.0, e * np.cos(varpi))
    kcomp = np.where(np.isnan(varpi), 0.0, e * np.sin(varpi))
    argp_unwrapped = (np.unwrap(argp) if not np.any(np.isnan(argp)) else None)
    rows = []
    for day, t_k in enumerate(boundaries):
        mask = trailing_window_mask(t, t_k, period_s, leading=(day == 0))
        if not mask.any() or t[-1] + 1e-6 < t_k:
            break                                    # truncated run
        row = {"case_id": case_id, "run": run, "day": day,
               "mean_a_m": float(a[mask].mean()),
               "mean_e": float(e[mask].mean()),
               "mean_i_rad": float(inc[mask].mean()),
               "mean_raan_rad": float(raan[mask].mean()),
               "ecc_cos_comp": float(hcomp[mask].mean()),
               "ecc_sin_comp": float(kcomp[mask].mean())}
        if not circular and argp_unwrapped is not None:
            row["mean_argp_rad"] = float(argp_unwrapped[mask].mean())
        rows.append(row)
    if not circular:
        prev = None
        for row in rows:
            cur = row.get("mean_argp_rad")
            row["argp_drift_sign"] = (None if prev is None or cur is None
                                      else int(np.sign(cur - prev)))
            prev = cur
    return rows


def growth_profile(daily_seps: list) -> dict:
    """D_k/(k*D_1) ratios with a near-zero D_1 guard; the raw day-1..7 series
    is the primary scientific output, the classification is secondary."""
    if not daily_seps or daily_seps[0] is None:
        return {"ratios": None, "classification": "indeterminate",
                "label": GROWTH_LABEL}
    d1 = daily_seps[0]
    if not math.isfinite(d1) or d1 < GROWTH_D1_MIN_M:
        return {"ratios": None, "classification": "indeterminate",
                "label": GROWTH_LABEL,
                "note": f"day-1 separation {d1!r} below "
                        f"{GROWTH_D1_MIN_M} m guard"}
    ratios = [round(dk / ((k + 1) * d1), 3) if dk is not None else None
              for k, dk in enumerate(daily_seps)]
    last = next((r for r in reversed(ratios) if r is not None), None)
    if last is None:
        cls = "indeterminate"
    elif 0.8 <= last <= 1.25:
        cls = "approximately linear"
    elif last < 0.8:
        cls = "sublinear"
    else:
        cls = "superlinear"
    finite = [d for d in daily_seps if d is not None]
    if finite and max(finite) / finite[-1] > 1.5:
        cls += " with oscillatory component"
    return {"ratios": ratios, "classification": cls, "label": GROWTH_LABEL}


def gl256_gate(measured_128_runtime_s: float) -> dict:
    projected = 4.0 * measured_128_runtime_s
    run = projected <= GL256_GATE_S
    return {"run": run, "projected_s": round(projected, 1),
            "limit_s": GL256_GATE_S,
            "basis": "rough empirical projection (4 x measured GL@128 "
                     "seven-day runtime); not a guarantee",
            "reason": None if run else
            f"projected {projected:.0f} s exceeds {GL256_GATE_S:.0f} s gate"}


# -- continuous seven-day runner (Option A, approved after preflight FAIL) ---
def run_case_continuous_sevenday(get_earth, get_sun, s0, t_end, out_step, *,
                                 j2_moon=0.0, model=None, rotation=None) -> dict:
    """ONE uninterrupted ``propagate_state`` call per model case.

    Wraps the validated ``run_case`` (no behavioral copy).  Surface crossing
    keeps the sampled-output-grid policy: the FULL continuous propagation
    completes first, the first crossing sample is found post-hoc, and the
    post-processing arrays are truncated there.  No day/chunk restarts: the
    chunked methodology was rejected by the continuous-vs-chunked preflight
    (multistep integrator restart artifact)."""
    teval = np.arange(0.0, t_end + out_step / 2.0, out_step)
    try:
        res = run_case(get_earth, get_sun, s0, teval, j2_moon=j2_moon,
                       harmonic_model=model, harmonic_rotation=rotation)
    except RuntimeError as exc:
        return {"t": np.array([]), "traj": np.empty((0, 6)),
                "runtime_s": float("nan"), "rhs_evals": 0,
                "status": "integrator_failure", "validity": "invalid",
                "error": str(exc), "crossing_detected": False,
                "crossing_index": None, "crossing_time_s": None,
                "first_crossing_t_s": None, "last_valid_time_s": None,
                "requested_samples": int(teval.size), "returned_samples": 0,
                "valid_duration_s": 0.0, "propagation_calls": 1}
    cut = truncate_at_first_crossing(teval, res["traj"])
    crossing = cut["validity"] != "valid"
    t_valid, traj_valid = cut["t"], cut["traj"]
    out = {
        "t": t_valid, "traj": traj_valid,
        "runtime_s": float(res["runtime_s"]), "rhs_evals": int(res["rhs_evals"]),
        "status": "surface_crossing_detected" if crossing else "complete",
        "validity": cut["validity"],
        "crossing_detected": bool(crossing),
        "crossing_index": (int(t_valid.size) if crossing else None),
        "crossing_time_s": cut["first_crossing_t_s"],
        "first_crossing_t_s": cut["first_crossing_t_s"],
        "last_valid_time_s": (float(t_valid[-1]) if t_valid.size else None),
        "requested_samples": int(teval.size),
        "returned_samples": int(t_valid.size),
        "valid_duration_s": (float(t_valid[-1]) if t_valid.size else 0.0),
        "propagation_calls": 1,
    }
    if traj_valid.shape[0]:
        radii = np.linalg.norm(traj_valid[:, :3], axis=1)
        out["min_altitude_m"] = float(radii.min() - R_MOON_M)
        out["max_altitude_m"] = float(radii.max() - R_MOON_M)
        out["minimum_radius_margin_before_crossing_m"] = (
            out["min_altitude_m"] if crossing else None)
    return out


def sevenday_daily_mask(t: np.ndarray, day: int) -> np.ndarray:
    """Samples belonging to calendar day ``day`` (1..7) on the shared output
    grid: half-open (lo, hi] so a boundary sample counts in ONE day only."""
    return (t > (day - 1) * 86400.0) & (t <= day * 86400.0)


def sevenday_comparison_row(name, run_a, run_b, res_a, res_b, *,
                            case_id: str, requested_duration_s: float,
                            day_boundaries, perilune: bool = False) -> dict:
    """Pure comparison over the MINIMUM COMMON VALID horizon: a model's
    post-crossing trajectory is never compared against another model."""
    n = min(res_a["traj"].shape[0], res_b["traj"].shape[0])
    dur_a = res_a.get("valid_duration_s",
                      float(res_a["t"][-1]) if len(res_a["t"]) else 0.0)
    dur_b = res_b.get("valid_duration_s",
                      float(res_b["t"][-1]) if len(res_b["t"]) else 0.0)
    if n == 0:
        return {"case_id": case_id, "comparison": name, "run": run_a,
                "reference": run_b, "status": "no_valid_overlap",
                "requested_duration_s": requested_duration_s,
                "run_a_valid_duration_s": dur_a,
                "run_b_valid_duration_s": dur_b,
                "common_comparison_duration_s": 0.0,
                "truncation_reason": "no_valid_overlap"}
    ta = res_a["t"][:n]
    dp = np.linalg.norm(res_a["traj"][:n, :3] - res_b["traj"][:n, :3], axis=1)
    dv = np.linalg.norm(res_a["traj"][:n, 3:] - res_b["traj"][:n, 3:], axis=1)
    rtn = rtn_summary(res_a["traj"][:n], res_b["traj"][:n])
    daily = []
    for t_k in day_boundaries:
        idx = np.nonzero(np.isclose(ta, t_k))[0]
        daily.append(float(dp[idx[0]]) if idx.size else None)
    truncated = (res_a.get("validity", "valid") != "valid"
                 or res_b.get("validity", "valid") != "valid")
    row = {"case_id": case_id, "comparison": name, "run": run_a,
           "reference": run_b,
           "status": "truncated" if truncated else "complete",
           "final_dpos_m": float(dp[-1]), "max_dpos_m": float(dp.max()),
           "rms_dpos_m": float(np.sqrt(np.mean(dp ** 2))),
           "final_dvel_mps": float(dv[-1]), **rtn,
           "daily_endpoint_dpos_m": daily,
           "growth": growth_profile(daily),
           "requested_duration_s": requested_duration_s,
           "run_a_valid_duration_s": dur_a,
           "run_b_valid_duration_s": dur_b,
           "common_comparison_duration_s": float(ta[-1]),
           "truncation_reason": ("surface_crossing" if truncated else None)}
    if perilune:
        mask = perilune_mask_from(res_b["traj"][:n], PERILUNE_HALF_WIDTH_DEG)
        if mask.any():
            row["perilune_rms_dpos_m"] = float(np.sqrt(np.mean(dp[mask] ** 2)))
            daily_peri = []
            for day in range(1, 8):
                dmask = mask & sevenday_daily_mask(ta, day)
                daily_peri.append(float(np.sqrt(np.mean(dp[dmask] ** 2)))
                                  if dmask.any() else None)
            row["daily_perilune_rms_dpos_m"] = daily_peri
    return row


def preflight_decision_block(preflight: dict) -> dict:
    """Preserve the continuous-vs-chunked preflight as METHODOLOGY EVIDENCE.

    The chunked methodology was rejected by this preflight; regardless of the
    gate outcome the campaign decision is fixed to continuous propagation."""
    gate = preflight.get("gate", {})
    return {**preflight,
            "status": "passed" if gate.get("pass") else "failed",
            "decision": "use_continuous_campaign_propagation",
            "reason": "multistep_integrator_restart_artifact"}


# -- atomic store persistence -------------------------------------------------
def _atomic_store_dump(store: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(store, indent=2)      # serialize BEFORE touching disk
    tmp = STORE_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, STORE_PATH)


def _init_sevenday_stage(store: dict) -> dict:
    """Fresh sevenday stage; any stale partial data is REPLACED (no --resume:
    the safest simple behavior is a clean rebuild, never silent mixing)."""
    stage = {
        "status": "partial",
        "planned_cases": [c["id"] for c in SEVENDAY_CASES],
        "completed_cases": [],
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "campaign_command": "python examples/phase13g_gravity_orbit_effects.py --sevenday",
        "propagation_contract": ("one uninterrupted continuous propagate_state "
                                 "call per model case (no day/chunk restarts); "
                                 "surface crossing handled post-hoc on the "
                                 "sampled output grid"),
        "out_step_s": dict(SEVENDAY_OUT_STEP_S),
        "cadence_s": CADENCE_S,
        "evidence_label": ("selected multi-day confirmation - Level-1 "
                           "internal model-vs-model; not estimator performance"),
        "chunk_preflight": None,
        "gl256_gate": None,
        "cases": {},
    }
    store["sevenday"] = stage
    return stage


def _sevenday_pair(et0: float, frame: str):
    margin = max(2.0 * CADENCE_S, 120.0)
    t_grid = np.arange(-margin, SEVENDAY_T_END_S + margin + CADENCE_S / 2.0,
                       CADENCE_S)
    from lunar_od.lunar_frames import sample_moon_pa_rotations
    rots = sample_moon_pa_rotations(et0, t_grid, frame=frame,
                                    load_kernels=False)
    return (t_grid, rots)


def run_sevenday(store: dict) -> dict:
    matrix = build_sevenday_matrix()
    stage = _init_sevenday_stage(store)
    stage["matrix_total_runs"] = matrix["_total_runs"]
    _atomic_store_dump(store)

    big660 = load_real_model("grgm660prim", 128)
    need_gl = any(spec["gl"] for spec in SEVENDAY_CASES)
    big1800 = load_real_model("gl1800f", 256) if need_gl else None
    zonal64 = make_variant(big660, "zonal_only", nmax=DECOMP_NMAX)

    eph, first_jd = load_ephemeris(SEVENDAY_T_END_S + 600.0)
    et0 = (first_jd - J2000_JD) * 86400.0

    load_kernel_profile("de421")
    pair421 = _sevenday_pair(et0, "MOON_PA_DE421")
    if need_gl:
        load_kernel_profile("de440")
        pair440 = _sevenday_pair(et0, "MOON_PA_DE440")

    # -- continuous-vs-chunked preflight: retained as METHODOLOGY EVIDENCE ---
    # The chunked methodology was rejected by this preflight (multistep
    # integrator restart artifact); the campaign therefore uses uninterrupted
    # continuous propagation for every scientific case, regardless of the
    # gate outcome recorded here.
    s0_pref = case_state(next(s for s in SENSITIVITY_CASES
                              if s["case"] == "S1_alt100_i45"))
    print("[sevenday] continuous-vs-chunked preflight (evidence record)")
    preflight = preflight_decision_block(
        run_chunk_preflight(eph, s0_pref, pair421,
                            truncate_model(big660, 128)))
    stage["chunk_preflight"] = preflight
    _atomic_store_dump(store)
    g = preflight["gate"]
    print(f"[sevenday] preflight: {preflight['status'].upper()} "
          f"(per-model {g['per_model_pass']}, comparison "
          f"{g['comparison_pass']}) -> decision: {preflight['decision']}")

    # -- pre-campaign configuration report -----------------------------------
    total_runs = matrix["_total_runs"]
    print(f"[sevenday] model cases: {len(stage['planned_cases'])} "
          f"(runs <= {total_runs}); one continuous propagation per run, "
          f"duration {SEVENDAY_T_END_S:.0f} s each")
    for cid_, step_ in SEVENDAY_OUT_STEP_S.items():
        print(f"[sevenday]   {cid_}: out_step {step_:.0f} s -> "
              f"{int(SEVENDAY_T_END_S / step_) + 1} samples")
    print("[sevenday] surface-crossing policy: sampled-output-grid detection, "
          "post-hoc truncation at the first crossing sample")
    print(f"[sevenday] artifacts: {OUT} (atomic per-case store writes)")
    print("[sevenday] resume policy: no mid-trajectory resume; an interrupted "
          "model case reruns continuous from scratch; completed case "
          "artifacts are preserved")

    day_boundaries = [k * 86400.0 for k in range(1, 8)]
    for spec in SEVENDAY_CASES:
        cid = spec["id"]
        case_ref = spec["case_ref"]
        out_step = SEVENDAY_OUT_STEP_S[cid]
        s0 = case_state(next(s for s in SENSITIVITY_CASES
                             if s["case"] == case_ref))
        period = 2.0 * math.pi * math.sqrt(
            rv2coe(s0[:3], s0[3:], MU_M)["a_m"] ** 3 / MU_M)
        print(f"[sevenday] === {cid} ({case_ref}) out_step={out_step:.0f} s ===")

        def do_run(name, **kw):
            res = run_case_continuous_sevenday(
                eph.earth_position, eph.sun_position, s0,
                SEVENDAY_T_END_S, out_step, **kw)
            print(f"  {cid} {name:10s} status={res['status']:<26s} "
                  f"rt {res['runtime_s']:7.1f} s  rhs {res['rhs_evals']:7d}  "
                  f"samples {res['returned_samples']}/{res['requested_samples']}  "
                  f"min_alt {res.get('min_altitude_m', float('nan'))/1e3:7.2f} km")
            return res

        results = {"j2_only": do_run("j2_only", j2_moon=J2)}
        for n in (32, 64, 128):
            results[f"full{n}"] = do_run(
                f"full{n}", model=truncate_model(big660, n), rotation=pair421)
        if spec["zonal"]:
            results["zonal64"] = do_run("zonal64", model=zonal64,
                                        rotation=pair421)
        gl_runs = list(spec["gl"])
        if 128 in gl_runs:
            results["g1800_128"] = do_run(
                "g1800_128", model=truncate_model(big1800, 128),
                rotation=pair440)
        if 256 in gl_runs:
            gate = gl256_gate(results["g1800_128"]["runtime_s"])
            stage["gl256_gate"] = gate
            if gate["run"]:
                results["g1800_256"] = do_run(
                    "g1800_256", model=truncate_model(big1800, 256),
                    rotation=pair440)
            else:
                print(f"  {cid} g1800_256 SKIPPED: {gate['reason']}")

        run_rows, comp_rows, elem_rows, daily_rows = [], [], [], []
        for name, res in results.items():
            run_rows.append({
                "case_id": cid, "run": name, "status": res["status"],
                "validity": res["validity"],
                "crossing_detected": res.get("crossing_detected", False),
                "crossing_index": res.get("crossing_index"),
                "crossing_time_s": res.get("crossing_time_s"),
                "last_valid_time_s": res.get("last_valid_time_s"),
                "minimum_radius_margin_before_crossing_m":
                    res.get("minimum_radius_margin_before_crossing_m"),
                "requested_samples": res.get("requested_samples"),
                "returned_samples": res.get("returned_samples"),
                "valid_duration_s": res.get("valid_duration_s"),
                "propagation_calls": res.get("propagation_calls", 1),
                "runtime_s": round(res["runtime_s"], 1),
                "rhs_evals": res["rhs_evals"],
                "min_altitude_m": round(res.get("min_altitude_m", float("nan")), 1),
                "max_altitude_m": round(res.get("max_altitude_m", float("nan")), 1),
                "out_step_s": out_step,
            })

        def compare(name, run_a, run_b, perilune=False):
            row = sevenday_comparison_row(
                name, run_a, run_b, results[run_a], results[run_b],
                case_id=cid, requested_duration_s=SEVENDAY_T_END_S,
                day_boundaries=day_boundaries, perilune=perilune)
            comp_rows.append(row)
            if "final_dpos_m" not in row:
                print(f"  {cid} {name}: {row['status']}")
                return
            daily = row["daily_endpoint_dpos_m"]
            print(f"  {cid} {name}: final {row['final_dpos_m']:.3e} m  "
                  f"daily {['%.0f' % d if d else '-' for d in daily]}  "
                  f"[{row['growth']['classification']}]")

        compare("j2only_vs_full128", "j2_only", "full128",
                perilune=(cid == "L3"))
        compare("ladder_32_vs_64", "full32", "full64")
        compare("ladder_64_vs_128", "full64", "full128",
                perilune=(cid == "L3"))
        if spec["zonal"]:
            compare("tesseral_contribution", "full128", "zonal64")
        if "g1800_128" in results:
            compare("cross_model_128", "full128", "g1800_128")
        if "g1800_256" in results:
            compare("g1800_ladder_128_vs_256", "g1800_128", "g1800_256")

        for name in ("j2_only", "full128"):
            res = results[name]
            elem_rows.extend(daily_element_rows(
                res["t"], res["traj"], period, run=name, case_id=cid,
                circular=spec["circular"]))
        for res_name, res in results.items():
            alt = np.linalg.norm(res["traj"][:, :3], axis=1) - R_MOON_M
            for k in range(1, 8):
                dmask = sevenday_daily_mask(res["t"], k)
                if not dmask.any():
                    break
                daily_rows.append({
                    "case_id": cid, "run": res_name, "day": k,
                    "min_altitude_m": float(alt[dmask].min()),
                    "max_altitude_m": float(alt[dmask].max())})

        stage["cases"][cid] = {
            "case_ref": case_ref, "circular": spec["circular"],
            "out_step_s": out_step, "t_end_s": SEVENDAY_T_END_S,
            "period_s": round(period, 1),
            "runs": run_rows, "comparisons": comp_rows,
            "elements": elem_rows, "daily": daily_rows,
        }
        stage["completed_cases"].append(cid)
        _atomic_store_dump(store)

    stage["status"] = "complete"
    _atomic_store_dump(store)
    return stage


# ---------------------------------------------------------------------------
# Outputs (cumulative store; CSV/MD regenerated from the merged store)
# ---------------------------------------------------------------------------
def _load_store() -> dict:
    if STORE_PATH.exists():
        try:
            return json.loads(STORE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _rows(store: dict, kind: str) -> list[dict]:
    rows = []
    for stage in ("baseline", "compare", "frames"):
        data = store.get(stage)
        if not data:
            continue
        for wdata in data["windows"].values():
            rows.extend(wdata[kind])
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[out] wrote {path}")


def _sensitivity_rows(store: dict, kind: str) -> list[dict]:
    data = store.get("sensitivity")
    if not data:
        return []
    rows = []
    for case_data in data["cases"].values():
        for wdata in case_data["windows"].values():
            rows.extend(wdata[kind])
    return rows


_NMAX_SUFFICIENT_M_PER_DAY = 10.0     # ladder-step threshold used in Section 10


def _day1_comp(case_data: dict, name: str) -> dict | None:
    for c in case_data["windows"].get("day1", {}).get("comparisons", []):
        if c["comparison"] == name:
            return c
    return None


def _recommended_nmax(case_data: dict) -> str:
    """Smallest ladder N whose step to 2N is below the threshold; the step
    full@2N - full@N is the truncation-error proxy of full@N."""
    for lo, hi in zip(LADDER_660[:-1], LADDER_660[1:]):
        c = _day1_comp(case_data, f"ladder_{lo}_vs_{hi}")
        if c is not None and c["final_dpos_m"] < _NMAX_SUFFICIENT_M_PER_DAY:
            return str(lo)
    return str(LADDER_660[-1])


def _write_sensitivity_md(store: dict) -> None:
    data = store.get("sensitivity")
    if not data:
        return
    lines = [
        "# Phase 13G-c1 — Gravity Sensitivity Campaign",
        "",
        f"- store updated (UTC): {store.get('generated_utc')}",
        f"- {TRUTH_NOTE}",
        f"- plan: {data['plan_note']}",
        f"- 128-run guard: {data['skip_128'] or 'never triggered'}",
        "",
        "## 0. Model strategy — why GRGM660PRIM everywhere, GL1800F only on "
        "selected cases",
        "",
        "GL1800F was NOT left out: it ran on the selected comparison cases "
        "S1 (100 km circular i=45), S7 (polar 100 km) and S8 (eccentric "
        "80x500 km), at nmax 64/128 (256 only on S8, runtime-gated).  It was "
        "deliberately NOT propagated through the whole sensitivity matrix:",
        "",
        "1. GRGM660PRIM is the primary science model because it is "
        "frame-EXACT with the project's default DE421 PA chain "
        "(MOON_PA_DE421), directly comparable with every Phase 12B/13G-b "
        "number, degree 660 is far beyond any truncation used here, and it "
        "keeps the campaign matrix and runtime controlled.",
        "2. GL1800F is the high-resolution cross-check: frame-exact with "
        "DE440 PA (MOON_PA_DE440), an independent JPL solution at higher "
        "degree — it tests whether the primary conclusions are "
        "model-dependent.",
        "3. Repeating the full orbit x model x nmax matrix with GL1800F "
        "would not change any conclusion: 13G-b measured 660PM@64 vs "
        "GL1800F@64 at the metres-per-day scale, and the selected 13G-c1 "
        "cases give 0.15-2.2 m/day — orders of magnitude below the J2-only "
        "error, the C22 effect, the tesseral contribution and the nmax "
        "convergence steps studied here.  Duplicating the matrix would only "
        "add runtime and report complexity.",
        "4. Role summary: GL1800F is NOT the primary model; it is the "
        "high-resolution sanity / cross-model validation model, applied on "
        "the hardest selected cases.",
        "",
        "## 1. Campaign design — what and why",
        "",
        "- Axes: altitude (100/200/500 km), inclination (0/30/60/90 deg) and "
        "eccentricity (80x500 km) are the three orbit parameters that "
        "control how a gravity field is SAMPLED: altitude sets the "
        "(R_ref/r)^n attenuation of degree-n terms, inclination sets which "
        "latitudes/longitudes are swept, eccentricity concentrates the "
        "sampling near perilune.",
        "- One-factor-at-a-time instead of a cross product: each axis is "
        "varied around the common anchor (100 km circular, i=45) so every "
        "observed change attributes to ONE parameter; a full cross product "
        "(3x4x2 orbits x 9 models x 2 windows) would multiply runtime and "
        "blur attribution without adding decisions.",
        "- 8 cases suffice for the DECISIONS this phase feeds (nmax per "
        "regime, estimator-model floor, 7-day shortlist); finer grids can "
        "be added later exactly where these 8 show gradients.",
        "- 7-day windows are deliberately excluded (13G-c2, separate "
        "approval): 1 orbit isolates the geometric signal, 1 day shows the "
        "secular accumulation; 7 days multiplies runtime and belongs to the "
        "selected shortlist this campaign produces.",
        "",
        "## 2. Orbit case construction",
        "",
        "- Circular cases: target altitude h gives the semi-major axis "
        "a = R_MOON + h (R_MOON = 1 737 400 m); e = 0 exactly, so the "
        "radius equals a on the whole orbit and 'altitude' is a single "
        "number; the inclination is set directly in coe2rv.  RAAN=30 deg, "
        "argp=20 deg, nu=10 deg are held at the baseline-style values so "
        "cases differ ONLY in the studied parameter.",
        "- Eccentric S8: r_p = R_MOON + 80 km, r_a = R_MOON + 500 km, "
        "a = (r_p + r_a)/2  (= R_MOON + 290 km), "
        "e = (r_a - r_p)/(r_a + r_p) (~0.1036).  This orbit was chosen "
        "because degree-n terms scale as (R_ref/r)^(n+2) in acceleration: "
        "a spacecraft that dips to 80 km feels the high-degree field "
        "strongly near perilune and almost not at all near 500 km apolune "
        "— the classic eccentric-orbit sampling question.",
        "- The 'orbit' window uses each case's OWN period "
        "T = 2*pi*sqrt(a^3/mu) (e.g. ~9498 s at 500 km vs ~7068 s at "
        "100 km); reusing the baseline period would compare unequal "
        "fractions of a revolution.",
        "",
        "## 3. Model variants — what each one isolates",
        "",
        "- V1 classical J2-only (j2_moon=J2, mean-pole frame): the simplest "
        "estimator candidate and the historical reference; everything else "
        "is measured against or beyond it.",
        "- V2 real C20-only (only Cbar20 kept, mmax=0): the real model's J2 "
        "equivalent (Cbar20 = -J2/sqrt(5) < 0, fully-normalized sign "
        "convention).  Exists to BRIDGE the classical and harmonic worlds: "
        "if masks/signs/frames were wrong, c20_bridge would explode.",
        "- V3 real C20+C22 (degree-2 m=0 and m=2 terms): C22 is the largest "
        "non-zonal lunar coefficient (~3.47e-5, comparable to C20/3!) — "
        "the first longitude-dependent (sectoral) signal.",
        "- V4 zonal-only@64 (all m>0 zeroed, mmax=0): "
        "longitude-INdependent field; drives the classical secular "
        "perigee/eccentricity/node evolution, so it is the right reference "
        "for splitting rotating-longitude effects from axisymmetric ones.",
        "- Full ladder nmax = 8/16/32/64/128: nmax is the truncation degree "
        "of the spherical-harmonic sum; doubling steps give a clean "
        "truncation-error proxy (Section 4).  256 only on S8 with GL1800F "
        "behind a runtime gate (measured 128-runtime x 4 <= 300 s), because "
        "Pines cost per RHS call grows ~nmax^2.",
        "- Tesseral contribution is DIFFERENCE-BASED (full@64 - zonal@64) "
        "instead of a direct tesseral-only run: 13G-b ran BOTH paths and "
        "showed they agree to ~4% (nonlinear coupling), so re-proving "
        "linearity per orbit would cost 2 runs/case without new "
        "information.",
        "",
        "## 4. Comparison definitions (what each row computes)",
        "",
        "- j2only_model_error = |traj(J2-only) - traj(full@64)|: the error a "
        "J2-only ESTIMATOR model accumulates against the real field — the "
        "headline number per regime.",
        "- c20_bridge = |traj(C20-only) - traj(J2-only)|: internal "
        "consistency check (GRAIL-vs-constants C20 ~0.12% + PA-vs-mean-pole "
        "frame difference); must stay small and did.",
        "- c22_effect = |traj(C20+C22) - traj(C20-only)|: the isolated "
        "C22/S22 signal.",
        "- zonal_beyond_c20 = |traj(zonal@64) - traj(C20-only)|: the "
        "C30/C40/... higher-zonal contribution.",
        "- tesseral_contribution = |traj(full@64) - traj(zonal@64)|: all "
        "longitude-dependent terms (plus coupling).",
        "- ladder_N_vs_2N = |traj(full@N) - traj(full@2N)|: truncation-error "
        "proxy of full@N; used for the nmax-sufficiency decision.",
        "",
        "## 5. Metrics — how each number is computed and why it exists",
        "",
        "- Cartesian final/max/RMS position difference and final velocity "
        "difference: raw trajectory divergence over the window.",
        "- RTN decomposition (R radial, T along-track, N cross-track, from "
        "the reference trajectory's r, r x v axes): separates along-track "
        "PHASE drift (period/energy differences accumulate here and "
        "dominate final dpos) from genuine geometric offsets; without it a "
        "large 'final dpos' can be misread.",
        "- Osculating elements via a script-local rv2coe: a (energy), e "
        "(shape), i (plane tilt), RAAN (node), argp (perilune direction), "
        "u/true longitude as singularity fallbacks.  Circular cases: "
        "argp/nu are undefined -> NaN by policy, and the e-series itself "
        "(growth from 0) is the signal; equatorial cases: RAAN/u undefined "
        "-> NaN, true longitude remains.",
        "- Drift estimation is ORBIT-AVERAGED (mean over the first orbital "
        "period vs mean over the last, divided by the time between them), "
        "NOT a plain linear fit: osculating elements oscillate with "
        "short-period amplitudes far above their secular slopes, and "
        "t*sin(wt) does not integrate to zero even over whole periods, so "
        "a least-squares line LEAKS the oscillation into the slope (found "
        "and fixed via a failing test in 13G-b).  short_period_amp is "
        "reported next to every drift for exactly this reason.",
        "- Safety: min/max altitude, surface-crossing flag and a finite "
        "check on every trajectory — a comparison between two unphysical "
        "trajectories would be meaningless.",
        "- Runtime: wall time and RHS-evaluation count per run; the runtime "
        "guard (skip rule + 256 gate) exists so a single expensive case "
        "cannot silently blow the <15-20 min campaign budget.",
        "",
    ]

    # ---- data-driven summary + interpretation --------------------------------
    alt_cases = [c for c in data["cases"] if data["cases"][c]["spec"]["axis"] == "altitude"]
    inc_cases = [c for c in data["cases"] if data["cases"][c]["spec"]["axis"] == "inclination"]
    ecc_cases = [c for c in data["cases"] if data["cases"][c]["spec"]["axis"] == "eccentric"]

    def _summary_table(case_ids):
        rows = ["| case | j2only err m/day | c22 m/day | tesseral m/day "
                "| 64->128 m/day | sufficient nmax |",
                "|---|---|---|---|---|---|"]
        for cid in case_ids:
            cd = data["cases"][cid]
            j2c = _day1_comp(cd, "j2only_model_error")
            c22 = _day1_comp(cd, "c22_effect")
            tes = _day1_comp(cd, "tesseral_contribution")
            l128 = _day1_comp(cd, "ladder_64_vs_128")
            rows.append(
                f"| {cid} | {j2c['final_dpos_m']:.3e} | {c22['final_dpos_m']:.3e} "
                f"| {tes['final_dpos_m']:.3e} "
                f"| {l128['final_dpos_m']:.3e} | {_recommended_nmax(cd)} |"
            )
        return rows

    lines += ["## 6. Altitude sensitivity — results and interpretation", ""]
    lines += _summary_table(alt_cases)
    lines += [
        "",
        f"- 'sufficient nmax' = smallest N whose N->2N ladder step is below "
        f"{_NMAX_SUFFICIENT_M_PER_DAY:.0f} m/day (truncation-error proxy); "
        "where 128 is quoted, the next step (128->256) was measured small "
        "elsewhere (2.2 m/day baseline, 0.8 m/day S8).",
        "- The nmax NEED falls sharply with altitude (128 at 100 km, 64 at "
        "200 km, 32 at 500 km): degree-n accelerations scale as "
        "(R_ref/r)^(n+2), so high degrees die off fastest.",
        "- OBSERVED IN THIS CAMPAIGN (not claimed as a law): the J2-only "
        "error is NOT monotonic in altitude — 200 km came out slightly "
        "ABOVE 100 km (15.1 vs 13.5 km/day) before dropping strongly at "
        "500 km (6.4 km/day).  Plausible mechanisms: the error is an "
        "ACCUMULATED along-track phase effect, so it depends not only on "
        "the field amplitude but on how the orbital period beats against "
        "the rotating body-fixed field (tesseral sampling geometry / "
        "near-resonance of the m-daily terms); a weaker field integrated "
        "with a different phase history can accumulate more displacement.",
        "",
        "## 7. Inclination sensitivity — results and interpretation", "",
    ]
    lines += _summary_table(inc_cases)
    zon_spike = _day1_comp(data["cases"][inc_cases[1]], "zonal_beyond_c20") if len(inc_cases) > 1 else None
    lines += [
        "",
        "- POLAR is the hardest regime (j2only error ~52.9 km/day, C22 "
        "~24.1 km/day, tesseral ~50.1 km/day): a polar orbit sweeps every "
        "longitude band each revolution while the Moon rotates underneath, "
        "so it samples the full tesseral/sectoral spectrum coherently; "
        "nothing averages out.",
        "- The equatorial case concentrates its sensitivity in SECTORAL "
        "(n=m) terms — it never leaves the equator band — and showed ~2x "
        "the RHS evaluations of the other cases (integrator stepping "
        "against the strong longitude-periodic forcing).",
        (f"- i=30 deg shows a zonal-beyond-C20 spike "
         f"({zon_spike['final_dpos_m']:.3e} m/day vs hundreds at other "
         "inclinations): consistent with odd-zonal (C30-driven) "
         "perigee/eccentricity coupling being geometry-dependent; flagged "
         "for a closer look in 13G-d, not over-interpreted here."
         if zon_spike else ""),
        "",
        "## 8. Eccentric case — results and interpretation", "",
    ]
    lines += _summary_table(ecc_cases)
    s8 = data["cases"][ecc_cases[0]]
    s8_j2 = _day1_comp(s8, "j2only_model_error")
    s8_l128 = _day1_comp(s8, "ladder_64_vs_128")
    lines += [
        "",
        "- High-degree terms are strongest near perilune ((R_ref/r)^(n+2) "
        "attenuation), which is why an 80 km-perilune orbit was expected "
        "to be demanding.  Yet the 64->128 step is only "
        f"{s8_l128['final_dpos_m']:.1f} m/day — ~35x SMALLER than the "
        "100 km circular case: the spacecraft spends most of each "
        "revolution near 500 km apolune where those terms are negligible, "
        "so the TIME-INTEGRATED high-degree effect is small.",
        "- The perilune-window metric (|nu| < 30 deg, measured on the "
        "reference full@64 trajectory) exists to separate the LOCAL effect "
        "from the window average: ratios ~1.17 on the high-degree ladder "
        "steps show the error is indeed elevated near perilune, while the "
        f"J2-only comparison sits at ~0.84 (its accumulated error is "
        "along-track-drift dominated, not perilune-localized). "
        f"J2-only headline here: {s8_j2['final_dpos_m']:.3e} m/day with "
        f"perilune RMS {s8_j2.get('perilune_rms_dpos_m', float('nan')):.3e} m.",
        "",
        "## 9. Cross-model check (GL1800F) — interpretation", "",
        "- Selected-case cross-model differences (660PM@64/DE421 vs "
        "GL1800F@64/DE440): 0.15-2.2 m/day across S1/S7/S8 — metres, while "
        "every physical effect studied here is kilometres.  The primary "
        "conclusions are therefore not artifacts of the model choice.",
        "- This is also why GL1800F was not propagated through the whole "
        "matrix (Section 0): the cross-check answers the model-dependence "
        "question at a tiny fraction of the cost.",
        "",
        "## 10. Preliminary model recommendations (final call in 13G-d)", "",
        "- Truth model: nmax=128 remains strong for 100 km / polar / low "
        "LLO (64->128 steps 39-86 m/day there; 128->256 measured at the "
        "metre scale); nmax=64 suffices at 200 km; nmax=32 at 500 km.",
        "- Estimator model: classical J2-only is inadequate in EVERY tested "
        "regime (6-53 km/day errors) and gets the perigee-drift sign wrong "
        "(13G-b); a minimum low-degree harmonic model is required — the "
        "exact floor (likely nmax ~32-64, regime-dependent) is fixed in "
        "13G-d together with the 7-day evidence.",
        "- 7-day shortlist candidates from this campaign: S1 (anchor), "
        "S7 (hardest), S8 (eccentric), plus S3 as a cheap high-altitude "
        "control.",
        "",
        "## Appendix — per-case comparison tables",
        "",
        "- singular-element policy: circular cases report argp/nu as NaN "
        "(e-growth stays meaningful); equatorial cases report RAAN/u as NaN.",
        "",
    ]
    for case_id, case_data in data["cases"].items():
        spec = case_data["spec"]
        lines += [f"### {case_id} ({spec.get('axis')})", ""]
        for wname, wdata in case_data["windows"].items():
            lines += [
                f"#### window: {wname} (t_end {wdata['t_end_s']:.0f} s, "
                f"period {wdata['period_s']:.0f} s)",
                "",
                "| comparison | final dpos m | R m | T m | N m | rms dpos m "
                "| perilune rms m |",
                "|---|---|---|---|---|---|---|",
            ]
            for c in wdata["comparisons"]:
                peri = (f"{c['perilune_rms_dpos_m']:.3e}"
                        if "perilune_rms_dpos_m" in c else "-")
                lines.append(
                    f"| {c['comparison']} | {c['final_dpos_m']:.3e} "
                    f"| {c['final_radial_m']:+.2e} | {c['final_along_m']:+.2e} "
                    f"| {c['final_cross_m']:+.2e} | {c['rms_dpos_m']:.3e} "
                    f"| {peri} |"
                )
            lines.append("")
    SENS_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[out] wrote {SENS_MD_PATH}")


def _sevenday_rows(store: dict, kind: str) -> list[dict]:
    data = store.get("sevenday")
    if not data:
        return []
    rows = []
    for case_data in data.get("cases", {}).values():
        for row in case_data.get(kind, []):
            flat = dict(row)
            for key in ("daily_endpoint_dpos_m", "daily_perilune_rms_dpos_m"):
                if key in flat:
                    flat[key] = json.dumps(flat[key])
            if isinstance(flat.get("growth"), dict):
                flat["growth_classification"] = flat["growth"]["classification"]
                flat["growth_ratios"] = json.dumps(flat["growth"].get("ratios"))
                del flat["growth"]
            rows.append(flat)
    return rows


def _write_sevenday_md(store: dict) -> None:
    data = store.get("sevenday")
    if not data:
        return
    pf = data.get("chunk_preflight") or {}
    gate = pf.get("gate", {})
    any_crossing = any(r.get("status") == "surface_crossing_detected"
                       for c in data.get("cases", {}).values()
                       for r in c.get("runs", []))
    lines = [
        "# Phase 13G-c2 — Selected Seven-Day Gravity Confirmation",
        "",
        f"- store updated (UTC): {store.get('generated_utc')}",
        f"- stage status: {data.get('status')} "
        f"(completed: {data.get('completed_cases')})",
        f"- evidence label: {data.get('evidence_label')}",
        f"- {TRUTH_NOTE}",
        "",
        "## Interpretation limits",
        "",
        "- trajectory separation != estimator error;",
        "- seven-day endpoint difference != measurement residual;",
        "- element divergence != covariance inconsistency;",
        "- model-vs-model agreement != external validation;",
        "- seven-day confirmation != mission lifetime prediction.",
        "- The seven-day eccentric-case analysis tests argument-of-perilune "
        "evolution for S8. It does not constitute a seven-day rerun of the "
        "Phase 13G-b e=0.01 baseline case.",
        "- Circular cases (L1/L2/L4) report eccentricity magnitude and "
        "nonsingular e*cos/e*sin components; no argument-of-perilune "
        "interpretation is made for them.",
        "",
        "## Methodology: continuous propagation (chunked rejected by preflight)",
        "",
        "**The chunked methodology was rejected by preflight. All scientific "
        "campaign cases use uninterrupted continuous propagation** (one "
        "`propagate_state` call per model case; no day/chunk restarts).",
        f"- preflight status: **{pf.get('status', '?')}**; decision: "
        f"`{pf.get('decision', '?')}`; reason: `{pf.get('reason', '?')}`.",
        f"- gate detail: {'PASS' if gate.get('pass') else 'FAIL'} — {gate.get('rule')}",
    ]
    for name, m in (pf.get("per_model") or {}).items():
        lines.append(f"- {name}: final dpos {m['final_dpos_m']:.3e} m, "
                     f"max {m['max_dpos_m']:.3e} m, final dvel "
                     f"{m['final_dvel_mps']:.3e} m/s, |da| {m['final_da_m']:.3e} m, "
                     f"|de| {m['final_de']:.3e}, |di| {m['final_di_rad']:.3e} rad")
    comp = pf.get("comparison") or {}
    if comp:
        lines.append(
            f"- comparison artifact (J2 vs full128): final "
            f"{comp['artifact_final_dpos_m']:.3e} m, max "
            f"{comp['artifact_max_dpos_m']:.3e} m against a measured "
            f"separation of {comp['continuous_final_dpos_m']:.3e} m "
            f"(limit {gate.get('rel_limit_m', float('nan')):.3e} m).")
    lines.append("")
    for cid, cdata in data.get("cases", {}).items():
        lines += [
            f"## {cid} ({cdata['case_ref']}; out_step {cdata['out_step_s']:.0f} s, "
            f"period {cdata['period_s']:.0f} s)",
            "",
            "| comparison | status | final dpos m | day1..day7 endpoint dpos m "
            "| growth |",
            "|---|---|---|---|---|",
        ]
        for c in cdata.get("comparisons", []):
            if "final_dpos_m" not in c:
                lines.append(f"| {c['comparison']} | {c.get('status')} | - | - | - |")
                continue
            daily = ", ".join("-" if d is None else f"{d:.0f}"
                              for d in c["daily_endpoint_dpos_m"])
            lines.append(
                f"| {c['comparison']} | {c['status']} | {c['final_dpos_m']:.3e} "
                f"| {daily} | {c['growth']['classification']} |")
        lines.append("")
        lines.append(f"- growth labels are a {GROWTH_LABEL}; the raw "
                     "day-1..7 series above is the primary output.")
        lines.append("")
    lines += [
        "## Safety",
        "",
        ("- No surface crossing was observed over seven days in the tested "
         "cases." if not any_crossing else
         "- SURFACE CROSSING detected in at least one run; scientific "
         "metrics were truncated post-hoc at the first sampled crossing "
         "within that run's single uninterrupted continuous propagation."),
        "- This statement concerns the tested seven-day windows only; no "
        "long-term orbital-stability claim is made.",
        "",
        "## Limitations",
        "",
        f"- {CROSSING_LIMITATION}",
        "- All scientific campaign runs above use uninterrupted continuous "
        "propagation (no chunk restarts). Chunked state-handoff propagation "
        "was rejected by the preflight above (multistep-integrator restart "
        "artifact) and is retained only as evidence/diagnostic tooling, not "
        "used for any reported case.",
        f"- GL1800F@256 gate: {data.get('gl256_gate')}",
        "- Runtime projections used for gating are rough empirical "
        "projections, not guarantees.",
        "",
    ]
    SEVEN_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[out] wrote {SEVEN_MD_PATH}")


def write_outputs(store: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    store["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    store["truth_note"] = TRUTH_NOTE
    _atomic_store_dump(store)
    print(f"[out] wrote {STORE_PATH}")
    _write_csv(RUNS_CSV, _rows(store, "runs"))
    _write_csv(COMP_CSV, _rows(store, "comparisons"))
    _write_csv(ELEM_CSV, _rows(store, "elements"))
    _write_csv(SENS_RUNS_CSV, _sensitivity_rows(store, "runs"))
    _write_csv(SENS_COMP_CSV, _sensitivity_rows(store, "comparisons"))
    _write_csv(SENS_ELEM_CSV, _sensitivity_rows(store, "elements"))
    _write_sensitivity_md(store)
    _write_csv(SEVEN_RUNS_CSV, _sevenday_rows(store, "runs"))
    _write_csv(SEVEN_COMP_CSV, _sevenday_rows(store, "comparisons"))
    _write_csv(SEVEN_ELEM_CSV, _sevenday_rows(store, "elements"))
    _write_csv(SEVEN_DAILY_CSV, _sevenday_rows(store, "daily"))
    _write_sevenday_md(store)

    lines = [
        "# Phase 13G — Real Lunar Gravity Orbit-Effect Campaign",
        "",
        f"- store updated (UTC): {store['generated_utc']}",
        f"- {TRUTH_NOTE}",
        "- baseline orbit: phase6 LLO (a=R_MOON+100 km, e=0.01, i=45 deg, "
        "perilune ~81.6 km); windows: 1 orbit + 1 day; cadence 60 s.",
        "- element drift = ORBIT-AVERAGED differencing (first vs last orbital "
        "period means) for windows >= 2 orbits, endpoint difference otherwise; "
        "a plain linear fit would leak short-period signal into the slope. "
        "Read drift_per_day together with short_period_amp.",
        "",
    ]
    for stage_name, title in (("baseline", "Baseline decomposition (GRGM660PRIM)"),
                              ("compare", "GL1800F comparison points"),
                              ("frames", "Frame diagnostics — INTENTIONALLY WRONG runs")):
        data = store.get(stage_name)
        if not data:
            continue
        lines += [f"## {title}", ""]
        if stage_name == "frames":
            lines += ["**Every row in this section is a deliberately wrong "
                      "configuration (diagnostic=True); none of these numbers "
                      "is a valid physical result.**", ""]
        for wname, wdata in data["windows"].items():
            lines += [f"### window: {wname}", "",
                      "| comparison | final dpos m | R m | T m | N m | rms T m | flag |",
                      "|---|---|---|---|---|---|---|"]
            for c in wdata["comparisons"]:
                flag = "DIAG-WRONG" if c["intentionally_wrong"] else ""
                lines.append(
                    f"| {c['comparison']} | {c['final_dpos_m']:.3e} "
                    f"| {c['final_radial_m']:+.2e} | {c['final_along_m']:+.2e} "
                    f"| {c['final_cross_m']:+.2e} | {c['rms_along_m']:.2e} | {flag} |"
                )
            lines.append("")
        if stage_name == "baseline" and data.get("kepler_check"):
            lines += [f"- Kepler-only integrator sanity: {data['kepler_check']}", ""]
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[out] wrote {MD_PATH}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--frames", action="store_true")
    parser.add_argument("--sensitivity", action="store_true",
                        help="13G-c1 altitude/inclination/eccentricity axes")
    parser.add_argument("--sevenday", action="store_true",
                        help="13G-c2 selected seven-day confirmation "
                             "(chunk-preflight gated)")
    parser.add_argument("--plots", action="store_true",
                        help="(deferred — refuses to run)")
    args = parser.parse_args(argv)
    if args.plots:
        print("[phase13g] --plots is deliberately not implemented.")
        return 2

    explicit = (args.baseline or args.compare or args.frames
                or args.sensitivity or args.sevenday)
    do_baseline = args.baseline or not explicit
    store = _load_store()
    if do_baseline:
        store["baseline"] = run_baseline()
    if args.compare:
        store["compare"] = run_compare()
    if args.frames:
        store["frames"] = run_frames()
    if args.sensitivity:
        store["sensitivity"] = run_sensitivity()
    if args.sevenday:
        run_sevenday(store)      # writes the stage into `store` atomically
    write_outputs(store)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
