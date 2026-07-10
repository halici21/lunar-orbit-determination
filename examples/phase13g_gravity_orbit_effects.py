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
  (default: --baseline)
Refused here (13G-c+ scope): --sensitivity, --sevenday, --plots.

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
from phase12b_real_grail_validation import (  # noqa: E402
    MODELS as GRAIL_FILES, diff_metrics, load_kernel_profile, rotation_pair,
    run_case, truncate_model,
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
def window_setup(window: str):
    """Baseline windows; day1 sampled at 120 s (denser than 13C's 600 s) so the
    osculating-element series resolves short-period content (~12 -> ~59
    samples/rev).  Sampling density does not change the integration itself."""
    if window == "orbit":
        a = R_MOON_M + 100e3
        period = 2.0 * math.pi * math.sqrt(a ** 3 / MU_M)
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
    return teval, t_end, eph, et0, initial_state()


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

    def __init__(self, window: str, stage: str):
        self.window = window
        self.stage = stage
        self.teval, self.t_end, self.eph, self.et0, self.s0 = window_setup(window)
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
            "stage": self.stage, "window": self.window, "run": name,
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
                    "stage": self.stage, "window": self.window, "run": name,
                    "diagnostic": diagnostic, **erow,
                })
        flag = " DIAG" if diagnostic else ""
        print(f"  [{self.window}] {name:26s} rt {res['runtime_s']:6.2f} s  "
              f"rhs {res['rhs_evals']:6d}  min_alt "
              f"{res['min_altitude_m'] / 1e3:6.2f} km{flag}"
              f"{'  SURFACE-CROSSING!' if res['surface_crossing'] else ''}")
        return res

    def compare(self, name: str, case: str, reference: str, note: str,
                diagnostic=False, intentionally_wrong=False) -> dict:
        traj, ref = self.trajs[case], self.trajs[reference]
        row = {
            "stage": self.stage, "window": self.window, "comparison": name,
            "case": case, "reference": reference,
            **diff_metrics(traj, ref), **rtn_summary(traj, ref),
            "diagnostic": diagnostic, "intentionally_wrong": intentionally_wrong,
            "note": note,
        }
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


def write_outputs(store: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    store["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    store["truth_note"] = TRUTH_NOTE
    STORE_PATH.write_text(json.dumps(store, indent=2), encoding="utf-8")
    print(f"[out] wrote {STORE_PATH}")
    _write_csv(RUNS_CSV, _rows(store, "runs"))
    _write_csv(COMP_CSV, _rows(store, "comparisons"))
    _write_csv(ELEM_CSV, _rows(store, "elements"))

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
                        help="(13G-c scope — refuses to run)")
    parser.add_argument("--sevenday", action="store_true",
                        help="(13G-c scope — refuses to run)")
    parser.add_argument("--plots", action="store_true",
                        help="(deferred — refuses to run)")
    args = parser.parse_args(argv)
    if args.sensitivity or args.sevenday or args.plots:
        print("[phase13g] --sensitivity/--sevenday/--plots are Phase 13G-c+ "
              "scope and are deliberately not implemented in 13G-b.")
        return 2

    do_baseline = args.baseline or not (args.baseline or args.compare or args.frames)
    store = _load_store()
    if do_baseline:
        store["baseline"] = run_baseline()
    if args.compare:
        store["compare"] = run_compare()
    if args.frames:
        store["frames"] = run_frames()
    write_outputs(store)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
