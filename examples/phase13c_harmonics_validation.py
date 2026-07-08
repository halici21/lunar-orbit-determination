"""Phase 13C — low-degree harmonics validation campaign + rotation cadence preflight.

Validates the Phase 13B dynamics splice end-to-end with REAL MOON_PA rotation
grids (SPICE ``sample_moon_pa_rotations``) on synthetic low-degree models, and
answers the rotation-cadence question (10 s reference / 60 s candidate /
300 s coarse) numerically.  Uses the production ``propagate_state`` directly;
NO production code is modified.  No real GRAIL coefficients: all models are
synthetic with ``mu = MU_MOON_M3S2`` and ``r_ref = R_MOON_M``.

Stages (shared infrastructure, one script):
  --preflight  rotation-matrix midpoint errors + M2 propagation at 10/60/300 s
               cadences; decision on the 60 s default via E/S22 ratios.
  --campaign   validation comparisons: C20-only vs Moon-J2 bridge (same
               constant frame), MOON_PA-vs-mean-pole frame effect, C22 and
               C30 contributions, nmax/mmax truncation ladder, polar sanity.
  (default: both)

Continuity: initial state and .mat ephemeris come from
``phase6_scenario_comparison`` (same representative oblique LLO, same TDB
epoch), so results are directly comparable with the Phase 6/7 campaign.
Windows are staged: 1 orbit first, then 1 day (7 days deliberately NOT run —
recommendation only).  Only summary CSV/JSON/MD (+ optional summary plots) are
written to results/phase13/ (git-ignored); dense trajectories are never saved.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
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
from lunar_od.dynamics import _MCI_TO_MOON_BF, f3body_moon, propagate_state  # noqa: E402
from lunar_od.gravity_harmonics import (  # noqa: E402
    SphericalHarmonicGravityModel, cbar_n0_from_unnormalized_jn,
)
from lunar_od.lunar_frames import (  # noqa: E402
    nearest_rotation_at_time, sample_moon_pa_rotations,
)
from lunar_od.orbit import coe2rv  # noqa: E402
from lunar_od.spice_loader import load_spice_kernels  # noqa: E402
from phase6_scenario_comparison import initial_state, load_ephemeris  # noqa: E402

MU_M, MU_E, MU_S = MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2
J2 = J2_MOON_UNNORMALIZED
J3 = 8.46e-6
J2000_JD = 2451545.0
OUT = ROOT / "results" / "phase13"
CADENCES_S = (10.0, 60.0, 300.0)
REF_CADENCE_S = 10.0


# ---------------------------------------------------------------------------
# Synthetic low-degree model set (NOT real GRAIL; mu/r_ref = production Moon)
# ---------------------------------------------------------------------------
def build_models() -> dict[str, SphericalHarmonicGravityModel]:
    def mk(cbar, sbar, nmax, mmax):
        return SphericalHarmonicGravityModel(
            mu_m3_s2=MU_M, r_ref_m=R_MOON_M, cbar=cbar, sbar=sbar,
            nmax=nmax, mmax=mmax,
        )

    c20 = cbar_n0_from_unnormalized_jn(J2, 2)      # -J2/sqrt(5)
    c30 = cbar_n0_from_unnormalized_jn(J3, 3)      # -J3/sqrt(7)

    cb1 = np.zeros((3, 3)); cb1[2, 0] = c20
    m1 = mk(cb1, np.zeros((3, 3)), 2, 0)                       # M1: C20-only

    cb2 = cb1.copy(); cb2[2, 2] = 3.47e-5
    sb2 = np.zeros((3, 3)); sb2[2, 2] = 1.0e-6
    m2 = mk(cb2, sb2, 2, 2)                                    # M2: C20+C22

    cb3 = np.zeros((4, 4)); cb3[:3, :3] = cb2; cb3[3, 0] = c30
    sb3 = np.zeros((4, 4)); sb3[:3, :3] = sb2
    m3 = mk(cb3, sb3, 3, 3)                                    # M3: +C30

    cb4 = np.zeros((5, 5)); cb4[:4, :4] = cb3
    sb4 = np.zeros((5, 5)); sb4[:4, :4] = sb3
    cb4[3, 1], sb4[3, 1] = 2.6e-5, 5.5e-6                      # extra synthetic
    cb4[3, 3], sb4[3, 3] = 1.2e-5, -2.0e-6
    cb4[4, 0] = -1.0e-6
    cb4[4, 2], sb4[4, 2] = 3.0e-6, -1.0e-6
    cb4[4, 4], sb4[4, 4] = 9.0e-7, 4.0e-7
    m4 = mk(cb4, sb4, 4, 4)                                    # M4: nmax=4 full
    m4_n3 = mk(cb4, sb4, 3, 3)
    m4_n2 = mk(cb4, sb4, 2, 2)
    m4_m0 = mk(cb4, sb4, 4, 0)                                 # zonal-only

    return {"M1": m1, "M2": m2, "M3": m3,
            "M4": m4, "M4_n3": m4_n3, "M4_n2": m4_n2, "M4_m0": m4_m0}


def polar_state() -> np.ndarray:
    a = R_MOON_M + 100e3
    r0, v0 = coe2rv(a, 0.01, np.radians(90.0), np.radians(30.0),
                    np.radians(20.0), np.radians(10.0), MU_M)
    return np.concatenate([r0, v0])


# ---------------------------------------------------------------------------
# Propagation harness (RHS-eval counting via the earth-position getter, which
# every RHS closure calls exactly once per evaluation)
# ---------------------------------------------------------------------------
class _CountingGetter:
    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, t):
        self.calls += 1
        return self.fn(t)


def run_case(get_earth, get_sun, state0, teval, *, j2_moon=0.0,
             harmonic_model=None, harmonic_rotation=None) -> dict:
    """One propagation with runtime, RHS-eval count and altitude sanity."""
    ge = _CountingGetter(get_earth)
    t0 = time.perf_counter()
    traj = propagate_state(
        teval, state0, MU_M, MU_E, MU_S, ge, get_sun,
        method="ADAMS", j2_moon=j2_moon, j2_earth=0.0,
        harmonic_model=harmonic_model, harmonic_rotation=harmonic_rotation,
    )
    runtime = time.perf_counter() - t0
    if not np.all(np.isfinite(traj)):
        raise RuntimeError("NaN/Inf in propagation output.")
    radii = np.linalg.norm(traj[:, :3], axis=1)
    min_radius = float(radii.min())
    return {
        "traj": traj,
        "runtime_s": float(runtime),
        "rhs_evals": int(ge.calls),
        "min_radius_m": min_radius,
        "min_altitude_m": min_radius - R_MOON_M,
        "surface_crossing": bool(min_radius <= R_MOON_M),
    }


def diff_metrics(traj, base) -> dict:
    dp = np.linalg.norm(traj[:, :3] - base[:, :3], axis=1)
    dv = np.linalg.norm(traj[:, 3:] - base[:, 3:], axis=1)
    return {
        "final_dpos_m": float(dp[-1]),
        "max_dpos_m": float(dp.max()),
        "rms_dpos_m": float(np.sqrt(np.mean(dp ** 2))),
        "final_dvel_mps": float(dv[-1]),
    }


def rotation_pair(et0, t_end_s, cadence_s):
    """MOON_PA rotation grid covering [-margin, T+margin] (kernels pre-loaded)."""
    margin = max(2.0 * cadence_s, 120.0)
    t_grid = np.arange(-margin, t_end_s + margin + cadence_s / 2.0, cadence_s)
    rots = sample_moon_pa_rotations(et0, t_grid, load_kernels=False)
    return (t_grid, rots)


def _rot_angle(a, b) -> float:
    x = (np.trace(b @ a.T) - 1.0) / 2.0
    return float(math.acos(min(1.0, max(-1.0, x))))


def midpoint_rotation_error(et0, pair, cadence_s, t_end_s, n_probe=8) -> float:
    """Max nearest-vs-direct-SPICE angle error at grid midpoints (rad)."""
    t_grid, rots = pair
    worst = 0.0
    for frac in np.linspace(0.05, 0.95, n_probe):
        t_q = float(frac * t_end_s + cadence_s / 2.0)
        c_near = nearest_rotation_at_time(rots, t_grid, t_q)
        c_true = sample_moon_pa_rotations(et0 + t_q, np.array([0.0]),
                                          load_kernels=False)[0]
        worst = max(worst, _rot_angle(c_near, c_true))
    return worst


def classify_ratio(ratio: float) -> str:
    if ratio < 0.01:
        return "ACCEPT (default)"
    if ratio < 0.1:
        return "MARGINAL (finer cadence for science runs)"
    return "REJECT (reduce cadence)"


# ---------------------------------------------------------------------------
# Campaign driver
# ---------------------------------------------------------------------------
def run_window(window: str, models: dict) -> tuple[list, list, dict]:
    """All propagations + comparisons for one window; returns rows and info."""
    if window == "orbit":
        a = R_MOON_M + 100e3
        period = 2.0 * math.pi * math.sqrt(a ** 3 / MU_M)      # ~7067 s
        t_end = math.ceil(period / 60.0) * 60.0
        out_step = 60.0
    elif window == "day1":
        t_end = 86400.0
        out_step = 600.0
    else:
        raise ValueError(f"unknown window {window!r}")

    teval = np.arange(0.0, t_end + 1.0, out_step)
    eph, first_jd = load_ephemeris(t_end + 600.0)
    et0 = (first_jd - J2000_JD) * 86400.0
    ge, gs = eph.earth_position, eph.sun_position
    s0 = initial_state()

    pairs = {c: rotation_pair(et0, t_end, c) for c in CADENCES_S}

    # -- all propagation cases (deduplicated; shared by both stages) --------
    print(f"[{window}] propagating (t_end={t_end:.0f} s, {teval.size} epochs) ...")
    runs: dict[str, dict] = {}

    def add(name, **kw):
        runs[name] = run_case(ge, gs, s0, teval, **kw)
        r = runs[name]
        print(f"  {name:12s} runtime {r['runtime_s']:7.2f} s  rhs {r['rhs_evals']:7d}  "
              f"min_alt {r['min_altitude_m']/1e3:7.2f} km  "
              f"{'SURFACE-CROSSING!' if r['surface_crossing'] else ''}")

    add("j2_baseline", j2_moon=J2)
    add("m1_const", harmonic_model=models["M1"], harmonic_rotation=_MCI_TO_MOON_BF)
    add("m1_pa10", harmonic_model=models["M1"], harmonic_rotation=pairs[10.0])
    add("m2_pa10", harmonic_model=models["M2"], harmonic_rotation=pairs[10.0])
    add("m2_pa60", harmonic_model=models["M2"], harmonic_rotation=pairs[60.0])
    add("m2_pa300", harmonic_model=models["M2"], harmonic_rotation=pairs[300.0])
    add("m3_pa10", harmonic_model=models["M3"], harmonic_rotation=pairs[10.0])
    add("m4n4_pa10", harmonic_model=models["M4"], harmonic_rotation=pairs[10.0])
    add("m4n3_pa10", harmonic_model=models["M4_n3"], harmonic_rotation=pairs[10.0])
    add("m4n2_pa10", harmonic_model=models["M4_n2"], harmonic_rotation=pairs[10.0])
    add("m4m0_pa10", harmonic_model=models["M4_m0"], harmonic_rotation=pairs[10.0])

    polar_runs = {}
    if window == "orbit":
        sp = polar_state()
        for name, model in (("m1_polar_pa60", models["M1"]),
                            ("m2_polar_pa60", models["M2"])):
            polar_runs[name] = run_case(ge, gs, sp, teval,
                                        harmonic_model=model,
                                        harmonic_rotation=pairs[60.0])
            r = polar_runs[name]
            print(f"  {name:14s} runtime {r['runtime_s']:5.2f} s  rhs {r['rhs_evals']:6d}  "
                  f"min_alt {r['min_altitude_m']/1e3:7.2f} km")

    invalid = [k for k, r in {**runs, **polar_runs}.items() if r["surface_crossing"]]
    if invalid:
        raise RuntimeError(f"[{window}] surface crossing in runs: {invalid} — window INVALID")

    # -- acceleration bridge at t0 -------------------------------------------
    a_j2 = f3body_moon(s0, MU_M, MU_E, MU_S, eph.earth_position(0.0),
                       eph.sun_position(0.0), j2_moon=J2)
    a_h = f3body_moon(s0, MU_M, MU_E, MU_S, eph.earth_position(0.0),
                      eph.sun_position(0.0), harmonic_model=models["M1"],
                      c_inertial_to_bf_harmonic=_MCI_TO_MOON_BF)
    accel_bridge = float(np.linalg.norm(a_h - a_j2))

    # -- validation comparison rows -------------------------------------------
    def comp(name, run_a, run_b, note, source=runs):
        m = diff_metrics(source[run_a]["traj"], source[run_b]["traj"])
        return {"window": window, "comparison": name, "case_a": run_a,
                "case_b": run_b, **m,
                "min_altitude_km_a": source[run_a]["min_altitude_m"] / 1e3,
                "surface_crossing": False, "note": note}

    val_rows = [
        {**comp("c20_vs_moon_j2_bridge", "m1_const", "j2_baseline",
                "same constant mean-pole frame; expect integrator-tolerance level"),
         "accel_bridge_mps2": accel_bridge},
        comp("frame_effect_pa_vs_meanpole", "m1_pa10", "m1_const",
             "PHYSICAL frame difference (MOON_PA w/ libration vs fixed mean pole), not an error"),
        comp("c22_effect", "m2_pa10", "m1_pa10", "S22 signal (longitude-dependent)"),
        comp("c30_effect", "m3_pa10", "m2_pa10", "J3/C30 contribution"),
        comp("trunc_nmax3_vs_4", "m4n3_pa10", "m4n4_pa10", "n=4 row contribution"),
        comp("trunc_nmax2_vs_4", "m4n2_pa10", "m4n4_pa10", "n=3+4 rows contribution"),
        comp("trunc_zonal_vs_full", "m4m0_pa10", "m4n4_pa10", "all m>0 contribution"),
    ]
    if polar_runs:
        val_rows.append(comp("polar_c22_effect", "m2_polar_pa60", "m1_polar_pa60",
                             "polar i=90 sanity (60 s cadence)", source=polar_runs))

    # -- cadence preflight rows -------------------------------------------------
    s22 = diff_metrics(runs["m2_pa10"]["traj"], runs["m1_pa10"]["traj"])
    cad_rows = []
    for cad, run_name in ((10.0, "m2_pa10"), (60.0, "m2_pa60"), (300.0, "m2_pa300")):
        row = {
            "window": window, "cadence_s": cad,
            "rotmat_midpoint_err_rad": midpoint_rotation_error(et0, pairs[cad], cad, t_end),
            "runtime_s": runs[run_name]["runtime_s"],
            "rhs_evals": runs[run_name]["rhs_evals"],
            "grid_points": pairs[cad][0].size,
        }
        if cad != REF_CADENCE_S:
            err = diff_metrics(runs[run_name]["traj"], runs["m2_pa10"]["traj"])
            ratio = max(
                err["final_dpos_m"] / s22["final_dpos_m"] if s22["final_dpos_m"] > 0 else 0.0,
                err["max_dpos_m"] / s22["max_dpos_m"] if s22["max_dpos_m"] > 0 else 0.0,
            )
            row.update({f"err_{k}": v for k, v in err.items()},
                       ratio_err_over_s22=float(ratio),
                       decision=classify_ratio(ratio))
        else:
            row.update(decision="REFERENCE")
        cad_rows.append(row)

    info = {
        "window": window, "t_end_s": t_end, "out_step_s": out_step,
        "epochs": int(teval.size), "first_jd_tdb": first_jd, "et0_s": et0,
        "accel_bridge_mps2": accel_bridge,
        "s22_final_dpos_m": s22["final_dpos_m"], "s22_max_dpos_m": s22["max_dpos_m"],
        "rotation_grids": {str(c): {"points": int(pairs[c][0].size),
                                    "range_s": [float(pairs[c][0][0]), float(pairs[c][0][-1])]}
                           for c in CADENCES_S},
        "runs": {k: {kk: vv for kk, vv in r.items() if kk != "traj"}
                 for k, r in {**runs, **polar_runs}.items()},
    }
    return val_rows, cad_rows, info


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_csv(path: Path, rows: list[dict]) -> None:
    import csv
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "window", k))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_plots(val_rows, cad_rows) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    plots_dir = OUT / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written = []

    fig, ax = plt.subplots(figsize=(7, 4))
    for window in sorted({r["window"] for r in cad_rows}):
        rows = [r for r in cad_rows if r["window"] == window and "err_final_dpos_m" in r]
        ax.loglog([r["cadence_s"] for r in rows],
                  [max(r["err_final_dpos_m"], 1e-12) for r in rows],
                  "o-", label=f"{window}: |traj(c) - traj(10s)| final")
    ax.set_xlabel("rotation cadence [s]"); ax.set_ylabel("final position error [m]")
    ax.set_title("Phase 13C rotation-cadence propagation error (M2, vs 10 s reference)")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    p = plots_dir / "cadence_position_error.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    written.append(str(p))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    comps = ["c22_effect", "c30_effect", "trunc_nmax3_vs_4",
             "trunc_nmax2_vs_4", "trunc_zonal_vs_full"]
    windows = sorted({r["window"] for r in val_rows})
    width = 0.8 / max(1, len(windows))
    x = np.arange(len(comps))
    for i, window in enumerate(windows):
        vals = []
        for c in comps:
            row = next((r for r in val_rows if r["window"] == window and r["comparison"] == c), None)
            vals.append(row["final_dpos_m"] if row else np.nan)
        ax.bar(x + i * width, vals, width, label=window)
    ax.set_yscale("log"); ax.set_xticks(x + width / 2)
    ax.set_xticklabels(comps, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("final position difference [m]")
    ax.set_title("Phase 13C low-degree contributions (synthetic models, 10 s MOON_PA grid)")
    ax.grid(True, axis="y", alpha=0.3); ax.legend()
    p = plots_dir / "low_degree_comparison.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    written.append(str(p))
    return written


def write_report(val_rows, cad_rows, infos, plots) -> None:
    lines = ["# Phase 13C — Low-Degree Harmonics Validation + Rotation Cadence Preflight", ""]
    lines += [f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} | "
              f"epoch first_jd(TDB)={infos[0]['first_jd_tdb']:.6f} | synthetic models only "
              f"(mu=MU_MOON, R_ref=R_MOON_M; NOT real GRAIL)", ""]

    lines += ["## Rotation cadence preflight", "",
              "| window | cadence [s] | rotmat midpoint err [rad] | final dpos vs 10s [m] | "
              "max dpos vs 10s [m] | E/S22 | runtime [s] | RHS evals | decision |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in cad_rows:
        lines.append(
            f"| {r['window']} | {r['cadence_s']:.0f} | {r['rotmat_midpoint_err_rad']:.2e} | "
            f"{r.get('err_final_dpos_m', 0.0):.3e} | {r.get('err_max_dpos_m', 0.0):.3e} | "
            f"{r.get('ratio_err_over_s22', 0.0):.2e} | {r['runtime_s']:.2f} | "
            f"{r['rhs_evals']} | {r['decision']} |")
    lines.append("")

    lines += ["## Validation comparisons", "",
              "| window | comparison | final dpos [m] | max dpos [m] | rms dpos [m] | "
              "final dvel [m/s] | note |", "|---|---|---:|---:|---:|---:|---|"]
    for r in val_rows:
        lines.append(
            f"| {r['window']} | {r['comparison']} | {r['final_dpos_m']:.3e} | "
            f"{r['max_dpos_m']:.3e} | {r['rms_dpos_m']:.3e} | {r['final_dvel_mps']:.3e} | "
            f"{r['note']} |")
    lines.append("")

    for info in infos:
        lines += [f"## Sanity — window {info['window']}",
                  f"- acceleration bridge (M1+const vs j2_moon) at t0: "
                  f"{info['accel_bridge_mps2']:.3e} m/s^2",
                  f"- S22 signal: final {info['s22_final_dpos_m']:.3e} m, "
                  f"max {info['s22_max_dpos_m']:.3e} m",
                  "- min altitude per run [km]: " + ", ".join(
                      f"{k}={v['min_altitude_m']/1e3:.1f}" for k, v in info["runs"].items()),
                  "- surface crossing: none (all runs valid)", ""]

    lines += ["## Interpretation / limitations",
              "- All coefficient sets are SYNTHETIC low-degree stand-ins; magnitudes are "
              "Moon-like but this is NOT a real GRAIL field and accuracy claims must not "
              "be read as high-fidelity gravity modelling.",
              "- The C20-only vs Moon-J2 bridge uses the SAME constant mean-pole frame, so "
              "it isolates the engine path; the MOON_PA-vs-mean-pole row is a PHYSICAL "
              "frame (libration/pole) difference, not an error.",
              "- Free-run trajectory separations are phase effects, not navigation errors.",
              "- The 10 s cadence run is itself a nearest-neighbour grid (best available "
              "reference), not continuous SPICE.",
              "- 7-day window NOT run (kept for separate approval); based on the 1-day "
              "runtimes below, a 7-day run is expected to scale roughly linearly and "
              "remain feasible; rotation-grid memory at 10 s for 7 days is ~60k x 3 x 3 "
              "floats (~4 MB), acceptable.", ""]
    if plots:
        lines += ["## Plots", *[f"- {p}" for p in plots], ""]
    lines.append("Production code NOT modified; outputs regenerable from this script.")
    (OUT / "phase13c_report.md").write_text("\n".join(lines), encoding="utf-8")


def git_info():
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip()
        return {"head": h}
    except Exception:
        return {"head": None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preflight", action="store_true", help="cadence preflight only")
    ap.add_argument("--campaign", action="store_true", help="validation campaign only")
    ap.add_argument("--windows", nargs="+", default=["orbit", "day1"],
                    choices=["orbit", "day1"])
    ap.add_argument("--plots", action="store_true", help="write summary plots")
    args = ap.parse_args(argv)
    do_pre = args.preflight or not args.campaign
    do_camp = args.campaign or not args.preflight

    OUT.mkdir(parents=True, exist_ok=True)
    load_spice_kernels()                      # once; samplers use load_kernels=False
    models = build_models()

    all_val, all_cad, infos = [], [], []
    for window in args.windows:
        val_rows, cad_rows, info = run_window(window, models)
        all_val += val_rows
        all_cad += cad_rows
        infos.append(info)

    if do_camp:
        write_csv(OUT / "phase13c_harmonics_validation.csv", all_val)
    if do_pre:
        write_csv(OUT / "phase13c_rotation_cadence.csv", all_cad)
    plots = write_plots(all_val, all_cad) if args.plots else []
    write_report(all_val, all_cad, infos, plots)

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_info(), "windows": args.windows,
        "cadences_s": list(CADENCES_S), "reference_cadence_s": REF_CADENCE_S,
        "models": {
            name: {"nmax": m.nmax, "mmax": m.mmax,
                   "nonzero": {f"C({n},{mm})": float(m.cbar[n, mm])
                               for n in range(m.cbar.shape[0])
                               for mm in range(m.cbar.shape[1])
                               if m.cbar[n, mm] != 0.0}}
            for name, m in models.items()},
        "windows_info": infos,
        "notes": ["synthetic models only (no real GRAIL)",
                  "dense trajectories not saved",
                  "j2_earth=0 in all runs (harmonics isolation); third body real"],
    }
    (OUT / "phase13c_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n=== Phase 13C summary ===")
    for r in all_cad:
        if "ratio_err_over_s22" in r:
            print(f"  [{r['window']}] cadence {r['cadence_s']:.0f}s: E/S22 = "
                  f"{r['ratio_err_over_s22']:.2e} -> {r['decision']}")
    b = [r for r in all_val if r["comparison"] == "c20_vs_moon_j2_bridge"]
    for r in b:
        print(f"  [{r['window']}] C20-vs-J2 bridge final dpos = {r['final_dpos_m']:.3e} m")
    print(f"  outputs -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
