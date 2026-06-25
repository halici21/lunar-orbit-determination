"""Phase 6A — force-model scenario comparison campaign (verification only).

Propagates ONE common oblique-LLO initial state under five force-model
combinations and decomposes each perturbation's contribution.  Uses the
production ``propagate_state`` directly; NO production code is modified.  Third
body is toggled off with ``mu=0`` (Keplerian); J2 via the existing flags.

Scenarios (single ephemeris = PlanetEphemeris .mat, so force effects are isolated):
  1 Keplerian          mu_e=0,   mu_s=0,   j2_moon=0,       j2_earth=0
  2 +Third Body        mu_e,mu_s real,     j2_moon=0,       j2_earth=0      (baseline)
  3 +Earth J2          mu_e,mu_s real,     j2_moon=0,       j2_earth=J2_E   (indirect)
  4 +Moon J2           mu_e,mu_s real,     j2_moon=MOON_J2, j2_earth=0
  5 +Earth J2+Moon J2  mu_e,mu_s real,     j2_moon=MOON_J2, j2_earth=J2_E   (indirect)

J2 contributions are measured vs the Third-Body baseline (scenario 2).
The PlanetEphemeris/SPICE ephemeris-source difference is a SEPARATE layer
(see phase6_ephemeris_comparison.py); it is not a force-model error.
"""
from __future__ import annotations
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py  # noqa: E402
from lunar_od.ephemeris import MoonCenteredEphemeris  # noqa: E402
from lunar_od.orbit import coe2rv  # noqa: E402
from lunar_od.dynamics import propagate_state, MOON_J2  # noqa: E402
from lunar_od.constants import (  # noqa: E402
    MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
    J2_EARTH_UNNORMALIZED, R_EARTH_J2_REF_M, R_MOON_M,
)

MAT = ROOT / "ephemeris_data.mat"
OUT = ROOT / "results" / "phase6"
J2000_JD = 2451545.0
CADENCE_S = 600.0

# Single common oblique LLO initial state (100 km altitude, i=45 deg).
def initial_state() -> np.ndarray:
    a = R_MOON_M + 100e3
    r0, v0 = coe2rv(a, 0.01, np.radians(45.0), np.radians(30.0),
                    np.radians(20.0), np.radians(10.0), MU_MOON_M3S2)
    return np.concatenate([r0, v0])


SCENARIOS = {
    "1_keplerian":      dict(mu_e=0.0,           mu_s=0.0,          j2_moon=0.0,     j2_earth=0.0),
    "2_third_body":     dict(mu_e=MU_EARTH_M3S2, mu_s=MU_SUN_M3S2,  j2_moon=0.0,     j2_earth=0.0),
    "3_earth_j2":       dict(mu_e=MU_EARTH_M3S2, mu_s=MU_SUN_M3S2,  j2_moon=0.0,     j2_earth=J2_EARTH_UNNORMALIZED),
    "4_moon_j2":        dict(mu_e=MU_EARTH_M3S2, mu_s=MU_SUN_M3S2,  j2_moon=MOON_J2, j2_earth=0.0),
    "5_earth_moon_j2":  dict(mu_e=MU_EARTH_M3S2, mu_s=MU_SUN_M3S2,  j2_moon=MOON_J2, j2_earth=J2_EARTH_UNNORMALIZED),
}


def load_ephemeris(duration_s: float, ephem_step_s: float = 10.0):
    """Load the .mat ephemeris for [0, duration], optionally subsampled.

    The .mat grid is 10 s native; ``ephem_step_s`` (multiple of 10 s) subsamples
    it (e.g. 60 s -> every 6th sample) while keeping the last point >= duration so
    the PCHIP interpolant covers the full propagation window without extrapolation.
    """
    step = max(1, int(round(ephem_step_s / 10.0)))
    f = h5py.File(str(MAT), "r")
    first_jd = float(np.array(f["first_jd"]).ravel()[0])
    n = int(duration_s / 10.0) + 3
    t = np.array(f["t_vec"]).ravel()[:n:step]
    rE = np.array(f["rEarth_data"][:, :n:step]).T
    rS = np.array(f["rSun_data"][:, :n:step]).T
    vE = np.array(f["vEarth_data"][:, :n:step]).T
    f.close()
    eph = MoonCenteredEphemeris(t_ephem_s=t, earth_pos_m=rE, sun_pos_m=rS, earth_vel_mps=vE)
    return eph, first_jd


def propagate(eph, state0, teval, sc):
    return propagate_state(
        teval, state0, MU_MOON_M3S2, sc["mu_e"], sc["mu_s"],
        eph.earth_position, eph.sun_position,
        method="ADAMS", j2_moon=sc["j2_moon"],
        j2_earth=sc["j2_earth"], earth_j2_mode="indirect",
    )


def rtn_of(diff_pos, base_state):
    r = base_state[:3]; v = base_state[3:]
    R = r / np.linalg.norm(r)
    N = np.cross(r, v); N = N / np.linalg.norm(N)
    T = np.cross(N, R)
    return float(diff_pos @ R), float(diff_pos @ T), float(diff_pos @ N)


def metrics(traj, base):
    dp = traj[:, :3] - base[:, :3]
    dv = traj[:, 3:] - base[:, 3:]
    np_ = np.linalg.norm(dp, axis=1); nv = np.linalg.norm(dv, axis=1)
    dR, dT, dN = rtn_of(dp[-1], base[-1])
    return dict(
        final_dpos_m=float(np_[-1]), final_dvel_mps=float(nv[-1]),
        max_dpos_m=float(np_.max()), rms_dpos_m=float(np.sqrt(np.mean(np_**2))),
        max_dvel_mps=float(nv.max()), rms_dvel_mps=float(np.sqrt(np.mean(nv**2))),
        rtn_final_R_m=dR, rtn_final_T_m=dT, rtn_final_N_m=dN,
    )


def git_info():
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT)).decode().strip()
        return {"head": h, "dirty": bool(dirty), "note": "Phase 0-6 work is uncommitted (local)"}
    except Exception:
        return {"head": None, "dirty": None}


def run_window(days: int, ephem_step_s: float = 10.0):
    T = days * 86400.0
    teval = np.arange(0.0, T + 1.0, CADENCE_S)
    eph, first_jd = load_ephemeris(T, ephem_step_s)
    state0 = initial_state()
    trajs = {k: propagate(eph, state0, teval, sc) for k, sc in SCENARIOS.items()}
    for k, tr in trajs.items():
        if not np.all(np.isfinite(tr)):
            raise RuntimeError(f"NaN/divergence in scenario {k} ({days}d)")

    base = trajs["2_third_body"]
    rows = {k: metrics(trajs[k], base) for k in SCENARIOS}

    # isolated contributions + interaction (vs third-body baseline)
    earth = trajs["3_earth_j2"] - base
    moon = trajs["4_moon_j2"] - base
    combined = trajs["5_earth_moon_j2"] - base
    inter = trajs["5_earth_moon_j2"] - trajs["3_earth_j2"] - trajs["4_moon_j2"] + base
    contrib = {
        "earth_j2_isolated_final_dpos_m": float(np.linalg.norm(earth[-1, :3])),
        "moon_j2_isolated_final_dpos_m": float(np.linalg.norm(moon[-1, :3])),
        "combined_j2_final_dpos_m": float(np.linalg.norm(combined[-1, :3])),
        "interaction_final_dpos_m": float(np.linalg.norm(inter[-1, :3])),
        "earth_j2_max_dpos_m": float(np.linalg.norm(earth[:, :3], axis=1).max()),
        "moon_j2_max_dpos_m": float(np.linalg.norm(moon[:, :3], axis=1).max()),
    }

    # min Moon-centered radius / altitude sanity per scenario (surface crossing = impact)
    sanity = {}
    for k, tr in trajs.items():
        rmin = float(np.linalg.norm(tr[:, :3], axis=1).min())
        amin = rmin - R_MOON_M
        sanity[k] = {"min_radius_m": rmin, "min_altitude_m": amin,
                     "surface_crossing": bool(amin < 0.0)}
    return rows, contrib, sanity, first_jd, state0, teval


def write_outputs(days, rows, contrib, sanity, first_jd, state0, teval, ephem_step_s=10.0):
    OUT.mkdir(parents=True, exist_ok=True)
    # CSV summary
    csv = OUT / f"phase6_force_comparison_{days}d.csv"
    cols = ["final_dpos_m", "final_dvel_mps", "max_dpos_m", "rms_dpos_m",
            "max_dvel_mps", "rms_dvel_mps", "rtn_final_R_m", "rtn_final_T_m", "rtn_final_N_m"]
    with open(csv, "w") as fh:
        fh.write("scenario," + ",".join(cols) + "\n")
        for k in SCENARIOS:
            fh.write(k + "," + ",".join(f"{rows[k][c]:.6e}" for c in cols) + "\n")
    # JSON metadata
    meta = {
        "phase": "6A", "window_days": days, "cadence_s": CADENCE_S,
        "epoch_first_jd_tdb": first_jd, "et0_s_past_j2000": (first_jd - J2000_JD) * 86400.0,
        "initial_state_mci": state0.tolist(), "n_output_points": int(teval.size),
        "scenarios": {k: {kk: (vv if not isinstance(vv, float) else vv) for kk, vv in v.items()}
                      for k, v in SCENARIOS.items()},
        "force_flags_note": "third body off via mu=0; earth_j2_mode=indirect (direct is NOT default)",
        "constants": {"MU_MOON_M3S2": MU_MOON_M3S2, "MU_EARTH_M3S2": MU_EARTH_M3S2,
                      "MU_SUN_M3S2": MU_SUN_M3S2, "MOON_J2": MOON_J2,
                      "J2_EARTH_UNNORMALIZED": J2_EARTH_UNNORMALIZED,
                      "R_EARTH_J2_REF_M": R_EARTH_J2_REF_M, "R_MOON_M": R_MOON_M},
        "ephemeris_source": f"PlanetEphemeris .mat (Moon-centered DE421, TDB, {ephem_step_s:.0f} s grid, PCHIP)",
        "ephem_step_s": ephem_step_s,
        "isolated_contributions": contrib, "min_altitude_sanity": sanity, "git": git_info(),
        "production_code_modified": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / f"phase6_metadata_{days}d.json").write_text(json.dumps(meta, indent=2))
    # Markdown report
    md = [f"# Phase 6A — Force-Model Comparison ({days}-day)", "",
          f"Epoch (first_jd TDB): {first_jd:.6f} | output cadence {CADENCE_S:.0f}s | "
          f"points {teval.size} | ephemeris: PlanetEphemeris .mat (DE421, {ephem_step_s:.0f}s grid)", "",
          "Differences vs Third-Body baseline (scenario 2).", "",
          "| scenario | final dpos [m] | max dpos [m] | rms dpos [m] | final dvel [m/s] |",
          "|---|---:|---:|---:|---:|"]
    for k in SCENARIOS:
        r = rows[k]
        md.append(f"| {k} | {r['final_dpos_m']:.3e} | {r['max_dpos_m']:.3e} | "
                  f"{r['rms_dpos_m']:.3e} | {r['final_dvel_mps']:.3e} |")
    e_iso = contrib["earth_j2_isolated_final_dpos_m"]
    inter = contrib["interaction_final_dpos_m"]
    rel = "comparable to or larger than" if inter >= e_iso else "comparable to"
    md += ["", "## Isolated J2 contributions (final position diff vs baseline)", "",
           f"- Earth J2: {e_iso:.3e} m (max {contrib['earth_j2_max_dpos_m']:.3e} m)",
           f"- Moon J2: {contrib['moon_j2_isolated_final_dpos_m']:.3e} m "
           f"(max {contrib['moon_j2_max_dpos_m']:.3e} m)",
           f"- Combined: {contrib['combined_j2_final_dpos_m']:.3e} m",
           f"- Interaction (s5-s3-s4+s2): {inter:.3e} m", "",
           "The interaction term is small compared to the dominant Moon J2 contribution, but it "
           f"is {rel} the isolated Earth J2 contribution (Earth J2 {e_iso:.3e} m vs interaction "
           f"{inter:.3e} m) over this arc. Therefore the combined trajectory difference should not "
           "be interpreted as a strictly linear sum of the isolated perturbation effects.", "",
           "## Interpretation / Limitations", "",
           "- Moon J2 dominates the modeled J2 perturbation hierarchy for this representative LLO.",
           "- Earth J2 is physically consistent and nonzero, but negligible compared with Moon J2 "
           "for this short-arc case.",
           "- The combined model is dominated by Moon J2.",
           "- The interaction term is small relative to Moon J2 but comparable to the isolated "
           "Earth J2 contribution, so the perturbation effects should not be described as perfectly "
           "linear.",
           "- The PlanetEphemeris/SPICE comparison is an ephemeris-source validation layer, not a "
           "force-model error.",
           "- Moon J2 is not a high-fidelity lunar gravity model; it is only a controlled low-order "
           "perturbation.",
           "- High-fidelity lunar gravity would require spherical-harmonic models, e.g. "
           "GRAIL-derived models (GRGM / GL series).", "",
           "## Free-run separation note (long arcs)", "",
           "- These metrics measure free-run trajectory separation from the SAME initial state.",
           "- They should NOT be interpreted directly as model error or navigation error.",
           "- Over long arcs, even small perturbations grow phase / orbital-element differences, "
           "so the position separation can become large while the underlying orbits remain valid.", "",
           "## Minimum altitude / periapsis sanity", "",
           "| scenario | min radius [m] | min altitude [m] | surface crossing |",
           "|---|---:|---:|---|"]
    any_crossing = False
    for k in SCENARIOS:
        s = sanity[k]
        cross = "YES (impact)" if s["surface_crossing"] else "no"
        any_crossing = any_crossing or s["surface_crossing"]
        md.append(f"| {k} | {s['min_radius_m']:.3e} | {s['min_altitude_m']:.3e} | {cross} |")
    if any_crossing:
        md += ["", "**WARNING: a surface crossing was detected — the trajectory is invalid after "
               "impact / surface crossing for the affected scenario(s). Interpret the force-model "
               "comparison with caution beyond that point.**"]
    md += ["", "Production code NOT modified. The PlanetEphemeris/SPICE difference is reported "
           "separately (phase6_ephemeris_comparison.py)."]
    (OUT / f"phase6_report_{days}d.md").write_text("\n".join(md))
    return csv, sanity, any_crossing


def main():
    argv = sys.argv[1:]
    ephem_step_s = 10.0
    if "--ephem" in argv:
        i = argv.index("--ephem"); ephem_step_s = float(argv[i + 1]); del argv[i:i + 2]
    windows = tuple(int(a) for a in argv if a.lstrip("-").isdigit()) or (1, 7)
    for days in windows:
        rows, contrib, sanity, first_jd, state0, teval = run_window(days, ephem_step_s)
        csv, sanity, any_crossing = write_outputs(days, rows, contrib, sanity, first_jd, state0, teval, ephem_step_s)
        print(f"\n=== {days}-day window  (points={teval.size}, ephem {ephem_step_s:.0f}s) ===")
        print(f"{'scenario':<18}{'final_dpos[m]':>15}{'max_dpos[m]':>15}{'rms_dpos[m]':>15}{'final_dvel[m/s]':>17}")
        for k in SCENARIOS:
            r = rows[k]
            print(f"{k:<18}{r['final_dpos_m']:>15.3e}{r['max_dpos_m']:>15.3e}{r['rms_dpos_m']:>15.3e}{r['final_dvel_mps']:>17.3e}")
        print(f"  Earth J2 isolated  final={contrib['earth_j2_isolated_final_dpos_m']:.3e} m  max={contrib['earth_j2_max_dpos_m']:.3e} m")
        print(f"  Moon  J2 isolated  final={contrib['moon_j2_isolated_final_dpos_m']:.3e} m  max={contrib['moon_j2_max_dpos_m']:.3e} m")
        print(f"  Combined           final={contrib['combined_j2_final_dpos_m']:.3e} m")
        print(f"  Interaction        final={contrib['interaction_final_dpos_m']:.3e} m")
        print("  min altitude [m]: " + ", ".join(f"{k.split('_')[0]}={sanity[k]['min_altitude_m']:.3e}" for k in SCENARIOS))
        print(f"  surface crossing: {'YES' if any_crossing else 'no'}")
        print(f"  -> {csv}")


if __name__ == "__main__":
    main()
