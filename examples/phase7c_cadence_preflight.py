"""Phase 7C preflight — ephemeris cadence sensitivity (10 s vs 60 s).

Before the 360-day run, verify that subsampling the .mat ephemeris from 10 s to
60 s does not change the force-model comparison at report scale.  Runs the same
scenarios at both cadences for 30-day and 120-day windows and compares the
reported quantities.  Writes results/phase6/phase6_cadence_sensitivity.{csv,md}.
NO production code; reuses the campaign module.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from phase6_scenario_comparison import (  # noqa: E402
    SCENARIOS, initial_state, load_ephemeris, propagate, CADENCE_S,
)
from lunar_od.constants import R_MOON_M  # noqa: E402

OUT = ROOT / "results" / "phase6"


def run(days, ephem_step_s):
    T = days * 86400.0
    teval = np.arange(0.0, T + 1.0, CADENCE_S)
    eph, _ = load_ephemeris(T, ephem_step_s)
    s0 = initial_state()
    tr = {k: propagate(eph, s0, teval, sc) for k, sc in SCENARIOS.items()}
    base = tr["2_third_body"]
    e_iso = float(np.linalg.norm((tr["3_earth_j2"] - base)[-1, :3]))
    m_iso = float(np.linalg.norm((tr["4_moon_j2"] - base)[-1, :3]))
    comb = float(np.linalg.norm((tr["5_earth_moon_j2"] - base)[-1, :3]))
    inter = float(np.linalg.norm((tr["5_earth_moon_j2"] - tr["3_earth_j2"] - tr["4_moon_j2"] + base)[-1, :3]))
    min_alt = float(np.linalg.norm(tr["5_earth_moon_j2"][:, :3], axis=1).min()) - R_MOON_M
    return dict(base_final=base[-1, :3].copy(), e_iso=e_iso, m_iso=m_iso,
               comb=comb, inter=inter, min_alt=min_alt)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [("window", "base_traj_diff_m", "dEarthJ2_m", "dMoonJ2_m",
             "dCombined_m", "dInteraction_m", "dMinAlt_m")]
    md_rows = []
    worst_science = 0.0
    for days in (30, 120):
        a = run(days, 10.0); b = run(days, 60.0)
        base_diff = float(np.linalg.norm(a["base_final"] - b["base_final"]))
        d_e = abs(a["e_iso"] - b["e_iso"]); d_m = abs(a["m_iso"] - b["m_iso"])
        d_c = abs(a["comb"] - b["comb"]); d_i = abs(a["inter"] - b["inter"])
        d_alt = abs(a["min_alt"] - b["min_alt"])
        # "science" = the small reported contributions whose interpretation could change
        worst_science = max(worst_science, d_e, d_i)
        rows.append((f"{days}d", f"{base_diff:.3e}", f"{d_e:.3e}", f"{d_m:.3e}",
                     f"{d_c:.3e}", f"{d_i:.3e}", f"{d_alt:.3e}"))
        md_rows.append(f"| {days}d | {base_diff:.3e} | {d_e:.3e} | {d_m:.3e} | "
                       f"{d_c:.3e} | {d_i:.3e} | {d_alt:.3e} |")
        print(f"{days}d: base_traj_diff={base_diff:.3e} m | dEarthJ2={d_e:.3e} | dMoonJ2={d_m:.3e} | "
              f"dCombined={d_c:.3e} | dInteraction={d_i:.3e} | dMinAlt={d_alt:.3e}")

    use_60 = worst_science < 1.0   # Earth-J2 / interaction (the small science) stable to < 1 m
    decision = "60 s" if use_60 else "10 s"
    print(f"\nworst science-metric (Earth J2 / interaction) cadence diff = {worst_science:.3e} m")
    print(f"DECISION for 360-day run: ephemeris cadence = {decision}")

    (OUT / "phase6_cadence_sensitivity.csv").write_text(
        "\n".join(",".join(r) for r in rows) + "\n")
    md = ["# Phase 7C — Ephemeris Cadence Sensitivity (10 s vs 60 s)", "",
          "Absolute difference between the 10 s and 60 s ephemeris results "
          "(same scenarios, same initial state).", "",
          "| window | base traj diff [m] | dEarthJ2 [m] | dMoonJ2 [m] | dCombined [m] | "
          "dInteraction [m] | dMinAlt [m] |", "|---|---:|---:|---:|---:|---:|---:|",
          *md_rows, "",
          f"Worst small-science (Earth J2 / interaction) cadence difference: {worst_science:.3e} m.", "",
          f"**Decision: use {decision} ephemeris cadence for the 360-day run.** "
          + ("The 10 s -> 60 s subsampling changes the small Earth-J2 / interaction "
             "contributions by < 1 m, i.e. below the report scale and without changing any "
             "interpretation; Moon J2 and combined are dominated by the orbit and shift only "
             "in proportion to the common baseline, which cancels in the isolated contributions."
             if use_60 else
             "The subsampling changes a small reported contribution by >= 1 m, so the full 10 s "
             "grid is retained for the 360-day run."), "",
          "Note: 'base traj diff' is the raw third-body trajectory shift from cadence change; it is "
          "a common-mode shift that largely cancels in the isolated J2 contributions (which are "
          "differences vs the same-cadence baseline)."]
    (OUT / "phase6_cadence_sensitivity.md").write_text("\n".join(md))
    print(f"-> {OUT / 'phase6_cadence_sensitivity.csv'}")
    return decision


if __name__ == "__main__":
    main()
