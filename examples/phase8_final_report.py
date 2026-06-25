"""Phase 8 — final report + visualization package (reporting only).

Reads the existing Phase 6/7 summary outputs (CSV/JSON) under results/phase6/ and
produces aggregate plots (results/phase6/plots/) plus a consolidated markdown
report (results/phase6/phase6_final_report.md).  Does NOT re-run any campaign, does
NOT modify any existing CSV/JSON/MD, and touches no production code.

Dense per-step trajectories were intentionally not saved, so time-series
separation plots cannot be produced from existing data (reported, not regenerated).
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase6"
PLOTS = OUT / "plots"
DURATIONS = [1, 7, 30, 120, 360]
SCEN = {"1_keplerian": "Keplerian", "3_earth_j2": "Earth J2",
        "4_moon_j2": "Moon J2", "5_earth_moon_j2": "Combined"}


def load_force(d):
    rows = {}
    with open(OUT / f"phase6_force_comparison_{d}d.csv") as f:
        for row in csv.DictReader(f):
            rows[row["scenario"]] = {k: float(v) for k, v in row.items() if k != "scenario"}
    return rows


def load_meta(d):
    return json.load(open(OUT / f"phase6_metadata_{d}d.json"))


FORCE = {d: load_force(d) for d in DURATIONS}
META = {d: load_meta(d) for d in DURATIONS}
INTER = [META[d]["isolated_contributions"]["interaction_final_dpos_m"] for d in DURATIONS]
EARTH = [FORCE[d]["3_earth_j2"]["final_dpos_m"] for d in DURATIONS]
MOON = [FORCE[d]["4_moon_j2"]["final_dpos_m"] for d in DURATIONS]
COMB = [FORCE[d]["5_earth_moon_j2"]["final_dpos_m"] for d in DURATIONS]


def _ser(scn, col):
    return [FORCE[d][scn][col] for d in DURATIONS]


def _line_plot(col, ylabel, title, fname):
    plt.figure(figsize=(7, 4.5))
    for scn, lab in SCEN.items():
        plt.loglog(DURATIONS, _ser(scn, col), marker="o", label=lab)
    plt.xlabel("duration [days]"); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, which="both", alpha=0.3); plt.legend()
    plt.savefig(PLOTS / fname, dpi=120, bbox_inches="tight"); plt.close()


def make_plots():
    PLOTS.mkdir(parents=True, exist_ok=True)
    made = []
    # 1-4: per-scenario vs duration
    _line_plot("final_dpos_m", "final position diff [m] (vs Third-Body)",
               "Final position difference vs duration", "final_position_vs_duration.png")
    _line_plot("max_dpos_m", "max position diff [m]",
               "Max position difference vs duration (note Moon-J2 oscillation)", "max_position_vs_duration.png")
    _line_plot("rms_dpos_m", "RMS position diff [m]",
               "RMS position difference vs duration", "rms_position_vs_duration.png")
    _line_plot("final_dvel_mps", "final velocity diff [m/s]",
               "Final velocity difference vs duration", "final_velocity_vs_duration.png")
    made += ["final_position_vs_duration.png", "max_position_vs_duration.png",
             "rms_position_vs_duration.png", "final_velocity_vs_duration.png"]

    # 5: J2 contributions
    plt.figure(figsize=(7, 4.5))
    plt.loglog(DURATIONS, EARTH, marker="o", label="Earth J2 isolated")
    plt.loglog(DURATIONS, MOON, marker="s", label="Moon J2 isolated")
    plt.loglog(DURATIONS, COMB, marker="^", label="Combined J2")
    plt.loglog(DURATIONS, INTER, marker="d", label="Interaction term")
    plt.xlabel("duration [days]"); plt.ylabel("final position contribution [m]")
    plt.title("J2 perturbation contributions vs duration\n(Moon J2 dominates; Earth J2 small; interaction not always < Earth J2)")
    plt.grid(True, which="both", alpha=0.3); plt.legend()
    plt.savefig(PLOTS / "j2_contributions_vs_duration.png", dpi=120, bbox_inches="tight"); plt.close()
    made.append("j2_contributions_vs_duration.png")

    # 6: interaction ratios
    plt.figure(figsize=(7, 4.5))
    plt.semilogx(DURATIONS, [i / m for i, m in zip(INTER, MOON)], marker="o", label="|interaction| / |Moon J2|")
    plt.semilogx(DURATIONS, [i / e for i, e in zip(INTER, EARTH)], marker="s", label="|interaction| / |Earth J2|")
    plt.axhline(1.0, color="k", ls="--", alpha=0.5, lw=1)
    plt.xlabel("duration [days]"); plt.ylabel("ratio"); plt.yscale("log")
    plt.title("Interaction-term ratios vs duration")
    plt.grid(True, which="both", alpha=0.3); plt.legend()
    plt.savefig(PLOTS / "interaction_ratios_vs_duration.png", dpi=120, bbox_inches="tight"); plt.close()
    made.append("interaction_ratios_vs_duration.png")

    # 7: earth/moon ratio
    plt.figure(figsize=(7, 4.5))
    plt.loglog(DURATIONS, [e / m for e, m in zip(EARTH, MOON)], marker="o", color="tab:red")
    plt.xlabel("duration [days]"); plt.ylabel("|Earth J2| / |Moon J2|")
    plt.title("Earth J2 vs Moon J2 isolated-contribution ratio\n(Earth J2 stays orders of magnitude below Moon J2)")
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(PLOTS / "earth_vs_moon_j2_ratio.png", dpi=120, bbox_inches="tight"); plt.close()
    made.append("earth_vs_moon_j2_ratio.png")

    # 8: ephemeris source comparison
    er = list(csv.DictReader(open(OUT / "phase6_ephemeris_comparison.csv")))
    ed = [int(r["window_days"]) for r in er]
    plt.figure(figsize=(7, 4.5))
    plt.semilogx(ed, [float(r["earth_pos_max_m"]) for r in er], marker="o", label="Earth pos max diff [m]")
    plt.semilogx(ed, [float(r["sun_pos_max_m"]) for r in er], marker="s", label="Sun pos max diff [m]")
    plt.xlabel("duration [days]"); plt.ylabel("max difference [m]")
    plt.title("Ephemeris Source Comparison (PlanetEphemeris .mat vs SPICE DE421)\nNOT a force-model error — ephemeris-source validation layer")
    plt.grid(True, which="both", alpha=0.3); plt.legend()
    plt.savefig(PLOTS / "ephemeris_source_comparison.png", dpi=120, bbox_inches="tight"); plt.close()
    made.append("ephemeris_source_comparison.png")

    # 9: cadence sensitivity (why 60 s was rejected)
    cr = list(csv.DictReader(open(OUT / "phase6_cadence_sensitivity.csv")))
    cw = [r["window"] for r in cr]
    import numpy as np
    x = np.arange(len(cw)); w = 0.35
    plt.figure(figsize=(7, 4.5))
    plt.bar(x - w / 2, [float(r["dEarthJ2_m"]) for r in cr], w, label="60s-vs-10s |dEarth J2| [m]")
    plt.bar(x + w / 2, [float(r["dInteraction_m"]) for r in cr], w, label="60s-vs-10s |dInteraction| [m]")
    # overlay the actual small-signal values at 30d/120d for comparison
    sig_e = [EARTH[DURATIONS.index(int(wd[:-1]))] for wd in cw]
    sig_i = [INTER[DURATIONS.index(int(wd[:-1]))] for wd in cw]
    plt.plot(x - w / 2, sig_e, "k_", ms=22, mew=2, label="actual Earth J2 signal [m]")
    plt.plot(x + w / 2, sig_i, "r_", ms=22, mew=2, label="actual interaction signal [m]")
    plt.xticks(x, cw); plt.ylabel("[m]")
    plt.title("Ephemeris cadence sensitivity: 10 s vs 60 s\n60 s noise approaches/exceeds the small signals -> 10 s kept for 360 d")
    plt.grid(True, axis="y", alpha=0.3); plt.legend(fontsize=8)
    plt.savefig(PLOTS / "cadence_sensitivity.png", dpi=120, bbox_inches="tight"); plt.close()
    made.append("cadence_sensitivity.png")

    # 10: min altitude sanity (only 120d / 360d have the data)
    alt_windows = [d for d in DURATIONS if "min_altitude_sanity" in META[d]]
    if alt_windows:
        scen_keys = list(SCEN.keys()) + ["2_third_body"]
        scen_keys = ["1_keplerian", "2_third_body", "3_earth_j2", "4_moon_j2", "5_earth_moon_j2"]
        xs = np.arange(len(scen_keys)); w2 = 0.8 / len(alt_windows)
        plt.figure(figsize=(7.5, 4.5))
        for i, d in enumerate(alt_windows):
            sa = META[d]["min_altitude_sanity"]
            vals = [sa[k]["min_altitude_m"] / 1e3 for k in scen_keys]
            plt.bar(xs + i * w2, vals, w2, label=f"{d}d")
        plt.axhline(0.0, color="r", ls="--", lw=1, label="surface (0 km)")
        plt.xticks(xs + w2 * (len(alt_windows) - 1) / 2, [k.split("_", 1)[1] for k in scen_keys], rotation=20)
        plt.ylabel("min altitude [km]")
        plt.title("Minimum altitude per scenario (no surface crossing)\n(min-altitude sanity available for 120 d / 360 d only)")
        plt.grid(True, axis="y", alpha=0.3); plt.legend()
        plt.savefig(PLOTS / "min_altitude_sanity.png", dpi=120, bbox_inches="tight"); plt.close()
        made.append("min_altitude_sanity.png")
    return made, alt_windows


def fmt(x):
    return f"{x:.3e}"


def write_final_report(made, alt_windows):
    L = []
    A = L.append
    A("# Phase 8 — Final Report: Lunar J2 Force-Model Verification & Comparison Campaign")
    A("")
    A("Consolidates Phase 0-7C. Reporting only; no production code, physics, or campaign re-run.")
    A("")
    # 1 executive summary
    A("## 1. Executive summary")
    A("")
    A("- **Moon J2 dominates** the modeled J2 perturbation hierarchy for this representative LLO "
      f"(Moon-J2 isolated final separation grows to ~{MOON[3]/1e3:.0f} km at 120 d).")
    A("- **Earth J2 is nonzero but negligible** relative to Moon J2 "
      f"(Earth-J2 final separation {fmt(EARTH[3])} m vs Moon-J2 {fmt(MOON[3])} m at 120 d; ~3-4 orders smaller).")
    A("- **Interaction is small relative to Moon J2 but can be comparable to Earth J2** "
      "(it should not be called negligible without that reference).")
    A("- **PlanetEphemeris .mat and SPICE DE421 agree at cm-level** for Earth position over all tested windows "
      f"(<= {1.49e-2*100:.1f} cm).")
    A("- **Moon J2 is NOT high-fidelity lunar gravity** — only a controlled low-order perturbation.")
    A("- **Next scientific step is GRAIL-based spherical-harmonic lunar gravity.**")
    A("")
    # 2 model hierarchy
    A("## 2. Model hierarchy")
    A("")
    A("1. Keplerian point-mass (Moon only; Earth/Sun off via mu=0)")
    A("2. + Third-body (Earth + Sun point-mass, indirect form) — comparison baseline")
    A("3. + Earth J2 (indirect, Moon-centered)")
    A("4. + Moon J2 (zonal, mean-pole body-fixed)")
    A("5. + Earth J2 + Moon J2 (combined)")
    A("6. Ephemeris source comparison layer (PlanetEphemeris .mat vs SPICE DE421) — separate, not a force model")
    A("")
    # 3 verification history
    A("## 3. Verification history")
    A("")
    A("| Phase | What | Result |")
    A("|---|---|---|")
    A("| 0 | Existing force-model verification (Py<->Numba, STM/FD, third-body, ephemeris) | PASS (machine precision) |")
    A("| 2 | Constants centralization (constants.py) | PASS (behavior bit-identical) |")
    A("| 3 | Generic Body-J2 helper (force_models.py + njit) | PASS (helper == existing path) |")
    A("| 4 | Moon J2 refactor onto the helper (both paths) | PASS (Phase 0 numbers unchanged) |")
    A("| 5 | Earth J2 implementation (indirect default) + config flags | PASS (Earth-off bit-identical) |")
    A("| 5B | Earth J2 hardening (magnitude, sign, FD, dual-tolerance, smoke) | PASS (0 fails) |")
    A("")
    # 4 campaign results
    A("## 4. Scenario campaign results (final position difference vs Third-Body baseline)")
    A("")
    A("| duration | Keplerian [m] | Earth J2 [m] | Moon J2 [m] | Combined [m] | Interaction [m] |")
    A("|---|---:|---:|---:|---:|---:|")
    for i, d in enumerate(DURATIONS):
        A(f"| {d}d | {fmt(FORCE[d]['1_keplerian']['final_dpos_m'])} | {fmt(EARTH[i])} | "
          f"{fmt(MOON[i])} | {fmt(COMB[i])} | {fmt(INTER[i])} |")
    A("")
    A(f"Moon-J2 max separation reaches {fmt(FORCE[360]['4_moon_j2']['max_dpos_m'])} m while its 360-day "
      f"*final* separation is only {fmt(MOON[4])} m — the separation is oscillatory (phase wrapping), not monotonic.")
    A("")
    # 5 interpretation
    A("## 5. Interpretation of the force-model hierarchy")
    A("")
    A("- Moon J2 dominates the modeled J2 hierarchy.")
    A("- Earth J2 is physically consistent but practically negligible in this representative LLO.")
    A("- The combined model is Moon-J2 dominated.")
    A("- Long-arc free-run separation is NOT a direct navigation or model error; it is trajectory "
      "separation from a shared initial state.")
    A("- Moon-J2 separation can be oscillatory: the final separation can be much smaller than the peak.")
    A("- The interaction term should not be called zero/negligible without a reference: it is small "
      "relative to Moon J2 but can be comparable to (or exceed) Earth J2 over medium arcs.")
    A("")
    # 6 ephemeris
    A("## 6. Ephemeris comparison (source-validation layer)")
    A("")
    A("- Both the `.mat` PlanetEphemeris data and SPICE use the DE421 layer.")
    A("- This comparison is an **ephemeris-source validation layer**, not a force-model error.")
    A("- Earth position max difference stays at cm-level over 1-360 days (<= 1.49 cm); Sun sub-metre.")
    A("")
    # 7 cadence
    A("## 7. Cadence sensitivity (preflight)")
    A("")
    A("- A 60 s ephemeris thinning was tested against the native 10 s grid.")
    A(f"- It distorted the small Earth-J2 / interaction signals (e.g. at 120 d it changed Earth J2 by "
      f"~32 m and interaction by ~105 m — larger than the signals themselves).")
    A("- Therefore the 360-day full campaign used the **native 10 s grid**, protecting small-signal interpretation.")
    A("")
    # 8 limitations
    A("## 8. Limitations")
    A("")
    A("- Moon J2 is a low-order perturbation, not a full lunar gravity model.")
    A("- Real lunar gravity is strongly shaped by mascons and longitude-dependent harmonics.")
    A("- High-fidelity lunar navigation requires spherical-harmonic gravity models (GRAIL-derived GRGM/GL).")
    A("- The Earth frame for Earth J2 is a fixed/identity (J2000 mean-equator) approximation, not high-fidelity Earth orientation.")
    A("- Long free-run separations are sensitive to phase drift and must not be read as OD error.")
    A("- The full 360-day run is computationally expensive (~4.7 h at native 10 s resolution).")
    A("")
    # 9 next steps
    A("## 9. Recommended next steps")
    A("")
    A("1. Commit the current validated J2 framework and campaign scripts (branch `feature/lunar-j2-force-models`).")
    A("2. Build the final figure/table package for the thesis/report (this phase).")
    A("3. Plan a Lunar Spherical-Harmonics gravity architecture.")
    A("4. Add a GRAIL model loader and a low-degree harmonic engine.")
    A("5. Only later consider OD-pipeline threading / Phase 6B if Earth-J2 toggling in the estimator is needed.")
    A("")
    A("## Figures")
    A("")
    for m in made:
        A(f"- `plots/{m}`")
    if 30 not in alt_windows:
        A("")
        A("_Note: min-altitude sanity is available only for 120 d / 360 d (the feature was added in Phase 7B); "
          "1/7/30 d would need a cheap re-run to populate it. Dense time-series separation was not saved by "
          "design, so time-series plots are not available without a (separately approved) re-run._")
    (OUT / "phase6_final_report.md").write_text("\n".join(L))


def main():
    made, alt_windows = make_plots()
    write_final_report(made, alt_windows)
    print("Plots written to results/phase6/plots/:")
    for m in made:
        print(f"  - {m}")
    print(f"Final report: results/phase6/phase6_final_report.md")
    print(f"min-altitude data available for windows: {alt_windows}")


if __name__ == "__main__":
    main()
