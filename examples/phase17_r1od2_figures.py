"""PHASE 17-R1O-D2 s48 - figures, drawn ONLY from the D2 result artifacts.

Every title states what the figure represents: measurement geometry,
information direction, information magnitude, systematic-bias sensitivity or
derivative qualification.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ART = Path(__file__).resolve().parent.parent / "artifacts"
OKABE = {"GC": "#0072B2", "GM": "#E69F00", "range": "#555555", "surr": "#009E73", "b3": "#D55E00",
         "b4": "#CC79A7"}
PRIMARY_ORDER = ["CANON", "E1_FIRST_QUARTER_PLUS", "E2_HALF_SYNODIC", "E3_THREE_QUARTER",
                 "O1_LOW_BETA_ECLIPSE_RICH", "O2_FULL_SUN_HIGH_BETA", "O3_HIGH_ALTITUDE_500KM",
                 "O4_POLAR", "O5_FAVOURABLE_GEO_MAX_FPERP"]
SHORT = {"CANON": "CANON", "E1_FIRST_QUARTER_PLUS": "E1", "E2_HALF_SYNODIC": "E2",
         "E3_THREE_QUARTER": "E3", "O1_LOW_BETA_ECLIPSE_RICH": "O1 low-beta",
         "O2_FULL_SUN_HIGH_BETA": "O2 full-Sun", "O3_HIGH_ALTITUDE_500KM": "O3 500 km",
         "O4_POLAR": "O4 polar", "O5_FAVOURABLE_GEO_MAX_FPERP": "O5 GEO-max"}


def _save(fig, name):
    fig.savefig(ART / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main() -> None:
    R = json.loads((ART / "phase17_r1od2_results.json").read_text())["geometries"]
    fd = json.loads((ART / "phase17_r1od2_fd.json").read_text()) if (ART / "phase17_r1od2_fd.json").exists() else None
    c = R["CANON"]["pairs"]

    # 1/2 quasar separation and position angle (systematic-bias sensitivity; formal direction invariant)
    for key, xlab, fname in (("sep_rad", "spacecraft-quasar separation (rad)", "phase17_r1od2_quasar_separation.png"),
                             ("pa_deg", "quasar position angle from celestial north (deg)", "phase17_r1od2_quasar_position_angle.png")):
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
        for ax, real in zip(axs, ("DENSE_5NRAD", "OPS_EVERY_WINDOW")):
            for p in ("GC", "GM"):
                q = [x for x in c[p]["quasar_systematics"] if x["realization"] == real]
                if key == "sep_rad":
                    xs = sorted({x["sep_rad"] for x in q})
                    tot = [np.median([x["k_bias_rss_total_frac"] for x in q if x["sep_rad"] == s]) for s in xs]
                    sl = [np.median([x["k_bias_rss_by_group_frac"]["station_location"] for x in q if x["sep_rad"] == s]) for s in xs]
                    ax.plot(xs, 100 * np.array(tot), "o-", color=OKABE[p], label="%s total (median over PA)" % p)
                    ax.plot(xs, 100 * np.array(sl), "s--", color=OKABE[p], alpha=0.6, label="%s station location only" % p)
                else:
                    ref = [x for x in q if abs(x["sep_rad"] - 0.1) < 1e-12]
                    ax.plot([x["pa_deg"] for x in ref], [100 * x["k_bias_rss_total_frac"] for x in ref], "o-",
                            color=OKABE[p], label="%s (sep 0.1 rad)" % p)
                ax.axhline(100 * q[0]["formal_sigma_k_frac"], color=OKABE[p], lw=0.8, ls=":",
                           label="%s formal sigma_K/K" % p)
            ax.set_xlabel(xlab)
            ax.set_ylabel("1-sigma systematic K bias (% of K)")
            ax.set_title(real, fontsize=9)
            _style(ax)
        axs[0].legend(fontsize=7)
        fig.suptitle("SYSTEMATIC-BIAS SENSITIVITY vs quasar geometry (CANON). Formal information direction "
                     "is quasar-invariant (rows do not depend on the quasar).", fontsize=9)
        _save(fig, fname)

    # 3 combined f_perp by station pair (information direction)
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    x = np.arange(len(PRIMARY_ORDER))
    ax.scatter(x, [R[k]["range_only"]["f_perp"] for k in PRIMARY_ORDER], marker="_", s=400,
               color=OKABE["range"], label="range only (same geometry)", zorder=3)
    for off, p in ((-0.15, "GC"), (0.15, "GM")):
        vals = [R[k]["pairs"][p]["dense_angular"]["5"]["f_perp"] if "dense_angular" in R[k]["pairs"][p]
                else np.nan for k in PRIMARY_ORDER]
        ax.bar(x + off, vals, width=0.3, color=OKABE[p], label="range + ΔDOR %s (5 nrad)" % p)
    ax.set_xticks(x, [SHORT[k] for k in PRIMARY_ORDER], rotation=20)
    ax.set_ylabel("f_perp")
    ax.set_title("INFORMATION DIRECTION: combined f_perp by station pair (production-visibility rows). "
                 "E3 GM and every MC case have no dual visibility.", fontsize=9)
    ax.legend(fontsize=8)
    _style(ax)
    _save(fig, "phase17_r1od2_pair_f_perp.png")

    # 4 delta vs baseline/K plane-of-sky angle (geometry -> direction)
    fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
    for p in ("GC", "GM"):
        xs, ys = [], []
        for k in PRIMARY_ORDER:
            pr = R[k]["pairs"][p]
            if "dense_angular" in pr:
                xs.append(pr["geometry"]["baseline_vs_k_plane_of_sky_angle_deg_median"])
                ys.append(pr["dense_angular"]["5"]["delta_f_perp"])
        ax.scatter(xs, ys, color=OKABE[p], label=p)
    ax.axhline(0.05, color="k", lw=0.6, ls="--")
    ax.axhline(0.0, color="k", lw=0.4)
    ax.set_xlabel("median angle between projected baseline and plane-of-sky dr/dK (deg)")
    ax.set_ylabel("delta f_perp (range+ΔDOR - range)")
    ax.set_title("MEASUREMENT GEOMETRY vs INFORMATION DIRECTION (5 nrad)", fontsize=9)
    ax.legend()
    _style(ax)
    _save(fig, "phase17_r1od2_delta_vs_baseline_angle.png")

    # 5 delta vs Sun-Moon-Earth angle (epoch)
    fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
    for p in ("GC", "GM"):
        xs, ys = [], []
        for k in ("CANON", "E1_FIRST_QUARTER_PLUS", "E2_HALF_SYNODIC", "E3_THREE_QUARTER"):
            pr = R[k]["pairs"][p]
            if "dense_angular" in pr:
                xs.append(R[k]["descriptors"]["sun_moon_earth_deg"])
                ys.append(pr["dense_angular"]["5"]["delta_f_perp"])
                ax.annotate(SHORT[k], (xs[-1], ys[-1]), fontsize=7)
        ax.plot(xs, ys, "o", color=OKABE[p], label=p)
    ax.axhline(0.05, color="k", lw=0.6, ls="--")
    ax.set_xlabel("Sun-Moon-Earth angle at mid-arc (deg)")
    ax.set_ylabel("delta f_perp")
    ax.set_title("INFORMATION DIRECTION vs epoch (same inertial orbit, 5 nrad)", fontsize=9)
    ax.legend()
    _style(ax)
    _save(fig, "phase17_r1od2_delta_vs_epoch.png")

    # 6 orbit families
    fams = ["CANON", "O1_LOW_BETA_ECLIPSE_RICH", "O2_FULL_SUN_HIGH_BETA", "O3_HIGH_ALTITUDE_500KM",
            "O4_POLAR", "O5_FAVOURABLE_GEO_MAX_FPERP"]
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    x = np.arange(len(fams))
    ax.bar(x - 0.25, [R[k]["range_only"]["f_perp"] for k in fams], 0.25, color=OKABE["range"], label="range only")
    for off, p in ((0.0, "GC"), (0.25, "GM")):
        ax.bar(x + off, [R[k]["pairs"][p]["dense_angular"]["5"]["f_perp"] for k in fams], 0.25,
               color=OKABE[p], label="range + ΔDOR %s" % p)
    ax.axhline(0.35, color="k", lw=0.6, ls=":", label="GEO moderate gate 0.35")
    ax.set_xticks(x, [SHORT[k] for k in fams], rotation=15)
    ax.set_ylabel("f_perp")
    ax.set_title("INFORMATION DIRECTION across representative orbit geometries (canonical epoch)", fontsize=9)
    ax.legend(fontsize=8)
    _style(ax)
    _save(fig, "phase17_r1od2_orbit_families.png")

    # 7 sigma_K/K vs precision (magnitude) + delta vs precision (direction)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for k, ls in (("CANON", "-"), ("E1_FIRST_QUARTER_PLUS", "--"), ("O5_FAVOURABLE_GEO_MAX_FPERP", ":")):
        for p in ("GC", "GM"):
            pr = R[k]["pairs"][p]
            if "dense_angular" not in pr:
                continue
            s = sorted(pr["dense_angular"], key=float)
            xs = [float(v) for v in s]
            axs[0].plot(xs, [100 * pr["dense_angular"][v]["sigma_k_frac"] for v in s], ls, marker="o",
                        color=OKABE[p], label="%s %s" % (SHORT[k], p))
            axs[1].plot(xs, [pr["dense_angular"][v]["delta_f_perp"] for v in s], ls, marker="o",
                        color=OKABE[p], label="%s %s" % (SHORT[k], p))
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlabel("ΔDOR angular sigma (nrad)")
        _style(ax)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("sigma_K / K (%)")
    axs[0].set_title("INFORMATION MAGNITUDE vs ΔDOR precision", fontsize=9)
    axs[1].set_ylabel("delta f_perp")
    axs[1].axhline(0.05, color="k", lw=0.6, ls="--")
    axs[1].set_title("INFORMATION DIRECTION vs ΔDOR precision (non-monotonic)", fontsize=9)
    axs[1].legend(fontsize=7)
    _save(fig, "phase17_r1od2_precision.png")

    # 8 nuisance
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True, sharey=True)
    for ax, real in zip(axs, ("DENSE_5NRAD", "OPS_EVERY_WINDOW")):
        labels, b0, b3, b4 = [], [], [], []
        for k in PRIMARY_ORDER:
            for p in ("GC", "GM"):
                b = R[k]["pairs"][p].get("bias", {}).get(real)
                if not b:
                    continue
                labels.append("%s %s" % (SHORT[k], p))
                b0.append(b["B0"]["delta_f_perp"])
                b3.append(b["B3"]["delta_f_perp"])
                b4.append(b["B4"]["delta_f_perp"])
        x = np.arange(len(labels))
        ax.bar(x - 0.27, b0, 0.27, color=OKABE["GC"], label="B0 ideal")
        ax.bar(x, b4, 0.27, color=OKABE["b4"], label="B4 session bias, prior 0.033 ns")
        ax.bar(x + 0.27, b3, 0.27, color=OKABE["b3"], label="B3 session bias, unconstrained")
        ax.axhline(0.05, color="k", lw=0.6, ls="--")
        ax.axhline(0.02, color="k", lw=0.4, ls=":")
        ax.set_xticks(x, labels, rotation=75, fontsize=7)
        ax.set_title(real, fontsize=9)
        _style(ax)
    axs[0].set_ylabel("delta f_perp")
    axs[0].legend(fontsize=7)
    fig.suptitle("INFORMATION DIRECTION with / without session-bias calibration freedom", fontsize=9)
    _save(fig, "phase17_r1od2_nuisance.png")

    # 9 production vs surrogate
    fig, ax = plt.subplots(figsize=(4.8, 4.5), constrained_layout=True)
    xs, ys = [], []
    for k in PRIMARY_ORDER:
        for p in ("GC", "GM"):
            s = R[k]["pairs"][p].get("surrogate")
            if s:
                xs.append(s["f_perp_combined_production"])
                ys.append(s["f_perp_combined_surrogate_weights"])
    h = R["CANON"]["elevation_only_historical"]["GC"]["surrogate"]
    ax.scatter(xs, ys, color=OKABE["surr"], label="primary cases")
    ax.scatter([h["f_perp_combined_production"]], [h["f_perp_combined_surrogate_weights"]], marker="*",
               s=120, color="k", label="CANON GC, historical elevation-only rows")
    lim = [0, max(xs + ys) * 1.05]
    ax.plot(lim, lim, "k", lw=0.5)
    ax.set_xlabel("production combined f_perp")
    ax.set_ylabel("independent surrogate combined f_perp")
    ax.set_title("INFORMATION DIRECTION: production vs surrogate", fontsize=9)
    ax.legend(fontsize=7)
    _style(ax)
    _save(fig, "phase17_r1od2_production_vs_surrogate.png")

    # 10 FD sweep
    if fd:
        keys = [k for k in fd if k.startswith("FD")]
        fig, axs = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4), constrained_layout=True, sharey=True)
        for ax, key in zip(np.atleast_1d(axs), keys):
            for n, col in fd[key]["columns"].items():
                steps = np.array(col["steps"], float)
                xs = steps if n == "K_SRP" else (steps if n in ("x0", "y0", "z0") else steps * 1e3)
                ax.loglog(np.arange(len(steps)), col["rel_err"], "o-", label=n)
            kref_path = ART / "phase17_r1od2_fd_k_refinement.json"
            if kref_path.exists():
                kr = json.loads(kref_path.read_text())[key]["rtol_1e-14"]["rel_err"]
                ax.loglog(np.arange(len(kr)), kr, "k*--", label="K_SRP, refined oracle rtol 1e-14")
            ax.axhline(1e-4, color="k", lw=0.6, ls="--")
            ax.axhline(1e-3, color="k", lw=0.6, ls=":")
            ax.set_xscale("linear")
            ax.set_xticks(range(5), ["s1", "s2", "s3", "s4", "s5"])
            ax.set_xlabel("step index s1..s5")
            ax.set_title("%s: %s %s" % (key, fd[key]["geometry"], fd[key]["pair"]), fontsize=9)
            _style(ax)
        np.atleast_1d(axs)[0].set_ylabel("relative error ||FD - analytic|| / ||analytic||")
        np.atleast_1d(axs)[0].legend(fontsize=7)
        fig.suptitle("DERIVATIVE QUALIFICATION: full production ΔDOR design vs end-to-end FD "
                     "(gates: state 1e-4 dashed, K 1e-3 dotted). Steps: position 10/30/100/300/1000 m; "
                     "velocity 0.01..1 m/s; K 1e-4/3e-4/1e-3/3e-3", fontsize=8)
        _save(fig, "phase17_r1od2_fd_sweep.png")


if __name__ == "__main__":
    main()
