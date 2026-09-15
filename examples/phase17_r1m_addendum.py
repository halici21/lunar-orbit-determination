"""PHASE 17-R1M addendum - Model-A fairness control, data-only sigma_K, figures.

Three things the main study leaves open, all of which matter for an honest
verdict:

1. FAIRNESS. The main campaign gives Model A 0.5-orbit windows, in which the
   K signature has almost no time to accumulate from the S_K=0 reset at each
   arc start. Model A would look better with longer arcs. The verified Schur
   sum rule already implies its ceiling analytically -- the global conditional
   information can never exceed the sum of the per-arc ones -- but that is an
   argument, and an argument is worth one measurement. So Model A is re-run
   with 2.0-orbit windows, the length at which a SINGLE arc reached
   I_K|x = 2658 in R1G.

2. PHYSICAL UNCERTAINTY. R1M-B showed the estimator-reported sigma_K is the
   eigenvalue floor rather than the data. Every conditional information in
   this study is therefore also reported as the data-only Schur sigma,
   1/sqrt(I_K|x), which IS scale-invariant and IS reproducible.

3. FIGURES required by s54.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase17_r1m_core import (  # noqa: E402
    K_TRUTH, SCALE_K, build_range_arc, campaign_epoch, information_matrix,
    orthogonal_decomposition, scale_matrix, schur_conditional, spectrum,
    whitened,
)
from phase17_r1m_multi_arc_study import (  # noqa: E402
    ALL_STATIONS, model_a_information, model_b_information,
)

ARTIFACTS = Path(r"C:/Users/erayh/Documents/Python/Grad/python_port_phase17r0/artifacts")
#: Model-A fairness control: same 2-arc / 3-arc pattern, but arcs long enough
#: that a single one of them would already be R1G's best stable case.
#: Starts are spaced so that 2.0-orbit windows leave a real 0.5-orbit gap and
#: never share an endpoint -- at 0.50/2.50 the first two windows would touch
#: exactly at t=2.50 and Model A would count that observation twice.
FAIR_WINDOW_ORBITS = 2.0
FAIR_CAMPAIGNS = {"F2": [0.50, 3.00], "F3": [0.50, 3.00, 5.50]}


def hdr(t):
    print()
    print("=" * 86)
    print(t)
    print("=" * 86)


def sigma_from_information(i_k_given_x: float) -> float:
    return float("inf") if i_k_given_x <= 0 else float(1.0 / np.sqrt(i_k_given_x))


def main() -> None:
    from lunar_od.spice_loader import load_spice_kernels

    load_spice_kernels(None, clear=True)
    _, t_orbit = campaign_epoch()
    rows = []

    hdr("MODEL-A FAIRNESS CONTROL  (2.0-orbit arcs)")
    print("  Each arc is now as long as R1G's best stable single-arc case.")
    for name, starts in FAIR_CAMPAIGNS.items():
        window_s = FAIR_WINDOW_ORBITS * t_orbit
        windows = [(s * t_orbit, s * t_orbit + window_s) for s in starts]
        arcs = [build_range_arc(w0, window_s, label="%s_a%d" % (name, i),
                                station_filter=ALL_STATIONS)
                for i, (w0, _) in enumerate(windows)]
        a = model_a_information(arcs)
        long_arc = build_range_arc(0.0, windows[-1][1], label="%s_long" % name,
                                   station_filter=ALL_STATIONS)
        b = model_b_information(long_arc, windows)
        per_arc = []
        for arc in arcs:
            ax, bk = whitened(arc.h_x0, arc.h_k, arc.w)
            per_arc.append(orthogonal_decomposition(ax, bk)["i_k_given_x"])
        elapsed = (windows[-1][1] - windows[0][0]) / t_orbit
        print("\n  %s  %d arcs of %.1f orbits, elapsed %.2f orbits"
              % (name, len(arcs), FAIR_WINDOW_ORBITS, elapsed))
        print("    per-arc I_K|x        : %s"
              % ", ".join("%.4e" % v for v in per_arc))
        print("    MODEL A  I_K|x=%.6e  sigma_K=%.6e  f_perp=%.6f  n_obs=%d"
              % (a["conditional_k"], sigma_from_information(a["conditional_k"]),
                 a["decomposition"]["orthogonal_fraction"], a["n_obs"]))
        print("    MODEL B  I_K|x=%.6e  sigma_K=%.6e  f_perp=%.6f  n_obs=%d"
              % (b["conditional_k"], sigma_from_information(b["conditional_k"]),
                 b["decomposition"]["orthogonal_fraction"], b["n_obs"]))
        print("    B/A information ratio: %.3e"
              % (b["conditional_k"] / max(a["conditional_k"], 1e-300)))
        for model, res in (("A_independent_arcs", a), ("B_continuity_linked", b)):
            rows.append(dict(
                case=name, model=model, arcs=len(arcs),
                window_orbits=FAIR_WINDOW_ORBITS,
                elapsed_orbits=round(elapsed, 4), observations=res["n_obs"],
                unknowns=res["n_unknown"],
                conditional_k_information=res["conditional_k"],
                sigma_k_data_only=sigma_from_information(res["conditional_k"]),
                sigma_k_over_k_truth=sigma_from_information(res["conditional_k"]) / K_TRUTH,
                orthogonal_fraction=res["decomposition"]["orthogonal_fraction"],
                weakest_mode_k_component=res["spectrum"]["weakest_k_component"],
                rank=res["spectrum"]["rank_default"]))

    with (ARTIFACTS / "r1m_model_a_fairness.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n  wrote r1m_model_a_fairness.csv (%d rows)" % len(rows))

    # ---- physical sigma_K for every case already computed ------------
    hdr("DATA-ONLY sigma_K FOR EVERY CASE  (scale-invariant, reproducible)")
    single = json.loads((ARTIFACTS / "r1m_multi_arc_results.json").read_text())
    sig_rows = []
    for r in single["single_arc"]:
        s = sigma_from_information(r["conditional_k_information"])
        sig_rows.append(dict(case=r["case"], model="single_contiguous",
                             elapsed_orbits=r["arc_orbits"],
                             observations=r["observations"],
                             conditional_k_information=r["conditional_k_information"],
                             sigma_k_data_only=s,
                             sigma_k_over_k_truth=s / K_TRUTH))
    for r in single["multi_arc"]:
        s = sigma_from_information(r["conditional_k_information"])
        sig_rows.append(dict(case=r["case"], model=r["model"],
                             elapsed_orbits=r["elapsed_orbits"],
                             observations=r["observations"],
                             conditional_k_information=r["conditional_k_information"],
                             sigma_k_data_only=s,
                             sigma_k_over_k_truth=s / K_TRUTH))
    print("  %-6s %-22s %8s %6s %14s %14s %10s"
          % ("case", "model", "elapsed", "n_obs", "I_K|x", "sigma_K", "sig/K"))
    for r in sig_rows:
        print("  %-6s %-22s %8.2f %6d %14.6e %14.6e %10.2f"
              % (r["case"], r["model"], r["elapsed_orbits"], r["observations"],
                 r["conditional_k_information"], r["sigma_k_data_only"],
                 r["sigma_k_over_k_truth"]))
    with (ARTIFACTS / "r1m_sigma_k_data_only.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sig_rows[0].keys()))
        w.writeheader()
        w.writerows(sig_rows)
    print("  wrote r1m_sigma_k_data_only.csv (%d rows)" % len(sig_rows))

    # ---- figures (s54) -----------------------------------------------
    hdr("FIGURES")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cov = json.loads((ARTIFACTS / "r1m_covariance_diagnostic.json").read_text())

    # 1. sigma_K by covariance method
    fig, ax = plt.subplots(figsize=(8, 4.5))
    methods = ["sigma_k_estimator", "sigma_k_direct", "sigma_k_svd",
               "sigma_k_schur", "sigma_k_full_posterior"]
    labels = ["estimator\n(floored)", "direct\ninverse", "SVD\npseudoinverse",
              "data-only\nSchur", "full\nposterior"]
    x = np.arange(len(methods))
    for off, case in zip((-0.18, 0.18), ("G0", "G1")):
        vals = [cov["cases"][case][m] for m in methods]
        ax.bar(x + off, vals, width=0.34, label=case)
    ax.axhline(K_TRUTH, ls="--", c="k", lw=1, label="K_truth = 0.01")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("sigma_K  [m^2/kg]")
    ax.set_title("R1M-B: sigma_K depends entirely on which inverse is used")
    ax.legend(); fig.tight_layout()
    fig.savefig(ARTIFACTS / "r1m_sigma_k_by_method.png", dpi=110); plt.close(fig)

    # 2. eigen spectrum vs floor
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for case, mk in (("G0", "o"), ("G1", "s")):
        c = cov["cases"][case]
        ax.axhline(c["floor"], ls=":", lw=1)
        ax.plot([0], [c["smallest_eigenvalue"]], mk, label="%s smallest eig" % case)
        ax.plot([1], [c["largest_eigenvalue"]], mk, label="%s largest eig" % case)
        ax.plot([0.5], [c["floor"]], "x", ms=10,
                label="%s floor = max*1e-14" % case)
    ax.set_yscale("log"); ax.set_xticks([])
    ax.set_ylabel("scaled eigenvalue")
    ax.set_title("R1M-B: the floor sits far ABOVE the true smallest eigenvalue")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(ARTIFACTS / "r1m_eigen_spectrum_vs_floor.png", dpi=110); plt.close(fig)

    # 3. orthogonal fraction vs contiguous arc length
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sa = single["single_arc"]
    ax.plot([r["arc_orbits"] for r in sa],
            [r["orthogonal_fraction"] for r in sa], "o-")
    for r in sa:
        ax.annotate(r["case"], (r["arc_orbits"], r["orthogonal_fraction"]),
                    textcoords="offset points", xytext=(5, -9), fontsize=8)
    ax.set_xlabel("contiguous arc length  [orbits]")
    ax.set_ylabel("orthogonal fraction  f_perp")
    ax.set_title("R1M-C: K's independent signature fraction saturates near 0.27")
    ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(ARTIFACTS / "r1m_orthogonal_fraction_vs_arc.png", dpi=110); plt.close(fig)

    # 4/5/6. multi-arc comparisons
    ma = single["multi_arc"]
    cases = ["M1", "M2", "M3"]
    a_vals = [next(r for r in ma if r["case"] == c and r["model"] == "A_independent_arcs")
              for c in cases]
    b_vals = [next(r for r in ma if r["case"] == c and r["model"] == "B_continuity_linked")
              for c in cases]
    n_arcs = [r["arcs"] for r in a_vals]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(n_arcs, [r["conditional_k_information"] for r in a_vals], "o-",
                label="Model A  pure multi-arc")
    ax.semilogy(n_arcs, [r["conditional_k_information"] for r in b_vals], "s-",
                label="Model B  continuity-linked")
    ax.set_xlabel("number of separated arcs"); ax.set_ylabel("conditional K information")
    ax.set_title("R1M-D/E: continuity is what preserves K information")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(ARTIFACTS / "r1m_conditional_k_vs_arcs.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(n_arcs, [r["orthogonal_fraction"] for r in a_vals], "o-",
            label="Model A (flat -> accumulation only)")
    ax.plot(n_arcs, [r["orthogonal_fraction"] for r in b_vals], "s-",
            label="Model B (rises -> genuine rotation)")
    ax.set_xlabel("number of separated arcs"); ax.set_ylabel("orthogonal fraction f_perp")
    ax.set_title("R1M-33: magnitude increase vs rotation out of the state subspace")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(ARTIFACTS / "r1m_orthogonal_fraction_vs_arcs.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(n_arcs, [1 - r["weakest_mode_k_component"] for r in a_vals], "o-",
            label="Model A")
    ax.plot(n_arcs, [1 - r["weakest_mode_k_component"] for r in b_vals], "s-",
            label="Model B")
    ax.set_yscale("log")
    ax.set_xlabel("number of separated arcs")
    ax.set_ylabel("1 - |K component of weakest mode|")
    ax.set_title("R1M: the weakest mode stays almost exactly K in both models")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(ARTIFACTS / "r1m_weakest_mode_k_vs_arcs.png", dpi=110); plt.close(fig)

    # 7. elapsed-time control
    ctrl = single["elapsed_control"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    idx = np.arange(len(ctrl))
    ax.bar(idx - 0.18, [r["contiguous_conditional_k"] for r in ctrl], 0.34,
           label="contiguous, same tracking time")
    ax.bar(idx + 0.18, [r["linked_conditional_k"] for r in ctrl], 0.34,
           label="continuity-linked, spread over gaps")
    ax.set_yscale("log"); ax.set_xticks(idx)
    ax.set_xticklabels(["%s (%d arcs)" % (r["case"], r["arcs"]) for r in ctrl])
    ax.set_ylabel("conditional K information")
    ax.set_title("R1M-s30: elapsed span buys information that more data does not")
    ax.legend(); ax.grid(alpha=.3, axis="y"); fig.tight_layout()
    fig.savefig(ARTIFACTS / "r1m_elapsed_time_control.png", dpi=110); plt.close(fig)

    print("  wrote 7 figures")


if __name__ == "__main__":
    main()
