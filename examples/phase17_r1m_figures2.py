"""PHASE 17-R1M - the remaining s54 figures.

Three plots the first figure pass did not cover:

  r1m_smallest_singular_vs_arcs.png   s54 item 5
  r1m_pure_vs_linked.png              s54 item 7, as a dedicated comparison
  r1m_model_a_window_length.png       the fairness control -- not on the s54
                                      list, but the single most important
                                      figure in the phase, because it shows
                                      that the headline "pure multi-arc loses
                                      10^6x" is a property of the WINDOW
                                      LENGTH I chose, not of pure multi-arc.

Reads only artifacts already written; runs no propagation.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARTIFACTS = Path(r"C:/Users/erayh/Documents/Python/Grad/python_port_phase17r0/artifacts")
CASES = ["M1", "M2", "M3"]


def load_csv(name):
    with (ARTIFACTS / name).open(newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    res = json.loads((ARTIFACTS / "r1m_multi_arc_results.json").read_text())
    ma = res["multi_arc"]

    def pick(case, model):
        return next(r for r in ma if r["case"] == case and r["model"] == model)

    a = [pick(c, "A_independent_arcs") for c in CASES]
    b = [pick(c, "B_continuity_linked") for c in CASES]
    n_arcs = [r["arcs"] for r in a]

    # ---- s54 #5  smallest singular value vs number of arcs -------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(n_arcs, [r["smallest_scaled_singular_value"] for r in a], "o-",
                label="Model A  pure multi-arc (6M+1 unknowns)")
    ax.semilogy(n_arcs, [r["smallest_scaled_singular_value"] for r in b], "s-",
                label="Model B  continuity-linked (7 unknowns)")
    ax.set_xlabel("number of separated arcs")
    ax.set_ylabel("smallest scaled singular value")
    ax.set_title("R1M s54: smallest singular value vs number of arcs")
    ax.set_xticks(n_arcs)
    ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(ARTIFACTS / "r1m_smallest_singular_vs_arcs.png", dpi=110)
    plt.close(fig)

    # ---- s54 #7  dedicated pure-vs-linked observability comparison -----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    idx = np.arange(len(CASES))
    lbl = ["%s\n(%d arcs)" % (c, n) for c, n in zip(CASES, n_arcs)]

    ax = axes[0]
    ax.bar(idx - .18, [r["conditional_k_information"] for r in a], .34, label="A pure")
    ax.bar(idx + .18, [r["conditional_k_information"] for r in b], .34, label="B linked")
    ax.set_yscale("log"); ax.set_xticks(idx); ax.set_xticklabels(lbl)
    ax.set_ylabel("conditional K information  I(K|x)")
    ax.set_title("independent K information")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(idx - .18, [r["orthogonal_fraction"] for r in a], .34, label="A pure")
    ax.bar(idx + .18, [r["orthogonal_fraction"] for r in b], .34, label="B linked")
    ax.set_xticks(idx); ax.set_xticklabels(lbl)
    ax.set_ylabel("orthogonal fraction  f_perp")
    ax.set_title("fraction of K's signature that is independent")
    ax.set_ylim(0, 1)
    ax.axhline(1.0, ls="--", c="k", lw=1)
    ax.text(0.02, 0.94, "f_perp = 1 would mean K fully separated",
            transform=ax.transAxes, fontsize=7, va="top")

    ax = axes[2]
    ax.bar(idx - .18, [1 - r["weakest_mode_k_component"] for r in a], .34, label="A pure")
    ax.bar(idx + .18, [1 - r["weakest_mode_k_component"] for r in b], .34, label="B linked")
    ax.set_yscale("log"); ax.set_xticks(idx); ax.set_xticklabels(lbl)
    ax.set_ylabel("1 - |K component of weakest mode|")
    ax.set_title("how nearly the weakest mode IS K\n(lower = more degenerate)")

    fig.suptitle("R1M s54: pure multi-arc vs continuity-linked observability "
                 "(0.5-orbit windows)", fontsize=11)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "r1m_pure_vs_linked.png", dpi=110)
    plt.close(fig)

    # ---- fairness control ----------------------------------------------
    fair = load_csv("r1m_model_a_fairness.csv")
    fa = [r for r in fair if r["model"] == "A_independent_arcs"]
    fb = [r for r in fair if r["model"] == "B_continuity_linked"]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    groups = [
        ("0.5-orbit arcs\n(pre-declared M1, 2 arcs)",
         a[0]["conditional_k_information"], b[0]["conditional_k_information"]),
        ("2.0-orbit arcs\n(fairness control F2, 2 arcs)",
         float(fa[0]["conditional_k_information"]),
         float(fb[0]["conditional_k_information"])),
        ("0.5-orbit arcs\n(pre-declared M2, 3 arcs)",
         a[1]["conditional_k_information"], b[1]["conditional_k_information"]),
        ("2.0-orbit arcs\n(fairness control F3, 3 arcs)",
         float(fa[1]["conditional_k_information"]),
         float(fb[1]["conditional_k_information"])),
    ]
    gi = np.arange(len(groups))
    ax.bar(gi - .19, [g[1] for g in groups], .36, label="Model A  pure multi-arc")
    ax.bar(gi + .19, [g[2] for g in groups], .36, label="Model B  continuity-linked")
    for i, g in enumerate(groups):
        ax.annotate("%.0fx" % (g[2] / g[1]), (i, max(g[1], g[2])),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=9, fontweight="bold")
    ax.set_yscale("log"); ax.set_xticks(gi)
    ax.set_xticklabels([g[0] for g in groups], fontsize=7.5)
    ax.set_ylabel("conditional K information  I(K|x)")
    ax.set_title("R1M fairness control: Model A's apparent collapse is set by\n"
                 "ARC LENGTH, not by the pure-multi-arc formulation",
                 fontsize=10.5)
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "r1m_model_a_window_length.png", dpi=110)
    plt.close(fig)

    print("wrote r1m_smallest_singular_vs_arcs.png")
    print("wrote r1m_pure_vs_linked.png")
    print("wrote r1m_model_a_window_length.png")


if __name__ == "__main__":
    main()
