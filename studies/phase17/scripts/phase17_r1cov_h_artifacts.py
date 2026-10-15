"""PHASE 17-R1COV - config record, historical correction table, figures.

s46 is explicit that historical reports must NOT be silently rewritten. The
correction table produced here is additive: it states what each historical
claim was, what it rested on, and what the qualified covariance now says,
leaving the original artifacts intact as historical evidence.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from phase17_r1cov_core import (  # noqa: E402
    K_INDEX, condition_numbers, exact_covariance, floored_covariance,
    relative_error, scale_matrix, square_root_covariance,
)

ARTIFACTS = REPO / "artifacts"
PRIOR_SWEEP = np.geomspace(1e-4, 1e2, 25)


def git(*a) -> str:
    return subprocess.run(["git", "-C", str(REPO), *a],
                          capture_output=True, text=True).stdout.strip()


def main() -> None:
    from lunar_od.spice_loader import load_spice_kernels
    from phase17_r1m_core import build_range_arc, campaign_epoch

    load_spice_kernels(None, clear=True)
    et0, t_orbit = campaign_epoch()
    scale = scale_matrix()

    arcs, designs = {}, {}
    for label, orbits in (("G0", 1.3), ("G1", 2.0), ("G3", 5.0)):
        a = build_range_arc(0.0, orbits * t_orbit, label=label)
        arcs[label] = a
        designs[label] = (np.hstack([a.h_x0, a.h_k[:, None]]), a.w)

    # ---------------- config ------------------------------------------
    cfg = dict(
        phase="PHASE 17-R1COV",
        title="SQUARE-ROOT COVARIANCE QUALIFICATION AND K_SRP FORMAL-UNCERTAINTY REPAIR",
        scope="COVARIANCE_METHOD_ONLY_K_SOLVE_FOR_PATH",
        git=dict(branch=git("rev-parse", "--abbrev-ref", "HEAD"),
                 head=git("rev-parse", "HEAD"),
                 tree=git("rev-parse", "HEAD^{tree}"),
                 main=git("rev-parse", "main"),
                 origin_main=git("rev-parse", "origin/main"),
                 r1m_content_commit="f2af3045e44860b0892fe84b788ab7b75c7c63a2"),
        oracle=dict(
            method="exact rational arithmetic (fractions.Fraction)",
            rationale=("every IEEE-754 double is a dyadic rational, so the "
                       "normal matrix is exactly representable and the inverse "
                       "carries zero arithmetic error"),
            convergence_study="decimal.Decimal at 50/80/120/200 digits",
            mpmath_available=False),
        method=dict(
            covariance="QR of the whitened scaled design, P = R^-1 R^-T via "
                       "two triangular solves; normal matrix never formed",
            prior_handling="prior square-root rows appended to the design",
            prior_sqrt="symmetric eigendecomposition (prior information is "
                       "routinely only semi-definite; Cholesky would raise)",
            rank_criterion="sigma_i > sigma_max * n * eps on the singular "
                           "values of R (a pure ratio, hence scale-invariant)",
            rank_failure="RankDeficientCovarianceError raised, never a "
                         "floor-derived finite covariance"),
        truth=dict(k_srp_truth_m2_per_kg=0.01, provenance="SYNTHETIC_CAMPAIGN_TRUTH"),
        epoch=dict(campaign_et0=et0, orbit_period_s=t_orbit),
        scaling=dict(position_m=1e6, velocity_m_s=1e3, k_srp_m2_per_kg=1e-2),
        production_change=dict(
            files=["lunar_od/estimators.py"],
            call_sites_modified=2,
            call_sites_protected=26,
            srukf_modified=False,
            default_six_state_modified=False,
            scenarios_duplicate_helper_modified=False),
    )
    (ARTIFACTS / "r1cov_config.json").write_text(json.dumps(cfg, indent=2, default=float))
    print("wrote r1cov_config.json")

    # ---------------- historical correction table (s46) ---------------
    def sig(label, prior=None):
        h, w = designs[label]
        return square_root_covariance(h, w, prior, scale).sigma_k

    def old_sig(label, prior=None):
        h, w = designs[label]
        return floored_covariance(h, w, prior, scale)[1]["sigma_k"]

    k_truth = 0.01
    rows = [
        dict(historical_claim="R1/R1G G0 formal sigma_K = 3.47e-03 m^2/kg",
             old_basis="floored normal-matrix inverse",
             r1cov_status="SUPERSEDED",
             correct_interpretation="qualified sigma_K = %.4e (%.0fx larger); "
             "K is unconstrained on this arc (%.0f%% of K_truth)"
             % (sig("G0"), sig("G0") / old_sig("G0"), 100 * sig("G0") / k_truth)),
        dict(historical_claim="R1G G1 formal sigma_K = 1.64e-03 m^2/kg",
             old_basis="floored normal-matrix inverse",
             r1cov_status="SUPERSEDED",
             correct_interpretation="qualified sigma_K = %.4e (%.1fx larger); "
             "%.0f%% of K_truth" % (sig("G1"), sig("G1") / old_sig("G1"),
                                    100 * sig("G1") / k_truth)),
        dict(historical_claim="R1G G3 formal sigma_K = 5.24e-04 m^2/kg",
             old_basis="floored normal-matrix inverse",
             r1cov_status="SUPERSEDED",
             correct_interpretation="qualified sigma_K = %.4e (%.1fx larger); "
             "%.0f%% of K_truth" % (sig("G3"), sig("G3") / old_sig("G3"),
                                    100 * sig("G3") / k_truth)),
        dict(historical_claim="augmented problem is 'rank 6/7'",
             old_basis="numerical rank of H^T W H in float64",
             r1cov_status="REINTERPRETED",
             correct_interpretation="rank deficiency is an artifact of forming "
             "the normal matrix (cond(I)=cond(A)^2 exactly). The design matrix "
             "is full rank with ~5 orders of margin; K is WEAK_BUT_RESOLVABLE, "
             "not absent from the data"),
        dict(historical_claim="broad (sigma=1.0) and moderate (sigma=0.01) K "
             "priors give identical posterior sigma_K",
             old_basis="both reported the floor value",
             r1cov_status="RESOLVED",
             correct_interpretation="neither prior was reaching the answer. "
             "Qualified posteriors are %.4e and %.4e for G0, a %.0f%% "
             "difference" % (sig("G0", _kprior(1.0)), sig("G0", _kprior(0.01)),
                             100 * relative_error(sig("G0", _kprior(1.0)),
                                                  sig("G0", _kprior(0.01))))),
        dict(historical_claim="prior classified DIAGNOSTIC_UNCERTAIN / "
             "PRIOR_INFLUENCED / PRIOR_DOMINATED (R1G s10)",
             old_basis="inference from an uninterpretable reported sigma",
             r1cov_status="SUPERSEDED",
             correct_interpretation="the posterior now responds correctly to "
             "the prior and matches an exact oracle, so prior influence can be "
             "read directly rather than inferred"),
        dict(historical_claim="R1M: estimator sigma_K is NOT_RELIABLE_IN_"
             "NEAR_SINGULAR_REGIME",
             old_basis="R1M covariance audit",
             r1cov_status="CONFIRMED_AND_REPAIRED",
             correct_interpretation="confirmed independently here, and repaired "
             "for the K solve-for path; the default six-state path still uses "
             "the floored helper and is unchanged"),
        dict(historical_claim="R1M: best radiometric case M3 Model B, "
             "sigma_K ~ 4.52e-03 from 1/sqrt(I(K|x))",
             old_basis="information-domain proxy (data-only Schur)",
             r1cov_status="CONFIRMED",
             correct_interpretation="the qualified covariance reproduces the "
             "data-only Schur value, so R1M's information-domain conclusion "
             "stands; see r1cov_requalification.json"),
    ]
    with (ARTIFACTS / "r1cov_historical_correction_table.csv").open("w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w_.writeheader(); w_.writerows(rows)
    print("wrote r1cov_historical_correction_table.csv (%d rows)" % len(rows))
    for r in rows:
        print("  - %s -> %s" % (r["historical_claim"][:58], r["r1cov_status"]))

    # ---------------- prior sweep -------------------------------------
    sweep = []
    for label in ("G0", "G1", "G3"):
        h, w = designs[label]
        for ps in PRIOR_SWEEP:
            pr = _kprior(ps)
            sweep.append(dict(case=label, prior_sigma_k=float(ps),
                              qualified_sigma_k=square_root_covariance(h, w, pr, scale).sigma_k,
                              floored_sigma_k=floored_covariance(h, w, pr, scale)[1]["sigma_k"],
                              exact_sigma_k=exact_covariance(h, w, pr)[1]))
    with (ARTIFACTS / "r1cov_prior_sweep.csv").open("w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(sweep[0].keys()))
        w_.writeheader(); w_.writerows(sweep)
    print("wrote r1cov_prior_sweep.csv (%d rows)" % len(sweep))

    # ---------------- figures (s57) -----------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = ["G0", "G1", "G3"]
    old = [old_sig(c) for c in cases]
    new = [sig(c) for c in cases]
    ex = [exact_covariance(*designs[c])[1] for c in cases]

    # 1. old vs qualified sigma_K
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(len(cases))
    ax.bar(x - 0.2, old, 0.38, label="old (floored normal matrix)")
    ax.bar(x + 0.2, new, 0.38, label="qualified (square-root)")
    ax.axhline(k_truth, ls="--", c="k", lw=1, label="K_truth = 0.01")
    for i, (o, n) in enumerate(zip(old, new)):
        ax.annotate("%.0fx" % (n / o), (i, max(o, n)), ha="center",
                    textcoords="offset points", xytext=(0, 5),
                    fontsize=9, fontweight="bold")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(cases)
    ax.set_ylabel("formal sigma_K  [m^2/kg]")
    ax.set_title("R1COV: the repaired formal uncertainty is 10-260x larger")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(ARTIFACTS / "r1cov_old_vs_qualified_sigma_k.png", dpi=110); plt.close(fig)

    # 2. sigma_K vs K scaling
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    kscales = np.geomspace(2.5e-3, 4e-2, 13)
    for c, mk in zip(cases, "os^"):
        q, f = [], []
        h, w = designs[c]
        for ks in kscales:
            s = scale_matrix(k_scale=ks)
            q.append(square_root_covariance(h, w, None, s).sigma_k)
            f.append(floored_covariance(h, w, None, s)[1]["sigma_k"])
        ax.plot(kscales, q, mk + "-", label="%s square-root" % c)
        ax.plot(kscales, f, mk + "--", alpha=.6, label="%s floored" % c)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("K_scale (an arbitrary numerical bookkeeping constant)")
    ax.set_ylabel("reported sigma_K  [m^2/kg]")
    ax.set_title("R1COV: a physical uncertainty must be FLAT here\n"
                 "(solid = repaired, dashed = old)", fontsize=10.5)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(ARTIFACTS / "r1cov_sigma_k_vs_scaling.png", dpi=110); plt.close(fig)

    # 3. raw singular spectrum
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for c, mk in zip(cases, "os^"):
        h, w = designs[c]
        a = np.sqrt(w)[:, None] * (h @ scale)
        sv = np.linalg.svd(a, compute_uv=False)
        ax.semilogy(range(1, 8), sv / sv[0], mk + "-", label="%s design" % c)
    ax.axhline(7 * np.finfo(float).eps, ls=":", c="r",
               label="rank threshold  n*eps")
    ax.set_xlabel("singular value index"); ax.set_ylabel("sigma_i / sigma_max")
    ax.set_title("R1COV: the design matrix's spectrum, with the rank threshold")
    ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(ARTIFACTS / "r1cov_design_singular_spectrum.png", dpi=110); plt.close(fig)

    # 4. normal-matrix eigenvalues vs old floor
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for i, c in enumerate(cases):
        h, w = designs[c]
        info = scale.T @ (h.T @ (w[:, None] * h)) @ scale
        ev = np.linalg.eigvalsh(0.5 * (info + info.T))
        floor = max(float(np.max(np.abs(ev))) * 1e-14, np.finfo(float).eps)
        ax.semilogy([i] * 7, ev, "o", ms=6, label="%s eigenvalues" % c if i == 0 else None,
                    color="C0")
        ax.semilogy([i], [floor], "rx", ms=13, mew=2,
                    label="old floor = max*1e-14" if i == 0 else None)
        ax.annotate("%.0fx" % (floor / ev[0]), (i, floor),
                    textcoords="offset points", xytext=(10, 0), fontsize=9,
                    color="r", fontweight="bold")
    ax.set_xticks(range(len(cases))); ax.set_xticklabels(cases)
    ax.set_ylabel("scaled information eigenvalue")
    ax.set_title("R1COV: the old floor sat far ABOVE the smallest real eigenvalue")
    ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(ARTIFACTS / "r1cov_eigenvalues_vs_floor.png", dpi=110); plt.close(fig)

    # 5. QR vs exact error
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    qr_err = [relative_error(new[i], ex[i]) for i in range(3)]
    fl_err = [relative_error(old[i], ex[i]) for i in range(3)]
    ax.bar(x - 0.2, np.maximum(qr_err, 1e-16), 0.38, label="square-root vs exact")
    ax.bar(x + 0.2, fl_err, 0.38, label="floored vs exact")
    ax.axhline(np.finfo(float).eps, ls=":", c="k", label="machine epsilon")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(cases)
    ax.set_ylabel("relative error in sigma_K vs the exact oracle")
    ax.set_title("R1COV: agreement with exact rational arithmetic")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(ARTIFACTS / "r1cov_qr_vs_oracle_error.png", dpi=110); plt.close(fig)

    # 6. posterior vs prior sigma
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for c, mk in zip(cases, "os^"):
        sub = [r for r in sweep if r["case"] == c]
        ax.loglog([r["prior_sigma_k"] for r in sub],
                  [r["qualified_sigma_k"] for r in sub], "-", label="%s repaired" % c)
        ax.loglog([r["prior_sigma_k"] for r in sub],
                  [r["floored_sigma_k"] for r in sub], "--", alpha=.6,
                  label="%s old (flat = ignoring the prior)" % c)
    ax.plot(PRIOR_SWEEP, PRIOR_SWEEP, ":", c="k", lw=1, label="posterior = prior")
    ax.set_xlabel("prior sigma_K  [m^2/kg]")
    ax.set_ylabel("posterior sigma_K  [m^2/kg]")
    ax.set_title("R1COV: posterior must follow the prior when data are weak")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(ARTIFACTS / "r1cov_posterior_vs_prior.png", dpi=110); plt.close(fig)

    print("wrote 6 figures")


def _kprior(sigma: float) -> np.ndarray:
    p = np.zeros((7, 7))
    p[K_INDEX, K_INDEX] = 1.0 / sigma ** 2
    return p


if __name__ == "__main__":
    main()
