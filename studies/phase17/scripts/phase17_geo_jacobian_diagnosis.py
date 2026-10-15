"""PHASE 17-GEO s48/s49 - diagnose the Jacobian gate FAIL; measure its materiality.

ANALYSIS SPACE ONLY.  phase17_geo_jacobian_checks.py FAILED its predeclared
criterion (error at the best-resolved step <= 10x the FD step-halving floor):
the analytic range Jacobian and the FD oracle disagree by ~1e-6..3e-5 relative,
FLAT across a 16x step range -- neither truncation (~h^2) nor roundoff (~1/h).
The criterion is not loosened.  This script asks the two questions that decide
what the FAIL means:

  1. SOURCE     Repeat the whole comparison (nominal STM AND every FD propagation)
                at 10x and 100x tighter integration tolerances.  If the
                disagreement falls with the tolerance, both sides are limited by
                integration accuracy; if it stays, it is a model-level difference.
  2. MATERIALITY  Replace the analytic 7-column design by the FD design and
                recompute f_perp, theta_K and sigma_K/K on the same arc.  The
                predeclared class thresholds are 0.02 in f_perp.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ARTIFACTS = HERE.parent / "artifacts"
TOLS = ((1e-12, 1e-13), (1e-13, 1e-14), (1e-14, 1e-15))
CASES = ("G0_consistent", "G2_alt30_b20_fixD", "G3_i26.7_b00")


def _init():
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels(None, clear=True)


def run(job) -> dict:
    import phase17_geo_core as G

    case, (rtol, atol) = job
    et, k = case["epoch_et"], case["k_srp"]
    x0 = np.asarray(case["x0"], float)
    t = np.arange(0.0, 4.0 * case["elements"]["period_s"] + 45.0, 90.0)
    nom = G.propagate48(t, x0, et, k, rtol=rtol, atol=atol)
    arc = G.build_geo_range_arc(x0, et, t[-1], label=case["id"], k_srp=k, nom48=nom)

    def z_of(x0p, kp):
        return G.predicted_range(arc, G.propagate48(t, x0p, et, kp, rtol=rtol, atol=atol))

    cols, errs = [], []
    for j in range(6):
        h = 1.0 if j < 3 else 1e-3
        e = np.zeros(6)
        e[j] = h
        fd = (z_of(x0 + e, k) - z_of(x0 - e, k)) / (2 * h)
        cols.append(fd)
        errs.append(float(np.max(np.abs(fd - arc.h_x0[:, j])) / np.max(np.abs(arc.h_x0[:, j]))))
    hk = 0.1 * k
    fd_k = (z_of(x0, k + hk) - z_of(x0, k - hk)) / (2 * hk)
    err_k = float(np.max(np.abs(fd_k - arc.h_k)) / np.max(np.abs(arc.h_k)))
    h_fd = np.column_stack(cols)
    m_an = G.information_metrics(arc.h_x0, arc.h_k, arc.w, k_truth=k)
    m_fd = G.information_metrics(h_fd, fd_k, arc.w, k_truth=k)
    return dict(case=case["id"], rtol=rtol, atol=atol, n_obs=arc.n_obs,
                state_rel_err=errs, max_state_rel_err=max(errs), k_rel_err=err_k,
                f_perp_analytic=m_an["f_perp"], f_perp_fd=m_fd["f_perp"],
                d_f_perp=m_fd["f_perp"] - m_an["f_perp"],
                sigma_k_frac_analytic=m_an["sigma_k_frac"], sigma_k_frac_fd=m_fd["sigma_k_frac"])


def main() -> None:
    grid = json.loads((ARTIFACTS / "phase17_geo_predeclared_grid.json").read_text())
    by = {c["id"]: c for c in grid["cases"]}
    jobs = [(by[c], tol) for c in CASES for tol in TOLS]
    with ProcessPoolExecutor(max_workers=len(jobs), initializer=_init) as ex:
        rows = list(ex.map(run, jobs))
    print(f"{'case':22s} {'rtol':>7s} {'max state err':>13s} {'K err':>9s} {'f_perp an':>10s} "
          f"{'f_perp FD':>10s} {'d f_perp':>10s} {'sigK/K an':>9s} {'sigK/K FD':>9s}")
    for r in rows:
        print(f"{r['case']:22s} {r['rtol']:7.0e} {r['max_state_rel_err']:13.3e} {r['k_rel_err']:9.2e} "
              f"{r['f_perp_analytic']:10.6f} {r['f_perp_fd']:10.6f} {r['d_f_perp']:+10.2e} "
              f"{100 * r['sigma_k_frac_analytic']:8.3f}% {100 * r['sigma_k_frac_fd']:8.3f}%")
    (ARTIFACTS / "phase17_geo_jacobian_diagnosis.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
