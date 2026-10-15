"""PHASE 17-GEO - POST-HOC robustness diagnostic: orbital phase at epoch.

ANALYSIS SPACE ONLY.  NOT part of the predeclared grid and NOT used by any
classification.  It exists because the grid's largest f_perp (G3_i26.7_bcan,
0.281) is the canonical plane in circular form, while the canonical orbit itself
(e = 0.0053, a different phase) gives 0.092.  A 3x swing from phase or
eccentricity alone would mean the coarse geometry descriptors (beta, altitude,
inclination) do not control f_perp.  This script measures that directly:

  PHASE SWEEP    argument of latitude at epoch u0 = 0, 45, ..., 315 deg on three
                 predeclared geometries, everything else identical
  ECC/PHASE      the canonical plane (a, i, RAAN) with (e, u0) = (canonical, canonical),
                 (0, canonical) and (0, 0), separating eccentricity from phase
"""
from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ARTIFACTS = HERE.parent / "artifacts"
PHASES = (0, 45, 90, 135, 180, 225, 270, 315)
SWEEP_IDS = ("G3_i26.7_bcan", "G3_i90.0_bcan", "G1_b88")


def _init():
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels(None, clear=True)


def canonical_u0_deg(x_mci: np.ndarray) -> tuple[float, float]:
    import phase17_geo_core as G
    c = G.mci_to_lme()
    r, v = c @ x_mci[:3], c @ x_mci[3:6]
    h = np.cross(r, v)
    n = np.cross([0.0, 0.0, 1.0], h)
    n /= np.linalg.norm(n)
    u = math.degrees(math.atan2(np.dot(np.cross(n, r), h / np.linalg.norm(h)), np.dot(n, r))) % 360.0
    mu = G.mu_moon()
    e_vec = np.cross(v, h) / mu - r / np.linalg.norm(r)
    argp = math.degrees(math.atan2(np.dot(np.cross(n, e_vec), h / np.linalg.norm(h)),
                                   np.dot(n, e_vec))) % 360.0
    return u, argp


def run(job: dict) -> dict:
    import phase17_geo_core as G
    arc = G.build_geo_range_arc(np.asarray(job["x0"]), job["epoch_et"], job["duration_s"],
                                label=job["label"])
    m = G.information_metrics(arc.h_x0, arc.h_k, arc.w)
    return dict(job, x0=None, n_obs=m["n_obs"], f_perp=m["f_perp"], theta_k_deg=m["theta_k_deg"],
                sigma_k_frac=m["sigma_k_frac"], k_column_norm=m["k_column_norm"])


def main() -> None:
    import phase17_geo_core as G
    from lunar_od.spice_loader import load_spice_kernels
    from phase17_r1m_core import campaign_initial_state

    load_spice_kernels(None, clear=True)
    grid = json.loads((ARTIFACTS / "phase17_geo_predeclared_grid.json").read_text())
    by_id = {c["id"]: c for c in grid["cases"]}
    jobs = []
    for cid in SWEEP_IDS:
        c = by_id[cid]
        el = c["elements"]
        for u0 in PHASES:
            x0 = G.state_from_elements(el["a_m"], 0.0, el["inc_deg"], el["raan_deg"], 0.0, float(u0))
            jobs.append(dict(label=f"{cid}_u{u0:03d}", group=cid, u0_deg=u0, e=0.0,
                             x0=x0.tolist(), epoch_et=c["epoch_et"], duration_s=c["duration_s"]))
    xc = campaign_initial_state()
    elc = G.elements_from_state(xc)
    u_c, argp_c = canonical_u0_deg(xc)
    base = by_id["G0_consistent"]
    nu_c = (u_c - argp_c) % 360.0
    for e, u0, tag in ((elc["e"], u_c, "canonical_e_canonical_u"),
                       (0.0, u_c, "circular_canonical_u"), (0.0, 0.0, "circular_u000")):
        if e > 0:
            x0 = G.state_from_elements(elc["a_m"], e, elc["inc_deg"], elc["raan_deg"], argp_c, nu_c)
        else:
            x0 = G.state_from_elements(elc["a_m"], 0.0, elc["inc_deg"], elc["raan_deg"], 0.0, u0)
        jobs.append(dict(label=f"ECC_{tag}", group="ECC_PHASE", u0_deg=u0, e=e, x0=x0.tolist(),
                         epoch_et=base["epoch_et"], duration_s=base["duration_s"]))
    print(f"canonical: e={elc['e']:.5f} argp={argp_c:.2f} u0={u_c:.2f} deg; "
          f"state rebuild error vs campaign x0 = "
          f"{np.max(np.abs(G.state_from_elements(elc['a_m'], elc['e'], elc['inc_deg'], elc['raan_deg'], argp_c, nu_c) - xc)):.3e}")

    with ProcessPoolExecutor(max_workers=10, initializer=_init) as ex:
        rows = list(ex.map(run, jobs))
    summary = {}
    for g in list(SWEEP_IDS) + ["ECC_PHASE"]:
        rs = [r for r in rows if r["group"] == g]
        print(f"\n{g}")
        for r in rs:
            print(f"  {r['label']:34s} u0={r['u0_deg']:6.1f} e={r['e']:.4f} n={r['n_obs']:5d} "
                  f"f_perp={r['f_perp']:.4f} theta={r['theta_k_deg']:6.2f} sigK/K={100 * r['sigma_k_frac']:7.2f}%")
        f = np.array([r["f_perp"] for r in rs])
        summary[g] = dict(f_min=float(f.min()), f_max=float(f.max()), f_mean=float(f.mean()),
                          f_spread=float(f.max() - f.min()))
        print(f"  -> f_perp range {f.min():.4f} .. {f.max():.4f} (spread {f.max() - f.min():.4f})")
    (ARTIFACTS / "phase17_geo_phase_robustness.json").write_text(
        json.dumps(dict(label="POST_HOC_ROBUSTNESS_DIAGNOSTIC (not used for classification)",
                        canonical_u0_deg=u_c, canonical_argp_deg=argp_c,
                        rows=rows, summary=summary), indent=1, default=float))


if __name__ == "__main__":
    main()
