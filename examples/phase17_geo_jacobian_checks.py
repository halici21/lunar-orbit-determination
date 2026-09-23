"""PHASE 17-GEO s49/s50 - cross-geometry Jacobian qualification, floor-aware.

ANALYSIS SPACE ONLY.  For the predeclared representative geometries (grid
rules 'jacobian_fd_cases'), on 4-revolution arcs built by the same production
path as the campaign:

  state columns   d(range)/dx0_j  vs central differences of the PRODUCTION
                  nominal range on re-propagated trajectories (x0 +/- h e_j)
  K column        d(range)/dK     vs central differences with K +/- h (x0 fixed)

Each derivative is swept over five steps (x4 .. x0.25 of a base step).  The
oracle's own precision floor is MEASURED as the smallest step-halving change
|FD(h) - FD(h/2)|; the analytic column passes when its error at the
best-resolved step is within 10x that floor (the R1O-R lesson: a fixed
threshold can fail a correct model against a noisy oracle, or pass a wrong one
against a coarse oracle).
"""
from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ARTIFACTS = HERE.parent / "artifacts"
STEP_FACTORS = (4.0, 2.0, 1.0, 0.5, 0.25)
BASE_POS_M, BASE_VEL_MPS = 1.0, 1e-3
FLOOR_MULTIPLE = 10.0


def _init():
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels(None, clear=True)


def _sweep(fd_fn, analytic: np.ndarray, base: float) -> dict:
    fds = [fd_fn(base * f) for f in STEP_FACTORS]
    scale = max(float(np.max(np.abs(analytic))), 1e-300)
    errs = [float(np.max(np.abs(fd - analytic))) / scale for fd in fds]
    halving = [float(np.max(np.abs(fds[i] - fds[i + 1]))) / scale for i in range(len(fds) - 1)]
    k = int(np.argmin(halving))              # best-resolved pair (h, h/2)
    floor = max(halving[k], 1e-15)
    err_best = min(errs[k], errs[k + 1])
    return dict(steps=[base * f for f in STEP_FACTORS], rel_err=errs, halving=halving,
                best_step=base * STEP_FACTORS[k + 1], oracle_floor=floor,
                err_at_best=err_best, model_to_oracle=err_best / floor,
                passed=bool(err_best <= FLOOR_MULTIPLE * floor or err_best < 1e-9))


def run(case: dict) -> dict:
    import phase17_geo_core as G

    et = case["epoch_et"]
    x0 = np.asarray(case["x0"], float)
    period = case["elements"]["period_s"]
    k = case["k_srp"]
    dur = 4.0 * period
    arc = G.build_geo_range_arc(x0, et, dur, label=case["id"], k_srp=k)
    t = arc.t_grid

    def z_of(x0p, kp):
        return G.predicted_range(arc, G.propagate48(t, x0p, et, kp))

    out = dict(id=case["id"], n_obs=arc.n_obs, arc_revolutions=4.0, components={})
    names = ("x", "y", "z", "vx", "vy", "vz")
    for j in range(6):
        base = BASE_POS_M if j < 3 else BASE_VEL_MPS

        def fd(h, j=j):
            e = np.zeros(6)
            e[j] = h
            return (z_of(x0 + e, k) - z_of(x0 - e, k)) / (2 * h)

        out["components"][names[j]] = _sweep(fd, arc.h_x0[:, j], base)

    def fd_k(h):
        return (z_of(x0, k + h) - z_of(x0, k - h)) / (2 * h)

    out["components"]["K"] = _sweep(fd_k, arc.h_k, 0.1 * k)
    out["passed"] = all(c["passed"] for c in out["components"].values())
    return out


def main() -> None:
    grid = json.loads((ARTIFACTS / "phase17_geo_predeclared_grid.json").read_text())
    ids = grid["rules"]["jacobian_fd_cases"]
    cases = [c for c in grid["cases"] if c["id"] in ids]
    with ProcessPoolExecutor(max_workers=len(cases), initializer=_init) as ex:
        results = list(ex.map(run, cases))
    rows = []
    for r in results:
        print(f"\n{r['id']}  (n_obs={r['n_obs']}, 4 rev)  PASS={r['passed']}")
        print(f"  {'col':3s} {'best step':>10s} {'err@best':>10s} {'floor':>10s} {'err/floor':>9s}  "
              f"rel_err over steps x4..x0.25")
        for name, c in r["components"].items():
            print(f"  {name:3s} {c['best_step']:10.3e} {c['err_at_best']:10.3e} {c['oracle_floor']:10.3e} "
                  f"{c['model_to_oracle']:9.3f}  " + " ".join(f"{e:.1e}" for e in c["rel_err"]))
            rows.append(dict(case=r["id"], column=name, best_step=c["best_step"],
                             err_at_best=c["err_at_best"], oracle_floor=c["oracle_floor"],
                             model_to_oracle=c["model_to_oracle"], passed=c["passed"],
                             **{f"rel_err_step{i}": e for i, e in enumerate(c["rel_err"])}))
    gate = all(r["passed"] for r in results)
    print(f"\nGEO_CROSS_GEOMETRY_JACOBIAN_GATE = {'PASS' if gate else 'FAIL'}")
    with (ARTIFACTS / "phase17_geo_jacobian_checks.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    (ARTIFACTS / "phase17_geo_jacobian_checks.json").write_text(
        json.dumps(dict(gate="PASS" if gate else "FAIL", results=results), indent=1))


if __name__ == "__main__":
    main()
