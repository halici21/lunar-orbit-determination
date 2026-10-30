"""Phase F2s diagnosis: why step A's method check fails (9.0e-8 vs 1e-9 m/s^2).

Own E2 trajectory only, no LOLA data.  Over the first 3 h of F, the same 5-point stencil + RHS as step A for several
stencil steps h and integrator tolerances.  Truncation error of the stencil grows like h^4, error from the
propagator's output values like 1/h; their scaling separates the two.  Writes results/phase_f2s_parts/diag.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_f2_lro_od as M  # noqa: E402
import phase_f2r_postfit_growth as R  # noqa: E402
import phase_f2s_lola_arc_structure as S  # noqa: E402

SPAN_S = 3 * 3600.0
H_LIST = (1.0, 2.0, 5.0, 10.0, 20.0)
TOLS = ((1e-10, 1e-11), (1e-12, 1e-13), (1e-13, 1e-14))


def check(pb, x0, k, emp, t, h):
    pts = [t + o for o in np.array([0, -2, -1, 1, 2]) * h]
    grid = np.unique(np.concatenate(pts))
    own = pb.propagate(x0, k, t_eval=np.concatenate([[pb.t_epoch], grid]), sensitivity=False, emp=emp)[1:]
    at = [own[np.searchsorted(grid, p)] for p in pts]
    a = S.stencil(*[s[:, 3:] for s in at[1:]], h)
    chk = S.to_rtn(a - S.model_accel(pb, t, at[0], k, emp), at[0])
    return np.sqrt(np.mean(chk ** 2, axis=0)).tolist()


def main():
    t0 = time.time()
    pb, spans = R.problem()
    x0, k, emp = S.e2_solution()
    t = np.arange(spans["F"][0], spans["F"][0] + SPAN_S, S.DT_A)
    out = dict(span_s=[float(t[0]), float(t[-1])], t_epoch=float(pb.t_epoch), by_h={}, by_tol={})
    for h in H_LIST:
        out["by_h"][str(h)] = check(pb, x0, k, emp, t, h)
        print("h", h, out["by_h"][str(h)], flush=True)
    for tol in TOLS:
        M.set_tolerance(*tol)
        out["by_tol"][f"{tol[0]:g}/{tol[1]:g}"] = check(pb, x0, k, emp, t, S.H_STENCIL)
        print("tol", tol, out["by_tol"][f"{tol[0]:g}/{tol[1]:g}"], flush=True)
    out["elapsed_s"] = time.time() - t0
    S.write_part("diag", out)


if __name__ == "__main__":
    main()
