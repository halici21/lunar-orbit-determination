"""Phase F2g gates G0 and G4e (studies/phase17/configs/phase_f2g_lro_empirical_predeclaration.json).

G0: with the empirical parameters at zero the study propagation equals production propagate_state bit for bit.
G4e: empirical-parameter range/Doppler partials vs direct central differences of the full measurement function.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_f2_lro_od as F2  # noqa: E402

OUT = HERE.parent / "results" / "phase_f2g_checks.json"


def main():
    t0 = time.time()
    d = json.loads((HERE.parent / "results" / "phase_f2d_lro_od_range_doppler.json").read_text(encoding="utf-8"))
    r = d["runs"]["N200_-"]
    pb = F2.Problem(300, doppler=True, days=F2.F2E_FIT_DAYS, n_emp=4)
    x0, k = np.array(r["x0"]), float(r["k_srp"])
    prod = pb.propagate(x0, k, sensitivity=False)                       # production propagate_state
    study = pb.propagate(x0, k, sensitivity=False, emp=np.zeros(4))     # study path, empirical = 0
    g0 = bool(np.array_equal(prod, study))
    g0_max = float(np.max(np.abs(prod - study)))

    # G4e: partials of 10 ranges and 10 Doppler intervals w.r.t. the 4 empirical parameters
    emp0 = np.array([2e-9, -1e-9, 1.5e-9, -2e-9])                       # away from zero, realistic size
    hist = pb.fd_history(x0, k, emp0)
    ir = list(np.linspace(0, len(pb.rows) - 1, 10).astype(int))
    idop = list(np.linspace(0, len(pb.dop) - 1, 10).astype(int))
    _, jr = pb.computed(hist, True, ir)
    _, jd = pb.computed_doppler(hist, True, idop)
    fd_r = np.zeros((len(ir), 4))
    fd_d = np.zeros((len(idop), 4))
    for j in range(4):
        for sgn in (1.0, -1.0):
            e = emp0.copy()
            e[j] += sgn * F2.EMP_FD_STEP
            hq = pb.propagate(x0, k, sensitivity=False, emp=e)
            cr, _ = pb.computed(hq, False, ir)
            cd, _ = pb.computed_doppler(hq, False, idop)
            fd_r[:, j] += sgn * cr / (2.0 * F2.EMP_FD_STEP)
            fd_d[:, j] += sgn * cd / (2.0 * F2.EMP_FD_STEP)
    col_r = [float(np.linalg.norm(jr[:, 7 + j] - fd_r[:, j]) / np.linalg.norm(fd_r[:, j])) for j in range(4)]
    col_d = [float(np.linalg.norm(jd[:, 7 + j] - fd_d[:, j]) / np.linalg.norm(fd_d[:, j])) for j in range(4)]
    worst = max(col_r + col_d)
    out = dict(G0=dict(PASS=g0, max_abs_diff_m=g0_max, rows=int(prod.shape[0])),
               G4e=dict(PASS=worst <= 1e-3, range_column_rel_err=col_r, doppler_column_rel_err=col_d, max=worst,
                        range_partial_norms=[float(np.linalg.norm(fd_r[:, j])) for j in range(4)]),
               elapsed_s=time.time() - t0)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
