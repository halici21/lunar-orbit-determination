"""Phase F2 gate G4e-R (studies/phase17/configs/phase_f2_g4er_predeclaration.json).

Requalifies the F2g/h/i partials at the qualified numerics after the G4e root-cause analysis:
G0R  empirical = 0 study propagation equals production bit for bit;
G4eR_dyn  fd_history's 11 dynamical columns vs an independent-step reference at the same tolerance;
G4eR_meas range/Doppler partials vs central differences of the full measurement function (historical G4e oracle);
G4eR_jump the historical counted-Doppler jump row is continuous.

usage: phase_f2_g4er_checks.py [--workers N]
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

OUT = HERE.parent / "results" / "phase_f2_g4er_checks.json"
EMP0 = np.array([2e-9, -1e-9, 1.5e-9, -2e-9])
NAMES = ("x", "y", "z", "vx", "vy", "vz", "K") + F2.EMP_NAMES
DYN_TOL, MEAS_TOL, JUMP_TOL = 1e-2, 1e-3, 1e-7


def _col_err(a, b):
    return [float(np.linalg.norm(a[:, j] - b[:, j]) / np.linalg.norm(b[:, j])) for j in range(a.shape[1])]


def main():
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 5
    t0 = time.time()
    d = json.loads((HERE.parent / "results" / "phase_f2d_lro_od_range_doppler.json").read_text(encoding="utf-8"))
    r = d["runs"]["N200_-"]
    x0, k = np.array(r["x0"]), float(r["k_srp"])
    pb = F2.Problem(300, doppler=True, days=F2.F2E_FIT_DAYS, n_emp=4)
    assert pb.numerics == F2._numerics_for_backend("trf"), "G4e-R qualifies the TRF numerics only"
    pb.attach_pool(workers)
    try:
        prod = pb.propagate(x0, k, sensitivity=False)
        study = pb.propagate(x0, k, sensitivity=False, emp=np.zeros(4))
        g0 = dict(PASS=bool(np.array_equal(prod, study)), max_abs_diff_m=float(np.max(np.abs(prod - study))))

        runs: list = []
        hist = pb.fd_history(x0, k, EMP0, keep=runs)
        q0 = runs[0][0]
        steps = list(F2.FD_STEPS) + [pb.emp_fd_step] * 4
        ref_steps = [100.0] * 3 + [0.1] * 3 + [5e-3] + [1e-7] * 4
        if q0[6] < ref_steps[6]:
            raise ValueError("reference K step would cross K = 0")
        qs = []
        for j, h in enumerate(ref_steps):
            for sgn in (1.0, -1.0):
                q = q0.copy()
                q[j] += sgn * h
                qs.append(q)
        ref_hists = list(pb.pool.map(F2._worker_propagate, qs))
    finally:
        pb.close_pool()

    n = hist.shape[0]
    phi = hist[:, 6:42].reshape(n, 6, 6)          # [:, j, :] = column j (d state / d x0_j)
    cols = [phi[:, j, :] for j in range(6)] + [hist[:, 42:48]] + [hist[:, 48 + 6 * i:54 + 6 * i] for i in range(4)]
    ref = [(ref_hists[2 * j] - ref_hists[2 * j + 1]) / (2.0 * ref_steps[j]) for j in range(11)]
    dyn = [float(np.linalg.norm(cols[j][:, :3] - ref[j][:, :3]) / np.linalg.norm(ref[j][:, :3])) for j in range(11)]
    g_dyn = dict(PASS=max(dyn) <= DYN_TOL, threshold=DYN_TOL, position_column_rel_err=dict(zip(NAMES, dyn)),
                 steps=steps, ref_steps=ref_steps)

    ir = list(np.linspace(0, len(pb.rows) - 1, 10).astype(int))
    idop = list(np.linspace(0, len(pb.dop) - 1, 10).astype(int))
    _, jr = pb.computed(hist, True, ir)
    _, jd = pb.computed_doppler(hist, True, idop)
    jr, jd = jr[:, :11], jd[:, :11]
    fr, fd = np.zeros_like(jr), np.zeros_like(jd)
    plus_minus = runs[1:]
    for j in range(11):
        (qp, hp), (qm, hm) = plus_minus[2 * j], plus_minus[2 * j + 1]
        h = 0.5 * (qp[j] - qm[j])
        fr[:, j] = (pb.computed(hp, False, ir)[0] - pb.computed(hm, False, ir)[0]) / (2.0 * h)
        fd[:, j] = (pb.computed_doppler(hp, False, idop)[0] - pb.computed_doppler(hm, False, idop)[0]) / (2.0 * h)
    err_r, err_d = _col_err(jr, fr), _col_err(jd, fd)
    g_meas = dict(PASS=max(err_r + err_d) <= MEAS_TOL, threshold=MEAS_TOL, range_column_rel_err=dict(zip(NAMES, err_r)),
                  doppler_column_rel_err=dict(zip(NAMES, err_d)))

    # historical jump row: Doppler row 5, column C_T, along the propagated C_T sensitivity
    base = runs[0][1]
    s_ct = cols[7]
    dd = pb.dop[idop[5]]
    s, R = pb.setup, pb.R
    grid = np.linspace(-3e-9, 3e-9, 61)
    vals = []
    for e in grid:
        sc = F2._spacecraft_history(pb.grid, base + e * s_ct, 4)
        v, _, _ = R.counted_doppler(dd["tm"], dd["tc"], s.tide[dd["pad"]], sc, s.eph, R.DSN_C2_LIGHT_TIME_MODEL,
                                    site=s.sites[dd["pad"]], media=dd["media"], media_site=s.media_sites[dd["pad"]])
        vals.append(v)
    vals = np.array(vals)
    resid = vals - np.polyval(np.polyfit(grid, vals, 1), grid)
    jump = float(np.max(np.abs(np.diff(resid))))
    g_jump = dict(PASS=jump <= JUMP_TOL, threshold=JUMP_TOL, max_neighbour_step_mps=jump,
                  detrended_std_mps=float(resid.std()))

    out = dict(G0R=g0, G4eR_dyn=g_dyn, G4eR_meas=g_meas, G4eR_jump=g_jump,
               PASS=all(g["PASS"] for g in (g0, g_dyn, g_meas, g_jump)),
               numerics=pb.numerics, grid_s=pb.grid_s, exact_epoch_stations=F2.F.EXACT_EPOCH_STATIONS,
               numerical_identity=F2.numerical_identity(), elapsed_s=time.time() - t0)
    path = F2._write_result(OUT, json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "numerical_identity"}, indent=1))
    print("written", path)


if __name__ == "__main__":
    main()
