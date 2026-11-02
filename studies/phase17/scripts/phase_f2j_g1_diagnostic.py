"""Phase F2j-G1 diagnostic: why the F2j fit ended at the damping limit with a large Gauss-Newton decrement.

Predeclaration: studies/phase17/configs/phase_f2j_g1_diagnostic_predeclaration.json.  Rebuilds F2j's Problem at
its end point and compares the actual cost along the Gauss-Newton and LM directions with the quadratic model, and
the solver's Jacobian with residual-level central differences at the A1 steps h and h/4.
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
M.use_historical_numerics()  # merge of main (G4e-R/TRF): keep the numerics of this phase's records

OUT = HERE.parent / "results" / "phase_f2j_g1_diagnostic.json"
LOG = M.F.REPO.parent / "f2j_g1_diagnostic.log"
DYN = list(range(7)) + [11, 12, 13, 14]
ALPHAS = (-0.01, 0.01, 0.1, 0.3, 1.0)
LAMBDAS = (1e-5, 1e-2, 1.0, 1e2, 1e4)
T0 = time.time()


def log(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.time() - T0:7.0f}s {msg}\n")


def end_point() -> np.ndarray:
    run = json.loads(M.OUT_F2J.read_text(encoding="utf-8"))["runs"]["S1"]
    return np.array(run["x0"] + [run["k_srp"]] + [run["biases_m"][str(k)] for k in M.PADS]
                    + [run["doppler_bias_mmps"] * 1e-3] + [run["empirical_mps2"][n] for n in M.EMP_NAMES], float)


PRIORS = sorted(M.F2H_PRIORS.items())
W_PRIOR = np.array([M.SIGMA_M / s for _, (_, s) in PRIORS])


def aug(r, h, p):
    r_out = np.concatenate([r, np.array([m - p[j] for j, (m, _) in PRIORS]) * W_PRIOR])
    if h is None:
        return r_out, None
    h_p = np.zeros((len(PRIORS), h.shape[1]))
    for row, (j, _) in enumerate(PRIORS):
        h_p[row, j] = W_PRIOR[row]
    return r_out, np.vstack([h, h_p])


def residual_from_history(pb, p, hist) -> np.ndarray:
    """Problem.residual(partials=False) for a given 6-state history, plus the prior rows."""
    comp, _ = pb.computed(hist, False)
    res = pb.obs - comp - p[7:10][pb.pad_index]
    cd, _ = pb.computed_doppler(hist, False)
    obs_d = np.array([d["obs"] for d in pb.dop])
    res = np.concatenate([res, (obs_d - cd - p[10]) * (M.SIGMA_M / M.SIGMA_DOP)])
    return aug(res, None, p)[0]


def batch_residuals(pb, ps) -> list:
    hists = list(pb.pool.map(M._worker_propagate, [np.concatenate([p[:7], p[11:15]]) for p in ps]))
    return [residual_from_history(pb, p, h) for p, h in zip(ps, hists)]


def scaled(ha, ra, scale):
    hs = ha * scale / M.SIGMA_M
    a = hs.T @ hs
    g = hs.T @ (ra / M.SIGMA_M)
    return a, g


def gn(ha, ra, scale):
    a, g = scaled(ha, ra, scale)
    return scale * np.linalg.solve(a, g), float(g @ np.linalg.solve(a, g)), g


def model_cost(ra, ha, step) -> float:
    r = ra - ha @ step
    return float(r @ r)


def main():
    LOG.unlink(missing_ok=True)
    p = end_point()
    pb = M.Problem(300, doppler=True, days=M.F2J_FIT_DAYS, n_emp=len(M.EMP_NAMES))
    pb.attach_pool(M._workers_arg() or 8)
    scale = np.array([1e3, 1e3, 1e3, 1.0, 1.0, 1.0, 1e-3, 1.0, 1.0, 1.0, 1e-3] + [M.EMP_SCALE] * pb.n_emp)
    try:
        # D0: the solver's residual and Jacobian at the end point
        res, h, _ = pb.residual(p)
        ra, ha = aug(res, h, p)
        cost = float(ra @ ra)
        d_gn, dec, g = gn(ha, ra, scale)
        log(f"D0 chi2 {cost:.6f} decrement {dec:.4f} |d_gn pos| {np.linalg.norm(d_gn[:3]):.3f} m")
        out = dict(predeclaration="studies/phase17/configs/phase_f2j_g1_diagnostic_predeclaration.json",
                   point=p.tolist(), D0=dict(chi2=cost, gn_decrement=dec, f2j_chi2=17153.652660468953,
                                             f2j_gn_decrement=977.4101992515072,
                                             gn_step=d_gn.tolist()))

        # D1 + D2: actual vs quadratic-model cost along the GN direction and for the solver's LM trials
        a, _ = scaled(ha, ra, scale)
        lm_steps = [scale * np.linalg.solve(a + lam * np.diag(np.diag(a)), g) for lam in LAMBDAS]
        steps = [al * d_gn for al in ALPHAS] + lm_steps
        actual = [float(r @ r) for r in batch_residuals(pb, [p + s for s in steps])]
        rows = [dict(kind="gn", alpha=al, actual=c, model=model_cost(ra, ha, s), pos_step_m=float(np.linalg.norm(s[:3])))
                for al, s, c in zip(ALPHAS, steps, actual)]
        rows += [dict(kind="lm", lam=lam, actual=c, model=model_cost(ra, ha, s), pos_step_m=float(np.linalg.norm(s[:3])))
                 for lam, s, c in zip(LAMBDAS, lm_steps, actual[len(ALPHAS):])]
        slope_actual = (actual[1] - actual[0]) / 0.02
        slope_model = -2.0 * float(ra @ (ha @ d_gn))
        out.update(D1_D2=rows, slope_actual_per_alpha=slope_actual, slope_model_per_alpha=slope_model)
        log(f"D1 slope actual {slope_actual:.3f} model {slope_model:.3f}")
        for r in rows:
            log(f"   {r}")

        # D3: residual-level central differences of every dynamical column at h and h/4
        hsteps = np.concatenate([M.FD_STEPS, [M.EMP_FD_STEP] * pb.n_emp])
        d3 = {}
        for name, frac in (("h", 1.0), ("h_over_4", 0.25)):
            ps = []
            for j, hj in zip(DYN, hsteps):
                e = np.zeros_like(p)
                e[j] = hj * frac
                ps += [p + e, p - e]
            rs = batch_residuals(pb, ps)
            jfd = ha.copy()
            cols = {}
            for k, (j, hj) in enumerate(zip(DYN, hsteps)):
                col = -(rs[2 * k] - rs[2 * k + 1]) / (2.0 * hj * frac)
                cols[str(j)] = dict(rel_diff=float(np.linalg.norm(col - ha[:, j]) / np.linalg.norm(ha[:, j])),
                                    grad_solver=float(ha[:, j] @ ra) * scale[j],
                                    grad_fd=float(col @ ra) * scale[j])
                jfd[:, j] = col
            d_fd, dec_fd, g_fd = gn(jfd, ra, scale)
            d3[name] = dict(columns=cols, gn_decrement=dec_fd, grad_rel_diff_vs_solver=float(
                np.linalg.norm(g_fd - g) / np.linalg.norm(g)), gn_step=d_fd.tolist(), jac=jfd)
            log(f"D3 {name}: decrement {dec_fd:.4f} grad rel diff {d3[name]['grad_rel_diff_vs_solver']:.4f}")
            for j, c in cols.items():
                log(f"   col {j}: {c}")
        g_h = scaled(d3["h"]["jac"], ra, scale)[1]
        g_q = scaled(d3["h_over_4"]["jac"], ra, scale)[1]
        fd_agreement = float(np.linalg.norm(g_h - g_q) / np.linalg.norm(g_q))
        # D3b: does the FD Jacobian's GN step (full and 0.1) lower the actual cost?
        trial = [np.array(d3[n]["gn_step"]) * al for n in ("h", "h_over_4") for al in (1.0, 0.1)]
        tc = [float(r @ r) for r in batch_residuals(pb, [p + s for s in trial])]
        for n in ("h", "h_over_4"):
            jac = d3[n].pop("jac")
            d3[n]["trial"] = [dict(alpha=al, actual=tc.pop(0), model=model_cost(ra, jac, np.array(d3[n]["gn_step"]) * al))
                              for al in (1.0, 0.1)]
        out.update(D3=d3, fd_h_vs_h_over_4_grad_rel_diff=fd_agreement, elapsed_s=time.time() - T0)
        log(f"D3 fd h vs h/4 grad rel diff {fd_agreement:.4f}; trials {d3['h']['trial']} {d3['h_over_4']['trial']}")
        OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
        log("done")
    finally:
        pb.close_pool()


if __name__ == "__main__":
    main()
