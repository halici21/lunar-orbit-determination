"""Phase F2j-G1 amendment A1, step D4: where in the residual vector does the cost jump come from?

Predeclaration: studies/phase17/configs/phase_f2j_g1_diagnostic_predeclaration.json (amendment A1).  Evaluates
the residual at the F2j end point twice (determinism), after a 1e-6 m change of x0[0], and at 0.01 of the
Gauss-Newton step saved by D0, and reports which rows carry the change in chi^2.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_f2_lro_od as M  # noqa: E402
M.use_historical_numerics()  # merge of main (G4e-R/TRF): keep the numerics of this phase's records
import phase_f2j_g1_diagnostic as G  # noqa: E402

OUT = HERE.parent / "results" / "phase_f2j_g1_rowdiff.json"
TOP = 25


def describe(pb, i: int) -> dict:
    nr, nd = len(pb.rows), len(pb.dop)
    if i < nr:
        return dict(kind="range", pad=int(pb.rows[i][0]), t_s=float(pb.setup.t3[i]))
    if i < nr + nd:
        d = pb.dop[i - nr]
        return dict(kind="doppler", pad=int(d["pad"]), t_s=float(d["tm"]), count_s=float(d["tc"]))
    return dict(kind="prior", index=int(G.PRIORS[i - nr - nd][0]))


def main():
    p = G.end_point()
    d_gn = np.array(json.loads(G.OUT.read_text(encoding="utf-8"))["D0"]["gn_step"])
    pb = M.Problem(300, doppler=True, days=M.F2J_FIT_DAYS, n_emp=len(M.EMP_NAMES))
    pb.attach_pool(M._workers_arg() or 8)
    try:
        e = np.zeros_like(p)
        e[0] = 1e-6
        cases = dict(base_repeat=p.copy(), x0_plus_1um=p + e, gn_alpha_0p01=p + 0.01 * d_gn)
        rs = G.batch_residuals(pb, [p] + list(cases.values()))
        r0 = rs[0]
        out = dict(predeclaration="studies/phase17/configs/phase_f2j_g1_diagnostic_predeclaration.json (A1, D4)",
                   chi2_base=float(r0 @ r0), cases={})
        for (name, _), r in zip(cases.items(), rs[1:]):
            dchi = r * r - r0 * r0
            order = np.argsort(-np.abs(dchi))
            dr = r - r0
            out["cases"][name] = dict(
                chi2=float(r @ r), dchi2=float(dchi.sum()), max_abs_dr=float(np.abs(dr).max()),
                rms_dr_range=float(np.sqrt(np.mean(dr[:len(pb.rows)] ** 2))),
                rms_dr_doppler_scaled=float(np.sqrt(np.mean(dr[len(pb.rows):len(pb.rows) + len(pb.dop)] ** 2))),
                share_of_top=float(dchi[order[:TOP]].sum() / dchi.sum()) if dchi.sum() != 0.0 else None,
                n_rows_abs_dr_gt_1cm=int(np.sum(np.abs(dr) > 0.01)),
                top=[dict(row=int(i), dchi2=float(dchi[i]), r0=float(r0[i]), r=float(r[i]), **describe(pb, int(i)))
                     for i in order[:TOP]])
        OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "top"} for k, v in out["cases"].items()},
                         indent=1))
    finally:
        pb.close_pool()


if __name__ == "__main__":
    main()
