"""Phase F1 diagnostic (not a gate): is the common residual slope a time-tag offset or an orbit signature?

Per pass, the M_FULL residual is fitted with (a) bias + slope, (b) bias + dt * rho_dot (rho_dot = d computed
range / d t3 by central difference, 0.5 s), (c) bias + dt * rho_dot + slope.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_f1_lro_residuals as F  # noqa: E402

OUT = HERE.parent / "results" / "phase_f1_slope_diagnostic.json"


def main(day_dir: str = F.DAY_DIR, out: Path = OUT):
    data = Path(F.REPO.parent / "lro_data")
    rows, _ = F.read_day(data, day_dir)
    setup = F.Setup(data, rows, "DE440")
    obs = np.array([F.C * r.rtlt_s / 2.0 for _, r in rows])
    comp, elev, _ = F.compute(setup, rows, "M_FULL")
    t3 = setup.t3.copy()
    setup.t3 = t3 + 0.5
    cp, _, _ = F.compute(setup, rows, "M_FULL")
    setup.t3 = t3 - 0.5
    cm, _, _ = F.compute(setup, rows, "M_FULL")
    setup.t3 = t3
    rdot = cp - cm
    res = obs - comp
    passes = []
    for idx in F.passes(rows, t3):
        tt = (t3[idx] - t3[idx].mean()) / 3600.0
        r = res[idx]
        fits = {}
        for name, cols in (("bias_slope", [np.ones_like(tt), tt]), ("bias_dt", [np.ones_like(tt), rdot[idx]]),
                           ("bias_dt_slope", [np.ones_like(tt), rdot[idx], tt])):
            a = np.column_stack(cols)
            x, *_ = np.linalg.lstsq(a, r, rcond=None)
            fits[name] = dict(params=x.tolist(), rms_m=float(np.std(r - a @ x)))
        passes.append(dict(pad=rows[idx[0]][0], n=int(idx.size), t_start_s=float(t3[idx].min()),
                           rdot_range_mps=[float(rdot[idx].min()), float(rdot[idx].max())],
                           mean_residual_m=float(r.mean()), fits=fits))
    out.write_text(json.dumps(dict(day_dir=day_dir, passes=passes), indent=1), encoding="utf-8")
    for p in passes:
        f = p["fits"]
        print(p["pad"], p["n"], "rdot %.0f..%.0f" % tuple(p["rdot_range_mps"]),
              "| slope %.2f m/h rms %.3f | dt %.3f ms rms %.3f | dt %.3f ms slope %.2f rms %.3f" % (
                  f["bias_slope"]["params"][1], f["bias_slope"]["rms_m"], f["bias_dt"]["params"][1] * 1e3,
                  f["bias_dt"]["rms_m"], f["bias_dt_slope"]["params"][1] * 1e3, f["bias_dt_slope"]["params"][2],
                  f["bias_dt_slope"]["rms_m"]))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1], HERE.parent / "results" / sys.argv[2])
    else:
        main()
